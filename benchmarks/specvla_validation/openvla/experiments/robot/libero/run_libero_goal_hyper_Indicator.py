"""
run_libero_goal_hyper_Indicator.py

基于综合指标（曲率半径+位移）动态切换检索与AR生成。

规则：
- 综合指标 > composite_threshold：使用检索（纯DB）
- 综合指标 <= composite_threshold：使用verify（AR生成）
- 无verify时：连续跑两次检索，加入一次AR执行

综合指标 = alpha * 曲率半径指标 + (1-alpha) * 位移指标
- 曲率半径指标归一化范围：[0.000001, 0.014615]
- 位移指标归一化范围：[0.000009, 0.120187]

Usage:
    python experiments/robot/libero/run_libero_goal_hyper_Indicator.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --spec_checkpoint <SPEC_CHECKPOINT_PATH> \
        --task_suite_name libero_goal \
        --center_crop True \
        --composite_threshold 0.4 \
        --alpha 0.5 \
        --run_id_note <OPTIONAL TAG>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm

# Ensure LIBERO is discoverable
sys.path.append("/path/to/SpecVLA/LIBERO")
from libero.libero import benchmark

import wandb
import json
import requests
from io import BytesIO
from PIL import Image
import time as time_module

RETRIEVAL_URL = "http://127.0.0.1:5002/pipeline"

# Append project root (for openvla package imports)
sys.path.append("/path/to/SpecVLA")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.libero.calc_r import CompositeMetricsCalculator


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    use_spec: bool = True
    parallel_draft: bool = False
    accept_threshold: int = 9

    #################################################################################################################
    # Composite metrics-based decision parameters
    #################################################################################################################
    window_size: int = 5
    composite_threshold: float = 0.4  # 综合指标阈值
    alpha: float = 0.5  # 曲率半径指标权重
    displacement_range_min: float = 0.000009
    displacement_range_max: float = 0.120187
    radius_range_min: float = 0.000001
    radius_range_max: float = 0.014615

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Union[str, Path] = "/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 10

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 7

    # fmt: on


def action_to_tokens(action, model, unnorm_key):
    """
    将连续的action转换为token IDs，使用与predict_action相同的离散化过程。
    
    参考 modeling_speculation.py 794-799行的逆过程：
    1. 归一化action到[-1, 1]
    2. 离散化到bin索引
    3. 转换为vocab token IDs
    
    Args:
        action: numpy array of shape (7,) - 连续动作
        model: VLA模型实例
        unnorm_key: 反归一化key
        
    Returns:
        token_ids: numpy array of shape (7,) - token IDs
    """
    # 1. 归一化action到[-1, 1]
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    
    # 归一化: normalized = 2 * (action - low) / (high - low) - 1
    normalized_actions = np.where(
        mask,
        2.0 * (action - action_low) / (action_high - action_low) - 1.0,
        action,
    )
    
    # 2. 离散化到bin索引
    # 找到每个归一化action值最接近的bin center
    bin_centers = model.bin_centers  # shape: (256,)
    discretized_actions = np.zeros(7, dtype=np.int64)
    
    for i in range(7):
        # 找到最接近的bin center索引
        distances = np.abs(bin_centers - normalized_actions[i])
        discretized_actions[i] = np.argmin(distances)
    
    # 3. 转换为vocab token IDs
    # 参考predict_action的逆过程：discretized_actions = vocab_size - predicted_action_token_ids - 1
    # 因此：predicted_action_token_ids = vocab_size - discretized_actions - 1
    vocab_size = model.vocab_size
    token_ids = vocab_size - discretized_actions - 1
    
    return token_ids


@draccus.wrap()
def rollout(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if cfg.use_spec:
        assert cfg.spec_checkpoint is not None, "cfg.spec_checkpoint must not be None when use_spec=True"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load VLA model
    model = get_model(cfg)

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize logging directory and file
    run_id = f"HYPER_INDICATOR-{cfg.task_suite_name}-alpha{cfg.alpha}-thresh{cfg.composite_threshold}"
    if cfg.run_id_note is not None:
        run_id += f"-{cfg.run_id_note}"
    run_id += f"-{DATE_TIME}"
    
    log_dir = Path(cfg.local_log_dir) / run_id
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = open(log_dir / "log.txt", "w")
    print(f"Logging to {log_dir}")
    log_file.write(f"Run ID: {run_id}\n")
    log_file.write(f"Composite threshold: {cfg.composite_threshold}\n")
    log_file.write(f"Alpha: {cfg.alpha}\n")
    log_file.write(f"Window size: {cfg.window_size}\n")

    # Initialize W&B
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Load task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    print("Experiment: Hyper Indicator - composite metric-based decision")
    log_file.write("Experiment: Hyper Indicator - composite metric-based decision\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_retrieval_data = []  # 存储所有检索验证数据
    all_accept_lengths = []  # 存储所有accept_length用于统计（仅AR模式）
    observations_data = {}  # 存储所有observations
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_retrieval_data = []  # 当前任务的检索数据
        task_accept_lengths = []  # 当前任务的accept_length列表
        observations_data[task_id] = {}  # 初始化当前任务的observations字典

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            episode_retrieval_data = []  # 当前episode的检索数据
            episode_actions = []  # 当前episode的action列表

            # 综合指标计算器（每个episode重置）
            metrics_calc = CompositeMetricsCalculator(
                window_size=cfg.window_size,
                displacement_range=(cfg.displacement_range_min, cfg.displacement_range_max),
                radius_range=(cfg.radius_range_min, cfg.radius_range_max),
            )

            # 用于无verify模式的计数器：连续两次检索后，强制一次AR
            retrieval_counter = 0  # 记录连续检索次数

            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []  # 第三人称视角
            replay_images_wrist = []  # 肘部摄像头视角
            frame_annotations = []  # 收集每帧的标注信息

            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed images from both cameras
                    img = get_libero_image(obs, resize_size)  # 第三人称
                    img_wrist = get_libero_wrist_image(obs, resize_size)  # 肘部摄像头
                    replay_images.append(img)
                    replay_images_wrist.append(img_wrist)

                    # Prepare observations dict (only for model input)
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # ========================================
                    # 使用实际机器人末端位置更新综合指标计算器
                    # ========================================
                    # 获取末端执行器的xyz位置（前3维）
                    eef_position = obs["robot0_eef_pos"]  # shape: (3,) - 从环境中获取的准确位置
                    # 构造一个7维的位置向量用于指标计算（只有前3维是位置，后4维用0填充）
                    position_for_metrics = np.concatenate([eef_position, np.zeros(4)])
                    metrics_calc.update_history(position_for_metrics)

                    # ========================================
                    # 基于综合指标决定是否使用检索
                    # ========================================
                    composite_metric = metrics_calc.compute_composite_metric(alpha=cfg.alpha)
                    metrics_info = metrics_calc.get_current_metrics(alpha=cfg.alpha)
                    
                    # 根据综合指标决定使用DB还是AR
                    if np.isnan(composite_metric):
                        # 历史不足时默认使用AR
                        use_db = False
                        decision_reason = "insufficient_history"
                    else:
                        # 综合指标 > threshold: 使用检索
                        # 综合指标 <= threshold: 使用AR
                        use_db = composite_metric > cfg.composite_threshold
                        decision_reason = f"composite={composite_metric:.4f}"

                    # 无verify模式：连续两次检索后，强制一次AR
                    if use_db:
                        retrieval_counter += 1
                        if retrieval_counter > 2:
                            use_db = False
                            retrieval_counter = 0
                            decision_reason = "forced_AR_after_2_retrievals"
                    else:
                        retrieval_counter = 0

                    # ========================================
                    # 步骤1: 检索相似的action（只在使用DB时检索，节省时间）
                    # ========================================
                    retrieved_action = None
                    retrieved_tokens = None
                    retrieval_time = 0.0
                    tokenization_time = 0.0
                    retrieval_success = False

                    if use_db:
                        # 只在DB模式时检索
                        try:
                            pil_img = Image.fromarray(img)
                            buf = BytesIO()
                            pil_img.save(buf, format='PNG')
                            buf.seek(0)

                            files = {"file": ("image.png", buf, "image/png")}
                            data = {
                                "instruction": task_description,
                                "dataset_type": cfg.task_suite_name
                            }

                            # 记录检索时间
                            t0_retrieval = time_module.time()
                            response = requests.post(RETRIEVAL_URL, files=files, data=data, timeout=30)
                            t1_retrieval = time_module.time()
                            retrieval_time = t1_retrieval - t0_retrieval

                            if response.status_code == 200:
                                result = response.json()
                                if result.get('success', False):
                                    # 尝试获取trajectory
                                    retrieved_traj = None
                                    if 'rtcache_trajectory' in result and result['rtcache_trajectory']:
                                        retrieved_traj = np.array(result['rtcache_trajectory'])
                                    elif 'averaged_trajectory' in result and result['averaged_trajectory']:
                                        retrieved_traj = np.array(result['averaged_trajectory'])

                                    if retrieved_traj is not None and len(retrieved_traj) > 0:
                                        # 取第一个action
                                        if retrieved_traj.ndim == 1:
                                            retrieved_action = retrieved_traj
                                        else:
                                            retrieved_action = retrieved_traj[0]
                                        retrieval_success = True
                                        
                                        # 将检索的action转换为tokens（用于后续统计）
                                        try:
                                            t0_token = time_module.time()
                                            retrieved_tokens = action_to_tokens(retrieved_action, model, cfg.unnorm_key)
                                            t1_token = time_module.time()
                                            tokenization_time = t1_token - t0_token
                                        except Exception as e:
                                            print(f"Tokenization error: {e}")
                                            log_file.write(f"Tokenization error: {e}\n")
                                            retrieved_tokens = None
                        except Exception as e:
                            print(f"Retrieval error: {e}")
                            log_file.write(f"Retrieval error: {e}\n")

                    # ========================================
                    # 步骤3: 生成action（使用检索的tokens进行verify或不使用）
                    # ========================================
                    # 记录生成时间
                    t0_gen = time_module.time()

                    if use_db:
                        # ========================================
                        # DB模式：使用检索的action
                        # ========================================
                        if retrieval_success and retrieved_tokens is not None:
                            action = retrieved_action.copy()
                            accept_length = 7  # DB模式假设全部接受
                            mode = "DB"
                        else:
                            # 检索失败，使用dummy action
                            action = get_libero_dummy_action(cfg.model_family)
                            accept_length = 0
                            mode = "DB_fallback"
                        
                        # DB action处理：只需要normalize gripper，不需要invert
                        action = normalize_gripper_action(action, binarize=True)
                        
                    else:
                        # ========================================
                        # AR模式：纯AR生成
                        # ========================================
                        action = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            generate_mode='AR',
                        )
                        mode = "AR"
                        
                        # 生成后，如果有retrieved_tokens，计算accept_length（只是统计，不影响速度）
                        if retrieved_tokens is not None:
                            try:
                                generated_tokens = action_to_tokens(action, model, cfg.unnorm_key)
                                
                                # 对比tokens计算accept_length
                                accept_length = 0
                                for i in range(7):
                                    if retrieved_tokens[i] == generated_tokens[i]:
                                        accept_length += 1
                                    else:
                                        break
                                
                                all_accept_lengths.append(accept_length)  # 统计accept_length
                            except Exception as e:
                                print(f"Accept length calculation error: {e}")
                                accept_length = 0
                        else:
                            accept_length = 0
                        
                        # AR action处理：需要normalize gripper + invert
                        action = normalize_gripper_action(action, binarize=True)
                        
                        # [OpenVLA] dataloader翻转了gripper符号，执行前需要翻转回来
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                    t1_gen = time_module.time()
                    generation_time = t1_gen - t0_gen

                    # ========================================
                    # 记录数据
                    # ========================================
                    step_data = {
                        "step": t - cfg.num_steps_wait,
                        "mode": mode,
                        "decision_reason": decision_reason,
                        "composite_metric": float(composite_metric) if not np.isnan(composite_metric) else None,
                        "raw_radius": float(metrics_info['raw']['radius']) if not np.isnan(metrics_info['raw']['radius']) else None,
                        "raw_displacement": float(metrics_info['raw']['displacement']) if not np.isnan(metrics_info['raw']['displacement']) else None,
                        "norm_radius": float(metrics_info['normalized']['radius']) if not np.isnan(metrics_info['normalized']['radius']) else None,
                        "norm_displacement": float(metrics_info['normalized']['displacement']) if not np.isnan(metrics_info['normalized']['displacement']) else None,
                        "retrieval_success": retrieval_success,
                        "accept_length": accept_length,
                        "retrieval_time": retrieval_time,
                        "tokenization_time": tokenization_time,
                        "generation_time": generation_time,
                        "total_time": retrieval_time + tokenization_time + generation_time,
                    }
                    episode_retrieval_data.append(step_data)

                    # Print step info (commented out to reduce verbosity)
                    # composite_str = f"{composite_metric:.4f}" if not np.isnan(composite_metric) else "nan"
                    # print(f"  Step {t-cfg.num_steps_wait}: mode={mode}, composite={composite_str}, accept_len={accept_length}")

                    # Step environment
                    obs, reward, done, info = env.step(action.tolist())
                    t += 1
                    
                    # 收集当前帧的标注信息
                    frame_annotations.append({
                        'composite_metric': float(composite_metric) if not np.isnan(composite_metric) else None,
                        'mode': mode,
                        'action': action.copy(),
                    })

                    # Check if task succeeded
                    if done:
                        task_successes += 1
                        total_successes += 1
                        print(f"Success! Episode {task_episodes+1} completed.")
                        log_file.write(f"Success! Episode {task_episodes+1} completed.\n")
                        break

                except Exception as e:
                    print(f"Error in episode: {e}")
                    log_file.write(f"Error in episode: {e}\n")
                    import traceback
                    traceback.print_exc()
                    break

            # ========================================
            # Episode结束，保存数据
            # ========================================
            task_episodes += 1
            total_episodes += 1

            # 保存episode数据
            observations_data[task_id][episode_idx] = {
                "retrieval_data": episode_retrieval_data,
                "success": done,
                "steps": t - cfg.num_steps_wait,
            }

            task_retrieval_data.extend(episode_retrieval_data)

            # 保存两个视频：第三人称视角和肘部摄像头视角
            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                frame_annotations=frame_annotations,
                camera_name="agentview",
            )
            save_rollout_video(
                replay_images_wrist,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                frame_annotations=frame_annotations,
                camera_name="wrist",
            )

        # ========================================
        # Task结束，打印详细统计
        # ========================================
        task_success_rate = 100.0 * task_successes / task_episodes if task_episodes > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Task {task_id} Completed: {task_description}")
        print(f"{'='*70}")
        print(f"Success Rate: {task_successes}/{task_episodes} = {task_success_rate:.1f}%")
        
        log_file.write(f"\n{'='*70}\n")
        log_file.write(f"Task {task_id} Completed: {task_description}\n")
        log_file.write(f"{'='*70}\n")
        log_file.write(f"Success Rate: {task_successes}/{task_episodes} = {task_success_rate:.1f}%\n")
        
        # 统计该任务的模式使用情况
        task_mode_counts = {}
        task_composite_metrics = []
        task_accept_lengths_task = []
        
        for data in task_retrieval_data:
            mode = data['mode']
            task_mode_counts[mode] = task_mode_counts.get(mode, 0) + 1
            
            if data['composite_metric'] is not None:
                task_composite_metrics.append(data['composite_metric'])
            
            if data['mode'] == 'AR' and data['accept_length'] > 0:
                task_accept_lengths_task.append(data['accept_length'])
        
        total_steps = sum(task_mode_counts.values())
        
        print(f"\nMode Usage Statistics:")
        log_file.write(f"\nMode Usage Statistics:\n")
        for mode in sorted(task_mode_counts.keys()):
            count = task_mode_counts[mode]
            percentage = 100.0 * count / total_steps if total_steps > 0 else 0
            print(f"  {mode:10s}: {count:4d} steps ({percentage:5.1f}%)")
            log_file.write(f"  {mode:10s}: {count:4d} steps ({percentage:5.1f}%)\n")
        
        # 综合指标统计
        if len(task_composite_metrics) > 0:
            print(f"\nComposite Metric Statistics:")
            print(f"  Mean:   {np.mean(task_composite_metrics):.4f}")
            print(f"  Std:    {np.std(task_composite_metrics):.4f}")
            print(f"  Min:    {np.min(task_composite_metrics):.4f}")
            print(f"  Max:    {np.max(task_composite_metrics):.4f}")
            print(f"  Median: {np.median(task_composite_metrics):.4f}")
            
            log_file.write(f"\nComposite Metric Statistics:\n")
            log_file.write(f"  Mean:   {np.mean(task_composite_metrics):.4f}\n")
            log_file.write(f"  Std:    {np.std(task_composite_metrics):.4f}\n")
            log_file.write(f"  Min:    {np.min(task_composite_metrics):.4f}\n")
            log_file.write(f"  Max:    {np.max(task_composite_metrics):.4f}\n")
            log_file.write(f"  Median: {np.median(task_composite_metrics):.4f}\n")
        
        # Accept length统计（AR模式）
        if len(task_accept_lengths_task) > 0:
            print(f"\nAccept Length Statistics (AR mode):")
            print(f"  Mean: {np.mean(task_accept_lengths_task):.2f}")
            print(f"  Std:  {np.std(task_accept_lengths_task):.2f}")
            
            log_file.write(f"\nAccept Length Statistics (AR mode):\n")
            log_file.write(f"  Mean: {np.mean(task_accept_lengths_task):.2f}\n")
            log_file.write(f"  Std:  {np.std(task_accept_lengths_task):.2f}\n")
        
        print(f"{'='*70}\n")
        log_file.write(f"{'='*70}\n\n")

        all_retrieval_data.extend(task_retrieval_data)

    # ========================================
    # 所有任务结束，打印总体统计
    # ========================================
    overall_success_rate = 100.0 * total_successes / total_episodes if total_episodes > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"OVERALL EXPERIMENT RESULTS")
    print(f"{'='*80}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Successes: {total_successes}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"OVERALL EXPERIMENT RESULTS\n")
    log_file.write(f"{'='*80}\n")
    log_file.write(f"Total Episodes: {total_episodes}\n")
    log_file.write(f"Total Successes: {total_successes}\n")
    log_file.write(f"Overall Success Rate: {overall_success_rate:.1f}%\n")
    
    # 统计各模式使用次数和比例
    mode_counts = {}
    all_composite_metrics = []
    all_retrieval_times = []
    all_generation_times = []
    
    for data in all_retrieval_data:
        mode = data['mode']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        if data['composite_metric'] is not None:
            all_composite_metrics.append(data['composite_metric'])
        
        if data['retrieval_time'] > 0:
            all_retrieval_times.append(data['retrieval_time'])
        
        if data['generation_time'] > 0:
            all_generation_times.append(data['generation_time'])
    
    total_steps = sum(mode_counts.values())
    
    print(f"\n{'='*80}")
    print(f"Mode Usage Statistics (Total {total_steps} steps)")
    print(f"{'='*80}")
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Mode Usage Statistics (Total {total_steps} steps)\n")
    log_file.write(f"{'='*80}\n")
    
    for mode in sorted(mode_counts.keys()):
        count = mode_counts[mode]
        percentage = 100.0 * count / total_steps if total_steps > 0 else 0
        print(f"  {mode:10s}: {count:6d} steps ({percentage:5.1f}%)")
        log_file.write(f"  {mode:10s}: {count:6d} steps ({percentage:5.1f}%)\n")
    
    # 综合指标统计
    if len(all_composite_metrics) > 0:
        print(f"\n{'='*80}")
        print(f"Composite Metric Statistics")
        print(f"{'='*80}")
        print(f"  Mean:       {np.mean(all_composite_metrics):.4f}")
        print(f"  Std:        {np.std(all_composite_metrics):.4f}")
        print(f"  Min:        {np.min(all_composite_metrics):.4f}")
        print(f"  Max:        {np.max(all_composite_metrics):.4f}")
        print(f"  Median:     {np.median(all_composite_metrics):.4f}")
        print(f"  25th %ile:  {np.percentile(all_composite_metrics, 25):.4f}")
        print(f"  75th %ile:  {np.percentile(all_composite_metrics, 75):.4f}")
        
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"Composite Metric Statistics\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"  Mean:       {np.mean(all_composite_metrics):.4f}\n")
        log_file.write(f"  Std:        {np.std(all_composite_metrics):.4f}\n")
        log_file.write(f"  Min:        {np.min(all_composite_metrics):.4f}\n")
        log_file.write(f"  Max:        {np.max(all_composite_metrics):.4f}\n")
        log_file.write(f"  Median:     {np.median(all_composite_metrics):.4f}\n")
        log_file.write(f"  25th %ile:  {np.percentile(all_composite_metrics, 25):.4f}\n")
        log_file.write(f"  75th %ile:  {np.percentile(all_composite_metrics, 75):.4f}\n")
        
        # 统计在阈值以上/以下的比例
        above_threshold = sum(1 for m in all_composite_metrics if m > cfg.composite_threshold)
        below_threshold = len(all_composite_metrics) - above_threshold
        print(f"\n  Above threshold ({cfg.composite_threshold}): {above_threshold} ({100.0*above_threshold/len(all_composite_metrics):.1f}%)")
        print(f"  Below threshold ({cfg.composite_threshold}): {below_threshold} ({100.0*below_threshold/len(all_composite_metrics):.1f}%)")
        
        log_file.write(f"\n  Above threshold ({cfg.composite_threshold}): {above_threshold} ({100.0*above_threshold/len(all_composite_metrics):.1f}%)\n")
        log_file.write(f"  Below threshold ({cfg.composite_threshold}): {below_threshold} ({100.0*below_threshold/len(all_composite_metrics):.1f}%)\n")
    
    # Accept length统计（AR模式）
    if len(all_accept_lengths) > 0:
        print(f"\n{'='*80}")
        print(f"Accept Length Statistics (AR mode, {len(all_accept_lengths)} samples)")
        print(f"{'='*80}")
        print(f"  Mean:   {np.mean(all_accept_lengths):.2f}")
        print(f"  Std:    {np.std(all_accept_lengths):.2f}")
        print(f"  Min:    {np.min(all_accept_lengths):.2f}")
        print(f"  Max:    {np.max(all_accept_lengths):.2f}")
        print(f"  Median: {np.median(all_accept_lengths):.2f}")
        
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"Accept Length Statistics (AR mode, {len(all_accept_lengths)} samples)\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"  Mean:   {np.mean(all_accept_lengths):.2f}\n")
        log_file.write(f"  Std:    {np.std(all_accept_lengths):.2f}\n")
        log_file.write(f"  Min:    {np.min(all_accept_lengths):.2f}\n")
        log_file.write(f"  Max:    {np.max(all_accept_lengths):.2f}\n")
        log_file.write(f"  Median: {np.median(all_accept_lengths):.2f}\n")
    
    # 时间统计
    if len(all_retrieval_times) > 0:
        print(f"\n{'='*80}")
        print(f"Timing Statistics")
        print(f"{'='*80}")
        print(f"Retrieval Time (DB mode, {len(all_retrieval_times)} samples):")
        print(f"  Mean: {np.mean(all_retrieval_times)*1000:.2f} ms")
        print(f"  Std:  {np.std(all_retrieval_times)*1000:.2f} ms")
        
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"Timing Statistics\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Retrieval Time (DB mode, {len(all_retrieval_times)} samples):\n")
        log_file.write(f"  Mean: {np.mean(all_retrieval_times)*1000:.2f} ms\n")
        log_file.write(f"  Std:  {np.std(all_retrieval_times)*1000:.2f} ms\n")
    
    if len(all_generation_times) > 0:
        print(f"\nGeneration Time (AR mode, {len(all_generation_times)} samples):")
        print(f"  Mean: {np.mean(all_generation_times)*1000:.2f} ms")
        print(f"  Std:  {np.std(all_generation_times)*1000:.2f} ms")
        
        log_file.write(f"\nGeneration Time (AR mode, {len(all_generation_times)} samples):\n")
        log_file.write(f"  Mean: {np.mean(all_generation_times)*1000:.2f} ms\n")
        log_file.write(f"  Std:  {np.std(all_generation_times)*1000:.2f} ms\n")
    
    print(f"{'='*80}\n")

    # 保存详细数据
    results_path = log_dir / "retrieval_data.json"
    with open(results_path, "w") as f:
        json.dump(all_retrieval_data, f, indent=2)
    print(f"\nDetailed retrieval data saved to {results_path}")

    # 保存observations数据
    obs_path = log_dir / "observations_data.json"
    with open(obs_path, "w") as f:
        json.dump(observations_data, f, indent=2)
    print(f"Observations data saved to {obs_path}")

    # W&B logging
    if cfg.use_wandb:
        wandb.log({
            "success_rate": 100.0 * total_successes / total_episodes,
            "mean_accept_length": mean_accept_length if len(all_accept_lengths) > 0 else 0,
        })
        wandb.finish()

    log_file.close()
    print("\nExperiment completed!")


if __name__ == "__main__":
    rollout()
