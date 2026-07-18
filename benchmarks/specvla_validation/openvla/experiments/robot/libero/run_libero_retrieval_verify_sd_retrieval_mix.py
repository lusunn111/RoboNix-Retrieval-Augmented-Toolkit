"""
run_libero_retrieval_verify_sd_retrieval_mix.py

基于综合指标动态切换SD（Speculative Decoding）与Retrieval策略。
使用Mix视角（第三人称+手腕相机）进行检索。

决策规则：
- 综合指标 > composite_threshold (0.143210)：使用Retrieval策略
  - Retrieval策略：2次verify(DB) + 1次noverify(AR) = 2:1
  - verify(DB)：直接使用检索的action
  - noverify(AR)：使用AR生成（不验证）
- 综合指标 <= composite_threshold：使用SD（Speculative Decoding）
- 历史不足（nan）：使用AR模式（避免SD状态问题）

归一化参数（minmax归一化，超过范围取0或1）：
- 位移指标：[0.000009, 0.139051]
- 曲率指标：[0.000001, 0.016873]
- alpha = 0.5（1:1构造综合指标）

Usage:
    python experiments/robot/libero/run_libero_retrieval_verify_sd_retrieval_mix.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --spec_checkpoint <SPEC_CHECKPOINT_PATH> \
        --task_suite_name libero_goal \
        --center_crop True \
        --composite_threshold 0.143210 \
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
from libero.libero import benchmark

import wandb
import json
import requests
from io import BytesIO
from PIL import Image
import time as time_module

# Mix retrieval server URL (port 5003 for mix view - third-person + wrist camera)
RETRIEVAL_URL = "http://127.0.0.1:5003/pipeline"

# Append current directory
sys.path.append("/path/to/SpecVLA/openvla")
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
    use_spec: bool = True  # 必须为True才能使用SD模式
    parallel_draft: bool = False
    accept_threshold: int = 9

    #################################################################################################################
    # Composite metrics-based decision parameters
    #################################################################################################################
    window_size: int = 5
    composite_threshold: float = 0.143210  # 阈值
    alpha: float = 0.5  # 1:1权重
    displacement_range_min: float = 0.000009
    displacement_range_max: float = 0.139051
    radius_range_min: float = 0.000001
    radius_range_max: float = 0.016873

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
    """
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    
    normalized_actions = np.where(
        mask,
        2.0 * (action - action_low) / (action_high - action_low) - 1.0,
        action,
    )
    
    bin_centers = model.bin_centers
    discretized_actions = np.zeros(7, dtype=np.int64)
    
    for i in range(7):
        distances = np.abs(bin_centers - normalized_actions[i])
        discretized_actions[i] = np.argmin(distances)
    
    vocab_size = model.vocab_size
    token_ids = vocab_size - discretized_actions - 1
    
    return token_ids


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.use_spec, "cfg.use_spec must be True to support both SD and AR modes!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model (SpecVLA model that supports both SD and AR modes)
    print("Loading SpecVLA model (supports both SD and AR modes)...")
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "../../../specdecoding/test-speed")
    target_dir = os.path.join(base_dir, "libero_sd_retrieval_mix")
    os.makedirs(target_dir, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-SD_Retrieval-alpha{cfg.alpha}-thresh{cfg.composite_threshold}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    local_log_datapath = os.path.join(target_dir, run_id + "_sd_retrieval.json")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    print("Experiment: SD + Retrieval (Mix View - Third-Person + Wrist)")
    log_file.write("Experiment: SD + Retrieval (Mix View - Third-Person + Wrist)\n")
    print(f"Retrieval URL: {RETRIEVAL_URL}")
    log_file.write(f"Retrieval URL: {RETRIEVAL_URL}\n")
    print(f"Composite threshold: {cfg.composite_threshold}")
    log_file.write(f"Composite threshold: {cfg.composite_threshold}\n")
    print(f"Alpha: {cfg.alpha} (1:1 ratio)")
    log_file.write(f"Alpha: {cfg.alpha} (1:1 ratio)\n")
    print(f"Displacement range: [{cfg.displacement_range_min}, {cfg.displacement_range_max}]")
    log_file.write(f"Displacement range: [{cfg.displacement_range_min}, {cfg.displacement_range_max}]\n")
    print(f"Radius range: [{cfg.radius_range_min}, {cfg.radius_range_max}]")
    log_file.write(f"Radius range: [{cfg.radius_range_min}, {cfg.radius_range_max}]\n")
    print("\nDecision Logic:")
    print("  综合指标 > threshold: 使用Retrieval策略（2次DB + 1次AR）")
    print("  综合指标 <= threshold: 使用SD (Speculative Decoding)")
    log_file.write("\nDecision Logic:\n")
    log_file.write("  综合指标 > threshold: 使用Retrieval策略（2次DB + 1次AR）\n")
    log_file.write("  综合指标 <= threshold: 使用SD (Speculative Decoding)\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_data = []  # 存储所有数据
    observations_data = {}  # 存储所有observations
    
    # 统计模式使用
    total_sd_steps = 0
    total_retrieval_db_steps = 0
    total_retrieval_ar_steps = 0
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_data = []  # 当前任务的数据
        observations_data[task_id] = {}  # 初始化当前任务的observations字典
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            episode_data = []  # 当前episode的数据
            episode_observations = []  # 当前episode的observations列表
            
            # 综合指标计算器（每个episode重置）
            # 使用 calc_r.py 中的 CompositeMetricsCalculator
            metrics_calc = CompositeMetricsCalculator(
                window_size=cfg.window_size,
                displacement_range=(cfg.displacement_range_min, cfg.displacement_range_max),
                radius_range=(cfg.radius_range_min, cfg.radius_range_max),
            )
            
            # Retrieval策略的计数器：2次DB后执行1次AR
            retrieval_consecutive_db_count = 0
            
            # 当前使用的策略：'SD' 或 'Retrieval'
            current_strategy = None
            
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            
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

                    # Get preprocessed images (both third-person and wrist views)
                    img = get_libero_image(obs, resize_size)
                    wrist_img = get_libero_wrist_image(obs, resize_size)
                    replay_images.append(img)

                    # Prepare observations dict
                    observation = {
                        "full_image": img,
                        "wrist_image": wrist_img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    
                    # 保存observation到列表
                    episode_observations.append(observation)

                    # ========================================
                    # 使用实际机器人末端位置更新综合指标计算器
                    # ========================================
                    eef_position = obs["robot0_eef_pos"]  # shape: (3,)
                    metrics_calc.update_history(eef_position)
                    
                    # ========================================
                    # 基于综合指标决定使用SD还是Retrieval策略
                    # ========================================
                    composite_metric = metrics_calc.compute_composite_metric(alpha=cfg.alpha)
                    metrics_info = metrics_calc.get_current_metrics(alpha=cfg.alpha)
                    
                    # 根据综合指标决定策略
                    if np.isnan(composite_metric):
                        # 历史不足时使用Retrieval策略（AR模式），避免SD模式的状态问题
                        use_retrieval_strategy = True
                        use_ar_for_insufficient = True  # 标记为AR模式
                        decision_reason = "insufficient_history_use_AR"
                    else:
                        use_ar_for_insufficient = False
                        # 综合指标 > threshold: 使用Retrieval策略
                        # 综合指标 <= threshold: 使用SD
                        use_retrieval_strategy = composite_metric > cfg.composite_threshold
                        decision_reason = f"composite={composite_metric:.4f}"
                    
                    # ========================================
                    # 根据策略生成action
                    # ========================================
                    # 重要：在调用模型前清理tree_mask，避免AR模式和SD模式之间的状态干扰
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'language_model'):
                        model.base_model.language_model.tree_mask = None
                    if hasattr(model, 'tree_mask'):
                        model.tree_mask = None
                    
                    action = None
                    mode = None
                    retrieval_success = False
                    retrieval_time = 0.0
                    generation_time = 0.0
                    
                    if use_retrieval_strategy:
                        # ========================================
                        # Retrieval策略：2次DB(verify) + 1次AR(noverify) = 2:1
                        # 如果是历史不足，直接用AR
                        # ========================================
                        current_strategy = "Retrieval"
                        
                        # 如果历史不足，直接使用AR
                        if use_ar_for_insufficient:
                            mode = "AR_insufficient_history"
                            total_retrieval_ar_steps += 1
                            
                            t0_generation = time_module.time()
                            action, time_tuple = get_action(
                                cfg,
                                model,
                                observation,
                                task_description,
                                processor=processor,
                                generate_mode='AR',
                                return_time=True
                            )
                            t1_generation = time_tuple[0]
                            t0_generation = time_tuple[1]
                            generation_time = t1_generation - t0_generation
                        # 检查是否应该执行AR（每3次中的第3次，即2:1比例，2次DB + 1次AR）
                        elif retrieval_consecutive_db_count >= 2:
                            # 执行AR（noverify）
                            mode = "Retrieval_AR"
                            total_retrieval_ar_steps += 1
                            retrieval_consecutive_db_count = 0  # 重置计数器
                            
                            t0_generation = time_module.time()
                            action, time_tuple = get_action(
                                cfg,
                                model,
                                observation,
                                task_description,
                                processor=processor,
                                generate_mode='AR',
                                return_time=True
                            )
                            t1_generation = time_tuple[0]
                            t0_generation = time_tuple[1]
                            generation_time = t1_generation - t0_generation
                        else:
                            # 执行DB（verify - 使用检索的action）
                            mode = "Retrieval_DB"
                            
                            # 尝试检索
                            try:
                                # Prepare third-person image
                                pil_third_person = Image.fromarray(img)
                                buf_third_person = BytesIO()
                                pil_third_person.save(buf_third_person, format='PNG')
                                buf_third_person.seek(0)
                                
                                # Prepare wrist image
                                pil_wrist = Image.fromarray(wrist_img)
                                buf_wrist = BytesIO()
                                pil_wrist.save(buf_wrist, format='PNG')
                                buf_wrist.seek(0)
                                
                                # Prepare request with two images (mix view)
                                files = {
                                    "third_person_image": ("third_person.png", buf_third_person, "image/png"),
                                    "wrist_image": ("wrist.png", buf_wrist, "image/png")
                                }
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
                                                action = retrieved_traj
                                            else:
                                                action = retrieved_traj[0]
                                            retrieval_success = True
                            except Exception as e:
                                print(f"Retrieval error: {e}")
                                log_file.write(f"Retrieval error: {e}\n")
                            
                            if retrieval_success:
                                # 检索成功，使用检索的action
                                total_retrieval_db_steps += 1
                                retrieval_consecutive_db_count += 1
                            else:
                                # 检索失败，fallback到AR
                                mode = "Retrieval_DB_fallback_AR"
                                total_retrieval_ar_steps += 1
                                retrieval_consecutive_db_count = 0  # 检索失败时重置计数器
                                
                                t0_generation = time_module.time()
                                action, time_tuple = get_action(
                                    cfg,
                                    model,
                                    observation,
                                    task_description,
                                    processor=processor,
                                    generate_mode='AR',
                                    return_time=True
                                )
                                t1_generation = time_tuple[0]
                                t0_generation = time_tuple[1]
                                generation_time = t1_generation - t0_generation
                    else:
                        # ========================================
                        # SD策略：使用Speculative Decoding
                        # ========================================
                        current_strategy = "SD"
                        mode = "SD"
                        total_sd_steps += 1
                        retrieval_consecutive_db_count = 0  # 切换到SD时重置计数器
                        
                        t0_generation = time_module.time()
                        action, time_tuple = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            generate_mode='speculative',
                            return_time=True
                        )
                        t1_generation = time_tuple[0]
                        t0_generation = time_tuple[1]
                        generation_time = t1_generation - t0_generation
                    
                    # 记录这一步的数据
                    step_data = {
                        'episode': task_episodes,
                        'step': t - cfg.num_steps_wait,
                        'strategy': current_strategy,
                        'mode': mode,
                        'decision_reason': decision_reason,
                        'composite_metric': float(composite_metric) if not np.isnan(composite_metric) else None,
                        'raw_radius': float(metrics_info['raw']['radius']) if not np.isnan(metrics_info['raw']['radius']) else None,
                        'raw_displacement': float(metrics_info['raw']['displacement']) if not np.isnan(metrics_info['raw']['displacement']) else None,
                        'norm_radius': float(metrics_info['normalized']['radius']) if not np.isnan(metrics_info['normalized']['radius']) else None,
                        'norm_displacement': float(metrics_info['normalized']['displacement']) if not np.isnan(metrics_info['normalized']['displacement']) else None,
                        'retrieval_success': retrieval_success,
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                        'retrieval_consecutive_db_count': retrieval_consecutive_db_count,
                    }
                    episode_data.append(step_data)

                    # Normalize gripper action
                    action = normalize_gripper_action(action, binarize=True)
                    
                    # Invert gripper action for OpenVLA
                    # 注意：检索的action（DB模式）已经是正确的格式，不需要invert
                    # 但AR和SD模式需要invert
                    if mode in ["Retrieval_AR", "Retrieval_DB_fallback_AR", "SD", "AR_insufficient_history"] and cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            
            # 将episode数据存入observations_data
            episode_success = done if 'done' in dir() else False
            observations_data[task_id][episode_idx] = {
                'observations': episode_observations,
                'success': episode_success,
                'task_description': task_description,
                'num_steps': len(episode_observations)
            }
            
            # 保存episode的数据
            task_data.append({
                'task_id': task_id,
                'task_description': task_description,
                'episode_idx': episode_idx,
                'success': bool(done),
                'steps': episode_data
            })

            # Save replay video
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
        
        # 计算并记录统计数据
        total_steps_task = sum(len(ep['steps']) for ep in task_data)
        
        # 分别统计各模式
        sd_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'SD')
            for ep in task_data
        )
        retrieval_db_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'Retrieval_DB')
            for ep in task_data
        )
        retrieval_ar_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] in ['Retrieval_AR', 'Retrieval_DB_fallback_AR', 'AR_insufficient_history'])
            for ep in task_data
        )
        
        print(f"\nTask {task_id} Statistics:")
        print(f"  Total steps: {total_steps_task}")
        print(f"  SD mode steps: {sd_steps_count}")
        print(f"  Retrieval_DB mode steps: {retrieval_db_steps_count}")
        print(f"  Retrieval_AR mode steps: {retrieval_ar_steps_count}")
        
        log_file.write(f"\nTask {task_id} Statistics:\n")
        log_file.write(f"  Total steps: {total_steps_task}\n")
        log_file.write(f"  SD mode steps: {sd_steps_count}\n")
        log_file.write(f"  Retrieval_DB mode steps: {retrieval_db_steps_count}\n")
        log_file.write(f"  Retrieval_AR mode steps: {retrieval_ar_steps_count}\n")
        
        # 将当前任务的数据添加到总体数据
        all_data.append({
            'task_id': task_id,
            'task_description': task_description,
            'episodes': task_data
        })
    
    # 打印总体统计
    print("\n" + "="*80)
    print("Overall Statistics:")
    print("="*80)
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Success rate: {total_successes/total_episodes*100:.1f}%")
    
    # 总体模式统计
    all_sd_steps = sum(
        sum(1 for step in ep['steps'] if step['mode'] == 'SD')
        for task_data in all_data
        for ep in task_data['episodes']
    )
    all_retrieval_db_steps = sum(
        sum(1 for step in ep['steps'] if step['mode'] == 'Retrieval_DB')
        for task_data in all_data
        for ep in task_data['episodes']
    )
    all_retrieval_ar_steps = sum(
        sum(1 for step in ep['steps'] if step['mode'] in ['Retrieval_AR', 'Retrieval_DB_fallback_AR', 'AR_insufficient_history'])
        for task_data in all_data
        for ep in task_data['episodes']
    )
    total_all_steps = all_sd_steps + all_retrieval_db_steps + all_retrieval_ar_steps
    
    print(f"\nMode Statistics:")
    print(f"  Total SD steps: {all_sd_steps} ({100.0*all_sd_steps/total_all_steps:.1f}%)" if total_all_steps > 0 else "  Total SD steps: 0")
    print(f"  Total Retrieval_DB steps: {all_retrieval_db_steps} ({100.0*all_retrieval_db_steps/total_all_steps:.1f}%)" if total_all_steps > 0 else "  Total Retrieval_DB steps: 0")
    print(f"  Total Retrieval_AR steps: {all_retrieval_ar_steps} ({100.0*all_retrieval_ar_steps/total_all_steps:.1f}%)" if total_all_steps > 0 else "  Total Retrieval_AR steps: 0")
    
    total_retrieval_steps = all_retrieval_db_steps + all_retrieval_ar_steps
    print(f"\n  ============ 关键比例 ============")
    print(f"  SD策略 : Retrieval策略 = {all_sd_steps}:{total_retrieval_steps}")
    print(f"  Retrieval内部 DB(verify) : AR(noverify) = {all_retrieval_db_steps}:{all_retrieval_ar_steps}")
    if all_retrieval_ar_steps > 0:
        print(f"  Retrieval内部 DB:AR 比值 = {all_retrieval_db_steps/all_retrieval_ar_steps:.2f}:1 (目标: 2:1)")
    print(f"  ==================================")
    
    # ============ 时间统计 ============
    # 收集所有时间数据
    all_sd_times = []
    all_retrieval_db_times = []
    all_retrieval_ar_times = []
    
    for task_data_item in all_data:
        for ep in task_data_item['episodes']:
            for step in ep['steps']:
                if step['mode'] == 'SD' and step['generation_time'] > 0:
                    all_sd_times.append(step['generation_time'])
                elif step['mode'] == 'Retrieval_DB' and step['retrieval_time'] > 0:
                    all_retrieval_db_times.append(step['retrieval_time'])
                elif step['mode'] in ['Retrieval_AR', 'Retrieval_DB_fallback_AR', 'AR_insufficient_history'] and step['generation_time'] > 0:
                    all_retrieval_ar_times.append(step['generation_time'])
    
    print(f"\n  ============ 时间统计 ============")
    if len(all_sd_times) > 0:
        avg_sd_time = np.mean(all_sd_times)
        std_sd_time = np.std(all_sd_times)
        print(f"  SD模式生成时间:")
        print(f"    Mean: {avg_sd_time*1000:.2f} ms")
        print(f"    Std:  {std_sd_time*1000:.2f} ms")
        print(f"    Samples: {len(all_sd_times)}")
    else:
        print(f"  SD模式生成时间: 无数据")
        avg_sd_time = 0
    
    if len(all_retrieval_db_times) > 0:
        avg_retrieval_db_time = np.mean(all_retrieval_db_times)
        std_retrieval_db_time = np.std(all_retrieval_db_times)
        print(f"  Retrieval_DB检索时间:")
        print(f"    Mean: {avg_retrieval_db_time*1000:.2f} ms")
        print(f"    Std:  {std_retrieval_db_time*1000:.2f} ms")
        print(f"    Samples: {len(all_retrieval_db_times)}")
    else:
        print(f"  Retrieval_DB检索时间: 无数据")
        avg_retrieval_db_time = 0
    
    if len(all_retrieval_ar_times) > 0:
        avg_retrieval_ar_time = np.mean(all_retrieval_ar_times)
        std_retrieval_ar_time = np.std(all_retrieval_ar_times)
        print(f"  Retrieval_AR生成时间:")
        print(f"    Mean: {avg_retrieval_ar_time*1000:.2f} ms")
        print(f"    Std:  {std_retrieval_ar_time*1000:.2f} ms")
        print(f"    Samples: {len(all_retrieval_ar_times)}")
    else:
        print(f"  Retrieval_AR生成时间: 无数据")
        avg_retrieval_ar_time = 0
    
    # 计算加权平均每步时间
    if total_all_steps > 0:
        weighted_avg_time = (
            all_sd_steps * avg_sd_time + 
            all_retrieval_db_steps * avg_retrieval_db_time + 
            all_retrieval_ar_steps * avg_retrieval_ar_time
        ) / total_all_steps
        print(f"\n  加权平均每步时间: {weighted_avg_time*1000:.2f} ms")
    print(f"  ==================================")
    
    print("="*80)
    
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("Overall Statistics:\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Total episodes: {total_episodes}\n")
    log_file.write(f"Total successes: {total_successes}\n")
    log_file.write(f"Success rate: {total_successes/total_episodes*100:.1f}%\n")
    log_file.write(f"\nMode Statistics:\n")
    log_file.write(f"  Total SD steps: {all_sd_steps} ({100.0*all_sd_steps/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "  Total SD steps: 0\n")
    log_file.write(f"  Total Retrieval_DB steps: {all_retrieval_db_steps} ({100.0*all_retrieval_db_steps/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "  Total Retrieval_DB steps: 0\n")
    log_file.write(f"  Total Retrieval_AR steps: {all_retrieval_ar_steps} ({100.0*all_retrieval_ar_steps/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "  Total Retrieval_AR steps: 0\n")
    
    log_file.write(f"\n  ============ 关键比例 ============\n")
    log_file.write(f"  SD策略 : Retrieval策略 = {all_sd_steps}:{total_retrieval_steps}\n")
    log_file.write(f"  Retrieval内部 DB(verify) : AR(noverify) = {all_retrieval_db_steps}:{all_retrieval_ar_steps}\n")
    if all_retrieval_ar_steps > 0:
        log_file.write(f"  Retrieval内部 DB:AR 比值 = {all_retrieval_db_steps/all_retrieval_ar_steps:.2f}:1 (目标: 2:1)\n")
    log_file.write(f"  ==================================\n")
    
    # 时间统计写入日志
    log_file.write(f"\n  ============ 时间统计 ============\n")
    if len(all_sd_times) > 0:
        log_file.write(f"  SD模式生成时间:\n")
        log_file.write(f"    Mean: {avg_sd_time*1000:.2f} ms\n")
        log_file.write(f"    Std:  {std_sd_time*1000:.2f} ms\n")
        log_file.write(f"    Samples: {len(all_sd_times)}\n")
    else:
        log_file.write(f"  SD模式生成时间: 无数据\n")
    
    if len(all_retrieval_db_times) > 0:
        log_file.write(f"  Retrieval_DB检索时间:\n")
        log_file.write(f"    Mean: {avg_retrieval_db_time*1000:.2f} ms\n")
        log_file.write(f"    Std:  {std_retrieval_db_time*1000:.2f} ms\n")
        log_file.write(f"    Samples: {len(all_retrieval_db_times)}\n")
    else:
        log_file.write(f"  Retrieval_DB检索时间: 无数据\n")
    
    if len(all_retrieval_ar_times) > 0:
        log_file.write(f"  Retrieval_AR生成时间:\n")
        log_file.write(f"    Mean: {avg_retrieval_ar_time*1000:.2f} ms\n")
        log_file.write(f"    Std:  {std_retrieval_ar_time*1000:.2f} ms\n")
        log_file.write(f"    Samples: {len(all_retrieval_ar_times)}\n")
    else:
        log_file.write(f"  Retrieval_AR生成时间: 无数据\n")
    
    if total_all_steps > 0:
        log_file.write(f"\n  加权平均每步时间: {weighted_avg_time*1000:.2f} ms\n")
    log_file.write(f"  ==================================\n")
    
    log_file.write("="*80 + "\n")
    
    # 保存所有数据
    with open(local_log_datapath, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\nData saved to: {local_log_datapath}")
    log_file.write(f"\nData saved to: {local_log_datapath}\n")
    
    # 保存observations数据（包含轨迹状态）
    local_log_obspath = os.path.join(target_dir, run_id + "_observations.npy")
    np.save(local_log_obspath, observations_data, allow_pickle=True)
    print(f"Observations data saved to: {local_log_obspath}")
    log_file.write(f"Observations data saved to: {local_log_obspath}\n")
    
    # Save local log file
    log_file.close()

    # Push metrics to wandb
    if cfg.use_wandb:
        wandb.log({
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        })
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
