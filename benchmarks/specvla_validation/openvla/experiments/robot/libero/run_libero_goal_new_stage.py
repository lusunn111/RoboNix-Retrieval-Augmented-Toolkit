"""
run_libero_goal_new_stage.py

基于曲率半径动态切换检索与AR生成。

规则：
- 曲率半径 > 0.006m：使用检索（纯DB）
- 曲率半径 <= 0.006m：使用verify（AR生成）

曲率半径计算使用滑动窗口最小二乘圆拟合（与 new_test_curvature.ipynb 一致）。
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
from experiments.robot.libero.calc_r import CurvatureCalculator


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
    # Curvature-based decision parameters
    #################################################################################################################
    curvature_window_size: int = 5
    curvature_threshold: float = 0.006

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
    # 1. 归一化action到[-1, 1]
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

    normalized_actions = np.where(
        mask,
        2.0 * (action - action_low) / (action_high - action_low) - 1.0,
        action,
    )

    # 2. 离散化到bin索引
    bin_centers = model.bin_centers  # shape: (256,)
    discretized_actions = np.zeros(7, dtype=np.int64)

    for i in range(7):
        distances = np.abs(bin_centers - normalized_actions[i])
        discretized_actions[i] = np.argmin(distances)

    # 3. 转换为vocab token IDs
    vocab_size = model.vocab_size
    token_ids = vocab_size - discretized_actions - 1

    return token_ids


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
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
    target_dir = os.path.join(base_dir, "libero_goal_new_stage")
    os.makedirs(target_dir, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    local_log_datapath = os.path.join(target_dir, run_id + "_retrieval_verify.json")
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
    print("Experiment: Curvature-based Retrieval/AR switching")
    log_file.write("Experiment: Curvature-based Retrieval/AR switching\n")
    print(f"Curvature threshold: {cfg.curvature_threshold}m, window size: {cfg.curvature_window_size}")
    log_file.write(f"Curvature threshold: {cfg.curvature_threshold}m, window size: {cfg.curvature_window_size}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_retrieval_data = []  # 存储所有检索验证数据
    all_accept_lengths = []  # 存储所有accept_length用于统计（仅Model模式）
    observations_data = {}  # 存储动作轨迹: task_id -> episode_idx -> {'actions': list, ...}

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
            episode_actions = []  # 当前episode的action列表（仅存action slice）

            # 曲率计算器（每个episode重置）
            curvature_calc = CurvatureCalculator(
                window_size=cfg.curvature_window_size,
                curvature_threshold=cfg.curvature_threshold,
            )

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

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    # Prepare observations dict (only for model input)
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # ========================================
                    # 使用实际机器人末端位置更新曲率计算器
                    # ========================================
                    # 获取末端执行器的xyz位置（前3维）
                    eef_position = obs["robot0_eef_pos"]  # shape: (3,)
                    # 构造一个7维的位置向量用于曲率计算（只有前3维是位置，后4维用0填充）
                    position_for_curvature = np.concatenate([eef_position, np.zeros(4)])
                    curvature_calc.update_history(position_for_curvature)

                    # ========================================
                    # 基于曲率半径决定是否使用检索
                    # ========================================
                    use_db = curvature_calc.should_use_retrieval()
                    decision_info = curvature_calc.get_decision_info()

                    # ========================================
                    # 步骤1: 检索相似的action
                    # ========================================
                    retrieved_action = None
                    retrieval_time = 0.0
                    retrieval_success = False

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
                    except Exception as e:
                        print(f"Retrieval error: {e}")
                        log_file.write(f"Retrieval error: {e}\n")

                    # ========================================
                    # 步骤2: 将检索的action转换为tokens（如果检索成功）
                    # ========================================
                    retrieved_tokens = None
                    tokenization_time = 0.0

                    if retrieval_success and retrieved_action is not None:
                        try:
                            t0_token = time_module.time()
                            retrieved_tokens = action_to_tokens(retrieved_action, model, cfg.unnorm_key)
                            t1_token = time_module.time()
                            tokenization_time = t1_token - t0_token
                        except Exception as e:
                            print(f"Tokenization error: {e}")
                            log_file.write(f"Tokenization error: {e}\n")
                            retrieved_tokens = None

                    # ========================================
                    # 步骤3: 根据模式生成或使用action
                    # ========================================
                    accept_length = -1

                    if use_db:
                        # DB模式：直接使用检索的action
                        if retrieval_success and retrieved_action is not None:
                            action = retrieved_action.copy()
                            generation_time = 0.0
                        else:
                            # 检索失败，使用dummy action
                            action = get_libero_dummy_action(cfg.model_family)
                            generation_time = 0.0
                    else:
                        # Model模式：使用AR生成，并与检索的tokens进行验证
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

                        # 如果有retrieved_tokens，计算accept_length
                        if retrieved_tokens is not None:
                            try:
                                generated_tokens = action_to_tokens(action, model, cfg.unnorm_key)

                                # 比较retrieved_tokens和generated_tokens
                                accept_length = 0
                                for i in range(7):
                                    if cfg.accept_threshold is None:
                                        if retrieved_tokens[i] == generated_tokens[i]:
                                            accept_length += 1
                                        else:
                                            break
                                    else:
                                        if abs(retrieved_tokens[i] - generated_tokens[i]) <= cfg.accept_threshold:
                                            accept_length += 1
                                        else:
                                            break

                                # 只在Model模式下统计accept_length
                                task_accept_lengths.append(accept_length)
                            except Exception as e:
                                print(f"Accept length calculation error: {e}")
                                log_file.write(f"Accept length calculation error: {e}\n")
                                accept_length = -1

                    # 存储action用于轨迹记录（注意：这是动作增量，不是位置）
                    episode_actions.append(np.array(action).tolist())

                    # 记录这一步的检索验证数据
                    step_data = {
                        'episode': task_episodes,
                        'step': t - cfg.num_steps_wait,
                        'mode': 'DB' if use_db else 'Model',
                        'retrieval_success': retrieval_success,
                        'retrieval_time': retrieval_time,
                        'tokenization_time': tokenization_time,
                        'generation_time': generation_time,
                        'accept_length': accept_length if not use_db else -1,  # Only record for Model mode
                        'has_retrieved_tokens': retrieved_tokens is not None,
                        'curvature_radius': decision_info['radius'],
                        'curvature_threshold': decision_info['threshold'],
                        'curvature_history_len': decision_info['history_length'],
                        'curvature_decision': decision_info['decision'],
                    }
                    episode_retrieval_data.append(step_data)

                    # Normalize gripper action
                    action = normalize_gripper_action(action, binarize=True)

                    # Invert gripper action for OpenVLA (only for Model mode, DB actions are already correct)
                    if not use_db and cfg.model_family == "openvla":
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

            # 将episode数据存入observations_data，包含动作轨迹
            episode_success = done if 'done' in dir() else False
            observations_data[task_id][episode_idx] = {
                'actions': episode_actions,
                'success': episode_success,
                'task_description': task_description,
                'num_steps': len(episode_actions)
            }

            # 保存episode的检索数据
            task_retrieval_data.append({
                'task_id': task_id,
                'task_description': task_description,
                'episode_idx': episode_idx,
                'success': bool(done),
                'steps': episode_retrieval_data
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

        all_accept_lengths.extend(task_accept_lengths)

        # 计算并记录统计数据
        total_retrievals = sum(len(ep['steps']) for ep in task_retrieval_data)
        successful_retrievals = sum(
            sum(1 for step in ep['steps'] if step['retrieval_success'])
            for ep in task_retrieval_data
        )

        # 分别统计DB和Model模式
        db_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'DB')
            for ep in task_retrieval_data
        )
        model_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'Model')
            for ep in task_retrieval_data
        )

        avg_retrieval_time = np.mean([
            step['retrieval_time']
            for ep in task_retrieval_data
            for step in ep['steps']
        ]) if total_retrievals > 0 else 0.0

        # DB模式平均时间（检索时间）
        db_times = [
            step['retrieval_time']
            for ep in task_retrieval_data
            for step in ep['steps']
            if step['mode'] == 'DB'
        ]
        avg_db_time = np.mean(db_times) if len(db_times) > 0 else 0.0

        # Model模式平均时间（生成时间）
        model_times = [
            step['generation_time']
            for ep in task_retrieval_data
            for step in ep['steps']
            if step['mode'] == 'Model'
        ]
        avg_model_time = np.mean(model_times) if len(model_times) > 0 else 0.0
        total_db_time = float(np.sum(db_times)) if len(db_times) > 0 else 0.0
        total_model_time = float(np.sum(model_times)) if len(model_times) > 0 else 0.0

        # 计算accept_length统计
        avg_accept_length = np.mean(task_accept_lengths) if len(task_accept_lengths) > 0 else 0.0
        median_accept_length = np.median(task_accept_lengths) if len(task_accept_lengths) > 0 else 0.0
        std_accept_length = np.std(task_accept_lengths) if len(task_accept_lengths) > 0 else 0.0
        max_accept_length = np.max(task_accept_lengths) if len(task_accept_lengths) > 0 else 0
        min_accept_length = np.min(task_accept_lengths) if len(task_accept_lengths) > 0 else 0

        print(f"\nTask {task_id} Statistics:")
        print(f"  Total retrieval attempts: {total_retrievals}")
        print(f"  DB mode steps: {db_steps_count} (avg time: {avg_db_time:.6f}s)")
        print(f"  Model mode steps: {model_steps_count} (avg time: {avg_model_time:.6f}s)")
        print(f"  DB total time: {total_db_time:.6f}s")
        print(f"  Model total time: {total_model_time:.6f}s")
        if task_episodes > 0:
            print(f"  Task success rate: {task_successes/task_episodes*100:.1f}% ({task_successes}/{task_episodes})")
        if total_retrievals > 0:
            print(f"  Successful retrievals: {successful_retrievals} ({successful_retrievals/total_retrievals*100:.1f}%)")
            if len(task_accept_lengths) > 0:
                print(f"  Accept Length Stats:")
                print(f"    Mean: {avg_accept_length:.2f}")
                print(f"    Median: {median_accept_length:.2f}")
                print(f"    Std: {std_accept_length:.2f}")
                print(f"    Min: {min_accept_length}, Max: {max_accept_length}")
        else:
            print(f"  No retrieval attempts (all episodes failed early)")

        log_file.write(f"\nTask {task_id} Statistics:\n")
        log_file.write(f"  Total retrieval attempts: {total_retrievals}\n")
        log_file.write(f"  DB mode steps: {db_steps_count} (avg time: {avg_db_time:.6f}s)\n")
        log_file.write(f"  Model mode steps: {model_steps_count} (avg time: {avg_model_time:.6f}s)\n")
        log_file.write(f"  DB total time: {total_db_time:.6f}s\n")
        log_file.write(f"  Model total time: {total_model_time:.6f}s\n")
        if task_episodes > 0:
            log_file.write(f"  Task success rate: {task_successes/task_episodes*100:.1f}% ({task_successes}/{task_episodes})\n")
        if total_retrievals > 0:
            log_file.write(f"  Successful retrievals: {successful_retrievals} ({successful_retrievals/total_retrievals*100:.1f}%)\n")
            if len(task_accept_lengths) > 0:
                log_file.write(f"  Accept Length Stats:\n")
                log_file.write(f"    Mean: {avg_accept_length:.2f}\n")
                log_file.write(f"    Median: {median_accept_length:.2f}\n")
                log_file.write(f"    Std: {std_accept_length:.2f}\n")
                log_file.write(f"    Min: {min_accept_length}, Max: {max_accept_length}\n")
        else:
            log_file.write(f"  No retrieval attempts (all episodes failed early)\n")

        # 将当前任务的数据添加到总体数据
        all_retrieval_data.append({
            'task_id': task_id,
            'task_description': task_description,
            'episodes': task_retrieval_data
        })

    # 打印总体统计
    print("\n" + "="*80)
    print("Overall Statistics:")
    print("="*80)
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Success rate: {total_successes/total_episodes*100:.1f}%")
    if total_episodes > 0:
        print(f"Success count: {total_successes}/{total_episodes}")

    # 总体DB/Model统计
    total_db_steps = sum(
        sum(1 for step in ep['steps'] if step['mode'] == 'DB')
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
    )
    total_model_steps = sum(
        sum(1 for step in ep['steps'] if step['mode'] == 'Model')
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
    )

    all_db_times = [
        step['retrieval_time']
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
        for step in ep['steps']
        if step['mode'] == 'DB'
    ]

    all_model_times = [
        step['generation_time']
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
        for step in ep['steps']
        if step['mode'] == 'Model'
    ]

    total_retrieval_attempts = sum(
        len(ep['steps'])
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
    )
    total_successful_retrievals = sum(
        sum(1 for step in ep['steps'] if step['retrieval_success'])
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
    )

    print(f"\nMode Statistics:")
    print(f"  Total DB steps: {total_db_steps} (avg time: {np.mean(all_db_times):.6f}s)" if len(all_db_times) > 0 else "  Total DB steps: 0")
    print(f"  Total Model steps: {total_model_steps} (avg time: {np.mean(all_model_times):.6f}s)" if len(all_model_times) > 0 else "  Total Model steps: 0")
    print(f"  Retrieval executions (DB): {total_db_steps}")
    print(f"  Verify executions (AR): {total_model_steps}")
    print(f"  DB total time: {float(np.sum(all_db_times)):.6f}s" if len(all_db_times) > 0 else "  DB total time: 0.000000s")
    print(f"  Model total time: {float(np.sum(all_model_times)):.6f}s" if len(all_model_times) > 0 else "  Model total time: 0.000000s")
    if total_retrieval_attempts > 0:
        print(f"  Retrieval success rate: {total_successful_retrievals/total_retrieval_attempts*100:.1f}% ({total_successful_retrievals}/{total_retrieval_attempts})")

    if len(all_accept_lengths) > 0:
        overall_avg_accept = np.mean(all_accept_lengths)
        overall_median_accept = np.median(all_accept_lengths)
        overall_std_accept = np.std(all_accept_lengths)
        overall_max_accept = np.max(all_accept_lengths)
        overall_min_accept = np.min(all_accept_lengths)

        print(f"\nAccept Length Statistics (all tasks):")
        print(f"  Mean: {overall_avg_accept:.2f}")
        print(f"  Median: {overall_median_accept:.2f}")
        print(f"  Std: {overall_std_accept:.2f}")
        print(f"  Min: {overall_min_accept}, Max: {overall_max_accept}")
        print(f"  Total samples: {len(all_accept_lengths)}")

    print("="*80)

    log_file.write("\n" + "="*80 + "\n")
    log_file.write("Overall Statistics:\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Total episodes: {total_episodes}\n")
    log_file.write(f"Total successes: {total_successes}\n")
    log_file.write(f"Success rate: {total_successes/total_episodes*100:.1f}%\n")
    if total_episodes > 0:
        log_file.write(f"Success count: {total_successes}/{total_episodes}\n")
    log_file.write(f"\nMode Statistics:\n")
    if len(all_db_times) > 0:
        log_file.write(f"  Total DB steps: {total_db_steps} (avg time: {np.mean(all_db_times):.6f}s)\n")
    else:
        log_file.write(f"  Total DB steps: 0\n")
    if len(all_model_times) > 0:
        log_file.write(f"  Total Model steps: {total_model_steps} (avg time: {np.mean(all_model_times):.6f}s)\n")
    else:
        log_file.write(f"  Total Model steps: 0\n")
    log_file.write(f"  Retrieval executions (DB): {total_db_steps}\n")
    log_file.write(f"  Verify executions (AR): {total_model_steps}\n")
    if len(all_db_times) > 0:
        log_file.write(f"  DB total time: {float(np.sum(all_db_times)):.6f}s\n")
    else:
        log_file.write("  DB total time: 0.000000s\n")
    if len(all_model_times) > 0:
        log_file.write(f"  Model total time: {float(np.sum(all_model_times)):.6f}s\n")
    else:
        log_file.write("  Model total time: 0.000000s\n")
    if total_retrieval_attempts > 0:
        log_file.write(f"  Retrieval success rate: {total_successful_retrievals/total_retrieval_attempts*100:.1f}% ({total_successful_retrievals}/{total_retrieval_attempts})\n")

    if len(all_accept_lengths) > 0:
        log_file.write(f"\nAccept Length Statistics (all tasks):\n")
        log_file.write(f"  Mean: {overall_avg_accept:.2f}\n")
        log_file.write(f"  Median: {overall_median_accept:.2f}\n")
        log_file.write(f"  Std: {overall_std_accept:.2f}\n")
        log_file.write(f"  Min: {overall_min_accept}, Max: {overall_max_accept}\n")
        log_file.write(f"  Total samples: {len(all_accept_lengths)}\n")

    log_file.write("="*80 + "\n")

    # 保存所有检索验证数据
    with open(local_log_datapath, 'w') as f:
        json.dump(all_retrieval_data, f, indent=2)

    print(f"\nRetrieval verification data saved to: {local_log_datapath}")
    log_file.write(f"\nRetrieval verification data saved to: {local_log_datapath}\n")

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
