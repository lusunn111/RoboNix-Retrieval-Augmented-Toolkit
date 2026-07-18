"""
run_fenxi_accept_len.py

分析每一步检索action slice的接受长度。
纯AR模式：每一步都使用AR生成action，同时检索并计算accept_length（阈值=9）。
将每个任务的accept_length列表存为.npy文件。

Usage:
    python experiments/robot/libero/run_fenxi_accept_len.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_goal \
        --center_crop True \
        --accept_threshold 9 \
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
    accept_threshold: int = 9  # 放松阈值

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Union[str, Path] = "/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1  # 每个任务只跑1次

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = 7

    # fmt: on


def compute_direction_consistency(positions, window_size=10, min_window=5):
    """
    计算方向一致性指标
    
    在窗口内：
    - 有 window_size 个位置点
    - 形成 window_size-1 个方向向量（相邻点之差）
    - 计算相邻方向向量之间的余弦相似度
    - 返回这些余弦值的方差（方差越小，方向越一致）
    
    Args:
        positions: 位置历史列表，每个元素是 (x, y, z) 或更高维
        window_size: 窗口大小，默认10
        min_window: 最小窗口大小，默认5
        
    Returns:
        direction_consistency: 方向一致性（余弦方差），如果数据不足返回 np.nan
    """
    if len(positions) < min_window:
        return np.nan
    
    # 取最近 window_size 个位置
    actual_window = min(window_size, len(positions))
    if actual_window < min_window:
        return np.nan
    
    recent_positions = positions[-actual_window:]
    
    # 计算方向向量
    direction_vectors = []
    for i in range(len(recent_positions) - 1):
        p1 = np.array(recent_positions[i][:3])  # 只取xyz
        p2 = np.array(recent_positions[i + 1][:3])
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm > 1e-8:  # 避免零向量
            direction_vectors.append(direction / norm)  # 归一化
    
    if len(direction_vectors) < 2:
        return np.nan
    
    # 计算相邻方向向量之间的余弦相似度
    cosines = []
    for i in range(len(direction_vectors) - 1):
        cos_sim = np.dot(direction_vectors[i], direction_vectors[i + 1])
        # 限制在 [-1, 1] 范围内（数值误差可能导致超出）
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        cosines.append(cos_sim)
    
    if len(cosines) < 1:
        return np.nan
    
    # 计算余弦值的方差
    variance = np.var(cosines)
    
    return variance


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


def sanitize_task_name(task_description):
    """
    将task_description转换为合法的文件名
    """
    # 替换空格为下划线，移除或替换特殊字符
    name = task_description.lower()
    name = name.replace(" ", "_")
    # 移除不合法的文件名字符
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        name = name.replace(char, '')
    return name


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

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
    target_dir = os.path.join(base_dir, "fenxi_accept_len")
    os.makedirs(target_dir, exist_ok=True)
    
    run_id = f"FENXI-{cfg.task_suite_name}-{cfg.model_family}-thresh{cfg.accept_threshold}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    
    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    print(f"Accept length .npy files will be saved to: {target_dir}")
    log_file.write(f"Accept length .npy files will be saved to: {target_dir}\n")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    print("Experiment: Pure AR with Accept Length Analysis")
    log_file.write("Experiment: Pure AR with Accept Length Analysis\n")
    print(f"Accept threshold: {cfg.accept_threshold}")
    log_file.write(f"Accept threshold: {cfg.accept_threshold}\n")
    print(f"Retrieval URL: {RETRIEVAL_URL}")
    log_file.write(f"Retrieval URL: {RETRIEVAL_URL}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_accept_lengths_global = []  # 全局accept_length统计
    all_execution_degrees_global = []  # 全局execution_degree统计
    all_similarity_scores_global = []  # 全局similarity_score统计
    all_direction_consistency_global = []  # 全局direction_consistency统计
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # 获取任务名（用于保存.npy）
        task_name = sanitize_task_name(task_description)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_accept_lengths = []  # 当前任务的accept_length列表
        task_execution_degrees = []  # 当前任务的execution_degree列表
        task_similarity_scores = []  # 当前任务的相似度分数列表
        task_direction_consistency = []  # 当前任务的方向一致性列表
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            episode_accept_lengths = []  # 当前episode的accept_length列表
            episode_execution_degrees = []  # 当前episode的execution_degree列表
            episode_similarity_scores = []  # 当前episode的检索相似度分数列表
            episode_direction_consistency = []  # 当前episode的方向一致性列表
            episode_positions = []  # 用于计算方向一致性的位置历史
            
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

                    # ========================================
                    # 步骤1: 检索相似的action
                    # ========================================
                    retrieved_action = None
                    retrieval_success = False
                    similarity_score = 0.0  # 检索的相似度分数
                    
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
                        
                        response = requests.post(RETRIEVAL_URL, files=files, data=data, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            if result.get('success', False):
                                # 获取相似度分数 (top_score)
                                similarity_score = result.get('top_score', 0.0)
                                
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
                    # 步骤2: 将检索的action转换为tokens
                    # ========================================
                    retrieved_tokens = None
                    
                    if retrieval_success and retrieved_action is not None:
                        try:
                            retrieved_tokens = action_to_tokens(retrieved_action, model, cfg.unnorm_key)
                        except Exception as e:
                            print(f"Tokenization error: {e}")
                            log_file.write(f"Tokenization error: {e}\n")
                            retrieved_tokens = None

                    # ========================================
                    # 步骤3: 使用AR生成action
                    # ========================================
                    action, _ = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        generate_mode='AR',
                        return_time=True
                    )

                    # ========================================
                    # 步骤4: 计算accept_length和execution_degree
                    # ========================================
                    accept_length = 0
                    execution_degree = 0  # 总共有多少维度在阈值内（不要求连续）
                    
                    if retrieved_tokens is not None:
                        try:
                            generated_tokens = action_to_tokens(action, model, cfg.unnorm_key)
                            
                            # 比较retrieved_tokens和generated_tokens
                            # accept_length: 从头开始连续匹配的长度
                            for i in range(7):
                                if abs(retrieved_tokens[i] - generated_tokens[i]) <= cfg.accept_threshold:
                                    accept_length += 1
                                else:
                                    break
                            
                            # execution_degree: 总共有多少维度在阈值内（不要求连续）
                            for i in range(7):
                                if abs(retrieved_tokens[i] - generated_tokens[i]) <= cfg.accept_threshold:
                                    execution_degree += 1
                            
                            episode_accept_lengths.append(accept_length)
                            episode_execution_degrees.append(execution_degree)
                            episode_similarity_scores.append(similarity_score)
                        except Exception as e:
                            print(f"Accept length calculation error: {e}")
                            log_file.write(f"Accept length calculation error: {e}\n")
                            # 检索失败时记录0
                            episode_accept_lengths.append(0)
                            episode_execution_degrees.append(0)
                            episode_similarity_scores.append(similarity_score)
                    else:
                        # 检索失败时记录0
                        episode_accept_lengths.append(0)
                        episode_execution_degrees.append(0)
                        episode_similarity_scores.append(0.0)
                    
                    # ========================================
                    # 步骤5: 记录位置并计算方向一致性
                    # ========================================
                    # 记录当前末端执行器位置
                    eef_pos = obs["robot0_eef_pos"]  # shape: (3,)
                    episode_positions.append(eef_pos.copy())
                    
                    # 计算方向一致性
                    dir_consistency = compute_direction_consistency(
                        episode_positions, window_size=10, min_window=5
                    )
                    episode_direction_consistency.append(dir_consistency)

                    # Normalize gripper action
                    action = normalize_gripper_action(action, binarize=True)
                    
                    # Invert gripper action for OpenVLA
                    if cfg.model_family == "openvla":
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
            
            # 将episode的数据添加到task列表
            task_accept_lengths.extend(episode_accept_lengths)
            task_execution_degrees.extend(episode_execution_degrees)
            task_similarity_scores.extend(episode_similarity_scores)
            task_direction_consistency.extend(episode_direction_consistency)

            # Save replay video
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            print(f"Episode accept_lengths count: {len(episode_accept_lengths)}")
            if len(episode_accept_lengths) > 0:
                print(f"Episode accept_length mean: {np.mean(episode_accept_lengths):.2f}")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.write(f"Episode accept_lengths count: {len(episode_accept_lengths)}\n")
            if len(episode_accept_lengths) > 0:
                log_file.write(f"Episode accept_length mean: {np.mean(episode_accept_lengths):.2f}\n")
        
        # ========================================
        # 保存当前任务的accept_lengths为.npy
        # ========================================
        # 保存accept_lengths
        task_npy_path = os.path.join(target_dir, f"{task_name}.npy")
        np.save(task_npy_path, np.array(task_accept_lengths))
        print(f"\nTask accept_lengths saved to: {task_npy_path}")
        log_file.write(f"\nTask accept_lengths saved to: {task_npy_path}\n")
        
        # 保存execution_degrees
        task_exec_npy_path = os.path.join(target_dir, f"{task_name}_exec_degree.npy")
        np.save(task_exec_npy_path, np.array(task_execution_degrees))
        print(f"Task execution_degrees saved to: {task_exec_npy_path}")
        log_file.write(f"Task execution_degrees saved to: {task_exec_npy_path}\n")
        
        # 保存similarity_scores
        task_sim_npy_path = os.path.join(target_dir, f"{task_name}_similarity.npy")
        np.save(task_sim_npy_path, np.array(task_similarity_scores))
        print(f"Task similarity_scores saved to: {task_sim_npy_path}")
        log_file.write(f"Task similarity_scores saved to: {task_sim_npy_path}\n")
        
        # 保存direction_consistency
        task_dir_npy_path = os.path.join(target_dir, f"{task_name}_direction.npy")
        np.save(task_dir_npy_path, np.array(task_direction_consistency))
        print(f"Task direction_consistency saved to: {task_dir_npy_path}")
        log_file.write(f"Task direction_consistency saved to: {task_dir_npy_path}\n")
        
        # 添加到全局统计
        all_accept_lengths_global.extend(task_accept_lengths)
        all_execution_degrees_global.extend(task_execution_degrees)
        all_similarity_scores_global.extend(task_similarity_scores)
        all_direction_consistency_global.extend(task_direction_consistency)
        
        # 打印当前任务统计
        if len(task_accept_lengths) > 0:
            print(f"Task '{task_description}' Statistics:")
            print(f"  Total steps: {len(task_accept_lengths)}")
            print(f"  [Accept Length] Mean: {np.mean(task_accept_lengths):.2f}, Median: {np.median(task_accept_lengths):.2f}")
            print(f"  [Exec Degree]   Mean: {np.mean(task_execution_degrees):.2f}, Median: {np.median(task_execution_degrees):.2f}")
            print(f"  [Similarity]    Mean: {np.mean(task_similarity_scores):.4f}, Median: {np.median(task_similarity_scores):.4f}, Min: {np.min(task_similarity_scores):.4f}, Max: {np.max(task_similarity_scores):.4f}")
            # 方向一致性统计（过滤nan值）
            valid_dir = [d for d in task_direction_consistency if not np.isnan(d)]
            if len(valid_dir) > 0:
                print(f"  [Dir Consist]   Mean: {np.mean(valid_dir):.6f}, Median: {np.median(valid_dir):.6f}, Valid: {len(valid_dir)}/{len(task_direction_consistency)}")
            
            # 统计各accept_length的分布
            accept_counts = {}
            for al in task_accept_lengths:
                accept_counts[al] = accept_counts.get(al, 0) + 1
            print(f"  Accept Length Distribution: {dict(sorted(accept_counts.items()))}")
            
            # 统计各execution_degree的分布
            exec_counts = {}
            for ed in task_execution_degrees:
                exec_counts[ed] = exec_counts.get(ed, 0) + 1
            print(f"  Exec Degree Distribution: {dict(sorted(exec_counts.items()))}")
            
            log_file.write(f"Task '{task_description}' Statistics:\n")
            log_file.write(f"  Total steps: {len(task_accept_lengths)}\n")
            log_file.write(f"  [Accept Length] Mean: {np.mean(task_accept_lengths):.2f}, Median: {np.median(task_accept_lengths):.2f}\n")
            log_file.write(f"  [Exec Degree]   Mean: {np.mean(task_execution_degrees):.2f}, Median: {np.median(task_execution_degrees):.2f}\n")
            log_file.write(f"  [Similarity]    Mean: {np.mean(task_similarity_scores):.4f}, Median: {np.median(task_similarity_scores):.4f}, Min: {np.min(task_similarity_scores):.4f}, Max: {np.max(task_similarity_scores):.4f}\n")
            log_file.write(f"  Accept Length Distribution: {dict(sorted(accept_counts.items()))}\n")
            log_file.write(f"  Exec Degree Distribution: {dict(sorted(exec_counts.items()))}\n")
    
    # ========================================
    # 打印总体统计
    # ========================================
    print("\n" + "="*80)
    print("Overall Statistics:")
    print("="*80)
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Success rate: {total_successes/total_episodes*100:.1f}%")
    
    if len(all_accept_lengths_global) > 0:
        overall_avg_accept = np.mean(all_accept_lengths_global)
        overall_median_accept = np.median(all_accept_lengths_global)
        overall_std_accept = np.std(all_accept_lengths_global)
        overall_max_accept = np.max(all_accept_lengths_global)
        overall_min_accept = np.min(all_accept_lengths_global)
        
        print(f"\nGlobal Accept Length Statistics:")
        print(f"  Total samples: {len(all_accept_lengths_global)}")
        print(f"  Mean: {overall_avg_accept:.2f}")
        print(f"  Median: {overall_median_accept:.2f}")
        print(f"  Std: {overall_std_accept:.2f}")
        print(f"  Min: {overall_min_accept}, Max: {overall_max_accept}")
        
        # 全局分布
        global_counts = {}
        for al in all_accept_lengths_global:
            global_counts[al] = global_counts.get(al, 0) + 1
        print(f"  Distribution: {dict(sorted(global_counts.items()))}")
        
        # Execution Degree统计
        overall_avg_exec = np.mean(all_execution_degrees_global)
        overall_median_exec = np.median(all_execution_degrees_global)
        overall_std_exec = np.std(all_execution_degrees_global)
        overall_max_exec = np.max(all_execution_degrees_global)
        overall_min_exec = np.min(all_execution_degrees_global)
        
        print(f"\nGlobal Execution Degree Statistics:")
        print(f"  Total samples: {len(all_execution_degrees_global)}")
        print(f"  Mean: {overall_avg_exec:.2f}")
        print(f"  Median: {overall_median_exec:.2f}")
        print(f"  Std: {overall_std_exec:.2f}")
        print(f"  Min: {overall_min_exec}, Max: {overall_max_exec}")
        
        # 全局分布
        global_exec_counts = {}
        for ed in all_execution_degrees_global:
            global_exec_counts[ed] = global_exec_counts.get(ed, 0) + 1
        print(f"  Distribution: {dict(sorted(global_exec_counts.items()))}")
        
        # Similarity Score统计
        overall_avg_sim = np.mean(all_similarity_scores_global)
        overall_median_sim = np.median(all_similarity_scores_global)
        overall_std_sim = np.std(all_similarity_scores_global)
        overall_max_sim = np.max(all_similarity_scores_global)
        overall_min_sim = np.min(all_similarity_scores_global)
        
        print(f"\nGlobal Similarity Score Statistics:")
        print(f"  Total samples: {len(all_similarity_scores_global)}")
        print(f"  Mean: {overall_avg_sim:.4f}")
        print(f"  Median: {overall_median_sim:.4f}")
        print(f"  Std: {overall_std_sim:.4f}")
        print(f"  Min: {overall_min_sim:.4f}, Max: {overall_max_sim:.4f}")
        
        # Direction Consistency统计
        valid_dir_global = [d for d in all_direction_consistency_global if not np.isnan(d)]
        if len(valid_dir_global) > 0:
            overall_avg_dir = np.mean(valid_dir_global)
            overall_median_dir = np.median(valid_dir_global)
            overall_std_dir = np.std(valid_dir_global)
            overall_max_dir = np.max(valid_dir_global)
            overall_min_dir = np.min(valid_dir_global)
            
            print(f"\nGlobal Direction Consistency Statistics:")
            print(f"  Total samples: {len(valid_dir_global)} (valid) / {len(all_direction_consistency_global)} (total)")
            print(f"  Mean: {overall_avg_dir:.6f}")
            print(f"  Median: {overall_median_dir:.6f}")
            print(f"  Std: {overall_std_dir:.6f}")
            print(f"  Min: {overall_min_dir:.6f}, Max: {overall_max_dir:.6f}")
    
    print("="*80)
    
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("Overall Statistics:\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Total episodes: {total_episodes}\n")
    log_file.write(f"Total successes: {total_successes}\n")
    log_file.write(f"Success rate: {total_successes/total_episodes*100:.1f}%\n")
    
    if len(all_accept_lengths_global) > 0:
        log_file.write(f"\nGlobal Accept Length Statistics:\n")
        log_file.write(f"  Total samples: {len(all_accept_lengths_global)}\n")
        log_file.write(f"  Mean: {overall_avg_accept:.2f}\n")
        log_file.write(f"  Median: {overall_median_accept:.2f}\n")
        log_file.write(f"  Std: {overall_std_accept:.2f}\n")
        log_file.write(f"  Min: {overall_min_accept}, Max: {overall_max_accept}\n")
        log_file.write(f"  Distribution: {dict(sorted(global_counts.items()))}\n")
        
        log_file.write(f"\nGlobal Execution Degree Statistics:\n")
        log_file.write(f"  Total samples: {len(all_execution_degrees_global)}\n")
        log_file.write(f"  Mean: {overall_avg_exec:.2f}\n")
        log_file.write(f"  Median: {overall_median_exec:.2f}\n")
        log_file.write(f"  Std: {overall_std_exec:.2f}\n")
        log_file.write(f"  Min: {overall_min_exec}, Max: {overall_max_exec}\n")
        log_file.write(f"  Distribution: {dict(sorted(global_exec_counts.items()))}\n")
        
        log_file.write(f"\nGlobal Similarity Score Statistics:\n")
        log_file.write(f"  Total samples: {len(all_similarity_scores_global)}\n")
        log_file.write(f"  Mean: {overall_avg_sim:.4f}\n")
        log_file.write(f"  Median: {overall_median_sim:.4f}\n")
        log_file.write(f"  Std: {overall_std_sim:.4f}\n")
        log_file.write(f"  Min: {overall_min_sim:.4f}, Max: {overall_max_sim:.4f}\n")
    
    log_file.write("="*80 + "\n")
    
    # 保存全局accept_lengths
    global_npy_path = os.path.join(target_dir, f"all_tasks_{cfg.task_suite_name}.npy")
    np.save(global_npy_path, np.array(all_accept_lengths_global))
    print(f"\nGlobal accept_lengths saved to: {global_npy_path}")
    log_file.write(f"\nGlobal accept_lengths saved to: {global_npy_path}\n")
    
    # 保存全局execution_degrees
    global_exec_npy_path = os.path.join(target_dir, f"all_tasks_{cfg.task_suite_name}_exec_degree.npy")
    np.save(global_exec_npy_path, np.array(all_execution_degrees_global))
    print(f"Global execution_degrees saved to: {global_exec_npy_path}")
    log_file.write(f"Global execution_degrees saved to: {global_exec_npy_path}\n")
    
    # 保存全局similarity_scores
    global_sim_npy_path = os.path.join(target_dir, f"all_tasks_{cfg.task_suite_name}_similarity.npy")
    np.save(global_sim_npy_path, np.array(all_similarity_scores_global))
    print(f"Global similarity_scores saved to: {global_sim_npy_path}")
    log_file.write(f"Global similarity_scores saved to: {global_sim_npy_path}\n")
    
    # 保存全局direction_consistency
    global_dir_npy_path = os.path.join(target_dir, f"all_tasks_{cfg.task_suite_name}_direction.npy")
    np.save(global_dir_npy_path, np.array(all_direction_consistency_global))
    print(f"Global direction_consistency saved to: {global_dir_npy_path}")
    log_file.write(f"Global direction_consistency saved to: {global_dir_npy_path}\n")
    
    # Save local log file
    log_file.close()
    print(f"\nLog file saved to: {local_log_filepath}")


if __name__ == "__main__":
    eval_libero()
