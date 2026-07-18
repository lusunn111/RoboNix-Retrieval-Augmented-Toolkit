"""
run_libero_naive_DB_mix_multy_stage.py

Experiment: Test the impact of different database states (base+10, base+20, ..., base+80) 
on pure DB retrieval success rate using mix view embeddings (third-person + wrist camera).

Usage:
    python experiments/robot/libero/run_libero_naive_DB_mix_multy_stage.py \
        --task_suite_name libero_goal \
        --db_state_name base+10 \
        --num_trials_per_task 10
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

# Mix retrieval server URL (port 5003 for mix view)
RETRIEVAL_URL = "http://127.0.0.1:5003/pipeline"

# Append current directory so that interpreter can find experiments.robot
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters (not used for pure DB)
    #################################################################################################################
    model_family: str = "openvla"                    # Model family (kept for compatibility)
    pretrained_checkpoint: Union[str, Path] = "none" # Not used for pure DB
    center_crop: bool = True                         # Center crop

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"             # Task suite: libero_spatial, libero_object, libero_goal, libero_10
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task
    
    #################################################################################################################
    # Database state parameter (NEW)
    #################################################################################################################
    db_state_name: str = "base"                      # Database state name: base, base+10, base+20, ..., base+80
    stage_index: int = 1                             # Stage index (1-8) for logging purposes

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    # Set random seed
    set_seed_everywhere(cfg.seed)

    print(f"\n{'='*80}")
    print(f"Pure DB Retrieval Experiment (Mix View) - Database State: {cfg.db_state_name}")
    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Trials per task: {cfg.num_trials_per_task}")
    print(f"Retrieval URL: {RETRIEVAL_URL}")
    print(f"{'='*80}\n")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Initialize local logging directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "../../../specdecoding/test-speed")
    target_dir = os.path.join(base_dir, "libero_naive_DB_mix_multy_stage")
    os.makedirs(target_dir, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.db_state_name}-MixView-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_timing_stats = []  # Store timing stats for each task
    observations_data = {}  # 存储所有observations: task_id -> episode_idx -> {'observations': list, 'success': bool, ...}
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_total_steps = 0  # Track total steps for this task
        task_db_times = []  # Track DB retrieval times
        observations_data[task_id] = {}  # 初始化当前任务的observations字典
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task {task_id}"):
            action_queue = []
            episode_db_times = []  # DB times for this episode
            episode_observations = []  # 当前episode的observations列表
            
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
            else:
                max_steps = 300
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed images (both third-person and wrist views)
                    third_person_img = get_libero_image(obs, resize_size)
                    wrist_img = get_libero_wrist_image(obs, resize_size)

                    # Save preprocessed image for replay video (use third-person view)
                    replay_images.append(third_person_img)

                    # Prepare and save observation
                    observation = {
                        "full_image": third_person_img,
                        "wrist_image": wrist_img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    episode_observations.append(observation)

                    # Pure DB Retrieval Mode (Mix View)
                    if len(action_queue) > 0:
                        # Use cached action from previous DB call
                        action = action_queue.pop(0)
                    else:
                        # Call Mix DB Retrieval API
                        try:
                            # Prepare third-person image
                            pil_third_person = Image.fromarray(third_person_img)
                            buf_third_person = BytesIO()
                            pil_third_person.save(buf_third_person, format='PNG')
                            buf_third_person.seek(0)
                            
                            # Prepare wrist image
                            pil_wrist = Image.fromarray(wrist_img)
                            buf_wrist = BytesIO()
                            pil_wrist.save(buf_wrist, format='PNG')
                            buf_wrist.seek(0)
                            
                            # Prepare request with two images
                            files = {
                                "third_person_image": ("third_person.png", buf_third_person, "image/png"),
                                "wrist_image": ("wrist.png", buf_wrist, "image/png")
                            }
                            data = {
                                "instruction": task_description,
                                "dataset_type": cfg.task_suite_name
                            }
                            
                            # Record time for DB API call
                            t0_req = time_module.time()
                            response = requests.post(RETRIEVAL_URL, files=files, data=data, timeout=30)
                            t1_req = time_module.time()
                            db_time = t1_req - t0_req
                            episode_db_times.append(db_time)
                            
                            if response.status_code == 200:
                                result = response.json()
                                # Check if retrieval was successful
                                if not result.get('success', False):
                                    action = np.zeros(7) 
                                    action[-1] = -1.0
                                else:
                                    # Try to get trajectory from result
                                    retrieved_traj = None
                                    if 'rtcache_trajectory' in result and result['rtcache_trajectory']:
                                        retrieved_traj = np.array(result['rtcache_trajectory'])
                                    elif 'averaged_trajectory' in result and result['averaged_trajectory']:
                                        retrieved_traj = np.array(result['averaged_trajectory'])
                                    
                                    if retrieved_traj is not None and len(retrieved_traj) > 0:
                                        # Store trajectory in queue (first 2 actions)
                                        if retrieved_traj.ndim == 1:
                                            action_queue = [retrieved_traj]
                                        else:
                                            action_queue = [a for a in retrieved_traj[:2]]
                                        
                                        # Pop first action
                                        action = action_queue.pop(0)
                                    else:
                                        action = np.zeros(7) 
                                        action[-1] = -1.0
                            else:
                                action = np.zeros(7)
                                action[-1] = -1.0
                        except Exception as e:
                            print(f"API Error: {e}")
                            action = np.zeros(7)
                            action[-1] = -1.0

                    # Normalize gripper action [0,1] -> [-1,+1]
                    action = normalize_gripper_action(action, binarize=True)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    break

            # 将episode数据存入observations_data，包含轨迹状态
            episode_success = done if 'done' in dir() else False
            observations_data[task_id][episode_idx] = {
                'observations': episode_observations,
                'success': episode_success,
                'task_description': task_description,
                'num_steps': len(episode_observations)
            }
            
            task_episodes += 1
            total_episodes += 1
            task_total_steps += len(episode_db_times)
            task_db_times.extend(episode_db_times)

        # Calculate timing statistics for this task
        num_db_calls = len(task_db_times)
        avg_db_time = np.mean(task_db_times) if num_db_calls > 0 else 0.0
        
        task_timing_stats.append({
            'task_id': task_id,
            'num_db_calls': num_db_calls,
            'avg_db_time': avg_db_time,
            'total_steps': task_total_steps,
            'num_episodes': task_episodes,
            'num_successes': task_successes
        })
        
        print(f"[Task {task_id}] Success: {task_successes}/{task_episodes} ({task_successes/task_episodes*100:.1f}%) | DB calls: {num_db_calls}, avg: {avg_db_time:.6f}s")

    # 保存observations数据（包含轨迹状态）
    local_log_obspath = os.path.join(target_dir, run_id + "_observations.npy")
    np.save(local_log_obspath, observations_data, allow_pickle=True)
    print(f"Observations data saved to: {local_log_obspath}")
    print(f"  Data structure: observations_data[task_id][episode_idx] = {{'observations': list, 'success': bool, 'task_description': str, 'num_steps': int}}")

    # Return results as a dictionary for shell script to use
    results = {
        'db_state_name': cfg.db_state_name,
        'stage_index': cfg.stage_index,
        'task_suite': cfg.task_suite_name,
        'num_trials_per_task': cfg.num_trials_per_task,
        'total_episodes': total_episodes,
        'total_successes': total_successes,
        'success_rate': total_successes / total_episodes if total_episodes > 0 else 0.0,
        'task_timing_stats': task_timing_stats,
        'overall_avg_db_time': np.mean([stat['avg_db_time'] for stat in task_timing_stats]) if task_timing_stats else 0.0,
        'total_db_calls': sum([stat['num_db_calls'] for stat in task_timing_stats])
    }
    
    return results


if __name__ == "__main__":
    results = eval_libero()
    
    # Print final summary
    print("\n" + "="*80)
    print(f"Database State: {results['db_state_name']} (Mix View) - Final Results")
    print("="*80)
    print(f"Success Rate: {results['total_successes']}/{results['total_episodes']} ({results['success_rate']*100:.1f}%)")
    print(f"Total DB Calls: {results['total_db_calls']}")
    print(f"Average DB Retrieval Time: {results['overall_avg_db_time']:.6f}s")
    print("="*80 + "\n")
