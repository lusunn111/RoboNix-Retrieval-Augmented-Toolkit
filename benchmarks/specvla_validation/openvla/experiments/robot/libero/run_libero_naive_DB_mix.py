"""
run_libero_naive_DB_mix.py

Runs pure DB retrieval (no model) in LIBERO simulation environment.
Uses mix view embeddings (third-person + wrist camera) for retrieval.

Usage:
    python experiments/robot/libero/run_libero_naive_DB_mix.py \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 ] \
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
    num_trials_per_task: int = 10                    # Number of rollouts per task (default 10)

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

    # Initialize local logging
    target_dir = f"/path/to/SpecVLA/openvla/specdecoding/test-speed/{cfg.task_suite_name}_naive_DB_mix"
    os.makedirs(target_dir, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-NaiveDB-Mix-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    local_log_timefilepath = os.path.join(target_dir, run_id + "_naive_DB_mix.json")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
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
    print(f"Mode: Pure DB Retrieval - Mix View (Third-Person + Wrist)")
    log_file.write(f"Mode: Pure DB Retrieval - Mix View (Third-Person + Wrist)\n")
    print(f"Retrieval URL: {RETRIEVAL_URL}")
    log_file.write(f"Retrieval URL: {RETRIEVAL_URL}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

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
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            action_queue = []
            episode_db_times = []  # DB times for this episode
            episode_observations = []  # 当前episode的observations列表
            
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
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed images (both third-person and wrist views)
                    third_person_img = get_libero_image(obs, resize_size)
                    wrist_img = get_libero_wrist_image(obs, resize_size)

                    # Save preprocessed image for replay video (use third-person view)
                    replay_images.append(third_person_img)

                    # Prepare observations dict
                    observation = {
                        "full_image": third_person_img,
                        "wrist_image": wrist_img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    
                    # 保存observation到列表
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
                                    print(f"Retrieval failed: {result.get('error', 'Unknown error')}")
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
                                        print("No trajectory found in API response.")
                                        action = np.zeros(7) 
                                        action[-1] = -1.0
                            else:
                                print(f"API Failed with status {response.status_code}: {response.text}")
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
                    log_file.write(f"Caught exception: {e}\n")
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

            # Save a replay video of the episode
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
            log_file.flush()

        # Calculate and log timing statistics for this task
        num_db_calls = len(task_db_times)
        avg_db_time = np.mean(task_db_times) if num_db_calls > 0 else 0.0
        
        timing_msg = f"[Timing] task {task_id}: DB retrieval: {num_db_calls} calls, avg {avg_db_time:.6f}s | Total steps: {task_total_steps} ({task_episodes} episodes)"
        print(timing_msg)
        log_file.write(timing_msg + "\n")
        
        task_timing_stats.append({
            'task_id': task_id,
            'num_db_calls': num_db_calls,
            'avg_db_time': avg_db_time,
            'total_steps': task_total_steps,
            'num_episodes': task_episodes
        })
        
        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
    
    # Save timing data
    timing_data = {
        'db_times': task_db_times,
        'task_timing_stats': task_timing_stats
    }
    with open(local_log_timefilepath, mode='w') as f:
        json.dump(timing_data, f, indent=2)
    
    # 保存observations数据（包含轨迹状态）
    local_log_obspath = os.path.join(target_dir, run_id + "_observations.npy")
    np.save(local_log_obspath, observations_data, allow_pickle=True)
    print(f"Observations data saved to: {local_log_obspath}")
    print(f"  Data structure: observations_data[task_id][episode_idx] = {{'observations': list, 'success': bool, 'task_description': str, 'num_steps': int}}")
    log_file.write(f"Observations data saved to: {local_log_obspath}\n")
    
    # Print and log timing summary
    print("\n" + "="*80)
    print("Timing Summary (Mix View):")
    print("="*80)
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("Timing Summary (Mix View):\n")
    log_file.write("="*80 + "\n")
    
    total_db_calls = 0
    total_db_time_sum = 0.0
    total_steps_all = 0
    
    for stat in task_timing_stats:
        msg = f"[Timing] task {stat['task_id']}: DB retrieval: {stat['num_db_calls']} calls, avg {stat['avg_db_time']:.6f}s | Total steps: {stat['total_steps']} ({stat['num_episodes']} episodes)"
        print(msg)
        log_file.write(msg + "\n")
        total_db_calls += stat['num_db_calls']
        total_db_time_sum += stat['avg_db_time'] * stat['num_db_calls']
        total_steps_all += stat['total_steps']
    
    overall_avg_db_time = total_db_time_sum / total_db_calls if total_db_calls > 0 else 0.0
    
    print("="*80)
    print(f"Overall Statistics (Mix View - Third-Person + Wrist):")
    print(f"  DB Retrieval: {total_db_calls} calls, average time: {overall_avg_db_time:.6f}s")
    print(f"  Total steps: {total_steps_all}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Success rate: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)")
    print("="*80)
    print(f"\n检索统计信息 (混合视角):")
    print(f"  检索次数: {total_db_calls}")
    print(f"  检索平均每步时间: {overall_avg_db_time:.6f}s")
    print(f"  检索总时间: {total_db_time_sum:.6f}s")
    print("="*80 + "\n")
    
    log_file.write("="*80 + "\n")
    log_file.write(f"Overall Statistics (Mix View - Third-Person + Wrist):\n")
    log_file.write(f"  DB Retrieval: {total_db_calls} calls, average time: {overall_avg_db_time:.6f}s\n")
    log_file.write(f"  Total steps: {total_steps_all}\n")
    log_file.write(f"  Total episodes: {total_episodes}\n")
    log_file.write(f"  Success rate: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"\n检索统计信息 (混合视角):\n")
    log_file.write(f"  检索次数: {total_db_calls}\n")
    log_file.write(f"  检索平均每步时间: {overall_avg_db_time:.6f}s\n")
    log_file.write(f"  检索总时间: {total_db_time_sum:.6f}s\n")
    log_file.write("="*80 + "\n")
    
    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
