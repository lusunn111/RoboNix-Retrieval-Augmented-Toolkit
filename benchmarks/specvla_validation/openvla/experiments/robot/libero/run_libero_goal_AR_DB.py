"""
run_libero_goal_AR_DB.py

Runs a model in a LIBERO simulation environment with alternating DB retrieval and AR generation.

Usage:
    python experiments/robot/libero/run_libero_goal_AR_DB.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --db_steps <N> \
        --model_steps <M> \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
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

RETRIEVAL_URL = "http://127.0.0.1:5002/pipeline"

# Append current directory so that interpreter can find experiments.robot
#sys.path.append("../..")
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


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "PATH_TO_SPECVLA/backbone_models/openvla-7b-finetuned-libero-goal"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    parallel_draft: bool = False
    accept_threshold: int = None
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    use_spec: bool = True
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Union[str, Path] = "PATH_TO_SPECVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/STATE_ID"
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    
    #################################################################################################################
    # Alternating execution parameters
    #################################################################################################################
    db_steps: int = 1                                # Number of consecutive DB retrieval actions
    model_steps: int = 1                              # Number of consecutive model generation actions

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


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
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    target_dir = f"PATH_TO_SPECVLA/openvla/specdecoding/test-speed/libero_goal_AR_DB_N{cfg.db_steps}_M{cfg.model_steps}"
    os.makedirs(target_dir, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    local_log_timefilepath = os.path.join(target_dir, run_id + "libero_Goal_AR_DB.json")
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
    print(f"Alternating mode: DB steps={cfg.db_steps}, Model steps={cfg.model_steps}")
    log_file.write(f"Alternating mode: DB steps={cfg.db_steps}, Model steps={cfg.model_steps}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_timing_stats = []  # Store timing stats for each task
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        total_episode_time = []
        task_total_steps = 0  # Track total steps for this task
        task_db_times = []  # Track DB retrieval times
        task_model_times = []  # Track model generation times
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            # Initialize alternating mode state
            # If model_steps is 0, start with DB mode; if db_steps is 0, start with model mode
            if cfg.model_steps == 0:
                use_db_mode = True  # Only use DB if model_steps=0
            elif cfg.db_steps == 0:
                use_db_mode = False  # Only use model if db_steps=0
            else:
                use_db_mode = False  # Start with model when both are enabled
            db_step_count = 0
            model_step_count = 0
            action_queue = []
            episode_db_times = []  # DB times for this episode
            episode_model_times = []  # Model times for this episode
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

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Alternating execution logic
                    if use_db_mode:
                        # ========================================
                        # DB Retrieval Mode - ONLY DB, NO AR
                        # ========================================
                        if len(action_queue) > 0:
                            # Use cached action from previous DB call, no timing needed
                            action = action_queue.pop(0)
                        else:
                            # Call DB Retrieval API - ONLY record API call time
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
                                
                                # ONLY record time for DB API call
                                t0_req = time_module.time()
                                response = requests.post(RETRIEVAL_URL, files=files, data=data, timeout=30)
                                t1_req = time_module.time()
                                db_time = t1_req - t0_req
                                episode_db_times.append(db_time)  # Record DB time
                                
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
                                            # Store trajectory in queue (first 2 actions as in DBcopy)
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
                        
                        # Update DB step counter and switch mode if needed
                        db_step_count += 1
                        if db_step_count >= cfg.db_steps:
                            # Only switch to model mode if model_steps > 0
                            use_db_mode = False if cfg.model_steps > 0 else True
                            db_step_count = 0
                        
                        # Note: DB actions don't need gripper inversion (as per DBcopy)
                        # Normalize gripper action [0,1] -> [-1,+1]
                        action = normalize_gripper_action(action, binarize=True)
                        
                        # ASSERT: In DB mode, we should NEVER call get_action (AR)
                        # DB mode is complete, proceed to execute action
                        
                    else:
                        # ========================================
                        # Model Generation Mode - ONLY AR, NO DB
                        # ========================================
                        if cfg.model_steps == 0:
                            # This should never happen now, but keep as safety check
                            raise RuntimeError("Entered model mode with model_steps=0! This is a logic error.")
                        
                        # ONLY record time for get_action (AR inference)
                        t0_model = time_module.time()
                        action = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            generate_mode='AR'
                        )
                        t1_model = time_module.time()
                        model_time = t1_model - t0_model
                        episode_model_times.append(model_time)  # Record model time
                        
                        # Update model step counter and switch mode if needed
                        model_step_count += 1
                        if model_step_count >= cfg.model_steps:
                            # Only switch to DB mode if db_steps > 0
                            use_db_mode = True if cfg.db_steps > 0 else False
                            model_step_count = 0
                        
                        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                        action = normalize_gripper_action(action, binarize=True)
                        
                        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)
                        
                        # ASSERT: In model mode, we should NEVER call DB API
                        # Model mode is complete, proceed to execute action

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    # Note: We don't append to total_time anymore since we track DB and model separately

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            # Remove total_episode_time since we track DB and model separately
            task_total_steps += len(episode_db_times) + len(episode_model_times)  # Total steps = DB calls + Model calls
            task_db_times.extend(episode_db_times)  # Add DB times
            task_model_times.extend(episode_model_times)  # Add model times

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
        # Separate statistics for DB retrieval and model generation
        num_db_calls = len(task_db_times)
        num_model_calls = len(task_model_times)
        avg_db_time = np.mean(task_db_times) if num_db_calls > 0 else 0.0
        avg_model_time = np.mean(task_model_times) if num_model_calls > 0 else 0.0
        
        timing_msg = f"[Timing] task {task_id}: DB retrieval: {num_db_calls} calls, avg {avg_db_time:.6f}s | Model (AR): {num_model_calls} calls, avg {avg_model_time:.6f}s | Total steps: {task_total_steps} ({task_episodes} episodes)"
        print(timing_msg)
        log_file.write(timing_msg + "\n")
        
        task_timing_stats.append({
            'task_id': task_id,
            'num_db_calls': num_db_calls,
            'avg_db_time': avg_db_time,
            'num_model_calls': num_model_calls,
            'avg_model_time': avg_model_time,
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
    # Save timing data separately for DB and Model
    timing_data = {
        'db_times': task_db_times,
        'model_times': task_model_times,
        'task_timing_stats': task_timing_stats
    }
    with open(local_log_timefilepath, mode='w') as f:
        json.dump(timing_data, f, indent=2)    
    # Print and log timing summary
    print("\n" + "="*80)
    print("Timing Summary:")
    print("="*80)
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("Timing Summary:\n")
    log_file.write("="*80 + "\n")
    
    total_db_calls = 0
    total_model_calls = 0
    total_db_time_sum = 0.0
    total_model_time_sum = 0.0
    total_steps_all = 0
    
    for stat in task_timing_stats:
        msg = f"[Timing] task {stat['task_id']}: DB retrieval: {stat['num_db_calls']} calls, avg {stat['avg_db_time']:.6f}s | Model (AR): {stat['num_model_calls']} calls, avg {stat['avg_model_time']:.6f}s | Total steps: {stat['total_steps']} ({stat['num_episodes']} episodes)"
        print(msg)
        log_file.write(msg + "\n")
        total_db_calls += stat['num_db_calls']
        total_model_calls += stat['num_model_calls']
        total_db_time_sum += stat['avg_db_time'] * stat['num_db_calls']
        total_model_time_sum += stat['avg_model_time'] * stat['num_model_calls']
        total_steps_all += stat['total_steps']
    
    overall_avg_db_time = total_db_time_sum / total_db_calls if total_db_calls > 0 else 0.0
    overall_avg_model_time = total_model_time_sum / total_model_calls if total_model_calls > 0 else 0.0
    
    print("="*80)
    print(f"Overall Statistics:")
    print(f"  DB Retrieval: {total_db_calls} calls, average time: {overall_avg_db_time:.6f}s")
    print(f"  Model (AR): {total_model_calls} calls, average time: {overall_avg_model_time:.6f}s")
    print(f"  Total steps: {total_steps_all}")
    print(f"  Total episodes: {total_episodes}")
    print("="*80)
    print(f"\n检索统计信息:")
    print(f"  检索次数: {total_db_calls}")
    print(f"  检索平均每步时间: {overall_avg_db_time:.6f}s")
    print(f"  检索总时间: {total_db_time_sum:.6f}s")
    print("="*80 + "\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Overall Statistics:\n")
    log_file.write(f"  DB Retrieval: {total_db_calls} calls, average time: {overall_avg_db_time:.6f}s\n")
    log_file.write(f"  Model (AR): {total_model_calls} calls, average time: {overall_avg_model_time:.6f}s\n")
    log_file.write(f"  Total steps: {total_steps_all}\n")
    log_file.write(f"  Total episodes: {total_episodes}\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"\n检索统计信息:\n")
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

