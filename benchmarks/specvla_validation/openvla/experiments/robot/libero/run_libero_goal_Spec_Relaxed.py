"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
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
import time as time_module
from collections import defaultdict

# Append current directory so that interpreter can find experiments.robot
# sys.path is set by PYTHONPATH environment variable
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

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    use_spec: bool = True
    parallel_draft: bool = False
    accept_threshold: int = 9
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Union[str, Path] = "PATH_TO_SPECVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/STATE_ID"
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task

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
    target_dir = "TGT_DIR"
    os.makedirs(target_dir,exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    local_log_timefilepath = os.path.join(target_dir, run_id + "libero_goal.json")
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

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_timing_stats = []  # Store timing stats for each task
    
    # Detailed timing breakdown storage
    detailed_timing_data = {
        'drafter': [],
        'tokenizer': [],
        'vit': [],
        'llm': [],
        'detokenizer': [],
        'total': [],
        'verification': []  # For speculative decoding verification
    }
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
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            total_time = []
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

                    # Query model to get action with detailed timing
                    t0 = time_module.time()
                    
                    # get_action 返回 (action, timing_dict)
                    result = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        generate_mode='speculative',
                        return_detailed_timing=True
                    )
                    
                    t1 = time_module.time()
                    time = t1 - t0
                    
                    # 处理返回值：可能是 (action, timing_dict) 或只是 action
                    if isinstance(result, tuple) and len(result) == 2:
                        action, timing_dict = result
                        # 收集详细计时信息
                        detailed_timing_data['drafter'].append(timing_dict.get('drafter', 0))
                        detailed_timing_data['tokenizer'].append(timing_dict.get('tokenizer', 0))
                        detailed_timing_data['vit'].append(timing_dict.get('vit', 0))
                        detailed_timing_data['llm'].append(timing_dict.get('llm', 0))
                        detailed_timing_data['detokenizer'].append(timing_dict.get('detokenizer', 0))
                        detailed_timing_data['verification'].append(timing_dict.get('verification', 0))
                        detailed_timing_data['total'].append(time)
                    else:
                        action = result
                        detailed_timing_data['total'].append(time)
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    total_time.append(time)

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break
            #exit()
            task_episodes += 1
            total_episodes += 1
            total_episode_time.append(total_time)
            task_total_steps += len(total_time)  # Add steps from this episode

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
        # Flatten all times and ensure they are scalars
        all_times = []
        for episode in total_episode_time:
            for t in episode:
                # Handle both scalar and array-like time values
                if isinstance(t, (list, np.ndarray)):
                    all_times.extend(np.atleast_1d(t).flatten().tolist())
                else:
                    all_times.append(float(t))
        
        if len(all_times) > 0:
            mean_latency = np.mean(all_times)
            timing_msg = f"[Timing] task {task_id}: mean step latency {mean_latency:.6f}s over {task_total_steps} steps ({task_episodes} episodes)"
            print(timing_msg)
            log_file.write(timing_msg + "\n")
            task_timing_stats.append({
                'task_id': task_id,
                'mean_latency': mean_latency,
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
        #exit()
    with open(local_log_timefilepath,mode='w') as f:
        json.dump(total_episode_time,f)
    
    # Save detailed timing breakdown
    detailed_timing_filepath = os.path.join(target_dir, run_id + "_detailed_timing.json")
    with open(detailed_timing_filepath, mode='w') as f:
        json.dump(detailed_timing_data, f, indent=2)
    
    # Print and log timing summary
    print("\n" + "="*60)
    print("Timing Summary:")
    print("="*60)
    log_file.write("\n" + "="*60 + "\n")
    log_file.write("Timing Summary:\n")
    log_file.write("="*60 + "\n")
    
    total_steps_all = 0
    total_latency_weighted = 0.0
    for stat in task_timing_stats:
        msg = f"[Timing] task {stat['task_id']}: mean step latency {stat['mean_latency']:.6f}s over {stat['total_steps']} steps ({stat['num_episodes']} episodes)"
        print(msg)
        log_file.write(msg + "\n")
        total_steps_all += stat['total_steps']
        total_latency_weighted += stat['mean_latency'] * stat['total_steps']
    
    if total_steps_all > 0:
        overall_avg_latency = total_latency_weighted / total_steps_all
        print("="*60)
        print(f"Overall average latency: {overall_avg_latency:.6f}s")
        print(f"Total steps: {total_steps_all}")
        print(f"Total episodes: {total_episodes}")
        print("="*60 + "\n")
        log_file.write("="*60 + "\n")
        log_file.write(f"Overall average latency: {overall_avg_latency:.6f}s\n")
        log_file.write(f"Total steps: {total_steps_all}\n")
        log_file.write(f"Total episodes: {total_episodes}\n")
        log_file.write("="*60 + "\n")
    
    # Print detailed timing breakdown
    print("\n" + "="*60)
    print("Detailed Timing Breakdown:")
    print("="*60)
    log_file.write("\n" + "="*60 + "\n")
    log_file.write("Detailed Timing Breakdown:\n")
    log_file.write("="*60 + "\n")
    
    component_names = ['drafter', 'tokenizer', 'vit', 'llm', 'detokenizer', 'verification', 'total']
    component_labels = ['Drafter (Small Model)', 'Tokenizer', 'ViT', 'LLM', 'De-Tokenizer', 'Verification', 'Total']
    
    for comp_name, comp_label in zip(component_names, component_labels):
        times = detailed_timing_data[comp_name]
        if len(times) > 0:
            times_array = np.array(times)
            mean_time = np.mean(times_array)
            std_time = np.std(times_array)
            min_time = np.min(times_array)
            max_time = np.max(times_array)
            total_time = np.sum(times_array)
            
            msg = f"{comp_label:20s}: mean={mean_time:.6f}s, std={std_time:.6f}s, min={min_time:.6f}s, max={max_time:.6f}s, total={total_time:.3f}s, count={len(times)}"
            print(msg)
            log_file.write(msg + "\n")
            
            # Calculate percentage of total time (if total is available)
            if comp_name != 'total' and len(detailed_timing_data['total']) > 0:
                total_mean = np.mean(detailed_timing_data['total'])
                if total_mean > 0:
                    percentage = (mean_time / total_mean) * 100
                    pct_msg = f"{'':20s}  └─ {percentage:.2f}% of total time"
                    print(pct_msg)
                    log_file.write(pct_msg + "\n")
    
    print("="*60)
    log_file.write("="*60 + "\n")
    
    # Save local log file
    log_file.close()
   # print('total time')
   # print(sum([sum(item) for item in total_episode_time]))

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