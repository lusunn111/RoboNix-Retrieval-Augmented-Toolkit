"""
run_libero_Spec_Exp_online_Memory.py

Experiment: Test the impact of runtime memory (successful trajectories) on success rate.

Workflow:
1. Warmup phase: Execute task until collecting N successful trajectories (insert to DB)
2. Test phase: Formally test with M trials (count success rate)

Usage:
    python experiments/robot/libero/run_libero_Spec_Exp_online_Memory.py \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --warmup_successes 10 \
        --num_trials_per_task 50
"""

import os
import sys
import time
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

# Append current directory
sys.path.append("/path/to/SpecVLA/openvla")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.online_rollout_inserter import (
    LiberoOnlineRolloutInserter,
    OnlineRolloutInsertConfig,
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
    accept_threshold: int = 9

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Union[str, Path] = "/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
    task_suite_name: str = "libero_goal"  # Options: libero_spatial, libero_object, libero_goal, libero_10
    num_steps_wait: int = 10
    
    #################################################################################################################
    # Experiment parameters (multi-stage warmup)
    #################################################################################################################
    warmup_stages: list = None                       # List of warmup targets [5, 10, 20, 30, 40, 50]
    num_trials_per_task: int = 50            # Number of test trials per task
    max_warmup_attempts_per_stage: int = 100         # Maximum attempts per warmup stage
    
    #################################################################################################################
    # Alternating execution (fixed 1:1 for this experiment)
    #################################################################################################################
    db_steps: int = 1                        # Fixed: 1 DB step
    model_steps: int = 1                     # Fixed: 1 Model step

    #################################################################################################################
    # Online DB insertion (always enabled for this experiment)
    #################################################################################################################
    online_db_dataset_name: str = "specvla_online"
    online_db_qdrant_url: str = "http://127.0.0.1:6333"
    online_db_embedding_server_url: str = "http://127.0.0.1:9020/predict"
    online_db_insert_stride: int = 1
    online_db_insert_max_steps: int = -1
    online_db_upsert_batch_size: int = 16
    online_db_request_timeout_s: float = 30.0
    online_db_upsert_wait: bool = True

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


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set default warmup stages if not provided
    if cfg.warmup_stages is None:
        cfg.warmup_stages = [5, 10, 20, 30, 40, 50]
    
    print(f"Multi-stage warmup experiment: {cfg.warmup_stages}")
    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Test trials per task per stage: {cfg.num_trials_per_task}")
    print("="*80)

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

    # Initialize online rollout inserter (always enabled)
    rollout_inserter = LiberoOnlineRolloutInserter(
        OnlineRolloutInsertConfig(
            embedding_server_url=cfg.online_db_embedding_server_url,
            qdrant_url=cfg.online_db_qdrant_url,
            request_timeout_s=cfg.online_db_request_timeout_s,
            upsert_wait=cfg.online_db_upsert_wait,
            upsert_batch_size=cfg.online_db_upsert_batch_size,
        )
    )

    # Initialize local logging (base directory)
    base_target_dir = f"/path/to/SpecVLA/openvla/specdecoding/test-speed/{cfg.task_suite_name}_Spec_Online_Memory_MultiStage"
    os.makedirs(base_target_dir, exist_ok=True)
    
    # Global log file for all stages
    run_id_base = f"EVAL-{cfg.task_suite_name}-SpecOnlineMem-MultiStage-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id_base += f"--{cfg.run_id_note}"
    global_log_filepath = os.path.join(base_target_dir, run_id_base + "_GLOBAL.txt")
    global_log_file = open(global_log_filepath, "w")
    
    print(f"Global log file: {global_log_filepath}")
    global_log_file.write(f"Multi-Stage Warmup Experiment\n")
    global_log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    global_log_file.write(f"Warmup stages: {cfg.warmup_stages}\n")
    global_log_file.write(f"Test trials per task: {cfg.num_trials_per_task}\n")
    global_log_file.write("="*80 + "\n")
    global_log_file.flush()

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

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Multi-stage warmup and testing
    all_stages_results = []
    
    # Calculate incremental warmup for each stage
    warmup_increments = []
    for i, target in enumerate(cfg.warmup_stages):
        if i == 0:
            increment = target  # First stage: 0 → 5
        else:
            increment = target - cfg.warmup_stages[i-1]  # Subsequent: 5→10, 10→20, etc.
        warmup_increments.append(increment)
    
    print(f"Warmup stages: {cfg.warmup_stages}")
    print(f"Incremental warmup per stage: {warmup_increments}")
    global_log_file.write(f"Warmup stages: {cfg.warmup_stages}\n")
    global_log_file.write(f"Incremental warmup: {warmup_increments}\n\n")
    
    for stage_idx, target_warmup in enumerate(cfg.warmup_stages):
        increment_warmup = warmup_increments[stage_idx]
        previous_warmup = cfg.warmup_stages[stage_idx - 1] if stage_idx > 0 else 0
        
        print(f"\n{'#'*80}")
        print(f"STAGE {stage_idx + 1}/{len(cfg.warmup_stages)}: {previous_warmup} → {target_warmup} (increment +{increment_warmup} per task)")
        print(f"{'#'*80}\n")
        global_log_file.write(f"\n{'#'*80}\n")
        global_log_file.write(f"STAGE {stage_idx + 1}/{len(cfg.warmup_stages)}: {previous_warmup} → {target_warmup} (increment +{increment_warmup})\n")
        global_log_file.write(f"{'#'*80}\n\n")
        global_log_file.flush()
        
        # Create stage-specific log
        stage_log_filepath = os.path.join(base_target_dir, f"{run_id_base}_Stage{stage_idx+1}_W{target_warmup}.txt")
        stage_log_file = open(stage_log_filepath, "w")
        stage_log_file.write(f"Stage {stage_idx + 1}: {previous_warmup} → {target_warmup} (increment +{increment_warmup} per task)\n")
        stage_log_file.write(f"Task suite: {cfg.task_suite_name}\n")
        stage_log_file.write(f"Test trials per task: {cfg.num_trials_per_task}\n")
        stage_log_file.write("="*80 + "\n")
        
        # Track if we need to reload retrieval service (only once per stage, after first task warmup)
        need_reload_after_warmup = True
        total_warmup_trajectories = 0  # Count total trajectories inserted in this stage
        
        # Start evaluation for this stage
        stage_test_episodes, stage_test_successes = 0, 0
        stage_timing_stats = []
        
        for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"Stage {stage_idx+1}"):
            # Get task
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

            print(f"\n{'='*80}")
            print(f"[Stage {stage_idx+1}] Task {task_id}: {task_description}")
            print(f"{'='*80}")
            stage_log_file.write(f"\n{'='*80}\n")
            stage_log_file.write(f"Task {task_id}: {task_description}\n")
            stage_log_file.write(f"{'='*80}\n")

            # ========================================
            # Phase 1: Warmup (每个任务增量 warmup increment_warmup 个成功轨迹)
            # ========================================
            warmup_db_times = []
            warmup_model_times = []
            task_warmup_successes = 0
            task_warmup_attempts = 0
            
            print(f"\n[WARMUP] Task {task_id}: Incremental warmup +{increment_warmup} trajectories ({previous_warmup} → {target_warmup})...")
            stage_log_file.write(f"\n[WARMUP] Task {task_id}: Incremental warmup +{increment_warmup} trajectories...\n")
            
            while task_warmup_successes < increment_warmup and task_warmup_attempts < cfg.max_warmup_attempts_per_stage:
                success = run_single_episode(
                    cfg, env, task_description, initial_states, task_warmup_attempts, model, processor,
                    resize_size, rollout_inserter, stage_log_file, phase="WARMUP",
                    db_times_list=warmup_db_times, model_times_list=warmup_model_times,
                    insert_on_success=True
                )
                
                if success:
                    task_warmup_successes += 1
                    total_warmup_trajectories += 1
                    print(f"  [WARMUP] Task {task_id}: {task_warmup_successes}/{increment_warmup} collected")
                    stage_log_file.write(f"  [WARMUP] Task {task_id}: {task_warmup_successes}/{increment_warmup} collected\n")
                    
                    # Check if this task reached increment target
                    if task_warmup_successes >= increment_warmup:
                        print(f"  [WARMUP] ✓ Task {task_id} reached increment target +{increment_warmup}!")
                        stage_log_file.write(f"  [WARMUP] ✓ Task {task_id} reached increment target!\n")
                        break
                
                task_warmup_attempts += 1
            
            print(f"  [WARMUP] Task {task_id} warmup complete: {task_warmup_successes} collected in {task_warmup_attempts} attempts")
            stage_log_file.write(f"  [WARMUP] Task {task_id}: {task_warmup_successes} collected in {task_warmup_attempts} attempts\n")
            
            # ========================================
            # CRITICAL: Reload retrieval service memory after first task warmup
            # Do this only once per stage (after first task completes warmup)
            # ========================================
            if need_reload_after_warmup and task_warmup_successes > 0:
                # First task in this stage completed warmup, trigger reload
                print(f"\n[RELOAD] First task warmup completed, signaling to reload retrieval service memory...")
                stage_log_file.write(f"\n[RELOAD] Creating reload marker for shell script...\n")
                
                # Create reload marker for shell script to detect
                reload_marker = os.path.join(base_target_dir, f".stage_{target_warmup}_reload_needed")
                with open(reload_marker, 'w') as f:
                    f.write(f"Stage {stage_idx+1} first task warmup complete, need to reload retrieval service\n")
                    f.write(f"Total trajectories inserted: {total_warmup_trajectories}\n")
                    f.write(f"Completed at task: {task_id}\n")
                
                print(f"  [RELOAD] Waiting for retrieval service reload...")
                stage_log_file.write(f"  [RELOAD] Waiting for service reload...\n")
                stage_log_file.flush()
                
                # Wait for shell script to complete reload (marked by .reload_complete file)
                reload_complete_marker = os.path.join(base_target_dir, f".stage_{target_warmup}_reload_complete")
                wait_count = 0
                max_wait = 300  # 5 minutes max
                
                while not os.path.exists(reload_complete_marker) and wait_count < max_wait:
                    time.sleep(1)
                    wait_count += 1
                    if wait_count % 10 == 0:
                        print(f"  [RELOAD] Waiting... ({wait_count}s)")
                
                if os.path.exists(reload_complete_marker):
                    print(f"  [RELOAD] ✓ Retrieval service reloaded successfully!")
                    stage_log_file.write(f"  [RELOAD] ✓ Service reloaded\n")
                    # Remove markers
                    os.remove(reload_marker)
                    os.remove(reload_complete_marker)
                    # Mark that we've already reloaded for this stage
                    need_reload_after_warmup = False
                else:
                    print(f"  [RELOAD] ✗ Timeout waiting for reload, continuing anyway...")
                    stage_log_file.write(f"  [RELOAD] ✗ Reload timeout\n")
                    need_reload_after_warmup = False
            
            # ========================================
            # Phase 2: Test - Formal evaluation
            # ========================================
            print(f"\n[TEST] Running {cfg.num_trials_per_task} test trials...")
            stage_log_file.write(f"\n[TEST] Running {cfg.num_trials_per_task} test trials...\n")
            
            task_test_episodes = 0
            task_test_successes = 0
            test_db_times = []
            test_model_times = []
            
            for trial_idx in range(cfg.num_trials_per_task):
                success = run_single_episode(
                    cfg, env, task_description, initial_states, trial_idx, model, processor,
                    resize_size, rollout_inserter, stage_log_file, phase="TEST",
                    db_times_list=test_db_times, model_times_list=test_model_times,
                    insert_on_success=False
                )
                
                if success:
                    task_test_successes += 1
                    stage_test_successes += 1
                
                task_test_episodes += 1
                stage_test_episodes += 1
                
                print(f"  [TEST] Trial {trial_idx+1}/{cfg.num_trials_per_task}: {'SUCCESS' if success else 'FAILED'} | Task: {task_test_successes}/{task_test_episodes} ({task_test_successes/task_test_episodes*100:.1f}%)")
                stage_log_file.write(f"  [TEST] Trial {trial_idx+1}: {'SUCCESS' if success else 'FAILED'} | {task_test_successes}/{task_test_episodes}\n")
                stage_log_file.flush()
            
            # Calculate timing statistics
            num_warmup_db = len(warmup_db_times)
            num_warmup_model = len(warmup_model_times)
            num_test_db = len(test_db_times)
            num_test_model = len(test_model_times)
            
            avg_warmup_db = np.mean(warmup_db_times) if num_warmup_db > 0 else 0.0
            avg_warmup_model = np.mean(warmup_model_times) if num_warmup_model > 0 else 0.0
            avg_test_db = np.mean(test_db_times) if num_test_db > 0 else 0.0
            avg_test_model = np.mean(test_model_times) if num_test_model > 0 else 0.0
            
            stage_timing_stats.append({
                'task_id': task_id,
                'task_description': task_description,
                'warmup_attempts': task_warmup_attempts,
                'warmup_successes': task_warmup_successes,
                'test_episodes': task_test_episodes,
                'test_successes': task_test_successes,
                'warmup_db_calls': num_warmup_db,
                'warmup_model_calls': num_warmup_model,
                'test_db_calls': num_test_db,
                'test_model_calls': num_test_model,
                'avg_warmup_db_time': avg_warmup_db,
                'avg_warmup_model_time': avg_warmup_model,
                'avg_test_db_time': avg_test_db,
                'avg_test_model_time': avg_test_model
            })
            
            print(f"Task {task_id} test success rate: {task_test_successes}/{task_test_episodes} ({task_test_successes/task_test_episodes*100:.1f}%)")
            stage_log_file.write(f"Task {task_id} test success rate: {task_test_successes}/{task_test_episodes}\n")
            stage_log_file.flush()
        
        # Save stage results
        stage_results = {
            'stage': stage_idx + 1,
            'previous_warmup': previous_warmup,
            'target_warmup': target_warmup,
            'increment_warmup': increment_warmup,
            'total_warmup_trajectories_inserted': total_warmup_trajectories,
            'test_episodes': stage_test_episodes,
            'test_successes': stage_test_successes,
            'success_rate': stage_test_successes / stage_test_episodes if stage_test_episodes > 0 else 0.0,
            'task_timing_stats': stage_timing_stats
        }
        all_stages_results.append(stage_results)
        
        # Save stage JSON
        stage_json_filepath = os.path.join(base_target_dir, f"{run_id_base}_Stage{stage_idx+1}_W{target_warmup}.json")
        with open(stage_json_filepath, 'w') as f:
            json.dump(stage_results, f, indent=2)
        
        # Print stage summary with detailed timing statistics
        stage_summary = f"\n{'='*80}\nSTAGE {stage_idx+1} SUMMARY ({previous_warmup} → {target_warmup}, increment +{increment_warmup}/task)\n{'='*80}\n"
        stage_summary += f"Incremental warmup trajectories inserted this stage: {total_warmup_trajectories}\n"
        stage_summary += f"Test episodes: {stage_test_episodes}\n"
        stage_summary += f"Test successes: {stage_test_successes}\n"
        stage_summary += f"Success rate: {stage_test_successes}/{stage_test_episodes} ({stage_test_successes/stage_test_episodes*100:.1f}%)\n"
        stage_summary += f"{'='*80}\n"
        
        print(stage_summary)
        stage_log_file.write(stage_summary)
        global_log_file.write(stage_summary)
        
        # Print detailed timing statistics for each task
        timing_section = f"\nTiming Statistics (Stage {stage_idx+1}):\n" + "="*80 + "\n"
        stage_log_file.write(timing_section)
        global_log_file.write(timing_section)
        print(timing_section)
        
        total_warmup_db = 0
        total_warmup_model = 0
        total_test_db = 0
        total_test_model = 0
        total_warmup_db_time = 0.0
        total_warmup_model_time = 0.0
        total_test_db_time = 0.0
        total_test_model_time = 0.0
        
        for stat in stage_timing_stats:
            warmup_msg = (
                f"[Task {stat['task_id']}] Warmup: "
                f"DB={stat['warmup_db_calls']} calls (avg {stat['avg_warmup_db_time']:.6f}s), "
                f"Model={stat['warmup_model_calls']} calls (avg {stat['avg_warmup_model_time']:.6f}s)"
            )
            test_msg = (
                f"[Task {stat['task_id']}] Test: "
                f"DB={stat['test_db_calls']} calls (avg {stat['avg_test_db_time']:.6f}s), "
                f"Model={stat['test_model_calls']} calls (avg {stat['avg_test_model_time']:.6f}s)"
            )
            print(warmup_msg)
            print(test_msg)
            stage_log_file.write(warmup_msg + "\n")
            stage_log_file.write(test_msg + "\n")
            global_log_file.write(warmup_msg + "\n")
            global_log_file.write(test_msg + "\n")
            
            total_warmup_db += stat['warmup_db_calls']
            total_warmup_model += stat['warmup_model_calls']
            total_test_db += stat['test_db_calls']
            total_test_model += stat['test_model_calls']
            total_warmup_db_time += stat['avg_warmup_db_time'] * stat['warmup_db_calls']
            total_warmup_model_time += stat['avg_warmup_model_time'] * stat['warmup_model_calls']
            total_test_db_time += stat['avg_test_db_time'] * stat['test_db_calls']
            total_test_model_time += stat['avg_test_model_time'] * stat['test_model_calls']
        
        overall_warmup_db = total_warmup_db_time / total_warmup_db if total_warmup_db > 0 else 0.0
        overall_warmup_model = total_warmup_model_time / total_warmup_model if total_warmup_model > 0 else 0.0
        overall_test_db = total_test_db_time / total_test_db if total_test_db > 0 else 0.0
        overall_test_model = total_test_model_time / total_test_model if total_test_model > 0 else 0.0
        
        overall_stats = "\n" + "="*80 + "\n"
        overall_stats += f"Overall Statistics (Stage {stage_idx+1}):\n"
        overall_stats += f"  Warmup Phase:\n"
        overall_stats += f"    DB Retrieval: {total_warmup_db} calls, average time: {overall_warmup_db:.6f}s\n"
        overall_stats += f"    Model (Spec): {total_warmup_model} calls, average time: {overall_warmup_model:.6f}s\n"
        overall_stats += f"  Test Phase:\n"
        overall_stats += f"    DB Retrieval: {total_test_db} calls, average time: {overall_test_db:.6f}s\n"
        overall_stats += f"    Model (Spec): {total_test_model} calls, average time: {overall_test_model:.6f}s\n"
        overall_stats += f"  Total steps (Warmup+Test): {total_warmup_db+total_warmup_model+total_test_db+total_test_model}\n"
        overall_stats += "="*80 + "\n"
        
        print(overall_stats)
        stage_log_file.write(overall_stats)
        global_log_file.write(overall_stats)
        global_log_file.flush()
        
        stage_log_file.close()
        
        # Signal to shell script that this stage is complete (for backup)
        stage_complete_marker = os.path.join(base_target_dir, f".stage_{target_warmup}_complete")
        with open(stage_complete_marker, 'w') as f:
            f.write(f"Stage {stage_idx+1} with warmup={target_warmup} completed\n")
        
        print(f"\n[INFO] Stage {stage_idx+1} complete. Marker created: {stage_complete_marker}")
        print(f"[INFO] Shell script can now backup database as 'base+{target_warmup}'\n")
    
    # Save final summary
    final_summary_filepath = os.path.join(base_target_dir, f"{run_id_base}_FINAL_SUMMARY.json")
    final_summary = {
        'task_suite': cfg.task_suite_name,
        'warmup_stages': cfg.warmup_stages,
        'test_trials_per_task': cfg.num_trials_per_task,
        'stages_results': all_stages_results
    }
    with open(final_summary_filepath, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    # Print final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON ACROSS ALL STAGES")
    print("="*80)
    global_log_file.write("\n" + "="*80 + "\n")
    global_log_file.write("FINAL COMPARISON ACROSS ALL STAGES\n")
    global_log_file.write("="*80 + "\n")
    
    for result in all_stages_results:
        line = f"Stage {result['stage']}: Warmup={result['previous_warmup']:2d}→{result['target_warmup']:2d} (+{result['increment_warmup']:2d}/task, Total inserted={result['total_warmup_trajectories_inserted']:3d}): Success Rate = {result['test_successes']:3d}/{result['test_episodes']:3d} ({result['success_rate']*100:5.1f}%)"
        print(line)
        global_log_file.write(line + "\n")
    
    print("="*80)
    global_log_file.write("="*80 + "\n")
    global_log_file.close()
    
    print(f"\nAll results saved to: {base_target_dir}")
    print(f"Final summary: {final_summary_filepath}")


def run_single_episode(cfg, env, task_description, initial_states, episode_idx, model, processor,
                       resize_size, rollout_inserter, log_file, phase="TEST",
                       db_times_list=None, model_times_list=None, insert_on_success=False):
    """Run a single episode and return success status"""
    
    # Reset environment
    env.reset()
    obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
    
    # Alternating state
    use_db_mode = False  # Start with model
    db_step_count = 0
    model_step_count = 0
    action_queue = []
    
    # Episode data
    t = 0
    replay_images = []
    episode_actions = []
    
    # Max steps
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
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            
            img = get_libero_image(obs, resize_size)
            replay_images.append(img)
            
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            
            # Alternating execution logic (1:1)
            if use_db_mode:
                # DB Retrieval Mode
                if len(action_queue) > 0:
                    action = action_queue.pop(0)
                else:
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
                        
                        t0_req = time_module.time()
                        response = requests.post(RETRIEVAL_URL, files=files, data=data, timeout=30)
                        t1_req = time_module.time()
                        db_time = t1_req - t0_req
                        if db_times_list is not None:
                            db_times_list.append(db_time)
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get('success', False):
                                retrieved_traj = None
                                if 'rtcache_trajectory' in result and result['rtcache_trajectory']:
                                    retrieved_traj = np.array(result['rtcache_trajectory'])
                                elif 'averaged_trajectory' in result and result['averaged_trajectory']:
                                    retrieved_traj = np.array(result['averaged_trajectory'])
                                
                                if retrieved_traj is not None and len(retrieved_traj) > 0:
                                    if retrieved_traj.ndim == 1:
                                        action_queue = [retrieved_traj]
                                    else:
                                        action_queue = [a for a in retrieved_traj[:1]]
                                    action = action_queue.pop(0)
                                else:
                                    action = np.zeros(7)
                                    action[-1] = -1.0
                            else:
                                action = np.zeros(7)
                                action[-1] = -1.0
                        else:
                            action = np.zeros(7)
                            action[-1] = -1.0
                    except Exception as e:
                        action = np.zeros(7)
                        action[-1] = -1.0
                
                db_step_count += 1
                if db_step_count >= cfg.db_steps:
                    use_db_mode = False if cfg.model_steps > 0 else True
                    db_step_count = 0
                
                action = normalize_gripper_action(action, binarize=True)
            
            else:
                # Model Generation Mode
                t0_model = time_module.time()
                action = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    generate_mode='speculative'
                )
                t1_model = time_module.time()
                model_time = t1_model - t0_model
                if model_times_list is not None:
                    model_times_list.append(model_time)
                
                model_step_count += 1
                if model_step_count >= cfg.model_steps:
                    use_db_mode = True if cfg.db_steps > 0 else False
                    model_step_count = 0
                
                action = normalize_gripper_action(action, binarize=True)
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)
            
            # Store action for potential DB insertion
            episode_actions.append(action.copy())
            
            # Execute action
            obs, reward, done, info = env.step(action.tolist())
            if done:
                # Episode succeeded
                if insert_on_success and rollout_inserter is not None:
                    # Insert successful trajectory to DB
                    try:
                        max_steps_cfg = None if cfg.online_db_insert_max_steps < 0 else cfg.online_db_insert_max_steps
                        insert_result = rollout_inserter.insert_trajectory(
                            replay_images,
                            episode_actions,
                            task_description,
                            cfg.task_suite_name,
                            dataset_name=cfg.online_db_dataset_name,
                            episode_idx=episode_idx,
                            stride=cfg.online_db_insert_stride,
                            max_steps=max_steps_cfg,
                        )
                        print(f"[OnlineInsert] {insert_result}")
                        log_file.write(f"[OnlineInsert] {insert_result}\n")
                    except Exception as e:
                        print(f"[WARNING] Failed to insert trajectory to DB: {e}")
                        log_file.write(f"[WARNING] Failed to insert trajectory to DB: {e}\n")
                
                return True
            
            t += 1
        
        except Exception as e:
            print(f"[{phase}] Caught exception: {e}")
            log_file.write(f"[{phase}] Caught exception: {e}\n")
            return False
    
    return False


if __name__ == "__main__":
    eval_libero()
