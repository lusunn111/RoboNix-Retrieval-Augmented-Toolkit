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
import requests
from io import BytesIO
from PIL import Image
import time as time_module

RETRIEVAL_URL = "http://127.0.0.1:5002/pipeline"

# Append current directory so that interpreter can find experiments.robot
# Get the script directory and add openvla to path
script_dir = os.path.dirname(os.path.abspath(__file__))
openvla_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if openvla_dir not in sys.path:
    sys.path.insert(0, openvla_dir)
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
    track_accept_length: bool = False               # Track DB retrieval action slice accept length
    use_db_retrieval: bool = False                  # Use DB retrieval for action slices
    retrieval_url: str = "http://127.0.0.1:5002/pipeline"  # DB retrieval API URL
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Union[str, Path] = "PATH_TO_SPECVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                    # Number of rollouts per task

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
    if "PATH_TO_SPECVLA" in str(cfg.pretrained_checkpoint):
        print(f"WARNING: Using placeholder path for pretrained_checkpoint: {cfg.pretrained_checkpoint}")
        print("Please set correct pretrained_checkpoint path!")
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
    # Get the SpecVLA directory (assuming script is run from SpecVLA/openvla or similar)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to find SpecVLA root - try multiple possible paths
    possible_roots = [
        os.path.abspath(os.path.join(script_dir, "..", "..", "..")),  # from experiments/robot/libero
        os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..")),  # alternative
    ]
    specvla_root = None
    for root in possible_roots:
        if os.path.exists(os.path.join(root, "exp")) or os.path.exists(os.path.join(root, "openvla")):
            specvla_root = root
            break
    if specvla_root is None:
        # Fallback to current working directory
        specvla_root = os.getcwd()
        # Try to find SpecVLA in current path
        if "SpecVLA" in specvla_root:
            specvla_root = os.path.join(specvla_root.split("SpecVLA")[0], "SpecVLA") if "SpecVLA" in specvla_root else specvla_root
    
    # Set target directory for logs
    target_dir = os.path.join(specvla_root, "openvla", "specdecoding", "test-speed", "libero_goal_Spec_Relaxed")
    os.makedirs(target_dir, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    local_log_timefilepath = os.path.join(target_dir, run_id + "libero_goal.json")
    print(f"Logging to local log file: {local_log_filepath}")
    
    # Create debug log file in SpecVLA/exp directory
    exp_dir = os.path.join(specvla_root, "exp")
    os.makedirs(exp_dir, exist_ok=True)
    debug_log_filepath = os.path.join(exp_dir, f"{run_id}.log")
    debug_log_file = open(debug_log_filepath, "w")
    print(f"Debug logging to: {debug_log_filepath}")
    
    # Store debug log file path as a global variable for use in other modules
    import experiments.robot.openvla_utils as openvla_utils_module
    openvla_utils_module.DEBUG_LOG_FILE = debug_log_file

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
    # Data collection for accept length tracking
    accept_length_data = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
    specvla_actions_data = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
    db_actions_data = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
    task_names = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
    
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
        # Per-task data collection for DB retrieval tracking (only if using DB retrieval)
        task_accept_lengths = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
        task_specvla_actions = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
        task_db_actions = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            total_time = []
            # Per-episode data collection (only if using DB retrieval)
            # Store per-frame data: (specvla_action, db_action, accept_length)
            episode_accept_lengths = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
            episode_specvla_actions = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
            episode_db_actions = [] if (cfg.track_accept_length and cfg.use_db_retrieval) else None
            
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

                    # DB Retrieval: Get action slice from DB if enabled
                    db_action_slice = None
                    if cfg.use_db_retrieval:
                        try:
                            pil_img = Image.fromarray(img)
                            buf = BytesIO()
                            pil_img.save(buf, format='PNG')
                            buf.seek(0)
                            
                            files = {"file": ("image.png", buf, "image/png")}
                            data = {"instruction": task_description}
                            
                            response = requests.post(cfg.retrieval_url, files=files, data=data, timeout=30)
                            
                            if response.status_code == 200:
                                result = response.json()
                                if result.get('success', False):
                                    # DB returns format: {'rtcache_trajectory': [[action1], [action2], [action3], [action4]], ...}
                                    # Each action is 7-dimensional: [x, y, z, roll, pitch, yaw, gripper]
                                    # We only need the FIRST action (current frame action)
                                    retrieved_traj = None
                                    if 'rtcache_trajectory' in result and result['rtcache_trajectory']:
                                        retrieved_traj = np.array(result['rtcache_trajectory'])
                                    elif 'averaged_trajectory' in result and result['averaged_trajectory']:
                                        retrieved_traj = np.array(result['averaged_trajectory'])
                                    
                                    if retrieved_traj is not None and len(retrieved_traj) > 0:
                                        # Take ONLY the first action (current frame action) from DB retrieval
                                        # retrieved_traj shape is (4, 7) - 4 future actions, each 7-dimensional
                                        # We only use the first one (index 0) as the current frame action
                                        db_action_slice = retrieved_traj[0]  # Shape: (7,) - current frame action only
                        except Exception as e:
                            # Silently fail DB retrieval - don't affect main execution
                            pass
                    
                    # Always use SpecVLA model to get action (main execution - keeps original accuracy)
                    # If use_db_action_slice is enabled, pass DB action slice to replace draft tokens
                    result = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        return_time=True,
                        generate_mode='speculative',
                        track_accept_length=cfg.track_accept_length and cfg.use_db_retrieval,  # Track if using DB retrieval
                        db_action_slice=db_action_slice if cfg.use_db_retrieval else None,
                        use_db_action_slice=cfg.use_db_retrieval
                    )
                    # Handle different return values from get_action
                    # get_action can return: 
                    # - (action, time) when track_accept_length=False
                    # - (action, time, accept_lengths, draft_tokens) when track_accept_length=True (no DB)
                    # - (action, time, accept_lengths, draft_tokens, db_accept_lengths, db_action_slices) when track_accept_length=True and use_db_action_slice=True
                    # Note: time can be a tuple (end_time, start_time) or a single value
                    db_accept_lengths_step = None
                    db_action_slice_step = None
                    try:
                        if isinstance(result, tuple):
                            if len(result) == 6 and cfg.use_db_retrieval:
                                # DB action slice tracking enabled
                                action, time, accept_lengths, draft_tokens, db_accept_lengths_step, db_action_slice_step = result
                                # time might be a tuple (end_time, start_time), calculate duration if needed
                                if isinstance(time, tuple) and len(time) == 2:
                                    time = time[0] - time[1]  # end_time - start_time
                                # Debug: log what we got
                                if cfg.track_accept_length:
                                    debug_log_file.write(f"  [DB Tracking] Step {t}: get_action returned {len(result)} values\n")
                                    debug_log_file.write(f"  [DB Tracking] Step {t}: db_accept_lengths_step={db_accept_lengths_step}, type={type(db_accept_lengths_step)}\n")
                                    if isinstance(db_accept_lengths_step, list):
                                        debug_log_file.write(f"  [DB Tracking] db_accept_lengths_step length: {len(db_accept_lengths_step)}\n")
                                    if db_action_slice_step is not None:
                                        debug_log_file.write(f"  [DB Tracking] db_action_slice_step shape={np.array(db_action_slice_step).shape if isinstance(db_action_slice_step, (list, np.ndarray)) else 'N/A'}\n")
                                    debug_log_file.flush()
                            elif len(result) == 4 and cfg.use_db_retrieval:
                                # This should not happen if DB tracking is working correctly
                                # But handle it gracefully - DB tracking data is missing
                                action, time, accept_lengths, draft_tokens = result
                                if isinstance(time, tuple) and len(time) == 2:
                                    time = time[0] - time[1]
                                # Don't log warning - this might be expected in some cases
                                # Just set db_accept_lengths_step to None
                                db_accept_lengths_step = None
                                db_action_slice_step = None
                            elif len(result) == 4:
                                # track_accept_length was True - get_action returns 4 values
                                action, time, accept_lengths, draft_tokens = result
                                # time might be a tuple (end_time, start_time), calculate duration if needed
                                if isinstance(time, tuple) and len(time) == 2:
                                    time = time[0] - time[1]  # end_time - start_time
                            elif len(result) == 2:
                                # Normal case: (action, time)
                                action, time = result
                                # time might be a tuple (end_time, start_time), calculate duration if needed
                                if isinstance(time, tuple) and len(time) == 2:
                                    time = time[0] - time[1]  # end_time - start_time
                            else:
                                # Fallback: just take first two
                                action = result[0]
                                time = result[1] if len(result) > 1 else None
                                if isinstance(time, tuple) and len(time) == 2:
                                    time = time[0] - time[1]
                        else:
                            # Single value returned
                            action = result
                            time = None
                    except (ValueError, TypeError) as e:
                        # If unpacking fails, try to extract what we can
                        import traceback
                        error_detail = f"Error unpacking get_action result: {e}\nResult type: {type(result)}\nResult length: {len(result) if isinstance(result, (tuple, list)) else 'N/A'}\n{traceback.format_exc()}"
                        print(f"Warning: {error_detail}")
                        log_file.write(f"Warning: {error_detail}\n")
                        if isinstance(result, (tuple, list)) and len(result) > 0:
                            action = result[0]
                            time = result[1] if len(result) > 1 else None
                            if isinstance(time, tuple) and len(time) == 2:
                                time = time[0] - time[1]
                        else:
                            action = result
                            time = None
                    
                    # Store SpecVLA action slice and DB action slice for tracking (before normalization)
                    specvla_action_slice = action.copy() if isinstance(action, np.ndarray) else np.array(action)
                    
                    # Store per-frame data if using DB retrieval
                    if cfg.use_db_retrieval and cfg.track_accept_length:
                        if episode_specvla_actions is not None:
                            episode_specvla_actions.append(specvla_action_slice.copy())
                        if episode_db_actions is not None:
                            # Store DB action slice (current frame action from DB)
                            if db_action_slice is not None:
                                episode_db_actions.append(db_action_slice.copy() if isinstance(db_action_slice, np.ndarray) else np.array(db_action_slice))
                            else:
                                episode_db_actions.append(None)  # No DB action for this frame
                        # Store accept length (from DB action slice tracking)
                        # IMPORTANT: Save the accept_length directly from db_accept_lengths_step
                        if episode_accept_lengths is not None:
                            if db_accept_lengths_step is not None:
                                # db_accept_lengths_step is a list from eagenerate
                                # It contains the accept length when DB action slice was used to replace draft_tokens
                                # Example: db_accept_lengths_step = [4] means accept_length=4
                                if isinstance(db_accept_lengths_step, list):
                                    if len(db_accept_lengths_step) > 0:
                                        # Take the first (and only) accept length value
                                        accept_len_value = db_accept_lengths_step[0]
                                        episode_accept_lengths.append(accept_len_value)
                                        # Debug log
                                        debug_log_file.write(f"  [DB Tracking] Step {t}: Saved accept_length={accept_len_value} to episode_accept_lengths\n")
                                        debug_log_file.flush()
                                    else:
                                        episode_accept_lengths.append(0)  # Empty list
                                        debug_log_file.write(f"  [DB Tracking] Step {t}: Warning - db_accept_lengths_step is empty list, saving 0\n")
                                        debug_log_file.flush()
                                else:
                                    # Single value (shouldn't happen, but handle it)
                                    episode_accept_lengths.append(db_accept_lengths_step)
                                    debug_log_file.write(f"  [DB Tracking] Step {t}: Saved accept_length={db_accept_lengths_step} (not a list)\n")
                                    debug_log_file.flush()
                            else:
                                # No DB accept length data for this frame (DB retrieval might have failed or not enabled)
                                episode_accept_lengths.append(0)
                                debug_log_file.write(f"  [DB Tracking] Step {t}: db_accept_lengths_step is None, saving 0\n")
                                debug_log_file.flush()
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
                    import traceback
                    error_msg = f"Caught exception in step {t}: {e}\n{traceback.format_exc()}"
                    print(error_msg)
                    log_file.write(error_msg + "\n")
                    log_file.flush()
                    # Only break on critical errors, otherwise continue
                    # For now, break to avoid infinite loops, but log the error
                    break
            #exit()
            task_episodes += 1
            total_episodes += 1
            total_episode_time.append(total_time)
            
            # Store episode data to task data
            if cfg.track_accept_length and cfg.use_db_retrieval:
                if task_accept_lengths is not None:
                    task_accept_lengths.append(episode_accept_lengths if episode_accept_lengths else [])
                if task_specvla_actions is not None:
                    task_specvla_actions.append(episode_specvla_actions if episode_specvla_actions else [])
                if task_db_actions is not None:
                    task_db_actions.append(episode_db_actions if episode_db_actions else [])

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

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        
        # Store task data (only if using DB retrieval)
        if cfg.track_accept_length and cfg.use_db_retrieval:
            if task_accept_lengths is not None:
                accept_length_data.append(task_accept_lengths)
            if task_specvla_actions is not None:
                specvla_actions_data.append(task_specvla_actions)
            if task_db_actions is not None:
                db_actions_data.append(task_db_actions)
            task_names.append(task_description)
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
    
    # Save accept length tracking data if enabled (only if using DB retrieval)
    if cfg.track_accept_length and cfg.use_db_retrieval:
        # numpy is already imported at the top
        # Get the SpecVLA directory (assuming script is run from SpecVLA/openvla or similar)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to find SpecVLA root - try multiple possible paths
        possible_roots = [
            os.path.abspath(os.path.join(script_dir, "..", "..", "..")),  # from experiments/robot/libero
            os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..")),  # alternative
        ]
        specvla_root = None
        for root in possible_roots:
            if os.path.exists(os.path.join(root, "exp")) or os.path.exists(os.path.join(root, "openvla")):
                specvla_root = root
                break
        if specvla_root is None:
            # Fallback to current working directory
            specvla_root = os.getcwd()
            # Try to find SpecVLA in current path
            if "SpecVLA" in specvla_root:
                specvla_root = os.path.join(specvla_root.split("SpecVLA")[0], "SpecVLA") if "SpecVLA" in specvla_root else specvla_root
        exp_dir = os.path.join(specvla_root, "exp")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save data as numpy file
        if accept_length_data is not None and specvla_actions_data is not None and db_actions_data is not None and len(accept_length_data) > 0:
            data_file = os.path.join(exp_dir, f"accept_length_data_{run_id}.npz")
            try:
                # 这些结构是 task -> episode -> 列表 / ndarray，是高度不规则的嵌套结构
                # 必须显式转成 dtype=object，避免 numpy 尝试拼成规则矩阵时报 ValueError
                np.savez(
                    data_file,
                    accept_lengths=np.array(accept_length_data, dtype=object),  # Accept lengths for each frame
                    specvla_actions=np.array(specvla_actions_data, dtype=object),  # SpecVLA action slices for each frame
                    db_actions=np.array(db_actions_data, dtype=object),  # DB action slices for each frame
                    task_names=np.array(task_names, dtype=object),
                    task_suite_name=cfg.task_suite_name,
                    accept_threshold=cfg.accept_threshold if hasattr(cfg, 'accept_threshold') else None,
                    use_db_retrieval=cfg.use_db_retrieval if hasattr(cfg, 'use_db_retrieval') else False,
                )
                print(f"Saved accept length tracking data to: {data_file}")
                log_file.write(f"Saved accept length tracking data to: {data_file}\n")
            except Exception as e:
                import traceback
                err_msg = f"Error saving accept length tracking data to {data_file}: {e}\n{traceback.format_exc()}"
                print(err_msg)
                log_file.write(err_msg + "\n")
            
            # Generate visualizations
            try:
                # Import visualization function
                vis_module_path = os.path.join(script_dir, "visualize_accept_length.py")
                if os.path.exists(vis_module_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("visualize_accept_length", vis_module_path)
                    vis_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(vis_module)
                    vis_module.generate_visualizations(data_file, exp_dir, run_id)
                    print(f"Generated visualizations in: {exp_dir}")
                    log_file.write(f"Generated visualizations in: {exp_dir}\n")
                else:
                    print(f"Warning: Visualization script not found at {vis_module_path}")
                    log_file.write(f"Warning: Visualization script not found at {vis_module_path}\n")
            except Exception as e:
                import traceback
                print(f"Warning: Could not generate visualizations: {e}")
                print(traceback.format_exc())
                log_file.write(f"Warning: Could not generate visualizations: {e}\n")
        else:
            print("Warning: No accept length data collected to save")
            log_file.write("Warning: No accept length data collected to save\n")
    
    # Save local log file
    log_file.close()
    if debug_log_file is not None:
        debug_log_file.close()
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
