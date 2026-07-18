import collections
import dataclasses
import datetime as dt
import json
import logging
import math
import pathlib
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 12  # client-side upper bound

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task: Optional[str] = None  # e.g. "0", "0-9", "0,2,5-7"
    initial_state_jitter_std: float = 0.0  # Tiny Gaussian jitter on LIBERO init state; 0 disables it.
    initial_state_jitter_seed_offset: int = 100000

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "../outputs/test"  # Base path to save per-run directories
    run_name: Optional[str] = None  # Optional run directory name under video_out_path
    save_videos: bool = True
    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # Dataset recording
    #################################################################################################################
    record_flash_dataset: bool = False
    flash_dataset_out_path: str = "../dataset/flash_episodes"
    record_flash_dataset_format: str = "episode_npz"  # episode_npz or chunk_dir
    record_max_chunks_per_task: int = 0  # 0 means no per-task cap.
    record_max_episodes_per_task: int = 0  # 0 means no per-task cap.
    record_max_chunks_per_episode: int = 0  # 0 means record every inference chunk in the episode.
    record_successful_episodes_only: bool = True
    record_stop_task_when_full: bool = True


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    run_dir = _make_run_dir(args.video_out_path, args.run_name)
    run_id = run_dir.name
    episode_log_path = run_dir / "episode_log.json"
    episode_records: List[Dict[str, Any]] = []
    selected_task_ids = _parse_task_spec(args.task, num_tasks_in_suite)
    logging.info(
        "Client will run task id(s): %s (LIBERO '[info] using task orders [...]' is suite reordering, not this list)",
        selected_task_ids,
    )

    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": str(run_id),
            "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "task_suite_name": str(args.task_suite_name),
            "task_spec": None if args.task is None else str(args.task),
            "selected_task_ids": list(map(int, selected_task_ids)),
            "num_trials_per_task": int(args.num_trials_per_task),
            "replan_steps": int(args.replan_steps),
            "resize_size": int(args.resize_size),
            "video_fps": 10,
            "save_videos": bool(args.save_videos),
            "seed": int(args.seed),
            "initial_state_jitter_std": float(args.initial_state_jitter_std),
            "initial_state_jitter_seed_offset": int(args.initial_state_jitter_seed_offset),
            "record_flash_dataset": bool(args.record_flash_dataset),
            "flash_dataset_out_path": str(args.flash_dataset_out_path),
            "record_flash_dataset_format": str(args.record_flash_dataset_format),
            "record_max_chunks_per_task": int(args.record_max_chunks_per_task),
            "record_max_episodes_per_task": int(args.record_max_episodes_per_task),
            "record_max_chunks_per_episode": int(args.record_max_chunks_per_episode),
            "record_successful_episodes_only": bool(args.record_successful_episodes_only),
        },
    )
    if bool(args.record_flash_dataset):
        _write_dataset_manifest(
            out_root=pathlib.Path(args.flash_dataset_out_path),
            run_id=run_id,
            args=args,
            selected_task_ids=selected_task_ids,
        )

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    dataset_record_counts: Dict[int, int] = {}
    dataset_episode_counts: Dict[int, int] = {}
    for task_id in tqdm.tqdm(selected_task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            if _task_recording_complete(
                args=args,
                task_id=task_id,
                chunk_counts=dataset_record_counts,
                episode_counts=dataset_episode_counts,
            ):
                logging.info(
                    "Stopping task %s dataset recording after chunks=%s episodes=%s",
                    task_id,
                    dataset_record_counts.get(int(task_id), 0),
                    dataset_episode_counts.get(int(task_id), 0),
                )
                break
            logging.info("\nTask: %s", task_description)
            env.reset()
            action_plan = collections.deque()

            init_state_base_idx = int(episode_idx % len(initial_states))
            init_state_jitter_seed = int(args.seed + args.initial_state_jitter_seed_offset + int(task_id) * 1000 + episode_idx)
            init_state = _make_episode_initial_state(
                initial_states=np.asarray(initial_states),
                base_index=init_state_base_idx,
                jitter_std=float(args.initial_state_jitter_std),
                jitter_seed=init_state_jitter_seed,
            )
            obs = env.set_init_state(init_state)

            t = 0
            replay_images = []
            done = False
            episode_error: Optional[str] = None
            infer_calls = 0
            infer_latency_sum_ms = 0.0
            policy_time_sum_ms = 0.0
            policy_time_count = 0
            serve_time_sum_ms = 0.0
            serve_time_count = 0
            accepted_action_sum = 0.0
            draft_rounds = 0
            vlm_rounds = 0
            full_rounds = 0
            infer_id_next = 0
            episode_trace: List[Dict[str, Any]] = []
            episode_infers: List[Dict[str, Any]] = []
            episode_dataset_samples: List[Dict[str, Any]] = []

            executed_steps = 0
            need_reset = True

            logging.info("Starting episode %s...", task_episodes + 1)
            episode_wall_t0 = time.perf_counter()
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)
                    frame_idx = int(len(replay_images) - 1)

                    if not action_plan:
                        infer_id = int(infer_id_next)
                        infer_id_next += 1
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "__executed_steps__": int(executed_steps),
                            "__trace_run_id__": str(run_id),
                            "__trace_task_id__": int(task_id),
                            "__trace_episode_idx__": int(episode_idx),
                            "__trace_infer_id__": int(infer_id),
                        }
                        if need_reset:
                            element["__reset_policy_state__"] = True
                            need_reset = False

                        client_send_timestamp_s = time.time()
                        client_t0 = time.perf_counter()
                        out = client.infer(element)
                        client_roundtrip_ms = (time.perf_counter() - client_t0) * 1000.0
                        client_recv_timestamp_s = time.time()
                        action_chunk = out["actions"]
                        policy_timing = out.get("policy_timing", {})
                        server_timing = out.get("server_timing", {})
                        if not isinstance(server_timing, dict):
                            server_timing = {}
                        infer_calls += 1
                        infer_latency_ms = policy_timing.get("sample_actions_ms", None)
                        if infer_latency_ms is not None:
                            infer_latency_sum_ms += float(infer_latency_ms)
                        policy_time_ms, serve_time_ms = _policy_and_serve_time_ms(server_timing)
                        if policy_time_ms is not None:
                            policy_time_sum_ms += float(policy_time_ms)
                            policy_time_count += 1
                        if serve_time_ms is not None:
                            serve_time_sum_ms += float(serve_time_ms)
                            serve_time_count += 1
                        route_type = _route_type_from_timing(policy_timing)
                        if route_type == "full":
                            full_rounds += 1
                        else:
                            draft_rounds += 1

                        accepted = int(out.get("accepted_prefix_len", args.replan_steps))
                        if bool(round(float(policy_timing.get("include_in_draft_accept_metrics", 1.0)))):
                            accepted_action_sum += float(accepted)
                        exec_len = int(min(int(args.replan_steps), int(accepted))) if accepted > 0 else 0
                        chunk_actions = np.asarray(action_chunk, dtype=np.float32)
                        infer_record = _make_infer_record(
                            run_id=run_id,
                            task_suite_name=args.task_suite_name,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            infer_id=infer_id,
                            frame_idx=frame_idx,
                            env_step=t,
                            route_type=route_type,
                            accepted_prefix_len=accepted,
                            chunk_exec_len=exec_len,
                            policy_timing=policy_timing,
                            server_timing=server_timing,
                            client_send_timestamp_s=client_send_timestamp_s,
                            client_recv_timestamp_s=client_recv_timestamp_s,
                            client_roundtrip_ms=client_roundtrip_ms,
                            chunk_actions=chunk_actions,
                        )
                        if _should_record_dataset_sample(
                            args=args,
                            task_id=task_id,
                            chunk_counts=dataset_record_counts,
                            episode_counts=dataset_episode_counts,
                            episode_sample_count=len(episode_dataset_samples),
                        ):
                            if _recording_format(args) == "chunk_dir":
                                sample_path = _record_flash_dataset_sample(
                                    out_root=pathlib.Path(args.flash_dataset_out_path),
                                    run_id=run_id,
                                    task_suite_name=args.task_suite_name,
                                    task_id=task_id,
                                    task_description=task_description,
                                    episode_idx=episode_idx,
                                    infer_id=infer_id,
                                    frame_idx=frame_idx,
                                    env_step=t,
                                    third_person_image=img,
                                    wrist_image=wrist_img,
                                    state=np.asarray(element["observation/state"], dtype=np.float32),
                                    chunk_actions=chunk_actions,
                                    infer_record=infer_record,
                                )
                                dataset_record_counts[int(task_id)] = dataset_record_counts.get(int(task_id), 0) + 1
                                infer_record["dataset_sample_path"] = str(
                                    sample_path.relative_to(pathlib.Path(args.flash_dataset_out_path))
                                )
                            else:
                                episode_dataset_samples.append(
                                    {
                                        "infer_id": int(infer_id),
                                        "frame_idx": int(frame_idx),
                                        "env_step": int(t),
                                        "third_person_image": np.asarray(img, dtype=np.uint8).copy(),
                                        "wrist_image": np.asarray(wrist_img, dtype=np.uint8).copy(),
                                        "state": np.asarray(element["observation/state"], dtype=np.float32).copy(),
                                        "action_chunk": np.asarray(chunk_actions, dtype=np.float32).copy(),
                                        "infer_record": dict(infer_record),
                                    }
                                )
                        episode_infers.append(infer_record)
                        if exec_len <= 0:
                            episode_trace.append(
                                _make_trace_record(
                                    run_id=run_id,
                                    task_suite_name=args.task_suite_name,
                                    task_id=task_id,
                                    episode_idx=episode_idx,
                                    frame_idx=frame_idx,
                                    env_step=t,
                                    infer_record=infer_record,
                                    action=None,
                                    action_offset_in_chunk=None,
                                    infer_start_frame=True,
                                    reward=None,
                                    done_after_step=False,
                                )
                            )
                            executed_steps = 0
                            continue
                        for action_offset in range(exec_len):
                            action_plan.append(
                                {
                                    "action": chunk_actions[action_offset].astype(np.float32, copy=True),
                                    "infer_record": infer_record,
                                    "action_offset_in_chunk": int(action_offset),
                                    "infer_start_frame": bool(action_offset == 0),
                                }
                            )

                        executed_steps = exec_len

                    action_meta = action_plan.popleft()
                    action = np.asarray(action_meta["action"], dtype=np.float32)
                    obs, reward, done, info = env.step(action.tolist())
                    episode_trace.append(
                        _make_trace_record(
                            run_id=run_id,
                            task_suite_name=args.task_suite_name,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            frame_idx=frame_idx,
                            env_step=t,
                            infer_record=action_meta["infer_record"],
                            action=action,
                            action_offset_in_chunk=int(action_meta["action_offset_in_chunk"]),
                            infer_start_frame=bool(action_meta["infer_start_frame"]),
                            reward=float(reward),
                            done_after_step=bool(done),
                        )
                    )
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    episode_error = str(e)
                    break

            episode_wall_time_s = time.perf_counter() - episode_wall_t0
            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            episode_dir = _episode_output_dir(
                run_dir=run_dir,
                task_description=task_description,
                suffix=suffix,
                task_id=task_id,
                episode_idx=episode_idx,
            )
            video_path = episode_dir / f"rollout_{suffix}.mp4"
            if bool(args.save_videos):
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            trace_path = episode_dir / "trace.jsonl"
            infer_path = episode_dir / "infer.jsonl"
            _write_jsonl(trace_path, episode_trace)
            _write_jsonl(infer_path, episode_infers)
            should_write_episode_dataset = (
                bool(args.record_flash_dataset)
                and _recording_format(args) == "episode_npz"
                and episode_dataset_samples
                and (bool(done) or not bool(args.record_successful_episodes_only))
            )
            if should_write_episode_dataset:
                episode_dataset_path = _record_flash_dataset_episode(
                    out_root=pathlib.Path(args.flash_dataset_out_path),
                    run_id=run_id,
                    task_suite_name=args.task_suite_name,
                    task_id=task_id,
                    task_description=task_description,
                    episode_idx=episode_idx,
                    success=bool(done),
                    failure_reason=episode_error,
                    max_steps=max_steps,
                    num_steps_wait=args.num_steps_wait,
                    initial_state=init_state,
                    initial_state_base_idx=init_state_base_idx,
                    initial_state_jitter_std=float(args.initial_state_jitter_std),
                    initial_state_jitter_seed=init_state_jitter_seed,
                    samples=episode_dataset_samples,
                    trace_records=episode_trace,
                    infer_records=episode_infers,
                )
                dataset_episode_counts[int(task_id)] = dataset_episode_counts.get(int(task_id), 0) + 1
                dataset_record_counts[int(task_id)] = dataset_record_counts.get(int(task_id), 0) + len(episode_dataset_samples)
            else:
                if bool(args.record_flash_dataset) and _recording_format(args) == "episode_npz" and episode_dataset_samples:
                    logging.info(
                        "Skipping dataset write for task %s episode %s because success=%s and success-only=%s",
                        task_id,
                        episode_idx,
                        bool(done),
                        bool(args.record_successful_episodes_only),
                    )
                episode_dataset_path = None
            mean_infer_latency_ms = (infer_latency_sum_ms / float(infer_calls)) if infer_calls > 0 else None
            mean_policy_time_ms = (
                policy_time_sum_ms / float(policy_time_count)
            ) if policy_time_count > 0 else None
            mean_serve_time_ms = (
                serve_time_sum_ms / float(serve_time_count)
            ) if serve_time_count > 0 else None
            accepted_metric_rounds = draft_rounds
            mean_accepted_action_len = (
                accepted_action_sum / float(accepted_metric_rounds)
            ) if accepted_metric_rounds > 0 else None
            infer_route_counts = _count_by_key(episode_infers, "route_type")
            executed_action_count = int(sum(1 for r in episode_trace if r.get("executed_action") is not None))
            action_route_counts = _count_by_key(
                [r for r in episode_trace if r.get("executed_action") is not None],
                "route_type",
            )
            episode_records.append(
                {
                    "run_id": str(run_id),
                    "task_suite_name": args.task_suite_name,
                    "task_id": int(task_id),
                    "task_description": str(task_description),
                    "episode_idx": int(episode_idx),
                    "success": bool(done),
                    "failure_reason": episode_error,
                    "max_steps": int(max_steps),
                    "num_steps_wait": int(args.num_steps_wait),
                    "initial_state_base_idx": int(init_state_base_idx),
                    "initial_state_jitter_std": float(args.initial_state_jitter_std),
                    "initial_state_jitter_seed": int(init_state_jitter_seed),
                    "env_steps_taken": int(t),
                    "episode_wall_time_s": float(episode_wall_time_s),
                    "infer_calls": int(infer_calls),
                    "draft_rounds": int(draft_rounds),
                    "vlm_rounds": int(vlm_rounds),
                    "full_rounds": int(full_rounds),
                    "executed_action_count": int(executed_action_count),
                    "infer_latency_mean_ms": None if mean_infer_latency_ms is None else float(mean_infer_latency_ms),
                    "infer_latency_sum_ms": float(infer_latency_sum_ms),
                    "policy_time_mean_ms": None if mean_policy_time_ms is None else float(mean_policy_time_ms),
                    "policy_time_sum_ms": float(policy_time_sum_ms),
                    "policy_time_count": int(policy_time_count),
                    "serve_time_mean_ms": None if mean_serve_time_ms is None else float(mean_serve_time_ms),
                    "serve_time_sum_ms": float(serve_time_sum_ms),
                    "serve_time_count": int(serve_time_count),
                    "avg_latency_per_action_ms": (
                        float(infer_latency_sum_ms) / float(executed_action_count) if executed_action_count > 0 else None
                    ),
                    "avg_policy_time_per_action_ms": (
                        float(policy_time_sum_ms) / float(executed_action_count)
                        if policy_time_count > 0 and executed_action_count > 0
                        else None
                    ),
                    "avg_serve_time_per_action_ms": (
                        float(serve_time_sum_ms) / float(executed_action_count)
                        if serve_time_count > 0 and executed_action_count > 0
                        else None
                    ),
                    "accepted_action_len_mean": None if mean_accepted_action_len is None else float(mean_accepted_action_len),
                    "route_ratio_by_infer": _ratio_dict(infer_route_counts, denom=int(len(episode_infers))),
                    "route_ratio_by_action": _ratio_dict(action_route_counts, denom=int(executed_action_count)),
                    "video_path": str(video_path.relative_to(run_dir)) if bool(args.save_videos) else None,
                    "trace_path": str(trace_path.relative_to(run_dir)),
                    "infer_path": str(infer_path.relative_to(run_dir)),
                    "dataset_episode_path": (
                        None
                        if episode_dataset_path is None
                        else str(episode_dataset_path.relative_to(pathlib.Path(args.flash_dataset_out_path)))
                    ),
                    "dataset_recorded_chunk_count": int(len(episode_dataset_samples)),
                    "frame_count": int(len(replay_images)),
                    "trace_record_count": int(len(episode_trace)),
                    "infer_record_count": int(len(episode_infers)),
                    "task_spec": None if args.task is None else str(args.task),
                    "seed": int(args.seed),
                }
            )
            _write_episode_log(episode_log_path, episode_records)

            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %s", total_episodes)
            logging.info("# successes: %s (%.1f%%)", total_successes, total_successes / total_episodes * 100.0)

        logging.info("Current task success rate: %.3f", float(task_successes) / float(task_episodes))
        logging.info("Current total success rate: %.3f", float(total_successes) / float(total_episodes))

    logging.info("Total success rate: %.3f", float(total_successes) / float(total_episodes))
    logging.info("Total episodes: %s", total_episodes)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _parse_task_spec(spec: Optional[str], num_tasks: int) -> List[int]:
    if spec is None or str(spec).strip() == "":
        return list(range(int(num_tasks)))
    task_ids: Set[int] = set()
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if match:
            lo = int(match.group(1))
            hi = int(match.group(2))
            if lo > hi:
                raise ValueError(f"invalid task range {token!r}: start > end")
            for task_id in range(lo, hi + 1):
                if task_id < 0 or task_id >= int(num_tasks):
                    raise ValueError(f"task id {task_id} out of range [0, {int(num_tasks) - 1}]")
                task_ids.add(int(task_id))
            continue
        if not token.isdigit():
            raise ValueError(f"unsupported task selector token {token!r}")
        task_id = int(token)
        if task_id < 0 or task_id >= int(num_tasks):
            raise ValueError(f"task id {task_id} out of range [0, {int(num_tasks) - 1}]")
        task_ids.add(int(task_id))
    return sorted(task_ids)


def _make_run_dir(video_out_path: str, run_name: Optional[str]) -> pathlib.Path:
    base_dir = pathlib.Path(video_out_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None or str(run_name).strip() == "":
        run_name = dt.datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_dir / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_episode_initial_state(
    *,
    initial_states: np.ndarray,
    base_index: int,
    jitter_std: float,
    jitter_seed: int,
) -> np.ndarray:
    init_state = np.asarray(initial_states[int(base_index)]).copy()
    if float(jitter_std) <= 0.0:
        return init_state
    rng = np.random.default_rng(int(jitter_seed))
    jitter_dtype = init_state.dtype if np.issubdtype(init_state.dtype, np.floating) else np.float32
    jitter = rng.normal(loc=0.0, scale=float(jitter_std), size=init_state.shape).astype(jitter_dtype)
    return init_state + jitter


def _recording_format(args: Args) -> str:
    value = str(args.record_flash_dataset_format).strip().lower()
    if value not in {"episode_npz", "chunk_dir"}:
        raise ValueError("record_flash_dataset_format must be either 'episode_npz' or 'chunk_dir'")
    return value


def _task_recording_complete(
    *,
    args: Args,
    task_id: int,
    chunk_counts: Dict[int, int],
    episode_counts: Dict[int, int],
) -> bool:
    if not bool(args.record_flash_dataset) or not bool(args.record_stop_task_when_full):
        return False
    if _recording_format(args) == "episode_npz":
        cap = int(args.record_max_episodes_per_task)
        return cap > 0 and episode_counts.get(int(task_id), 0) >= cap
    cap = int(args.record_max_chunks_per_task)
    return cap > 0 and chunk_counts.get(int(task_id), 0) >= cap


def _should_record_dataset_sample(
    *,
    args: Args,
    task_id: int,
    chunk_counts: Dict[int, int],
    episode_counts: Dict[int, int],
    episode_sample_count: int,
) -> bool:
    if not bool(args.record_flash_dataset):
        return False
    if _recording_format(args) == "episode_npz":
        episode_cap = int(args.record_max_episodes_per_task)
        if episode_cap > 0 and episode_counts.get(int(task_id), 0) >= episode_cap:
            return False
        per_episode_cap = int(args.record_max_chunks_per_episode)
        return per_episode_cap <= 0 or int(episode_sample_count) < per_episode_cap
    chunk_cap = int(args.record_max_chunks_per_task)
    return chunk_cap <= 0 or chunk_counts.get(int(task_id), 0) < chunk_cap


def _write_dataset_manifest(
    *,
    out_root: pathlib.Path,
    run_id: str,
    args: Args,
    selected_task_ids: List[int],
) -> None:
    manifest_path = out_root / "manifest.json"
    existing: Dict[str, Any]
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    runs = list(existing.get("runs", []))
    record_format = _recording_format(args)
    if record_format == "episode_npz":
        schema_version = "flash_episode_v1"
        sample_layout = "{suite}/task_{task_id:02d}/{run_id}_episode_{episode_idx:04d}.npz"
        description = "One compressed NPZ per LIBERO episode with inference-time images, state, payload, trace, and returned 50-step action chunks."
    else:
        schema_version = "flash_chunk_v1"
        sample_layout = "{suite}/task_{task_id:02d}/{run_id}/episode_{episode_idx:03d}/infer_{infer_id:06d}"
        description = "FLASH LIBERO client-side observations and returned 50-step action chunks."
    runs.append(
        {
            "run_id": str(run_id),
            "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "task_suite_name": str(args.task_suite_name),
            "task_spec": None if args.task is None else str(args.task),
            "selected_task_ids": list(map(int, selected_task_ids)),
            "num_trials_per_task": int(args.num_trials_per_task),
            "record_flash_dataset_format": str(record_format),
            "record_max_chunks_per_task": int(args.record_max_chunks_per_task),
            "record_max_episodes_per_task": int(args.record_max_episodes_per_task),
            "record_max_chunks_per_episode": int(args.record_max_chunks_per_episode),
            "record_successful_episodes_only": bool(args.record_successful_episodes_only),
            "initial_state_jitter_std": float(args.initial_state_jitter_std),
            "initial_state_jitter_seed_offset": int(args.initial_state_jitter_seed_offset),
            "schema_version": str(schema_version),
            "sample_layout": str(sample_layout),
            "image_space": f"uint8_rgb_rot180_resize_pad_{int(args.resize_size)}",
            "action_space": "flash_client_output_env_action_chunk_7",
            "action_chunk_shape": [50, 7],
        }
    )
    payload = {
        "schema_version": str(schema_version),
        "description": str(description),
        "runs": runs,
    }
    _write_json(manifest_path, payload)


def _record_flash_dataset_sample(
    *,
    out_root: pathlib.Path,
    run_id: str,
    task_suite_name: str,
    task_id: int,
    task_description: str,
    episode_idx: int,
    infer_id: int,
    frame_idx: int,
    env_step: int,
    third_person_image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray,
    chunk_actions: np.ndarray,
    infer_record: Dict[str, Any],
) -> pathlib.Path:
    sample_dir = (
        out_root
        / str(task_suite_name)
        / f"task_{int(task_id):02d}"
        / str(run_id)
        / f"episode_{int(episode_idx):03d}"
        / f"infer_{int(infer_id):06d}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    imageio.imwrite(sample_dir / "third_person.png", np.asarray(third_person_image, dtype=np.uint8))
    imageio.imwrite(sample_dir / "wrist.png", np.asarray(wrist_image, dtype=np.uint8))
    np.save(sample_dir / "state.npy", np.asarray(state, dtype=np.float32))
    np.save(sample_dir / "action_chunk.npy", np.asarray(chunk_actions, dtype=np.float32))

    chunk = np.asarray(chunk_actions, dtype=np.float32)
    metadata = {
        "schema_version": "flash_chunk_v1",
        "run_id": str(run_id),
        "task_suite_name": str(task_suite_name),
        "task_id": int(task_id),
        "task_description": str(task_description),
        "episode_idx": int(episode_idx),
        "infer_id": int(infer_id),
        "frame_idx": int(frame_idx),
        "env_step": int(env_step),
        "third_person_image": "third_person.png",
        "wrist_image": "wrist.png",
        "state": "state.npy",
        "action_chunk": "action_chunk.npy",
        "action_chunk_shape": list(map(int, chunk.shape)),
        "action_space": "flash_client_output_env_action_chunk_7",
        "image_space": "client_preprocessed_uint8_rgb_rot180_resize_pad",
        "route_type": str(infer_record.get("route_type", "unknown")),
        "accepted_prefix_len": int(infer_record.get("accepted_prefix_len", 0)),
        "chunk_exec_len": int(infer_record.get("chunk_exec_len", 0)),
        "sample_actions_ms": infer_record.get("sample_actions_ms"),
        "total_ms": infer_record.get("total_ms"),
        "radius_dist_mean": infer_record.get("radius_dist_mean"),
    }
    _write_json(sample_dir / "metadata.json", metadata)

    index_record = {
        **metadata,
        "sample_dir": str(sample_dir.relative_to(out_root)),
    }
    index_path = out_root / str(task_suite_name) / f"task_{int(task_id):02d}" / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(index_record, ensure_ascii=False) + "\n")
    return sample_dir


def _record_flash_dataset_episode(
    *,
    out_root: pathlib.Path,
    run_id: str,
    task_suite_name: str,
    task_id: int,
    task_description: str,
    episode_idx: int,
    success: bool,
    failure_reason: Optional[str],
    max_steps: int,
    num_steps_wait: int,
    initial_state: np.ndarray,
    initial_state_base_idx: int,
    initial_state_jitter_std: float,
    initial_state_jitter_seed: int,
    samples: List[Dict[str, Any]],
    trace_records: List[Dict[str, Any]],
    infer_records: List[Dict[str, Any]],
) -> pathlib.Path:
    if not samples:
        raise ValueError("cannot write an episode dataset file without samples")

    task_dir = out_root / str(task_suite_name) / f"task_{int(task_id):02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    episode_path = task_dir / f"{run_id}_episode_{int(episode_idx):04d}.npz"

    third_person_images = np.stack([np.asarray(x["third_person_image"], dtype=np.uint8) for x in samples], axis=0)
    wrist_images = np.stack([np.asarray(x["wrist_image"], dtype=np.uint8) for x in samples], axis=0)
    states = np.stack([np.asarray(x["state"], dtype=np.float32) for x in samples], axis=0)
    action_chunks = np.stack([np.asarray(x["action_chunk"], dtype=np.float32) for x in samples], axis=0)
    infer_ids = np.asarray([int(x["infer_id"]) for x in samples], dtype=np.int32)
    frame_idxs = np.asarray([int(x["frame_idx"]) for x in samples], dtype=np.int32)
    env_steps = np.asarray([int(x["env_step"]) for x in samples], dtype=np.int32)

    sample_infer_records = [dict(x["infer_record"]) for x in samples]
    executed_trace_records = [dict(x) for x in trace_records if x.get("executed_action") is not None]
    if executed_trace_records:
        executed_actions = np.stack(
            [np.asarray(x["executed_action"], dtype=np.float32) for x in executed_trace_records],
            axis=0,
        )
    else:
        executed_actions = np.zeros((0, int(action_chunks.shape[-1])), dtype=np.float32)
    executed_infer_ids = np.asarray([int(x["infer_id"]) for x in executed_trace_records], dtype=np.int32)
    executed_action_offsets = np.asarray(
        [int(x["action_offset_in_chunk"]) for x in executed_trace_records],
        dtype=np.int32,
    )
    executed_frame_idxs = np.asarray([int(x["frame_idx"]) for x in executed_trace_records], dtype=np.int32)
    executed_env_steps = np.asarray([int(x["env_step"]) for x in executed_trace_records], dtype=np.int32)
    executed_rewards = np.asarray(
        [np.nan if x.get("reward") is None else float(x["reward"]) for x in executed_trace_records],
        dtype=np.float32,
    )
    executed_done_after_step = np.asarray(
        [bool(x.get("done_after_step", False)) for x in executed_trace_records],
        dtype=np.bool_,
    )
    executed_route_types = np.asarray([str(x.get("route_type", "unknown")) for x in executed_trace_records])

    infer_payloads = []
    for idx, record in enumerate(sample_infer_records):
        payload = dict(record)
        payload.pop("chunk_actions", None)
        payload["action_chunk_ref"] = f"action_chunks[{idx}]"
        infer_payloads.append(payload)

    route_types = np.asarray([str(x.get("route_type", "unknown")) for x in sample_infer_records])
    accepted_prefix_lens = np.asarray(
        [int(x.get("accepted_prefix_len", 0)) for x in sample_infer_records],
        dtype=np.int32,
    )
    chunk_exec_lens = np.asarray(
        [int(x.get("chunk_exec_len", 0)) for x in sample_infer_records],
        dtype=np.int32,
    )

    metadata = {
        "schema_version": "flash_episode_v1",
        "run_id": str(run_id),
        "task_suite_name": str(task_suite_name),
        "task_id": int(task_id),
        "task_description": str(task_description),
        "episode_idx": int(episode_idx),
        "success": bool(success),
        "failure_reason": failure_reason,
        "max_steps": int(max_steps),
        "num_steps_wait": int(num_steps_wait),
        "initial_state_base_idx": int(initial_state_base_idx),
        "initial_state_jitter_std": float(initial_state_jitter_std),
        "initial_state_jitter_seed": int(initial_state_jitter_seed),
        "infer_count": int(len(samples)),
        "trace_record_count": int(len(trace_records)),
        "executed_action_count": int(len(executed_trace_records)),
        "action_space": "flash_client_output_env_action_chunk_7",
        "action_chunk_shape": list(map(int, action_chunks.shape[1:])),
        "executed_action_shape": list(map(int, executed_actions.shape)),
        "image_space": "client_preprocessed_uint8_rgb_rot180_resize_pad",
        "arrays": {
            "third_person_images": list(map(int, third_person_images.shape)),
            "wrist_images": list(map(int, wrist_images.shape)),
            "states": list(map(int, states.shape)),
            "initial_state": list(map(int, np.asarray(initial_state).shape)),
            "action_chunks": list(map(int, action_chunks.shape)),
            "infer_ids": list(map(int, infer_ids.shape)),
            "frame_idxs": list(map(int, frame_idxs.shape)),
            "env_steps": list(map(int, env_steps.shape)),
            "route_types": list(map(int, route_types.shape)),
            "executed_actions": list(map(int, executed_actions.shape)),
            "executed_infer_ids": list(map(int, executed_infer_ids.shape)),
            "executed_action_offsets": list(map(int, executed_action_offsets.shape)),
            "executed_frame_idxs": list(map(int, executed_frame_idxs.shape)),
            "executed_env_steps": list(map(int, executed_env_steps.shape)),
            "executed_rewards": list(map(int, executed_rewards.shape)),
            "executed_done_after_step": list(map(int, executed_done_after_step.shape)),
            "executed_route_types": list(map(int, executed_route_types.shape)),
            "infer_payloads_json": [int(len(infer_payloads))],
            "trace_records_json": [int(len(trace_records))],
        },
        "payload_note": "Per-inference payloads omit duplicate chunk_actions; use action_chunks[i] via action_chunk_ref. executed_actions stores the actual environment trajectory.",
    }

    np.savez_compressed(
        episode_path,
        third_person_images=third_person_images,
        wrist_images=wrist_images,
        states=states,
        initial_state=np.asarray(initial_state, dtype=np.float32),
        action_chunks=action_chunks,
        infer_ids=infer_ids,
        frame_idxs=frame_idxs,
        env_steps=env_steps,
        route_types=route_types,
        accepted_prefix_lens=accepted_prefix_lens,
        chunk_exec_lens=chunk_exec_lens,
        executed_actions=executed_actions,
        executed_infer_ids=executed_infer_ids,
        executed_action_offsets=executed_action_offsets,
        executed_frame_idxs=executed_frame_idxs,
        executed_env_steps=executed_env_steps,
        executed_rewards=executed_rewards,
        executed_done_after_step=executed_done_after_step,
        executed_route_types=executed_route_types,
        sample_actions_ms=_float_field_array(sample_infer_records, "sample_actions_ms"),
        policy_time_ms=_float_field_array(sample_infer_records, "policy_time_ms"),
        serve_time_ms=_float_field_array(sample_infer_records, "serve_time_ms"),
        client_roundtrip_ms=_float_field_array(sample_infer_records, "client_roundtrip_ms"),
        total_ms=_float_field_array(sample_infer_records, "total_ms"),
        radius_dist_mean=_float_field_array(sample_infer_records, "radius_dist_mean"),
        infer_payloads_json=np.asarray([_json_dumps_compact(x) for x in infer_payloads]),
        trace_records_json=np.asarray(_json_dumps_compact(trace_records)),
        infer_records_json=np.asarray(_json_dumps_compact([_infer_record_without_actions(x) for x in infer_records])),
        metadata_json=np.asarray(_json_dumps_compact(metadata)),
    )

    index_record = {
        **metadata,
        "episode_file": str(episode_path.relative_to(out_root)),
    }
    index_path = task_dir / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(index_record, ensure_ascii=False) + "\n")
    return episode_path


def _infer_record_without_actions(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(record)
    payload.pop("chunk_actions", None)
    return payload


def _float_field_array(records: List[Dict[str, Any]], field: str) -> np.ndarray:
    values = []
    for record in records:
        value = _float_or_none(record.get(field))
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float32)


def _json_dumps_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _episode_output_dir(
    *,
    run_dir: pathlib.Path,
    task_description: str,
    suffix: str,
    task_id: int,
    episode_idx: int,
) -> pathlib.Path:
    task_segment = task_description.replace(" ", "_")
    path = run_dir / "episodes" / f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_episode_log(path: pathlib.Path, records: List[Dict[str, Any]]) -> None:
    _write_json(path, records)


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: pathlib.Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _route_type_from_timing(policy_timing: Dict[str, Any]) -> str:
    is_full_round = bool(round(float(policy_timing.get("is_full_pipeline_round", 0.0))))
    if is_full_round:
        return "full"
    return "draft"


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _policy_and_serve_time_ms(server_timing: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    policy_time_ms = _float_or_none(server_timing.get("policy_time_ms"))
    if policy_time_ms is None:
        policy_time_ms = _float_or_none(server_timing.get("infer_ms"))

    serve_time_ms = _float_or_none(server_timing.get("serve_time_ms"))
    if serve_time_ms is None and policy_time_ms is not None:
        ws_unpack_ms = _float_or_none(server_timing.get("ws_unpack_ms"))
        ws_pack_ms = _float_or_none(server_timing.get("ws_pack_ms"))
        if ws_unpack_ms is not None and ws_pack_ms is not None:
            serve_time_ms = float(ws_unpack_ms) + float(policy_time_ms) + float(ws_pack_ms)
    return policy_time_ms, serve_time_ms


def _make_infer_record(
    *,
    run_id: str,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    infer_id: int,
    frame_idx: int,
    env_step: int,
    route_type: str,
    accepted_prefix_len: int,
    chunk_exec_len: int,
    policy_timing: Dict[str, Any],
    server_timing: Dict[str, Any],
    client_send_timestamp_s: float,
    client_recv_timestamp_s: float,
    client_roundtrip_ms: float,
    chunk_actions: np.ndarray,
) -> Dict[str, Any]:
    policy_time_ms, serve_time_ms = _policy_and_serve_time_ms(server_timing)
    return {
        "timestamp": float(client_recv_timestamp_s),
        "client_send_timestamp_s": float(client_send_timestamp_s),
        "client_recv_timestamp_s": float(client_recv_timestamp_s),
        "client_roundtrip_ms": float(client_roundtrip_ms),
        "server_recv_timestamp_s": _float_or_none(server_timing.get("server_recv_timestamp_s")),
        "server_response_timestamp_s": _float_or_none(server_timing.get("server_response_timestamp_s")),
        "run_id": str(run_id),
        "task_suite_name": str(task_suite_name),
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "infer_id": int(infer_id),
        "frame_idx": int(frame_idx),
        "env_step": int(env_step),
        "route_type": str(route_type),
        "accepted_prefix_len": int(accepted_prefix_len),
        "chunk_exec_len": int(chunk_exec_len),
        "chunk_actions": np.asarray(chunk_actions, dtype=np.float32).tolist(),
        "sample_actions_ms": _float_or_none(policy_timing.get("sample_actions_ms")),
        "policy_time_ms": policy_time_ms,
        "serve_time_ms": serve_time_ms,
        "ws_unpack_ms": _float_or_none(server_timing.get("ws_unpack_ms")),
        "ws_pack_ms": _float_or_none(server_timing.get("ws_pack_ms")),
        "encoder_ms": _float_or_none(policy_timing.get("encoder_ms")),
        "vlm_prefill_ms": _float_or_none(policy_timing.get("vlm_prefill_ms")),
        "draft_ms": _float_or_none(policy_timing.get("draft_ms")),
        "action_verify_ms": _float_or_none(policy_timing.get("action_verify_ms")),
        "full_fallback_ms": _float_or_none(policy_timing.get("full_fallback_ms")),
        "total_ms": _float_or_none(policy_timing.get("total_ms")),
        "accepted_prefix_len_mean": _float_or_none(policy_timing.get("accepted_prefix_len_mean")),
        "radius_dist_mean": _float_or_none(policy_timing.get("radius_dist_mean")),
        "gripper_override_rate": _float_or_none(policy_timing.get("gripper_override_rate")),
        "did_prefill": int(round(float(policy_timing.get("did_prefill", 0.0)))),
        "is_full_pipeline_round": int(round(float(policy_timing.get("is_full_pipeline_round", 0.0)))),
        "used_full_fallback": int(round(float(policy_timing.get("used_full_fallback", 0.0)))),
        "scheduled_full_fallback": int(round(float(policy_timing.get("scheduled_full_fallback", 0.0)))),
        "verify_mode_random": int(round(float(policy_timing.get("verify_mode_random", 0.0)))),
        "gripper_verify_enabled": int(round(float(policy_timing.get("gripper_verify_enabled", 0.0)))),
        "rtcache_draft": int(round(float(policy_timing.get("rtcache_draft", 0.0)))),
        "rtcache_runtime_feature": int(round(float(policy_timing.get("rtcache_runtime_feature", 0.0)))),
        "rtcache_with_embedding_ms": _float_or_none(policy_timing.get("rtcache_with_embedding_ms")),
        "rtcache_retrieval_ms": _float_or_none(policy_timing.get("rtcache_retrieval_ms")),
        "rtcache_top_score": _float_or_none(policy_timing.get("rtcache_top_score")),
        "rtcache_selected_score": _float_or_none(policy_timing.get("rtcache_selected_score")),
        "rtcache_num_candidates": _float_or_none(policy_timing.get("rtcache_num_candidates")),
        "rtcache_selected_rank": _float_or_none(policy_timing.get("rtcache_selected_rank")),
        "rtcache_best_accept_len": _float_or_none(policy_timing.get("rtcache_best_accept_len")),
        "rtcache_rerank_verify_ms": _float_or_none(policy_timing.get("rtcache_rerank_verify_ms")),
        "rtcache_rerank_extra_verifies": _float_or_none(policy_timing.get("rtcache_rerank_extra_verifies")),
        "rtcache_noverify": _float_or_none(policy_timing.get("rtcache_noverify")),
        "rtcache_composite": _float_or_none(policy_timing.get("rtcache_composite")),
        "rtcache_composite_threshold": _float_or_none(policy_timing.get("rtcache_composite_threshold")),
        "rtcache_composite_displacement": _float_or_none(policy_timing.get("rtcache_composite_displacement")),
        "rtcache_composite_radius": _float_or_none(policy_timing.get("rtcache_composite_radius")),
        "rtcache_composite_norm_displacement": _float_or_none(policy_timing.get("rtcache_composite_norm_displacement")),
        "rtcache_composite_norm_radius": _float_or_none(policy_timing.get("rtcache_composite_norm_radius")),
        "rtcache_noverify_streak": _float_or_none(policy_timing.get("rtcache_noverify_streak")),
        "rtcache_record_index": _float_or_none(policy_timing.get("rtcache_record_index")),
        "rtcache_record_task_id": _float_or_none(policy_timing.get("rtcache_record_task_id")),
        "rtcache_record_episode_idx": _float_or_none(policy_timing.get("rtcache_record_episode_idx")),
        "rtcache_record_infer_id": _float_or_none(policy_timing.get("rtcache_record_infer_id")),
        "rtcache_prompt_task_id": _float_or_none(policy_timing.get("rtcache_prompt_task_id")),
        "rtcache_trace_task_id": _float_or_none(policy_timing.get("rtcache_trace_task_id")),
    }


def _make_trace_record(
    *,
    run_id: str,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    frame_idx: int,
    env_step: int,
    infer_record: Dict[str, Any],
    action: Optional[np.ndarray],
    action_offset_in_chunk: Optional[int],
    infer_start_frame: bool,
    reward: Optional[float],
    done_after_step: bool,
) -> Dict[str, Any]:
    return {
        "run_id": str(run_id),
        "task_suite_name": str(task_suite_name),
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "frame_idx": int(frame_idx),
        "env_step": int(env_step),
        "infer_id": int(infer_record["infer_id"]),
        "infer_start_frame": bool(infer_start_frame),
        "action_offset_in_chunk": None if action_offset_in_chunk is None else int(action_offset_in_chunk),
        "executed_action": None if action is None else np.asarray(action, dtype=np.float32).tolist(),
        "route_type": str(infer_record["route_type"]),
        "accepted_prefix_len": int(infer_record["accepted_prefix_len"]),
        "chunk_exec_len": int(infer_record["chunk_exec_len"]),
        "sample_actions_ms": infer_record["sample_actions_ms"],
        "policy_time_ms": infer_record.get("policy_time_ms"),
        "serve_time_ms": infer_record.get("serve_time_ms"),
        "client_roundtrip_ms": infer_record.get("client_roundtrip_ms"),
        "total_ms": infer_record["total_ms"],
        "radius_dist_mean": infer_record["radius_dist_mean"],
        "did_prefill": int(infer_record["did_prefill"]),
        "is_full_pipeline_round": int(infer_record["is_full_pipeline_round"]),
        "used_full_fallback": int(infer_record["used_full_fallback"]),
        "scheduled_full_fallback": int(infer_record["scheduled_full_fallback"]),
        "reward": reward,
        "done_after_step": bool(done_after_step),
    }


def _count_by_key(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = str(record.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _ratio_dict(counts: Dict[str, int], *, denom: int) -> Dict[str, float]:
    if denom <= 0:
        return {str(k): 0.0 for k in counts}
    return {str(k): float(v) / float(denom) for k, v in sorted(counts.items())}


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
