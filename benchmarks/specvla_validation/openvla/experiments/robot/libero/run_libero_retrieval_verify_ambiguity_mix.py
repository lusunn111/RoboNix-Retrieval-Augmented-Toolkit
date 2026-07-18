"""
run_libero_retrieval_verify_ambiguity_mix.py

基于综合指标（曲率半径+位移）+ Ambiguity指标动态切换检索与AR生成。
使用Mix视角（第三人称+手腕相机）进行检索。
直接使用Qdrant进行检索，获取top-k结果计算ambiguity。

规则：
1. 首先用综合指标判断：
   - 综合指标 > composite_threshold (0.143210, 25%分位数)：初始判定为 noverify (使用DB检索)
   - 综合指标 <= composite_threshold：verify (使用AR生成)

2. 当综合指标判定为 noverify 时，再看 ambiguity 变化趋势来决定是否需要强制 verify：
   - ambiguity 上升 → 强制改为 verify (AR)
   - ambiguity 下降 → 继续 noverify (DB)
   - ambiguity 平稳 → 交替切换 (state ^= 1)

归一化参数（minmax归一化，超过范围取0或1）：
- 位移指标：[0.000009, 0.123381]
- 曲率半径：[0.000001, 0.014989]
- alpha = 0.5（1:1构造综合指标）

Usage:
    python experiments/robot/libero/run_libero_retrieval_verify_ambiguity_mix.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_goal \
        --center_crop True \
        --composite_threshold 0.143210 \
        --top_k 5 \
        --rise_threshold 0.01 \
        --fall_threshold 0.01 \
        --run_id_note <OPTIONAL TAG>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import hashlib
import base64
from io import BytesIO

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
import json
import requests
from PIL import Image
import time as time_module
import torch
from qdrant_client import QdrantClient
import logging

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
from experiments.robot.libero.calc_r import AmbiguityStateController, CompositeMetricsCalculator


# ============================================
# Qdrant 和 Embedding 配置
# ============================================
EMBEDDING_URL = "http://127.0.0.1:9021/predict"  # Mix embedding server
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection前缀
DATASET_CONFIGS = {
    "goal": "libero_goal_mix_task_",
    "10": "libero_10_mix_task_",
    "object": "libero_object_mix_task_",
    "spatial": "libero_spatial_mix_task_",
}


def normalize_dataset_type(name: str) -> Optional[str]:
    """Normalize dataset type name"""
    if not name:
        return None
    n = name.lower().strip()
    if n.startswith("libero_"):
        n = n.replace("libero_", "", 1)
    if n.endswith("_no_noops"):
        n = n[:-9]
    if n.endswith("_mix"):
        n = n[:-4]
    return n if n in DATASET_CONFIGS else None


def get_task_id(instruction: str) -> int:
    """根据instruction计算task_id（使用MD5哈希）"""
    instruction_lower = instruction.lower().strip()
    instruction_hash = int(hashlib.md5(instruction_lower.encode('utf-8')).hexdigest(), 16)
    task_id = instruction_hash % 1001
    return task_id


def generate_mix_embedding(third_person_image: Image.Image, wrist_image: Image.Image, 
                           instruction: str = "") -> Optional[torch.Tensor]:
    """
    通过远程服务器生成mix embedding
    """
    try:
        # Prepare third-person image
        buf_third = BytesIO()
        third_person_image.save(buf_third, format='PNG')
        buf_third.seek(0)
        
        # Prepare wrist image
        buf_wrist = BytesIO()
        wrist_image.save(buf_wrist, format='PNG')
        buf_wrist.seek(0)
        
        files = {
            "third_person_image": ("third_person.png", buf_third, "image/png"),
            "wrist_image": ("wrist.png", buf_wrist, "image/png")
        }
        data = {
            "instruction": instruction,
            "return_individual": "false"
        }
        
        response = requests.post(EMBEDDING_URL, files=files, data=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if "mix_features" in result:
            b64_string = result["mix_features"]
            binary_data = base64.b64decode(b64_string)
            buffer = BytesIO(binary_data)
            tensor = torch.load(buffer, map_location="cpu")
            return tensor.squeeze(0)
        else:
            print("No mix_features in embedding response")
            return None
            
    except Exception as e:
        print(f"Mix embedding generation failed: {e}")
        return None


def search_points(qdrant_client: QdrantClient, collection_name: str, 
                  query_vector: List[float], limit: int = 10):
    """Version-agnostic Qdrant search"""
    # 尝试旧版API (search)
    if hasattr(qdrant_client, "search"):
        try:
            return qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logging.warning(f"Legacy search failed, will try query_points: {e}")
    
    # 尝试新版API (query_points)
    try:
        result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        if hasattr(result, "points"):
            return result.points
        return result
    except Exception as e:
        logging.error(f"query_points fallback failed: {e}")
        raise


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
    # Composite metrics-based decision parameters (综合指标)
    #################################################################################################################
    window_size: int = 5
    composite_threshold: float = 0.143210  # 25%分位数阈值
    alpha: float = 0.5  # 1:1权重
    displacement_range_min: float = 0.000009
    displacement_range_max: float = 0.123381
    radius_range_min: float = 0.000001
    radius_range_max: float = 0.014989

    #################################################################################################################
    # Ambiguity-based decision parameters (用于替代"两步强制verify")
    #################################################################################################################
    top_k: int = 5  # 检索top-k用于计算ambiguity
    history_window: int = 3  # ambiguity变化趋势的历史窗口
    rise_threshold: float = 0.01  # ambiguity上升阈值
    fall_threshold: float = 0.01  # ambiguity下降阈值

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

    # 初始化Qdrant客户端
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # 获取数据集类型
    dataset_type = normalize_dataset_type(cfg.task_suite_name)
    if dataset_type is None:
        print(f"Unknown dataset type: {cfg.task_suite_name}, will try to use default")
        dataset_type = "goal"
    collection_prefix = DATASET_CONFIGS[dataset_type]
    print(f"Using collection prefix: {collection_prefix}")

    # Initialize local logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "../../../specdecoding/test-speed")
    target_dir = os.path.join(base_dir, "libero_retrieval_verify_ambiguity_mix")
    os.makedirs(target_dir, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-composite{cfg.composite_threshold}-topk{cfg.top_k}-{DATE_TIME}"
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
    
    # 打印配置信息
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    print("Experiment: Composite + Ambiguity Retrieval Verify (Mix View - Direct Qdrant)")
    log_file.write("Experiment: Composite + Ambiguity Retrieval Verify (Mix View - Direct Qdrant)\n")
    print(f"\n=== 综合指标参数 ===")
    print(f"Composite threshold: {cfg.composite_threshold} (25% percentile)")
    print(f"Alpha: {cfg.alpha} (1:1 ratio)")
    print(f"Displacement range: [{cfg.displacement_range_min}, {cfg.displacement_range_max}]")
    print(f"Radius range: [{cfg.radius_range_min}, {cfg.radius_range_max}]")
    log_file.write(f"\n=== 综合指标参数 ===\n")
    log_file.write(f"Composite threshold: {cfg.composite_threshold} (25% percentile)\n")
    log_file.write(f"Alpha: {cfg.alpha} (1:1 ratio)\n")
    log_file.write(f"Displacement range: [{cfg.displacement_range_min}, {cfg.displacement_range_max}]\n")
    log_file.write(f"Radius range: [{cfg.radius_range_min}, {cfg.radius_range_max}]\n")
    
    print(f"\n=== Ambiguity指标参数 ===")
    print(f"Top-K: {cfg.top_k}")
    print(f"History window: {cfg.history_window}")
    print(f"Rise threshold: {cfg.rise_threshold}")
    print(f"Fall threshold: {cfg.fall_threshold}")
    log_file.write(f"\n=== Ambiguity指标参数 ===\n")
    log_file.write(f"Top-K: {cfg.top_k}\n")
    log_file.write(f"History window: {cfg.history_window}\n")
    log_file.write(f"Rise threshold: {cfg.rise_threshold}\n")
    log_file.write(f"Fall threshold: {cfg.fall_threshold}\n")
    
    print(f"\n=== 决策逻辑 ===")
    print(f"1. 综合指标 > {cfg.composite_threshold} → 初始判定 noverify (DB)")
    print(f"2. 综合指标 <= {cfg.composite_threshold} → verify (AR)")
    print(f"3. 当判定为 noverify 时，根据 ambiguity 变化趋势调整：")
    print(f"   - ambiguity 上升 → 强制 verify (AR)")
    print(f"   - ambiguity 下降 → 继续 noverify (DB)")
    print(f"   - ambiguity 平稳 → 交替切换")
    log_file.write(f"\n=== 决策逻辑 ===\n")
    log_file.write(f"1. 综合指标 > {cfg.composite_threshold} → 初始判定 noverify (DB)\n")
    log_file.write(f"2. 综合指标 <= {cfg.composite_threshold} → verify (AR)\n")
    log_file.write(f"3. 当判定为 noverify 时，根据 ambiguity 变化趋势调整：\n")
    log_file.write(f"   - ambiguity 上升 → 强制 verify (AR)\n")
    log_file.write(f"   - ambiguity 下降 → 继续 noverify (DB)\n")
    log_file.write(f"   - ambiguity 平稳 → 交替切换\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # 预加载payload缓存
    print("\nPreloading payload cache...")
    payload_cache = {}

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_retrieval_data = []  # 存储所有检索验证数据
    all_accept_lengths = []  # 存储所有accept_length用于统计（仅AR模式）
    
    # 统计模式使用
    total_db_steps = 0
    total_ar_steps = 0
    total_ar_by_composite = 0  # 由综合指标触发的AR
    total_ar_by_ambiguity = 0  # 由ambiguity触发的AR（本应是DB但被改为AR）
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # 获取collection名
        qdrant_task_id = get_task_id(task_description)
        collection_name = f"{collection_prefix}{qdrant_task_id}"
        
        # 检查collection是否存在
        if not qdrant_client.collection_exists(collection_name):
            print(f"Collection {collection_name} does not exist, skipping task: {task_description}")
            log_file.write(f"Collection {collection_name} does not exist, skipping task: {task_description}\n")
            continue
        
        # 预加载这个collection的payload
        if collection_name not in payload_cache:
            print(f"Loading payloads from {collection_name}...")
            points_dict = {}
            offset = None
            while True:
                records, offset = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                if not records:
                    break
                for record in records:
                    points_dict[str(record.id)] = record.payload
                if offset is None:
                    break
            payload_cache[collection_name] = points_dict
            print(f"  Loaded {len(points_dict)} points")

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_retrieval_data = []  # 当前任务的检索数据
        task_accept_lengths = []  # 当前任务的accept_length列表
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            episode_retrieval_data = []  # 当前episode的检索数据
            
            # 综合指标计算器（每个episode重置）
            composite_calc = CompositeMetricsCalculator(
                window_size=cfg.window_size,
                displacement_range=(cfg.displacement_range_min, cfg.displacement_range_max),
                radius_range=(cfg.radius_range_min, cfg.radius_range_max),
            )
            
            # Ambiguity状态控制器（每个episode重置）
            amb_controller = AmbiguityStateController(
                history_window=cfg.history_window,
                rise_threshold=cfg.rise_threshold,
                fall_threshold=cfg.fall_threshold,
                initial_state=0  # 初始为noverify
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
            else:
                max_steps = 300

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
                    # 步骤1: 更新综合指标计算器
                    # ========================================
                    eef_position = obs["robot0_eef_pos"]  # shape: (3,)
                    composite_calc.update_history(eef_position)
                    
                    # 计算综合指标
                    composite_metric = composite_calc.compute_composite_metric(alpha=cfg.alpha)
                    metrics_info = composite_calc.get_current_metrics(alpha=cfg.alpha)

                    # ========================================
                    # 步骤2: 生成embedding并检索top-k
                    # ========================================
                    pil_third = Image.fromarray(img)
                    pil_wrist = Image.fromarray(wrist_img)
                    
                    t0_embedding = time_module.time()
                    embedding = generate_mix_embedding(pil_third, pil_wrist, task_description)
                    t1_embedding = time_module.time()
                    embedding_time = t1_embedding - t0_embedding
                    
                    top_k_actions = []
                    top_k_scores = []
                    retrieval_success = False
                    retrieval_time = 0.0
                    
                    if embedding is not None:
                        try:
                            t0_search = time_module.time()
                            search_results = search_points(
                                qdrant_client,
                                collection_name=collection_name,
                                query_vector=embedding.tolist(),
                                limit=cfg.top_k,
                            )
                            t1_search = time_module.time()
                            retrieval_time = t1_search - t0_search
                            
                            if search_results and len(search_results) > 0:
                                for result in search_results:
                                    top_k_scores.append(result.score)
                                    
                                    # 获取payload
                                    point_id = str(result.id)
                                    payload = payload_cache.get(collection_name, {}).get(point_id)
                                    if payload is None:
                                        payload = result.payload
                                    
                                    if payload and 'current_action' in payload:
                                        action = np.array(payload['current_action'])
                                        top_k_actions.append(action)
                                
                                if len(top_k_actions) > 0:
                                    retrieval_success = True
                        except Exception as e:
                            print(f"Search failed: {e}")
                            log_file.write(f"Search failed: {e}\n")

                    # ========================================
                    # 步骤3: 基于综合指标做初始判断
                    # ========================================
                    if np.isnan(composite_metric):
                        # 历史不足时默认使用AR
                        initial_decision = "verify"
                        decision_by_composite = "insufficient_history"
                    elif composite_metric > cfg.composite_threshold:
                        # 综合指标 > threshold: 初始判定为 noverify (DB)
                        initial_decision = "noverify"
                        decision_by_composite = f"composite={composite_metric:.4f}>{cfg.composite_threshold}"
                    else:
                        # 综合指标 <= threshold: verify (AR)
                        initial_decision = "verify"
                        decision_by_composite = f"composite={composite_metric:.4f}<={cfg.composite_threshold}"

                    # ========================================
                    # 步骤4: 当初始判定为 noverify 时，用 ambiguity 调整
                    # ========================================
                    final_decision = initial_decision
                    ambiguity_trend = "N/A"
                    current_ambiguity = np.nan
                    decision_by_ambiguity = "N/A"
                    
                    if initial_decision == "noverify":
                        # 计算 ambiguity 并更新状态
                        if len(top_k_actions) >= 2:
                            amb_state, ambiguity_trend, current_ambiguity = amb_controller.update_and_get_state(top_k_actions)
                            
                            # 根据 ambiguity 趋势决定是否改变决策
                            if ambiguity_trend == 'rising':
                                # ambiguity 上升 → 强制改为 verify
                                final_decision = "verify"
                                decision_by_ambiguity = f"rising→forced_verify"
                            elif ambiguity_trend == 'falling':
                                # ambiguity 下降 → 继续 noverify
                                final_decision = "noverify"
                                decision_by_ambiguity = f"falling→keep_noverify"
                            elif ambiguity_trend == 'stable':
                                # ambiguity 平稳 → 交替
                                # amb_state 已经在 update_and_get_state 中 ^= 1 了
                                final_decision = "verify" if amb_state == 1 else "noverify"
                                decision_by_ambiguity = f"stable→toggle({final_decision})"
                            else:
                                # initial 或其他情况
                                decision_by_ambiguity = f"{ambiguity_trend}→keep"
                        else:
                            # 检索结果不足，默认保持 noverify
                            decision_by_ambiguity = "insufficient_actions→keep_noverify"

                    # ========================================
                    # 步骤5: 根据最终决策执行
                    # ========================================
                    accept_length = -1
                    generation_time = 0.0
                    retrieved_action = None
                    
                    if final_decision == "noverify" and retrieval_success and len(top_k_actions) > 0:
                        # DB模式：直接使用top1检索结果
                        total_db_steps += 1
                        retrieved_action = top_k_actions[0].copy()
                        action = retrieved_action
                        mode = "DB"
                    else:
                        # AR模式：使用AR生成
                        total_ar_steps += 1
                        mode = "AR"
                        
                        # 统计是由综合指标还是ambiguity触发的AR
                        if initial_decision == "verify":
                            total_ar_by_composite += 1
                        else:
                            total_ar_by_ambiguity += 1
                        
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
                        
                        # 如果有检索结果，计算accept_length（用于统计）
                        if retrieval_success and len(top_k_actions) > 0:
                            try:
                                retrieved_action = top_k_actions[0]
                                retrieved_tokens = action_to_tokens(retrieved_action, model, cfg.unnorm_key)
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
                                
                                task_accept_lengths.append(accept_length)
                            except Exception as e:
                                print(f"Accept length calculation error: {e}")
                                log_file.write(f"Accept length calculation error: {e}\n")
                                accept_length = -1
                    
                    # 记录这一步的检索验证数据
                    step_data = {
                        'episode': task_episodes,
                        'step': t - cfg.num_steps_wait,
                        'mode': mode,
                        
                        # 综合指标
                        'composite_metric': float(composite_metric) if not np.isnan(composite_metric) else None,
                        'raw_radius': float(metrics_info['raw']['radius']) if not np.isnan(metrics_info['raw']['radius']) else None,
                        'raw_displacement': float(metrics_info['raw']['displacement']) if not np.isnan(metrics_info['raw']['displacement']) else None,
                        'norm_radius': float(metrics_info['normalized']['radius']) if not np.isnan(metrics_info['normalized']['radius']) else None,
                        'norm_displacement': float(metrics_info['normalized']['displacement']) if not np.isnan(metrics_info['normalized']['displacement']) else None,
                        
                        # 决策过程
                        'initial_decision': initial_decision,
                        'decision_by_composite': decision_by_composite,
                        'final_decision': final_decision,
                        'decision_by_ambiguity': decision_by_ambiguity,
                        
                        # Ambiguity 相关
                        'ambiguity': float(current_ambiguity) if not np.isnan(current_ambiguity) else None,
                        'ambiguity_trend': ambiguity_trend,
                        
                        # 检索相关
                        'top_k_scores': [float(s) for s in top_k_scores],
                        'retrieval_success': retrieval_success,
                        'num_retrieved_actions': len(top_k_actions),
                        
                        # 时间统计
                        'embedding_time': embedding_time,
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                        
                        # Accept Length (仅 AR 模式)
                        'accept_length': accept_length,
                    }
                    episode_retrieval_data.append(step_data)

                    # Normalize gripper action
                    action = normalize_gripper_action(action, binarize=True)
                    
                    # Invert gripper action for OpenVLA (only for AR mode)
                    if mode == "AR" and cfg.model_family == "openvla":
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
        total_steps_in_task = sum(len(ep['steps']) for ep in task_retrieval_data)
        
        # 分别统计各模式
        db_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'DB')
            for ep in task_retrieval_data
        )
        ar_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'AR')
            for ep in task_retrieval_data
        )
        
        # 统计决策来源
        ar_by_composite_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'AR' and step['initial_decision'] == 'verify')
            for ep in task_retrieval_data
        )
        ar_by_ambiguity_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'AR' and step['initial_decision'] == 'noverify')
            for ep in task_retrieval_data
        )
        
        # 统计trend分布
        trend_counts = {'rising': 0, 'falling': 0, 'stable': 0, 'initial': 0, 'N/A': 0}
        for ep in task_retrieval_data:
            for step in ep['steps']:
                trend = step.get('ambiguity_trend', 'N/A')
                if trend in trend_counts:
                    trend_counts[trend] += 1
        
        print(f"\nTask {task_id} Statistics:")
        print(f"  Total steps: {total_steps_in_task}")
        print(f"  DB mode steps (noverify): {db_steps_count}")
        print(f"  AR mode steps (verify): {ar_steps_count}")
        print(f"    - AR by composite: {ar_by_composite_count}")
        print(f"    - AR by ambiguity: {ar_by_ambiguity_count}")
        print(f"  Ambiguity trend distribution: {trend_counts}")
        if len(task_accept_lengths) > 0:
            print(f"  Accept Length Stats:")
            print(f"    Mean: {np.mean(task_accept_lengths):.2f}")
            print(f"    Median: {np.median(task_accept_lengths):.2f}")
        
        log_file.write(f"\nTask {task_id} Statistics:\n")
        log_file.write(f"  Total steps: {total_steps_in_task}\n")
        log_file.write(f"  DB mode steps (noverify): {db_steps_count}\n")
        log_file.write(f"  AR mode steps (verify): {ar_steps_count}\n")
        log_file.write(f"    - AR by composite: {ar_by_composite_count}\n")
        log_file.write(f"    - AR by ambiguity: {ar_by_ambiguity_count}\n")
        log_file.write(f"  Ambiguity trend distribution: {trend_counts}\n")
        if len(task_accept_lengths) > 0:
            log_file.write(f"  Accept Length Stats:\n")
            log_file.write(f"    Mean: {np.mean(task_accept_lengths):.2f}\n")
            log_file.write(f"    Median: {np.median(task_accept_lengths):.2f}\n")
        
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
    
    # 总体DB/AR统计
    all_db_steps = sum(
        sum(1 for step in ep['steps'] if step['mode'] == 'DB')
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
    )
    all_ar_steps = sum(
        sum(1 for step in ep['steps'] if step['mode'] == 'AR')
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
    )
    total_all_steps = all_db_steps + all_ar_steps
    
    # 统计AR来源
    all_ar_by_composite = sum(
        sum(1 for step in ep['steps'] if step['mode'] == 'AR' and step['initial_decision'] == 'verify')
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
    )
    all_ar_by_ambiguity = sum(
        sum(1 for step in ep['steps'] if step['mode'] == 'AR' and step['initial_decision'] == 'noverify')
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
    )
    
    # 总体trend分布
    overall_trend_counts = {'rising': 0, 'falling': 0, 'stable': 0, 'initial': 0, 'N/A': 0}
    for task_data in all_retrieval_data:
        for ep in task_data['episodes']:
            for step in ep['steps']:
                trend = step.get('ambiguity_trend', 'N/A')
                if trend in overall_trend_counts:
                    overall_trend_counts[trend] += 1
    
    print(f"\nMode Statistics:")
    print(f"  Total DB steps (noverify/检索): {all_db_steps} ({100.0*all_db_steps/total_all_steps:.1f}%)" if total_all_steps > 0 else "  Total DB steps: 0")
    print(f"  Total AR steps (verify): {all_ar_steps} ({100.0*all_ar_steps/total_all_steps:.1f}%)" if total_all_steps > 0 else "  Total AR steps: 0")
    print(f"    - AR triggered by composite: {all_ar_by_composite} ({100.0*all_ar_by_composite/total_all_steps:.1f}%)" if total_all_steps > 0 else "")
    print(f"    - AR triggered by ambiguity: {all_ar_by_ambiguity} ({100.0*all_ar_by_ambiguity/total_all_steps:.1f}%)" if total_all_steps > 0 else "")
    
    print(f"\nAmbiguity Trend Distribution (only for noverify initial decisions):")
    for trend, count in overall_trend_counts.items():
        if trend != 'N/A':
            print(f"  {trend}: {count}")
    
    print(f"\n  ============ 关键比例 ============")
    print(f"  verify (AR) : noverify (DB) = {all_ar_steps}:{all_db_steps}")
    if all_db_steps > 0:
        print(f"  verify:noverify 比值 = {all_ar_steps/all_db_steps:.2f}:1")
    if all_ar_steps > 0:
        print(f"  noverify:verify 比值 = {all_db_steps/all_ar_steps:.2f}:1")
    print(f"  ==================================")
    
    # ============ 时间统计 ============
    all_embedding_times = []
    all_retrieval_times = []
    all_generation_times = []
    
    for task_data in all_retrieval_data:
        for ep in task_data['episodes']:
            for step in ep['steps']:
                if step['embedding_time'] > 0:
                    all_embedding_times.append(step['embedding_time'])
                if step['retrieval_time'] > 0:
                    all_retrieval_times.append(step['retrieval_time'])
                if step['mode'] == 'AR' and step['generation_time'] > 0:
                    all_generation_times.append(step['generation_time'])
    
    print(f"\n  ============ 时间统计 ============")
    if len(all_embedding_times) > 0:
        avg_embedding_time = np.mean(all_embedding_times)
        std_embedding_time = np.std(all_embedding_times)
        print(f"  Embedding生成时间:")
        print(f"    Mean: {avg_embedding_time*1000:.2f} ms")
        print(f"    Std:  {std_embedding_time*1000:.2f} ms")
    else:
        avg_embedding_time = 0
    
    if len(all_retrieval_times) > 0:
        avg_retrieval_time = np.mean(all_retrieval_times)
        std_retrieval_time = np.std(all_retrieval_times)
        print(f"  Qdrant检索时间 (top-{cfg.top_k}):")
        print(f"    Mean: {avg_retrieval_time*1000:.2f} ms")
        print(f"    Std:  {std_retrieval_time*1000:.2f} ms")
    else:
        avg_retrieval_time = 0
    
    if len(all_generation_times) > 0:
        avg_generation_time = np.mean(all_generation_times)
        std_generation_time = np.std(all_generation_times)
        print(f"  AR生成时间 (get_action):")
        print(f"    Mean: {avg_generation_time*1000:.2f} ms")
        print(f"    Std:  {std_generation_time*1000:.2f} ms")
    else:
        avg_generation_time = 0
    
    # 计算平均每步时间（加权）
    if total_all_steps > 0:
        db_time = avg_embedding_time + avg_retrieval_time
        ar_time = avg_embedding_time + avg_retrieval_time + avg_generation_time
        weighted_avg_time = (all_db_steps * db_time + all_ar_steps * ar_time) / total_all_steps
        print(f"\n  加权平均每步时间: {weighted_avg_time*1000:.2f} ms")
        if avg_generation_time > 0:
            pure_ar_time = avg_generation_time
            speedup = pure_ar_time / weighted_avg_time if weighted_avg_time > 0 else 0
            print(f"  相对纯AR加速比: {speedup:.2f}x")
    print(f"  ==================================")
    
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
    
    # 写入日志文件
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("Overall Statistics:\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Total episodes: {total_episodes}\n")
    log_file.write(f"Total successes: {total_successes}\n")
    log_file.write(f"Success rate: {total_successes/total_episodes*100:.1f}%\n")
    log_file.write(f"\nMode Statistics:\n")
    log_file.write(f"  Total DB steps (noverify/检索): {all_db_steps} ({100.0*all_db_steps/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "  Total DB steps: 0\n")
    log_file.write(f"  Total AR steps (verify): {all_ar_steps} ({100.0*all_ar_steps/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "  Total AR steps: 0\n")
    log_file.write(f"    - AR triggered by composite: {all_ar_by_composite} ({100.0*all_ar_by_composite/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "")
    log_file.write(f"    - AR triggered by ambiguity: {all_ar_by_ambiguity} ({100.0*all_ar_by_ambiguity/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "")
    
    log_file.write(f"\nAmbiguity Trend Distribution:\n")
    for trend, count in overall_trend_counts.items():
        if trend != 'N/A':
            log_file.write(f"  {trend}: {count}\n")
    
    log_file.write(f"\n  ============ 关键比例 ============\n")
    log_file.write(f"  verify (AR) : noverify (DB) = {all_ar_steps}:{all_db_steps}\n")
    if all_db_steps > 0:
        log_file.write(f"  verify:noverify 比值 = {all_ar_steps/all_db_steps:.2f}:1\n")
    if all_ar_steps > 0:
        log_file.write(f"  noverify:verify 比值 = {all_db_steps/all_ar_steps:.2f}:1\n")
    log_file.write(f"  ==================================\n")
    
    # 时间统计写入日志
    log_file.write(f"\n  ============ 时间统计 ============\n")
    if len(all_embedding_times) > 0:
        log_file.write(f"  Embedding生成时间:\n")
        log_file.write(f"    Mean: {avg_embedding_time*1000:.2f} ms\n")
        log_file.write(f"    Std:  {std_embedding_time*1000:.2f} ms\n")
    
    if len(all_retrieval_times) > 0:
        log_file.write(f"  Qdrant检索时间 (top-{cfg.top_k}):\n")
        log_file.write(f"    Mean: {avg_retrieval_time*1000:.2f} ms\n")
        log_file.write(f"    Std:  {std_retrieval_time*1000:.2f} ms\n")
    
    if len(all_generation_times) > 0:
        log_file.write(f"  AR生成时间 (get_action):\n")
        log_file.write(f"    Mean: {avg_generation_time*1000:.2f} ms\n")
        log_file.write(f"    Std:  {std_generation_time*1000:.2f} ms\n")
    
    if total_all_steps > 0:
        log_file.write(f"\n  加权平均每步时间: {weighted_avg_time*1000:.2f} ms\n")
        if avg_generation_time > 0:
            log_file.write(f"  相对纯AR加速比: {speedup:.2f}x\n")
    log_file.write(f"  ==================================\n")
    
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
