"""
run_libero_block_sd.py

Block-wise Speculative Decoding: 使用检索的 Top-K 候选作为 Draft，
通过 Target Model 进行 Block 级别的联合概率验证。

Block 划分：
- Block 1: [0, 1, 2] - 位置 (x, y, z)
- Block 2: [3, 4, 5] - 姿态 (roll, pitch, yaw)
- Block 3: [6] - 夹爪状态

验证策略：
- 使用 tree_mask 并行验证 K 条候选链
- 基于联合概率判断 Block 是否通过
- Block 通过 → 接受该 Block
- Block 失败 → AR 生成该 Block

Usage:
    python experiments/robot/libero/run_libero_block_sd.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name libero_goal \
        --top_k 5 \
        --prob_threshold 0.001 \
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

try:
    import wandb  # noqa: F401
except Exception:
    wandb = None
import json
import requests
from PIL import Image
import time as time_module
import torch
from qdrant_client import QdrantClient
import logging


# ============================================
# 自定义 JSON 编码器，处理 numpy 类型
# ============================================
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif torch.is_tensor(obj):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
        return super().default(obj)


class ComponentProfiler:
    """Lightweight CUDA-aware forward-hook profiler for per-step component timing."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.times = {}
        self._starts = {}
        self._handles = []
        self._method_wrappers = []

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset(self):
        self.times = {}
        self._starts = {}

    def register(self, name: str, module):
        if not self.enabled or module is None:
            return

        def pre_hook(_module, _inputs):
            self._sync()
            self._starts.setdefault(name, []).append(time_module.perf_counter())

        def post_hook(_module, _inputs, _outputs):
            self._sync()
            starts = self._starts.get(name, [])
            if starts:
                start = starts.pop()
                self.times[name] = self.times.get(name, 0.0) + (time_module.perf_counter() - start)

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def wrap_method(self, name: str, obj, method_name: str):
        if not self.enabled or obj is None or not hasattr(obj, method_name):
            return
        original = getattr(obj, method_name)

        def wrapped(*args, **kwargs):
            self._sync()
            start = time_module.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self._sync()
                self.times[name] = self.times.get(name, 0.0) + (time_module.perf_counter() - start)

        setattr(obj, method_name, wrapped)
        self._method_wrappers.append((obj, method_name, original))

    def snapshot(self):
        return dict(self.times)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        for obj, method_name, original in self._method_wrappers:
            setattr(obj, method_name, original)
        self._method_wrappers = []


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
from experiments.robot.libero.calc_r import CompositeMetricsCalculator


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

# Block 配置
BLOCKS = [
    {'name': 'position', 'indices': [0, 1, 2]},
    {'name': 'orientation', 'indices': [3, 4, 5]},
    {'name': 'gripper', 'indices': [6]},
]


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


def parse_task_ids(task_ids: Optional[str], num_tasks: int) -> List[int]:
    """Parse task id specs like "0", "0,2,5", or "0-3"."""
    if task_ids is None or str(task_ids).strip() == "":
        return list(range(num_tasks))

    selected = []
    for part in str(task_ids).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid task range: {token}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(token))

    unique_selected = []
    for task_id in selected:
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(f"Task id {task_id} out of range [0, {num_tasks - 1}]")
        if task_id not in unique_selected:
            unique_selected.append(task_id)
    return unique_selected


def safe_float(value):
    """Convert numpy/python scalars to JSON-friendly floats, keeping NaN as None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def payload_to_action_trajectory(payload):
    """Return [current_action] + next_actions from an RT-Cache payload."""
    if not payload or "current_action" not in payload:
        return None

    actions = [np.asarray(payload["current_action"], dtype=np.float64)]
    next_actions = payload.get("next_actions") or []
    for action in next_actions:
        actions.append(np.asarray(action, dtype=np.float64))

    valid_actions = []
    for action in actions:
        if action.ndim != 1 or action.shape[0] < 3:
            continue
        valid_actions.append(action)
    if not valid_actions:
        return None
    return np.stack(valid_actions, axis=0)


def annotate_episode_trajectory_overlap(episode_data, overlap_eps=0.01):
    """Add DB-vs-executed action trajectory overlap diagnostics after an episode."""
    executed_actions = []
    for step in episode_data:
        action = step.get("final_action_model_space")
        if isinstance(action, list) and len(action) >= 3:
            executed_actions.append(np.asarray(action, dtype=np.float64))
        else:
            executed_actions.append(None)

    for idx, step in enumerate(episode_data):
        action_traj = step.get("top1_retrieved_action_trajectory")
        if action_traj is None:
            continue

        future_available = len(executed_actions) - idx
        if future_available <= 0:
            continue

        try:
            action_traj_arr = np.asarray(action_traj, dtype=np.float64)
        except Exception:
            continue
        if action_traj_arr.ndim != 2 or action_traj_arr.shape[1] < 3:
            continue

        horizon = min(action_traj_arr.shape[0], future_available)
        if horizon <= 0:
            continue

        exec_slice = executed_actions[idx: idx + horizon]
        if any(point is None for point in exec_slice):
            continue
        exec_traj = np.stack(exec_slice, axis=0)
        db_traj = action_traj_arr[:horizon]
        if db_traj.shape != exec_traj.shape:
            continue

        distances = np.linalg.norm(exec_traj[:, :3] - db_traj[:, :3], axis=1)
        full_distances = np.linalg.norm(exec_traj - db_traj, axis=1)
        step["trajectory_space"] = "action_xyz"
        step["trajectory_overlap_horizon"] = int(horizon)
        step["trajectory_mean_deviation"] = safe_float(np.mean(distances))
        step["trajectory_max_deviation"] = safe_float(np.max(distances))
        step["trajectory_endpoint_error"] = safe_float(distances[-1])
        step["trajectory_action_l2_mean_deviation"] = safe_float(np.mean(full_distances))
        step["trajectory_action_l2_endpoint_error"] = safe_float(full_distances[-1])
        step["trajectory_overlap_ratio"] = safe_float(np.mean(distances <= overlap_eps))
        step["trajectory_overlap_eps"] = safe_float(overlap_eps)
        step["executed_action_trajectory"] = exec_traj.tolist()
        step["retrieved_action_trajectory_used"] = db_traj.tolist()


def set_action_diagnostics(step_stat, top_k_actions, final_action, model, unnorm_key, include_arrays=False):
    """Add top-1 retrieval vs executed-model action diagnostics for overlap analysis."""
    if len(top_k_actions) == 0 or final_action is None:
        return

    try:
        top1_action = np.asarray(top_k_actions[0], dtype=np.float64)
        final_action_arr = np.asarray(final_action, dtype=np.float64)
        if top1_action.shape != final_action_arr.shape:
            return

        diff = top1_action - final_action_arr
        step_stat["top1_action_l2_error"] = safe_float(np.linalg.norm(diff))
        step_stat["top1_action_l1_error"] = safe_float(np.mean(np.abs(diff)))
        if top1_action.shape[0] >= 3:
            step_stat["top1_action_pos_l2_error"] = safe_float(np.linalg.norm(diff[:3]))

        try:
            top1_tokens = action_to_tokens(top1_action, model, unnorm_key)
            final_tokens = action_to_tokens(final_action_arr, model, unnorm_key)
            token_abs_diff = np.abs(top1_tokens - final_tokens)
            step_stat["top1_token_diff_sum"] = int(np.sum(token_abs_diff))
            step_stat["top1_token_diff_max"] = int(np.max(token_abs_diff))
            step_stat["top1_token_match_count"] = int(np.sum(token_abs_diff == 0))
        except Exception:
            pass

        if include_arrays:
            step_stat["top1_action"] = top1_action.tolist()
            step_stat["final_action_model_space"] = final_action_arr.tolist()
    except Exception as e:
        step_stat["action_diagnostic_error"] = str(e)


def infer_accepted_length(step_stat):
    """Normalize different verify modes into one accepted_length field."""
    mode = step_stat.get("mode", "")
    if mode in ["Retrieval_DB", "Ablation2_Noverify", "Retrieval_BlockSD_fully_verified"]:
        return 7
    if mode == "Retrieval_BlockSD_partial_AR":
        return step_stat.get("accepted_tokens", 0)
    if mode == "SD":
        return step_stat.get("sd_accept_length")
    if mode in [
        "AR_retrieval_failed",
        "AR_insufficient_history",
        "Ablation1_AR",
        "Ablation2_AR_verify",
        "AR_unknown_ablation",
        "Retrieval_BlockSD_error_AR",
    ]:
        return 1
    return step_stat.get("accepted_tokens")


def generate_mix_embedding(third_person_image: Image.Image, wrist_image: Image.Image, 
                           instruction: str = "") -> Optional[torch.Tensor]:
    """通过远程服务器生成mix embedding"""
    try:
        buf_third = BytesIO()
        third_person_image.save(buf_third, format='PNG')
        buf_third.seek(0)
        
        buf_wrist = BytesIO()
        wrist_image.save(buf_wrist, format='PNG')
        buf_wrist.seek(0)
        
        files = {
            "third_person_image": ("third_person.png", buf_third, "image/png"),
            "wrist_image": ("wrist.png", buf_wrist, "image/png")
        }
        data = {"instruction": instruction, "return_individual": "false"}
        
        response = requests.post(EMBEDDING_URL, files=files, data=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "mix_features" in result:
            b64_string = result["mix_features"]
            binary_data = base64.b64decode(b64_string)
            buffer = BytesIO(binary_data)
            tensor = torch.load(buffer, map_location="cpu")
            return tensor.squeeze(0)
        return None
    except Exception as e:
        print(f"Mix embedding generation failed: {e}")
        return None


def search_points(qdrant_client: QdrantClient, collection_name: str, 
                  query_vector: List[float], limit: int = 10):
    """Version-agnostic Qdrant search"""
    if hasattr(qdrant_client, "search"):
        try:
            return qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            pass
    
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


def action_to_tokens(action, model, unnorm_key):
    """将连续的action转换为token IDs"""
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


def tokens_to_action(token_ids, model, unnorm_key):
    """将token IDs转换回连续的action"""
    vocab_size = model.vocab_size
    discretized_actions = vocab_size - token_ids - 1
    discretized_actions = np.clip(discretized_actions, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    
    normalized_actions = model.bin_centers[discretized_actions]
    
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    
    return actions


def build_block_tree_mask(K: int, block_size: int) -> torch.Tensor:
    """
    构建 Block 级别的 tree attention mask
    
    K 条候选链并行，链内 causal，链间不可见
    """
    total_len = K * block_size
    mask = torch.zeros(total_len, total_len)
    
    for k in range(K):
        start = k * block_size
        for i in range(block_size):
            for j in range(i + 1):
                mask[start + i, start + j] = 1
    
    return mask.unsqueeze(0).unsqueeze(0)


def build_block_position_ids(K: int, block_size: int, prefix_len: int, device) -> torch.Tensor:
    """构建 position ids，所有候选的同一位置共享 position"""
    base_positions = torch.arange(prefix_len, prefix_len + block_size, device=device)
    position_ids = base_positions.repeat(K)
    return position_ids.unsqueeze(0)


def evaluate_posterior_block_joint_prob(
    logits: torch.Tensor,      # [K, block_size, vocab_size]
    candidates: torch.Tensor,  # [K, block_size]
    prob_threshold: float = 0.001,
    use_avg_prob: bool = True,
):
    """
    基于联合概率的 Block 验证
    
    Args:
        logits: Target model 输出 [K, block_size, vocab_size]
        candidates: Draft tokens [K, block_size]
        prob_threshold: 联合概率阈值
        use_avg_prob: 是否使用平均概率（几何平均）而非联合概率
    
    Returns:
        best_candidate: 最佳候选索引
        best_prob: 最佳概率值
        block_passed: 是否通过
        all_probs: 所有候选的概率
    """
    K, block_size, vocab_size = logits.shape
    
    # 计算 softmax 概率
    probs = torch.softmax(logits, dim=-1)  # [K, block_size, vocab_size]
    
    # 取出 draft token 对应的概率
    token_probs = torch.gather(
        probs, dim=-1, index=candidates.unsqueeze(-1).to(logits.device)
    ).squeeze(-1)  # [K, block_size]
    
    # 计算 log 联合概率
    log_token_probs = torch.log(token_probs + 1e-10)  # [K, block_size]
    log_joint_probs = log_token_probs.sum(dim=1)  # [K]
    
    if use_avg_prob:
        # 使用几何平均（归一化到每个 token）
        avg_log_probs = log_joint_probs / block_size
        probs_to_compare = torch.exp(avg_log_probs)
    else:
        # 使用联合概率
        probs_to_compare = torch.exp(log_joint_probs)
    
    # 选择最佳候选
    best_prob, best_candidate = probs_to_compare.max(dim=0)
    
    # 判断是否通过
    block_passed = (best_prob > prob_threshold)
    
    return best_candidate.item(), best_prob.item(), block_passed, probs_to_compare.cpu().numpy()


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
    # 综合指标参数 (决定使用 Retrieval 策略还是 SD 策略)
    #################################################################################################################
    window_size: int = 5  # 滑动窗口大小
    composite_threshold: float = 0.143210  # 综合指标阈值
    alpha: float = 0.5  # 曲率权重 (1:1)
    displacement_range_min: float = 0.000009
    displacement_range_max: float = 0.139051
    radius_range_min: float = 0.000001
    radius_range_max: float = 0.016873

    #################################################################################################################
    # Block SD parameters (用于 Retrieval 策略的 2:1 中的验证部分)
    #################################################################################################################
    top_k: int = 5  # 检索候选数量
    prob_threshold: float = 0.1  # (已弃用) Block 验证概率阈值
    use_avg_prob: bool = True  # (已弃用) 使用几何平均概率
    
    # Block 差值验证阈值 (新版) - 放松阈值以提高接受率
    block_sum_threshold: int = 45  # α: Block 内 token 差值之和的阈值 (15*3)
    block_max_threshold: int = 25  # μ: Block 内单个 token 差值的阈值
    # 接受条件: sum(diff) < α AND max(diff) < μ
    
    # Block 特定阈值 (可选，如果设置则覆盖 prob_threshold)
    prob_threshold_position: Optional[float] = None
    prob_threshold_orientation: Optional[float] = None
    prob_threshold_gripper: Optional[float] = None

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Union[str, Path] = "/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    task_ids: Optional[str] = None
    save_videos: bool = True

    #################################################################################################################
    # 消融实验模式
    #################################################################################################################
    # ablation_mode:
    #   1 = 只有阈值分割: 阈值之下 SD，阈值之上每步 AR（不使用 noverify）
    #   2 = + Verify-Skip: 阈值之下 SD，阈值之上 2:1（2步 noverify + 1步 AR）
    #   3 = + Seq-Wise: 阈值之下 SD，阈值之上 2:1（2步 noverify + 1步 BlockSD verify）[主实验]
    ablation_mode: int = 3

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 7
    mmrebuttal_record_step_metrics: bool = False
    mmrebuttal_output_dir: Optional[str] = None
    mmrebuttal_overlap_eps: float = 0.01
    mmrebuttal_profile_component_times: bool = False

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert cfg.use_spec, "cfg.use_spec must be True for Block SD!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    print("Loading SpecVLA model for Block SD...")
    model = get_model(cfg)
    component_profiler = ComponentProfiler(enabled=cfg.mmrebuttal_profile_component_times)
    if cfg.mmrebuttal_profile_component_times:
        base_model = getattr(model, "base_model", None)
        component_profiler.register("vit", getattr(base_model, "vision_backbone", None))
        component_profiler.register("llm", getattr(base_model, "language_model", None))
        component_profiler.wrap_method("draft_model", getattr(model, "ea_layer", None), "topK_genrate")
        print("MMRebuttal component profiling enabled: ViT/LLM hooks and Draft Model method timer registered.")

    # [OpenVLA] Check action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found!"

    # [OpenVLA] Get Hugging Face processor
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    # 初始化 Qdrant 客户端
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # 获取数据集类型
    dataset_type = normalize_dataset_type(cfg.task_suite_name)
    if dataset_type is None:
        dataset_type = "goal"
    collection_prefix = DATASET_CONFIGS[dataset_type]
    print(f"Using collection prefix: {collection_prefix}")

    # 设置 block 阈值
    block_thresholds = {
        'position': cfg.prob_threshold_position if cfg.prob_threshold_position else cfg.prob_threshold,
        'orientation': cfg.prob_threshold_orientation if cfg.prob_threshold_orientation else cfg.prob_threshold,
        'gripper': cfg.prob_threshold_gripper if cfg.prob_threshold_gripper else cfg.prob_threshold,
    }

    # Initialize local logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "../../../specdecoding/test-speed")
    if cfg.mmrebuttal_record_step_metrics and cfg.mmrebuttal_output_dir:
        target_dir = os.path.abspath(os.path.expanduser(str(cfg.mmrebuttal_output_dir)))
    else:
        target_dir = os.path.join(base_dir, "libero_block_sd")
    os.makedirs(target_dir, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-Ablation{cfg.ablation_mode}-topk{cfg.top_k}-sum{cfg.block_sum_threshold}-max{cfg.block_max_threshold}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    local_log_datapath = os.path.join(target_dir, run_id + "_block_sd.json")
    print(f"Logging to: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    selected_task_ids = parse_task_ids(cfg.task_ids, num_tasks_in_suite)

    # 打印配置
    ablation_mode_names = {
        1: "只有阈值分割 (阈值之上: 每步AR)",
        2: "+ Verify-Skip (阈值之上: 2:1, 2步noverify + 1步AR)",
        3: "+ Seq-Wise (阈值之上: 2:1, 2步noverify + 1步BlockSD) [主实验]",
    }
    print(f"\n{'='*60}")
    print(f"Block SD Configuration")
    print(f"{'='*60}")
    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Selected tasks: {selected_task_ids}")
    print(f"Ablation Mode: {cfg.ablation_mode} - {ablation_mode_names.get(cfg.ablation_mode, 'Unknown')}")
    print(f"Top-K: {cfg.top_k}")
    print(f"Block 验证阈值 (Token ID 差值):")
    print(f"  BLOCK_SUM_THRESHOLD (α): {cfg.block_sum_threshold}")
    print(f"  BLOCK_MAX_THRESHOLD (μ): {cfg.block_max_threshold}")
    if cfg.block_sum_threshold < 0 or cfg.block_max_threshold < 0:
        print(f"  模式: 纯 AR 基线 (跳过验证)")
    else:
        print(f"  模式: Block 验证 (sum < {cfg.block_sum_threshold} AND max < {cfg.block_max_threshold})")
    print(f"Composite threshold: {cfg.composite_threshold}")
    print(f"{'='*60}\n")
    
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Selected tasks: {selected_task_ids}\n")
    log_file.write(f"Ablation Mode: {cfg.ablation_mode} - {ablation_mode_names.get(cfg.ablation_mode, 'Unknown')}\n")
    log_file.write(f"Top-K: {cfg.top_k}\n")
    log_file.write(f"Block thresholds: sum={cfg.block_sum_threshold}, max={cfg.block_max_threshold}\n")
    log_file.write(f"Composite threshold: {cfg.composite_threshold}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # 预加载 payload 缓存
    payload_cache = {}

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_data = []
    
    # 统计
    total_block_verified = {'position': 0, 'orientation': 0, 'gripper': 0}
    total_block_ar = {'position': 0, 'orientation': 0, 'gripper': 0}
    total_full_verify = 0  # 三个 block 都通过的 step 数
    
    for task_id in tqdm.tqdm(selected_task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # 获取 collection 名
        qdrant_task_id = get_task_id(task_description)
        collection_name = f"{collection_prefix}{qdrant_task_id}"
        
        if not qdrant_client.collection_exists(collection_name):
            print(f"Collection {collection_name} does not exist, skipping: {task_description}")
            continue
        
        # 预加载 payload
        if collection_name not in payload_cache:
            print(f"Loading payloads from {collection_name}...")
            points_dict = {}
            offset = None
            while True:
                records, offset = qdrant_client.scroll(
                    collection_name=collection_name, limit=100, offset=offset,
                    with_payload=True, with_vectors=False
                )
                if not records:
                    break
                for record in records:
                    points_dict[str(record.id)] = record.payload
                if offset is None:
                    break
            payload_cache[collection_name] = points_dict
            print(f"  Loaded {len(points_dict)} points")

        task_episodes, task_successes = 0, 0
        task_data = []
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            episode_data = []
            
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            
            # ========================================
            # 初始化综合指标计算器（每个 episode 重置）
            # ========================================
            metrics_calc = CompositeMetricsCalculator(
                window_size=cfg.window_size,
                displacement_range=(cfg.displacement_range_min, cfg.displacement_range_max),
                radius_range=(cfg.radius_range_min, cfg.radius_range_max),
            )
            
            # 2:1 策略计数器
            retrieval_consecutive_db_count = 0

            t = 0
            replay_images = []
            
            # 设置 max_steps
            max_steps_map = {
                "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
                "libero_10": 520, "libero_90": 400
            }
            max_steps = max_steps_map.get(cfg.task_suite_name, 300)

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    wrist_img = get_libero_wrist_image(obs, resize_size)
                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "wrist_image": wrist_img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # ========================================
                    # Step 1: 检索 Top-K 候选
                    # ========================================
                    pil_third = Image.fromarray(img)
                    pil_wrist = Image.fromarray(wrist_img)
                    
                    t0_embedding = time_module.time()
                    embedding = generate_mix_embedding(pil_third, pil_wrist, task_description)
                    embedding_time = time_module.time() - t0_embedding
                    
                    top_k_actions = []
                    top_k_scores = []
                    top_k_trajectories = []
                    top_k_payload_refs = []
                    retrieval_success = False
                    retrieval_time = 0.0
                    
                    if embedding is not None:
                        try:
                            t0_search = time_module.time()
                            search_results = search_points(
                                qdrant_client, collection_name=collection_name,
                                query_vector=embedding.tolist(), limit=cfg.top_k,
                            )
                            retrieval_time = time_module.time() - t0_search
                            
                            if search_results and len(search_results) > 0:
                                for result in search_results:
                                    top_k_scores.append(result.score)
                                    point_id = str(result.id)
                                    payload = payload_cache.get(collection_name, {}).get(point_id)
                                    if payload is None:
                                        payload = result.payload
                                    action_traj = payload_to_action_trajectory(payload)
                                    if action_traj is not None:
                                        action = action_traj[0]
                                        top_k_actions.append(action)
                                        top_k_trajectories.append(action_traj)
                                        top_k_payload_refs.append({
                                            "point_id": point_id,
                                            "dataset_name": payload.get("dataset_name") if payload else None,
                                            "episode_idx": payload.get("episode_idx") if payload else None,
                                            "step_idx": payload.get("step_idx") if payload else None,
                                            "language_instruction": payload.get("language_instruction") if payload else None,
                                        })
                                
                                if len(top_k_actions) >= 1:
                                    retrieval_success = True
                        except Exception as e:
                            print(f"Search failed: {e}")

                    # ========================================
                    # Step 2: 计算综合指标，决定策略
                    # ========================================
                    t0_metric = time_module.perf_counter()
                    composite_metric = metrics_calc.compute_composite_metric(alpha=cfg.alpha)
                    metrics_info = metrics_calc.get_current_metrics(alpha=cfg.alpha)
                    metric_time = time_module.perf_counter() - t0_metric
                    raw_metrics = metrics_info.get('raw', {})
                    norm_metrics = metrics_info.get('normalized', {})
                    
                    step_stat = {
                        'step': t - cfg.num_steps_wait,
                        'task_suite_name': cfg.task_suite_name,
                        'task_id': task_id,
                        'episode_idx': episode_idx,
                        'retrieval_success': retrieval_success,
                        'num_candidates': len(top_k_actions),
                        'embedding_time': embedding_time,
                        'retrieval_time': retrieval_time,
                        'retrieval_total_time': embedding_time + retrieval_time,
                        'metric_time': metric_time,
                        'top1_score': safe_float(top_k_scores[0]) if len(top_k_scores) > 0 else None,
                        'top_k_scores': [safe_float(score) for score in top_k_scores],
                        'top1_retrieved_action_trajectory_len': int(top_k_trajectories[0].shape[0]) if len(top_k_trajectories) > 0 else 0,
                        'raw_radius': safe_float(raw_metrics.get('radius')),
                        'raw_displacement': safe_float(raw_metrics.get('displacement')),
                        'norm_radius': safe_float(norm_metrics.get('radius')),
                        'norm_displacement': safe_float(norm_metrics.get('displacement')),
                        'composite_metric': float(composite_metric) if not np.isnan(composite_metric) else None,
                        'metric_history_length': metrics_info.get('history_length'),
                        'blocks': [],
                    }
                    if cfg.mmrebuttal_record_step_metrics and len(top_k_trajectories) > 0:
                        step_stat['top1_retrieved_action_trajectory'] = top_k_trajectories[0].tolist()
                        step_stat['top1_payload_ref'] = top_k_payload_refs[0] if len(top_k_payload_refs) > 0 else None
                    
                    final_action = None
                    generation_time = 0.0
                    component_profiler.reset()
                    
                    # 决定使用哪个策略
                    use_ar_for_insufficient = False
                    if np.isnan(composite_metric):
                        # 历史不足时使用 Retrieval 策略（AR模式），避免 SD 模式的状态问题
                        use_retrieval_strategy = True
                        use_ar_for_insufficient = True
                        decision_reason = "insufficient_history_use_AR"
                    else:
                        # 综合指标 > threshold: Retrieval 策略 (2:1)
                        # 综合指标 <= threshold: SD 策略
                        use_retrieval_strategy = composite_metric > cfg.composite_threshold
                        decision_reason = f"composite={composite_metric:.4f}"
                    
                    step_stat['use_retrieval_strategy'] = use_retrieval_strategy
                    step_stat['use_ar_for_insufficient'] = use_ar_for_insufficient
                    step_stat['decision_reason'] = decision_reason
                    
                    # 清理 tree_mask 状态
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'language_model'):
                        model.base_model.language_model.tree_mask = None
                    if hasattr(model, 'tree_mask'):
                        model.tree_mask = None
                    
                    # ========================================
                    # Step 3: 根据策略生成 action
                    # ========================================
                    if use_retrieval_strategy:
                        # Retrieval 策略 - 根据 ablation_mode 选择不同行为
                        
                        # 如果历史不足，直接使用 AR
                        if use_ar_for_insufficient:
                            t0_gen = time_module.time()
                            final_action, time_tuple = get_action(
                                cfg, model, observation, task_description,
                                processor=processor, generate_mode='AR', return_time=True
                            )
                            generation_time = time_tuple[0] - time_tuple[1]
                            step_stat['mode'] = 'AR_insufficient_history'
                        
                        # 检索失败，fallback 到 AR
                        elif not retrieval_success:
                            t0_gen = time_module.time()
                            final_action, time_tuple = get_action(
                                cfg, model, observation, task_description,
                                processor=processor, generate_mode='AR', return_time=True
                            )
                            generation_time = time_tuple[0] - time_tuple[1]
                            step_stat['mode'] = 'AR_retrieval_failed'
                            retrieval_consecutive_db_count = 0
                        
                        # ========================================
                        # 消融模式 1: 只有阈值分割 (每步都 AR)
                        # ========================================
                        elif cfg.ablation_mode == 1:
                            # 阈值之上，每步都使用 AR（不使用 noverify）
                            t0_gen = time_module.time()
                            final_action, time_tuple = get_action(
                                cfg, model, observation, task_description,
                                processor=processor, generate_mode='AR', return_time=True
                            )
                            generation_time = time_tuple[0] - time_tuple[1]
                            step_stat['mode'] = 'Ablation1_AR'
                        
                        # ========================================
                        # 消融模式 2: + Verify-Skip (2:1, verify 用 AR)
                        # ========================================
                        elif cfg.ablation_mode == 2:
                            if retrieval_consecutive_db_count < 2:
                                # noverify: 直接使用检索的 top-1 action
                                t0_gen = time_module.time()
                                final_action = top_k_actions[0]  # 使用 top-1
                                generation_time = time_module.time() - t0_gen
                                retrieval_consecutive_db_count += 1
                                step_stat['mode'] = 'Ablation2_Noverify'
                            else:
                                # verify: 使用 AR
                                t0_gen = time_module.time()
                                final_action, time_tuple = get_action(
                                    cfg, model, observation, task_description,
                                    processor=processor, generate_mode='AR', return_time=True
                                )
                                generation_time = time_tuple[0] - time_tuple[1]
                                step_stat['mode'] = 'Ablation2_AR_verify'
                                retrieval_consecutive_db_count = 0  # 重置计数器
                        
                        # ========================================
                        # 消融模式 3: + Seq-Wise (2:1, verify 用 BlockSD) [主实验]
                        # ========================================
                        elif cfg.ablation_mode == 3:
                            if retrieval_consecutive_db_count < 2:
                                # noverify: 直接使用检索的 top-1 action
                                t0_gen = time_module.time()
                                final_action = top_k_actions[0]  # 使用 top-1
                                generation_time = time_module.time() - t0_gen
                                retrieval_consecutive_db_count += 1
                                step_stat['mode'] = 'Retrieval_DB'
                            else:
                                # verify: Block SD 验证模式
                                K = len(top_k_actions)
                                top_k_tokens = np.stack([
                                    action_to_tokens(a, model, cfg.unnorm_key) for a in top_k_actions
                                ])  # [K, 7]
                                top_k_tokens_tensor = torch.tensor(top_k_tokens, dtype=torch.long)
                                
                                # 准备输入
                                from PIL import Image as PILImage
                                image = PILImage.fromarray(img).convert("RGB")
                                prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
                                inputs = processor(prompt, image).to(model.base_model.device, dtype=torch.bfloat16)
                                
                                input_ids = inputs['input_ids']
                                if not torch.all(input_ids[:, -1] == 29871):
                                    input_ids = torch.cat(
                                        (input_ids, torch.tensor([[29871]], device=input_ids.device)), dim=1
                                    )
                                    inputs['attention_mask'] = torch.cat(
                                        (inputs['attention_mask'], torch.tensor([[1]], device=inputs['attention_mask'].device)), dim=1
                                    )
                                
                                t0_gen = time_module.time()
                                
                                try:
                                    final_tokens_tensor, block_stats = model.block_sd_verify(
                                        input_ids=input_ids,
                                        top_k_tokens=top_k_tokens_tensor,
                                        blocks=BLOCKS,
                                        prob_threshold=cfg.prob_threshold,
                                        use_avg_prob=cfg.use_avg_prob,
                                        accept_threshold=cfg.accept_threshold,
                                        block_thresholds=block_thresholds,
                                        block_sum_threshold=cfg.block_sum_threshold,
                                        block_max_threshold=cfg.block_max_threshold,
                                        pixel_values=inputs.get('pixel_values'),
                                        attention_mask=inputs.get('attention_mask'),
                                    )
                                    
                                    generation_time = time_module.time() - t0_gen
                                    final_tokens = final_tokens_tensor.cpu().numpy()
                                    final_action = tokens_to_action(final_tokens, model, cfg.unnorm_key)
                                    
                                    for bs in block_stats['blocks']:
                                        step_stat['blocks'].append(bs)
                                        # 统计每个 block 的验证/AR 情况
                                        if bs['mode'] == 'verified':
                                            total_block_verified[bs['name']] += 1
                                        else:
                                            total_block_ar[bs['name']] += 1
                                    
                                    # 记录接受的 token 数
                                    step_stat['accepted_tokens'] = block_stats.get('accepted_tokens', 0)
                                    
                                    if block_stats.get('accepted', False):
                                        # 全部 block 验证通过
                                        total_full_verify += 1
                                        step_stat['mode'] = 'Retrieval_BlockSD_fully_verified'
                                    else:
                                        # 部分 block 使用 AR
                                        step_stat['mode'] = 'Retrieval_BlockSD_partial_AR'
                                        step_stat['num_ar_blocks'] = block_stats.get('num_ar_blocks', 0)
                                        
                                except Exception as e:
                                    print(f"block_sd_verify failed: {e}, fallback to AR")
                                    import traceback
                                    traceback.print_exc()
                                    
                                    final_action, time_tuple = get_action(
                                        cfg, model, observation, task_description,
                                        processor=processor, generate_mode='AR', return_time=True
                                    )
                                    generation_time = time_tuple[0] - time_tuple[1]
                                    step_stat['mode'] = 'Retrieval_BlockSD_error_AR'
                                    for block in BLOCKS:
                                        total_block_ar[block['name']] += 1
                                
                                retrieval_consecutive_db_count = 0  # 重置计数器
                        
                        else:
                            # 未知的消融模式，默认使用 AR
                            print(f"Unknown ablation_mode: {cfg.ablation_mode}, fallback to AR")
                            t0_gen = time_module.time()
                            final_action, time_tuple = get_action(
                                cfg, model, observation, task_description,
                                processor=processor, generate_mode='AR', return_time=True
                            )
                            generation_time = time_tuple[0] - time_tuple[1]
                            step_stat['mode'] = 'AR_unknown_ablation'
                    
                    else:
                        # SD 策略: 使用原始 SpecVLA SD
                        t0_gen = time_module.time()
                        final_action, time_tuple, sd_stats = get_action(
                            cfg, model, observation, task_description,
                            processor=processor, generate_mode='Speculative', return_time=True,
                            return_sd_stats=True
                        )
                        generation_time = time_tuple[0] - time_tuple[1]
                        step_stat['mode'] = 'SD'
                        # 记录 SD 的接受长度信息
                        if sd_stats is not None:
                            # 平均接受长度 = new_token / num_iterations
                            sd_accept_len = sd_stats['new_token'] / sd_stats['num_iterations'] if sd_stats['num_iterations'] > 0 else 0
                            step_stat['sd_accept_length'] = sd_accept_len
                            step_stat['sd_new_token'] = sd_stats['new_token']
                            step_stat['sd_num_iterations'] = sd_stats['num_iterations']
                        retrieval_consecutive_db_count = 0  # 重置计数器
                    
                    step_stat['generation_time'] = generation_time
                    if cfg.mmrebuttal_profile_component_times:
                        component_times = component_profiler.snapshot()
                        step_stat['profile_vit_time'] = safe_float(component_times.get('vit', 0.0))
                        step_stat['profile_llm_time'] = safe_float(component_times.get('llm', 0.0))
                        step_stat['profile_draft_model_time'] = safe_float(component_times.get('draft_model', 0.0))
                        mode_for_profile = step_stat.get('mode', '')
                        is_block_verify = mode_for_profile in {
                            'Retrieval_BlockSD_fully_verified',
                            'Retrieval_BlockSD_partial_AR',
                            'Retrieval_BlockSD_error_AR',
                        }
                        is_ar_generation = (
                            mode_for_profile.startswith('AR_')
                            or mode_for_profile in {
                                'Ablation1_AR',
                                'Ablation2_AR_verify',
                                'AR_unknown_ablation',
                            }
                        )
                        step_stat['profile_verification_with_metric_time'] = safe_float(
                            (generation_time + metric_time) if is_block_verify else 0.0
                        )
                        step_stat['profile_ar_generation_time'] = safe_float(
                            generation_time if is_ar_generation else 0.0
                        )
                        step_stat['profile_sd_generation_time'] = safe_float(
                            generation_time if mode_for_profile == 'SD' else 0.0
                        )
                    step_stat['retrieval_consecutive_db_count'] = retrieval_consecutive_db_count
                    accepted_length = infer_accepted_length(step_stat)
                    step_stat['accepted_length'] = safe_float(accepted_length)
                    if 'accepted_tokens' not in step_stat and accepted_length is not None:
                        step_stat['accepted_tokens'] = safe_float(accepted_length)
                    set_action_diagnostics(
                        step_stat, top_k_actions, final_action, model, cfg.unnorm_key,
                        include_arrays=cfg.mmrebuttal_record_step_metrics
                    )
                    episode_data.append(step_stat)
                    
                    # 更新综合指标计算器的历史
                    # 注意：必须使用机器人末端的绝对位置 (eef_pos)，而不是增量动作 (action)
                    # CompositeMetricsCalculator 是基于轨迹的绝对位置来计算曲率和位移的
                    eef_position = obs["robot0_eef_pos"]  # shape: (3,)
                    if cfg.mmrebuttal_record_step_metrics:
                        step_stat['eef_position'] = np.asarray(eef_position).tolist()
                    metrics_calc.update_history(eef_position)

                    # Normalize and execute
                    action = normalize_gripper_action(final_action, binarize=True)
                    
                    # Invert gripper action for OpenVLA
                    # 注意：检索的action（DB模式/noverify模式）已经是正确的格式，不需要invert
                    # 但 Block SD (tokens_to_action)、AR、SD 模式需要 invert
                    mode = step_stat.get('mode', '')
                    # 不需要 invert 的模式：直接使用检索结果的模式
                    no_invert_modes = ['Retrieval_DB', 'Ablation2_Noverify']
                    needs_invert = mode not in no_invert_modes
                    if needs_invert and cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    import traceback
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1
            if cfg.mmrebuttal_record_step_metrics:
                annotate_episode_trajectory_overlap(
                    episode_data,
                    overlap_eps=cfg.mmrebuttal_overlap_eps,
                )
            
            task_data.append({
                'task_id': task_id,
                'task_description': task_description,
                'episode_idx': episode_idx,
                'success': bool(done),
                'steps': episode_data
            })

            if cfg.save_videos:
                save_rollout_video(
                    replay_images, total_episodes, success=done, 
                    task_description=task_description, log_file=log_file
                )

            print(f"Success: {done}")
            print(f"# episodes: {total_episodes}, successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)")
            log_file.write(f"Success: {done}\n")
        
        # ============================================
        # 任务级别统计
        # ============================================
        total_steps_task = sum(len(ep['steps']) for ep in task_data)
        
        # 统计各模式步数 (包括消融实验的模式)
        task_db_steps = sum(
            sum(1 for step in ep['steps'] if step.get('mode') in ['Retrieval_DB', 'Ablation2_Noverify'])
            for ep in task_data
        )
        task_blocksd_verified = sum(
            sum(1 for step in ep['steps'] if step.get('mode') == 'Retrieval_BlockSD_fully_verified')
            for ep in task_data
        )
        task_blocksd_partial = sum(
            sum(1 for step in ep['steps'] if step.get('mode') == 'Retrieval_BlockSD_partial_AR')
            for ep in task_data
        )
        task_sd_steps = sum(
            sum(1 for step in ep['steps'] if step.get('mode') == 'SD')
            for ep in task_data
        )
        task_ar_steps = sum(
            sum(1 for step in ep['steps'] if step.get('mode') in ['AR_retrieval_failed', 'AR_insufficient_history', 'Ablation1_AR', 'Ablation2_AR_verify'])
            for ep in task_data
        )
        
        # 收集时间数据
        task_db_times = []
        task_blocksd_times = []
        task_sd_times = []
        task_ar_times = []
        for ep in task_data:
            for step in ep['steps']:
                mode = step.get('mode', '')
                gen_time = step.get('generation_time', 0)
                ret_time = step.get('retrieval_time', 0)
                if mode in ['Retrieval_DB', 'Ablation2_Noverify'] and ret_time > 0:
                    task_db_times.append(ret_time)
                elif mode in ['Retrieval_BlockSD_fully_verified', 'Retrieval_BlockSD_partial_AR'] and gen_time > 0:
                    task_blocksd_times.append(gen_time)
                elif mode == 'SD' and gen_time > 0:
                    task_sd_times.append(gen_time)
                elif mode in ['AR_retrieval_failed', 'AR_insufficient_history', 'Ablation1_AR', 'Ablation2_AR_verify'] and gen_time > 0:
                    task_ar_times.append(gen_time)
        
        print(f"\n{'='*60}")
        print(f"Task {task_id} Statistics: {task_description}")
        print(f"{'='*60}")
        print(f"  Successes: {task_successes}/{cfg.num_trials_per_task} ({100*task_successes/cfg.num_trials_per_task:.1f}%)")
        print(f"  Total steps: {total_steps_task}")
        print(f"  Mode breakdown:")
        print(f"    Retrieval_DB: {task_db_steps}")
        print(f"    BlockSD_verified: {task_blocksd_verified}")
        print(f"    BlockSD_partial_AR: {task_blocksd_partial}")
        print(f"    SD: {task_sd_steps}")
        print(f"    AR: {task_ar_steps}")
        
        print(f"  Time statistics:")
        if len(task_db_times) > 0:
            print(f"    DB: {np.mean(task_db_times)*1000:.2f} ms (n={len(task_db_times)})")
        if len(task_blocksd_times) > 0:
            print(f"    BlockSD: {np.mean(task_blocksd_times)*1000:.2f} ms (n={len(task_blocksd_times)})")
        if len(task_sd_times) > 0:
            print(f"    SD: {np.mean(task_sd_times)*1000:.2f} ms (n={len(task_sd_times)})")
        if len(task_ar_times) > 0:
            print(f"    AR: {np.mean(task_ar_times)*1000:.2f} ms (n={len(task_ar_times)})")
        print(f"{'='*60}\n")
        
        # 写入日志
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Task {task_id} Statistics: {task_description}\n")
        log_file.write(f"  Successes: {task_successes}/{cfg.num_trials_per_task}\n")
        log_file.write(f"  Total steps: {total_steps_task}\n")
        log_file.write(f"  Retrieval_DB: {task_db_steps}, BlockSD_verified: {task_blocksd_verified}, ")
        log_file.write(f"BlockSD_partial: {task_blocksd_partial}, SD: {task_sd_steps}, AR: {task_ar_steps}\n")
        if len(task_db_times) > 0:
            log_file.write(f"  DB time: {np.mean(task_db_times)*1000:.2f} ms\n")
        if len(task_blocksd_times) > 0:
            log_file.write(f"  BlockSD time: {np.mean(task_blocksd_times)*1000:.2f} ms\n")
        if len(task_sd_times) > 0:
            log_file.write(f"  SD time: {np.mean(task_sd_times)*1000:.2f} ms\n")
        if len(task_ar_times) > 0:
            log_file.write(f"  AR time: {np.mean(task_ar_times)*1000:.2f} ms\n")
        log_file.write(f"{'='*60}\n")
        
        all_data.append({
            'task_id': task_id,
            'task_description': task_description,
            'episodes': task_data
        })
    
    # ============================================
    # Profile: 收集所有概率值并分析分布
    # ============================================
    profile_data = {
        'position': {'probs': [], 'passed': [], 'all_candidate_probs': []},
        'orientation': {'probs': [], 'passed': [], 'all_candidate_probs': []},
        'gripper': {'probs': [], 'passed': [], 'all_candidate_probs': []},
    }
    
    # 从 all_data 中提取概率
    for task_data_item in all_data:
        for ep in task_data_item['episodes']:
            for step in ep['steps']:
                if 'blocks' in step:
                    for block_stat in step['blocks']:
                        block_name = block_stat.get('name')
                        if block_name and block_name in profile_data:
                            # best_prob
                            if 'best_prob' in block_stat:
                                profile_data[block_name]['probs'].append(block_stat['best_prob'])
                                profile_data[block_name]['passed'].append(block_stat.get('mode') == 'verified')
                            # all_probs (所有候选的概率)
                            if 'all_probs' in block_stat and block_stat['all_probs']:
                                profile_data[block_name]['all_candidate_probs'].extend(block_stat['all_probs'])
    
    # ============================================
    # 打印总体统计
    # ============================================
    print("\n" + "="*80)
    print("Overall Statistics:")
    print("="*80)
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Success rate (SR): {total_successes/total_episodes*100:.1f}%")
    
    # ============================================
    # 平均步数统计 (Avg Steps)
    # ============================================
    episode_steps = []
    for task_data_item in all_data:
        for ep in task_data_item['episodes']:
            episode_steps.append(len(ep['steps']))
    
    avg_steps = np.mean(episode_steps) if len(episode_steps) > 0 else 0
    std_steps = np.std(episode_steps) if len(episode_steps) > 0 else 0
    print(f"\nAvg Steps per Episode: {avg_steps:.1f} ± {std_steps:.1f}")
    
    print(f"\nBlock Statistics:")
    total_blocks = sum(total_block_verified.values()) + sum(total_block_ar.values())
    for name in ['position', 'orientation', 'gripper']:
        verified = total_block_verified[name]
        ar = total_block_ar[name]
        total = verified + ar
        if total > 0:
            print(f"  {name}: verified={verified} ({100*verified/total:.1f}%), AR={ar} ({100*ar/total:.1f}%)")
    
    print(f"\nFull verify steps (all 3 blocks passed): {total_full_verify}")
    
    # ============================================
    # 模式统计 & 时间统计
    # ============================================
    # 统计各模式步数
    all_db_steps = 0
    all_blocksd_verified_steps = 0
    all_blocksd_partial_ar_steps = 0
    all_sd_steps = 0
    all_ar_steps = 0
    
    # 收集时间数据
    db_times = []
    blocksd_verified_times = []
    blocksd_partial_ar_times = []
    sd_times = []
    ar_times = []
    
    for task_data_item in all_data:
        for ep in task_data_item['episodes']:
            for step in ep['steps']:
                mode = step.get('mode', '')
                gen_time = step.get('generation_time', 0)
                retrieval_time = step.get('retrieval_time', 0)
                
                # noverify 模式：直接使用检索结果
                if mode in ['Retrieval_DB', 'Ablation2_Noverify']:
                    all_db_steps += 1
                    if retrieval_time > 0:
                        db_times.append(retrieval_time)
                elif mode == 'Retrieval_BlockSD_fully_verified':
                    all_blocksd_verified_steps += 1
                    if gen_time > 0:
                        blocksd_verified_times.append(gen_time)
                elif mode == 'Retrieval_BlockSD_partial_AR':
                    all_blocksd_partial_ar_steps += 1
                    if gen_time > 0:
                        blocksd_partial_ar_times.append(gen_time)
                elif mode == 'SD':
                    all_sd_steps += 1
                    if gen_time > 0:
                        sd_times.append(gen_time)
                # AR 模式：包括消融实验1的每步AR、消融实验2的AR verify、以及其他AR fallback
                elif mode in ['AR_retrieval_failed', 'AR_insufficient_history', 'Ablation1_AR', 'Ablation2_AR_verify']:
                    all_ar_steps += 1
                    if gen_time > 0:
                        ar_times.append(gen_time)
    
    total_all_steps = all_db_steps + all_blocksd_verified_steps + all_blocksd_partial_ar_steps + all_sd_steps + all_ar_steps
    
    print(f"\n  ============ 模式统计 ============")
    if total_all_steps > 0:
        print(f"  Retrieval_DB steps: {all_db_steps} ({100.0*all_db_steps/total_all_steps:.1f}%)")
        print(f"  BlockSD_fully_verified steps: {all_blocksd_verified_steps} ({100.0*all_blocksd_verified_steps/total_all_steps:.1f}%)")
        print(f"  BlockSD_partial_AR steps: {all_blocksd_partial_ar_steps} ({100.0*all_blocksd_partial_ar_steps/total_all_steps:.1f}%)")
        print(f"  SD steps: {all_sd_steps} ({100.0*all_sd_steps/total_all_steps:.1f}%)")
        print(f"  AR steps: {all_ar_steps} ({100.0*all_ar_steps/total_all_steps:.1f}%)")
    
    total_retrieval_steps = all_db_steps + all_blocksd_verified_steps + all_blocksd_partial_ar_steps
    print(f"\n  ============ 关键比例 ============")
    print(f"  SD策略 : Retrieval策略 = {all_sd_steps}:{total_retrieval_steps}")
    print(f"  Retrieval内部 DB : BlockSD = {all_db_steps}:{all_blocksd_verified_steps + all_blocksd_partial_ar_steps}")
    if all_blocksd_verified_steps + all_blocksd_partial_ar_steps > 0:
        print(f"  BlockSD内部 fully_verified : partial_AR = {all_blocksd_verified_steps}:{all_blocksd_partial_ar_steps}")
    print(f"  ==================================")
    
    print(f"\n  ============ 时间统计 ============")
    avg_db_time = np.mean(db_times) if len(db_times) > 0 else 0
    avg_blocksd_verified_time = np.mean(blocksd_verified_times) if len(blocksd_verified_times) > 0 else 0
    avg_blocksd_partial_ar_time = np.mean(blocksd_partial_ar_times) if len(blocksd_partial_ar_times) > 0 else 0
    avg_sd_time = np.mean(sd_times) if len(sd_times) > 0 else 0
    avg_ar_time = np.mean(ar_times) if len(ar_times) > 0 else 0
    
    if len(db_times) > 0:
        print(f"  Retrieval_DB 检索时间:")
        print(f"    Mean: {avg_db_time*1000:.2f} ms, Std: {np.std(db_times)*1000:.2f} ms, Samples: {len(db_times)}")
    
    if len(blocksd_verified_times) > 0:
        print(f"  BlockSD_fully_verified 生成时间:")
        print(f"    Mean: {avg_blocksd_verified_time*1000:.2f} ms, Std: {np.std(blocksd_verified_times)*1000:.2f} ms, Samples: {len(blocksd_verified_times)}")
    
    if len(blocksd_partial_ar_times) > 0:
        print(f"  BlockSD_partial_AR 生成时间:")
        print(f"    Mean: {avg_blocksd_partial_ar_time*1000:.2f} ms, Std: {np.std(blocksd_partial_ar_times)*1000:.2f} ms, Samples: {len(blocksd_partial_ar_times)}")
    
    if len(sd_times) > 0:
        print(f"  SD 生成时间:")
        print(f"    Mean: {avg_sd_time*1000:.2f} ms, Std: {np.std(sd_times)*1000:.2f} ms, Samples: {len(sd_times)}")
    
    if len(ar_times) > 0:
        print(f"  AR 生成时间:")
        print(f"    Mean: {avg_ar_time*1000:.2f} ms, Std: {np.std(ar_times)*1000:.2f} ms, Samples: {len(ar_times)}")
    
    # 加权平均每步时间
    if total_all_steps > 0:
        weighted_avg_time = (
            all_db_steps * avg_db_time +
            all_blocksd_verified_steps * avg_blocksd_verified_time +
            all_blocksd_partial_ar_steps * avg_blocksd_partial_ar_time +
            all_sd_steps * avg_sd_time +
            all_ar_steps * avg_ar_time
        ) / total_all_steps
        print(f"\n  加权平均每步时间: {weighted_avg_time*1000:.2f} ms")
    print(f"  ==================================")
    
    # ============================================
    # Accept Length (AL) 统计
    # ============================================
    print(f"\n  ============ Accept Length (AL) 统计 ============")
    
    # 收集各模式的 Accept Length
    # - Retrieval_DB (noverify): AL = 7 (直接使用检索结果，无需验证)
    # - BlockSD_fully_verified: AL = 7 (全部接受)
    # - BlockSD_partial_AR: AL = accepted_tokens (部分接受，其余 AR)
    # - SD: AL = SpecVLA 的实际接受长度 (从 sd_stats 获取)
    # - AR: AL = 1 (逐 token 生成，每步生成 1 个 token)
    
    al_db = []  # noverify
    al_blocksd_verified = []
    al_blocksd_partial = []
    al_sd = []
    al_ar = []
    
    for task_data_item in all_data:
        for ep in task_data_item['episodes']:
            for step in ep['steps']:
                mode = step.get('mode', '')
                
                # noverify 模式：直接使用检索结果，AL=7
                if mode in ['Retrieval_DB', 'Ablation2_Noverify']:
                    al_db.append(7)  # noverify 直接使用，AL=7
                elif mode == 'Retrieval_BlockSD_fully_verified':
                    al_blocksd_verified.append(7)  # 全部通过，AL=7
                elif mode == 'Retrieval_BlockSD_partial_AR':
                    accepted = step.get('accepted_tokens', 0)
                    al_blocksd_partial.append(safe_float(accepted))
                elif mode == 'SD':
                    # SD 模式使用实际的接受长度
                    sd_al = step.get('sd_accept_length', 7)  # 默认 7
                    al_sd.append(safe_float(sd_al))
                # AR 模式：包括消融实验1的每步AR、消融实验2的AR verify、以及其他AR fallback
                elif mode in ['AR_retrieval_failed', 'AR_insufficient_history', 'Ablation1_AR', 'Ablation2_AR_verify']:
                    al_ar.append(1)  # AR 逐 token，AL=1
    
    # 打印各模式的 AL
    if len(al_db) > 0:
        print(f"  Retrieval_DB (noverify): AL = 7.0 (fixed), Samples: {len(al_db)}")
    if len(al_blocksd_verified) > 0:
        print(f"  BlockSD_fully_verified: AL = 7.0 (all accepted), Samples: {len(al_blocksd_verified)}")
    if len(al_blocksd_partial) > 0:
        print(f"  BlockSD_partial_AR: AL = {np.mean(al_blocksd_partial):.2f} ± {np.std(al_blocksd_partial):.2f}, Samples: {len(al_blocksd_partial)}")
    if len(al_sd) > 0:
        print(f"  SD: AL = {np.mean(al_sd):.2f} ± {np.std(al_sd):.2f}, Samples: {len(al_sd)}")
    if len(al_ar) > 0:
        print(f"  AR: AL = 1.0 (fixed), Samples: {len(al_ar)}")
    
    # 计算加权平均 AL
    all_als = al_db + al_blocksd_verified + al_blocksd_partial + al_sd + al_ar
    if len(all_als) > 0:
        weighted_al = np.mean(all_als)
        print(f"\n  加权平均 Accept Length (AL): {weighted_al:.2f}")
    
    print(f"  ==================================")
    
    # ============================================
    # Speedup 计算
    # ============================================
    print(f"\n  ============ Speedup 统计 ============")
    
    # AR baseline 时间 (假设纯 AR 每步约 175ms)
    ar_baseline_time = avg_ar_time if avg_ar_time > 0 else 0.175  # 秒
    
    if total_all_steps > 0 and weighted_avg_time > 0:
        speedup = ar_baseline_time / weighted_avg_time
        print(f"  AR Baseline 时间: {ar_baseline_time*1000:.2f} ms")
        print(f"  当前策略平均时间: {weighted_avg_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    print(f"  ==================================")
    
    # ============================================
    # 汇总表格
    # ============================================
    print(f"\n" + "="*60)
    print(f"  ============ 汇总指标 ============")
    print(f"  SR (Success Rate): {total_successes/total_episodes*100:.1f}%")
    print(f"  Avg Steps: {avg_steps:.1f}")
    if len(all_als) > 0:
        print(f"  AL (Accept Length): {weighted_al:.2f}")
    if total_all_steps > 0 and weighted_avg_time > 0:
        print(f"  Speedup: {speedup:.2f}x")
    print(f"  ==================================")
    print(f"="*60)
    
    # ============================================
    # Profile 分析: 概率分布
    # ============================================
    print("\n" + "="*80)
    print("PROFILE: Block Probability Distribution")
    print("="*80)
    
    for block_name in ['position', 'orientation', 'gripper']:
        probs = profile_data[block_name]['probs']
        passed = profile_data[block_name]['passed']
        all_cand_probs = profile_data[block_name]['all_candidate_probs']
        
        if len(probs) > 0:
            probs_arr = np.array(probs)
            passed_arr = np.array(passed)
            
            print(f"\n  [{block_name.upper()}]")
            print(f"    样本数: {len(probs)}")
            print(f"    Best Prob 统计:")
            print(f"      Mean:   {np.mean(probs_arr):.6f}")
            print(f"      Std:    {np.std(probs_arr):.6f}")
            print(f"      Min:    {np.min(probs_arr):.6f}")
            print(f"      Max:    {np.max(probs_arr):.6f}")
            print(f"      Median: {np.median(probs_arr):.6f}")
            
            # 分位数
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            print(f"    Percentiles:")
            for p in percentiles:
                val = np.percentile(probs_arr, p)
                print(f"      P{p:02d}: {val:.6f}")
            
            # 通过率 vs 概率阈值分析
            print(f"    通过率 @ 不同阈值:")
            thresholds_to_test = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
            for thresh in thresholds_to_test:
                pass_rate = np.mean(probs_arr > thresh) * 100
                print(f"      thresh={thresh:.2f}: {pass_rate:.1f}%")
            
            # 如果有 all_candidate_probs，分析所有候选
            if len(all_cand_probs) > 0:
                all_cand_arr = np.array(all_cand_probs)
                print(f"    All Candidates Prob 统计 (含非 best):")
                print(f"      Mean:   {np.mean(all_cand_arr):.6f}")
                print(f"      Std:    {np.std(all_cand_arr):.6f}")
                print(f"      Max:    {np.max(all_cand_arr):.6f}")
    
    print("="*80)
    
    # ============================================
    # 写入日志
    # ============================================
    log_file.write("\n" + "="*80 + "\n")
    log_file.write(f"Total episodes: {total_episodes}\n")
    log_file.write(f"Total successes: {total_successes}\n")
    log_file.write(f"Success rate: {total_successes/total_episodes*100:.1f}%\n")
    log_file.write(f"Block verified: {total_block_verified}\n")
    log_file.write(f"Block AR: {total_block_ar}\n")
    log_file.write(f"Full verify steps: {total_full_verify}\n")
    
    # 模式统计写入日志
    log_file.write(f"\n  ============ 模式统计 ============\n")
    if total_all_steps > 0:
        log_file.write(f"  Retrieval_DB steps: {all_db_steps} ({100.0*all_db_steps/total_all_steps:.1f}%)\n")
        log_file.write(f"  BlockSD_fully_verified steps: {all_blocksd_verified_steps} ({100.0*all_blocksd_verified_steps/total_all_steps:.1f}%)\n")
        log_file.write(f"  BlockSD_partial_AR steps: {all_blocksd_partial_ar_steps} ({100.0*all_blocksd_partial_ar_steps/total_all_steps:.1f}%)\n")
        log_file.write(f"  SD steps: {all_sd_steps} ({100.0*all_sd_steps/total_all_steps:.1f}%)\n")
        log_file.write(f"  AR steps: {all_ar_steps} ({100.0*all_ar_steps/total_all_steps:.1f}%)\n")
    
    log_file.write(f"\n  ============ 关键比例 ============\n")
    log_file.write(f"  SD策略 : Retrieval策略 = {all_sd_steps}:{total_retrieval_steps}\n")
    log_file.write(f"  Retrieval内部 DB : BlockSD = {all_db_steps}:{all_blocksd_verified_steps + all_blocksd_partial_ar_steps}\n")
    if all_blocksd_verified_steps + all_blocksd_partial_ar_steps > 0:
        log_file.write(f"  BlockSD内部 fully_verified : partial_AR = {all_blocksd_verified_steps}:{all_blocksd_partial_ar_steps}\n")
    
    # 时间统计写入日志
    log_file.write(f"\n  ============ 时间统计 ============\n")
    if len(db_times) > 0:
        log_file.write(f"  Retrieval_DB 检索时间: Mean={avg_db_time*1000:.2f}ms, Std={np.std(db_times)*1000:.2f}ms, Samples={len(db_times)}\n")
    if len(blocksd_verified_times) > 0:
        log_file.write(f"  BlockSD_fully_verified 生成时间: Mean={avg_blocksd_verified_time*1000:.2f}ms, Std={np.std(blocksd_verified_times)*1000:.2f}ms, Samples={len(blocksd_verified_times)}\n")
    if len(blocksd_partial_ar_times) > 0:
        log_file.write(f"  BlockSD_partial_AR 生成时间: Mean={avg_blocksd_partial_ar_time*1000:.2f}ms, Std={np.std(blocksd_partial_ar_times)*1000:.2f}ms, Samples={len(blocksd_partial_ar_times)}\n")
    if len(sd_times) > 0:
        log_file.write(f"  SD 生成时间: Mean={avg_sd_time*1000:.2f}ms, Std={np.std(sd_times)*1000:.2f}ms, Samples={len(sd_times)}\n")
    if len(ar_times) > 0:
        log_file.write(f"  AR 生成时间: Mean={avg_ar_time*1000:.2f}ms, Std={np.std(ar_times)*1000:.2f}ms, Samples={len(ar_times)}\n")
    if total_all_steps > 0:
        log_file.write(f"\n  加权平均每步时间: {weighted_avg_time*1000:.2f}ms\n")
    log_file.write(f"  ==================================\n")
    
    # Accept Length 统计写入日志
    log_file.write(f"\n  ============ Accept Length (AL) 统计 ============\n")
    if len(al_db) > 0:
        log_file.write(f"  Retrieval_DB (noverify): AL = 7.0, Samples: {len(al_db)}\n")
    if len(al_blocksd_verified) > 0:
        log_file.write(f"  BlockSD_fully_verified: AL = 7.0, Samples: {len(al_blocksd_verified)}\n")
    if len(al_blocksd_partial) > 0:
        log_file.write(f"  BlockSD_partial_AR: AL = {np.mean(al_blocksd_partial):.2f} ± {np.std(al_blocksd_partial):.2f}, Samples: {len(al_blocksd_partial)}\n")
    if len(al_sd) > 0:
        log_file.write(f"  SD: AL = {np.mean(al_sd):.2f} ± {np.std(al_sd):.2f}, Samples: {len(al_sd)}\n")
    if len(al_ar) > 0:
        log_file.write(f"  AR: AL = 1.0, Samples: {len(al_ar)}\n")
    if len(all_als) > 0:
        log_file.write(f"\n  加权平均 Accept Length (AL): {weighted_al:.2f}\n")
    log_file.write(f"  ==================================\n")
    
    # Speedup 统计写入日志
    log_file.write(f"\n  ============ Speedup 统计 ============\n")
    if total_all_steps > 0 and weighted_avg_time > 0:
        log_file.write(f"  AR Baseline 时间: {ar_baseline_time*1000:.2f}ms\n")
        log_file.write(f"  当前策略平均时间: {weighted_avg_time*1000:.2f}ms\n")
        log_file.write(f"  Speedup: {speedup:.2f}x\n")
    log_file.write(f"  ==================================\n")
    
    # 汇总指标写入日志
    log_file.write(f"\n" + "="*60 + "\n")
    log_file.write(f"  ============ 汇总指标 ============\n")
    log_file.write(f"  SR (Success Rate): {total_successes/total_episodes*100:.1f}%\n")
    log_file.write(f"  Avg Steps: {avg_steps:.1f}\n")
    if len(all_als) > 0:
        log_file.write(f"  AL (Accept Length): {weighted_al:.2f}\n")
    if total_all_steps > 0 and weighted_avg_time > 0:
        log_file.write(f"  Speedup: {speedup:.2f}x\n")
    log_file.write(f"  ==================================\n")
    log_file.write(f"="*60 + "\n")
    
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("PROFILE: Block Probability Distribution\n")
    log_file.write("="*80 + "\n")
    
    for block_name in ['position', 'orientation', 'gripper']:
        probs = profile_data[block_name]['probs']
        if len(probs) > 0:
            probs_arr = np.array(probs)
            log_file.write(f"\n[{block_name.upper()}]\n")
            log_file.write(f"  样本数: {len(probs)}\n")
            log_file.write(f"  Mean: {np.mean(probs_arr):.6f}\n")
            log_file.write(f"  Std:  {np.std(probs_arr):.6f}\n")
            log_file.write(f"  Min:  {np.min(probs_arr):.6f}\n")
            log_file.write(f"  Max:  {np.max(probs_arr):.6f}\n")
            log_file.write(f"  Median: {np.median(probs_arr):.6f}\n")
            
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            log_file.write(f"  Percentiles:\n")
            for p in percentiles:
                val = np.percentile(probs_arr, p)
                log_file.write(f"    P{p:02d}: {val:.6f}\n")
            
            log_file.write(f"  通过率 @ 不同阈值:\n")
            thresholds_to_test = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
            for thresh in thresholds_to_test:
                pass_rate = np.mean(probs_arr > thresh) * 100
                log_file.write(f"    thresh={thresh:.2f}: {pass_rate:.1f}%\n")
    
    # 保存数据 (使用自定义编码器处理 numpy 类型)
    with open(local_log_datapath, 'w') as f:
        json.dump(all_data, f, indent=2, cls=NumpyEncoder)
    print(f"\nData saved to: {local_log_datapath}")
    
    # 保存 profile 数据 (方便后续分析)
    profile_path = os.path.join(target_dir, run_id + "_profile.json")
    profile_export = {}
    for block_name in ['position', 'orientation', 'gripper']:
        profile_export[block_name] = {
            'probs': [float(p) for p in profile_data[block_name]['probs']],  # 确保是 float
            'passed': [bool(p) for p in profile_data[block_name]['passed']],
        }
    with open(profile_path, 'w') as f:
        json.dump(profile_export, f, indent=2, cls=NumpyEncoder)
    print(f"Profile data saved to: {profile_path}")
    
    log_file.close()


if __name__ == "__main__":
    eval_libero()
