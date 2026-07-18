"""
run_libero_retrieval_verify_mix.py

基于综合指标（曲率半径+位移）动态切换检索与AR生成。
使用Mix视角（第三人称+手腕相机）进行检索。

规则：
- 综合指标 > composite_threshold (0.143210, 25%分位数)：使用检索（DB）
- 综合指标 <= composite_threshold：使用AR生成
- 两次连续检索后，强制执行一次AR

归一化参数（minmax归一化，超过范围取0或1）：
- 位移指标：[0.000009, 0.123381]
- 曲率半径：[0.000001, 0.014989]
- alpha = 0.5（1:1构造综合指标）

Usage:
    python experiments/robot/libero/run_libero_retrieval_verify_mix.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --spec_checkpoint <SPEC_CHECKPOINT_PATH> \
        --task_suite_name libero_goal \
        --center_crop True \
        --composite_threshold 0.143210 \
        --alpha 0.5 \
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

import wandb
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


class CompositeMetricsCalculatorMix:
    """
    综合指标计算器（Mix视角版本）
    结合曲率半径指标和位移指标，进行归一化后加权求和
    
    新的归一化范围：
    - 位移指标：[0.000009, 0.123381]
    - 曲率半径：[0.000001, 0.014989]
    """
    
    def __init__(self, window_size=5, 
                 displacement_range=(0.000009, 0.123381),
                 radius_range=(0.000001, 0.014989)):
        """
        初始化综合指标计算器
        
        参数:
            window_size: 滑动窗口大小
            displacement_range: 位移指标的归一化范围 (min, max)
            radius_range: 曲率半径指标的归一化范围 (min, max)
        """
        self.window_size = window_size
        self.displacement_range = displacement_range
        self.radius_range = radius_range
        self.action_history = []  # 存储位置历史
        
    def update_history(self, position):
        """
        更新位置历史记录
        
        参数:
            position: numpy array, shape (3,) 或 (7,) - 位置坐标（只取前3维）
        """
        if len(position) >= 3:
            pos = position[:3].copy()
        else:
            pos = position.copy()
        self.action_history.append(pos)
        
    def clear_history(self):
        """清空历史记录（用于新episode开始时）"""
        self.action_history = []
        
    def normalize_value(self, value, value_range):
        """
        按照指定范围进行minmax归一化
        低于下限为0，高于上限为1
        
        参数:
            value: 待归一化的值
            value_range: (min, max) 归一化范围
            
        返回:
            normalized_value: 归一化后的值 [0, 1]
        """
        if np.isnan(value):
            return np.nan
        
        min_val, max_val = value_range
        
        # 低于下限为0
        if value <= min_val:
            return 0.0
        # 高于上限为1
        elif value >= max_val:
            return 1.0
        # 线性归一化
        else:
            return (value - min_val) / (max_val - min_val)
    
    def compute_radius_least_squares(self, points):
        """
        使用最小二乘法拟合圆，直接返回曲率半径
        
        参数:
            points: (N, 3) 的点数组
            
        返回:
            radius: 曲率半径（单位：米），如果无法计算则返回np.nan
        """
        from scipy.optimize import least_squares
        
        if len(points) < 3:
            return np.nan
        
        # 使用3D点的投影到最佳拟合平面
        center = np.mean(points, axis=0)
        points_centered = points - center
        
        try:
            _, _, vh = np.linalg.svd(points_centered)
            normal = vh[2, :]
            
            # 投影到平面
            points_2d = points_centered - np.outer(np.dot(points_centered, normal), normal)
            
            # 使用前两个主成分作为2D坐标
            u = vh[0, :]
            v = vh[1, :]
            x = np.dot(points_2d, u)
            y = np.dot(points_2d, v)
            
            # 拟合圆
            def calc_R(xc, yc):
                return np.sqrt((x - xc)**2 + (y - yc)**2)
            
            def f(c):
                Ri = calc_R(*c)
                return Ri - Ri.mean()
            
            center_estimate = np.array([x.mean(), y.mean()])
            result = least_squares(f, center_estimate)
            xc, yc = result.x
            Ri = calc_R(xc, yc)
            R = Ri.mean()
            
            return R if R > 1e-6 else np.nan
        except Exception as e:
            return np.nan
    
    def compute_displacement_metric(self, points):
        """
        计算位移指标：窗口最后一个点与前面所有点的欧式距离之和
        
        参数:
            points: (N, 3) 的点数组
            
        返回:
            metric: 位移指标值，如果无法计算则返回np.nan
        """
        if len(points) < 2:
            return np.nan
        
        last_point = points[-1]
        total_distance = 0.0
        for i in range(len(points) - 1):
            dist = np.linalg.norm(last_point - points[i])
            total_distance += dist
        
        return total_distance
    
    def get_current_radius(self):
        """计算当前轨迹的曲率半径"""
        if len(self.action_history) < 3:
            return np.nan
        
        start_idx = max(0, len(self.action_history) - self.window_size)
        window_points = np.array(self.action_history[start_idx:])
        
        return self.compute_radius_least_squares(window_points)
    
    def get_current_displacement(self):
        """计算当前的位移指标"""
        if len(self.action_history) < 2:
            return np.nan
        
        start_idx = max(0, len(self.action_history) - self.window_size)
        window_points = np.array(self.action_history[start_idx:])
        
        return self.compute_displacement_metric(window_points)
    
    def compute_composite_metric(self, alpha=0.5):
        """
        计算综合指标
        
        参数:
            alpha: 曲率半径指标的权重 [0, 1]
                  综合指标 = alpha * 曲率半径指标 + (1-alpha) * 位移指标
                  
        返回:
            composite_metric: 综合指标值 [0, 1]，如果无法计算则返回np.nan
        """
        radius = self.get_current_radius()
        displacement = self.get_current_displacement()
        
        radius_norm = self.normalize_value(radius, self.radius_range)
        displacement_norm = self.normalize_value(displacement, self.displacement_range)
        
        if np.isnan(radius_norm) or np.isnan(displacement_norm):
            return np.nan
        
        composite = alpha * radius_norm + (1 - alpha) * displacement_norm
        
        return composite
    
    def get_current_metrics(self, alpha=0.5):
        """
        获取当前所有指标（原始值、归一化值、综合值）
        """
        radius = self.get_current_radius()
        displacement = self.get_current_displacement()
        
        radius_norm = self.normalize_value(radius, self.radius_range)
        displacement_norm = self.normalize_value(displacement, self.displacement_range)
        
        composite = self.compute_composite_metric(alpha)
        
        return {
            'raw': {
                'radius': radius,
                'displacement': displacement
            },
            'normalized': {
                'radius': radius_norm,
                'displacement': displacement_norm
            },
            'composite': composite,
            'alpha': alpha,
            'history_length': len(self.action_history)
        }
    
    def get_history_length(self):
        """返回当前历史记录的长度"""
        return len(self.action_history)


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
    # Composite metrics-based decision parameters
    #################################################################################################################
    window_size: int = 5
    composite_threshold: float = 0.143210  # 25%分位数阈值
    alpha: float = 0.5  # 1:1权重
    displacement_range_min: float = 0.000009
    displacement_range_max: float = 0.123381
    radius_range_min: float = 0.000001
    radius_range_max: float = 0.014989

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

    # Initialize local logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "../../../specdecoding/test-speed")
    target_dir = os.path.join(base_dir, "libero_retrieval_verify_mix")
    os.makedirs(target_dir, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-alpha{cfg.alpha}-thresh{cfg.composite_threshold}-{DATE_TIME}"
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
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    print("Experiment: Composite Metric Retrieval Verify (Mix View - Third-Person + Wrist)")
    log_file.write("Experiment: Composite Metric Retrieval Verify (Mix View - Third-Person + Wrist)\n")
    print(f"Retrieval URL: {RETRIEVAL_URL}")
    log_file.write(f"Retrieval URL: {RETRIEVAL_URL}\n")
    print(f"Composite threshold: {cfg.composite_threshold} (25% percentile)")
    log_file.write(f"Composite threshold: {cfg.composite_threshold} (25% percentile)\n")
    print(f"Alpha: {cfg.alpha} (1:1 ratio)")
    log_file.write(f"Alpha: {cfg.alpha} (1:1 ratio)\n")
    print(f"Displacement range: [{cfg.displacement_range_min}, {cfg.displacement_range_max}]")
    log_file.write(f"Displacement range: [{cfg.displacement_range_min}, {cfg.displacement_range_max}]\n")
    print(f"Radius range: [{cfg.radius_range_min}, {cfg.radius_range_max}]")
    log_file.write(f"Radius range: [{cfg.radius_range_min}, {cfg.radius_range_max}]\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_retrieval_data = []  # 存储所有检索验证数据
    all_accept_lengths = []  # 存储所有accept_length用于统计（仅AR模式）
    observations_data = {}  # 存储所有observations
    
    # 统计模式使用
    total_db_steps = 0
    total_ar_steps = 0
    total_forced_ar_steps = 0
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_retrieval_data = []  # 当前任务的检索数据
        task_accept_lengths = []  # 当前任务的accept_length列表
        observations_data[task_id] = {}  # 初始化当前任务的observations字典
        
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            episode_retrieval_data = []  # 当前episode的检索数据
            episode_observations = []  # 当前episode的observations列表
            
            # 综合指标计算器（每个episode重置）
            metrics_calc = CompositeMetricsCalculatorMix(
                window_size=cfg.window_size,
                displacement_range=(cfg.displacement_range_min, cfg.displacement_range_max),
                radius_range=(cfg.radius_range_min, cfg.radius_range_max),
            )
            
            # 连续检索计数器：两次检索后强制AR
            consecutive_retrieval_count = 0
            
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
                    
                    # 保存observation到列表
                    episode_observations.append(observation)

                    # ========================================
                    # 使用实际机器人末端位置更新综合指标计算器
                    # ========================================
                    eef_position = obs["robot0_eef_pos"]  # shape: (3,)
                    metrics_calc.update_history(eef_position)
                    
                    # ========================================
                    # 基于综合指标决定是否使用检索
                    # ========================================
                    composite_metric = metrics_calc.compute_composite_metric(alpha=cfg.alpha)
                    metrics_info = metrics_calc.get_current_metrics(alpha=cfg.alpha)
                    
                    # 根据综合指标决定使用DB还是AR
                    if np.isnan(composite_metric):
                        # 历史不足时默认使用AR
                        use_db = False
                        decision_reason = "insufficient_history"
                    else:
                        # 综合指标 > threshold: 使用检索（DB）
                        # 综合指标 <= threshold: 使用AR
                        use_db = composite_metric > cfg.composite_threshold
                        decision_reason = f"composite={composite_metric:.4f}"
                    
                    # ========================================
                    # 两次连续检索后，强制执行一次AR
                    # ========================================
                    forced_ar = False
                    if use_db:
                        consecutive_retrieval_count += 1
                        if consecutive_retrieval_count > 2:
                            # 已经连续两次检索了，强制AR
                            use_db = False
                            forced_ar = True
                            consecutive_retrieval_count = 0
                            decision_reason = "forced_AR_after_2_retrievals"
                    else:
                        # 不使用DB（AR），重置计数器
                        consecutive_retrieval_count = 0

                    # ========================================
                    # 步骤1: 检索相似的action（只在DB模式时检索）
                    # ========================================
                    retrieved_action = None
                    retrieval_time = 0.0
                    retrieval_success = False
                    
                    if use_db:
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
                            
                            # 记录检索时间
                            t0_retrieval = time_module.time()
                            response = requests.post(RETRIEVAL_URL, files=files, data=data, timeout=30)
                            t1_retrieval = time_module.time()
                            retrieval_time = t1_retrieval - t0_retrieval
                            
                            if response.status_code == 200:
                                result = response.json()
                                if result.get('success', False):
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
                    # 步骤2: 将检索的action转换为tokens（如果检索成功）
                    # ========================================
                    retrieved_tokens = None
                    tokenization_time = 0.0
                    
                    if retrieval_success and retrieved_action is not None:
                        try:
                            t0_token = time_module.time()
                            retrieved_tokens = action_to_tokens(retrieved_action, model, cfg.unnorm_key)
                            t1_token = time_module.time()
                            tokenization_time = t1_token - t0_token
                        except Exception as e:
                            print(f"Tokenization error: {e}")
                            log_file.write(f"Tokenization error: {e}\n")
                            retrieved_tokens = None

                    # ========================================
                    # 步骤3: 根据模式生成或使用action
                    # ========================================
                    accept_length = -1
                    generation_time = 0.0
                    
                    if use_db:
                        # DB模式：直接使用检索的action
                        total_db_steps += 1
                        if retrieval_success and retrieved_action is not None:
                            action = retrieved_action.copy()
                            mode = "DB"
                        else:
                            # 检索失败，fallback到AR
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
                            mode = "DB_fallback_AR"
                    else:
                        # AR模式：使用AR生成
                        if forced_ar:
                            total_forced_ar_steps += 1
                            mode = "AR_forced"
                        else:
                            total_ar_steps += 1
                            mode = "AR"
                        
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
                        
                        # 如果有retrieved_tokens，计算accept_length（用于统计）
                        if retrieved_tokens is not None:
                            try:
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
                        'decision_reason': decision_reason,
                        'composite_metric': float(composite_metric) if not np.isnan(composite_metric) else None,
                        'raw_radius': float(metrics_info['raw']['radius']) if not np.isnan(metrics_info['raw']['radius']) else None,
                        'raw_displacement': float(metrics_info['raw']['displacement']) if not np.isnan(metrics_info['raw']['displacement']) else None,
                        'norm_radius': float(metrics_info['normalized']['radius']) if not np.isnan(metrics_info['normalized']['radius']) else None,
                        'norm_displacement': float(metrics_info['normalized']['displacement']) if not np.isnan(metrics_info['normalized']['displacement']) else None,
                        'retrieval_success': retrieval_success,
                        'retrieval_time': retrieval_time,
                        'tokenization_time': tokenization_time,
                        'generation_time': generation_time,
                        'accept_length': accept_length,
                        'has_retrieved_tokens': retrieved_tokens is not None,
                    }
                    episode_retrieval_data.append(step_data)

                    # Normalize gripper action
                    action = normalize_gripper_action(action, binarize=True)
                    
                    # Invert gripper action for OpenVLA (only for AR mode, DB actions are already correct)
                    if mode in ["AR", "AR_forced", "DB_fallback_AR"] and cfg.model_family == "openvla":
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
            
            # 将episode数据存入observations_data
            episode_success = done if 'done' in dir() else False
            observations_data[task_id][episode_idx] = {
                'observations': episode_observations,
                'success': episode_success,
                'task_description': task_description,
                'num_steps': len(episode_observations)
            }
            
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
        total_retrievals = sum(len(ep['steps']) for ep in task_retrieval_data)
        successful_retrievals = sum(
            sum(1 for step in ep['steps'] if step['retrieval_success']) 
            for ep in task_retrieval_data
        )
        
        # 分别统计各模式
        db_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'DB')
            for ep in task_retrieval_data
        )
        ar_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'AR')
            for ep in task_retrieval_data
        )
        ar_forced_steps_count = sum(
            sum(1 for step in ep['steps'] if step['mode'] == 'AR_forced')
            for ep in task_retrieval_data
        )
        
        print(f"\nTask {task_id} Statistics:")
        print(f"  Total steps: {total_retrievals}")
        print(f"  DB mode steps: {db_steps_count}")
        print(f"  AR mode steps: {ar_steps_count}")
        print(f"  AR_forced mode steps: {ar_forced_steps_count}")
        if total_retrievals > 0:
            print(f"  Successful retrievals: {successful_retrievals}")
            if len(task_accept_lengths) > 0:
                print(f"  Accept Length Stats:")
                print(f"    Mean: {np.mean(task_accept_lengths):.2f}")
                print(f"    Median: {np.median(task_accept_lengths):.2f}")
        
        log_file.write(f"\nTask {task_id} Statistics:\n")
        log_file.write(f"  Total steps: {total_retrievals}\n")
        log_file.write(f"  DB mode steps: {db_steps_count}\n")
        log_file.write(f"  AR mode steps: {ar_steps_count}\n")
        log_file.write(f"  AR_forced mode steps: {ar_forced_steps_count}\n")
        if total_retrievals > 0:
            log_file.write(f"  Successful retrievals: {successful_retrievals}\n")
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
    all_ar_forced_steps = sum(
        sum(1 for step in ep['steps'] if step['mode'] == 'AR_forced')
        for task_data in all_retrieval_data
        for ep in task_data['episodes']
    )
    total_all_steps = all_db_steps + all_ar_steps + all_ar_forced_steps
    
    print(f"\nMode Statistics:")
    print(f"  Total DB steps (noverify/检索): {all_db_steps} ({100.0*all_db_steps/total_all_steps:.1f}%)" if total_all_steps > 0 else "  Total DB steps: 0")
    print(f"  Total AR steps (verify): {all_ar_steps} ({100.0*all_ar_steps/total_all_steps:.1f}%)" if total_all_steps > 0 else "  Total AR steps: 0")
    print(f"  Total AR_forced steps: {all_ar_forced_steps} ({100.0*all_ar_forced_steps/total_all_steps:.1f}%)" if total_all_steps > 0 else "  Total AR_forced steps: 0")
    
    total_ar_all = all_ar_steps + all_ar_forced_steps
    print(f"\n  ============ 关键比例 ============")
    print(f"  verify (AR总计) : noverify (DB) = {total_ar_all}:{all_db_steps}")
    if all_db_steps > 0:
        print(f"  verify:noverify 比值 = {total_ar_all/all_db_steps:.2f}:1")
    if total_ar_all > 0:
        print(f"  noverify:verify 比值 = {all_db_steps/total_ar_all:.2f}:1")
    print(f"  ==================================")
    
    # ============ 时间统计 ============
    # 收集所有时间数据
    all_retrieval_times = []
    all_generation_times = []
    
    for task_data in all_retrieval_data:
        for ep in task_data['episodes']:
            for step in ep['steps']:
                if step['mode'] == 'DB' and step['retrieval_time'] > 0:
                    all_retrieval_times.append(step['retrieval_time'])
                if step['mode'] in ['AR', 'AR_forced', 'DB_fallback_AR'] and step['generation_time'] > 0:
                    all_generation_times.append(step['generation_time'])
    
    print(f"\n  ============ 时间统计 ============")
    if len(all_retrieval_times) > 0:
        avg_retrieval_time = np.mean(all_retrieval_times)
        std_retrieval_time = np.std(all_retrieval_times)
        print(f"  检索时间 (DB mode):")
        print(f"    Mean: {avg_retrieval_time*1000:.2f} ms")
        print(f"    Std:  {std_retrieval_time*1000:.2f} ms")
        print(f"    Samples: {len(all_retrieval_times)}")
    else:
        print(f"  检索时间: 无数据")
        avg_retrieval_time = 0
        std_retrieval_time = 0
    
    if len(all_generation_times) > 0:
        avg_generation_time = np.mean(all_generation_times)
        std_generation_time = np.std(all_generation_times)
        print(f"  AR生成时间 (get_action):")
        print(f"    Mean: {avg_generation_time*1000:.2f} ms")
        print(f"    Std:  {std_generation_time*1000:.2f} ms")
        print(f"    Samples: {len(all_generation_times)}")
    else:
        print(f"  AR生成时间: 无数据")
        avg_generation_time = 0
        std_generation_time = 0
    
    # 计算平均每步时间（加权）
    if total_all_steps > 0 and (len(all_retrieval_times) > 0 or len(all_generation_times) > 0):
        # 估算平均每步时间
        weighted_avg_time = (all_db_steps * avg_retrieval_time + total_ar_all * avg_generation_time) / total_all_steps
        print(f"\n  加权平均每步时间: {weighted_avg_time*1000:.2f} ms")
        if avg_generation_time > 0:
            speedup = avg_generation_time / weighted_avg_time
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
    
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("Overall Statistics:\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Total episodes: {total_episodes}\n")
    log_file.write(f"Total successes: {total_successes}\n")
    log_file.write(f"Success rate: {total_successes/total_episodes*100:.1f}%\n")
    log_file.write(f"\nMode Statistics:\n")
    log_file.write(f"  Total DB steps (noverify/检索): {all_db_steps} ({100.0*all_db_steps/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "  Total DB steps: 0\n")
    log_file.write(f"  Total AR steps (verify): {all_ar_steps} ({100.0*all_ar_steps/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "  Total AR steps: 0\n")
    log_file.write(f"  Total AR_forced steps: {all_ar_forced_steps} ({100.0*all_ar_forced_steps/total_all_steps:.1f}%)\n" if total_all_steps > 0 else "  Total AR_forced steps: 0\n")
    
    log_file.write(f"\n  ============ 关键比例 ============\n")
    log_file.write(f"  verify (AR总计) : noverify (DB) = {total_ar_all}:{all_db_steps}\n")
    if all_db_steps > 0:
        log_file.write(f"  verify:noverify 比值 = {total_ar_all/all_db_steps:.2f}:1\n")
    if total_ar_all > 0:
        log_file.write(f"  noverify:verify 比值 = {all_db_steps/total_ar_all:.2f}:1\n")
    log_file.write(f"  ==================================\n")
    
    # 时间统计写入日志
    log_file.write(f"\n  ============ 时间统计 ============\n")
    if len(all_retrieval_times) > 0:
        log_file.write(f"  检索时间 (DB mode):\n")
        log_file.write(f"    Mean: {avg_retrieval_time*1000:.2f} ms\n")
        log_file.write(f"    Std:  {std_retrieval_time*1000:.2f} ms\n")
        log_file.write(f"    Samples: {len(all_retrieval_times)}\n")
    else:
        log_file.write(f"  检索时间: 无数据\n")
    
    if len(all_generation_times) > 0:
        log_file.write(f"  AR生成时间 (get_action):\n")
        log_file.write(f"    Mean: {avg_generation_time*1000:.2f} ms\n")
        log_file.write(f"    Std:  {std_generation_time*1000:.2f} ms\n")
        log_file.write(f"    Samples: {len(all_generation_times)}\n")
    else:
        log_file.write(f"  AR生成时间: 无数据\n")
    
    if total_all_steps > 0 and (len(all_retrieval_times) > 0 or len(all_generation_times) > 0):
        weighted_avg_time = (all_db_steps * avg_retrieval_time + total_ar_all * avg_generation_time) / total_all_steps
        log_file.write(f"\n  加权平均每步时间: {weighted_avg_time*1000:.2f} ms\n")
        if avg_generation_time > 0:
            speedup = avg_generation_time / weighted_avg_time
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
    
    # 保存observations数据（包含轨迹状态）
    local_log_obspath = os.path.join(target_dir, run_id + "_observations.npy")
    np.save(local_log_obspath, observations_data, allow_pickle=True)
    print(f"Observations data saved to: {local_log_obspath}")
    log_file.write(f"Observations data saved to: {local_log_obspath}\n")
    
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
