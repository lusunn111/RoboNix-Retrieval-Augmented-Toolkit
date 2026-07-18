"""
run_fenxi_top5_ambiguity.py

分析每一步检索top5的歧义性和置信度。
- 歧义性：top5 action相对于7维重心的平均距离
- 置信度：top1-5的相似度分数

直接调用Qdrant而非检索服务器，以获取top5结果。

Usage:
    python experiments/robot/libero/run_fenxi_top5_ambiguity.py \
        --task_suite_name libero_goal \
        --num_trials_per_task 1
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Dict
import hashlib
import base64
from io import BytesIO

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import requests
from PIL import Image
import torch
import matplotlib.pyplot as plt
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
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    set_seed_everywhere,
)


# ============================================
# 配置
# ============================================
EMBEDDING_URL = "http://127.0.0.1:9021/predict"  # Mix embedding server
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TOP_K = 5  # 获取top5

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


def compute_ambiguity(actions: List[np.ndarray]) -> float:
    """
    计算top-k actions的歧义性
    定义：所有action相对于7维重心的平均欧氏距离
    
    Args:
        actions: List of action arrays, each shape (7,)
        
    Returns:
        ambiguity: 平均距离（歧义性越大表示检索结果越分散）
    """
    if len(actions) == 0:
        return np.nan
    
    actions_array = np.array(actions)  # shape: (k, 7)
    
    # 计算重心
    centroid = np.mean(actions_array, axis=0)  # shape: (7,)
    
    # 计算每个action到重心的距离
    distances = [np.linalg.norm(a - centroid) for a in actions_array]
    
    # 返回平均距离
    return np.mean(distances)


@dataclass
class GenerateConfig:
    # fmt: off
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    run_id_note: Optional[str] = None
    seed: int = 7
    # fmt: on


@draccus.wrap()
def run_analysis(cfg: GenerateConfig) -> None:
    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    # 初始化Qdrant客户端
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # 获取数据集类型
    dataset_type = normalize_dataset_type(cfg.task_suite_name)
    if dataset_type is None:
        print(f"Unknown dataset type: {cfg.task_suite_name}")
        return
    collection_prefix = DATASET_CONFIGS[dataset_type]
    print(f"Using collection prefix: {collection_prefix}")
    
    # 初始化输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "figs", "top5_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}, {num_tasks_in_suite} tasks")
    
    # 预加载payload缓存
    print("Preloading payload cache...")
    payload_cache = {}
    
    # 存储所有数据
    all_task_data = {}
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        
        # 获取collection名
        qdrant_task_id = get_task_id(task_description)
        collection_name = f"{collection_prefix}{qdrant_task_id}"
        
        # 检查collection是否存在
        if not qdrant_client.collection_exists(collection_name):
            print(f"Collection {collection_name} does not exist, skipping task: {task_description}")
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
        
        # 任务数据
        task_data = {
            'task_description': task_description,
            'ambiguities': [],  # 每步的歧义性
            'top5_scores': [],  # 每步的top5置信度 shape: (steps, 5)
        }
        
        for episode_idx in range(cfg.num_trials_per_task):
            print(f"\nTask: {task_description}, Episode: {episode_idx + 1}")
            
            # Reset environment
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            
            t = 0
            
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
                        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                        t += 1
                        continue
                    
                    # Get images
                    img = get_libero_image(obs, (224, 224))
                    wrist_img = get_libero_wrist_image(obs, (224, 224))
                    
                    # Generate embedding
                    pil_third = Image.fromarray(img)
                    pil_wrist = Image.fromarray(wrist_img)
                    
                    embedding = generate_mix_embedding(pil_third, pil_wrist, task_description)
                    
                    if embedding is None:
                        task_data['ambiguities'].append(np.nan)
                        task_data['top5_scores'].append([np.nan] * TOP_K)
                        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                        t += 1
                        continue
                    
                    # Search in Qdrant (version-agnostic)
                    try:
                        search_results = search_points(
                            qdrant_client,
                            collection_name=collection_name,
                            query_vector=embedding.tolist(),
                            limit=TOP_K,
                        )
                    except Exception as e:
                        print(f"Search failed: {e}")
                        task_data['ambiguities'].append(np.nan)
                        task_data['top5_scores'].append([np.nan] * TOP_K)
                        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                        t += 1
                        continue
                    
                    if not search_results or len(search_results) == 0:
                        task_data['ambiguities'].append(np.nan)
                        task_data['top5_scores'].append([np.nan] * TOP_K)
                        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                        t += 1
                        continue
                    
                    # 提取top5的scores
                    scores = []
                    actions = []
                    
                    for i, result in enumerate(search_results[:TOP_K]):
                        scores.append(result.score)
                        
                        # 获取payload
                        point_id = str(result.id)
                        payload = payload_cache.get(collection_name, {}).get(point_id)
                        if payload is None:
                            payload = result.payload
                        
                        if payload and 'current_action' in payload:
                            action = np.array(payload['current_action'])
                            actions.append(action)
                    
                    # 补齐scores到TOP_K
                    while len(scores) < TOP_K:
                        scores.append(np.nan)
                    
                    # 计算歧义性
                    if len(actions) >= 2:
                        ambiguity = compute_ambiguity(actions)
                    else:
                        ambiguity = np.nan
                    
                    task_data['ambiguities'].append(ambiguity)
                    task_data['top5_scores'].append(scores)
                    
                    # Execute dummy action (we don't care about execution result)
                    obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                    if done:
                        break
                    t += 1
                    
                except Exception as e:
                    print(f"Error at step {t}: {e}")
                    task_data['ambiguities'].append(np.nan)
                    task_data['top5_scores'].append([np.nan] * TOP_K)
                    break
        
        # 保存任务数据
        all_task_data[task_id] = task_data
        
        # 绘制当前任务的图
        plot_task_analysis(task_data, task_id, output_dir)
    
    # 绘制汇总图
    plot_summary(all_task_data, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


def plot_task_analysis(task_data: Dict, task_id: int, output_dir: str):
    """绘制单个任务的分析图"""
    task_desc = task_data['task_description']
    ambiguities = np.array(task_data['ambiguities'])
    scores = np.array(task_data['top5_scores'])  # shape: (steps, 5)
    
    # 创建安全的文件名
    safe_name = task_desc.lower().replace(' ', '_').replace('/', '_')[:50]
    
    # ====== 图1: 歧义性折线图 ======
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    
    steps = np.arange(len(ambiguities))
    valid_mask = ~np.isnan(ambiguities)
    
    if np.sum(valid_mask) > 0:
        ax1.plot(steps[valid_mask], ambiguities[valid_mask], 
                 'b-', linewidth=1.5, alpha=0.8, label='Ambiguity')
        ax1.scatter(steps[valid_mask], ambiguities[valid_mask], 
                    c='blue', s=20, alpha=0.5)
        
        mean_amb = np.nanmean(ambiguities)
        ax1.axhline(y=mean_amb, color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {mean_amb:.4f}')
    
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Ambiguity (Avg Distance to Centroid)', fontsize=12)
    ax1.set_title(f'Top-5 Retrieval Ambiguity\n{task_desc}', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, f'{safe_name}_ambiguity.png'), dpi=150)
    plt.close(fig1)
    
    # ====== 图2: 置信度折线图（5条线，每条代表一个top-i） ======
    if len(scores) > 0 and scores.shape[1] == TOP_K:
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        
        steps = np.arange(len(scores))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5种颜色
        
        # 绘制每个top-i的置信度曲线
        for i in range(TOP_K):
            top_i_scores = scores[:, i]
            valid_mask = ~np.isnan(top_i_scores)
            if np.sum(valid_mask) > 0:
                ax2.plot(steps[valid_mask], top_i_scores[valid_mask], 
                         color=colors[i], linewidth=1.5, alpha=0.8, 
                         label=f'Top-{i+1}')
        
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Confidence (Similarity Score)', fontsize=12)
        ax2.set_title(f'Top-5 Confidence over Steps\n{task_desc}', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, f'{safe_name}_confidence_lines.png'), dpi=150)
        plt.close(fig2)
        
        # ====== 图3: 置信度折线图（MinMax归一化版本） ======
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        
        # MinMax归一化每一步的scores
        scores_normalized = np.zeros_like(scores)
        for i in range(len(scores)):
            row = scores[i]
            valid = ~np.isnan(row)
            if np.sum(valid) > 1:
                min_val = np.nanmin(row)
                max_val = np.nanmax(row)
                if max_val > min_val:
                    scores_normalized[i] = (row - min_val) / (max_val - min_val)
                else:
                    scores_normalized[i] = 0.5  # 所有值相同
            else:
                scores_normalized[i] = row
        
        # 绘制每个top-i的归一化置信度曲线
        for i in range(TOP_K):
            top_i_scores = scores_normalized[:, i]
            valid_mask = ~np.isnan(top_i_scores)
            if np.sum(valid_mask) > 0:
                ax3.plot(steps[valid_mask], top_i_scores[valid_mask], 
                         color=colors[i], linewidth=1.5, alpha=0.8, 
                         label=f'Top-{i+1}')
        
        ax3.set_xlabel('Step', fontsize=12)
        ax3.set_ylabel('Normalized Confidence (MinMax per Step)', fontsize=12)
        ax3.set_title(f'Top-5 Confidence (MinMax Normalized)\n{task_desc}', fontsize=12)
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir, f'{safe_name}_confidence_normalized.png'), dpi=150)
        plt.close(fig3)
    
    print(f"  Saved plots for: {task_desc}")


def plot_summary(all_task_data: Dict, output_dir: str):
    """绘制所有任务的汇总图"""
    # 收集所有歧义性数据
    all_ambiguities = []
    task_names = []
    
    for task_id, data in all_task_data.items():
        valid_amb = [a for a in data['ambiguities'] if not np.isnan(a)]
        if len(valid_amb) > 0:
            all_ambiguities.append(valid_amb)
            task_names.append(data['task_description'][:30])
    
    if len(all_ambiguities) == 0:
        print("No valid data for summary plot")
        return
    
    # ====== 汇总图1: 各任务歧义性箱线图 ======
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    
    ax1.boxplot(all_ambiguities, labels=range(1, len(task_names) + 1))
    ax1.set_xlabel('Task ID', fontsize=12)
    ax1.set_ylabel('Ambiguity', fontsize=12)
    ax1.set_title('Top-5 Retrieval Ambiguity Distribution by Task', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加任务名注释
    for i, name in enumerate(task_names):
        ax1.annotate(name, (i + 1, ax1.get_ylim()[0]), 
                     rotation=45, ha='right', fontsize=7, alpha=0.7)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'summary_ambiguity_boxplot.png'), dpi=150)
    plt.close(fig1)
    
    # ====== 汇总图2: 全局歧义性分布 ======
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    all_amb_flat = [a for amb_list in all_ambiguities for a in amb_list]
    ax2.hist(all_amb_flat, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=np.mean(all_amb_flat), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(all_amb_flat):.4f}')
    ax2.axvline(x=np.median(all_amb_flat), color='green', linestyle=':', 
                linewidth=2, label=f'Median: {np.median(all_amb_flat):.4f}')
    
    ax2.set_xlabel('Ambiguity', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Global Ambiguity Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'summary_ambiguity_distribution.png'), dpi=150)
    plt.close(fig2)
    
    print(f"\nSummary plots saved to {output_dir}")


if __name__ == "__main__":
    run_analysis()
