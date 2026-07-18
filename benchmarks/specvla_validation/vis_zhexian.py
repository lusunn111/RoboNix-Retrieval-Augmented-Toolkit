"""
vis_zhexian.py

用于生成折线图数据并绘制的脚本
包含：位移指标、曲率半径、加权指标的折线图

Usage:
    python vis_zhexian.py  # 生成数据并绘图
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ================================
# 配置
# ================================
# libero_goal 数据路径
RETRIEVAL_NPY_PATH = "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_goal_naive_DB_mix/EVAL-libero_goal-NaiveDB-Mix-2026_01_21-14_38_17_observations.npy"
SD_NPY_PATH = "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_goal_Retrieval_Verify/EVAL-libero_goal-openvla-2026_01_17-12_50_04--record_guiji_suc_fail_pure_sd_observations.npy"

# 目标任务名
TARGET_TASK_NAME = "push the plate to the front of the stove"

# 目标轨迹对 (r_eid, s_eid) - r3_s1 表示 retrieval episode 3, sd episode 1
TARGET_R_EID = 3
TARGET_S_EID = 1

# 输出目录
OUTPUT_DIR = "/path/to/SpecVLA/zhexian_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 窗口大小
WINDOW_SIZE = 5


# ================================
# 数据加载函数
# ================================
def load_observations(npy_path):
    """加载 observations.npy 文件"""
    data = np.load(npy_path, allow_pickle=True)
    return data.item()


def get_trajectory_info(obs_dict):
    """获取轨迹信息摘要"""
    info = {}
    for task_id in obs_dict.keys():
        info[task_id] = {}
        for episode_idx in obs_dict[task_id].keys():
            episode_data = obs_dict[task_id][episode_idx]
            info[task_id][episode_idx] = {
                "success": episode_data["success"],
                "task_description": episode_data["task_description"],
                "num_steps": episode_data["num_steps"],
            }
    return info


def get_task_name_from_description(task_description):
    """将任务描述转换为文件名友好的格式"""
    clean_desc = re.sub(r'[^\w\s]', '', task_description.lower())
    task_name = '_'.join(clean_desc.split())
    if len(task_name) > 50:
        task_name = task_name[:50]
    return task_name


def find_trajectory_pairs(retrieval_info, sd_info, target_task_name=None):
    """找到不同成功状态组合的轨迹对（同一任务内交叉配对）"""
    both_success = []
    sd_success_retrieval_fail = []
    retrieval_success_sd_fail = []
    both_fail = []

    for task_id in retrieval_info.keys():
        if task_id not in sd_info:
            continue

        retrieval_success_episodes = []
        retrieval_fail_episodes = []
        sd_success_episodes = []
        sd_fail_episodes = []

        task_desc = None
        for r_eid in retrieval_info[task_id].keys():
            if task_desc is None:
                task_desc = retrieval_info[task_id][r_eid]["task_description"]

        if target_task_name is not None and task_desc is not None:
            if target_task_name.lower() not in task_desc.lower():
                continue

        for r_eid in retrieval_info[task_id].keys():
            if retrieval_info[task_id][r_eid]["success"]:
                retrieval_success_episodes.append(r_eid)
            else:
                retrieval_fail_episodes.append(r_eid)

        for s_eid in sd_info[task_id].keys():
            if sd_info[task_id][s_eid]["success"]:
                sd_success_episodes.append(s_eid)
            else:
                sd_fail_episodes.append(s_eid)

        for r_eid in retrieval_success_episodes:
            for s_eid in sd_success_episodes:
                both_success.append((task_id, r_eid, s_eid, task_desc))

        for r_eid in retrieval_fail_episodes:
            for s_eid in sd_success_episodes:
                sd_success_retrieval_fail.append((task_id, r_eid, s_eid, task_desc))

        for r_eid in retrieval_success_episodes:
            for s_eid in sd_fail_episodes:
                retrieval_success_sd_fail.append((task_id, r_eid, s_eid, task_desc))

        for r_eid in retrieval_fail_episodes:
            for s_eid in sd_fail_episodes:
                both_fail.append((task_id, r_eid, s_eid, task_desc))

    return both_success, sd_success_retrieval_fail, retrieval_success_sd_fail, both_fail


def extract_trajectory_xyz(episode_data):
    """从 episode 数据中提取 xyz 轨迹坐标"""
    observations = episode_data["observations"]
    states = np.array([obs["state"] for obs in observations])
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    return x, y, z


# ================================
# 指标计算函数
# ================================
def least_squares_circle_fit_radius(points):
    """使用最小二乘法拟合圆，返回半径"""
    if len(points) < 3:
        return np.nan

    center = np.mean(points, axis=0)
    points_centered = points - center

    _, _, vh = np.linalg.svd(points_centered)
    normal = vh[2, :]

    points_2d = points_centered - np.outer(np.dot(points_centered, normal), normal)

    u = vh[0, :]
    v = vh[1, :]
    x = np.dot(points_2d, u)
    y = np.dot(points_2d, v)

    def calc_R(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.array([x.mean(), y.mean()])
    result = least_squares(f, center_estimate)
    xc, yc = result.x
    Ri = calc_R(xc, yc)
    R = Ri.mean()

    return R if R > 1e-6 else np.nan


def compute_radius_least_squares(x, y, z, window_size=5):
    """使用滑动窗口最小二乘圆拟合计算曲率半径"""
    trajectory = np.column_stack([x, y, z])
    n = len(trajectory)
    radii = []

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)

        if end - start < 3:
            radii.append(np.nan)
            continue

        window_points = trajectory[start:end]
        radius = least_squares_circle_fit_radius(window_points)
        radii.append(radius)

    return np.array(radii)


def compute_displacement_metric(x, y, z, window_size=5):
    """计算位移指标：在滑动窗口内，窗口最后一个点与前面所有点的欧式距离之和"""
    trajectory = np.column_stack([x, y, z])
    n = len(trajectory)
    displacement_metrics = []

    for i in range(n):
        start = max(0, i - window_size + 1)
        end = i + 1

        if end - start < 2:
            displacement_metrics.append(np.nan)
            continue

        window_points = trajectory[start:end]
        last_point = window_points[-1]

        total_distance = 0.0
        for j in range(len(window_points) - 1):
            dist = np.linalg.norm(last_point - window_points[j])
            total_distance += dist

        displacement_metrics.append(total_distance)

    return np.array(displacement_metrics)


def minmax_normalize(values):
    """MinMax 归一化"""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return values

    min_val = np.nanmin(values)
    max_val = np.nanmax(values)

    if max_val - min_val < 1e-10:
        return np.where(~np.isnan(values), 0.5, np.nan)

    return (values - min_val) / (max_val - min_val)


def compute_weighted_metric(radius, displacement, weight_r=0.5):
    """计算加权平均指标"""
    radius_norm = minmax_normalize(radius)
    displacement_norm = minmax_normalize(displacement)
    weighted = weight_r * radius_norm + (1 - weight_r) * displacement_norm
    return weighted, radius_norm, displacement_norm


# ================================
# 绘图函数 - 保持与 notebook 一致
# ================================
def plot_curvature_radius_comparison(r_radius_norm, s_radius_norm, save_path=None):
    """绘制曲率半径对比折线图（归一化）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 从第5个点开始绘制（跳过前5个点）
    start_idx = 5
    
    # 绘制检索轨迹的归一化曲率半径
    steps_r = np.arange(len(r_radius_norm))
    valid_r = ~np.isnan(r_radius_norm)
    valid_r = valid_r & (steps_r >= start_idx)
    ax.plot(steps_r[valid_r], r_radius_norm[valid_r], 
            'b-', linewidth=4, alpha=0.8, 
            marker='^', markersize=7, 
            markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=1)
    
    # 绘制SD轨迹的归一化曲率半径
    steps_s = np.arange(len(s_radius_norm))
    valid_s = ~np.isnan(s_radius_norm)
    valid_s = valid_s & (steps_s >= start_idx)
    ax.plot(steps_s[valid_s], s_radius_norm[valid_s], 
            color='darkorange', linewidth=4, alpha=0.8, 
            marker='^', markersize=7, 
            markerfacecolor='darkorange', markeredgecolor='orangered', markeredgewidth=1)
    
    # 移除轴标签、标题、刻度标签，但保留grid
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"曲率半径图已保存: {save_path}")
    
    plt.close()
    return save_path


def plot_displacement_metric_comparison(r_displacement_norm, s_displacement_norm, save_path=None):
    """绘制位移指标对比折线图（归一化）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 从第5个点开始绘制
    start_idx = 5
    
    # 绘制检索轨迹的归一化位移指标
    steps_r = np.arange(len(r_displacement_norm))
    valid_r = ~np.isnan(r_displacement_norm)
    valid_r = valid_r & (steps_r >= start_idx)
    ax.plot(steps_r[valid_r], r_displacement_norm[valid_r], 
            'b-', linewidth=4, alpha=0.8, 
            marker='*', markersize=10, 
            markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=1)
    
    # 绘制SD轨迹的归一化位移指标
    steps_s = np.arange(len(s_displacement_norm))
    valid_s = ~np.isnan(s_displacement_norm)
    valid_s = valid_s & (steps_s >= start_idx)
    ax.plot(steps_s[valid_s], s_displacement_norm[valid_s], 
            color='darkorange', linewidth=4, alpha=0.8, 
            marker='*', markersize=12, 
            markerfacecolor='darkorange', markeredgecolor='orangered', markeredgewidth=1)
    
    # 移除轴标签、标题、刻度标签
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"位移指标图已保存: {save_path}")
    
    plt.close()
    return save_path


def plot_weighted_metric_comparison(r_weighted, s_weighted, save_path=None):
    """绘制加权指标对比折线图（归一化）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 从第5个点开始绘制
    start_idx = 5
    
    # 绘制检索轨迹的归一化加权指标
    steps_r = np.arange(len(r_weighted))
    valid_r = ~np.isnan(r_weighted)
    valid_r = valid_r & (steps_r >= start_idx)
    ax.plot(steps_r[valid_r], r_weighted[valid_r], 
            'b-', linewidth=4, alpha=0.8, 
            marker='s', markersize=7, 
            markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=1)
    
    # 绘制SD轨迹的归一化加权指标
    steps_s = np.arange(len(s_weighted))
    valid_s = ~np.isnan(s_weighted)
    valid_s = valid_s & (steps_s >= start_idx)
    ax.plot(steps_s[valid_s], s_weighted[valid_s], 
            color='darkorange', linewidth=4, alpha=0.8, 
            marker='s', markersize=7, 
            markerfacecolor='darkorange', markeredgecolor='orangered', markeredgewidth=1)
    
    # 移除轴标签、标题、刻度标签
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"加权指标图已保存: {save_path}")
    
    plt.close()
    return save_path


# ================================
# 从 .npy 文件绘图的函数
# ================================
def plot_from_npy(npy_path, output_dir=None):
    """
    从 .npy 文件加载数据并绘制折线图
    
    参数:
    - npy_path: .npy 文件路径
    - output_dir: 输出目录（可选，默认与 npy 文件同目录）
    """
    print(f"加载数据: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()
    
    if output_dir is None:
        output_dir = os.path.dirname(npy_path)
    
    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    
    # 提取数据
    r_radius_norm = data['r_radius_norm']
    s_radius_norm = data['s_radius_norm']
    r_displacement_norm = data['r_displacement_norm']
    s_displacement_norm = data['s_displacement_norm']
    r_weighted = data['r_weighted']
    s_weighted = data['s_weighted']
    
    # 绘制曲率半径图
    radius_path = os.path.join(output_dir, f"{base_name}_radius.png")
    plot_curvature_radius_comparison(r_radius_norm, s_radius_norm, save_path=radius_path)
    
    # 绘制位移指标图
    displacement_path = os.path.join(output_dir, f"{base_name}_displacement.png")
    plot_displacement_metric_comparison(r_displacement_norm, s_displacement_norm, save_path=displacement_path)
    
    # 绘制加权指标图
    weighted_path = os.path.join(output_dir, f"{base_name}_weighted.png")
    plot_weighted_metric_comparison(r_weighted, s_weighted, save_path=weighted_path)
    
    print(f"所有图片已保存到: {output_dir}")
    
    return {
        'radius_path': radius_path,
        'displacement_path': displacement_path,
        'weighted_path': weighted_path
    }


# ================================
# 主函数
# ================================
def main():
    print("=" * 60)
    print("折线图数据生成与绘制")
    print("=" * 60)
    
    # 加载数据
    print(f"\n加载数据...")
    print(f"Retrieval: {RETRIEVAL_NPY_PATH}")
    print(f"SD: {SD_NPY_PATH}")
    
    retrieval_data = load_observations(RETRIEVAL_NPY_PATH)
    sd_data = load_observations(SD_NPY_PATH)
    
    retrieval_info = get_trajectory_info(retrieval_data)
    sd_info = get_trajectory_info(sd_data)
    
    print(f"检索数据 - 任务数: {len(retrieval_data)}")
    print(f"SD数据 - 任务数: {len(sd_data)}")
    
    # 打印所有任务
    print("\n可用任务列表:")
    for task_id in retrieval_info.keys():
        for eid in retrieval_info[task_id].keys():
            task_desc = retrieval_info[task_id][eid]['task_description']
            print(f"  Task {task_id}: {task_desc}")
            break
    
    # 找到目标任务的轨迹对
    print(f"\n目标任务: {TARGET_TASK_NAME}")
    both_success, _, _, _ = find_trajectory_pairs(
        retrieval_info, sd_info, target_task_name=TARGET_TASK_NAME
    )
    
    print(f"两种方法都成功的轨迹对数量: {len(both_success)}")
    
    if len(both_success) == 0:
        print("错误: 没有找到符合条件的轨迹对!")
        return
    
    # 打印所有轨迹对
    print("\n所有轨迹对:")
    for i, (tid, r_eid, s_eid, desc) in enumerate(both_success):
        print(f"  [{i}] Task {tid}, r{r_eid}_s{s_eid}: {desc[:50]}...")
    
    # 查找目标轨迹对
    target_pair = None
    for tid, r_eid, s_eid, desc in both_success:
        if r_eid == TARGET_R_EID and s_eid == TARGET_S_EID:
            target_pair = (tid, r_eid, s_eid, desc)
            break
    
    if target_pair is None:
        print(f"\n警告: 未找到 r{TARGET_R_EID}_s{TARGET_S_EID} 轨迹对!")
        print(f"使用第一个轨迹对代替...")
        target_pair = both_success[0]
    
    task_id, r_eid, s_eid, task_desc = target_pair
    print(f"\n处理轨迹对: Task {task_id}, r{r_eid}_s{s_eid}")
    print(f"任务描述: {task_desc}")
    
    # 提取轨迹
    r_episode = retrieval_data[task_id][r_eid]
    s_episode = sd_data[task_id][s_eid]
    
    r_xyz = extract_trajectory_xyz(r_episode)
    s_xyz = extract_trajectory_xyz(s_episode)
    
    print(f"检索轨迹长度: {len(r_xyz[0])} steps")
    print(f"SD轨迹长度: {len(s_xyz[0])} steps")
    
    # 计算指标
    print("\n计算指标...")
    
    # 曲率半径
    r_radius = compute_radius_least_squares(r_xyz[0], r_xyz[1], r_xyz[2], window_size=WINDOW_SIZE)
    s_radius = compute_radius_least_squares(s_xyz[0], s_xyz[1], s_xyz[2], window_size=WINDOW_SIZE)
    
    # 位移指标
    r_displacement = compute_displacement_metric(r_xyz[0], r_xyz[1], r_xyz[2], window_size=WINDOW_SIZE)
    s_displacement = compute_displacement_metric(s_xyz[0], s_xyz[1], s_xyz[2], window_size=WINDOW_SIZE)
    
    # 归一化
    r_radius_norm = minmax_normalize(r_radius)
    s_radius_norm = minmax_normalize(s_radius)
    r_displacement_norm = minmax_normalize(r_displacement)
    s_displacement_norm = minmax_normalize(s_displacement)
    
    # 加权指标 (weight_r=0.5)
    r_weighted, _, _ = compute_weighted_metric(r_radius, r_displacement, weight_r=0.5)
    s_weighted, _, _ = compute_weighted_metric(s_radius, s_displacement, weight_r=0.5)
    
    # 保存数据到 .npy 文件
    task_name = get_task_name_from_description(task_desc)
    npy_filename = f"{task_name}_both_success_r{r_eid}_s{s_eid}_zhexian_data.npy"
    npy_path = os.path.join(OUTPUT_DIR, npy_filename)
    
    save_data = {
        'task_id': task_id,
        'r_eid': r_eid,
        's_eid': s_eid,
        'task_desc': task_desc,
        'task_name': task_name,
        'window_size': WINDOW_SIZE,
        # 原始轨迹坐标
        'r_xyz': r_xyz,
        's_xyz': s_xyz,
        # 原始指标
        'r_radius': r_radius,
        's_radius': s_radius,
        'r_displacement': r_displacement,
        's_displacement': s_displacement,
        # 归一化后的指标（用于绘图）
        'r_radius_norm': r_radius_norm,
        's_radius_norm': s_radius_norm,
        'r_displacement_norm': r_displacement_norm,
        's_displacement_norm': s_displacement_norm,
        # 加权指标
        'r_weighted': r_weighted,
        's_weighted': s_weighted,
    }
    
    np.save(npy_path, save_data)
    print(f"\n数据已保存: {npy_path}")
    
    # 绘制折线图
    print("\n绘制折线图...")
    
    # 曲率半径图
    radius_path = os.path.join(OUTPUT_DIR, f"{task_name}_both_success_r{r_eid}_s{s_eid}_radius.png")
    plot_curvature_radius_comparison(r_radius_norm, s_radius_norm, save_path=radius_path)
    
    # 位移指标图
    displacement_path = os.path.join(OUTPUT_DIR, f"{task_name}_both_success_r{r_eid}_s{s_eid}_displacement.png")
    plot_displacement_metric_comparison(r_displacement_norm, s_displacement_norm, save_path=displacement_path)
    
    # 加权指标图
    weighted_path = os.path.join(OUTPUT_DIR, f"{task_name}_both_success_r{r_eid}_s{s_eid}_weighted.png")
    plot_weighted_metric_comparison(r_weighted, s_weighted, save_path=weighted_path)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  数据文件: {npy_path}")
    print(f"  曲率半径图: {radius_path}")
    print(f"  位移指标图: {displacement_path}")
    print(f"  加权指标图: {weighted_path}")
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"  Retrieval 位移指标: min={np.nanmin(r_displacement):.4f}, max={np.nanmax(r_displacement):.4f}, mean={np.nanmean(r_displacement):.4f}")
    print(f"  SD 位移指标: min={np.nanmin(s_displacement):.4f}, max={np.nanmax(s_displacement):.4f}, mean={np.nanmean(s_displacement):.4f}")
    print(f"  Retrieval 曲率半径: min={np.nanmin(r_radius):.4f}, max={np.nanmax(r_radius):.4f}, mean={np.nanmean(r_radius):.4f}")
    print(f"  SD 曲率半径: min={np.nanmin(s_radius):.4f}, max={np.nanmax(s_radius):.4f}, mean={np.nanmean(s_radius):.4f}")


if __name__ == "__main__":
    main()
