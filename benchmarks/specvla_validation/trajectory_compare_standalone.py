import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy.optimize import least_squares


def load_observations(npy_path):
    data = np.load(npy_path, allow_pickle=True)
    return data.item()


def get_trajectory_info(obs_dict):
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
    clean_desc = re.sub(r"[^\w\s]", "", task_description.lower())
    task_name = "_".join(clean_desc.split())
    if len(task_name) > 50:
        task_name = task_name[:50]
    return task_name


def extract_trajectory_xyz(episode_data):
    observations = episode_data["observations"]
    states = np.array([obs["state"] for obs in observations])
    return states[:, 0], states[:, 1], states[:, 2]


def compute_point_distances(traj1_xyz, traj2_xyz):
    x1, y1, z1 = traj1_xyz
    x2, y2, z2 = traj2_xyz
    min_len = min(len(x1), len(x2))
    distances = []
    for i in range(min_len):
        dist = np.sqrt((x1[i] - x2[i]) ** 2 + (y1[i] - y2[i]) ** 2 + (z1[i] - z2[i]) ** 2)
        distances.append(dist)
    return np.array(distances)


def find_overlapping_points(distances, threshold=0.02):
    return np.where(distances < threshold)[0]


def least_squares_circle_fit_radius(points):
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

    def calc_r(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def residuals(c):
        ri = calc_r(*c)
        return ri - ri.mean()

    center_estimate = np.array([x.mean(), y.mean()])
    result = least_squares(residuals, center_estimate)
    xc, yc = result.x
    ri = calc_r(xc, yc)
    radius = ri.mean()
    return radius if radius > 1e-6 else np.nan


def compute_radius_least_squares(x, y, z, window_size=5):
    trajectory = np.column_stack([x, y, z])
    radii = []
    for i in range(len(trajectory)):
        start = max(0, i - window_size // 2)
        end = min(len(trajectory), i + window_size // 2 + 1)
        if end - start < 3:
            radii.append(np.nan)
            continue
        radii.append(least_squares_circle_fit_radius(trajectory[start:end]))
    return np.array(radii)


def compute_displacement_metric(x, y, z, window_size=5):
    trajectory = np.column_stack([x, y, z])
    displacement_metrics = []
    for i in range(len(trajectory)):
        start = max(0, i - window_size + 1)
        end = i + 1
        if end - start < 2:
            displacement_metrics.append(np.nan)
            continue
        window_points = trajectory[start:end]
        last_point = window_points[-1]
        total_distance = 0.0
        for j in range(len(window_points) - 1):
            total_distance += np.linalg.norm(last_point - window_points[j])
        displacement_metrics.append(total_distance)
    return np.array(displacement_metrics)


def minmax_normalize(values):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return values
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    if max_val - min_val < 1e-10:
        return np.where(~np.isnan(values), 0.5, np.nan)
    return (values - min_val) / (max_val - min_val)


def compute_weighted_metric(radius, displacement, weight_r=0.5):
    radius_norm = minmax_normalize(radius)
    displacement_norm = minmax_normalize(displacement)
    weighted = weight_r * radius_norm + (1 - weight_r) * displacement_norm
    return weighted, radius_norm, displacement_norm


def find_trajectory_pairs(retrieval_info, sd_info, target_task_name=None):
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


def plot_trajectory_comparison(
    retrieval_xyz,
    sd_xyz,
    retrieval_success,
    sd_success,
    task_description,
    task_id,
    episode_idx,
    overlap_threshold=0.02,
    save_path=None,
):
    r_x, r_y, r_z = retrieval_xyz
    s_x, s_y, s_z = sd_xyz

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(r_x, r_y, r_z, "b-", linewidth=1.5, alpha=0.5)
    ax.scatter(r_x, r_y, r_z, c="royalblue", s=60, alpha=0.7, edgecolors="none", label="Retrieval")

    ax.plot(s_x, s_y, s_z, color="darkorange", linewidth=1.5, alpha=0.5)
    ax.scatter(s_x, s_y, s_z, c="darkorange", s=60, alpha=0.7, edgecolors="none", label="SD 1:1")

    ax.scatter(r_x[0], r_y[0], r_z[0], c="royalblue", s=200, marker="o", edgecolors="black", linewidth=2.5, zorder=10)
    ax.scatter(s_x[0], s_y[0], s_z[0], c="darkorange", s=200, marker="o", edgecolors="black", linewidth=2.5, zorder=10)
    ax.scatter(r_x[-1], r_y[-1], r_z[-1], c="royalblue", s=200, marker="s", edgecolors="black", linewidth=2.5, zorder=10)
    ax.scatter(s_x[-1], s_y[-1], s_z[-1], c="darkorange", s=200, marker="^", edgecolors="black", linewidth=2.5, zorder=10)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    plt.close()


def plot_curvature_radius_comparison(retrieval_xyz, sd_xyz, window_size=5, save_path=None):
    r_radius = compute_radius_least_squares(*retrieval_xyz, window_size=window_size)
    s_radius = compute_radius_least_squares(*sd_xyz, window_size=window_size)
    r_radius_norm = minmax_normalize(r_radius)
    s_radius_norm = minmax_normalize(s_radius)

    fig, ax = plt.subplots(figsize=(14, 6))
    start_idx = 5
    steps_r = np.arange(len(r_radius_norm))
    valid_r = (~np.isnan(r_radius_norm)) & (steps_r >= start_idx)
    ax.plot(
        steps_r[valid_r],
        r_radius_norm[valid_r],
        "b-",
        linewidth=4,
        alpha=0.8,
        marker="^",
        markersize=7,
        markerfacecolor="blue",
        markeredgecolor="darkblue",
        markeredgewidth=1,
    )

    steps_s = np.arange(len(s_radius_norm))
    valid_s = (~np.isnan(s_radius_norm)) & (steps_s >= start_idx)
    ax.plot(
        steps_s[valid_s],
        s_radius_norm[valid_s],
        color="darkorange",
        linewidth=4,
        alpha=0.8,
        marker="^",
        markersize=7,
        markerfacecolor="darkorange",
        markeredgecolor="orangered",
        markeredgewidth=1,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"曲率半径图已保存: {save_path}")
    plt.close()


def plot_displacement_metric_comparison(retrieval_xyz, sd_xyz, window_size=5, save_path=None):
    r_displacement = compute_displacement_metric(*retrieval_xyz, window_size=window_size)
    s_displacement = compute_displacement_metric(*sd_xyz, window_size=window_size)
    r_displacement_norm = minmax_normalize(r_displacement)
    s_displacement_norm = minmax_normalize(s_displacement)

    fig, ax = plt.subplots(figsize=(14, 6))
    start_idx = 5
    steps_r = np.arange(len(r_displacement_norm))
    valid_r = (~np.isnan(r_displacement_norm)) & (steps_r >= start_idx)
    ax.plot(
        steps_r[valid_r],
        r_displacement_norm[valid_r],
        "b-",
        linewidth=4,
        alpha=0.8,
        marker="*",
        markersize=10,
        markerfacecolor="blue",
        markeredgecolor="darkblue",
        markeredgewidth=1,
    )

    steps_s = np.arange(len(s_displacement_norm))
    valid_s = (~np.isnan(s_displacement_norm)) & (steps_s >= start_idx)
    ax.plot(
        steps_s[valid_s],
        s_displacement_norm[valid_s],
        color="darkorange",
        linewidth=4,
        alpha=0.8,
        marker="*",
        markersize=12,
        markerfacecolor="darkorange",
        markeredgecolor="orangered",
        markeredgewidth=1,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"位移指标图已保存: {save_path}")
    plt.close()


def plot_weighted_metric_comparison(retrieval_xyz, sd_xyz, window_size=5, weight_r=0.5, save_path=None):
    r_radius = compute_radius_least_squares(*retrieval_xyz, window_size=window_size)
    s_radius = compute_radius_least_squares(*sd_xyz, window_size=window_size)
    r_displacement = compute_displacement_metric(*retrieval_xyz, window_size=window_size)
    s_displacement = compute_displacement_metric(*sd_xyz, window_size=window_size)
    r_weighted, _, _ = compute_weighted_metric(r_radius, r_displacement, weight_r)
    s_weighted, _, _ = compute_weighted_metric(s_radius, s_displacement, weight_r)

    fig, ax = plt.subplots(figsize=(14, 6))
    start_idx = 5
    steps_r = np.arange(len(r_weighted))
    valid_r = (~np.isnan(r_weighted)) & (steps_r >= start_idx)
    ax.plot(
        steps_r[valid_r],
        r_weighted[valid_r],
        "b-",
        linewidth=4,
        alpha=0.8,
        marker="s",
        markersize=7,
        markerfacecolor="blue",
        markeredgecolor="darkblue",
        markeredgewidth=1,
    )

    steps_s = np.arange(len(s_weighted))
    valid_s = (~np.isnan(s_weighted)) & (steps_s >= start_idx)
    ax.plot(
        steps_s[valid_s],
        s_weighted[valid_s],
        color="darkorange",
        linewidth=4,
        alpha=0.8,
        marker="s",
        markersize=7,
        markerfacecolor="darkorange",
        markeredgecolor="orangered",
        markeredgewidth=1,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"加权指标图已保存: {save_path}")
    plt.close()


def batch_plot_comparisons(trajectory_pairs, retrieval_data, sd_data, retrieval_info, sd_info, output_dir, category_name, category_short_name, max_plots=100, window_size=5, weight_r=0.5, overlap_threshold=0.02):
    print(f"\n{'=' * 60}")
    print(f"批量绘制: {category_name}")
    print(f"{'=' * 60}")

    if len(trajectory_pairs) == 0:
        print(f"没有找到 {category_name} 的轨迹对")
        return []

    saved_paths = []
    num_to_plot = min(len(trajectory_pairs), max_plots)

    for i, (task_id, r_eid, s_eid, task_desc) in enumerate(trajectory_pairs[:num_to_plot]):
        print(f"\n[{i + 1}/{num_to_plot}] Task {task_id}, Retrieval Episode {r_eid} vs SD Episode {s_eid}")

        r_episode = retrieval_data[task_id][r_eid]
        s_episode = sd_data[task_id][s_eid]
        r_xyz = extract_trajectory_xyz(r_episode)
        s_xyz = extract_trajectory_xyz(s_episode)
        r_success = retrieval_info[task_id][r_eid]["success"]
        s_success = sd_info[task_id][s_eid]["success"]

        task_name = get_task_name_from_description(task_desc)
        category_folder = os.path.join(output_dir, task_name, category_short_name)
        os.makedirs(category_folder, exist_ok=True)

        filename = f"{task_name}_{category_short_name}_r{r_eid}_s{s_eid}.png"
        save_path = os.path.join(category_folder, filename)
        plot_trajectory_comparison(r_xyz, s_xyz, r_success, s_success, task_desc, task_id, f"R{r_eid}_S{s_eid}", overlap_threshold=overlap_threshold, save_path=save_path)
        saved_paths.append(save_path)

        radius_path = save_path.replace(".png", "_radius_squares.png")
        plot_curvature_radius_comparison(r_xyz, s_xyz, window_size=window_size, save_path=radius_path)
        saved_paths.append(radius_path)

        displacement_path = save_path.replace(".png", "_displacement.png")
        plot_displacement_metric_comparison(r_xyz, s_xyz, window_size=window_size, save_path=displacement_path)
        saved_paths.append(displacement_path)

        weighted_path = save_path.replace(".png", f"_weighted_r{weight_r}.png")
        plot_weighted_metric_comparison(r_xyz, s_xyz, window_size=window_size, weight_r=weight_r, save_path=weighted_path)
        saved_paths.append(weighted_path)

    return saved_paths


def compute_overall_statistics(trajectory_pairs, retrieval_data, sd_data, category_name, overlap_threshold=0.02):
    if len(trajectory_pairs) == 0:
        return None

    all_avg_distances = []
    all_overlap_ratios = []
    for task_id, r_eid, s_eid, _ in trajectory_pairs:
        r_xyz = extract_trajectory_xyz(retrieval_data[task_id][r_eid])
        s_xyz = extract_trajectory_xyz(sd_data[task_id][s_eid])
        distances = compute_point_distances(r_xyz, s_xyz)
        overlap_indices = find_overlapping_points(distances, threshold=overlap_threshold)
        all_avg_distances.append(np.mean(distances))
        overlap_ratio = len(overlap_indices) / len(distances) if len(distances) > 0 else 0
        all_overlap_ratios.append(overlap_ratio)

    return {
        "category": category_name,
        "count": len(trajectory_pairs),
        "avg_distance_mean": np.mean(all_avg_distances),
        "avg_distance_std": np.std(all_avg_distances),
        "overlap_ratio_mean": np.mean(all_overlap_ratios) * 100,
        "overlap_ratio_std": np.std(all_overlap_ratios) * 100,
    }


def write_analysis(output_dir, stats_list, both_success, sd_success_retrieval_fail, retrieval_success_sd_fail, both_fail):
    analysis_file = os.path.join(output_dir, "Analysis.txt")
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("总体统计汇总\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Category':<40} {'Count':>8} {'Avg Dist (m)':>15} {'Overlap %':>12}\n")
        f.write("-" * 70 + "\n")
        for s in stats_list:
            f.write(f"{s['category']:<40} {s['count']:>8} {s['avg_distance_mean']:>10.4f}±{s['avg_distance_std']:.4f} {s['overlap_ratio_mean']:>8.1f}±{s['overlap_ratio_std']:.1f}%\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("分析结论:\n")
        f.write("=" * 70 + "\n")
        f.write("- 两种方法都成功时，轨迹重叠度高说明两种方法产生了相似的路径\n")
        f.write("- SD成功但检索失败时，可以观察SD如何纠正错误的检索结果\n")
        f.write("- 检索成功但SD失败时，说明检索效果好但模型生成能力不足\n")
        f.write("- 两种方法都失败时，可能是任务本身难度较大\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("详细统计:\n")
        f.write("=" * 70 + "\n")
        f.write(f"SD Success & Retrieval Success: {len(both_success)} 对\n")
        f.write(f"SD Success & Retrieval Fail: {len(sd_success_retrieval_fail)} 对\n")
        f.write(f"SD Fail & Retrieval Success: {len(retrieval_success_sd_fail)} 对\n")
        f.write(f"SD Fail & Retrieval Fail: {len(both_fail)} 对\n")
        f.write(f"总计: {len(both_success) + len(sd_success_retrieval_fail) + len(retrieval_success_sd_fail) + len(both_fail)} 对\n")
    print(f"\n统计结果已保存到: {analysis_file}")


def write_metric_statistics(output_dir, all_displacement_values, all_radius_values):
    stats_file = os.path.join(output_dir, "Metrics_Statistics.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("位移指标和曲率半径全局统计\n")
        f.write("=" * 60 + "\n\n")
        f.write("位移指标统计\n")
        f.write("-" * 60 + "\n")
        f.write(f"样本数: {len(all_displacement_values)}\n")
        f.write(f"最小值: {np.min(all_displacement_values):.6f} m\n")
        f.write(f"最大值: {np.max(all_displacement_values):.6f} m\n")
        f.write(f"均值: {np.mean(all_displacement_values):.6f} m\n")
        f.write(f"中位数: {np.median(all_displacement_values):.6f} m\n")
        f.write(f"标准差: {np.std(all_displacement_values):.6f} m\n")
        f.write(f"25%分位数: {np.percentile(all_displacement_values, 25):.6f} m\n")
        f.write(f"75%分位数: {np.percentile(all_displacement_values, 75):.6f} m\n")
        f.write(f"95%分位数: {np.percentile(all_displacement_values, 95):.6f} m\n")
        f.write(f"99%分位数: {np.percentile(all_displacement_values, 99):.6f} m\n")
        f.write("\n曲率半径统计\n")
        f.write("-" * 60 + "\n")
        f.write(f"样本数: {len(all_radius_values)}\n")
        f.write(f"最小值: {np.min(all_radius_values):.6f} m\n")
        f.write(f"最大值: {np.max(all_radius_values):.6f} m\n")
        f.write(f"均值: {np.mean(all_radius_values):.6f} m\n")
        f.write(f"中位数: {np.median(all_radius_values):.6f} m\n")
        f.write(f"标准差: {np.std(all_radius_values):.6f} m\n")
        f.write(f"25%分位数: {np.percentile(all_radius_values, 25):.6f} m\n")
        f.write(f"75%分位数: {np.percentile(all_radius_values, 75):.6f} m\n")
        f.write(f"95%分位数: {np.percentile(all_radius_values, 95):.6f} m\n")
        f.write(f"99%分位数: {np.percentile(all_radius_values, 99):.6f} m\n")
    print(f"统计结果已保存到: {stats_file}")


def write_metric_plots(output_dir, all_displacement_values, all_radius_values):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.boxplot([
        all_displacement_values
    ], vert=True, patch_artist=True, labels=["Displacement Metric"], boxprops=dict(facecolor="lightblue", alpha=0.7), medianprops=dict(color="red", linewidth=2), whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
    ax1.set_ylabel("Displacement Metric (m)", fontsize=14, fontweight="bold")
    ax1.set_title("Displacement Metric - Box Plot", fontsize=16, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.boxplot([
        all_radius_values
    ], vert=True, patch_artist=True, labels=["Radius of Curvature"], boxprops=dict(facecolor="lightcoral", alpha=0.7), medianprops=dict(color="darkblue", linewidth=2), whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
    ax2.set_ylabel("Radius of Curvature (m)", fontsize=14, fontweight="bold")
    ax2.set_title("Radius of Curvature - Box Plot", fontsize=16, fontweight="bold", pad=15)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "Metrics_BoxPlot.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.hist(all_displacement_values, bins=50, color="skyblue", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(all_displacement_values), color="red", linestyle="--", linewidth=2)
    ax.axvline(np.median(all_displacement_values), color="green", linestyle="--", linewidth=2)
    ax.set_title("Displacement Metric - Histogram (Full Range)")

    ax = axes[0, 1]
    disp_95 = np.percentile(all_displacement_values, 95)
    disp_filtered = all_displacement_values[all_displacement_values <= disp_95]
    ax.hist(disp_filtered, bins=50, color="lightblue", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(disp_filtered), color="red", linestyle="--", linewidth=2)
    ax.axvline(np.median(disp_filtered), color="green", linestyle="--", linewidth=2)
    ax.set_title("Displacement Metric - Histogram (≤95% Percentile)")

    ax = axes[1, 0]
    ax.hist(all_radius_values, bins=50, color="lightcoral", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(all_radius_values), color="darkblue", linestyle="--", linewidth=2)
    ax.axvline(np.median(all_radius_values), color="purple", linestyle="--", linewidth=2)
    ax.set_title("Radius of Curvature - Histogram (Full Range)")

    ax = axes[1, 1]
    radius_95 = np.percentile(all_radius_values, 95)
    radius_filtered = all_radius_values[all_radius_values <= radius_95]
    ax.hist(radius_filtered, bins=50, color="mistyrose", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(radius_filtered), color="darkblue", linestyle="--", linewidth=2)
    ax.axvline(np.median(radius_filtered), color="purple", linestyle="--", linewidth=2)
    ax.set_title("Radius of Curvature - Histogram (≤95% Percentile)")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    histogram_path = os.path.join(output_dir, "Metrics_Histogram.png")
    plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"箱线图已保存: {boxplot_path}")
    print(f"直方图已保存: {histogram_path}")


def collect_metric_values(all_pairs, retrieval_data, sd_data, window_size):
    all_displacement_values = []
    all_radius_values = []
    for task_id, r_eid, s_eid, _ in all_pairs:
        r_xyz = extract_trajectory_xyz(retrieval_data[task_id][r_eid])
        s_xyz = extract_trajectory_xyz(sd_data[task_id][s_eid])
        r_displacement = compute_displacement_metric(*r_xyz, window_size=window_size)
        s_displacement = compute_displacement_metric(*s_xyz, window_size=window_size)
        r_radius = compute_radius_least_squares(*r_xyz, window_size=window_size)
        s_radius = compute_radius_least_squares(*s_xyz, window_size=window_size)
        all_displacement_values.extend(r_displacement[~np.isnan(r_displacement)])
        all_displacement_values.extend(s_displacement[~np.isnan(s_displacement)])
        all_radius_values.extend(r_radius[~np.isnan(r_radius)])
        all_radius_values.extend(s_radius[~np.isnan(s_radius)])
    return np.array(all_displacement_values), np.array(all_radius_values)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone trajectory comparison script extracted from notebook.")
    parser.add_argument("--retrieval-npy-path", required=True, help="Retrieval observations.npy path")
    parser.add_argument("--sd-npy-path", required=True, help="SD observations.npy path")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "trajectory_visualizations"), help="Output directory")
    parser.add_argument("--max-plots", type=int, default=100, help="Maximum pairs per category")
    parser.add_argument("--window-size", type=int, default=5, help="Sliding window size")
    parser.add_argument("--weight-r", type=float, default=0.5, help="Radius weight for weighted metric")
    parser.add_argument("--overlap-threshold", type=float, default=0.02, help="Overlap threshold in meters")
    parser.add_argument("--target-task-name", default=None, help="Optional substring filter for task description")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("加载数据中...")
    retrieval_data = load_observations(args.retrieval_npy_path)
    sd_data = load_observations(args.sd_npy_path)
    retrieval_info = get_trajectory_info(retrieval_data)
    sd_info = get_trajectory_info(sd_data)

    both_success, sd_success_retrieval_fail, retrieval_success_sd_fail, both_fail = find_trajectory_pairs(
        retrieval_info, sd_info, target_task_name=args.target_task_name
    )

    batch_plot_comparisons(both_success, retrieval_data, sd_data, retrieval_info, sd_info, output_dir, "SD Success & Retrieval Success", "both_success", max_plots=args.max_plots, window_size=args.window_size, weight_r=args.weight_r, overlap_threshold=args.overlap_threshold)
    batch_plot_comparisons(sd_success_retrieval_fail, retrieval_data, sd_data, retrieval_info, sd_info, output_dir, "SD Success & Retrieval Fail", "sd_success_retrieval_fail", max_plots=args.max_plots, window_size=args.window_size, weight_r=args.weight_r, overlap_threshold=args.overlap_threshold)
    batch_plot_comparisons(retrieval_success_sd_fail, retrieval_data, sd_data, retrieval_info, sd_info, output_dir, "SD Fail & Retrieval Success", "sd_fail_retrieval_success", max_plots=args.max_plots, window_size=args.window_size, weight_r=args.weight_r, overlap_threshold=args.overlap_threshold)
    batch_plot_comparisons(both_fail, retrieval_data, sd_data, retrieval_info, sd_info, output_dir, "SD Fail & Retrieval Fail", "both_fail", max_plots=args.max_plots, window_size=args.window_size, weight_r=args.weight_r, overlap_threshold=args.overlap_threshold)

    stats_list = []
    for pairs, category_name in [
        (both_success, "SD Success & Retrieval Success"),
        (sd_success_retrieval_fail, "SD Success & Retrieval Fail"),
        (retrieval_success_sd_fail, "SD Fail & Retrieval Success"),
        (both_fail, "SD Fail & Retrieval Fail"),
    ]:
        stats = compute_overall_statistics(pairs, retrieval_data, sd_data, category_name, overlap_threshold=args.overlap_threshold)
        if stats is not None:
            stats_list.append(stats)
    write_analysis(output_dir, stats_list, both_success, sd_success_retrieval_fail, retrieval_success_sd_fail, both_fail)

    all_pairs = both_success + sd_success_retrieval_fail + retrieval_success_sd_fail + both_fail
    all_displacement_values, all_radius_values = collect_metric_values(all_pairs, retrieval_data, sd_data, window_size=args.window_size)
    if len(all_displacement_values) and len(all_radius_values):
        write_metric_statistics(output_dir, all_displacement_values, all_radius_values)
        write_metric_plots(output_dir, all_displacement_values, all_radius_values)

    print("\n完成")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()