import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ================================
# 配置：四组数据路径
# ================================
DATASETS = [
    {
        "name": "libero_goal",
        "retrieval_npy_path": "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_goal_naive_DB_mix/EVAL-libero_goal-NaiveDB-Mix-2026_01_21-14_38_17_observations.npy",
        "sd_npy_path": "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_goal_Retrieval_Verify/EVAL-libero_goal-openvla-2026_01_17-12_50_04--record_guiji_suc_fail_pure_sd_observations.npy",
    },
    {
        "name": "libero_spatial",
        "retrieval_npy_path": "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_spatial_naive_DB_mix/EVAL-libero_spatial-NaiveDB-Mix-2026_01_21-18_41_51_observations.npy",
        "sd_npy_path": "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_spatial_naive_DB_mix/EVAL-libero_spatial-NaiveDB-Mix-2026_01_21-18_41_51_observations.npy",
    },
    {
        "name": "libero_object",
        "retrieval_npy_path": "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_object_naive_DB_mix/EVAL-libero_object-NaiveDB-Mix-2026_01_21-19_07_14_observations.npy",
        "sd_npy_path": "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_object_naive_DB_mix/EVAL-libero_object-NaiveDB-Mix-2026_01_21-19_07_14_observations.npy",
    },
    {
        "name": "libero_10",
        "retrieval_npy_path": "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_10_naive_DB_mix/EVAL-libero_10-NaiveDB-Mix-2026_01_21-19_35_44_observations.npy",
        "sd_npy_path": "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_10_naive_DB_mix/EVAL-libero_10-NaiveDB-Mix-2026_01_21-19_35_44_observations.npy",
    },
]

OUTPUT_DIR = "/path/to/SpecVLA/vis_distri"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# 数据与指标函数
# ================================
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


def extract_trajectory_xyz(episode_data):
    observations = episode_data["observations"]
    states = np.array([obs["state"] for obs in observations])
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    return x, y, z


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
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return values

    min_val = np.nanmin(values)
    max_val = np.nanmax(values)

    if max_val - min_val < 1e-10:
        return np.where(~np.isnan(values), 0.5, np.nan)

    return (values - min_val) / (max_val - min_val)


# ================================
# 绘图函数
# ================================
def plot_hist(values, title, xlabel, output_path, bins=50):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(values, bins=bins, color="#5DA5DA", alpha=0.8, edgecolor="black")

    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel("Statistics", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ================================
# 主流程
# ================================
def collect_metrics_for_env(retrieval_data, sd_data, window_size=5):
    retrieval_info = get_trajectory_info(retrieval_data)
    sd_info = get_trajectory_info(sd_data)

    both_success, sd_success_retrieval_fail, retrieval_success_sd_fail, both_fail = find_trajectory_pairs(
        retrieval_info, sd_info, target_task_name=None
    )
    all_pairs = both_success + sd_success_retrieval_fail + retrieval_success_sd_fail + both_fail

    all_displacement_values = []
    all_radius_values = []

    for task_id, r_eid, s_eid, _ in all_pairs:
        r_episode = retrieval_data[task_id][r_eid]
        s_episode = sd_data[task_id][s_eid]

        r_xyz = extract_trajectory_xyz(r_episode)
        s_xyz = extract_trajectory_xyz(s_episode)

        r_displacement = compute_displacement_metric(r_xyz[0], r_xyz[1], r_xyz[2], window_size=window_size)
        s_displacement = compute_displacement_metric(s_xyz[0], s_xyz[1], s_xyz[2], window_size=window_size)

        r_radius = compute_radius_least_squares(r_xyz[0], r_xyz[1], r_xyz[2], window_size=window_size)
        s_radius = compute_radius_least_squares(s_xyz[0], s_xyz[1], s_xyz[2], window_size=window_size)

        all_displacement_values.extend(r_displacement[~np.isnan(r_displacement)])
        all_displacement_values.extend(s_displacement[~np.isnan(s_displacement)])

        all_radius_values.extend(r_radius[~np.isnan(r_radius)])
        all_radius_values.extend(s_radius[~np.isnan(s_radius)])

    all_displacement_values = np.array(all_displacement_values)
    all_radius_values = np.array(all_radius_values)

    # Fused Metric (1:1)
    disp_norm = minmax_normalize(all_displacement_values)
    radius_norm = minmax_normalize(all_radius_values)
    fused = 0.5 * disp_norm + 0.5 * radius_norm

    return all_displacement_values, all_radius_values, fused


def filter_p95(values):
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return values
    p95 = np.percentile(values, 95)
    return values[values <= p95]


def main():
    for ds in DATASETS:
        name = ds["name"]
        print(f"处理环境: {name}")

        retrieval_data = load_observations(ds["retrieval_npy_path"])
        sd_data = load_observations(ds["sd_npy_path"])

        d_vals, r_vals, f_vals = collect_metrics_for_env(retrieval_data, sd_data, window_size=5)

        d_vals_p95 = filter_p95(d_vals)
        r_vals_p95 = filter_p95(r_vals)
        f_vals_p95 = filter_p95(f_vals)

        # D: Cumulative Spatial Displacement
        d_title = f"{name} | D (≤95%)"
        d_xlabel = "D: Cumulative Spatial Displacement (m)"
        d_out = os.path.join(OUTPUT_DIR, f"{name}_D_p95.png")
        plot_hist(d_vals_p95, d_title, d_xlabel, d_out)

        # R: Radius of Curvature
        r_title = f"{name} | R (≤95%)"
        r_xlabel = "R: Radius of Curvature (m)"
        r_out = os.path.join(OUTPUT_DIR, f"{name}_R_p95.png")
        plot_hist(r_vals_p95, r_title, r_xlabel, r_out)

        # F: Fused Metric
        f_title = f"{name} | F (≤95%)"
        f_xlabel = "F: Fused Metric (a.u.)"
        f_out = os.path.join(OUTPUT_DIR, f"{name}_F_p95.png")
        plot_hist(f_vals_p95, f_title, f_xlabel, f_out)

        print(f"已保存: {d_out}")
        print(f"已保存: {r_out}")
        print(f"已保存: {f_out}")


if __name__ == "__main__":
    main()
