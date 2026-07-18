"""
edit_zhexian_local.py

本地编辑折线图数据的脚本
下载到本地后运行，可以修改数据并重新生成折线图

使用方法：
1. 将此脚本和 .npy 文件下载到本地同一目录
2. 修改下方 DATA_FILE 为你的 .npy 文件名
3. 在 modify_data() 函数中修改数据
4. 运行脚本: python edit_zhexian_local.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ================================
# 配置 - 修改这里
# ================================
DATA_FILE = "push_the_plate_to_the_front_of_the_stove_both_success_r3_s1_zhexian_data.npy"
OUTPUT_DIR = "./"  # 输出目录，默认当前目录


# ================================
# 加载数据
# ================================
def load_data(npy_path):
    """加载 .npy 数据文件"""
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def print_data_info(data):
    """打印数据信息"""
    print("=" * 60)
    print("数据信息")
    print("=" * 60)
    print(f"任务: {data['task_desc']}")
    print(f"轨迹对: r{data['r_eid']}_s{data['s_eid']}")
    print(f"窗口大小: {data['window_size']}")
    print()
    print("数据字段:")
    for key in data.keys():
        val = data[key]
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        elif isinstance(val, tuple):
            print(f"  {key}: tuple of {len(val)} arrays")
        else:
            print(f"  {key}: {val}")
    print()
    
    # 打印绘图用数据的长度
    print("绘图数据长度:")
    print(f"  Retrieval (r_): {len(data['r_radius_norm'])} 点")
    print(f"  SD (s_): {len(data['s_radius_norm'])} 点")
    print("=" * 60)


# ================================
# 修改数据 - 在这里编辑你的数据！
# ================================
def modify_data(data):
    """
    在这里修改数据！
    
    可修改的字段（用于绘图）：
    - r_radius_norm: Retrieval 的归一化曲率半径
    - s_radius_norm: SD 的归一化曲率半径
    - r_displacement_norm: Retrieval 的归一化位移指标
    - s_displacement_norm: SD 的归一化位移指标
    - r_weighted: Retrieval 的加权指标
    - s_weighted: SD 的加权指标
    
    数据都是 numpy 数组，可以用索引修改：
    - data['r_radius_norm'][10] = 0.5  # 修改第10个点
    - data['r_radius_norm'][10:20] = 0.5  # 修改第10-19个点
    - data['r_radius_norm'] = data['r_radius_norm'] * 1.2  # 整体缩放
    """
    
    # ========== 示例：修改数据 ==========
    
    # 示例1：打印前20个点的值
    print("\n修改前 - Retrieval 曲率半径 (前20个点):")
    print(data['r_radius_norm'][:20])
    
    print("\n修改前 - SD 曲率半径 (前20个点):")
    print(data['s_radius_norm'][:20])
    
    # ----- 在下面添加你的修改 -----
    
    # 示例2：修改某个点
    # data['r_radius_norm'][15] = 0.8
    
    # 示例3：修改一段区间
    # data['s_radius_norm'][20:30] = np.linspace(0.3, 0.7, 10)
    
    # 示例4：整体乘以系数
    # data['r_displacement_norm'] = data['r_displacement_norm'] * 1.1
    
    # 示例5：添加偏移
    # data['s_weighted'] = data['s_weighted'] + 0.05
    
    # 示例6：平滑处理（简单移动平均）
    # def smooth(arr, window=3):
    #     return np.convolve(arr, np.ones(window)/window, mode='same')
    # data['r_radius_norm'] = smooth(data['r_radius_norm'])
    
    # ----- 修改结束 -----
    
    print("\n修改后 - Retrieval 曲率半径 (前20个点):")
    print(data['r_radius_norm'][:20])
    
    return data


# ================================
# 绘图函数 - 与服务器版本一致
# ================================
def plot_curvature_radius_comparison(r_radius_norm, s_radius_norm, save_path=None):
    """绘制曲率半径对比折线图（归一化）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    start_idx = 5
    
    steps_r = np.arange(len(r_radius_norm))
    valid_r = ~np.isnan(r_radius_norm)
    valid_r = valid_r & (steps_r >= start_idx)
    ax.plot(steps_r[valid_r], r_radius_norm[valid_r], 
            'b-', linewidth=4, alpha=0.8, 
            marker='^', markersize=7, 
            markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=1,
            label='Retrieval')
    
    steps_s = np.arange(len(s_radius_norm))
    valid_s = ~np.isnan(s_radius_norm)
    valid_s = valid_s & (steps_s >= start_idx)
    ax.plot(steps_s[valid_s], s_radius_norm[valid_s], 
            color='darkorange', linewidth=4, alpha=0.8, 
            marker='^', markersize=7, 
            markerfacecolor='darkorange', markeredgecolor='orangered', markeredgewidth=1,
            label='SD')
    
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


def plot_displacement_metric_comparison(r_displacement_norm, s_displacement_norm, save_path=None):
    """绘制位移指标对比折线图（归一化）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    start_idx = 5
    
    steps_r = np.arange(len(r_displacement_norm))
    valid_r = ~np.isnan(r_displacement_norm)
    valid_r = valid_r & (steps_r >= start_idx)
    ax.plot(steps_r[valid_r], r_displacement_norm[valid_r], 
            'b-', linewidth=4, alpha=0.8, 
            marker='*', markersize=10, 
            markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=1,
            label='Retrieval')
    
    steps_s = np.arange(len(s_displacement_norm))
    valid_s = ~np.isnan(s_displacement_norm)
    valid_s = valid_s & (steps_s >= start_idx)
    ax.plot(steps_s[valid_s], s_displacement_norm[valid_s], 
            color='darkorange', linewidth=4, alpha=0.8, 
            marker='*', markersize=12, 
            markerfacecolor='darkorange', markeredgecolor='orangered', markeredgewidth=1,
            label='SD')
    
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


def plot_weighted_metric_comparison(r_weighted, s_weighted, save_path=None):
    """绘制加权指标对比折线图（归一化）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    start_idx = 5
    
    steps_r = np.arange(len(r_weighted))
    valid_r = ~np.isnan(r_weighted)
    valid_r = valid_r & (steps_r >= start_idx)
    ax.plot(steps_r[valid_r], r_weighted[valid_r], 
            'b-', linewidth=4, alpha=0.8, 
            marker='s', markersize=7, 
            markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=1,
            label='Retrieval')
    
    steps_s = np.arange(len(s_weighted))
    valid_s = ~np.isnan(s_weighted)
    valid_s = valid_s & (steps_s >= start_idx)
    ax.plot(steps_s[valid_s], s_weighted[valid_s], 
            color='darkorange', linewidth=4, alpha=0.8, 
            marker='s', markersize=7, 
            markerfacecolor='darkorange', markeredgecolor='orangered', markeredgewidth=1,
            label='SD')
    
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


def generate_all_plots(data, output_dir="./", prefix="modified"):
    """生成所有折线图"""
    import os
    
    # 曲率半径图
    radius_path = os.path.join(output_dir, f"{prefix}_radius.png")
    plot_curvature_radius_comparison(
        data['r_radius_norm'], 
        data['s_radius_norm'], 
        save_path=radius_path
    )
    
    # 位移指标图
    displacement_path = os.path.join(output_dir, f"{prefix}_displacement.png")
    plot_displacement_metric_comparison(
        data['r_displacement_norm'], 
        data['s_displacement_norm'], 
        save_path=displacement_path
    )
    
    # 加权指标图
    weighted_path = os.path.join(output_dir, f"{prefix}_weighted.png")
    plot_weighted_metric_comparison(
        data['r_weighted'], 
        data['s_weighted'], 
        save_path=weighted_path
    )
    
    print(f"\n所有图片已保存到: {output_dir}")


# ================================
# 交互式预览（可选）
# ================================
def preview_data(data):
    """交互式预览数据（需要 GUI 支持）"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    start_idx = 5
    
    # 曲率半径
    ax = axes[0]
    steps_r = np.arange(len(data['r_radius_norm']))
    valid_r = ~np.isnan(data['r_radius_norm']) & (steps_r >= start_idx)
    steps_s = np.arange(len(data['s_radius_norm']))
    valid_s = ~np.isnan(data['s_radius_norm']) & (steps_s >= start_idx)
    
    ax.plot(steps_r[valid_r], data['r_radius_norm'][valid_r], 'b-', linewidth=2, marker='^', markersize=5, label='Retrieval')
    ax.plot(steps_s[valid_s], data['s_radius_norm'][valid_s], color='darkorange', linewidth=2, marker='^', markersize=5, label='SD')
    ax.set_title('Curvature Radius (Normalized)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 位移指标
    ax = axes[1]
    valid_r = ~np.isnan(data['r_displacement_norm']) & (steps_r >= start_idx)
    valid_s = ~np.isnan(data['s_displacement_norm']) & (steps_s >= start_idx)
    
    ax.plot(steps_r[valid_r], data['r_displacement_norm'][valid_r], 'b-', linewidth=2, marker='*', markersize=6, label='Retrieval')
    ax.plot(steps_s[valid_s], data['s_displacement_norm'][valid_s], color='darkorange', linewidth=2, marker='*', markersize=6, label='SD')
    ax.set_title('Displacement Metric (Normalized)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 加权指标
    ax = axes[2]
    valid_r = ~np.isnan(data['r_weighted']) & (steps_r >= start_idx)
    valid_s = ~np.isnan(data['s_weighted']) & (steps_s >= start_idx)
    
    ax.plot(steps_r[valid_r], data['r_weighted'][valid_r], 'b-', linewidth=2, marker='s', markersize=5, label='Retrieval')
    ax.plot(steps_s[valid_s], data['s_weighted'][valid_s], color='darkorange', linewidth=2, marker='s', markersize=5, label='SD')
    ax.set_title('Weighted Metric (Normalized)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("preview.png", dpi=150)
    print("预览图已保存: preview.png")
    
    # 如果有 GUI，显示图片
    try:
        plt.show()
    except:
        pass


# ================================
# 主函数
# ================================
def main():
    print("=" * 60)
    print("折线图数据编辑器")
    print("=" * 60)
    
    # 1. 加载数据
    print(f"\n加载数据: {DATA_FILE}")
    data = load_data(DATA_FILE)
    print_data_info(data)
    
    # 2. 修改数据（在 modify_data 函数中编辑）
    print("\n" + "=" * 60)
    print("修改数据")
    print("=" * 60)
    data = modify_data(data)
    
    # 3. 预览（可选）
    print("\n生成预览图...")
    preview_data(data)
    
    # 4. 生成最终图片
    print("\n" + "=" * 60)
    print("生成最终图片")
    print("=" * 60)
    generate_all_plots(data, output_dir=OUTPUT_DIR, prefix="modified")
    
    # 5. 保存修改后的数据（可选）
    save_modified = input("\n是否保存修改后的数据到新文件？(y/n): ").strip().lower()
    if save_modified == 'y':
        new_path = "modified_data.npy"
        np.save(new_path, data)
        print(f"数据已保存: {new_path}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
