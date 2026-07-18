"""
plot_accept_len.py

读取fenxi_accept_len目录下的.npy文件，绘制散点图
横轴：步数(step)
纵轴：接受长度(accept_length) 或 执行度(execution_degree)

每个任务生成一张图，保存到figs目录下
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def plot_execution_degree(npy_path, output_dir):
    """
    绘制单个任务的execution_degree散点图
    
    Args:
        npy_path: .npy文件路径 (以_exec_degree.npy结尾)
        output_dir: 输出目录
    """
    # 加载数据
    exec_degrees = np.load(npy_path)
    
    # 获取任务名（从文件名）
    task_name = os.path.basename(npy_path).replace('_exec_degree.npy', '')
    
    # 跳过全局统计文件
    if task_name.startswith('all_tasks'):
        print(f"跳过全局统计文件: {task_name}")
        return
    
    # 创建步数数组
    steps = np.arange(len(exec_degrees))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制散点图
    scatter = ax.scatter(steps, exec_degrees, 
                        c=exec_degrees, 
                        cmap='plasma', 
                        alpha=0.7, 
                        s=30,
                        edgecolors='none')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Execution Degree', fontsize=12)
    
    # 添加水平参考线
    for y in range(8):
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 计算统计量
    mean_val = np.mean(exec_degrees)
    median_val = np.median(exec_degrees)
    std_val = np.std(exec_degrees)
    max_val = np.max(exec_degrees)
    min_val = np.min(exec_degrees)
    
    # 添加均值线
    ax.axhline(y=mean_val, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axhline(y=median_val, color='blue', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.2f}')
    
    # 设置标签和标题
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Execution Degree (# matching dims)', fontsize=14)
    
    # 格式化任务名用于标题
    title_name = task_name.replace('_', ' ').title()
    ax.set_title(f'Execution Degree vs Step\n{title_name}', fontsize=14)
    
    # 设置y轴范围
    ax.set_ylim(-0.5, 7.5)
    ax.set_yticks(range(8))
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息文本框
    stats_text = f'Total Steps: {len(exec_degrees)}\n'
    stats_text += f'Mean: {mean_val:.2f}\n'
    stats_text += f'Median: {median_val:.2f}\n'
    stats_text += f'Std: {std_val:.2f}\n'
    stats_text += f'Min: {min_val}, Max: {max_val}'
    
    # 计算分布
    unique, counts = np.unique(exec_degrees, return_counts=True)
    dist_text = '\nDistribution:\n'
    for u, c in zip(unique, counts):
        pct = 100.0 * c / len(exec_degrees)
        dist_text += f'  {int(u)}: {c} ({pct:.1f}%)\n'
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.98, stats_text + dist_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{task_name}_exec_degree.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {output_path}")
    print(f"  - 总步数: {len(exec_degrees)}, 均值: {mean_val:.2f}, 中位数: {median_val:.2f}")
    
    return {
        'task_name': task_name,
        'total_steps': len(exec_degrees),
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    }

def plot_accept_length(npy_path, output_dir):
    """
    绘制单个任务的accept_length散点图
    
    Args:
        npy_path: .npy文件路径
        output_dir: 输出目录
    """
    # 获取任务名（从文件名）
    basename = os.path.basename(npy_path)
    
    # 跳过特殊文件
    if basename.startswith('all_tasks'):
        print(f"跳过全局统计文件: {basename}")
        return None
    if basename.endswith('_exec_degree.npy') or basename.endswith('_similarity.npy') or basename.endswith('_direction.npy'):
        return None
    
    task_name = basename.replace('.npy', '')
    
    # 加载数据
    accept_lengths = np.load(npy_path)
    
    # 创建步数数组
    steps = np.arange(len(accept_lengths))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制散点图
    scatter = ax.scatter(steps, accept_lengths, 
                        c=accept_lengths, 
                        cmap='viridis', 
                        alpha=0.7, 
                        s=30,
                        edgecolors='none')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Accept Length', fontsize=12)
    
    # 添加水平参考线
    for y in range(8):
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 计算统计量
    mean_val = np.mean(accept_lengths)
    median_val = np.median(accept_lengths)
    std_val = np.std(accept_lengths)
    max_val = np.max(accept_lengths)
    min_val = np.min(accept_lengths)
    
    # 添加均值线
    ax.axhline(y=mean_val, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axhline(y=median_val, color='blue', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.2f}')
    
    # 设置标签和标题
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Accept Length', fontsize=14)
    
    # 格式化任务名用于标题（将下划线替换为空格）
    title_name = task_name.replace('_', ' ').title()
    ax.set_title(f'Accept Length vs Step\n{title_name}', fontsize=14)
    
    # 设置y轴范围
    ax.set_ylim(-0.5, 7.5)
    ax.set_yticks(range(8))
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息文本框
    stats_text = f'Total Steps: {len(accept_lengths)}\n'
    stats_text += f'Mean: {mean_val:.2f}\n'
    stats_text += f'Median: {median_val:.2f}\n'
    stats_text += f'Std: {std_val:.2f}\n'
    stats_text += f'Min: {min_val}, Max: {max_val}'
    
    # 计算分布（过滤NaN值）
    valid_accept = accept_lengths[~np.isnan(accept_lengths)]
    if len(valid_accept) > 0:
        unique, counts = np.unique(valid_accept, return_counts=True)
        dist_text = '\nDistribution:\n'
        for u, c in zip(unique, counts):
            pct = 100.0 * c / len(valid_accept)
            dist_text += f'  {int(u)}: {c} ({pct:.1f}%)\n'
    else:
        dist_text = '\nDistribution: N/A\n'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text + dist_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{task_name}_accept_len.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {output_path}")
    print(f"  - 总步数: {len(accept_lengths)}, 均值: {mean_val:.2f}, 中位数: {median_val:.2f}")
    
    return {
        'task_name': task_name,
        'total_steps': len(accept_lengths),
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    }


def plot_all_tasks_combined(npy_dir, output_dir):
    """
    将所有任务的accept_length绘制在一张图上（子图形式）
    """
    # 获取所有accept_length文件（排除全局统计文件和其他类型文件）
    npy_files = glob.glob(os.path.join(npy_dir, '*.npy'))
    npy_files = [f for f in npy_files 
                 if not os.path.basename(f).startswith('all_tasks')
                 and not f.endswith('_exec_degree.npy')
                 and not f.endswith('_similarity.npy')
                 and not f.endswith('_direction.npy')]
    npy_files = sorted(npy_files)
    
    if len(npy_files) == 0:
        print("没有找到任务.npy文件")
        return
    
    # 计算子图布局
    n_tasks = len(npy_files)
    n_cols = min(2, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    if n_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    all_stats = []
    
    for idx, npy_path in enumerate(npy_files):
        accept_lengths = np.load(npy_path)
        task_name = os.path.basename(npy_path).replace('.npy', '')
        steps = np.arange(len(accept_lengths))
        
        ax = axes[idx]
        
        # 绘制散点图
        scatter = ax.scatter(steps, accept_lengths, 
                            c=accept_lengths, 
                            cmap='viridis', 
                            alpha=0.6, 
                            s=15,
                            edgecolors='none')
        
        # 计算统计量
        mean_val = np.mean(accept_lengths)
        median_val = np.median(accept_lengths)
        
        # 添加均值线
        ax.axhline(y=mean_val, color='red', linestyle='-', alpha=0.8, linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        
        # 设置标签和标题
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Accept Length', fontsize=10)
        
        # 格式化任务名
        short_name = task_name.replace('_', ' ')
        if len(short_name) > 40:
            short_name = short_name[:40] + '...'
        ax.set_title(short_name, fontsize=10)
        
        ax.set_ylim(-0.5, 7.5)
        ax.set_yticks(range(8))
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        all_stats.append({
            'task_name': task_name,
            'mean': mean_val,
            'median': median_val,
            'total_steps': len(accept_lengths)
        })
    
    # 隐藏多余的子图
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Accept Length Analysis - All Tasks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'all_tasks_combined.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n已保存汇总图: {output_path}")
    
    return all_stats


def plot_exec_degree_distribution(npy_dir, output_dir):
    """
    绘制所有任务的execution_degree分布直方图
    """
    # 获取全局统计文件
    global_files = glob.glob(os.path.join(npy_dir, 'all_tasks_*_exec_degree.npy'))
    
    if len(global_files) == 0:
        # 如果没有全局文件，合并所有任务的数据
        npy_files = glob.glob(os.path.join(npy_dir, '*_exec_degree.npy'))
        npy_files = [f for f in npy_files if not os.path.basename(f).startswith('all_tasks')]
        
        if len(npy_files) == 0:
            print("没有找到execution_degree .npy文件")
            return
        
        all_exec_degrees = []
        for npy_path in npy_files:
            all_exec_degrees.extend(np.load(npy_path).tolist())
        all_exec_degrees = np.array(all_exec_degrees)
    else:
        all_exec_degrees = np.load(global_files[0])
    
    if len(all_exec_degrees) == 0:
        print("execution_degree数据为空")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图
    counts, bins, patches = ax1.hist(all_exec_degrees, bins=np.arange(-0.5, 8.5, 1), 
                                      edgecolor='black', alpha=0.7, color='coral')
    
    # 添加数值标签
    for i, (count, patch) in enumerate(zip(counts, patches)):
        ax1.text(patch.get_x() + patch.get_width()/2, count + max(counts)*0.02, 
                f'{int(count)}\n({100*count/len(all_exec_degrees):.1f}%)', 
                ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Execution Degree', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Execution Degree Distribution (Histogram)', fontsize=12)
    ax1.set_xticks(range(8))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：累积分布
    unique_vals = np.arange(8)
    cum_pcts = []
    for v in unique_vals:
        pct = np.sum(all_exec_degrees <= v) / len(all_exec_degrees) * 100
        cum_pcts.append(pct)
    
    ax2.bar(unique_vals, cum_pcts, edgecolor='black', alpha=0.7, color='mediumpurple')
    
    for i, pct in enumerate(cum_pcts):
        ax2.text(i, pct + 2, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Execution Degree', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('Cumulative Distribution', fontsize=12)
    ax2.set_xticks(range(8))
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    mean_val = np.mean(all_exec_degrees)
    median_val = np.median(all_exec_degrees)
    std_val = np.std(all_exec_degrees)
    
    stats_text = f'Total: {len(all_exec_degrees)}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Execution Degree Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    output_path = os.path.join(output_dir, 'exec_degree_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存分布图: {output_path}")


def plot_distribution_histogram(npy_dir, output_dir):
    """
    绘制所有任务的accept_length分布直方图
    """
    # 获取全局统计文件（排除_exec_degree, _similarity, _direction后缀）
    global_files = glob.glob(os.path.join(npy_dir, 'all_tasks_*.npy'))
    global_files = [f for f in global_files 
                    if not f.endswith('_exec_degree.npy')
                    and not f.endswith('_similarity.npy')
                    and not f.endswith('_direction.npy')]
    
    if len(global_files) == 0:
        # 如果没有全局文件，合并所有任务的数据
        npy_files = glob.glob(os.path.join(npy_dir, '*.npy'))
        npy_files = [f for f in npy_files 
                     if not os.path.basename(f).startswith('all_tasks')
                     and not f.endswith('_exec_degree.npy')
                     and not f.endswith('_similarity.npy')
                     and not f.endswith('_direction.npy')]
        
        if len(npy_files) == 0:
            print("没有找到accept_length .npy文件")
            return
        
        all_accept_lengths = []
        for npy_path in npy_files:
            all_accept_lengths.extend(np.load(npy_path).tolist())
        all_accept_lengths = np.array(all_accept_lengths)
    else:
        all_accept_lengths = np.load(global_files[0])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图
    counts, bins, patches = ax1.hist(all_accept_lengths, bins=np.arange(-0.5, 8.5, 1), 
                                      edgecolor='black', alpha=0.7, color='steelblue')
    
    # 添加数值标签
    for i, (count, patch) in enumerate(zip(counts, patches)):
        ax1.text(patch.get_x() + patch.get_width()/2, count + max(counts)*0.02, 
                f'{int(count)}\n({100*count/len(all_accept_lengths):.1f}%)', 
                ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Accept Length', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Accept Length Distribution (Histogram)', fontsize=12)
    ax1.set_xticks(range(8))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：累积分布
    sorted_data = np.sort(all_accept_lengths)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # 计算每个accept_length的累积百分比
    unique_vals = np.arange(8)
    cum_pcts = []
    for v in unique_vals:
        pct = np.sum(all_accept_lengths <= v) / len(all_accept_lengths) * 100
        cum_pcts.append(pct)
    
    ax2.bar(unique_vals, cum_pcts, edgecolor='black', alpha=0.7, color='coral')
    
    for i, pct in enumerate(cum_pcts):
        ax2.text(i, pct + 2, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Accept Length', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('Cumulative Distribution', fontsize=12)
    ax2.set_xticks(range(8))
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    mean_val = np.mean(all_accept_lengths)
    median_val = np.median(all_accept_lengths)
    std_val = np.std(all_accept_lengths)
    
    stats_text = f'Total: {len(all_accept_lengths)}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Accept Length Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    output_path = os.path.join(output_dir, 'accept_len_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存分布图: {output_path}")


def plot_similarity_score(npy_path, output_dir):
    """
    绘制单个任务的similarity_score散点图
    
    Args:
        npy_path: .npy文件路径 (以_similarity.npy结尾)
        output_dir: 输出目录
    """
    # 现代风格的颜色列表
    MODERN_COLORS = [
        '#3498db',  # 明亮蓝
        '#e74c3c',  # 珊瑚红
        '#2ecc71',  # 翡翠绿
        '#9b59b6',  # 紫罗兰
        '#f39c12',  # 橙黄色
        '#1abc9c',  # 青绿色
        '#e91e63',  # 粉红色
        '#00bcd4',  # 青色
        '#ff5722',  # 深橙色
        '#673ab7',  # 深紫色
        '#4caf50',  # 绿色
        '#ff9800',  # 琥珀色
        '#03a9f4',  # 浅蓝色
        '#795548',  # 棕色
        '#607d8b',  # 蓝灰色
    ]
    
    # 加载数据
    similarity_scores = np.load(npy_path)
    
    # 获取任务名（从文件名）
    task_name = os.path.basename(npy_path).replace('_similarity.npy', '')
    
    # 跳过全局统计文件
    if task_name.startswith('all_tasks'):
        print(f"跳过全局统计文件: {task_name}")
        return
    
    # 根据任务名选择颜色（使用哈希值确保同一任务始终使用相同颜色）
    color_idx = hash(task_name) % len(MODERN_COLORS)
    point_color = MODERN_COLORS[color_idx]
    
    # 创建步数数组
    steps = np.arange(len(similarity_scores))
    
    # 创建图形 - 宽度7英寸，高度2英寸
    fig, ax = plt.subplots(figsize=(7, 2))
    
    # 绘制散点图 - 使用任务对应的颜色
    scatter = ax.scatter(steps, similarity_scores, 
                        c=point_color, 
                        alpha=0.7, 
                        s=15,  # 点的大小也适当缩小
                        edgecolors='none')
    
    # 计算统计量
    mean_val = np.mean(similarity_scores)
    median_val = np.median(similarity_scores)
    std_val = np.std(similarity_scores)
    max_val = np.max(similarity_scores)
    min_val = np.min(similarity_scores)
    
    # 只添加均值线（红色，细一点）
    ax.axhline(y=mean_val, color='red', linestyle='-', alpha=0.8, linewidth=1, label=f'Mean: {mean_val:.4f}')
    
    # 设置标签和标题
    ax.set_xlabel('Step', fontsize=10)
    ax.set_ylabel('Confidence', fontsize=10)
    
    # 格式化任务名用于标题（两行标题）
    title_name = task_name.replace('_', ' ').title()
    ax.set_title(f'Confidence of Retrieval Drafts in Our Database\n{title_name}', fontsize=10)
    
    # 设置y轴范围从0.6开始
    ax.set_ylim(0.6, 1.02)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=8)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{task_name}_similarity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {output_path}")
    print(f"  - 总步数: {len(similarity_scores)}, 均值: {mean_val:.4f}, 中位数: {median_val:.4f}")
    
    return {
        'task_name': task_name,
        'total_steps': len(similarity_scores),
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    }


def plot_similarity_distribution(npy_dir, output_dir):
    """
    绘制所有任务的similarity_score分布直方图
    """
    # 获取全局统计文件
    global_files = glob.glob(os.path.join(npy_dir, 'all_tasks_*_similarity.npy'))
    
    if len(global_files) == 0:
        # 如果没有全局文件，合并所有任务的数据
        npy_files = glob.glob(os.path.join(npy_dir, '*_similarity.npy'))
        npy_files = [f for f in npy_files if not os.path.basename(f).startswith('all_tasks')]
        
        if len(npy_files) == 0:
            print("没有找到similarity .npy文件")
            return
        
        all_sim_scores = []
        for npy_path in npy_files:
            all_sim_scores.extend(np.load(npy_path).tolist())
        all_sim_scores = np.array(all_sim_scores)
    else:
        all_sim_scores = np.load(global_files[0])
    
    if len(all_sim_scores) == 0:
        print("similarity数据为空")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图
    counts, bins, patches = ax1.hist(all_sim_scores, bins=20, 
                                      edgecolor='black', alpha=0.7, color='seagreen')
    
    # 为直方图添加颜色渐变
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', plt.cm.RdYlGn(c))
    
    ax1.set_xlabel('Similarity Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Similarity Score Distribution (Histogram)', fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加均值和中位数线
    mean_val = np.mean(all_sim_scores)
    median_val = np.median(all_sim_scores)
    ax1.axvline(x=mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.4f}')
    ax1.axvline(x=median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    ax1.legend()
    
    # 右图：累积分布曲线 (CDF)
    sorted_scores = np.sort(all_sim_scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    
    ax2.plot(sorted_scores, cdf * 100, color='seagreen', linewidth=2)
    ax2.fill_between(sorted_scores, cdf * 100, alpha=0.3, color='seagreen')
    
    ax2.set_xlabel('Similarity Score', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 添加百分位线
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(all_sim_scores, p)
        ax2.axhline(y=p, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=val, color='orange', linestyle='--', alpha=0.7)
        ax2.text(val + 0.02, p + 2, f'P{p}={val:.3f}', fontsize=9)
    
    # 添加统计信息
    std_val = np.std(all_sim_scores)
    stats_text = f'Total: {len(all_sim_scores)}\nMean: {mean_val:.4f}\nMedian: {median_val:.4f}\nStd: {std_val:.4f}'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Similarity Score Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    output_path = os.path.join(output_dir, 'similarity_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存分布图: {output_path}")


def plot_direction_consistency(npy_path, output_dir):
    """
    绘制单个任务的方向一致性散点图
    
    Args:
        npy_path: .npy文件路径 (以_direction.npy结尾)
        output_dir: 输出目录 (figs/direct)
    """
    # 加载数据
    dir_consistency = np.load(npy_path)
    
    # 获取任务名（从文件名）
    task_name = os.path.basename(npy_path).replace('_direction.npy', '')
    
    # 跳过全局统计文件
    if task_name.startswith('all_tasks'):
        print(f"跳过全局统计文件: {task_name}")
        return
    
    # 过滤nan值用于统计，但保留原始索引用于绘图
    steps = np.arange(len(dir_consistency))
    valid_mask = ~np.isnan(dir_consistency)
    valid_steps = steps[valid_mask]
    valid_values = dir_consistency[valid_mask]
    
    if len(valid_values) == 0:
        print(f"跳过（无有效数据）: {task_name}")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制散点图（方差越小越好，用反向色阶）
    scatter = ax.scatter(valid_steps, valid_values, 
                        c=valid_values, 
                        cmap='RdYlGn_r',  # 反向：绿色表示小（好），红色表示大（差）
                        alpha=0.7, 
                        s=30,
                        edgecolors='none')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Direction Consistency (Variance)', fontsize=12)
    
    # 计算统计量
    mean_val = np.mean(valid_values)
    median_val = np.median(valid_values)
    std_val = np.std(valid_values)
    max_val = np.max(valid_values)
    min_val = np.min(valid_values)
    
    # 添加均值线
    ax.axhline(y=mean_val, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.6f}')
    ax.axhline(y=median_val, color='blue', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.6f}')
    
    # 设置标签和标题
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Direction Consistency (Variance of Cosines)', fontsize=14)
    
    # 格式化任务名用于标题
    title_name = task_name.replace('_', ' ').title()
    ax.set_title(f'Direction Consistency vs Step\n{title_name}\n(Lower = More Consistent)', fontsize=14)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息文本框
    stats_text = f'Total Steps: {len(dir_consistency)}\n'
    stats_text += f'Valid Steps: {len(valid_values)}\n'
    stats_text += f'Mean: {mean_val:.6f}\n'
    stats_text += f'Median: {median_val:.6f}\n'
    stats_text += f'Std: {std_val:.6f}\n'
    stats_text += f'Min: {min_val:.6f}\n'
    stats_text += f'Max: {max_val:.6f}'
    
    props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{task_name}_direction.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {output_path}")
    print(f"  - 有效步数: {len(valid_values)}/{len(dir_consistency)}, 均值: {mean_val:.6f}")
    
    return {
        'task_name': task_name,
        'total_steps': len(dir_consistency),
        'valid_steps': len(valid_values),
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    }


def plot_direction_distribution(npy_dir, output_dir):
    """
    绘制所有任务的方向一致性分布直方图
    """
    # 获取全局统计文件
    global_files = glob.glob(os.path.join(npy_dir, 'all_tasks_*_direction.npy'))
    
    if len(global_files) == 0:
        # 如果没有全局文件，合并所有任务的数据
        npy_files = glob.glob(os.path.join(npy_dir, '*_direction.npy'))
        npy_files = [f for f in npy_files if not os.path.basename(f).startswith('all_tasks')]
        
        if len(npy_files) == 0:
            print("没有找到direction .npy文件")
            return
        
        all_dir_values = []
        for npy_path in npy_files:
            data = np.load(npy_path)
            all_dir_values.extend([d for d in data if not np.isnan(d)])
        all_dir_values = np.array(all_dir_values)
    else:
        data = np.load(global_files[0])
        all_dir_values = np.array([d for d in data if not np.isnan(d)])
    
    if len(all_dir_values) == 0:
        print("direction数据为空")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图
    counts, bins, patches = ax1.hist(all_dir_values, bins=30, 
                                      edgecolor='black', alpha=0.7, color='teal')
    
    ax1.set_xlabel('Direction Consistency (Variance)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Direction Consistency Distribution', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加均值和中位数线
    mean_val = np.mean(all_dir_values)
    median_val = np.median(all_dir_values)
    ax1.axvline(x=mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.6f}')
    ax1.axvline(x=median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.6f}')
    ax1.legend()
    
    # 右图：累积分布曲线 (CDF)
    sorted_values = np.sort(all_dir_values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    
    ax2.plot(sorted_values, cdf * 100, color='teal', linewidth=2)
    ax2.fill_between(sorted_values, cdf * 100, alpha=0.3, color='teal')
    
    ax2.set_xlabel('Direction Consistency (Variance)', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('CDF (Lower Variance = Better)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 添加百分位线
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(all_dir_values, p)
        ax2.axhline(y=p, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=val, color='orange', linestyle='--', alpha=0.7)
        ax2.text(val, p + 2, f'P{p}={val:.4f}', fontsize=8)
    
    # 添加统计信息
    std_val = np.std(all_dir_values)
    stats_text = f'Total: {len(all_dir_values)}\nMean: {mean_val:.6f}\nMedian: {median_val:.6f}\nStd: {std_val:.6f}'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('Direction Consistency Distribution Analysis\n(Lower Variance = More Consistent Direction)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    
    output_path = os.path.join(output_dir, 'direction_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存分布图: {output_path}")


def plot_comparison(npy_dir, output_dir):
    """
    绘制accept_length和execution_degree的对比图
    """
    # 获取全局统计文件
    accept_file = glob.glob(os.path.join(npy_dir, 'all_tasks_*.npy'))
    accept_file = [f for f in accept_file if '_exec_degree' not in f]
    
    exec_file = glob.glob(os.path.join(npy_dir, 'all_tasks_*_exec_degree.npy'))
    
    if len(accept_file) == 0 or len(exec_file) == 0:
        print("缺少全局统计文件，跳过对比图")
        return
    
    accept_lengths = np.load(accept_file[0])
    exec_degrees = np.load(exec_file[0])
    
    if len(accept_lengths) != len(exec_degrees):
        print("数据长度不匹配，跳过对比图")
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上：散点图对比
    ax1 = axes[0, 0]
    ax1.scatter(accept_lengths + np.random.uniform(-0.1, 0.1, len(accept_lengths)), 
                exec_degrees + np.random.uniform(-0.1, 0.1, len(exec_degrees)), 
                alpha=0.3, s=20)
    ax1.plot([0, 7], [0, 7], 'r--', linewidth=2, label='y=x (perfect match)')
    ax1.set_xlabel('Accept Length (consecutive)', fontsize=12)
    ax1.set_ylabel('Execution Degree (total matching)', fontsize=12)
    ax1.set_title('Accept Length vs Execution Degree', fontsize=12)
    ax1.set_xlim(-0.5, 7.5)
    ax1.set_ylim(-0.5, 7.5)
    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右上：差值分布
    ax2 = axes[0, 1]
    diff = exec_degrees - accept_lengths
    counts, bins, patches = ax2.hist(diff, bins=np.arange(-0.5, 8.5, 1), 
                                      edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Execution Degree - Accept Length', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Difference Distribution\n(Mean diff: {np.mean(diff):.2f})', fontsize=12)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加百分比标签
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count > 0:
            ax2.text(patch.get_x() + patch.get_width()/2, count + max(counts)*0.02, 
                    f'{int(count)}\n({100*count/len(diff):.1f}%)', 
                    ha='center', va='bottom', fontsize=8)
    
    # 左下：两个分布对比
    ax3 = axes[1, 0]
    x = np.arange(8)
    width = 0.35
    
    accept_counts = [np.sum(accept_lengths == i) for i in range(8)]
    exec_counts = [np.sum(exec_degrees == i) for i in range(8)]
    
    bars1 = ax3.bar(x - width/2, accept_counts, width, label='Accept Length', color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, exec_counts, width, label='Execution Degree', color='coral', alpha=0.8)
    
    ax3.set_xlabel('Value', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Distribution Comparison', fontsize=12)
    ax3.set_xticks(x)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 右下：统计表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Accept Length', 'Execution Degree'],
        ['Mean', f'{np.mean(accept_lengths):.2f}', f'{np.mean(exec_degrees):.2f}'],
        ['Median', f'{np.median(accept_lengths):.2f}', f'{np.median(exec_degrees):.2f}'],
        ['Std', f'{np.std(accept_lengths):.2f}', f'{np.std(exec_degrees):.2f}'],
        ['Min', f'{np.min(accept_lengths)}', f'{np.min(exec_degrees)}'],
        ['Max', f'{np.max(accept_lengths)}', f'{np.max(exec_degrees)}'],
        ['Total', f'{len(accept_lengths)}', f'{len(exec_degrees)}'],
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Statistics Summary', fontsize=12, pad=20)
    
    plt.suptitle('Accept Length vs Execution Degree Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'accept_vs_exec_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存对比图: {output_path}")


def main():
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 正确的路径: openvla/experiments/robot/libero -> openvla/specdecoding/test-speed/fenxi_accept_len
    npy_dir = os.path.join(script_dir, '../../../specdecoding/test-speed/fenxi_accept_len')
    output_dir = os.path.join(script_dir, 'figs')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Accept Length 散点图绘制")
    print("="*60)
    print(f"输入目录: {npy_dir}")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    # 检查输入目录是否存在
    if not os.path.exists(npy_dir):
        print(f"错误: 输入目录不存在: {npy_dir}")
        return
    
    # 获取所有.npy文件
    all_npy_files = glob.glob(os.path.join(npy_dir, '*.npy'))
    
    if len(all_npy_files) == 0:
        print("错误: 没有找到.npy文件")
        print("请先运行 run_fenxi_accept_len.sh 生成数据")
        return
    
    print(f"\n找到 {len(all_npy_files)} 个.npy文件:")
    for f in sorted(all_npy_files):
        print(f"  - {os.path.basename(f)}")
    print()
    
    # 过滤出accept_length文件（不包含_exec_degree, _similarity, _direction后缀的文件）
    npy_files = [f for f in all_npy_files 
                 if not os.path.basename(f).startswith('all_tasks')
                 and not f.endswith('_exec_degree.npy')
                 and not f.endswith('_similarity.npy')
                 and not f.endswith('_direction.npy')]
    
    # 绘制每个任务的accept_length散点图
    print("\n" + "-"*60)
    print("绘制各任务 Accept Length 散点图...")
    print("-"*60)
    
    all_stats = []
    for npy_path in sorted(npy_files):
        stats = plot_accept_length(npy_path, output_dir)
        if stats:
            all_stats.append(stats)
    
    # 绘制每个任务的execution_degree散点图
    exec_npy_files = glob.glob(os.path.join(npy_dir, '*_exec_degree.npy'))
    exec_npy_files = [f for f in exec_npy_files if not os.path.basename(f).startswith('all_tasks')]
    
    if len(exec_npy_files) > 0:
        print("\n" + "-"*60)
        print("绘制各任务 Execution Degree 散点图...")
        print("-"*60)
        
        for npy_path in sorted(exec_npy_files):
            plot_execution_degree(npy_path, output_dir)
    
    # 绘制汇总图
    print("\n" + "-"*60)
    print("绘制汇总图...")
    print("-"*60)
    plot_all_tasks_combined(npy_dir, output_dir)
    
    # 绘制accept_length分布直方图
    print("\n" + "-"*60)
    print("绘制 Accept Length 分布直方图...")
    print("-"*60)
    plot_distribution_histogram(npy_dir, output_dir)
    
    # 绘制execution_degree分布直方图
    if len(exec_npy_files) > 0:
        print("\n" + "-"*60)
        print("绘制 Execution Degree 分布直方图...")
        print("-"*60)
        plot_exec_degree_distribution(npy_dir, output_dir)
        
        # 绘制对比图
        print("\n" + "-"*60)
        print("绘制 Accept Length vs Execution Degree 对比图...")
        print("-"*60)
        plot_comparison(npy_dir, output_dir)
    
    # 绘制similarity_score相关图表
    sim_npy_files = glob.glob(os.path.join(npy_dir, '*_similarity.npy'))
    sim_npy_files = [f for f in sim_npy_files if not os.path.basename(f).startswith('all_tasks')]
    
    if len(sim_npy_files) > 0:
        print("\n" + "-"*60)
        print("绘制各任务 Similarity Score 散点图...")
        print("-"*60)
        
        for npy_path in sorted(sim_npy_files):
            plot_similarity_score(npy_path, output_dir)
        
        print("\n" + "-"*60)
        print("绘制 Similarity Score 分布直方图...")
        print("-"*60)
        plot_similarity_distribution(npy_dir, output_dir)
    
    # 绘制direction_consistency相关图表
    dir_npy_files = glob.glob(os.path.join(npy_dir, '*_direction.npy'))
    dir_npy_files = [f for f in dir_npy_files if not os.path.basename(f).startswith('all_tasks')]
    
    if len(dir_npy_files) > 0:
        # 创建figs/direct目录
        direct_output_dir = os.path.join(output_dir, 'direct')
        os.makedirs(direct_output_dir, exist_ok=True)
        
        print("\n" + "-"*60)
        print("绘制各任务 Direction Consistency 散点图...")
        print(f"输出目录: {direct_output_dir}")
        print("-"*60)
        
        for npy_path in sorted(dir_npy_files):
            plot_direction_consistency(npy_path, direct_output_dir)
        
        print("\n" + "-"*60)
        print("绘制 Direction Consistency 分布直方图...")
        print("-"*60)
        plot_direction_distribution(npy_dir, direct_output_dir)
    
    # 打印汇总统计
    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)
    if all_stats:
        print(f"{'Task Name':<50} {'Steps':>8} {'Mean':>8} {'Median':>8}")
        print("-"*80)
        for s in all_stats:
            name = s['task_name'][:47] + '...' if len(s['task_name']) > 50 else s['task_name']
            print(f"{name:<50} {s['total_steps']:>8} {s['mean']:>8.2f} {s['median']:>8.2f}")
    
    print("\n" + "="*60)
    print(f"所有图片已保存到: {output_dir}")
    if len(dir_npy_files) > 0:
        print(f"方向一致性图片已保存到: {os.path.join(output_dir, 'direct')}")
    print("="*60)


if __name__ == "__main__":
    main()
