"""
demo.py - 可视化机器人末端执行器轨迹

从observations.npy文件中加载数据，并生成3D轨迹图

使用方法:
    python demo.py --npy_path observations.npy --task_id 0 --episode_id 0 --output trajectory_plot.png
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_observations(npy_path):
    """加载observations数据"""
    observations_data = np.load(npy_path, allow_pickle=True)
    obs_dict = observations_data.item()
    
    print(f"数据已加载: {npy_path}")
    print(f"任务数量: {len(obs_dict)}")
    print(f"任务ID列表: {list(obs_dict.keys())}")
    
    return obs_dict


def normalize_episode_data(episode_data):
    """统一不同 observations 结构，返回轨迹列表和附加元信息。"""
    if isinstance(episode_data, dict):
        if 'observations' in episode_data:
            trajectory = episode_data['observations']
            metadata = {
                'success': episode_data.get('success'),
                'num_steps': episode_data.get('num_steps', len(trajectory)),
                'task_description': episode_data.get('task_description'),
            }
            return trajectory, metadata
        raise ValueError('不支持的 episode 数据格式：缺少 observations 字段')

    if isinstance(episode_data, list):
        return episode_data, {
            'success': None,
            'num_steps': len(episode_data),
            'task_description': None,
        }

    raise ValueError(f'不支持的 episode 数据类型: {type(episode_data)}')


def sanitize_filename(text):
    safe = ''.join(ch if ch.isalnum() else '_' for ch in str(text))
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_') or 'unknown'


def extract_trajectory(obs_dict, task_id, episode_id):
    """提取指定任务和episode的轨迹"""
    if task_id not in obs_dict:
        raise ValueError(f"任务ID {task_id} 不存在！可用任务: {list(obs_dict.keys())}")
    
    if episode_id not in obs_dict[task_id]:
        raise ValueError(f"Episode ID {episode_id} 不存在！可用episodes: {list(obs_dict[task_id].keys())}")
    
    trajectory, metadata = normalize_episode_data(obs_dict[task_id][episode_id])
    print(f"\n任务 {task_id}, Episode {episode_id}")
    print(f"轨迹长度: {len(trajectory)} steps")
    if metadata.get('task_description'):
        print(f"任务描述: {metadata['task_description']}")
    
    return trajectory, metadata


def plot_trajectory_3d(trajectory, output_path, task_id=None, episode_id=None, dpi=300):
    """
    绘制3D轨迹图
    
    Args:
        trajectory: 轨迹数据列表，每个元素包含'state'和'full_image'
        output_path: 输出图片路径
        task_id: 任务ID（可选，用于标题）
        episode_id: Episode ID（可选，用于标题）
        dpi: 图片分辨率
    """
    # 提取state（前3个维度是xyz坐标）
    state_list = [obs['state'] for obs in trajectory]
    states = np.array(state_list)
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹线
    ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.6, label='Trajectory')
    
    # 绘制散点（颜色映射到时间步）
    scatter = ax.scatter(x, y, z, c=range(len(x)), cmap='viridis', 
                         s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # 标记起点和终点
    ax.scatter(x[0], y[0], z[0], c='green', s=200, marker='o', 
              edgecolors='black', linewidth=2, label='Start', zorder=5)
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=200, marker='s', 
              edgecolors='black', linewidth=2, label='End', zorder=5)
    
    # 设置标签
    ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Position (m)', fontsize=12, labelpad=10)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Time Step', fontsize=11)
    
    # 设置图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # 设置网格
    ax.grid(True, alpha=0.3)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    # 打印统计信息
    print(f"\n轨迹统计:")
    print(f"  X范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Y范围: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Z范围: [{z.min():.4f}, {z.max():.4f}]")
    print(f"  轨迹长度: {len(x)} steps")
    
    plt.close()


def batch_plot_trajectories(obs_dict, npy_path, output_dir, dpi=300, metadata_output=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_records = []
    for task_id in sorted(obs_dict.keys()):
        for episode_id in sorted(obs_dict[task_id].keys()):
            trajectory, episode_metadata = extract_trajectory(obs_dict, task_id, episode_id)

            task_desc = episode_metadata.get('task_description')
            if task_desc:
                prefix = sanitize_filename(task_desc)
                filename = f"{prefix}_task{task_id}_ep{episode_id}.png"
            else:
                filename = f"task{task_id}_ep{episode_id}.png"

            output_path = output_dir / filename
            plot_trajectory_3d(trajectory, str(output_path), task_id, episode_id, dpi)

            metadata_records.append({
                'task_id': task_id,
                'episode_id': episode_id,
                'num_steps': len(trajectory),
                'task_description': task_desc,
                'success': episode_metadata.get('success'),
                'source_npy': str(Path(npy_path).resolve()),
                'output_png': str(output_path.resolve()),
            })

    if metadata_output is None:
        metadata_output = output_dir / 'metadata.json'
    else:
        metadata_output = Path(metadata_output)

    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata_records, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\n批量绘图完成，共生成 {len(metadata_records)} 张图片")
    print(f"元数据已保存到: {metadata_output}")
    return metadata_records, metadata_output


def main():
    parser = argparse.ArgumentParser(description='可视化机器人末端执行器轨迹')
    parser.add_argument('--npy_path', type=str, 
                        default='/path/to/SpecVLA/observations.npy',
                        help='observations.npy文件路径')
    parser.add_argument('--task_id', type=int, default=0,
                        help='任务ID')
    parser.add_argument('--episode_id', type=int, default=0,
                        help='Episode ID')
    parser.add_argument('--output', type=str, 
                        default='/path/to/SpecVLA/trajectory_plot.png',
                        help='输出图片路径')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/SpecVLA/trajectory_visualizations/all_tasks_from_observations',
                        help='批量模式输出目录')
    parser.add_argument('--metadata_output', type=str, default=None,
                        help='批量模式元数据 JSON 输出路径')
    parser.add_argument('--all_tasks', action='store_true',
                        help='批量绘制所有 task / episode')
    parser.add_argument('--dpi', type=int, default=300,
                        help='图片分辨率（DPI）')
    
    args = parser.parse_args()
    
    # 加载数据
    obs_dict = load_observations(args.npy_path)

    if args.all_tasks:
        batch_plot_trajectories(
            obs_dict,
            args.npy_path,
            args.output_dir,
            dpi=args.dpi,
            metadata_output=args.metadata_output,
        )
        print("\n完成！")
        return
    
    # 提取轨迹
    trajectory, _ = extract_trajectory(obs_dict, args.task_id, args.episode_id)
    
    # 绘制轨迹
    plot_trajectory_3d(trajectory, args.output, args.task_id, args.episode_id, args.dpi)
    
    print("\n完成！")


if __name__ == '__main__':
    main()
