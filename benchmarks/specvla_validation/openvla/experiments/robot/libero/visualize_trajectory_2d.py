"""
visualize_trajectory_2d.py

在LIBERO背景图上可视化2D轨迹

从observations.npy中提取图像和轨迹，在第一帧图像上叠加xy坐标轨迹

Usage:
    python experiments/robot/libero/visualize_trajectory_2d.py \
        --npy_path <NPY_PATH> \
        --task_id 0 \
        --episode_id 0 \
        --output trajectory_2d.png
"""

import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def normalize_to_image_coords(positions, img_width, img_height, margin=50):
    """
    将世界坐标归一化到图像坐标
    
    Args:
        positions: (N, 2) array of xy positions
        img_width: 图像宽度
        img_height: 图像高度
        margin: 边距
        
    Returns:
        (N, 2) array of image coordinates
    """
    # 找到xy的范围
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    
    # 归一化到[0, 1]
    x_norm = (positions[:, 0] - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(positions[:, 0])
    y_norm = (positions[:, 1] - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(positions[:, 1])
    
    # 映射到图像坐标（注意y轴翻转）
    img_x = margin + x_norm * (img_width - 2 * margin)
    img_y = img_height - (margin + y_norm * (img_height - 2 * margin))  # 翻转y轴
    
    return np.stack([img_x, img_y], axis=1).astype(np.int32)


def world_to_image_coords(positions, img_width, img_height):
    """
    将3D世界坐标投影到2D图像坐标
    LIBERO相机大致是俯视视角，使用简单的正交投影
    
    Args:
        positions: (N, 3) array of xyz positions
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        (N, 2) array of image coordinates
    """
    # 提取xy坐标
    x = positions[:, 0]
    y = positions[:, 1]
    
    # 找到范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # 添加边距
    x_range = x_max - x_min
    y_range = y_max - y_min
    margin_x = x_range * 0.2 if x_range > 0 else 0.1
    margin_y = y_range * 0.2 if y_range > 0 else 0.1
    
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y
    
    # 归一化到[0, 1]
    if x_max > x_min:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = np.ones_like(x) * 0.5
    
    if y_max > y_min:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = np.ones_like(y) * 0.5
    
    # 映射到图像坐标
    # x对应图像的水平方向，y对应垂直方向（需要翻转）
    # 相机俯视，所以机器人在图像上半部分
    img_x = (x_norm * img_width * 0.6 + img_width * 0.2).astype(np.int32)
    img_y = ((1 - y_norm) * img_height * 0.4 + img_height * 0.1).astype(np.int32)  # 翻转y并映射到上半部分
    
    return np.stack([img_x, img_y], axis=1)


def visualize_trajectory_on_image(obs_data, task_id, episode_id, output_path, use_video=False):
    """
    在LIBERO图像上可视化轨迹（使用xyz坐标投影）
    
    Args:
        obs_data: observations字典
        task_id: 任务ID
        episode_id: Episode ID
        output_path: 输出路径
        use_video: 是否制作GIF动画
    """
    if task_id not in obs_data:
        raise ValueError(f"Task ID {task_id} not found! Available: {list(obs_data.keys())}")
    
    if episode_id not in obs_data[task_id]:
        raise ValueError(f"Episode ID {episode_id} not found! Available: {list(obs_data[task_id].keys())}")
    
    trajectory = obs_data[task_id][episode_id]
    print(f"Task {task_id}, Episode {episode_id}: {len(trajectory)} steps")
    
    # 提取xyz坐标
    positions = np.array([obs['state'][:3] for obs in trajectory])
    
    # 使用第一帧作为背景
    ref_img = trajectory[0]['full_image']
    img_height, img_width = ref_img.shape[:2]
    
    # 将世界坐标转换为图像坐标
    print("Projecting world coordinates to image...")
    img_coords = world_to_image_coords(positions, img_width, img_height)
    
    if use_video:
        # 制作gif动画 - 在每一帧上叠加已走过的轨迹
        import imageio
        output_gif = str(Path(output_path).with_suffix('.gif'))
        frames = []
        
        step_interval = max(1, len(trajectory) // 50)  # 最多50帧
        
        for i in range(0, len(trajectory), step_interval):
            frame = trajectory[i]['full_image'].copy()
            
            # 绘制已经走过的轨迹
            for j in range(min(i, len(img_coords) - 1)):
                # 渐变颜色
                ratio = j / max(1, len(img_coords) - 1)
                color = (int(255 * (1 - ratio)), int(100), int(255 * ratio))
                
                pt1 = tuple(img_coords[j])
                pt2 = tuple(img_coords[j + 1])
                cv2.line(frame, pt1, pt2, color, 3, lineType=cv2.LINE_AA)
            
            # 绘制当前点
            if i < len(img_coords):
                cv2.circle(frame, tuple(img_coords[i]), 8, (0, 0, 255), -1)
                cv2.circle(frame, tuple(img_coords[i]), 10, (255, 255, 255), 2)
            
            # 绘制起点
            cv2.circle(frame, tuple(img_coords[0]), 10, (0, 255, 0), -1)
            cv2.circle(frame, tuple(img_coords[0]), 12, (255, 255, 255), 2)
            
            # 在图像上添加文本信息
            cv2.putText(frame, f"Step {i}/{len(trajectory)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # 显示当前xyz位置
            x, y, z = positions[i][:3]
            cv2.putText(frame, f"XYZ: ({x:.3f}, {y:.3f}, {z:.3f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            frames.append(frame)
        
        # 保存为gif
        imageio.mimsave(output_gif, frames, fps=10, loop=0)
        print(f"GIF saved to: {output_gif}")
    
    else:
        # 静态图：在第一帧上绘制完整轨迹
        result_image = ref_img.copy()
        
        # 绘制轨迹线
        for i in range(len(img_coords) - 1):
            # 渐变颜色（从蓝到红）
            ratio = i / max(1, len(img_coords) - 1)
            color = (int(255 * (1 - ratio)), int(100), int(255 * ratio))
            
            pt1 = tuple(img_coords[i])
            pt2 = tuple(img_coords[i + 1])
            cv2.line(result_image, pt1, pt2, color, 4, lineType=cv2.LINE_AA)
        
        # 绘制轨迹点
        for i, coord in enumerate(img_coords):
            ratio = i / max(1, len(img_coords) - 1)
            color = (int(255 * (1 - ratio)), int(100), int(255 * ratio))
            cv2.circle(result_image, tuple(coord), 5, color, -1)
        
        # 标记起点（绿色大圆）
        cv2.circle(result_image, tuple(img_coords[0]), 12, (0, 255, 0), -1)
        cv2.circle(result_image, tuple(img_coords[0]), 15, (255, 255, 255), 2)
        
        # 标记终点（红色方块）
        end_pt = img_coords[-1]
        cv2.rectangle(result_image, 
                     (end_pt[0]-10, end_pt[1]-10), 
                     (end_pt[0]+10, end_pt[1]+10), 
                     (0, 0, 255), -1)
        cv2.rectangle(result_image, 
                     (end_pt[0]-13, end_pt[1]-13), 
                     (end_pt[0]+13, end_pt[1]+13), 
                     (255, 255, 255), 2)
        
        # 保存图像
        plt.figure(figsize=(12, 9))
        plt.imshow(result_image)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Image saved to: {output_path}")
    
    # 打印统计
    print(f"\nTrajectory Statistics:")
    print(f"  X range: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]")
    print(f"  Y range: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")
    print(f"  Z range: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}]")
    print(f"  Trajectory length: {len(positions)} steps")


def main():
    parser = argparse.ArgumentParser(description='在LIBERO图像上可视化2D轨迹')
    parser.add_argument('--npy_path', type=str, required=True,
                        help='observations.npy文件路径')
    parser.add_argument('--task_id', type=int, default=0,
                        help='任务ID')
    parser.add_argument('--episode_id', type=int, default=0,
                        help='Episode ID')
    parser.add_argument('--output', type=str, default='trajectory_2d.png',
                        help='输出图片路径')
    parser.add_argument('--video', action='store_true',
                        help='生成视频而不是静态图')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading observations from: {args.npy_path}")
    obs_data = np.load(args.npy_path, allow_pickle=True).item()
    print(f"Loaded {len(obs_data)} tasks")
    
    # 可视化
    visualize_trajectory_on_image(
        obs_data, 
        args.task_id, 
        args.episode_id, 
        args.output,
        args.video
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
