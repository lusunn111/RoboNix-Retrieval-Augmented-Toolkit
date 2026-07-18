#!/bin/bash
# 运行2D轨迹可视化
# 在LIBERO背景图上叠加机器人轨迹
#
# 使用方法: bash run_visualize_2d.sh

set -e

# =============================================================================
# 配置参数
# =============================================================================
SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# 默认参数
NPY_PATH="$SPECVLA_ROOT/openvla/specdecoding/test-speed/libero_goal_Retrieval_Verify/EVAL-libero_goal-openvla-*_observations.npy"
TASK_ID=0
EPISODE_ID=0
OUTPUT_DIR="$SPECVLA_ROOT/trajectory_visualizations"
USE_VIDEO=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --npy_path)
            NPY_PATH="$2"
            shift 2
            ;;
        --task_id)
            TASK_ID="$2"
            shift 2
            ;;
        --episode_id)
            EPISODE_ID="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --video)
            USE_VIDEO=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 查找最新的observations.npy文件（如果使用通配符）
if [[ "$NPY_PATH" == *"*"* ]]; then
    NPY_FILE=$(ls -t $NPY_PATH 2>/dev/null | head -1)
    if [ -z "$NPY_FILE" ]; then
        echo "错误: 未找到匹配的.npy文件: $NPY_PATH"
        exit 1
    fi
    NPY_PATH="$NPY_FILE"
fi

# 检查文件是否存在
if [ ! -f "$NPY_PATH" ]; then
    echo "错误: observations文件不存在: $NPY_PATH"
    exit 1
fi

echo "=========================================="
echo "2D轨迹可视化配置"
echo "=========================================="
echo "NPY文件: $NPY_PATH"
echo "任务ID: $TASK_ID"
echo "Episode ID: $EPISODE_ID"
echo "输出目录: $OUTPUT_DIR"
echo "生成视频: $USE_VIDEO"
echo "=========================================="
echo ""

# 激活conda环境
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate specvla || {
        echo "错误: 无法激活conda环境 'specvla'"
        exit 1
    }
    echo "Conda环境已激活: specvla"
else
    echo "警告: 未找到conda命令"
fi

# 设置输出文件名
if [ "$USE_VIDEO" = true ]; then
    OUTPUT_FILE="$OUTPUT_DIR/trajectory_task${TASK_ID}_ep${EPISODE_ID}.gif"
    VIDEO_FLAG="--video"
else
    OUTPUT_FILE="$OUTPUT_DIR/trajectory_task${TASK_ID}_ep${EPISODE_ID}.png"
    VIDEO_FLAG=""
fi

# 运行可视化脚本
echo "开始生成轨迹可视化..."
python openvla/experiments/robot/libero/visualize_trajectory_2d.py \
    --npy_path "$NPY_PATH" \
    --task_id $TASK_ID \
    --episode_id $EPISODE_ID \
    --output "$OUTPUT_FILE" \
    $VIDEO_FLAG

echo ""
echo "=========================================="
echo "可视化完成！"
echo "=========================================="
echo "输出文件: $OUTPUT_FILE"
echo "=========================================="

# 可选：为所有任务生成可视化
read -p "是否为所有任务生成可视化？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始批量生成..."
    
    # 获取任务数量（假设有10个任务）
    for task_id in {0..9}; do
        echo "处理任务 $task_id..."
        if [ "$USE_VIDEO" = true ]; then
            output_file="$OUTPUT_DIR/trajectory_task${task_id}_ep0.gif"
        else
            output_file="$OUTPUT_DIR/trajectory_task${task_id}_ep0.png"
        fi
        
        python openvla/experiments/robot/libero/visualize_trajectory_2d.py \
            --npy_path "$NPY_PATH" \
            --task_id $task_id \
            --episode_id 0 \
            --output "$output_file" \
            $VIDEO_FLAG || echo "任务 $task_id 失败，跳过..."
    done
    
    echo "批量生成完成！"
    echo "所有文件保存在: $OUTPUT_DIR"
fi
