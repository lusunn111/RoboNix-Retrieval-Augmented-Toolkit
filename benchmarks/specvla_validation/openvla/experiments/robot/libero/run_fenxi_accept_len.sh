#!/bin/bash
# 分析每一步检索action slice的接受长度
# 
# 纯AR模式：每一步都使用AR生成action，同时检索并计算accept_length（阈值=9）
# 将每个任务的accept_length列表存为.npy文件
#
# 使用方法: bash run_fenxi_accept_len.sh
#
# 前置条件：
# 1. 确保 Mix视角检索服务器正在运行 (http://127.0.0.1:5003/pipeline)
# 2. 确保 embedding 服务器正在运行 (http://127.0.0.1:9021/predict)
# 3. 确保 Qdrant 数据库正在运行 (localhost:6333)

set -e  # 遇到错误立即退出

# =============================================================================
# 设置工作目录
# =============================================================================
SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

# 检查目录是否存在
if [ ! -d "$SPECVLA_ROOT" ]; then
    echo "错误: SpecVLA 根目录不存在: $SPECVLA_ROOT"
    exit 1
fi

# =============================================================================
# 激活conda环境
# =============================================================================
echo "Activating conda environment: specvla"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate specvla

# Check if conda environment is activated
if [ "$CONDA_DEFAULT_ENV" != "specvla" ]; then
    echo "[WARNING] Conda environment may not be activated correctly. Current: $CONDA_DEFAULT_ENV"
fi

# =============================================================================
# 设置环境变量（关键！）
# =============================================================================
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=$SPECVLA_ROOT/robosuite.log
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

echo "[INFO] Working directory: $SPECVLA_ROOT"
echo "[INFO] Conda environment: $CONDA_DEFAULT_ENV"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
echo "[INFO] PYTHONPATH=$PYTHONPATH"

# =============================================================================
# 检查服务可用性
# =============================================================================
echo "=========================================="
echo "检查服务可用性..."
echo "=========================================="

# 检查Qdrant
echo -n "检查Qdrant服务 (localhost:6333)... "
if curl -s "http://localhost:6333/collections" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "错误: Qdrant服务未运行！请先启动Qdrant服务。"
    exit 1
fi

# 检查Mix视角Retrieval API (端口5003)
echo -n "检查Mix视角Retrieval API (http://127.0.0.1:5003)... "
if curl -s --connect-timeout 2 "http://127.0.0.1:5003/health" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "警告: Mix视角Retrieval API可能未运行（端口5003），但将继续执行..."
fi

# 检查Embedding服务 (端口9021)
echo -n "检查Mix Embedding服务 (http://127.0.0.1:9021)... "
if curl -s --connect-timeout 2 "http://127.0.0.1:9021/health" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "警告: Mix Embedding服务可能未运行（端口9021），但将继续执行..."
fi

echo "=========================================="
echo ""

# =============================================================================
# 实验参数设置
# =============================================================================
MODEL_FAMILY="openvla"
PRETRAINED_CHECKPOINT="/path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
TASK_SUITE="libero_goal"
CENTER_CROP="True"
ACCEPT_THRESHOLD=9  # 放松阈值

NUM_TRIALS=1  # 每个任务只跑1次
RUN_ID_NOTE="fenxi_accept_len"

echo "=========================================="
echo "实验配置"
echo "=========================================="
echo "模型: $MODEL_FAMILY"
echo "预训练检查点: $PRETRAINED_CHECKPOINT"
echo "任务套件: $TASK_SUITE"
echo "Center crop: $CENTER_CROP"
echo "Accept threshold (放松阈值): $ACCEPT_THRESHOLD"
echo "----------------------------------------"
echo "实验说明:"
echo "  纯AR模式，每一步都检索并计算accept_length"
echo "  每个任务跑1次"
echo "  结果保存为 task_name.npy"
echo "----------------------------------------"
echo "Mix视角检索服务: http://127.0.0.1:5003/pipeline"
echo "每任务试验次数: $NUM_TRIALS"
echo "运行标记: $RUN_ID_NOTE"
echo "=========================================="
echo ""

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "开始运行Accept Length分析实验..."
echo "=========================================="
echo ""

# 运行Python脚本
python openvla/experiments/robot/libero/run_fenxi_accept_len.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop "$CENTER_CROP" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_spec True \
    --parallel_draft False

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "检查结果目录: ./specdecoding/test-speed/fenxi_accept_len/"
echo "每个任务的accept_length保存为: task_name.npy"
echo "=========================================="
