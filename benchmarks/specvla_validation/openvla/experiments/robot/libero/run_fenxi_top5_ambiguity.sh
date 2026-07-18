#!/bin/bash
# 分析每一步检索top5的歧义性和置信度
# 
# - 歧义性：top5 action相对于7维重心的平均距离（折线图）
# - 置信度热力图：top1-5的相似度分数（MinMax归一化）
#
# 使用方法: bash run_fenxi_top5_ambiguity.sh
#
# 前置条件：
# 1. 确保 Mix embedding 服务器正在运行 (http://127.0.0.1:9021/predict)
# 2. 确保 Qdrant 数据库正在运行 (localhost:6333)

set -e

# =============================================================================
# 设置工作目录
# =============================================================================
SPECVLA_ROOT="/path/to/SpecVLA"
cd $SPECVLA_ROOT

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

if [ "$CONDA_DEFAULT_ENV" != "specvla" ]; then
    echo "[WARNING] Conda environment may not be activated correctly. Current: $CONDA_DEFAULT_ENV"
fi

# =============================================================================
# 设置环境变量
# =============================================================================
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=$SPECVLA_ROOT/robosuite.log
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

echo "[INFO] Working directory: $SPECVLA_ROOT"
echo "[INFO] Conda environment: $CONDA_DEFAULT_ENV"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] PYTHONPATH=$PYTHONPATH"

# =============================================================================
# 安装依赖
# =============================================================================
echo "检查并安装 qdrant-client..."
pip install qdrant-client -q 2>/dev/null || pip install qdrant-client

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

# 检查Mix Embedding服务 (端口9021)
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
TASK_SUITE="libero_goal"
NUM_TRIALS=1  # 每个任务只跑1次
RUN_ID_NOTE="top5_ambiguity"

echo "=========================================="
echo "实验配置"
echo "=========================================="
echo "任务套件: $TASK_SUITE"
echo "每任务试验次数: $NUM_TRIALS"
echo "----------------------------------------"
echo "分析内容:"
echo "  1. Top-5 歧义性（折线图）"
echo "  2. Top-5 置信度热力图（MinMax归一化）"
echo "=========================================="
echo ""

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "开始运行 Top-5 歧义性分析..."
echo "=========================================="
echo ""

python openvla/experiments/robot/libero/run_fenxi_top5_ambiguity.py \
    --task_suite_name "$TASK_SUITE" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "$RUN_ID_NOTE"

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "检查结果目录: openvla/experiments/robot/libero/figs/top5_analysis/"
echo "=========================================="
