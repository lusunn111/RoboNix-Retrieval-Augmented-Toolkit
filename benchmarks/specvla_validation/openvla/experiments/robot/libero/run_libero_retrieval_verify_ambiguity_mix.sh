#!/bin/bash
# 运行基于综合指标 + Ambiguity指标的检索验证实验（Mix视角）
# 
# 规则：
# 1. 首先用综合指标判断：
#    - 综合指标 > 0.143210（25%分位数）→ 初始判定 noverify (DB)
#    - 综合指标 <= 0.143210 → verify (AR)
#
# 2. 当初始判定为 noverify 时，用 ambiguity 变化趋势调整：
#    - ambiguity 上升 → 强制 verify (AR)
#    - ambiguity 下降 → 继续 noverify (DB)
#    - ambiguity 平稳 → 交替切换
#
# 归一化参数：
# - 位移指标：[0.000009, 0.123381]
# - 曲率半径：[0.000001, 0.014989]
# - alpha = 0.5（1:1构造综合指标）
#
# 使用方法: bash run_libero_retrieval_verify_ambiguity_mix.sh
#
# 前置条件：
# 1. 确保 Mix embedding 服务器正在运行 (http://127.0.0.1:9021/predict)
# 2. 确保 Qdrant 数据库正在运行 (localhost:6333)

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

# =============================================================================
# 设置环境变量
# =============================================================================
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=$SPECVLA_ROOT/robosuite.log
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

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
MODEL_FAMILY="openvla"
PRETRAINED_CHECKPOINT="/path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
TASK_SUITE="libero_goal"
CENTER_CROP="True"
ACCEPT_THRESHOLD=9

# 综合指标参数
WINDOW_SIZE=5
COMPOSITE_THRESHOLD=0.143210  # 25%分位数阈值
ALPHA=0.5  # 1:1权重
DISPLACEMENT_RANGE_MIN=0.000009
DISPLACEMENT_RANGE_MAX=0.123381
RADIUS_RANGE_MIN=0.000001
RADIUS_RANGE_MAX=0.014989

# Ambiguity指标参数（用于替代"两步强制verify"）
TOP_K=5  # 检索top-k用于计算ambiguity
HISTORY_WINDOW=3  # ambiguity变化趋势的历史窗口
RISE_THRESHOLD=0.01  # ambiguity上升阈值
FALL_THRESHOLD=0.01  # ambiguity下降阈值

NUM_TRIALS=10  # 每个任务运行10次
RUN_ID_NOTE="Composite_Ambiguity_Mix"

# 日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiments/logs/retrieval_verify_ambiguity_mix_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "实验配置"
echo "=========================================="
echo "模型: $MODEL_FAMILY"
echo "预训练检查点: $PRETRAINED_CHECKPOINT"
echo "Spec检查点: $SPEC_CHECKPOINT"
echo "任务套件: $TASK_SUITE"
echo "Center crop: $CENTER_CROP"
echo "Accept threshold: $ACCEPT_THRESHOLD"
echo "----------------------------------------"
echo "综合指标参数:"
echo "  窗口大小: $WINDOW_SIZE"
echo "  综合指标阈值: $COMPOSITE_THRESHOLD (25%分位数)"
echo "  Alpha (曲率半径权重): $ALPHA (1:1权重)"
echo "  位移归一化范围: [$DISPLACEMENT_RANGE_MIN, $DISPLACEMENT_RANGE_MAX]"
echo "  曲率半径归一化范围: [$RADIUS_RANGE_MIN, $RADIUS_RANGE_MAX]"
echo "----------------------------------------"
echo "Ambiguity指标参数（替代两步强制verify）:"
echo "  Top-K: $TOP_K"
echo "  历史窗口: $HISTORY_WINDOW"
echo "  上升阈值: $RISE_THRESHOLD"
echo "  下降阈值: $FALL_THRESHOLD"
echo "----------------------------------------"
echo "决策逻辑:"
echo "  1. 综合指标 > $COMPOSITE_THRESHOLD → 初始判定 noverify (DB)"
echo "  2. 综合指标 <= $COMPOSITE_THRESHOLD → verify (AR)"
echo "  3. 当判定为 noverify 时，根据 ambiguity 变化趋势调整："
echo "     - ambiguity 上升 → 强制 verify (AR)"
echo "     - ambiguity 下降 → 继续 noverify (DB)"
echo "     - ambiguity 平稳 → 交替切换"
echo "----------------------------------------"
echo "检索方式: 直接Qdrant (无检索服务器)"
echo "Embedding服务: http://127.0.0.1:9021/predict"
echo "Qdrant服务: localhost:6333"
echo "每任务试验次数: $NUM_TRIALS"
echo "运行标记: $RUN_ID_NOTE"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "开始运行综合指标 + Ambiguity指标检索验证实验（Mix视角）..."
echo "=========================================="
echo ""

# 运行Python脚本
python openvla/experiments/robot/libero/run_libero_retrieval_verify_ambiguity_mix.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop "$CENTER_CROP" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --window_size "$WINDOW_SIZE" \
    --composite_threshold "$COMPOSITE_THRESHOLD" \
    --alpha "$ALPHA" \
    --displacement_range_min "$DISPLACEMENT_RANGE_MIN" \
    --displacement_range_max "$DISPLACEMENT_RANGE_MAX" \
    --radius_range_min "$RADIUS_RANGE_MIN" \
    --radius_range_max "$RADIUS_RANGE_MAX" \
    --top_k "$TOP_K" \
    --history_window "$HISTORY_WINDOW" \
    --rise_threshold "$RISE_THRESHOLD" \
    --fall_threshold "$FALL_THRESHOLD" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "检查日志目录: ./experiments/logs/"
echo "检查结果目录: ./specdecoding/test-speed/libero_retrieval_verify_ambiguity_mix/"
echo "=========================================="
