#!/bin/bash
# 运行基于综合指标的检索验证实验（Mix视角）
# 
# 基于综合指标（曲率半径+位移）动态切换检索与AR生成
# - 综合指标 > 0.143210（25%分位数）：使用检索（DB）
# - 综合指标 <= 0.143210：使用AR生成
# - 两次连续检索后，强制执行一次AR
#
# 归一化参数（minmax归一化，超过范围取0或1）：
# - 位移指标：[0.000009, 0.123381]
# - 曲率半径：[0.000001, 0.014989]
# - alpha = 0.5（1:1构造综合指标）
#
# 使用方法: bash run_libero_retrieval_verify_mix.sh
#
# 前置条件：
# 1. 确保 Mix视角检索服务器正在运行 (http://127.0.0.1:5003/pipeline)
# 2. 确保 embedding 服务器正在运行 (http://127.0.0.1:9020/predict)
# 3. 确保 Qdrant 数据库正在运行 (localhost:6333)
# 注意：不需要恢复数据库，假设服务都已启动好

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
if curl -s "http://127.0.0.1:5003" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "错误: Mix视角Retrieval API未运行！请先启动检索服务（端口5003）。"
    exit 1
fi

# 检查Embedding服务
echo -n "检查Embedding服务 (http://127.0.0.1:9020)... "
if curl -s "http://127.0.0.1:9020" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "警告: Embedding服务可能未运行（端口9020），但将继续执行..."
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

# 综合指标参数（新的归一化范围）
WINDOW_SIZE=5
COMPOSITE_THRESHOLD=0.143210  # 25%分位数阈值
ALPHA=0.5  # 1:1权重
DISPLACEMENT_RANGE_MIN=0.000009
DISPLACEMENT_RANGE_MAX=0.123381
RADIUS_RANGE_MIN=0.000001
RADIUS_RANGE_MAX=0.014989

NUM_TRIALS=10
RUN_ID_NOTE="Composite_Metric_Mix"

# 日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiments/logs/retrieval_verify_mix_${TIMESTAMP}"
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
echo "决策逻辑:"
echo "  综合指标 > $COMPOSITE_THRESHOLD: 使用检索（DB）"
echo "  综合指标 <= $COMPOSITE_THRESHOLD: 使用AR生成"
echo "  两次连续检索后，强制执行一次AR"
echo "----------------------------------------"
echo "Mix视角检索服务: http://127.0.0.1:5003/pipeline"
echo "每任务试验次数: $NUM_TRIALS"
echo "运行标记: $RUN_ID_NOTE"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "开始运行综合指标检索验证实验（Mix视角）..."
echo "=========================================="
echo ""

# 激活SpecVLA环境
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

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo ""

# 运行Python脚本
python openvla/experiments/robot/libero/run_libero_retrieval_verify_mix.py \
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
echo "检查结果目录: ./specdecoding/test-speed/libero_retrieval_verify_mix/"
echo "=========================================="
