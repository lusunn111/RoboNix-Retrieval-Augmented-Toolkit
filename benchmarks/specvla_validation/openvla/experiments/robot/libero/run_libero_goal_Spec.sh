#!/bin/bash
# 运行Speculative Decoding实验
# 使用Spec模型进行推理加速
#
# 使用方法: bash run_libero_goal_Spec.sh

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
# 实验参数设置
# =============================================================================
MODEL_FAMILY="openvla"
PRETRAINED_CHECKPOINT="/path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"
SPEC_CHECKPOINT="/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
TASK_SUITE="libero_goal"
CENTER_CROP="True"
USE_SPEC="True"
PARALLEL_DRAFT="False"
ACCEPT_THRESHOLD=0
NUM_TRIALS=10
RUN_ID_NOTE="Spec_Decoding"

# 日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiments/logs/spec_decoding_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Speculative Decoding 实验配置"
echo "=========================================="
echo "模型: $MODEL_FAMILY"
echo "预训练检查点: $PRETRAINED_CHECKPOINT"
echo "Spec检查点: $SPEC_CHECKPOINT"
echo "任务套件: $TASK_SUITE"
echo "Center crop: $CENTER_CROP"
echo "使用Spec: $USE_SPEC"
echo "Parallel draft: $PARALLEL_DRAFT"
echo "Accept threshold: $ACCEPT_THRESHOLD"
echo "每任务试验次数: $NUM_TRIALS"
echo "运行标记: $RUN_ID_NOTE"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "开始运行Speculative Decoding实验..."
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
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo ""

# 运行Python脚本
python openvla/experiments/robot/libero/run_libero_goal_Spec.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --spec_checkpoint "$SPEC_CHECKPOINT" \
    --task_suite_name "$TASK_SUITE" \
    --center_crop "$CENTER_CROP" \
    --use_spec "$USE_SPEC" \
    --parallel_draft "$PARALLEL_DRAFT" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_wandb False

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "检查日志目录: $LOG_DIR"
echo "检查结果目录: ./experiments/logs/spec_results/"
echo "=========================================="
