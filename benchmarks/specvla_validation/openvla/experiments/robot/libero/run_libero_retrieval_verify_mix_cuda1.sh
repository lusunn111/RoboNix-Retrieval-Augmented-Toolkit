#!/bin/bash
# GPU 1: libero_10 (Long)
# 基于综合指标的检索验证实验（Mix视角）
#
# 使用方法: bash run_libero_retrieval_verify_mix_cuda1.sh

set -e

SPECVLA_ROOT="/path/to/SpecVLA"
cd "$SPECVLA_ROOT"

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

# 通用参数
MODEL_FAMILY="openvla"
CENTER_CROP="True"
ACCEPT_THRESHOLD=9
WINDOW_SIZE=5
ALPHA=0.5  # 1:1权重
NUM_TRIALS=10

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiments/logs/retrieval_verify_mix_cuda1_$TIMESTAMP"
mkdir -p "$LOG_DIR"

# 激活conda环境
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate specvla || exit 1
fi

echo "=========================================="
echo "GPU 1: libero_10 (Long)"
echo "综合指标检索验证实验 (Mix View)"
echo "=========================================="
echo "Log dir: $LOG_DIR"
echo ""

# ============================================
# [1/1] libero_10 (Long)
# ============================================
echo "=========================================="
echo "[1/1] Running libero_10 (Long)..."
echo "=========================================="
echo "参数配置:"
echo "  位移指标范围: [0.000008, 0.102298]"
echo "  曲率半径范围: [0.000001, 0.012479]"
echo "  阈值: 0.076252"
echo "=========================================="

python openvla/experiments/robot/libero/run_libero_retrieval_verify_mix.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-10" \
    --spec_checkpoint "$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_10_debug_ckpt/state_190" \
    --task_suite_name "libero_10" \
    --center_crop "$CENTER_CROP" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --window_size "$WINDOW_SIZE" \
    --alpha "$ALPHA" \
    --displacement_range_min 0.000008 \
    --displacement_range_max 0.102298 \
    --radius_range_min 0.000001 \
    --radius_range_max 0.012479 \
    --composite_threshold 0.076252 \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "Composite_Metric_Mix_Long" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False \
    2>&1 | tee "$LOG_DIR/libero_10.log"

echo "[1/1] libero_10 completed!"
echo ""

echo "=========================================="
echo "GPU 1 experiment complete!"
echo "=========================================="
echo "Log dir: $LOG_DIR"
echo "Results dir: $SPECVLA_ROOT/openvla/specdecoding/test-speed/libero_retrieval_verify_mix/"
echo "=========================================="
