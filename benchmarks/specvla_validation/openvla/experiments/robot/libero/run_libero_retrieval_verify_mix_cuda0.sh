#!/bin/bash
# GPU 0: libero_spatial + libero_object
# 基于综合指标的检索验证实验（Mix视角）
#
# 使用方法: bash run_libero_retrieval_verify_mix_cuda0.sh

set -e

SPECVLA_ROOT="/path/to/SpecVLA"
cd "$SPECVLA_ROOT"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

# 通用参数
MODEL_FAMILY="openvla"
CENTER_CROP="True"
ACCEPT_THRESHOLD=9
WINDOW_SIZE=5
ALPHA=0.5  # 1:1权重
NUM_TRIALS=10

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiments/logs/retrieval_verify_mix_cuda0_$TIMESTAMP"
mkdir -p "$LOG_DIR"

# 激活conda环境
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate specvla || exit 1
fi

echo "=========================================="
echo "GPU 0: libero_spatial + libero_object"
echo "综合指标检索验证实验 (Mix View)"
echo "=========================================="
echo "Log dir: $LOG_DIR"
echo ""

# ============================================
# [1/2] libero_spatial
# ============================================
echo "=========================================="
echo "[1/2] Running libero_spatial..."
echo "=========================================="
echo "参数配置:"
echo "  位移指标范围: [0.000027, 0.128629]"
echo "  曲率半径范围: [0.000019, 0.015654]"
echo "  阈值: 0.217119"
echo "=========================================="

python openvla/experiments/robot/libero/run_libero_retrieval_verify_mix.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-spatial" \
    --spec_checkpoint "$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_spatial_debug_ckpt/state_190" \
    --task_suite_name "libero_spatial" \
    --center_crop "$CENTER_CROP" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --window_size "$WINDOW_SIZE" \
    --alpha "$ALPHA" \
    --displacement_range_min 0.000027 \
    --displacement_range_max 0.128629 \
    --radius_range_min 0.000019 \
    --radius_range_max 0.015654 \
    --composite_threshold 0.217119 \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "Composite_Metric_Mix_Spatial" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False \
    2>&1 | tee "$LOG_DIR/libero_spatial.log"

echo "[1/2] libero_spatial completed!"
echo ""

# ============================================
# [2/2] libero_object
# ============================================
echo "=========================================="
echo "[2/2] Running libero_object..."
echo "=========================================="
echo "参数配置:"
echo "  位移指标范围: [0.000098, 0.116458]"
echo "  曲率半径范围: [0.000010, 0.014151]"
echo "  阈值: 0.188199"
echo "=========================================="

python openvla/experiments/robot/libero/run_libero_retrieval_verify_mix.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-object" \
    --spec_checkpoint "$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_object_debug_ckpt/state_190" \
    --task_suite_name "libero_object" \
    --center_crop "$CENTER_CROP" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --window_size "$WINDOW_SIZE" \
    --alpha "$ALPHA" \
    --displacement_range_min 0.000098 \
    --displacement_range_max 0.116458 \
    --radius_range_min 0.000010 \
    --radius_range_max 0.014151 \
    --composite_threshold 0.188199 \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "Composite_Metric_Mix_Object" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False \
    2>&1 | tee "$LOG_DIR/libero_object.log"

echo "[2/2] libero_object completed!"
echo ""

echo "=========================================="
echo "GPU 0 experiments complete!"
echo "=========================================="
echo "Log dir: $LOG_DIR"
echo "Results dir: $SPECVLA_ROOT/openvla/specdecoding/test-speed/libero_retrieval_verify_mix/"
echo "=========================================="
