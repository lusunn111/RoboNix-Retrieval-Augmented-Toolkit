#!/bin/bash
# GPU 1: libero_10
# Mix View Retrieval Verify - Pure AR mode

set -e

SPECVLA_ROOT="/path/to/SpecVLA"
cd "$SPECVLA_ROOT"

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1

MODEL_FAMILY="openvla"
CENTER_CROP="True"
ACCEPT_THRESHOLD=9
DB_MODEL_RATIO="0 1"  # Pure AR mode
NUM_TRIALS=10
RUN_ID_NOTE="Retrieval_Verify_MixView"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiments/logs/retrieval_verify_mixview_gpu1_$TIMESTAMP"
mkdir -p "$LOG_DIR"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate specvla || exit 1
fi

echo "=========================================="
echo "GPU 1: libero_10"
echo "=========================================="
echo "Log dir: $LOG_DIR"
echo ""

# Run libero_10
echo "[1/1] Running libero_10..."
python openvla/experiments/robot/libero/run_libero_10_Retrieval_Verify.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-10" \
    --spec_checkpoint "$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_10_debug_ckpt/state_190" \
    --task_suite_name "libero_10" \
    --center_crop "$CENTER_CROP" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --db_model_ratio "$DB_MODEL_RATIO" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False \
    2>&1 | tee "$LOG_DIR/libero_10_mixview.log"

echo "=========================================="
echo "GPU 1 experiment complete!"
echo "Log dir: $LOG_DIR"
echo "=========================================="
