#!/bin/bash
# GPU 0: libero_spatial + libero_object (串行)
# Mix View Retrieval Verify - Pure AR mode

set -e

SPECVLA_ROOT="/path/to/SpecVLA"
cd "$SPECVLA_ROOT"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

MODEL_FAMILY="openvla"
CENTER_CROP="True"
ACCEPT_THRESHOLD=9
DB_MODEL_RATIO="0 1"  # Pure AR mode
NUM_TRIALS=10
RUN_ID_NOTE="Retrieval_Verify_MixView"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiments/logs/retrieval_verify_mixview_gpu0_$TIMESTAMP"
mkdir -p "$LOG_DIR"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate specvla || exit 1
fi

echo "=========================================="
echo "GPU 0: libero_spatial + libero_object"
echo "=========================================="
echo "Log dir: $LOG_DIR"
echo ""

# Run libero_spatial
echo "[1/2] Running libero_spatial..."
python openvla/experiments/robot/libero/run_libero_spatial_Retrieval_Verify.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-spatial" \
    --spec_checkpoint "$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_spatial_debug_ckpt/state_190" \
    --task_suite_name "libero_spatial" \
    --center_crop "$CENTER_CROP" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --db_model_ratio "$DB_MODEL_RATIO" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False \
    2>&1 | tee "$LOG_DIR/libero_spatial_mixview.log"

# Run libero_object
echo "[2/2] Running libero_object..."
python openvla/experiments/robot/libero/run_libero_object_Retrieval_Verify.py \
    --model_family "$MODEL_FAMILY" \
    --pretrained_checkpoint "$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-object" \
    --spec_checkpoint "$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_object_debug_ckpt/state_190" \
    --task_suite_name "libero_object" \
    --center_crop "$CENTER_CROP" \
    --accept_threshold "$ACCEPT_THRESHOLD" \
    --db_model_ratio "$DB_MODEL_RATIO" \
    --num_trials_per_task "$NUM_TRIALS" \
    --run_id_note "$RUN_ID_NOTE" \
    --use_spec True \
    --parallel_draft False \
    --use_wandb False \
    2>&1 | tee "$LOG_DIR/libero_object_mixview.log"

echo "=========================================="
echo "GPU 0 experiments complete!"
echo "Log dir: $LOG_DIR"
echo "=========================================="
