#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/MMRebuttal"
SPECVLA_ROOT="/path/to/SpecVLA"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/zhihao/mmrebuttal_outputs/small_formal}"

SUITE="${SUITE:-libero_goal}"
TASK_IDS="${TASK_IDS:-}"
TRIALS="${TRIALS:-2}"
GPU="${GPU:-0}"
SEED="${SEED:-7}"

MODEL_FAMILY="openvla"
CENTER_CROP="True"
TOP_K="${TOP_K:-5}"
PROB_THRESHOLD="${PROB_THRESHOLD:-0.1}"
USE_AVG_PROB="${USE_AVG_PROB:-True}"
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-9}"
BLOCK_SUM_THRESHOLD="${BLOCK_SUM_THRESHOLD:-45}"
BLOCK_MAX_THRESHOLD="${BLOCK_MAX_THRESHOLD:-25}"
ABLATION_MODE="${ABLATION_MODE:-3}"
ALPHA="${ALPHA:-0.5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"

case "$SUITE" in
  libero_goal)
    PRETRAINED_CHECKPOINT="$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-goal"
    SPEC_CHECKPOINT="$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190"
    DISPLACEMENT_MIN="0.000009"
    DISPLACEMENT_MAX="0.139051"
    RADIUS_MIN="0.000001"
    RADIUS_MAX="0.016873"
    COMPOSITE_THRESHOLD="${COMPOSITE_THRESHOLD:-0.143210}"
    ;;
  libero_spatial)
    PRETRAINED_CHECKPOINT="$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-spatial"
    SPEC_CHECKPOINT="$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_spatial_debug_ckpt/state_190"
    DISPLACEMENT_MIN="0.000027"
    DISPLACEMENT_MAX="0.128629"
    RADIUS_MIN="0.000019"
    RADIUS_MAX="0.015654"
    COMPOSITE_THRESHOLD="${COMPOSITE_THRESHOLD:-0.217119}"
    ;;
  libero_object)
    PRETRAINED_CHECKPOINT="$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-object"
    SPEC_CHECKPOINT="$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_object_debug_ckpt/state_190"
    DISPLACEMENT_MIN="0.000098"
    DISPLACEMENT_MAX="0.116458"
    RADIUS_MIN="0.000010"
    RADIUS_MAX="0.014151"
    COMPOSITE_THRESHOLD="${COMPOSITE_THRESHOLD:-0.188199}"
    ;;
  libero_10)
    PRETRAINED_CHECKPOINT="$SPECVLA_ROOT/backbone_models/openvla-7b-finetuned-libero-10"
    SPEC_CHECKPOINT="$SPECVLA_ROOT/openvla/specdecoding/train-scripts/ckpt_libero_10_debug_ckpt/state_190"
    DISPLACEMENT_MIN="0.000008"
    DISPLACEMENT_MAX="0.102298"
    RADIUS_MIN="0.000001"
    RADIUS_MAX="0.012479"
    COMPOSITE_THRESHOLD="${COMPOSITE_THRESHOLD:-0.4}"
    ;;
  *)
    echo "Unknown SUITE: $SUITE" >&2
    exit 2
    ;;
esac

RAW_DIR="$OUTPUT_ROOT/raw_runs/$SUITE"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$RAW_DIR" "$LOG_DIR"

if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)"
  conda activate specvla
  set -u
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_EGL_DEVICE_ID="$GPU"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$SPECVLA_ROOT:$SPECVLA_ROOT/openvla:$SPECVLA_ROOT/LIBERO:${PYTHONPATH:-}"
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "$SPECVLA_ROOT"

run_note="mmrebuttal_${SUITE}_trials${TRIALS}"
if [[ -n "$TASK_IDS" ]]; then
  run_note="${run_note}_tasks${TASK_IDS//,/_}"
else
  run_note="${run_note}_alltasks"
fi

cmd=(
  python openvla/experiments/robot/libero/run_libero_block_sd.py
  --model_family "$MODEL_FAMILY"
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
  --spec_checkpoint "$SPEC_CHECKPOINT"
  --task_suite_name "$SUITE"
  --center_crop "$CENTER_CROP"
  --top_k "$TOP_K"
  --prob_threshold "$PROB_THRESHOLD"
  --use_avg_prob "$USE_AVG_PROB"
  --accept_threshold "$ACCEPT_THRESHOLD"
  --block_sum_threshold "$BLOCK_SUM_THRESHOLD"
  --block_max_threshold "$BLOCK_MAX_THRESHOLD"
  --num_trials_per_task "$TRIALS"
  --run_id_note "$run_note"
  --use_spec True
  --parallel_draft False
  --use_wandb False
  --ablation_mode "$ABLATION_MODE"
  --alpha "$ALPHA"
  --seed "$SEED"
  --composite_threshold "$COMPOSITE_THRESHOLD"
  --displacement_range_min "$DISPLACEMENT_MIN"
  --displacement_range_max "$DISPLACEMENT_MAX"
  --radius_range_min "$RADIUS_MIN"
  --radius_range_max "$RADIUS_MAX"
  --save_videos "$SAVE_VIDEOS"
  --mmrebuttal_record_step_metrics True
  --mmrebuttal_output_dir "$RAW_DIR"
  --mmrebuttal_overlap_eps "${MMREBUTTAL_OVERLAP_EPS:-0.01}"
  --mmrebuttal_profile_component_times "${MMREBUTTAL_PROFILE_COMPONENT_TIMES:-False}"
)

if [[ -n "$TASK_IDS" ]]; then
  cmd+=(--task_ids "$TASK_IDS")
fi

echo "Running MMRebuttal suite"
echo "  suite=$SUITE"
echo "  task_ids=${TASK_IDS:-all}"
echo "  trials=$TRIALS"
echo "  gpu=$GPU"
echo "  output=$RAW_DIR"
echo "  log=$LOG_DIR/${SUITE}.log"

"${cmd[@]}" 2>&1 | tee "$LOG_DIR/${SUITE}.log"
