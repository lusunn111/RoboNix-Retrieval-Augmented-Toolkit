#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/libero_common.sh"

PORT=${PORT:-8103}
CONFIG_NAME=${CONFIG_NAME:-pi0_libero}
PI0_TORCH_CKPT=${PI0_TORCH_CKPT:-"$ROOT_DIR/models/openpi/openpi-assets/checkpoints/pi0_libero_torch"}
DRAFT_CHECKPOINT=${DRAFT_CHECKPOINT:-"$ROOT_DIR/models/realtime-vla-flash/draft_${SUITE}.pt"}
INDEX_DIR=${INDEX_DIR:-"$ROOT_DIR/database/flash_index_pytorch"}
RTCACHE_TOP_K=${RTCACHE_TOP_K:-${TOP_K:-1}}
RTCACHE_RERANK_MIN_ACCEPT_LEN=${RTCACHE_RERANK_MIN_ACCEPT_LEN:-8}
RTCACHE_NOVERIFY_POLICY=${RTCACHE_NOVERIFY_POLICY:-off}
RTCACHE_NOVERIFY_MAX_CONSECUTIVE=${RTCACHE_NOVERIFY_MAX_CONSECUTIVE:-2}
RTCACHE_COMPOSITE_WINDOW_SIZE=${RTCACHE_COMPOSITE_WINDOW_SIZE:-5}
RTCACHE_COMPOSITE_ALPHA=${RTCACHE_COMPOSITE_ALPHA:-0.5}
TAU_RADIUS=${TAU_RADIUS:-0.3}
T_LIST=${T_LIST:-"0.10 0.05"}
DIST_DIMS=${DIST_DIMS:-7}
GRIPPER_VERIFY=${GRIPPER_VERIFY:-1}
GRIPPER_POST_VERIFY=${GRIPPER_POST_VERIFY:-1}
GRIPPER_FULL_WINDOW=${GRIPPER_FULL_WINDOW:-1}
RUN_NAME=${RUN_NAME:-"flash_db_${SUITE}_task_${TASK}_trials_${TRIALS}"}
VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-"$OUTPUT_ROOT/flash_db_draft_pytorch"}
LOG_DIR=${LOG_DIR:-"$VIDEO_OUT_PATH/logs"}
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server_${RUN_NAME}.log"
CLIENT_LOG="$LOG_DIR/client_${RUN_NAME}.log"

if [[ ! -f "$PI0_TORCH_CKPT/model.safetensors" ]]; then
  echo "Missing PyTorch checkpoint: $PI0_TORCH_CKPT/model.safetensors" >&2
  echo "Run experiments/00_prepare_pi0_torch.sh first." >&2
  exit 1
fi

if [[ ! -f "$DRAFT_CHECKPOINT" ]]; then
  echo "Missing draft checkpoint: $DRAFT_CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$INDEX_DIR/manifest.json" ]]; then
  echo "Missing PyTorch retrieval index: $INDEX_DIR/manifest.json" >&2
  echo "Run experiments/00_build_flash_index_pytorch.sh first." >&2
  exit 1
fi

ensure_transformers_replace "$FLASH_DIR"

cd "$FLASH_DIR"
read -r -a T_LIST_ARGS <<<"$T_LIST"
GRIPPER_VERIFY_ARG=--enable-gripper-verify
if [[ "$GRIPPER_VERIFY" == "0" || "$GRIPPER_VERIFY" == "false" ]]; then
  GRIPPER_VERIFY_ARG=--no-enable-gripper-verify
fi
GRIPPER_POST_VERIFY_ARG=--enable-gripper-post-verify
if [[ "$GRIPPER_POST_VERIFY" == "0" || "$GRIPPER_POST_VERIFY" == "false" ]]; then
  GRIPPER_POST_VERIFY_ARG=--no-enable-gripper-post-verify
fi
uv run scripts/spec/spec_serve_policy.py \
  --config "$CONFIG_NAME" \
  --checkpoint-dir "$PI0_TORCH_CKPT" \
  --draft-checkpoint "$DRAFT_CHECKPOINT" \
  --task-suite-name "$SUITE" \
  --backend compiled \
  --pytorch-device "$PYTORCH_DEVICE" \
  --max-exec-steps "$REPLAN_STEPS" \
  --t-list "${T_LIST_ARGS[@]}" \
  --tau-radius "$TAU_RADIUS" \
  --dist-dims "$DIST_DIMS" \
  "$GRIPPER_VERIFY_ARG" \
  "$GRIPPER_POST_VERIFY_ARG" \
  --gripper-full-window "$GRIPPER_FULL_WINDOW" \
  --port "$PORT" \
  --disable-torch-compile \
  --rtcache-draft \
  --rtcache-index-dir "$INDEX_DIR" \
  --rtcache-top-k "$RTCACHE_TOP_K" \
  --rtcache-device "$PYTORCH_DEVICE" \
  --rtcache-pytorch-checkpoint-dir "$PI0_TORCH_CKPT" \
  --rtcache-suite-name "$SUITE" \
  --rtcache-rerank-min-accept-len "$RTCACHE_RERANK_MIN_ACCEPT_LEN" \
  --rtcache-noverify-policy "$RTCACHE_NOVERIFY_POLICY" \
  --rtcache-noverify-max-consecutive "$RTCACHE_NOVERIFY_MAX_CONSECUTIVE" \
  --rtcache-composite-window-size "$RTCACHE_COMPOSITE_WINDOW_SIZE" \
  --rtcache-composite-alpha "$RTCACHE_COMPOSITE_ALPHA" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'cleanup_server "$SERVER_PID"' EXIT

wait_for_server "$SERVER_PID" "$SERVER_LOG"
run_libero_client "$PORT" "$VIDEO_OUT_PATH" "$RUN_NAME" "$CLIENT_LOG"

echo "FLASH database-draft PyTorch experiment finished:"
echo "  output: $VIDEO_OUT_PATH/$RUN_NAME"
echo "  server log: $SERVER_LOG"
echo "  client log: $CLIENT_LOG"
