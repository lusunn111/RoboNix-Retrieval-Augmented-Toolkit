#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/libero_common.sh"

PORT=${PORT:-8102}
CONFIG_NAME=${CONFIG_NAME:-pi0_libero}
PI0_TORCH_CKPT=${PI0_TORCH_CKPT:-"$ROOT_DIR/models/openpi/openpi-assets/checkpoints/pi0_libero_torch"}
DRAFT_CHECKPOINT=${DRAFT_CHECKPOINT:-"$ROOT_DIR/models/realtime-vla-flash/draft_${SUITE}.pt"}
RUN_NAME=${RUN_NAME:-"flash_${SUITE}_task_${TASK}_trials_${TRIALS}"}
VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-"$OUTPUT_ROOT/flash_pytorch"}
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

ensure_transformers_replace "$FLASH_DIR"

cd "$FLASH_DIR"
uv run scripts/spec/spec_serve_policy.py \
  --config "$CONFIG_NAME" \
  --checkpoint-dir "$PI0_TORCH_CKPT" \
  --draft-checkpoint "$DRAFT_CHECKPOINT" \
  --task-suite-name "$SUITE" \
  --backend compiled \
  --pytorch-device "$PYTORCH_DEVICE" \
  --max-exec-steps "$REPLAN_STEPS" \
  --port "$PORT" \
  --disable-torch-compile \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'cleanup_server "$SERVER_PID"' EXIT

wait_for_server "$SERVER_PID" "$SERVER_LOG"
run_libero_client "$PORT" "$VIDEO_OUT_PATH" "$RUN_NAME" "$CLIENT_LOG"

echo "FLASH PyTorch experiment finished:"
echo "  output: $VIDEO_OUT_PATH/$RUN_NAME"
echo "  server log: $SERVER_LOG"
echo "  client log: $CLIENT_LOG"
