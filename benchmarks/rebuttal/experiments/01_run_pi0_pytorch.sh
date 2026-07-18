#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/libero_common.sh"

PORT=${PORT:-8101}
CONFIG_NAME=${CONFIG_NAME:-pi0_libero}
PI0_TORCH_CKPT=${PI0_TORCH_CKPT:-"$ROOT_DIR/models/openpi/openpi-assets/checkpoints/pi0_libero_torch"}
RUN_NAME=${RUN_NAME:-"pi0_${SUITE}_task_${TASK}_trials_${TRIALS}"}
VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-"$OUTPUT_ROOT/pi0_pytorch"}
LOG_DIR=${LOG_DIR:-"$VIDEO_OUT_PATH/logs"}
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server_${RUN_NAME}.log"
CLIENT_LOG="$LOG_DIR/client_${RUN_NAME}.log"

if [[ ! -f "$PI0_TORCH_CKPT/model.safetensors" ]]; then
  echo "Missing PyTorch checkpoint: $PI0_TORCH_CKPT/model.safetensors" >&2
  echo "Run experiments/00_prepare_pi0_torch.sh first." >&2
  exit 1
fi

ensure_transformers_replace "$FLASH_DIR"

cd "$FLASH_DIR"
uv run scripts/serve_policy.py \
  --port "$PORT" \
  --disable-torch-compile \
  policy:checkpoint \
  --policy.config="$CONFIG_NAME" \
  --policy.dir="$PI0_TORCH_CKPT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'cleanup_server "$SERVER_PID"' EXIT

wait_for_server "$SERVER_PID" "$SERVER_LOG" "Creating server"
run_libero_client "$PORT" "$VIDEO_OUT_PATH" "$RUN_NAME" "$CLIENT_LOG"

echo "pi0 PyTorch experiment finished:"
echo "  output: $VIDEO_OUT_PATH/$RUN_NAME"
echo "  server log: $SERVER_LOG"
echo "  client log: $CLIENT_LOG"
