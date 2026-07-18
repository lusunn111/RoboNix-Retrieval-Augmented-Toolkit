#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/path/to/MMRebuttal}
FLASH_DIR="$ROOT_DIR/realtime-vla-flash"
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT_DIR/outputs/experiments"}
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_ROOT"
fi
OUTPUT_ROOT=$(realpath -m "$OUTPUT_ROOT")

SUITE=${SUITE:-libero_goal}
TASK=${TASK:-0}
TRIALS=${TRIALS:-1}
REPLAN_STEPS=${REPLAN_STEPS:-12}
MUJOCO_GL=${MUJOCO_GL:-egl}
SAVE_VIDEOS=${SAVE_VIDEOS:-0}
SEED=${SEED:-7777}
INITIAL_STATE_JITTER_STD=${INITIAL_STATE_JITTER_STD:-0.0005}
INITIAL_STATE_JITTER_SEED_OFFSET=${INITIAL_STATE_JITTER_SEED_OFFSET:-900000}

GPU=${GPU:-cuda:0}
GPU_INDEX=${GPU#cuda:}
GPU_INDEX=${GPU_INDEX#gpu:}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}
PYTORCH_DEVICE=${PYTORCH_DEVICE:-cuda:0}

export CUDA_VISIBLE_DEVICES
export OPENPI_ENABLE_TORCH_COMPILE=0
export OPENPI_DISABLE_TORCH_COMPILE=1
export SPEC_TRITON_INPUT_PREPARE_FAST=0
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_ROOT"

wait_for_server() {
  local pid="$1"
  local log_path="$2"
  local ready_pattern="${3:-Creating .*server|Serving pure database policy}"
  for _ in $(seq 1 "${SERVER_START_TIMEOUT:-240}"); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "Server exited before it was ready. Log: $log_path" >&2
      tail -n 160 "$log_path" >&2 || true
      return 1
    fi
    if rg -q "$ready_pattern" "$log_path"; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for server. Log: $log_path" >&2
  tail -n 160 "$log_path" >&2 || true
  return 1
}

run_libero_client() {
  local port="$1"
  local video_out_path="$2"
  local run_name="$3"
  local client_log="$4"

  cd "$FLASH_DIR"
  source examples/libero/.venv/bin/activate
  export PYTHONPATH="$PWD:$PWD/src:$PWD/third_party/libero"
  export MUJOCO_GL

  local video_args=()
  if [[ "$SAVE_VIDEOS" == "0" || "$SAVE_VIDEOS" == "false" ]]; then
    video_args+=(--no-save-videos)
  fi

  python scripts/spec/spec_client_libero.py \
    --host 127.0.0.1 \
    --port "$port" \
    --task-suite-name "$SUITE" \
    --task "$TASK" \
    --num-trials-per-task "$TRIALS" \
    --replan-steps "$REPLAN_STEPS" \
    --video-out-path "$video_out_path" \
    --run-name "$run_name" \
    --seed "$SEED" \
    --initial-state-jitter-std "$INITIAL_STATE_JITTER_STD" \
    --initial-state-jitter-seed-offset "$INITIAL_STATE_JITTER_SEED_OFFSET" \
    "${video_args[@]}" \
    >"$client_log" 2>&1
}

cleanup_server() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

ensure_transformers_replace() {
  local repo_dir="$1"
  local patch_dir="$repo_dir/src/openpi/models_pytorch/transformers_replace"
  if [[ ! -d "$patch_dir" ]]; then
    echo "Missing transformers_replace patch directory: $patch_dir" >&2
    return 1
  fi

  local transformers_dir
  transformers_dir=$(
    cd "$repo_dir"
    uv run python - <<'PY'
from pathlib import Path
import transformers
print(Path(transformers.__file__).resolve().parent)
PY
  )
  if [[ ! -d "$transformers_dir" ]]; then
    echo "Unable to locate transformers package directory for $repo_dir" >&2
    return 1
  fi
  cp -a "$patch_dir"/. "$transformers_dir"/
}
