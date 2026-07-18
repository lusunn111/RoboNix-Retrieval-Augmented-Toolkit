#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
MAX_EXEC_STEPS="${MAX_EXEC_STEPS:-12}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-80}"
MAX_CHUNKS_PER_TASK="${MAX_CHUNKS_PER_TASK:-40}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-40}"
MAX_CHUNKS_PER_EPISODE="${MAX_CHUNKS_PER_EPISODE:-0}"
RECORD_FORMAT="${RECORD_FORMAT:-episode_npz}"
RECORD_SUCCESSFUL_EPISODES_ONLY="${RECORD_SUCCESSFUL_EPISODES_ONLY:-1}"
DATASET_OUT="${DATASET_OUT:-../dataset/flash_episodes}"
VIDEO_OUT="${VIDEO_OUT:-data/flash_episode_collect}"
RUN_PREFIX="${RUN_PREFIX:-flash_episodes}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
INITIAL_STATE_JITTER_STD="${INITIAL_STATE_JITTER_STD:-0.0005}"
BASE_TRITON_PATH="${BASE_TRITON_PATH:-../models/realtime-vla-flash/triton/pi0_libero_base}"
DRAFT_TRITON_PATH="${DRAFT_TRITON_PATH:-../models/realtime-vla-flash/triton/draft_libero_goal}"
SUITES="${SUITES:-libero_goal libero_spatial libero_object libero_10}"
TASK_SPEC="${TASK_SPEC:-}"

cd "$(dirname "$0")/../.."

mkdir -p "$VIDEO_OUT"

port_open() {
  python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

wait_port() {
  for _ in $(seq 1 120); do
    if port_open; then
      return 0
    fi
    sleep 1
  done
  return 1
}

SERVER_PID=""
if port_open; then
  echo "Using existing FLASH server on port ${PORT}"
else
  echo "Starting FLASH full-teacher server on port ${PORT}"
  uv run scripts/spec/spec_serve_policy.py \
    --config pi0_libero \
    --base-triton-path "$BASE_TRITON_PATH" \
    --draft-triton-path "$DRAFT_TRITON_PATH" \
    --task-suite-name libero_goal \
    --backend triton \
    --pytorch-device "$DEVICE" \
    --max-exec-steps "$MAX_EXEC_STEPS" \
    --force-full-each-round \
    --port "$PORT" \
    > "${VIDEO_OUT}/server_${PORT}.log" 2>&1 &
  SERVER_PID="$!"
  trap 'if [[ -n "${SERVER_PID}" ]]; then kill "${SERVER_PID}" 2>/dev/null || true; fi' EXIT
  wait_port
fi

source examples/libero/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/src:$PWD/third_party/libero"
export MUJOCO_GL
TASK_ARGS=()
if [[ -n "$TASK_SPEC" ]]; then
  TASK_ARGS=(--task "$TASK_SPEC")
fi
SUCCESS_ONLY_ARGS=()
if [[ "$RECORD_SUCCESSFUL_EPISODES_ONLY" == "0" ]]; then
  SUCCESS_ONLY_ARGS=(--no-record-successful-episodes-only)
fi

for suite in $SUITES; do
  echo "Collecting ${suite}"
  python scripts/spec/spec_client_libero.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --task-suite-name "$suite" \
    "${TASK_ARGS[@]}" \
    --num-trials-per-task "$TRIALS_PER_TASK" \
    --replan-steps "$MAX_EXEC_STEPS" \
    --video-out-path "$VIDEO_OUT" \
    --run-name "${RUN_PREFIX}_${suite}" \
    --no-save-videos \
    --record-flash-dataset \
    --record-flash-dataset-format "$RECORD_FORMAT" \
    --flash-dataset-out-path "$DATASET_OUT" \
    --record-max-chunks-per-task "$MAX_CHUNKS_PER_TASK" \
    --record-max-episodes-per-task "$MAX_EPISODES_PER_TASK" \
    --record-max-chunks-per-episode "$MAX_CHUNKS_PER_EPISODE" \
    "${SUCCESS_ONLY_ARGS[@]}" \
    --initial-state-jitter-std "$INITIAL_STATE_JITTER_STD"
done

deactivate
