#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/path/to/MMRebuttal}
SESSION_NAME=${SESSION_NAME:-libero_db_tau_sweep}
JOBS_FILE=${JOBS_FILE:-"$ROOT_DIR/experiments/db_tau_sweep_jobs.tsv"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT_DIR/outputs/experiments/db_tau_sweep"}
TASK=${TASK:-0}
TRIALS=${TRIALS:-10}
SEED=${SEED:-7777}
REPLAN_STEPS=${REPLAN_STEPS:-12}
SAVE_VIDEOS=${SAVE_VIDEOS:-0}
INITIAL_STATE_JITTER_STD=${INITIAL_STATE_JITTER_STD:-0.0005}
INITIAL_STATE_JITTER_SEED_OFFSET=${INITIAL_STATE_JITTER_SEED_OFFSET:-900000}
MUJOCO_GL=${MUJOCO_GL:-egl}
T_LIST=${T_LIST:-"0.10 0.05"}
DIST_DIMS=${DIST_DIMS:-7}
GRIPPER_VERIFY=${GRIPPER_VERIFY:-1}
GRIPPER_POST_VERIFY=${GRIPPER_POST_VERIFY:-1}
GRIPPER_FULL_WINDOW=${GRIPPER_FULL_WINDOW:-1}
FORCE_SESSION=${FORCE_SESSION:-0}
FORCE_JOB=${FORCE_JOB:-0}

sanitize() {
  echo "$1" | tr ',/: .-' '______'
}

expected_episode_count() {
  python3 - "$TASK" "$TRIALS" <<'PY'
import sys
task_spec = sys.argv[1]
trials = int(sys.argv[2])
tasks = set()
for token in task_spec.split(","):
    token = token.strip()
    if not token:
        continue
    if "-" in token:
        lo, hi = map(int, token.split("-", 1))
        tasks.update(range(lo, hi + 1))
    else:
        tasks.add(int(token))
print(len(tasks) * trials)
PY
}

assert_port_free() {
  local port="$1"
  if ss -ltn | awk '{print $4}' | grep -Eq "(:|\\])${port}$"; then
    echo "Port still listening: $port" >&2
    ss -ltnp | grep -E "(:|\\])${port}\\b" >&2 || true
    return 1
  fi
}

validate_job() {
  local tau_dir="$1"
  local run_name="$2"
  local expected="$3"
  local episode_log="$tau_dir/flash_db_draft_pytorch/$run_name/episode_log.json"
  local server_log="$tau_dir/flash_db_draft_pytorch/logs/server_${run_name}.log"
  local client_log="$tau_dir/flash_db_draft_pytorch/logs/client_${run_name}.log"
  python3 - "$episode_log" "$expected" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"missing episode_log: {path}")
rows = json.loads(path.read_text())
if len(rows) != expected:
    raise SystemExit(f"episode count mismatch for {path}: got {len(rows)}, expected {expected}")
print(f"validated {path}: episodes={len(rows)} successes={sum(bool(r.get('success')) for r in rows)}")
PY
  test -f "$server_log"
  test -f "$client_log"
}

job_complete() {
  local tau_dir="$1"
  local run_name="$2"
  local expected="$3"
  validate_job "$tau_dir" "$run_name" "$expected" >/dev/null 2>&1
}

run_worker() {
  local worker_id="${WORKER_ID:?missing WORKER_ID}"
  local gpu_filter="${GPU_FILTER:?missing GPU_FILTER}"
  local expected safe_task worker_log

  mkdir -p "$OUTPUT_ROOT/logs"
  worker_log="$OUTPUT_ROOT/logs/${worker_id}.log"
  exec > >(tee -a "$worker_log") 2>&1

  safe_task=$(sanitize "$TASK")
  expected=$(expected_episode_count)
  echo "[$(date --iso-8601=seconds)] $worker_id: running DB tau sweep jobs assigned to $gpu_filter"
  echo "output_root=$OUTPUT_ROOT task=$TASK trials=$TRIALS seed=$SEED t_list=$T_LIST"

  while IFS=$'\t' read -r tau suite gpu port; do
    [[ -z "${tau:-}" || "$tau" == \#* ]] && continue
    [[ "$gpu" == "$gpu_filter" ]] || continue

    local safe_tau tau_dir run_name
    safe_tau=$(sanitize "$tau")
    tau_dir="$OUTPUT_ROOT/tau_${safe_tau}"
    run_name="dbtau_${safe_tau}_${suite}_task${safe_task}_trials${TRIALS}_seed${SEED}"

    if [[ "$FORCE_JOB" != "1" ]] && job_complete "$tau_dir" "$run_name" "$expected"; then
      echo "[$(date --iso-8601=seconds)] $worker_id: skip completed tau=$tau suite=$suite"
      continue
    fi

    echo "[$(date --iso-8601=seconds)] $worker_id: start tau=$tau suite=$suite gpu=$gpu port=$port run=$run_name"
    assert_port_free "$port"
    env \
      OUTPUT_ROOT="$tau_dir" \
      SUITE="$suite" \
      TASK="$TASK" \
      TRIALS="$TRIALS" \
      GPU="$gpu" \
      PORT="$port" \
      SEED="$SEED" \
      REPLAN_STEPS="$REPLAN_STEPS" \
      SAVE_VIDEOS="$SAVE_VIDEOS" \
      INITIAL_STATE_JITTER_STD="$INITIAL_STATE_JITTER_STD" \
      INITIAL_STATE_JITTER_SEED_OFFSET="$INITIAL_STATE_JITTER_SEED_OFFSET" \
      MUJOCO_GL="$MUJOCO_GL" \
      RUN_NAME="$run_name" \
      TAU_RADIUS="$tau" \
      T_LIST="$T_LIST" \
      DIST_DIMS="$DIST_DIMS" \
      GRIPPER_VERIFY="$GRIPPER_VERIFY" \
      GRIPPER_POST_VERIFY="$GRIPPER_POST_VERIFY" \
      GRIPPER_FULL_WINDOW="$GRIPPER_FULL_WINDOW" \
      "$ROOT_DIR/experiments/03_run_flash_db_draft_pytorch.sh"

    validate_job "$tau_dir" "$run_name" "$expected"
    assert_port_free "$port"
    echo "[$(date --iso-8601=seconds)] $worker_id: done tau=$tau suite=$suite"
  done < "$JOBS_FILE"

  echo "[$(date --iso-8601=seconds)] $worker_id: all assigned jobs finished"
}

start_tmux() {
  command -v tmux >/dev/null
  test -f "$JOBS_FILE"
  mkdir -p "$OUTPUT_ROOT/logs"

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    if [[ "$FORCE_SESSION" == "1" ]]; then
      tmux kill-session -t "$SESSION_NAME"
    else
      echo "tmux session already exists: $SESSION_NAME" >&2
      echo "Attach with: tmux attach -t $SESSION_NAME" >&2
      exit 1
    fi
  fi

  local common_env
  common_env="ROOT_DIR='$ROOT_DIR' SESSION_NAME='$SESSION_NAME' JOBS_FILE='$JOBS_FILE' OUTPUT_ROOT='$OUTPUT_ROOT' TASK='$TASK' TRIALS='$TRIALS' SEED='$SEED' REPLAN_STEPS='$REPLAN_STEPS' SAVE_VIDEOS='$SAVE_VIDEOS' INITIAL_STATE_JITTER_STD='$INITIAL_STATE_JITTER_STD' INITIAL_STATE_JITTER_SEED_OFFSET='$INITIAL_STATE_JITTER_SEED_OFFSET' MUJOCO_GL='$MUJOCO_GL' T_LIST='$T_LIST' DIST_DIMS='$DIST_DIMS' GRIPPER_VERIFY='$GRIPPER_VERIFY' GRIPPER_POST_VERIFY='$GRIPPER_POST_VERIFY' GRIPPER_FULL_WINDOW='$GRIPPER_FULL_WINDOW' FORCE_JOB='$FORCE_JOB'"

  tmux new-session -d -s "$SESSION_NAME" -n worker0 \
    "cd '$ROOT_DIR' && $common_env WORKER_ID='worker0' GPU_FILTER='cuda:0' '$ROOT_DIR/experiments/run_db_tau_sweep_tmux.sh' --worker"
  tmux new-window -t "$SESSION_NAME" -n worker1 \
    "cd '$ROOT_DIR' && $common_env WORKER_ID='worker1' GPU_FILTER='cuda:1' '$ROOT_DIR/experiments/run_db_tau_sweep_tmux.sh' --worker"

  echo "Started tmux session: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  echo "Worker logs: $OUTPUT_ROOT/logs/worker0.log and worker1.log"
}

case "${1:-}" in
  --worker) run_worker ;;
  *) start_tmux ;;
esac
