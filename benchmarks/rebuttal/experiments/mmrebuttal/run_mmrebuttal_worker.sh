#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/MMRebuttal"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/zhihao/mmrebuttal_outputs/small_formal}"
WORKER="${WORKER:-worker0}"
GPU="${GPU:-0}"
TRIALS="${TRIALS:-2}"
TASK_IDS="${TASK_IDS:-}"

case "$WORKER" in
  worker0)
    SUITES=("libero_goal" "libero_object")
    ;;
  worker1)
    SUITES=("libero_spatial" "libero_10")
    ;;
  *)
    echo "Unknown WORKER: $WORKER" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT/logs"
cd "$PROJECT_ROOT"

echo "[$WORKER] output=$OUTPUT_ROOT gpu=$GPU trials=$TRIALS task_ids=${TASK_IDS:-all}"

for suite in "${SUITES[@]}"; do
  echo "[$WORKER] starting $suite"
  SUITE="$suite" \
  TASK_IDS="$TASK_IDS" \
  TRIALS="$TRIALS" \
  GPU="$GPU" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  SAVE_VIDEOS="${SAVE_VIDEOS:-False}" \
  bash experiments/mmrebuttal/run_mmrebuttal_suite.sh
  echo "[$WORKER] finished $suite"
done

touch "$OUTPUT_ROOT/logs/${WORKER}.done"
echo "[$WORKER] done"

