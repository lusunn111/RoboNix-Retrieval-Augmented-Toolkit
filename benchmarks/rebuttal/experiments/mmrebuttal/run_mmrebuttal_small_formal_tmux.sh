#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/MMRebuttal"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/zhihao/mmrebuttal_outputs/small_formal}"
SESSION="${SESSION:-mmrebuttal_small_formal}"
TRIALS="${TRIALS:-2}"
TASK_IDS="${TASK_IDS:-}"

mkdir -p "$OUTPUT_ROOT/logs"
rm -f "$OUTPUT_ROOT/logs/worker0.done" "$OUTPUT_ROOT/logs/worker1.done"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n worker0 \
  "cd '$PROJECT_ROOT' && OUTPUT_ROOT='$OUTPUT_ROOT' WORKER=worker0 GPU=0 TRIALS='$TRIALS' TASK_IDS='$TASK_IDS' bash experiments/mmrebuttal/run_mmrebuttal_worker.sh; exec bash"

tmux new-window -t "$SESSION" -n worker1 \
  "cd '$PROJECT_ROOT' && OUTPUT_ROOT='$OUTPUT_ROOT' WORKER=worker1 GPU=1 TRIALS='$TRIALS' TASK_IDS='$TASK_IDS' bash experiments/mmrebuttal/run_mmrebuttal_worker.sh; exec bash"

tmux new-window -t "$SESSION" -n analyze \
  "cd '$PROJECT_ROOT'; while [ ! -f '$OUTPUT_ROOT/logs/worker0.done' ] || [ ! -f '$OUTPUT_ROOT/logs/worker1.done' ]; do date; echo 'waiting for workers...'; sleep 120; done; OUTPUT_ROOT='$OUTPUT_ROOT' bash experiments/mmrebuttal/run_mmrebuttal_analysis.sh; exec bash"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Output root: $OUTPUT_ROOT"
