#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/MMRebuttal"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/zhihao/mmrebuttal_outputs/small_formal}"

if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)"
  conda activate specvla
  set -u
fi

cd "$PROJECT_ROOT"
python3 experiments/mmrebuttal/analyze_small_formal.py --root "$OUTPUT_ROOT"

