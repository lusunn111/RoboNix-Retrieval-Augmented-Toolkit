# Non-Triton PyTorch LIBERO Experiments

These scripts run the three LIBERO smoke experiments without the FLASH Triton
backend or Triton artifact paths. Outputs are written under:

```bash
/path/to/MMRebuttal/outputs/experiments
```

Large folders are symlinked from the project root:

```text
models   -> /data/Zhihao/models
dataset  -> /data/Zhihao/dataset
database -> /data/Zhihao/database
```

## Prepare

Convert the JAX `pi0_libero` checkpoint to PyTorch:

```bash
./experiments/00_prepare_pi0_torch.sh
```

Expected checkpoint:

```text
models/openpi/openpi-assets/checkpoints/pi0_libero_torch/model.safetensors
```

Rebuild the local retrieval index from the existing FLASH episode dataset using
the PyTorch pi0 vision tower:

```bash
FORCE_INDEX=1 ./experiments/00_build_flash_index_pytorch.sh
```

Expected index:

```text
database/flash_index_pytorch
```

The index uses 4096-d vectors:

```text
third-person pi0 projected image token mean 2048
wrist pi0 projected image token mean        2048
```

## Run

Pure pi0 PyTorch baseline:

```bash
./experiments/01_run_pi0_pytorch.sh
```

FLASH PyTorch path with the learned draft checkpoint:

```bash
./experiments/02_run_flash_pytorch.sh
```

FLASH PyTorch path with database retrieval replacing the draft proposal:

```bash
./experiments/03_run_flash_db_draft_pytorch.sh
```

All scripts default to:

```text
SUITE=libero_goal
TASK=0
TRIALS=1
GPU=cuda:0
REPLAN_STEPS=12
MUJOCO_GL=egl
```

Override them with environment variables, for example:

```bash
SUITE=libero_goal TASK=0-3 TRIALS=5 GPU=cuda:1 ./experiments/03_run_flash_db_draft_pytorch.sh
```

The scripts set:

```text
OPENPI_ENABLE_TORCH_COMPILE=0
OPENPI_DISABLE_TORCH_COMPILE=1
SPEC_TRITON_INPUT_PREPARE_FAST=0
```

They intentionally do not pass Triton backend or Triton artifact arguments.

## Current Smoke Result

Default `libero_goal task 0 x 1 episode` was verified on 2026-06-01:

```text
pi0_pytorch             success=true  infer=11  mean_infer_ms~=201.32
flash_pytorch           success=true  infer=10  mean_infer_ms~=86.17
flash_db_draft_pytorch  success=true  infer=10  mean_infer_ms~=65.59
```

The database-draft run writes `rtcache_*` fields into `infer.jsonl`, including
`rtcache_draft`, `rtcache_top_score`, retrieval latency, matched record IDs, and
`rtcache_trace_task_id`. Its server log also prints `rtcache_draft=1` for draft
rounds while keeping FLASH verify timing.

## Full-30 tmux Run

The full experiment runs all 12 method/suite jobs in a tmux session with two
workers:

```bash
./experiments/run_full_30_tmux.sh
tmux attach -t libero_full30
```

Defaults:

```text
TASK=0-9
TRIALS=30
SEED=7777
SAVE_VIDEOS=0
INITIAL_STATE_JITTER_STD=0.0005
INITIAL_STATE_JITTER_SEED_OFFSET=900000
OUTPUT_ROOT=outputs/experiments/full_30
```

Use the same runner for the pilot:

```bash
SESSION_NAME=libero_pilot2 \
TASK=0 \
TRIALS=2 \
OUTPUT_ROOT=/path/to/MMRebuttal/outputs/experiments/pilot_2 \
./experiments/run_full_30_tmux.sh
```

After the run:

```bash
cd /path/to/MMRebuttal
uv run python experiments/analyze_full_30.py \
  --root outputs/experiments/full_30 \
  --out outputs/experiments/full_30/summary
```
