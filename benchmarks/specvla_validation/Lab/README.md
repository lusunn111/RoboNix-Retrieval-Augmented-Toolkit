# Lab

这里用于集中存放：**idea 构思文档**、**实验脚本**、以及你自己跑出来的 **结果/图表**（避免散落在工程主代码里）。

## 目录结构

- `Lab/idea/`：论文/idea 叙事与实验设计
- `Lab/scripts/`：分析脚本（读取 `exp/` 或 `openvla/specdecoding/test-speed/` 的日志/npz/json）
- `Lab/results/`：你跑脚本生成的 csv/png 等（建议按日期或 run_id 再建子目录）

## 环境约定（按你当前习惯）

- VLA/仿真/SpecVLA：`conda activate specvla`（跑 LIBERO 时通常使用 `CUDA_VISIBLE_DEVICES=1`、`MUJOCO_EGL_DEVICE_ID=1`）
- DB/数据处理相关：`conda activate rt-mzh`（你已启动 Qdrant + embedding + retrieval）

## 快速入口

- 叙事与实验路线：`Lab/idea/retrieval_sd_memory_reasoning_boundary.md`
- 分析 DB-as-draft 的 accept length（从 `.npz` 里重算并画图）：`Lab/scripts/analyze_db_draft_acceptance.py`
- 汇总 `openvla/specdecoding/test-speed/` 下各 run 的速度/成功率：`Lab/scripts/summarize_libero_runs.py`
- 合并 accept length 与 success（按 task 对齐）：`Lab/scripts/join_acceptance_with_success.py`
- DB-as-draft 汇总表（accept + speed + success）：`Lab/scripts/summarize_db_draft_runs.py`

## 常用命令（示例）

```bash
# 1) 分析 accept length（会在 Lab/results/<npz_stem>/ 下生成 csv + png）
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate specvla
python Lab/scripts/analyze_db_draft_acceptance.py \
  --npz exp/accept_length_data_EVAL-libero_goal-openvla-2025_12_24-17_52_18--DB_Tracking.npz

# 2) 汇总 test-speed 下的所有 run（生成一张 csv 表）
python Lab/scripts/summarize_libero_runs.py \
  --log-root openvla/specdecoding/test-speed \
  --out-csv Lab/results/libero_runs_summary.csv

# 3) 生成 DB-as-draft 汇总表（把 accept + speed + success 合到一张表）
python Lab/scripts/summarize_db_draft_runs.py \
  --results-root Lab/results \
  --runs-csv Lab/results/libero_runs_summary.csv \
  --out-dir Lab/results
```
