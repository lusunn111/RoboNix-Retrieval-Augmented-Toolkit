#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_TASK_RE = re.compile(r"^Task:\s+(.*)\s*$")
_SUCCESS_RE = re.compile(r"^Success:\s+(True|False)\s*$")


def _parse_success_by_task(txt_path: Path) -> dict[str, bool]:
    success_by_task: dict[str, bool] = {}
    current_task: Optional[str] = None
    for line in txt_path.read_text(errors="ignore").splitlines():
        task_match = _TASK_RE.match(line.strip())
        if task_match:
            current_task = task_match.group(1)
            continue

        success_match = _SUCCESS_RE.match(line.strip())
        if success_match and current_task is not None and current_task not in success_by_task:
            success_by_task[current_task] = success_match.group(1) == "True"
    return success_by_task


def join(per_task_csv: Path, txt_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(per_task_csv)
    success_by_task = _parse_success_by_task(txt_path)
    df["success"] = df["task_name"].map(success_by_task)
    df.to_csv(out_dir / "per_task_with_success.csv", index=False)

    labeled = df.dropna(subset=["success"]).copy()
    if labeled.empty:
        raise SystemExit(f"No success labels matched tasks in {txt_path}")

    correlation = float(labeled[["mean_accept", "success"]].corr().iloc[0, 1])
    summary = {
        "tasks": int(df.shape[0]),
        "labeled_tasks": int(labeled.shape[0]),
        "success_rate": float(labeled["success"].mean()),
        "mean_accept_success": float(labeled[labeled["success"] == True]["mean_accept"].mean()),
        "mean_accept_fail": float(labeled[labeled["success"] == False]["mean_accept"].mean()),
        "corr_mean_accept_vs_success": correlation,
    }
    (out_dir / "summary.json").write_text(pd.Series(summary).to_json(indent=2, force_ascii=False))

    plt.figure(figsize=(8, max(4, 0.25 * len(df))))
    df = df.sort_values("mean_accept")
    y = np.arange(len(df))
    colors = df["success"].map(lambda x: "#2CA02C" if x is True else "#D62728" if x is False else "#7F7F7F")
    plt.barh(y, df["mean_accept"], color=colors)
    plt.yticks(y, df["task_name"])
    plt.xlabel("mean accept length")
    plt.title("Mean accept length per task (green=success, red=fail)")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_accept_with_success.png", dpi=200)
    plt.close()

    print(f"[OK] Wrote: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Join accept-length per-task summary with success labels from txt.")
    parser.add_argument("--per-task", type=Path, required=True, help="Path to per_task_summary.csv")
    parser.add_argument("--txt", type=Path, required=True, help="Path to EVAL-*.txt")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Lab/results/_accept_success"),
        help="Output directory.",
    )
    args = parser.parse_args()
    join(args.per_task, args.txt, args.out_dir)


if __name__ == "__main__":
    main()

