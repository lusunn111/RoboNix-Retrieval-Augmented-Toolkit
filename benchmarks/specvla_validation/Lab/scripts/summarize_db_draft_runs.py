#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize(results_root: Path, runs_csv: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    accept_dirs = sorted(results_root.glob("accept_length_data_*/overall_summary.json"))
    if not accept_dirs:
        raise SystemExit(f"No overall_summary.json found under {results_root}")

    runs_df = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for overall_path in accept_dirs:
        overall = json.loads(overall_path.read_text())
        run_dir = overall_path.parent.name
        run_id = run_dir.replace("accept_length_data_", "")

        match = None
        if not runs_df.empty and "txt_file" in runs_df.columns:
            candidates = runs_df[runs_df["txt_file"].fillna("").str.endswith(f"{run_id}.txt")]
            if not candidates.empty:
                match = candidates.iloc[0].to_dict()

        row = {
            "run_id": run_id,
            "accept_threshold": overall.get("accept_threshold"),
            "steps": overall.get("steps"),
            "mean_accept": overall.get("mean_accept"),
            "p50_accept": overall.get("p50_accept"),
            "p95_accept": overall.get("p95_accept"),
            "pct_accept_gt0": overall.get("pct_accept_gt0"),
            "mean_l2": overall.get("mean_l2"),
            "mean_time": match.get("mean_time") if match else None,
            "p95_time": match.get("p95_time") if match else None,
            "success_rate": match.get("success_rate") if match else None,
            "txt_file": match.get("txt_file") if match else None,
            "time_json": match.get("json_file") if match else None,
            "accept_results_dir": str(overall_path.parent),
            "npz": overall.get("npz"),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["accept_threshold", "run_id"])
    out_csv = out_dir / "db_draft_runs_summary.csv"
    df.to_csv(out_csv, index=False)

    if "accept_threshold" in df.columns and df["accept_threshold"].notna().any():
        plt.figure(figsize=(6, 4))
        plt.scatter(df["accept_threshold"], df["mean_accept"], s=60)
        plt.xlabel("accept_threshold")
        plt.ylabel("mean_accept")
        plt.title("DB-as-draft: accept_threshold vs mean_accept")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "accept_threshold_vs_mean_accept.png", dpi=200)
        plt.close()

        if df["success_rate"].notna().any():
            plt.figure(figsize=(6, 4))
            plt.scatter(df["mean_accept"], df["success_rate"], s=60)
            plt.xlabel("mean_accept")
            plt.ylabel("success_rate")
            plt.title("DB-as-draft: mean_accept vs success_rate")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(out_dir / "mean_accept_vs_success.png", dpi=200)
            plt.close()

    print(f"[OK] Wrote: {out_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DB-as-draft runs (accept + speed + success).")
    parser.add_argument("--results-root", type=Path, default=Path("Lab/results"), help="Root of accept_length_data_* dirs")
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("Lab/results/libero_runs_summary.csv"),
        help="CSV from summarize_libero_runs.py (for speed/success join).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("Lab/results"), help="Output directory")
    args = parser.parse_args()
    summarize(args.results_root, args.runs_csv, args.out_dir)


if __name__ == "__main__":
    main()

