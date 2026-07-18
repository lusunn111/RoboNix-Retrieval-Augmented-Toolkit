#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_SUCCESS_LINE_RE = re.compile(r"^Success:\s+(True|False)\s*$")
_TOTAL_SUCCESS_RATE_RE = re.compile(r"^Current total success rate:\s+([0-9.]+)\s*$")


@dataclass(frozen=True)
class RunStats:
    log_dir: str
    json_file: str
    txt_file: Optional[str]
    episodes: int
    steps: int
    mean_time: float
    p50_time: float
    p95_time: float
    success_rate: Optional[float]


def _flatten_times(raw: object) -> list[float]:
    if not isinstance(raw, list):
        return []
    flat: list[float] = []
    for episode in raw:
        if not isinstance(episode, list):
            continue
        for value in episode:
            if isinstance(value, (int, float)) and not np.isnan(value):
                flat.append(float(value))
    return flat


def _infer_txt_for_json(json_path: Path) -> Optional[Path]:
    name = json_path.name
    libero_index = name.rfind("libero")
    if libero_index == -1:
        return None
    prefix = name[:libero_index]
    candidate = json_path.with_name(prefix + ".txt")
    return candidate if candidate.exists() else None


def _parse_success_rate(txt_path: Path) -> Optional[float]:
    lines = txt_path.read_text(errors="ignore").splitlines()

    for line in reversed(lines):
        match = _TOTAL_SUCCESS_RATE_RE.match(line.strip())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None

    successes = 0
    total = 0
    for line in lines:
        match = _SUCCESS_LINE_RE.match(line.strip())
        if not match:
            continue
        total += 1
        if match.group(1) == "True":
            successes += 1
    if total == 0:
        return None
    return successes / total


def _compute_percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def summarize(log_root: Path) -> list[RunStats]:
    runs: list[RunStats] = []
    for json_path in sorted(log_root.rglob("*.json")):
        raw = json.loads(json_path.read_text(errors="ignore"))
        times = _flatten_times(raw)
        if not times:
            continue

        txt_path = _infer_txt_for_json(json_path)
        success_rate = _parse_success_rate(txt_path) if txt_path is not None else None

        episodes = sum(1 for episode in raw if isinstance(episode, list))
        steps = len(times)
        runs.append(
            RunStats(
                log_dir=str(json_path.parent),
                json_file=str(json_path),
                txt_file=str(txt_path) if txt_path is not None else None,
                episodes=episodes,
                steps=steps,
                mean_time=float(np.mean(np.array(times, dtype=np.float32))),
                p50_time=_compute_percentile(times, 50),
                p95_time=_compute_percentile(times, 95),
                success_rate=success_rate,
            )
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LIBERO run speed (and success rate if available).")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("openvla/specdecoding/test-speed"),
        help="Root folder containing run logs (default: openvla/specdecoding/test-speed).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("Lab/results/libero_runs_summary.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    stats = summarize(args.log_root)
    if not stats:
        raise SystemExit(f"No usable *.json logs found under {args.log_root}")

    df = pd.DataFrame([s.__dict__ for s in stats])
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
