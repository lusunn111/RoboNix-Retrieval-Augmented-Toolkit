#!/usr/bin/env python3
"""Summarize AL and rollout steps from LIBERO episode logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


SUITE_LABELS = {
    "libero_goal": "LIBERO-Goal",
    "libero_object": "LIBERO-Object",
    "libero_spatial": "LIBERO-Spatial",
    "libero_10": "LIBERO-Long",
}

METHOD_LABELS = {
    "pi0_pytorch": "pi0",
    "flash_pytorch": "FLASH",
    "flash_db_draft_pytorch": "Ours",
}


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = int(math.ceil(0.9 * len(values))) - 1
    return values[max(0, min(idx, len(values) - 1))]


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _load_episode_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_log in sorted(root.glob("*/*/episode_log.json")):
        method = episode_log.relative_to(root).parts[0]
        run_rows = json.loads(episode_log.read_text(encoding="utf-8"))
        for row in run_rows:
            rows.append({"method": method, **row})
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("method")), str(row.get("task_suite_name")))].append(row)

    out: list[dict[str, Any]] = []
    for (method, suite), group in sorted(groups.items()):
        successes = [1.0 if row.get("success") else 0.0 for row in group]
        executed_steps = [v for row in group if (v := _float(row.get("executed_action_count"))) is not None]
        env_steps = [v for row in group if (v := _float(row.get("env_steps_taken"))) is not None]
        infer_calls = [v for row in group if (v := _float(row.get("infer_calls"))) is not None]
        al_values = [v for row in group if (v := _float(row.get("accepted_action_len_mean"))) is not None]
        al_mean = None if method == "pi0_pytorch" else (mean(al_values) if al_values else None)
        out.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "suite": suite,
                "suite_label": SUITE_LABELS.get(suite, suite),
                "episodes": len(group),
                "successes": int(sum(successes)),
                "success_rate": mean(successes) if successes else None,
                "al_mean": al_mean,
                "executed_steps_mean": mean(executed_steps) if executed_steps else None,
                "executed_steps_median": median(executed_steps) if executed_steps else None,
                "executed_steps_p90": _p90(executed_steps),
                "env_steps_taken_mean": mean(env_steps) if env_steps else None,
                "infer_calls_mean": mean(infer_calls) if infer_calls else None,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "method_label",
        "suite",
        "suite_label",
        "episodes",
        "successes",
        "success_rate",
        "al_mean",
        "executed_steps_mean",
        "executed_steps_median",
        "executed_steps_p90",
        "env_steps_taken_mean",
        "infer_calls_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    order_methods = ["pi0_pytorch", "flash_pytorch", "flash_db_draft_pytorch"]
    order_suites = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]
    by_key = {(row["method"], row["suite"]): row for row in rows}
    lines = [
        "# AL / Steps Summary",
        "",
        "AL is not applicable to pure pi0 because it has no speculative draft verification.",
        "Steps uses `executed_action_count`, i.e. environment actions after the initial wait steps.",
        "",
        "| Env. | Method | Episodes | SR | AL | Steps | Env Steps | Infer Calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for suite in order_suites:
        for method in order_methods:
            row = by_key.get((method, suite))
            if row is None:
                continue
            al = "-" if method == "pi0_pytorch" else _fmt(row["al_mean"], 2)
            lines.append(
                "| {suite} | {method} | {episodes} | {sr} | {al} | {steps} | {env_steps} | {infer_calls} |".format(
                    suite=row["suite_label"],
                    method=row["method_label"],
                    episodes=row["episodes"],
                    sr=_fmt(row["success_rate"], 3),
                    al=al,
                    steps=_fmt(row["executed_steps_mean"], 1),
                    env_steps=_fmt(row["env_steps_taken_mean"], 1),
                    infer_calls=_fmt(row["infer_calls_mean"], 1),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_episode_rows(args.root)
    if not rows:
        raise SystemExit(f"No episode_log.json files found under {args.root}")
    summary = _summarize(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "al_steps_by_suite.csv", summary)
    _write_md(args.out / "al_steps_summary.md", summary)
    print(f"Wrote {args.out / 'al_steps_summary.md'}")


if __name__ == "__main__":
    main()
