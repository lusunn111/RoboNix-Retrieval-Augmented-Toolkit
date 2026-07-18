#!/usr/bin/env python3
"""Aggregate LIBERO full-30 experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


EPISODE_FIELDS = [
    "method",
    "run_id",
    "task_suite_name",
    "task_id",
    "episode_idx",
    "success",
    "episode_wall_time_s",
    "infer_calls",
    "executed_action_count",
    "infer_latency_mean_ms",
    "policy_time_mean_ms",
    "serve_time_mean_ms",
    "avg_latency_per_action_ms",
    "avg_policy_time_per_action_ms",
    "avg_serve_time_per_action_ms",
    "draft_rounds",
    "full_rounds",
    "vlm_rounds",
    "env_steps_taken",
    "trace_path",
    "infer_path",
]

INFER_FIELDS = [
    "method",
    "run_id",
    "task_suite_name",
    "task_id",
    "episode_idx",
    "infer_id",
    "route_type",
    "sample_actions_ms",
    "policy_time_ms",
    "serve_time_ms",
    "client_roundtrip_ms",
    "encoder_ms",
    "draft_ms",
    "action_verify_ms",
    "full_fallback_ms",
    "total_ms",
    "rtcache_draft",
    "rtcache_runtime_feature",
    "rtcache_retrieval_ms",
    "rtcache_with_embedding_ms",
    "rtcache_top_score",
    "rtcache_selected_score",
    "rtcache_num_candidates",
    "rtcache_selected_rank",
    "rtcache_best_accept_len",
    "rtcache_rerank_verify_ms",
    "rtcache_rerank_extra_verifies",
    "rtcache_noverify",
    "rtcache_composite",
    "rtcache_composite_threshold",
    "rtcache_composite_displacement",
    "rtcache_composite_radius",
    "rtcache_composite_norm_displacement",
    "rtcache_composite_norm_radius",
    "rtcache_noverify_streak",
    "rtcache_record_index",
    "rtcache_record_task_id",
    "rtcache_trace_task_id",
]


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


def _avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def _med(values: list[float]) -> float | None:
    return median(values) if values else None


def _method_from_episode_log(root: Path, episode_log: Path) -> str:
    rel = episode_log.relative_to(root)
    return rel.parts[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    infers: list[dict[str, Any]] = []
    for episode_log in sorted(root.glob("*/*/episode_log.json")):
        method = _method_from_episode_log(root, episode_log)
        rows = json.loads(episode_log.read_text(encoding="utf-8"))
        run_dir = episode_log.parent
        for row in rows:
            episode = {"method": method, **row}
            episodes.append(episode)
            infer_rel = row.get("infer_path")
            if not infer_rel:
                continue
            infer_path = run_dir / str(infer_rel)
            if not infer_path.is_file():
                continue
            for line in infer_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                infer = json.loads(line)
                infers.append({"method": method, **infer})
    return episodes, infers


def _summarize_groups(episodes: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        groups[tuple(row.get(key) for key in group_keys)].append(row)

    out = []
    for key, rows in sorted(groups.items()):
        success = [1.0 if bool(row.get("success")) else 0.0 for row in rows]
        wall = [v for row in rows if (v := _float(row.get("episode_wall_time_s"))) is not None]
        infer = [v for row in rows if (v := _float(row.get("infer_latency_mean_ms"))) is not None]
        policy = [v for row in rows if (v := _float(row.get("policy_time_mean_ms"))) is not None]
        action = [v for row in rows if (v := _float(row.get("avg_latency_per_action_ms"))) is not None]
        record = {name: value for name, value in zip(group_keys, key, strict=True)}
        record.update(
            {
                "episodes": len(rows),
                "successes": int(sum(success)),
                "success_rate": _avg(success),
                "episode_wall_time_s_mean": _avg(wall),
                "episode_wall_time_s_median": _med(wall),
                "episode_wall_time_s_p90": _p90(wall),
                "infer_latency_mean_ms": _avg(infer),
                "policy_time_mean_ms": _avg(policy),
                "avg_latency_per_action_ms": _avg(action),
            }
        )
        out.append(record)
    return out


def _summarize_suite_macro(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        groups[(row["method"], row["task_suite_name"])].append(row)

    out = []
    for (method, suite), rows in sorted(groups.items()):
        task_rates = [v for row in rows if (v := _float(row.get("success_rate"))) is not None]
        out.append(
            {
                "method": method,
                "task_suite_name": suite,
                "tasks": len(rows),
                "episodes": sum(int(row.get("episodes", 0)) for row in rows),
                "successes": sum(int(row.get("successes", 0)) for row in rows),
                "success_rate_macro": _avg(task_rates),
                "success_rate_micro": (
                    sum(int(row.get("successes", 0)) for row in rows)
                    / max(1, sum(int(row.get("episodes", 0)) for row in rows))
                ),
                "episode_wall_time_s_mean": _avg(
                    [v for row in rows if (v := _float(row.get("episode_wall_time_s_mean"))) is not None]
                ),
                "infer_latency_mean_ms": _avg(
                    [v for row in rows if (v := _float(row.get("infer_latency_mean_ms"))) is not None]
                ),
                "avg_latency_per_action_ms": _avg(
                    [v for row in rows if (v := _float(row.get("avg_latency_per_action_ms"))) is not None]
                ),
            }
        )
    return out


def _infer_summary(infers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in infers:
        groups[(row.get("method"), row.get("task_suite_name"), row.get("route_type"))].append(row)
    for (method, suite, route), group in sorted(groups.items()):
        rows.append(
            {
                "method": method,
                "task_suite_name": suite,
                "route_type": route,
                "count": len(group),
                "sample_actions_ms_mean": _avg(
                    [v for row in group if (v := _float(row.get("sample_actions_ms"))) is not None]
                ),
                "draft_ms_mean": _avg([v for row in group if (v := _float(row.get("draft_ms"))) is not None]),
                "action_verify_ms_mean": _avg(
                    [v for row in group if (v := _float(row.get("action_verify_ms"))) is not None]
                ),
                "rtcache_retrieval_ms_mean": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_retrieval_ms"))) is not None]
                ),
                "rtcache_runtime_feature_mean": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_runtime_feature"))) is not None]
                ),
                "rtcache_with_embedding_ms_mean": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_with_embedding_ms"))) is not None]
                ),
                "rtcache_top_score_mean": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_top_score"))) is not None]
                ),
                "rtcache_selected_rank_mean": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_selected_rank"))) is not None]
                ),
                "rtcache_noverify_rate": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_noverify"))) is not None]
                ),
                "rtcache_best_accept_len_mean": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_best_accept_len"))) is not None]
                ),
                "rtcache_rerank_extra_verifies_mean": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_rerank_extra_verifies"))) is not None]
                ),
                "rtcache_rerank_verify_ms_mean": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_rerank_verify_ms"))) is not None]
                ),
                "rtcache_composite_mean": _avg(
                    [v for row in group if (v := _float(row.get("rtcache_composite"))) is not None]
                ),
            }
        )
    return rows


def _write_markdown(path: Path, suite_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LIBERO Full-30 Summary",
        "",
        "| method | suite | tasks | episodes | success_macro | success_micro | infer_ms | action_ms | wall_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in suite_rows:
        lines.append(
            "| {method} | {task_suite_name} | {tasks} | {episodes} | {success_rate_macro:.4f} | "
            "{success_rate_micro:.4f} | {infer_latency_mean_ms:.2f} | "
            "{avg_latency_per_action_ms:.2f} | {episode_wall_time_s_mean:.2f} |".format(
                **{
                    key: (0.0 if row.get(key) is None else row.get(key))
                    for key in [
                        "method",
                        "task_suite_name",
                        "tasks",
                        "episodes",
                        "success_rate_macro",
                        "success_rate_micro",
                        "infer_latency_mean_ms",
                        "avg_latency_per_action_ms",
                        "episode_wall_time_s_mean",
                    ]
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments/full_30"))
    parser.add_argument("--out", type=Path, default=Path("outputs/experiments/full_30/summary"))
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    episodes, infers = _load_rows(root)
    if not episodes:
        raise SystemExit(f"No episode_log.json files found under {root}")

    task_rows = _summarize_groups(episodes, ("method", "task_suite_name", "task_id"))
    suite_rows = _summarize_suite_macro(task_rows)
    infer_rows = _infer_summary(infers)

    _write_csv(out / "episode_results.csv", episodes, EPISODE_FIELDS)
    _write_csv(
        out / "summary_by_task.csv",
        task_rows,
        [
            "method",
            "task_suite_name",
            "task_id",
            "episodes",
            "successes",
            "success_rate",
            "episode_wall_time_s_mean",
            "episode_wall_time_s_median",
            "episode_wall_time_s_p90",
            "infer_latency_mean_ms",
            "policy_time_mean_ms",
            "avg_latency_per_action_ms",
        ],
    )
    _write_csv(
        out / "summary_by_suite.csv",
        suite_rows,
        [
            "method",
            "task_suite_name",
            "tasks",
            "episodes",
            "successes",
            "success_rate_macro",
            "success_rate_micro",
            "episode_wall_time_s_mean",
            "infer_latency_mean_ms",
            "avg_latency_per_action_ms",
        ],
    )
    _write_csv(out / "inference_timing.csv", infers, INFER_FIELDS)
    _write_csv(
        out / "inference_summary.csv",
        infer_rows,
        [
            "method",
            "task_suite_name",
            "route_type",
            "count",
            "sample_actions_ms_mean",
            "draft_ms_mean",
            "action_verify_ms_mean",
            "rtcache_retrieval_ms_mean",
            "rtcache_runtime_feature_mean",
            "rtcache_with_embedding_ms_mean",
            "rtcache_top_score_mean",
            "rtcache_selected_rank_mean",
            "rtcache_noverify_rate",
            "rtcache_best_accept_len_mean",
            "rtcache_rerank_extra_verifies_mean",
            "rtcache_rerank_verify_ms_mean",
            "rtcache_composite_mean",
        ],
    )
    _write_markdown(out / "summary.md", suite_rows)
    print(f"Wrote summaries to {out}")


if __name__ == "__main__":
    main()
