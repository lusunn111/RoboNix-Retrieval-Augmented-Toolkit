#!/usr/bin/env python3
"""Analyze small-formal MMRebuttal runs.

The input is the original HeiSD JSON output from run_libero_block_sd.py.
Outputs are written under /data/zhihao by default and intentionally exclude
environment/video/wait timing from the timing summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("/data/zhihao/mmrebuttal_outputs/small_formal")
SUITE_RE = re.compile(r"EVAL-(libero_(?:goal|spatial|object|10))-")

METRIC_FIELDS = [
    "raw_radius",
    "raw_displacement",
    "norm_radius",
    "norm_displacement",
    "composite_metric",
    "accepted_length",
    "top1_score",
    "top1_action_l2_error",
    "top1_action_pos_l2_error",
    "top1_token_diff_sum",
    "top1_token_diff_max",
]

SIGNAL_METRICS = [
    "raw_radius",
    "raw_displacement",
    "norm_radius",
    "norm_displacement",
    "composite_metric",
]

TRAJECTORY_FIELDS = [
    "top1_retrieved_action_trajectory_len",
    "trajectory_overlap_horizon",
    "trajectory_mean_deviation",
    "trajectory_max_deviation",
    "trajectory_endpoint_error",
    "trajectory_action_l2_mean_deviation",
    "trajectory_action_l2_endpoint_error",
    "trajectory_overlap_ratio",
    "trajectory_overlap_eps",
]

TIME_FIELDS = [
    "embedding_time",
    "retrieval_time",
    "retrieval_total_time",
    "metric_time",
    "generation_time",
    "model_retrieval_time",
    "profile_vit_time",
    "profile_llm_time",
    "profile_draft_model_time",
    "profile_verification_with_metric_time",
    "profile_ar_generation_time",
    "profile_sd_generation_time",
]

EEF_FIELDS = ["eef_x", "eef_y", "eef_z"]

BASE_FIELDS = [
    "run_file",
    "suite",
    "task_id",
    "task_description",
    "episode_idx",
    "episode_success",
    "step_idx",
    "mode",
    "overlap_label",
    "overlap_label_source",
    "trajectory_space",
    "retrieval_success",
    "num_candidates",
    "decision_reason",
    "use_retrieval_strategy",
    "use_ar_for_insufficient",
    "metric_history_length",
    "accepted_tokens",
    "sd_accept_length",
    "sd_new_token",
    "sd_num_iterations",
    "num_ar_blocks",
    "retrieval_consecutive_db_count",
]


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float:
    arr = np.array([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.mean(arr)) if arr.size else math.nan


def std(values: list[float]) -> float:
    arr = np.array([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.std(arr)) if arr.size else math.nan


def percentile(values: list[float], q: float) -> float:
    arr = np.array([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.percentile(arr, q)) if arr.size else math.nan


def finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [as_float(row.get(field)) for row in rows if math.isfinite(as_float(row.get(field)))]


def format_float(value: float) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def infer_suite(path: Path, step: dict[str, Any]) -> str:
    suite = step.get("task_suite_name")
    if suite:
        return str(suite)
    match = SUITE_RE.search(path.name)
    if match:
        return match.group(1)
    return path.parent.name


def accepted_length_overlap_label(row: dict[str, Any]) -> str:
    accepted = as_float(row.get("accepted_length"))
    if math.isfinite(accepted):
        if accepted >= 5:
            return "overlap"
        if accepted <= 2:
            return "non_overlap"
    return "neutral"


def has_trajectory_overlap(row: dict[str, Any]) -> bool:
    horizon = as_float(row.get("trajectory_overlap_horizon"))
    mean_dev = as_float(row.get("trajectory_mean_deviation"))
    endpoint = as_float(row.get("trajectory_endpoint_error"))
    return math.isfinite(horizon) and horizon >= 1 and math.isfinite(mean_dev) and math.isfinite(endpoint)


def assign_overlap_labels(rows: list[dict[str, Any]]) -> None:
    """Prefer geometric DB-vs-executed trajectory overlap labels.

    Old runs did not record trajectory diagnostics. For those, fall back to the
    accepted-length proxy and mark the source explicitly.
    """
    if not any(has_trajectory_overlap(row) for row in rows):
        for row in rows:
            row["overlap_label"] = accepted_length_overlap_label(row)
            row["overlap_label_source"] = "accepted_length_proxy_no_trajectory"
        return

    suites = sorted({row.get("suite") for row in rows})
    for suite in suites:
        suite_rows = [row for row in rows if row.get("suite") == suite]
        trajectory_rows = [row for row in suite_rows if has_trajectory_overlap(row)]
        if len(trajectory_rows) < 4:
            for row in suite_rows:
                row["overlap_label"] = "neutral"
                row["overlap_label_source"] = "insufficient_trajectory_samples"
            continue

        mean_devs = [as_float(row.get("trajectory_mean_deviation")) for row in trajectory_rows]
        endpoints = [as_float(row.get("trajectory_endpoint_error")) for row in trajectory_rows]
        mean_q25 = percentile(mean_devs, 25)
        mean_q75 = percentile(mean_devs, 75)
        endpoint_q50 = percentile(endpoints, 50)
        endpoint_q75 = percentile(endpoints, 75)

        for row in suite_rows:
            if not has_trajectory_overlap(row):
                row["overlap_label"] = "neutral"
                row["overlap_label_source"] = "missing_trajectory_overlap"
                continue

            mean_dev = as_float(row.get("trajectory_mean_deviation"))
            endpoint = as_float(row.get("trajectory_endpoint_error"))
            if mean_dev <= mean_q25 and endpoint <= endpoint_q50:
                label = "overlap"
            elif mean_dev >= mean_q75 or endpoint >= endpoint_q75:
                label = "non_overlap"
            else:
                label = "neutral"
            row["overlap_label"] = label
            row["overlap_label_source"] = "trajectory_deviation_quantiles"


def load_step_records(root: Path) -> list[dict[str, Any]]:
    raw_dir = root / "raw_runs"
    records: list[dict[str, Any]] = []
    json_paths = sorted(p for p in raw_dir.rglob("*_block_sd.json") if "_profile" not in p.name)

    for json_path in json_paths:
        with json_path.open("r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON: {json_path}: {exc}")
                continue

        if not isinstance(data, list):
            continue

        for task_item in data:
            task_id = task_item.get("task_id")
            task_description = task_item.get("task_description")
            for episode in task_item.get("episodes", []):
                episode_idx = episode.get("episode_idx")
                episode_success = bool(episode.get("success"))
                for step in episode.get("steps", []):
                    suite = infer_suite(json_path, step)
                    row: dict[str, Any] = {
                        "run_file": str(json_path),
                        "suite": suite,
                        "task_id": step.get("task_id", task_id),
                        "task_description": task_description,
                        "episode_idx": step.get("episode_idx", episode_idx),
                        "episode_success": episode_success,
                        "step_idx": step.get("step"),
                        "mode": step.get("mode", ""),
                        "trajectory_space": step.get("trajectory_space", ""),
                        "retrieval_success": step.get("retrieval_success"),
                        "num_candidates": step.get("num_candidates"),
                        "decision_reason": step.get("decision_reason"),
                        "use_retrieval_strategy": step.get("use_retrieval_strategy"),
                        "use_ar_for_insufficient": step.get("use_ar_for_insufficient"),
                        "metric_history_length": step.get("metric_history_length"),
                        "accepted_tokens": step.get("accepted_tokens"),
                        "sd_accept_length": step.get("sd_accept_length"),
                        "sd_new_token": step.get("sd_new_token"),
                        "sd_num_iterations": step.get("sd_num_iterations"),
                        "num_ar_blocks": step.get("num_ar_blocks"),
                        "retrieval_consecutive_db_count": step.get("retrieval_consecutive_db_count"),
                    }

                    eef_position = step.get("eef_position")
                    if isinstance(eef_position, list) and len(eef_position) >= 3:
                        row["eef_x"] = eef_position[0]
                        row["eef_y"] = eef_position[1]
                        row["eef_z"] = eef_position[2]
                    else:
                        row["eef_x"] = None
                        row["eef_y"] = None
                        row["eef_z"] = None

                    for field in METRIC_FIELDS:
                        row[field] = step.get(field)
                    for field in TRAJECTORY_FIELDS:
                        row[field] = step.get(field)
                    for field in TIME_FIELDS[:-1]:
                        row[field] = step.get(field)

                    row["model_retrieval_time"] = (
                        as_float(row.get("embedding_time"))
                        + as_float(row.get("retrieval_time"))
                        + as_float(row.get("generation_time"))
                    )
                    row["overlap_label"] = step.get("trajectory_overlap_label", "neutral")
                    row["overlap_label_source"] = "unassigned"
                    records.append(row)

    assign_overlap_labels(records)
    return records


def write_records(root: Path, rows: list[dict[str, Any]]) -> None:
    jsonl_path = root / "step_records.jsonl"
    csv_path = root / "step_records.csv"
    fields = BASE_FIELDS + METRIC_FIELDS + TRAJECTORY_FIELDS + TIME_FIELDS + EEF_FIELDS

    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cohen_d(a: list[float], b: list[float]) -> float:
    a_arr = np.array([v for v in a if math.isfinite(v)], dtype=float)
    b_arr = np.array([v for v in b if math.isfinite(v)], dtype=float)
    if a_arr.size < 2 or b_arr.size < 2:
        return math.nan
    pooled = math.sqrt(((a_arr.size - 1) * np.var(a_arr) + (b_arr.size - 1) * np.var(b_arr)) / (a_arr.size + b_arr.size - 2))
    if pooled == 0:
        return math.nan
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled)


def rank_auc(labels: list[int], scores: list[float]) -> float:
    pairs = [(score, label) for label, score in zip(labels, scores) if math.isfinite(score)]
    n_pos = sum(1 for _, label in pairs if label == 1)
    n_neg = sum(1 for _, label in pairs if label == 0)
    if n_pos == 0 or n_neg == 0:
        return math.nan

    pairs.sort(key=lambda item: item[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    rank_sum_pos = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def write_overlap_distribution_plots(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        (plot_dir / "plot_error.txt").write_text(str(exc))
        return

    plot_metrics = [
        ("raw_radius", "Radius"),
        ("raw_displacement", "Displacement"),
        ("composite_metric", "Composite"),
    ]
    labels = [("overlap", "overlap"), ("non_overlap", "non-overlap")]

    for metric, title in plot_metrics:
        metric_rows = [
            row for row in rows
            if row.get("overlap_label") in {"overlap", "non_overlap"}
            and math.isfinite(as_float(row.get(metric)))
        ]
        if not metric_rows:
            continue

        values_by_label = [
            np.array([as_float(row.get(metric)) for row in metric_rows if row.get("overlap_label") == label], dtype=float)
            for label, _ in labels
        ]
        if any(values.size == 0 for values in values_by_label):
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].boxplot(values_by_label, labels=[display for _, display in labels], showfliers=False)
        axes[0].set_ylabel(title)
        axes[0].set_title(f"{title}: overlap vs non-overlap")
        axes[0].grid(True, axis="y", alpha=0.25)

        all_values = np.concatenate(values_by_label)
        lo = float(np.percentile(all_values, 1))
        hi = float(np.percentile(all_values, 99))
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        bins = np.linspace(lo, hi, 45)
        for values, (_, display) in zip(values_by_label, labels):
            axes[1].hist(values, bins=bins, alpha=0.55, density=True, label=display)
        axes[1].set_xlabel(title)
        axes[1].set_ylabel("density")
        axes[1].set_title(f"{title} distribution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_distribution_by_overlap.png", dpi=180)
        plt.close(fig)

        suites = sorted({row.get("suite") for row in metric_rows})
        fig, axes = plt.subplots(len(suites), 1, figsize=(8, max(2.2 * len(suites), 3.2)), sharex=False)
        if len(suites) == 1:
            axes = [axes]
        for ax, suite in zip(axes, suites):
            suite_values = [
                np.array([
                    as_float(row.get(metric)) for row in metric_rows
                    if row.get("suite") == suite and row.get("overlap_label") == label
                ], dtype=float)
                for label, _ in labels
            ]
            if any(values.size == 0 for values in suite_values):
                ax.text(0.5, 0.5, f"{suite}: insufficient samples", ha="center", va="center")
                ax.set_axis_off()
                continue
            ax.boxplot(suite_values, labels=[display for _, display in labels], showfliers=False, vert=False)
            ax.set_title(suite)
            ax.grid(True, axis="x", alpha=0.25)
        fig.suptitle(f"{title} distribution by suite", y=0.995)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_distribution_by_suite_overlap.png", dpi=180)
        plt.close(fig)

    scatter_rows = [
        row for row in rows
        if row.get("overlap_label") in {"overlap", "non_overlap"}
        and math.isfinite(as_float(row.get("raw_radius")))
        and math.isfinite(as_float(row.get("raw_displacement")))
    ]
    if scatter_rows:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        colors = {"overlap": "#1f77b4", "non_overlap": "#d62728"}
        displays = {"overlap": "overlap", "non_overlap": "non-overlap"}
        for label in ["overlap", "non_overlap"]:
            subset = [row for row in scatter_rows if row.get("overlap_label") == label]
            ax.scatter(
                [as_float(row.get("raw_radius")) for row in subset],
                [as_float(row.get("raw_displacement")) for row in subset],
                s=9,
                alpha=0.35,
                label=displays[label],
                color=colors[label],
            )
        ax.set_xlabel("Radius")
        ax.set_ylabel("Displacement")
        ax.set_title("Radius vs displacement by trajectory overlap")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "radius_displacement_scatter_by_overlap.png", dpi=180)
        plt.close(fig)


def optional_stats():
    try:
        from scipy import stats  # type: ignore

        return stats
    except Exception:
        return None


def analyze_overlap(root: Path, rows: list[dict[str, Any]]) -> None:
    out_dir = root / "overlap_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_metrics = METRIC_FIELDS + TRAJECTORY_FIELDS

    summary_fields = [
        "suite",
        "label",
        "n_steps",
        "success_rate",
    ]
    for metric in summary_metrics:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_median", f"{metric}_p25", f"{metric}_p75"])

    suites = sorted({row["suite"] for row in rows})
    labels = ["overlap", "non_overlap", "neutral"]
    with (out_dir / "summary_by_suite.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for suite in suites:
            suite_rows = [row for row in rows if row["suite"] == suite]
            for label in labels:
                subset = [row for row in suite_rows if row["overlap_label"] == label]
                if not subset:
                    continue
                out = {
                    "suite": suite,
                    "label": label,
                    "n_steps": len(subset),
                    "success_rate": mean([1.0 if row.get("episode_success") else 0.0 for row in subset]),
                }
                for metric in summary_metrics:
                    vals = finite_values(subset, metric)
                    out[f"{metric}_mean"] = mean(vals)
                    out[f"{metric}_std"] = std(vals)
                    out[f"{metric}_median"] = percentile(vals, 50)
                    out[f"{metric}_p25"] = percentile(vals, 25)
                    out[f"{metric}_p75"] = percentile(vals, 75)
                writer.writerow(out)

    task_fields = [
        "suite",
        "task_id",
        "label",
        "n_steps",
        "accepted_length_mean",
        "top1_action_l2_error_mean",
        "trajectory_mean_deviation_mean",
        "trajectory_endpoint_error_mean",
        "raw_radius_mean",
        "raw_displacement_mean",
        "composite_metric_mean",
    ]
    task_groups: dict[tuple[Any, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_groups[(row.get("suite"), row.get("task_id"), row.get("overlap_label"))].append(row)
    with (out_dir / "summary_by_task.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=task_fields)
        writer.writeheader()
        for (suite, task_id, label), subset in sorted(task_groups.items()):
            writer.writerow({
                "suite": suite,
                "task_id": task_id,
                "label": label,
                "n_steps": len(subset),
                "accepted_length_mean": mean(finite_values(subset, "accepted_length")),
                "top1_action_l2_error_mean": mean(finite_values(subset, "top1_action_l2_error")),
                "trajectory_mean_deviation_mean": mean(finite_values(subset, "trajectory_mean_deviation")),
                "trajectory_endpoint_error_mean": mean(finite_values(subset, "trajectory_endpoint_error")),
                "raw_radius_mean": mean(finite_values(subset, "raw_radius")),
                "raw_displacement_mean": mean(finite_values(subset, "raw_displacement")),
                "composite_metric_mean": mean(finite_values(subset, "composite_metric")),
            })

    stats_mod = optional_stats()
    stat_fields = [
        "suite",
        "metric",
        "n_overlap",
        "n_non_overlap",
        "overlap_mean",
        "non_overlap_mean",
        "cohen_d",
        "mannwhitney_p",
        "ks_p",
        "spearman_target",
        "spearman_r",
        "spearman_p",
        "auc_overlap_high",
        "auc_separable",
    ]
    with (out_dir / "stat_tests.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stat_fields)
        writer.writeheader()
        for suite in suites:
            suite_rows = [row for row in rows if row["suite"] == suite]
            overlap = [row for row in suite_rows if row["overlap_label"] == "overlap"]
            non_overlap = [row for row in suite_rows if row["overlap_label"] == "non_overlap"]
            labels_binary = [1 if row["overlap_label"] == "overlap" else 0 for row in overlap + non_overlap]

            for metric in SIGNAL_METRICS:
                a = finite_values(overlap, metric)
                b = finite_values(non_overlap, metric)
                scores = [as_float(row.get(metric)) for row in overlap + non_overlap]

                mann_p = math.nan
                ks_p = math.nan
                spearman_r = math.nan
                spearman_p = math.nan
                spearman_target = "trajectory_mean_deviation"
                if stats_mod is not None and len(a) >= 2 and len(b) >= 2:
                    try:
                        mann_p = float(stats_mod.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                    except Exception:
                        pass
                    try:
                        ks_p = float(stats_mod.ks_2samp(a, b).pvalue)
                    except Exception:
                        pass
                if stats_mod is not None:
                    target_field = "trajectory_mean_deviation"
                    paired = [
                        (as_float(row.get(metric)), as_float(row.get(target_field)))
                        for row in suite_rows
                    ]
                    paired = [(x, y) for x, y in paired if math.isfinite(x) and math.isfinite(y)]
                    if len(paired) < 3:
                        target_field = "accepted_length"
                        spearman_target = "accepted_length_proxy"
                        paired = [
                            (as_float(row.get(metric)), as_float(row.get(target_field)))
                            for row in suite_rows
                        ]
                        paired = [(x, y) for x, y in paired if math.isfinite(x) and math.isfinite(y)]
                    if len(paired) >= 3:
                        try:
                            res = stats_mod.spearmanr([x for x, _ in paired], [y for _, y in paired])
                            spearman_r, spearman_p = float(res.correlation), float(res.pvalue)
                        except Exception:
                            pass

                auc = rank_auc(labels_binary, scores)
                writer.writerow({
                    "suite": suite,
                    "metric": metric,
                    "n_overlap": len(a),
                    "n_non_overlap": len(b),
                    "overlap_mean": mean(a),
                    "non_overlap_mean": mean(b),
                    "cohen_d": cohen_d(a, b),
                    "mannwhitney_p": mann_p,
                    "ks_p": ks_p,
                    "spearman_target": spearman_target,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "auc_overlap_high": auc,
                    "auc_separable": max(auc, 1.0 - auc) if math.isfinite(auc) else math.nan,
                })

    write_overlap_distribution_plots(out_dir, rows)


def summarize_timing(root: Path, rows: list[dict[str, Any]]) -> None:
    out_dir = root / "timing_breakdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    timing_record_fields = [
        "suite",
        "task_id",
        "episode_idx",
        "step_idx",
        "mode",
        "embedding_time_ms",
        "qdrant_search_time_ms",
        "retrieval_total_ms",
        "metric_time_ms",
        "generation_time_ms",
        "model_retrieval_time_ms",
        "profile_vit_time_ms",
        "profile_llm_time_ms",
        "profile_draft_model_time_ms",
        "profile_verification_with_metric_time_ms",
        "profile_ar_generation_time_ms",
        "profile_sd_generation_time_ms",
    ]
    with (out_dir / "timing_records.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=timing_record_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "suite": row.get("suite"),
                "task_id": row.get("task_id"),
                "episode_idx": row.get("episode_idx"),
                "step_idx": row.get("step_idx"),
                "mode": row.get("mode"),
                "embedding_time_ms": as_float(row.get("embedding_time")) * 1000.0,
                "qdrant_search_time_ms": as_float(row.get("retrieval_time")) * 1000.0,
                "retrieval_total_ms": as_float(row.get("retrieval_total_time")) * 1000.0,
                "metric_time_ms": as_float(row.get("metric_time")) * 1000.0,
                "generation_time_ms": as_float(row.get("generation_time")) * 1000.0,
                "model_retrieval_time_ms": as_float(row.get("model_retrieval_time")) * 1000.0,
                "profile_vit_time_ms": as_float(row.get("profile_vit_time")) * 1000.0,
                "profile_llm_time_ms": as_float(row.get("profile_llm_time")) * 1000.0,
                "profile_draft_model_time_ms": as_float(row.get("profile_draft_model_time")) * 1000.0,
                "profile_verification_with_metric_time_ms": as_float(row.get("profile_verification_with_metric_time")) * 1000.0,
                "profile_ar_generation_time_ms": as_float(row.get("profile_ar_generation_time")) * 1000.0,
                "profile_sd_generation_time_ms": as_float(row.get("profile_sd_generation_time")) * 1000.0,
            })

    def write_grouped(path: Path, group_keys: list[str]) -> None:
        fields = group_keys + ["n_steps"]
        for field in TIME_FIELDS:
            fields.extend([f"{field}_mean_ms", f"{field}_median_ms", f"{field}_p90_ms"])

        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row.get(key, "") for key in group_keys)].append(row)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for key, subset in sorted(groups.items()):
                out = dict(zip(group_keys, key))
                out["n_steps"] = len(subset)
                for field in TIME_FIELDS:
                    vals_ms = [as_float(row.get(field)) * 1000.0 for row in subset]
                    out[f"{field}_mean_ms"] = mean(vals_ms)
                    out[f"{field}_median_ms"] = percentile(vals_ms, 50)
                    out[f"{field}_p90_ms"] = percentile(vals_ms, 90)
                writer.writerow(out)

    write_grouped(out_dir / "summary_by_mode.csv", ["suite", "mode"])
    write_grouped(out_dir / "summary_by_suite.csv", ["suite"])

    try:
        import matplotlib.pyplot as plt  # type: ignore

        suites = sorted({row["suite"] for row in rows})
        embedding = []
        qdrant = []
        generation = []
        for suite in suites:
            subset = [row for row in rows if row["suite"] == suite]
            embedding.append(mean([as_float(row.get("embedding_time")) * 1000.0 for row in subset]))
            qdrant.append(mean([as_float(row.get("retrieval_time")) * 1000.0 for row in subset]))
            generation.append(mean([as_float(row.get("generation_time")) * 1000.0 for row in subset]))

        x = np.arange(len(suites))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x, embedding, label="embedding")
        ax.bar(x, qdrant, bottom=embedding, label="qdrant")
        bottom = np.array(embedding) + np.array(qdrant)
        ax.bar(x, generation, bottom=bottom, label="generation/verify")
        ax.set_xticks(x)
        ax.set_xticklabels(suites, rotation=20, ha="right")
        ax.set_ylabel("mean step time (ms)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "timing_stacked_bar.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (out_dir / "plot_error.txt").write_text(str(exc))


def write_case_studies(root: Path, rows: list[dict[str, Any]], max_cases: int = 4) -> None:
    out_dir = root / "case_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes: dict[tuple[str, Any, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["suite"], row.get("task_id"), row.get("episode_idx"), row.get("run_file"))
        episodes[key].append(row)

    candidates = []
    for key, subset in episodes.items():
        if len(subset) < 3:
            continue
        modes = Counter(row.get("mode", "") for row in subset)
        avg_accept = mean([as_float(row.get("accepted_length")) for row in subset])
        avg_error = mean([as_float(row.get("top1_action_l2_error")) for row in subset])
        avg_comp = mean([as_float(row.get("composite_metric")) for row in subset])
        candidates.append({
            "key": key,
            "rows": sorted(subset, key=lambda row: as_int(row.get("step_idx")) or 0),
            "avg_accept": avg_accept,
            "avg_error": avg_error,
            "avg_comp": avg_comp,
            "unique_modes": len(modes),
            "modes": modes,
            "success": bool(subset[0].get("episode_success")),
        })

    picks = []
    if candidates:
        picks.append(("high_overlap", max(candidates, key=lambda c: (-math.inf if not math.isfinite(c["avg_accept"]) else c["avg_accept"]))))
        picks.append(("low_overlap", min(candidates, key=lambda c: (math.inf if not math.isfinite(c["avg_accept"]) else c["avg_accept"]))))
        picks.append(("high_error", max(candidates, key=lambda c: (-math.inf if not math.isfinite(c["avg_error"]) else c["avg_error"]))))
        picks.append(("mixed_modes", max(candidates, key=lambda c: c["unique_modes"])))

    seen = set()
    selected = []
    for tag, item in picks:
        if item["key"] in seen:
            continue
        seen.add(item["key"])
        selected.append((tag, item))
        if len(selected) >= max_cases:
            break

    for tag, item in selected:
        suite, task_id, episode_idx, _ = item["key"]
        case_dir = out_dir / f"{tag}_{suite}_task{task_id}_ep{episode_idx}"
        case_dir.mkdir(parents=True, exist_ok=True)

        timeline_fields = [
            "step_idx",
            "mode",
            "overlap_label",
            "composite_metric",
            "raw_radius",
            "raw_displacement",
            "norm_radius",
            "norm_displacement",
            "accepted_length",
            "top1_score",
        "top1_action_l2_error",
        "trajectory_mean_deviation",
        "trajectory_endpoint_error",
        "trajectory_overlap_ratio",
        "eef_x",
        "eef_y",
        "eef_z",
            "embedding_time",
            "retrieval_time",
            "generation_time",
            "model_retrieval_time",
        ]
        with (case_dir / "timeline.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=timeline_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(item["rows"])

        try:
            import matplotlib.pyplot as plt  # type: ignore

            steps = [as_float(row.get("step_idx")) for row in item["rows"]]
            composite = [as_float(row.get("composite_metric")) for row in item["rows"]]
            accepted = [as_float(row.get("accepted_length")) for row in item["rows"]]
            error = [as_float(row.get("top1_action_l2_error")) for row in item["rows"]]
            traj_dev = [as_float(row.get("trajectory_mean_deviation")) for row in item["rows"]]

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(steps, composite, label="composite", color="#1f77b4")
            ax1.plot(steps, accepted, label="accepted_length", color="#2ca02c")
            ax1.set_xlabel("step")
            ax1.set_ylabel("metric / accepted length")
            ax2 = ax1.twinx()
            ax2.plot(steps, error, label="top1 action L2", color="#d62728", alpha=0.7)
            ax2.set_ylabel("top1 action L2")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="best")
            ax1.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(case_dir / "timeline.png", dpi=180)
            plt.close(fig)

            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            axes[0].plot(steps, [as_float(row.get("raw_radius")) for row in item["rows"]], label="raw_radius")
            axes[0].plot(steps, [as_float(row.get("raw_displacement")) for row in item["rows"]], label="raw_displacement")
            axes[0].plot(steps, composite, label="composite")
            axes[0].set_ylabel("metric")
            axes[0].legend(loc="best")
            axes[0].grid(True, alpha=0.25)

            axes[1].plot(steps, accepted, color="#2ca02c", label="accepted_length")
            axes[1].set_ylabel("accepted")
            axes[1].set_ylim(bottom=0)
            axes[1].legend(loc="best")
            axes[1].grid(True, alpha=0.25)

            mode_names = sorted({row.get("mode", "") for row in item["rows"]})
            mode_to_id = {mode: idx for idx, mode in enumerate(mode_names)}
            axes[2].step(steps, [mode_to_id.get(row.get("mode", ""), -1) for row in item["rows"]], where="post")
            axes[2].set_yticks(list(mode_to_id.values()))
            axes[2].set_yticklabels(mode_names, fontsize=7)
            axes[2].set_xlabel("step")
            axes[2].set_ylabel("mode")
            axes[2].grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(case_dir / "metric_accept_mode_timeline.png", dpi=180)
            plt.close(fig)

            for name, values, ylabel, color in [
                ("metric_timeline.png", composite, "composite", "#1f77b4"),
                ("accept_timeline.png", accepted, "accepted_length", "#2ca02c"),
                ("action_error_timeline.png", error, "top1 action L2", "#d62728"),
                ("trajectory_deviation_timeline.png", traj_dev, "trajectory mean deviation", "#9467bd"),
            ]:
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.plot(steps, values, color=color)
                ax.set_xlabel("step")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(case_dir / name, dpi=180)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 3))
            mode_names = sorted({row.get("mode", "") for row in item["rows"]})
            mode_to_id = {mode: idx for idx, mode in enumerate(mode_names)}
            ax.step(steps, [mode_to_id.get(row.get("mode", ""), -1) for row in item["rows"]], where="post")
            ax.set_yticks(list(mode_to_id.values()))
            ax.set_yticklabels(mode_names, fontsize=7)
            ax.set_xlabel("step")
            ax.set_ylabel("mode")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(case_dir / "mode_timeline.png", dpi=180)
            plt.close(fig)

            eef = [
                (as_float(row.get("eef_x")), as_float(row.get("eef_y")), as_float(row.get("eef_z")))
                for row in item["rows"]
            ]
            eef = [point for point in eef if all(math.isfinite(v) for v in point)]
            if len(eef) >= 2:
                arr = np.array(eef)
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="#1f77b4")
                ax.scatter(arr[0, 0], arr[0, 1], arr[0, 2], color="#2ca02c", label="start")
                ax.scatter(arr[-1, 0], arr[-1, 1], arr[-1, 2], color="#d62728", label="end")
                ax.set_xlabel("eef x")
                ax.set_ylabel("eef y")
                ax.set_zlabel("eef z")
                ax.legend()
                fig.tight_layout()
                fig.savefig(case_dir / "trajectory_3d.png", dpi=180)
                plt.close(fig)
        except Exception as exc:
            (case_dir / "plot_error.txt").write_text(str(exc))

        with (case_dir / "summary.md").open("w") as f:
            f.write(f"# {tag}: {suite} task {task_id} episode {episode_idx}\n\n")
            f.write(f"- success: {item['success']}\n")
            f.write(f"- steps: {len(item['rows'])}\n")
            f.write(f"- avg accepted length: {format_float(item['avg_accept'])}\n")
            f.write(f"- avg composite: {format_float(item['avg_comp'])}\n")
            f.write(f"- avg top1 action L2 error: {format_float(item['avg_error'])}\n")
            f.write(f"- mode counts: {dict(item['modes'])}\n")
            f.write("\nFiles: `timeline.csv`, `timeline.png`, metric/accept/mode timelines, and optional `trajectory_3d.png`.\n")


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    suites = sorted({row["suite"] for row in rows})
    episodes = {
        (row["suite"], row.get("task_id"), row.get("episode_idx"), row.get("run_file"))
        for row in rows
    }
    labels = Counter(row["overlap_label"] for row in rows)
    label_sources = Counter(row.get("overlap_label_source", "") for row in rows)
    modes = Counter(row.get("mode", "") for row in rows)

    lines = [
        "# MMRebuttal Small Formal Summary",
        "",
        f"- root: `{root}`",
        f"- suites: {', '.join(suites) if suites else 'none'}",
        f"- episodes with records: {len(episodes)}",
        f"- step records: {len(rows)}",
        f"- overlap labels: {dict(labels)}",
        f"- overlap label sources: {dict(label_sources)}",
        f"- mode counts: {dict(modes)}",
        "",
        "Generated files:",
        "- `step_records.csv` / `step_records.jsonl`",
        "- `overlap_correlation/summary_by_suite.csv`",
        "- `overlap_correlation/summary_by_task.csv`",
        "- `overlap_correlation/stat_tests.csv`",
        "- `overlap_correlation/plots/*_distribution_by_overlap.png`",
        "- `timing_breakdown/timing_records.csv`",
        "- `timing_breakdown/summary_by_mode.csv`",
        "- `timing_breakdown/summary_by_suite.csv`",
        "- `timing_breakdown/timing_stacked_bar.png`",
        "- `case_study/*/summary.md`",
        "",
        "Timing note: timing summaries use only embedding, Qdrant retrieval, and model generation/verify time. They do not include environment stepping, video saving, rendering, or wait steps.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    rows = load_step_records(root)
    if not rows:
        raise SystemExit(f"No *_block_sd.json records found under {root / 'raw_runs'}")

    write_records(root, rows)
    analyze_overlap(root, rows)
    summarize_timing(root, rows)
    write_case_studies(root, rows)
    write_summary(root, rows)

    print(f"Analyzed {len(rows)} steps from {root / 'raw_runs'}")
    print(f"Wrote summary: {root / 'summary.md'}")


if __name__ == "__main__":
    main()
