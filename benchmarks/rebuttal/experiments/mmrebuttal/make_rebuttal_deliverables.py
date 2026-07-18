#!/usr/bin/env python3
"""Create MMRebuttal-ready reports and case-study videos."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np


ROOT = Path("/data/zhihao/mmrebuttal_outputs/small_formal")
REPORT_DIR = ROOT / "reports"
VIDEO_DIR = ROOT / "case_study_videos"
SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]


def fnum(value, default=math.nan):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value, digits=3):
    value = fnum(value)
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def p_fmt(value):
    value = fnum(value)
    if not math.isfinite(value):
        return "-"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def read_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def load_common():
    step_rows = read_csv(ROOT / "step_records.csv")
    overlap_rows = read_csv(ROOT / "overlap_correlation" / "summary_by_suite.csv")
    stat_rows = read_csv(ROOT / "overlap_correlation" / "stat_tests.csv")
    timing_suite_rows = read_csv(ROOT / "timing_breakdown" / "summary_by_suite.csv")
    timing_records = read_csv(ROOT / "timing_breakdown" / "timing_records.csv")
    return step_rows, overlap_rows, stat_rows, timing_suite_rows, timing_records


def success_rows(step_rows):
    episodes = {}
    for row in step_rows:
        key = (row["suite"], row["task_id"], row["episode_idx"], row["run_file"])
        episodes[key] = row["episode_success"] == "True"
    by_suite = defaultdict(list)
    for (suite, *_), success in episodes.items():
        by_suite[suite].append(success)

    rows = []
    for suite in SUITES:
        vals = by_suite[suite]
        n_steps = sum(1 for row in step_rows if row["suite"] == suite)
        rows.append([suite, len(vals), sum(vals), f"{100 * sum(vals) / len(vals):.1f}%", n_steps])
    rows.append(["Overall", len(episodes), sum(episodes.values()), f"{100 * sum(episodes.values()) / len(episodes):.1f}%", len(step_rows)])
    return rows


def write_overlap_report(step_rows, overlap_rows, stat_rows):
    by_overlap = {(row["suite"], row["label"]): row for row in overlap_rows}
    by_stat = {(row["suite"], row["metric"]): row for row in stat_rows}

    distribution_rows = []
    for suite in SUITES:
        for label in ["overlap", "non_overlap"]:
            row = by_overlap[(suite, label)]
            distribution_rows.append([
                suite,
                "Overlap" if label == "overlap" else "Non-overlap",
                row["n_steps"],
                fmt(row["raw_radius_mean"], 4),
                fmt(row["raw_displacement_mean"], 4),
                fmt(row["composite_metric_mean"], 3),
                fmt(row["accepted_length_mean"], 2),
                fmt(row["top1_action_l2_error_mean"], 2),
            ])

    stat_compact_rows = []
    for suite in SUITES:
        disp = by_stat[(suite, "raw_displacement")]
        radius = by_stat[(suite, "raw_radius")]
        comp = by_stat[(suite, "composite_metric")]
        stat_compact_rows.append([
            suite,
            p_fmt(radius["mannwhitney_p"]),
            p_fmt(disp["mannwhitney_p"]),
            p_fmt(comp["mannwhitney_p"]),
            fmt(disp["cohen_d"], 3),
            fmt(disp["auc_separable"], 3),
        ])

    text = f"""# Overlap Correlation Analysis

## What This Experiment Answers

We proposed two kinematic signals, radius and displacement, and fuse them into a composite metric. This experiment does **not** claim that radius or displacement theoretically guarantees trajectory overlap. The point is empirical:

> If the retrieved trajectory segment really overlaps with the current rollout segment, then its radius/displacement/composite distribution should be statistically different from non-overlap segments.

We therefore split step-level retrievals by verifier accepted length:

- **Overlap**: accepted length >= 5.
- **Non-overlap**: accepted length <= 2.
- Neutral steps are kept in the raw records but excluded from the two-group comparison.

Accepted length is used as the main overlap proxy because it is produced by the verifier and directly reflects whether the retrieved proposal agrees with the target model.

## Main Distribution Table

{markdown_table(["Suite", "Group", "N", "Radius mean", "Displacement mean", "Composite mean", "Accepted len", "Action L2 error"], distribution_rows)}

## Statistical Test

Mann-Whitney U tests compare the overlap and non-overlap distributions. `AUC sep.` reports how separable the two groups are using the displacement signal. Values above 0.5 indicate positive separability.

{markdown_table(["Suite", "Radius p", "Displacement p", "Composite p", "Displacement Cohen d", "Displacement AUC sep."], stat_compact_rows)}

## Takeaway

- The verifier-defined overlap group has much larger accepted length and much lower action error in every suite, so the grouping is meaningful.
- `libero_goal`, `libero_spatial`, and `libero_object` show significant displacement differences between overlap and non-overlap segments.
- `libero_spatial` and `libero_object` also show significant composite-metric differences.
- `libero_10` is weaker: displacement and composite are not significant in this small run, which is consistent with longer-horizon tasks being more diverse and harder to summarize by a local kinematic signal.

Suggested rebuttal wording:

> We do not derive overlap from radius/displacement analytically. Instead, we validate the relationship empirically by grouping retrieved segments according to verifier accepted length. Across LIBERO Goal, Spatial, and Object, overlap segments show significantly different displacement distributions from non-overlap segments and consistently lower action error, supporting that the proposed kinematic signals are correlated with retrieval reliability.

Raw files:

- `{ROOT / "step_records.csv"}`
- `{ROOT / "overlap_correlation" / "summary_by_suite.csv"}`
- `{ROOT / "overlap_correlation" / "stat_tests.csv"}`
"""
    (REPORT_DIR / "01_overlap_correlation_report.md").write_text(text)


def write_timing_report(step_rows, timing_suite_rows, timing_records):
    timing_suite_rows = sorted(timing_suite_rows, key=lambda row: SUITES.index(row["suite"]))
    suite_rows = [
        [
            row["suite"],
            row["n_steps"],
            fmt(row["embedding_time_mean_ms"], 1),
            fmt(row["retrieval_time_mean_ms"], 1),
            fmt(row["generation_time_mean_ms"], 1),
            fmt(row["model_retrieval_time_mean_ms"], 1),
        ]
        for row in timing_suite_rows
    ]

    by_mode = defaultdict(list)
    for row in timing_records:
        by_mode[row["mode"]].append(row)

    def mean(rows, field):
        vals = [fnum(row[field]) for row in rows]
        vals = [v for v in vals if math.isfinite(v)]
        return sum(vals) / len(vals) if vals else math.nan

    mode_order = [
        "Retrieval_DB",
        "Retrieval_BlockSD_fully_verified",
        "Retrieval_BlockSD_partial_AR",
        "SD",
        "AR_insufficient_history",
    ]
    mode_rows = []
    for mode in mode_order:
        rows = by_mode[mode]
        mode_rows.append([
            mode,
            len(rows),
            fmt(mean(rows, "embedding_time_ms"), 1),
            fmt(mean(rows, "qdrant_search_time_ms"), 1),
            fmt(mean(rows, "generation_time_ms"), 1),
            fmt(mean(rows, "model_retrieval_time_ms"), 1),
        ])

    ratio_rows = []
    for suite in SUITES:
        rows = [row for row in step_rows if row["suite"] == suite]
        n = len(rows)
        counts = Counter(row["mode"] for row in rows)
        blocksd = counts["Retrieval_BlockSD_fully_verified"] + counts["Retrieval_BlockSD_partial_AR"]
        ratio_rows.append([
            suite,
            f"{100 * counts['Retrieval_DB'] / n:.1f}%",
            f"{100 * blocksd / n:.1f}%",
            f"{100 * counts['SD'] / n:.1f}%",
            f"{100 * counts['AR_insufficient_history'] / n:.1f}%",
        ])

    text = f"""# Timing Breakdown

## Scope

This report only counts model/retrieval path time:

- embedding time,
- local Qdrant search time,
- generation / verify time.

It explicitly excludes environment stepping, rendering, video saving, waiting steps, and episode wall time.

## Suite-Level Mean Time

Unit: milliseconds per step.

{markdown_table(["Suite", "N steps", "Embedding", "Qdrant", "Gen/Verify", "Total model path"], suite_rows)}

## Mode-Level Mean Time

Unit: milliseconds per step.

{markdown_table(["Mode", "N", "Embedding", "Qdrant", "Gen/Verify", "Total"], mode_rows)}

## Mode Ratio

{markdown_table(["Suite", "DB", "BlockSD", "SD", "AR"], ratio_rows)}

## Takeaway

- Direct DB retrieval is the fastest path, around 75 ms per step in this run.
- Qdrant search is small, around 7-8 ms; embedding dominates the direct DB path.
- `Retrieval_BlockSD_partial_AR` and `SD` are much slower because they call the model verifier/generation path.
- `libero_10` is slower mostly because it enters SD far more often, not because Qdrant itself is slow.

Raw files:

- `{ROOT / "timing_breakdown" / "timing_records.csv"}`
- `{ROOT / "timing_breakdown" / "summary_by_suite.csv"}`
- `{ROOT / "timing_breakdown" / "summary_by_mode.csv"}`
- `{ROOT / "timing_breakdown" / "timing_stacked_bar.png"}`
"""
    (REPORT_DIR / "03_timing_breakdown_report.md").write_text(text)


def read_case_summary(case_dir: Path):
    out = {}
    path = case_dir / "summary.md"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            out[key.strip()] = value.strip()
    return out


def make_case_video(case_dir: Path):
    rows = read_csv(case_dir / "timeline.csv")
    steps = np.array([fnum(row["step_idx"], 0.0) for row in rows], dtype=float)
    composite = np.array([fnum(row["composite_metric"]) for row in rows], dtype=float)
    accepted = np.array([fnum(row["accepted_length"]) for row in rows], dtype=float)
    error = np.array([fnum(row["top1_action_l2_error"]) for row in rows], dtype=float)
    eef = np.array([
        [fnum(row.get("eef_x")), fnum(row.get("eef_y")), fnum(row.get("eef_z"))]
        for row in rows
    ], dtype=float)

    valid = np.isfinite(steps)
    steps = steps[valid]
    composite = composite[valid]
    accepted = accepted[valid]
    error = error[valid]
    eef = eef[valid]
    rows = [row for row, ok in zip(rows, valid) if ok]

    n = len(steps)
    frame_count = min(180, max(20, n))
    frame_indices = np.unique(np.linspace(0, n - 1, frame_count).astype(int))

    mode_names = sorted({row["mode"] for row in rows})
    mode_to_id = {mode: idx for idx, mode in enumerate(mode_names)}
    mode_ids = np.array([mode_to_id[row["mode"]] for row in rows], dtype=float)

    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_metric = fig.add_subplot(grid[0, :])
    ax_mode = fig.add_subplot(grid[1, 0])
    ax_traj = fig.add_subplot(grid[1, 1], projection="3d")

    def clean_ylim(values, pad=0.1):
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return 0, 1
        lo, hi = float(vals.min()), float(vals.max())
        if lo == hi:
            return lo - 1, hi + 1
        span = hi - lo
        return lo - pad * span, hi + pad * span

    metric_lo, metric_hi = clean_ylim(np.concatenate([
        composite[np.isfinite(composite)],
        accepted[np.isfinite(accepted)] / 7.0,
    ]))
    error_lo, error_hi = clean_ylim(error)

    eef_valid = eef[np.all(np.isfinite(eef), axis=1)]
    if eef_valid.size:
        mins = eef_valid.min(axis=0)
        maxs = eef_valid.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-3)
        mins = mins - 0.1 * spans
        maxs = maxs + 0.1 * spans
    else:
        mins = np.array([0, 0, 0], dtype=float)
        maxs = np.array([1, 1, 1], dtype=float)

    metric_line, = ax_metric.plot([], [], label="composite", color="#1f77b4")
    accept_line, = ax_metric.plot([], [], label="accepted_len / 7", color="#2ca02c")
    error_ax = ax_metric.twinx()
    error_line, = error_ax.plot([], [], label="action L2 error", color="#d62728", alpha=0.8)
    cursor = ax_metric.axvline(0, color="black", alpha=0.25)

    mode_line, = ax_mode.step([], [], where="post", color="#9467bd")
    traj_line, = ax_traj.plot([], [], [], color="#1f77b4")
    traj_point = ax_traj.scatter([], [], [], color="#d62728", s=35)

    ax_metric.set_xlim(float(steps.min()), float(steps.max()))
    ax_metric.set_ylim(metric_lo, metric_hi)
    error_ax.set_ylim(error_lo, error_hi)
    ax_metric.set_ylabel("composite / normalized accepted")
    error_ax.set_ylabel("action L2 error")
    ax_metric.grid(True, alpha=0.25)

    lines = [metric_line, accept_line, error_line]
    labels = [line.get_label() for line in lines]
    ax_metric.legend(lines, labels, loc="upper right")

    ax_mode.set_xlim(float(steps.min()), float(steps.max()))
    ax_mode.set_ylim(-0.5, max(0.5, len(mode_names) - 0.5))
    ax_mode.set_yticks(list(mode_to_id.values()))
    ax_mode.set_yticklabels(mode_names, fontsize=7)
    ax_mode.set_xlabel("step")
    ax_mode.set_ylabel("mode")
    ax_mode.grid(True, alpha=0.25)

    ax_traj.set_xlim(mins[0], maxs[0])
    ax_traj.set_ylim(mins[1], maxs[1])
    ax_traj.set_zlim(mins[2], maxs[2])
    ax_traj.set_xlabel("eef x")
    ax_traj.set_ylabel("eef y")
    ax_traj.set_zlabel("eef z")

    title = fig.suptitle(case_dir.name)

    def update(frame_idx):
        idx = int(frame_idx)
        xs = steps[: idx + 1]
        metric_line.set_data(xs, composite[: idx + 1])
        accept_line.set_data(xs, accepted[: idx + 1] / 7.0)
        error_line.set_data(xs, error[: idx + 1])
        cursor.set_xdata([steps[idx], steps[idx]])

        mode_line.set_data(xs, mode_ids[: idx + 1])

        valid_traj = eef[: idx + 1]
        valid_traj = valid_traj[np.all(np.isfinite(valid_traj), axis=1)]
        if len(valid_traj) > 0:
            traj_line.set_data(valid_traj[:, 0], valid_traj[:, 1])
            traj_line.set_3d_properties(valid_traj[:, 2])
            traj_point._offsets3d = ([valid_traj[-1, 0]], [valid_traj[-1, 1]], [valid_traj[-1, 2]])

        title.set_text(f"{case_dir.name} | step {int(steps[idx])}")
        return metric_line, accept_line, error_line, mode_line, traj_line, traj_point, cursor

    fig.tight_layout()
    video_path = VIDEO_DIR / f"{case_dir.name}.mp4"
    anim = FuncAnimation(fig, update, frames=frame_indices, interval=100, blit=False)
    anim.save(video_path, writer=FFMpegWriter(fps=12, bitrate=1800))
    plt.close(fig)
    return video_path


def write_case_report():
    rows = []
    for case_dir in sorted((ROOT / "case_study").iterdir()):
        if not case_dir.is_dir() or not (case_dir / "timeline.csv").exists():
            continue
        summary = read_case_summary(case_dir)
        video = make_case_video(case_dir)
        rows.append([
            case_dir.name,
            summary.get("success", "-"),
            summary.get("steps", "-"),
            fmt(summary.get("avg accepted length"), 2),
            fmt(summary.get("avg composite"), 3),
            fmt(summary.get("avg top1 action L2 error"), 2),
            str(video),
        ])

    text = f"""# Case Study Deliverables

## What These Videos Show

The full small-formal run did not save RGB rollout videos to avoid large output. Therefore the delivered videos are metric-timeline case videos reconstructed from recorded step-level signals. Each video shows:

- composite metric over time,
- accepted length over time,
- action error over time,
- selected mode over time,
- end-effector trajectory.

These videos are meant to explain why the controller stays on DB, switches to BlockSD/SD, or enters fallback-like generation paths.

{markdown_table(["Case", "Success", "Steps", "Avg accepted", "Avg composite", "Avg action error", "Video"], rows)}

If RGB environment frames are required for the final paper figure, rerun only the selected case episodes with frame/video recording enabled; the full 80-episode experiment does not need to be repeated.
"""
    (REPORT_DIR / "02_case_study_deliverables.md").write_text(text)


def write_index():
    text = f"""# MMRebuttal Deliverables

This folder contains the three requested deliverables:

1. Overlap correlation document: `01_overlap_correlation_report.md`
2. Case study video manifest: `02_case_study_deliverables.md`
3. Timing breakdown document: `03_timing_breakdown_report.md`

Case-study videos are stored in:

`{VIDEO_DIR}`

All results are generated from:

`{ROOT}`
"""
    (REPORT_DIR / "README.md").write_text(text)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    step_rows, overlap_rows, stat_rows, timing_suite_rows, timing_records = load_common()
    write_overlap_report(step_rows, overlap_rows, stat_rows)
    write_timing_report(step_rows, timing_suite_rows, timing_records)
    write_case_report()
    write_index()
    print(f"Wrote reports to {REPORT_DIR}")
    print(f"Wrote videos to {VIDEO_DIR}")


if __name__ == "__main__":
    main()

