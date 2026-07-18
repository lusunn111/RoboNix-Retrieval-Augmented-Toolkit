#!/usr/bin/env python3
"""Export compact PPT-ready overlap distribution figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path("/data/zhihao/mmrebuttal_outputs/trajectory_overlap_small_formal")
OUT_DIR = Path("/path/to/MMRebuttal/Outputs")
ARIAL_CANDIDATES = [
    Path("/path/to/external/third_party_baselines/ShortV/lmms-eval/lmms_eval/tasks/mmmu/arial.ttf"),
    Path("/data/models/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2/assets/Arial.ttf"),
]

METRICS = [
    (
        "raw_radius",
        "Radius",
        "Radius Distribution",
        "PPT_半径_分布图_2.55cm_300dpi.png",
        "半径_重叠与非重叠_直方图拟合曲线_300dpi.png",
    ),
    (
        "raw_displacement",
        "Displacement",
        "Displacement Distribution",
        "PPT_位移_分布图_2.55cm_300dpi.png",
        "位移_重叠与非重叠_直方图拟合曲线_300dpi.png",
    ),
    (
        "composite_metric",
        "Composite",
        "Composite Distribution",
        "PPT_综合指标_分布图_2.55cm_300dpi.png",
        "综合指标_重叠与非重叠_直方图拟合曲线_300dpi.png",
    ),
]

COLORS = {
    "Overlap": "#4c78a8",
    "Non-Overlap": "#f58518",
}


def cm_to_in(value: float) -> float:
    return value / 2.54


def fnum(value: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def setup_font(font_path: Path | None) -> None:
    if font_path is not None and font_path.exists():
        fm.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.linewidth": 0.5,
            "axes.unicode_minus": False,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_font(cli_font: Path | None) -> Path | None:
    if cli_font is not None:
        return cli_font
    for candidate in ARIAL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_metric_values(step_csv: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    overlap = []
    non_overlap = []
    with step_csv.open() as f:
        for row in csv.DictReader(f):
            label = row.get("overlap_label")
            if label not in {"overlap", "non_overlap"}:
                continue
            value = fnum(row.get(metric, ""))
            if not math.isfinite(value):
                # Missing radius/displacement/composite means there was not enough
                # motion history yet. Keep it as a zero-valued unstable segment
                # instead of dropping it; otherwise the near-zero non-overlap mass
                # disappears from the distribution figure.
                value = 0.0
            if label == "overlap":
                overlap.append(value)
            else:
                non_overlap.append(value)
    return np.asarray(overlap, dtype=float), np.asarray(non_overlap, dtype=float)


def smooth_density(values: np.ndarray, lo: float, hi: float, bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel_x = np.arange(-3, 4)
    kernel = np.exp(-0.5 * (kernel_x / 1.15) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    return centers, smooth


def kde_density(values: np.ndarray, lo: float, hi: float, points: int = 240) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    values = values[(values >= lo) & (values <= hi)]
    if values.size < 2:
        return np.array([]), np.array([])

    x = np.linspace(lo, hi, points)
    std = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])
    robust_std = float((q75 - q25) / 1.349) if q75 > q25 else std
    sigma = min(std, robust_std) if robust_std > 0 else std
    bandwidth = 0.9 * sigma * (values.size ** (-1 / 5)) if sigma > 0 else 0.0
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = max((hi - lo) / 80.0, 1e-8)

    z = (x[:, None] - values[None, :]) / bandwidth
    y = np.exp(-0.5 * z * z).sum(axis=1) / (values.size * bandwidth * math.sqrt(2 * math.pi))
    return x, y


def fitted_hist_curve(values: np.ndarray, lo: float, hi: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    values = values[(values >= lo) & (values <= hi)]
    if values.size == 0:
        return np.array([]), np.array([])
    density, edges = np.histogram(values, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel_x = np.arange(-3, 4)
    kernel = np.exp(-0.5 * (kernel_x / 0.8) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(density, kernel, mode="same")
    return centers, smooth


def fitted_hist_count_curve(values: np.ndarray, lo: float, hi: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    values = values[(values >= lo) & (values <= hi)]
    if values.size == 0:
        return np.array([]), np.array([])
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel_x = np.arange(-3, 4)
    kernel = np.exp(-0.5 * (kernel_x / 0.8) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(counts.astype(float), kernel, mode="same")
    return centers, smooth


def metric_limits(overlap: np.ndarray, non_overlap: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([overlap, non_overlap])
    values = values[np.isfinite(values)]
    lo = 0.0
    hi = float(np.percentile(values, 99.5))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo = 0.0
        hi = float(np.nanmax(values)) + 1e-6
    return lo, hi * 1.03


def style_axis(ax: plt.Axes, title: str, xlabel: str, *, compact: bool) -> None:
    title_size = 6.4 if compact else 20
    label_size = 5.4 if compact else 17
    tick_size = 4.8 if compact else 14
    ax.set_title(title, fontsize=title_size, pad=2.0 if compact else 10)
    ax.set_xlabel(xlabel, fontsize=label_size, labelpad=1.0 if compact else 6)
    ax.set_ylabel("Density", fontsize=5.4, labelpad=1.0)
    if not compact:
        ax.set_ylabel("Density", fontsize=label_size, labelpad=8)
    ax.tick_params(axis="both", labelsize=tick_size, pad=1.0 if compact else 4)
    ax.grid(True, color="#d9d9d9", linewidth=0.35 if compact else 0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))


def plot_metric(
    ax: plt.Axes,
    overlap: np.ndarray,
    non_overlap: np.ndarray,
    title: str,
    xlabel: str,
    show_legend: bool,
    *,
    compact: bool,
) -> None:
    lo, hi = metric_limits(overlap, non_overlap)
    bins = 32 if compact else 48
    for label, values in [("Overlap", overlap), ("Non-Overlap", non_overlap)]:
        clipped = values[(values >= lo) & (values <= hi)]
        if clipped.size == 0:
            continue
        ax.hist(
            clipped,
            bins=bins,
            range=(lo, hi),
            density=True,
            color=COLORS[label],
            alpha=0.34,
            edgecolor="white",
            linewidth=0.15 if compact else 0.35,
            label=label,
        )
        x, y = fitted_hist_curve(clipped, lo, hi, bins)
        if x.size:
            ax.plot(x, y, color=COLORS[label], linewidth=0.9 if compact else 2.0)
    ax.set_xlim(-0.015 * (hi - lo), hi)
    style_axis(ax, title, xlabel, compact=compact)
    if show_legend:
        ax.legend(
            loc="upper right",
            fontsize=4.6 if compact else 14,
            frameon=not compact,
            borderpad=0.1 if compact else 0.4,
            handlelength=1.0 if compact else 1.4,
            handletextpad=0.3 if compact else 0.6,
            labelspacing=0.2 if compact else 0.5,
        )


def style_camera_ready_axis(ax: plt.Axes, title: str, xlabel: str, *, scale: float = 1.0) -> None:
    ax.set_title(title, fontsize=10.0 * scale, pad=3 * scale)
    ax.set_xlabel(xlabel, fontsize=9.0 * scale, labelpad=2 * scale)
    ax.set_ylabel("Frequency", fontsize=9.0 * scale, labelpad=3 * scale)
    ax.tick_params(axis="both", labelsize=8.0 * scale, width=0.65, length=3.0 * scale, pad=2)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color="#e5e7eb", linewidth=0.6, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4 if scale >= 0.9 else 3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4 if scale >= 0.9 else 3))
    if xlabel in {"Radius", "Displacement"}:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    else:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def plot_metric_camera_ready(
    ax: plt.Axes,
    overlap: np.ndarray,
    non_overlap: np.ndarray,
    title: str,
    xlabel: str,
    *,
    show_legend: bool,
    ylabel: bool = True,
    scale: float = 1.0,
    ymax: float | None = None,
) -> None:
    lo, hi = metric_limits(overlap, non_overlap)
    bins = 52
    line_styles = {"Overlap": "-", "Non-Overlap": "-"}
    for label, values in [("Overlap", overlap), ("Non-Overlap", non_overlap)]:
        clipped = values[(values >= lo) & (values <= hi)]
        if clipped.size == 0:
            continue
        ax.hist(
            clipped,
            bins=bins,
            range=(lo, hi),
            density=False,
            color=COLORS[label],
            alpha=0.30,
            edgecolor="white",
            linewidth=0.18,
            label=label,
            zorder=2,
        )
        x, y = fitted_hist_count_curve(clipped, lo, hi, bins)
        if x.size:
            ax.plot(
                x,
                y,
                color=COLORS[label],
                linewidth=1.45 * scale,
                linestyle=line_styles[label],
                solid_capstyle="round",
                zorder=4,
            )

    ax.set_xlim(-0.012 * (hi - lo), hi)
    style_camera_ready_axis(ax, title, xlabel, scale=scale)
    if ymax is not None:
        current_top = ax.get_ylim()[1]
        if current_top > ymax:
            ax.set_ylim(0, ymax)
    if not ylabel:
        ax.set_ylabel("")
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[:2],
            labels[:2],
            loc="upper right",
            ncol=1,
            frameon=False,
            fontsize=7.2 * scale,
            handlelength=1.1,
            handletextpad=0.45,
            labelspacing=0.25,
            borderaxespad=0.2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument("--single-width-cm", type=float, default=2.55)
    parser.add_argument("--single-height-cm", type=float, default=3.20)
    parser.add_argument("--panel-width-cm", type=float, default=8.382)
    parser.add_argument("--panel-height-cm", type=float, default=3.81)
    parser.add_argument("--camera-width-cm", type=float, default=8.382)
    parser.add_argument("--camera-height-cm", type=float, default=3.81)
    parser.add_argument("--camera-ymax", type=float, default=250.0)
    args = parser.parse_args()

    setup_font(resolve_font(args.font))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_csv = args.root / "step_records.csv"

    loaded = []
    camera_files = []
    for metric, short_title, full_title, filename, large_filename in METRICS:
        overlap, non_overlap = load_metric_values(step_csv, metric)
        loaded.append((short_title, overlap, non_overlap, filename))

        fig, ax = plt.subplots(figsize=(cm_to_in(args.single_width_cm), cm_to_in(args.single_height_cm)))
        plot_metric(ax, overlap, non_overlap, short_title, short_title, show_legend=False, compact=True)
        fig.subplots_adjust(left=0.26, right=0.97, bottom=0.20, top=0.90)
        fig.savefig(args.out_dir / filename, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        plot_metric(ax, overlap, non_overlap, full_title, short_title, show_legend=True, compact=False)
        fig.subplots_adjust(left=0.14, right=0.93, bottom=0.16, top=0.86)
        fig.savefig(args.out_dir / large_filename, dpi=300)
        plt.close(fig)

        cn_title = {"Radius": "半径", "Displacement": "位移", "Composite": "综合指标"}[short_title]
        camera_name = f"顶会版_{cn_title}_重叠与非重叠_分布图_300dpi.png"
        fig, ax = plt.subplots(figsize=(cm_to_in(args.camera_width_cm), cm_to_in(args.camera_height_cm)))
        plot_metric_camera_ready(
            ax,
            overlap,
            non_overlap,
            short_title,
            "Composite Metric" if short_title == "Composite" else short_title,
            show_legend=True,
            scale=1.0,
            ymax=args.camera_ymax,
        )
        fig.subplots_adjust(left=0.14, right=0.985, bottom=0.24, top=0.88)
        fig.savefig(args.out_dir / camera_name, dpi=300)
        plt.close(fig)
        camera_files.append(camera_name)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(cm_to_in(args.panel_width_cm), cm_to_in(args.panel_height_cm)),
        sharey=True,
    )
    for idx, (title, overlap, non_overlap, _filename) in enumerate(loaded):
        plot_metric(axes[idx], overlap, non_overlap, title, title, show_legend=(idx == 2), compact=True)
        if idx > 0:
            axes[idx].set_ylabel("")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.89, wspace=0.40)
    panel_name = "PPT_半径位移综合指标_三联分布图_8.382x3.81cm_300dpi.png"
    fig.savefig(args.out_dir / panel_name, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(cm_to_in(args.panel_width_cm), cm_to_in(args.panel_height_cm)))
    for idx, (title, overlap, non_overlap, _filename) in enumerate(loaded):
        plot_metric_camera_ready(
            axes[idx],
            overlap,
            non_overlap,
            title,
            "Composite Metric" if title == "Composite" else title,
            show_legend=True,
            ylabel=(idx == 0),
            scale=0.64,
            ymax=args.camera_ymax,
        )
        if idx > 0:
            axes[idx].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.24, top=0.86, wspace=0.26)
    camera_panel_name = "顶会版_半径位移综合指标_三联分布图_8.382x3.81cm_300dpi.png"
    fig.savefig(args.out_dir / camera_panel_name, dpi=300)
    plt.close(fig)

    print(args.out_dir)
    for *_rest, filename in loaded:
        print(args.out_dir / filename)
    for filename in camera_files:
        print(args.out_dir / filename)
    print(args.out_dir / panel_name)
    print(args.out_dir / camera_panel_name)


if __name__ == "__main__":
    main()
