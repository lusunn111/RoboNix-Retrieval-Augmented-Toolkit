#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int
    n_action_bins: int


def _load_tokenizer_config(model_config_path: Path) -> TokenizerConfig:
    model_config = json.loads(model_config_path.read_text())
    n_action_bins = int(model_config["n_action_bins"])
    text_vocab_size = int(model_config["text_config"]["vocab_size"])
    pad_to_multiple_of = int(model_config["pad_to_multiple_of"])
    vocab_size = text_vocab_size - pad_to_multiple_of
    return TokenizerConfig(vocab_size=vocab_size, n_action_bins=n_action_bins)


def _build_action_bins(n_action_bins: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n_action_bins, dtype=np.float32)


def _action_to_token_ids(
    action: np.ndarray,
    tokenizer_config: TokenizerConfig,
    bins: np.ndarray,
) -> np.ndarray:
    clipped_action = np.clip(action.astype(np.float32), a_min=-1.0, a_max=1.0)
    discretized = np.digitize(clipped_action, bins)
    discretized = np.clip(discretized - 1, a_min=0, a_max=(len(bins) - 2))
    token_ids = tokenizer_config.vocab_size - discretized - 1
    return token_ids.astype(np.int64)


def _prefix_match_length(tokens_a: np.ndarray, tokens_b: np.ndarray) -> int:
    matched = 0
    for token_a, token_b in zip(tokens_a.tolist(), tokens_b.tolist()):
        if token_a != token_b:
            break
        matched += 1
    return matched


def _iter_steps(spec_actions: Any, db_actions: Any) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    if not isinstance(spec_actions, list) or not isinstance(db_actions, list):
        return
    max_len = min(len(spec_actions), len(db_actions))
    for step_index in range(max_len):
        spec_action = spec_actions[step_index]
        db_action = db_actions[step_index]
        if db_action is None:
            continue
        yield step_index, np.asarray(spec_action), np.asarray(db_action)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.array(values, dtype=np.float32)))


def _safe_pctl(values: list[float], pctl: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float32), pctl))


def _flatten_object_array(value: np.ndarray) -> list:
    return value.tolist() if isinstance(value, np.ndarray) else list(value)


def analyze_npz(npz_path: Path, model_config_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_config = _load_tokenizer_config(model_config_path)
    bins = _build_action_bins(tokenizer_config.n_action_bins)

    npz = np.load(npz_path, allow_pickle=True)
    task_suite_name = str(npz["task_suite_name"])
    accept_threshold = int(npz["accept_threshold"]) if "accept_threshold" in npz else None

    task_names = _flatten_object_array(npz["task_names"])
    specvla_actions = npz["specvla_actions"]
    db_actions = npz["db_actions"]
    recorded_accept_lengths = npz["accept_lengths"] if "accept_lengths" in npz else None

    rows: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(task_names):
        for episode_index in range(specvla_actions.shape[1]):
            episode_spec_actions = specvla_actions[task_index, episode_index]
            episode_db_actions = db_actions[task_index, episode_index]

            episode_recorded = None
            if recorded_accept_lengths is not None:
                episode_recorded = recorded_accept_lengths[task_index, episode_index]

            if not isinstance(episode_spec_actions, list):
                continue

            episode_db_list = episode_db_actions if isinstance(episode_db_actions, list) else []
            episode_accept_list = episode_recorded if isinstance(episode_recorded, list) else []

            for step_index in range(len(episode_spec_actions)):
                spec_action = np.asarray(episode_spec_actions[step_index])
                db_action = episode_db_list[step_index] if step_index < len(episode_db_list) else None

                recorded_accept_len = None
                if step_index < len(episode_accept_list):
                    recorded_accept_len = episode_accept_list[step_index]
                    if isinstance(recorded_accept_len, (list, tuple)) and recorded_accept_len:
                        recorded_accept_len = recorded_accept_len[0]

                recomputed_accept_len = None
                l2_distance = float("nan")
                db_token_ids = None
                spec_token_ids = None
                if db_action is not None:
                    db_action_arr = np.asarray(db_action)
                    spec_token_ids = _action_to_token_ids(spec_action, tokenizer_config, bins)
                    db_token_ids = _action_to_token_ids(db_action_arr, tokenizer_config, bins)
                    recomputed_accept_len = _prefix_match_length(spec_token_ids, db_token_ids)
                    l2_distance = float(np.linalg.norm((spec_action - db_action_arr).astype(np.float32), ord=2))

                accept_len = recorded_accept_len if recorded_accept_len is not None else recomputed_accept_len
                rows.append(
                    {
                        "task_suite": task_suite_name,
                        "task_name": task_name,
                        "task_index": task_index,
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "accept_len_recorded": recorded_accept_len,
                        "accept_len": accept_len,
                        "accept_len_recomputed": recomputed_accept_len,
                        "l2_distance": l2_distance,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No valid steps found in {npz_path}")

    df.to_csv(out_dir / "per_step.csv", index=False)

    per_task = (
        df.groupby(["task_suite", "task_index", "task_name"], as_index=False)
        .agg(
            steps=("accept_len", "count"),
            mean_accept=("accept_len", "mean"),
            p50_accept=("accept_len", lambda x: float(np.percentile(np.asarray(x, dtype=np.float32), 50))),
            p95_accept=("accept_len", lambda x: float(np.percentile(np.asarray(x, dtype=np.float32), 95))),
            pct_accept_gt0=("accept_len", lambda x: float(np.mean(np.asarray(x, dtype=np.float32) > 0))),
            mean_l2=("l2_distance", lambda x: float(np.nanmean(np.asarray(x, dtype=np.float32)))),
        )
        .sort_values(["task_index"])
    )
    per_task.to_csv(out_dir / "per_task_summary.csv", index=False)

    if df["accept_len_recomputed"].notna().any() and df["accept_len_recorded"].notna().any():
        comparable = df.dropna(subset=["accept_len_recorded", "accept_len_recomputed"])
        accept_agreement = float(
            np.mean(comparable["accept_len_recorded"].to_numpy() == comparable["accept_len_recomputed"].to_numpy())
        )
    else:
        accept_agreement = float("nan")

    overall = {
        "npz": str(npz_path),
        "task_suite": task_suite_name,
        "accept_threshold": accept_threshold,
        "steps": int(df.shape[0]),
        "mean_accept": _safe_mean(df["accept_len"].dropna().tolist()),
        "p50_accept": _safe_pctl(df["accept_len"].dropna().tolist(), 50),
        "p95_accept": _safe_pctl(df["accept_len"].dropna().tolist(), 95),
        "pct_accept_gt0": float(np.mean(df["accept_len"].fillna(0).to_numpy() > 0)),
        "mean_l2": float(np.nanmean(df["l2_distance"].to_numpy(dtype=np.float32))),
        "accept_len_recorded_vs_recomputed_agreement": accept_agreement,
    }
    (out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2))

    plt.figure(figsize=(7, 4))
    plt.hist(df["accept_len"].fillna(0), bins=np.arange(-0.5, 8.5, 1), rwidth=0.9)
    plt.title("DB-as-draft accept length")
    plt.xlabel("accept length (#tokens)")
    plt.ylabel("count")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "accept_len_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sampled = df.dropna(subset=["l2_distance", "accept_len"]).sample(n=min(5000, len(df)), random_state=0)
    plt.scatter(sampled["l2_distance"], sampled["accept_len"], s=6, alpha=0.3)
    plt.title("accept length vs L2(action) (sampled)")
    plt.xlabel("L2 distance between SpecVLA action and DB action")
    plt.ylabel("accept length (#tokens)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "accept_vs_l2.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, max(4, 0.25 * len(per_task))))
    y_pos = np.arange(len(per_task))
    plt.barh(y_pos, per_task["mean_accept"], color="#4C72B0")
    plt.yticks(y_pos, per_task["task_name"])
    plt.gca().invert_yaxis()
    plt.title("Mean accept length per task")
    plt.xlabel("mean accept length (#tokens)")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_accept_per_task.png", dpi=200)
    plt.close()

    print(f"[OK] Wrote: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DB-as-draft accept length from SpecVLA npz outputs.")
    parser.add_argument("--npz", type=Path, required=True, help="Path to accept_length_data_*.npz")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("backbone_models/openvla-7b-finetuned-libero-goal/config.json"),
        help="Path to OpenVLA model config.json (for n_action_bins/pad_token_id).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: Lab/results/<npz_stem>/)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else Path("Lab/results") / args.npz.stem
    analyze_npz(args.npz, args.model_config, out_dir)


if __name__ == "__main__":
    main()
