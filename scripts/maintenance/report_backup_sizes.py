#!/usr/bin/env python3
"""Report Qdrant snapshot backup sizes under scripts/retrieval/qdrant_backups.

This script is intentionally dependency-free.

It helps answer questions like:
- Is there an apparent fixed overhead in snapshots?
- How does snapshot size grow from backup_base to backup_base+N or diff_backup_base+N?

Usage:
  python scripts/retrieval/report_backup_sizes.py \
    --root scripts/retrieval/qdrant_backups \
    --base backup_base

Notes:
- We only sum *.snapshot sizes (not directory metadata).
- "Episodes per collection" is inferred from directory name suffix base+N.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class BackupRow:
    name: str
    snapshots: int
    total_bytes: int
    delta_vs_base: int
    episodes_per_collection: Optional[int]


def _fmt_bytes(num: int) -> str:
    n = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(n)}B"
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{num}B"


def _sum_snapshot_bytes(path: Path) -> tuple[int, int]:
    files = list(path.glob("*.snapshot"))
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except FileNotFoundError:
            # Best-effort if files are being written.
            continue
    return len(files), total


def _extract_base_plus(name: str) -> Optional[int]:
    m = re.search(r"base\+(\d+)", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _iter_backup_dirs(root: Path) -> Iterable[Path]:
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name == "latest":
            continue
        if p.name == "backup_base" or p.name.startswith("backup_base+") or p.name.startswith(
            "diff_backup_base+"
        ):
            yield p


def _sort_key(p: Path) -> tuple[int, int, str]:
    if p.name == "backup_base":
        return (0, 0, p.name)
    n = _extract_base_plus(p.name) or 0
    if p.name.startswith("backup_base+"):
        return (1, n, p.name)
    if p.name.startswith("diff_backup_base+"):
        return (2, n, p.name)
    return (3, n, p.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="scripts/retrieval/qdrant_backups",
        help="Backup root directory",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="backup_base",
        help="Base backup directory name under root",
    )
    args = parser.parse_args()

    root = Path(args.root)
    base_dir = root / args.base
    if not base_dir.exists():
        raise SystemExit(f"Base backup dir not found: {base_dir}")

    base_snapshots, base_bytes = _sum_snapshot_bytes(base_dir)
    if base_snapshots == 0:
        raise SystemExit(f"No *.snapshot files under: {base_dir}")

    rows: list[BackupRow] = []
    for p in sorted(_iter_backup_dirs(root), key=_sort_key):
        n_snaps, total = _sum_snapshot_bytes(p)
        rows.append(
            BackupRow(
                name=p.name,
                snapshots=n_snaps,
                total_bytes=total,
                delta_vs_base=total - base_bytes,
                episodes_per_collection=_extract_base_plus(p.name),
            )
        )

    print(f"Backup root: {root}")
    print(f"Base: {base_dir.name} (snapshots={base_snapshots}, total={_fmt_bytes(base_bytes)})")
    print()
    print(
        "name\tsnapshots\ttotal_snap_bytes\tdelta_vs_base\tavg_delta_per_collection\tavg_delta_per_episode"
    )

    collections = base_snapshots

    for r in rows:
        per_collection = r.delta_vs_base / collections if collections else 0.0
        per_episode = "-"
        if r.episodes_per_collection is not None and r.episodes_per_collection > 0:
            total_episodes = collections * r.episodes_per_collection
            per_episode_val = r.delta_vs_base / total_episodes
            per_episode = _fmt_bytes(int(per_episode_val))
        print(
            f"{r.name}\t{r.snapshots}\t{_fmt_bytes(r.total_bytes)}\t{_fmt_bytes(r.delta_vs_base)}\t{_fmt_bytes(int(per_collection))}\t{per_episode}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
