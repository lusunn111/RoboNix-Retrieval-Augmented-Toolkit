#!/usr/bin/env python3

import argparse
import os
import stat
from pathlib import Path
from typing import Dict, Tuple

import tensorflow_datasets as tfds


DEFAULT_SNAPSHOT_ROOT = (
    "/path/to/rtcache/libero/"
    "datasets--openvla--modified_libero_rlds/snapshots/"
    "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"
)

SUITES = {
    "goal": "libero_goal_no_noops",
    "spatial": "libero_spatial_no_noops",
    "object": "libero_object_no_noops",
    "long": "libero_10_no_noops",
}


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{int(num_bytes)}B"


def compute_disk_usage_bytes(root: Path) -> Tuple[int, int, Dict[str, int]]:
    """Compute real disk usage for a dataset dir.

    Notes:
      - 目录里大量是 symlink 指向 blobs/；这里会 follow 到真实文件。
      - 用 (st_dev, st_ino) 去重，避免重复计算硬链接/重复引用。

    Returns:
      (disk_bytes, apparent_bytes, debug_counts)
    """
    seen_inodes = set()
    disk_bytes = 0
    apparent_bytes = 0
    counts = {"files": 0, "symlinks": 0, "unique_targets": 0}

    for dirpath, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                st_l = os.lstat(path)
            except OSError:
                continue

            if stat.S_ISLNK(st_l.st_mode):
                counts["symlinks"] += 1
                target_path = Path(os.path.realpath(path))
            else:
                target_path = path

            try:
                st = os.stat(target_path)
            except OSError:
                continue

            if not stat.S_ISREG(st.st_mode):
                continue

            inode_key = (st.st_dev, st.st_ino)
            if inode_key in seen_inodes:
                continue
            seen_inodes.add(inode_key)
            counts["unique_targets"] += 1
            counts["files"] += 1

            disk_bytes += int(st.st_blocks) * 512
            apparent_bytes += int(st.st_size)

    return disk_bytes, apparent_bytes, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check LIBERO TFDS episode counts and on-disk size (following blobs symlinks)."
    )
    parser.add_argument(
        "--snapshot-root",
        type=str,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="Path to TFDS snapshot root (contains libero_*_no_noops dirs).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="TFDS version subdir (default: 1.0.0)",
    )

    args = parser.parse_args()
    snapshot_root = Path(args.snapshot_root)

    if not snapshot_root.exists():
        print(f"ERROR: snapshot root not found: {snapshot_root}")
        return 1

    print(f"Snapshot root: {snapshot_root}")
    print("=" * 80)

    totals = {"disk": 0, "apparent": 0, "episodes": 0}

    for suite_name, suite_dirname in SUITES.items():
        dataset_dir = snapshot_root / suite_dirname / args.version
        print(f"Suite: {suite_name}")
        print(f"Path:  {dataset_dir}")

        if not dataset_dir.exists():
            print("  WARNING: dataset dir missing, skip")
            print("-" * 80)
            continue

        # Episode count
        num_train = None
        ds_name = None
        try:
            builder = tfds.builder_from_directory(str(dataset_dir))
            ds_name = builder.name
            if "train" in builder.info.splits:
                num_train = builder.info.splits["train"].num_examples
        except Exception as e:
            print(f"  ERROR: tfds load failed: {e}")

        if ds_name:
            print(f"  Dataset: {ds_name}")
        if num_train is not None:
            print(f"  Episodes(train): {num_train}")
            totals["episodes"] += int(num_train)

        # Storage usage (follow symlinks into blobs)
        disk_b, apparent_b, counts = compute_disk_usage_bytes(dataset_dir)
        totals["disk"] += disk_b
        totals["apparent"] += apparent_b
        print(f"  Disk usage (real):     {human_bytes(disk_b)}")
        print(f"  Apparent size:         {human_bytes(apparent_b)}")
        print(
            "  Targets: files={files} symlinks={symlinks} unique_targets={unique_targets}".format(
                **counts
            )
        )
        print("-" * 80)

    print("TOTAL")
    print(f"  Episodes(train): {totals['episodes']}")
    print(f"  Disk usage (real): {human_bytes(totals['disk'])}")
    print(f"  Apparent size:     {human_bytes(totals['apparent'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
