#!/usr/bin/env python

"""Inspect a local LeRobot dataset and print a few sample values.

Example:
    python scripts/inspect_lerobot_dataset.py
    python scripts/inspect_lerobot_dataset.py --root /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from pprint import pprint


os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_datasets_cache")


DEFAULT_DATASET_GLOB = "/home/robot/.cache/huggingface/lerobot/kuimin/test_ur_gello_*"


def find_latest_dataset() -> Path:
    candidates = [path for path in Path("/").glob(DEFAULT_DATASET_GLOB.lstrip("/")) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No dataset directories matched {DEFAULT_DATASET_GLOB}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def repo_id_from_root(root: Path) -> str:
    # Local cache layout is usually .../lerobot/{owner}/{dataset_name}.
    if len(root.parts) >= 2:
        return f"{root.parent.name}/{root.name}"
    return root.name


def to_python(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def summarize_value(value, max_items: int = 12):
    value = to_python(value)
    if isinstance(value, list):
        flat = value
        if flat and isinstance(flat[0], list):
            return f"nested list, outer_len={len(flat)}"
        shown = flat[:max_items]
        suffix = "" if len(flat) <= max_items else f" ... ({len(flat)} values)"
        return f"{shown}{suffix}"
    return value


def load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inspect_with_lerobot(root: Path, sample_index: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = repo_id_from_root(root)
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    print("\nLoaded with LeRobotDataset")
    print(f"repo_id: {repo_id}")
    print(f"root: {root}")
    print(f"num_frames: {len(dataset)}")
    print(f"num_episodes: {dataset.num_episodes}")
    print(f"fps: {dataset.fps}")

    idx = min(max(sample_index, 0), len(dataset) - 1)
    sample = dataset[idx]
    print(f"\nSample frame index: {idx}")
    for key in sorted(sample):
        value = sample[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        print(f"{key}: shape={shape}, value={summarize_value(value)}")

    effort_key = "observation.effort"
    if effort_key in sample:
        print(f"\n{effort_key}: {summarize_value(sample[effort_key], max_items=24)}")


def inspect_metadata(root: Path) -> None:
    info = load_info(root)
    print("LeRobot metadata")
    print(f"root: {root}")
    print(f"codebase_version: {info.get('codebase_version')}")
    print(f"robot_type: {info.get('robot_type')}")
    print(f"fps: {info.get('fps')}")
    print("\nfeatures:")
    pprint(info.get("features", {}), sort_dicts=False)


def inspect_parquet_fallback(root: Path, sample_index: int) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("\nPandas is not installed; skipping parquet fallback.")
        return

    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        print("\nNo data parquet files found.")
        return

    df = pd.read_parquet(parquet_files[0])
    idx = min(max(sample_index, 0), len(df) - 1)
    print("\nParquet fallback")
    print(f"file: {parquet_files[0]}")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    row = df.iloc[idx]
    print(f"\nRaw row {idx}:")
    for key in df.columns:
        print(f"{key}: {summarize_value(row[key])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local LeRobot dataset.")
    parser.add_argument("--root", type=Path, default=None, help="Path to a local LeRobot dataset root.")
    parser.add_argument("--sample-index", type=int, default=0, help="Frame index to print.")
    parser.add_argument("--parquet-only", action="store_true", help="Skip LeRobotDataset and read parquet directly.")
    args = parser.parse_args()

    root = args.root or find_latest_dataset()
    root = root.expanduser().resolve()

    inspect_metadata(root)
    if args.parquet_only:
        inspect_parquet_fallback(root, args.sample_index)
        return

    try:
        inspect_with_lerobot(root, args.sample_index)
    except Exception as exc:
        print(f"\nLeRobotDataset load failed: {exc}")
        inspect_parquet_fallback(root, args.sample_index)


if __name__ == "__main__":
    main()
