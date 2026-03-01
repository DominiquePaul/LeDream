#!/usr/bin/env python3
"""Repair LeRobot `meta/episodes` index ranges from observed data rows.

This script fixes metadata drift where episode rows in `meta/episodes` no longer
match the actual data parquet rows (`data/chunk-*/file-*.parquet`), which can
cause DataLoader sampler out-of-bounds errors during training.

What it does:
1. Loads the dataset with `LeRobotDataset`.
2. Rebuilds per-episode `[dataset_from_index, dataset_to_index)` and `length`
   from `hf_dataset` (`index` + `episode_index` columns).
3. Resolves duplicate metadata rows per episode by selecting the best matching
   source row and replacing index fields with observed values.
4. Rewrites `meta/episodes` as a single parquet file after backing up the
   existing directory.

By default it runs in dry-run mode and prints what would change.

Usage:
    python scripts/repair_dataset_episodes_metadata.py username/dataset
    python scripts/repair_dataset_episodes_metadata.py username/dataset --apply
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass(frozen=True)
class ObservedEpisodeRange:
    start: int
    end_exclusive: int
    length: int


def _to_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _compute_observed_ranges(ds: LeRobotDataset) -> dict[int, ObservedEpisodeRange]:
    episode_ids = [_to_int(v) for v in ds.hf_dataset["episode_index"]]
    indices = [_to_int(v) for v in ds.hf_dataset["index"]]
    if len(episode_ids) != len(indices):
        raise ValueError("Column length mismatch between episode_index and index.")

    observed: dict[int, dict[str, int]] = {}
    for ep, idx in zip(episode_ids, indices, strict=True):
        if ep not in observed:
            observed[ep] = {"start": idx, "end_exclusive": idx + 1, "length": 1}
            continue
        row = observed[ep]
        if idx < row["start"]:
            row["start"] = idx
        if idx + 1 > row["end_exclusive"]:
            row["end_exclusive"] = idx + 1
        row["length"] += 1

    return {
        ep: ObservedEpisodeRange(
            start=vals["start"], end_exclusive=vals["end_exclusive"], length=vals["length"]
        )
        for ep, vals in observed.items()
    }


def _choose_best_metadata_row(rows: list[dict[str, Any]], observed: ObservedEpisodeRange) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[int, int]:
        from_i = _to_int(row["dataset_from_index"])
        to_i = _to_int(row["dataset_to_index"])
        length = _to_int(row["length"])
        exact_match = int(
            from_i == observed.start and to_i == observed.end_exclusive and length == observed.length
        )
        boundary_match = int(from_i == observed.start and to_i == observed.end_exclusive)
        length_match = int(length == observed.length)
        # Higher is better.
        primary = exact_match * 100 + boundary_match * 10 + length_match
        # Prefer earlier row as deterministic tie-breaker by returning 0 for all.
        return (primary, 0)

    return max(rows, key=score)


def _build_repaired_rows(ds: LeRobotDataset) -> tuple[list[dict[str, Any]], dict[str, int]]:
    observed_by_ep = _compute_observed_ranges(ds)
    observed_eps = sorted(observed_by_ep.keys())

    meta = ds.meta.episodes
    meta_rows = [meta[i] for i in range(len(meta))]
    meta_by_ep: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in meta_rows:
        meta_by_ep[_to_int(row["episode_index"])].append(row)

    meta_eps = set(meta_by_ep.keys())
    missing_in_meta = sorted(set(observed_eps) - meta_eps)
    extra_in_meta = sorted(meta_eps - set(observed_eps))

    if missing_in_meta:
        raise ValueError(
            "Cannot safely repair: some episodes exist in data but not in metadata. "
            f"Missing metadata episode ids: {missing_in_meta[:10]}"
            + ("..." if len(missing_in_meta) > 10 else "")
        )

    repaired: list[dict[str, Any]] = []
    duplicate_meta_episodes = 0
    for ep in observed_eps:
        candidates = meta_by_ep[ep]
        if len(candidates) > 1:
            duplicate_meta_episodes += 1
        base = _choose_best_metadata_row(candidates, observed_by_ep[ep])
        repaired_row = dict(base)
        repaired_row["episode_index"] = ep
        repaired_row["dataset_from_index"] = observed_by_ep[ep].start
        repaired_row["dataset_to_index"] = observed_by_ep[ep].end_exclusive
        repaired_row["length"] = observed_by_ep[ep].length
        # Rewriting into a single parquet file.
        repaired_row["meta/episodes/chunk_index"] = 0
        repaired_row["meta/episodes/file_index"] = 0
        repaired.append(repaired_row)

    repaired.sort(key=lambda r: _to_int(r["episode_index"]))

    stats = {
        "meta_rows_before": len(meta_rows),
        "meta_rows_after": len(repaired),
        "duplicate_episode_ids_in_meta": duplicate_meta_episodes,
        "episodes_missing_in_meta": len(missing_in_meta),
        "episodes_extra_in_meta": len(extra_in_meta),
        "episodes_observed": len(observed_eps),
    }
    return repaired, stats


def _rewrite_episodes_dir(dataset_root: Path, repaired_rows: list[dict[str, Any]]) -> Path:
    meta_dir = dataset_root / "meta"
    episodes_dir = meta_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Expected episodes directory at: {episodes_dir}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = meta_dir / f"episodes.backup.{timestamp}"
    shutil.move(str(episodes_dir), str(backup_dir))

    out_file = episodes_dir / "chunk-000" / "file-000.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(repaired_rows).to_parquet(out_file, index=False)
    return backup_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair LeRobot meta/episodes index metadata from observed data rows."
    )
    parser.add_argument("repo_id", help="Dataset id in form username/datasetname")
    parser.add_argument("--root", type=Path, default=None, help="Optional local dataset root override")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this flag, runs as dry-run.",
    )
    args = parser.parse_args()

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    repaired_rows, stats = _build_repaired_rows(ds)

    print(f"Dataset root: {ds.root}")
    print(f"Repo: {args.repo_id}")
    print("Planned repair summary:")
    print(f"  - meta rows: {stats['meta_rows_before']} -> {stats['meta_rows_after']}")
    print(f"  - observed episodes: {stats['episodes_observed']}")
    print(f"  - duplicate episode ids in meta: {stats['duplicate_episode_ids_in_meta']}")
    print(f"  - episodes missing in meta: {stats['episodes_missing_in_meta']}")
    print(f"  - episodes extra in meta: {stats['episodes_extra_in_meta']}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to rewrite meta/episodes.")
        return

    backup_dir = _rewrite_episodes_dir(ds.root, repaired_rows)
    print("\nRepair complete.")
    print(f"Backup created at: {backup_dir}")
    print("Re-run validation:")
    print(f"  python scripts/check_dataset_indices.py {args.repo_id} --strict")


if __name__ == "__main__":
    main()
