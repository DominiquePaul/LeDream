#!/usr/bin/env python3
"""Check index integrity for a LeRobot dataset.

This script validates that core indexing fields are self-consistent and flags
irregularities often introduced by filtering/subsetting operations.

Checks include:
- Required fields exist: `index`, `episode_index`, `frame_index`
- Global `index` is unique, strictly increasing by 1, and starts at 0
- Rows are sorted by `index`
- Per-episode `frame_index` is contiguous from 0..N-1
- Per-episode metadata ranges (`dataset_from_index`, `dataset_to_index`) match
  actual data rows and global indices
- Episode IDs are contiguous from 0..N-1

Usage:
    python scripts/check_dataset_indices.py username/datasetname
    python scripts/check_dataset_indices.py username/datasetname --strict
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    # Allow running from repository root without setting PYTHONPATH manually.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _to_scalar_int(value: Any) -> int:
    """Convert tensor/scalar-like values to Python int."""
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _to_list_of_ints(values: Any) -> list[int]:
    """Convert a sequence of tensor/scalar-like values to list[int]."""
    return [_to_scalar_int(v) for v in values]


def validate_dataset_indices(repo_id: str, root: Path | None = None) -> list[str]:
    ds = LeRobotDataset(repo_id=repo_id, root=root)
    hf_ds = ds.hf_dataset

    errors: list[str] = []

    required = ("index", "episode_index", "frame_index")
    missing = [k for k in required if k not in hf_ds.features]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors

    indices = _to_list_of_ints(hf_ds["index"])
    episodes = _to_list_of_ints(hf_ds["episode_index"])
    frame_indices = _to_list_of_ints(hf_ds["frame_index"])

    n_rows = len(indices)
    if not (len(episodes) == n_rows and len(frame_indices) == n_rows):
        errors.append(
            "Column length mismatch: "
            f"index={len(indices)}, episode_index={len(episodes)}, frame_index={len(frame_indices)}."
        )
        return errors

    if n_rows == 0:
        errors.append("Dataset has zero rows.")
        return errors

    # Global index checks.
    if indices[0] != 0:
        errors.append(f"Global index should start at 0 but starts at {indices[0]}.")

    prev_idx = indices[0]
    for i in range(1, n_rows):
        cur_idx = indices[i]
        if cur_idx <= prev_idx:
            if cur_idx == prev_idx:
                errors.append(f"Duplicate global index {cur_idx} at row {i}.")
            else:
                errors.append(
                    f"Global index decreases at row {i}: previous={prev_idx}, current={cur_idx}."
                )
        if cur_idx != prev_idx + 1:
            errors.append(
                "Global index gap detected at row "
                f"{i}: expected {prev_idx + 1}, found {cur_idx}."
            )
        prev_idx = cur_idx

    expected_last = n_rows - 1
    if indices[-1] != expected_last:
        errors.append(
            "Global index end mismatch: "
            f"expected last index {expected_last}, found {indices[-1]}."
        )

    # Build per-episode view from data rows.
    by_episode: dict[int, list[tuple[int, int]]] = {}
    for idx, ep, frame in zip(indices, episodes, frame_indices, strict=True):
        by_episode.setdefault(ep, []).append((idx, frame))

    observed_episode_ids = sorted(by_episode)
    expected_episode_ids = list(range(len(observed_episode_ids)))
    if observed_episode_ids != expected_episode_ids:
        errors.append(
            "Episode indices are not contiguous from 0..N-1: "
            f"found {observed_episode_ids[:10]}"
            + ("..." if len(observed_episode_ids) > 10 else "")
        )

    # Per-episode checks from rows.
    for ep in observed_episode_ids:
        rows = by_episode[ep]
        rows_sorted = sorted(rows, key=lambda x: x[0])
        ep_indices = [x[0] for x in rows_sorted]
        ep_frames = [x[1] for x in rows_sorted]

        # Frame index should be contiguous from 0 within each episode.
        if ep_frames and ep_frames[0] != 0:
            errors.append(f"Episode {ep}: frame_index should start at 0, found {ep_frames[0]}.")
        for i in range(1, len(ep_frames)):
            if ep_frames[i] != ep_frames[i - 1] + 1:
                errors.append(
                    f"Episode {ep}: frame_index discontinuity at local row {i}: "
                    f"expected {ep_frames[i - 1] + 1}, found {ep_frames[i]}."
                )
                break

        # Within an episode, global indices should also be contiguous.
        for i in range(1, len(ep_indices)):
            if ep_indices[i] != ep_indices[i - 1] + 1:
                errors.append(
                    f"Episode {ep}: global index discontinuity inside episode at local row {i}: "
                    f"expected {ep_indices[i - 1] + 1}, found {ep_indices[i]}."
                )
                break

        # Common invariant in LeRobot datasets: index == episode_start + frame_index.
        start_idx = ep_indices[0]
        for idx_val, frame_val in zip(ep_indices, ep_frames, strict=True):
            if idx_val != start_idx + frame_val:
                errors.append(
                    f"Episode {ep}: index/frame mismatch at global index {idx_val}: "
                    f"expected frame {idx_val - start_idx}, found {frame_val}."
                )
                break

    # Cross-check with metadata episode ranges.
    meta_eps = ds.meta.episodes
    meta_episode_ids = _to_list_of_ints(meta_eps["episode_index"])
    meta_from = _to_list_of_ints(meta_eps["dataset_from_index"])
    meta_to = _to_list_of_ints(meta_eps["dataset_to_index"])

    meta_by_ep = {ep: (start, end) for ep, start, end in zip(meta_episode_ids, meta_from, meta_to, strict=True)}

    for ep in observed_episode_ids:
        if ep not in meta_by_ep:
            errors.append(f"Episode {ep} is present in data but missing from metadata.")
            continue

        start, end = meta_by_ep[ep]
        if end <= start:
            errors.append(f"Episode {ep}: invalid metadata range [{start}, {end}).")
            continue

        rows = by_episode[ep]
        ep_indices = sorted(idx for idx, _ in rows)
        observed_start = ep_indices[0]
        observed_end_exclusive = ep_indices[-1] + 1
        observed_len = len(ep_indices)
        expected_len = end - start

        if observed_start != start or observed_end_exclusive != end:
            errors.append(
                f"Episode {ep}: metadata range [{start}, {end}) does not match "
                f"observed indices [{observed_start}, {observed_end_exclusive})."
            )

        if observed_len != expected_len:
            errors.append(
                f"Episode {ep}: metadata length {expected_len} does not match observed {observed_len}."
            )

    # Metadata episodes not present in data.
    for ep in sorted(meta_by_ep):
        if ep not in by_episode:
            errors.append(f"Episode {ep} is present in metadata but has no rows in data.")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check LeRobot dataset index consistency and filtering/subsetting irregularities."
    )
    parser.add_argument("repo_id", help="Dataset id in form username/datasetname")
    parser.add_argument("--root", type=Path, default=None, help="Optional local dataset root override")
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 if any issue is found")
    args = parser.parse_args()

    issues = validate_dataset_indices(repo_id=args.repo_id, root=args.root)

    if issues:
        print(f"Found {len(issues)} index irregularit{'y' if len(issues) == 1 else 'ies'} in {args.repo_id}:")
        for i, issue in enumerate(issues, start=1):
            print(f"{i:>3}. {issue}")
        if args.strict:
            raise SystemExit(2)
    else:
        print(f"OK: index fields are consistent for dataset '{args.repo_id}'.")


if __name__ == "__main__":
    main()
