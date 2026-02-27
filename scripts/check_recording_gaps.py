#!/usr/bin/env python3
"""Validate recorded LeRobot data quality and timing consistency.

Checks performed:
- Timestamp spacing vs expected FPS
- Joint-state smoothness (spike/outlier jumps)
- Action zero-value anomalies
- TCP pose consistency with joint state (optional, requires TCP key and FK fn)

Usage:
    # Basic check on all episodes from a dataset
    python scripts/check_recording_gaps.py \
        --repo-id dopaul/pcb_placement_v1

    # Also validate expected episode duration (frames ~= fps * duration)
    .venv/bin/python scripts/check_recording_gaps.py \
        --repo-id dopaul/pcb_placement_v1 \
        --expected-episode-time-s 5.0

    # Check only specific episodes
    .venv/bin/python scripts/check_recording_gaps.py \
        --repo-id dopaul/pcb_placement_v1 \
        --episodes 0,1

    # Optional TCP-vs-FK consistency check (when dataset has a TCP key)
    .venv/bin/python scripts/check_recording_gaps.py \
        --repo-id dopaul/pcb_placement_v1 \
        --tcp-key observation.tcp_pose \
        --fk-module path/to/kinematics.py \
        --fk-function forward_kinematics

Notes:
    - Use --root if the dataset is stored outside the default cache location.
    - Use --strict to return exit code 2 when anomalies are detected.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _parse_episodes(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(tok.strip()) for tok in raw.split(",") if tok.strip()]


def _find_default_key(keys: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in keys:
            return c
    return None


def _auto_find_tcp_key(keys: list[str]) -> str | None:
    lowered = {k: k.lower() for k in keys}
    exact = [
        "observation.tcp_pose",
        "observation.ee_pose",
        "observation.tcp",
        "tcp_pose",
        "ee_pose",
    ]
    for k in exact:
        if k in lowered:
            return k
    for key, lkey in lowered.items():
        if "tcp" in lkey or "ee" in lkey:
            if "pose" in lkey or "state" in lkey or "position" in lkey:
                return key
    return None


def _safe_mad(arr: np.ndarray) -> np.ndarray:
    med = np.median(arr, axis=0)
    return np.median(np.abs(arr - med), axis=0)


@dataclass
class EpisodeResult:
    episode: int
    n_frames: int
    timestamp_violations: int
    timestamp_max_gap_ratio: float
    smooth_spikes: int
    jerk_spikes: int
    action_all_zero_frames: int
    action_any_zero_ratio_max: float
    tcp_check_run: bool
    tcp_mae: float | None
    tcp_rmse: float | None
    notes: list[str]


def _load_fk_callable(fk_module: str | None, fk_function: str | None) -> Callable[[np.ndarray], np.ndarray] | None:
    if not fk_module:
        return None
    if not fk_function:
        raise ValueError("--fk-function is required when --fk-module is set.")

    module_obj = None
    module_path = Path(fk_module)
    if module_path.exists():
        spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load module from path: {fk_module}")
        module_obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_obj)
    else:
        module_obj = importlib.import_module(fk_module)

    fn = getattr(module_obj, fk_function, None)
    if fn is None:
        raise ValueError(f"Function '{fk_function}' not found in module '{fk_module}'.")
    return fn


def validate_episode(
    ep_idx: int,
    rows: list[dict[str, Any]],
    expected_fps: float,
    joint_key: str,
    action_key: str,
    tcp_key: str | None,
    action_zero_eps: float,
    timestamp_tol_s: float,
    smooth_spike_sigma: float,
    expected_episode_time_s: float | None,
    fk_fn: Callable[[np.ndarray], np.ndarray] | None,
) -> EpisodeResult:
    notes: list[str] = []
    n_frames = len(rows)

    timestamps = np.array([float(_to_numpy(r["timestamp"])) for r in rows], dtype=np.float64)
    joints = np.stack([_to_numpy(r[joint_key]).astype(np.float64) for r in rows], axis=0)
    actions = np.stack([_to_numpy(r[action_key]).astype(np.float64) for r in rows], axis=0)

    # 1) Timestamp quality.
    expected_dt = 1.0 / expected_fps
    dts = np.diff(timestamps)
    if dts.size == 0:
        timestamp_violations = 0
        max_gap_ratio = 1.0
    else:
        too_small = dts < (expected_dt - timestamp_tol_s)
        too_large = dts > (expected_dt + timestamp_tol_s)
        timestamp_violations = int(np.count_nonzero(too_small | too_large))
        max_gap_ratio = float(np.max(dts / expected_dt))
        if np.any(dts <= 0):
            notes.append("Non-monotonic timestamps detected.")

    # 2) Joint smoothness.
    if joints.shape[0] < 3:
        smooth_spikes = 0
        jerk_spikes = 0
    else:
        joint_deltas = np.abs(np.diff(joints, axis=0))
        med = np.median(joint_deltas, axis=0)
        mad = _safe_mad(joint_deltas)
        sigma = 1.4826 * np.maximum(mad, 1e-9)
        threshold = med + smooth_spike_sigma * sigma
        smooth_spikes = int(np.count_nonzero(np.any(joint_deltas > threshold, axis=1)))

        # Jerk-like spikes are more indicative of discontinuities than pure velocity.
        joint_vel = np.diff(joints, axis=0)
        joint_acc = np.diff(joint_vel, axis=0)
        acc_norm = np.linalg.norm(joint_acc, axis=1)
        if acc_norm.size == 0:
            jerk_spikes = 0
        else:
            acc_med = np.median(acc_norm)
            acc_mad = np.median(np.abs(acc_norm - acc_med))
            acc_sigma = max(1.4826 * acc_mad, 1e-9)
            jerk_threshold = acc_med + smooth_spike_sigma * acc_sigma
            jerk_spikes = int(np.count_nonzero(acc_norm > jerk_threshold))

    # 3) Action zeros.
    abs_actions = np.abs(actions)
    all_zero_mask = np.all(abs_actions <= action_zero_eps, axis=1)
    all_zero_frames = int(np.count_nonzero(all_zero_mask))
    dim_zero_ratio = np.mean(abs_actions <= action_zero_eps, axis=0)
    any_zero_ratio_max = float(np.max(dim_zero_ratio))

    # 4) TCP vs FK(joints), optional.
    tcp_check_run = False
    tcp_mae = None
    tcp_rmse = None
    if tcp_key and tcp_key in rows[0] and fk_fn is not None:
        tcp_check_run = True
        tcp_obs = np.stack([_to_numpy(r[tcp_key]).astype(np.float64) for r in rows], axis=0)
        tcp_pred = np.asarray(fk_fn(joints), dtype=np.float64)
        if tcp_pred.shape != tcp_obs.shape:
            notes.append(
                f"FK output shape mismatch: predicted {tcp_pred.shape}, observed {tcp_obs.shape}. Skipping TCP metric."
            )
        else:
            err = tcp_pred - tcp_obs
            tcp_mae = float(np.mean(np.abs(err)))
            tcp_rmse = float(math.sqrt(np.mean(err**2)))
    elif tcp_key and tcp_key in rows[0] and fk_fn is None:
        notes.append("TCP key is present but FK function not provided; TCP consistency skipped.")
    elif tcp_key and tcp_key not in rows[0]:
        notes.append(f"TCP key '{tcp_key}' not found in rows.")

    if expected_episode_time_s is not None:
        expected_frames = int(round(expected_episode_time_s * expected_fps))
        frame_delta = n_frames - expected_frames
        if frame_delta != 0:
            notes.append(
                f"Episode length mismatch: got {n_frames} frames, expected ~{expected_frames} at {expected_fps:.2f} FPS."
            )

    return EpisodeResult(
        episode=ep_idx,
        n_frames=n_frames,
        timestamp_violations=timestamp_violations,
        timestamp_max_gap_ratio=max_gap_ratio,
        smooth_spikes=smooth_spikes,
        jerk_spikes=jerk_spikes,
        action_all_zero_frames=all_zero_frames,
        action_any_zero_ratio_max=any_zero_ratio_max,
        tcp_check_run=tcp_check_run,
        tcp_mae=tcp_mae,
        tcp_rmse=tcp_rmse,
        notes=notes,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Check LeRobot recording quality for timing and signal anomalies.")
    p.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. dopaul/pcb_placement_v1")
    p.add_argument("--root", type=Path, default=None, help="Local dataset root override")
    p.add_argument("--episodes", default=None, help="Comma-separated episode indices, e.g. 0,1,2")
    p.add_argument("--expected-fps", type=float, default=None, help="Override expected fps (default: dataset fps)")
    p.add_argument("--joint-key", default=None, help="Joint-state key (default: auto -> observation.state)")
    p.add_argument("--action-key", default=None, help="Action key (default: auto -> action)")
    p.add_argument("--tcp-key", default=None, help="TCP pose key (default: auto detect if present)")
    p.add_argument("--action-zero-eps", type=float, default=1e-6, help="Abs threshold to consider action value zero")
    p.add_argument(
        "--timestamp-tol-s",
        type=float,
        default=5e-3,
        help="Allowed absolute timestamp jitter around 1/fps in seconds",
    )
    p.add_argument(
        "--smooth-spike-sigma",
        type=float,
        default=10.0,
        help="Outlier threshold: median + sigma * robust_std on |delta joint|",
    )
    p.add_argument("--fk-module", default=None, help="Module path or import path providing FK function")
    p.add_argument("--fk-function", default=None, help="FK function name. Signature: fn(joints[N,D])->pose[N,M]")
    p.add_argument(
        "--expected-episode-time-s",
        type=float,
        default=None,
        help="Expected episode duration in seconds to flag missing/extra frames.",
    )
    p.add_argument("--strict", action="store_true", help="Exit non-zero if checks fail thresholds")
    args = p.parse_args()

    episodes = _parse_episodes(args.episodes)
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes)
    expected_fps = float(args.expected_fps or ds.fps)

    available_keys = list(ds.features.keys())
    joint_key = args.joint_key or _find_default_key(available_keys, ["observation.state"])
    action_key = args.action_key or _find_default_key(available_keys, ["action"])
    tcp_key = args.tcp_key or _auto_find_tcp_key(available_keys)

    if joint_key is None or action_key is None:
        raise ValueError(
            f"Could not resolve required keys. joint_key={joint_key}, action_key={action_key}. "
            f"Available keys: {available_keys}"
        )

    fk_fn = _load_fk_callable(args.fk_module, args.fk_function)

    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    needed = {"timestamp", "episode_index", joint_key, action_key}
    if tcp_key:
        needed.add(tcp_key)

    # Iterate without image/video decode by only selecting scalar/vector columns we need.
    hf_ds = ds.hf_dataset.select_columns(list(needed))
    for i in range(len(hf_ds)):
        row = hf_ds[i]
        ep = int(_to_numpy(row["episode_index"]))
        rows_by_episode.setdefault(ep, []).append(row)

    results: list[EpisodeResult] = []
    for ep_idx in sorted(rows_by_episode):
        results.append(
            validate_episode(
                ep_idx=ep_idx,
                rows=rows_by_episode[ep_idx],
                expected_fps=expected_fps,
                joint_key=joint_key,
                action_key=action_key,
                tcp_key=tcp_key,
                action_zero_eps=args.action_zero_eps,
                timestamp_tol_s=args.timestamp_tol_s,
                smooth_spike_sigma=args.smooth_spike_sigma,
                expected_episode_time_s=args.expected_episode_time_s,
                fk_fn=fk_fn,
            )
        )

    print(f"Dataset: {args.repo_id}")
    print(f"Resolved keys: joint='{joint_key}', action='{action_key}', tcp='{tcp_key}'")
    print(f"Expected FPS: {expected_fps:.3f}")
    print("-" * 100)
    print("episode frames ts_viol ts_max_gap(xdt) smooth_spikes jerk_spikes all_zero_actions max_dim_zero_ratio tcp_check tcp_mae tcp_rmse")
    for r in results:
        print(
            f"{r.episode:>7} {r.n_frames:>6} {r.timestamp_violations:>7} "
            f"{r.timestamp_max_gap_ratio:>14.3f} {r.smooth_spikes:>13} {r.jerk_spikes:>11} {r.action_all_zero_frames:>16} "
            f"{r.action_any_zero_ratio_max:>18.3f} {str(r.tcp_check_run):>9} "
            f"{'-' if r.tcp_mae is None else f'{r.tcp_mae:.6f}':>7} "
            f"{'-' if r.tcp_rmse is None else f'{r.tcp_rmse:.6f}':>8}"
        )
        for n in r.notes:
            print(f"  - note: {n}")

    # Strict mode: fail on clear anomalies.
    if args.strict:
        fail = False
        for r in results:
            if r.timestamp_violations > 0:
                fail = True
            if r.smooth_spikes > 0:
                fail = True
            if r.jerk_spikes > 0:
                fail = True
            if r.action_all_zero_frames > 0:
                fail = True
        if fail:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
