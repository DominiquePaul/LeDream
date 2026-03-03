#!/usr/bin/env python
"""Pre-decode all video frames in a LeRobot dataset to a JPEG image cache.

This eliminates the expensive per-sample H.264 video decode during training.
Random video access requires decoding from the nearest keyframe (often ~100+
intermediate frames), making each __getitem__ ~100x slower than a JPEG read.

Usage:
    python scripts/predecode_videos.py --repo-id dopaul/pcb_placement_100x_1st_item

The cache is stored alongside the videos at:
    {dataset_root}/image_cache/{camera_key}/frame_{global_idx:06d}.jpg

Training will automatically use the cache when present (see lerobot_dataset.py).
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def decode_video_sequential(video_path: str, frame_indices: list[int],
                            global_indices: list[int], output_dir: Path,
                            quality: int = 95) -> int:
    """Decode all requested frames from one video file sequentially.

    Sequential decoding is ~100x faster than random access because H.264
    only needs to decode forward from keyframes once.
    """
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path, seek_mode="approximate")
    count = 0

    # Decode frames in sorted order for sequential access efficiency
    pairs = sorted(zip(frame_indices, global_indices))
    batch_size = 64
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start:batch_start + batch_size]
        f_indices = [p[0] for p in batch]
        g_indices = [p[1] for p in batch]

        frames_batch = decoder.get_frames_at(indices=f_indices)
        for frame, gidx in zip(frames_batch.data, g_indices):
            out_path = output_dir / f"frame_{gidx:06d}.jpg"
            if out_path.exists():
                continue
            frame_np = frame.permute(1, 2, 0).numpy()
            Image.fromarray(frame_np).save(str(out_path), "JPEG", quality=quality)
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Pre-decode video frames to JPEG cache")
    parser.add_argument("--repo-id", required=True, help="HuggingFace dataset repo ID")
    parser.add_argument("--root", default=None, help="Local dataset root (auto-detected if omitted)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logging.info(f"Loading dataset metadata: {args.repo_id}")
    ds = LeRobotDataset(args.repo_id, root=args.root)

    cache_dir = ds.root / "image_cache"
    video_keys = ds.meta.video_keys
    num_episodes = ds.num_episodes

    logging.info(f"Dataset: {ds.num_frames} frames, {num_episodes} episodes, {len(video_keys)} cameras")
    logging.info(f"Cache dir: {cache_dir}")

    # Group frames by (camera, video_file) for sequential decoding
    video_tasks: dict[tuple[str, str], tuple[list[int], list[int]]] = {}

    for vid_key in video_keys:
        cam_cache = cache_dir / vid_key
        cam_cache.mkdir(parents=True, exist_ok=True)

        for ep_idx in range(num_episodes):
            ep = ds.meta.episodes[ep_idx]
            video_path = str(ds.root / ds.meta.get_video_file_path(ep_idx, vid_key))
            from_ts = ep[f"videos/{vid_key}/from_timestamp"]

            metadata_key = f"videos/{vid_key}/from_index"
            if metadata_key in ep:
                from_frame = ep[metadata_key]
            else:
                from_frame = round(from_ts * 30)

            ep_from_idx = ep["dataset_from_index"]
            ep_to_idx = ep["dataset_to_index"]

            task_key = (vid_key, video_path)
            if task_key not in video_tasks:
                video_tasks[task_key] = ([], [])

            for i, global_idx in enumerate(range(ep_from_idx, ep_to_idx)):
                video_tasks[task_key][0].append(from_frame + i)
                video_tasks[task_key][1].append(global_idx)

    total_frames = sum(len(v[0]) for v in video_tasks.values())
    logging.info(f"Total frames to decode: {total_frames} across {len(video_tasks)} video files")

    t0 = time.time()
    decoded = 0

    for (vid_key, video_path), (frame_indices, global_indices) in tqdm(
        video_tasks.items(), desc="Video files", unit="file"
    ):
        cam_cache = cache_dir / vid_key
        n = decode_video_sequential(
            video_path, frame_indices, global_indices, cam_cache, args.quality
        )
        decoded += n

    elapsed = time.time() - t0
    logging.info(f"Done: {decoded} frames decoded in {elapsed:.1f}s ({decoded/max(elapsed,1):.0f} frames/s)")

    import subprocess
    result = subprocess.run(["du", "-sh", str(cache_dir)], capture_output=True, text=True)
    logging.info(f"Cache size: {result.stdout.strip()}")


if __name__ == "__main__":
    main()
