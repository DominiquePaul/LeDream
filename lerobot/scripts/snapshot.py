#!/usr/bin/env python3
"""Capture a single frame from each camera and save as PNG."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def set_v4l2_controls(device: str, controls: dict[str, int]) -> None:
    """Set V4L2 controls via v4l2-ctl (more reliable than OpenCV's cap.set)."""
    ctrl_str = ",".join(f"{k}={v}" for k, v in controls.items())
    subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", ctrl_str],
                   capture_output=True)


def reset_v4l2_defaults(device: str) -> None:
    """Reset all V4L2 controls to their per-device hardware defaults."""
    result = subprocess.run(
        ["v4l2-ctl", "-d", device, "--list-ctrls"],
        capture_output=True, text=True,
    )
    controls: dict[str, int] = {}
    for line in result.stdout.splitlines():
        tokens = line.split()
        if len(tokens) < 2:
            continue
        name = tokens[0]
        for tok in tokens:
            if tok.startswith("default="):
                try:
                    controls[name] = int(tok.split("=", 1)[1])
                except ValueError:
                    pass
                break
    if controls:
        set_v4l2_controls(device, controls)


def capture(device: str, name: str, width: int, height: int,
            wb_temp: int | None = None, sat: int | None = None, hue: int | None = None,
            warmup: int = 60) -> np.ndarray | None:
    resolved = str(Path(device).resolve())

    if wb_temp is not None or sat is not None or hue is not None:
        controls: dict[str, int] = {"white_balance_automatic": 0}
        if wb_temp is not None:
            controls["white_balance_temperature"] = wb_temp
        if sat is not None:
            controls["saturation"] = sat
        if hue is not None:
            controls["hue"] = hue
        set_v4l2_controls(resolved, controls)
    else:
        reset_v4l2_defaults(resolved)

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"  {name}: FAILED to open {device}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(warmup):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  {name}: FAILED to capture")
        return None

    return frame


def gray_world_wb(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance: scale each channel so all means match."""
    float_img = image.astype(np.float32)
    b, g, r = cv2.split(float_img)
    avg = (r.mean() + g.mean() + b.mean()) / 3
    r = r * (avg / r.mean())
    g = g * (avg / g.mean())
    b = b * (avg / b.mean())
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def correct_ca(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Correct lateral chromatic aberration by scaling R and B channels around center."""
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    r_scale = 1.0 - 0.002 * strength
    b_scale = 1.0 + 0.002 * strength
    M_r = cv2.getRotationMatrix2D((cx, cy), 0, r_scale)
    M_b = cv2.getRotationMatrix2D((cx, cy), 0, b_scale)
    b, g, r = cv2.split(image)
    r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
    b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b, g, r])


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.putText(labeled, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(labeled, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
    return labeled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", required=True, help="Comma-separated name=path pairs, e.g. top=/dev/video0,left=/dev/video2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--ca-strength", type=float, default=0.0, help="Chromatic aberration correction strength (0=off, try 1-5)")
    parser.add_argument("--wb-temperature", type=int, default=None, help="Manual white balance temperature in Kelvin (2800-6500). Omit for auto.")
    parser.add_argument("--saturation", type=int, default=None, help="Color saturation (0-100). Omit for camera default.")
    parser.add_argument("--hue", type=int, default=None, help="Hue shift (-180 to 180). Omit for camera default.")
    parser.add_argument("--swap-cr-cb", action="store_true", help="Swap Cr/Cb chroma channels to fix YUV decoding mismatch")
    parser.add_argument("--gray-world-wb", action="store_true", help="Apply gray-world white balance correction in software")
    args = parser.parse_args()

    cameras = {}
    for pair in args.cameras.split(","):
        name, path = pair.split("=", 1)
        cameras[name.strip()] = path.strip()

    frames = {}
    for name, path in cameras.items():
        frame = capture(path, name, args.width, args.height,
                        wb_temp=args.wb_temperature, sat=args.saturation, hue=args.hue)
        if frame is not None:
            if args.swap_cr_cb:
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 1], ycrcb[:, :, 2] = ycrcb[:, :, 2].copy(), ycrcb[:, :, 1].copy()
                frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            if args.gray_world_wb:
                frame = gray_world_wb(frame)
            if args.ca_strength > 0:
                frame = correct_ca(frame, args.ca_strength)
            frames[name] = frame

    if not frames:
        print("No frames captured.")
        sys.exit(1)

    labeled = [add_label(f, n) for n, f in frames.items()]
    combined = np.hstack(labeled)

    output_dir = Path("snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(out_path), combined)
    print(out_path)


if __name__ == "__main__":
    main()
