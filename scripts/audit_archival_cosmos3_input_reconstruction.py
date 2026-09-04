#!/usr/bin/env python3
"""Audit Cosmos current-frame reconstruction from archived RoboLab MP4s."""

from __future__ import annotations

import argparse
import json
import math
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def remap_recording(path_text: str, screen_root: Path) -> Path:
    marker = "cosmos3_population_screen_v1/"
    if marker not in path_text:
        raise ValueError(f"cannot map archived recording path: {path_text}")
    return screen_root / path_text.split(marker, 1)[1]


def read_frames(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open {path}")
    wanted = sorted(index for index in indices if index >= 0)
    frames: dict[int, np.ndarray] = {}
    try:
        next_wanted = 0
        frame_index = 0
        while next_wanted < len(wanted):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index == wanted[next_wanted]:
                frames[frame_index] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                next_wanted += 1
            frame_index += 1
    finally:
        capture.release()
    missing = set(wanted) - set(frames)
    if missing:
        raise RuntimeError(f"missing frames {sorted(missing)} from {path}")
    return frames


def half_size(image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(180, 320), mode="bilinear")
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(image.dtype)


def compose(frame: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    if frame.shape != (360, 2560, 3):
        raise ValueError(f"unexpected archived frame shape: {frame.shape}")
    panels = np.split(frame, 4, axis=1)
    wrist_index, left_index, right_index = order
    bottom = np.concatenate((half_size(panels[left_index]), half_size(panels[right_index])), axis=1)
    return np.concatenate((panels[wrist_index], bottom), axis=0)


def metrics(value: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    difference = value.astype(np.float32) - target.astype(np.float32)
    absolute = np.abs(difference)
    mse = float(np.square(difference).mean())
    return {
        "maximum_absolute_error": int(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "root_mean_squared_error": math.sqrt(mse),
        "exact_pixel_channel_fraction": float(np.mean(difference == 0)),
        "psnr_db": float("inf") if mse == 0.0 else float(20 * math.log10(255) - 10 * math.log10(mse)),
    }


def main() -> None:
    args = parse_args()
    expected_order = (3, 1, 2)  # MP4 panels: head, left, right, wrist.
    rows = []
    for summary_path in sorted(args.cohort_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        self_path = summary_path.parent / "self.npz"
        if not self_path.exists():
            raise FileNotFoundError(self_path)
        with np.load(self_path, allow_pickle=False) as payload:
            target = np.asarray(payload["video"][0], dtype=np.uint8)
        if target.shape != (540, 640, 3):
            raise ValueError(f"unexpected target shape at {self_path}: {target.shape}")

        recording = remap_recording(str(summary["recorded_hdf5"]), args.screen_root)
        videos = sorted(recording.parent.glob("*.mp4"))
        if len(videos) != 1:
            raise RuntimeError(f"expected one MP4 beside {recording}, got {videos}")
        expected_index = int(summary["branch_step"]) - 1
        candidate_indices = set(range(max(0, expected_index - 2), expected_index + 3))
        decoded = read_frames(videos[0], candidate_indices)

        expected = compose(decoded[expected_index], expected_order)
        expected_metrics = metrics(expected, target)
        candidates = []
        for frame_index, frame in decoded.items():
            for order in permutations(range(4), 3):
                candidate = compose(frame, order)
                candidate_metrics = metrics(candidate, target)
                candidates.append(
                    {
                        "frame_index": frame_index,
                        "frame_offset_from_branch_minus_one": frame_index - expected_index,
                        "order_wrist_left_right": list(order),
                        **candidate_metrics,
                    }
                )
        best = min(
            candidates,
            key=lambda row: (
                row["root_mean_squared_error"],
                abs(row["frame_offset_from_branch_minus_one"]),
                row["order_wrist_left_right"],
            ),
        )
        rows.append(
            {
                "state": summary_path.parent.name,
                "task": summary["task"],
                "environment_seed": summary["environment_seed"],
                "branch_step": summary["branch_step"],
                "expected_mp4_frame_index": expected_index,
                "mp4": str(videos[0]),
                "self_npz": str(self_path),
                "expected_order_wrist_left_right": list(expected_order),
                "expected": expected_metrics,
                "best": best,
                "expected_is_best": (
                    best["frame_index"] == expected_index
                    and best["order_wrist_left_right"] == list(expected_order)
                ),
            }
        )

    report = {
        "scope": "input reconstruction audit only; no new model intervention outcome",
        "cohort_state_count": len(rows),
        "mp4_panel_order": ["head", "over_shoulder_left", "over_shoulder_right", "wrist"],
        "reconstruction": (
            "MP4 frame branch_step-1; wrist panel full resolution above bilinear half-scale "
            "left/right panels; H.264 source is lossy"
        ),
        "expected_is_best_count": sum(row["expected_is_best"] for row in rows),
        "expected_mean_absolute_error_mean": float(
            np.mean([row["expected"]["mean_absolute_error"] for row in rows])
        ),
        "expected_mean_absolute_error_median": float(
            np.median([row["expected"]["mean_absolute_error"] for row in rows])
        ),
        "expected_rmse_mean": float(
            np.mean([row["expected"]["root_mean_squared_error"] for row in rows])
        ),
        "expected_psnr_db_mean": float(np.mean([row["expected"]["psnr_db"] for row in rows])),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
