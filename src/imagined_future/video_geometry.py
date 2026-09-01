"""Geometry helpers for auditing tokenizer video round trips."""

from __future__ import annotations

import numpy as np


def center_crop_video(video: np.ndarray, height: int, width: int) -> np.ndarray:
    """Center-crop ``[T,H,W,C]`` video to a tokenizer-decoded spatial shape."""

    if video.ndim != 4:
        raise ValueError(f"expected [T,H,W,C] video, got shape {video.shape}")
    source_height, source_width = video.shape[1:3]
    if height <= 0 or width <= 0:
        raise ValueError("crop dimensions must be positive")
    if height > source_height or width > source_width:
        raise ValueError(
            f"cannot crop {source_height}x{source_width} video to {height}x{width}"
        )
    top = (source_height - height) // 2
    left = (source_width - width) // 2
    return video[:, top : top + height, left : left + width, :]


def difference_alignment(
    raw_low: np.ndarray,
    raw_high: np.ndarray,
    decoded_low: np.ndarray,
    decoded_high: np.ndarray,
) -> dict[str, float | bool]:
    """Measure whether an encode/decode round trip preserves a paired edit."""

    shapes = {array.shape for array in (raw_low, raw_high, decoded_low, decoded_high)}
    if len(shapes) != 1:
        raise ValueError(f"paired videos must have one common shape, got {sorted(shapes)}")
    raw_difference = raw_high.astype(np.float32) - raw_low.astype(np.float32)
    decoded_difference = decoded_high.astype(np.float32) - decoded_low.astype(np.float32)
    raw_squared_norm = float(np.sum(raw_difference * raw_difference, dtype=np.float64))
    decoded_squared_norm = float(
        np.sum(decoded_difference * decoded_difference, dtype=np.float64)
    )
    if raw_squared_norm == 0.0 or decoded_squared_norm == 0.0:
        cosine = 0.0
        projection = 0.0
    else:
        dot = float(np.sum(raw_difference * decoded_difference, dtype=np.float64))
        cosine = float(dot / np.sqrt(raw_squared_norm * decoded_squared_norm))
        projection = float(dot / raw_squared_norm)
    return {
        "raw_decoded_difference_cosine": cosine,
        "decoded_on_raw_difference_projection": projection,
        "direction_preserved": bool(cosine > 0.0),
    }
