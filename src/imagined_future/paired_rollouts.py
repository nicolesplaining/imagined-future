"""Metrics for paired natural rollouts from the same LIBERO initial state."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PIL import Image


def pixel_l1(left: np.ndarray, right: np.ndarray) -> float:
    """Mean absolute pixel difference on the native 0--255 scale."""

    if left.shape != right.shape:
        raise ValueError(f"image shapes differ: {left.shape} != {right.shape}")
    return float(np.abs(left.astype(np.float64) - right.astype(np.float64)).mean())


def resize_uint8_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize an RGB image to ``(height, width)`` with explicit bilinear sampling."""

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HxWx3 uint8 image")
    height, width = shape
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(Image.fromarray(image).resize((width, height), resampling.BILINEAR))


def paired_query_metrics(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], index: int) -> dict:
    """Calculate symmetric action, state, future, and realization distances."""

    left_prediction = left["predicted_primary_images"][index]
    right_prediction = right["predicted_primary_images"][index]
    left_endpoint = left["endpoint_primary_images"][index]
    right_endpoint = right["endpoint_primary_images"][index]
    prediction_shape = left_prediction.shape[:2]
    if right_prediction.shape[:2] != prediction_shape:
        raise ValueError("paired predicted images use different resolutions")
    left_endpoint_resized = resize_uint8_image(left_endpoint, prediction_shape)
    right_endpoint_resized = resize_uint8_image(right_endpoint, prediction_shape)
    left_own_error = pixel_l1(left_prediction, left_endpoint_resized)
    right_own_error = pixel_l1(right_prediction, right_endpoint_resized)
    left_cross_error = pixel_l1(left_prediction, right_endpoint_resized)
    right_cross_error = pixel_l1(right_prediction, left_endpoint_resized)
    return {
        "query_index": index,
        "query_step": int(left["query_steps"][index]),
        "current_state_l2": float(
            np.linalg.norm(
                left["current_states"][index].astype(np.float64)
                - right["current_states"][index].astype(np.float64)
            )
        ),
        "current_primary_pixel_l1": pixel_l1(
            left["current_primary_images"][index], right["current_primary_images"][index]
        ),
        "normalized_action_l2": float(
            np.linalg.norm(
                left["normalized_action_chunks"][index].astype(np.float64)
                - right["normalized_action_chunks"][index].astype(np.float64)
            )
        ),
        "predicted_primary_pixel_l1": pixel_l1(left_prediction, right_prediction),
        "endpoint_primary_pixel_l1": pixel_l1(left_endpoint, right_endpoint),
        "left_prediction_own_endpoint_l1": left_own_error,
        "right_prediction_own_endpoint_l1": right_own_error,
        "left_prediction_cross_endpoint_l1": left_cross_error,
        "right_prediction_cross_endpoint_l1": right_cross_error,
        "mean_own_endpoint_advantage": float(
            ((left_cross_error - left_own_error) + (right_cross_error - right_own_error)) / 2
        ),
    }
