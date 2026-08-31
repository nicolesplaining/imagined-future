import numpy as np

from imagined_future.paired_rollouts import paired_query_metrics, pixel_l1, resize_uint8_image


def test_pixel_l1_uses_native_scale():
    assert pixel_l1(np.array([0, 255], dtype=np.uint8), np.array([2, 250], dtype=np.uint8)) == 3.5


def test_resize_uint8_image_uses_requested_height_width():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    assert resize_uint8_image(image, (2, 3)).shape == (2, 3, 3)


def test_paired_query_metrics_reports_own_endpoint_advantage():
    left = {
        "query_steps": np.array([0]),
        "current_states": np.zeros((1, 2)),
        "current_primary_images": np.zeros((1, 1, 1, 3), dtype=np.uint8),
        "normalized_action_chunks": np.zeros((1, 1, 1)),
        "predicted_primary_images": np.zeros((1, 1, 1, 3), dtype=np.uint8),
        "endpoint_primary_images": np.zeros((1, 1, 1, 3), dtype=np.uint8),
    }
    right = {
        **left,
        "current_states": np.array([[3.0, 4.0]]),
        "normalized_action_chunks": np.ones((1, 1, 1)),
        "predicted_primary_images": np.full((1, 1, 1, 3), 10, dtype=np.uint8),
        "endpoint_primary_images": np.full((1, 1, 1, 3), 10, dtype=np.uint8),
    }
    metrics = paired_query_metrics(left, right, 0)
    assert metrics["current_state_l2"] == 5.0
    assert metrics["normalized_action_l2"] == 1.0
    assert metrics["predicted_primary_pixel_l1"] == 10.0
    assert metrics["mean_own_endpoint_advantage"] == 10.0
