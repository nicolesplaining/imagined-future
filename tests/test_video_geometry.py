import numpy as np
import pytest

from imagined_future.video_geometry import center_crop_video, difference_alignment


def test_center_crop_video_matches_cosmos_multiple_of_16_height() -> None:
    video = np.arange(2 * 540 * 640 * 3, dtype=np.int64).reshape(2, 540, 640, 3)

    cropped = center_crop_video(video, height=528, width=640)

    np.testing.assert_array_equal(cropped, video[:, 6:534])


def test_center_crop_video_rejects_expansion() -> None:
    with pytest.raises(ValueError, match="cannot crop"):
        center_crop_video(np.zeros((1, 10, 20, 3)), height=11, width=20)


def test_difference_alignment_detects_preserved_and_reversed_edits() -> None:
    low = np.zeros((2, 3, 4, 1), dtype=np.uint8)
    high = low.copy()
    high[:, 1, 2] = 20

    preserved = difference_alignment(low, high, low, high)
    reversed_edit = difference_alignment(low, high, high, low)

    assert preserved["direction_preserved"] is True
    assert preserved["raw_decoded_difference_cosine"] == pytest.approx(1.0)
    assert reversed_edit["direction_preserved"] is False
    assert reversed_edit["raw_decoded_difference_cosine"] == pytest.approx(-1.0)
