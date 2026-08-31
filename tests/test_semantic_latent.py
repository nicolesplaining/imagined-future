from __future__ import annotations

import pytest
import torch

from imagined_future.semantic_latent import raw_frame_span, splice_target_images


def test_raw_frame_span_matches_one_plus_t_layout() -> None:
    assert raw_frame_span(0) == slice(0, 1)
    assert raw_frame_span(1) == slice(1, 5)
    assert raw_frame_span(6) == slice(21, 25)
    assert raw_frame_span(7) == slice(25, 29)
    with pytest.raises(ValueError):
        raw_frame_span(-1)


def test_splice_target_images_changes_only_future_camera_raw_frames() -> None:
    recipient_video = torch.zeros(1, 3, 33, 1, 1)
    target_video = torch.arange(33, dtype=torch.float32).reshape(1, 1, 33, 1, 1).repeat(1, 3, 1, 1, 1)
    indices = {
        "current_wrist_image_latent_idx": torch.tensor([2]),
        "current_image_latent_idx": torch.tensor([3]),
        "future_wrist_image_latent_idx": torch.tensor([6]),
        "future_image_latent_idx": torch.tensor([7]),
    }
    recipient = {"video": recipient_video, **indices}
    target = {"video": target_video, **indices}

    result = splice_target_images(recipient, target)

    assert torch.equal(recipient["video"], torch.zeros_like(recipient_video))
    assert torch.equal(result["video"][:, :, 21:25], target_video[:, :, 5:9])
    assert torch.equal(result["video"][:, :, 25:29], target_video[:, :, 9:13])
    unchanged = torch.ones(33, dtype=torch.bool)
    unchanged[21:29] = False
    assert torch.count_nonzero(result["video"][:, :, unchanged]) == 0
