from __future__ import annotations

import pytest
import torch

from imagined_future.frames import LatentFrameGroups


def test_resolves_libero_frame_groups() -> None:
    batch = {
        "current_proprio_latent_idx": torch.tensor([1, 1]),
        "current_wrist_image_latent_idx": torch.tensor([2, 2]),
        "current_wrist_image2_latent_idx": torch.tensor([-1, -1]),
        "current_image_latent_idx": torch.tensor([3, 3]),
        "current_image2_latent_idx": torch.tensor([-1, -1]),
        "action_latent_idx": torch.tensor([4, 4]),
        "future_proprio_latent_idx": torch.tensor([5, 5]),
        "future_wrist_image_latent_idx": torch.tensor([6, 6]),
        "future_wrist_image2_latent_idx": torch.tensor([-1, -1]),
        "future_image_latent_idx": torch.tensor([7, 7]),
        "future_image2_latent_idx": torch.tensor([-1, -1]),
        "value_latent_idx": torch.tensor([8, 8]),
    }

    groups = LatentFrameGroups.from_batch(batch)

    assert groups.current == (1, 2, 3)
    assert groups.action == (4,)
    assert groups.future == (5, 6, 7)
    assert groups.value == (8,)


def test_rejects_mixed_indices_in_paired_batch() -> None:
    with pytest.raises(ValueError, match="constant"):
        LatentFrameGroups.from_batch({"action_latent_idx": torch.tensor([4, 5])})
