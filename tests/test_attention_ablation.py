from __future__ import annotations

import pytest
import torch

from imagined_future.attention_ablation import gated_attention_output, temporal_token_indices


def test_temporal_token_indices_follow_thw_flattening() -> None:
    assert temporal_token_indices((1, 3), time=4, height=2, width=2) == (
        4,
        5,
        6,
        7,
        12,
        13,
        14,
        15,
    )


def test_temporal_token_indices_reject_invalid_frame() -> None:
    with pytest.raises(IndexError, match="invalid"):
        temporal_token_indices((4,), time=4, height=2, width=2)


def test_gated_attention_output_has_registered_endpoints() -> None:
    original = torch.tensor([1.0, 3.0])
    ablated = torch.tensor([5.0, -1.0])
    assert torch.equal(gated_attention_output(original, ablated, 0.0), original)
    assert torch.equal(gated_attention_output(original, ablated, 1.0), ablated)
    assert torch.equal(gated_attention_output(original, ablated, 0.25), torch.tensor([2.0, 2.0]))


def test_gated_attention_output_rejects_extrapolation() -> None:
    with pytest.raises(ValueError, match="within"):
        gated_attention_output(torch.zeros(1), torch.ones(1), 1.1)
