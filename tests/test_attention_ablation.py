from __future__ import annotations

import pytest

from imagined_future.attention_ablation import temporal_token_indices


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
