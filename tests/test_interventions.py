from __future__ import annotations

import pytest
import torch

from imagined_future.interventions import (
    SemanticFutureClamp,
    norm_distance_matched_random_target,
    replace_frames,
    resample_frames,
)


def test_replace_frames_is_functional_and_localized() -> None:
    base = torch.zeros(2, 1, 4, 2, 2)
    donor = torch.ones_like(base)

    result = replace_frames(base, donor, (1, 3))

    assert torch.count_nonzero(base) == 0
    assert torch.all(result[:, :, (1, 3)] == 1)
    assert torch.all(result[:, :, (0, 2)] == 0)


def test_resampling_preserves_nonintervened_frames_and_is_repeatable() -> None:
    base = torch.zeros(1, 2, 5, 2, 2)

    first = resample_frames(base, (2, 4), seed=17, standard_deviation=4.0)
    second = resample_frames(base, (2, 4), seed=17, standard_deviation=4.0)

    assert torch.equal(first, second)
    assert torch.count_nonzero(first[:, :, (0, 1, 3)]) == 0
    assert torch.count_nonzero(first[:, :, (2, 4)]) > 0


def test_semantic_clamp_noise_matches_sigma_and_stabilizes_output() -> None:
    donor_clean = torch.full((1, 1, 3, 1, 1), 2.0)
    donor_noise = torch.full_like(donor_clean, 3.0)
    observed_inputs: list[torch.Tensor] = []

    def identity_x0(noisy: torch.Tensor, _sigma: torch.Tensor) -> torch.Tensor:
        observed_inputs.append(noisy.clone())
        return noisy

    clamp = SemanticFutureClamp(donor_clean, donor_noise, (1,))
    result = clamp.wrap(identity_x0)(torch.zeros_like(donor_clean), torch.tensor([4.0]))

    assert observed_inputs[0][0, 0, 1, 0, 0].item() == 14.0
    assert result[0, 0, 1, 0, 0].item() == 2.0
    assert torch.all(result[:, :, (0, 2)] == 0)
    assert clamp.calls == [4.0]


def test_random_target_matches_donor_norm_and_distance() -> None:
    recipient = torch.arange(1, 17, dtype=torch.float64).reshape(1, 1, 1, 4, 4)
    donor = torch.flip(recipient, dims=(-1,)) * 0.8

    matched = norm_distance_matched_random_target(recipient, donor, seed=73)

    assert torch.linalg.vector_norm(matched) == pytest.approx(torch.linalg.vector_norm(donor))
    assert torch.linalg.vector_norm(matched - recipient) == pytest.approx(
        torch.linalg.vector_norm(donor - recipient)
    )
    assert not torch.allclose(matched, donor)
