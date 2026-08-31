from __future__ import annotations

import pytest
import torch

from imagined_future.metrics import donor_steering


def test_donor_steering_has_interpretable_endpoints() -> None:
    recipient = torch.tensor([[0.0, 0.0]])
    donor = torch.tensor([[2.0, 0.0]])

    assert torch.allclose(donor_steering(recipient, recipient, donor), torch.tensor([0.0]))
    assert torch.allclose(donor_steering(donor, recipient, donor), torch.tensor([1.0]))
    assert torch.allclose(donor_steering(torch.tensor([[1.0, 1.0]]), recipient, donor), torch.tensor([0.5]))


def test_donor_steering_rejects_degenerate_pair() -> None:
    action = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="indistinguishable"):
        donor_steering(action, action, action)
