"""Predeclared causal outcome metrics for paired action trajectories."""

from __future__ import annotations

import torch


def donor_steering(
    patched_action: torch.Tensor,
    recipient_action: torch.Tensor,
    donor_action: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Signed fraction of the recipient-to-donor action displacement recovered.

    The final dimensions may contain an action chunk of any shape. Only the first
    dimension is treated as batch when present. A score of zero is no movement;
    one is complete movement to the donor action under Euclidean projection.
    """

    if patched_action.shape != recipient_action.shape or donor_action.shape != recipient_action.shape:
        raise ValueError("patched, recipient, and donor actions must have identical shapes")
    if recipient_action.ndim == 0:
        raise ValueError("actions must have at least one dimension")
    if recipient_action.ndim == 1:
        patched_action = patched_action.unsqueeze(0)
        recipient_action = recipient_action.unsqueeze(0)
        donor_action = donor_action.unsqueeze(0)
    patched_delta = (patched_action - recipient_action).flatten(start_dim=1)
    donor_delta = (donor_action - recipient_action).flatten(start_dim=1)
    denominator = donor_delta.square().sum(dim=1)
    if torch.any(denominator <= eps):
        raise ValueError("donor and recipient actions are indistinguishable")
    return (patched_delta * donor_delta).sum(dim=1) / denominator
