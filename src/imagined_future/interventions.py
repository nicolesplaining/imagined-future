"""Low-level interventions on temporal frames in a latent diffusion trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

Tensor = torch.Tensor
X0Function = Callable[[Tensor, Tensor], Tensor]


def _validate_latents(base: Tensor, replacement: Tensor) -> None:
    if base.ndim != 5:
        raise ValueError(f"expected latent shape (B, C, T, H, W), got {tuple(base.shape)}")
    if base.shape != replacement.shape:
        raise ValueError(f"latent shapes differ: {tuple(base.shape)} != {tuple(replacement.shape)}")


def replace_frames(base: Tensor, replacement: Tensor, frame_indices: Sequence[int]) -> Tensor:
    """Functionally replace selected temporal frames without mutating either input."""

    _validate_latents(base, replacement)
    output = base.clone()
    indices = tuple(int(index) for index in frame_indices)
    if not indices:
        return output
    if min(indices) < 0 or max(indices) >= base.shape[2]:
        raise IndexError(f"frame indices {indices} are invalid for T={base.shape[2]}")
    output[:, :, indices, :, :] = replacement[:, :, indices, :, :]
    return output


def resample_frames(
    base: Tensor,
    frame_indices: Sequence[int],
    *,
    seed: int,
    standard_deviation: float = 1.0,
) -> Tensor:
    """Resample only selected Gaussian-noise frames using an isolated RNG.

    ``standard_deviation`` must match the scale of ``base``. For Cosmos's
    ``x_sigma_max`` tensor, pass the sampling schedule's ``sigma_max``.
    """

    if base.ndim != 5:
        raise ValueError(f"expected latent shape (B, C, T, H, W), got {tuple(base.shape)}")
    indices = tuple(int(index) for index in frame_indices)
    if not indices:
        return base.clone()
    generator = torch.Generator(device=base.device)
    generator.manual_seed(seed)
    sampled = torch.randn(
        (base.shape[0], base.shape[1], len(indices), base.shape[3], base.shape[4]),
        generator=generator,
        device=base.device,
        dtype=base.dtype,
    ) * standard_deviation
    output = base.clone()
    output[:, :, indices, :, :] = sampled
    return output


def _sigma_view(sigma: Tensor, reference: Tensor) -> Tensor:
    if sigma.ndim == 1:
        if sigma.shape[0] != reference.shape[0]:
            raise ValueError("sigma batch dimension does not match latent batch")
        return sigma.reshape(-1, 1, 1, 1, 1)
    if sigma.ndim == 2:
        if sigma.shape[0] != reference.shape[0] or sigma.shape[1] not in (1, reference.shape[2]):
            raise ValueError("per-frame sigma must have shape (B, 1) or (B, T)")
        return sigma.reshape(sigma.shape[0], 1, sigma.shape[1], 1, 1)
    raise ValueError(f"unsupported sigma shape {tuple(sigma.shape)}")


@dataclass
class SemanticFutureClamp:
    """Clamp future frames to a donor trajectory at every denoiser evaluation.

    Input clamping presents a noise-matched donor future to the denoiser. Output
    clamping stabilizes that future in the numerical solver. Keeping both enabled
    defines the primary semantic-sufficiency intervention; each can be ablated.
    """

    donor_clean: Tensor
    donor_noise: Tensor
    frame_indices: tuple[int, ...]
    clamp_input: bool = True
    clamp_output: bool = True
    calls: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        _validate_latents(self.donor_clean, self.donor_noise)
        if not self.frame_indices:
            raise ValueError("semantic future clamp requires at least one frame")

    def wrap(self, x0_fn: X0Function) -> X0Function:
        """Return an interventional denoiser with the same callable contract."""

        def interventional_x0(noisy_latent: Tensor, sigma: Tensor) -> Tensor:
            _validate_latents(noisy_latent, self.donor_clean)
            self.calls.append(float(sigma.detach().float().mean().cpu()))
            denoiser_input = noisy_latent
            if self.clamp_input:
                donor_at_sigma = self.donor_clean + _sigma_view(sigma, noisy_latent) * self.donor_noise
                denoiser_input = replace_frames(noisy_latent, donor_at_sigma, self.frame_indices)
            prediction = x0_fn(denoiser_input, sigma)
            if self.clamp_output:
                prediction = replace_frames(prediction, self.donor_clean, self.frame_indices)
            return prediction

        return interventional_x0
