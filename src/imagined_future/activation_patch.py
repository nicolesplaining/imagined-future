"""Capture and transplant temporal residual streams in Cosmos Policy's DiT."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import torch


def _validate_block_output(output: Any, frame_indices: Sequence[int]) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"expected a tensor block output, got {type(output)!r}")
    if output.ndim != 5:
        raise ValueError(f"expected DiT residual shape (B, T, H, W, D), got {tuple(output.shape)}")
    indices = tuple(int(index) for index in frame_indices)
    if not indices:
        raise ValueError("at least one temporal frame is required")
    if min(indices) < 0 or max(indices) >= output.shape[1]:
        raise IndexError(f"frame indices {indices} are invalid for T={output.shape[1]}")
    return output


@dataclass
class ActivationBank:
    """Per-module, per-denoiser-call residual slices kept on their source device."""

    frame_indices: tuple[int, ...]
    calls: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def capture(self, name: str, output: Any) -> None:
        tensor = _validate_block_output(output, self.frame_indices)
        self.calls.setdefault(name, []).append(tensor[:, self.frame_indices].detach().clone())


@contextmanager
def capture_module_outputs(
    modules: Mapping[str, torch.nn.Module], frame_indices: Sequence[int]
) -> Iterator[ActivationBank]:
    """Capture selected temporal slices after each named module."""

    bank = ActivationBank(tuple(int(index) for index in frame_indices))
    with ExitStack() as stack:
        for name, module in modules.items():
            handle = module.register_forward_hook(
                lambda _module, _inputs, output, name=name: bank.capture(name, output)
            )
            stack.callback(handle.remove)
        yield bank


@dataclass
class ActivationTransplant:
    """Replace one module's temporal residual slice call-by-call."""

    donor_calls: Sequence[torch.Tensor]
    frame_indices: tuple[int, ...]
    scale: float = 1.0
    calls: int = 0
    patch_l2: list[float] = field(default_factory=list)
    donor_recipient_l2: list[float] = field(default_factory=list)
    recipient_l2: list[float] = field(default_factory=list)
    donor_l2: list[float] = field(default_factory=list)

    def hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> torch.Tensor:
        tensor = _validate_block_output(output, self.frame_indices)
        if self.calls >= len(self.donor_calls):
            raise RuntimeError("recipient made more denoiser calls than the donor")
        donor = self.donor_calls[self.calls].to(device=tensor.device, dtype=tensor.dtype)
        recipient = tensor[:, self.frame_indices]
        if donor.shape != recipient.shape:
            raise ValueError(f"donor slice shape {tuple(donor.shape)} != recipient {tuple(recipient.shape)}")
        difference = donor.float() - recipient.float()
        applied = difference * self.scale
        self.donor_recipient_l2.append(float(torch.linalg.vector_norm(difference).cpu()))
        self.patch_l2.append(float(torch.linalg.vector_norm(applied).cpu()))
        self.recipient_l2.append(float(torch.linalg.vector_norm(recipient.float()).cpu()))
        self.donor_l2.append(float(torch.linalg.vector_norm(donor.float()).cpu()))
        patched = tensor.clone()
        patched[:, self.frame_indices] = recipient + applied.to(dtype=recipient.dtype)
        self.calls += 1
        return patched

    def validate_complete(self) -> None:
        if self.calls != len(self.donor_calls):
            raise RuntimeError(f"recipient used {self.calls} donor calls; expected {len(self.donor_calls)}")


@contextmanager
def transplant_module_output(
    module: torch.nn.Module,
    donor_calls: Sequence[torch.Tensor],
    frame_indices: Sequence[int],
    *,
    scale: float = 1.0,
) -> Iterator[ActivationTransplant]:
    """Install a temporary call-aligned residual-stream transplant hook."""

    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"transplant scale must be in [0, 1], got {scale}")
    transplant = ActivationTransplant(donor_calls, tuple(int(index) for index in frame_indices), scale=scale)
    handle = module.register_forward_hook(transplant.hook)
    try:
        yield transplant
    finally:
        handle.remove()
