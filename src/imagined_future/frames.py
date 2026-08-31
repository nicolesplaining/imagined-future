"""Resolve semantic frame groups from a Cosmos Policy inference batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _uniform_index(batch: Mapping[str, Any], key: str) -> int | None:
    """Return a batch-wide latent index, rejecting mixed-index batches."""

    if key not in batch:
        return None
    value = batch[key]
    if hasattr(value, "detach"):
        flattened = value.detach().reshape(-1).cpu().tolist()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        flattened = list(value)
    else:
        flattened = [value]
    indices = {int(item) for item in flattened}
    if len(indices) != 1:
        raise ValueError(f"{key} must be constant within a paired inference batch; got {sorted(indices)}")
    index = indices.pop()
    return None if index < 0 else index


def _present_indices(batch: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[int, ...]:
    indices = tuple(index for key in keys if (index := _uniform_index(batch, key)) is not None)
    if len(indices) != len(set(indices)):
        raise ValueError(f"latent frame groups contain duplicate indices: {indices}")
    return indices


@dataclass(frozen=True)
class LatentFrameGroups:
    """Semantic positions in Cosmos Policy's ``(B, C, T, H, W)`` latent sequence."""

    current: tuple[int, ...]
    action: tuple[int, ...]
    future: tuple[int, ...]
    value: tuple[int, ...]

    @classmethod
    def from_batch(cls, batch: Mapping[str, Any]) -> "LatentFrameGroups":
        current = _present_indices(
            batch,
            (
                "current_proprio_latent_idx",
                "current_wrist_image_latent_idx",
                "current_wrist_image2_latent_idx",
                "current_image_latent_idx",
                "current_image2_latent_idx",
            ),
        )
        action = _present_indices(batch, ("action_latent_idx",))
        future = _present_indices(
            batch,
            (
                "future_proprio_latent_idx",
                "future_wrist_image_latent_idx",
                "future_wrist_image2_latent_idx",
                "future_image_latent_idx",
                "future_image2_latent_idx",
            ),
        )
        value = _present_indices(batch, ("value_latent_idx",))
        groups = cls(current=current, action=action, future=future, value=value)
        groups.validate_disjoint()
        return groups

    def validate_disjoint(self) -> None:
        named = {
            "current": set(self.current),
            "action": set(self.action),
            "future": set(self.future),
            "value": set(self.value),
        }
        for left_name, left in named.items():
            for right_name, right in named.items():
                if left_name >= right_name:
                    continue
                overlap = left & right
                if overlap:
                    raise ValueError(f"{left_name} and {right_name} overlap at {sorted(overlap)}")
