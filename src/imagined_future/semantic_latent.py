"""Construct VAE-consistent semantic future targets using public Cosmos APIs."""

from __future__ import annotations

from typing import Any, Mapping

import torch


IMAGE_FRAME_PAIRS = (
    ("future_wrist_image_latent_idx", "current_wrist_image_latent_idx"),
    ("future_wrist_image2_latent_idx", "current_wrist_image2_latent_idx"),
    ("future_image_latent_idx", "current_image_latent_idx"),
    ("future_image2_latent_idx", "current_image2_latent_idx"),
)


def _index(batch: Mapping[str, Any], key: str) -> int | None:
    if key not in batch:
        return None
    value = batch[key]
    flattened = value.detach().reshape(-1).cpu().tolist() if hasattr(value, "detach") else list(value)
    indices = {int(item) for item in flattened}
    if len(indices) != 1:
        raise ValueError(f"{key} must be constant within a paired batch; got {sorted(indices)}")
    index = indices.pop()
    return None if index < 0 else index


def raw_frame_span(latent_index: int, temporal_compression_factor: int = 4) -> slice:
    """Map a Cosmos latent frame to raw frames for its 1+T temporal VAE layout."""

    if latent_index < 0:
        raise ValueError("latent index must be non-negative")
    if temporal_compression_factor < 1:
        raise ValueError("temporal compression factor must be positive")
    if latent_index == 0:
        return slice(0, 1)
    start = 1 + (latent_index - 1) * temporal_compression_factor
    return slice(start, start + temporal_compression_factor)


def splice_target_images(
    recipient_batch: Mapping[str, Any],
    target_batch: Mapping[str, Any],
    *,
    temporal_compression_factor: int = 4,
) -> dict[str, Any]:
    """Copy target current-view raw frames into recipient future-view slots."""

    output = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in recipient_batch.items()
    }
    recipient_video = output["video"]
    target_video = target_batch["video"]
    if recipient_video.shape != target_video.shape:
        raise ValueError(f"paired video shapes differ: {recipient_video.shape} != {target_video.shape}")
    for future_key, current_key in IMAGE_FRAME_PAIRS:
        future_index = _index(recipient_batch, future_key)
        current_index = _index(target_batch, current_key)
        if (future_index is None) != (current_index is None):
            raise ValueError(f"paired modality availability differs for {future_key} and {current_key}")
        if future_index is None:
            continue
        destination = raw_frame_span(future_index, temporal_compression_factor)
        source = raw_frame_span(current_index, temporal_compression_factor)
        recipient_video[:, :, destination] = target_video[:, :, source]
    return output


def encode_semantic_future(
    model: Any,
    recipient_batch: Mapping[str, Any],
    target_batch: Mapping[str, Any],
    recipient_clean_latent: torch.Tensor,
    target_proprio: Any,
    *,
    temporal_compression_factor: int = 4,
) -> torch.Tensor:
    """Encode target views in future slots and inject target normalized proprio."""

    modified_batch = splice_target_images(
        recipient_batch,
        target_batch,
        temporal_compression_factor=temporal_compression_factor,
    )
    _raw_state, encoded, _condition = model.get_data_and_condition(modified_batch)
    semantic = recipient_clean_latent.clone()
    for future_key, _current_key in IMAGE_FRAME_PAIRS:
        future_index = _index(recipient_batch, future_key)
        if future_index is not None:
            semantic[:, :, future_index] = encoded[:, :, future_index].type_as(semantic)

    future_proprio_index = _index(recipient_batch, "future_proprio_latent_idx")
    if future_proprio_index is not None:
        from cosmos_policy.models.policy_text2world_model import replace_latent_with_proprio

        proprio = torch.as_tensor(target_proprio, device=semantic.device, dtype=semantic.dtype).reshape(
            semantic.shape[0], -1
        )
        indices = torch.full(
            (semantic.shape[0],),
            future_proprio_index,
            device=semantic.device,
            dtype=torch.int64,
        )
        semantic = replace_latent_with_proprio(semantic, proprio, indices)
    return semantic
