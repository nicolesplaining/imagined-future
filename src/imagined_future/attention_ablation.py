"""Selective future-to-action self-attention ablations for Cosmos Policy."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch


def temporal_token_indices(frames: Sequence[int], *, time: int, height: int, width: int) -> tuple[int, ...]:
    """Map temporal frame indices to flattened ``(T H W)`` token indices."""

    selected = tuple(int(frame) for frame in frames)
    if not selected:
        raise ValueError("at least one frame is required")
    if min(selected) < 0 or max(selected) >= time:
        raise IndexError(f"frame indices {selected} are invalid for T={time}")
    spatial = height * width
    return tuple(index for frame in selected for index in range(frame * spatial, (frame + 1) * spatial))


def gated_attention_output(original: torch.Tensor, ablated: torch.Tensor, gate: float) -> torch.Tensor:
    """Interpolate from native attention to a key-excluded recomputation."""

    if not 0.0 <= gate <= 1.0:
        raise ValueError("attention ablation gate must be within [0, 1]")
    if original.shape != ablated.shape:
        raise ValueError("native and ablated attention outputs must have equal shape")
    return original + gate * (ablated - original)


@dataclass
class AttentionAblationStats:
    """Numerical diagnostics collected for each patched block and denoiser call."""

    calls_by_block: dict[int, int] = field(default_factory=dict)
    selected_output_l2_by_block: dict[int, list[float]] = field(default_factory=dict)
    selected_output_max_abs_by_block: dict[int, list[float]] = field(default_factory=dict)

    def record(self, block: int, original: torch.Tensor, replacement: torch.Tensor) -> None:
        difference = replacement.float() - original.float()
        self.calls_by_block[block] = self.calls_by_block.get(block, 0) + 1
        self.selected_output_l2_by_block.setdefault(block, []).append(
            float(torch.linalg.vector_norm(difference).cpu())
        )
        self.selected_output_max_abs_by_block.setdefault(block, []).append(
            float(torch.max(torch.abs(difference)).cpu())
        )


@contextmanager
def restrict_future_to_action_attention(
    blocks: Sequence[torch.nn.Module],
    *,
    action_frames: Sequence[int],
    future_frames: Sequence[int],
    exclude_future_keys: bool,
    gate: float = 1.0,
    block_ids: Sequence[int] | None = None,
) -> Iterator[AttentionAblationStats]:
    """Recompute action queries with all keys or with future-frame keys removed.

    The implementation reuses each public block's configured attention operator,
    normalization, output projection, and dtype. Recomputing with all keys is the
    numerical control for the changed query batching.
    """

    action_frames = tuple(int(index) for index in action_frames)
    future_frames = tuple(int(index) for index in future_frames)
    if set(action_frames) & set(future_frames):
        raise ValueError("action and future frames must not overlap")
    if not 0.0 <= gate <= 1.0:
        raise ValueError("attention ablation gate must be within [0, 1]")
    stats = AttentionAblationStats()
    labels = tuple(range(len(blocks))) if block_ids is None else tuple(int(index) for index in block_ids)
    if len(labels) != len(blocks) or len(set(labels)) != len(labels):
        raise ValueError("block_ids must be unique and match blocks")
    originals = []
    try:
        for block_index, block in zip(labels, blocks, strict=True):
            attention = block.self_attn
            if getattr(attention, "backend", None) not in {"transformer_engine", "minimal_a2a"}:
                raise ValueError(
                    "attention ablation requires a public backend whose native operator supports "
                    "different query and key lengths"
                )
            original = attention.compute_attention
            originals.append((attention, original))

            def wrapped(q, k, v, video_size=None, kv_cache_cfg=None, *, _index=block_index, _attention=attention, _original=original):
                if kv_cache_cfg is not None:
                    raise ValueError("attention ablation does not support a KV cache")
                if video_size is None:
                    raise ValueError("self-attention did not provide video_size")
                full = _original(q, k, v, video_size=video_size, kv_cache_cfg=kv_cache_cfg)
                time, height, width = int(video_size.T), int(video_size.H), int(video_size.W)
                if q.shape[1] != time * height * width:
                    raise ValueError("flattened token count does not match video_size")
                query_indices = temporal_token_indices(
                    action_frames, time=time, height=height, width=width
                )
                if exclude_future_keys:
                    excluded = set(
                        temporal_token_indices(future_frames, time=time, height=height, width=width)
                    )
                    key_indices = tuple(index for index in range(k.shape[1]) if index not in excluded)
                else:
                    key_indices = tuple(range(k.shape[1]))
                query_index = torch.as_tensor(query_indices, device=q.device)
                key_index = torch.as_tensor(key_indices, device=k.device)
                query = torch.index_select(q, 1, query_index)
                key = torch.index_select(k, 1, key_index)
                value = torch.index_select(v, 1, key_index)
                replacement = _attention.attn_op(query, key, value)
                replacement = _attention.output_dropout(_attention.output_proj(replacement))
                original_selected = torch.index_select(full, 1, query_index)
                if exclude_future_keys:
                    replacement = gated_attention_output(original_selected, replacement, gate)
                stats.record(_index, original_selected, replacement)
                output = full.clone()
                output[:, query_index] = replacement
                return output

            attention.compute_attention = wrapped
        yield stats
    finally:
        for attention, original in originals:
            attention.compute_attention = original
