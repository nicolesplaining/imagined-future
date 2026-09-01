"""Cosmos 3 action-query attention interventions.

The released two-way attention packs vision tokens first and action tokens last
in the generation stream. This wrapper recomputes only the action-query rows
after removing future-video keys/values, then uses device-resident gates to
select layers without changing the compiled graph between requests.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class AttentionRuntimeOps:
    attention: Callable[..., torch.Tensor]
    from_mode_splits: Callable[..., dict[str, Any]]
    get_all_seq: Callable[[dict[str, Any]], torch.Tensor]
    get_causal_seq: Callable[[dict[str, Any]], torch.Tensor]
    get_full_only_seq: Callable[[dict[str, Any]], torch.Tensor]


def sequence_without_offsets(value: Any) -> torch.Tensor:
    """Normalize public runtime accessors, which return ``(sequence, offsets)``."""

    return value[0] if isinstance(value, tuple) else value


def public_attention_runtime_ops() -> AttentionRuntimeOps:
    """Load the pinned public Cosmos Framework attention operations."""

    from cosmos_framework.data.generator.sequence_packing.runtime import (
        from_mode_splits,
        get_all_seq,
        get_causal_seq,
        get_full_only_seq,
    )
    from cosmos_framework.model.attention import attention

    return AttentionRuntimeOps(
        attention=attention,
        from_mode_splits=from_mode_splits,
        get_all_seq=get_all_seq,
        get_causal_seq=get_causal_seq,
        get_full_only_seq=get_full_only_seq,
    )


class ActionQueryFutureKVExcluder:
    """Remove future-video K/V at direct or full non-future attention interfaces."""

    def __init__(
        self,
        *,
        num_layers: int,
        action_tokens: int,
        video_latent_frames: int,
        device: torch.device,
        ops: AttentionRuntimeOps | None = None,
    ) -> None:
        if num_layers <= 0 or action_tokens <= 0 or video_latent_frames <= 1:
            raise ValueError("layer, action-token, and video-frame counts must be positive")
        self.num_layers = int(num_layers)
        self.action_tokens = int(action_tokens)
        self.video_latent_frames = int(video_latent_frames)
        self.action_gates = torch.zeros(self.num_layers, device=device, dtype=torch.float32)
        self.barrier_gates = torch.zeros(self.num_layers, device=device, dtype=torch.float32)
        self.ops = ops or public_attention_runtime_ops()

    def set_layers(
        self, layers: tuple[int, ...] | list[int], *, scope: str = "action"
    ) -> None:
        selected = tuple(int(layer) for layer in layers)
        if scope not in {"action", "nonfuture"}:
            raise ValueError("attention exclusion scope must be 'action' or 'nonfuture'")
        if len(set(selected)) != len(selected):
            raise ValueError("attention exclusion layers must be unique")
        if any(layer < 0 or layer >= self.num_layers for layer in selected):
            raise ValueError(f"attention exclusion layers must be in [0,{self.num_layers})")
        self.action_gates.zero_()
        self.barrier_gates.zero_()
        if selected:
            gates = self.action_gates if scope == "action" else self.barrier_gates
            gates[list(selected)] = 1.0

    def active_layers(self) -> dict[str, list[int]]:
        return {
            "action": self.action_gates.nonzero(as_tuple=False).reshape(-1).cpu().tolist(),
            "nonfuture": self.barrier_gates.nonzero(as_tuple=False).reshape(-1).cpu().tolist(),
        }

    @contextmanager
    def activate(self, layers: tuple[int, ...] | list[int], *, scope: str = "action"):
        self.set_layers(layers, scope=scope)
        try:
            yield self
        finally:
            self.set_layers(())

    def wrap(self, layer: int, original: Callable[..., tuple[dict[str, Any], Any]]):
        """Wrap one public ``dispatch_attention_fn`` without changing its signature."""

        if layer < 0 or layer >= self.num_layers:
            raise ValueError(f"layer must be in [0,{self.num_layers})")
        ops = self.ops

        def dispatch(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
            natten_metadata=None,
            memory_value=None,
            packed_key_states_normalized=None,
        ):
            native, kv_to_store = original(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                attention_mask,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
                packed_key_states_normalized=packed_key_states_normalized,
            )
            if memory_value is not None:
                raise ValueError("future K/V exclusion supports the released non-memory policy path only")
            if bool(getattr(attention_mask, "is_three_way", False)):
                raise ValueError("future K/V exclusion requires released two-way attention")
            if getattr(attention_mask, "control_stream_token_ranges", None) is not None:
                raise ValueError("future K/V exclusion does not support multi-control attention")
            if getattr(attention_mask, "flex_block_mask", None) is not None:
                raise ValueError("future K/V exclusion does not support FlexAttention masks")

            query_gen = sequence_without_offsets(ops.get_full_only_seq(packed_query_states))
            actual_gen_tokens = int(packed_query_states["_full_indices"].shape[0])
            vision_tokens = actual_gen_tokens - self.action_tokens
            if vision_tokens <= 0 or vision_tokens % self.video_latent_frames:
                raise ValueError(
                    f"cannot divide {vision_tokens} vision tokens into "
                    f"{self.video_latent_frames} latent frames"
                )
            tokens_per_video_frame = vision_tokens // self.video_latent_frames
            current_stop = tokens_per_video_frame
            action_start = vision_tokens
            action_stop = actual_gen_tokens

            key_source = (
                packed_key_states_normalized
                if packed_key_states_normalized is not None
                else packed_key_states
            )
            key_causal = sequence_without_offsets(ops.get_causal_seq(key_source))
            key_gen = sequence_without_offsets(ops.get_full_only_seq(key_source))
            value_causal = sequence_without_offsets(ops.get_causal_seq(packed_value_states))
            value_gen = sequence_without_offsets(ops.get_full_only_seq(packed_value_states))
            key_nonfuture = torch.cat(
                (key_causal, key_gen[:current_stop], key_gen[action_start:action_stop]),
                dim=0,
            )
            value_nonfuture = torch.cat(
                (
                    value_causal,
                    value_gen[:current_stop],
                    value_gen[action_start:action_stop],
                ),
                dim=0,
            )
            query_nonfuture = torch.cat(
                (query_gen[:current_stop], query_gen[action_start:action_stop]),
                dim=0,
            )
            alternative_nonfuture = ops.attention(
                query_nonfuture.unsqueeze(0),
                key_nonfuture.unsqueeze(0),
                value_nonfuture.unsqueeze(0),
            )
            alternative_nonfuture = alternative_nonfuture.squeeze(0).flatten(-2, -1)
            alternative_current = alternative_nonfuture[:current_stop]
            alternative_action = alternative_nonfuture[current_stop:]
            native_gen = sequence_without_offsets(ops.get_full_only_seq(native))
            native_current = native_gen[:current_stop]
            native_action = native_gen[action_start:action_stop]
            action_gate = self.action_gates[layer].bool()
            barrier_gate = self.barrier_gates[layer].bool()
            gated_current = torch.where(
                barrier_gate, alternative_current, native_current
            )
            gated_action = torch.where(
                torch.logical_or(action_gate, barrier_gate), alternative_action, native_action
            )
            gated_gen = torch.cat(
                (
                    gated_current,
                    native_gen[current_stop:action_start],
                    gated_action,
                    native_gen[action_stop:],
                ),
                dim=0,
            )
            native_causal = sequence_without_offsets(ops.get_causal_seq(native))
            output = ops.from_mode_splits(native_causal, gated_gen, native)
            return output, kv_to_store

        return dispatch
