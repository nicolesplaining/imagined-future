"""Cosmos 3 future-to-action attention interventions.

The released two-way attention packs vision tokens first and action tokens last
in the generation stream. This wrapper recomputes only the action-query rows
after excluding or content-patching future-video keys/values. The research
server uses the public eager path because layer-specific Python dispatchers are
not compatible with the released full-graph repeated-layer compiler.
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
    """Exclude or content-patch future-video K/V at action interfaces."""

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
        self.mode = "exclude"
        self.cache_id: str | None = None
        self.kv_caches: dict[
            str, dict[int, list[tuple[torch.Tensor, torch.Tensor]]]
        ] = {}
        self.call_indices: dict[int, int] = {}

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
    def activate(
        self,
        layers: tuple[int, ...] | list[int],
        *,
        scope: str = "action",
        mode: str = "exclude",
        cache_id: str | None = None,
    ):
        if mode not in {"exclude", "record", "patch"}:
            raise ValueError("attention mode must be 'exclude', 'record', or 'patch'")
        if mode in {"record", "patch"} and not cache_id:
            raise ValueError(f"attention mode {mode!r} requires a cache id")
        if mode == "record":
            self.kv_caches = {str(cache_id): {}}
        elif mode == "patch" and str(cache_id) not in self.kv_caches:
            raise KeyError(f"unknown attention K/V cache: {cache_id!r}")
        self.set_layers(layers, scope=scope)
        self.mode = mode
        self.cache_id = str(cache_id) if cache_id is not None else None
        self.call_indices = {int(layer): 0 for layer in layers}
        try:
            yield self
        finally:
            self.set_layers(())
            self.mode = "exclude"
            self.cache_id = None
            self.call_indices = {}

    def cache_summary(self, cache_id: str) -> dict[str, int]:
        """Return recorded forward-call counts per layer for audit output."""

        if cache_id not in self.kv_caches:
            raise KeyError(f"unknown attention K/V cache: {cache_id!r}")
        return {
            str(layer): len(entries)
            for layer, entries in self.kv_caches[cache_id].items()
        }

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
            action_gate = bool(self.action_gates[layer].item())
            barrier_gate = bool(self.barrier_gates[layer].item())
            if not action_gate and not barrier_gate:
                return native, kv_to_store
            if memory_value is not None:
                raise ValueError("future K/V intervention supports the released non-memory policy path only")
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
            key_all = ops.get_all_seq(key_source)
            key_gen = sequence_without_offsets(ops.get_full_only_seq(key_source))
            value_all = ops.get_all_seq(packed_value_states)
            value_gen = sequence_without_offsets(ops.get_full_only_seq(packed_value_states))
            future_gen = torch.arange(
                current_stop,
                action_start,
                device=packed_query_states["_full_indices"].device,
            )
            future_all = packed_query_states["_full_indices"][future_gen]
            if self.mode == "record":
                if self.cache_id is None:
                    raise RuntimeError("record mode has no cache id")
                layer_cache = self.kv_caches[self.cache_id].setdefault(layer, [])
                layer_cache.append(
                    (
                        key_gen[current_stop:action_start].detach().clone(),
                        value_gen[current_stop:action_start].detach().clone(),
                    )
                )
                self.call_indices[layer] += 1
                return native, kv_to_store

            if self.mode == "patch":
                if self.cache_id is None:
                    raise RuntimeError("patch mode has no cache id")
                reference_calls = self.kv_caches[self.cache_id].get(layer, [])
                call_index = self.call_indices[layer]
                if call_index >= len(reference_calls):
                    raise RuntimeError(
                        f"attention cache {self.cache_id!r} layer {layer} has no "
                        f"reference for call {call_index}"
                    )
                reference_key, reference_value = reference_calls[call_index]
                self.call_indices[layer] += 1
                native_future_key = key_gen[current_stop:action_start]
                native_future_value = value_gen[current_stop:action_start]
                if reference_key.shape != native_future_key.shape:
                    raise ValueError(
                        f"cached key shape {reference_key.shape} differs from "
                        f"native future key shape {native_future_key.shape}"
                    )
                if reference_value.shape != native_future_value.shape:
                    raise ValueError(
                        f"cached value shape {reference_value.shape} differs from "
                        f"native future value shape {native_future_value.shape}"
                    )
                alternative_key = key_all.clone()
                alternative_value = value_all.clone()
                alternative_key[future_all] = reference_key.to(
                    device=key_all.device, dtype=key_all.dtype
                )
                alternative_value[future_all] = reference_value.to(
                    device=value_all.device, dtype=value_all.dtype
                )
            else:
                keep = torch.ones(
                    key_all.shape[0], device=key_all.device, dtype=torch.bool
                )
                keep[future_all] = False
                alternative_key = key_all[keep]
                alternative_value = value_all[keep]
            # Preserve the native generation-query length so the attention
            # backend uses identical row tiling. Only current/action rows are
            # inserted below; future-query outputs remain native.
            alternative_gen = ops.attention(
                query_gen.unsqueeze(0),
                alternative_key.unsqueeze(0),
                alternative_value.unsqueeze(0),
            )
            alternative_gen = alternative_gen.squeeze(0).flatten(-2, -1)
            alternative_current = alternative_gen[:current_stop]
            alternative_action = alternative_gen[action_start:action_stop]
            native_gen = sequence_without_offsets(ops.get_full_only_seq(native))
            native_current = native_gen[:current_stop]
            native_action = native_gen[action_start:action_stop]
            gated_current = torch.where(
                torch.tensor(barrier_gate, device=native_current.device),
                alternative_current,
                native_current,
            )
            gated_action = torch.where(
                torch.tensor(action_gate or barrier_gate, device=native_action.device),
                alternative_action,
                native_action,
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
