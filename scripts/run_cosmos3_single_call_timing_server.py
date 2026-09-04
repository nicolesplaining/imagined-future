#!/usr/bin/env python3
"""Serve Cosmos 3 with auditable native and future-clamp research modes.

The default request path remains compatible with RoboLab's OpenPI client.  A
research client can additionally provide ``research_mode``, ``research_seed``,
and ``research_id`` fields.  Native predictions and tokenizer-encoded executed
rollouts are kept in an in-memory registry and can be used as future-only
donors after an exact transformed-observation fingerprint check.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from imagined_future.cosmos3_interventions import (
    GuidedFutureClamp,
    GuidedX0Recorder,
    PreparedLayoutCapture,
    SamplerInitialStateCapture,
    SamplerVelocityWrapper,
    gaussian_target_on_mask,
    temporal_mask,
)
from imagined_future.cosmos3_attention import ActionQueryFutureKVExcluder

DEFAULT_ALLOWED_DONOR_ROOTS = (
    Path("/research/results"),
    Path("/lambda/nfs/imagined-future/results"),
)


def resolve_donor_path(value: str) -> Path:
    """Resolve a donor under one of the explicit container result mounts."""

    path = Path(value).expanduser().resolve()
    configured = os.environ.get("IMAGINED_FUTURE_DONOR_ROOTS")
    roots = (
        tuple(Path(item).expanduser().resolve() for item in configured.split(":"))
        if configured
        else tuple(root.resolve() for root in DEFAULT_ALLOWED_DONOR_ROOTS)
    )
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError(f"donor path must be under one of {roots}")
    return path


def tensor_digest(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact storage bytes, including bfloat16."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sample_fingerprint(sample: dict[str, Any]) -> str:
    """Fingerprint the transformed current vision, state, and instruction."""

    digest = hashlib.sha256()
    for key in ("ai_caption", "video", "action", "history_action", "domain_id", "conditioning_fps"):
        if key not in sample:
            continue
        digest.update(key.encode())
        value = sample[key]
        if isinstance(value, torch.Tensor):
            digest.update(tensor_digest(value).encode())
        else:
            digest.update(str(value).encode())
    return digest.hexdigest()


def parameter_probe_fingerprint(module: torch.nn.Module, *, values_per_edge: int = 16) -> str:
    """Hash small deterministic samples from every parameter without copying the 16B model."""

    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        flat = parameter.detach().reshape(-1)
        edge = min(values_per_edge, flat.numel())
        sample = torch.cat((flat[:edge], flat[-edge:])).cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(np.asarray(parameter.shape, dtype=np.int64).tobytes())
        digest.update(sample.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class FutureRecord:
    record_id: str
    source: str
    state_hash: str
    seed: int | None
    vision_shape: tuple[int, ...]
    target: torch.Tensor
    path_noise: torch.Tensor | None
    initial_state_hash: str | None
    action: np.ndarray | None
    sigmas: tuple[float, ...]
    x0_vision_hashes: tuple[str, ...]
    x0_action_hashes: tuple[str, ...]


class ResearchPolicyService:
    """Thin wrapper around the pinned public RoboLab policy service."""

    def __init__(
        self,
        args: Any,
        *,
        registry_limit: int = 256,
        attention_instrumentation: bool = False,
    ) -> None:
        from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService

        self._base = RobolabPolicyService(args)
        if self._base.cfg.action_space != "joint_pos":
            raise ValueError("the research server currently supports the released joint-position policy only")
        self.registry_limit = int(registry_limit)
        self.registry: OrderedDict[str, FutureRecord] = OrderedDict()
        net = self._base.model.net
        self.attention_excluder = None
        if attention_instrumentation:
            self.attention_excluder = ActionQueryFutureKVExcluder(
                num_layers=36,
                action_tokens=self._base.cfg.action_chunk_size + self._base.cfg.history_length,
                video_latent_frames=9,
                device=next(net.parameters()).device,
            )
            wrapped_layers = set()
            for _name, module in net.named_modules():
                if not hasattr(module, "dispatch_attention_fn") or not hasattr(module, "layer_idx"):
                    continue
                layer = int(module.layer_idx)
                if layer in wrapped_layers:
                    raise RuntimeError(f"multiple attention modules reported layer {layer}")
                module.dispatch_attention_fn = self.attention_excluder.wrap(
                    layer, module.dispatch_attention_fn
                )
                wrapped_layers.add(layer)
            expected_layers = set(range(self.attention_excluder.num_layers))
            if wrapped_layers != expected_layers:
                raise RuntimeError(
                    f"attention interface layer census mismatch: {sorted(wrapped_layers)}"
                )
            # Torch's full-graph compiled decoder cannot swap dispatch functions
            # after its first trace. Keep the audited wrapper installed with zero
            # gates by default and expose full text/video/action K/V on every
            # attention-server request. This disables only request-local text K/V
            # reuse; every experimental arm therefore follows the same graph.
            self._base.model._can_reuse_inference_text_kv = lambda *_args, **_kwargs: False
        self.parameter_probe_hash = parameter_probe_fingerprint(self._base.model.net)

    def _remember(self, record: FutureRecord) -> None:
        if record.record_id in self.registry:
            raise ValueError(f"research_id already exists: {record.record_id!r}")
        self.registry[record.record_id] = record
        while len(self.registry) > self.registry_limit:
            self.registry.popitem(last=False)

    def _batch(self, sample: dict[str, Any]) -> dict[str, Any]:
        from cosmos_framework.scripts.action_policy_server_robolab import _build_data_batch_from_sample

        return _build_data_batch_from_sample(sample)

    def _format_action(self, samples: dict[str, Any]) -> np.ndarray:
        action = samples["action"][0][:, : self._base.cfg.action_dim]
        action = action[self._base.cfg.history_length :]
        output = action.detach().float().cpu().numpy()
        output[:, -1] = 1.0 - output[:, -1]
        return output

    def _native(self, data_batch: dict[str, Any], seed: int) -> tuple[dict[str, Any], FutureRecord]:
        layout = PreparedLayoutCapture()
        initial = SamplerInitialStateCapture(self._base.model.sampler)
        x0 = GuidedX0Recorder(layout)
        sampler = SamplerVelocityWrapper(initial, x0.wrap_velocity)
        samples = self._base.model.generate_samples_from_batch(
            data_batch,
            sampler=sampler,
            velocity_postprocess_builder=layout,
            guidance=self._base.cfg.guidance,
            guidance_interval=(
                list(self._base.cfg.guidance_interval) if self._base.cfg.guidance_interval is not None else None
            ),
            seed=[seed],
            num_steps=self._base.cfg.num_steps,
            shift=self._base.cfg.shift,
        )
        if layout.layout is None or initial.initial_state is None:
            raise RuntimeError("native sampler audit capture did not initialize")
        vision = layout.layout.modality(0, "vision_0")
        path_noise = initial.initial_state[0][vision.start : vision.stop].detach().cpu()
        action = self._format_action(samples)
        record = FutureRecord(
            record_id="",
            source="predicted",
            state_hash="",
            seed=seed,
            vision_shape=tuple(int(item) for item in samples["vision"][0].shape),
            target=samples["vision"][0].detach().cpu().reshape(-1),
            path_noise=path_noise,
            initial_state_hash=tensor_digest(initial.initial_state[0]),
            action=action,
            sigmas=tuple(float(item["sigma"]) for item in x0.records),
            x0_vision_hashes=tuple(tensor_digest(item["samples"][0]["vision_0"]) for item in x0.records),
            x0_action_hashes=tuple(tensor_digest(item["samples"][0]["action"]) for item in x0.records),
        )
        return samples, record

    def _intervene(
        self,
        data_batch: dict[str, Any],
        *,
        seed: int,
        recipient: FutureRecord,
        donor: FutureRecord,
        mode: str,
        gaussian_seed: int,
        timing_steps: tuple[int, ...] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        layout = PreparedLayoutCapture()
        if donor.vision_shape != recipient.vision_shape:
            raise ValueError(
                f"recipient/donor latent shapes differ: {recipient.vision_shape} versus {donor.vision_shape}"
            )
        future_frames = tuple(range(1, donor.vision_shape[-3]))
        mask = temporal_mask(donor.vision_shape, future_frames, device=torch.device("cpu"))
        if mode in {"self", "none"}:
            target = recipient.target
            target_source = "recipient"
        elif mode == "donor":
            target = donor.target
            target_source = "donor"
        elif mode == "gaussian":
            target = gaussian_target_on_mask(recipient.target, donor.target, mask, seed=gaussian_seed)
            target_source = "gaussian_geometry"
        else:
            raise ValueError(f"unsupported intervention mode: {mode!r}")
        if mode == "none" and timing_steps != ():
            raise ValueError("none mode requires an explicit empty research_timing_steps list")
        if recipient.path_noise is None:
            raise ValueError("recipient record has no captured path noise")

        clamp = GuidedFutureClamp(
            layout,
            target,
            recipient.path_noise,
            future_frames,
            active_call_indices=timing_steps,
        )
        x0 = GuidedX0Recorder(layout)

        def transform(velocity_fn):
            return x0.wrap_velocity(clamp.wrap_velocity(velocity_fn))

        initial = SamplerInitialStateCapture(self._base.model.sampler)
        sampler = SamplerVelocityWrapper(initial, transform)
        samples = self._base.model.generate_samples_from_batch(
            data_batch,
            sampler=sampler,
            velocity_postprocess_builder=layout,
            guidance=self._base.cfg.guidance,
            guidance_interval=(
                list(self._base.cfg.guidance_interval) if self._base.cfg.guidance_interval is not None else None
            ),
            seed=[seed],
            num_steps=self._base.cfg.num_steps,
            shift=self._base.cfg.shift,
        )
        if initial.initial_state is None:
            raise RuntimeError("intervention sampler audit capture did not initialize")
        output_target = samples["vision"][0].detach().cpu().reshape(-1)
        requested_active_call_indices = (
            tuple(range(len(clamp.calls))) if timing_steps is None else timing_steps
        )
        if (
            len(set(requested_active_call_indices)) != len(requested_active_call_indices)
            or tuple(sorted(requested_active_call_indices)) != requested_active_call_indices
            or any(index < 0 or index >= len(clamp.calls) for index in requested_active_call_indices)
        ):
            raise ValueError(
                f"requested active calls are not unique ordered valid indices: "
                f"{requested_active_call_indices} for {len(clamp.calls)} calls"
            )
        requested_active_sigmas = tuple(
            clamp.calls[index] for index in requested_active_call_indices
        )
        if tuple(clamp.clamped_call_indices) != requested_active_call_indices:
            raise RuntimeError(
                f"observed active calls {clamp.clamped_call_indices} differ from requested "
                f"{requested_active_call_indices}"
            )
        if tuple(clamp.active_call_sigmas) != requested_active_sigmas:
            raise RuntimeError("observed active sigmas differ from requested call sigmas")
        if not (
            len(clamp.model_input_future_clamp_errors)
            == len(clamp.returned_future_velocity_overwrite_errors)
            == len(requested_active_call_indices)
        ):
            raise RuntimeError("intervention-site audit cardinality differs from active calls")
        inactive_call_indices = tuple(
            index
            for index in range(len(clamp.calls))
            if index not in requested_active_call_indices
        )
        if tuple(clamp.inactive_call_indices) != inactive_call_indices:
            raise RuntimeError("observed inactive calls differ from requested complement")
        if not (
            len(clamp.action_input_errors)
            == len(clamp.action_output_errors)
            == len(clamp.calls)
        ):
            raise RuntimeError("per-call action nonwrite audit cardinality differs")
        final_sampler_delta = output_target[mask] - target[mask]
        audit = {
            "target_hash": tensor_digest(target),
            "target_source": target_source,
            "target_source_record_ids": [recipient.record_id]
            if target_source == "recipient"
            else [donor.record_id]
            if target_source == "donor"
            else [recipient.record_id, donor.record_id],
            "recipient_future_hash": tensor_digest(recipient.target),
            "donor_future_hash": tensor_digest(donor.target),
            "recipient_path_noise_hash": tensor_digest(recipient.path_noise),
            "initial_state_hash": tensor_digest(initial.initial_state[0]),
            "output_future_hash": tensor_digest(output_target),
            "final_sampler_target_max_abs_error": float(
                final_sampler_delta.abs().max()
            ),
            "final_sampler_target_l2": float(
                torch.linalg.vector_norm(final_sampler_delta.double())
            ),
            "maximum_action_input_error": clamp.maximum_action_input_error,
            "maximum_action_output_error": clamp.maximum_action_output_error,
            "sigmas": np.asarray(clamp.calls, dtype=np.float32),
            "vision_shape": np.asarray(donor.vision_shape, dtype=np.int64),
            "future_frame_indices": np.asarray(future_frames, dtype=np.int64),
            "vision_coordinate_count": int(mask.numel()),
            "future_mask_coordinate_count": int(mask.sum()),
            "future_mask_index_hash": tensor_digest(mask),
            "requested_active_call_indices": np.asarray(
                requested_active_call_indices, dtype=np.int64
            ),
            "observed_active_call_indices": np.asarray(
                clamp.clamped_call_indices, dtype=np.int64
            ),
            "inactive_call_indices": np.asarray(
                inactive_call_indices, dtype=np.int64
            ),
            "inactive_wrapper_write_count": int(clamp.inactive_wrapper_write_count),
            "requested_active_sigmas": np.asarray(
                requested_active_sigmas, dtype=np.float32
            ),
            "observed_active_sigmas": np.asarray(
                clamp.active_call_sigmas, dtype=np.float32
            ),
            "model_input_future_clamp_errors": np.asarray(
                clamp.model_input_future_clamp_errors, dtype=np.float64
            ),
            "returned_future_velocity_overwrite_errors": np.asarray(
                clamp.returned_future_velocity_overwrite_errors, dtype=np.float64
            ),
            "action_input_errors": np.asarray(
                clamp.action_input_errors, dtype=np.float64
            ),
            "action_output_errors": np.asarray(
                clamp.action_output_errors, dtype=np.float64
            ),
            "clamped_call_indices": np.asarray(clamp.clamped_call_indices, dtype=np.int64),
            "x0_sigmas": np.asarray([item["sigma"] for item in x0.records], dtype=np.float32),
            "x0_vision_hashes": [tensor_digest(item["samples"][0]["vision_0"]) for item in x0.records],
            "x0_action_hashes": [tensor_digest(item["samples"][0]["action"]) for item in x0.records],
        }
        if mode == "gaussian":
            recipient_future = recipient.target[mask].double()
            donor_future = donor.target[mask].double()
            target_future = target[mask].double()
            donor_norm = torch.linalg.vector_norm(donor_future)
            donor_distance = torch.linalg.vector_norm(donor_future - recipient_future)
            target_norm = torch.linalg.vector_norm(target_future)
            target_distance = torch.linalg.vector_norm(target_future - recipient_future)
            audit.update(
                {
                    "gaussian_donor_norm": float(donor_norm),
                    "gaussian_target_norm": float(target_norm),
                    "gaussian_donor_distance": float(donor_distance),
                    "gaussian_target_distance": float(target_distance),
                    "gaussian_norm_relative_error": float(
                        torch.abs(target_norm - donor_norm) / torch.clamp(donor_norm, min=1e-12)
                    ),
                    "gaussian_distance_relative_error": float(
                        torch.abs(target_distance - donor_distance)
                        / torch.clamp(donor_distance, min=1e-12)
                    ),
                }
            )
        return samples, audit

    def _encode_executed_donor(
        self,
        obs: dict[str, Any],
        current_sample: dict[str, Any],
        state_hash: str,
        record_id: str,
    ) -> FutureRecord:
        path = resolve_donor_path(str(obs["research_donor_path"]))
        with np.load(path, allow_pickle=False) as payload:
            video = np.asarray(payload["video"])
            action = np.asarray(payload["action"], dtype=np.float32) if "action" in payload else None
        expected = (self._base.cfg.action_chunk_size + 1, self._base.cfg.image_height, self._base.cfg.image_width, 3)
        if video.shape != expected or video.dtype != np.uint8:
            raise ValueError(f"executed donor video must be uint8 with shape {expected}, got {video.shape} {video.dtype}")
        if action is not None and action.shape != (self._base.cfg.action_chunk_size, self._base.cfg.action_dim):
            raise ValueError(f"executed donor action has invalid shape {action.shape}")

        processed_frames = []
        for frame in video:
            frame_obs = dict(obs)
            frame_obs["observation/image"] = frame
            processed = self._base._build_sample(frame_obs)["video"][:, 0]
            processed_frames.append(processed)
        processed_video_uint8 = torch.stack(processed_frames, dim=1).unsqueeze(0).cuda()
        current_frame = current_sample["video"][:, 0]
        first_frame_error = float((processed_video_uint8[0, :, 0].cpu() - current_frame.cpu()).abs().max())
        if first_frame_error != 0.0:
            raise ValueError(f"executed donor current frame differs after official preprocessing: {first_frame_error}")
        processed_video = processed_video_uint8.to(torch.float32) / 127.5 - 1.0
        encoded_padded = self._base.model.encode(processed_video).detach().cpu()
        image_size = current_sample["image_size"].reshape(-1)
        content_height = int(image_size[2].item())
        content_width = int(image_size[3].item())
        spatial_factor = int(self._base.model.config.latent_downsample_factor)
        latent_height = content_height // spatial_factor
        latent_width = content_width // spatial_factor
        encoded = encoded_padded[..., :latent_height, :latent_width].contiguous()
        latent = encoded.reshape(-1)
        return FutureRecord(
            record_id=record_id,
            source="executed",
            state_hash=state_hash,
            seed=None,
            vision_shape=tuple(int(item) for item in encoded.shape),
            target=latent,
            path_noise=None,
            initial_state_hash=None,
            action=action,
            sigmas=(),
            x0_vision_hashes=(),
            x0_action_hashes=(),
        )

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        mode = str(obs.get("research_mode", "native"))
        seed = int(obs["research_seed"]) if "research_seed" in obs else self._base._next_seed()
        record_id = str(obs.get("research_id", f"native-{seed}-{time.time_ns()}"))
        sample = self._base._build_sample(obs)
        state_hash = sample_fingerprint(sample)
        attention_layers = (
            tuple(int(item) for item in np.asarray(obs["research_attention_exclude_layers"]).reshape(-1))
            if "research_attention_exclude_layers" in obs
            else ()
        )
        attention_scope = str(obs.get("research_attention_exclude_scope", "action"))
        attention_mode = str(obs.get("research_attention_mode", "exclude"))
        attention_cache_id = (
            str(obs["research_attention_cache_id"])
            if "research_attention_cache_id" in obs
            else None
        )
        attention_instrumented = (
            "research_attention_exclude_layers" in obs or attention_mode != "exclude"
        )
        if attention_instrumented and self.attention_excluder is None:
            raise ValueError("server was not started with --attention-instrumentation")
        attention_context = (
            self.attention_excluder.activate(
                attention_layers,
                scope=attention_scope,
                mode=attention_mode,
                cache_id=attention_cache_id,
            )
            if self.attention_excluder is not None
            else nullcontext()
        )

        with (
            self._base._lock,
            torch.inference_mode(),
            attention_context,
        ):
            if mode == "register_executed":
                record = self._encode_executed_donor(obs, sample, state_hash, record_id)
                self._remember(record)
                outputs = {
                    "action": np.zeros((self._base.cfg.action_chunk_size, self._base.cfg.action_dim), np.float32),
                    "research_id": record_id,
                    "research_mode": mode,
                    "research_state_hash": state_hash,
                    "research_parameter_probe_hash": self.parameter_probe_hash,
                    "research_future_hash": tensor_digest(record.target),
                    "research_source": record.source,
                    "research_attention_exclude_layers": np.asarray(attention_layers, dtype=np.int64),
                    "research_attention_exclude_scope": attention_scope,
                }
                if bool(obs.get("research_return_video", False)):
                    encoded = record.target.reshape(record.vision_shape).to(
                        device=next(self._base.model.net.parameters()).device
                    )
                    decoded = self._base.model.decode(encoded)
                    video = (
                        ((decoded[0].clamp(-1, 1) + 1) * 127.5)
                        .to(torch.uint8)
                        .permute(1, 2, 3, 0)
                    )
                    outputs["video"] = video.cpu().numpy()
                return outputs

            data_batch = self._batch(sample)
            if mode == "native":
                samples, record = self._native(data_batch, seed)
                record.record_id = record_id
                record.state_hash = state_hash
                self._remember(record)
                outputs: dict[str, Any] = {
                    "action": record.action,
                    "research_id": record_id,
                    "research_mode": mode,
                    "research_seed": seed,
                    "research_state_hash": state_hash,
                    "research_parameter_probe_hash": self.parameter_probe_hash,
                    "research_future_hash": tensor_digest(record.target),
                    "research_path_noise_hash": tensor_digest(record.path_noise),
                    "research_initial_state_hash": record.initial_state_hash,
                    "research_sigmas": np.asarray(record.sigmas, dtype=np.float32),
                    "research_x0_sigmas": np.asarray(record.sigmas, dtype=np.float32),
                    "research_x0_vision_hashes": list(record.x0_vision_hashes),
                    "research_x0_action_hashes": list(record.x0_action_hashes),
                    "research_attention_exclude_layers": np.asarray(attention_layers, dtype=np.int64),
                    "research_attention_exclude_scope": attention_scope,
                }
                if bool(obs.get("research_return_video", False)):
                    decoded = self._base.model.decode(samples["vision"][0])
                    video = ((decoded[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0)
                    outputs["video"] = video.cpu().numpy()
            elif mode in {"none", "self", "donor", "gaussian"}:
                recipient_id = str(obs["research_recipient_id"])
                donor_id = str(obs.get("research_donor_id", recipient_id))
                recipient = self.registry[recipient_id]
                donor = self.registry[donor_id]
                for role, record in (("recipient", recipient), ("donor", donor)):
                    if record.state_hash != state_hash:
                        raise ValueError(
                            f"{role} state mismatch: request={state_hash}, registered={record.state_hash}"
                        )
                if recipient.seed != seed:
                    raise ValueError(f"recipient seed is {recipient.seed}, intervention requested seed {seed}")
                samples, audit = self._intervene(
                    data_batch,
                    seed=seed,
                    recipient=recipient,
                    donor=donor,
                    mode=mode,
                    gaussian_seed=int(obs.get("research_gaussian_seed", 1223)),
                    timing_steps=(
                        tuple(int(item) for item in np.asarray(obs["research_timing_steps"]).reshape(-1))
                        if "research_timing_steps" in obs
                        else None
                    ),
                )
                action = self._format_action(samples)
                outputs = {
                    "action": action,
                    "research_id": record_id,
                    "research_mode": mode,
                    "research_seed": seed,
                    "research_state_hash": state_hash,
                    "research_parameter_probe_hash": self.parameter_probe_hash,
                    "research_recipient_id": recipient_id,
                    "research_donor_id": donor_id,
                    **{f"research_{key}": value for key, value in audit.items()},
                    "research_attention_exclude_layers": np.asarray(attention_layers, dtype=np.int64),
                    "research_attention_exclude_scope": attention_scope,
                }
                if donor.action is not None:
                    direction = donor.action.astype(np.float64) - recipient.action.astype(np.float64)
                    denominator = float(np.square(direction).sum())
                    projection_applicable = denominator > 0
                    outputs["research_action_donor_projection_applicable"] = (
                        projection_applicable
                    )
                    outputs["research_action_donor_projection"] = (
                        float(((action - recipient.action) * direction).sum() / denominator)
                        if projection_applicable
                        else None
                    )
            else:
                raise ValueError(f"unknown research_mode: {mode!r}")

        outputs["research_infer_ms"] = (time.monotonic() - start) * 1000.0
        outputs["research_attention_interface"] = {
            "layers": self.attention_excluder.num_layers if self.attention_excluder else 36,
            "action_tokens": (
                self.attention_excluder.action_tokens
                if self.attention_excluder
                else self._base.cfg.action_chunk_size + self._base.cfg.history_length
            ),
            "video_latent_frames": (
                self.attention_excluder.video_latent_frames if self.attention_excluder else 9
            ),
            "excluded_keys_values": "future_video",
            "instrumented_server": self.attention_excluder is not None,
            "intervention_requested": attention_instrumented,
            "text_kv_reuse": self.attention_excluder is None,
            "mode": attention_mode,
            "cache_id": attention_cache_id,
            "cache_call_counts": (
                self.attention_excluder.cache_summary(attention_cache_id)
                if self.attention_excluder is not None
                and attention_cache_id is not None
                and attention_cache_id in self.attention_excluder.kv_caches
                else {}
            ),
            "scopes": {
                "action": "exclude only for action queries",
                "nonfuture": (
                    "exclude for current-video and action queries to block indirect "
                    "future-to-current-to-action paths"
                ),
            },
        }
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--registry-limit", type=int, default=256)
    parser.add_argument("--attention-instrumentation", action="store_true")
    args = parser.parse_args()

    # The public attention backend filter emits a Loguru debug call from inside
    # a torch.compile region when I4_ATTN_BACKENDS is set. Loguru reaches
    # sys._getframe, which Dynamo cannot trace. Silence only that debug method
    # before model construction so an explicitly pinned public backend remains
    # usable without changing attention math.
    if os.environ.get("I4_ATTN_BACKENDS"):
        from cosmos_framework.model.attention.utils import environment as attention_environment

        attention_environment.log.debug = lambda *_args, **_kwargs: None

    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
        _load_openpi_websocket_policy_server,
    )

    native_build_setup_args = RobolabPolicyService._build_setup_args

    def build_without_guardrails(self, server_args):
        updates = {"guardrails": False}
        if args.attention_instrumentation:
            updates.update(use_torch_compile=False, use_cuda_graphs=False)
        return native_build_setup_args(self, server_args).model_copy(update=updates)

    RobolabPolicyService._build_setup_args = build_without_guardrails
    service = ResearchPolicyService(
        RobolabServerArgs(checkpoint_path=str(args.checkpoint), seed=args.seed),
        registry_limit=args.registry_limit,
        attention_instrumentation=args.attention_instrumentation,
    )
    if service.registry:
        raise RuntimeError("research registry was not empty at server startup")
    server_cls = _load_openpi_websocket_policy_server()
    hostname = socket.gethostname()
    print(
        f"research server {hostname} listening on 0.0.0.0:{args.port} "
        "registry_entries=0",
        flush=True,
    )
    server_cls(policy=service, host="0.0.0.0", port=args.port, metadata={}).serve_forever()


if __name__ == "__main__":
    main()
