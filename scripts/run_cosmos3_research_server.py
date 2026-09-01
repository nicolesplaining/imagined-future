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

ALLOWED_DONOR_ROOT = Path("/lambda/nfs/imagined-future/results")


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

    def __init__(self, args: Any, *, registry_limit: int = 256) -> None:
        from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService

        self._base = RobolabPolicyService(args)
        if self._base.cfg.action_space != "joint_pos":
            raise ValueError("the research server currently supports the released joint-position policy only")
        self.registry_limit = int(registry_limit)
        self.registry: OrderedDict[str, FutureRecord] = OrderedDict()
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
        if mode == "self":
            target = recipient.target
        elif mode == "donor":
            target = donor.target
        elif mode == "gaussian":
            target = gaussian_target_on_mask(recipient.target, donor.target, mask, seed=gaussian_seed)
        else:
            raise ValueError(f"unsupported intervention mode: {mode!r}")
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

        sampler = SamplerVelocityWrapper(self._base.model.sampler, transform)
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
        output_target = samples["vision"][0].detach().cpu().reshape(-1)
        audit = {
            "target_hash": tensor_digest(target),
            "output_future_hash": tensor_digest(output_target),
            "target_future_max_error": float((output_target[mask] - target[mask]).abs().max()),
            "maximum_action_input_error": clamp.maximum_action_input_error,
            "maximum_action_output_error": clamp.maximum_action_output_error,
            "sigmas": np.asarray(clamp.calls, dtype=np.float32),
            "clamped_call_indices": np.asarray(clamp.clamped_call_indices, dtype=np.int64),
            "x0_sigmas": np.asarray([item["sigma"] for item in x0.records], dtype=np.float32),
            "x0_vision_hashes": [tensor_digest(item["samples"][0]["vision_0"]) for item in x0.records],
            "x0_action_hashes": [tensor_digest(item["samples"][0]["action"]) for item in x0.records],
        }
        return samples, audit

    def _encode_executed_donor(
        self,
        obs: dict[str, Any],
        current_sample: dict[str, Any],
        state_hash: str,
        record_id: str,
    ) -> FutureRecord:
        path = Path(str(obs["research_donor_path"])).expanduser().resolve()
        if not path.is_relative_to(ALLOWED_DONOR_ROOT.resolve()):
            raise ValueError(f"donor path must be under {ALLOWED_DONOR_ROOT}")
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

        with self._base._lock, torch.inference_mode():
            if mode == "register_executed":
                record = self._encode_executed_donor(obs, sample, state_hash, record_id)
                self._remember(record)
                return {
                    "action": np.zeros((self._base.cfg.action_chunk_size, self._base.cfg.action_dim), np.float32),
                    "research_id": record_id,
                    "research_mode": mode,
                    "research_state_hash": state_hash,
                    "research_parameter_probe_hash": self.parameter_probe_hash,
                    "research_future_hash": tensor_digest(record.target),
                    "research_source": record.source,
                }

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
                    "research_x0_vision_hashes": list(record.x0_vision_hashes),
                    "research_x0_action_hashes": list(record.x0_action_hashes),
                }
                if bool(obs.get("research_return_video", False)):
                    decoded = self._base.model.decode(samples["vision"][0])
                    video = ((decoded[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0)
                    outputs["video"] = video.cpu().numpy()
            elif mode in {"self", "donor", "gaussian"}:
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
                }
                if donor.action is not None:
                    direction = donor.action.astype(np.float64) - recipient.action.astype(np.float64)
                    denominator = float(np.square(direction).sum())
                    outputs["research_action_donor_projection"] = (
                        float(((action - recipient.action) * direction).sum() / denominator)
                        if denominator > 0
                        else float("nan")
                    )
            else:
                raise ValueError(f"unknown research_mode: {mode!r}")

        outputs["research_infer_ms"] = (time.monotonic() - start) * 1000.0
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--registry-limit", type=int, default=256)
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
        return native_build_setup_args(self, server_args).model_copy(update={"guardrails": False})

    RobolabPolicyService._build_setup_args = build_without_guardrails
    service = ResearchPolicyService(
        RobolabServerArgs(checkpoint_path=str(args.checkpoint), seed=args.seed),
        registry_limit=args.registry_limit,
    )
    server_cls = _load_openpi_websocket_policy_server()
    hostname = socket.gethostname()
    print(f"research server {hostname} listening on 0.0.0.0:{args.port}", flush=True)
    server_cls(policy=service, host="0.0.0.0", port=args.port, metadata={}).serve_forever()


if __name__ == "__main__":
    main()
