#!/usr/bin/env python3
"""Run an outcome-labeled synthetic-observation Cosmos 3 donor-clamp pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from imagined_future.cosmos3_interventions import (
    GuidedFutureClamp,
    PreparedLayoutCapture,
    SamplerInitialStateCapture,
    SamplerVelocityWrapper,
    gaussian_target_on_mask,
    temporal_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipient-seed", type=int, default=0)
    parser.add_argument("--donor-seed", type=int, default=1)
    parser.add_argument("--gaussian-seed", type=int, default=1223)
    return parser.parse_args()


def frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def projection(value: torch.Tensor, origin: torch.Tensor, target: torch.Tensor) -> float:
    direction = (target - origin).double().reshape(-1)
    denominator = torch.dot(direction, direction)
    if denominator == 0:
        raise ValueError("native action donor and recipient are identical")
    return float(torch.dot((value - origin).double().reshape(-1), direction) / denominator)


def main() -> None:
    args = parse_args()
    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
        _build_data_batch_from_sample,
    )

    native_build_setup_args = RobolabPolicyService._build_setup_args

    def build_without_guardrails(self, server_args):
        return native_build_setup_args(self, server_args).model_copy(update={"guardrails": False})

    RobolabPolicyService._build_setup_args = build_without_guardrails
    service = RobolabPolicyService(
        RobolabServerArgs(checkpoint_path=str(args.checkpoint), deterministic_seed=True, seed=args.recipient_seed)
    )
    image = frame(args.asset_video)
    observation = {
        "observation/image": image,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "Pick up the banana and place it in the bowl.",
    }

    def batch() -> dict:
        return _build_data_batch_from_sample(service._build_sample(observation))

    def native(seed: int, sampler=None, builder=None):
        return service.model.generate_samples_from_batch(
            batch(),
            sampler=sampler,
            velocity_postprocess_builder=builder,
            guidance=service.cfg.guidance,
            seed=[seed],
            num_steps=service.cfg.num_steps,
            shift=service.cfg.shift,
        )

    with torch.inference_mode():
        initial_capture = SamplerInitialStateCapture(service.model.sampler)
        recipient = native(args.recipient_seed, sampler=initial_capture)
        donor = native(args.donor_seed)
        if initial_capture.initial_state is None:
            raise RuntimeError("recipient initial noise was not captured")

        layout_capture = PreparedLayoutCapture()
        # Capture geometry without changing the native CFG path or sampler.
        _ = native(args.recipient_seed, builder=layout_capture)
        if layout_capture.layout is None:
            raise RuntimeError("layout was not captured")
        vision_slice = layout_capture.layout.modality(0, "vision_0")
        temporal_size = vision_slice.shape[-3]
        future_frames = tuple(range(1, temporal_size))
        selected = temporal_mask(vision_slice.shape, future_frames, device=torch.device("cpu"))
        recipient_target = recipient["vision"][0].detach().cpu().reshape(-1)
        donor_target = donor["vision"][0].detach().cpu().reshape(-1)
        path_noise = initial_capture.initial_state[0][vision_slice.start : vision_slice.stop].detach().cpu()
        gaussian_target = gaussian_target_on_mask(
            recipient_target,
            donor_target,
            selected,
            seed=args.gaussian_seed,
        )

        intervention_results = {}
        clamp_audits = {}
        for name, target in {
            "self": recipient_target,
            "donor": donor_target,
            "gaussian": gaussian_target,
        }.items():
            capture = PreparedLayoutCapture()
            clamp = GuidedFutureClamp(capture, target, path_noise, future_frames)
            sampler = SamplerVelocityWrapper(service.model.sampler, clamp.wrap_velocity)
            intervention_results[name] = native(args.recipient_seed, sampler=sampler, builder=capture)
            clamp_audits[name] = {
                "sigmas": clamp.calls,
                "maximum_action_input_error": clamp.maximum_action_input_error,
                "maximum_action_output_error": clamp.maximum_action_output_error,
            }

        decoded = {}
        for name, latent in {
            "recipient": recipient["vision"][0],
            "donor": donor["vision"][0],
            **{key: value["vision"][0] for key, value in intervention_results.items()},
        }.items():
            decoded[name] = service.model.decode(latent).detach().float().cpu()

    recipient_action = recipient["action"][0][1:, :8].detach().cpu()
    donor_action = donor["action"][0][1:, :8].detach().cpu()
    action_distance = float(torch.linalg.vector_norm(donor_action - recipient_action))
    results = {}
    for name, sample in intervention_results.items():
        action = sample["action"][0][1:, :8].detach().cpu()
        vision = sample["vision"][0].detach().cpu().reshape(-1)
        target = {"self": recipient_target, "donor": donor_target, "gaussian": gaussian_target}[name]
        results[name] = {
            "action_donor_projection": projection(action, recipient_action, donor_action),
            "action_l2_from_recipient": float(torch.linalg.vector_norm(action - recipient_action)),
            "future_target_maximum_absolute_error": float((vision[selected] - target[selected]).abs().max()),
            "decoded_l1_to_recipient": float((decoded[name] - decoded["recipient"]).abs().mean()),
            "decoded_l1_to_donor": float((decoded[name] - decoded["donor"]).abs().mean()),
        }

    report = {
        "label": "synthetic-observation engineering pilot; not a RoboLab causal result",
        "recipient_seed": args.recipient_seed,
        "donor_seed": args.donor_seed,
        "future_latent_frames": list(future_frames),
        "native_action_l2": action_distance,
        "gaussian_relative_norm_error": float(
            abs(torch.linalg.vector_norm(gaussian_target[selected]) - torch.linalg.vector_norm(donor_target[selected]))
            / torch.linalg.vector_norm(donor_target[selected])
        ),
        "gaussian_relative_distance_error": float(
            abs(
                torch.linalg.vector_norm(gaussian_target[selected] - recipient_target[selected])
                - torch.linalg.vector_norm(donor_target[selected] - recipient_target[selected])
            )
            / torch.linalg.vector_norm(donor_target[selected] - recipient_target[selected])
        ),
        "clamp_audits": clamp_audits,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    for audit in clamp_audits.values():
        if audit["maximum_action_input_error"] != 0.0 or audit["maximum_action_output_error"] != 0.0:
            raise SystemExit("action-coordinate localization failed")


if __name__ == "__main__":
    main()
