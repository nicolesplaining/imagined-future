#!/usr/bin/env python3
"""Run a real-checkpoint Cosmos 3 token census and exact no-op audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from imagined_future.cosmos3_interventions import (
    GuidedX0Recorder,
    PreparedLayoutCapture,
    SamplerVelocityWrapper,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode first frame of {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def request(image: np.ndarray) -> dict[str, object]:
    return {
        "observation/image": image,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "Pick up the banana and place it in the bowl.",
    }


def main() -> None:
    args = parse_args()
    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
        _build_data_batch_from_sample,
    )

    # The action policy does not use output guardrails, but the generic public
    # inference setup enables a separate gated guardrail checkpoint by default.
    # Keep this audit limited to the released policy checkpoint.
    native_build_setup_args = RobolabPolicyService._build_setup_args

    def build_without_guardrails(self, server_args):
        setup = native_build_setup_args(self, server_args)
        return setup.model_copy(update={"guardrails": False})

    RobolabPolicyService._build_setup_args = build_without_guardrails

    service = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=str(args.checkpoint),
            decode_video=False,
            deterministic_seed=True,
            seed=args.seed,
        )
    )
    image = first_frame(args.asset_video)

    def batch() -> dict:
        return _build_data_batch_from_sample(service._build_sample(request(image)))

    generation_kwargs = {
        "guidance": service.cfg.guidance,
        "seed": [args.seed],
        "num_steps": service.cfg.num_steps,
        "shift": service.cfg.shift,
    }
    with torch.inference_mode():
        native = service.model.generate_samples_from_batch(batch(), **generation_kwargs)
        layout_capture = PreparedLayoutCapture()
        recorder = GuidedX0Recorder(layout_capture)
        wrapped_sampler = SamplerVelocityWrapper(service.model.sampler, recorder.wrap_velocity)
        audited = service.model.generate_samples_from_batch(
            batch(),
            sampler=wrapped_sampler,
            velocity_postprocess_builder=layout_capture,
            **generation_kwargs,
        )

    native_action = native["action"][0].detach().cpu()
    audited_action = audited["action"][0].detach().cpu()
    native_vision = native["vision"][0].detach().cpu()
    audited_vision = audited["vision"][0].detach().cpu()
    if layout_capture.layout is None or layout_capture.generation_data is None:
        raise RuntimeError("layout capture did not run")

    action_error = float((native_action - audited_action).abs().max())
    vision_error = float((native_vision - audited_vision).abs().max())
    report = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "native_action_shape": list(native_action.shape),
        "native_vision_shape": list(native_vision.shape),
        "flat_layout": [
            [
                {"name": item.name, "start": item.start, "stop": item.stop, "shape": list(item.shape)}
                for item in sample
            ]
            for sample in layout_capture.layout.samples
        ],
        "denoising_sigmas": [record["sigma"] for record in recorder.records],
        "post_guidance_x0_shapes": [
            {
                name: list(value.shape)
                for name, value in record["samples"][0].items()
            }
            for record in recorder.records
        ],
        "exact_action_equal": bool(torch.equal(native_action, audited_action)),
        "exact_vision_equal": bool(torch.equal(native_vision, audited_vision)),
        "maximum_action_absolute_error": action_error,
        "maximum_vision_absolute_error": vision_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if action_error != 0.0 or vision_error != 0.0:
        raise SystemExit("exact no-op audit failed")


if __name__ == "__main__":
    main()
