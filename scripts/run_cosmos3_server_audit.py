#!/usr/bin/env python3
"""Exercise the Cosmos 3 research server on one fixed public observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--asset-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--donor-video", type=Path, required=True)
    return parser.parse_args()


def first_frame(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        decoded = next(container.decode(video=0), None)
    if decoded is None:
        raise RuntimeError(f"could not decode {path}")
    rgb = decoded.to_ndarray(format="rgb24")
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    resized = torch.nn.functional.interpolate(tensor, size=(540, 640), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).to(torch.uint8).numpy()


def metadata(response: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in response.items():
        if key == "action" or key == "video":
            continue
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def main() -> None:
    args = parse_args()
    image = first_frame(args.asset_video)
    base = {
        "observation/image": image,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "Pick up the banana and place it in the bowl.",
    }
    client = WebsocketClientPolicy(args.host, args.port)

    def request(**research: Any) -> dict[str, Any]:
        return client.infer({**base, **research})

    recipient = request(research_mode="native", research_seed=0, research_id="audit-recipient")
    repeated = request(research_mode="native", research_seed=0, research_id="audit-recipient-repeat")
    donor = request(research_mode="native", research_seed=1, research_id="audit-donor")
    common = {"research_seed": 0, "research_recipient_id": "audit-recipient"}
    self_clamp = request(research_mode="self", research_id="audit-self", **common)
    donor_clamp = request(
        research_mode="donor",
        research_id="audit-donor-clamp",
        research_donor_id="audit-donor",
        **common,
    )
    gaussian = request(
        research_mode="gaussian",
        research_id="audit-gaussian",
        research_donor_id="audit-donor",
        research_gaussian_seed=1223,
        **common,
    )

    args.donor_video.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.donor_video, video=np.repeat(image[None], 33, axis=0))
    registered = request(
        research_mode="register_executed",
        research_id="audit-executed",
        research_donor_path=str(args.donor_video),
    )
    executed_clamp = request(
        research_mode="donor",
        research_id="audit-executed-clamp",
        research_donor_id="audit-executed",
        **common,
    )

    native_action_error = float(np.max(np.abs(recipient["action"] - repeated["action"])))
    report = {
        "label": "fixed public-observation server engineering audit; not a RoboLab causal result",
        "native_recomputation": {
            "action_maximum_absolute_error": native_action_error,
            "future_hash_equal": recipient["research_future_hash"] == repeated["research_future_hash"],
            "x0_vision_hashes_equal": (
                recipient["research_x0_vision_hashes"] == repeated["research_x0_vision_hashes"]
            ),
            "x0_action_hashes_equal": recipient["research_x0_action_hashes"] == repeated["research_x0_action_hashes"],
        },
        "native_action_l2": float(np.linalg.norm(donor["action"] - recipient["action"])),
        "responses": {
            "recipient": metadata(recipient),
            "repeated": metadata(repeated),
            "donor": metadata(donor),
            "self": metadata(self_clamp),
            "donor_clamp": metadata(donor_clamp),
            "gaussian": metadata(gaussian),
            "registered_executed": metadata(registered),
            "executed_clamp": metadata(executed_clamp),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

    hashes = {value["research_state_hash"] for value in report["responses"].values()}
    if len(hashes) != 1:
        raise SystemExit(f"state fingerprints diverged: {hashes}")
    if native_action_error != 0.0 or not report["native_recomputation"]["future_hash_equal"]:
        raise SystemExit("native recomputation was not exact")
    for name in ("self", "donor_clamp", "gaussian", "executed_clamp"):
        response = report["responses"][name]
        if response["research_maximum_action_input_error"] != 0.0:
            raise SystemExit(f"{name}: action input coordinates were mutated")
        if response["research_maximum_action_output_error"] != 0.0:
            raise SystemExit(f"{name}: action output coordinates were overwritten")


if __name__ == "__main__":
    main()
