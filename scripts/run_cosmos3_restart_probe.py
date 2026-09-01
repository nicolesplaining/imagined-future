#!/usr/bin/env python3
"""Capture a fixed Cosmos 3 native request for cross-process determinism audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from run_cosmos3_server_audit import first_frame


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--asset-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    request = {
        "observation/image": first_frame(args.asset_video),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "Pick up the banana and place it in the bowl.",
        "research_mode": "native",
        "research_seed": args.seed,
        "research_id": args.study_id,
    }
    response = WebsocketClientPolicy(args.host, args.port).infer(request)
    action = np.asarray(response["action"])
    report = {
        "study_id": args.study_id,
        "seed": args.seed,
        "state_hash": response["research_state_hash"],
        "parameter_probe_hash": response["research_parameter_probe_hash"],
        "future_hash": response["research_future_hash"],
        "path_noise_hash": response["research_path_noise_hash"],
        "initial_state_hash": response["research_initial_state_hash"],
        "action_hash": array_digest(action),
        "action": action.tolist(),
        "x0_vision_hashes": response["research_x0_vision_hashes"],
        "x0_action_hashes": response["research_x0_action_hashes"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
