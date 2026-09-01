#!/usr/bin/env python3
"""Localize Cosmos 3 future-video K/V mediation into action queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from run_cosmos3_server_audit import first_frame


def projection(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> float:
    direction = donor.astype(np.float64) - recipient.astype(np.float64)
    denominator = float(np.square(direction).sum())
    return float(((value.astype(np.float64) - recipient) * direction).sum() / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--asset-video", type=Path, required=True)
    parser.add_argument("--recorded-hdf5", type=Path)
    parser.add_argument("--branch-summary", type=Path)
    parser.add_argument("--branch-step", type=int)
    parser.add_argument("--prompt", default="Pick up the banana and place it in the bowl.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--recipient-seed", type=int, default=0)
    parser.add_argument("--donor-seed", type=int, default=1)
    parser.add_argument("--scan-nonfuture-layers", action="store_true")
    args = parser.parse_args()

    prompt = args.prompt
    branch_step = args.branch_step
    if args.branch_summary is not None:
        branch_summary = json.loads(args.branch_summary.read_text())
        prompt = str(branch_summary["instruction"])
        if branch_step is None:
            branch_step = int(branch_summary["branch_step"])

    client = WebsocketClientPolicy(args.host, args.port)
    joint_position = np.zeros(7, dtype=np.float32)
    gripper_position = np.zeros(1, dtype=np.float32)
    proprio_source = "zeros"
    if args.recorded_hdf5 is not None:
        if branch_step is None or branch_step <= 0:
            raise ValueError("--recorded-hdf5 requires a positive --branch-step")
        import h5py

        with h5py.File(args.recorded_hdf5, "r") as stream:
            recorded_joint_position = np.asarray(
                stream["data/demo_0/states/articulation/robot/joint_position"]
                [branch_step - 1],
                dtype=np.float32,
            )
        joint_position = recorded_joint_position[:7].copy()
        gripper_position = np.clip(
            recorded_joint_position[7:8] / (np.pi / 4), 0.0, 1.0
        ).astype(np.float32)
        proprio_source = "noise-free recorded post-step simulator state"

    base = {
        "observation/image": first_frame(args.asset_video),
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }

    def native(seed: int, label: str) -> dict:
        return client.infer(
            {
                **base,
                "research_mode": "native",
                "research_seed": seed,
                "research_id": f"{args.study_id}-{label}",
            }
        )

    recipient_response = native(args.recipient_seed, "recipient")
    donor_response = native(args.donor_seed, "donor")
    recipient = np.asarray(recipient_response["action"])
    donor = np.asarray(donor_response["action"])

    def transplant(
        label: str,
        layers: list[int],
        *,
        scope: str = "action",
        instrument: bool = True,
    ) -> tuple[dict, np.ndarray]:
        attention_fields = (
            {
                "research_attention_exclude_layers": layers,
                "research_attention_exclude_scope": scope,
            }
            if instrument
            else {}
        )
        response = client.infer(
            {
                **base,
                "research_mode": "donor",
                "research_seed": args.recipient_seed,
                "research_id": f"{args.study_id}-{label}",
                "research_recipient_id": f"{args.study_id}-recipient",
                "research_donor_id": f"{args.study_id}-donor",
                **attention_fields,
            }
        )
        return response, np.asarray(response["action"])

    _ordinary_response, ordinary = transplant("ordinary", [], instrument=False)
    baseline_response, baseline = transplant("baseline", [])
    ordinary_zero_gate_error = float(np.abs(ordinary - baseline).max())
    if ordinary_zero_gate_error != 0.0:
        raise RuntimeError(
            "an explicit empty attention intervention differs from the implicit zero-gate path by "
            f"{ordinary_zero_gate_error}"
        )
    noop_response, noop = transplant("noop", [])
    noop_error = float(np.abs(noop - baseline).max())
    if noop_error != 0.0:
        raise RuntimeError(f"empty attention exclusion changed the action by {noop_error}")
    _barrier_noop_response, barrier_noop = transplant(
        "nonfuture-noop", [], scope="nonfuture"
    )
    barrier_noop_error = float(np.abs(barrier_noop - baseline).max())
    if barrier_noop_error != 0.0:
        raise RuntimeError(
            f"empty nonfuture attention exclusion changed the action by {barrier_noop_error}"
        )

    rows = []
    for layer in range(36):
        response, action = transplant(f"layer-{layer}", [layer])
        rows.append(
            {
                "layer": layer,
                "action_donor_projection": projection(action, recipient, donor),
                "action_l2_from_baseline_transplant": float(np.linalg.norm(action - baseline)),
                "target_future_max_error": float(response["research_target_future_max_error"]),
            }
        )
    all_response, all_action = transplant("all-action-layers", list(range(36)))
    barrier_response, barrier_action = transplant(
        "all-nonfuture-layers", list(range(36)), scope="nonfuture"
    )
    baseline_projection = projection(baseline, recipient, donor)
    for row in rows:
        row["mediation_loss_from_baseline"] = baseline_projection - row["action_donor_projection"]

    nonfuture_rows = []
    if args.scan_nonfuture_layers:
        for layer in range(36):
            response, action = transplant(f"nonfuture-layer-{layer}", [layer], scope="nonfuture")
            action_projection = projection(action, recipient, donor)
            nonfuture_rows.append(
                {
                    "layer": layer,
                    "action_donor_projection": action_projection,
                    "mediation_loss_from_baseline": baseline_projection - action_projection,
                    "action_l2_from_baseline_transplant": float(np.linalg.norm(action - baseline)),
                    "target_future_max_error": float(response["research_target_future_max_error"]),
                }
            )

    report = {
        "scope": "excluded public-observation attention-interface calibration",
        "study_id": args.study_id,
        "recipient_seed": args.recipient_seed,
        "donor_seed": args.donor_seed,
        "recorded_hdf5": str(args.recorded_hdf5) if args.recorded_hdf5 else None,
        "branch_step": branch_step,
        "branch_summary": str(args.branch_summary) if args.branch_summary else None,
        "prompt": prompt,
        "proprio_source": proprio_source,
        "native_action_l2": float(np.linalg.norm(donor - recipient)),
        "empty_exclusion_maximum_action_error": noop_error,
        "empty_nonfuture_exclusion_maximum_action_error": barrier_noop_error,
        "implicit_vs_explicit_zero_gate_maximum_action_error": ordinary_zero_gate_error,
        "baseline_transplant_action_donor_projection": baseline_projection,
        "all_layer_exclusion_action_donor_projection": projection(all_action, recipient, donor),
        "all_layer_mediation_loss": baseline_projection - projection(all_action, recipient, donor),
        "all_layer_target_future_max_error": float(all_response["research_target_future_max_error"]),
        "all_layer_nonfuture_barrier_action_donor_projection": projection(
            barrier_action, recipient, donor
        ),
        "all_layer_nonfuture_barrier_mediation_loss": (
            baseline_projection - projection(barrier_action, recipient, donor)
        ),
        "all_layer_nonfuture_barrier_target_future_max_error": float(
            barrier_response["research_target_future_max_error"]
        ),
        "attention_interface": baseline_response.get("research_attention_interface"),
        "action_query_layers": rows,
        "nonfuture_barrier_layers": nonfuture_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
