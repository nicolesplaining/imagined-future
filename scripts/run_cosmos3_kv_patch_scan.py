#!/usr/bin/env python3
"""Scan token-count-preserving Cosmos 3 future-K/V content mediation."""

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


def current_request(args: argparse.Namespace) -> tuple[dict, dict]:
    prompt = args.prompt
    branch_step = args.branch_step
    if args.branch_summary is not None:
        branch_summary = json.loads(args.branch_summary.read_text())
        prompt = str(branch_summary["instruction"])
        if branch_step is None:
            branch_step = int(branch_summary["branch_step"])

    joint_position = np.zeros(7, dtype=np.float32)
    gripper_position = np.zeros(1, dtype=np.float32)
    proprio_source = "zeros"
    if args.recorded_hdf5 is not None:
        if branch_step is None or branch_step <= 0:
            raise ValueError("--recorded-hdf5 requires a positive branch step")
        import h5py

        with h5py.File(args.recorded_hdf5, "r") as stream:
            recorded = np.asarray(
                stream["data/demo_0/states/articulation/robot/joint_position"]
                [branch_step - 1],
                dtype=np.float32,
            )
        joint_position = recorded[:7].copy()
        gripper_position = np.clip(recorded[7:8] / (np.pi / 4), 0, 1).astype(
            np.float32
        )
        proprio_source = "noise-free recorded post-step simulator state"

    request = {
        "observation/image": first_frame(args.asset_video),
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }
    audit = {
        "asset_video": str(args.asset_video),
        "recorded_hdf5": str(args.recorded_hdf5) if args.recorded_hdf5 else None,
        "branch_summary": str(args.branch_summary) if args.branch_summary else None,
        "branch_step": branch_step,
        "prompt": prompt,
        "proprio_source": proprio_source,
    }
    return request, audit


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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite completed scan: {args.output}")

    client = WebsocketClientPolicy(args.host, args.port)
    base, current_audit = current_request(args)
    all_layers = list(range(36))
    recipient_id = f"{args.study_id}-recipient"
    donor_id = f"{args.study_id}-donor"
    cache_id = f"{args.study_id}-self-future-kv"

    def native(seed: int, record_id: str) -> dict:
        return client.infer(
            {
                **base,
                "research_mode": "native",
                "research_seed": seed,
                "research_id": record_id,
            }
        )

    recipient_response = native(args.recipient_seed, recipient_id)
    donor_response = native(args.donor_seed, donor_id)
    recipient = np.asarray(recipient_response["action"])
    donor = np.asarray(donor_response["action"])

    def intervene(
        label: str,
        *,
        future_mode: str,
        attention_mode: str | None = None,
        layers: list[int] | None = None,
        scope: str = "action",
    ) -> tuple[dict, np.ndarray]:
        attention = {}
        if attention_mode is not None:
            attention = {
                "research_attention_mode": attention_mode,
                "research_attention_cache_id": cache_id,
                "research_attention_exclude_layers": layers or [],
                "research_attention_exclude_scope": scope,
            }
        response = client.infer(
            {
                **base,
                "research_mode": future_mode,
                "research_seed": args.recipient_seed,
                "research_id": f"{args.study_id}-{label}",
                "research_recipient_id": recipient_id,
                "research_donor_id": donor_id,
                **attention,
            }
        )
        return response, np.asarray(response["action"])

    self_response, self_action = intervene("self", future_mode="self")
    record_response, record_action = intervene(
        "self-record-kv",
        future_mode="self",
        attention_mode="record",
        layers=all_layers,
    )
    record_action_error = float(np.abs(record_action - self_action).max())
    if record_action_error != 0.0:
        raise RuntimeError(f"recording future K/V changed self action by {record_action_error}")
    cache_counts = record_response["research_attention_interface"]["cache_call_counts"]
    if set(int(value) for value in cache_counts.values()) != {8} or len(cache_counts) != 36:
        raise RuntimeError(f"unexpected attention cache call census: {cache_counts}")

    self_patch_response, self_patch_action = intervene(
        "self-patch-all",
        future_mode="self",
        attention_mode="patch",
        layers=all_layers,
    )
    self_patch_error = float(np.abs(self_patch_action - self_action).max())
    if self_patch_error != 0.0:
        raise RuntimeError(f"self-cache K/V patch changed self action by {self_patch_error}")

    baseline_response, baseline = intervene("donor-baseline", future_mode="donor")
    repeat_response, repeat = intervene("donor-repeat", future_mode="donor")
    repeat_error = float(np.abs(repeat - baseline).max())
    if repeat_error != 0.0:
        raise RuntimeError(f"donor baseline did not recompute exactly: {repeat_error}")
    baseline_projection = projection(baseline, recipient, donor)

    rows = []
    for layer in all_layers:
        response, action = intervene(
            f"donor-patch-layer-{layer}",
            future_mode="donor",
            attention_mode="patch",
            layers=[layer],
        )
        patched_projection = projection(action, recipient, donor)
        rows.append(
            {
                "layer": layer,
                "action_donor_projection": patched_projection,
                "mediation_loss_from_baseline": baseline_projection - patched_projection,
                "action_l2_from_baseline": float(np.linalg.norm(action - baseline)),
                "target_future_max_error": float(
                    response["research_target_future_max_error"]
                ),
            }
        )

    all_direct_response, all_direct = intervene(
        "donor-patch-all-direct",
        future_mode="donor",
        attention_mode="patch",
        layers=all_layers,
    )
    all_barrier_response, all_barrier = intervene(
        "donor-patch-all-barrier",
        future_mode="donor",
        attention_mode="patch",
        layers=all_layers,
        scope="nonfuture",
    )
    all_direct_projection = projection(all_direct, recipient, donor)
    all_barrier_projection = projection(all_barrier, recipient, donor)

    report = {
        "scope": "excluded token-count-preserving future-K/V content mediation scan",
        "study_id": args.study_id,
        **current_audit,
        "recipient_seed": args.recipient_seed,
        "donor_seed": args.donor_seed,
        "native_action_l2": float(np.linalg.norm(donor - recipient)),
        "self_action_donor_projection": projection(self_action, recipient, donor),
        "record_action_maximum_error": record_action_error,
        "self_patch_maximum_action_error": self_patch_error,
        "donor_repeat_maximum_action_error": repeat_error,
        "cache_call_counts": cache_counts,
        "baseline_transplant_action_donor_projection": baseline_projection,
        "baseline_target_future_max_error": float(
            baseline_response["research_target_future_max_error"]
        ),
        "all_direct_patch_action_donor_projection": all_direct_projection,
        "all_direct_patch_mediation_loss": baseline_projection - all_direct_projection,
        "all_direct_target_future_max_error": float(
            all_direct_response["research_target_future_max_error"]
        ),
        "all_barrier_patch_action_donor_projection": all_barrier_projection,
        "all_barrier_patch_mediation_loss": baseline_projection - all_barrier_projection,
        "all_barrier_target_future_max_error": float(
            all_barrier_response["research_target_future_max_error"]
        ),
        "layer_rows": rows,
        "attention_interface": record_response["research_attention_interface"],
        "self_patch_interface": self_patch_response["research_attention_interface"],
        "repeat_target_future_max_error": float(
            repeat_response["research_target_future_max_error"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
