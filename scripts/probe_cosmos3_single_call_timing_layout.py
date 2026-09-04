#!/usr/bin/env python3
"""Run an outcome-blind layout probe on one excluded archival state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from build_cosmos3_single_call_timing_manifest import SNAPSHOT_FILES, closure_hash
from imagined_future.cosmos3_archival import atomic_json, sha256
from imagined_future.cosmos3_single_call_timing import BRANCH_SEEDS, RESEARCH_SIGMAS, TASKS
from run_cosmos3_single_call_timing import build_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-excluded-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--expected-parameter-probe-hash", required=True)
    parser.add_argument("--checkpoint-verification-receipt", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_exact_array(
    response: Mapping[str, Any], key: str, expected: np.ndarray
) -> None:
    if key not in response or response[key] is None:
        raise ValueError(f"layout probe response lacks {key}")
    actual = np.asarray(response[key], dtype=expected.dtype)
    if (
        actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
        or not np.array_equal(actual, expected)
    ):
        raise ValueError(f"layout probe {key} differs from the frozen value")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite layout audit: {args.output}")
    if args.port != 8004:
        raise ValueError("timing layout probe must use dedicated port 8004")
    source_hash = sha256(args.source_excluded_manifest)
    if source_hash != args.expected_source_sha256:
        raise ValueError("excluded source-manifest SHA mismatch")
    source = json.loads(args.source_excluded_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "frozen_before_model_outcomes":
        raise ValueError("excluded source is not a pre-outcome freeze")
    if source.get("admission") != "excluded_development_smoke":
        raise ValueError("layout source is not excluded development data")
    if source.get("selection_uses_model_or_intervention_outcomes") is not False:
        raise ValueError("layout source used model outcomes")
    states = list(source.get("states", []))
    if len(states) != 1:
        raise ValueError("layout probe requires exactly one excluded state")
    unit = states[0]
    if str(unit["task"]) in TASKS:
        raise ValueError("layout-probe task overlaps the evaluation tasks")
    if tuple(int(seed) for seed in unit["branch_seeds"]) != BRANCH_SEEDS:
        raise ValueError("layout-probe branch order differs")

    snapshot_root = args.snapshot_root.resolve()
    snapshot_hashes = {}
    for relative in SNAPSHOT_FILES:
        path = snapshot_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        snapshot_hashes[relative] = sha256(path)
    snapshot_closure = closure_hash(snapshot_hashes)
    if sha256(args.checkpoint_verification_receipt) != (
        args.expected_checkpoint_receipt_sha256
    ):
        raise ValueError("checkpoint-verification receipt SHA mismatch")
    checkpoint_receipt = json.loads(
        args.checkpoint_verification_receipt.read_text(encoding="utf-8")
    )
    if (
        checkpoint_receipt.get("status") != "pass"
        or checkpoint_receipt.get("scope")
        != "full_runtime_checkpoint_content_pre_timing_calls"
        or checkpoint_receipt.get("snapshot_closure_sha256") != snapshot_closure
        or int(checkpoint_receipt.get("symlink_count", -1)) != 0
        or int(checkpoint_receipt.get("file_count", -1)) != 87
        or int(checkpoint_receipt.get("total_size_bytes", -1)) != 32937437706
    ):
        raise ValueError("checkpoint verification receipt does not bind this snapshot")
    base, input_audit = build_request(unit, args.screen_root)
    client = WebsocketClientPolicy(args.host, args.port)
    prefix = f"timing-layout-{snapshot_closure[:16]}-{unit['unit_id']}"
    native: dict[int, Mapping[str, Any]] = {}
    for seed in BRANCH_SEEDS[:2]:
        request_id = f"{prefix}-native-{seed}"
        native[seed] = client.infer(
            {
                **base,
                "research_id": request_id,
                "research_mode": "native",
                "research_seed": seed,
            }
        )
        if native[seed].get("research_parameter_probe_hash") != (
            args.expected_parameter_probe_hash
        ):
            raise ValueError("layout native parameter probe differs")
    recipient, donor = BRANCH_SEEDS[:2]
    response = client.infer(
        {
            **base,
            "research_id": f"{prefix}-all-calls-{recipient}-{donor}",
            "research_mode": "donor",
            "research_seed": recipient,
            "research_recipient_id": f"{prefix}-native-{recipient}",
            "research_donor_id": f"{prefix}-native-{donor}",
            "research_timing_steps": [0, 1, 2, 3],
        }
    )
    if response.get("research_parameter_probe_hash") != args.expected_parameter_probe_hash:
        raise ValueError("layout intervention parameter probe differs")
    sigmas = np.asarray(RESEARCH_SIGMAS, dtype=np.float32)
    active = np.arange(4, dtype=np.int64)
    inactive = np.asarray([], dtype=np.int64)
    for key, expected in (
        ("research_sigmas", sigmas),
        ("research_x0_sigmas", sigmas),
        ("research_requested_active_call_indices", active),
        ("research_observed_active_call_indices", active),
        ("research_clamped_call_indices", active),
        ("research_inactive_call_indices", inactive),
        ("research_requested_active_sigmas", sigmas),
        ("research_observed_active_sigmas", sigmas),
        ("research_model_input_future_clamp_errors", np.zeros(4)),
        ("research_returned_future_velocity_overwrite_errors", np.zeros(4)),
        ("research_action_input_errors", np.zeros(4)),
        ("research_action_output_errors", np.zeros(4)),
    ):
        require_exact_array(response, key, expected)
    if int(response.get("research_inactive_wrapper_write_count", -1)) != 0:
        raise ValueError("layout probe observed an inactive wrapper write")
    if float(response.get("research_maximum_action_input_error", np.nan)) != 0.0:
        raise ValueError("layout probe observed an action-input write")
    if float(response.get("research_maximum_action_output_error", np.nan)) != 0.0:
        raise ValueError("layout probe observed an action-output write")
    for key in (
        "research_final_sampler_target_max_abs_error",
        "research_final_sampler_target_l2",
    ):
        value = float(response.get(key, np.nan))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"layout probe has invalid descriptive residual {key}")
    vision_shape = [int(value) for value in response["research_vision_shape"]]
    future_frames = [int(value) for value in response["research_future_frame_indices"]]
    vision_count = int(response["research_vision_coordinate_count"])
    mask_count = int(response["research_future_mask_coordinate_count"])
    mask_hash = str(response["research_future_mask_index_hash"])
    if len(vision_shape) < 3 or any(value <= 0 for value in vision_shape):
        raise ValueError("layout probe vision shape is invalid")
    if future_frames != list(range(1, vision_shape[-3])):
        raise ValueError("layout probe future frames are not frames 1 through final")
    if vision_count != int(np.prod(vision_shape)):
        raise ValueError("layout probe vision-coordinate count differs")
    expected_mask_count = int(
        np.prod(vision_shape[:-3] + [len(future_frames)] + vision_shape[-2:])
    )
    if mask_count != expected_mask_count:
        raise ValueError("layout probe future-mask count differs")
    if len(mask_hash) != 64:
        raise ValueError("layout probe future-mask coordinate hash is invalid")
    if response.get("research_recipient_future_hash") != native[recipient].get(
        "research_future_hash"
    ):
        raise ValueError("layout probe recipient future hash differs")
    if response.get("research_donor_future_hash") != native[donor].get(
        "research_future_hash"
    ):
        raise ValueError("layout probe donor future hash differs")
    if response.get("research_target_hash") != native[donor].get("research_future_hash"):
        raise ValueError("layout probe target hash differs")
    audit = {
        "status": "pass",
        "scope": "excluded_development_layout_probe",
        "source_excluded_manifest": str(args.source_excluded_manifest.resolve()),
        "source_excluded_manifest_sha256": source_hash,
        "unit_id": unit["unit_id"],
        "task": unit["task"],
        "server_port": args.port,
        "model_call_count": 3,
        "scientific_action_or_future_outputs_retained": False,
        "snapshot_root": str(snapshot_root),
        "snapshot_file_sha256": snapshot_hashes,
        "snapshot_closure_sha256": snapshot_closure,
        "expected_parameter_probe_hash": args.expected_parameter_probe_hash,
        "checkpoint_verification_receipt": str(
            args.checkpoint_verification_receipt.resolve()
        ),
        "checkpoint_verification_receipt_sha256": (
            args.expected_checkpoint_receipt_sha256
        ),
        "checkpoint_content_manifest_sha256": checkpoint_receipt[
            "checkpoint_content_manifest_sha256"
        ],
        "research_sigmas": sigmas.tolist(),
        "research_x0_sigmas": sigmas.tolist(),
        "vision_shape": vision_shape,
        "future_frame_indices": future_frames,
        "vision_coordinate_count": vision_count,
        "future_mask_coordinate_count": mask_count,
        "future_mask_index_hash": mask_hash,
        "all_live_site_errors_exactly_zero": True,
        "all_action_coordinate_errors_exactly_zero": True,
        "inactive_wrapper_write_count": 0,
        "final_sampler_target_max_abs_error_finite": True,
        "final_sampler_target_l2_finite": True,
        "input_reconstruction_hashes": input_audit,
    }
    atomic_json(args.output, audit)
    print(
        json.dumps(
            {
                "status": "pass",
                "scope": audit["scope"],
                "snapshot_closure_sha256": snapshot_closure,
                "output": str(args.output),
                "output_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
