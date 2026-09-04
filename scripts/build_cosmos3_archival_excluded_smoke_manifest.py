#!/usr/bin/env python3
"""Freeze one excluded Bagels archival integration smoke for the exact runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py

from imagined_future.cosmos3_archival import (
    BRANCH_SEEDS,
    atomic_json,
    canonical_json,
    deterministic_shuffled_source,
    deterministic_wrong_donor,
    sha256,
)
from imagined_future.cosmos3_protocol import ordered_recipient_donor_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--package-init", type=Path, required=True)
    parser.add_argument("--archival-module", type=Path, required=True)
    parser.add_argument("--protocol-module", type=Path, required=True)
    parser.add_argument("--server-script", type=Path, required=True)
    parser.add_argument("--interventions-module", type=Path, required=True)
    parser.add_argument("--attention-module", type=Path, required=True)
    parser.add_argument("--upstream-robolab-policy-hash", required=True)
    parser.add_argument("--server-container", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--expected-parameter-probe-hash", required=True)
    parser.add_argument("--checkpoint-content-manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-content-manifest-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-verifier", type=Path, required=True)
    parser.add_argument("--checkpoint-verification-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256(args.checkpoint_content_manifest) != args.expected_checkpoint_content_manifest_sha256:
        raise ValueError("checkpoint content-manifest hash does not match frozen CLI value")
    checkpoint_manifest = json.loads(
        args.checkpoint_content_manifest.read_text(encoding="utf-8")
    )
    if (
        checkpoint_manifest.get("schema_version") != "checkpoint-content-manifest-v1"
        or int(checkpoint_manifest.get("file_count", -1)) != 87
        or int(checkpoint_manifest.get("total_size_bytes", -1)) != 32937437706
    ):
        raise ValueError("checkpoint content-manifest metadata does not match the frozen model")
    checkpoint_verification = json.loads(
        args.checkpoint_verification_receipt.read_text(encoding="utf-8")
    )
    if (
        checkpoint_verification.get("status") != "pass"
        or checkpoint_verification.get("content_manifest_sha256")
        != args.expected_checkpoint_content_manifest_sha256
    ):
        raise ValueError("checkpoint verification receipt is not a pass")
    relative = Path("seed_101/BagelsOnPlateTask")
    episode = args.screen_root / relative
    mp4s = sorted(episode.glob("*.mp4"))
    hdf5s = sorted(episode.glob("*.hdf5"))
    if len(mp4s) != 1 or len(hdf5s) != 1:
        raise RuntimeError("excluded Bagels episode does not have exactly one MP4/HDF5")
    mp4, hdf5 = mp4s[0], hdf5s[0]
    env_cfg = episode / "env_cfg.json"
    config = json.loads(env_cfg.read_text(encoding="utf-8"))
    with h5py.File(hdf5, "r") as stream:
        length = int(stream["data/demo_0/actions"].shape[0])
    candidates = [step for step in range(16, length, 32) if 1 <= step <= length - 1]
    target = 0.50 * (length - 1)
    step = min(candidates, key=lambda value: (abs(value - target), value))
    if step % 32 != 16:
        raise AssertionError("excluded smoke step is not a half-chunk point")
    capture = cv2.VideoCapture(str(mp4))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count != length - 1 or step - 1 >= frame_count:
        raise RuntimeError("excluded smoke recording length invariant failed")
    pairs = ordered_recipient_donor_pairs(BRANCH_SEEDS)
    retrieval = [
        (recipient, source) for recipient in BRANCH_SEEDS for source in BRANCH_SEEDS
    ]
    hashes = {"mp4": sha256(mp4), "hdf5": sha256(hdf5), "env_cfg": sha256(env_cfg)}
    unit_id = f"BagelsOnPlateTask_seed_101_phase_middle_step_{step}"
    body = {
        "schema_version": 1,
        "status": "frozen_before_model_outcomes",
        "study_name": "cosmos3-archival-excluded-development-smoke",
        "admission": "excluded_development_smoke",
        "selection_uses_model_or_intervention_outcomes": False,
        "model_called_during_manifest_build": False,
        "scope": {
            "archival": True,
            "lossy_input_reconstruction": True,
            "action_only": True,
            "physical_endpoint_evidence": False,
            "fresh_simulator_validation": False,
            "excluded_development_smoke": True,
            "admitted_to_evaluation": False,
        },
        "design": {
            "task": "BagelsOnPlateTask",
            "environment_seed": 101,
            "phase": "middle",
            "phase_fraction": 0.5,
            "phase_target_formula": "0.5 * (episode_action_count - 1)",
            "branchpoint_mapping": "nearest 16 mod 32 timestep; ties lower",
            "branch_seeds": list(BRANCH_SEEDS),
            "ordered_pairs": [list(pair) for pair in pairs],
            "future_source_retrieval_cells": [list(cell) for cell in retrieval],
        },
        "runtime": {
            "server_image": (
                "sha256:95a8933deb69f2fc73697d846f7e17066a4d4e05dc9ef85718a7ee12cfea2a8c"
            ),
            "server_container": args.server_container,
            "server_port": args.server_port,
            "runner_sha256": sha256(args.runner),
            "runner_path": str(args.runner.resolve()),
            "research_server_script_sha256": sha256(args.server_script),
            "interventions_module_sha256": sha256(args.interventions_module),
            "attention_module_sha256": sha256(args.attention_module),
            "archival_module_sha256": sha256(args.archival_module),
            "protocol_module_sha256": sha256(args.protocol_module),
            "manifest_builder_sha256": sha256(Path(__file__).resolve()),
            "upstream_robolab_policy_service_sha256": args.upstream_robolab_policy_hash,
            "client_dependency_paths_sha256": {
                str(args.package_init.resolve()): sha256(args.package_init),
                str(args.archival_module.resolve()): sha256(args.archival_module),
                str(args.protocol_module.resolve()): sha256(args.protocol_module),
            },
            "host_mounted_server_dependency_paths_sha256": {
                str(args.package_init.resolve()): sha256(args.package_init),
                str(args.server_script.resolve()): sha256(args.server_script),
                str(args.interventions_module.resolve()): sha256(
                    args.interventions_module
                ),
                str(args.attention_module.resolve()): sha256(args.attention_module),
            },
            "expected_parameter_probe_hash": args.expected_parameter_probe_hash,
            "checkpoint_content_manifest_path": str(
                args.checkpoint_content_manifest.resolve()
            ),
            "checkpoint_content_manifest_sha256": (
                args.expected_checkpoint_content_manifest_sha256
            ),
            "checkpoint_content_manifest_audit_path": str(
                args.checkpoint_content_manifest_audit.resolve()
            ),
            "checkpoint_content_manifest_audit_sha256": sha256(
                args.checkpoint_content_manifest_audit
            ),
            "checkpoint_verifier_path": str(args.checkpoint_verifier.resolve()),
            "checkpoint_verifier_sha256": sha256(args.checkpoint_verifier),
            "checkpoint_verification_receipt_path": str(
                args.checkpoint_verification_receipt.resolve()
            ),
            "checkpoint_verification_receipt_sha256": sha256(
                args.checkpoint_verification_receipt
            ),
            "checkpoint_root": str(args.checkpoint_root.resolve()),
        },
        "states": [
            {
                "unit_id": unit_id,
                "episode_id": "BagelsOnPlateTask_seed_101",
                "task": "BagelsOnPlateTask",
                "environment_seed": 101,
                "instruction": str(config["instruction"]),
                "phase": "middle",
                "phase_fraction": 0.5,
                "continuous_target_step": target,
                "branch_step": step,
                "mp4_frame_index": step - 1,
                "hdf5_state_index": step - 1,
                "branch_seeds": list(BRANCH_SEEDS),
                "ordered_pairs": [list(pair) for pair in pairs],
                "future_source_retrieval_cells": [list(cell) for cell in retrieval],
                "frozen_source_label_permutation": [
                    {
                        "source_seed": source,
                        "shuffled_source_seed": deterministic_shuffled_source(
                            source, BRANCH_SEEDS
                        ),
                    }
                    for source in BRANCH_SEEDS
                ],
                "frozen_wrong_donor_mapping": [
                    {
                        "recipient_seed": recipient,
                        "donor_seed": donor,
                        "wrong_donor_seed": deterministic_wrong_donor(
                            recipient, donor, BRANCH_SEEDS
                        ),
                    }
                    for recipient, donor in pairs
                ],
                "assets": {
                    "relative_episode_directory": str(relative),
                    "mp4_filename": mp4.name,
                    "hdf5_filename": hdf5.name,
                    "env_cfg_filename": env_cfg.name,
                },
                "input_sha256": hashes,
            }
        ],
    }
    identifier = "cosmos3-archival-excluded-smoke-" + hashlib.sha256(
        canonical_json(body)
    ).hexdigest()[:16]
    atomic_json(args.output, {"manifest_id": identifier, **body})
    print(json.dumps({"manifest_id": identifier, "unit_id": unit_id, "step": step}, indent=2))


if __name__ == "__main__":
    main()
