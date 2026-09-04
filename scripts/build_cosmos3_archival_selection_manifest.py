#!/usr/bin/env python3
"""Freeze the archival selection-free Cosmos 3 action-only cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import h5py

from imagined_future.cosmos3_archival import (
    BRANCH_SEEDS,
    ENVIRONMENT_SEEDS,
    PAPER_TASKS,
    PHASES,
    atomic_json,
    canonical_json,
    deterministic_wrong_donor,
    deterministic_shuffled_source,
    phase_branch_steps,
    sha256,
)
from imagined_future.cosmos3_protocol import ordered_recipient_donor_pairs


EXPECTED_EPISODES = len(PAPER_TASKS) * len(ENVIRONMENT_SEEDS)
EXPECTED_STATES = EXPECTED_EPISODES * len(PHASES)
COSMOS_COMMIT = "d4599e2e43fbd06168e9884205b9b66c3902d8f6"
COSMOS_SERVER_IMAGE = (
    "sha256:95a8933deb69f2fc73697d846f7e17066a4d4e05dc9ef85718a7ee12cfea2a8c"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--reconstruction-audit", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--package-init", type=Path, required=True)
    parser.add_argument("--archival-module", type=Path, required=True)
    parser.add_argument("--protocol-module", type=Path, required=True)
    parser.add_argument("--server-script", type=Path, required=True)
    parser.add_argument("--interventions-module", type=Path, required=True)
    parser.add_argument("--attention-module", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--checkpoint-content-manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-content-manifest-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-verifier", type=Path, required=True)
    parser.add_argument("--checkpoint-verification-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-parameter-probe-hash", required=True)
    parser.add_argument("--upstream-robolab-policy-hash", required=True)
    parser.add_argument("--server-container", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--evaluation-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def one_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {pattern} in {directory}, got {matches}")
    return matches[0]


def mp4_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open {path}")
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count <= 0:
        raise RuntimeError(f"invalid MP4 frame count {count}: {path}")
    return count


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
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
        or int(checkpoint_verification.get("file_count", -1)) != 87
        or int(checkpoint_verification.get("total_size_bytes", -1)) != 32937437706
    ):
        raise ValueError("checkpoint verification receipt is not a pass for the frozen model")
    screen_root = args.screen_root.resolve()
    reconstruction_audit = json.loads(
        args.reconstruction_audit.read_text(encoding="utf-8")
    )
    if int(reconstruction_audit["cohort_state_count"]) != 22:
        raise ValueError("reconstruction audit does not cover the canonical 22-state check")
    if int(reconstruction_audit["expected_is_best_count"]) != 22:
        raise ValueError("reconstruction audit did not identify the expected mapping for all states")
    prior_steps = sorted({int(row["branch_step"]) for row in reconstruction_audit["rows"]})
    if not prior_steps or any(step % 32 != 0 for step in prior_steps):
        raise ValueError(f"prior cohort branchpoints are not all 0 mod 32: {prior_steps}")

    runner_hash = sha256(args.runner)
    archival_module_hash = sha256(args.archival_module)
    protocol_module_hash = sha256(args.protocol_module)
    pairs = ordered_recipient_donor_pairs(BRANCH_SEEDS)
    retrieval_cells = [
        (recipient, source) for recipient in BRANCH_SEEDS for source in BRANCH_SEEDS
    ]
    states: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for task in PAPER_TASKS:
        for environment_seed in ENVIRONMENT_SEEDS:
            relative_dir = Path(f"seed_{environment_seed}") / task
            episode_dir = screen_root / relative_dir
            if not episode_dir.is_dir():
                raise FileNotFoundError(episode_dir)
            mp4 = one_file(episode_dir, "*.mp4")
            hdf5 = one_file(episode_dir, "*.hdf5")
            env_cfg = episode_dir / "env_cfg.json"
            if not env_cfg.is_file():
                raise FileNotFoundError(env_cfg)
            config = json.loads(env_cfg.read_text(encoding="utf-8"))
            if int(config["seed"]) != environment_seed:
                raise ValueError(f"environment seed mismatch in {env_cfg}")
            instruction = str(config["instruction"]).strip()
            if not instruction:
                raise ValueError(f"empty instruction in {env_cfg}")
            with h5py.File(hdf5, "r") as stream:
                actions = stream["data/demo_0/actions"]
                joints = stream["data/demo_0/states/articulation/robot/joint_position"]
                action_count = int(actions.shape[0])
                joint_count = int(joints.shape[0])
                archived_success = bool(stream["data/demo_0"].attrs["success"])
            frame_count = mp4_frame_count(mp4)
            if joint_count != action_count:
                raise ValueError(
                    f"joint/action length mismatch in {hdf5}: {joint_count} != {action_count}"
                )
            if frame_count != action_count - 1:
                raise ValueError(
                    f"MP4/action invariant failed in {episode_dir}: "
                    f"{frame_count} != {action_count}-1"
                )
            phase_rows = phase_branch_steps(action_count)
            episode_id = f"{task}_seed_{environment_seed}"
            hashes = {
                "mp4": sha256(mp4),
                "hdf5": sha256(hdf5),
                "env_cfg": sha256(env_cfg),
            }
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "task": task,
                    "environment_seed": environment_seed,
                    "instruction": instruction,
                    "action_count": action_count,
                    "joint_state_count": joint_count,
                    "mp4_frame_count": frame_count,
                    "archived_success_attribute": archived_success,
                    "relative_episode_directory": str(relative_dir),
                    "input_sha256": hashes,
                }
            )
            for phase_row in phase_rows:
                branch_step = int(phase_row["branch_step"])
                if branch_step % 32 != 16 or branch_step in prior_steps:
                    raise AssertionError(f"branchpoint nonoverlap gate failed: {branch_step}")
                phase = str(phase_row["phase"])
                unit_id = f"{episode_id}_phase_{phase}_step_{branch_step}"
                states.append(
                    {
                        "unit_id": unit_id,
                        "episode_id": episode_id,
                        "task": task,
                        "environment_seed": environment_seed,
                        "instruction": instruction,
                        **phase_row,
                        "branch_seeds": list(BRANCH_SEEDS),
                        "ordered_pairs": [list(pair) for pair in pairs],
                        "future_source_retrieval_cells": [
                            list(cell) for cell in retrieval_cells
                        ],
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
                            "relative_episode_directory": str(relative_dir),
                            "mp4_filename": mp4.name,
                            "hdf5_filename": hdf5.name,
                            "env_cfg_filename": env_cfg.name,
                        },
                        "input_sha256": hashes,
                    }
                )

    if len(episode_rows) != EXPECTED_EPISODES or len(states) != EXPECTED_STATES:
        raise RuntimeError(
            f"frozen cohort count failed: {len(episode_rows)} episodes, {len(states)} states"
        )
    if len({state["unit_id"] for state in states}) != EXPECTED_STATES:
        raise RuntimeError("frozen unit IDs are not unique")
    if any(int(state["branch_step"]) % 32 != 16 for state in states):
        raise RuntimeError("a frozen branchpoint overlaps the prior 0 mod 32 design")

    body = {
        "schema_version": 1,
        "status": "frozen_before_model_outcomes",
        "study_name": "cosmos3-archival-selection-free-action-only-v7",
        "admission": "frozen_archival_selection_free_action_level_evaluation",
        "selection_uses_model_or_intervention_outcomes": False,
        "model_called_during_manifest_build": False,
        "scope": {
            "archival": True,
            "lossy_input_reconstruction": True,
            "action_only": True,
            "physical_endpoint_evidence": False,
            "fresh_simulator_validation": False,
            "claim": (
                "selection-free donor-identity steering across fixed archival task, episode, "
                "and phase cells"
            ),
        },
        "source": {
            "dataset": "cosmos3_population_screen_v1",
            "inclusion": (
                "all six paper tasks at every available frozen environment seed "
                "101,103,107,109,113; no within-cell filtering"
            ),
            "episode_count": EXPECTED_EPISODES,
            "state_count": EXPECTED_STATES,
            "reconstruction_audit": str(args.reconstruction_audit),
            "reconstruction_audit_sha256": sha256(args.reconstruction_audit),
            "reconstruction_audit_state_count": 22,
            "reconstruction_expected_mapping_best_count": 22,
            "reconstruction_mean_absolute_error_255": float(
                reconstruction_audit["expected_mean_absolute_error_mean"]
            ),
            "prior_branch_steps": prior_steps,
        },
        "predecessor_failure": {
            "manifest_id": "cosmos3-archival-sf-d2df8d9d1d9f0c19",
            "manifest_sha256": (
                "d8394fb17900d2c9a0d032317fba2cf554aaaf1b3213908782995b4829c03211"
            ),
            "status": "permanently_failed_incomplete_not_admitted",
            "completed_output_count_preserved_unopened": 18,
            "failure_unit_position": 19,
            "disclosed_failure_scalar": 0.03341507911682129,
            "diagnosis": (
                "the previous gate measured final sampler-state drift, not fidelity at the "
                "model-input clamp or returned-velocity overwrite sites"
            ),
            "v7_policy": (
                "rerun all 90 states from a new immutable root; no v6 output is resumed, "
                "admitted, or used for state selection"
            ),
        },
        "design": {
            "tasks": list(PAPER_TASKS),
            "environment_seeds": list(ENVIRONMENT_SEEDS),
            "phases": [
                {"phase": phase, "episode_fraction": quantile}
                for phase, quantile in PHASES
            ],
            "phase_target_formula": "q * (episode_action_count - 1)",
            "branchpoint_mapping": (
                "nearest valid timestep congruent to 16 mod 32 in [1,length-1]; "
                "ties choose lower; assert exactly three distinct steps or stop"
            ),
            "nonoverlap": "new steps are 16 mod 32; existing cohort steps are 0 mod 32",
            "branch_seeds": list(BRANCH_SEEDS),
            "ordered_pair_count_per_state": len(pairs),
            "ordered_pairs": [list(pair) for pair in pairs],
            "future_source_retrieval_cell_count_per_state": len(retrieval_cells),
            "future_source_retrieval_cells": [
                list(cell) for cell in retrieval_cells
            ],
            "recipient_action_noise": "fixed to each ordered pair's recipient branch seed",
            "gaussian_seed": 1223,
        },
        "controls": {
            "native_repeat": "exact action, future, and x0 replay for all four seeds",
            "self": "clean self-future clamp repeated exactly for all four recipients",
            "donor_replay": "every one of 12 donor clamps repeated exactly",
            "wrong_donor": (
                "deterministic nonrecipient/nondonor label from frozen branch-seed order"
            ),
            "shuffle": (
                "balanced cyclic derangement across all four future-source labels; "
                "source-label permutation includes self"
            ),
            "gaussian": (
                "one geometry-matched Gaussian future per ordered pair; norm and distance "
                "relative errors <= 1e-5"
            ),
            "input_fingerprint": "exactly one transformed-input fingerprint within each unit",
            "parameter_probe": (
                "every response must carry the singleton frozen checkpoint parameter-probe hash"
            ),
            "none": (
                "one explicit zero-active-site request per recipient; require exact native "
                "action, future, x0 traces, sigmas, and deterministic trace signature"
            ),
            "coordinates": (
                "zero direct writes to action input and output coordinates in all 48 "
                "interventional requests per state"
            ),
            "intervention_sites": (
                "at all 176 active calls per state, require <=1e-7 model-input future error "
                "against (1-sigma)*target + sigma*recipient_noise and <=1e-7 returned-future-"
                "velocity error against (sampler_future-target)/sigma; require exact requested/"
                "observed calls and sigmas, target source/hash, mask cardinality, and frames"
            ),
            "final_sampler_target_residual": (
                "finite continuous descriptive audit only; report maximum, quantiles, and "
                "count above 0.03; never use it for admission, exclusion, stopping, or evidence"
            ),
            "native_future_distinctness": (
                "report whether all four native future hashes are distinct; do not exclude "
                "or adapt a state when they are not"
            ),
        },
        "primary_chance_rate": 0.25,
        "primary_outcome": "four-way correct future-source action identification",
        "secondary_outcomes": [
            "distance reduction to correct donor action",
            "cosine alignment",
            "orthogonal residual normalized by native donor separation",
            "normalized projection",
            "native donor separation quartile",
            "phase-stratified estimates",
        ],
        "analysis": {
            "independent_sampling_hierarchy": "task -> archived episode -> state",
            "within_state_measurements": (
                "four branches, 12 ordered directions, repeats, and controls"
            ),
            "bootstrap": "hierarchical task -> episode -> state; 10000 draws",
            "bootstrap_seed": 20260903,
            "strata": [
                "task",
                "phase",
                "native donor separation quartile",
                "leave-one-task-out",
            ],
            "permutation": (
                "conditional source-label permutation Monte Carlo among all four sources "
                "within recipient (chance 0.25); 10000 samples seeded 20260903, plus a "
                "frozen cyclic derangement descriptive control"
            ),
            "directional_metric_denominator": (
                "prespecified nondegenerate off-diagonal native action axes; null/degenerate "
                "counts are reported explicitly and zero degenerate axes is a success gate"
            ),
            "native_separation_quartiles": (
                "compute deterministic cohort-global quartile boundaries over all 1080 "
                "off-diagonal native action separations; assign donor arms before averaging "
                "within state inside each quartile; report boundaries and arm/state counts"
            ),
            "evidence_criteria": {
                "four_way_source_retrieval": "hierarchical 95% CI lower bound > 0.25",
                "donor_distance_reduction": "hierarchical 95% CI lower bound > 0",
                "complete_cohort": "90/90 immutable state outputs; no missing or extra states",
                "exact_controls": (
                    "all native/self/donor repeats exact; 4/4 zero-active-site no-op controls; "
                    "48/48 action-coordinate audits zero; 176/176 input-clamp and 176/176 "
                    "returned-velocity site audits <=1e-7 per state; singleton input and "
                    "checkpoint fingerprints"
                ),
                "directional_axes": "zero degenerate off-diagonal native action axes",
            },
            "no_pooling": "donors and ordered directions are averaged within state",
        },
        "runtime": {
            "cosmos_commit": COSMOS_COMMIT,
            "server_image": COSMOS_SERVER_IMAGE,
            "runner_sha256": runner_hash,
            "runner_path": str(args.runner.resolve()),
            "archival_module_sha256": archival_module_hash,
            "protocol_module_sha256": protocol_module_hash,
            "research_server_script_sha256": sha256(args.server_script),
            "interventions_module_sha256": sha256(args.interventions_module),
            "attention_module_sha256": sha256(args.attention_module),
            "launcher_sha256": sha256(args.launcher),
            "launcher_path": str(args.launcher.resolve()),
            "analyzer_sha256": sha256(args.analyzer),
            "analyzer_path": str(args.analyzer.resolve()),
            "manifest_builder_sha256": sha256(Path(__file__).resolve()),
            "expected_parameter_probe_hash": args.expected_parameter_probe_hash,
            "checkpoint_content_manifest_path": str(
                args.checkpoint_content_manifest.resolve()
            ),
            "checkpoint_content_manifest_sha256": (
                args.expected_checkpoint_content_manifest_sha256
            ),
            "checkpoint_content_manifest_file_count": 87,
            "checkpoint_content_manifest_total_size_bytes": 32937437706,
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
            "upstream_robolab_policy_service_sha256": args.upstream_robolab_policy_hash,
            "research_metadata_isolation": (
                "pinned RobolabPolicyService._build_sample reads prompt, image, joint_position, "
                "and gripper_position only; research_* fields are consumed by the wrapper after "
                "sample construction and are absent from the transformed model batch"
            ),
            "client_dependency_paths_sha256": {
                str(args.package_init.resolve()): sha256(args.package_init),
                str(args.archival_module.resolve()): archival_module_hash,
                str(args.protocol_module.resolve()): protocol_module_hash,
            },
            "host_mounted_server_dependency_paths_sha256": {
                str(args.package_init.resolve()): sha256(args.package_init),
                str(args.server_script.resolve()): sha256(args.server_script),
                str(args.interventions_module.resolve()): sha256(
                    args.interventions_module
                ),
                str(args.attention_module.resolve()): sha256(
                    args.attention_module
                ),
            },
            "server_container": args.server_container,
            "server_port": args.server_port,
            "evaluation_output_root": str(args.evaluation_output_root.resolve()),
            "server_registry_limit": 4096,
            "server_seed": 0,
            "expected_denoising_calls": 4,
            "expected_future_latent_frames": list(range(1, 9)),
            "intervention_site_error_tolerance": 1e-7,
            "interventional_requests_per_state": 48,
            "active_interventional_requests_per_state": 44,
            "active_intervention_sites_per_state": 176,
            "server_code_mount": (
                "isolated immutable v7 snapshot is host-mounted from NFS; absolute server and "
                "intervention source paths and hashes are pinned because the image digest does "
                "not transitively pin them"
            ),
        },
        "episodes": episode_rows,
        "states": states,
    }
    manifest_id = "cosmos3-archival-sf-" + hashlib.sha256(canonical_json(body)).hexdigest()[:16]
    manifest = {"manifest_id": manifest_id, **body}
    atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "manifest_id": manifest_id,
                "episode_count": len(episode_rows),
                "state_count": len(states),
                "runner_sha256": runner_hash,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
