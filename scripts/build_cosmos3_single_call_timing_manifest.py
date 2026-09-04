#!/usr/bin/env python3
"""Freeze the prospective 30-state Cosmos 3 single-call timing manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from imagined_future.cosmos3_archival import atomic_json, canonical_json, sha256
from imagined_future.cosmos3_single_call_timing import (
    ACTION_COORDINATE_COUNT,
    ACTION_SHAPE,
    BRANCH_SEEDS,
    ENVIRONMENT_SEEDS,
    EXPECTED_REQUEST_COUNT,
    EXPECTED_STATE_COUNT,
    RESEARCH_SIGMAS,
    REQUESTS_PER_STATE,
    TASKS,
    TIMING_CONDITIONS,
    expected_request_labels,
    ordered_off_diagonal_pairs,
    ordered_source_cells,
)


SNAPSHOT_FILES = (
    "docs/overnight_2026-09-03/cosmos3_single_call_timing_protocol_v2.md",
    "docs/overnight_2026-09-03/cosmos3_single_call_timing_outcome_blind_audit_checklist.md",
    "docs/overnight_2026-09-03/cosmos3_single_call_timing_protocol_v2_finiteness_amendment.md",
    "docs/overnight_2026-09-03/cosmos3_single_call_timing_protocol_v2_action_shape_amendment.md",
    "scripts/build_cosmos3_single_call_timing_manifest.py",
    "scripts/build_cosmos3_single_call_timing_smoke_manifest.py",
    "scripts/hash_checkpoint_content_manifest.py",
    "scripts/launch_cosmos3_single_call_timing.py",
    "scripts/probe_cosmos3_single_call_timing_layout.py",
    "scripts/run_cosmos3_single_call_timing.py",
    "scripts/run_cosmos3_single_call_timing_server.py",
    "scripts/summarize_cosmos3_single_call_timing.py",
    "scripts/verify_cosmos3_checkpoint_content.py",
    "src/imagined_future/cosmos3_archival.py",
    "src/imagined_future/cosmos3_attention.py",
    "src/imagined_future/cosmos3_interventions.py",
    "src/imagined_future/cosmos3_protocol.py",
    "src/imagined_future/cosmos3_single_call_timing.py",
    "src/imagined_future/__init__.py",
    "tests/test_cosmos3_single_call_timing.py",
    "tests/test_cosmos3_interventions.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archival-manifest", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--layout-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--checkpoint-content-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-content-manifest-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-verification-root", type=Path, required=True)
    parser.add_argument("--server-container", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def closure_hash(files: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json(files)).hexdigest()


def require_exact_float32(value: Any, expected: tuple[np.float32, ...], label: str) -> None:
    actual = np.asarray(value, dtype=np.float32)
    target = np.asarray(expected, dtype=np.float32)
    if actual.shape != target.shape or not np.array_equal(actual, target):
        raise ValueError(f"{label} differs: {actual.tolist()} != {target.tolist()}")


def validate_checkpoint_provenance(
    provenance_path: Path,
    content_manifest_path: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = json.loads(provenance_path.read_text(encoding="utf-8"))
    if checkpoint.get("status") != "verified_pre_outcome":
        raise ValueError("checkpoint provenance is not verified pre-outcome")
    if checkpoint.get("checkpoint_identity_kind") != (
        "sha256_of_canonical_full_file_content_manifest"
    ):
        raise ValueError("checkpoint identity is not a full file-content manifest")
    checkpoint_identity = str(checkpoint.get("checkpoint_identity_sha256", ""))
    if len(checkpoint_identity) != 64:
        raise ValueError("checkpoint provenance lacks a SHA-256 identity")
    if not content_manifest_path.is_file():
        raise FileNotFoundError(content_manifest_path)
    if sha256(content_manifest_path) != checkpoint_identity:
        raise ValueError("checkpoint content-manifest bytes differ from frozen identity")
    content = json.loads(content_manifest_path.read_text(encoding="utf-8"))
    if content.get("schema_version") != "checkpoint-content-manifest-v1":
        raise ValueError("unsupported checkpoint content-manifest schema")
    expected_bytes = canonical_json(content) + b"\n"
    if content_manifest_path.read_bytes() != expected_bytes:
        raise ValueError("checkpoint content manifest is not canonical JSON bytes")
    entries = list(content.get("files", []))
    paths = [str(entry.get("relative_path", "")) for entry in entries]
    if not entries or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("checkpoint content-manifest paths are not sorted and unique")
    for entry in entries:
        if (
            not entry["relative_path"]
            or Path(entry["relative_path"]).is_absolute()
            or ".." in Path(entry["relative_path"]).parts
            or int(entry["size_bytes"]) < 0
            or len(str(entry["sha256"])) != 64
        ):
            raise ValueError(f"invalid checkpoint file entry: {entry}")
    if int(content.get("file_count", -1)) != len(entries):
        raise ValueError("checkpoint content-manifest file count differs")
    if int(content.get("total_size_bytes", -1)) != sum(
        int(entry["size_bytes"]) for entry in entries
    ):
        raise ValueError("checkpoint content-manifest byte count differs")
    if Path(str(content.get("checkpoint_root", ""))).resolve() != Path(
        str(checkpoint.get("checkpoint_root", ""))
    ).resolve():
        raise ValueError("checkpoint roots differ across provenance artifacts")
    if str(checkpoint.get("checkpoint_content_manifest_sha256", "")) != checkpoint_identity:
        raise ValueError("provenance content-manifest SHA differs from identity")
    audit_hash = str(checkpoint.get("checkpoint_content_manifest_audit_sha256", ""))
    if not audit_path.is_file() or len(audit_hash) != 64 or sha256(audit_path) != audit_hash:
        raise ValueError("checkpoint content-manifest audit artifact differs")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("content_manifest_sha256") != checkpoint_identity
        or int(audit.get("file_count", -1)) != len(entries)
        or int(audit.get("total_size_bytes", -1)) != int(content["total_size_bytes"])
        or not np.isfinite(float(audit.get("hashing_elapsed_seconds", np.nan)))
        or float(audit["hashing_elapsed_seconds"]) <= 0.0
    ):
        raise ValueError("checkpoint content-manifest audit contents differ")
    return checkpoint, content


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
    source_hash = sha256(args.source_archival_manifest)
    source = json.loads(args.source_archival_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "frozen_before_model_outcomes":
        raise ValueError("source archival manifest is not a pre-outcome freeze")
    if source.get("selection_uses_model_or_intervention_outcomes") is not False:
        raise ValueError("source archival population is not selection-free")
    source_states = list(source.get("states", []))
    middle = [state for state in source_states if state.get("phase") == "middle"]
    expected_order = [
        (task, environment_seed) for task in TASKS for environment_seed in ENVIRONMENT_SEEDS
    ]
    observed_order = [
        (str(state["task"]), int(state["environment_seed"])) for state in middle
    ]
    if observed_order != expected_order or len(middle) != EXPECTED_STATE_COUNT:
        raise ValueError(
            f"source middle cohort differs from frozen 6x5 order: {observed_order}"
        )
    for state in middle:
        if tuple(int(seed) for seed in state["branch_seeds"]) != BRANCH_SEEDS:
            raise ValueError(f"branch order mismatch in {state['unit_id']}")
        if not all(key in state for key in (
            "unit_id", "episode_id", "branch_step", "mp4_frame_index",
            "hdf5_state_index", "assets", "input_sha256", "instruction",
        )):
            raise ValueError(f"source state lacks required archival fields: {state['unit_id']}")

    snapshot_root = args.snapshot_root.resolve()
    snapshot_hashes: dict[str, str] = {}
    for relative in SNAPSHOT_FILES:
        path = snapshot_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        snapshot_hashes[relative] = sha256(path)
    snapshot_closure = closure_hash(snapshot_hashes)
    protocol_relative = (
        "docs/overnight_2026-09-03/cosmos3_single_call_timing_protocol_v2.md"
    )
    checklist_relative = (
        "docs/overnight_2026-09-03/"
        "cosmos3_single_call_timing_outcome_blind_audit_checklist.md"
    )
    amendment_relative = (
        "docs/overnight_2026-09-03/"
        "cosmos3_single_call_timing_protocol_v2_finiteness_amendment.md"
    )
    action_shape_amendment_relative = (
        "docs/overnight_2026-09-03/"
        "cosmos3_single_call_timing_protocol_v2_action_shape_amendment.md"
    )
    protocol_hash = snapshot_hashes[protocol_relative]
    checklist_hash = snapshot_hashes[checklist_relative]
    amendment_hash = snapshot_hashes[amendment_relative]
    action_shape_amendment_hash = snapshot_hashes[action_shape_amendment_relative]
    if protocol_hash != "526d126a1ffe6cc8216f8be1a0aaa93732faf637932f8155eea3d024fcb38c57":
        raise ValueError("snapshot does not contain the reviewed timing protocol v2")
    if checklist_hash != "288472c4f8ea914333916b6bff68c9777b4d07eb892646e70b0ebfa3749d7012":
        raise ValueError("snapshot does not contain the reviewed outcome-blind checklist")
    if amendment_hash != "23a5c922fb1abc7c3feaf409b764308060266ef171458bd8ba58fafaacab3f83":
        raise ValueError("snapshot does not contain the frozen finiteness amendment")
    if action_shape_amendment_hash != (
        "70767fda042b3ba8dab888ea5c4325f34aa20d6a9e494b68dc47cbacf653e88f"
    ):
        raise ValueError("snapshot does not contain the frozen action-shape amendment")

    layout = json.loads(args.layout_audit.read_text(encoding="utf-8"))
    if layout.get("status") != "pass":
        raise ValueError("excluded layout audit did not pass")
    if layout.get("scope") != "excluded_development_layout_probe":
        raise ValueError("layout audit is not explicitly excluded from evaluation")
    if layout.get("snapshot_closure_sha256") != snapshot_closure:
        raise ValueError("layout audit was not run against this exact snapshot")
    require_exact_float32(layout.get("research_sigmas"), RESEARCH_SIGMAS, "layout sigmas")
    require_exact_float32(layout.get("research_x0_sigmas"), RESEARCH_SIGMAS, "layout x0 sigmas")
    future_frames = [int(value) for value in layout["future_frame_indices"]]
    vision_shape = [int(value) for value in layout["vision_shape"]]
    vision_count = int(layout["vision_coordinate_count"])
    mask_count = int(layout["future_mask_coordinate_count"])
    mask_hash = str(layout["future_mask_index_hash"])
    if future_frames != list(range(1, vision_shape[-3])):
        raise ValueError("layout future frames are not exactly 1 through final latent frame")
    if vision_count != int(np.prod(vision_shape)):
        raise ValueError("layout vision coordinate count differs from vision shape")
    expected_mask_count = int(np.prod(vision_shape[:-3] + [len(future_frames)] + vision_shape[-2:]))
    if mask_count != expected_mask_count:
        raise ValueError("layout future-mask count differs from frozen frames/shape")
    if len(mask_hash) != 64:
        raise ValueError("layout future-mask index hash is not SHA-256")

    checkpoint, checkpoint_content = validate_checkpoint_provenance(
        args.checkpoint_provenance,
        args.checkpoint_content_manifest,
        args.checkpoint_content_manifest_audit,
    )
    checkpoint_identity = str(checkpoint["checkpoint_identity_sha256"])
    if layout.get("checkpoint_content_manifest_sha256") != checkpoint_identity:
        raise ValueError("layout probe used a different full checkpoint manifest")
    checkpoint_receipt_path = Path(
        str(layout.get("checkpoint_verification_receipt", ""))
    )
    checkpoint_receipt_hash = str(
        layout.get("checkpoint_verification_receipt_sha256", "")
    )
    if (
        not checkpoint_receipt_path.is_file()
        or len(checkpoint_receipt_hash) != 64
        or sha256(checkpoint_receipt_path) != checkpoint_receipt_hash
    ):
        raise ValueError("layout probe checkpoint receipt differs")
    checkpoint_receipt = json.loads(
        checkpoint_receipt_path.read_text(encoding="utf-8")
    )
    if (
        checkpoint_receipt.get("status") != "pass"
        or checkpoint_receipt.get("snapshot_closure_sha256") != snapshot_closure
        or checkpoint_receipt.get("checkpoint_content_manifest_sha256")
        != checkpoint_identity
    ):
        raise ValueError("checkpoint receipt does not bind snapshot and checkpoint")
    checkpoint_verification_root = args.checkpoint_verification_root.resolve()
    if not checkpoint_verification_root.is_dir():
        raise FileNotFoundError(checkpoint_verification_root)
    source_runtime = source["runtime"]
    source_dependency_expectations = {
        "src/imagined_future/__init__.py": next(
            value
            for path, value in source_runtime[
                "client_dependency_paths_sha256"
            ].items()
            if path.endswith("/src/imagined_future/__init__.py")
        ),
        "src/imagined_future/cosmos3_archival.py": source_runtime[
            "archival_module_sha256"
        ],
        "src/imagined_future/cosmos3_attention.py": source_runtime[
            "attention_module_sha256"
        ],
        "src/imagined_future/cosmos3_interventions.py": source_runtime[
            "interventions_module_sha256"
        ],
        "src/imagined_future/cosmos3_protocol.py": source_runtime[
            "protocol_module_sha256"
        ],
    }
    for relative, expected_hash in source_dependency_expectations.items():
        if snapshot_hashes.get(relative) != expected_hash:
            raise ValueError(
                f"timing snapshot changed v7 causal dependency {relative}: "
                f"{snapshot_hashes.get(relative)} != {expected_hash}"
            )
    if source_runtime["research_server_script_sha256"] != (
        "64980c631d4bec71be3e41cb574c0b84c759e80ebbaf1e58eeb558f66be17073"
    ):
        raise ValueError("source v7 server provenance differs from reviewed base")
    expected_probe = str(source_runtime["expected_parameter_probe_hash"])
    if len(expected_probe) != 64:
        raise ValueError("source parameter-probe hash is invalid")
    if args.server_port != 8004:
        raise ValueError("reviewed timing topology reserves dedicated port 8004")

    labels = list(expected_request_labels(BRANCH_SEEDS))
    body: dict[str, Any] = {
        "schema_version": 5,
        "status": "frozen_before_model_outcomes",
        "study_name": "cosmos3-single-call-timing-v5",
        "admission": "frozen_single_call_timing_evaluation",
        "launch_authorization": "independent_outcome_blind_go_required",
        "selection_uses_model_or_intervention_outcomes": False,
        "model_called_during_manifest_build": False,
        "scope": {
            "archival": True,
            "lossy_input_reconstruction": True,
            "action_only": True,
            "imposed_intervention_timing_strength": True,
            "natural_mediation": False,
            "physical_endpoint_evidence": False,
            "semantic_planning": False,
        },
        "source": {
            "protocol_relative_path": protocol_relative,
            "protocol_sha256": protocol_hash,
            "outcome_blind_checklist_relative_path": checklist_relative,
            "outcome_blind_checklist_sha256": checklist_hash,
            "finiteness_amendment_relative_path": amendment_relative,
            "finiteness_amendment_sha256": amendment_hash,
            "action_shape_amendment_relative_path": action_shape_amendment_relative,
            "action_shape_amendment_sha256": action_shape_amendment_hash,
            "supersedes_failed_timing_manifest": {
                "manifest_id": "cosmos3-timing-1a1e2733084791f0",
                "manifest_sha256": (
                    "aeaebdf3e8ebb2acd81dfae3d083ebe366045fa62fd0dca943bb7d056b95fa77"
                ),
                "smoke_manifest_id": (
                    "cosmos3-timing-excluded-smoke-8247bc17a53b432a"
                ),
                "smoke_manifest_sha256": (
                    "86a9f91f1939776264ac4166ebe12d1a400c2ddc58215d88ffa1167cc34d7318"
                ),
                "snapshot_checksum_list_sha256": (
                    "87888ce836c87a515df92943e0a588aef60bd2efb8619d170cfec7dbcab495f2"
                ),
                "failure_stage": "before_atomic_state_output",
                "admission": "permanently_failed_and_non_admitted",
            },
            "supersedes_pre_smoke_engineering_freeze": {
                "root": (
                    "/lambda/nfs/imagined-future/results/overnight_2026_09_03/"
                    "cosmos3_single_call_timing_v3"
                ),
                "snapshot_checksum_list_sha256": (
                    "9c64737ba1d0e40865301ff69bac9807178e302e606cc7223d00d40a7faaff10"
                ),
                "reason": "ambiguous_unexecuted_generic_server_present_in_snapshot",
                "model_or_layout_call_count": 0,
                "admission": "permanently_superseded_and_non_admitted",
            },
            "supersedes_timing_v4_no_go": {
                "evaluation_manifest_id": "cosmos3-timing-89823a363d824c4a",
                "evaluation_manifest_sha256": (
                    "5c4e4d50c8d0132b0ac8d57fe509784b6badc28441c004f0a10311f61deef073"
                ),
                "smoke_manifest_id": (
                    "cosmos3-timing-excluded-smoke-ac67e566390c2204"
                ),
                "smoke_manifest_sha256": (
                    "5c924e46fd403fcb32f71367531dfc3f63436099ada6d927e7462ca117c5e7f6"
                ),
                "smoke_artifact_sha256": (
                    "17ce67034159bb2d98892e64ece8f3e88e5a6785e9652cf4dea4c07df75e4e66"
                ),
                "snapshot_checksum_list_sha256": (
                    "d764bf5b949cfc56f33d16f55db256c8fd63271400b20111eabf6e0274514513"
                ),
                "sole_launch_blocker": "32x7_protocol_text_vs_32x8_policy_action",
                "admitted_evaluation_call_count": 0,
                "admission": "permanently_smoke_only_and_non_admitted",
            },
            "archival_manifest": str(args.source_archival_manifest.resolve()),
            "archival_manifest_id": source["manifest_id"],
            "archival_manifest_sha256": source_hash,
            "state_copy_rule": "all and only 30 middle-phase states in source manifest order",
            "layout_audit": str(args.layout_audit.resolve()),
            "layout_audit_sha256": sha256(args.layout_audit),
            "base_v7_research_server_sha256": source_runtime[
                "research_server_script_sha256"
            ],
            "checkpoint_provenance": str(args.checkpoint_provenance.resolve()),
            "checkpoint_provenance_sha256": sha256(args.checkpoint_provenance),
            "checkpoint_identity_sha256": checkpoint_identity,
            "checkpoint_identity_kind": checkpoint["checkpoint_identity_kind"],
            "checkpoint_root": checkpoint["checkpoint_root"],
            "checkpoint_verification_root": str(checkpoint_verification_root),
            "checkpoint_content_manifest": str(
                args.checkpoint_content_manifest.resolve()
            ),
            "checkpoint_content_manifest_sha256": checkpoint_identity,
            "checkpoint_content_manifest_file_count": checkpoint_content[
                "file_count"
            ],
            "checkpoint_content_manifest_total_size_bytes": checkpoint_content[
                "total_size_bytes"
            ],
            "checkpoint_content_manifest_audit": str(
                args.checkpoint_content_manifest_audit.resolve()
            ),
            "checkpoint_content_manifest_audit_sha256": sha256(
                args.checkpoint_content_manifest_audit
            ),
            "checkpoint_runtime_verification_receipt": str(
                checkpoint_receipt_path.resolve()
            ),
            "checkpoint_runtime_verification_receipt_sha256": (
                checkpoint_receipt_hash
            ),
        },
        "design": {
            "tasks": list(TASKS),
            "environment_seeds": list(ENVIRONMENT_SEEDS),
            "phase": "middle",
            "branch_seeds": list(BRANCH_SEEDS),
            "timing_conditions": [
                {"name": name, "active_call_indices": list(indices)}
                for name, indices in TIMING_CONDITIONS
            ],
            "research_sigmas": [float(value) for value in RESEARCH_SIGMAS],
            "research_x0_sigmas": [float(value) for value in RESEARCH_SIGMAS],
            "future_source_cells": [list(cell) for cell in ordered_source_cells()],
            "ordered_off_diagonal_pairs": [
                list(pair) for pair in ordered_off_diagonal_pairs()
            ],
            "request_labels": labels,
            "action_shape": list(ACTION_SHAPE),
            "action_coordinate_count": ACTION_COORDINATE_COUNT,
            "action_coordinate_semantics": (
                "seven joint coordinates plus gripper per timestep; all coordinates "
                "enter every frozen primary action estimand"
            ),
            "shape_valid_response_actions_per_state": REQUESTS_PER_STATE,
            "shape_valid_stored_actions_per_state": 100,
            "requests_per_state": REQUESTS_PER_STATE,
            "state_count": EXPECTED_STATE_COUNT,
            "total_request_count": EXPECTED_REQUEST_COUNT,
            "request_breakdown": {
                "native": 4,
                "native_replay": 4,
                "six_complete_4x4_timing_grids": 96,
                "all_calls_diagonal_replay": 4,
            },
            "vision_shape": vision_shape,
            "future_frame_indices": future_frames,
            "vision_coordinate_count": vision_count,
            "future_mask_coordinate_count": mask_count,
            "future_mask_index_hash": mask_hash,
            "recipient_rng": "recipient branch seed fixes initial state and path noise",
            "source_rng": "source seed identifies only a registered native future target",
        },
        "controls": {
            "native_replay": "one exact replay per native branch",
            "none": "all 16 source-labeled cells are full no-ops",
            "all_calls_diagonal_replay": "one extra exact replay per diagonal",
            "active_data_path": "live model-input clamp and returned-velocity errors exactly zero",
            "mask": "frames, shape, cardinality, and selected-coordinate hash exact",
            "coordinates": "per-call action input/output errors exactly zero",
            "rng": "initial-state and path-noise hashes exact within recipient",
            "target": "target/recipient/donor hashes match named native records",
            "final_residual": (
                "max-absolute and L2 distance from final sampled future to target are "
                "descriptive only and never gate admission"
            ),
            "final_residual_protocol_term_mapping": {
                "max_absolute": "final_sampler_target_max_abs_error",
                "euclidean": "final_sampler_target_l2",
                "role": "descriptive_only_not_an_admission_gate",
            },
            "projection_applicability": {
                "per_state_structural_null_diagonal_interventions": 28,
                "per_state_finite_off_diagonal_interventions": 72,
                "per_state_native_field_absent": 8,
                "off_diagonal_none_projection": "finite_exactly_zero",
                "only_other_allowed_null_path": (
                    "research_attention_interface.cache_id when inactive"
                ),
                "all_other_numeric_fields": "required_and_finite",
            },
            "action_schema": {
                "shape": list(ACTION_SHAPE),
                "coordinate_count": ACTION_COORDINATE_COUNT,
                "wire_responses_per_state": 108,
                "stored_native_actions_per_state": 4,
                "stored_timing_grid_actions_per_state": 96,
                "shape_failures_or_exclusions_allowed": 0,
            },
        },
        "analysis": {
            "independent_unit": "saved state",
            "task_weighting": "five states per task then six equal-weight tasks",
            "within_state_measurements": "six timings and all 16 recipient/source cells",
            "bootstrap": "shared task-to-state hierarchical resample, 10000 draws",
            "bootstrap_rng": "numpy Generator(PCG64(20260903))",
            "primary": (
                "average of four timing-matched single-call retrieval and distance gains; "
                "both hierarchical lower bounds must exceed zero"
            ),
            "sustained": (
                "all_calls minus mean(single calls), separately for matched retrieval and "
                "distance gain; both lower bounds must exceed zero"
            ),
            "call_local": (
                "one-sided null-centered component p-values; per-call conjunction is max; "
                "Holm across four calls"
            ),
            "source_retrieval_chance": 0.25,
            "native_separation_quartiles": (
                "global boundaries from all 360 directed pairs before state aggregation; "
                "numpy searchsorted side=right retains boundary ties deterministically"
            ),
            "null_boundary": (
                "failed positive gate means not detected under this design; no equivalence, "
                "necessity, or no-effect conclusion"
            ),
        },
        "runtime": {
            "cosmos_commit": source_runtime["cosmos_commit"],
            "server_image": source_runtime["server_image"],
            "server_container": args.server_container,
            "server_port": args.server_port,
            "server_gpu": 3,
            "server_topology": (
                "dedicated isolated uninstrumented timing server; no arm mixing with archival v7"
            ),
            "timing_server_relative_path": (
                "scripts/run_cosmos3_single_call_timing_server.py"
            ),
            "timing_server_sha256": snapshot_hashes[
                "scripts/run_cosmos3_single_call_timing_server.py"
            ],
            "timing_server_delta": (
                "v7 server with only structural donor-projection JSON null/applicability "
                "serialization added for the frozen finiteness amendment"
            ),
            "server_registry_initial_size_required": 0,
            "server_registry_limit": source_runtime.get("server_registry_limit", 4096),
            "expected_parameter_probe_hash": expected_probe,
            "upstream_robolab_policy_service_sha256": source_runtime[
                "upstream_robolab_policy_service_sha256"
            ],
            "source_host_mounted_server_dependency_paths_sha256": source_runtime[
                "host_mounted_server_dependency_paths_sha256"
            ],
            "source_client_dependency_paths_sha256": source_runtime[
                "client_dependency_paths_sha256"
            ],
            "checkpoint_identity_sha256": checkpoint_identity,
            "checkpoint_identity_kind": checkpoint["checkpoint_identity_kind"],
            "checkpoint_root": checkpoint["checkpoint_root"],
            "checkpoint_verification_root": str(checkpoint_verification_root),
            "checkpoint_content_manifest": str(
                args.checkpoint_content_manifest.resolve()
            ),
            "checkpoint_content_manifest_sha256": checkpoint_identity,
            "checkpoint_content_manifest_file_count": checkpoint_content[
                "file_count"
            ],
            "checkpoint_content_manifest_total_size_bytes": checkpoint_content[
                "total_size_bytes"
            ],
            "checkpoint_runtime_verification_receipt": str(
                checkpoint_receipt_path.resolve()
            ),
            "checkpoint_runtime_verification_receipt_sha256": (
                checkpoint_receipt_hash
            ),
            "snapshot_root": str(snapshot_root),
            "snapshot_file_sha256": snapshot_hashes,
            "snapshot_closure_sha256": snapshot_closure,
            "runner_relative_path": "scripts/run_cosmos3_single_call_timing.py",
            "runner_sha256": snapshot_hashes[
                "scripts/run_cosmos3_single_call_timing.py"
            ],
            "launcher_relative_path": "scripts/launch_cosmos3_single_call_timing.py",
            "launcher_sha256": snapshot_hashes[
                "scripts/launch_cosmos3_single_call_timing.py"
            ],
            "analyzer_relative_path": "scripts/summarize_cosmos3_single_call_timing.py",
            "analyzer_sha256": snapshot_hashes[
                "scripts/summarize_cosmos3_single_call_timing.py"
            ],
            "timing_module_sha256": snapshot_hashes[
                "src/imagined_future/cosmos3_single_call_timing.py"
            ],
            "research_metadata_isolation": source_runtime[
                "research_metadata_isolation"
            ],
        },
        "states": middle,
    }
    manifest_id = "cosmos3-timing-" + hashlib.sha256(canonical_json(body)).hexdigest()[:16]
    manifest = {"manifest_id": manifest_id, **body}
    atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "status": "frozen",
                "manifest_id": manifest_id,
                "manifest_sha256": sha256(args.output),
                "state_count": len(middle),
                "request_count": EXPECTED_REQUEST_COUNT,
                "snapshot_closure_sha256": snapshot_closure,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
