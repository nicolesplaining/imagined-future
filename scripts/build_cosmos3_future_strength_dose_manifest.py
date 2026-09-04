#!/usr/bin/env python3
"""Freeze the prospective 30-state Cosmos 3 future-strength dose cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from imagined_future.cosmos3_archival import atomic_json, canonical_json, sha256
from imagined_future.cosmos3_dose_response import (
    ALPHAS,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    EXPECTED_ACTIVE_RESPONSES_PER_STATE,
    EXPECTED_ACTIVE_SITES_PER_STATE,
    EXPECTED_CALLS_PER_STATE,
    FROZEN_BRANCH_SEED_ORDER,
    frozen_request_specs,
)


PROTOCOL_SHA256 = "7b559902b640610d88f54b1953151249118ad6d162aa6389ac3cbcaca01cb0cc"
AMENDMENT_SHA256 = "a02e3a74d0d5b7f4a9d72401c8f869519acf7a4f808067ee2a2180e68775158f"
PARENT_V7_MANIFEST_SHA256 = (
    "8ab294c7a581424cfa02faae1db85941e3f43c1e28100e68e21fa5f533703b9e"
)
COSMOS_COMMIT = "d4599e2e43fbd06168e9884205b9b66c3902d8f6"
COSMOS_SERVER_IMAGE = (
    "sha256:95a8933deb69f2fc73697d846f7e17066a4d4e05dc9ef85718a7ee12cfea2a8c"
)
EXPECTED_TASKS = (
    "BananaInBowlTask",
    "RubiksCubeTask",
    "MustardInLeftBinTask",
    "SpoonInMugTask",
    "MarkerInMugTask",
    "SmartphoneInBinTask",
)
EXPECTED_ENVIRONMENT_SEEDS = (101, 103, 107, 109, 113)
EXPECTED_BRANCH_SEEDS = FROZEN_BRANCH_SEED_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-v7-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-v2-amendment", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-checksum-list", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--server-script", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--smoke-validator", type=Path, required=True)
    parser.add_argument("--package-init", type=Path, required=True)
    parser.add_argument("--archival-module", type=Path, required=True)
    parser.add_argument("--protocol-module", type=Path, required=True)
    parser.add_argument("--dose-module", type=Path, required=True)
    parser.add_argument("--interventions-module", type=Path, required=True)
    parser.add_argument("--attention-module", type=Path, required=True)
    parser.add_argument("--checkpoint-content-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256", required=True
    )
    parser.add_argument("--checkpoint-verifier", type=Path, required=True)
    parser.add_argument("--checkpoint-verification-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-parameter-probe-hash", required=True)
    parser.add_argument("--upstream-robolab-policy-hash", required=True)
    parser.add_argument("--server-container", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--evaluation-output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("pre_smoke", "evaluation_ready"), required=True)
    parser.add_argument("--excluded-smoke-manifest", type=Path)
    parser.add_argument("--excluded-smoke-artifact", type=Path)
    parser.add_argument("--excluded-smoke-controls-report", type=Path)
    parser.add_argument("--server-registry-empty-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _hashed_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256(path) for path in paths.values()}


def _snapshot_files(root: Path, checksum_list: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("snapshot root must be an existing nonsymlink directory")
    resolved_root = root.resolve()
    rows: dict[str, str] = {}
    for raw_line in checksum_list.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in rows:
            raise ValueError(f"invalid or duplicate snapshot row: {line}")
        path = (resolved_root / relative).resolve()
        if not path.is_relative_to(resolved_root) or not path.is_file():
            raise ValueError(f"invalid snapshot entry: {relative}")
        if sha256(path) != digest:
            raise ValueError(f"snapshot hash differs for {relative}")
        rows[relative] = digest
    if not rows:
        raise ValueError("snapshot checksum list is empty")
    descendants = list(resolved_root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise ValueError("snapshot closure contains a symlink")
    actual_files = {
        path.relative_to(resolved_root).as_posix()
        for path in descendants
        if path.is_file()
    }
    if actual_files != set(rows):
        raise ValueError(
            "snapshot checksum list is not the exact file closure: "
            f"unlisted={sorted(actual_files-set(rows))}, "
            f"stale={sorted(set(rows)-actual_files)}"
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
    if sha256(args.parent_v7_manifest) != PARENT_V7_MANIFEST_SHA256:
        raise ValueError("parent archival v7 manifest hash differs from the frozen source")
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("dose-response protocol hash differs from the frozen design")
    if sha256(args.protocol_v2_amendment) != AMENDMENT_SHA256:
        raise ValueError("dose-response v2 amendment hash differs from the frozen design")
    final_artifacts = (
        args.excluded_smoke_manifest,
        args.excluded_smoke_artifact,
        args.excluded_smoke_controls_report,
        args.server_registry_empty_receipt,
    )
    if args.stage == "pre_smoke" and any(item is not None for item in final_artifacts):
        raise ValueError("pre-smoke manifest must not bind post-smoke artifacts")
    if args.stage == "evaluation_ready" and any(item is None for item in final_artifacts):
        raise ValueError("evaluation-ready manifest must bind smoke, validator, and registry receipt")
    if args.stage == "evaluation_ready":
        for path in final_artifacts:
            assert path is not None
            if not path.is_file():
                raise FileNotFoundError(path)
        assert args.excluded_smoke_manifest is not None
        assert args.excluded_smoke_artifact is not None
        assert args.excluded_smoke_controls_report is not None
        assert args.server_registry_empty_receipt is not None
        smoke_manifest = json.loads(
            args.excluded_smoke_manifest.read_text(encoding="utf-8")
        )
        smoke_manifest_sha256 = sha256(args.excluded_smoke_manifest)
        smoke_artifact = json.loads(
            args.excluded_smoke_artifact.read_text(encoding="utf-8")
        )
        smoke_artifact_sha256 = sha256(args.excluded_smoke_artifact)
        controls_report = json.loads(
            args.excluded_smoke_controls_report.read_text(encoding="utf-8")
        )
        if (
            smoke_manifest.get("admission") != "excluded_development_smoke"
            or len(smoke_manifest.get("states", [])) != 1
            or smoke_artifact.get("status") != "complete"
            or smoke_artifact.get("admission") != "excluded_development_smoke"
            or smoke_artifact.get("manifest_id") != smoke_manifest.get("manifest_id")
            or smoke_artifact.get("manifest_sha256") != smoke_manifest_sha256
            or controls_report.get("status") != "pass"
            or controls_report.get("scope")
            != "excluded_development_smoke_controls_only"
            or controls_report.get("manifest_id") != smoke_manifest.get("manifest_id")
            or controls_report.get("manifest_sha256") != smoke_manifest_sha256
            or controls_report.get("smoke_artifact_sha256")
            != smoke_artifact_sha256
            or controls_report.get("request_count") != EXPECTED_CALLS_PER_STATE
            or controls_report.get("shape_valid_action_count")
            != EXPECTED_CALLS_PER_STATE
            or controls_report.get("authorization_audit_sha256")
            != smoke_artifact.get("authorization_audit_sha256")
            or controls_report.get("scientific_outcomes_reported") is not False
        ):
            raise ValueError("excluded-smoke artifact/control binding is not an exact pass")
        registry_receipt = json.loads(
            args.server_registry_empty_receipt.read_text(encoding="utf-8")
        )
        if (
            registry_receipt.get("status") != "pass"
            or registry_receipt.get("scope")
            != "post_smoke_pre_evaluation_empty_registry_evidence"
            or registry_receipt.get("registry_empty_before_evaluation") is not True
            or registry_receipt.get("evaluation_calls_before_receipt") != 0
            or registry_receipt.get("container_name") != args.server_container
            or registry_receipt.get("dedicated_server_port") != args.server_port
            or registry_receipt.get("dedicated_server_image") != COSMOS_SERVER_IMAGE
            or registry_receipt.get("dedicated_server_entrypoint_sha256")
            != sha256(args.server_script)
            or registry_receipt.get("checkpoint_content_manifest_sha256")
            != args.expected_checkpoint_content_manifest_sha256
            or registry_receipt.get("snapshot_checksum_list_sha256")
            != sha256(args.snapshot_checksum_list)
        ):
            raise ValueError("post-smoke empty-registry receipt is not an exact pass")
    if (
        sha256(args.checkpoint_content_manifest)
        != args.expected_checkpoint_content_manifest_sha256
    ):
        raise ValueError("checkpoint content-manifest hash differs")
    checkpoint = json.loads(
        args.checkpoint_content_manifest.read_text(encoding="utf-8")
    )
    if (
        checkpoint.get("schema_version") != "checkpoint-content-manifest-v1"
        or int(checkpoint.get("file_count", -1)) != 87
        or int(checkpoint.get("total_size_bytes", -1)) != 32937437706
    ):
        raise ValueError("checkpoint manifest does not identify the frozen model")
    receipt = json.loads(
        args.checkpoint_verification_receipt.read_text(encoding="utf-8")
    )
    if (
        receipt.get("status") != "pass"
        or receipt.get("content_manifest_sha256")
        != args.expected_checkpoint_content_manifest_sha256
    ):
        raise ValueError("checkpoint verification receipt is not a matching pass")

    parent = json.loads(args.parent_v7_manifest.read_text(encoding="utf-8"))
    if (
        parent.get("manifest_id") != "cosmos3-archival-sf-507feb24297971eb"
        or parent.get("status") != "frozen_before_model_outcomes"
        or len(parent.get("states", [])) != 90
    ):
        raise ValueError("parent archival v7 manifest metadata differs")
    middle = [row for row in parent["states"] if row.get("phase") == "middle"]
    if len(middle) != 30:
        raise ValueError(f"expected exactly 30 middle states, got {len(middle)}")
    expected_cells = {
        (task, seed) for task in EXPECTED_TASKS for seed in EXPECTED_ENVIRONMENT_SEEDS
    }
    actual_cells = {
        (str(row["task"]), int(row["environment_seed"])) for row in middle
    }
    if actual_cells != expected_cells:
        raise ValueError("middle-state task x episode grid is incomplete or altered")

    request_sequence = list(frozen_request_specs(EXPECTED_BRANCH_SEEDS))
    states: list[dict[str, Any]] = []
    for row in middle:
        if tuple(int(seed) for seed in row["branch_seeds"]) != EXPECTED_BRANCH_SEEDS:
            raise ValueError(f"branch seeds differ in {row['unit_id']}")
        states.append(
            {
                "unit_id": row["unit_id"],
                "episode_id": row["episode_id"],
                "task": row["task"],
                "environment_seed": row["environment_seed"],
                "instruction": row["instruction"],
                "phase": row["phase"],
                "phase_fraction": row["phase_fraction"],
                "continuous_target_step": row["continuous_target_step"],
                "branch_step": row["branch_step"],
                "mp4_frame_index": row["mp4_frame_index"],
                "hdf5_state_index": row["hdf5_state_index"],
                "branch_seeds": list(EXPECTED_BRANCH_SEEDS),
                "ordered_pairs": [
                    [spec["recipient_seed"], spec["donor_seed"]]
                    for spec in request_sequence[20:80:5]
                ],
                "alpha_grid": list(ALPHAS),
                "request_sequence": request_sequence,
                "assets": row["assets"],
                "input_sha256": row["input_sha256"],
            }
        )

    source_paths = {
        "package_init": args.package_init,
        "archival": args.archival_module,
        "protocol": args.protocol_module,
        "dose": args.dose_module,
        "server": args.server_script,
        "interventions": args.interventions_module,
        "attention": args.attention_module,
        "smoke_validator": args.smoke_validator,
    }
    snapshot_files = _snapshot_files(args.snapshot_root, args.snapshot_checksum_list)
    body = {
        "schema_version": 1,
        "status": "frozen_before_model_outcomes",
        "study_name": "cosmos3-future-strength-dose-response-v2",
        "admission": "prospective_action_level_future_strength_dose_response",
        "freeze_stage": args.stage,
        "launch_authorization": (
            "powered_evaluation_after_independent_go"
            if args.stage == "evaluation_ready"
            else "excluded_smoke_only_no_powered_evaluation"
        ),
        "selection_uses_model_or_intervention_outcomes": False,
        "model_called_during_manifest_build": False,
        "scope": {
            "archival": True,
            "lossy_input_reconstruction": True,
            "action_only": True,
            "physical_endpoint_evidence": False,
            "natural_mediation": False,
            "policy_native_noisy_trajectory": False,
            "claim": "graded action response under a continuously imposed latent future path",
        },
        "source": {
            "parent_v7_manifest_id": parent["manifest_id"],
            "parent_v7_manifest_path": str(args.parent_v7_manifest.resolve()),
            "parent_v7_manifest_sha256": PARENT_V7_MANIFEST_SHA256,
            "selection": "all and only 30 middle-phase parent-manifest states",
            "task_count": 6,
            "episode_count": 30,
            "state_count": 30,
        },
        "design": {
            "tasks": list(EXPECTED_TASKS),
            "environment_seeds": list(EXPECTED_ENVIRONMENT_SEEDS),
            "phase": "middle",
            "phase_fraction": 0.5,
            "branch_seeds": list(EXPECTED_BRANCH_SEEDS),
            "ordered_pair_count_per_state": 12,
            "alpha_grid": list(ALPHAS),
            "target_formula": "F_A + alpha * (F_B - F_A)",
            "endpoint_target_construction": (
                "alpha 0 is the registered recipient tensor; alpha 1 is the registered "
                "donor tensor; interior alphas use the fixed native-dtype operation order"
            ),
            "future_latent_frames": list(range(1, 9)),
            "current_latent_frame_source": "recipient",
            "active_denoising_calls": [0, 1, 2, 3],
            "recipient_action_noise": "fixed within ordered pair at every alpha",
            "request_count_per_state": EXPECTED_CALLS_PER_STATE,
            "active_response_count_per_state": EXPECTED_ACTIVE_RESPONSES_PER_STATE,
            "active_site_count_per_state": EXPECTED_ACTIVE_SITES_PER_STATE,
            "released_action_shape": [32, 8],
            "released_action_coordinate_count": 256,
            "request_sequence": request_sequence,
        },
        "controls": {
            "native_replay": "four exact action/future/x0/metadata replays",
            "none": "four exact zero-active-site native no-ops",
            "self_replay": "four exact full-call self-clamp replays",
            "midpoint_replay": "twelve exact alpha=0.50 replays",
            "interpolation": "zero formula and nonfuture-coordinate error",
            "endpoint_identity": "alpha=0 recipient future and alpha=1 donor future exact on mask",
            "alpha_zero_routing_invariance": (
                "each alpha-0 response is bit-exact to explicit self clamp and mutually "
                "identical across all three donor labels under the frozen behavior signature"
            ),
            "rng": "recipient initial-state and path-noise hashes fixed across alpha",
            "sites": "zero input-clamp and returned-future-velocity error at 320 active sites",
            "nonwrite": "zero action-coordinate and inactive-wrapper writes",
            "final_sampler_residual": "finite descriptive metric only; never an admission or evidence gate",
        },
        "outcomes": {
            "primary": (
                "equal mean of 12 pairwise five-alpha OLS-with-intercept slopes per state; "
                "beta=sum((alpha-0.5)*distance_reduction)/0.625"
            ),
            "primary_evidence_criterion": "task->episode/state bootstrap 95% CI lower bound > 0",
            "secondary": [
                "alpha=1 minus alpha=0 distance-reduction contrast",
                "four adjacent-alpha distance-reduction contrasts",
                "normalized-projection slope and endpoint contrast",
                "four-way correct-donor identification at every alpha",
                "fraction of ordered pairs with nonincreasing donor distance",
            ],
            "metric_definitions": {
                "distance_reduction": "(d - ||A-N_q||_2) / d",
                "donor_projection": "dot(A-N_r,N_q-N_r) / d^2",
                "cosine_alignment": (
                    "dot(A-N_r,N_q-N_r)/(||A-N_r||_2*d); exactly 0 when "
                    "||A-N_r||_2 <= 1e-12"
                ),
                "orthogonal_residual_normalized": (
                    "||(A-N_r)-donor_projection*(N_q-N_r)||_2/d"
                ),
                "axis_gate": "finite d=||N_q-N_r||_2 strictly greater than 1e-12",
                "retrieval": (
                    "Euclidean distance to all four native actions; exact ties broken in "
                    "frozen seed order 211,223,227,229; report ties and top-two margin"
                ),
                "coordinates": (
                    "flatten all 256 float64 coordinates without truncation, padding, "
                    "rescaling, or coordinate-specific weighting"
                ),
            },
            "wording": {
                "positive_linear_dose_trend_in_donor_directed_action_distance_under_imposed_all_call_future_interpolation": (
                    "primary slope lower confidence bound > 0"
                ),
                "task_weighted_mean_profile_increased_strictly_at_every_adjacent_alpha_step": (
                    "all four adjacent hierarchical lower confidence bounds > 0"
                ),
                "monotonic": "not licensed by this protocol",
            },
        },
        "analysis": {
            "hierarchy": "task -> archived episode/state; arms are repeated measurements",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_generator": "numpy.random.Generator(numpy.random.PCG64(20260903))",
            "bootstrap_draw_table": (
                "one shared table: six tasks with replacement, then five states independently "
                "within each sampled task occurrence; reused for every estimand"
            ),
            "confidence_interval": "2.5th/97.5th percentiles; numpy.quantile method=linear",
            "equal_task_primary": True,
            "sensitivity": ["state-weighted", "per-task", "leave-one-task-out"],
            "complete_cohort_gate": "exactly 30 manifest states; no missing or extra files",
            "no_early_stopping": True,
        },
        "runtime": {
            "cosmos_commit": COSMOS_COMMIT,
            "server_image": COSMOS_SERVER_IMAGE,
            "protocol_path": str(args.protocol.resolve()),
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_v2_amendment_path": str(args.protocol_v2_amendment.resolve()),
            "protocol_v2_amendment_sha256": AMENDMENT_SHA256,
            "runner_path": str(args.runner.resolve()),
            "runner_sha256": sha256(args.runner),
            "server_script_path": str(args.server_script.resolve()),
            "server_script_sha256": sha256(args.server_script),
            "launcher_path": str(args.launcher.resolve()),
            "launcher_sha256": sha256(args.launcher),
            "analyzer_path": str(args.analyzer.resolve()),
            "analyzer_sha256": sha256(args.analyzer),
            "smoke_validator_path": str(args.smoke_validator.resolve()),
            "smoke_validator_sha256": sha256(args.smoke_validator),
            "manifest_builder_sha256": sha256(Path(__file__).resolve()),
            "snapshot_root": str(args.snapshot_root.resolve()),
            "snapshot_checksum_list_path": str(
                args.snapshot_checksum_list.resolve()
            ),
            "snapshot_checksum_list_sha256": sha256(
                args.snapshot_checksum_list
            ),
            "snapshot_file_sha256": snapshot_files,
            "client_dependency_paths_sha256": _hashed_paths(source_paths),
            "host_mounted_server_dependency_paths_sha256": _hashed_paths(
                {
                    key: value
                    for key, value in source_paths.items()
                    if key
                    in {
                        "package_init",
                        "protocol",
                        "dose",
                        "server",
                        "interventions",
                        "attention",
                    }
                }
            ),
            "checkpoint_content_manifest_path": str(
                args.checkpoint_content_manifest.resolve()
            ),
            "checkpoint_content_manifest_sha256": (
                args.expected_checkpoint_content_manifest_sha256
            ),
            "checkpoint_content_manifest_file_count": 87,
            "checkpoint_content_manifest_total_size_bytes": 32937437706,
            "checkpoint_verifier_path": str(args.checkpoint_verifier.resolve()),
            "checkpoint_verifier_sha256": sha256(args.checkpoint_verifier),
            "checkpoint_verification_receipt_path": str(
                args.checkpoint_verification_receipt.resolve()
            ),
            "checkpoint_verification_receipt_sha256": sha256(
                args.checkpoint_verification_receipt
            ),
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "expected_parameter_probe_hash": args.expected_parameter_probe_hash,
            "upstream_robolab_policy_service_sha256": args.upstream_robolab_policy_hash,
            "server_container": args.server_container,
            "server_port": args.server_port,
            "server_registry_limit": 4096,
            "server_seed": 0,
            "evaluation_output_root": str(args.evaluation_output_root.resolve()),
            "intervention_site_error_tolerance": 1e-7,
            "excluded_smoke_artifact": (
                {
                    "path": str(args.excluded_smoke_artifact.resolve()),
                    "sha256": sha256(args.excluded_smoke_artifact),
                }
                if args.excluded_smoke_artifact is not None
                else None
            ),
            "excluded_smoke_manifest": (
                {
                    "path": str(args.excluded_smoke_manifest.resolve()),
                    "sha256": sha256(args.excluded_smoke_manifest),
                }
                if args.excluded_smoke_manifest is not None
                else None
            ),
            "excluded_smoke_controls_report": (
                {
                    "path": str(args.excluded_smoke_controls_report.resolve()),
                    "sha256": sha256(args.excluded_smoke_controls_report),
                }
                if args.excluded_smoke_controls_report is not None
                else None
            ),
            "server_registry_empty_receipt": (
                {
                    "path": str(args.server_registry_empty_receipt.resolve()),
                    "sha256": sha256(args.server_registry_empty_receipt),
                }
                if args.server_registry_empty_receipt is not None
                else None
            ),
        },
        "states": states,
    }
    manifest_id = "cosmos3-dose-" + hashlib.sha256(canonical_json(body)).hexdigest()[:16]
    manifest = {"manifest_id": manifest_id, **body}
    atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "manifest_id": manifest_id,
                "state_count": len(states),
                "calls_per_state": EXPECTED_CALLS_PER_STATE,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
