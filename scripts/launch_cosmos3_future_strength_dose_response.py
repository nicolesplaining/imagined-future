#!/usr/bin/env python3
"""Resume-safe single-lane launcher for the prospective Cosmos 3 dose study."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

from imagined_future.cosmos3_archival import sha256


def load_analyzer(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_dose_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def validate_completed(
    path: Path,
    *,
    manifest_id: str,
    manifest_sha256: str,
    unit_id: str,
    unit: dict,
    manifest: dict,
    analyzer_module,
    expected_request_sequence: list[str],
    expected_parameter_probe_hash: str,
    intervention_site_error_tolerance: float,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "admission": "prospective_action_level_future_strength_dose_response",
        "manifest_id": manifest_id,
        "manifest_sha256": manifest_sha256,
        "unit_id": unit_id,
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"existing output is not an admissible exact resume: {path}: {actual}")
    validated_rows = analyzer_module.validate_report(
        payload, unit, manifest, manifest_sha256
    )
    if len(validated_rows) != 60:
        raise RuntimeError(f"existing output lacks the exact validated 12x5 grid: {path}")
    if payload.get("input_fingerprint_count") != 1:
        raise RuntimeError(f"existing output failed input-fingerprint gate: {path}")
    if (
        payload.get("parameter_probe_hash_count") != 1
        or payload.get("parameter_probe_hashes") != [expected_parameter_probe_hash]
        or payload.get("expected_parameter_probe_hash") != expected_parameter_probe_hash
    ):
        raise RuntimeError(f"existing output failed parameter-probe equality gate: {path}")
    if len(payload.get("dose_rows", [])) != 60:
        raise RuntimeError(f"existing output lacks the frozen 12x5 dose grid: {path}")
    if payload.get("request_count") != 92:
        raise RuntimeError(f"existing output lacks exactly 92 frozen requests: {path}")
    if payload.get("request_sequence") != expected_request_sequence:
        raise RuntimeError(f"existing output lacks the frozen request trace: {path}")
    if payload.get("request_class_census") != {
        "native": 4,
        "native_replay": 4,
        "none": 4,
        "self": 4,
        "self_replay": 4,
        "dose": 60,
        "midpoint_replay": 12,
    }:
        raise RuntimeError(f"existing output request-class census failed: {path}")
    response_actions = payload.get("response_actions", {})
    response_metadata = payload.get("response_metadata", {})
    if (
        set(response_actions) != set(expected_request_sequence)
        or set(response_metadata) != set(expected_request_sequence)
        or payload.get("wire_schema_validated_response_count") != 92
    ):
        raise RuntimeError(f"existing output response census failed: {path}")
    actions = list(response_actions.values())
    if len(actions) != 92 or any(
        not isinstance(action, list)
        or len(action) != 32
        or any(not isinstance(step, list) or len(step) != 8 for step in action)
        for action in actions
    ):
        raise RuntimeError(f"existing output action shape census failed: {path}")
    if any(
        not math.isfinite(float(coordinate))
        for action in actions
        for step in action
        for coordinate in step
    ):
        raise RuntimeError(f"existing output contains a nonfinite action coordinate: {path}")
    coordinate_errors = payload.get("action_coordinate_errors", {})
    if len(coordinate_errors) != 84 or any(
        pair != {"input": 0.0, "output": 0.0}
        for pair in coordinate_errors.values()
    ):
        raise RuntimeError(f"existing output lacks 84 exact action-coordinate audits: {path}")
    max_abs_residuals = payload.get("final_sampler_target_max_abs_errors", {})
    l2_residuals = payload.get("final_sampler_target_l2_errors", {})
    if (
        len(max_abs_residuals) != 80
        or len(l2_residuals) != 80
        or set(max_abs_residuals) != set(l2_residuals)
        or any(
            not math.isfinite(float(value))
            for value in [*max_abs_residuals.values(), *l2_residuals.values()]
        )
    ):
        raise RuntimeError(f"existing output lacks 80 paired finite residuals: {path}")
    none_controls = payload.get("none_controls", [])
    if len(none_controls) != 4 or any(
        row.get("action_max_abs_error_vs_native") != 0.0
        or row.get("native_trace_exact") is not True
        or row.get("projection_structural_null") is not True
        for row in none_controls
    ):
        raise RuntimeError(f"existing output failed zero-active-site no-op controls: {path}")
    site_audits = payload.get("intervention_site_audits", {})
    if len(site_audits) != 84:
        raise RuntimeError(f"existing output lacks 84 intervention-site audits: {path}")
    active_sites = sum(int(row.get("active_site_count", -1)) for row in site_audits.values())
    if active_sites != 320 or payload.get("active_intervention_site_count") != 320:
        raise RuntimeError(f"existing output lacks exactly 320 active sites: {path}")
    for row in site_audits.values():
        for key in ("model_input_max_error", "returned_velocity_max_error"):
            value = float(row.get(key, float("nan")))
            if not math.isfinite(value) or value > intervention_site_error_tolerance:
                raise RuntimeError(f"existing output failed {key} audit: {path}")
    native_repeat = payload.get("native_repeat_gates", {})
    if len(native_repeat) != 4 or any(
        row.get("action_max_abs_error") != 0.0
        or row.get("deterministic_metadata_exact") is not True
        or row.get("future_trace_exact") is not True
        for row in native_repeat.values()
    ):
        raise RuntimeError(f"existing output failed native replay: {path}")
    midpoint = payload.get("midpoint_replay_controls", [])
    if len(midpoint) != 12 or any(
        row.get("action_max_abs_error") != 0.0
        or row.get("deterministic_metadata_exact") is not True
        for row in midpoint
    ):
        raise RuntimeError(f"existing output failed midpoint replay: {path}")
    self_controls = payload.get("self_controls", [])
    if len(self_controls) != 4 or any(
        row.get("repeat_action_max_abs_error") != 0.0
        or row.get("repeat_signature_exact") is not True
        or row.get("projection_structural_null") is not True
        for row in self_controls
    ):
        raise RuntimeError(f"existing output failed self-clamp replay: {path}")


def main() -> None:
    args = parse_args()
    if args.shard_count != 1 or args.shard_index != 0:
        raise ValueError(
            "the frozen launch topology is one sequential shard on the general server"
        )
    actual_manifest_hash = sha256(args.manifest)
    if actual_manifest_hash != args.expected_manifest_sha256:
        raise ValueError("manifest SHA does not match the frozen CLI value")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_model_outcomes":
        raise ValueError("manifest is not frozen before model outcomes")
    if manifest.get("admission") != "prospective_action_level_future_strength_dose_response":
        raise ValueError("launcher refuses a non-evaluation manifest")
    if (
        manifest.get("freeze_stage") != "evaluation_ready"
        or manifest.get("launch_authorization")
        != "powered_evaluation_after_independent_go"
    ):
        raise ValueError("launcher refuses a pre-smoke/non-evaluation-ready manifest")
    if len(manifest.get("states", [])) != 30:
        raise ValueError("frozen dose evaluation must contain exactly 30 states")
    if sha256(args.audit_report) != args.expected_audit_sha256:
        raise ValueError("independent GO audit SHA differs from the frozen CLI value")
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "pass"
        or audit.get("verdict") != "GO"
        or audit.get("scope") != "outcome_blind_prelaunch_audit"
        or audit.get("manifest_id") != manifest.get("manifest_id")
        or audit.get("manifest_sha256") != actual_manifest_hash
        or audit.get("snapshot_checksum_list_sha256")
        != manifest["runtime"]["snapshot_checksum_list_sha256"]
        or audit.get("authorized_state_count") != 30
        or audit.get("authorized_call_count") != 2760
    ):
        raise ValueError("independent GO audit does not authorize this exact powered cohort")
    if sha256(args.runner) != manifest["runtime"]["runner_sha256"]:
        raise ValueError("runner differs from frozen manifest")
    if sha256(Path(__file__).resolve()) != manifest["runtime"]["launcher_sha256"]:
        raise ValueError("launcher differs from frozen manifest")
    if sha256(args.analyzer) != manifest["runtime"]["analyzer_sha256"]:
        raise ValueError("analyzer differs from frozen manifest")
    analyzer_module = load_analyzer(args.analyzer)
    checkpoint_verifier = Path(manifest["runtime"]["checkpoint_verifier_path"])
    if sha256(checkpoint_verifier) != manifest["runtime"]["checkpoint_verifier_sha256"]:
        raise ValueError("checkpoint verifier differs from frozen manifest")
    checkpoint_content_manifest = Path(
        manifest["runtime"]["checkpoint_content_manifest_path"]
    )
    if sha256(checkpoint_content_manifest) != manifest["runtime"][
        "checkpoint_content_manifest_sha256"
    ]:
        raise ValueError("checkpoint content manifest differs from frozen manifest")
    checkpoint_verification_receipt = Path(
        manifest["runtime"]["checkpoint_verification_receipt_path"]
    )
    if sha256(checkpoint_verification_receipt) != manifest["runtime"][
        "checkpoint_verification_receipt_sha256"
    ]:
        raise ValueError("checkpoint verification receipt differs from frozen manifest")
    for key in (
        "excluded_smoke_manifest",
        "excluded_smoke_artifact",
        "excluded_smoke_controls_report",
        "server_registry_empty_receipt",
    ):
        record = manifest["runtime"].get(key)
        if not isinstance(record, dict):
            raise ValueError(f"evaluation manifest does not bind {key}")
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"{key} differs from the frozen evaluation manifest")
    snapshot_root = Path(manifest["runtime"]["snapshot_root"])
    checksum_list = Path(manifest["runtime"]["snapshot_checksum_list_path"])
    if sha256(checksum_list) != manifest["runtime"]["snapshot_checksum_list_sha256"]:
        raise ValueError("snapshot checksum list differs from the frozen manifest")
    for relative, expected in manifest["runtime"]["snapshot_file_sha256"].items():
        candidate = (snapshot_root / relative).resolve()
        if not candidate.is_relative_to(snapshot_root.resolve()):
            raise ValueError(f"snapshot path escapes the frozen root: {relative}")
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ValueError(f"snapshot file differs from the frozen manifest: {relative}")
    for raw_path, expected in manifest["runtime"][
        "client_dependency_paths_sha256"
    ].items():
        path = Path(raw_path)
        if sha256(path) != expected:
            raise ValueError(f"client-side causal dependency changed: {path}")
    for raw_path, expected in manifest["runtime"][
        "host_mounted_server_dependency_paths_sha256"
    ].items():
        path = Path(raw_path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"host-mounted causal dependency changed: {path}")
    upstream_policy = Path(
        "/source/cosmos_framework/scripts/action_policy_server_robolab.py"
    )
    if sha256(upstream_policy) != manifest["runtime"][
        "upstream_robolab_policy_service_sha256"
    ]:
        raise ValueError("upstream RoboLab policy service changed")
    if args.port != int(manifest["runtime"]["server_port"]):
        raise ValueError("requested server port differs from frozen topology")
    if args.output_root.resolve() != Path(
        manifest["runtime"]["evaluation_output_root"]
    ).resolve():
        raise ValueError("requested output root differs from the frozen dose root")
    selected = manifest["states"][args.shard_index :: args.shard_count]
    if len(selected) != 30:
        raise RuntimeError("the sequential dose shard must assign all 30 states")
    args.output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(checkpoint_verifier),
            "--checkpoint-root",
            manifest["runtime"]["checkpoint_root"],
            "--content-manifest",
            str(checkpoint_content_manifest),
            "--expected-content-manifest-sha256",
            manifest["runtime"]["checkpoint_content_manifest_sha256"],
        ],
        check=True,
    )
    completed = 0
    skipped = 0
    for position, unit in enumerate(selected, start=1):
        unit_id = str(unit["unit_id"])
        output = args.output_root / f"{unit_id}.json"
        if output.exists():
            validate_completed(
                output,
                manifest_id=manifest["manifest_id"],
                manifest_sha256=args.expected_manifest_sha256,
                unit_id=unit_id,
                unit=unit,
                manifest=manifest,
                analyzer_module=analyzer_module,
                expected_request_sequence=[
                    str(row["label"]) for row in unit["request_sequence"]
                ],
                expected_parameter_probe_hash=manifest["runtime"][
                    "expected_parameter_probe_hash"
                ],
                intervention_site_error_tolerance=float(
                    manifest["runtime"]["intervention_site_error_tolerance"]
                ),
            )
            skipped += 1
            print(
                json.dumps(
                    {
                        "event": "resume_skip",
                        "shard_index": args.shard_index,
                        "position": position,
                        "unit_id": unit_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(args.runner),
            "--manifest",
            str(args.manifest),
            "--expected-manifest-sha256",
            args.expected_manifest_sha256,
            "--unit-id",
            unit_id,
            "--screen-root",
            str(args.screen_root),
            "--audit-report",
            str(args.audit_report),
            "--expected-audit-sha256",
            args.expected_audit_sha256,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--output",
            str(output),
        ]
        print(
            json.dumps(
                {
                    "event": "unit_start",
                    "shard_index": args.shard_index,
                    "position": position,
                    "unit_id": unit_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        subprocess.run(command, check=True)
        validate_completed(
            output,
            manifest_id=manifest["manifest_id"],
            manifest_sha256=args.expected_manifest_sha256,
            unit_id=unit_id,
            unit=unit,
            manifest=manifest,
            analyzer_module=analyzer_module,
            expected_request_sequence=[
                str(row["label"]) for row in unit["request_sequence"]
            ],
            expected_parameter_probe_hash=manifest["runtime"][
                "expected_parameter_probe_hash"
            ],
            intervention_site_error_tolerance=float(
                manifest["runtime"]["intervention_site_error_tolerance"]
            ),
        )
        completed += 1
        print(
            json.dumps(
                {
                    "event": "unit_complete",
                    "shard_index": args.shard_index,
                    "position": position,
                    "unit_id": unit_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "shard_complete",
                "shard_index": args.shard_index,
                "assigned": len(selected),
                "completed": completed,
                "resume_skipped": skipped,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
