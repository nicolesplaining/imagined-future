#!/usr/bin/env python3
"""Control-only validator for the excluded 92-call dose-response smoke."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from imagined_future.cosmos3_archival import atomic_json, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--smoke-artifact", type=Path, required=True)
    parser.add_argument("--expected-smoke-artifact-sha256", required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_analyzer(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_dose_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke validation: {args.output}")
    if sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("smoke manifest hash differs from frozen CLI value")
    if sha256(args.smoke_artifact) != args.expected_smoke_artifact_sha256:
        raise ValueError("smoke artifact hash differs from frozen CLI value")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("study_name") != "cosmos3-future-strength-dose-response-v2"
        or manifest.get("admission") != "excluded_development_smoke"
        or len(manifest.get("states", [])) != 1
    ):
        raise ValueError("validator requires the one-state excluded dose manifest")
    if sha256(args.analyzer) != manifest["runtime"]["analyzer_sha256"]:
        raise ValueError("analyzer hash differs from smoke manifest")
    report = json.loads(args.smoke_artifact.read_text(encoding="utf-8"))
    analyzer = load_analyzer(args.analyzer)
    validated_rows = analyzer.validate_report(
        report,
        manifest["states"][0],
        manifest,
        args.expected_manifest_sha256,
    )
    if len(validated_rows) != 60:
        raise RuntimeError("smoke validator did not admit the exact 12x5 dose grid")
    control_report = {
        "status": "pass",
        "scope": "excluded_development_smoke_controls_only",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.expected_manifest_sha256,
        "smoke_artifact_path": str(args.smoke_artifact.resolve()),
        "smoke_artifact_sha256": args.expected_smoke_artifact_sha256,
        "analyzer_sha256": sha256(args.analyzer),
        "authorization_audit_path": report["authorization_audit_path"],
        "authorization_audit_sha256": report["authorization_audit_sha256"],
        "request_count": 92,
        "shape_valid_action_count": 92,
        "action_shape": [32, 8],
        "action_coordinate_count": 256,
        "request_class_census": report["request_class_census"],
        "native_replay_count": 4,
        "none_noop_count": 4,
        "self_replay_count": 4,
        "midpoint_replay_count": 12,
        "alpha_zero_routing_invariance_count": 12,
        "intervention_response_count": 84,
        "active_response_count": 80,
        "active_site_count": 320,
        "input_clamp_max_error": report["model_input_future_clamp_max_error"],
        "returned_velocity_max_error": report[
            "returned_future_velocity_overwrite_max_error"
        ],
        "action_nonwrite_count": 84,
        "input_fingerprint_count": report["input_fingerprint_count"],
        "parameter_probe_hash_count": report["parameter_probe_hash_count"],
        "wire_schema_validated_response_count": report[
            "wire_schema_validated_response_count"
        ],
        "scientific_outcomes_reported": False,
    }
    atomic_json(args.output, control_report)
    print(json.dumps(control_report, sort_keys=True))


if __name__ == "__main__":
    main()
