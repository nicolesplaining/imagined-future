#!/usr/bin/env python3
"""Outcome-blind control validator for one frozen timing smoke artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject_nonstandard(value: str) -> None:
        raise ValueError(f"nonstandard JSON numeric literal: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard)


def null_paths(value: Any, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if value is None:
        return {path}
    if isinstance(value, Mapping):
        output: set[tuple[str, ...]] = set()
        for key, item in value.items():
            output.update(null_paths(item, path + (str(key),)))
        return output
    if isinstance(value, (list, tuple)):
        output = set()
        for index, item in enumerate(value):
            output.update(null_paths(item, path + (str(index),)))
        return output
    return set()


def numeric_leaves_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(numeric_leaves_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(numeric_leaves_finite(item) for item in value)
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return bool(np.isfinite(float(value)))
    return True


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    if sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("smoke manifest SHA differs")
    manifest = strict_json(args.manifest)
    report = strict_json(args.state)
    if (
        manifest.get("admission") != "excluded_development_smoke"
        or manifest.get("study_name") != "cosmos3-single-call-timing-v5"
        or len(manifest.get("states", [])) != 1
    ):
        raise ValueError("manifest is not the frozen v4 excluded smoke")
    unit = manifest["states"][0]
    expected = {
        "status": "complete",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.expected_manifest_sha256,
        "unit_id": unit["unit_id"],
        "request_count": 108,
        "structural_projection_null_count": 28,
        "finite_off_diagonal_projection_count": 72,
        "native_projection_absent_count": 8,
        "action_shape": [32, 8],
        "action_coordinate_count": 256,
        "shape_valid_response_action_count": 108,
        "action_shape_failure_count": 0,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{key} differs from frozen smoke: {report.get(key)!r}")

    rows = report.get("timing_rows")
    if not isinstance(rows, list) or len(rows) != 96:
        raise ValueError("timing row count is not 96")
    allowed_nulls: set[tuple[str, ...]] = set()
    diagonal_nulls = 0
    off_diagonal_finite = 0
    none_off_diagonal_zero = 0
    native_actions = report.get("native_actions")
    if not isinstance(native_actions, Mapping) or len(native_actions) != 4:
        raise ValueError("stored native action map is incomplete")
    for seed, value in native_actions.items():
        action = np.asarray(value, dtype=np.float64)
        if action.shape != (32, 8) or action.size != 256 or not np.all(np.isfinite(action)):
            raise ValueError(f"native action {seed} fails exact 32x8/256 schema")
    for index, row in enumerate(rows):
        recipient = int(row["recipient_seed"])
        source = int(row["source_seed"])
        timing = str(row["timing_condition"])
        server = row["server"]
        action = np.asarray(row.get("action"), dtype=np.float64)
        if action.shape != (32, 8) or action.size != 256 or not np.all(np.isfinite(action)):
            raise ValueError(f"row {index}: action fails exact 32x8/256 schema")
        attention = server.get("research_attention_interface")
        if (
            not isinstance(attention, Mapping)
            or attention.get("cache_id") is not None
            or attention.get("instrumented_server") is not False
            or attention.get("intervention_requested") is not False
            or attention.get("mode") != "exclude"
        ):
            raise ValueError(f"row {index}: invalid inactive attention routing metadata")
        allowed_nulls.add(
            (
                "timing_rows", str(index), "server",
                "research_attention_interface", "cache_id",
            )
        )
        projection = server.get("research_action_donor_projection", "missing")
        applicable = server.get("research_action_donor_projection_applicable")
        if recipient == source:
            if projection is not None or applicable is not False:
                raise ValueError(f"row {index}: invalid diagonal projection schema")
            allowed_nulls.add(
                (
                    "timing_rows", str(index), "server",
                    "research_action_donor_projection",
                )
            )
            diagonal_nulls += 1
        else:
            if isinstance(projection, bool) or not isinstance(projection, (int, float)):
                raise ValueError(f"row {index}: off-diagonal projection is not numeric")
            if not np.isfinite(float(projection)) or applicable is not True:
                raise ValueError(f"row {index}: off-diagonal projection is invalid")
            off_diagonal_finite += 1
            if timing == "none":
                if float(projection) != 0.0:
                    raise ValueError(f"row {index}: none projection is not exactly zero")
                none_off_diagonal_zero += 1
    if (diagonal_nulls, off_diagonal_finite, none_off_diagonal_zero) != (24, 72, 12):
        raise ValueError("stored projection census is not 24/72/12")
    if null_paths(report) != allowed_nulls:
        raise ValueError("state artifact contains a null outside frozen paths")
    if not numeric_leaves_finite(report):
        raise ValueError("state artifact contains NaN or infinity")

    zero_scalars = (
        "native_replay_max_action_error",
        "all_calls_diagonal_replay_max_action_error",
        "none_noop_max_action_error",
        "none_source_invariance_max_action_error",
        "maximum_action_input_error",
        "maximum_action_output_error",
        "maximum_active_model_input_future_clamp_error",
        "maximum_active_returned_future_velocity_error",
        "inactive_wrapper_write_count",
    )
    for key in zero_scalars:
        if report.get(key) != 0 and report.get(key) != 0.0:
            raise ValueError(f"control {key} is nonzero")
    runtime = report.get("runtime_gate")
    required_runtime = (
        "passed",
        "exact_schedule",
        "exact_active_site_captures",
        "exact_mask",
        "zero_action_coordinate_writes",
        "zero_inactive_wrapper_writes",
        "exact_none_noop",
        "exact_replays",
        "exact_rng_and_target_hashes",
        "required_numeric_fields_finite",
        "structural_null_census_exact",
        "exact_projection_applicability_census",
        "exact_action_shape_and_count",
    )
    if not isinstance(runtime, Mapping) or any(runtime.get(key) is not True for key in required_runtime):
        raise ValueError("one or more frozen runtime gates failed")

    audit = {
        "status": "pass",
        "scope": "excluded_smoke_controls_only_no_causal_outcomes_reported",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.expected_manifest_sha256,
        "state_artifact": str(args.state.resolve()),
        "state_artifact_sha256": sha256(args.state),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "request_count": 108,
        "projection_census": {
            "structural_null_intervention_responses": 28,
            "finite_off_diagonal_intervention_responses": 72,
            "native_field_absent_responses": 8,
            "stored_diagonal_null_rows": 24,
            "stored_off_diagonal_finite_rows": 72,
            "off_diagonal_none_exact_zero_rows": 12,
        },
        "required_numeric_fields_finite": True,
        "action_shape": [32, 8],
        "action_coordinate_count": 256,
        "shape_valid_wire_response_actions": 108,
        "shape_valid_stored_actions": 100,
        "action_shape_failure_count": 0,
        "structural_null_paths_exact": True,
        "all_frozen_runtime_controls_passed": True,
        "causal_action_or_timing_effect_metrics_included": False,
    }
    atomic_json(args.output, audit)
    print(json.dumps({**audit, "output": str(args.output), "output_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
