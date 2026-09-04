#!/usr/bin/env python3
"""Validate and summarize the complete prospective Cosmos 3 dose cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from imagined_future.cosmos3_archival import atomic_json, sha256
from imagined_future.cosmos3_dose_response import (
    ALPHAS,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    EXPECTED_ACTIVE_RESPONSES_PER_STATE,
    EXPECTED_ACTIVE_SITES_PER_STATE,
    EXPECTED_CALLS_PER_STATE,
    adjacent_contrasts,
    dose_action_metrics,
    dose_label,
    frozen_request_specs,
    ols_slope,
    pair_is_nondecreasing_proximity,
    validate_released_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is nonfinite: {value!r}")
    return result


def null_and_nonfinite_paths(
    value: Any, prefix: str = ""
) -> tuple[set[str], set[str]]:
    nulls: set[str] = set()
    nonfinite: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            child_nulls, child_nonfinite = null_and_nonfinite_paths(item, path)
            nulls.update(child_nulls)
            nonfinite.update(child_nonfinite)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_nulls, child_nonfinite = null_and_nonfinite_paths(
                item, f"{prefix}[{index}]"
            )
            nulls.update(child_nulls)
            nonfinite.update(child_nonfinite)
    elif value is None:
        nulls.add(prefix)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            nonfinite.add(prefix)
    return nulls, nonfinite


def torch_bool_tensor_digest(value: np.ndarray) -> str:
    flat = np.ascontiguousarray(np.asarray(value, dtype=np.bool_).reshape(-1))
    digest = hashlib.sha256()
    digest.update(b"torch.bool")
    digest.update(np.asarray(flat.shape, dtype=np.int64).tobytes())
    digest.update(flat.view(np.uint8).tobytes())
    return digest.hexdigest()


def behavior_signature(action: Any, metadata: dict[str, Any]) -> str:
    keys = (
        "research_target_hash",
        "research_recipient_future_hash",
        "research_recipient_path_noise_hash",
        "research_initial_state_hash",
        "research_output_future_hash",
        "research_final_sampler_target_max_abs_error",
        "research_final_sampler_target_l2",
        "research_sigmas",
        "research_x0_sigmas",
        "research_x0_vision_hashes",
        "research_x0_action_hashes",
        "research_vision_shape",
        "research_future_frame_indices",
        "research_vision_coordinate_count",
        "research_future_mask_coordinate_count",
        "research_future_mask_index_hash",
        "research_requested_active_call_indices",
        "research_observed_active_call_indices",
        "research_clamped_call_indices",
        "research_inactive_call_indices",
        "research_requested_active_sigmas",
        "research_observed_active_sigmas",
        "research_model_input_future_clamp_errors",
        "research_returned_future_velocity_overwrite_errors",
        "research_maximum_action_input_error",
        "research_maximum_action_output_error",
        "research_action_input_errors",
        "research_action_output_errors",
        "research_inactive_wrapper_write_count",
    )
    missing = [key for key in keys if key not in metadata]
    if missing:
        raise ValueError(f"behavior signature missing fields: {missing}")
    payload = {"action": action, **{key: metadata[key] for key in keys}}
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def deterministic_metadata_signature(metadata: dict[str, Any]) -> str:
    """Mirror the runner replay signature after actions are checked separately."""

    excluded = {"research_id", "research_infer_ms", "server_timing"}
    payload = {key: value for key, value in metadata.items() if key not in excluded}
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def require_audit_matches_metadata(
    audit: dict[str, Any], metadata: dict[str, Any], *, label: str
) -> None:
    """Bind every retained manipulation-site audit to the raw server response."""

    direct_fields = (
        "mode",
        "target_source",
        "target_hash",
        "target_source_record_ids",
        "recipient_future_hash",
        "donor_future_hash",
        "recipient_path_noise_hash",
        "initial_state_hash",
        "sigmas",
        "requested_active_call_indices",
        "observed_active_call_indices",
        "clamped_call_indices",
        "inactive_call_indices",
        "requested_active_sigmas",
        "observed_active_sigmas",
        "future_frame_indices",
        "vision_shape",
        "future_mask_index_hash",
        "model_input_future_clamp_errors",
        "returned_future_velocity_overwrite_errors",
        "final_sampler_target_max_abs_error",
        "final_sampler_target_l2",
        "vision_coordinate_count",
        "target_coordinate_count",
        "target_finite_coordinate_count",
        "maximum_action_input_error",
        "maximum_action_output_error",
        "action_input_errors",
        "action_output_errors",
        "inactive_wrapper_write_count",
        "alpha",
        "interpolation_formula_max_abs_error",
        "nonfuture_recipient_target_max_abs_error",
        "alpha_zero_recipient_future_max_abs_error",
        "alpha_one_donor_future_max_abs_error",
        "alpha_zero_target_hash_matches_recipient",
        "alpha_one_target_hash_matches_donor",
        "interpolated_future_hash",
        "recipient_future_mask_hash",
        "donor_future_mask_hash",
        "current_frame_hash",
        "recipient_current_frame_hash",
        "future_mask_hash",
    )
    for field in direct_fields:
        metadata_key = f"research_{field}"
        expected = metadata.get(metadata_key)
        if audit.get(field) != expected:
            raise ValueError(
                f"{label}: retained audit field {field} differs from raw response"
            )
    if audit.get("mask_coordinate_count") != metadata.get(
        "research_future_mask_coordinate_count"
    ):
        raise ValueError(f"{label}: retained mask count differs from raw response")
    if (
        audit.get("recipient_id") != metadata.get("research_recipient_id")
        or audit.get("donor_id") != metadata.get("research_donor_id")
    ):
        raise ValueError(f"{label}: retained registry IDs differ from raw response")
    active_count = len(metadata.get("research_observed_active_call_indices", []))
    if audit.get("active_site_count") != active_count:
        raise ValueError(f"{label}: retained active-site count differs from raw response")
    raw_input = [float(value) for value in metadata["research_model_input_future_clamp_errors"]]
    raw_velocity = [
        float(value)
        for value in metadata["research_returned_future_velocity_overwrite_errors"]
    ]
    if audit.get("model_input_max_error") != (max(raw_input) if raw_input else 0.0):
        raise ValueError(f"{label}: retained input-error maximum differs")
    if audit.get("returned_velocity_max_error") != (
        max(raw_velocity) if raw_velocity else 0.0
    ):
        raise ValueError(f"{label}: retained velocity-error maximum differs")


def _ci(values: np.ndarray, point: float) -> dict[str, float]:
    return {
        "estimate": float(point),
        "lower": float(np.quantile(values, 0.025, method="linear")),
        "upper": float(np.quantile(values, 0.975, method="linear")),
    }


def hierarchical_draw_table(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> tuple[tuple[str, ...], dict[str, tuple[dict[str, Any], ...]], np.ndarray, np.ndarray]:
    """Create the one shared PCG64 task->state resampling table."""

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    tasks = sorted(by_task)
    if len(tasks) != 6 or any(len(by_task[task]) != 5 for task in tasks):
        raise ValueError("hierarchical bootstrap requires six tasks x five states")
    generator = np.random.Generator(np.random.PCG64(seed))
    task_indices = np.empty((samples, 6), dtype=np.int64)
    state_indices = np.empty((samples, 6, 5), dtype=np.int64)
    for draw in range(samples):
        task_indices[draw] = generator.integers(0, 6, size=6)
        for occurrence in range(6):
            state_indices[draw, occurrence] = generator.integers(0, 5, size=5)
    return (
        tuple(tasks),
        {task: tuple(by_task[task]) for task in tasks},
        task_indices,
        state_indices,
    )


def hierarchical_estimate(
    draw_table: tuple[
        tuple[str, ...],
        dict[str, tuple[dict[str, Any], ...]],
        np.ndarray,
        np.ndarray,
    ],
    value: Callable[[dict[str, Any]], float],
) -> dict[str, float]:
    """Evaluate one estimand on the single shared hierarchical draw table."""

    tasks, by_task, task_indices, state_indices = draw_table
    point = float(
        np.mean(
            [np.mean([value(row) for row in by_task[task]]) for task in tasks]
        )
    )
    draws = np.empty(len(task_indices), dtype=np.float64)
    for draw in range(len(task_indices)):
        occurrence_means: list[float] = []
        for occurrence, task_index in enumerate(task_indices[draw]):
            task_rows = by_task[tasks[int(task_index)]]
            occurrence_means.append(
                float(
                    np.mean(
                        [
                            value(task_rows[int(index)])
                            for index in state_indices[draw, occurrence]
                        ]
                    )
                )
            )
        draws[draw] = float(np.mean(occurrence_means))
    return _ci(draws, point)


def _action(value: Any, *, label: str) -> np.ndarray:
    return validate_released_action(value, label=label)


def validate_report(
    report: dict[str, Any],
    unit: dict[str, Any],
    manifest: dict[str, Any],
    manifest_hash: str,
) -> list[dict[str, Any]]:
    expected_header = {
        "status": "complete",
        "admission": manifest["admission"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_hash,
        "unit_id": unit["unit_id"],
        "task": unit["task"],
        "environment_seed": unit["environment_seed"],
        "episode_id": unit["episode_id"],
        "phase": "middle",
    }
    if {key: report.get(key) for key in expected_header} != expected_header:
        raise ValueError(f"report header differs for {unit['unit_id']}")
    authorization_path = Path(str(report.get("authorization_audit_path", "")))
    authorization_sha256 = str(report.get("authorization_audit_sha256", ""))
    if (
        len(authorization_sha256) != 64
        or not authorization_path.is_file()
        or sha256(authorization_path) != authorization_sha256
    ):
        raise ValueError(f"authorization audit binding failed for {unit['unit_id']}")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    expected_authorization = (
        {
            "status": "pass",
            "verdict": "GO",
            "scope": "outcome_blind_prelaunch_audit",
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest_hash,
            "snapshot_checksum_list_sha256": manifest["runtime"][
                "snapshot_checksum_list_sha256"
            ],
            "authorized_state_count": 30,
            "authorized_call_count": 2760,
        }
        if manifest["admission"]
        == "prospective_action_level_future_strength_dose_response"
        else {
            "status": "pass",
            "verdict": "GO_SMOKE",
            "scope": "outcome_blind_excluded_smoke_prelaunch_audit",
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest_hash,
            "snapshot_checksum_list_sha256": manifest["runtime"][
                "snapshot_checksum_list_sha256"
            ],
            "authorized_state_count": 1,
            "authorized_call_count": EXPECTED_CALLS_PER_STATE,
        }
    )
    if (
        {key: authorization.get(key) for key in expected_authorization}
        != expected_authorization
    ):
        raise ValueError(f"authorization audit does not match {unit['unit_id']}")
    expected_sequence = [
        str(row["label"]) for row in frozen_request_specs(unit["branch_seeds"])
    ]
    if (
        report.get("request_count") != EXPECTED_CALLS_PER_STATE
        or report.get("request_sequence") != expected_sequence
        or report.get("alpha_grid") != list(ALPHAS)
        or report.get("ordered_pairs") != unit["ordered_pairs"]
    ):
        raise ValueError(f"request design differs for {unit['unit_id']}")
    if report.get("request_class_census") != {
        "native": 4,
        "native_replay": 4,
        "none": 4,
        "self": 4,
        "self_replay": 4,
        "dose": 60,
        "midpoint_replay": 12,
    }:
        raise ValueError(f"request-class census differs for {unit['unit_id']}")
    response_actions = report.get("response_actions", {})
    if set(response_actions) != set(expected_sequence) or len(response_actions) != 92:
        raise ValueError(f"response-action order/census failed for {unit['unit_id']}")
    for label, action in response_actions.items():
        _action(action, label=f"{unit['unit_id']}:{label}")
    response_metadata = report.get("response_metadata", {})
    if (
        set(response_metadata) != set(expected_sequence)
        or len(response_metadata) != 92
        or report.get("wire_schema_validated_response_count") != 92
    ):
        raise ValueError(f"response-metadata census failed for {unit['unit_id']}")
    spec_by_label = {
        str(row["label"]): row for row in frozen_request_specs(unit["branch_seeds"])
    }
    for label, metadata in response_metadata.items():
        nulls, nonfinite_paths = null_and_nonfinite_paths(metadata)
        allowed = {"research_attention_interface.cache_id"}
        kind = str(spec_by_label[label]["kind"])
        if kind in {"none", "self", "self_repeat"}:
            allowed.add("research_action_donor_projection")
        if kind in {"dose", "dose_midpoint_repeat"}:
            alpha = float(spec_by_label[label]["alpha"])
            if alpha == 0.0:
                allowed.update(
                    {
                        "research_alpha_one_donor_future_max_abs_error",
                        "research_alpha_one_target_hash_matches_donor",
                    }
                )
            elif alpha == 1.0:
                allowed.update(
                    {
                        "research_alpha_zero_recipient_future_max_abs_error",
                        "research_alpha_zero_target_hash_matches_recipient",
                    }
                )
            else:
                allowed.update(
                    {
                        "research_alpha_zero_recipient_future_max_abs_error",
                        "research_alpha_one_donor_future_max_abs_error",
                        "research_alpha_zero_target_hash_matches_recipient",
                        "research_alpha_one_target_hash_matches_donor",
                    }
                )
        if nonfinite_paths or nulls != allowed:
            raise ValueError(
                f"{unit['unit_id']}:{label} wire schema failed: "
                f"nonfinite={sorted(nonfinite_paths)}, nulls={sorted(nulls)}, "
                f"allowed={sorted(allowed)}"
            )
    raw_input_fingerprints = {
        metadata.get("research_state_hash") for metadata in response_metadata.values()
    }
    raw_parameter_probes = {
        metadata.get("research_parameter_probe_hash")
        for metadata in response_metadata.values()
    }
    expected_probe = manifest["runtime"]["expected_parameter_probe_hash"]
    if (
        len(raw_input_fingerprints) != 1
        or None in raw_input_fingerprints
        or raw_parameter_probes != {expected_probe}
        or report.get("input_fingerprint_count") != 1
        or report.get("input_fingerprints") != sorted(raw_input_fingerprints)
        or report.get("parameter_probe_hash_count") != 1
        or report.get("parameter_probe_hashes") != [expected_probe]
        or report.get("expected_parameter_probe_hash") != expected_probe
    ):
        raise ValueError(f"input/checkpoint singleton gate failed for {unit['unit_id']}")
    native_repeat = report.get("native_repeat_gates", {})
    if len(native_repeat) != 4 or any(
        row != {
            "action_max_abs_error": 0.0,
            "deterministic_metadata_exact": True,
            "future_trace_exact": True,
        }
        for row in native_repeat.values()
    ):
        raise ValueError(f"native replay gate failed for {unit['unit_id']}")
    native_trace_keys = (
        "research_future_hash",
        "research_x0_vision_hashes",
        "research_x0_action_hashes",
        "research_path_noise_hash",
        "research_initial_state_hash",
        "research_sigmas",
        "research_x0_sigmas",
    )
    for seed in unit["branch_seeds"]:
        base_label = f"native-{int(seed)}"
        repeat_label = f"native-repeat-{int(seed)}"
        base_metadata = response_metadata[base_label]
        repeat_metadata = response_metadata[repeat_label]
        if (
            not np.array_equal(
                _action(response_actions[base_label], label=base_label),
                _action(response_actions[repeat_label], label=repeat_label),
            )
            or deterministic_metadata_signature(base_metadata)
            != deterministic_metadata_signature(repeat_metadata)
            or any(base_metadata.get(key) != repeat_metadata.get(key) for key in native_trace_keys)
        ):
            raise ValueError(f"native raw replay differs for {unit['unit_id']}:{seed}")
    none = report.get("none_controls", [])
    if len(none) != 4 or any(
        row.get("action_max_abs_error_vs_native") != 0.0
        or row.get("native_trace_exact") is not True
        or row.get("projection_structural_null") is not True
        for row in none
    ):
        raise ValueError(f"none control gate failed for {unit['unit_id']}")
    for row in none:
        seed = int(row["recipient_seed"])
        label = f"none-{seed}"
        metadata = response_metadata[label]
        native_metadata = response_metadata[f"native-{seed}"]
        if (
            row.get("server") != metadata
            or metadata.get("research_action_donor_projection_applicable") is not False
            or metadata.get("research_action_donor_projection") is not None
            or not np.array_equal(
                _action(response_actions[label], label=label),
                _action(response_actions[f"native-{seed}"], label=f"native-{seed}"),
            )
            or metadata.get("research_output_future_hash")
            != native_metadata.get("research_future_hash")
            or any(
                metadata.get(key) != native_metadata.get(key)
                for key in (
                    "research_x0_vision_hashes",
                    "research_x0_action_hashes",
                    "research_sigmas",
                    "research_x0_sigmas",
                )
            )
        ):
            raise ValueError(f"none raw no-op differs for {unit['unit_id']}:{seed}")
    self_rows = report.get("self_controls", [])
    if len(self_rows) != 4 or any(
        row.get("repeat_action_max_abs_error") != 0.0
        or row.get("repeat_signature_exact") is not True
        or row.get("projection_structural_null") is not True
        for row in self_rows
    ):
        raise ValueError(f"self replay gate failed for {unit['unit_id']}")
    for row in self_rows:
        seed = int(row["recipient_seed"])
        base_label = f"self-{seed}"
        repeat_label = f"self-repeat-{seed}"
        base_metadata = response_metadata[base_label]
        repeat_metadata = response_metadata[repeat_label]
        base_action = _action(response_actions[base_label], label=base_label)
        repeat_action = _action(response_actions[repeat_label], label=repeat_label)
        native_action = _action(response_actions[f"native-{seed}"], label=f"native-{seed}")
        if (
            row.get("server") != base_metadata
            or base_metadata.get("research_action_donor_projection_applicable") is not False
            or repeat_metadata.get("research_action_donor_projection_applicable") is not False
            or base_metadata.get("research_action_donor_projection") is not None
            or repeat_metadata.get("research_action_donor_projection") is not None
            or not np.array_equal(base_action, repeat_action)
            or deterministic_metadata_signature(base_metadata)
            != deterministic_metadata_signature(repeat_metadata)
            or row.get("action") != response_actions[base_label]
            or finite(row.get("clean_clamp_vs_native_max_abs_error"), label=base_label)
            != float(np.max(np.abs(base_action.astype(np.float64) - native_action.astype(np.float64))))
            or finite(row.get("clean_clamp_vs_native_l2"), label=base_label)
            != float(np.linalg.norm(base_action.astype(np.float64) - native_action.astype(np.float64)))
        ):
            raise ValueError(f"self raw replay differs for {unit['unit_id']}:{seed}")
    midpoint = report.get("midpoint_replay_controls", [])
    if len(midpoint) != 12 or any(
        row.get("action_max_abs_error") != 0.0
        or row.get("deterministic_metadata_exact") is not True
        for row in midpoint
    ):
        raise ValueError(f"midpoint replay gate failed for {unit['unit_id']}")
    midpoint_cells = []
    for row in midpoint:
        recipient_seed = int(row["recipient_seed"])
        donor_seed = int(row["donor_seed"])
        midpoint_cells.append((recipient_seed, donor_seed))
        base_label = dose_label(recipient_seed, donor_seed, 0.5)
        repeat_label = base_label + "-repeat"
        if (
            float(row.get("alpha")) != 0.5
            or not np.array_equal(
                _action(response_actions[base_label], label=base_label),
                _action(response_actions[repeat_label], label=repeat_label),
            )
            or deterministic_metadata_signature(response_metadata[base_label])
            != deterministic_metadata_signature(response_metadata[repeat_label])
        ):
            raise ValueError(
                f"midpoint raw replay differs for {unit['unit_id']}:{recipient_seed}:{donor_seed}"
            )
    if midpoint_cells != [tuple(int(item) for item in pair) for pair in unit["ordered_pairs"]]:
        raise ValueError(f"midpoint replay order differs for {unit['unit_id']}")
    alpha_zero = report.get("alpha_zero_routing_controls", [])
    if len(alpha_zero) != 12 or any(
        row.get("action_max_abs_error_vs_self") != 0.0
        or row.get("behavior_signature_exact_vs_self") is not True
        or not isinstance(row.get("behavior_signature"), str)
        for row in alpha_zero
    ):
        raise ValueError(f"alpha-zero routing-invariance gate failed for {unit['unit_id']}")
    for recipient_seed in unit["branch_seeds"]:
        self_label = f"self-{int(recipient_seed)}"
        expected_signature = behavior_signature(
            response_actions[self_label], response_metadata[self_label]
        )
        recipient_rows = [
            row
            for row in alpha_zero
            if int(row["recipient_seed"]) == int(recipient_seed)
        ]
        signatures: set[str] = set()
        for row in recipient_rows:
            label = dose_label(int(recipient_seed), int(row["donor_seed"]), 0.0)
            actual_signature = behavior_signature(
                response_actions[label], response_metadata[label]
            )
            if (
                actual_signature != expected_signature
                or row["behavior_signature"] != actual_signature
            ):
                raise ValueError(
                    f"alpha-zero behavior signature differs for {unit['unit_id']}:{label}"
                )
            signatures.add(actual_signature)
        if len(signatures) != 1:
            raise ValueError(
                f"alpha-zero donor-label invariance failed for {unit['unit_id']}:{recipient_seed}"
            )

    coordinate_errors = report.get("action_coordinate_errors", {})
    expected_intervention_labels = {
        label
        for label, spec in spec_by_label.items()
        if spec["kind"] not in {"native", "native_repeat"}
    }
    if set(coordinate_errors) != expected_intervention_labels or any(
        row != {"input": 0.0, "output": 0.0}
        for row in coordinate_errors.values()
    ):
        raise ValueError(f"action nonwrite census failed for {unit['unit_id']}")
    audits = report.get("intervention_site_audits", {})
    if set(audits) != expected_intervention_labels:
        raise ValueError(f"intervention audit census failed for {unit['unit_id']}")
    if (
        report.get("intervention_response_count") != 84
        or report.get("active_intervention_response_count")
        != EXPECTED_ACTIVE_RESPONSES_PER_STATE
        or report.get("active_intervention_site_count")
        != EXPECTED_ACTIVE_SITES_PER_STATE
        or sum(int(row.get("active_site_count", -1)) for row in audits.values())
        != EXPECTED_ACTIVE_SITES_PER_STATE
    ):
        raise ValueError(f"active-site census failed for {unit['unit_id']}")
    native_server = report.get("native_server", {})
    if set(native_server) != {str(seed) for seed in unit["branch_seeds"]}:
        raise ValueError(f"native server metadata grid failed for {unit['unit_id']}")
    for seed in unit["branch_seeds"]:
        if native_server[str(seed)] != response_metadata[f"native-{int(seed)}"]:
            raise ValueError(f"native metadata copy differs for {unit['unit_id']}:{seed}")
    expected_future_hashes = {
        str(seed): response_metadata[f"native-{int(seed)}"].get("research_future_hash")
        for seed in unit["branch_seeds"]
    }
    if (
        report.get("native_future_hashes") != expected_future_hashes
        or report.get("native_future_hashes_distinct")
        != (len(set(expected_future_hashes.values())) == 4)
    ):
        raise ValueError(f"native future-hash record differs for {unit['unit_id']}")
    study_id = f"{manifest['manifest_id']}-{unit['unit_id']}"
    native_by_id = {
        f"{study_id}-native-{seed}": native_server[str(seed)]
        for seed in unit["branch_seeds"]
    }
    if report.get("recipient_schedule_identity_count") != 84:
        raise ValueError(f"schedule-identity census failed for {unit['unit_id']}")
    null_projection_count = 0
    finite_projection_count = 0
    absent_projection_count = 0
    for label, metadata in response_metadata.items():
        kind = str(spec_by_label[label]["kind"])
        has_projection = "research_action_donor_projection" in metadata
        has_applicability = "research_action_donor_projection_applicable" in metadata
        if kind in {"native", "native_repeat"}:
            if has_projection or has_applicability:
                raise ValueError(f"{unit['unit_id']}:{label} native projection must be absent")
            absent_projection_count += 1
        elif kind in {"none", "self", "self_repeat"}:
            if (
                not has_projection
                or not has_applicability
                or metadata["research_action_donor_projection"] is not None
                or metadata["research_action_donor_projection_applicable"] is not False
            ):
                raise ValueError(f"{unit['unit_id']}:{label} diagonal projection schema failed")
            null_projection_count += 1
        else:
            if (
                not has_projection
                or not has_applicability
                or metadata["research_action_donor_projection_applicable"] is not True
            ):
                raise ValueError(f"{unit['unit_id']}:{label} off-diagonal projection schema failed")
            finite(metadata["research_action_donor_projection"], label=f"{label}:projection")
            finite_projection_count += 1
    if (null_projection_count, finite_projection_count, absent_projection_count) != (12, 72, 8):
        raise ValueError(
            f"{unit['unit_id']}: projection census differs: "
            f"{null_projection_count}/{finite_projection_count}/{absent_projection_count}"
        )
    if report.get("structural_null_schema") != {
        "research_action_donor_projection": (
            "null iff recipient and donor are identical in none/self controls; "
            "finite for all off-diagonal dose responses"
        ),
        "expected_null_count": 12,
        "expected_finite_count": 72,
        "expected_absent_native_count": 8,
    }:
        raise ValueError(f"{unit['unit_id']}: structural-null schema record differs")
    schedule_count = 0
    for label, metadata in response_metadata.items():
        if metadata.get("research_mode") not in {"none", "self", "dose"}:
            continue
        recipient_id = metadata.get("research_recipient_id")
        if recipient_id not in native_by_id:
            raise ValueError(f"{unit['unit_id']}:{label} unknown recipient ID")
        native_metadata = native_by_id[recipient_id]
        if (
            metadata.get("research_sigmas") != native_metadata.get("research_sigmas")
            or metadata.get("research_x0_sigmas")
            != native_metadata.get("research_x0_sigmas")
        ):
            raise ValueError(f"{unit['unit_id']}:{label} recipient schedule differs")
        schedule_count += 1
    if schedule_count != 84:
        raise ValueError(f"schedule response count failed for {unit['unit_id']}")
    tolerance = float(manifest["runtime"]["intervention_site_error_tolerance"])
    for label, row in audits.items():
        if label not in response_metadata:
            raise ValueError(f"{unit['unit_id']}:{label} audit has no raw response")
        require_audit_matches_metadata(
            row, response_metadata[label], label=f"{unit['unit_id']}:{label}"
        )
        request_kind = str(spec_by_label[label]["kind"])
        expected_mode = (
            "none"
            if request_kind == "none"
            else "self"
            if request_kind in {"self", "self_repeat"}
            else "dose"
        )
        expected_target_source = (
            "recipient"
            if expected_mode in {"none", "self"}
            else "recipient_donor_linear_interpolation"
        )
        expected_source_ids = (
            [row.get("recipient_id")]
            if expected_target_source == "recipient"
            else [row.get("recipient_id"), row.get("donor_id")]
        )
        if (
            row.get("mode") != expected_mode
            or row.get("target_source") != expected_target_source
            or row.get("target_source_record_ids") != expected_source_ids
        ):
            raise ValueError(f"{unit['unit_id']}:{label} mode/source routing differs")
        expected_alpha = (
            float(spec_by_label[label]["alpha"])
            if request_kind in {"dose", "dose_midpoint_repeat"}
            else None
        )
        if row.get("alpha") != expected_alpha:
            raise ValueError(f"{unit['unit_id']}:{label} alpha differs from request")
        is_active = row.get("mode") != "none"
        expected_indices = [0, 1, 2, 3] if is_active else []
        expected_inactive = [] if is_active else [0, 1, 2, 3]
        sigmas = [finite(value, label=f"{label}:sigma") for value in row.get("sigmas", [])]
        if (
            len(sigmas) != 4
            or row.get("requested_active_call_indices") != expected_indices
            or row.get("observed_active_call_indices") != expected_indices
            or row.get("clamped_call_indices") != expected_indices
            or row.get("inactive_call_indices") != expected_inactive
            or row.get("requested_active_sigmas") != [sigmas[index] for index in expected_indices]
            or row.get("observed_active_sigmas") != [sigmas[index] for index in expected_indices]
            or int(row.get("active_site_count", -1)) != len(expected_indices)
        ):
            raise ValueError(f"{unit['unit_id']}:{label} call/sigma audit failed")
        for key in (
            "model_input_future_clamp_errors",
            "returned_future_velocity_overwrite_errors",
        ):
            values = [finite(value, label=f"{label}:{key}") for value in row.get(key, [])]
            if len(values) != len(expected_indices) or any(value > tolerance for value in values):
                raise ValueError(f"{unit['unit_id']}:{label}:{key} audit failed")
        for key in ("action_input_errors", "action_output_errors"):
            values = [finite(value, label=f"{label}:{key}") for value in row.get(key, [])]
            if len(values) != 4 or any(value != 0.0 for value in values):
                raise ValueError(f"{unit['unit_id']}:{label}:{key} audit failed")
        shape = tuple(int(value) for value in row.get("vision_shape", []))
        frames = tuple(int(value) for value in row.get("future_frame_indices", []))
        if len(shape) not in (4, 5) or frames != tuple(range(1, 9)):
            raise ValueError(f"{unit['unit_id']}:{label} vision/frame schema failed")
        temporal_axis = len(shape) - 3
        expected_mask = np.zeros(shape, dtype=np.bool_)
        index = [slice(None)] * len(shape)
        index[temporal_axis] = list(frames)
        expected_mask[tuple(index)] = True
        if (
            int(row.get("vision_coordinate_count", -1)) != int(expected_mask.size)
            or int(row.get("mask_coordinate_count", -1)) != int(expected_mask.sum())
            or int(row.get("target_coordinate_count", -1)) != int(expected_mask.size)
            or int(row.get("target_finite_coordinate_count", -1))
            != int(expected_mask.size)
            or row.get("future_mask_index_hash") != torch_bool_tensor_digest(expected_mask)
        ):
            raise ValueError(f"{unit['unit_id']}:{label} mask audit failed")
        for key in (
            "model_input_max_error",
            "returned_velocity_max_error",
            "maximum_action_input_error",
            "maximum_action_output_error",
            "inactive_wrapper_write_count",
        ):
            value = finite(row.get(key), label=f"{unit['unit_id']}:{label}:{key}")
            if key in {"model_input_max_error", "returned_velocity_max_error"}:
                if value > tolerance:
                    raise ValueError(f"{unit['unit_id']}:{label}:{key} exceeds tolerance")
            elif value != 0.0:
                raise ValueError(f"{unit['unit_id']}:{label}:{key} is nonzero")
        recipient_id = row.get("recipient_id")
        donor_id = row.get("donor_id")
        if recipient_id not in native_by_id or donor_id not in native_by_id:
            raise ValueError(f"{unit['unit_id']}:{label} record IDs are not native IDs")
        recipient_native = native_by_id[recipient_id]
        donor_native = native_by_id[donor_id]
        if (
            row.get("recipient_future_hash") != recipient_native.get("research_future_hash")
            or row.get("donor_future_hash") != donor_native.get("research_future_hash")
            or row.get("recipient_path_noise_hash")
            != recipient_native.get("research_path_noise_hash")
            or row.get("initial_state_hash")
            != recipient_native.get("research_initial_state_hash")
        ):
            raise ValueError(f"{unit['unit_id']}:{label} target/RNG provenance failed")
        if (
            expected_target_source == "recipient"
            and row.get("target_hash") != recipient_native.get("research_future_hash")
        ):
            raise ValueError(f"{unit['unit_id']}:{label} recipient target hash differs")
        if row.get("alpha") is not None:
            metadata = response_metadata[label]
            if (
                metadata.get("research_alpha") != row.get("alpha")
                or metadata.get("research_alpha_grid") != list(ALPHAS)
                or metadata.get("research_interpolation_formula")
                != "F_A + alpha * (F_B - F_A)"
                or finite(
                    row.get("interpolation_formula_max_abs_error"),
                    label=f"{unit['unit_id']}:{label}:formula",
                )
                != 0.0
                or finite(
                    row.get("nonfuture_recipient_target_max_abs_error"),
                    label=f"{unit['unit_id']}:{label}:nonfuture",
                )
                != 0.0
            ):
                raise ValueError(f"{unit['unit_id']}:{label} interpolation gate failed")
            alpha = float(row["alpha"])
            zero_error = row.get("alpha_zero_recipient_future_max_abs_error")
            one_error = row.get("alpha_one_donor_future_max_abs_error")
            if alpha == 0.0:
                if (
                    zero_error != 0.0
                    or one_error is not None
                    or row.get("target_hash")
                    != recipient_native.get("research_future_hash")
                    or row.get("alpha_zero_target_hash_matches_recipient") is not True
                    or row.get("alpha_one_target_hash_matches_donor") is not None
                    or row.get("interpolated_future_hash")
                    != row.get("recipient_future_mask_hash")
                ):
                    raise ValueError(f"{unit['unit_id']}:{label} alpha=0 identity failed")
            elif alpha == 1.0:
                if (
                    one_error != 0.0
                    or zero_error is not None
                    or row.get("target_hash")
                    != donor_native.get("research_future_hash")
                    or row.get("alpha_one_target_hash_matches_donor") is not True
                    or row.get("alpha_zero_target_hash_matches_recipient") is not None
                    or row.get("interpolated_future_hash")
                    != row.get("donor_future_mask_hash")
                ):
                    raise ValueError(f"{unit['unit_id']}:{label} alpha=1 identity failed")
            elif (
                zero_error is not None
                or one_error is not None
                or row.get("alpha_zero_target_hash_matches_recipient") is not None
                or row.get("alpha_one_target_hash_matches_donor") is not None
            ):
                raise ValueError(f"{unit['unit_id']}:{label} endpoint fields malformed")
            if (
                row.get("current_frame_hash")
                != row.get("recipient_current_frame_hash")
                or row.get("future_mask_hash") != row.get("future_mask_index_hash")
            ):
                raise ValueError(f"{unit['unit_id']}:{label} frame/mask hash gate failed")
    site_input_max = max(finite(row["model_input_max_error"], label="site input") for row in audits.values())
    site_velocity_max = max(
        finite(row["returned_velocity_max_error"], label="site velocity")
        for row in audits.values()
    )
    if (
        report.get("model_input_future_clamp_max_error") != site_input_max
        or report.get("returned_future_velocity_overwrite_max_error")
        != site_velocity_max
        or report.get("fixed_recipient_noise") is not True
    ):
        raise ValueError(f"{unit['unit_id']}: aggregate site/RNG record differs")
    max_abs = report.get("final_sampler_target_max_abs_errors", {})
    l2 = report.get("final_sampler_target_l2_errors", {})
    if (
        len(max_abs) != EXPECTED_ACTIVE_RESPONSES_PER_STATE
        or len(l2) != EXPECTED_ACTIVE_RESPONSES_PER_STATE
        or set(max_abs) != set(l2)
    ):
        raise ValueError(f"descriptive residual census failed for {unit['unit_id']}")
    for label, value in [*max_abs.items(), *l2.items()]:
        finite(value, label=f"{unit['unit_id']}:{label}:residual")
    expected_residual_labels = {
        label
        for label, spec in spec_by_label.items()
        if spec["kind"] in {"self", "self_repeat", "dose", "dose_midpoint_repeat"}
    }
    if set(max_abs) != expected_residual_labels:
        raise ValueError(f"descriptive residual label set failed for {unit['unit_id']}")
    for label in expected_residual_labels:
        metadata = response_metadata[label]
        if (
            max_abs[label] != metadata.get("research_final_sampler_target_max_abs_error")
            or l2[label] != metadata.get("research_final_sampler_target_l2")
        ):
            raise ValueError(f"descriptive residual copy differs for {unit['unit_id']}:{label}")

    native = {
        int(seed): _action(action, label=f"{unit['unit_id']}:native-{seed}")
        for seed, action in report.get("native_actions", {}).items()
    }
    if set(native) != set(int(seed) for seed in unit["branch_seeds"]):
        raise ValueError(f"native action seed grid failed for {unit['unit_id']}")
    for seed, action in native.items():
        if not np.array_equal(
            action,
            _action(
                response_actions[f"native-{seed}"],
                label=f"{unit['unit_id']}:response-native-{seed}",
            ),
        ):
            raise ValueError(f"native action copies differ for {unit['unit_id']}:{seed}")
    recomputed_axis_l2 = {
        f"{int(pair[0])}:{int(pair[1])}": float(
            np.linalg.norm(
                native[int(pair[1])].astype(np.float64).reshape(-1)
                - native[int(pair[0])].astype(np.float64).reshape(-1)
            )
        )
        for pair in unit["ordered_pairs"]
    }
    if (
        report.get("degenerate_native_action_axis_count") != 0
        or report.get("native_action_pair_l2") != recomputed_axis_l2
        or any(value <= 1e-12 or not math.isfinite(value) for value in recomputed_axis_l2.values())
    ):
        raise ValueError(f"native donor-axis gate failed for {unit['unit_id']}")
    rows = report.get("dose_rows", [])
    if len(rows) != 60:
        raise ValueError(f"dose row census failed for {unit['unit_id']}")
    expected_cells = [
        (int(pair[0]), int(pair[1]), alpha)
        for pair in unit["ordered_pairs"]
        for alpha in ALPHAS
    ]
    actual_cells = [
        (int(row["recipient_seed"]), int(row["donor_seed"]), float(row["alpha"]))
        for row in rows
    ]
    if actual_cells != expected_cells or len(set(actual_cells)) != 60:
        raise ValueError(f"dose cell grid/order failed for {unit['unit_id']}")

    validated: list[dict[str, Any]] = []
    for row in rows:
        recipient_seed = int(row["recipient_seed"])
        donor_seed = int(row["donor_seed"])
        alpha = float(row["alpha"])
        action = _action(
            row["action"],
            label=f"{unit['unit_id']}:{recipient_seed}:{donor_seed}:{alpha}",
        )
        response_label = dose_label(recipient_seed, donor_seed, alpha)
        if not np.array_equal(
            action,
            _action(
                response_actions[response_label],
                label=f"{unit['unit_id']}:{response_label}:response-copy",
            ),
        ):
            raise ValueError(f"{unit['unit_id']}:{response_label} action copies differ")
        recomputed = dose_action_metrics(
            action,
            native[recipient_seed],
            native[donor_seed],
            native,
            donor_seed,
        )
        metadata = response_metadata[response_label]
        server_projection = finite(
            metadata.get("research_action_donor_projection"),
            label=f"{unit['unit_id']}:{response_label}:server projection",
        )
        for key in (
            "native_donor_l2",
            "l2_to_donor",
            "distance_reduction_to_donor",
            "normalized_projection",
            "cosine_alignment",
            "orthogonal_residual_normalized",
        ):
            observed = finite(row.get(key), label=f"{unit['unit_id']}:{key}")
            expected = finite(recomputed[key], label=f"recomputed:{unit['unit_id']}:{key}")
            if observed != expected:
                raise ValueError(f"{unit['unit_id']}:{key} differs from recomputation")
        if (
            metadata.get("research_action_donor_projection_applicable") is not True
            or abs(
                server_projection
                - finite(recomputed["normalized_projection"], label="recomputed projection")
            )
            > 1e-6
            or finite(row.get("server_action_donor_projection"), label="stored projection")
            != server_projection
            or row.get("target_hash") != metadata.get("research_target_hash")
            or finite(
                row.get("final_sampler_target_max_abs_error"), label="stored residual"
            )
            != finite(
                metadata.get("research_final_sampler_target_max_abs_error"),
                label="raw residual",
            )
            or finite(row.get("final_sampler_target_l2"), label="stored residual l2")
            != finite(metadata.get("research_final_sampler_target_l2"), label="raw residual l2")
        ):
            raise ValueError(f"{unit['unit_id']}:{response_label} raw/copy audit differs")
        if (
            int(row.get("nearest_native_seed"))
            != int(recomputed["nearest_native_seed"])
            or bool(row.get("correct_donor_top1"))
            != bool(recomputed["correct_donor_top1"])
            or bool(row.get("nearest_native_exact_tie"))
            != bool(recomputed["nearest_native_exact_tie"])
            or int(row.get("nearest_native_tie_count"))
            != int(recomputed["nearest_native_tie_count"])
            or row.get("nearest_native_tied_seeds")
            != recomputed["nearest_native_tied_seeds"]
            or finite(row.get("nearest_native_top_two_margin"), label="top-two margin")
            != finite(recomputed["nearest_native_top_two_margin"], label="recomputed margin")
            or row.get("distances_to_native_actions")
            != recomputed["distances_to_native_actions"]
        ):
            raise ValueError(f"{unit['unit_id']}: donor identification differs")
        validated.append(row)
    return validated


def state_summary(
    unit: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_pair: dict[tuple[int, int], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair = (int(row["recipient_seed"]), int(row["donor_seed"]))
        by_pair[pair][float(row["alpha"])] = row
    if len(by_pair) != 12 or any(set(grid) != set(ALPHAS) for grid in by_pair.values()):
        raise ValueError(f"{unit['unit_id']}: pair x alpha grid is incomplete")

    distance_slopes: list[float] = []
    projection_slopes: list[float] = []
    monotonic_pairs: list[bool] = []
    alpha_distance_means: dict[float, float] = {}
    alpha_projection_means: dict[float, float] = {}
    alpha_identification: dict[float, float] = {}
    alpha_cosine_means: dict[float, float] = {}
    alpha_orthogonal_means: dict[float, float] = {}
    pair_profiles: dict[str, dict[str, Any]] = {}
    for alpha in ALPHAS:
        alpha_rows = [grid[alpha] for grid in by_pair.values()]
        alpha_distance_means[alpha] = float(
            np.mean([finite(row["distance_reduction_to_donor"], label="distance") for row in alpha_rows])
        )
        alpha_projection_means[alpha] = float(
            np.mean([finite(row["normalized_projection"], label="projection") for row in alpha_rows])
        )
        alpha_identification[alpha] = float(
            np.mean([bool(row["correct_donor_top1"]) for row in alpha_rows])
        )
        alpha_cosine_means[alpha] = float(
            np.mean([finite(row["cosine_alignment"], label="cosine") for row in alpha_rows])
        )
        alpha_orthogonal_means[alpha] = float(
            np.mean(
                [
                    finite(row["orthogonal_residual_normalized"], label="orthogonal")
                    for row in alpha_rows
                ]
            )
        )
    for pair, grid in by_pair.items():
        distance_profile = {
            alpha: finite(grid[alpha]["distance_reduction_to_donor"], label="distance")
            for alpha in ALPHAS
        }
        projection_profile = {
            alpha: finite(grid[alpha]["normalized_projection"], label="projection")
            for alpha in ALPHAS
        }
        l2_profile = {
            alpha: finite(grid[alpha]["l2_to_donor"], label="l2") for alpha in ALPHAS
        }
        distance_slope = ols_slope(ALPHAS, [distance_profile[alpha] for alpha in ALPHAS])
        projection_slope = ols_slope(ALPHAS, [projection_profile[alpha] for alpha in ALPHAS])
        nondecreasing = pair_is_nondecreasing_proximity(l2_profile)
        distance_slopes.append(distance_slope)
        projection_slopes.append(projection_slope)
        monotonic_pairs.append(nondecreasing)
        pair_profiles[f"{pair[0]}:{pair[1]}"] = {
            "recipient_seed": pair[0],
            "donor_seed": pair[1],
            "distance_reduction_slope": distance_slope,
            "projection_slope": projection_slope,
            "distance_reduction_by_alpha": {
                str(alpha): distance_profile[alpha] for alpha in ALPHAS
            },
            "projection_by_alpha": {
                str(alpha): projection_profile[alpha] for alpha in ALPHAS
            },
            "l2_to_donor_by_alpha": {
                str(alpha): l2_profile[alpha] for alpha in ALPHAS
            },
            "correct_donor_top1_by_alpha": {
                str(alpha): bool(grid[alpha]["correct_donor_top1"])
                for alpha in ALPHAS
            },
            "donor_proximity_nondecreasing": nondecreasing,
        }
    distance_adjacent = adjacent_contrasts(alpha_distance_means)
    projection_adjacent = adjacent_contrasts(alpha_projection_means)
    distance_pair_slope_mean = float(np.mean(distance_slopes))
    distance_profile_slope = ols_slope(
        ALPHAS, [alpha_distance_means[alpha] for alpha in ALPHAS]
    )
    projection_pair_slope_mean = float(np.mean(projection_slopes))
    projection_profile_slope = ols_slope(
        ALPHAS, [alpha_projection_means[alpha] for alpha in ALPHAS]
    )
    if abs(distance_pair_slope_mean - distance_profile_slope) > 1e-12:
        raise ValueError(f"{unit['unit_id']}: two equivalent distance slopes differ")
    if abs(projection_pair_slope_mean - projection_profile_slope) > 1e-12:
        raise ValueError(f"{unit['unit_id']}: two equivalent projection slopes differ")
    return {
        "unit_id": unit["unit_id"],
        "task": unit["task"],
        "episode_id": unit["episode_id"],
        "environment_seed": unit["environment_seed"],
        "distance_reduction_slope": distance_pair_slope_mean,
        "distance_reduction_profile_slope": distance_profile_slope,
        "distance_slope_equivalence_abs_error": abs(
            distance_pair_slope_mean - distance_profile_slope
        ),
        "projection_slope": projection_pair_slope_mean,
        "projection_profile_slope": projection_profile_slope,
        "projection_slope_equivalence_abs_error": abs(
            projection_pair_slope_mean - projection_profile_slope
        ),
        "distance_reduction_endpoint_contrast": float(
            alpha_distance_means[1.0] - alpha_distance_means[0.0]
        ),
        "projection_endpoint_contrast": float(
            alpha_projection_means[1.0] - alpha_projection_means[0.0]
        ),
        "distance_reduction_by_alpha": {str(alpha): value for alpha, value in alpha_distance_means.items()},
        "projection_by_alpha": {str(alpha): value for alpha, value in alpha_projection_means.items()},
        "donor_identification_by_alpha": {str(alpha): value for alpha, value in alpha_identification.items()},
        "cosine_alignment_by_alpha": {
            str(alpha): value for alpha, value in alpha_cosine_means.items()
        },
        "orthogonal_residual_by_alpha": {
            str(alpha): value for alpha, value in alpha_orthogonal_means.items()
        },
        "distance_reduction_adjacent_contrasts": distance_adjacent,
        "projection_adjacent_contrasts": projection_adjacent,
        "nondecreasing_pair_fraction": float(np.mean(monotonic_pairs)),
        "ordered_pair_profiles": pair_profiles,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis artifact: {args.output}")
    if args.bootstrap_samples != BOOTSTRAP_SAMPLES or args.bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError("bootstrap count/seed differ from the prospective protocol")
    manifest_hash = sha256(args.manifest)
    if manifest_hash != args.expected_manifest_sha256:
        raise ValueError("manifest SHA differs from the frozen CLI value")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("study_name") != "cosmos3-future-strength-dose-response-v2"
        or manifest.get("admission")
        != "prospective_action_level_future_strength_dose_response"
        or manifest.get("freeze_stage") != "evaluation_ready"
        or manifest.get("launch_authorization")
        != "powered_evaluation_after_independent_go"
    ):
        raise ValueError("analyzer received the wrong study manifest")
    if (
        manifest.get("design", {}).get("alpha_grid") != list(ALPHAS)
        or manifest.get("design", {}).get("request_count_per_state")
        != EXPECTED_CALLS_PER_STATE
        or manifest.get("design", {}).get("released_action_shape") != [32, 8]
        or manifest.get("design", {}).get("released_action_coordinate_count") != 256
        or manifest.get("analysis", {}).get("bootstrap_samples") != BOOTSTRAP_SAMPLES
        or manifest.get("analysis", {}).get("bootstrap_seed") != BOOTSTRAP_SEED
        or manifest.get("runtime", {}).get("protocol_sha256")
        != "7b559902b640610d88f54b1953151249118ad6d162aa6389ac3cbcaca01cb0cc"
        or manifest.get("runtime", {}).get("protocol_v2_amendment_sha256")
        != "a02e3a74d0d5b7f4a9d72401c8f869519acf7a4f808067ee2a2180e68775158f"
    ):
        raise ValueError("manifest design/analysis constants differ from the frozen protocol")
    if sha256(Path(__file__).resolve()) != manifest["runtime"]["analyzer_sha256"]:
        raise ValueError("analyzer source differs from the frozen manifest")
    states = manifest.get("states", [])
    if len(states) != 30:
        raise ValueError("dose analysis requires exactly 30 manifest states")
    if args.input_root.resolve() != Path(
        manifest["runtime"]["evaluation_output_root"]
    ).resolve():
        raise ValueError("analysis input root differs from the frozen evaluation root")
    if args.input_root.is_symlink() or not args.input_root.is_dir():
        raise ValueError("input root must be an existing nonsymlink directory")
    expected_names = {f"{unit['unit_id']}.json" for unit in states}
    paths = sorted(args.input_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("dose output set contains a symlink or non-file")
    actual_names = {path.name for path in paths}
    if actual_names != expected_names or len(paths) != 30:
        raise ValueError(
            f"dose output set differs: missing={sorted(expected_names-actual_names)}, "
            f"extra={sorted(actual_names-expected_names)}"
        )

    state_rows: list[dict[str, Any]] = []
    residual_max_abs: list[float] = []
    residual_l2: list[float] = []
    for unit in states:
        path = args.input_root / f"{unit['unit_id']}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        validated = validate_report(report, unit, manifest, manifest_hash)
        state_rows.append(state_summary(unit, validated))
        residual_max_abs.extend(
            finite(value, label=f"{unit['unit_id']}:max_abs")
            for value in report["final_sampler_target_max_abs_errors"].values()
        )
        residual_l2.extend(
            finite(value, label=f"{unit['unit_id']}:l2")
            for value in report["final_sampler_target_l2_errors"].values()
        )

    draw_table = hierarchical_draw_table(
        state_rows,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    def boot(key: str) -> dict[str, float]:
        return hierarchical_estimate(
            draw_table,
            lambda row: finite(row[key], label=key),
        )

    primary = boot("distance_reduction_slope")
    distance_endpoint = boot("distance_reduction_endpoint_contrast")
    projection_slope = boot("projection_slope")
    projection_endpoint = boot("projection_endpoint_contrast")
    monotonic_fraction = boot("nondecreasing_pair_fraction")
    distance_adjacent: dict[str, dict[str, float]] = {}
    for index, key in enumerate(
        ("0.00_to_0.25", "0.25_to_0.50", "0.50_to_0.75", "0.75_to_1.00")
    ):
        distance_adjacent[key] = hierarchical_estimate(
            draw_table,
            lambda row, key=key: finite(
                row["distance_reduction_adjacent_contrasts"][key], label=key
            ),
        )
    distance_by_alpha: dict[str, dict[str, float]] = {}
    projection_by_alpha: dict[str, dict[str, float]] = {}
    donor_id_by_alpha: dict[str, dict[str, float]] = {}
    cosine_by_alpha: dict[str, dict[str, float]] = {}
    orthogonal_by_alpha: dict[str, dict[str, float]] = {}
    for index, alpha in enumerate(ALPHAS):
        token = str(alpha)
        distance_by_alpha[token] = hierarchical_estimate(
            draw_table,
            lambda row, token=token: finite(row["distance_reduction_by_alpha"][token], label=token),
        )
        projection_by_alpha[token] = hierarchical_estimate(
            draw_table,
            lambda row, token=token: finite(row["projection_by_alpha"][token], label=token),
        )
        donor_id_by_alpha[token] = hierarchical_estimate(
            draw_table,
            lambda row, token=token: finite(row["donor_identification_by_alpha"][token], label=token),
        )
        cosine_by_alpha[token] = hierarchical_estimate(
            draw_table,
            lambda row, token=token: finite(row["cosine_alignment_by_alpha"][token], label=token),
        )
        orthogonal_by_alpha[token] = hierarchical_estimate(
            draw_table,
            lambda row, token=token: finite(row["orthogonal_residual_by_alpha"][token], label=token),
        )

    extractors: dict[str, Callable[[dict[str, Any]], float]] = {
        "distance_reduction_slope": lambda row: finite(
            row["distance_reduction_slope"], label="distance slope"
        ),
        "distance_reduction_endpoint_contrast": lambda row: finite(
            row["distance_reduction_endpoint_contrast"], label="distance endpoint"
        ),
        "projection_slope": lambda row: finite(row["projection_slope"], label="projection slope"),
        "projection_endpoint_contrast": lambda row: finite(
            row["projection_endpoint_contrast"], label="projection endpoint"
        ),
        "nondecreasing_pair_fraction": lambda row: finite(
            row["nondecreasing_pair_fraction"], label="nondecreasing fraction"
        ),
    }
    for alpha in ALPHAS:
        token = str(alpha)
        extractors[f"distance_reduction_alpha_{token}"] = (
            lambda row, token=token: finite(row["distance_reduction_by_alpha"][token], label=token)
        )
        extractors[f"projection_alpha_{token}"] = (
            lambda row, token=token: finite(row["projection_by_alpha"][token], label=token)
        )
        extractors[f"donor_identification_alpha_{token}"] = (
            lambda row, token=token: finite(row["donor_identification_by_alpha"][token], label=token)
        )
        extractors[f"cosine_alpha_{token}"] = (
            lambda row, token=token: finite(row["cosine_alignment_by_alpha"][token], label=token)
        )
        extractors[f"orthogonal_alpha_{token}"] = (
            lambda row, token=token: finite(row["orthogonal_residual_by_alpha"][token], label=token)
        )
    for contrast in ("0.00_to_0.25", "0.25_to_0.50", "0.50_to_0.75", "0.75_to_1.00"):
        extractors[f"distance_adjacent_{contrast}"] = (
            lambda row, contrast=contrast: finite(
                row["distance_reduction_adjacent_contrasts"][contrast], label=contrast
            )
        )
    tasks = sorted({str(row["task"]) for row in state_rows})
    per_task_all_estimands: dict[str, dict[str, float]] = {}
    leave_one_task_out_all_estimands: dict[str, dict[str, float]] = {}
    for name, extractor in extractors.items():
        task_means = {
            task: float(
                np.mean([extractor(row) for row in state_rows if row["task"] == task])
            )
            for task in tasks
        }
        per_task_all_estimands[name] = task_means
        leave_one_task_out_all_estimands[name] = {
            held_out: float(
                np.mean([value for task, value in task_means.items() if task != held_out])
            )
            for held_out in tasks
        }
    per_task_primary = per_task_all_estimands["distance_reduction_slope"]
    leave_one_task_out = leave_one_task_out_all_estimands[
        "distance_reduction_slope"
    ]
    pooled_state_point = float(
        np.mean([finite(row["distance_reduction_slope"], label="slope") for row in state_rows])
    )
    pooled_equal_task_abs_error = abs(pooled_state_point - primary["estimate"])
    residual_summary = {
        "max_abs": {
            "count": len(residual_max_abs),
            "maximum": float(np.max(residual_max_abs)),
            "quantiles": {
                str(q): float(np.quantile(residual_max_abs, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
            "count_gt_0_03": int(np.count_nonzero(np.asarray(residual_max_abs) > 0.03)),
        },
        "l2": {
            "count": len(residual_l2),
            "maximum": float(np.max(residual_l2)),
            "quantiles": {
                str(q): float(np.quantile(residual_l2, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
            "count_gt_0_03": int(np.count_nonzero(np.asarray(residual_l2) > 0.03)),
        },
        "role": "descriptive_only_not_an_admission_or_evidence_criterion",
    }
    all_adjacent_lower_positive = all(
        row["lower"] > 0.0 for row in distance_adjacent.values()
    )
    all_adjacent_point_nonnegative = all(
        row["estimate"] >= 0.0 for row in distance_adjacent.values()
    )
    summary = {
        "status": "complete",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_hash,
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "state_count": len(state_rows),
        "expected_state_count": 30,
        "missing_state_count": 0,
        "extra_state_count": 0,
        "exclusion_count": 0,
        "call_count": len(state_rows) * EXPECTED_CALLS_PER_STATE,
        "shape_valid_response_action_count": len(state_rows)
        * EXPECTED_CALLS_PER_STATE,
        "released_action_coordinate_census": len(state_rows)
        * EXPECTED_CALLS_PER_STATE
        * 256,
        "action_shape": [32, 8],
        "action_coordinate_count": 256,
        "primary_distance_reduction_slope": primary,
        "primary_criterion_pass": primary["lower"] > 0.0,
        "permitted_wording": {
            "positive_linear_dose_trend_in_donor_directed_action_distance_under_imposed_all_call_future_interpolation": (
                primary["lower"] > 0.0
            ),
            "task_weighted_mean_profile_increased_strictly_at_every_adjacent_alpha_step": (
                all_adjacent_lower_positive
            ),
            "numerically_nondecreasing_sample_mean_profile": (
                all_adjacent_point_nonnegative
            ),
            "monotonic": False,
        },
        "distance_reduction_endpoint_contrast": distance_endpoint,
        "distance_reduction_adjacent_contrasts": distance_adjacent,
        "projection_slope": projection_slope,
        "projection_endpoint_contrast": projection_endpoint,
        "distance_reduction_by_alpha": distance_by_alpha,
        "projection_by_alpha": projection_by_alpha,
        "donor_identification_by_alpha": donor_id_by_alpha,
        "cosine_alignment_by_alpha": cosine_by_alpha,
        "orthogonal_residual_by_alpha": orthogonal_by_alpha,
        "nondecreasing_pair_fraction": monotonic_fraction,
        "pooled_30_state_primary_point_sensitivity": pooled_state_point,
        "pooled_equal_task_primary_point_abs_error": pooled_equal_task_abs_error,
        "pooled_equals_equal_task_point_within_1e_15": (
            pooled_equal_task_abs_error <= 1e-15
        ),
        "per_task_primary": per_task_primary,
        "leave_one_task_out_primary": leave_one_task_out,
        "per_task_all_estimands": per_task_all_estimands,
        "leave_one_task_out_all_estimands": leave_one_task_out_all_estimands,
        "final_sampler_target_residual": residual_summary,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "hierarchy": "task -> episode/state",
            "bit_generator": "PCG64",
            "shared_draw_table_reused_for_every_estimand": True,
            "quantile_method": "linear",
        },
        "state_rows": state_rows,
    }
    atomic_json(args.output, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "state_count": len(state_rows),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
