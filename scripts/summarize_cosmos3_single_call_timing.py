#!/usr/bin/env python3
"""Analyze only a complete, frozen Cosmos 3 single-call timing cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from imagined_future.cosmos3_single_call_timing import (
    ACTION_COORDINATE_COUNT,
    ACTION_SHAPE,
    BRANCH_SEEDS,
    ENVIRONMENT_SEEDS,
    EXPECTED_REQUEST_COUNT,
    EXPECTED_STATE_COUNT,
    REQUESTS_PER_STATE,
    RESEARCH_SIGMAS,
    SINGLE_CALL_CONDITIONS,
    TASKS,
    TIMING_CONDITIONS,
    all_finite,
    holm_adjust,
    make_hierarchical_draws,
    ordered_source_cells,
    separation_quartiles,
    state_estimands,
    summarize_state_values,
)


PRIMARY_METRICS = ("matched_retrieval_gain", "matched_distance_gain")
TIMING_METRICS = (
    "complete_source_retrieval",
    "raw_off_diagonal_donor_retrieval",
    "matched_retrieval_gain",
    "matched_distance_gain",
    "distance_reduction",
    "donor_projection",
    "cosine_alignment",
    "orthogonal_residual_normalized",
    "distance_to_donor",
    "minimum_top2_margin",
    "final_sampler_target_max_abs_error_mean",
    "final_sampler_target_max_abs_error_max",
    "final_sampler_target_l2_mean",
    "final_sampler_target_l2_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "bootstrap_values"}


def state_value_map(
    states: Sequence[Mapping[str, Any]],
    accessor,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for state in states:
        value = float(accessor(state))
        if not np.isfinite(value):
            raise ValueError(f"nonfinite state estimand for {state['unit_id']}")
        result[str(state["unit_id"])] = value
    if len(result) != EXPECTED_STATE_COUNT:
        raise ValueError("state estimand map is incomplete or duplicated")
    return result


def leave_one_task_out(
    states: Sequence[Mapping[str, Any]],
    values: Mapping[str, float],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for held_out in TASKS:
        per_task = []
        for task in TASKS:
            if task == held_out:
                continue
            task_values = [
                values[str(state["unit_id"])]
                for state in states
                if str(state["task"]) == task
            ]
            if len(task_values) != 5:
                raise ValueError(f"task {task} does not have five states")
            per_task.append(float(np.mean(task_values)))
        output[held_out] = float(np.mean(per_task))
    return output


def require_exact_numeric_array(
    mapping: Mapping[str, Any], key: str, expected: np.ndarray, label: str
) -> None:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{label}: missing required numeric array {key}")
    actual = np.asarray(mapping[key], dtype=expected.dtype)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
        raise ValueError(f"{label}: invalid required numeric array {key}")
    if not np.array_equal(actual, expected):
        raise ValueError(f"{label}: {key} differs from frozen value")


def require_finite_scalar(
    mapping: Mapping[str, Any], key: str, label: str, *, nonnegative: bool = False
) -> float:
    if key not in mapping or mapping[key] is None or isinstance(mapping[key], bool):
        raise ValueError(f"{label}: missing required numeric scalar {key}")
    value = float(mapping[key])
    if not np.isfinite(value) or (nonnegative and value < 0.0):
        raise ValueError(f"{label}: invalid required numeric scalar {key}")
    return value


def require_exact_action(value: Any, label: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{label}: action is missing")
    try:
        action = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: action is not a rectangular numeric array") from error
    if action.shape != ACTION_SHAPE or action.size != ACTION_COORDINATE_COUNT:
        raise ValueError(
            f"{label}: action shape/count {action.shape}/{action.size} differs from "
            f"{ACTION_SHAPE}/{ACTION_COORDINATE_COUNT}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError(f"{label}: action contains NaN or infinity")
    return action


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


def validate_timing_server_rows(
    report: Mapping[str, Any], manifest: Mapping[str, Any], unit: Mapping[str, Any]
) -> None:
    rows = list(report.get("timing_rows", []))
    expected_rows = [
        (timing, recipient, source, tuple(active))
        for timing, active in TIMING_CONDITIONS
        for recipient, source in ordered_source_cells(BRANCH_SEEDS)
    ]
    actual_rows = [
        (
            str(row.get("timing_condition")),
            int(row.get("recipient_seed", -1)),
            int(row.get("source_seed", -1)),
            tuple(int(index) for index in row.get("active_call_indices", [])),
        )
        for row in rows
    ]
    if actual_rows != expected_rows:
        raise ValueError(f"{unit['unit_id']}: timing rows/order differs from frozen matrix")
    sigmas = np.asarray(RESEARCH_SIGMAS, dtype=np.float32)
    future_frames = np.asarray(manifest["design"]["future_frame_indices"], dtype=np.int64)
    vision_shape = np.asarray(manifest["design"]["vision_shape"], dtype=np.int64)
    vision_count = int(manifest["design"]["vision_coordinate_count"])
    mask_count = int(manifest["design"]["future_mask_coordinate_count"])
    mask_hash = str(manifest["design"]["future_mask_index_hash"])
    native_future_hashes = report.get("native_future_hashes")
    native_path_hashes = report.get("native_path_noise_hashes")
    native_initial_hashes = report.get("native_initial_state_hashes")
    native_actions = report.get("native_actions")
    expected_seed_keys = {str(seed) for seed in BRANCH_SEEDS}
    for name, mapping in (
        ("native_future_hashes", native_future_hashes),
        ("native_path_noise_hashes", native_path_hashes),
        ("native_initial_state_hashes", native_initial_hashes),
    ):
        if (
            not isinstance(mapping, Mapping)
            or set(mapping) != expected_seed_keys
            or any(not isinstance(value, str) or not value for value in mapping.values())
        ):
            raise ValueError(f"{unit['unit_id']}: invalid {name}")
    if not isinstance(native_actions, Mapping) or set(native_actions) != expected_seed_keys:
        raise ValueError(f"{unit['unit_id']}: native action map is incomplete")
    for seed in BRANCH_SEEDS:
        require_exact_action(
            native_actions[str(seed)], f"{unit['unit_id']}:native:{seed}"
        )
    projection_null_count = 0
    finite_off_diagonal_projection_count = 0
    for row, (timing, recipient, source, active) in zip(
        rows, expected_rows, strict=True
    ):
        label = f"{unit['unit_id']}:{timing}:{recipient}:{source}"
        require_exact_action(row.get("action"), label)
        max_abs = require_finite_scalar(
            row, "final_sampler_target_max_abs_error", label, nonnegative=True
        )
        l2_value = require_finite_scalar(
            row, "final_sampler_target_l2", label, nonnegative=True
        )
        server = row.get("server")
        if not isinstance(server, Mapping):
            raise ValueError(f"{label}: missing server audit mapping")
        projection_key = "research_action_donor_projection"
        applicability_key = "research_action_donor_projection_applicable"
        routing_null = (
            server.get("research_attention_interface", {}).get("cache_id") is None
        )
        allowed_nulls = (
            {("research_attention_interface", "cache_id")} if routing_null else set()
        )
        if recipient == source:
            if (
                server.get(projection_key, "missing") is not None
                or server.get(applicability_key) is not False
            ):
                raise ValueError(f"{label}: diagonal projection is not structural null")
            allowed_nulls.add((projection_key,))
            projection_null_count += 1
        else:
            projection = require_finite_scalar(server, projection_key, label)
            if server.get(applicability_key) is not True:
                raise ValueError(f"{label}: off-diagonal projection not applicable")
            if timing == "none" and projection != 0.0:
                raise ValueError(f"{label}: off-diagonal none projection is not zero")
            finite_off_diagonal_projection_count += 1
        if null_paths(server) != allowed_nulls:
            raise ValueError(f"{label}: server null paths differ from frozen schema")
        attention = server.get("research_attention_interface")
        if (
            not isinstance(attention, Mapping)
            or attention.get("instrumented_server") is not False
            or attention.get("intervention_requested") is not False
            or attention.get("mode") != "exclude"
        ):
            raise ValueError(f"{label}: attention routing metadata differs")
        active_array = np.asarray(active, dtype=np.int64)
        inactive_array = np.asarray(
            [index for index in range(4) if index not in active], dtype=np.int64
        )
        active_sigmas = sigmas[active_array]
        for key, expected in (
            ("research_sigmas", sigmas),
            ("research_x0_sigmas", sigmas),
            ("research_requested_active_call_indices", active_array),
            ("research_observed_active_call_indices", active_array),
            ("research_clamped_call_indices", active_array),
            ("research_inactive_call_indices", inactive_array),
            ("research_requested_active_sigmas", active_sigmas),
            ("research_observed_active_sigmas", active_sigmas),
            ("research_future_frame_indices", future_frames),
            ("research_vision_shape", vision_shape),
            (
                "research_model_input_future_clamp_errors",
                np.zeros(active_array.shape, dtype=np.float64),
            ),
            (
                "research_returned_future_velocity_overwrite_errors",
                np.zeros(active_array.shape, dtype=np.float64),
            ),
            ("research_action_input_errors", np.zeros(4, dtype=np.float64)),
            ("research_action_output_errors", np.zeros(4, dtype=np.float64)),
        ):
            require_exact_numeric_array(server, key, expected, label)
        for key, expected in (
            ("research_vision_coordinate_count", vision_count),
            ("research_future_mask_coordinate_count", mask_count),
            ("research_inactive_wrapper_write_count", 0),
        ):
            if server.get(key) != expected:
                raise ValueError(f"{label}: {key} differs from frozen value")
        if server.get("research_future_mask_index_hash") != mask_hash:
            raise ValueError(f"{label}: future-mask index hash differs")
        for key in (
            "research_state_hash",
            "research_parameter_probe_hash",
            "research_recipient_path_noise_hash",
            "research_initial_state_hash",
            "research_recipient_future_hash",
            "research_donor_future_hash",
            "research_target_hash",
        ):
            if not isinstance(server.get(key), str) or not server[key]:
                raise ValueError(f"{label}: missing required hash {key}")
        recipient_key = str(recipient)
        source_key = str(source)
        expected_recipient_id = (
            f"{manifest['manifest_id']}-{unit['unit_id']}-native-{recipient}"
        )
        expected_source_id = (
            f"{manifest['manifest_id']}-{unit['unit_id']}-native-{source}"
        )
        expected_target_hash = (
            native_future_hashes[recipient_key]
            if timing == "none" or recipient == source
            else native_future_hashes[source_key]
        )
        expected_target_id = (
            expected_recipient_id
            if timing == "none" or recipient == source
            else expected_source_id
        )
        expected_target_source = (
            "recipient" if timing == "none" or recipient == source else "donor"
        )
        if (
            server.get("research_recipient_id") != expected_recipient_id
            or server.get("research_donor_id") != expected_source_id
            or server.get("research_recipient_future_hash")
            != native_future_hashes[recipient_key]
            or server.get("research_donor_future_hash") != native_future_hashes[source_key]
            or server.get("research_target_hash") != expected_target_hash
            or server.get("research_target_source") != expected_target_source
            or server.get("research_target_source_record_ids") != [expected_target_id]
            or server.get("research_recipient_path_noise_hash")
            != native_path_hashes[recipient_key]
            or server.get("research_initial_state_hash")
            != native_initial_hashes[recipient_key]
            or server.get("research_state_hash") != report["input_fingerprints"][0]
            or server.get("research_parameter_probe_hash")
            != report["parameter_probe_hashes"][0]
        ):
            raise ValueError(f"{label}: native/source/RNG hash identity differs")
        if require_finite_scalar(
            server, "research_maximum_action_input_error", label
        ) != 0.0:
            raise ValueError(f"{label}: maximum action-input error is nonzero")
        if require_finite_scalar(
            server, "research_maximum_action_output_error", label
        ) != 0.0:
            raise ValueError(f"{label}: maximum action-output error is nonzero")
        if require_finite_scalar(
            server,
            "research_final_sampler_target_max_abs_error",
            label,
            nonnegative=True,
        ) != max_abs:
            raise ValueError(f"{label}: row/server max-abs residual differs")
        if require_finite_scalar(
            server, "research_final_sampler_target_l2", label, nonnegative=True
        ) != l2_value:
            raise ValueError(f"{label}: row/server L2 residual differs")
    if projection_null_count != 24 or finite_off_diagonal_projection_count != 72:
        raise ValueError(
            f"{unit['unit_id']}: stored timing projection census is not 24/72"
        )


def validate_exact_control_map(
    report: Mapping[str, Any], key: str, expected_value: Any, unit_id: str
) -> None:
    mapping = report.get(key)
    expected_keys = {str(seed) for seed in BRANCH_SEEDS}
    if not isinstance(mapping, Mapping) or set(mapping) != expected_keys:
        raise ValueError(f"{unit_id}: {key} does not contain all four branches")
    if any(value != expected_value for value in mapping.values()):
        raise ValueError(f"{unit_id}: {key} failed")


def validate_report(
    report: Mapping[str, Any],
    unit: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_hash: str,
) -> None:
    if report.get("status") != "complete":
        raise ValueError(f"unit {unit['unit_id']} is not complete")
    exact = {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_hash,
        "unit_id": unit["unit_id"],
        "task": unit["task"],
        "environment_seed": unit["environment_seed"],
        "phase": "middle",
        "request_count": REQUESTS_PER_STATE,
        "action_shape": list(ACTION_SHAPE),
        "action_coordinate_count": ACTION_COORDINATE_COUNT,
        "shape_valid_response_action_count": REQUESTS_PER_STATE,
        "action_shape_failure_count": 0,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise ValueError(
                f"{unit['unit_id']}: {key}={report.get(key)!r}, expected {expected!r}"
            )
    if tuple(int(seed) for seed in report.get("branch_seeds", [])) != BRANCH_SEEDS:
        raise ValueError(f"{unit['unit_id']}: branch order mismatch")
    if report.get("request_labels") != manifest["design"]["request_labels"]:
        raise ValueError(f"{unit['unit_id']}: request labels/order mismatch")
    if report.get("runtime_gate", {}).get("passed") is not True:
        raise ValueError(f"{unit['unit_id']}: runtime gate did not pass")
    if report.get("input_fingerprint_count") != 1:
        raise ValueError(f"{unit['unit_id']}: input fingerprint is not singleton")
    if report.get("parameter_probe_hash_count") != 1:
        raise ValueError(f"{unit['unit_id']}: parameter probe is not singleton")
    if report.get("parameter_probe_hashes") != [
        manifest["runtime"]["expected_parameter_probe_hash"]
    ]:
        raise ValueError(f"{unit['unit_id']}: parameter probe differs from manifest")
    if report.get("native_replay_max_action_error") != 0.0:
        raise ValueError(f"{unit['unit_id']}: native replay is not exact")
    if report.get("all_calls_diagonal_replay_max_action_error") != 0.0:
        raise ValueError(f"{unit['unit_id']}: all-calls diagonal replay is not exact")
    if report.get("none_noop_max_action_error") != 0.0:
        raise ValueError(f"{unit['unit_id']}: none action is not an exact no-op")
    if report.get("none_source_invariance_max_action_error") != 0.0:
        raise ValueError(f"{unit['unit_id']}: none output depends on source metadata")
    if report.get("maximum_action_input_error") != 0.0:
        raise ValueError(f"{unit['unit_id']}: action input coordinate changed")
    if report.get("maximum_action_output_error") != 0.0:
        raise ValueError(f"{unit['unit_id']}: action output coordinate changed")
    if report.get("maximum_active_model_input_future_clamp_error") != 0.0:
        raise ValueError(f"{unit['unit_id']}: live model-input clamp mismatch")
    if report.get("maximum_active_returned_future_velocity_error") != 0.0:
        raise ValueError(f"{unit['unit_id']}: live returned-velocity overwrite mismatch")
    if report.get("inactive_wrapper_write_count") != 0:
        raise ValueError(f"{unit['unit_id']}: inactive timing arm wrote through wrapper")
    if report.get("schedule_and_index_gate_exact") is not True:
        raise ValueError(f"{unit['unit_id']}: call schedule/index gate failed")
    if report.get("target_hash_gate_exact") is not True:
        raise ValueError(f"{unit['unit_id']}: target/source hash gate failed")
    if report.get("rng_hash_gate_exact") is not True:
        raise ValueError(f"{unit['unit_id']}: recipient RNG gate failed")
    if report.get("replay_signature_gate_exact") is not True:
        raise ValueError(f"{unit['unit_id']}: replay signatures are not exact")
    for key, expected in (
        ("structural_projection_null_count", 28),
        ("finite_off_diagonal_projection_count", 72),
        ("native_projection_absent_count", 8),
    ):
        if report.get(key) != expected:
            raise ValueError(f"{unit['unit_id']}: {key} differs from frozen census")
    for key in (
        "exact_schedule",
        "exact_active_site_captures",
        "exact_mask",
        "zero_action_coordinate_writes",
        "zero_inactive_wrapper_writes",
        "exact_none_noop",
        "exact_replays",
        "exact_rng_and_target_hashes",
        "all_finite",
        "required_numeric_fields_finite",
        "structural_null_census_exact",
        "exact_projection_applicability_census",
        "exact_action_shape_and_count",
    ):
        if report.get("runtime_gate", {}).get(key) is not True:
            raise ValueError(f"{unit['unit_id']}: runtime gate {key} did not pass")
    for key in (
        "native_replay_action_errors",
        "all_calls_diagonal_replay_action_errors",
        "none_noop_action_errors",
        "none_source_action_errors",
    ):
        validate_exact_control_map(report, key, 0.0, str(unit["unit_id"]))
    for key in (
        "native_replay_signature_exact",
        "all_calls_diagonal_replay_signature_exact",
        "none_source_invariance_exact",
    ):
        validate_exact_control_map(report, key, True, str(unit["unit_id"]))
    validate_timing_server_rows(report, manifest, unit)
    expected_nulls: set[tuple[str, ...]] = set()
    for index, row in enumerate(report["timing_rows"]):
        server = row["server"]
        if server["research_attention_interface"]["cache_id"] is None:
            expected_nulls.add(
                (
                    "timing_rows", str(index), "server",
                    "research_attention_interface", "cache_id",
                )
            )
        if row["recipient_seed"] == row["source_seed"]:
            expected_nulls.add(
                (
                    "timing_rows", str(index), "server",
                    "research_action_donor_projection",
                )
            )
    if null_paths(report) != expected_nulls:
        raise ValueError(f"{unit['unit_id']}: report null paths differ from frozen schema")
    if not all_finite(report):
        raise ValueError(f"{unit['unit_id']}: report contains a nonfinite numeric leaf")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("residual distribution is empty or nonfinite")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def plot_timing(
    path: Path,
    aggregate: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name, _ in TIMING_CONDITIONS]
    labels = ["none", "call 0", "call 1", "call 2", "call 3", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "matched_retrieval_gain", "Matched donor-retrieval gain"),
        (axes[1], "matched_distance_gain", "Matched donor-distance gain"),
    ):
        summaries = [aggregate[name][metric] for name in names]
        means = np.asarray([summary["mean"] for summary in summaries])
        lower = np.maximum(
            0.0, means - np.asarray([summary["ci95_low"] for summary in summaries])
        )
        upper = np.maximum(
            0.0, np.asarray([summary["ci95_high"] for summary in summaries]) - means
        )
        axis.errorbar(
            np.arange(len(names)), means, yerr=np.stack([lower, upper]), fmt="o-",
            color="#1f6f8b", capsize=3, linewidth=1.8, zorder=3,
        )
        for task_index, task in enumerate(TASKS):
            jitter = (task_index - 2.5) * 0.025
            axis.scatter(
                np.arange(len(names)) + jitter,
                [summary["task_means"][task] for summary in summaries],
                s=16, alpha=0.45, color="#d95f02", zorder=2,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel("State/task-balanced effect")
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    actual_manifest_hash = sha256(args.manifest)
    if actual_manifest_hash != args.expected_manifest_sha256:
        raise ValueError(
            f"manifest SHA mismatch: {actual_manifest_hash} != {args.expected_manifest_sha256}"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_model_outcomes":
        raise ValueError("manifest is not a frozen pre-outcome artifact")
    if manifest.get("study_name") != "cosmos3-single-call-timing-v5":
        raise ValueError("manifest is not the single-call timing v5 study")
    if args.bootstrap_samples != 10_000 or args.bootstrap_seed != 20260903:
        raise ValueError("analyzer requires frozen 10000-draw PCG64(20260903) bootstrap")
    if (
        tuple(int(value) for value in manifest.get("design", {}).get("action_shape", []))
        != ACTION_SHAPE
        or int(manifest.get("design", {}).get("action_coordinate_count", -1))
        != ACTION_COORDINATE_COUNT
    ):
        raise ValueError("manifest action shape/count differs from frozen 32x8/256 schema")
    if sha256(Path(__file__).resolve()) != manifest["runtime"]["analyzer_sha256"]:
        raise ValueError("analyzer differs from frozen manifest")
    states = list(manifest.get("states", []))
    if len(states) != EXPECTED_STATE_COUNT:
        raise ValueError(f"expected 30 manifest states, got {len(states)}")
    expected_ids = [str(unit["unit_id"]) for unit in states]
    if len(set(expected_ids)) != EXPECTED_STATE_COUNT:
        raise ValueError("manifest state IDs are duplicated")
    expected_paths = {args.output_root / f"{unit_id}.json" for unit_id in expected_ids}
    actual_paths = set(args.output_root.glob("*.json"))
    missing = sorted(str(path) for path in expected_paths - actual_paths)
    unexpected = sorted(str(path) for path in actual_paths - expected_paths)
    if missing or unexpected:
        raise ValueError(f"incomplete/extra cohort: missing={missing}, unexpected={unexpected}")
    if args.summary_dir.exists():
        raise FileExistsError(f"refusing to overwrite summary directory: {args.summary_dir}")

    reports: list[dict[str, Any]] = []
    derived: list[dict[str, Any]] = []
    for unit, path in zip(states, [args.output_root / f"{unit_id}.json" for unit_id in expected_ids]):
        report = json.loads(path.read_text(encoding="utf-8"))
        validate_report(report, unit, manifest, actual_manifest_hash)
        reports.append(report)
        derived.append(state_estimands(report))
    if sum(int(report["request_count"]) for report in reports) != EXPECTED_REQUEST_COUNT:
        raise ValueError("complete cohort does not contain exactly 3,240 model calls")

    draws = make_hierarchical_draws(
        derived, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    per_task_rows: list[dict[str, Any]] = []
    loto_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for state in derived:
        flat = {
            "unit_id": state["unit_id"],
            "task": state["task"],
            "environment_seed": state["environment_seed"],
            "minimum_native_separation": state["minimum_native_separation"],
        }
        for timing, _ in TIMING_CONDITIONS:
            for metric, value in state["timing"][timing].items():
                flat[f"{timing}__{metric}"] = value
        for metric, value in state["average_single"].items():
            flat[f"average_single__{metric}"] = value
        for metric, value in state["sustained_minus_single"].items():
            flat[f"sustained_minus_single__{metric}"] = value
        state_rows.append(flat)

    def add_summary(group: str, label: str, metric: str, accessor) -> dict[str, Any]:
        values = state_value_map(derived, accessor)
        raw = summarize_state_values(derived, values, draws)
        result = clean_summary(raw)
        aggregate.setdefault(label, {})[metric] = result
        for task, value in result["task_means"].items():
            per_task_rows.append(
                {"group": group, "label": label, "metric": metric, "task": task, "value": value}
            )
        for task, value in leave_one_task_out(derived, values).items():
            loto_rows.append(
                {"group": group, "label": label, "metric": metric, "held_out_task": task, "value": value}
            )
        return raw

    raw_bootstrap: dict[tuple[str, str], np.ndarray] = {}
    for timing, _ in TIMING_CONDITIONS:
        for metric in TIMING_METRICS:
            raw = add_summary(
                "timing", timing, metric,
                lambda state, timing=timing, metric=metric: state["timing"][timing][metric],
            )
            raw_bootstrap[(timing, metric)] = raw["bootstrap_values"]
    for label in ("average_single", "sustained_minus_single"):
        for metric in PRIMARY_METRICS:
            raw = add_summary(
                "contrast", label, metric,
                lambda state, label=label, metric=metric: state[label][metric],
            )
            raw_bootstrap[(label, metric)] = raw["bootstrap_values"]

    call_local: dict[str, Any] = {}
    composite_raw: dict[str, float] = {}
    for timing in SINGLE_CALL_CONDITIONS:
        components = {
            metric: aggregate[timing][metric]["one_sided_null_centered_p"]
            for metric in PRIMARY_METRICS
        }
        composite = float(max(components.values()))
        composite_raw[timing] = composite
        call_local[timing] = {
            "component_p_values": components,
            "conjunctive_raw_p": composite,
        }
    holm = holm_adjust(composite_raw)
    for timing in SINGLE_CALL_CONDITIONS:
        call_local[timing]["holm"] = holm[timing]

    primary_pass = all(
        aggregate["average_single"][metric]["ci95_low"] > 0.0
        for metric in PRIMARY_METRICS
    )
    sustained_pass = all(
        aggregate["sustained_minus_single"][metric]["ci95_low"] > 0.0
        for metric in PRIMARY_METRICS
    )
    all_pair_rows = [row for state in derived for row in state["pair_rows"]]
    quartiles = separation_quartiles(all_pair_rows)
    boundaries = np.asarray(quartiles["boundaries"], dtype=np.float64)
    quartile_rows: list[dict[str, Any]] = []
    for row in all_pair_rows:
        q = int(np.searchsorted(boundaries, float(row["native_separation"]), side="right")) + 1
        quartile_rows.append({**row, "quartile": f"q{q}"})
    quartile_summary: list[dict[str, Any]] = []
    for timing, _ in TIMING_CONDITIONS:
        for quartile in ("q1", "q2", "q3", "q4"):
            selected = [
                row for row in quartile_rows
                if row["timing_condition"] == timing and row["quartile"] == quartile
            ]
            retrieval_values = [row["retrieval_gain"] for row in selected]
            distance_values = [row["distance_gain"] for row in selected]
            quartile_summary.append(
                {
                    "timing_condition": timing,
                    "quartile": quartile,
                    "pair_count": len(selected),
                    "state_count": len({row["unit_id"] for row in selected}),
                    "task_count": len({row["task"] for row in selected}),
                    "retrieval_gain_mean": (
                        float(np.mean(retrieval_values)) if retrieval_values else None
                    ),
                    "distance_gain_mean": (
                        float(np.mean(distance_values)) if distance_values else None
                    ),
                }
            )

    residual_rows: list[dict[str, Any]] = []
    for report in reports:
        for row in report["timing_rows"]:
            residual_rows.append(
                {
                    "unit_id": report["unit_id"],
                    "task": report["task"],
                    "environment_seed": report["environment_seed"],
                    "timing_condition": row["timing_condition"],
                    "recipient_seed": row["recipient_seed"],
                    "source_seed": row["source_seed"],
                    "final_sampler_target_max_abs_error": row[
                        "final_sampler_target_max_abs_error"
                    ],
                    "final_sampler_target_l2": row["final_sampler_target_l2"],
                }
            )
    if len(residual_rows) != EXPECTED_STATE_COUNT * 96:
        raise ValueError("final-target residual table is incomplete")
    residual_distributions = {
        timing: {
            metric: distribution_summary(
                [
                    float(row[metric])
                    for row in residual_rows
                    if row["timing_condition"] == timing
                ]
            )
            for metric in (
                "final_sampler_target_max_abs_error",
                "final_sampler_target_l2",
            )
        }
        for timing, _ in TIMING_CONDITIONS
    }

    controls = {
        "native_replay_max_action_error": max(report["native_replay_max_action_error"] for report in reports),
        "all_calls_diagonal_replay_max_action_error": max(report["all_calls_diagonal_replay_max_action_error"] for report in reports),
        "none_noop_max_action_error": max(report["none_noop_max_action_error"] for report in reports),
        "none_source_invariance_max_action_error": max(report["none_source_invariance_max_action_error"] for report in reports),
        "maximum_action_input_error": max(report["maximum_action_input_error"] for report in reports),
        "maximum_action_output_error": max(report["maximum_action_output_error"] for report in reports),
        "maximum_active_model_input_future_clamp_error": max(report["maximum_active_model_input_future_clamp_error"] for report in reports),
        "maximum_active_returned_future_velocity_error": max(report["maximum_active_returned_future_velocity_error"] for report in reports),
        "inactive_wrapper_write_count": sum(report["inactive_wrapper_write_count"] for report in reports),
        "minimum_native_separation": min(state["minimum_native_separation"] for state in derived),
        "schedule_and_index_gate_exact": all(report["schedule_and_index_gate_exact"] for report in reports),
        "target_hash_gate_exact": all(report["target_hash_gate_exact"] for report in reports),
        "rng_hash_gate_exact": all(report["rng_hash_gate_exact"] for report in reports),
        "replay_signature_gate_exact": all(report["replay_signature_gate_exact"] for report in reports),
        "structural_projection_null_count": sum(
            report["structural_projection_null_count"] for report in reports
        ),
        "finite_off_diagonal_projection_count": sum(
            report["finite_off_diagonal_projection_count"] for report in reports
        ),
        "native_projection_absent_count": sum(
            report["native_projection_absent_count"] for report in reports
        ),
        "shape_valid_response_action_count": sum(
            report["shape_valid_response_action_count"] for report in reports
        ),
        "action_shape_failure_count": sum(
            report["action_shape_failure_count"] for report in reports
        ),
    }
    runtime_pass = all(
        report.get("runtime_gate", {}).get("passed") is True for report in reports
    )
    result = {
        "status": "complete",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": actual_manifest_hash,
        "audit": {
            "state_count": len(reports),
            "request_count": sum(report["request_count"] for report in reports),
            "expected_state_count": EXPECTED_STATE_COUNT,
            "expected_request_count": EXPECTED_REQUEST_COUNT,
            "missing_state_count": 0,
            "unexpected_state_count": 0,
            "required_numeric_nonfinite_count": 0,
            "structural_projection_null_count": 840,
            "finite_off_diagonal_projection_count": 2160,
            "native_projection_absent_count": 240,
            "action_shape": list(ACTION_SHAPE),
            "action_coordinate_count": ACTION_COORDINATE_COUNT,
            "shape_valid_response_action_count": 3240,
            "action_shape_failure_count": 0,
            "degenerate_axis_count": 0,
        },
        "analysis": {
            "independent_unit": "state",
            "top_level_clusters": "six equal-weight tasks",
            "bootstrap": "task-to-state hierarchical",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "research_sigmas": [float(value) for value in RESEARCH_SIGMAS],
        },
        "timing": aggregate,
        "call_local_tests": call_local,
        "quartiles": quartiles,
        "quartile_timing_summaries": quartile_summary,
        "final_sampler_target_residual_distributions": residual_distributions,
        "controls": controls,
        "evidence_gates": {
            "runtime_and_completeness": runtime_pass,
            "average_single_donor_specific_effect": primary_pass,
            "sustained_strength": sustained_pass,
            "timing_local_holm_rejections": {
                timing: bool(call_local[timing]["holm"]["rejected"])
                for timing in SINGLE_CALL_CONDITIONS
            },
        },
        "claim_boundary": (
            "imposed action-space timing/strength audit; not natural mediation, physical "
            "success, semantic planning, necessity, or an isolated local direct effect"
        ),
    }
    if (
        controls["structural_projection_null_count"] != 840
        or controls["finite_off_diagonal_projection_count"] != 2160
        or controls["native_projection_absent_count"] != 240
        or controls["shape_valid_response_action_count"] != 3240
        or controls["action_shape_failure_count"] != 0
        or null_paths(result)
        or not all_finite(result)
    ):
        raise ValueError("cohort projection census or result finiteness gate failed")

    temp_parent = args.summary_dir.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.summary_dir.name}.", dir=temp_parent))
    try:
        write_csv(temporary / "cosmos3_single_call_timing_states.csv", state_rows)
        write_csv(temporary / "cosmos3_single_call_timing_pairs.csv", quartile_rows)
        write_csv(temporary / "cosmos3_single_call_timing_quartiles.csv", quartile_summary)
        write_csv(temporary / "cosmos3_single_call_timing_residuals.csv", residual_rows)
        write_csv(temporary / "cosmos3_single_call_timing_per_task.csv", per_task_rows)
        write_csv(temporary / "cosmos3_single_call_timing_leave_one_task_out.csv", loto_rows)
        aggregate_rows = []
        for label, metrics in aggregate.items():
            for metric, summary in metrics.items():
                aggregate_rows.append(
                    {"label": label, "metric": metric, **{k: v for k, v in summary.items() if k != "task_means"}}
                )
        write_csv(temporary / "cosmos3_single_call_timing_aggregate.csv", aggregate_rows)
        (temporary / "cosmos3_single_call_timing_results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        latex_lines = [
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Timing & Retrieval gain & Distance gain \\",
            r"\midrule",
        ]
        for timing, _ in TIMING_CONDITIONS:
            retrieval = aggregate[timing]["matched_retrieval_gain"]
            distance = aggregate[timing]["matched_distance_gain"]
            latex_timing = timing.replace("_", "\\_")
            latex_lines.append(
                f"{latex_timing} & "
                f"{retrieval['mean']:.3f} [{retrieval['ci95_low']:.3f}, {retrieval['ci95_high']:.3f}] & "
                f"{distance['mean']:.3f} [{distance['ci95_low']:.3f}, {distance['ci95_high']:.3f}] \\\\"  # noqa: E501
            )
        latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
        (temporary / "cosmos3_single_call_timing_results.tex").write_text(
            "\n".join(latex_lines) + "\n", encoding="utf-8"
        )
        plot_timing(temporary / "cosmos3_single_call_timing_summary.png", aggregate)
        os.replace(temporary, args.summary_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest_id": manifest["manifest_id"],
                "summary_dir": str(args.summary_dir),
                "primary_pass": primary_pass,
                "sustained_pass": sustained_pass,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
