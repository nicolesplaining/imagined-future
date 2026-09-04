#!/usr/bin/env python3
"""Strict complete-cohort analysis for archival Cosmos 3 action steering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from imagined_future.cosmos3_archival import atomic_json, sha256


BOOTSTRAP_SAMPLES = 10_000
ANALYSIS_SEED = 20260903
INTERVENTION_SITE_ERROR_TOLERANCE = 1e-7
EXPECTED_DENOISING_CALLS = 4
EXPECTED_FUTURE_FRAME_INDICES = list(range(1, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def metric_or_none(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return finite(value, label=label)


def describe_final_sampler_residuals(
    values: Iterable[Any], *, label: str = "final_sampler_target_residual"
) -> dict[str, Any]:
    """Summarize final sampler-state drift without using it as an admission gate."""

    array = np.asarray(
        [finite(value, label=label) for value in values],
        dtype=np.float64,
    )
    if len(array) == 0:
        raise ValueError("final sampler residual summary requires at least one value")
    return {
        "count": int(len(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "quantiles": {
            str(q): float(np.quantile(array, q))
            for q in (0.5, 0.9, 0.95, 0.99)
        },
        "count_gt_0_03": int(np.count_nonzero(array > 0.03)),
    }


def torch_bool_tensor_digest(value: np.ndarray) -> str:
    flat = np.ascontiguousarray(np.asarray(value, dtype=np.bool_).reshape(-1))
    digest = hashlib.sha256()
    digest.update(b"torch.bool")
    digest.update(np.asarray(flat.shape, dtype=np.int64).tobytes())
    digest.update(flat.view(np.uint8).tobytes())
    return digest.hexdigest()


def hierarchical_task_episode_state_bootstrap(
    rows: list[dict[str, Any]],
    value_key: str,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = ANALYSIS_SEED,
) -> dict[str, Any]:
    nested: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        nested[str(row["task"])][str(row["episode_id"])].append(
            finite(value, label=value_key)
        )
    eligible_states = sum(
        row.get(value_key) is not None for row in rows
    )
    null_states = len(rows) - eligible_states
    if not nested:
        return {
            "mean": None,
            "ci95": None,
            "tasks": 0,
            "episodes": 0,
            "states": 0,
            "input_states": len(rows),
            "eligible_states": eligible_states,
            "null_states": null_states,
        }
    tasks = sorted(nested)
    point_task_means = [
        np.mean([np.mean(values) for values in nested[task].values()]) for task in tasks
    ]
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        sampled_task_means = []
        for task_index in generator.integers(0, len(tasks), size=len(tasks)):
            task = tasks[int(task_index)]
            episodes = sorted(nested[task])
            sampled_episode_means = []
            for episode_index in generator.integers(0, len(episodes), size=len(episodes)):
                values = np.asarray(nested[task][episodes[int(episode_index)]], dtype=np.float64)
                sampled_states = generator.integers(0, len(values), size=len(values))
                sampled_episode_means.append(float(values[sampled_states].mean()))
            sampled_task_means.append(float(np.mean(sampled_episode_means)))
        draws[draw] = float(np.mean(sampled_task_means))
    return {
        "mean": float(np.mean(point_task_means)),
        "state_weighted_mean": float(
            np.mean([float(row[value_key]) for row in rows if row.get(value_key) is not None])
        ),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "tasks": len(tasks),
        "episodes": sum(len(episodes) for episodes in nested.values()),
        "states": sum(len(values) for episodes in nested.values() for values in episodes.values()),
        "input_states": len(rows),
        "eligible_states": eligible_states,
        "null_states": null_states,
        "samples": samples,
        "seed": seed,
    }


def per_task(rows: list[dict[str, Any]], value_key: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        values = [
            float(row[value_key])
            for row in rows
            if row["task"] == task and row.get(value_key) is not None
        ]
        output[task] = float(np.mean(values))
    return output


def leave_one_task_out(rows: list[dict[str, Any]], value_key: str) -> dict[str, float]:
    tasks = sorted({str(row["task"]) for row in rows})
    result: dict[str, float] = {}
    for held_out in tasks:
        means = per_task([row for row in rows if row["task"] != held_out], value_key)
        result[held_out] = float(np.mean(list(means.values())))
    return result


def source_permutation_test(
    reports: list[dict[str, Any]], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = ANALYSIS_SEED
) -> dict[str, Any]:
    observed_state: list[float] = []
    state_predictions: list[dict[int, list[int]]] = []
    state_tasks: list[str] = []
    for report in reports:
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row in report["future_source_retrieval_rows"]:
            grouped[int(row["recipient_seed"])].append(
                (int(row["future_source_seed"]), int(row["nearest_native_seed"]))
            )
        predictions: dict[int, list[int]] = {}
        correct = []
        for recipient, pairs in sorted(grouped.items()):
            pairs.sort()
            sources = [source for source, _nearest in pairs]
            nearest = [nearest for _source, nearest in pairs]
            if len(sources) != 4 or len(set(sources)) != 4:
                raise ValueError("retrieval permutation requires four unique sources per recipient")
            predictions[recipient] = nearest
            correct.extend(int(source == prediction) for source, prediction in zip(sources, nearest))
        observed_state.append(float(np.mean(correct)))
        state_predictions.append(predictions)
        state_tasks.append(str(report["task"]))
    tasks = sorted(set(state_tasks))
    observed_equal_task = float(
        np.mean(
            [
                np.mean([value for value, task_value in zip(observed_state, state_tasks) if task_value == task])
                for task in tasks
            ]
        )
    )
    generator = np.random.default_rng(seed)
    null = np.zeros(samples, dtype=np.float64)
    sources = np.asarray([211, 223, 227, 229], dtype=np.int64)
    for draw in range(samples):
        state_values = []
        for predictions in state_predictions:
            matches = []
            for recipient in sorted(predictions):
                labels = generator.permutation(sources)
                matches.extend(int(label == prediction) for label, prediction in zip(labels, predictions[recipient]))
            state_values.append(float(np.mean(matches)))
        null[draw] = float(
            np.mean(
                [
                    np.mean([value for value, task_value in zip(state_values, state_tasks) if task_value == task])
                    for task in tasks
                ]
            )
        )
    return {
        "observed_equal_task_mean": observed_equal_task,
        "chance": 0.25,
        "p_greater_monte_carlo": float((1 + np.sum(null >= observed_equal_task)) / (samples + 1)),
        "null_mean": float(null.mean()),
        "null_ci95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "samples": samples,
        "seed": seed,
        "permutation_unit": "four future-source labels independently within each recipient and state",
    }


def donor_pair_separation_quartiles(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stratify off-diagonal donor arms before within-state aggregation."""

    field_map = {
        "donor_top1": "correct_donor_top1",
        "wrong_donor_top1": "wrong_donor_top1",
        "distance_reduction": "distance_reduction_to_target",
        "cosine_alignment": "cosine_alignment",
        "orthogonal_residual_normalized": "orthogonal_residual_normalized",
        "normalized_projection": "normalized_projection",
        "native_separation": "native_target_l2",
    }
    arms: list[dict[str, Any]] = []
    for report in reports:
        for donor in report["donor_rows"]:
            separation = finite(donor["native_target_l2"], label="native_target_l2")
            arm = {
                "unit_id": report["unit_id"],
                "task": report["task"],
                "episode_id": report["episode_id"],
                "native_separation": separation,
            }
            for output_field, source_field in field_map.items():
                value = donor.get(source_field)
                if output_field in {"donor_top1", "wrong_donor_top1"}:
                    if not isinstance(value, bool):
                        raise ValueError(f"{source_field} is not Boolean")
                    arm[output_field] = float(value)
                else:
                    arm[output_field] = metric_or_none(value, label=source_field)
            arms.append(arm)
    if len(arms) != len(reports) * 12:
        raise ValueError("pair-level quartiles require exactly 12 donor arms per state")
    separations = np.asarray([arm["native_separation"] for arm in arms], dtype=np.float64)
    boundaries = np.quantile(separations, [0.25, 0.5, 0.75]).tolist()
    for arm in arms:
        arm["quartile"] = int(
            np.searchsorted(boundaries, arm["native_separation"], side="right") + 1
        )

    metrics = list(field_map)
    output: dict[str, Any] = {
        "definition": (
            "cohort-global quartiles of all off-diagonal native action separations; "
            "donor arms are assigned first, then averaged within state and bootstrapped "
            "task -> archived episode -> state"
        ),
        "boundaries": boundaries,
        "total_arms": len(arms),
        "quartiles": {},
    }
    for quartile in range(1, 5):
        selected = [arm for arm in arms if arm["quartile"] == quartile]
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for arm in selected:
            buckets[str(arm["unit_id"])].append(arm)
        within_state: list[dict[str, Any]] = []
        for unit_id, state_arms in sorted(buckets.items()):
            row: dict[str, Any] = {
                "unit_id": unit_id,
                "task": state_arms[0]["task"],
                "episode_id": state_arms[0]["episode_id"],
                "arm_count": len(state_arms),
            }
            for metric in metrics:
                values = [arm[metric] for arm in state_arms if arm[metric] is not None]
                row[metric] = float(np.mean(values)) if values else None
                row[f"{metric}_valid_arm_count"] = len(values)
                row[f"{metric}_null_arm_count"] = len(state_arms) - len(values)
            within_state.append(row)
        output["quartiles"][str(quartile)] = {
            "arm_count": len(selected),
            "state_count": len(within_state),
            "episode_count": len({row["episode_id"] for row in within_state}),
            "task_count": len({row["task"] for row in within_state}),
            "metric_valid_arm_counts": {
                metric: sum(
                    int(row[f"{metric}_valid_arm_count"]) for row in within_state
                )
                for metric in metrics
            },
            "metric_null_arm_counts": {
                metric: sum(
                    int(row[f"{metric}_null_arm_count"]) for row in within_state
                )
                for metric in metrics
            },
            "aggregate": {
                metric: hierarchical_task_episode_state_bootstrap(within_state, metric)
                for metric in metrics
            },
        }
    if sum(cell["arm_count"] for cell in output["quartiles"].values()) != len(arms):
        raise AssertionError("pair-level quartiles did not partition all donor arms")
    return output


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if path.exists():
        raise FileExistsError(path)
    fields = list(materialized[0])
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_report(report: dict[str, Any], unit: dict[str, Any], manifest: dict[str, Any], manifest_hash: str) -> None:
    source = str(unit["unit_id"])
    expected_identity = {
        "status": "complete",
        "admission": manifest["admission"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_hash,
        "unit_id": unit["unit_id"],
        "task": unit["task"],
        "environment_seed": unit["environment_seed"],
        "episode_id": unit["episode_id"],
        "phase": unit["phase"],
        "branch_step": unit["branch_step"],
    }
    for field, expected in expected_identity.items():
        if report.get(field) != expected:
            raise ValueError(f"{source}: {field} differs from frozen manifest")
    if report.get("scope") != manifest["scope"]:
        raise ValueError(f"{source}: scope differs from manifest")
    seeds = [int(seed) for seed in unit["branch_seeds"]]
    if report.get("branch_seeds") != seeds:
        raise ValueError(f"{source}: branch seeds differ")
    if report.get("input_fingerprint_count") != 1:
        raise ValueError(f"{source}: transformed-input fingerprint count is not one")
    if report.get("request_count") != 56:
        raise ValueError(f"{source}: request count differs from the frozen 56-call design")
    expected_probe = manifest["runtime"]["expected_parameter_probe_hash"]
    if (
        report.get("parameter_probe_hash_count") != 1
        or report.get("parameter_probe_hashes") != [expected_probe]
        or report.get("expected_parameter_probe_hash") != expected_probe
    ):
        raise ValueError(f"{source}: checkpoint parameter probe differs from manifest")
    tolerance = finite(
        report.get("intervention_site_error_tolerance"),
        label="intervention_site_error_tolerance",
    )
    manifest_tolerance = finite(
        manifest["runtime"]["intervention_site_error_tolerance"],
        label="manifest_intervention_site_error_tolerance",
    )
    if tolerance != INTERVENTION_SITE_ERROR_TOLERANCE or tolerance != manifest_tolerance:
        raise ValueError(f"{source}: intervention-site tolerance changed")

    site_audits = report.get("intervention_site_audits", {})
    if len(site_audits) != 48:
        raise ValueError(f"{source}: expected exactly 48 intervention-site audits")
    none_site_count = 0
    active_response_count = 0
    active_site_count = 0
    mode_counts: dict[str, int] = defaultdict(int)
    input_site_errors: list[float] = []
    velocity_site_errors: list[float] = []
    for label, audit in site_audits.items():
        mode = str(audit.get("mode"))
        mode_counts[mode] += 1
        is_none = mode == "none"
        expected_active_indices = [] if is_none else list(range(EXPECTED_DENOISING_CALLS))
        expected_inactive_indices = [
            index
            for index in range(EXPECTED_DENOISING_CALLS)
            if index not in expected_active_indices
        ]
        if is_none:
            none_site_count += 1
        else:
            active_response_count += 1
        sigmas = [finite(value, label=f"{label}.sigma") for value in audit.get("sigmas", [])]
        if len(sigmas) != EXPECTED_DENOISING_CALLS:
            raise ValueError(f"{source}: {label} sigma cardinality differs")
        for key in (
            "requested_active_call_indices",
            "observed_active_call_indices",
            "clamped_call_indices",
        ):
            if audit.get(key) != expected_active_indices:
                raise ValueError(f"{source}: {label} {key} differs")
        if audit.get("inactive_call_indices") != expected_inactive_indices:
            raise ValueError(f"{source}: {label} inactive-call telemetry differs")
        expected_sigmas = [sigmas[index] for index in expected_active_indices]
        if (
            audit.get("requested_active_sigmas") != expected_sigmas
            or audit.get("observed_active_sigmas") != expected_sigmas
        ):
            raise ValueError(f"{source}: {label} active sigma audit differs")
        if audit.get("future_frame_indices") != EXPECTED_FUTURE_FRAME_INDICES:
            raise ValueError(f"{source}: {label} future-frame audit differs")
        vision_shape = tuple(int(value) for value in audit.get("vision_shape", []))
        if len(vision_shape) not in (4, 5):
            raise ValueError(f"{source}: {label} vision shape differs")
        temporal_axis = len(vision_shape) - 3
        expected_mask = np.zeros(vision_shape, dtype=np.bool_)
        mask_index = [slice(None)] * len(vision_shape)
        mask_index[temporal_axis] = EXPECTED_FUTURE_FRAME_INDICES
        expected_mask[tuple(mask_index)] = True
        vision_count = int(audit.get("vision_coordinate_count", -1))
        mask_count = int(audit.get("mask_coordinate_count", -1))
        if (
            vision_count <= 0
            or int(np.prod(vision_shape)) != vision_count
            or vision_count % 9 != 0
            or mask_count != (vision_count // 9) * 8
            or audit.get("future_mask_index_hash")
            != torch_bool_tensor_digest(expected_mask)
        ):
            raise ValueError(f"{source}: {label} mask cardinality differs")
        per_input = [
            finite(value, label=f"{label}.model_input_error")
            for value in audit.get("model_input_future_clamp_errors", [])
        ]
        per_velocity = [
            finite(value, label=f"{label}.returned_velocity_error")
            for value in audit.get("returned_future_velocity_overwrite_errors", [])
        ]
        expected_sites = len(expected_active_indices)
        if (
            int(audit.get("active_site_count", -1)) != expected_sites
            or len(per_input) != expected_sites
            or len(per_velocity) != expected_sites
            or any(value > tolerance for value in per_input)
            or any(value > tolerance for value in per_velocity)
        ):
            raise ValueError(f"{source}: {label} active intervention-site gate failed")
        if (
            finite(audit.get("model_input_max_error"), label="model_input_max_error")
            != (max(per_input) if per_input else 0.0)
            or finite(
                audit.get("returned_velocity_max_error"),
                label="returned_velocity_max_error",
            )
            != (max(per_velocity) if per_velocity else 0.0)
        ):
            raise ValueError(f"{source}: {label} site-error summary differs")
        if (
            finite(audit.get("maximum_action_input_error"), label="action_input_error")
            != 0.0
            or finite(
                audit.get("maximum_action_output_error"), label="action_output_error"
            )
            != 0.0
        ):
            raise ValueError(f"{source}: {label} wrote action coordinates")
        per_call_action_input = [
            finite(value, label=f"{label}.action_input_error")
            for value in audit.get("action_input_errors", [])
        ]
        per_call_action_output = [
            finite(value, label=f"{label}.action_output_error")
            for value in audit.get("action_output_errors", [])
        ]
        if (
            len(per_call_action_input) != EXPECTED_DENOISING_CALLS
            or len(per_call_action_output) != EXPECTED_DENOISING_CALLS
            or any(value != 0.0 for value in per_call_action_input)
            or any(value != 0.0 for value in per_call_action_output)
            or audit.get("inactive_wrapper_write_count") != 0
        ):
            raise ValueError(f"{source}: {label} per-call nonwrite telemetry differs")
        for key in (
            "target_hash",
            "recipient_future_hash",
            "donor_future_hash",
            "recipient_path_noise_hash",
            "initial_state_hash",
        ):
            if not isinstance(audit.get(key), str) or len(audit[key]) != 64:
                raise ValueError(f"{source}: {label} lacks {key} provenance")
        expected_source = (
            "recipient"
            if mode in {"none", "self"}
            else "donor"
            if mode == "donor"
            else "gaussian_geometry"
            if mode == "gaussian"
            else None
        )
        if expected_source is None or audit.get("target_source") != expected_source:
            raise ValueError(f"{source}: {label} target-source provenance differs")
        source_ids = audit.get("target_source_record_ids")
        expected_source_id_count = 2 if expected_source == "gaussian_geometry" else 1
        if (
            not isinstance(source_ids, list)
            or len(source_ids) != expected_source_id_count
            or any(not isinstance(value, str) or not value for value in source_ids)
        ):
            raise ValueError(f"{source}: {label} target-source IDs differ")
        if expected_source == "recipient" and audit["target_hash"] != audit["recipient_future_hash"]:
            raise ValueError(f"{source}: {label} recipient target hash differs")
        if expected_source == "donor" and audit["target_hash"] != audit["donor_future_hash"]:
            raise ValueError(f"{source}: {label} donor target hash differs")
        finite(
            audit.get("final_sampler_target_max_abs_error"),
            label=f"{label}.final_sampler_target_max_abs_error",
        )
        finite(
            audit.get("final_sampler_target_l2"),
            label=f"{label}.final_sampler_target_l2",
        )
        active_site_count += expected_sites
        input_site_errors.extend(per_input)
        velocity_site_errors.extend(per_velocity)
    if (
        none_site_count != 4
        or active_response_count != 44
        or dict(mode_counts)
        != {"none": 4, "self": 8, "donor": 24, "gaussian": 12}
        or active_site_count != 176
        or len(input_site_errors) != 176
        or len(velocity_site_errors) != 176
        or report.get("active_intervention_response_count") != 44
        or report.get("active_intervention_site_count") != 176
        or finite(
            report.get("model_input_future_clamp_max_error"),
            label="model_input_future_clamp_max_error",
        )
        != max(input_site_errors)
        or finite(
            report.get("returned_future_velocity_overwrite_max_error"),
            label="returned_future_velocity_overwrite_max_error",
        )
        != max(velocity_site_errors)
    ):
        raise ValueError(f"{source}: aggregate intervention-site audit differs")

    max_abs_residuals = {
        str(label): finite(value, label="final_sampler_target_max_abs_error")
        for label, value in report.get(
            "final_sampler_target_max_abs_errors", {}
        ).items()
    }
    l2_residuals = {
        str(label): finite(value, label="final_sampler_target_l2")
        for label, value in report.get("final_sampler_target_l2_errors", {}).items()
    }
    if (
        len(max_abs_residuals) != 44
        or len(l2_residuals) != 44
        or set(max_abs_residuals) != set(l2_residuals)
    ):
        raise ValueError(
            f"{source}: expected 44 paired finite descriptive final residuals"
        )
    residual_summary = report.get("final_sampler_target_residual_summary", {})
    expected_residual_summary = {
        "max_abs": describe_final_sampler_residuals(
            max_abs_residuals.values(), label="final_sampler_target_max_abs_error"
        ),
        "l2": describe_final_sampler_residuals(
            l2_residuals.values(), label="final_sampler_target_l2"
        ),
    }
    if residual_summary != expected_residual_summary:
        raise ValueError(f"{source}: descriptive final-residual summary differs")
    if set(report.get("native_repeat_action_maximum_error", {}).values()) != {0.0}:
        raise ValueError(f"{source}: native repeat was not exact")
    if set(report.get("native_future_replay_exact", {}).values()) != {True}:
        raise ValueError(f"{source}: native future replay was not exact")
    if set(
        report.get("native_deterministic_metadata_replay_exact", {}).values()
    ) != {True}:
        raise ValueError(f"{source}: native deterministic metadata replay was not exact")
    if not isinstance(report.get("native_future_hashes_distinct"), bool):
        raise ValueError(f"{source}: native distinctness audit is absent")
    coordinate_errors = report.get("action_coordinate_errors", {})
    if len(coordinate_errors) != 48:
        raise ValueError(f"{source}: expected exactly 48 action-coordinate audits")
    if any(
        max(
            finite(error, label="action_coordinate_error")
            for error in pair.values()
        )
        != 0.0
        for pair in coordinate_errors.values()
    ):
        raise ValueError(f"{source}: an arm wrote action coordinates")

    none_rows = report.get("none_controls", [])
    if len(none_rows) != 4:
        raise ValueError(f"{source}: zero-active-site control count differs")
    for row in none_rows:
        if (
            row.get("active_call_count") != 0
            or finite(
                row.get("action_maximum_error_vs_native"),
                label="none_action_maximum_error",
            )
            != 0.0
            or row.get("future_exact_vs_native") is not True
            or row.get("x0_exact_vs_native") is not True
            or row.get("sigma_exact_vs_native") is not True
            or row.get("trace_signature_exact_vs_native") is not True
        ):
            raise ValueError(f"{source}: zero-active-site no-op gate failed")

    self_rows = report.get("self_controls", [])
    if len(self_rows) != 4:
        raise ValueError(f"{source}: self control count differs")
    for row in self_rows:
        if row["repeat_action_maximum_error"] != 0.0 or not row["repeat_signature_exact"] or not row["target_matches_native_future"]:
            raise ValueError(f"{source}: self replay/signature gate failed")
    donor_rows = report.get("donor_rows", [])
    expected_pairs = [tuple(pair) for pair in unit["ordered_pairs"]]
    actual_pairs = [(row["recipient_seed"], row["target_donor_seed"]) for row in donor_rows]
    if len(donor_rows) != 12 or actual_pairs != expected_pairs:
        raise ValueError(f"{source}: donor subset differs from frozen grid")
    for row in donor_rows:
        if row["repeat_action_maximum_error"] != 0.0 or not row["repeat_signature_exact"] or not row["target_matches_native_future"]:
            raise ValueError(f"{source}: donor replay/signature gate failed")
        separation = finite(row["native_target_l2"], label="native_target_l2")
        directional_fields = (
            "distance_reduction_to_target",
            "cosine_alignment",
            "orthogonal_residual_normalized",
            "normalized_projection",
        )
        if separation <= 1e-12:
            if any(row.get(field) is not None for field in directional_fields):
                raise ValueError(
                    f"{source}: degenerate donor axis has a non-null directional metric"
                )
        elif any(
            metric_or_none(row.get(field), label=field) is None
            for field in directional_fields
        ):
            raise ValueError(
                f"{source}: nondegenerate donor axis lacks a directional metric"
            )
    retrieval = report.get("future_source_retrieval_rows", [])
    actual_cells = [(row["recipient_seed"], row["future_source_seed"]) for row in retrieval]
    if len(retrieval) != 16 or actual_cells != [tuple(cell) for cell in unit["future_source_retrieval_cells"]]:
        raise ValueError(f"{source}: 4x4 retrieval grid differs")
    gaussian = report.get("gaussian_rows", [])
    if len(gaussian) != 12:
        raise ValueError(f"{source}: Gaussian control count differs")
    for row in gaussian:
        norm_error = finite(row["norm_relative_error"], label="norm_relative_error")
        distance_error = finite(
            row["distance_relative_error"], label="distance_relative_error"
        )
        if norm_error > 1e-5 or distance_error > 1e-5:
            raise ValueError(f"{source}: Gaussian geometry control failed")


def main() -> None:
    args = parse_args()
    if sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_model_outcomes":
        raise ValueError("manifest is not frozen before model outcomes")
    if sha256(Path(__file__).resolve()) != manifest["runtime"]["analyzer_sha256"]:
        raise ValueError("analyzer differs from the hash frozen in the manifest")
    states = manifest.get("states", [])
    if len(states) != 90 or manifest.get("primary_chance_rate") != 0.25:
        raise ValueError("manifest is not the frozen 90-state four-way study")
    expected_files = {f"{unit['unit_id']}.json" for unit in states}
    observed_files = {path.name for path in args.input_root.glob("*.json")}
    if observed_files != expected_files:
        missing = sorted(expected_files - observed_files)
        extra = sorted(observed_files - expected_files)
        raise RuntimeError(
            f"refusing partial analysis: expected 90 exact files; missing={missing[:5]} "
            f"({len(missing)}), extra={extra[:5]} ({len(extra)})"
        )

    reports = []
    state_rows: list[dict[str, Any]] = []
    for unit in states:
        path = args.input_root / f"{unit['unit_id']}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        validate_report(report, unit, manifest, args.expected_manifest_sha256)
        reports.append(report)
        retrieval = report["future_source_retrieval_rows"]
        donors = report["donor_rows"]
        gaussian = report["gaussian_rows"]

        def mean_bool(rows: list[dict[str, Any]], field: str) -> float:
            values = [row[field] for row in rows]
            if not all(isinstance(value, bool) for value in values):
                raise ValueError(f"{unit['unit_id']}: {field} is not Boolean")
            return float(np.mean(values))

        def mean_metric(
            rows: list[dict[str, Any]], field: str
        ) -> tuple[float | None, int, int]:
            values = [metric_or_none(row.get(field), label=field) for row in rows]
            finite_values = [value for value in values if value is not None]
            return (
                float(np.mean(finite_values)) if finite_values else None,
                len(finite_values),
                len(values) - len(finite_values),
            )

        donor_metric_fields = (
            "distance_reduction_to_target",
            "cosine_alignment",
            "orthogonal_residual_normalized",
            "normalized_projection",
        )
        donor_metric_summaries = {
            field: mean_metric(donors, field) for field in donor_metric_fields
        }
        donor_degenerate = sum(
            finite(row["native_target_l2"], label="native_target_l2") <= 1e-12
            for row in donors
        )
        if any(
            summary[2] != donor_degenerate
            for summary in donor_metric_summaries.values()
        ):
            raise ValueError(
                f"{unit['unit_id']}: directional null counts do not match degenerate axes"
            )
        native_separation_summary = mean_metric(donors, "native_target_l2")
        gaussian_projection_summary = mean_metric(gaussian, "normalized_projection")
        gaussian_distance_summary = mean_metric(
            gaussian, "distance_reduction_to_target"
        )
        if (
            native_separation_summary[2] != 0
            or gaussian_projection_summary[2] != donor_degenerate
            or gaussian_distance_summary[2] != donor_degenerate
        ):
            raise ValueError(
                f"{unit['unit_id']}: Gaussian/native metric denominators do not match "
                "the prespecified donor axes"
            )

        state_rows.append(
            {
                "unit_id": unit["unit_id"],
                "task": unit["task"],
                "episode_id": unit["episode_id"],
                "environment_seed": unit["environment_seed"],
                "phase": unit["phase"],
                "branch_step": unit["branch_step"],
                "retrieval_top1": mean_bool(retrieval, "correct_future_source_top1"),
                "retrieval_shuffled_top1": mean_bool(retrieval, "shuffled_source_top1"),
                "donor_top1": mean_bool(donors, "correct_donor_top1"),
                "wrong_donor_top1": mean_bool(donors, "wrong_donor_top1"),
                "gaussian_top1": mean_bool(gaussian, "correct_donor_top1"),
                "distance_reduction": donor_metric_summaries[
                    "distance_reduction_to_target"
                ][0],
                "cosine_alignment": donor_metric_summaries["cosine_alignment"][0],
                "orthogonal_residual_normalized": donor_metric_summaries[
                    "orthogonal_residual_normalized"
                ][0],
                "normalized_projection": donor_metric_summaries[
                    "normalized_projection"
                ][0],
                "native_separation_mean": native_separation_summary[0],
                "gaussian_projection": gaussian_projection_summary[0],
                "gaussian_distance_reduction": gaussian_distance_summary[0],
                "donor_directional_valid_arms": len(donors) - donor_degenerate,
                "donor_directional_degenerate_arms": donor_degenerate,
                "native_future_hashes_distinct": float(report["native_future_hashes_distinct"]),
                "final_sampler_target_max_abs_error": max(
                    float(value)
                    for value in report[
                        "final_sampler_target_max_abs_errors"
                    ].values()
                ),
                "final_sampler_target_l2": max(
                    float(value)
                    for value in report["final_sampler_target_l2_errors"].values()
                ),
            }
        )

    metric_names = [
        "retrieval_top1",
        "retrieval_shuffled_top1",
        "donor_top1",
        "wrong_donor_top1",
        "gaussian_top1",
        "distance_reduction",
        "cosine_alignment",
        "orthogonal_residual_normalized",
        "normalized_projection",
        "native_separation_mean",
        "gaussian_projection",
        "gaussian_distance_reduction",
        "native_future_hashes_distinct",
        "final_sampler_target_max_abs_error",
        "final_sampler_target_l2",
    ]
    aggregate = {
        metric: hierarchical_task_episode_state_bootstrap(state_rows, metric)
        for metric in metric_names
    }
    task_results = {metric: per_task(state_rows, metric) for metric in metric_names}
    loto = {metric: leave_one_task_out(state_rows, metric) for metric in metric_names}
    phase_results = {
        phase: {
            metric: hierarchical_task_episode_state_bootstrap(
                [row for row in state_rows if row["phase"] == phase], metric
            )
            for metric in metric_names
        }
        for phase in ("early", "middle", "late")
    }
    pair_separation_quartiles = donor_pair_separation_quartiles(reports)
    separations = np.asarray([float(row["native_separation_mean"]) for row in state_rows])
    boundaries = np.quantile(separations, [0.25, 0.5, 0.75]).tolist()
    for row in state_rows:
        row["native_separation_quartile"] = int(
            np.searchsorted(boundaries, row["native_separation_mean"], side="right") + 1
        )
    quartiles = {
        str(quartile): {
            metric: hierarchical_task_episode_state_bootstrap(
                [row for row in state_rows if row["native_separation_quartile"] == quartile],
                metric,
            )
            for metric in metric_names
        }
        for quartile in range(1, 5)
    }
    total_donor_arms = sum(len(report["donor_rows"]) for report in reports)
    total_degenerate_donor_arms = sum(
        int(row["donor_directional_degenerate_arms"]) for row in state_rows
    )
    total_valid_donor_arms = sum(
        int(row["donor_directional_valid_arms"]) for row in state_rows
    )
    expected_coordinate_audits = len(state_rows) * 48
    observed_coordinate_audits = sum(
        len(report["action_coordinate_errors"]) for report in reports
    )
    expected_active_site_audits = len(state_rows) * 176
    observed_input_site_audits = sum(
        sum(
            len(audit["model_input_future_clamp_errors"])
            for audit in report["intervention_site_audits"].values()
        )
        for report in reports
    )
    observed_velocity_site_audits = sum(
        sum(
            len(audit["returned_future_velocity_overwrite_errors"])
            for audit in report["intervention_site_audits"].values()
        )
        for report in reports
    )
    final_sampler_max_abs_residuals = np.asarray(
        [
            finite(value, label="final_sampler_target_max_abs_error")
            for report in reports
            for value in report["final_sampler_target_max_abs_errors"].values()
        ],
        dtype=np.float64,
    )
    final_sampler_l2_residuals = np.asarray(
        [
            finite(value, label="final_sampler_target_l2")
            for report in reports
            for value in report["final_sampler_target_l2_errors"].values()
        ],
        dtype=np.float64,
    )
    expected_final_sampler_residuals = len(state_rows) * 44
    if (
        len(final_sampler_max_abs_residuals) != expected_final_sampler_residuals
        or len(final_sampler_l2_residuals) != expected_final_sampler_residuals
    ):
        raise RuntimeError("descriptive final sampler residual cardinality differs")
    final_sampler_residual_summary = {
        "interpretation": (
            "finite descriptive final sampler-state residual; never an admission, "
            "exclusion, stopping, or evidence criterion"
        ),
        "max_abs": {
            **describe_final_sampler_residuals(
                final_sampler_max_abs_residuals,
                label="final_sampler_target_max_abs_error",
            ),
            "fraction_gt_0_03": float(
                np.mean(final_sampler_max_abs_residuals > 0.03)
            ),
        },
        "l2": {
            **describe_final_sampler_residuals(
                final_sampler_l2_residuals,
                label="final_sampler_target_l2",
            ),
            "fraction_gt_0_03": float(np.mean(final_sampler_l2_residuals > 0.03)),
        },
    }
    exact_replay_count = {
        "native": sum(
            sum(value is True for value in report["native_deterministic_metadata_replay_exact"].values())
            for report in reports
        ),
        "self": sum(
            sum(row["repeat_signature_exact"] is True for row in report["self_controls"])
            for report in reports
        ),
        "donor": sum(
            sum(row["repeat_signature_exact"] is True for row in report["donor_rows"])
            for report in reports
        ),
        "none_no_op": sum(
            sum(
                row["action_maximum_error_vs_native"] == 0.0
                and row["future_exact_vs_native"] is True
                and row["x0_exact_vs_native"] is True
                and row["sigma_exact_vs_native"] is True
                and row["trace_signature_exact_vs_native"] is True
                for row in report["none_controls"]
            )
            for report in reports
        ),
    }
    evidence_criteria = {
        "complete_cohort_90_of_90": len(state_rows) == 90,
        "retrieval_hierarchical_ci_lower_gt_chance_0_25": (
            aggregate["retrieval_top1"]["ci95"] is not None
            and aggregate["retrieval_top1"]["ci95"][0] > 0.25
        ),
        "donor_distance_reduction_hierarchical_ci_lower_gt_zero": (
            aggregate["distance_reduction"]["ci95"] is not None
            and aggregate["distance_reduction"]["ci95"][0] > 0.0
        ),
        "zero_degenerate_donor_axes": total_degenerate_donor_arms == 0,
        "exact_control_cardinality": (
            observed_coordinate_audits == expected_coordinate_audits
            and observed_input_site_audits == expected_active_site_audits
            and observed_velocity_site_audits == expected_active_site_audits
            and exact_replay_count == {
                "native": len(state_rows) * 4,
                "self": len(state_rows) * 4,
                "donor": len(state_rows) * 12,
                "none_no_op": len(state_rows) * 4,
            }
        ),
        "intervention_site_errors_within_frozen_tolerance": all(
            report["model_input_future_clamp_max_error"]
            <= manifest["runtime"]["intervention_site_error_tolerance"]
            and report["returned_future_velocity_overwrite_max_error"]
            <= manifest["runtime"]["intervention_site_error_tolerance"]
            for report in reports
        ),
        "singleton_expected_input_and_parameter_fingerprints": all(
            report["input_fingerprint_count"] == 1
            and report["parameter_probe_hash_count"] == 1
            and report["parameter_probe_hashes"]
            == [manifest["runtime"]["expected_parameter_probe_hash"]]
            for report in reports
        ),
    }
    evidence_criteria["all_prespecified_criteria_pass"] = all(
        evidence_criteria.values()
    )
    summary = {
        "status": "complete",
        "scope": manifest["scope"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.expected_manifest_sha256,
        "state_count": len(state_rows),
        "episode_count": len({row["episode_id"] for row in state_rows}),
        "task_count": len({row["task"] for row in state_rows}),
        "primary": aggregate["retrieval_top1"],
        "primary_permutation": source_permutation_test(reports),
        "evidence_criteria": evidence_criteria,
        "directional_metric_denominator": {
            "prespecified_off_diagonal_donor_arms": total_donor_arms,
            "valid_non_degenerate_arms": total_valid_donor_arms,
            "null_degenerate_arms": total_degenerate_donor_arms,
        },
        "control_audit_counts": {
            "action_coordinate_expected": expected_coordinate_audits,
            "action_coordinate_observed_and_zero": observed_coordinate_audits,
            "model_input_future_clamp_site_expected": expected_active_site_audits,
            "model_input_future_clamp_site_observed_within_tolerance": (
                observed_input_site_audits
            ),
            "returned_future_velocity_site_expected": expected_active_site_audits,
            "returned_future_velocity_site_observed_within_tolerance": (
                observed_velocity_site_audits
            ),
            "exact_replays": exact_replay_count,
        },
        "final_sampler_target_residual_descriptive": final_sampler_residual_summary,
        "aggregate": aggregate,
        "phase": phase_results,
        "native_pair_separation_quartiles": pair_separation_quartiles,
        "native_state_mean_separation_quartiles_secondary": {
            "boundaries": boundaries,
            "quartiles": quartiles,
        },
        "per_task": task_results,
        "leave_one_task_out": loto,
        "analysis_hierarchy": "task -> archived episode -> state; within-state arms averaged",
        "limitations": "archival lossy-input action-only study; no fresh physical endpoint estimate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_rows_path = args.output.with_name("state_rows.csv")
    if args.output.exists() or state_rows_path.exists():
        raise FileExistsError("refusing to overwrite a completed or partial analysis artifact")
    # The summary JSON is the completion marker, so write it last.
    atomic_csv(state_rows_path, state_rows)
    atomic_json(args.output, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "state_count": len(state_rows),
                "summary": str(args.output),
                "state_rows": str(state_rows_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
