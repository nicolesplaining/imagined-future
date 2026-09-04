#!/usr/bin/env python3
"""Independent, inventory-gated audit of the complete Cosmos archival v7 cohort.

The script does not import the frozen runner, protocol helpers, or analyzer.  It
reconstructs action metrics from stored action arrays and will not parse any outcome
JSON until an exact 90-file, hash-inventoried, mode-0444 package has been verified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from cosmos3_raw_audit_common import (
    atomic_json_no_overwrite,
    compare_tree,
    finite,
    load_json,
    load_reference_after_hash,
    verify_frozen_package,
)


SEEDS = (211, 223, 227, 229)
PHASES = ("early", "middle", "late")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260903
METRICS = (
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
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--state-csv", type=Path, required=True)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def action(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (32, 8) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: expected a finite [32,8] action")
    return array


def nearest(value: np.ndarray, native: Mapping[int, np.ndarray]) -> tuple[int, dict[str, float]]:
    # The archival runner performs its nearest-native norm on float32 policy
    # arrays (directional estimands below intentionally promote to float64).
    value32 = np.asarray(value, dtype=np.float32)
    distances = {
        str(seed): float(np.linalg.norm(value32 - np.asarray(native[seed], dtype=np.float32)))
        for seed in SEEDS
    }
    winner = min(SEEDS, key=lambda seed: (distances[str(seed)], seed))
    return winner, distances


def directional(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> dict[str, float | None]:
    result = value.reshape(-1) - recipient.reshape(-1)
    axis = donor.reshape(-1) - recipient.reshape(-1)
    separation = float(np.linalg.norm(axis))
    distance = float(np.linalg.norm(value.reshape(-1) - donor.reshape(-1)))
    if separation <= 1e-12:
        return {
            "native_target_l2": separation,
            "l2_to_target": distance,
            "distance_reduction_to_target": None,
            "cosine_alignment": None,
            "orthogonal_residual_normalized": None,
            "normalized_projection": None,
        }
    coefficient = float(np.dot(result, axis) / np.dot(axis, axis))
    result_norm = float(np.linalg.norm(result))
    return {
        "native_target_l2": separation,
        "l2_to_target": distance,
        "distance_reduction_to_target": 1.0 - distance / separation,
        "cosine_alignment": (
            float(np.dot(result, axis) / (result_norm * separation))
            if result_norm > 1e-12
            else None
        ),
        "orthogonal_residual_normalized": float(
            np.linalg.norm(result - coefficient * axis) / separation
        ),
        "normalized_projection": coefficient,
    }


def same_number(actual: Any, expected: Any, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label}: null/non-null mismatch")
        return
    if not math.isclose(finite(actual, label), finite(expected, label), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}: stored metric does not equal action-derived metric")


def same_distances(actual: Any, expected: Mapping[str, float], label: str) -> None:
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        raise ValueError(f"{label}: native-distance keys differ")
    # np.linalg.norm accumulation may vary by a few float32 ulps across NumPy/BLAS
    # builds.  Classification is checked exactly; scalar distances use a tight
    # cross-runtime tolerance.
    for key, value in expected.items():
        if not math.isclose(finite(actual[key], label), value, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"{label}: native distance {key} does not reproduce")


def mean_optional(rows: Iterable[Mapping[str, Any]], field: str) -> tuple[float | None, int, int]:
    materialized = list(rows)
    values = [None if row.get(field) is None else finite(row[field], field) for row in materialized]
    valid = [value for value in values if value is not None]
    return (float(np.mean(valid)) if valid else None, len(valid), len(values) - len(valid))


def compare_state_csv(states: list[dict[str, Any]], path: Path, expected_sha256: str) -> list[str]:
    from cosmos3_raw_audit_common import require_sha

    require_sha(path, expected_sha256, "archival state CSV")
    with path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))
    if len(reference) != len(states):
        return [f"state_rows.csv: row count {len(reference)} != {len(states)}"]
    problems: list[str] = []
    for index, (computed, observed) in enumerate(zip(states, reference, strict=True)):
        if set(computed) != set(observed):
            problems.append(f"state_rows.csv[{index}]: columns differ")
            continue
        for key, expected in computed.items():
            actual = observed[key]
            if expected is None:
                if actual not in ("", "None"):
                    problems.append(f"state_rows.csv[{index}].{key}: expected null")
            elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
                try:
                    parsed = float(actual)
                except ValueError:
                    problems.append(f"state_rows.csv[{index}].{key}: not numeric")
                    continue
                if not math.isclose(float(expected), parsed, rel_tol=1e-12, abs_tol=1e-12):
                    problems.append(f"state_rows.csv[{index}].{key}: numeric value differs")
            elif str(expected) != actual:
                problems.append(f"state_rows.csv[{index}].{key}: value differs")
    return problems


def hierarchy(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get(metric) is not None:
            grouped[str(row["task"])][str(row["episode_id"])].append(finite(row[metric], metric))
    eligible = sum(row.get(metric) is not None for row in rows)
    if not grouped:
        return {
            "mean": None, "ci95": None, "tasks": 0, "episodes": 0, "states": 0,
            "input_states": len(rows), "eligible_states": eligible,
            "null_states": len(rows) - eligible,
        }
    tasks = sorted(grouped)
    point_tasks = [
        np.mean([np.mean(values) for values in grouped[task].values()]) for task in tasks
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty(BOOTSTRAPS, dtype=np.float64)
    for draw in range(BOOTSTRAPS):
        sampled_tasks: list[float] = []
        for task_index in rng.integers(0, len(tasks), size=len(tasks)):
            task = tasks[int(task_index)]
            episodes = sorted(grouped[task])
            sampled_episodes: list[float] = []
            for episode_index in rng.integers(0, len(episodes), size=len(episodes)):
                values = np.asarray(grouped[task][episodes[int(episode_index)]], dtype=np.float64)
                state_indices = rng.integers(0, len(values), size=len(values))
                sampled_episodes.append(float(values[state_indices].mean()))
            sampled_tasks.append(float(np.mean(sampled_episodes)))
        boot[draw] = float(np.mean(sampled_tasks))
    values = [finite(row[metric], metric) for row in rows if row.get(metric) is not None]
    return {
        "mean": float(np.mean(point_tasks)),
        "state_weighted_mean": float(np.mean(values)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "tasks": len(tasks),
        "episodes": sum(len(value) for value in grouped.values()),
        "states": sum(len(states) for episodes in grouped.values() for states in episodes.values()),
        "input_states": len(rows),
        "eligible_states": eligible,
        "null_states": len(rows) - eligible,
        "samples": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
    }


def task_means(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        values = [finite(row[metric], metric) for row in rows if row["task"] == task and row.get(metric) is not None]
        result[task] = float(np.mean(values))
    return result


def loto(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    tasks = sorted({str(row["task"]) for row in rows})
    return {
        held: float(np.mean(list(task_means([row for row in rows if row["task"] != held], metric).values())))
        for held in tasks
    }


def residual_description(values: Iterable[Any]) -> dict[str, Any]:
    array = np.asarray([finite(value, "sampler residual") for value in values], dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "quantiles": {str(q): float(np.quantile(array, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        "count_gt_0_03": int(np.count_nonzero(array > 0.03)),
    }


def permutation(reports: list[dict[str, Any]]) -> dict[str, Any]:
    state_scores: list[float] = []
    predictions: list[dict[int, list[int]]] = []
    state_tasks: list[str] = []
    for report in reports:
        recipients: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row in report["future_source_retrieval_rows"]:
            recipients[int(row["recipient_seed"])].append(
                (int(row["future_source_seed"]), int(row["nearest_native_seed"]))
            )
        state_prediction: dict[int, list[int]] = {}
        correct: list[int] = []
        for recipient, pairs in sorted(recipients.items()):
            pairs.sort()
            sources = [source for source, _ in pairs]
            guessed = [guess for _, guess in pairs]
            if len(sources) != 4 or len(set(sources)) != 4:
                raise ValueError("retrieval permutation cell is not four-way complete")
            state_prediction[recipient] = guessed
            correct.extend(int(left == right) for left, right in zip(sources, guessed, strict=True))
        state_scores.append(float(np.mean(correct)))
        predictions.append(state_prediction)
        state_tasks.append(str(report["task"]))
    tasks = sorted(set(state_tasks))
    observed = float(np.mean([
        np.mean([score for score, task_value in zip(state_scores, state_tasks, strict=True) if task_value == task])
        for task in tasks
    ]))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    null = np.empty(BOOTSTRAPS, dtype=np.float64)
    source_array = np.asarray(SEEDS, dtype=np.int64)
    for draw in range(BOOTSTRAPS):
        scores: list[float] = []
        for state_prediction in predictions:
            matches: list[int] = []
            for recipient in sorted(state_prediction):
                permuted = rng.permutation(source_array)
                matches.extend(
                    int(label == guess)
                    for label, guess in zip(permuted, state_prediction[recipient], strict=True)
                )
            scores.append(float(np.mean(matches)))
        null[draw] = float(np.mean([
            np.mean([score for score, state_task in zip(scores, state_tasks, strict=True) if state_task == task])
            for task in tasks
        ]))
    return {
        "observed_equal_task_mean": observed,
        "chance": 0.25,
        "p_greater_monte_carlo": float((1 + np.sum(null >= observed)) / (BOOTSTRAPS + 1)),
        "null_mean": float(null.mean()),
        "null_ci95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "samples": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "permutation_unit": "four future-source labels independently within each recipient and state",
    }


def pair_quartiles(arms: list[dict[str, Any]]) -> dict[str, Any]:
    if len(arms) != 1080:
        raise ValueError(f"off-diagonal arm count is {len(arms)}, expected 1080")
    boundaries = np.quantile(
        np.asarray([arm["native_separation"] for arm in arms], dtype=np.float64),
        [0.25, 0.5, 0.75],
    ).tolist()
    for arm in arms:
        arm["quartile"] = int(np.searchsorted(boundaries, arm["native_separation"], side="right") + 1)
    names = (
        "donor_top1", "wrong_donor_top1", "distance_reduction", "cosine_alignment",
        "orthogonal_residual_normalized", "normalized_projection", "native_separation",
    )
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
        states: list[dict[str, Any]] = []
        for unit_id, state_arms in sorted(buckets.items()):
            row: dict[str, Any] = {
                "unit_id": unit_id,
                "task": state_arms[0]["task"],
                "episode_id": state_arms[0]["episode_id"],
                "arm_count": len(state_arms),
            }
            for metric in names:
                valid = [arm[metric] for arm in state_arms if arm[metric] is not None]
                row[metric] = float(np.mean(valid)) if valid else None
                row[f"{metric}_valid_arm_count"] = len(valid)
                row[f"{metric}_null_arm_count"] = len(state_arms) - len(valid)
            states.append(row)
        output["quartiles"][str(quartile)] = {
            "arm_count": len(selected),
            "state_count": len(states),
            "episode_count": len({row["episode_id"] for row in states}),
            "task_count": len({row["task"] for row in states}),
            "metric_valid_arm_counts": {
                metric: sum(row[f"{metric}_valid_arm_count"] for row in states) for metric in names
            },
            "metric_null_arm_counts": {
                metric: sum(row[f"{metric}_null_arm_count"] for row in states) for metric in names
            },
            "aggregate": {metric: hierarchy(states, metric) for metric in names},
        }
    return output


def validate_and_derive(
    report: dict[str, Any], unit: dict[str, Any], manifest: dict[str, Any], manifest_hash: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    label = str(unit["unit_id"])
    exact = {
        "status": "complete", "admission": manifest["admission"],
        "manifest_id": manifest["manifest_id"], "manifest_sha256": manifest_hash,
        "unit_id": unit["unit_id"], "task": unit["task"],
        "environment_seed": unit["environment_seed"], "episode_id": unit["episode_id"],
        "phase": unit["phase"], "branch_step": unit["branch_step"], "scope": manifest["scope"],
        "branch_seeds": list(SEEDS), "ordered_pairs": unit["ordered_pairs"],
        "future_source_retrieval_cells": unit["future_source_retrieval_cells"],
        "request_count": 56, "input_fingerprint_count": 1, "parameter_probe_hash_count": 1,
        "parameter_probe_hashes": [manifest["runtime"]["expected_parameter_probe_hash"]],
        "expected_parameter_probe_hash": manifest["runtime"]["expected_parameter_probe_hash"],
        "active_intervention_response_count": 44, "active_intervention_site_count": 176,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise ValueError(f"{label}: {key} differs from manifest/design")
    native = {seed: action(report["native_actions"][str(seed)], f"{label}:native:{seed}") for seed in SEEDS}
    expected_cells = [(recipient, source) for recipient in SEEDS for source in SEEDS]
    expected_pairs = [(recipient, donor) for recipient in SEEDS for donor in SEEDS if donor != recipient]

    self_rows = list(report.get("self_controls", []))
    donor_rows = list(report.get("donor_rows", []))
    gaussian_rows = list(report.get("gaussian_rows", []))
    if len(self_rows) != 4 or len(donor_rows) != 12 or len(gaussian_rows) != 12:
        raise ValueError(f"{label}: self/donor/Gaussian arm census differs")
    by_cell: dict[tuple[int, int], dict[str, Any]] = {}
    arms: list[dict[str, Any]] = []
    degenerate = 0
    for row in self_rows:
        seed = int(row["recipient_seed"])
        if int(row["future_source_seed"]) != seed:
            raise ValueError(f"{label}: self cell is off diagonal")
        value = action(row["action"], f"{label}:self:{seed}")
        winner, distances = nearest(value, native)
        same_distances(row["distances_to_native_actions"], distances, f"{label}:self distances")
        if winner != int(row["nearest_native_seed"]):
            raise ValueError(f"{label}: self nearest-native fields do not reproduce")
        if bool(row["correct_future_source_top1"]) != (winner == seed):
            raise ValueError(f"{label}: self retrieval Boolean does not reproduce")
        by_cell[(seed, seed)] = row
    for row, expected_pair in zip(donor_rows, expected_pairs, strict=True):
        pair = (int(row["recipient_seed"]), int(row["target_donor_seed"]))
        if pair != expected_pair or int(row["future_source_seed"]) != pair[1]:
            raise ValueError(f"{label}: donor row order/cell differs")
        value = action(row["action"], f"{label}:donor:{pair}")
        winner, distances = nearest(value, native)
        derived_metrics = directional(value, native[pair[0]], native[pair[1]])
        for key, expected_value in derived_metrics.items():
            same_number(row.get(key), expected_value, f"{label}:{pair}:{key}")
        same_distances(row["distances_to_native_actions"], distances, f"{label}:donor distances")
        if winner != int(row["nearest_native_seed"]):
            raise ValueError(f"{label}: donor nearest-native fields do not reproduce")
        if bool(row["correct_donor_top1"]) != (winner == pair[1]):
            raise ValueError(f"{label}: donor retrieval Boolean does not reproduce")
        wrong = next(
            int(entry["wrong_donor_seed"])
            for entry in unit["frozen_wrong_donor_mapping"]
            if (int(entry["recipient_seed"]), int(entry["donor_seed"])) == pair
        )
        if int(row["frozen_wrong_donor_seed"]) != wrong or bool(row["wrong_donor_top1"]) != (winner == wrong):
            raise ValueError(f"{label}: frozen wrong-donor control does not reproduce")
        by_cell[pair] = row
        is_degenerate = derived_metrics["native_target_l2"] <= 1e-12
        degenerate += int(is_degenerate)
        arms.append({
            "unit_id": label, "task": report["task"], "episode_id": report["episode_id"],
            "donor_top1": float(winner == pair[1]), "wrong_donor_top1": float(winner == wrong),
            "distance_reduction": derived_metrics["distance_reduction_to_target"],
            "cosine_alignment": derived_metrics["cosine_alignment"],
            "orthogonal_residual_normalized": derived_metrics["orthogonal_residual_normalized"],
            "normalized_projection": derived_metrics["normalized_projection"],
            "native_separation": derived_metrics["native_target_l2"],
        })
    for row, expected_pair in zip(gaussian_rows, expected_pairs, strict=True):
        pair = (int(row["recipient_seed"]), int(row["target_donor_seed"]))
        if pair != expected_pair:
            raise ValueError(f"{label}: Gaussian row order/cell differs")
        value = action(row["action"], f"{label}:gaussian:{pair}")
        winner, distances = nearest(value, native)
        derived_metrics = directional(value, native[pair[0]], native[pair[1]])
        for key, expected_value in derived_metrics.items():
            same_number(row.get(key), expected_value, f"{label}:gaussian:{pair}:{key}")
        same_distances(row["distances_to_native_actions"], distances, f"{label}:Gaussian distances")
        if winner != int(row["nearest_native_seed"]):
            raise ValueError(f"{label}: Gaussian nearest-native fields do not reproduce")
        if finite(row["norm_relative_error"], "Gaussian norm error") > 1e-5 or finite(row["distance_relative_error"], "Gaussian distance error") > 1e-5:
            raise ValueError(f"{label}: Gaussian geometry gate failed")

    retrieval = list(report.get("future_source_retrieval_rows", []))
    actual_cells = [(int(row["recipient_seed"]), int(row["future_source_seed"])) for row in retrieval]
    if actual_cells != expected_cells:
        raise ValueError(f"{label}: retrieval grid/order differs")
    shuffle_map = {int(entry["source_seed"]): int(entry["shuffled_source_seed"]) for entry in unit["frozen_source_label_permutation"]}
    for row in retrieval:
        cell = (int(row["recipient_seed"]), int(row["future_source_seed"]))
        source = by_cell[cell]
        for key in ("distances_to_native_actions", "nearest_native_seed", "correct_future_source_top1"):
            if row.get(key) != source.get(key):
                raise ValueError(f"{label}: retrieval cell {cell} is not sourced from its raw arm")
        winner = int(row["nearest_native_seed"])
        shuffled = shuffle_map[cell[1]]
        if int(row["shuffled_source_seed"]) != shuffled or bool(row["shuffled_source_top1"]) != (winner == shuffled):
            raise ValueError(f"{label}: shuffled-source control does not reproduce")

    none = list(report.get("none_controls", []))
    if len(none) != 4 or any(
        row.get("active_call_count") != 0
        or finite(row.get("action_maximum_error_vs_native"), "none action error") != 0.0
        or any(row.get(key) is not True for key in (
            "future_exact_vs_native", "x0_exact_vs_native", "sigma_exact_vs_native",
            "trace_signature_exact_vs_native",
        ))
        for row in none
    ):
        raise ValueError(f"{label}: none control failed")
    if any(
        finite(row["repeat_action_maximum_error"], "repeat error") != 0.0
        or row["repeat_signature_exact"] is not True
        or row["target_matches_native_future"] is not True
        for row in self_rows + donor_rows
    ):
        raise ValueError(f"{label}: self/donor replay control failed")
    if set(report["native_repeat_action_maximum_error"].values()) != {0.0} \
            or set(report["native_future_replay_exact"].values()) != {True} \
            or set(report["native_deterministic_metadata_replay_exact"].values()) != {True}:
        raise ValueError(f"{label}: native replay control failed")
    coordinates = report.get("action_coordinate_errors", {})
    if len(coordinates) != 48 or any(
        finite(value, "coordinate error") != 0.0 for pair in coordinates.values() for value in pair.values()
    ):
        raise ValueError(f"{label}: coordinate nonwrite control failed")
    audits = report.get("intervention_site_audits", {})
    if len(audits) != 48:
        raise ValueError(f"{label}: intervention-site audit census differs")
    modes = defaultdict(int)
    input_sites = velocity_sites = 0
    for site in audits.values():
        mode = str(site["mode"]); modes[mode] += 1
        expected_active = 0 if mode == "none" else 4
        if int(site["active_site_count"]) != expected_active:
            raise ValueError(f"{label}: active-site count differs")
        input_errors = [finite(value, "input site error") for value in site["model_input_future_clamp_errors"]]
        velocity_errors = [finite(value, "velocity site error") for value in site["returned_future_velocity_overwrite_errors"]]
        if len(input_errors) != expected_active or len(velocity_errors) != expected_active:
            raise ValueError(f"{label}: active-site telemetry length differs")
        if any(value > 1e-7 for value in input_errors + velocity_errors):
            raise ValueError(f"{label}: intervention-site error exceeds tolerance")
        if finite(site["maximum_action_input_error"], "site action input") != 0.0 \
                or finite(site["maximum_action_output_error"], "site action output") != 0.0 \
                or int(site["inactive_wrapper_write_count"]) != 0:
            raise ValueError(f"{label}: site nonwrite control failed")
        input_sites += len(input_errors); velocity_sites += len(velocity_errors)
    if dict(modes) != {"none": 4, "self": 8, "donor": 24, "gaussian": 12} or input_sites != 176 or velocity_sites != 176:
        raise ValueError(f"{label}: site mode/count census differs")

    donor_distance = mean_optional(donor_rows, "distance_reduction_to_target")
    donor_cosine = mean_optional(donor_rows, "cosine_alignment")
    donor_orthogonal = mean_optional(donor_rows, "orthogonal_residual_normalized")
    donor_projection = mean_optional(donor_rows, "normalized_projection")
    separation = mean_optional(donor_rows, "native_target_l2")
    gaussian_projection = mean_optional(gaussian_rows, "normalized_projection")
    gaussian_distance = mean_optional(gaussian_rows, "distance_reduction_to_target")
    bool_mean = lambda rows, key: float(np.mean([float(row[key]) for row in rows]))
    state = {
        "unit_id": label, "task": unit["task"], "episode_id": unit["episode_id"],
        "environment_seed": unit["environment_seed"], "phase": unit["phase"],
        "branch_step": unit["branch_step"],
        "retrieval_top1": bool_mean(retrieval, "correct_future_source_top1"),
        "retrieval_shuffled_top1": bool_mean(retrieval, "shuffled_source_top1"),
        "donor_top1": bool_mean(donor_rows, "correct_donor_top1"),
        "wrong_donor_top1": bool_mean(donor_rows, "wrong_donor_top1"),
        "gaussian_top1": bool_mean(gaussian_rows, "correct_donor_top1"),
        "distance_reduction": donor_distance[0], "cosine_alignment": donor_cosine[0],
        "orthogonal_residual_normalized": donor_orthogonal[0],
        "normalized_projection": donor_projection[0], "native_separation_mean": separation[0],
        "gaussian_projection": gaussian_projection[0],
        "gaussian_distance_reduction": gaussian_distance[0],
        "donor_directional_valid_arms": 12 - degenerate,
        "donor_directional_degenerate_arms": degenerate,
        "native_future_hashes_distinct": float(report["native_future_hashes_distinct"]),
        "final_sampler_target_max_abs_error": max(float(value) for value in report["final_sampler_target_max_abs_errors"].values()),
        "final_sampler_target_l2": max(float(value) for value in report["final_sampler_target_l2_errors"].values()),
    }
    return state, arms, {"coordinates": len(coordinates), "input_sites": input_sites, "velocity_sites": velocity_sites}


def main() -> None:
    cli = args()
    manifest, paths, inventory = verify_frozen_package(
        manifest_path=cli.manifest,
        expected_manifest_sha256=cli.expected_manifest_sha256,
        output_root=cli.run_root,
        inventory_path=cli.inventory,
        expected_inventory_sha256=cli.expected_inventory_sha256,
        expected_count=90,
        expected_inventory_schema="cosmos3-archival-output-inventory-v1",
    )
    if manifest.get("study_name") != "cosmos3-archival-selection-free-action-only-v7" \
            or manifest.get("primary_chance_rate") != 0.25:
        raise ValueError("manifest is not the frozen archival v7 four-way study")
    by_name = {path.name: path for path in paths}
    reports: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    counts = {"coordinates": 0, "input_sites": 0, "velocity_sites": 0}
    for unit in manifest["states"]:
        report = load_json(by_name[f"{unit['unit_id']}.json"])
        state, state_arms, state_counts = validate_and_derive(
            report, unit, manifest, cli.expected_manifest_sha256
        )
        reports.append(report); states.append(state); arms.extend(state_arms)
        for key in counts:
            counts[key] += state_counts[key]
    aggregates = {metric: hierarchy(states, metric) for metric in METRICS}
    phases = {
        phase: {metric: hierarchy([row for row in states if row["phase"] == phase], metric) for metric in METRICS}
        for phase in PHASES
    }
    per_task = {metric: task_means(states, metric) for metric in METRICS}
    leave_out = {metric: loto(states, metric) for metric in METRICS}
    state_boundaries = np.quantile(
        np.asarray([row["native_separation_mean"] for row in states]), [0.25, 0.5, 0.75]
    ).tolist()
    for row in states:
        row["native_separation_quartile"] = int(
            np.searchsorted(state_boundaries, row["native_separation_mean"], side="right") + 1
        )
    state_quartiles = {
        str(q): {metric: hierarchy([row for row in states if row["native_separation_quartile"] == q], metric) for metric in METRICS}
        for q in range(1, 5)
    }
    degenerate = sum(int(row["donor_directional_degenerate_arms"]) for row in states)
    max_abs = [value for report in reports for value in report["final_sampler_target_max_abs_errors"].values()]
    l2_values = [value for report in reports for value in report["final_sampler_target_l2_errors"].values()]
    replay = {
        "native": sum(sum(value is True for value in report["native_deterministic_metadata_replay_exact"].values()) for report in reports),
        "self": sum(sum(row["repeat_signature_exact"] is True for row in report["self_controls"]) for report in reports),
        "donor": sum(sum(row["repeat_signature_exact"] is True for row in report["donor_rows"]) for report in reports),
        "none_no_op": sum(sum(
            row["action_maximum_error_vs_native"] == 0.0 and row["future_exact_vs_native"] is True
            and row["x0_exact_vs_native"] is True and row["sigma_exact_vs_native"] is True
            and row["trace_signature_exact_vs_native"] is True
            for row in report["none_controls"]
        ) for report in reports),
    }
    evidence = {
        "complete_cohort_90_of_90": len(states) == 90,
        "retrieval_hierarchical_ci_lower_gt_chance_0_25": aggregates["retrieval_top1"]["ci95"][0] > 0.25,
        "donor_distance_reduction_hierarchical_ci_lower_gt_zero": aggregates["distance_reduction"]["ci95"][0] > 0.0,
        "zero_degenerate_donor_axes": degenerate == 0,
        "exact_control_cardinality": counts == {"coordinates": 4320, "input_sites": 15840, "velocity_sites": 15840}
        and replay == {"native": 360, "self": 360, "donor": 1080, "none_no_op": 360},
        "intervention_site_errors_within_frozen_tolerance": all(
            report["model_input_future_clamp_max_error"] <= 1e-7
            and report["returned_future_velocity_overwrite_max_error"] <= 1e-7 for report in reports
        ),
        "singleton_expected_input_and_parameter_fingerprints": all(
            report["input_fingerprint_count"] == 1 and report["parameter_probe_hash_count"] == 1
            and report["parameter_probe_hashes"] == [manifest["runtime"]["expected_parameter_probe_hash"]]
            for report in reports
        ),
    }
    evidence["all_prespecified_criteria_pass"] = all(evidence.values())
    computed = {
        "state_count": 90, "episode_count": 30, "task_count": 6,
        "primary": aggregates["retrieval_top1"],
        "primary_permutation": permutation(reports),
        "evidence_criteria": evidence,
        "directional_metric_denominator": {
            "prespecified_off_diagonal_donor_arms": 1080,
            "valid_non_degenerate_arms": 1080 - degenerate,
            "null_degenerate_arms": degenerate,
        },
        "control_audit_counts": {
            "action_coordinate_expected": 4320,
            "action_coordinate_observed_and_zero": counts["coordinates"],
            "model_input_future_clamp_site_expected": 15840,
            "model_input_future_clamp_site_observed_within_tolerance": counts["input_sites"],
            "returned_future_velocity_site_expected": 15840,
            "returned_future_velocity_site_observed_within_tolerance": counts["velocity_sites"],
            "exact_replays": replay,
        },
        "final_sampler_target_residual_descriptive": {
            "interpretation": "finite descriptive final sampler-state residual; never an admission, exclusion, stopping, or evidence criterion",
            "max_abs": {**residual_description(max_abs), "fraction_gt_0_03": float(np.mean(np.asarray(max_abs) > 0.03))},
            "l2": {**residual_description(l2_values), "fraction_gt_0_03": float(np.mean(np.asarray(l2_values) > 0.03))},
        },
        "aggregate": aggregates,
        "phase": phases,
        "native_pair_separation_quartiles": pair_quartiles(arms),
        "native_state_mean_separation_quartiles_secondary": {
            "boundaries": state_boundaries, "quartiles": state_quartiles,
        },
        "per_task": per_task,
        "leave_one_task_out": leave_out,
        "analysis_hierarchy": "task -> archived episode -> state; within-state arms averaged",
    }
    reference = load_reference_after_hash(cli.summary_json, cli.expected_summary_sha256)
    discrepancies = compare_tree(computed, reference, path="summary")
    discrepancies.extend(compare_state_csv(states, cli.state_csv, cli.expected_state_sha256))
    result = {
        "status": "pass" if not discrepancies else "fail",
        "audit": "independent-cosmos3-archival-v7-raw-output-audit-v1",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": cli.expected_manifest_sha256,
        "inventory_sha256": cli.expected_inventory_sha256,
        "summary_sha256": cli.expected_summary_sha256,
        "state_csv_sha256": cli.expected_state_sha256,
        "outcome_file_count": len(paths),
        "raw_action_metrics_recomputed": True,
        "frozen_analyzer_or_helpers_imported": False,
        "computed": computed,
        "comparison_discrepancies": discrepancies,
    }
    atomic_json_no_overwrite(cli.output, result)
    print(json.dumps({"status": result["status"], "output": str(cli.output), "discrepancy_count": len(discrepancies)}, sort_keys=True))
    if discrepancies:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
