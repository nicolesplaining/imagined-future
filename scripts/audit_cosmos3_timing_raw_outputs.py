#!/usr/bin/env python3
"""Independent, inventory-gated audit of complete Cosmos timing-v5 outputs.

No frozen timing module, runner, or analyzer is imported.  All 30 reports remain
unparsed until an exact hash inventory and mode-0444 file-set check succeeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cosmos3_raw_audit_common import (
    atomic_json_no_overwrite,
    compare_tree,
    finite,
    load_json,
    load_reference_after_hash,
    require_sha,
    verify_frozen_package,
)


TASKS = (
    "BananaInBowlTask", "RubiksCubeTask", "MustardInLeftBinTask",
    "SpoonInMugTask", "MarkerInMugTask", "SmartphoneInBinTask",
)
SEEDS = (211, 223, 227, 229)
CONDITIONS = (
    ("none", ()), ("call_0_only", (0,)), ("call_1_only", (1,)),
    ("call_2_only", (2,)), ("call_3_only", (3,)),
    ("all_calls", (0, 1, 2, 3)),
)
SINGLES = tuple(name for name, _ in CONDITIONS[1:5])
PRIMARY = ("matched_retrieval_gain", "matched_distance_gain")
TIMING_METRICS = (
    "complete_source_retrieval", "raw_off_diagonal_donor_retrieval",
    "matched_retrieval_gain", "matched_distance_gain", "distance_reduction",
    "donor_projection", "cosine_alignment", "orthogonal_residual_normalized",
    "distance_to_donor", "minimum_top2_margin",
    "final_sampler_target_max_abs_error_mean", "final_sampler_target_max_abs_error_max",
    "final_sampler_target_l2_mean", "final_sampler_target_l2_max",
)
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260903


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    for name in ("states", "pairs", "per_task", "loto"):
        parser.add_argument(f"--{name.replace('_', '-')}-csv", type=Path, required=True)
        parser.add_argument(f"--expected-{name.replace('_', '-')}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cells() -> list[tuple[int, int]]:
    return [(recipient, source) for recipient in SEEDS for source in SEEDS]


def off_diagonal() -> list[tuple[int, int]]:
    return [(recipient, source) for recipient, source in cells() if recipient != source]


def parse_action(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (32, 8) or array.size != 256 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: expected finite [32,8]/256 action")
    return array


def null_paths(value: Any, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if value is None:
        return {path}
    if isinstance(value, Mapping):
        result: set[tuple[str, ...]] = set()
        for key, item in value.items():
            result.update(null_paths(item, path + (str(key),)))
        return result
    if isinstance(value, (list, tuple)):
        result: set[tuple[str, ...]] = set()
        for index, item in enumerate(value):
            result.update(null_paths(item, path + (str(index),)))
        return result
    return set()


def nearest(value: np.ndarray, natives: Mapping[int, np.ndarray]) -> tuple[int, float, bool]:
    ordered = sorted(
        (float(np.linalg.norm(value.reshape(-1) - natives[seed].reshape(-1))), SEEDS.index(seed), seed)
        for seed in SEEDS
    )
    return ordered[0][2], ordered[1][0] - ordered[0][0], ordered[0][0] == ordered[1][0]


def directional(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> dict[str, float]:
    flat = value.reshape(-1); base = recipient.reshape(-1); target = donor.reshape(-1)
    axis = target - base; displacement = flat - base
    separation = float(np.linalg.norm(axis))
    if not math.isfinite(separation) or separation <= 1e-12:
        raise ValueError("native action axis is degenerate")
    coefficient = float(np.dot(displacement, axis) / np.dot(axis, axis))
    norm = float(np.linalg.norm(displacement))
    return {
        "native_separation": separation,
        "distance_to_donor": float(np.linalg.norm(flat - target)),
        "distance_reduction": float((separation - np.linalg.norm(flat - target)) / separation),
        "donor_projection": coefficient,
        "cosine_alignment": float(np.dot(displacement, axis) / (norm * separation)) if norm > 1e-12 else 0.0,
        "orthogonal_residual_normalized": float(np.linalg.norm(displacement - coefficient * axis) / separation),
    }


def derive_state(report: dict[str, Any], unit: dict[str, Any], manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
    unit_id = str(unit["unit_id"])
    exact = {
        "status": "complete", "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_hash, "unit_id": unit_id,
        "task": unit["task"], "environment_seed": unit["environment_seed"],
        "phase": "middle", "request_count": 108, "branch_seeds": list(SEEDS),
        "action_shape": [32, 8], "action_coordinate_count": 256,
        "shape_valid_response_action_count": 108, "action_shape_failure_count": 0,
        "input_fingerprint_count": 1, "parameter_probe_hash_count": 1,
        "parameter_probe_hashes": [manifest["runtime"]["expected_parameter_probe_hash"]],
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise ValueError(f"{unit_id}: {key} differs from frozen manifest/design")
    if report.get("request_labels") != manifest["design"]["request_labels"]:
        raise ValueError(f"{unit_id}: request label order differs")
    controls = {
        "native_replay_max_action_error": 0.0,
        "all_calls_diagonal_replay_max_action_error": 0.0,
        "none_noop_max_action_error": 0.0,
        "none_source_invariance_max_action_error": 0.0,
        "maximum_action_input_error": 0.0,
        "maximum_action_output_error": 0.0,
        "maximum_active_model_input_future_clamp_error": 0.0,
        "maximum_active_returned_future_velocity_error": 0.0,
        "inactive_wrapper_write_count": 0,
        "schedule_and_index_gate_exact": True,
        "target_hash_gate_exact": True,
        "rng_hash_gate_exact": True,
        "replay_signature_gate_exact": True,
        "structural_projection_null_count": 28,
        "finite_off_diagonal_projection_count": 72,
        "native_projection_absent_count": 8,
    }
    for key, expected in controls.items():
        if report.get(key) != expected:
            raise ValueError(f"{unit_id}: frozen control {key} failed")
    if report.get("runtime_gate", {}).get("passed") is not True:
        raise ValueError(f"{unit_id}: runtime gate is not true")
    for key in (
        "exact_schedule", "exact_active_site_captures", "exact_mask",
        "zero_action_coordinate_writes", "zero_inactive_wrapper_writes",
        "exact_none_noop", "exact_replays", "exact_rng_and_target_hashes",
        "all_finite", "required_numeric_fields_finite",
        "structural_null_census_exact", "exact_projection_applicability_census",
        "exact_action_shape_and_count",
    ):
        if report["runtime_gate"].get(key) is not True:
            raise ValueError(f"{unit_id}: runtime gate {key} failed")
    for key, expected in (
        ("native_replay_action_errors", 0.0),
        ("all_calls_diagonal_replay_action_errors", 0.0),
        ("none_noop_action_errors", 0.0),
        ("none_source_action_errors", 0.0),
        ("native_replay_signature_exact", True),
        ("all_calls_diagonal_replay_signature_exact", True),
        ("none_source_invariance_exact", True),
    ):
        mapping = report.get(key)
        if not isinstance(mapping, Mapping) or set(mapping) != {str(seed) for seed in SEEDS} \
                or any(value != expected for value in mapping.values()):
            raise ValueError(f"{unit_id}: exact control map {key} failed")
    natives = {seed: parse_action(report["native_actions"][str(seed)], f"{unit_id}:native:{seed}") for seed in SEEDS}
    native_futures = report.get("native_future_hashes", {})
    native_paths = report.get("native_path_noise_hashes", {})
    native_initial = report.get("native_initial_state_hashes", {})
    expected_seed_keys = {str(seed) for seed in SEEDS}
    for name, mapping in (
        ("native_future_hashes", native_futures),
        ("native_path_noise_hashes", native_paths),
        ("native_initial_state_hashes", native_initial),
    ):
        if not isinstance(mapping, Mapping) or set(mapping) != expected_seed_keys \
                or any(not isinstance(value, str) or not value for value in mapping.values()):
            raise ValueError(f"{unit_id}: {name} is incomplete")
    separations = {
        pair: float(np.linalg.norm(natives[pair[1]].reshape(-1) - natives[pair[0]].reshape(-1)))
        for pair in off_diagonal()
    }
    if any(value <= 1e-12 or not math.isfinite(value) for value in separations.values()):
        raise ValueError(f"{unit_id}: degenerate native axis")
    rows = list(report.get("timing_rows", []))
    expected_order = [(name, recipient, source, tuple(active)) for name, active in CONDITIONS for recipient, source in cells()]
    actual_order = [
        (str(row.get("timing_condition")), int(row.get("recipient_seed", -1)),
         int(row.get("source_seed", -1)), tuple(int(value) for value in row.get("active_call_indices", [])))
        for row in rows
    ]
    if actual_order != expected_order:
        raise ValueError(f"{unit_id}: timing grid/order differs")
    expected_nulls: set[tuple[str, ...]] = set()
    indexed = {(name, recipient, source): row for row, (name, recipient, source, _) in zip(rows, expected_order, strict=True)}
    timing: dict[str, dict[str, float | int]] = {}
    pair_rows: list[dict[str, Any]] = []
    for name, _active in CONDITIONS:
        complete: list[float] = []; donor_correct: list[float] = []
        retrieval_gain: list[float] = []; distance_gain: list[float] = []
        vectors: dict[str, list[float]] = {
            key: [] for key in (
                "distance_reduction", "donor_projection", "cosine_alignment",
                "orthogonal_residual_normalized", "distance_to_donor",
            )
        }
        margins: list[float] = []; ties = 0; max_abs: list[float] = []; residual_l2: list[float] = []
        for recipient, source in cells():
            row = indexed[(name, recipient, source)]
            row_index = rows.index(row)
            value = parse_action(row.get("action"), f"{unit_id}:{name}:{recipient}:{source}")
            max_abs.append(finite(row["final_sampler_target_max_abs_error"], "sampler max-abs"))
            residual_l2.append(finite(row["final_sampler_target_l2"], "sampler l2"))
            winner, margin, tied = nearest(value, natives)
            complete.append(float(winner == source)); margins.append(margin); ties += int(tied)
            server = row.get("server", {})
            projection = server.get("research_action_donor_projection")
            applicable = server.get("research_action_donor_projection_applicable")
            if recipient == source:
                if projection is not None or applicable is not False:
                    raise ValueError(f"{unit_id}: diagonal projection is not structural null")
                expected_nulls.add(("timing_rows", str(row_index), "server", "research_action_donor_projection"))
            else:
                if applicable is not True or not math.isfinite(float(projection)):
                    raise ValueError(f"{unit_id}: off-diagonal server projection invalid")
                if name == "none" and float(projection) != 0.0:
                    raise ValueError(f"{unit_id}: none projection is nonzero")
            for key in ("research_maximum_action_input_error", "research_maximum_action_output_error"):
                if finite(server[key], f"{unit_id}:{key}") != 0.0:
                    raise ValueError(f"{unit_id}: per-row live control {key} failed")
            requested = list(_active)
            inactive = [index for index in range(4) if index not in _active]
            for key in (
                "research_requested_active_call_indices",
                "research_observed_active_call_indices",
                "research_clamped_call_indices",
            ):
                if server.get(key) != requested:
                    raise ValueError(f"{unit_id}: {key} differs from the timing condition")
            if server.get("research_inactive_call_indices") != inactive:
                raise ValueError(f"{unit_id}: inactive call indices differ")
            sigmas = np.asarray(manifest["design"]["research_sigmas"], dtype=np.float32)
            expected_arrays = {
                "research_sigmas": sigmas,
                "research_x0_sigmas": np.asarray(manifest["design"]["research_x0_sigmas"], dtype=np.float32),
                "research_requested_active_sigmas": sigmas[np.asarray(_active, dtype=np.int64)],
                "research_observed_active_sigmas": sigmas[np.asarray(_active, dtype=np.int64)],
                "research_future_frame_indices": np.asarray(manifest["design"]["future_frame_indices"], dtype=np.int64),
                "research_vision_shape": np.asarray(manifest["design"]["vision_shape"], dtype=np.int64),
            }
            for key, expected_array in expected_arrays.items():
                actual_array = np.asarray(server.get(key), dtype=expected_array.dtype)
                if actual_array.shape != expected_array.shape or not np.array_equal(actual_array, expected_array):
                    raise ValueError(f"{unit_id}: {key} differs from the frozen design")
            for key, expected_length in (
                ("research_model_input_future_clamp_errors", len(_active)),
                ("research_returned_future_velocity_overwrite_errors", len(_active)),
                ("research_action_input_errors", 4),
                ("research_action_output_errors", 4),
            ):
                errors = [finite(item, f"{unit_id}:{key}") for item in server.get(key, [])]
                if len(errors) != expected_length or any(item != 0.0 for item in errors):
                    raise ValueError(f"{unit_id}: {key} is not the expected all-zero vector")
            if int(server["research_inactive_wrapper_write_count"]) != 0:
                raise ValueError(f"{unit_id}: inactive wrapper write observed")
            attention = server.get("research_attention_interface", {})
            if attention.get("cache_id") is None:
                expected_nulls.add(("timing_rows", str(row_index), "server", "research_attention_interface", "cache_id"))
            if attention.get("instrumented_server") is not False \
                    or attention.get("intervention_requested") is not False \
                    or attention.get("mode") != "exclude":
                raise ValueError(f"{unit_id}: attention-routing control differs")
            expected_target_seed = recipient if name == "none" or recipient == source else source
            expected_source = "recipient" if expected_target_seed == recipient else "donor"
            expected_recipient_id = f"{manifest['manifest_id']}-{unit_id}-native-{recipient}"
            expected_source_id = f"{manifest['manifest_id']}-{unit_id}-native-{source}"
            expected_target_id = expected_recipient_id if expected_target_seed == recipient else expected_source_id
            if (
                server.get("research_recipient_id") != expected_recipient_id
                or server.get("research_donor_id") != expected_source_id
                or server.get("research_recipient_future_hash") != native_futures[str(recipient)]
                or server.get("research_donor_future_hash") != native_futures[str(source)]
                or server.get("research_target_hash") != native_futures[str(expected_target_seed)]
                or server.get("research_target_source") != expected_source
                or server.get("research_target_source_record_ids") != [expected_target_id]
                or server.get("research_recipient_path_noise_hash") != native_paths[str(recipient)]
                or server.get("research_initial_state_hash") != native_initial[str(recipient)]
                or server.get("research_state_hash") != report["input_fingerprints"][0]
                or server.get("research_parameter_probe_hash") != report["parameter_probe_hashes"][0]
            ):
                raise ValueError(f"{unit_id}: source/recipient/RNG hash identity differs")
            if recipient == source:
                continue
            self_value = parse_action(indexed[(name, recipient, recipient)]["action"], "matched self")
            self_winner, _, _ = nearest(self_value, natives)
            metrics = directional(value, natives[recipient], natives[source])
            # The server diagnostic is intentionally separate from the protocol
            # estimand below.  `_format_action` and the registered native actions
            # are float32 arrays, so its displacement is formed in float32 before
            # multiplication by the float64-cast native direction.  Reconstruct
            # that exact operation order here; casting both operands to float64
            # before subtraction can differ by a few 1e-9 through cancellation.
            recipient_f32 = natives[recipient].astype(np.float32)
            source_f32 = natives[source].astype(np.float32)
            value_f32 = value.astype(np.float32)
            server_direction = source_f32.astype(np.float64) - recipient_f32.astype(np.float64)
            server_projection = float(
                ((value_f32 - recipient_f32) * server_direction).sum()
                / np.square(server_direction).sum()
            )
            if not math.isclose(
                float(projection), server_projection, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{unit_id}: server donor projection does not reproduce from action")
            gain_retrieval = float(winner == source) - float(self_winner == source)
            gain_distance = float(
                (np.linalg.norm(self_value.reshape(-1) - natives[source].reshape(-1))
                 - np.linalg.norm(value.reshape(-1) - natives[source].reshape(-1)))
                / separations[(recipient, source)]
            )
            donor_correct.append(float(winner == source)); retrieval_gain.append(gain_retrieval)
            distance_gain.append(gain_distance)
            for key in vectors:
                vectors[key].append(metrics[key])
            pair_rows.append({
                "task": report["task"], "unit_id": unit_id, "timing_condition": name,
                "recipient_seed": recipient, "source_seed": source,
                "native_separation": separations[(recipient, source)],
                "retrieval_gain": gain_retrieval, "distance_gain": gain_distance,
                "correct_donor": float(winner == source), **metrics,
            })
        timing[name] = {
            "complete_source_retrieval": float(np.mean(complete)),
            "raw_off_diagonal_donor_retrieval": float(np.mean(donor_correct)),
            "matched_retrieval_gain": float(np.mean(retrieval_gain)),
            "matched_distance_gain": float(np.mean(distance_gain)),
            "tie_count": ties, "minimum_top2_margin": float(min(margins)),
            "final_sampler_target_max_abs_error_mean": float(np.mean(max_abs)),
            "final_sampler_target_max_abs_error_max": float(max(max_abs)),
            "final_sampler_target_l2_mean": float(np.mean(residual_l2)),
            "final_sampler_target_l2_max": float(max(residual_l2)),
            **{key: float(np.mean(values)) for key, values in vectors.items()},
        }
    if null_paths(report) != expected_nulls:
        raise ValueError(f"{unit_id}: report null-path topology differs")
    average = {metric: float(np.mean([timing[name][metric] for name in SINGLES])) for metric in PRIMARY}
    sustained = {metric: float(timing["all_calls"][metric] - average[metric]) for metric in PRIMARY}
    return {
        "unit_id": unit_id, "task": report["task"], "environment_seed": report["environment_seed"],
        "timing": timing, "average_single": average, "sustained_minus_single": sustained,
        "minimum_native_separation": float(min(separations.values())), "pair_rows": pair_rows,
    }


def bootstrap_indices(states: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    ids: dict[str, list[str]] = {
        task: [str(state["unit_id"]) for state in states if state["task"] == task] for task in TASKS
    }
    if any(len(values) != 5 or len(set(values)) != 5 for values in ids.values()):
        raise ValueError("each of six tasks must have exactly five distinct states")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return (
        rng.integers(0, 6, size=(BOOTSTRAPS, 6), endpoint=False),
        rng.integers(0, 5, size=(BOOTSTRAPS, 6, 5), endpoint=False),
        ids,
    )


def summarize(values: Mapping[str, float], indices: tuple[np.ndarray, np.ndarray, dict[str, list[str]]]) -> dict[str, Any]:
    task_draws, state_draws, ids = indices
    matrix = np.asarray([[finite(values[unit_id], "state estimand") for unit_id in ids[task]] for task in TASKS])
    task_means = matrix.mean(axis=1); point = float(task_means.mean())
    selected = matrix[task_draws[:, :, None], state_draws]
    boot = selected.mean(axis=2).mean(axis=1)
    return {
        "mean": point, "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "one_sided_null_centered_p": float((1 + np.count_nonzero((boot - point) >= point)) / (BOOTSTRAPS + 1)),
        "bootstrap_samples": BOOTSTRAPS,
        "task_means": {task: float(task_means[index]) for index, task in enumerate(TASKS)},
    }


def leave_out(states: list[dict[str, Any]], values: Mapping[str, float]) -> dict[str, float]:
    output: dict[str, float] = {}
    for held in TASKS:
        output[held] = float(np.mean([
            np.mean([values[str(state["unit_id"])] for state in states if state["task"] == task])
            for task in TASKS if task != held
        ]))
    return output


def holm(raw: Mapping[str, float]) -> dict[str, dict[str, float | bool]]:
    ordered = sorted((float(value), name) for name, value in raw.items())
    running = 0.0; stop = False; output: dict[str, dict[str, float | bool]] = {}
    for rank, (value, name) in enumerate(ordered):
        running = max(running, min(1.0, (4 - rank) * value))
        threshold = 0.05 / (4 - rank); rejected = bool(not stop and value <= threshold)
        if not rejected: stop = True
        output[name] = {"raw_p": value, "holm_adjusted_p": running, "step_threshold": threshold, "rejected": rejected}
    return output


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)), "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)), "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def csv_rows(path: Path, expected_hash: str) -> list[dict[str, str]]:
    require_sha(path, expected_hash, path.name)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_csv(computed: list[dict[str, Any]], reference: list[dict[str, str]], label: str) -> list[str]:
    if len(computed) != len(reference):
        return [f"{label}: row count {len(reference)} != {len(computed)}"]
    problems: list[str] = []
    for index, (left, right) in enumerate(zip(computed, reference, strict=True)):
        if set(left) != set(right):
            problems.append(f"{label}[{index}]: columns differ")
            continue
        for key, expected in left.items():
            actual = right[key]
            if expected is None:
                if actual not in ("", "None"):
                    problems.append(f"{label}[{index}].{key}: expected empty/null")
            elif isinstance(expected, bool):
                if actual != str(expected): problems.append(f"{label}[{index}].{key}: Boolean differs")
            elif isinstance(expected, (int, float)):
                try: parsed = float(actual)
                except ValueError:
                    problems.append(f"{label}[{index}].{key}: not numeric"); continue
                if not math.isclose(float(expected), parsed, rel_tol=1e-12, abs_tol=1e-12):
                    problems.append(f"{label}[{index}].{key}: {expected!r} != {actual!r}")
            elif str(expected) != actual:
                problems.append(f"{label}[{index}].{key}: {expected!r} != {actual!r}")
    return problems


def main() -> None:
    cli = parse_args()
    manifest, paths, _inventory = verify_frozen_package(
        manifest_path=cli.manifest, expected_manifest_sha256=cli.expected_manifest_sha256,
        output_root=cli.run_root, inventory_path=cli.inventory,
        expected_inventory_sha256=cli.expected_inventory_sha256, expected_count=30,
        expected_inventory_schema="cosmos3-timing-output-inventory-v1",
    )
    if manifest.get("study_name") != "cosmos3-single-call-timing-v5":
        raise ValueError("manifest is not timing v5")
    by_name = {path.name: path for path in paths}
    reports: list[dict[str, Any]] = []; states: list[dict[str, Any]] = []
    for unit in manifest["states"]:
        report = load_json(by_name[f"{unit['unit_id']}.json"])
        reports.append(report)
        states.append(derive_state(report, unit, manifest, cli.expected_manifest_sha256))
    if sum(int(report["request_count"]) for report in reports) != 3240:
        raise ValueError("cohort does not contain exactly 3240 requests")
    indices = bootstrap_indices(states)
    timing: dict[str, dict[str, dict[str, Any]]] = {}
    state_rows: list[dict[str, Any]] = []; per_task_rows: list[dict[str, Any]] = []; loto_rows: list[dict[str, Any]] = []
    for state in states:
        row: dict[str, Any] = {
            "unit_id": state["unit_id"], "task": state["task"],
            "environment_seed": state["environment_seed"],
            "minimum_native_separation": state["minimum_native_separation"],
        }
        for name, _ in CONDITIONS:
            for metric, value in state["timing"][name].items(): row[f"{name}__{metric}"] = value
        for group in ("average_single", "sustained_minus_single"):
            for metric, value in state[group].items(): row[f"{group}__{metric}"] = value
        state_rows.append(row)
    for group, labels, metrics in (
        ("timing", [name for name, _ in CONDITIONS], TIMING_METRICS),
        ("contrast", ["average_single", "sustained_minus_single"], PRIMARY),
    ):
        for label in labels:
            timing.setdefault(label, {})
            for metric in metrics:
                values = {str(state["unit_id"]): float(state[label][metric] if label in ("average_single", "sustained_minus_single") else state["timing"][label][metric]) for state in states}
                summary = summarize(values, indices); timing[label][metric] = summary
                for task, value in summary["task_means"].items():
                    per_task_rows.append({"group": group, "label": label, "metric": metric, "task": task, "value": value})
                for task, value in leave_out(states, values).items():
                    loto_rows.append({"group": group, "label": label, "metric": metric, "held_out_task": task, "value": value})
    local_tests: dict[str, Any] = {}
    composite = {name: max(timing[name][metric]["one_sided_null_centered_p"] for metric in PRIMARY) for name in SINGLES}
    adjusted = holm(composite)
    for name in SINGLES:
        local_tests[name] = {
            "component_p_values": {metric: timing[name][metric]["one_sided_null_centered_p"] for metric in PRIMARY},
            "conjunctive_raw_p": composite[name], "holm": adjusted[name],
        }
    all_pairs = [row for state in states for row in state["pair_rows"]]
    all_calls_pairs = [row for row in all_pairs if row["timing_condition"] == "all_calls"]
    if len(all_calls_pairs) != 360:
        raise ValueError("all-calls separation reference does not contain 360 pairs")
    boundaries = np.quantile([row["native_separation"] for row in all_calls_pairs], [0.25, 0.5, 0.75])
    pair_counts: dict[str, int] = {}; state_counts: dict[str, int] = {}; task_counts: dict[str, int] = {}
    all_call_bins = np.searchsorted(boundaries, [row["native_separation"] for row in all_calls_pairs], side="right")
    for index in range(4):
        key = f"q{index + 1}"; pair_counts[key] = int(np.count_nonzero(all_call_bins == index))
        state_counts[key] = len({row["unit_id"] for row, value in zip(all_calls_pairs, all_call_bins, strict=True) if value == index})
        task_counts[key] = len({row["task"] for row, value in zip(all_calls_pairs, all_call_bins, strict=True) if value == index})
    quartiles = {
        "boundaries": [float(value) for value in boundaries], "pair_counts": pair_counts,
        "state_counts": state_counts, "task_counts": task_counts,
        "boundary_assignment": "numpy.searchsorted(boundaries, value, side='right')",
    }
    quartile_rows: list[dict[str, Any]] = []
    for row in all_pairs:
        q = int(np.searchsorted(boundaries, float(row["native_separation"]), side="right")) + 1
        quartile_rows.append({**row, "quartile": f"q{q}"})
    quartile_summaries: list[dict[str, Any]] = []
    for name, _ in CONDITIONS:
        for q in ("q1", "q2", "q3", "q4"):
            selected = [row for row in quartile_rows if row["timing_condition"] == name and row["quartile"] == q]
            quartile_summaries.append({
                "timing_condition": name, "quartile": q, "pair_count": len(selected),
                "state_count": len({row["unit_id"] for row in selected}),
                "task_count": len({row["task"] for row in selected}),
                "retrieval_gain_mean": float(np.mean([row["retrieval_gain"] for row in selected])) if selected else None,
                "distance_gain_mean": float(np.mean([row["distance_gain"] for row in selected])) if selected else None,
            })
    residuals = [
        {"unit_id": report["unit_id"], "task": report["task"], "environment_seed": report["environment_seed"],
         "timing_condition": row["timing_condition"], "recipient_seed": row["recipient_seed"], "source_seed": row["source_seed"],
         "final_sampler_target_max_abs_error": row["final_sampler_target_max_abs_error"],
         "final_sampler_target_l2": row["final_sampler_target_l2"]}
        for report in reports for row in report["timing_rows"]
    ]
    residual_distributions = {
        name: {metric: distribution([float(row[metric]) for row in residuals if row["timing_condition"] == name])
               for metric in ("final_sampler_target_max_abs_error", "final_sampler_target_l2")}
        for name, _ in CONDITIONS
    }
    controls = {
        key: operation(report[key] for report in reports)
        for key, operation in (
            ("native_replay_max_action_error", max), ("all_calls_diagonal_replay_max_action_error", max),
            ("none_noop_max_action_error", max), ("none_source_invariance_max_action_error", max),
            ("maximum_action_input_error", max), ("maximum_action_output_error", max),
            ("maximum_active_model_input_future_clamp_error", max),
            ("maximum_active_returned_future_velocity_error", max),
            ("inactive_wrapper_write_count", sum), ("structural_projection_null_count", sum),
            ("finite_off_diagonal_projection_count", sum), ("native_projection_absent_count", sum),
            ("shape_valid_response_action_count", sum), ("action_shape_failure_count", sum),
        )
    }
    controls.update({
        "minimum_native_separation": min(state["minimum_native_separation"] for state in states),
        "schedule_and_index_gate_exact": all(report["schedule_and_index_gate_exact"] for report in reports),
        "target_hash_gate_exact": all(report["target_hash_gate_exact"] for report in reports),
        "rng_hash_gate_exact": all(report["rng_hash_gate_exact"] for report in reports),
        "replay_signature_gate_exact": all(report["replay_signature_gate_exact"] for report in reports),
    })
    primary_pass = all(timing["average_single"][metric]["ci95_low"] > 0 for metric in PRIMARY)
    sustained_pass = all(timing["sustained_minus_single"][metric]["ci95_low"] > 0 for metric in PRIMARY)
    computed = {
        "audit": {
            "state_count": 30, "request_count": 3240, "expected_state_count": 30,
            "expected_request_count": 3240, "missing_state_count": 0, "unexpected_state_count": 0,
            "required_numeric_nonfinite_count": 0, "structural_projection_null_count": 840,
            "finite_off_diagonal_projection_count": 2160, "native_projection_absent_count": 240,
            "action_shape": [32, 8], "action_coordinate_count": 256,
            "shape_valid_response_action_count": 3240, "action_shape_failure_count": 0,
            "degenerate_axis_count": 0,
        },
        "analysis": {
            "independent_unit": "state", "top_level_clusters": "six equal-weight tasks",
            "bootstrap": "task-to-state hierarchical", "bootstrap_samples": 10000,
            "bootstrap_seed": 20260903,
            "research_sigmas": [0.9990000128746033, 0.9369999766349792, 0.8330000042915344, 0.6240000128746033],
        },
        "timing": timing, "call_local_tests": local_tests, "quartiles": quartiles,
        "quartile_timing_summaries": quartile_summaries,
        "final_sampler_target_residual_distributions": residual_distributions,
        "controls": controls,
        "evidence_gates": {
            "runtime_and_completeness": all(report["runtime_gate"]["passed"] is True for report in reports),
            "average_single_donor_specific_effect": primary_pass,
            "sustained_strength": sustained_pass,
            "timing_local_holm_rejections": {name: bool(local_tests[name]["holm"]["rejected"]) for name in SINGLES},
        },
        "claim_boundary": "imposed action-space timing/strength audit; not natural mediation, physical success, semantic planning, necessity, or an isolated local direct effect",
    }
    reference = load_reference_after_hash(cli.summary_json, cli.expected_summary_sha256)
    discrepancies = compare_tree(computed, reference, path="summary")
    state_ref = csv_rows(cli.states_csv, cli.expected_states_sha256)
    pair_ref = csv_rows(cli.pairs_csv, cli.expected_pairs_sha256)
    per_task_ref = csv_rows(cli.per_task_csv, cli.expected_per_task_sha256)
    loto_ref = csv_rows(cli.loto_csv, cli.expected_loto_sha256)
    discrepancies.extend(compare_csv(state_rows, state_ref, "states_csv"))
    discrepancies.extend(compare_csv(quartile_rows, pair_ref, "pairs_csv"))
    discrepancies.extend(compare_csv(per_task_rows, per_task_ref, "per_task_csv"))
    discrepancies.extend(compare_csv(loto_rows, loto_ref, "loto_csv"))
    result = {
        "status": "pass" if not discrepancies else "fail",
        "audit": "independent-cosmos3-timing-v5-raw-output-audit-v1",
        "manifest_id": manifest["manifest_id"], "manifest_sha256": cli.expected_manifest_sha256,
        "inventory_sha256": cli.expected_inventory_sha256, "summary_sha256": cli.expected_summary_sha256,
        "outcome_file_count": len(paths), "request_count": 3240,
        "raw_action_metrics_recomputed": True, "frozen_analyzer_or_helpers_imported": False,
        "computed": computed, "leave_one_task_out": loto_rows,
        "state_row_count": len(state_rows), "pair_row_count": len(quartile_rows),
        "comparison_discrepancies": discrepancies,
    }
    atomic_json_no_overwrite(cli.output, result)
    print(json.dumps({"status": result["status"], "output": str(cli.output), "discrepancy_count": len(discrepancies)}, sort_keys=True))
    if discrepancies: raise SystemExit(2)


if __name__ == "__main__":
    main()
