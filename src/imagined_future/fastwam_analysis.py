"""Deterministic, state-unit analysis for the frozen FastWAM smoke study.

The analyzer deliberately separates completeness auditing from outcome loading.
It will not calculate a scale decision from a partial set of states or arms.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fastwam_optional_idm import (
    CORE_CONDITIONS,
    FastWAMCondition,
    FastWAMRunSpec,
    action_metrics,
    atomic_write_json,
    expand_run_specs,
    load_frozen_manifest,
    state_from_dict,
)


DIRECTIONAL_CONDITIONS = (
    FastWAMCondition.FIRST_FRAME.value,
    FastWAMCondition.WRONG_LATENT.value,
    FastWAMCondition.SHUFFLED_CACHE.value,
    FastWAMCondition.DONOR_LATENT.value,
    FastWAMCondition.DONOR_CACHE.value,
)

DIRECTIONAL_METRICS = (
    "correct_donor_retrieval_rate",
    "donor_distance_reduction",
    "donor_projection",
    "cosine_alignment",
    "orthogonal_residual",
    "orthogonal_residual_ratio",
    "distance_to_donor",
)

COMPARISONS = (
    ("donor_latent_minus_wrong_latent", "donor_latent", "wrong_latent"),
    ("donor_cache_minus_shuffled_cache", "donor_cache", "shuffled_cache"),
)

SOURCE_GRID_CONDITIONS = {
    "latent": (
        FastWAMCondition.SELF_LATENT.value,
        FastWAMCondition.DONOR_LATENT.value,
    ),
    "cache": (
        FastWAMCondition.SELF_CACHE.value,
        FastWAMCondition.DONOR_CACHE.value,
    ),
}

POWERED_ANALYSIS_VERSION = "source-grid-v2"


def atomic_write_text(path: str | Path, value: str) -> None:
    """Write a text artifact atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(
    path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    atomic_write_text(path, buffer.getvalue())


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _result_root(output_root: Path, manifest_id: str) -> Path:
    output_root = output_root.resolve()
    if output_root.name == manifest_id:
        return output_root
    return output_root / manifest_id


def _spec_matches(recorded: Mapping[str, Any], expected: FastWAMRunSpec) -> bool:
    return dict(recorded) == expected.to_dict()


def audit_fastwam_outputs(
    manifest: Mapping[str, Any],
    output_root: str | Path,
    *,
    required_state_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit the exact frozen matrix without reading any action outcomes."""

    manifest_id = str(manifest["manifest_id"])
    result_root = _result_root(Path(output_root), manifest_id)
    state_rows: list[dict[str, Any]] = []
    malformed_records: list[str] = []
    total_expected = 0
    total_present = 0
    for frozen_state in manifest["states"]:
        state = state_from_dict(frozen_state)
        expected_specs = expand_run_specs(manifest_id, state)
        expected_by_id = {spec.run_id: spec for spec in expected_specs}
        expected_ids = set(expected_by_id)
        state_root = result_root / state.state_id
        run_dir = state_root / "runs"
        json_ids = {path.stem for path in run_dir.glob("run-*.json")}
        npz_ids = {path.stem for path in run_dir.glob("run-*.npz")}
        complete_ids: set[str] = set()
        state_malformed: list[str] = []
        for run_id in sorted(expected_ids & json_ids & npz_ids):
            json_path = run_dir / f"{run_id}.json"
            try:
                record = json.loads(json_path.read_text(encoding="utf-8"))
                valid = (
                    record.get("status") == "complete"
                    and record.get("manifest_id") == manifest_id
                    and _spec_matches(record.get("run", {}), expected_by_id[run_id])
                    and record.get("array_file") == f"{run_id}.npz"
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                valid = False
            if valid:
                complete_ids.add(run_id)
            else:
                state_malformed.append(run_id)
                malformed_records.append(f"{state.state_id}/{run_id}")

        summary_path = state_root / "summary.json"
        summary_valid = False
        summary_problem = "missing"
        if summary_path.exists():
            try:
                state_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary_valid = (
                    state_summary.get("status") == "complete"
                    and state_summary.get("manifest_id") == manifest_id
                    and state_summary.get("state_id") == state.state_id
                    and set(state_summary.get("completed_conditions", []))
                    == set(CORE_CONDITIONS)
                )
                summary_problem = "" if summary_valid else "invalid_or_partial"
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                summary_problem = "unreadable"

        missing_json = sorted(expected_ids - json_ids)
        missing_npz = sorted(expected_ids - npz_ids)
        unexpected_json = sorted(json_ids - expected_ids)
        unexpected_npz = sorted(npz_ids - expected_ids)
        state_complete = (
            complete_ids == expected_ids
            and summary_valid
            and not unexpected_json
            and not unexpected_npz
            and not state_malformed
        )
        state_rows.append(
            {
                "state_id": state.state_id,
                "expected_runs": len(expected_ids),
                "complete_runs": len(complete_ids),
                "missing_json_count": len(missing_json),
                "missing_npz_count": len(missing_npz),
                "malformed_count": len(state_malformed),
                "unexpected_json_count": len(unexpected_json),
                "unexpected_npz_count": len(unexpected_npz),
                "summary_valid": summary_valid,
                "summary_problem": summary_problem,
                "state_complete": state_complete,
                "missing_json_ids": missing_json,
                "missing_npz_ids": missing_npz,
                "malformed_ids": state_malformed,
                "unexpected_json_ids": unexpected_json,
                "unexpected_npz_ids": unexpected_npz,
            }
        )
        total_expected += len(expected_ids)
        total_present += len(complete_ids)

    expected_state_ids = [str(value["state_id"]) for value in manifest["states"]]
    actual_state_ids = sorted(
        path.name
        for path in result_root.iterdir()
        if path.is_dir() and path.name != "analysis"
    ) if result_root.exists() else []
    unexpected_states = sorted(set(actual_state_ids) - set(expected_state_ids))
    complete_state_count = sum(bool(row["state_complete"]) for row in state_rows)
    required_count = (
        len(expected_state_ids) if required_state_count is None else required_state_count
    )
    audit = {
        "manifest_id": manifest_id,
        "result_root": str(result_root),
        "expected_state_count": len(expected_state_ids),
        "required_frozen_state_count": required_count,
        "complete_state_count": complete_state_count,
        "expected_run_count": total_expected,
        "valid_complete_run_count": total_present,
        "unexpected_state_ids": unexpected_states,
        "malformed_records": malformed_records,
        "all_frozen_outputs_complete": (
            len(expected_state_ids) == required_count
            and complete_state_count == required_count
            and total_present == total_expected
            and not unexpected_states
            and not malformed_records
        ),
    }
    return audit, state_rows


def _finite_mean(values: Sequence[Any]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def _seed_for_key(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def state_bootstrap(
    values: Sequence[float | None], *, samples: int, seed: int, key: str
) -> dict[str, Any]:
    """Bootstrap state-level values, explicitly reporting unavailable states."""

    finite = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "n_states_total": len(values),
        "n_states_valid": int(finite.size),
        "n_states_missing": int(len(values) - finite.size),
        "mean": None,
        "ci95_low": None,
        "ci95_high": None,
    }
    if not finite.size:
        return result
    result["mean"] = float(finite.mean())
    if finite.size == 1 or samples <= 0:
        result["ci95_low"] = float(finite.mean())
        result["ci95_high"] = float(finite.mean())
        return result
    rng = np.random.default_rng(_seed_for_key(seed, key))
    indices = rng.integers(0, finite.size, size=(samples, finite.size))
    means = finite[indices].mean(axis=1)
    result["ci95_low"] = float(np.quantile(means, 0.025))
    result["ci95_high"] = float(np.quantile(means, 0.975))
    return result


def hierarchical_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    samples: int,
    seed: int,
    key: str,
) -> dict[str, Any]:
    """Balanced suite->task->state bootstrap for the powered LIBERO grid."""

    suites = sorted({str(row["suite"]) for row in rows})
    grouped: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["suite"]), {}).setdefault(
            int(row["task_id"]), []
        ).append(row)
    task_counts = {len(grouped[suite]) for suite in suites}
    state_counts = {
        len(task_rows)
        for suite in suites
        for task_rows in grouped[suite].values()
    }
    if not suites or len(task_counts) != 1 or len(state_counts) != 1:
        raise ValueError(
            "hierarchical bootstrap requires a nonempty balanced suite/task/state grid"
        )
    task_count = next(iter(task_counts))
    state_count = next(iter(state_counts))
    if task_count <= 0 or state_count <= 0:
        raise ValueError("hierarchical bootstrap found an empty task or state group")
    value_grid = np.full(
        (len(suites), task_count, state_count), np.nan, dtype=np.float64
    )
    for suite_index, suite in enumerate(suites):
        tasks = sorted(grouped[suite])
        if len(tasks) != task_count:
            raise ValueError("hierarchical suite has the wrong number of tasks")
        for task_index, task_id in enumerate(tasks):
            task_rows = sorted(
                grouped[suite][task_id], key=lambda row: str(row["state_id"])
            )
            for state_index, row in enumerate(task_rows):
                value = row.get(value_key)
                if value is not None and np.isfinite(value):
                    value_grid[suite_index, task_index, state_index] = float(value)

    finite = value_grid[np.isfinite(value_grid)]
    result: dict[str, Any] = {
        "n_suites": len(suites),
        "n_tasks": len(suites) * task_count,
        "n_states_total": int(value_grid.size),
        "n_states_valid": int(finite.size),
        "n_states_missing": int(value_grid.size - finite.size),
        "mean": None,
        "ci95_low": None,
        "ci95_high": None,
    }
    if not finite.size:
        return result
    result["mean"] = float(np.nanmean(value_grid))
    if samples <= 0:
        result["ci95_low"] = result["mean"]
        result["ci95_high"] = result["mean"]
        return result

    rng = np.random.default_rng(_seed_for_key(seed, key))
    suite_draw = rng.integers(0, len(suites), size=(samples, len(suites)))
    task_draw = rng.integers(
        0, task_count, size=(samples, len(suites), task_count)
    )
    state_draw = rng.integers(
        0,
        state_count,
        size=(samples, len(suites), task_count, state_count),
    )
    target_shape = (samples, len(suites), task_count, state_count)
    suite_indices = np.broadcast_to(suite_draw[:, :, None, None], target_shape)
    task_indices = np.broadcast_to(task_draw[:, :, :, None], target_shape)
    sampled = value_grid[suite_indices, task_indices, state_draw]
    valid_counts = np.sum(np.isfinite(sampled), axis=(1, 2, 3))
    sums = np.nansum(sampled, axis=(1, 2, 3))
    draw_means = np.divide(
        sums,
        valid_counts,
        out=np.full(samples, np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )
    draw_means = draw_means[np.isfinite(draw_means)]
    if not draw_means.size:
        return result
    result["ci95_low"] = float(np.quantile(draw_means, 0.025))
    result["ci95_high"] = float(np.quantile(draw_means, 0.975))
    return result


def _empty_directional_metrics() -> dict[str, Any]:
    return {
        "axis_degenerate": None,
        "donor_projection": None,
        "cosine_alignment": None,
        "distance_to_recipient": None,
        "distance_to_donor": None,
        "native_recipient_to_donor_distance": None,
        "donor_distance_reduction": None,
        "orthogonal_residual": None,
        "orthogonal_residual_ratio": None,
        "nearest_branch_id": None,
        "correct_donor_retrieval": None,
    }


def _load_complete_matrix(
    manifest: Mapping[str, Any], output_root: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    manifest_id = str(manifest["manifest_id"])
    result_root = _result_root(output_root, manifest_id)
    run_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    source_grid_state_rows: list[dict[str, Any]] = []
    controls_by_state: dict[str, Any] = {}

    for frozen_state in manifest["states"]:
        state = state_from_dict(frozen_state)
        run_dir = result_root / state.state_id / "runs"
        specs = expand_run_specs(manifest_id, state)
        action_by_run: dict[str, np.ndarray] = {}
        metadata_by_run: dict[str, dict[str, Any]] = {}
        video_by_branch: dict[str, np.ndarray] = {}
        finite_all_arrays = True
        invalid_run_ids: list[str] = []
        for spec in specs:
            record = json.loads((run_dir / f"{spec.run_id}.json").read_text(encoding="utf-8"))
            metadata_by_run[spec.run_id] = record
            try:
                with np.load(run_dir / f"{spec.run_id}.npz", allow_pickle=False) as arrays:
                    action = np.asarray(arrays["action_model"], dtype=np.float64)
                    action_env = np.asarray(arrays["action_env"], dtype=np.float64)
                    if spec.condition == FastWAMCondition.NATIVE.value:
                        video = np.asarray(arrays["video_latent"], dtype=np.float64)
                        video_by_branch[spec.recipient_id] = video
            except (OSError, ValueError, KeyError):
                action = np.asarray([np.nan], dtype=np.float64)
                action_env = np.asarray([np.nan], dtype=np.float64)
            if not np.isfinite(action).all() or not np.isfinite(action_env).all():
                finite_all_arrays = False
                invalid_run_ids.append(spec.run_id)
            if spec.condition == FastWAMCondition.NATIVE.value:
                video = video_by_branch.get(spec.recipient_id)
                if video is None or not np.isfinite(video).all():
                    finite_all_arrays = False
                    invalid_run_ids.append(spec.run_id)
            action_by_run[spec.run_id] = action

        native_specs = {
            spec.recipient_id: spec
            for spec in specs
            if spec.condition == FastWAMCondition.NATIVE.value
        }
        native_actions = {
            branch_id: action_by_run[spec.run_id]
            for branch_id, spec in native_specs.items()
        }
        replay_errors = {"self_latent": [], "self_cache": []}
        first_frame_actions: dict[str, list[np.ndarray]] = {}

        for spec in specs:
            action = action_by_run[spec.run_id]
            recipient_action = native_actions[spec.recipient_id]
            row: dict[str, Any] = {
                "manifest_id": manifest_id,
                "state_id": state.state_id,
                "suite": state.suite,
                "task_id": state.task_id,
                "initial_state_index": state.initial_state_index,
                "condition": spec.condition,
                "run_id": spec.run_id,
                "recipient_id": spec.recipient_id,
                "donor_id": spec.donor_id,
                "source_id": spec.source_id,
                "action_seed": spec.action_seed,
                "shuffle_seed": spec.shuffle_seed,
                "array_finite": bool(np.isfinite(action).all()),
                "nearest_source_id": None,
                "correct_source_retrieval": None,
                **_empty_directional_metrics(),
            }
            if spec.condition in replay_errors and row["array_finite"]:
                error = float(np.max(np.abs(action - recipient_action)))
                replay_errors[spec.condition].append(error)
                row["replay_max_abs_error"] = error
            else:
                row["replay_max_abs_error"] = None
            if spec.condition == FastWAMCondition.FIRST_FRAME.value and row["array_finite"]:
                first_frame_actions.setdefault(spec.recipient_id, []).append(action)
            candidates_finite = all(
                np.isfinite(candidate).all() for candidate in native_actions.values()
            )
            shapes_match = all(
                candidate.shape == action.shape for candidate in native_actions.values()
            )
            if (
                row["array_finite"]
                and candidates_finite
                and shapes_match
                and spec.source_id in native_actions
            ):
                source_distances = {
                    branch_id: float(np.linalg.norm(action - candidate))
                    for branch_id, candidate in native_actions.items()
                }
                row["nearest_source_id"] = min(
                    source_distances, key=source_distances.get
                )
                row["correct_source_retrieval"] = (
                    row["nearest_source_id"] == spec.source_id
                )
            if spec.donor_id is not None:
                donor_action = native_actions[spec.donor_id]
                if row["array_finite"] and candidates_finite and shapes_match:
                    metrics = action_metrics(
                        action,
                        recipient_action,
                        donor_action,
                        native_actions,
                        donor_id=spec.donor_id,
                    )
                    for key in _empty_directional_metrics():
                        row[key] = metrics[key]
            run_rows.append(row)

        latent_pairwise: list[float] = []
        native_videos_finite = len(video_by_branch) == len(state.branches)
        if native_videos_finite:
            ordered_videos = [video_by_branch[branch.branch_id] for branch in state.branches]
            native_videos_finite = all(np.isfinite(video).all() for video in ordered_videos)
            if native_videos_finite:
                for left in range(len(ordered_videos)):
                    for right in range(left + 1, len(ordered_videos)):
                        latent_pairwise.append(
                            float(np.linalg.norm(ordered_videos[left] - ordered_videos[right]))
                        )
        first_frame_errors: dict[str, float | None] = {}
        for branch in state.branches:
            actions = first_frame_actions.get(branch.branch_id, [])
            first_frame_errors[branch.branch_id] = (
                float(np.max(np.abs(np.stack(actions) - actions[0])))
                if len(actions) == len(state.branches) - 1
                else None
            )
        state_control = {
            "finite_all_arrays": finite_all_arrays,
            "invalid_run_ids": sorted(set(invalid_run_ids)),
            "self_latent_max_abs_error": (
                max(replay_errors["self_latent"])
                if len(replay_errors["self_latent"]) == len(state.branches)
                else None
            ),
            "self_cache_max_abs_error": (
                max(replay_errors["self_cache"])
                if len(replay_errors["self_cache"]) == len(state.branches)
                else None
            ),
            "first_frame_by_recipient_max_abs_error": first_frame_errors,
            "first_frame_global_max_abs_error": (
                max(float(value) for value in first_frame_errors.values() if value is not None)
                if all(value is not None for value in first_frame_errors.values())
                else None
            ),
            "native_video_latent_pairwise_min_l2": (
                min(latent_pairwise) if latent_pairwise else None
            ),
            "native_video_latent_pairwise_max_l2": (
                max(latent_pairwise) if latent_pairwise else None
            ),
            "native_video_latent_pair_count": len(latent_pairwise),
        }
        controls_by_state[state.state_id] = state_control

        state_run_rows = [row for row in run_rows if row["state_id"] == state.state_id]
        for condition in DIRECTIONAL_CONDITIONS:
            condition_rows = [row for row in state_run_rows if row["condition"] == condition]
            valid_axes = [row for row in condition_rows if row["axis_degenerate"] is False]
            degenerate_axes = [row for row in condition_rows if row["axis_degenerate"] is True]
            invalid_rows = [row for row in condition_rows if row["axis_degenerate"] is None]
            retrieval_values = [
                float(bool(row["correct_donor_retrieval"]))
                for row in condition_rows
                if row["correct_donor_retrieval"] is not None
            ]
            state_rows.append(
                {
                    "state_id": state.state_id,
                    "suite": state.suite,
                    "task_id": state.task_id,
                    "initial_state_index": state.initial_state_index,
                    "condition": condition,
                    "n_rows": len(condition_rows),
                    "valid_axes": len(valid_axes),
                    "degenerate_axes": len(degenerate_axes),
                    "invalid_rows": len(invalid_rows),
                    "correct_donor_retrieval_rate": _finite_mean(retrieval_values),
                    "donor_distance_reduction": _finite_mean(
                        [row["donor_distance_reduction"] for row in valid_axes]
                    ),
                    "donor_projection": _finite_mean(
                        [row["donor_projection"] for row in valid_axes]
                    ),
                    "cosine_alignment": _finite_mean(
                        [row["cosine_alignment"] for row in valid_axes]
                    ),
                    "orthogonal_residual": _finite_mean(
                        [row["orthogonal_residual"] for row in valid_axes]
                    ),
                    "orthogonal_residual_ratio": _finite_mean(
                        [row["orthogonal_residual_ratio"] for row in valid_axes]
                    ),
                    "distance_to_donor": _finite_mean(
                        [row["distance_to_donor"] for row in condition_rows]
                    ),
                }
            )

        for modality, conditions in SOURCE_GRID_CONDITIONS.items():
            grid_rows = [
                row for row in state_run_rows if row["condition"] in conditions
            ]
            valid = [
                row
                for row in grid_rows
                if row["correct_source_retrieval"] is not None
            ]
            source_grid_state_rows.append(
                {
                    "state_id": state.state_id,
                    "suite": state.suite,
                    "task_id": state.task_id,
                    "initial_state_index": state.initial_state_index,
                    "modality": modality,
                    "n_rows": len(grid_rows),
                    "n_valid": len(valid),
                    "n_missing": len(grid_rows) - len(valid),
                    "correct_source_retrieval_rate": _finite_mean(
                        [
                            float(bool(row["correct_source_retrieval"]))
                            for row in valid
                        ]
                    ),
                }
            )

    return run_rows, state_rows, source_grid_state_rows, controls_by_state


def _aggregate_state_rows(
    state_rows: Sequence[Mapping[str, Any]],
    *,
    state_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    by_condition: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    state_ids = sorted({str(row["state_id"]) for row in state_rows})
    if len(state_ids) != state_count:
        raise ValueError(
            f"state-row matrix has {len(state_ids)} states; expected {state_count}"
        )
    for condition in DIRECTIONAL_CONDITIONS:
        rows = [row for row in state_rows if row["condition"] == condition]
        condition_result: dict[str, Any] = {
            "n_states": len(rows),
            "n_rows": sum(int(row["n_rows"]) for row in rows),
            "valid_axes": sum(int(row["valid_axes"]) for row in rows),
            "degenerate_axes": sum(int(row["degenerate_axes"]) for row in rows),
            "invalid_rows": sum(int(row["invalid_rows"]) for row in rows),
        }
        for metric in DIRECTIONAL_METRICS:
            values_by_state = {
                str(row["state_id"]): row.get(metric)
                for row in rows
            }
            ordered_values = [values_by_state.get(state_id) for state_id in state_ids]
            stats = state_bootstrap(
                ordered_values,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                key=f"condition:{condition}:{metric}",
            )
            condition_result[metric] = stats
            flat_rows.append(
                {
                    "bootstrap_mode": "state",
                    "kind": "condition",
                    "label": condition,
                    "metric": metric,
                    **stats,
                }
            )
        by_condition[condition] = condition_result

    comparisons: dict[str, Any] = {}
    row_index = {
        (str(row["state_id"]), str(row["condition"])): row for row in state_rows
    }
    for comparison_name, treatment, control in COMPARISONS:
        comparison_result: dict[str, Any] = {
            "treatment": treatment,
            "control": control,
        }
        for metric in DIRECTIONAL_METRICS:
            differences: list[float | None] = []
            for state_id in state_ids:
                treatment_value = row_index[(state_id, treatment)].get(metric)
                control_value = row_index[(state_id, control)].get(metric)
                if treatment_value is None or control_value is None:
                    differences.append(None)
                else:
                    differences.append(float(treatment_value) - float(control_value))
            stats = state_bootstrap(
                differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                key=f"comparison:{comparison_name}:{metric}",
            )
            comparison_result[metric] = stats
            flat_rows.append(
                {
                    "bootstrap_mode": "state",
                    "kind": "comparison",
                    "label": comparison_name,
                    "metric": metric,
                    **stats,
                }
            )
        comparisons[comparison_name] = comparison_result
    return by_condition, comparisons, flat_rows


def _aggregate_hierarchical(
    state_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    by_condition: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for condition in DIRECTIONAL_CONDITIONS:
        rows = [row for row in state_rows if row["condition"] == condition]
        condition_result: dict[str, Any] = {
            "n_states": len(rows),
            "n_rows": sum(int(row["n_rows"]) for row in rows),
            "valid_axes": sum(int(row["valid_axes"]) for row in rows),
            "degenerate_axes": sum(int(row["degenerate_axes"]) for row in rows),
            "invalid_rows": sum(int(row["invalid_rows"]) for row in rows),
        }
        for metric in DIRECTIONAL_METRICS:
            stats = hierarchical_bootstrap(
                rows,
                value_key=metric,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                key=f"hierarchical:condition:{condition}:{metric}",
            )
            condition_result[metric] = stats
            flat_rows.append(
                {
                    "bootstrap_mode": "suite_task_state_hierarchical",
                    "kind": "condition",
                    "label": condition,
                    "metric": metric,
                    **stats,
                }
            )
        by_condition[condition] = condition_result

    row_index = {
        (str(row["state_id"]), str(row["condition"])): row for row in state_rows
    }
    template_rows = [
        row for row in state_rows if row["condition"] == DIRECTIONAL_CONDITIONS[0]
    ]
    comparisons: dict[str, Any] = {}
    for comparison_name, treatment, control in COMPARISONS:
        comparison_result: dict[str, Any] = {
            "treatment": treatment,
            "control": control,
        }
        for metric in DIRECTIONAL_METRICS:
            difference_rows: list[dict[str, Any]] = []
            for template in template_rows:
                state_id = str(template["state_id"])
                treatment_value = row_index[(state_id, treatment)].get(metric)
                control_value = row_index[(state_id, control)].get(metric)
                difference = (
                    None
                    if treatment_value is None or control_value is None
                    else float(treatment_value) - float(control_value)
                )
                difference_rows.append(
                    {
                        "state_id": state_id,
                        "suite": template["suite"],
                        "task_id": template["task_id"],
                        "difference": difference,
                    }
                )
            stats = hierarchical_bootstrap(
                difference_rows,
                value_key="difference",
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                key=f"hierarchical:comparison:{comparison_name}:{metric}",
            )
            comparison_result[metric] = stats
            flat_rows.append(
                {
                    "bootstrap_mode": "suite_task_state_hierarchical",
                    "kind": "comparison",
                    "label": comparison_name,
                    "metric": metric,
                    **stats,
                }
            )
        comparisons[comparison_name] = comparison_result
    return by_condition, comparisons, flat_rows


def _aggregate_source_grid(
    source_grid_state_rows: Sequence[Mapping[str, Any]],
    *,
    state_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    bootstrap_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate the 4x4 source-identification grid.

    Each state contributes four diagonal self-source cells and twelve
    off-diagonal donor-source cells. With all four labels represented equally,
    exact label-permutation chance is 1/4. The off-diagonal donor-only retrieval
    remains a separate descriptive metric in ``conditions``.
    """

    result: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for modality in SOURCE_GRID_CONDITIONS:
        rows = [row for row in source_grid_state_rows if row["modality"] == modality]
        if len(rows) != state_count:
            raise ValueError(
                f"{modality} source grid has {len(rows)} state rows; expected {state_count}"
            )
        if any(int(row["n_rows"]) != 16 for row in rows):
            raise ValueError(f"{modality} source grid must contain 16 cells per state")
        if bootstrap_mode == "hierarchical":
            stats = hierarchical_bootstrap(
                rows,
                value_key="correct_source_retrieval_rate",
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                key=f"hierarchical:source_grid:{modality}:retrieval",
            )
            mode_label = "suite_task_state_hierarchical"
        else:
            ordered = sorted(rows, key=lambda row: str(row["state_id"]))
            stats = state_bootstrap(
                [row["correct_source_retrieval_rate"] for row in ordered],
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                key=f"source_grid:{modality}:retrieval",
            )
            mode_label = "state"
        modality_result = {
            "n_states": len(rows),
            "n_rows": sum(int(row["n_rows"]) for row in rows),
            "n_valid": sum(int(row["n_valid"]) for row in rows),
            "n_missing": sum(int(row["n_missing"]) for row in rows),
            "chance_rate": 0.25,
            "grid_cells_per_state": 16,
            "correct_source_retrieval_rate": stats,
        }
        result[modality] = modality_result
        flat_rows.append(
            {
                "bootstrap_mode": mode_label,
                "kind": "source_grid",
                "label": modality,
                "metric": "correct_source_retrieval_rate",
                **stats,
            }
        )
    return result, flat_rows


def _aggregate_task_and_suite_rows(
    state_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    suites = sorted({str(row["suite"]) for row in state_rows})
    for suite in suites:
        task_ids = sorted(
            {int(row["task_id"]) for row in state_rows if row["suite"] == suite}
        )
        for task_id in task_ids:
            for condition in DIRECTIONAL_CONDITIONS:
                rows = [
                    row
                    for row in state_rows
                    if row["suite"] == suite
                    and int(row["task_id"]) == task_id
                    and row["condition"] == condition
                ]
                task_rows.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "condition": condition,
                        "n_states": len(rows),
                        "n_rows": sum(int(row["n_rows"]) for row in rows),
                        "valid_axes": sum(int(row["valid_axes"]) for row in rows),
                        "degenerate_axes": sum(
                            int(row["degenerate_axes"]) for row in rows
                        ),
                        "invalid_rows": sum(int(row["invalid_rows"]) for row in rows),
                        **{
                            metric: _finite_mean([row.get(metric) for row in rows])
                            for metric in DIRECTIONAL_METRICS
                        },
                    }
                )
    suite_rows: list[dict[str, Any]] = []
    for suite in suites:
        for condition in DIRECTIONAL_CONDITIONS:
            rows = [
                row
                for row in task_rows
                if row["suite"] == suite and row["condition"] == condition
            ]
            suite_rows.append(
                {
                    "suite": suite,
                    "condition": condition,
                    "n_tasks": len(rows),
                    "n_states": sum(int(row["n_states"]) for row in rows),
                    "n_rows": sum(int(row["n_rows"]) for row in rows),
                    "valid_axes": sum(int(row["valid_axes"]) for row in rows),
                    "degenerate_axes": sum(int(row["degenerate_axes"]) for row in rows),
                    "invalid_rows": sum(int(row["invalid_rows"]) for row in rows),
                    **{
                        metric: _finite_mean([row.get(metric) for row in rows])
                        for metric in DIRECTIONAL_METRICS
                    },
                }
            )
    return task_rows, suite_rows


def _leave_one_task_out_rows(
    state_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Descriptive sensitivity analysis excluding each task in turn."""

    tasks = sorted(
        {(str(row["suite"]), int(row["task_id"])) for row in state_rows}
    )
    results: list[dict[str, Any]] = []
    for held_out_suite, held_out_task_id in tasks:
        for condition in DIRECTIONAL_CONDITIONS:
            rows = [
                row
                for row in state_rows
                if row["condition"] == condition
                and not (
                    str(row["suite"]) == held_out_suite
                    and int(row["task_id"]) == held_out_task_id
                )
            ]
            results.append(
                {
                    "held_out_suite": held_out_suite,
                    "held_out_task_id": held_out_task_id,
                    "condition": condition,
                    "remaining_states": len(rows),
                    "remaining_rows": sum(int(row["n_rows"]) for row in rows),
                    "valid_axes": sum(int(row["valid_axes"]) for row in rows),
                    "degenerate_axes": sum(
                        int(row["degenerate_axes"]) for row in rows
                    ),
                    "invalid_rows": sum(int(row["invalid_rows"]) for row in rows),
                    **{
                        metric: _finite_mean([row.get(metric) for row in rows])
                        for metric in DIRECTIONAL_METRICS
                    },
                }
            )
    return results


def _control_aggregate(controls_by_state: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def max_or_none(key: str) -> float | None:
        values = [value.get(key) for value in controls_by_state.values()]
        if not values or any(value is None for value in values):
            return None
        return max(float(value) for value in values)

    latent_mins = [
        value.get("native_video_latent_pairwise_min_l2")
        for value in controls_by_state.values()
    ]
    return {
        "all_arrays_finite": all(
            bool(value["finite_all_arrays"]) for value in controls_by_state.values()
        ),
        "self_latent_global_max_abs_error": max_or_none("self_latent_max_abs_error"),
        "self_cache_global_max_abs_error": max_or_none("self_cache_max_abs_error"),
        "first_frame_global_max_abs_error": max_or_none(
            "first_frame_global_max_abs_error"
        ),
        "native_video_latent_global_min_pairwise_l2": (
            min(float(value) for value in latent_mins if value is not None)
            if latent_mins and all(value is not None for value in latent_mins)
            else None
        ),
    }


def _gate_value(aggregate: Mapping[str, Any], condition: str, metric: str) -> float | None:
    value = aggregate[condition][metric]["mean"]
    return None if value is None else float(value)


def _scale_gate(
    *,
    audit: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    controls: Mapping[str, Any],
    replay_tolerance: float,
    latent_distinct_tolerance: float,
) -> dict[str, Any]:
    latent_retrieval = _gate_value(
        aggregate, "donor_latent", "correct_donor_retrieval_rate"
    )
    cache_retrieval = _gate_value(
        aggregate, "donor_cache", "correct_donor_retrieval_rate"
    )
    donor_latent_attraction = _gate_value(
        aggregate, "donor_latent", "donor_distance_reduction"
    )
    wrong_latent_attraction = _gate_value(
        aggregate, "wrong_latent", "donor_distance_reduction"
    )
    donor_cache_attraction = _gate_value(
        aggregate, "donor_cache", "donor_distance_reduction"
    )
    shuffled_cache_attraction = _gate_value(
        aggregate, "shuffled_cache", "donor_distance_reduction"
    )
    replay_values = (
        controls.get("self_latent_global_max_abs_error"),
        controls.get("self_cache_global_max_abs_error"),
    )
    criteria = {
        "all_eight_frozen_states_and_arms_complete": bool(
            audit["all_frozen_outputs_complete"]
        ),
        "all_arrays_finite": bool(controls.get("all_arrays_finite")),
        "self_latent_and_cache_replay_within_tolerance": (
            all(value is not None for value in replay_values)
            and max(float(value) for value in replay_values) <= replay_tolerance
        ),
        "native_video_latents_nonidentical": (
            controls.get("native_video_latent_global_min_pairwise_l2") is not None
            and float(controls["native_video_latent_global_min_pairwise_l2"])
            > latent_distinct_tolerance
        ),
        "donor_latent_retrieval_above_four_way_chance": (
            latent_retrieval is not None and latent_retrieval > 0.25
        ),
        "donor_cache_retrieval_above_four_way_chance": (
            cache_retrieval is not None and cache_retrieval > 0.25
        ),
        "donor_latent_attraction_exceeds_wrong_latent": (
            donor_latent_attraction is not None
            and wrong_latent_attraction is not None
            and donor_latent_attraction > wrong_latent_attraction
        ),
        "donor_cache_attraction_exceeds_shuffled_cache": (
            donor_cache_attraction is not None
            and shuffled_cache_attraction is not None
            and donor_cache_attraction > shuffled_cache_attraction
        ),
        "first_frame_invariant_to_donor_video_seed": (
            controls.get("first_frame_global_max_abs_error") is not None
            and float(controls["first_frame_global_max_abs_error"])
            <= replay_tolerance
        ),
    }
    passed = all(criteria.values())
    return {
        "passed": passed,
        "decision": "scale" if passed else "do_not_scale",
        "criteria": criteria,
        "thresholds": {
            "required_frozen_states": 8,
            "four_way_retrieval_chance": 0.25,
            "replay_max_abs_tolerance": replay_tolerance,
            "native_video_latent_min_pairwise_l2": latent_distinct_tolerance,
            "attraction_metric": "state-mean donor_distance_reduction",
        },
    }


def _powered_evidence_gate(
    *,
    audit: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    source_grid: Mapping[str, Any],
    controls: Mapping[str, Any],
    replay_tolerance: float,
    latent_distinct_tolerance: float,
) -> dict[str, Any]:
    def lower(condition: str, metric: str) -> float | None:
        value = aggregate[condition][metric]["ci95_low"]
        return None if value is None else float(value)

    def comparison_lower(comparison: str, metric: str) -> float | None:
        value = comparisons[comparison][metric]["ci95_low"]
        return None if value is None else float(value)

    replay_values = (
        controls.get("self_latent_global_max_abs_error"),
        controls.get("self_cache_global_max_abs_error"),
    )
    latent_source_grid_lower = source_grid["latent"][
        "correct_source_retrieval_rate"
    ]["ci95_low"]
    cache_source_grid_lower = source_grid["cache"][
        "correct_source_retrieval_rate"
    ]["ci95_low"]
    latent_retrieval_contrast = comparison_lower(
        "donor_latent_minus_wrong_latent", "correct_donor_retrieval_rate"
    )
    latent_distance_contrast = comparison_lower(
        "donor_latent_minus_wrong_latent", "donor_distance_reduction"
    )
    cache_retrieval_contrast = comparison_lower(
        "donor_cache_minus_shuffled_cache", "correct_donor_retrieval_rate"
    )
    cache_distance_contrast = comparison_lower(
        "donor_cache_minus_shuffled_cache", "donor_distance_reduction"
    )
    criteria = {
        "all_registered_states_and_arms_complete": bool(
            audit["all_frozen_outputs_complete"]
        ),
        "all_arrays_finite": bool(controls.get("all_arrays_finite")),
        "self_latent_and_cache_replay_within_tolerance": (
            all(value is not None for value in replay_values)
            and max(float(value) for value in replay_values) <= replay_tolerance
        ),
        "native_video_latents_nonidentical": (
            controls.get("native_video_latent_global_min_pairwise_l2") is not None
            and float(controls["native_video_latent_global_min_pairwise_l2"])
            > latent_distinct_tolerance
        ),
        "first_frame_invariant_to_donor_video_seed": (
            controls.get("first_frame_global_max_abs_error") is not None
            and float(controls["first_frame_global_max_abs_error"])
            <= replay_tolerance
        ),
        "latent_4x4_source_retrieval_hierarchical_lower_above_chance": (
            latent_source_grid_lower is not None
            and float(latent_source_grid_lower) > 0.25
        ),
        "cache_4x4_source_retrieval_hierarchical_lower_above_chance": (
            cache_source_grid_lower is not None
            and float(cache_source_grid_lower) > 0.25
        ),
        "latent_minus_wrong_retrieval_lower_above_zero": (
            latent_retrieval_contrast is not None and latent_retrieval_contrast > 0
        ),
        "latent_minus_wrong_distance_reduction_lower_above_zero": (
            latent_distance_contrast is not None and latent_distance_contrast > 0
        ),
        "cache_minus_shuffle_retrieval_lower_above_zero": (
            cache_retrieval_contrast is not None and cache_retrieval_contrast > 0
        ),
        "cache_minus_shuffle_distance_reduction_lower_above_zero": (
            cache_distance_contrast is not None and cache_distance_contrast > 0
        ),
    }
    passed = all(criteria.values())
    return {
        "passed": passed,
        "decision": (
            "powered_external_replication_criteria_met"
            if passed
            else "powered_external_replication_criteria_not_met"
        ),
        "criteria": criteria,
        "thresholds": {
            "four_by_four_source_grid_label_permutation_chance": 0.25,
            "donor_only_three_label_reference_rate": 1.0 / 3.0,
            "donor_only_retrieval_role": (
                "secondary; primary gate uses the balanced 16-cell source grid and "
                "donor-only contrasts use the empirical wrong/shuffled controls"
            ),
            "population_interval": "suite-task-state hierarchical bootstrap 95%",
            "bootstrap_lower_bound_rule": "strictly greater than threshold",
            "replay_max_abs_tolerance": replay_tolerance,
            "native_video_latent_min_pairwise_l2": latent_distinct_tolerance,
        },
    }


def _format_ci(stats: Mapping[str, Any]) -> str:
    if stats.get("mean") is None:
        return "--"
    return (
        f"{float(stats['mean']):.3f} "
        f"[{float(stats['ci95_low']):.3f}, {float(stats['ci95_high']):.3f}]"
    )


def render_latex_table(
    aggregate: Mapping[str, Any],
    controls: Mapping[str, Any],
    source_grid: Mapping[str, Any],
) -> str:
    labels = {
        "first_frame": "First-frame control",
        "wrong_latent": "Wrong future latent",
        "shuffled_cache": "Shuffled donor cache",
        "donor_latent": "Correct future latent",
        "donor_cache": "Correct future cache",
    }
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Future source grid & Correct-source retrieval $\uparrow$ \\",
        r"\midrule",
        f"Future latent (4$\\times$4) & {_format_ci(source_grid['latent']['correct_source_retrieval_rate'])} \\\\",
        f"Video cache (4$\\times$4) & {_format_ci(source_grid['cache']['correct_source_retrieval_rate'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"% Exact label-permutation chance for each balanced 4x4 grid is 0.25.",
        "",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Condition & Donor-only retrieval $\uparrow$ & $\Delta d$ $\uparrow$ & Projection $\uparrow$ & Orth. ratio $\downarrow$ & Valid axes \\",
        r"\midrule",
    ]
    for condition in DIRECTIONAL_CONDITIONS:
        result = aggregate[condition]
        lines.append(
            f"{labels[condition]} & "
            f"{_format_ci(result['correct_donor_retrieval_rate'])} & "
            f"{_format_ci(result['donor_distance_reduction'])} & "
            f"{_format_ci(result['donor_projection'])} & "
            f"{_format_ci(result['orthogonal_residual_ratio'])} & "
            f"{result['valid_axes']}/{result['n_rows']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            "% Control maxima across all registered frozen states:",
            f"% self-latent replay = {controls['self_latent_global_max_abs_error']}",
            f"% self-cache replay = {controls['self_cache_global_max_abs_error']}",
            f"% first-frame donor-seed invariance = {controls['first_frame_global_max_abs_error']}",
            f"% native latent minimum pairwise L2 = {controls['native_video_latent_global_min_pairwise_l2']}",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix
    )
    os.close(fd)
    try:
        figure.savefig(temporary_name, dpi=220, bbox_inches="tight", facecolor="white")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def render_compact_plot(
    state_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    destination: str | Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(DIRECTIONAL_CONDITIONS)
    labels = ["First\nframe", "Wrong\nlatent", "Shuffled\ncache", "Donor\nlatent", "Donor\ncache"]
    colors = ["#9CA3AF", "#B8B8B8", "#777777", "#E6862A", "#2A9D8F"]
    panels = (
        ("correct_donor_retrieval_rate", "Correct-donor retrieval", 0.25),
        ("donor_distance_reduction", "Distance reduction", 0.0),
        ("donor_projection", "Donor-axis projection", 0.0),
        ("orthogonal_residual_ratio", "Orthogonal residual / axis", 0.0),
    )
    state_ids = sorted({str(row["state_id"]) for row in state_rows})
    index = {
        (str(row["state_id"]), str(row["condition"])): row for row in state_rows
    }
    figure, axes = plt.subplots(2, 2, figsize=(8.1, 5.4), constrained_layout=True)
    for axis, (metric, title, baseline) in zip(axes.flat, panels, strict=True):
        for condition_index, (condition, color) in enumerate(zip(conditions, colors, strict=True)):
            values = [index[(state_id, condition)].get(metric) for state_id in state_ids]
            finite_values = np.asarray(
                [float(value) for value in values if value is not None and np.isfinite(value)],
                dtype=np.float64,
            )
            if finite_values.size:
                offsets = np.linspace(-0.105, 0.105, finite_values.size)
                axis.scatter(
                    condition_index + offsets,
                    finite_values,
                    s=15,
                    color=color,
                    alpha=0.58,
                    linewidths=0,
                    zorder=2,
                )
            stats = aggregate[condition][metric]
            if stats["mean"] is not None:
                mean = float(stats["mean"])
                low = float(stats["ci95_low"])
                high = float(stats["ci95_high"])
                axis.errorbar(
                    condition_index,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="o",
                    markersize=5.5,
                    capsize=2.5,
                    color="#111827",
                    markerfacecolor=color,
                    markeredgewidth=0.8,
                    zorder=3,
                )
        axis.axhline(baseline, color="#6B7280", linewidth=0.9, linestyle="--", zorder=1)
        axis.set_title(title, fontsize=10, loc="left", fontweight="semibold")
        axis.set_xticks(range(len(conditions)), labels, fontsize=7.5)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylim(-0.03, 1.03)
    figure.suptitle(title, fontsize=11, fontweight="semibold")
    _atomic_save_figure(figure, Path(destination))
    plt.close(figure)


def analyze_fastwam_smoke(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    summary_dir: str | Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_903,
    replay_tolerance: float = 1e-6,
    latent_distinct_tolerance: float = 1e-6,
    required_state_count: int | None = None,
    bootstrap_mode: str = "state",
    gate_mode: str = "smoke_scale",
    make_plot: bool = True,
) -> tuple[dict[str, Any], int]:
    """Analyze a complete frozen FastWAM matrix.

    Returns ``(report, exit_code)``. Exit code 2 means the frozen matrix is
    incomplete; exit code 1 means complete but its frozen gate failed.
    """

    manifest = load_frozen_manifest(manifest_path)
    if bootstrap_mode not in {"state", "hierarchical"}:
        raise ValueError("bootstrap_mode must be 'state' or 'hierarchical'")
    if gate_mode not in {"smoke_scale", "powered_evidence"}:
        raise ValueError("gate_mode must be 'smoke_scale' or 'powered_evidence'")
    if gate_mode == "powered_evidence" and bootstrap_mode != "hierarchical":
        raise ValueError("powered_evidence gate requires hierarchical bootstrap")
    summary_root = Path(summary_dir)
    summary_root.mkdir(parents=True, exist_ok=True)
    audit, missingness_rows = audit_fastwam_outputs(
        manifest,
        output_root,
        required_state_count=required_state_count,
    )
    atomic_write_json(summary_root / "fastwam_completeness.json", audit)
    missingness_fields = (
        "state_id",
        "expected_runs",
        "complete_runs",
        "missing_json_count",
        "missing_npz_count",
        "malformed_count",
        "unexpected_json_count",
        "unexpected_npz_count",
        "summary_valid",
        "summary_problem",
        "state_complete",
        "missing_json_ids",
        "missing_npz_ids",
        "malformed_ids",
        "unexpected_json_ids",
        "unexpected_npz_ids",
    )
    atomic_write_csv(
        summary_root / "fastwam_missingness.csv", missingness_rows, missingness_fields
    )
    if not audit["all_frozen_outputs_complete"]:
        unavailable_gate = {
            "passed": False,
            "decision": "unavailable_incomplete_matrix",
            "criteria": {"all_registered_states_and_arms_complete": False},
        }
        report = {
            "manifest_id": manifest["manifest_id"],
            "status": "incomplete",
            "audit": audit,
            "gate": unavailable_gate,
            "scale_gate": unavailable_gate,
        }
        atomic_write_json(summary_root / "fastwam_results.json", report)
        return report, 2

    run_rows, state_rows, source_grid_state_rows, controls_by_state = _load_complete_matrix(
        manifest, Path(output_root)
    )
    state_aggregate, state_comparisons, state_aggregate_rows = _aggregate_state_rows(
        state_rows,
        state_count=len(manifest["states"]),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    task_rows, suite_rows = _aggregate_task_and_suite_rows(state_rows)
    leave_one_task_out_rows = _leave_one_task_out_rows(state_rows)
    hierarchical_aggregate: dict[str, Any] | None = None
    hierarchical_comparisons: dict[str, Any] | None = None
    hierarchical_rows: list[dict[str, Any]] = []
    if bootstrap_mode == "hierarchical":
        (
            hierarchical_aggregate,
            hierarchical_comparisons,
            hierarchical_rows,
        ) = _aggregate_hierarchical(
            state_rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        aggregate = hierarchical_aggregate
        comparisons = hierarchical_comparisons
    else:
        aggregate = state_aggregate
        comparisons = state_comparisons
    source_grid, source_grid_aggregate_rows = _aggregate_source_grid(
        source_grid_state_rows,
        state_count=len(manifest["states"]),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_mode=bootstrap_mode,
    )
    controls = _control_aggregate(controls_by_state)
    if gate_mode == "powered_evidence":
        gate = _powered_evidence_gate(
            audit=audit,
            aggregate=aggregate,
            comparisons=comparisons,
            source_grid=source_grid,
            controls=controls,
            replay_tolerance=replay_tolerance,
            latent_distinct_tolerance=latent_distinct_tolerance,
        )
    else:
        gate = _scale_gate(
            audit=audit,
            aggregate=aggregate,
            controls=controls,
            replay_tolerance=replay_tolerance,
            latent_distinct_tolerance=latent_distinct_tolerance,
        )
    report = {
        "manifest_id": manifest["manifest_id"],
        "status": "complete",
        "analysis": {
            "version": POWERED_ANALYSIS_VERSION,
            "independent_unit": "state",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_mode": bootstrap_mode,
            "gate_mode": gate_mode,
            "directional_axes": (
                "Degenerate native recipient-to-donor axes are retained, counted, "
                "and reported; only mathematically undefined directional metrics "
                "are absent from their means."
            ),
        },
        "audit": audit,
        "controls_by_state": controls_by_state,
        "controls": controls,
        "conditions": aggregate,
        "paired_state_comparisons": comparisons,
        "source_grid_retrieval": source_grid,
        "state_bootstrap_conditions": state_aggregate,
        "state_bootstrap_comparisons": state_comparisons,
        "hierarchical_conditions": hierarchical_aggregate,
        "hierarchical_comparisons": hierarchical_comparisons,
        "leave_one_task_out": leave_one_task_out_rows,
        "gate": gate,
        "scale_gate": gate,
    }
    atomic_write_json(summary_root / "fastwam_results.json", report)
    run_fields = (
        "manifest_id",
        "state_id",
        "suite",
        "task_id",
        "initial_state_index",
        "condition",
        "run_id",
        "recipient_id",
        "donor_id",
        "source_id",
        "action_seed",
        "shuffle_seed",
        "array_finite",
        "replay_max_abs_error",
        "axis_degenerate",
        "correct_donor_retrieval",
        "nearest_branch_id",
        "nearest_source_id",
        "correct_source_retrieval",
        "donor_distance_reduction",
        "donor_projection",
        "cosine_alignment",
        "orthogonal_residual",
        "orthogonal_residual_ratio",
        "distance_to_recipient",
        "distance_to_donor",
        "native_recipient_to_donor_distance",
    )
    atomic_write_csv(summary_root / "fastwam_run_metrics.csv", run_rows, run_fields)
    state_fields = (
        "state_id",
        "suite",
        "task_id",
        "initial_state_index",
        "condition",
        "n_rows",
        "valid_axes",
        "degenerate_axes",
        "invalid_rows",
        *DIRECTIONAL_METRICS,
    )
    atomic_write_csv(summary_root / "fastwam_state_metrics.csv", state_rows, state_fields)
    source_grid_state_fields = (
        "state_id",
        "suite",
        "task_id",
        "initial_state_index",
        "modality",
        "n_rows",
        "n_valid",
        "n_missing",
        "correct_source_retrieval_rate",
    )
    atomic_write_csv(
        summary_root / "fastwam_source_grid_state_metrics.csv",
        source_grid_state_rows,
        source_grid_state_fields,
    )
    level_fields = (
        "suite",
        "task_id",
        "condition",
        "n_tasks",
        "n_states",
        "n_rows",
        "valid_axes",
        "degenerate_axes",
        "invalid_rows",
        *DIRECTIONAL_METRICS,
    )
    atomic_write_csv(
        summary_root / "fastwam_task_metrics.csv", task_rows, level_fields
    )
    atomic_write_csv(
        summary_root / "fastwam_suite_metrics.csv", suite_rows, level_fields
    )
    leave_one_task_out_fields = (
        "held_out_suite",
        "held_out_task_id",
        "condition",
        "remaining_states",
        "remaining_rows",
        "valid_axes",
        "degenerate_axes",
        "invalid_rows",
        *DIRECTIONAL_METRICS,
    )
    atomic_write_csv(
        summary_root / "fastwam_leave_one_task_out_metrics.csv",
        leave_one_task_out_rows,
        leave_one_task_out_fields,
    )
    aggregate_fields = (
        "bootstrap_mode",
        "kind",
        "label",
        "metric",
        "n_states_total",
        "n_states_valid",
        "n_states_missing",
        "mean",
        "ci95_low",
        "ci95_high",
        "n_suites",
        "n_tasks",
    )
    atomic_write_csv(
        summary_root / "fastwam_aggregate_metrics.csv",
        [
            *state_aggregate_rows,
            *hierarchical_rows,
            *source_grid_aggregate_rows,
        ],
        aggregate_fields,
    )
    atomic_write_text(
        summary_root / "fastwam_results.tex",
        render_latex_table(aggregate, controls, source_grid),
    )
    if make_plot:
        render_compact_plot(
            state_rows,
            aggregate,
            summary_root / "fastwam_summary.png",
            title=(
                f"FastWAM Optional-IDM — frozen {len(manifest['states'])}-state study "
                "(points are state means)"
            ),
        )
    return report, 0 if gate["passed"] else 1
