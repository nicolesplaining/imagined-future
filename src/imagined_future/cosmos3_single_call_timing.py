"""Frozen design and analysis helpers for the Cosmos 3 timing audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASKS = (
    "BananaInBowlTask",
    "RubiksCubeTask",
    "MustardInLeftBinTask",
    "SpoonInMugTask",
    "MarkerInMugTask",
    "SmartphoneInBinTask",
)
ENVIRONMENT_SEEDS = (101, 103, 107, 109, 113)
BRANCH_SEEDS = (211, 223, 227, 229)
ACTION_SHAPE = (32, 8)
ACTION_COORDINATE_COUNT = 256
RESEARCH_SIGMAS = (
    np.float32(0.9990000128746033),
    np.float32(0.9369999766349792),
    np.float32(0.8330000042915344),
    np.float32(0.6240000128746033),
)
TIMING_CONDITIONS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("none", ()),
    ("call_0_only", (0,)),
    ("call_1_only", (1,)),
    ("call_2_only", (2,)),
    ("call_3_only", (3,)),
    ("all_calls", (0, 1, 2, 3)),
)
SINGLE_CALL_CONDITIONS = tuple(name for name, _ in TIMING_CONDITIONS[1:5])
REQUESTS_PER_STATE = 108
EXPECTED_STATE_COUNT = 30
EXPECTED_REQUEST_COUNT = EXPECTED_STATE_COUNT * REQUESTS_PER_STATE


def ordered_source_cells(
    branch_seeds: Sequence[int] = BRANCH_SEEDS,
) -> tuple[tuple[int, int], ...]:
    seeds = tuple(int(seed) for seed in branch_seeds)
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("the timing design requires four unique branch seeds")
    return tuple((recipient, source) for recipient in seeds for source in seeds)


def ordered_off_diagonal_pairs(
    branch_seeds: Sequence[int] = BRANCH_SEEDS,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (recipient, source)
        for recipient, source in ordered_source_cells(branch_seeds)
        if recipient != source
    )


def expected_request_labels(
    branch_seeds: Sequence[int] = BRANCH_SEEDS,
) -> tuple[str, ...]:
    """Return the immutable 108-label request order for one state."""

    seeds = tuple(int(seed) for seed in branch_seeds)
    labels = [f"native-{seed}" for seed in seeds]
    labels.extend(f"native-replay-{seed}" for seed in seeds)
    for timing, _ in TIMING_CONDITIONS:
        labels.extend(
            f"timing-{timing}-recipient-{recipient}-source-{source}"
            for recipient, source in ordered_source_cells(seeds)
        )
    labels.extend(f"all-calls-diagonal-replay-{seed}" for seed in seeds)
    if len(labels) != REQUESTS_PER_STATE or len(set(labels)) != len(labels):
        raise AssertionError("timing request construction did not produce 108 unique labels")
    return tuple(labels)


def maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    if left_value.shape != right_value.shape:
        raise ValueError(f"shape mismatch: {left_value.shape} != {right_value.shape}")
    if not np.all(np.isfinite(left_value)) or not np.all(np.isfinite(right_value)):
        raise ValueError("maximum-error operands must be finite")
    return float(np.max(np.abs(left_value - right_value), initial=0.0))


def nearest_native_seed(
    action: np.ndarray,
    native_actions: Mapping[int, np.ndarray],
    branch_seeds: Sequence[int] = BRANCH_SEEDS,
) -> tuple[int, dict[int, float], bool, float]:
    """Return deterministic top-1, distances, tie flag, and top-2 margin."""

    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(value)):
        raise ValueError("action contains a nonfinite value")
    distances: dict[int, float] = {}
    for seed in branch_seeds:
        native = np.asarray(native_actions[int(seed)], dtype=np.float64).reshape(-1)
        if native.shape != value.shape or not np.all(np.isfinite(native)):
            raise ValueError("native actions must be finite and shape-matched")
        distances[int(seed)] = float(np.linalg.norm(value - native))
    ordered = sorted((distance, list(branch_seeds).index(seed), seed) for seed, distance in distances.items())
    tie = bool(np.isclose(ordered[0][0], ordered[1][0], rtol=0.0, atol=0.0))
    return int(ordered[0][2]), distances, tie, float(ordered[1][0] - ordered[0][0])


def directional_metrics(
    action: np.ndarray,
    recipient: np.ndarray,
    donor: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    recipient_value = np.asarray(recipient, dtype=np.float64).reshape(-1)
    donor_value = np.asarray(donor, dtype=np.float64).reshape(-1)
    if value.shape != recipient_value.shape or donor_value.shape != recipient_value.shape:
        raise ValueError("directional-metric inputs must have equal flattened shapes")
    if not all(np.all(np.isfinite(item)) for item in (value, recipient_value, donor_value)):
        raise ValueError("directional-metric inputs must be finite")
    axis = donor_value - recipient_value
    displacement = value - recipient_value
    separation = float(np.linalg.norm(axis))
    if separation <= eps:
        raise ValueError(f"degenerate native axis: {separation}")
    denominator = float(np.dot(axis, axis))
    projection = float(np.dot(displacement, axis) / denominator)
    displacement_norm = float(np.linalg.norm(displacement))
    orthogonal = displacement - projection * axis
    return {
        "native_separation": separation,
        "distance_to_donor": float(np.linalg.norm(value - donor_value)),
        "distance_reduction": float(
            (separation - np.linalg.norm(value - donor_value)) / separation
        ),
        "donor_projection": projection,
        "cosine_alignment": float(
            np.dot(displacement, axis) / (displacement_norm * separation)
        )
        if displacement_norm > eps
        else 0.0,
        "orthogonal_residual_normalized": float(np.linalg.norm(orthogonal) / separation),
    }


def validate_native_axes(
    native_actions: Mapping[int, np.ndarray],
    branch_seeds: Sequence[int] = BRANCH_SEEDS,
    *,
    eps: float = 1e-12,
) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for recipient, donor in ordered_off_diagonal_pairs(branch_seeds):
        separation = float(
            np.linalg.norm(
                np.asarray(native_actions[recipient], dtype=np.float64).reshape(-1)
                - np.asarray(native_actions[donor], dtype=np.float64).reshape(-1)
            )
        )
        if not np.isfinite(separation) or separation <= eps:
            raise ValueError(f"degenerate native axis {(recipient, donor)}: {separation}")
        values[(recipient, donor)] = separation
    return values


def state_estimands(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compute all frozen within-state timing estimands from one complete report."""

    seeds = tuple(int(seed) for seed in report["branch_seeds"])
    if seeds != BRANCH_SEEDS:
        raise ValueError(f"unexpected branch order: {seeds}")
    native_actions = {
        int(seed): np.asarray(action, dtype=np.float64)
        for seed, action in report["native_actions"].items()
    }
    separations = validate_native_axes(native_actions, seeds)
    raw_rows = list(report["timing_rows"])
    expected = [
        (timing, recipient, source)
        for timing, _ in TIMING_CONDITIONS
        for recipient, source in ordered_source_cells(seeds)
    ]
    actual = [
        (str(row["timing_condition"]), int(row["recipient_seed"]), int(row["source_seed"]))
        for row in raw_rows
    ]
    if actual != expected:
        raise ValueError("timing rows do not match the frozen 96-row order")
    indexed = {
        (str(row["timing_condition"]), int(row["recipient_seed"]), int(row["source_seed"])): row
        for row in raw_rows
    }

    timing_metrics: dict[str, dict[str, float | int]] = {}
    pair_rows: list[dict[str, Any]] = []
    for timing, _ in TIMING_CONDITIONS:
        complete_correct: list[float] = []
        raw_donor_correct: list[float] = []
        matched_retrieval_gain: list[float] = []
        matched_distance_gain: list[float] = []
        directional: dict[str, list[float]] = {
            "distance_reduction": [],
            "donor_projection": [],
            "cosine_alignment": [],
            "orthogonal_residual_normalized": [],
            "distance_to_donor": [],
        }
        tie_count = 0
        margins: list[float] = []
        final_target_max_abs_errors: list[float] = []
        final_target_l2_distances: list[float] = []
        for recipient, source in ordered_source_cells(seeds):
            row = indexed[(timing, recipient, source)]
            action = np.asarray(row["action"], dtype=np.float64)
            max_abs_error = float(row["final_sampler_target_max_abs_error"])
            l2_distance = float(row["final_sampler_target_l2"])
            if not np.isfinite(max_abs_error) or not np.isfinite(l2_distance):
                raise ValueError("final sampler target residuals must be finite")
            final_target_max_abs_errors.append(max_abs_error)
            final_target_l2_distances.append(l2_distance)
            nearest, _distances, tied, margin = nearest_native_seed(
                action, native_actions, seeds
            )
            complete_correct.append(float(nearest == source))
            tie_count += int(tied)
            margins.append(margin)
            if recipient == source:
                continue
            self_action = np.asarray(
                indexed[(timing, recipient, recipient)]["action"], dtype=np.float64
            )
            native_donor = native_actions[source]
            denominator = separations[(recipient, source)]
            retrieval_gain = float(nearest == source) - float(
                nearest_native_seed(self_action, native_actions, seeds)[0] == source
            )
            distance_gain = float(
                (
                    np.linalg.norm(self_action.reshape(-1) - native_donor.reshape(-1))
                    - np.linalg.norm(action.reshape(-1) - native_donor.reshape(-1))
                )
                / denominator
            )
            native_metrics = directional_metrics(
                action, native_actions[recipient], native_actions[source]
            )
            raw_donor_correct.append(float(nearest == source))
            matched_retrieval_gain.append(retrieval_gain)
            matched_distance_gain.append(distance_gain)
            for metric in directional:
                directional[metric].append(native_metrics[metric])
            pair_rows.append(
                {
                    "task": str(report["task"]),
                    "unit_id": str(report["unit_id"]),
                    "timing_condition": timing,
                    "recipient_seed": recipient,
                    "source_seed": source,
                    "native_separation": denominator,
                    "retrieval_gain": retrieval_gain,
                    "distance_gain": distance_gain,
                    "correct_donor": float(nearest == source),
                    **native_metrics,
                }
            )
        timing_metrics[timing] = {
            "complete_source_retrieval": float(np.mean(complete_correct)),
            "raw_off_diagonal_donor_retrieval": float(np.mean(raw_donor_correct)),
            "matched_retrieval_gain": float(np.mean(matched_retrieval_gain)),
            "matched_distance_gain": float(np.mean(matched_distance_gain)),
            "tie_count": tie_count,
            "minimum_top2_margin": float(min(margins)),
            "final_sampler_target_max_abs_error_mean": float(
                np.mean(final_target_max_abs_errors)
            ),
            "final_sampler_target_max_abs_error_max": float(
                max(final_target_max_abs_errors)
            ),
            "final_sampler_target_l2_mean": float(
                np.mean(final_target_l2_distances)
            ),
            "final_sampler_target_l2_max": float(max(final_target_l2_distances)),
            **{name: float(np.mean(values)) for name, values in directional.items()},
        }

    average_single: dict[str, float] = {}
    sustained_minus_single: dict[str, float] = {}
    for metric in ("matched_retrieval_gain", "matched_distance_gain"):
        single = float(np.mean([timing_metrics[name][metric] for name in SINGLE_CALL_CONDITIONS]))
        average_single[metric] = single
        sustained_minus_single[metric] = float(
            timing_metrics["all_calls"][metric] - single
        )
    return {
        "unit_id": str(report["unit_id"]),
        "task": str(report["task"]),
        "environment_seed": int(report["environment_seed"]),
        "timing": timing_metrics,
        "average_single": average_single,
        "sustained_minus_single": sustained_minus_single,
        "minimum_native_separation": float(min(separations.values())),
        "pair_rows": pair_rows,
    }


@dataclass(frozen=True)
class HierarchicalDraws:
    tasks: tuple[str, ...]
    state_ids_by_task: Mapping[str, tuple[str, ...]]
    task_draws: np.ndarray
    state_draws: np.ndarray


def make_hierarchical_draws(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 20260903,
) -> HierarchicalDraws:
    tasks = tuple(TASKS)
    observed_tasks = {str(row["task"]) for row in rows}
    if observed_tasks != set(tasks):
        raise ValueError(f"task set mismatch: {sorted(observed_tasks)}")
    state_ids_by_task: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        ids = tuple(
            str(row["unit_id"])
            for row in rows
            if str(row["task"]) == task
        )
        if len(ids) != 5 or len(set(ids)) != 5:
            raise ValueError(f"task {task} must have exactly five unique states")
        state_ids_by_task[task] = ids
    rng = np.random.Generator(np.random.PCG64(seed))
    task_draws = rng.integers(0, len(tasks), size=(samples, len(tasks)), endpoint=False)
    state_draws = rng.integers(
        0, 5, size=(samples, len(tasks), 5), endpoint=False
    )
    return HierarchicalDraws(tasks, state_ids_by_task, task_draws, state_draws)


def summarize_state_values(
    rows: Sequence[Mapping[str, Any]],
    values: Mapping[str, float],
    draws: HierarchicalDraws,
) -> dict[str, Any]:
    if len(rows) != EXPECTED_STATE_COUNT or len(values) != EXPECTED_STATE_COUNT:
        raise ValueError("summary requires all 30 state values")
    task_values: list[np.ndarray] = []
    for task in draws.tasks:
        ids = draws.state_ids_by_task[task]
        try:
            vector = np.asarray([values[state_id] for state_id in ids], dtype=np.float64)
        except KeyError as error:
            raise ValueError(f"missing state value: {error}") from error
        if vector.shape != (5,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"task {task} has invalid state values")
        task_values.append(vector)
    matrix = np.stack(task_values, axis=0)
    task_means = matrix.mean(axis=1)
    point = float(task_means.mean())
    boot = np.empty(draws.task_draws.shape[0], dtype=np.float64)
    for draw_index, sampled_tasks in enumerate(draws.task_draws):
        sampled_task_means = [
            float(matrix[task_index, draws.state_draws[draw_index, occurrence]].mean())
            for occurrence, task_index in enumerate(sampled_tasks)
        ]
        boot[draw_index] = float(np.mean(sampled_task_means))
    centered_p = float(
        (1 + np.count_nonzero((boot - point) >= point)) / (boot.size + 1)
    )
    return {
        "mean": point,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "one_sided_null_centered_p": centered_p,
        "bootstrap_samples": int(boot.size),
        "task_means": {
            task: float(task_means[index]) for index, task in enumerate(draws.tasks)
        },
        "bootstrap_values": boot,
    }


def holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, dict[str, float | bool]]:
    if len(raw_p_values) != 4:
        raise ValueError("Holm timing family must contain exactly four tests")
    ordered = sorted((float(value), str(name)) for name, value in raw_p_values.items())
    adjusted_sorted: list[float] = []
    running = 0.0
    for rank, (value, _name) in enumerate(ordered):
        candidate = min(1.0, (len(ordered) - rank) * value)
        running = max(running, candidate)
        adjusted_sorted.append(running)
    stop = False
    result: dict[str, dict[str, float | bool]] = {}
    for rank, ((raw, name), adjusted) in enumerate(zip(ordered, adjusted_sorted)):
        threshold = 0.05 / (len(ordered) - rank)
        rejected = bool(not stop and raw <= threshold)
        if not rejected:
            stop = True
        result[name] = {
            "raw_p": raw,
            "holm_adjusted_p": adjusted,
            "step_threshold": threshold,
            "rejected": rejected,
        }
    return result


def separation_quartiles(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    off_diagonal = [
        row
        for row in pair_rows
        if str(row["timing_condition"]) == "all_calls"
    ]
    if len(off_diagonal) != 360:
        raise ValueError(f"expected 360 directed native separations, got {len(off_diagonal)}")
    values = np.asarray([row["native_separation"] for row in off_diagonal], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("native separations must be finite")
    boundaries = np.quantile(values, [0.25, 0.5, 0.75])
    bins = np.searchsorted(boundaries, values, side="right")
    counts = {f"q{index + 1}": int(np.count_nonzero(bins == index)) for index in range(4)}
    state_counts = {
        f"q{index + 1}": len(
            {
                str(row["unit_id"])
                for row, bin_index in zip(off_diagonal, bins)
                if bin_index == index
            }
        )
        for index in range(4)
    }
    task_counts = {
        f"q{index + 1}": len(
            {
                str(row["task"])
                for row, bin_index in zip(off_diagonal, bins)
                if bin_index == index
            }
        )
        for index in range(4)
    }
    return {
        "boundaries": [float(value) for value in boundaries],
        "pair_counts": counts,
        "state_counts": state_counts,
        "task_counts": task_counts,
        "boundary_assignment": "numpy.searchsorted(boundaries, value, side='right')",
    }


def all_finite(value: Any) -> bool:
    """Recursively require every numeric leaf to be finite."""

    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(float(value)))
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    # Optional nonnumeric metadata can legitimately be null (for example an
    # unused attention-cache ID). Required numeric leaves are checked by their
    # explicit schema validators before this recursive finite audit.
    if value is None:
        return True
    return True
