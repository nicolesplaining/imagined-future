"""Frozen unit construction and donor selection for confirmatory studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StudyUnit:
    task_id: int
    initial_state_index: int
    prefix_chunks: int

    @property
    def unit_id(self) -> str:
        return f"task{self.task_id:02d}_state{self.initial_state_index:02d}_prefix{self.prefix_chunks:02d}"

    def to_dict(self) -> dict[str, int | str]:
        return {"unit_id": self.unit_id, **asdict(self)}


def fixed_unit_grid(
    task_ids: Sequence[int], initial_state_indices: Sequence[int], prefix_chunks: Sequence[int]
) -> tuple[StudyUnit, ...]:
    """Cross tasks with registered state/timing strata."""

    if len(initial_state_indices) != len(prefix_chunks):
        raise ValueError("initial-state indices and prefix chunks must define equal strata")
    units = tuple(
        StudyUnit(int(task), int(state), int(prefix))
        for task in task_ids
        for state, prefix in zip(initial_state_indices, prefix_chunks, strict=True)
    )
    if len({unit.unit_id for unit in units}) != len(units):
        raise ValueError("study grid contains duplicate units")
    return units


def pairwise_l2(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return np.linalg.norm(flattened[:, None] - flattened[None, :], axis=-1)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def select_primary_pair(
    normalized_actions: np.ndarray,
    goal_endpoint_features: np.ndarray,
    *,
    minimum_action_l2: float,
    minimum_endpoint_l2: float = 1e-8,
) -> tuple[int, int, dict[str, float]]:
    """Select the action/endpoint-divergent pair without intervention outcomes."""

    action_distances = pairwise_l2(normalized_actions)
    endpoint_distances = pairwise_l2(goal_endpoint_features)
    pairs = [
        (left, right)
        for left in range(len(normalized_actions))
        for right in range(left + 1, len(normalized_actions))
    ]
    eligible = [
        pair
        for pair in pairs
        if action_distances[pair] >= minimum_action_l2
        and endpoint_distances[pair] >= minimum_endpoint_l2
    ]
    if not eligible:
        raise ValueError("no branch pair passes the registered action and endpoint distance floors")
    action_values = np.asarray([action_distances[pair] for pair in eligible])
    endpoint_values = np.asarray([endpoint_distances[pair] for pair in eligible])
    scores = (_average_ranks(action_values) + _average_ranks(endpoint_values)) / 2.0
    best_index = max(
        range(len(eligible)),
        key=lambda index: (scores[index], -eligible[index][0], -eligible[index][1]),
    )
    left, right = eligible[best_index]
    return left, right, {
        "score": float(scores[best_index]),
        "normalized_action_l2": float(action_distances[left, right]),
        "physical_endpoint_l2": float(endpoint_distances[left, right]),
    }


def matched_same_label_donor(
    *,
    recipient: int,
    primary_donor: int,
    labels: Sequence[bool | None],
    action_distances: np.ndarray,
    endpoint_distances: np.ndarray,
) -> int | None:
    """Choose the same-label donor closest to the primary intervention size."""

    recipient_label = labels[recipient]
    if recipient_label is None:
        return None
    candidates = [
        index
        for index, label in enumerate(labels)
        if index not in (recipient, primary_donor) and label == recipient_label
    ]
    if not candidates:
        return None
    action_target = float(action_distances[recipient, primary_donor])
    endpoint_target = float(endpoint_distances[recipient, primary_donor])
    action_scale = max(action_target, np.finfo(np.float64).eps)
    endpoint_scale = max(endpoint_target, np.finfo(np.float64).eps)
    return min(
        candidates,
        key=lambda index: (
            abs(float(action_distances[recipient, index]) - action_target) / action_scale
            + abs(float(endpoint_distances[recipient, index]) - endpoint_target) / endpoint_scale,
            index,
        ),
    )


def distance_matched_control_donor(
    *,
    recipient: int,
    primary_donor: int,
    action_distances: np.ndarray,
    endpoint_distances: np.ndarray,
) -> int:
    """Preselect a natural control with primary-like intervention size."""

    candidates = [
        index
        for index in range(action_distances.shape[0])
        if index not in (recipient, primary_donor)
    ]
    if not candidates:
        raise ValueError("distance-matched control requires at least three branches")
    action_target = float(action_distances[recipient, primary_donor])
    endpoint_target = float(endpoint_distances[recipient, primary_donor])
    action_scale = max(action_target, np.finfo(np.float64).eps)
    endpoint_scale = max(endpoint_target, np.finfo(np.float64).eps)
    return min(
        candidates,
        key=lambda index: (
            abs(float(action_distances[recipient, index]) - action_target) / action_scale
            + abs(float(endpoint_distances[recipient, index]) - endpoint_target) / endpoint_scale,
            index,
        ),
    )
