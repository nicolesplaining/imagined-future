from __future__ import annotations

import numpy as np

from imagined_future.study_design import (
    distance_matched_control_donor,
    fixed_unit_grid,
    matched_same_label_donor,
    pairwise_l2,
    select_primary_pair,
)


def test_fixed_unit_grid_crosses_tasks_and_strata() -> None:
    units = fixed_unit_grid((0, 1), (10, 27), (0, 3))
    assert [unit.unit_id for unit in units] == [
        "task00_state10_prefix00",
        "task00_state27_prefix03",
        "task01_state10_prefix00",
        "task01_state27_prefix03",
    ]


def test_select_primary_pair_uses_action_and_endpoint_ranks() -> None:
    actions = np.asarray([[0.0], [5.0], [6.0]])
    endpoints = np.asarray([[0.0], [1.0], [10.0]])
    left, right, diagnostics = select_primary_pair(actions, endpoints, minimum_action_l2=0.1)
    assert (left, right) == (0, 2)
    assert diagnostics["normalized_action_l2"] == 6.0
    assert diagnostics["physical_endpoint_l2"] == 10.0


def test_matched_same_label_donor_matches_intervention_size() -> None:
    action = pairwise_l2(np.asarray([[0.0], [4.0], [3.8], [1.0]]))
    endpoint = pairwise_l2(np.asarray([[0.0], [8.0], [7.9], [2.0]]))
    selected = matched_same_label_donor(
        recipient=0,
        primary_donor=1,
        labels=(False, True, False, False),
        action_distances=action,
        endpoint_distances=endpoint,
    )
    assert selected == 2
    assert (
        distance_matched_control_donor(
            recipient=0,
            primary_donor=1,
            action_distances=action,
            endpoint_distances=endpoint,
        )
        == 2
    )
