from __future__ import annotations

import numpy as np
import pytest

from imagined_future.cosmos3_protocol import (
    FROZEN_TASK_OBJECT_NAMES,
    MINIMAL_KV_PHYSICAL_LABELS,
    directional_target_metrics,
    donor_kv_factorial_interventions,
    donor_selection_description,
    native_execution_seeds,
    ordered_recipient_donor_pairs,
    should_execute_intervention,
)


def test_frozen_tasks_have_explicit_task_object_names() -> None:
    assert FROZEN_TASK_OBJECT_NAMES == {
        "BananaInBowlTask": "banana",
        "RubiksCubeTask": "rubiks_cube",
        "MustardInLeftBinTask": "mustard",
        "SpoonInMugTask": "spoon_big",
        "MarkerInMugTask": "marker",
        "SmartphoneInBinTask": "smartphone",
    }


def test_directional_target_metrics_separate_axis_progress_from_residual() -> None:
    metrics = directional_target_metrics(
        np.asarray([1.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([2.0, 0.0]),
    )

    assert metrics["l2_to_target"] == pytest.approx(np.sqrt(2.0))
    assert metrics["native_target_l2"] == 2.0
    assert metrics["distance_reduction_to_target"] == pytest.approx(1.0 - np.sqrt(2.0) / 2.0)
    assert metrics["cosine_alignment"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert metrics["orthogonal_residual_normalized"] == 0.5


def test_directional_target_metrics_preserve_nulls() -> None:
    value = np.asarray([1.0, 0.0])
    recipient = np.asarray([0.0, 0.0])

    assert all(
        result is None
        for result in directional_target_metrics(value, recipient, None).values()
    )
    degenerate = directional_target_metrics(value, recipient, recipient)
    assert degenerate["l2_to_target"] == 1.0
    assert degenerate["native_target_l2"] == 0.0
    assert degenerate["distance_reduction_to_target"] is None
    assert degenerate["cosine_alignment"] is None
    assert degenerate["orthogonal_residual_normalized"] is None


def test_frozen_multi_donor_executes_every_native_branch() -> None:
    seeds = [211, 223, 227, 229]

    assert native_execution_seeds(seeds, (211, 223), multi_donor=True) == set(seeds)
    assert native_execution_seeds(seeds, (211, 223), multi_donor=False) == {211, 223}
    assert native_execution_seeds(seeds, None, multi_donor=False) == set(seeds)


def test_all_recipient_grid_contains_every_directed_nonself_pair() -> None:
    pairs = ordered_recipient_donor_pairs([211, 223, 227, 229])

    assert len(pairs) == 12
    assert len(set(pairs)) == 12
    assert all(recipient != donor for recipient, donor in pairs)
    assert {recipient for recipient, _donor in pairs} == {211, 223, 227, 229}
    assert all(
        sum(recipient == seed for recipient, _donor in pairs) == 3
        for seed in (211, 223, 227, 229)
    )


def test_frozen_pair_metadata_does_not_claim_unobserved_selection_rule() -> None:
    description = donor_selection_description((211, 223), multi_donor=True)

    assert "externally supplied" in description
    assert "prespecified additional donors" in description
    assert "maximum" not in description


def test_donor_kv_factorial_is_sequential_and_uses_one_donor_cache() -> None:
    specs, targets = donor_kv_factorial_interventions(
        study_id="held-out-state",
        recipient_seed=211,
        donor_seed=223,
        layers=range(36),
    )

    labels = list(specs)
    assert labels == [
        "predicted_donor_kv_record",
        "predicted_donor_kv_replay",
        "self_with_predicted_donor_kv",
        "executed_donor_kv_record",
        "executed_donor_kv_replay",
        "self_with_executed_donor_kv",
    ]
    assert [specs[label]["research_attention_mode"] for label in labels] == [
        "record",
        "patch",
        "patch",
        "record",
        "patch",
        "patch",
    ]
    assert [specs[label]["research_attention_cache_id"] for label in labels] == [
        "held-out-state-predicted-donor-future-kv",
        "held-out-state-predicted-donor-future-kv",
        "held-out-state-predicted-donor-future-kv",
        "held-out-state-executed-donor-future-kv",
        "held-out-state-executed-donor-future-kv",
        "held-out-state-executed-donor-future-kv",
    ]
    assert specs["predicted_donor_kv_record"]["research_donor_id"] == (
        "held-out-state-native-223"
    )
    assert specs["executed_donor_kv_record"]["research_donor_id"] == (
        "held-out-state-executed-223"
    )
    assert specs["self_with_predicted_donor_kv"]["research_mode"] == "self"
    assert specs["self_with_executed_donor_kv"]["research_donor_id"] == (
        "held-out-state-native-211"
    )
    assert all(target == 223 for target in targets.values())


def test_minimal_kv_execution_allowlist_excludes_duplicate_controls() -> None:
    assert should_execute_intervention("predicted_donor", minimal_kv_factorial=True)
    assert should_execute_intervention(
        "self_with_executed_donor_kv", minimal_kv_factorial=True
    )
    assert not should_execute_intervention(
        "predicted_donor_kv_record", minimal_kv_factorial=True
    )
    assert not should_execute_intervention("gaussian_executed", minimal_kv_factorial=True)
    assert should_execute_intervention("gaussian_executed", minimal_kv_factorial=False)
    assert len(MINIMAL_KV_PHYSICAL_LABELS) == 7
