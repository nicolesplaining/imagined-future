from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_cosmos3_archival_selection_free.py"
SPEC = importlib.util.spec_from_file_location("archival_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_rows() -> list[dict[str, object]]:
    return [
        {
            "task": task,
            "episode_id": f"{task}-{episode}",
            "value": value,
        }
        for task, task_value in (("a", 0.0), ("b", 1.0))
        for episode in range(2)
        for value in (task_value, task_value)
    ]


def test_hierarchical_bootstrap_uses_equal_task_point_estimate() -> None:
    result = MODULE.hierarchical_task_episode_state_bootstrap(
        synthetic_rows(), "value", samples=100, seed=7
    )
    assert result["mean"] == 0.5
    assert result["tasks"] == 2
    assert result["episodes"] == 4
    assert result["states"] == 8
    assert result["input_states"] == 8
    assert result["eligible_states"] == 8
    assert result["null_states"] == 0


def test_hierarchical_bootstrap_reports_null_denominator() -> None:
    rows = synthetic_rows()
    rows[0]["value"] = None
    result = MODULE.hierarchical_task_episode_state_bootstrap(
        rows, "value", samples=100, seed=7
    )
    assert result["input_states"] == 8
    assert result["eligible_states"] == 7
    assert result["null_states"] == 1


def test_per_task_and_loto_are_equal_task_summaries() -> None:
    rows = synthetic_rows()
    assert MODULE.per_task(rows, "value") == {"a": 0.0, "b": 1.0}
    assert MODULE.leave_one_task_out(rows, "value") == {"a": 1.0, "b": 0.0}


def test_finite_rejects_nan() -> None:
    with pytest.raises(ValueError, match="not finite"):
        MODULE.finite(float("nan"), label="test")


def test_final_sampler_residual_above_old_threshold_is_descriptive() -> None:
    result = MODULE.describe_final_sampler_residuals([0.01, 0.03341507911682129])
    assert result["count"] == 2
    assert result["maximum"] == pytest.approx(0.03341507911682129)
    assert result["count_gt_0_03"] == 1


def test_pair_level_quartiles_partition_arms_before_state_aggregation() -> None:
    reports = []
    for state_index in range(2):
        donor_rows = []
        for arm_index in range(12):
            separation = float(state_index * 12 + arm_index + 1)
            donor_rows.append(
                {
                    "native_target_l2": separation,
                    "correct_donor_top1": arm_index % 2 == 0,
                    "wrong_donor_top1": False,
                    "distance_reduction_to_target": separation / 10,
                    "cosine_alignment": 0.5,
                    "orthogonal_residual_normalized": 0.25,
                    "normalized_projection": 0.75,
                }
            )
        reports.append(
            {
                "unit_id": f"u{state_index}",
                "task": "task",
                "episode_id": f"episode{state_index}",
                "donor_rows": donor_rows,
            }
        )
    result = MODULE.donor_pair_separation_quartiles(reports)
    assert result["total_arms"] == 24
    assert sum(cell["arm_count"] for cell in result["quartiles"].values()) == 24
    assert all(cell["arm_count"] == 6 for cell in result["quartiles"].values())
