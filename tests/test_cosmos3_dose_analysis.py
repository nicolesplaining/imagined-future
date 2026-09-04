from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from imagined_future.cosmos3_dose_response import ALPHAS


def load_analyzer():
    path = Path(__file__).parents[1] / "scripts" / "summarize_cosmos3_future_strength_dose_response.py"
    spec = importlib.util.spec_from_file_location("dose_analyzer_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_state_rows() -> list[dict[str, object]]:
    return [
        {"task": f"task-{task}", "episode_id": f"task-{task}-state-{state}", "value": task + state / 10}
        for task in range(6)
        for state in range(5)
    ]


def test_shared_hierarchical_draw_table_is_exact_pcg64_and_deterministic() -> None:
    analyzer = load_analyzer()
    rows = synthetic_state_rows()
    first = analyzer.hierarchical_draw_table(rows, samples=10_000, seed=20260903)
    second = analyzer.hierarchical_draw_table(rows, samples=10_000, seed=20260903)
    assert np.array_equal(first[2], second[2])
    assert np.array_equal(first[3], second[3])

    generator = np.random.Generator(np.random.PCG64(20260903))
    expected_tasks = generator.integers(0, 6, size=6)
    expected_states = np.stack([generator.integers(0, 5, size=5) for _ in range(6)])
    assert np.array_equal(first[2][0], expected_tasks)
    assert np.array_equal(first[3][0], expected_states)

    estimate = analyzer.hierarchical_estimate(first, lambda row: float(row["value"]))
    task_means = [
        np.mean([float(row["value"]) for row in rows if row["task"] == f"task-{task}"])
        for task in range(6)
    ]
    assert estimate["estimate"] == float(np.mean(task_means))


def test_state_summary_uses_equal_pair_weights_and_exact_pairwise_ols() -> None:
    analyzer = load_analyzer()
    unit = {
        "unit_id": "synthetic-middle",
        "task": "synthetic",
        "episode_id": "synthetic-episode",
        "environment_seed": 101,
    }
    pairs = [
        (recipient, donor)
        for recipient in (211, 223, 227, 229)
        for donor in (211, 223, 227, 229)
        if recipient != donor
    ]
    rows = []
    for recipient, donor in pairs:
        for alpha in ALPHAS:
            rows.append(
                {
                    "recipient_seed": recipient,
                    "donor_seed": donor,
                    "alpha": alpha,
                    "distance_reduction_to_donor": alpha,
                    "normalized_projection": 2.0 * alpha,
                    "cosine_alignment": alpha,
                    "orthogonal_residual_normalized": 0.0,
                    "l2_to_donor": 1.0 - alpha,
                    "correct_donor_top1": alpha >= 0.75,
                }
            )
    summary = analyzer.state_summary(unit, rows)
    assert summary["distance_reduction_slope"] == 1.0
    assert summary["distance_reduction_profile_slope"] == 1.0
    assert summary["distance_slope_equivalence_abs_error"] == 0.0
    assert summary["projection_slope"] == 2.0
    assert summary["distance_reduction_endpoint_contrast"] == 1.0
    assert summary["distance_reduction_adjacent_contrasts"] == {
        "0.00_to_0.25": 0.25,
        "0.25_to_0.50": 0.25,
        "0.50_to_0.75": 0.25,
        "0.75_to_1.00": 0.25,
    }
    assert summary["nondecreasing_pair_fraction"] == 1.0
