from __future__ import annotations

import numpy as np
import pytest

from imagined_future.cosmos3_dose_response import (
    ALPHAS,
    EXPECTED_CALLS_PER_STATE,
    adjacent_contrasts,
    dose_action_metrics,
    frozen_request_specs,
    interpolation_target,
    ols_slope,
    pair_is_nondecreasing_proximity,
    validate_released_action,
)


BRANCH_SEEDS = (211, 223, 227, 229)


def test_frozen_request_specs_has_exact_order_and_census() -> None:
    rows = frozen_request_specs(BRANCH_SEEDS)
    assert len(rows) == EXPECTED_CALLS_PER_STATE == 92
    assert len({row["label"] for row in rows}) == 92
    assert [row["kind"] for row in rows[:8]] == ["native"] * 4 + [
        "native_repeat"
    ] * 4
    assert [row["kind"] for row in rows[8:12]] == ["none"] * 4
    assert [row["kind"] for row in rows[12:20]] == ["self"] * 4 + [
        "self_repeat"
    ] * 4
    assert [row["kind"] for row in rows[20:80]] == ["dose"] * 60
    assert [row["kind"] for row in rows[80:]] == ["dose_midpoint_repeat"] * 12
    first_pair = rows[20:25]
    assert [(row["recipient_seed"], row["donor_seed"]) for row in first_pair] == [
        (211, 223)
    ] * 5
    assert tuple(row["alpha"] for row in first_pair) == ALPHAS


def test_interpolation_formula_and_endpoint_identities() -> None:
    recipient = np.asarray([1.0, -2.0, 5.0], dtype=np.float32)
    donor = np.asarray([5.0, 6.0, -3.0], dtype=np.float32)
    assert np.array_equal(interpolation_target(recipient, donor, 0.0), recipient)
    assert np.array_equal(interpolation_target(recipient, donor, 1.0), donor)
    midpoint = interpolation_target(recipient, donor, 0.5)
    assert np.array_equal(midpoint, np.asarray([3.0, 2.0, 1.0], dtype=np.float32))
    with pytest.raises(ValueError):
        interpolation_target(recipient, donor, 0.3)


def test_dose_metrics_recover_exact_axis_values_and_four_way_id() -> None:
    native = {
        211: np.asarray([0.0, 0.0]),
        223: np.asarray([2.0, 0.0]),
        227: np.asarray([0.0, 3.0]),
        229: np.asarray([-3.0, 0.0]),
    }
    metrics = dose_action_metrics(
        np.asarray([1.0, 0.0]), native[211], native[223], native, 223
    )
    assert metrics["distance_reduction_to_donor"] == pytest.approx(0.5)
    assert metrics["normalized_projection"] == pytest.approx(0.5)
    assert metrics["cosine_alignment"] == pytest.approx(1.0)
    assert metrics["orthogonal_residual_normalized"] == pytest.approx(0.0)
    # Equal distance to 211 and 223 is resolved deterministically by seed.
    assert metrics["nearest_native_seed"] == 211
    assert metrics["correct_donor_top1"] is False
    assert metrics["nearest_native_exact_tie"] is True
    assert metrics["nearest_native_tie_count"] == 2
    assert metrics["nearest_native_tied_seeds"] == [211, 223]
    assert metrics["nearest_native_top_two_margin"] == 0.0
    reversed_native = dict(reversed(list(native.items())))
    reordered = dose_action_metrics(
        np.asarray([1.0, 0.0]),
        reversed_native[211],
        reversed_native[223],
        reversed_native,
        223,
    )
    assert reordered["nearest_native_seed"] == 211
    with pytest.raises(ValueError):
        dose_action_metrics(
            np.asarray([1.0, 0.0]),
            native[211],
            native[223],
            {211: native[211], 223: native[223], 227: native[227]},
            223,
        )


def test_zero_displacement_cosine_is_exactly_zero() -> None:
    native = {
        211: np.zeros((32, 8), dtype=np.float32),
        223: np.ones((32, 8), dtype=np.float32),
        227: np.full((32, 8), 4.0, dtype=np.float32),
        229: np.full((32, 8), -4.0, dtype=np.float32),
    }
    metrics = dose_action_metrics(native[211], native[211], native[223], native, 223)
    assert metrics["cosine_alignment"] == 0.0
    assert metrics["distance_reduction_to_donor"] == 0.0
    assert metrics["normalized_projection"] == 0.0


def test_ols_adjacent_and_monotonic_helpers() -> None:
    values = {alpha: 2.0 * alpha - 0.25 for alpha in ALPHAS}
    assert ols_slope(ALPHAS, [values[alpha] for alpha in ALPHAS]) == pytest.approx(2.0)
    explicit = sum((alpha - 0.5) * values[alpha] for alpha in ALPHAS) / 0.625
    assert ols_slope(ALPHAS, [values[alpha] for alpha in ALPHAS]) == explicit
    assert adjacent_contrasts(values) == {
        "0.00_to_0.25": pytest.approx(0.5),
        "0.25_to_0.50": pytest.approx(0.5),
        "0.50_to_0.75": pytest.approx(0.5),
        "0.75_to_1.00": pytest.approx(0.5),
    }
    assert pair_is_nondecreasing_proximity(
        {alpha: 1.0 - alpha for alpha in ALPHAS}
    )
    assert not pair_is_nondecreasing_proximity(
        {0.0: 1.0, 0.25: 0.8, 0.5: 0.9, 0.75: 0.4, 1.0: 0.0}
    )


def test_full_released_action_shape_is_mandatory() -> None:
    valid = np.zeros((32, 8), dtype=np.float32)
    assert validate_released_action(valid).shape == (32, 8)
    with pytest.raises(ValueError):
        validate_released_action(np.zeros((32, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        validate_released_action(np.zeros(256, dtype=np.float32))
    invalid = valid.copy()
    invalid[0, 7] = np.nan
    with pytest.raises(ValueError):
        validate_released_action(invalid)
    invalid[0, 7] = np.inf
    with pytest.raises(ValueError):
        validate_released_action(invalid)
    with pytest.raises(ValueError):
        validate_released_action([[0.0] * 8] * 31 + [[0.0] * 7])


def test_request_grid_rejects_duplicate_branch_seeds() -> None:
    with pytest.raises(ValueError):
        frozen_request_specs((211, 211, 227, 229))


def test_gripper_only_change_contributes_to_dose_metrics() -> None:
    native = {seed: np.zeros((32, 8), dtype=np.float32) for seed in BRANCH_SEEDS}
    native[223][:, 7] = 2.0
    native[227][:, 0] = 10.0
    native[229][:, 1] = -10.0
    halfway = np.zeros((32, 8), dtype=np.float32)
    halfway[:, 7] = 1.0
    metrics = dose_action_metrics(halfway, native[211], native[223], native, 223)
    assert metrics["distance_reduction_to_donor"] == pytest.approx(0.5)
    assert metrics["normalized_projection"] == pytest.approx(0.5)
