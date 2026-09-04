"""Pure helpers for the prospective Cosmos 3 future-strength dose study."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from imagined_future.cosmos3_protocol import ordered_recipient_donor_pairs


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_DENOISING_CALLS = 4
EXPECTED_FUTURE_FRAME_INDICES = tuple(range(1, 9))
EXPECTED_CALLS_PER_STATE = 92
EXPECTED_ACTIVE_RESPONSES_PER_STATE = 80
EXPECTED_ACTIVE_SITES_PER_STATE = 320
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260903
RELEASED_ACTION_SHAPE = (32, 8)
RELEASED_ACTION_COORDINATE_COUNT = 256
FROZEN_BRANCH_SEED_ORDER = (211, 223, 227, 229)


def validate_released_action(value: Any, *, label: str = "action") -> np.ndarray:
    """Require all 32x8 released coordinates, including the gripper channel."""

    action = np.asarray(value, dtype=np.float32)
    if (
        action.shape != RELEASED_ACTION_SHAPE
        or action.size != RELEASED_ACTION_COORDINATE_COUNT
        or not np.isfinite(action).all()
    ):
        raise ValueError(
            f"{label} must be a finite {RELEASED_ACTION_SHAPE} released action "
            f"with {RELEASED_ACTION_COORDINATE_COUNT} coordinates"
        )
    return action


def frozen_request_specs(branch_seeds: Sequence[int]) -> tuple[dict[str, Any], ...]:
    """Return the exact, outcome-independent 92-request sequence for one state."""

    seeds = tuple(int(seed) for seed in branch_seeds)
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("dose response requires exactly four unique branch seeds")
    pairs = ordered_recipient_donor_pairs(seeds)
    rows: list[dict[str, Any]] = []
    rows.extend(
        {"label": f"native-{seed}", "kind": "native", "recipient_seed": seed}
        for seed in seeds
    )
    rows.extend(
        {
            "label": f"native-repeat-{seed}",
            "kind": "native_repeat",
            "recipient_seed": seed,
        }
        for seed in seeds
    )
    rows.extend(
        {"label": f"none-{seed}", "kind": "none", "recipient_seed": seed}
        for seed in seeds
    )
    rows.extend(
        {"label": f"self-{seed}", "kind": "self", "recipient_seed": seed}
        for seed in seeds
    )
    rows.extend(
        {
            "label": f"self-repeat-{seed}",
            "kind": "self_repeat",
            "recipient_seed": seed,
        }
        for seed in seeds
    )
    for recipient_seed, donor_seed in pairs:
        for alpha in ALPHAS:
            rows.append(
                {
                    "label": dose_label(recipient_seed, donor_seed, alpha),
                    "kind": "dose",
                    "recipient_seed": recipient_seed,
                    "donor_seed": donor_seed,
                    "alpha": alpha,
                }
            )
    rows.extend(
        {
            "label": dose_label(recipient_seed, donor_seed, 0.5) + "-repeat",
            "kind": "dose_midpoint_repeat",
            "recipient_seed": recipient_seed,
            "donor_seed": donor_seed,
            "alpha": 0.5,
        }
        for recipient_seed, donor_seed in pairs
    )
    if len(rows) != EXPECTED_CALLS_PER_STATE:
        raise AssertionError(f"dose request census is {len(rows)}, expected 92")
    if len({row["label"] for row in rows}) != len(rows):
        raise AssertionError("dose request labels are not unique")
    return tuple(rows)


def dose_label(recipient_seed: int, donor_seed: int, alpha: float) -> str:
    """Create a stable label that does not depend on float formatting defaults."""

    value = validate_alpha(alpha)
    alpha_token = {0.0: "000", 0.25: "025", 0.5: "050", 0.75: "075", 1.0: "100"}[value]
    return f"recipient-{int(recipient_seed)}-donor-{int(donor_seed)}-alpha-{alpha_token}"


def validate_alpha(alpha: float) -> float:
    """Require exact membership in the prospective five-level grid."""

    value = float(alpha)
    if value not in ALPHAS:
        raise ValueError(f"alpha {value!r} is outside the frozen grid {ALPHAS}")
    return value


def interpolation_target(
    recipient: np.ndarray, donor: np.ndarray, alpha: float
) -> np.ndarray:
    """Construct ``F_A + alpha * (F_B - F_A)`` without changing shape."""

    value = validate_alpha(alpha)
    left = np.asarray(recipient)
    right = np.asarray(donor)
    if left.shape != right.shape:
        raise ValueError(f"recipient/donor shapes differ: {left.shape} != {right.shape}")
    if value == 0.0:
        return left.copy()
    if value == 1.0:
        return right.copy()
    return left + value * (right - left)


def dose_action_metrics(
    value: np.ndarray,
    recipient: np.ndarray,
    donor: np.ndarray,
    all_native_actions: dict[int, np.ndarray],
    donor_seed: int,
    *,
    eps: float = 1e-12,
) -> dict[str, float | int | bool | None]:
    """Compute every frozen action-level metric for one off-diagonal dose arm."""

    action = np.asarray(value, dtype=np.float64).reshape(-1)
    native_recipient = np.asarray(recipient, dtype=np.float64).reshape(-1)
    native_donor = np.asarray(donor, dtype=np.float64).reshape(-1)
    if action.shape != native_recipient.shape or action.shape != native_donor.shape:
        raise ValueError("dose action and native actions must have identical shapes")
    direction = native_donor - native_recipient
    displacement = action - native_recipient
    separation = float(np.linalg.norm(direction))
    distance = float(np.linalg.norm(action - native_donor))
    if separation <= eps:
        projection = None
        cosine = None
        orthogonal = None
        reduction = None
    else:
        denominator = float(np.dot(direction, direction))
        projection = float(np.dot(displacement, direction) / denominator)
        displacement_norm = float(np.linalg.norm(displacement))
        cosine = (
            float(np.dot(displacement, direction) / (displacement_norm * separation))
            if displacement_norm > eps
            else 0.0
        )
        orthogonal_vector = displacement - projection * direction
        orthogonal = float(np.linalg.norm(orthogonal_vector) / separation)
        reduction = float(1.0 - distance / separation)
    if set(int(seed) for seed in all_native_actions) != set(FROZEN_BRANCH_SEED_ORDER):
        raise ValueError(
            "four-way source identification requires exactly the frozen seeds "
            f"{FROZEN_BRANCH_SEED_ORDER}"
        )
    candidate_order = list(FROZEN_BRANCH_SEED_ORDER)
    distances = {
        int(seed): float(
            np.linalg.norm(action - np.asarray(candidate, dtype=np.float64).reshape(-1))
        )
        for seed, candidate in all_native_actions.items()
    }
    nearest = min(
        candidate_order,
        key=lambda seed: (distances[seed], candidate_order.index(seed)),
    )
    sorted_candidates = sorted(
        candidate_order,
        key=lambda seed: (distances[seed], candidate_order.index(seed)),
    )
    minimum_distance = distances[nearest]
    exact_ties = [
        seed for seed in candidate_order if distances[seed] == minimum_distance
    ]
    top_two_margin = distances[sorted_candidates[1]] - distances[sorted_candidates[0]]
    return {
        "native_donor_l2": separation,
        "l2_to_donor": distance,
        "distance_reduction_to_donor": reduction,
        "normalized_projection": projection,
        "cosine_alignment": cosine,
        "orthogonal_residual_normalized": orthogonal,
        "nearest_native_seed": int(nearest),
        "correct_donor_top1": bool(nearest == int(donor_seed)),
        "nearest_native_exact_tie": len(exact_ties) > 1,
        "nearest_native_tie_count": len(exact_ties),
        "nearest_native_tied_seeds": exact_ties,
        "nearest_native_top_two_margin": float(top_two_margin),
        "distances_to_native_actions": {str(seed): value for seed, value in distances.items()},
    }


def ols_slope(alphas: Iterable[float], values: Iterable[float]) -> float:
    """Return the OLS slope with an intercept for one complete alpha grid."""

    x = np.asarray(tuple(float(item) for item in alphas), dtype=np.float64)
    y = np.asarray(tuple(float(item) for item in values), dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("alphas and values must be equal-length one-dimensional vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OLS inputs must be finite")
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        raise ValueError("OLS alpha grid is degenerate")
    if tuple(float(item) for item in x) == ALPHAS:
        if float(x.mean()) != 0.5 or denominator != 0.625:
            raise AssertionError("frozen alpha-grid OLS constants differ")
        return float(np.dot(x - 0.5, y) / 0.625)
    return float(np.dot(centered, y) / denominator)


def adjacent_contrasts(values_by_alpha: dict[float, float]) -> dict[str, float]:
    """Compute all four adjacent contrasts in frozen alpha order."""

    missing = [alpha for alpha in ALPHAS if alpha not in values_by_alpha]
    if missing:
        raise ValueError(f"alpha grid is incomplete: missing {missing}")
    return {
        f"{left:.2f}_to_{right:.2f}": float(
            values_by_alpha[right] - values_by_alpha[left]
        )
        for left, right in zip(ALPHAS[:-1], ALPHAS[1:])
    }


def pair_is_nondecreasing_proximity(
    l2_to_donor_by_alpha: dict[float, float], *, tolerance: float = 0.0
) -> bool:
    """Test whether donor distance never increases as alpha increases."""

    missing = [alpha for alpha in ALPHAS if alpha not in l2_to_donor_by_alpha]
    if missing:
        raise ValueError(f"alpha grid is incomplete: missing {missing}")
    distances = [float(l2_to_donor_by_alpha[alpha]) for alpha in ALPHAS]
    if not np.isfinite(distances).all():
        raise ValueError("donor distances must be finite")
    return all(
        right <= left + float(tolerance)
        for left, right in zip(distances[:-1], distances[1:])
    )
