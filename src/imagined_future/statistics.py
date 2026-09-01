"""Small, dependency-light confirmatory estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def cluster_bootstrap_mean(
    unit_values: Mapping[str, float],
    *,
    resamples: int = 10_000,
    seed: int = 20260831,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile interval resampling independent saved-state units."""

    if not unit_values:
        raise ValueError("cluster bootstrap requires at least one saved-state unit")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be within (0, 1)")
    values = np.asarray(list(unit_values.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("unit effects must be finite")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "units": len(values),
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, alpha / 2)),
        "upper": float(np.quantile(means, 1 - alpha / 2)),
        "resamples": resamples,
        "seed": seed,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm familywise-error adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=np.float64)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be within [0, 1]")
    order = np.argsort(values, kind="mergesort")
    adjusted_sorted = np.maximum.accumulate(
        np.asarray([(len(values) - rank) * values[index] for rank, index in enumerate(order)])
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()
