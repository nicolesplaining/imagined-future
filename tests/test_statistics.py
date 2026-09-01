from __future__ import annotations

import pytest

from imagined_future.statistics import cluster_bootstrap_mean, holm_adjust


def test_cluster_bootstrap_resamples_units() -> None:
    estimate = cluster_bootstrap_mean({"a": 0.0, "b": 1.0, "c": 2.0}, resamples=1000, seed=7)
    assert estimate["units"] == 3
    assert estimate["mean"] == 1.0
    assert estimate["lower"] <= 1.0 <= estimate["upper"]


def test_cluster_bootstrap_rejects_pseudoreplication_input_without_units() -> None:
    with pytest.raises(ValueError, match="saved-state"):
        cluster_bootstrap_mean({})


def test_holm_adjust_preserves_order_and_monotonicity() -> None:
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
