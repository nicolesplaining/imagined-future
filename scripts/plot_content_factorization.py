"""Render hybrid and natural robot-versus-object result panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _unit_means(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return (
        frame.groupby("unit_id", as_index=False)[metric]
        .mean()[metric]
        .to_numpy(dtype=np.float64)
    )


def _interval(
    values: np.ndarray, *, seed: int
) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(np.mean(values)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _point_interval(
    axis,
    position: int,
    values: np.ndarray,
    *,
    color: str,
    seed: int,
) -> None:
    mean, lower, upper = _interval(values, seed=seed)
    axis.scatter(
        np.full(len(values), position) + np.linspace(-0.08, 0.08, len(values)),
        values,
        s=14,
        color=color,
        alpha=0.45,
        linewidths=0,
        zorder=1,
    )
    axis.errorbar(
        position,
        mean,
        yerr=[[mean - lower], [upper - mean]],
        fmt="D",
        color="black",
        markerfacecolor=color,
        markersize=4,
        capsize=2,
        linewidth=1,
        zorder=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial-dir", type=Path, required=True)
    parser.add_argument("--natural-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    factorial_effects = pd.read_csv(
        args.factorial_dir / "factorial_effect_repetitions.csv"
    )
    factorial_cells = pd.read_csv(
        args.factorial_dir / "factorial_cell_repetitions.csv"
    )
    natural = pd.read_csv(args.natural_dir / "factorization_repetitions.csv")
    natural_units = natural["unit_id"].nunique()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.25), constrained_layout=True)

    identification = (
        ("decoded_primary_target_top1", "decoded primary", "#5B8E55"),
        ("decoded_wrist_target_top1", "decoded wrist", "#5B8E55"),
        ("correct_target_cell", "executed endpoint", "#A7463A"),
    )
    for position, (metric, _label, color) in enumerate(identification):
        _point_interval(
            axes[0],
            position,
            _unit_means(factorial_cells, metric),
            color=color,
            seed=20262001 + position,
        )
    axes[0].axhline(0.25, color="#7A7A7A", linestyle="--", linewidth=0.8)
    axes[0].set_ylim(0, 1.06)
    axes[0].set_xticks(
        range(len(identification)), [label for _metric, label, _color in identification]
    )
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_ylabel("four-cell top-1 identification")
    axes[0].set_title("a  Future changes; execution does not", loc="left", fontweight="bold")

    all_effects = factorial_effects.loc[factorial_effects["modality"] == "all"]
    hybrid_metrics = (
        (
            "goal_endpoint_donor_steering__object_main_effect",
            "object → goal",
            "#2457A7",
        ),
        (
            "goal_endpoint_donor_steering__robot_main_effect",
            "robot → goal",
            "#E28743",
        ),
        (
            "robot_endpoint_donor_steering__object_main_effect",
            "object → robot",
            "#2457A7",
        ),
        (
            "robot_endpoint_donor_steering__robot_main_effect",
            "robot → robot",
            "#E28743",
        ),
    )
    for position, (metric, _label, color) in enumerate(hybrid_metrics):
        _point_interval(
            axes[1],
            position,
            _unit_means(all_effects, metric),
            color=color,
            seed=20262101 + position,
        )
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].axhline(0.10, color="#7A7A7A", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(
        range(len(hybrid_metrics)), [label for _metric, label, _color in hybrid_metrics]
    )
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylabel("factorial endpoint steering")
    axes[1].set_title("b  Hybrid content effects (n = 20)", loc="left", fontweight="bold")

    natural_specs = (
        ("object", "action_donor_steering", "object action", "#2457A7"),
        (
            "object",
            "goal_endpoint_donor_steering",
            "object goal",
            "#2457A7",
        ),
        ("robot", "action_donor_steering", "robot action", "#E28743"),
        (
            "robot",
            "robot_endpoint_donor_steering",
            "robot endpoint",
            "#E28743",
        ),
    )
    for position, (pair, metric, _label, color) in enumerate(natural_specs):
        selected = natural.loc[
            natural["contrast"] == f"{pair}_all_donor_minus_recipient"
        ]
        _point_interval(
            axes[2],
            position,
            _unit_means(selected, metric),
            color=color,
            seed=20262201 + position,
        )
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].axhline(0.10, color="#7A7A7A", linestyle="--", linewidth=0.8)
    axes[2].set_xticks(
        range(len(natural_specs)),
        [label for _pair, _metric, label, _color in natural_specs],
    )
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].set_ylabel("donor − recipient steering")
    axes[2].set_title(
        f"c  Natural reachable pairs (n = {natural_units})",
        loc="left",
        fontweight="bold",
    )

    for suffix in ("pdf", "png"):
        figure.savefig(
            args.output_dir / f"content_factorization_results.{suffix}", dpi=300
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
