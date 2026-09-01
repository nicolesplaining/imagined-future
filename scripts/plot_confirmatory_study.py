"""Render the registered state-level confirmatory panels as PDF and PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {0: "#2457A7", 3: "#E28743", 6: "#5B8E55"}
PREFIX_LABELS = {0: "first query", 3: "after 3 chunks", 6: "after 6 chunks"}


def _state_means(frame: pd.DataFrame, contrast: str, metric: str) -> pd.DataFrame:
    selected = frame.loc[frame["contrast"] == contrast].copy()
    return (
        selected.groupby(["unit_id", "task_id", "prefix_chunks"], as_index=False)[
            metric
        ]
        .mean()
        .sort_values(["task_id", "prefix_chunks"])
    )


def _bootstrap_interval(
    values: np.ndarray, seed: int = 20260831
) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--robocasa-summary-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    semantic = pd.read_csv(args.summary_dir / "semantic_state_repetitions.csv")
    attention = pd.read_csv(args.summary_dir / "attention_state_repetitions.csv")
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
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)

    primary = _state_means(
        semantic,
        "semantic_all_donor_minus_recipient",
        "action_donor_steering",
    )
    for task_id, task in primary.groupby("task_id"):
        axes[0].plot(
            task["prefix_chunks"].map({0: 0, 3: 1, 6: 2}),
            task["action_donor_steering"],
            color="#B8B8B8",
            linewidth=0.8,
            alpha=0.75,
            zorder=1,
        )
        axes[0].scatter(
            task["prefix_chunks"].map({0: 0, 3: 1, 6: 2}),
            task["action_donor_steering"],
            c=[COLORS[int(prefix)] for prefix in task["prefix_chunks"]],
            s=18,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
            label=f"task {int(task_id)}",
        )
    for position, prefix in enumerate((0, 3, 6)):
        values = primary.loc[
            primary["prefix_chunks"] == prefix, "action_donor_steering"
        ].to_numpy()
        mean, lower, upper = _bootstrap_interval(values, seed=20260831 + prefix)
        axes[0].errorbar(
            position + 0.12,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt="D",
            color="black",
            markersize=3.5,
            capsize=2,
            linewidth=1,
            zorder=3,
        )
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].axhline(0.10, color="#7A7A7A", linewidth=0.7, linestyle="--")
    axes[0].set_xticks(range(3), ["first", "3 chunks", "6 chunks"])
    axes[0].set_ylabel("donor − self steering")
    axes[0].set_title("a  Semantic future sufficiency", loc="left", fontweight="bold")

    contrast_specs = [
        ("semantic_all_donor_minus_recipient", "all future"),
        ("semantic_wrist_donor_minus_recipient", "wrist"),
        ("semantic_primary_donor_minus_recipient", "primary"),
        ("semantic_proprio_donor_minus_recipient", "proprio"),
        ("semantic_all_donor_minus_gaussian", "vs Gaussian"),
        ("semantic_all_donor_minus_natural_control", "vs natural"),
        ("semantic_all_donor_minus_shuffled", "vs shuffled"),
    ]
    early = semantic.loc[semantic["prefix_chunks"] == 0]
    for position, (contrast, label) in enumerate(contrast_specs):
        values = _state_means(early, contrast, "action_donor_steering")[
            "action_donor_steering"
        ].to_numpy()
        mean, lower, upper = _bootstrap_interval(values, seed=20260920 + position)
        axes[1].errorbar(
            mean,
            position,
            xerr=[[mean - lower], [upper - mean]],
            fmt="o",
            color="#2457A7" if position < 4 else "#5B8E55",
            markersize=4,
            capsize=2,
            linewidth=1,
        )
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].axvline(0.10, color="#7A7A7A", linewidth=0.7, linestyle="--")
    axes[1].set_yticks(
        range(len(contrast_specs)), [label for _contrast, label in contrast_specs]
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("first-query steering contrast")
    axes[1].set_title("b  Modalities and controls", loc="left", fontweight="bold")

    gates = [0.25, 0.5, 0.75, 1.0]
    for prefix in (0, 3, 6):
        means = []
        lowers = []
        uppers = []
        for gate in gates:
            contrast = f"block27_future_gate{gate:g}"
            values = _state_means(
                attention.loc[attention["prefix_chunks"] == prefix],
                contrast,
                "action_l2_from_baseline",
            )["action_l2_from_baseline"].to_numpy()
            mean, lower, upper = _bootstrap_interval(
                values, seed=20261000 + prefix + int(gate * 10)
            )
            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)
        axes[2].plot(
            gates,
            means,
            marker="o",
            markersize=3.5,
            color=COLORS[prefix],
            label=PREFIX_LABELS[prefix],
        )
        axes[2].fill_between(
            gates, lowers, uppers, color=COLORS[prefix], alpha=0.16, linewidth=0
        )
    axes[2].set_xlabel("future-key removal gate")
    axes[2].set_ylabel("action L2 from baseline")
    axes[2].set_xticks(gates)
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].set_title("c  Attention-path dose response", loc="left", fontweight="bold")

    for suffix in ("pdf", "png"):
        figure.savefig(
            args.output_dir / f"confirmatory_action_results.{suffix}", dpi=300
        )
    plt.close(figure)

    if args.robocasa_summary_dir is None:
        return
    robocasa = pd.read_csv(args.robocasa_summary_dir / "semantic_repetitions.csv")
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)

    physical = _state_means(
        semantic,
        "semantic_all_donor_minus_recipient",
        "physical_endpoint_donor_steering",
    )
    for _task_id, task in physical.groupby("task_id"):
        positions = task["prefix_chunks"].map({0: 0, 3: 1, 6: 2})
        axes[0].plot(
            positions,
            task["physical_endpoint_donor_steering"],
            color="#B8B8B8",
            linewidth=0.8,
        )
        axes[0].scatter(
            positions,
            task["physical_endpoint_donor_steering"],
            c=[COLORS[int(prefix)] for prefix in task["prefix_chunks"]],
            s=18,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
    for position, prefix in enumerate((0, 3, 6)):
        values = physical.loc[
            physical["prefix_chunks"] == prefix, "physical_endpoint_donor_steering"
        ].to_numpy()
        mean, lower, upper = _bootstrap_interval(values, seed=20261100 + prefix)
        axes[0].errorbar(
            position + 0.12,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt="D",
            color="black",
            markersize=3.5,
            capsize=2,
            linewidth=1,
            zorder=3,
        )
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].axhline(0.10, color="#7A7A7A", linewidth=0.7, linestyle="--")
    axes[0].set_xticks(range(3), ["first", "3 chunks", "6 chunks"])
    axes[0].set_ylabel("physical donor − self steering")
    axes[0].set_title("a  Executed semantic effect", loc="left", fontweight="bold")

    endpoint_conditions = [
        ("block27_future_gate1", "future keys"),
        ("block27_random_gate1", "random keys"),
        ("block27_current_gate1", "current keys"),
        ("block27_all_key_control", "no-op control"),
    ]
    early_attention = attention.loc[attention["prefix_chunks"] == 0]
    for position, (contrast, _label) in enumerate(endpoint_conditions):
        values = _state_means(
            early_attention,
            contrast,
            "physical_endpoint_donor_steering",
        )["physical_endpoint_donor_steering"].to_numpy()
        mean, lower, upper = _bootstrap_interval(values, seed=20261200 + position)
        axes[1].errorbar(
            mean,
            position,
            xerr=[[mean - lower], [upper - mean]],
            fmt="o",
            color="#2457A7" if position == 0 else "#7A7A7A",
            markersize=4,
            capsize=2,
            linewidth=1,
        )
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].set_yticks(
        range(len(endpoint_conditions)),
        [label for _contrast, label in endpoint_conditions],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("absolute endpoint steering change")
    axes[1].set_title("b  Attention endpoint controls", loc="left", fontweight="bold")

    robo = (
        robocasa.groupby(
            ["unit_id", "task_name", "episode_index", "contrast"], as_index=False
        )["physical_endpoint_donor_steering"]
        .mean()
        .sort_values(["task_name", "episode_index"])
    )
    unit_order = robo.loc[
        robo["contrast"] == "semantic_donor_minus_recipient", "unit_id"
    ].tolist()
    unit_labels = {
        unit: unit.replace("PnPCounterToCab", "Pick/place")
        .replace("TurnOffMicrowave", "Microwave")
        .replace("OpenDrawer", "Drawer")
        .replace("_episode00_prefix00", " · first")
        .replace("_episode10_prefix03", " · 3 chunks")
        for unit in unit_order
    }
    for contrast, label, color, marker in (
        ("semantic_donor_minus_recipient", "donor − self", "#2457A7", "o"),
        ("semantic_donor_minus_gaussian", "donor − Gaussian", "#5B8E55", "s"),
    ):
        selected = robo.loc[robo["contrast"] == contrast].set_index("unit_id")
        values = [
            selected.loc[unit, "physical_endpoint_donor_steering"]
            for unit in unit_order
        ]
        axes[2].scatter(
            values,
            range(len(unit_order)),
            color=color,
            marker=marker,
            s=22,
            label=label,
        )
    axes[2].axvline(0, color="black", linewidth=0.6)
    axes[2].set_yticks(
        range(len(unit_order)), [unit_labels[unit] for unit in unit_order]
    )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("physical steering contrast")
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].set_title("c  RoboCasa replication", loc="left", fontweight="bold")

    for suffix in ("pdf", "png"):
        figure.savefig(
            args.output_dir / f"confirmatory_endpoint_results.{suffix}", dpi=300
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
