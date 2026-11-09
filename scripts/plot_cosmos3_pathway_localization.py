#!/usr/bin/env python3
"""Plot single-layer K/V localization and the population route decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imagined_future.statistics import cluster_bootstrap_mean

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, INK, MUTED = "#b9b7b0", "#0b0b0b", "#52514e"


def decomposition(population: Path) -> dict[str, dict[str, float]]:
    parts = {"layer16": {}, "other_direct": {}, "via_current": {}}
    for path in sorted(population.glob("*/summary.json")):
        runs = json.loads(path.read_text())["interventions"]
        value = lambda label: float(runs[label]["action_donor_projection"])  # noqa: E731
        base = value("predicted_donor")
        layer16 = value("predicted_donor_kv_patch_selected")
        direct = value("predicted_donor_kv_patch_all_action")
        barrier = value("predicted_donor_kv_patch_all_nonfuture")
        parts["layer16"][path.parent.name] = base - layer16
        parts["other_direct"][path.parent.name] = layer16 - direct
        parts["via_current"][path.parent.name] = direct - barrier
    return {name: cluster_bootstrap_mean(values, seed=20261109) for name, values in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=Path("results/cosmos3_kv_patch_v4/summary.json"))
    parser.add_argument("--population", type=Path, default=Path("results/cosmos3_population_confirmatory_v1"))
    parser.add_argument("--replication", type=Path, default=Path("results/cosmos3_mechanism_v1/aggregate_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/figure_pathway_localization.pdf"))
    args = parser.parse_args()

    layer_means = json.loads(args.calibration.read_text())["calibration"]["all_layer_means"]
    layers = np.arange(36)
    losses = np.asarray([layer_means[str(layer)] for layer in layers])
    parts = decomposition(args.population)

    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    figure, (top, bottom, third) = plt.subplots(
        3, 1, figsize=(3.35, 4.1), gridspec_kw={"height_ratios": [2.1, 1.0, 1.9], "hspace": 1.0}
    )

    colors = [BLUE if layer == 16 else NEUTRAL for layer in layers]
    top.bar(layers, losses, width=0.72, color=colors, linewidth=0)
    top.axhline(0, color=MUTED, linewidth=0.6)
    top.set_xlim(-0.8, 35.8)
    top.set_xticks([0, 8, 16, 24, 35])
    top.set_xlabel("Layer whose predicted-future K/V is restored (action queries)")
    top.set_ylabel("Replacement loss")
    top.annotate(f"layer 16: {losses[16]:.2f}", xy=(16, losses[16]), xytext=(19.5, losses[16] * 0.92),
                 color=INK, fontsize=7, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})
    top.set_title("a  Single-layer scan, 4 excluded calibration tasks", loc="left", fontsize=7.5, color=INK)
    for side in ("top", "right"):
        top.spines[side].set_visible(False)
    top.grid(axis="y", color="#e6e5e0", linewidth=0.5)
    top.set_axisbelow(True)

    left = 0.0
    segments = [
        ("layer16", "Layer 16", BLUE),
        ("other_direct", "Other 35 layers", ORANGE),
        ("via_current", "Via current frame", AQUA),
    ]
    for key, label, color in segments:
        width = parts[key]["mean"]
        bottom.barh(0, width, left=left, height=0.55, color=color, edgecolor="white", linewidth=1.0)
        bottom.text(left + width / 2, 0.52, f"{label}\n{width:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        left += width
    bottom.set_xlim(0, 1.05)
    bottom.set_ylim(-0.45, 1.35)
    bottom.set_yticks([])
    bottom.set_xlabel("Share of donor steering removed (predicted action, 22 held-out states)")
    bottom.set_title("b  Route decomposition", loc="left", fontsize=7.5, color=INK)
    for side in ("top", "right", "left"):
        bottom.spines[side].set_visible(False)

    metrics = json.loads(args.replication.read_text())["metrics"]
    rows = [
        ("Layers 0-11", "kv_loss_layers_00_11"),
        ("Layers 12-23", "kv_loss_layers_12_23"),
        ("Layers 24-35", "kv_loss_layers_24_35"),
        ("Keys only", "kv_loss_keys_only"),
        ("Values only", "kv_loss_values_only"),
    ]
    positions = np.asarray([0, 1, 2, 3.6, 4.6])
    means = np.asarray([metrics[key]["mean"] for _, key in rows])
    lower = means - np.asarray([metrics[key]["lower"] for _, key in rows])
    upper = np.asarray([metrics[key]["upper"] for _, key in rows]) - means
    third.barh(positions, means, height=0.62, color=BLUE, linewidth=0)
    third.errorbar(means, positions, xerr=[lower, upper], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
    for position, mean, high in zip(positions, means, means + upper):
        third.text(high + 0.03, position, f"{mean:.2f}", va="center", fontsize=6.5, color=INK)
    third.set_yticks(positions)
    third.set_yticklabels([label for label, _ in rows])
    third.invert_yaxis()
    third.set_xlim(0, 1.05)
    third.set_xlabel("Replacement loss (replication run, 22 states)")
    third.set_title("c  Which layers and which component", loc="left", fontsize=7.5, color=INK)
    for side in ("top", "right"):
        third.spines[side].set_visible(False)
    third.grid(axis="x", color="#e6e5e0", linewidth=0.5)
    third.set_axisbelow(True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(json.dumps({key: {k: round(v, 3) for k, v in value.items() if k in ("mean", "lower", "upper")}
                      for key, value in parts.items()}))


if __name__ == "__main__":
    main()
