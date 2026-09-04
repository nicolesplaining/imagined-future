#!/usr/bin/env python3
"""Render the frozen Cosmos 3 archival selection-free analysis for handoff.

This script is presentation-only: it consumes the complete-cohort analyzer's
JSON and CSV, verifies their fixed cohort identity, and emits a compact table,
Markdown summary, and an overview figure. It performs no refitting or outcome
selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_STATES = 90
EXPECTED_TASKS = 6
EXPECTED_EPISODES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--state-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_inputs(summary_path: Path, state_rows_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with state_rows_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if summary.get("status") != "complete":
        raise ValueError("analysis summary is not complete")
    if (
        summary.get("state_count") != EXPECTED_STATES
        or summary.get("task_count") != EXPECTED_TASKS
        or summary.get("episode_count") != EXPECTED_EPISODES
        or len(rows) != EXPECTED_STATES
    ):
        raise ValueError("input is not the frozen 90-state/30-episode/six-task cohort")
    criteria = summary.get("evidence_criteria", {})
    if set(criteria) == set() or criteria.get("complete_cohort_90_of_90") is not True:
        raise ValueError("complete-cohort evidence gate is absent or false")
    return summary, rows


def estimate(cell: dict[str, Any]) -> tuple[float, float, float]:
    mean = float(cell["mean"])
    lower, upper = (float(value) for value in cell["ci95"])
    if not np.isfinite([mean, lower, upper]).all() or lower > mean or mean > upper:
        raise ValueError(f"invalid estimate cell: {cell}")
    return mean, lower, upper


def interval_text(cell: dict[str, Any], digits: int = 3) -> str:
    mean, lower, upper = estimate(cell)
    return f"{mean:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def errorbar(ax: Any, x: np.ndarray, cells: list[dict[str, Any]], **kwargs: Any) -> None:
    values = np.asarray([estimate(cell)[0] for cell in cells])
    lower = np.asarray([estimate(cell)[1] for cell in cells])
    upper = np.asarray([estimate(cell)[2] for cell in cells])
    ax.errorbar(x, values, yerr=np.vstack([values - lower, upper - values]), **kwargs)


def task_points(summary: dict[str, Any], metric: str) -> list[float]:
    values = summary["per_task"][metric]
    if len(values) != EXPECTED_TASKS:
        raise ValueError(f"{metric}: expected six task values")
    result = [float(values[key]) for key in sorted(values)]
    if not np.isfinite(result).all():
        raise ValueError(f"{metric}: nonfinite task value")
    return result


def render_figure(summary: dict[str, Any], output_dir: Path) -> None:
    aggregate = summary["aggregate"]
    phases = summary["phase"]
    quartiles = summary["native_pair_separation_quartiles"]["quartiles"]
    colors = {"effect": "#0B7285", "control": "#ADB5BD", "accent": "#E67E22"}

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)

    # A: balanced four-source retrieval and prespecified controls.
    ax = axes[0, 0]
    retrieval_metrics = ["retrieval_top1", "retrieval_shuffled_top1", "gaussian_top1"]
    retrieval_labels = ["Coherent source", "Shuffled label", "Gaussian target"]
    cells = [aggregate[name] for name in retrieval_metrics]
    x = np.arange(len(cells))
    means = [estimate(cell)[0] for cell in cells]
    ax.bar(x, means, color=[colors["effect"], colors["control"], colors["control"]], width=0.65)
    errorbar(ax, x, cells, fmt="none", ecolor="black", capsize=3, linewidth=1.1)
    for task_value in task_points(summary, "retrieval_top1"):
        ax.scatter(0, task_value, s=18, facecolors="white", edgecolors="black", linewidths=0.7, zorder=3)
    ax.axhline(0.25, color=colors["accent"], linestyle="--", linewidth=1.2, label="chance = 0.25")
    ax.set_xticks(x, retrieval_labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Nearest-source accuracy")
    ax.set_title("A  Future identity is recoverable")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # B: all preregistered directional summaries, with task points on the two
    # central scale-compatible metrics.
    ax = axes[0, 1]
    directional_metrics = ["distance_reduction", "cosine_alignment", "normalized_projection"]
    directional_labels = ["Distance\nreduction", "Cosine\nalignment", "Projection"]
    cells = [aggregate[name] for name in directional_metrics]
    x = np.arange(len(cells))
    ax.bar(x, [estimate(cell)[0] for cell in cells], color=colors["effect"], width=0.65)
    errorbar(ax, x, cells, fmt="none", ecolor="black", capsize=3, linewidth=1.1)
    for index, metric in enumerate(directional_metrics):
        for task_value in task_points(summary, metric):
            ax.scatter(index, task_value, s=18, facecolors="white", edgecolors="black", linewidths=0.7, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, directional_labels)
    ax.set_title("B  Donor-directed action change")

    # C: phase generalization, fixed a priori.
    ax = axes[1, 0]
    phase_order = ["early", "middle", "late"]
    cells = [phases[phase]["retrieval_top1"] for phase in phase_order]
    x = np.arange(3)
    errorbar(
        ax,
        x,
        cells,
        fmt="o-",
        color=colors["effect"],
        markerfacecolor="white",
        capsize=3,
        linewidth=1.8,
    )
    ax.axhline(0.25, color=colors["accent"], linestyle="--", linewidth=1.2)
    ax.set_xticks(x, [value.title() for value in phase_order])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Nearest-source accuracy")
    ax.set_title("C  Steering across decision phase")

    # D: the key anti-selection diagnostic, using pair-level separation bins
    # assigned before state aggregation.
    ax = axes[1, 1]
    quartile_order = ["1", "2", "3", "4"]
    cells = [quartiles[key]["aggregate"]["distance_reduction"] for key in quartile_order]
    x = np.arange(4)
    errorbar(
        ax,
        x,
        cells,
        fmt="o-",
        color=colors["effect"],
        markerfacecolor="white",
        capsize=3,
        linewidth=1.8,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ["Q1\nsmallest", "Q2", "Q3", "Q4\nlargest"])
    ax.set_ylabel("Distance reduction")
    ax.set_title("D  Effect by native separation")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E9ECEF", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Selection-free multi-donor evaluation in Cosmos 3", fontsize=14, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cosmos3_archival_selection_free_summary.{suffix}", dpi=240)
    plt.close(fig)


def render_table(summary: dict[str, Any], output_dir: Path) -> None:
    aggregate = summary["aggregate"]
    rows = [
        ("Future-source retrieval (chance 0.25)", "retrieval_top1"),
        ("Off-diagonal donor retrieval", "donor_top1"),
        ("Shuffled-label retrieval", "retrieval_shuffled_top1"),
        ("Gaussian-target donor retrieval", "gaussian_top1"),
        ("Distance reduction toward donor", "distance_reduction"),
        ("Cosine alignment", "cosine_alignment"),
        ("Normalized orthogonal residual", "orthogonal_residual_normalized"),
        ("Normalized donor-axis projection", "normalized_projection"),
    ]
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Estimand & Mean & Hierarchical 95\% CI \\",
        r"\midrule",
    ]
    markdown = [
        "# Cosmos 3 archival selection-free result",
        "",
        "90 states, 30 archived episodes, six tasks; all four future sources per state.",
        "",
        "| Estimand | Equal-task mean | Hierarchical 95% CI |",
        "|---|---:|---:|",
    ]
    for label, metric in rows:
        mean, lower, upper = estimate(aggregate[metric])
        lines.append(f"{label} & {mean:.3f} & [{lower:.3f}, {upper:.3f}] \\\\")
        markdown.append(f"| {label} | {mean:.3f} | [{lower:.3f}, {upper:.3f}] |")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    permutation = summary["primary_permutation"]
    markdown.extend(
        [
            "",
            f"Permutation p-value for four-source retrieval: {float(permutation['p_greater_monte_carlo']):.6g}.",
            "",
            "Scope: predicted actions from frozen archival observations; no physical-endpoint or task-success estimate.",
        ]
    )
    (output_dir / "cosmos3_archival_selection_free_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_dir / "cosmos3_archival_selection_free_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    summary, _rows = load_inputs(args.summary, args.state_rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    render_table(summary, args.output_dir)
    render_figure(summary, args.output_dir)
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
