#!/usr/bin/env python3
"""Render presentation artifacts from a frozen Cosmos 3 dose summary.

This script performs no statistical recomputation. It validates and displays
the estimates and intervals already stored by the frozen analyzer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ALPHAS = ("0.0", "0.25", "0.5", "0.75", "1.0")
ALPHA_LABELS = ("0", ".25", ".50", ".75", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def estimate(summary: dict, field: str, alpha: str) -> tuple[float, float, float]:
    row = summary[field][alpha]
    return float(row["estimate"]), float(row["lower"]), float(row["upper"])


def validate(summary: dict) -> None:
    if summary.get("status") != "complete" or summary.get("state_count") != 30:
        raise ValueError("renderer requires the complete 30-state frozen summary")
    if summary.get("missing_state_count") != 0 or summary.get("exclusion_count") != 0:
        raise ValueError("renderer refuses an incomplete or filtered summary")
    state_rows = summary.get("state_rows", [])
    if len(state_rows) != 30 or any(
        len(row.get("ordered_pair_profiles", {})) != 12 for row in state_rows
    ):
        raise ValueError("renderer requires 30 state rows with 12 ordered pairs each")
    for field in (
        "distance_reduction_by_alpha",
        "projection_by_alpha",
        "donor_identification_by_alpha",
    ):
        if tuple(summary.get(field, {}).keys()) != ALPHAS:
            raise ValueError(f"unexpected alpha grid for {field}")


def render_plot(summary: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.asarray([float(value) for value in ALPHAS])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    panels = (
        ("distance_reduction_by_alpha", "Distance reduction toward donor"),
        ("donor_identification_by_alpha", "Correct-donor retrieval"),
    )
    task_rows = summary["per_task_all_estimands"]
    task_names = sorted(summary["per_task_primary"])
    for ax, (field, ylabel) in zip(axes, panels):
        prefix = (
            "distance_reduction_alpha_"
            if field == "distance_reduction_by_alpha"
            else "donor_identification_alpha_"
        )
        for task in task_names:
            task_values = [float(task_rows[prefix + alpha][task]) for alpha in ALPHAS]
            ax.plot(x, task_values, color="#E8893D", alpha=0.30, linewidth=1.1)
        triples = [estimate(summary, field, alpha) for alpha in ALPHAS]
        means = np.asarray([row[0] for row in triples])
        lows = np.asarray([row[1] for row in triples])
        highs = np.asarray([row[2] for row in triples])
        ax.errorbar(
            x,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            color="#16778A",
            marker="o",
            markersize=7,
            linewidth=2.7,
            capsize=4,
            zorder=5,
        )
        ax.axhline(0.0, color="#6B7280", linewidth=1.0)
        ax.set_xlabel(r"Donor-future mixture $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, ALPHA_LABELS)
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(0.25, color="#E8893D", linestyle="--", linewidth=1.5)
    axes[1].text(0.02, 0.265, "4-way chance = .25", color="#A4541E", fontsize=9)
    fig.suptitle("Future-strength dose response in Cosmos 3", fontsize=17, weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cosmos3_future_strength_dose_response.{suffix}", dpi=240)
    plt.close(fig)


def fmt(row: tuple[float, float, float]) -> str:
    return f"{row[0]:.3f} [{row[1]:.3f}, {row[2]:.3f}]"


def render_tables(summary: dict, output_dir: Path) -> None:
    rows = []
    for label, alpha in zip(ALPHA_LABELS, ALPHAS):
        rows.append(
            (
                label,
                fmt(estimate(summary, "distance_reduction_by_alpha", alpha)),
                fmt(estimate(summary, "projection_by_alpha", alpha)),
                fmt(estimate(summary, "donor_identification_by_alpha", alpha)),
            )
        )

    md = [
        "# Cosmos 3 future-strength dose response",
        "",
        "| Donor mixture $\\alpha$ | Distance reduction | Donor-axis projection | Correct-donor retrieval |",
        "|---:|---:|---:|---:|",
    ]
    md.extend(f"| {a} | {d} | {p} | {r} |" for a, d, p, r in rows)
    primary = summary["primary_distance_reduction_slope"]
    md.extend(
        [
            "",
            f"Primary distance-reduction slope: {primary['estimate']:.3f} "
            f"[{primary['lower']:.3f}, {primary['upper']:.3f}].",
            "",
            "Values are equal-task means with task-to-state hierarchical 95% "
            "bootstrap intervals over 30 fixed states from six tasks.",
        ]
    )
    (output_dir / "cosmos3_future_strength_dose_response.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    tex_rows = "\n".join(
        f"    {a} & {d} & {p} & {r} \\\\" for a, d, p, r in rows
    )
    tex = rf"""\begin{{table}}[t]
  \centering
  \small
  \setlength{{\tabcolsep}}{{3.2pt}}
  \caption{{\textbf{{Future-strength dose response in Cosmos 3.}}
  We interpolate the imposed future from recipient ($\alpha=0$) to donor
  ($\alpha=1$). Values are equal-task means with hierarchical 95\% bootstrap
  intervals over 30 fixed states from six tasks.}}
  \label{{tab:future_dose}}
  \begin{{tabular}}{{lccc}}
    \toprule
    $\alpha$ & Distance reduction & Projection & Donor retrieval \\
    \midrule
{tex_rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    (output_dir / "cosmos3_future_strength_dose_response_table.tex").write_text(
        tex, encoding="utf-8"
    )


def render_state_csvs(summary: dict, output_dir: Path) -> None:
    state_fields = (
        "unit_id",
        "task",
        "episode_id",
        "environment_seed",
        "distance_reduction_slope",
        "projection_slope",
        "distance_reduction_endpoint_contrast",
        "projection_endpoint_contrast",
        "nondecreasing_pair_fraction",
    )
    for alpha in ALPHAS:
        state_fields += (
            f"distance_reduction_alpha_{alpha}",
            f"projection_alpha_{alpha}",
            f"donor_identification_alpha_{alpha}",
            f"cosine_alignment_alpha_{alpha}",
            f"orthogonal_residual_alpha_{alpha}",
        )
    state_path = output_dir / "cosmos3_future_strength_dose_response_states.csv"
    with state_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=state_fields)
        writer.writeheader()
        for state in summary["state_rows"]:
            row = {field: state[field] for field in state_fields[:9]}
            for alpha in ALPHAS:
                row[f"distance_reduction_alpha_{alpha}"] = state[
                    "distance_reduction_by_alpha"
                ][alpha]
                row[f"projection_alpha_{alpha}"] = state["projection_by_alpha"][alpha]
                row[f"donor_identification_alpha_{alpha}"] = state[
                    "donor_identification_by_alpha"
                ][alpha]
                row[f"cosine_alignment_alpha_{alpha}"] = state[
                    "cosine_alignment_by_alpha"
                ][alpha]
                row[f"orthogonal_residual_alpha_{alpha}"] = state[
                    "orthogonal_residual_by_alpha"
                ][alpha]
            writer.writerow(row)

    pair_fields = (
        "unit_id",
        "task",
        "recipient_seed",
        "donor_seed",
        "distance_reduction_slope",
        "projection_slope",
        "donor_proximity_nondecreasing",
    )
    for alpha in ALPHAS:
        pair_fields += (
            f"distance_reduction_alpha_{alpha}",
            f"projection_alpha_{alpha}",
            f"l2_to_donor_alpha_{alpha}",
            f"correct_donor_top1_alpha_{alpha}",
        )
    pair_path = output_dir / "cosmos3_future_strength_dose_response_pairs.csv"
    with pair_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for state in summary["state_rows"]:
            profiles = state["ordered_pair_profiles"]
            for pair_name in sorted(profiles):
                profile = profiles[pair_name]
                row = {
                    "unit_id": state["unit_id"],
                    "task": state["task"],
                    **{field: profile[field] for field in pair_fields[2:7]},
                }
                for alpha in ALPHAS:
                    row[f"distance_reduction_alpha_{alpha}"] = profile[
                        "distance_reduction_by_alpha"
                    ][alpha]
                    row[f"projection_alpha_{alpha}"] = profile[
                        "projection_by_alpha"
                    ][alpha]
                    row[f"l2_to_donor_alpha_{alpha}"] = profile[
                        "l2_to_donor_by_alpha"
                    ][alpha]
                    row[f"correct_donor_top1_alpha_{alpha}"] = profile[
                        "correct_donor_top1_by_alpha"
                    ][alpha]
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    validate(summary)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    render_plot(summary, args.output_dir)
    render_tables(summary, args.output_dir)
    render_state_csvs(summary, args.output_dir)


if __name__ == "__main__":
    main()
