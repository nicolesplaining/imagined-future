#!/usr/bin/env python3
"""Render a presentation-only FastWAM summary from frozen analysis outputs.

This script does not recompute any aggregate or confidence interval. It reads
the frozen state CSV for display points and the audited JSON for displayed
means and intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CONDITIONS = (
    "first_frame",
    "wrong_latent",
    "shuffled_cache",
    "donor_latent",
    "donor_cache",
)
LABELS = ("First\nframe", "Wrong\nlatent", "Shuffled\ncache", "Donor\nlatent", "Donor\ncache")
COLORS = ("#9CA3AF", "#B8B8B8", "#777777", "#E6862A", "#2A9D8F")
PANELS = (
    ("correct_donor_retrieval_rate", "Correct-donor retrieval", 0.25),
    ("donor_distance_reduction", "Distance reduction", 0.0),
    ("donor_projection", "Donor-axis projection", 0.0),
    ("orthogonal_residual_ratio", "Orthogonal residual / native separation", 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--state-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("manifest_id") != "fastwam-813f0233b9a2c083":
        raise ValueError("renderer requires the complete audited powered FastWAM summary")
    aggregate = report["conditions"]
    with args.state_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    state_ids = sorted({row["state_id"] for row in rows})
    if len(state_ids) != 120 or len(rows) != 120 * len(CONDITIONS):
        raise ValueError("renderer requires 120 states and five displayed conditions")
    index = {(row["state_id"], row["condition"]): row for row in rows}
    if set(row["condition"] for row in rows) != set(CONDITIONS):
        raise ValueError("unexpected condition set in state CSV")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args.output_dir.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    for axis, (metric, panel_title, baseline) in zip(axes.flat, PANELS, strict=True):
        for condition_index, (condition, color) in enumerate(
            zip(CONDITIONS, COLORS, strict=True)
        ):
            values = np.asarray(
                [float(index[(state_id, condition)][metric]) for state_id in state_ids],
                dtype=np.float64,
            )
            offsets = np.linspace(-0.105, 0.105, values.size)
            axis.scatter(
                condition_index + offsets,
                values,
                s=18,
                color=color,
                alpha=0.52,
                linewidths=0,
                zorder=2,
            )
            stats = aggregate[condition][metric]
            mean = float(stats["mean"])
            low = float(stats["ci95_low"])
            high = float(stats["ci95_high"])
            axis.errorbar(
                condition_index,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                markersize=6.5,
                capsize=3,
                color="#111827",
                markerfacecolor=color,
                markeredgewidth=0.9,
                zorder=3,
            )
        axis.axhline(baseline, color="#6B7280", linewidth=0.9, linestyle="--", zorder=1)
        axis.set_title(panel_title, fontsize=13, loc="left", fontweight="semibold")
        axis.set_xticks(range(len(CONDITIONS)), LABELS, fontsize=9)
        axis.tick_params(axis="y", labelsize=9)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylim(-0.03, 1.03)
    figure.suptitle(
        "Directional future steering in FastWAM Optional-IDM",
        fontsize=16,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"fastwam_optional_idm_summary.{suffix}",
            dpi=240,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
