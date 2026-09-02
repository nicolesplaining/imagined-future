"""Generate manuscript figures directly from frozen result summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1F4E79"
BLUE = "#4C78A8"
ORANGE = "#E07A3F"
TEAL = "#2A9D8F"
GRAY = "#7A7A7A"
LIGHT_BLUE = "#E8F1F8"
LIGHT_ORANGE = "#FBEDE4"
LIGHT_GRAY = "#F2F2F2"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, fc, ec, fontsize=8, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#202020",
    )


def arrow(ax, start, end, color="#333333", style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.2,
            linestyle=style,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def make_overview() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("a  Natural alternatives", loc="left", weight="bold", fontsize=8)
    rounded_box(ax, (0.04, 0.39), 0.18, 0.22, "state\n$S$", LIGHT_GRAY, GRAY, weight="bold")
    rounded_box(ax, (0.42, 0.63), 0.28, 0.20, "future $F_A$", LIGHT_BLUE, BLUE)
    rounded_box(ax, (0.42, 0.17), 0.28, 0.20, "future $F_B$", LIGHT_ORANGE, ORANGE)
    rounded_box(ax, (0.79, 0.63), 0.17, 0.20, "action\n$a_A$", "white", BLUE)
    rounded_box(ax, (0.79, 0.17), 0.17, 0.20, "action\n$a_B$", "white", ORANGE)
    arrow(ax, (0.22, 0.52), (0.42, 0.73), BLUE)
    arrow(ax, (0.22, 0.48), (0.42, 0.27), ORANGE)
    arrow(ax, (0.70, 0.73), (0.79, 0.73), BLUE)
    arrow(ax, (0.70, 0.27), (0.79, 0.27), ORANGE)
    ax.text(0.49, 0.93, "Two policy-native branches", ha="center", va="center", fontsize=7)

    ax = axes[1]
    ax.set_title("b  Transplant future content", loc="left", weight="bold", fontsize=8)
    rounded_box(ax, (0.02, 0.66), 0.28, 0.16, "current $S$", LIGHT_GRAY, GRAY, fontsize=7)
    rounded_box(ax, (0.02, 0.42), 0.28, 0.16, "recipient\nnoise", LIGHT_BLUE, BLUE, fontsize=7)
    rounded_box(ax, (0.02, 0.18), 0.28, 0.16, "donor future\n$F_B$", LIGHT_ORANGE, ORANGE, fontsize=7)
    rounded_box(ax, (0.43, 0.34), 0.22, 0.30, "frozen\npolicy", "white", NAVY, weight="bold")
    rounded_box(ax, (0.77, 0.40), 0.20, 0.18, "patched\naction", LIGHT_ORANGE, ORANGE)
    for y in (0.74, 0.50, 0.26):
        arrow(ax, (0.30, y), (0.43, 0.50), GRAY if y != 0.26 else ORANGE)
    arrow(ax, (0.65, 0.49), (0.77, 0.49), ORANGE)
    ax.text(0.50, 0.09, "Only future content changes", ha="center", fontsize=7, color="#333333")

    ax = axes[2]
    ax.set_title("c  Signed steering + K/V patch", loc="left", weight="bold", fontsize=8)
    ax.plot([0.10, 0.88], [0.72, 0.72], color="#444444", linewidth=1.5)
    ax.scatter([0.10], [0.72], s=34, color=BLUE, zorder=3)
    ax.scatter([0.88], [0.72], s=34, color=ORANGE, zorder=3)
    ax.scatter([0.70], [0.72], s=42, marker="D", color=TEAL, zorder=4)
    ax.text(0.10, 0.61, "recipient\n0", ha="center", va="top", fontsize=7)
    ax.text(0.88, 0.61, "donor\n1", ha="center", va="top", fontsize=7)
    ax.text(0.70, 0.83, "transplant", ha="center", fontsize=7, color=TEAL, weight="bold")
    rounded_box(ax, (0.08, 0.15), 0.34, 0.18, "donor future\nK/V", LIGHT_ORANGE, ORANGE, fontsize=7)
    rounded_box(ax, (0.57, 0.15), 0.34, 0.18, "patch self\nK/V", LIGHT_BLUE, BLUE, fontsize=7)
    arrow(ax, (0.42, 0.24), (0.57, 0.24), NAVY)
    ax.text(0.50, 0.04, "Does donor steering disappear?", ha="center", fontsize=7)

    fig.subplots_adjust(wspace=0.17)
    save(fig, "method_overview")


def ci_point(ax, x, stat, color, marker, label=None, offset=0.0):
    y = stat["mean"] + offset
    ax.errorbar(
        [x],
        [y],
        yerr=[[y - stat["lower"]], [stat["upper"] - y]],
        fmt=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=6,
        linewidth=1.4,
        capsize=2.5,
        label=label,
        zorder=3,
    )


def make_predict2_results() -> None:
    summary = load_json("results/confirmatory_v1/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45))

    ax = axes[0]
    timings = ["0", "3", "6"]
    labels = ["First query", "After 3 chunks", "After 6 chunks"]
    for i, key in enumerate(timings):
        cell = summary["semantic_by_prefix_chunks"][key]["semantic_all_donor_minus_recipient"]
        ci_point(ax, i - 0.08, cell["action_donor_steering"], BLUE, "o", "Action" if i == 0 else None)
        ci_point(
            ax,
            i + 0.08,
            cell["physical_endpoint_donor_steering"],
            ORANGE,
            "s",
            "Executed endpoint" if i == 0 else None,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.text(2.42, 0.107, "SEOI", fontsize=6.5, color=GRAY, ha="right", va="bottom")
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Donor - self steering")
    ax.set_title("a  Strong early steering is state dependent", loc="left", weight="bold")
    ax.set_ylim(-0.12, 0.82)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    ax = axes[1]
    cell = summary["semantic_by_prefix_chunks"]["0"]
    keys = [
        ("All future", "semantic_all_donor_minus_recipient"),
        ("Wrist video", "semantic_wrist_donor_minus_recipient"),
        ("Primary video", "semantic_primary_donor_minus_recipient"),
        ("Proprioception", "semantic_proprio_donor_minus_recipient"),
    ]
    for i, (_, key) in enumerate(keys):
        color = NAVY if i == 0 else (ORANGE if i == 1 else BLUE if i == 2 else GRAY)
        ci_point(ax, i, cell[key]["action_donor_steering"], color, "o")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.text(3.45, 0.107, "SEOI", fontsize=6.5, color=GRAY, ha="right", va="bottom")
    ax.set_xticks(range(4), [k[0] for k in keys], rotation=18, ha="right")
    ax.set_ylabel("Action steering")
    ax.set_title("b  The useful future is visual", loc="left", weight="bold")
    ax.set_ylim(-0.12, 0.82)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    fig.subplots_adjust(wspace=0.34, bottom=0.22)
    save(fig, "predict2_results")


def make_cosmos3_results() -> None:
    branches = load_json("results/cosmos3_multitask_branches/summary.json")
    kv = load_json("results/cosmos3_kv_patch_v4/summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.1, 1.0, 1.2]})

    ax = axes[0]
    labels = ["Predicted", "Executed"]
    for i, source in enumerate(("predicted", "executed")):
        stat = branches["source_summaries"][source]
        for j, (metric, color, marker) in enumerate(
            (("action", BLUE, "o"), ("endpoint", ORANGE, "s"))
        ):
            mean = stat[f"{metric}_projection_mean"]
            lo, hi = stat[f"{metric}_projection_task_bootstrap_95ci"]
            x = i + (-0.08 if j == 0 else 0.08)
            ax.errorbar(
                [x],
                [mean],
                yerr=[[mean - lo], [hi - mean]],
                fmt=marker,
                color=color,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2.5,
                linewidth=1.3,
                label=metric.capitalize() if i == 0 else None,
            )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(1, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(2), labels)
    ax.set_ylim(-0.12, 1.30)
    ax.set_ylabel("Donor projection")
    ax.set_title("a  All 24 transplants steer", loc="left", weight="bold", fontsize=8.5)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    ax = axes[1]
    metrics = kv["calibration"]["metrics"]
    names = ["Layer 16", "All direct", "Full barrier"]
    keys = ["layer16_mediation_loss", "all_direct_mediation_loss", "all_barrier_mediation_loss"]
    colors = [BLUE, TEAL, ORANGE]
    for i, (key, color) in enumerate(zip(keys, colors)):
        stat = metrics[key]
        lo, hi = stat["task_bootstrap_95_interval"]
        ax.errorbar(
            [i],
            [stat["mean"]],
            yerr=[[stat["mean"] - lo], [hi - stat["mean"]]],
            fmt="o",
            color=color,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            capsize=2.5,
            linewidth=1.3,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(1, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(3), names, rotation=18, ha="right")
    ax.set_ylim(-0.12, 1.15)
    ax.set_ylabel("Mediation loss")
    ax.set_title("b  Future K/V mediates", loc="left", weight="bold", fontsize=8.5)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    ax = axes[2]
    holdout = kv["holdout"]["arms"]
    conditions = ["Donor", "Layer 16", "All direct", "Barrier"]
    for source, color, marker in (("predicted", BLUE, "o"), ("executed", ORANGE, "s")):
        arm = holdout[source]
        vals = [
            arm["baseline_action_projection"],
            arm["patches"]["selected"]["action_projection"],
            arm["patches"]["all_direct"]["action_projection"],
            arm["patches"]["all_barrier"]["action_projection"],
        ]
        ax.plot(range(4), vals, color=color, marker=marker, linewidth=1.4, markersize=5, label=source.capitalize())
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(range(4), conditions, rotation=18, ha="right")
    ax.set_ylim(-0.18, 1.78)
    ax.set_ylabel("Held-out action projection")
    ax.set_title("c  Held-out suppression", loc="left", weight="bold", fontsize=8.5)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    fig.subplots_adjust(wspace=0.43, bottom=0.24)
    save(fig, "cosmos3_results")


if __name__ == "__main__":
    make_overview()
    make_predict2_results()
    make_cosmos3_results()
