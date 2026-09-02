"""Generate manuscript figures from the paper summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
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


def load_summary() -> dict:
    return json.loads((ROOT / "results" / "paper_summary.json").read_text())


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def rounded_box(ax, xy, width, height, label, face, edge, fontsize=8, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#202020",
    )


def arrow(ax, start, end, color="#333333"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.2,
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
    ax.set_title("a  Alternative runs", loc="left", weight="bold", fontsize=7.5)
    rounded_box(ax, (0.04, 0.39), 0.18, 0.22, "saved\nstate $S$", LIGHT_GRAY, GRAY, fontsize=6.3, weight="bold")
    rounded_box(ax, (0.42, 0.63), 0.28, 0.20, "recipient future\n$F_A$", LIGHT_BLUE, BLUE, fontsize=5.8)
    rounded_box(ax, (0.42, 0.17), 0.28, 0.20, "donor future\n$F_B$", LIGHT_ORANGE, ORANGE, fontsize=5.8)
    rounded_box(ax, (0.79, 0.63), 0.17, 0.20, "action\n$a_A$", "white", BLUE, fontsize=6.3)
    rounded_box(ax, (0.79, 0.17), 0.17, 0.20, "action\n$a_B$", "white", ORANGE, fontsize=6.3)
    arrow(ax, (0.22, 0.52), (0.42, 0.73), BLUE)
    arrow(ax, (0.22, 0.48), (0.42, 0.27), ORANGE)
    arrow(ax, (0.70, 0.73), (0.79, 0.73), BLUE)
    arrow(ax, (0.70, 0.27), (0.79, 0.27), ORANGE)
    ax.text(0.49, 0.93, "Two runs from the same state", ha="center", va="center", fontsize=7)

    ax = axes[1]
    ax.set_title("b  Replace the future", loc="left", weight="bold", fontsize=7.5)
    rounded_box(ax, (0.02, 0.66), 0.28, 0.16, "state $S$", LIGHT_GRAY, GRAY, fontsize=7)
    rounded_box(ax, (0.02, 0.42), 0.28, 0.16, "noise from\nrun $A$", LIGHT_BLUE, BLUE, fontsize=7)
    rounded_box(ax, (0.02, 0.18), 0.28, 0.16, "donor future\n$F_B$ only", LIGHT_ORANGE, ORANGE, fontsize=7)
    rounded_box(ax, (0.43, 0.34), 0.22, 0.30, "same\npolicy", "white", NAVY, weight="bold")
    rounded_box(ax, (0.77, 0.40), 0.20, 0.18, "action after\nreplacement", LIGHT_ORANGE, ORANGE, fontsize=5.7)
    for y in (0.74, 0.50, 0.26):
        arrow(ax, (0.30, y), (0.43, 0.50), GRAY if y != 0.26 else ORANGE)
    arrow(ax, (0.65, 0.49), (0.77, 0.49), ORANGE)
    ax.text(0.50, 0.09, "Recipient inputs and noise stay fixed", ha="center", fontsize=6.5, color="#333333")

    ax = axes[2]
    ax.set_title("c  Measure the effect", loc="left", weight="bold", fontsize=7.5)
    ax.plot([0.10, 0.88], [0.72, 0.72], color="#444444", linewidth=1.5)
    ax.scatter([0.10], [0.72], s=34, color=BLUE, zorder=3)
    ax.scatter([0.88], [0.72], s=34, color=ORANGE, zorder=3)
    ax.scatter([0.70], [0.72], s=42, marker="D", color=TEAL, zorder=4)
    ax.text(0.10, 0.61, "recipient $A$\n0", ha="center", va="top", fontsize=7)
    ax.text(0.88, 0.61, "donor $B$\n1", ha="center", va="top", fontsize=7)
    ax.text(0.70, 0.83, "action after replacement", ha="center", fontsize=6.5, color=TEAL, weight="bold")
    rounded_box(ax, (0.08, 0.15), 0.34, 0.18, "future K/V\nfrom run $B$", LIGHT_ORANGE, ORANGE, fontsize=7)
    rounded_box(ax, (0.57, 0.15), 0.34, 0.18, "replace with K/V\nfrom run $A$", LIGHT_BLUE, BLUE, fontsize=7)
    arrow(ax, (0.42, 0.24), (0.57, 0.24), NAVY)
    ax.text(0.50, 0.04, "Donor action is used only for scoring", ha="center", fontsize=6.4)

    fig.subplots_adjust(wspace=0.17)
    save(fig, "method_overview")


def error_point(ax, x, stat, color, marker="o", label=None):
    mean = stat["mean"]
    if "ci" in stat:
        low, high = stat["ci"]
        yerr = [[mean - low], [high - mean]]
    else:
        yerr = None
    ax.errorbar(
        [x],
        [mean],
        yerr=yerr,
        fmt=marker,
        color=color,
        markerfacecolor=color if yerr is not None else "white",
        markeredgecolor=color,
        markeredgewidth=1.0,
        markersize=6,
        linewidth=1.3,
        capsize=2.5,
        label=label,
        zorder=3,
    )


def make_directionality(summary: dict) -> None:
    """Cosmos Policy: state dependence and modality localization."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.28), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    timing = summary["cosmos_policy"]["timing"]
    for i, cell in enumerate(timing):
        error_point(ax, i - 0.08, cell["action"], BLUE, "o", "Action chunk" if i == 0 else None)
        error_point(ax, i + 0.08, cell["endpoint"], ORANGE, "s", "Executed endpoint" if i == 0 else None)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.text(2.43, 0.107, "Prespecified 0.10 threshold", fontsize=5.8, color=GRAY, ha="right", va="bottom")
    ax.set_xticks(range(3), ["First decision", "After 3 action chunks", "After 6 action chunks"])
    ax.set_ylabel("Projection difference\n(donor future $-$ self future)")
    ax.set_title("a  Steering is strongest at the first query", loc="left", weight="bold")
    ax.set_ylim(-0.12, 0.82)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    ax = axes[1]
    all_future = summary["cosmos_policy"]["timing"][0]["action"]
    modalities = summary["cosmos_policy"]["modalities"]
    for i, stat in enumerate([all_future, *modalities.values()]):
        error_point(ax, i, stat, [TEAL, ORANGE, BLUE, GRAY][i])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(4), ["All\nfuture", "Wrist\nvideo", "Primary\nvideo", "Future\nproprio."])
    ax.set_ylim(-0.08, 0.68)
    ax.set_ylabel("Difference in action projection")
    ax.set_title("b  Steering is carried by visual futures", loc="left", weight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    fig.subplots_adjust(wspace=0.35, bottom=0.20)
    save(fig, "directionality_results")


def make_content(summary: dict) -> None:
    data = summary["cosmos_policy"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    ax = axes[0]
    modalities = data["modalities"]
    for i, (name, stat) in enumerate(modalities.items()):
        error_point(ax, i, stat, [ORANGE, BLUE, GRAY][i])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(3), ["Wrist\ncamera", "Primary\ncamera", "Proprioception"])
    ax.set_ylim(-0.08, 0.68)
    ax.set_ylabel("Difference in action projection")
    ax.set_title("a  Input modality", loc="left", weight="bold", fontsize=8.5)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    ax = axes[1]
    effects = data["object_main_effect"]
    error_point(ax, 0, effects["action"], BLUE, "o")
    error_point(ax, 1, effects["endpoint"], ORANGE, "s")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 1], ["Action\nchunk", "Goal-state\nendpoint"])
    ax.set_ylim(-0.015, 0.115)
    ax.set_ylabel("Estimated effect of object state")
    ax.set_title("b  Object-state intervention", loc="left", weight="bold", fontsize=8.5)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    ax = axes[2]
    natural = data["natural_robot_only"]
    error_point(ax, 0, natural["action"], BLUE, "o")
    error_point(ax, 1, natural["endpoint"], ORANGE, "s")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 1], ["Action\nchunk", "Executed robot\nendpoint"])
    ax.set_ylim(-0.08, 0.68)
    ax.set_ylabel("Difference in projection")
    ax.set_title("c  Natural robot-motion pairs", loc="left", weight="bold", fontsize=8.5)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    fig.subplots_adjust(wspace=0.43, bottom=0.20)
    save(fig, "content_results")


def make_pathway(summary: dict) -> None:
    """Cosmos 3: steering, K/V replacement, and isolated pixel factors."""
    data = summary["cosmos3"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    ax = axes[0]
    steering = data["directionality"]
    primary = [steering["predicted_action"], steering["executed_action"], steering["executed_endpoint"]]
    for i, stat in enumerate(primary):
        error_point(ax, i - 0.08, stat, BLUE if i < 2 else ORANGE, "o" if i < 2 else "s")
    for i, stat in enumerate([steering["predicted_minus_natural"], steering["executed_minus_natural"]]):
        error_point(ax, i + 0.10, stat, TEAL, "D")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(1, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(3), ["Predicted\naction", "Executed\naction", "Executed\nendpoint"])
    ax.set_ylim(-0.08, 1.17)
    ax.set_ylabel("Donor-directed projection")
    ax.set_title("a  Coherent futures steer", loc="left", weight="bold")
    ax.plot([], [], color=BLUE, marker="o", linestyle="none", label="Donor $-$ self")
    ax.plot([], [], color=TEAL, marker="D", linestyle="none", label="Donor $-$ natural")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    ax = axes[1]
    losses = data["kv_replacement_loss"]
    for i, stat in enumerate(losses.values()):
        error_point(ax, i, stat, [BLUE, TEAL, ORANGE][i], "s" if i == 2 else "o")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(3), ["Predicted\naction", "Executed\naction", "Executed\nendpoint"])
    ax.set_ylim(-0.05, 1.02)
    ax.set_ylabel("Reduction in donor projection")
    ax.set_title("b  Future K/V carries steering", loc="left", weight="bold", fontsize=8)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    ax = axes[2]
    factors = data["factor_study"]
    for i, name in enumerate(("robot", "object")):
        error_point(ax, i - 0.08, factors[f"{name}_action"], BLUE, "o", "Action" if i == 0 else None)
        error_point(ax, i + 0.08, factors[f"{name}_endpoint"], ORANGE, "s", "Endpoint" if i == 0 else None)
    ax.axhspan(-0.10, 0.10, color=LIGHT_GRAY, zorder=0)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(-0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.axhline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(2), ["Robot pixels", "Object pixels"])
    ax.set_ylim(-0.14, 0.14)
    ax.set_ylabel("Isolated-factor effect")
    ax.set_title("c  Isolated factors are negligible", loc="left", weight="bold", fontsize=8)
    ax.legend(frameon=False, loc="upper center")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    fig.subplots_adjust(wspace=0.48, bottom=0.22)
    save(fig, "pathway_results")


if __name__ == "__main__":
    summary = load_summary()
    make_overview()
    make_directionality(summary)
    make_pathway(summary)
