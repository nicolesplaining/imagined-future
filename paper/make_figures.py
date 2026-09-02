"""Generate manuscript figures from the paper summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


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
    """Four-step schematic with one question per panel."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.65))
    axes = axes.ravel()
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("a  Generate two reachable alternatives", loc="left", weight="bold", fontsize=8)
    rounded_box(ax, (0.04, 0.39), 0.18, 0.22, "saved\nstate $S$", LIGHT_GRAY, GRAY, fontsize=6.3, weight="bold")
    rounded_box(ax, (0.42, 0.63), 0.28, 0.20, "recipient future\n$F_A$", LIGHT_BLUE, BLUE, fontsize=5.8)
    rounded_box(ax, (0.42, 0.17), 0.28, 0.20, "donor future\n$F_B$", LIGHT_ORANGE, ORANGE, fontsize=5.8)
    rounded_box(ax, (0.79, 0.63), 0.17, 0.20, "action\n$a_A$", "white", BLUE, fontsize=6.3)
    rounded_box(ax, (0.79, 0.17), 0.17, 0.20, "action\n$a_B$", "white", ORANGE, fontsize=6.3)
    arrow(ax, (0.22, 0.52), (0.42, 0.73), BLUE)
    arrow(ax, (0.22, 0.48), (0.42, 0.27), ORANGE)
    arrow(ax, (0.70, 0.73), (0.79, 0.73), BLUE)
    arrow(ax, (0.70, 0.27), (0.79, 0.27), ORANGE)
    ax.text(0.49, 0.93, "Same state and instruction; different native draws", ha="center", va="center", fontsize=6.8)

    ax = axes[1]
    ax.set_title("b  Transplant only the donor future", loc="left", weight="bold", fontsize=8)
    rounded_box(ax, (0.02, 0.66), 0.28, 0.16, "state $S$", LIGHT_GRAY, GRAY, fontsize=7)
    rounded_box(ax, (0.02, 0.42), 0.28, 0.16, "noise from\nrun $A$", LIGHT_BLUE, BLUE, fontsize=7)
    rounded_box(ax, (0.02, 0.18), 0.28, 0.16, "donor future\n$F_B$ only", LIGHT_ORANGE, ORANGE, fontsize=7)
    rounded_box(ax, (0.43, 0.34), 0.22, 0.30, "same\npolicy", "white", NAVY, weight="bold")
    rounded_box(ax, (0.77, 0.40), 0.20, 0.18, "action after\nreplacement", LIGHT_ORANGE, ORANGE, fontsize=5.7)
    for y in (0.74, 0.50, 0.26):
        arrow(ax, (0.30, y), (0.43, 0.50), GRAY if y != 0.26 else ORANGE)
    arrow(ax, (0.65, 0.49), (0.77, 0.49), ORANGE)
    ax.text(0.50, 0.09, "Observation, instruction, and recipient noise stay fixed", ha="center", fontsize=6.3, color="#333333")

    ax = axes[2]
    ax.set_title("c  Score the direction of the change", loc="left", weight="bold", fontsize=8)
    ax.plot([0.10, 0.88], [0.72, 0.72], color="#444444", linewidth=1.5)
    ax.scatter([0.10], [0.72], s=34, color=BLUE, zorder=3)
    ax.scatter([0.88], [0.72], s=34, color=ORANGE, zorder=3)
    ax.scatter([0.70], [0.72], s=42, marker="D", color=TEAL, zorder=4)
    ax.text(0.10, 0.61, "recipient $A$\n0", ha="center", va="top", fontsize=7)
    ax.text(0.88, 0.61, "donor $B$\n1", ha="center", va="top", fontsize=7)
    ax.text(0.70, 0.83, "action after replacement", ha="center", fontsize=6.5, color=TEAL, weight="bold")
    ax.text(0.50, 0.25, "Projection = 0: recipient\nProjection = 1: donor", ha="center", va="center", fontsize=7)
    ax.text(0.50, 0.05, "The donor action is never given to the policy", ha="center", fontsize=6.4, weight="bold")

    ax = axes[3]
    ax.set_title("d  Test the future-to-action pathway", loc="left", weight="bold", fontsize=8)
    rounded_box(ax, (0.05, 0.61), 0.34, 0.18, "donor-future K/V", LIGHT_ORANGE, ORANGE, fontsize=7)
    rounded_box(ax, (0.61, 0.61), 0.34, 0.18, "donor-directed\naction", LIGHT_ORANGE, ORANGE, fontsize=7)
    arrow(ax, (0.39, 0.70), (0.61, 0.70), ORANGE)
    rounded_box(ax, (0.05, 0.23), 0.34, 0.18, "self-future K/V", LIGHT_BLUE, BLUE, fontsize=7)
    rounded_box(ax, (0.61, 0.23), 0.34, 0.18, "reduced donor\nprojection", LIGHT_BLUE, BLUE, fontsize=7)
    arrow(ax, (0.39, 0.32), (0.61, 0.32), BLUE)
    ax.text(0.50, 0.06, "Only future-token keys and values are replaced", ha="center", fontsize=6.4)

    legend = [
        Patch(facecolor=LIGHT_BLUE, edgecolor=BLUE, label="Blue = recipient / self information"),
        Patch(facecolor=LIGHT_ORANGE, edgecolor=ORANGE, label="Orange = donor information"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=TEAL,
               markeredgecolor=TEAL, label="Teal = action after intervention"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=3, frameon=False, fontsize=6.5)
    fig.subplots_adjust(wspace=0.15, hspace=0.34, bottom=0.10)
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


def horizontal_error_point(ax, y, stat, color, marker="o", label=None):
    mean = stat["mean"]
    low, high = stat["ci"]
    ax.errorbar(
        [mean],
        [y],
        xerr=[[mean - low], [high - mean]],
        fmt=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=5.5,
        linewidth=1.3,
        capsize=2.3,
        label=label,
        zorder=3,
    )


def make_directionality(summary: dict) -> None:
    """Show the paired donor against the strongest available control."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharex=True,
                             gridspec_kw={"width_ratios": [1.05, 0.95]})

    cosmos3 = summary["cosmos3"]
    policy = summary["cosmos_policy"]
    panels = [
        (
            axes[0],
            "a  Cosmos 3",
            [
                ("Predicted action", cosmos3["condition_projections"]["predicted_action"]["natural"],
                 "Natural", cosmos3["condition_projections"]["predicted_action"]["donor"], 0.433),
                ("Executed action", cosmos3["condition_projections"]["executed_action"]["natural"],
                 "Natural", cosmos3["condition_projections"]["executed_action"]["donor"], 0.391),
                ("Executed endpoint", cosmos3["condition_projections"]["executed_endpoint"]["self"],
                 "Self", cosmos3["condition_projections"]["executed_endpoint"]["donor"], 1.003),
            ],
        ),
        (
            axes[1],
            "b  Cosmos Policy",
            [
                ("Predicted action", policy["condition_projections"]["predicted_action"]["natural"],
                 "Natural", policy["condition_projections"]["predicted_action"]["donor"], 0.103),
                ("Predicted endpoint", policy["condition_projections"]["predicted_endpoint"]["natural"],
                 "Natural", policy["condition_projections"]["predicted_endpoint"]["donor"], 0.110),
            ],
        ),
    ]

    for ax, title, rows in panels:
        for y, (outcome, control, control_name, donor, advantage) in enumerate(rows):
            ax.plot([control["mean"], donor["mean"]], [y, y], color="#C8C8C8",
                    linewidth=2.2, solid_capstyle="round", zorder=1)
            horizontal_error_point(ax, y, control, GRAY, "o")
            horizontal_error_point(ax, y, donor, ORANGE, "D")
            control_label_x = 0.025 if control_name == "Self" else control["mean"]
            control_label_alignment = "left" if control_name == "Self" else "center"
            ax.text(control_label_x, y - 0.18, f"{control_name} {control['mean']:.3f}",
                    ha=control_label_alignment, va="top", fontsize=6.1, color=GRAY)
            ax.text(donor["mean"], y + 0.18, f"Ours {donor['mean']:.3f}",
                    ha="center", va="bottom", fontsize=6.1, color=ORANGE, weight="bold")
        labels = [f"{outcome}\nadvantage +{advantage:.3f}" for outcome, _, _, _, advantage in rows]
        ax.set_yticks(range(len(rows)), labels)
        ax.set_ylim(len(rows) - 0.45, -0.55)
        ax.axvline(0, color="#777777", linewidth=0.7)
        ax.axvline(1.0, color=ORANGE, linewidth=0.75, linestyle=":")
        ax.set_xlim(-0.05, 1.13)
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.5)
        ax.set_xlabel("Donor-directed projection\n0 = self, 1 = paired donor")

    legend = [
        Line2D([0], [0], marker="o", color=GRAY, linestyle="none",
               markerfacecolor=GRAY, label="Strongest available control"),
        Line2D([0], [0], marker="D", color=ORANGE, linestyle="none",
               markerfacecolor=ORANGE, label="Paired donor (ours)"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               ncol=2, frameon=False, fontsize=6.7, columnspacing=2.2)
    fig.subplots_adjust(wspace=0.42, bottom=0.29, left=0.18, right=0.98)
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
    """Which content and pathway carry the effect?"""
    data = summary["cosmos3"]
    policy = summary["cosmos_policy"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.80), gridspec_kw={"width_ratios": [1.0, 1.08, 1.08]})

    ax = axes[0]
    modalities = policy["modalities"]
    policy_rows = [
        ("Whole future", policy["timing"][0]["action"]),
        ("Wrist video only", modalities["wrist"]),
        ("Primary video only", modalities["primary"]),
        ("Proprioception only", modalities["proprioception"]),
    ]
    for y, (_, stat) in enumerate(reversed(policy_rows)):
        horizontal_error_point(ax, y, stat, ORANGE)
        ax.text(stat["mean"] + 0.025, y, f"{stat['mean']:.3f}", va="center", fontsize=6.1)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.axvline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(policy_rows)), [r[0] for r in reversed(policy_rows)])
    ax.set_xlim(-0.05, 0.69)
    ax.set_xlabel("Action steering")
    ax.set_title("a  Cosmos Policy\nWrist video carries the effect", loc="left", weight="bold", fontsize=7.6)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)

    ax = axes[1]
    factors = data["factor_study"]
    factor_rows = [("Robot — action", factors["robot_action"], GRAY, "o"),
                   ("Robot — endpoint", factors["robot_endpoint"], GRAY, "s"),
                   ("Object — action", factors["object_action"], GRAY, "o"),
                   ("Object — endpoint", factors["object_endpoint"], GRAY, "s")]
    ax.axvspan(-0.10, 0.10, color=LIGHT_GRAY, zorder=0)
    for y, (_, stat, color, marker) in enumerate(reversed(factor_rows)):
        horizontal_error_point(ax, y, stat, color, marker)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.axvline(-0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.axvline(0.10, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(factor_rows)), [r[0] for r in reversed(factor_rows)])
    ax.set_xlim(-0.14, 0.14)
    ax.set_xlabel("Isolated-factor effect\nGray = robot/object-only transplant")
    ax.set_title("b  Cosmos 3 content\nIsolated pixels are insufficient", loc="left", weight="bold", fontsize=7.6)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)

    ax = axes[2]
    losses = data["kv_replacement_loss"]
    labels = ["Predicted action", "Executed action", "Executed endpoint"]
    fractions = [losses["predicted_action"]["fraction_of_unpatched"],
                 losses["executed_action"]["fraction_of_unpatched"],
                 losses["executed_endpoint"]["fraction_of_unpatched"]]
    for y, frac in enumerate(reversed(fractions)):
        ax.barh(y, frac, height=0.45, color=BLUE, label="Removed by self-future K/V" if y == 0 else None)
        ax.barh(y, 1 - frac, left=frac, height=0.45, color=LIGHT_GRAY, edgecolor=GRAY,
                label="Remaining donor effect" if y == 0 else None)
        ax.text(frac / 2, y, f"{100 * frac:.0f}% removed", ha="center", va="center",
                color="white", fontsize=6.2, weight="bold")
    ax.set_yticks(range(3), list(reversed(labels)))
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("Fraction of donor effect\nBlue = removed; gray = remaining")
    ax.set_title("c  Cosmos 3 pathway\nSelf-future K/V removes most effect", loc="left", weight="bold", fontsize=7.6)
    ax.grid(axis="x", color="#FFFFFF", linewidth=0.6)

    fig.subplots_adjust(wspace=0.68, bottom=0.20)
    save(fig, "pathway_results")


if __name__ == "__main__":
    summary = load_summary()
    make_overview()
    make_directionality(summary)
    make_pathway(summary)
