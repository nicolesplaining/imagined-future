"""Generate manuscript figures from the paper summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
    """Plain-language causal test illustrated with one held-out LIBERO trace."""
    data = np.load(OUT / "method_example_data.npz")
    fig = plt.figure(figsize=(7.2, 3.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def frame(image, extent, title, color=GRAY):
        x0, x1, y0, y1 = extent
        ax.imshow(image, extent=extent, aspect="auto", zorder=1)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor=color, linewidth=1.6, zorder=2))
        ax.text((x0 + x1) / 2, y1 + 0.018, title, ha="center", va="bottom",
                fontsize=6.8, weight="bold", color=color)

    ax.text(0.01, 0.965, "a  Two normal rollouts from exactly the same state", weight="bold", fontsize=8.3)
    frame(data["start"], (0.02, 0.19, 0.34, 0.79), "same start", NAVY)
    ax.text(0.105, 0.305, "same observation\nand instruction", ha="center", va="top", fontsize=6.1)

    frame(data["future_a"], (0.28, 0.405, 0.60, 0.87), "Recipient future", BLUE)
    frame(data["endpoint_a"], (0.47, 0.595, 0.60, 0.87), "Recipient action", BLUE)
    frame(data["future_b"], (0.28, 0.405, 0.20, 0.47), "Donor future", ORANGE)
    frame(data["endpoint_b"], (0.47, 0.595, 0.20, 0.47), "Donor action", ORANGE)
    arrow(ax, (0.19, 0.58), (0.28, 0.735), BLUE)
    arrow(ax, (0.19, 0.52), (0.28, 0.335), ORANGE)
    arrow(ax, (0.405, 0.735), (0.47, 0.735), BLUE)
    arrow(ax, (0.405, 0.335), (0.47, 0.335), ORANGE)
    ax.text(0.438, 0.765, "+ recipient action", ha="center", fontsize=5.4, color=BLUE)
    ax.text(0.438, 0.365, "+ donor action", ha="center", fontsize=5.4, color=ORANGE)
    ax.text(0.435, 0.09, "Agreement within each rollout is only correlation.",
            ha="center", fontsize=6.7, weight="bold")

    ax.plot([0.625, 0.625], [0.06, 0.94], color="#C8C8C8", linewidth=0.9)
    ax.text(0.65, 0.965, "b  Stage 1: insert donor future into recipient", weight="bold", fontsize=8.0)
    rounded_box(ax, (0.65, 0.72), 0.15, 0.13, "held fixed\nstart, instruction,\noriginal noise + solver", LIGHT_GRAY, GRAY,
                fontsize=5.5, weight="bold")
    frame(data["future_b"], (0.83, 0.965, 0.70, 0.87), "only the inserted future changes", ORANGE)
    arrow(ax, (0.80, 0.785), (0.83, 0.785), ORANGE)

    # Project the three observed action chunks into a common two-dimensional
    # PCA plane. This is a display of the actual 16-step action sequences, not
    # the metric used for inference.
    action_a = data["action_a"]
    action_b = data["action_b"]
    action_patch = data["action_patch"]
    all_steps = np.concatenate([action_a, action_b, action_patch], axis=0)
    centered = all_steps - all_steps.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    projected = [(action - all_steps.mean(axis=0, keepdims=True)) @ basis
                 for action in (action_a, action_b, action_patch)]
    trajectory_ax = fig.add_axes([0.655, 0.31, 0.19, 0.29])
    for points, color, label, width in (
        (projected[0], BLUE, "Recipient action", 1.2),
        (projected[1], ORANGE, "Donor action", 1.2),
        (projected[2], TEAL, "after donor insert", 2.0),
    ):
        trajectory_ax.plot(points[:, 0], points[:, 1], color=color, linewidth=width, label=label)
        trajectory_ax.scatter(points[0, 0], points[0, 1], s=9, color=color, zorder=3)
    trajectory_ax.set_xticks([])
    trajectory_ax.set_yticks([])
    trajectory_ax.set_title("actual 16-step actions (PCA)", fontsize=6.2, pad=2)
    trajectory_ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    trajectory_ax.spines[["top", "right", "bottom", "left"]].set_color("#BBBBBB")

    frame(data["endpoint_patch"], (0.865, 0.985, 0.32, 0.59), "new action executed", TEAL)
    arrow(ax, (0.845, 0.455), (0.865, 0.455), TEAL)

    direction = action_b - action_a
    denom = float(np.sum(direction * direction))
    self_projection = float(np.sum((data["action_self"] - action_a) * direction) / denom)
    patch_projection = float(np.sum((action_patch - action_a) * direction) / denom)
    ax.plot([0.67, 0.96], [0.17, 0.17], color="#444444", linewidth=1.3)
    ax.scatter([0.67, 0.96], [0.17, 0.17], s=22, color=[BLUE, ORANGE], zorder=3)
    scale = lambda value: 0.67 + 0.29 * value
    ax.scatter([scale(self_projection)], [0.17], marker="|", s=85, color=BLUE, linewidths=2, zorder=4)
    ax.scatter([scale(patch_projection)], [0.17], marker="D", s=30, color=TEAL, zorder=4)
    ax.text(0.67, 0.13, "Original action\n(recipient)  0", ha="center", va="top", fontsize=5.8, color=BLUE)
    ax.text(0.96, 0.13, "Reference action\n(donor)  1", ha="center", va="top", fontsize=5.8, color=ORANGE)
    ax.text(scale(self_projection), 0.205, "self rerun", ha="center", va="bottom", fontsize=5.6, color=BLUE)
    ax.text(scale(patch_projection), 0.205, f"new action {patch_projection:.2f}", ha="center", va="bottom",
            fontsize=5.8, color=TEAL, weight="bold")
    ax.text(0.815, 0.035, "The reference action is never given to the model; it is used only for scoring.",
            ha="center", fontsize=6.2, weight="bold")
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
            ax.text(donor["mean"], y + 0.18, f"Donor {donor['mean']:.3f}",
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
               markerfacecolor=ORANGE, label="Paired donor"),
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
    ax.set_xlabel("Isolated-factor effect\nGray = only donor robot/object pixels inserted")
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
