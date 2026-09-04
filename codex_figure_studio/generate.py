"""Generate a self-contained figure set directly from repository result data.

This directory is intentionally independent of ``paper/make_figures.py`` and
``scripts/plot_main_data_figures.py``.  The main figures expose restored-state
means and ordinary paired relationships; repetitions are averaged within state
before any point or interval is drawn.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "rendered"
DEFAULT_TABLES = HERE / "tables"

TASK_ORDER = [
    "BananaInBowlTask",
    "MarkerInMugTask",
    "MustardInLeftBinTask",
    "RubiksCubeTask",
    "SmartphoneInBinTask",
    "SpoonInMugTask",
]
TASK_LABEL = {
    "BananaInBowlTask": "banana → bowl",
    "MarkerInMugTask": "marker → mug",
    "MustardInLeftBinTask": "mustard → bin",
    "RubiksCubeTask": "Rubik's cube",
    "SmartphoneInBinTask": "phone → bin",
    "SpoonInMugTask": "spoon → mug",
}
TASK_SHORT = {
    "BananaInBowlTask": "banana",
    "MarkerInMugTask": "marker",
    "MustardInLeftBinTask": "mustard",
    "RubiksCubeTask": "cube",
    "SmartphoneInBinTask": "phone",
    "SpoonInMugTask": "spoon",
}
TASK_COLOR = {
    "BananaInBowlTask": "#0072B2",
    "MarkerInMugTask": "#D55E00",
    "MustardInLeftBinTask": "#009E73",
    "RubiksCubeTask": "#CC79A7",
    "SmartphoneInBinTask": "#E69F00",
    "SpoonInMugTask": "#56B4E9",
}

RECIPIENT = "#2F6F9F"
DONOR = "#D36B2D"
ACTION = "#245B8A"
ENDPOINT = "#B45F3C"
INTERVENTION = "#278F78"
GRID = "#E7E7E7"
REFERENCE = "#777777"
TEXT = "#222222"
EQUIVALENCE = "#F1F1F1"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def deep_get(mapping: dict, dotted: str):
    value = mapping
    for key in dotted.split("."):
        value = value[key]
    return value


def load_population(root: Path) -> list[dict]:
    folder = root / "results" / "cosmos3_population_confirmatory_v1"
    fields = {
        "pred_self": "self.action_donor_projection",
        "pred_natural": "natural_control.action_donor_projection",
        "pred_donor": "predicted_donor.action_donor_projection",
        "exec_self": "executed_self.action_donor_projection",
        "exec_gaussian": "gaussian_executed.action_donor_projection",
        "exec_natural": "natural_control.action_donor_projection",
        "exec_donor": "executed_donor.action_donor_projection",
        "endpoint_self": "executed_self.endpoint_donor_projection.all",
        "endpoint_gaussian": "gaussian_executed.endpoint_donor_projection.all",
        "endpoint_natural": "natural_control.endpoint_donor_projection.all",
        "endpoint_donor": "executed_donor.endpoint_donor_projection.all",
        "pred_l2": "predicted_donor.action_l2_from_recipient",
        "exec_l2": "executed_donor.action_l2_from_recipient",
        "endpoint_l2": "executed_donor.endpoint_l2_from_recipient.all",
        "pred_patch": "predicted_donor_kv_patch_all_action.action_donor_projection",
        "exec_patch": "executed_donor_kv_patch_all_action.action_donor_projection",
        "endpoint_patch": (
            "executed_donor_kv_patch_all_action.endpoint_donor_projection.all"
        ),
    }
    rows = []
    for path in folder.glob("*/summary.json"):
        data = json.loads(path.read_text())
        interventions = data["interventions"]
        row = {
            "unit": f"{data['task']}-seed-{data['environment_seed']}",
            "task": data["task"],
            "seed": int(data["environment_seed"]),
            "native_action_l2": float(data["native_action_l2"]),
            "native_endpoint_l2": float(data["native_endpoint_l2"]["all"]),
        }
        row.update({name: float(deep_get(interventions, field)) for name, field in fields.items()})
        row["pred_loss"] = row["pred_donor"] - row["pred_patch"]
        row["exec_loss"] = row["exec_donor"] - row["exec_patch"]
        row["endpoint_loss"] = row["endpoint_donor"] - row["endpoint_patch"]
        rows.append(row)
    rows.sort(key=lambda row: (TASK_ORDER.index(row["task"]), row["seed"]))
    if len(rows) != 22:
        raise ValueError(f"expected 22 Cosmos 3 population states, found {len(rows)}")
    return rows


def load_factorization(root: Path) -> list[dict]:
    folder = root / "results" / "cosmos3_population_confirmatory_v1"
    rows = []
    for path in folder.glob("*/summary.json"):
        data = json.loads(path.read_text())
        effects = (data.get("factorial_effects") or {}).get("composite")
        if not effects:
            continue
        action = effects["action_donor_projection_effects"]
        endpoint = effects["endpoint_donor_projection_effects"]["all"]
        row = {
            "unit": f"{data['task']}-seed-{data['environment_seed']}",
            "task": data["task"],
            "seed": int(data["environment_seed"]),
            "full_action": float(data["interventions"]["executed_donor"]["action_donor_projection"]),
            "full_endpoint": float(
                data["interventions"]["executed_donor"]["endpoint_donor_projection"]["all"]
            ),
            "action_robot": float(action["robot_main_effect"]),
            "action_object": float(action["object_main_effect"]),
            "action_interaction": float(action["interaction"]),
            "endpoint_robot": float(endpoint["robot_main_effect"]),
            "endpoint_object": float(endpoint["object_main_effect"]),
            "endpoint_interaction": float(endpoint["interaction"]),
        }
        for cell, value in effects["action_donor_projection_cells"].items():
            row[f"action_{cell}"] = float(value)
        for cell, value in effects["endpoint_donor_projection_cells"]["all"].items():
            row[f"endpoint_{cell}"] = float(value)
        rows.append(row)
    rows.sort(key=lambda row: (TASK_ORDER.index(row["task"]), row["seed"]))
    if len(rows) != 10:
        raise ValueError(f"expected 10 Cosmos 3 factor states, found {len(rows)}")
    return rows


def load_policy_state_means(root: Path) -> list[dict]:
    path = root / "results" / "confirmatory_v1" / "semantic_state_repetitions.csv"
    groups: dict[tuple, dict] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["unit_id"],
                int(row["task_id"]),
                int(row["prefix_chunks"]),
                row["contrast"],
            )
            accumulator = groups.setdefault(
                key,
                {"action": [], "endpoint": []},
            )
            accumulator["action"].append(float(row["action_donor_steering"]))
            accumulator["endpoint"].append(
                float(row["physical_endpoint_donor_steering"])
            )
    rows = []
    for (unit, task, prefix, contrast), accumulator in groups.items():
        rows.append(
            {
                "unit": unit,
                "task": task,
                "prefix": prefix,
                "contrast": contrast,
                "action": float(np.mean(accumulator["action"])),
                "endpoint": float(np.mean(accumulator["endpoint"])),
                "repetitions": len(accumulator["action"]),
            }
        )
    rows.sort(key=lambda row: (row["task"], row["prefix"], row["contrast"]))
    return rows


def values(rows: Sequence[dict], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def bootstrap_mean(
    sample: Iterable[float], *, seed: int, resamples: int = 10_000
) -> tuple[float, float, float]:
    sample = np.asarray(list(sample), dtype=float)
    if sample.ndim != 1 or len(sample) == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("bootstrap sample must be a nonempty finite vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(sample), size=(resamples, len(sample)))
    means = sample[indices].mean(axis=1)
    return (
        float(sample.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def add_mean_ci(
    ax: plt.Axes,
    x: float,
    sample: Iterable[float],
    *,
    seed: int,
    color: str = "black",
    marker: str = "D",
    facecolor: str | None = None,
    size: float = 4.0,
    statistic: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    mean, lower, upper = (
        bootstrap_mean(sample, seed=seed) if statistic is None else statistic
    )
    ax.errorbar(
        x,
        mean,
        yerr=[[mean - lower], [upper - mean]],
        fmt=marker,
        color=color,
        markerfacecolor=color if facecolor is None else facecolor,
        markeredgecolor=color,
        markeredgewidth=0.8,
        markersize=size,
        linewidth=1.0,
        capsize=2.2,
        zorder=6,
    )
    return mean, lower, upper


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def task_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=TASK_COLOR[task],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=5,
            label=TASK_LABEL[task],
        )
        for task in TASK_ORDER
    ]


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    figure.savefig(
        output / f"{name}.png", dpi=360, bbox_inches="tight", pad_inches=0.04
    )
    plt.close(figure)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    edge: str,
    face: str = "white",
    fontsize: float = 7,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = REFERENCE,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
        )
    )


def transplant_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(
        ax,
        (0.02, 0.67),
        0.20,
        0.15,
        r"$S,\,o_S,\,\ell$",
        edge=REFERENCE,
        face="#F4F4F4",
        fontsize=7.5,
    )
    rounded_box(
        ax,
        (0.37, 0.79),
        0.29,
        0.14,
        "recipient $A$\n" + r"$(\hat F_A,\,F_A,\,a_A,\,e_A)$",
        edge=RECIPIENT,
        face="#EAF2F8",
    )
    rounded_box(
        ax,
        (0.37, 0.55),
        0.29,
        0.14,
        "donor $B$\n" + r"$(\hat F_B,\,F_B,\,a_B,\,e_B)$",
        edge=DONOR,
        face="#FBEFE8",
    )
    arrow(ax, (0.22, 0.75), (0.37, 0.86), color=RECIPIENT)
    arrow(ax, (0.22, 0.73), (0.37, 0.62), color=DONOR)

    rounded_box(
        ax,
        (0.02, 0.22),
        0.30,
        0.18,
        "future target $z^*$\nself $A$ · natural $C$ · $G$ · donor $B$",
        edge=INTERVENTION,
        face="#E9F5F1",
        fontsize=5.8,
    )
    rounded_box(
        ax,
        (0.42, 0.23),
        0.25,
        0.16,
        "recipient run\n" + r"$(S,o_S,\ell,\epsilon_A)$ fixed",
        edge="#444444",
        fontsize=6.3,
    )
    rounded_box(
        ax,
        (0.77, 0.23),
        0.20,
        0.16,
        r"$(a^*,\,e^*)$",
        edge=INTERVENTION,
        face="#E9F5F1",
        fontsize=7.2,
    )
    arrow(ax, (0.32, 0.31), (0.42, 0.31), color=INTERVENTION)
    arrow(ax, (0.67, 0.31), (0.77, 0.31), color=INTERVENTION)
    arrow(
        ax,
        (0.58, 0.55),
        (0.24, 0.40),
        color=DONOR,
        connectionstyle="arc3,rad=0.13",
    )
    ax.text(
        0.50,
        0.08,
        r"$\pi_y(z^*)=\langle y(z^*)-y_A,\,y_B-y_A\rangle/\|y_B-y_A\|_2^2$"
        r"$,\quad y\in\{a,e\}$",
        ha="center",
        va="center",
        fontsize=7.0,
        color=TEXT,
    )
    panel_label(ax, "a", x=-0.03, y=0.98)


def condition_axis(
    ax: plt.Axes,
    rows: Sequence[dict],
    conditions: Sequence[tuple[str, str]],
    *,
    title: str,
    seed: int,
) -> list[tuple[str, tuple[float, float, float]]]:
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-0.055, 0.055, len(rows))
    positions = np.arange(len(conditions), dtype=float)
    for row_index, row in enumerate(rows):
        row_values = [float(row[column]) for _, column in conditions]
        xs = positions + offsets[row_index]
        ax.plot(
            xs,
            row_values,
            color=TASK_COLOR[row["task"]],
            alpha=0.15,
            linewidth=0.55,
            zorder=1,
        )
        ax.scatter(
            xs,
            row_values,
            s=17,
            color=TASK_COLOR[row["task"]],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            zorder=3,
        )
    summaries = []
    for position, (label, column) in zip(positions, conditions):
        stat = add_mean_ci(
            ax,
            position,
            values(rows, column),
            seed=seed + 100 + int(position),
        )
        summaries.append((label, stat))
    ax.axhline(0, color="black", linewidth=0.7, zorder=0)
    ax.axhline(0.10, color="#A0A0A0", linestyle=":", linewidth=0.7, zorder=0)
    ax.axhline(1, color=REFERENCE, linestyle="--", linewidth=0.75, zorder=0)
    ax.set_xticks(positions, [label for label, _ in conditions])
    ax.set_xlim(-0.35, len(conditions) - 0.65)
    ax.set_ylim(-1.10, 1.35)
    ax.set_title(title, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    return summaries


def figure1(population: list[dict], output: Path, stats: list[dict]) -> None:
    figure = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.0, 1.15])
    schematic = figure.add_subplot(grid[0, 0])
    pred = figure.add_subplot(grid[0, 1])
    executed = figure.add_subplot(grid[1, 0])
    endpoint = figure.add_subplot(grid[1, 1])
    transplant_schematic(schematic)
    panels = [
        (
            pred,
            [
                ("Self $A$", "pred_self"),
                ("Natural $C$", "pred_natural"),
                ("Donor $B$", "pred_donor"),
            ],
            "Action: predicted-donor comparison",
            6101,
            "b",
        ),
        (
            executed,
            [
                ("Executed\nself $A$", "exec_self"),
                ("Gaussian\n$G$", "exec_gaussian"),
                ("Natural $C$\n(predicted)", "exec_natural"),
                ("Executed\ndonor $B$", "exec_donor"),
            ],
            "Action: executed-donor comparison",
            6102,
            "c",
        ),
        (
            endpoint,
            [
                ("Executed\nself $A$", "endpoint_self"),
                ("Gaussian\n$G$", "endpoint_gaussian"),
                ("Natural $C$\n(predicted)", "endpoint_natural"),
                ("Executed\ndonor $B$", "endpoint_donor"),
            ],
            "Endpoint: executed-donor comparison",
            6103,
            "d",
        ),
    ]
    for ax, conditions, title, seed, label in panels:
        summaries = condition_axis(
            ax, population, conditions, title=title, seed=seed
        )
        panel_label(ax, label)
        for condition, (mean, lower, upper) in summaries:
            stats.append(
                {
                    "figure": "fig1",
                    "panel": label,
                    "measure": title,
                    "condition": condition.replace("\n", " "),
                    "n": len(population),
                    "mean": mean,
                    "lower": lower,
                    "upper": upper,
                }
            )
    executed.set_ylabel(r"Normalized donor projection, $\pi$")
    pred.set_ylabel(r"Normalized donor projection, $\pi$")
    endpoint.set_yticklabels([])
    figure.legend(
        handles=task_legend(),
        loc="outside lower center",
        ncol=6,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    save_figure(figure, output, "fig1_transplantation_and_steering")


def task_group_spans(rows: Sequence[dict]) -> list[tuple[int, int, str]]:
    spans = []
    start = 0
    while start < len(rows):
        task = rows[start]["task"]
        end = start
        while end + 1 < len(rows) and rows[end + 1]["task"] == task:
            end += 1
        spans.append((start, end, task))
        start = end + 1
    return spans


def kv_dumbbell_axis(
    ax: plt.Axes,
    rows: Sequence[dict],
    *,
    before: str,
    after: str,
    loss: str,
    title: str,
    aggregate_stat: dict,
    show_labels: bool,
) -> None:
    count = len(rows)
    y_positions = np.arange(count - 1, -1, -1)
    for group_index, (start, end, _task) in enumerate(task_group_spans(rows)):
        ys = [count - 1 - start, count - 1 - end]
        if group_index % 2 == 0:
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color="#F7F7F7", zorder=0)
        if end < count - 1:
            separator = count - 1 - end - 0.5
            ax.axhline(separator, color="#D8D8D8", linewidth=0.5, zorder=0)
    for y, row in zip(y_positions, rows):
        color = TASK_COLOR[row["task"]]
        ax.plot(
            [row[after], row[before]],
            [y, y],
            color=color,
            linewidth=1.15,
            alpha=0.72,
            zorder=2,
        )
        ax.scatter(
            row[before],
            y,
            s=23,
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
        ax.scatter(
            row[after],
            y,
            s=24,
            marker="s",
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=4,
        )
    ax.axvline(0, color="black", linewidth=0.65, zorder=1)
    ax.axvline(1, color=REFERENCE, linestyle="--", linewidth=0.75, zorder=1)
    ax.set_xlim(-0.10, 1.38)
    ax.set_ylim(-0.75, count - 0.25)
    ax.set_xlabel(r"Normalized donor projection, $\pi$")
    ax.set_title(title, pad=16)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.text(
        0.5,
        1.015,
        rf"mean $\Delta\pi={aggregate_stat['mean']:.3f}$ "
        rf"$[{aggregate_stat['lower']:.3f},{aggregate_stat['upper']:.3f}]$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#444444",
    )
    if show_labels:
        labels = [f"{TASK_SHORT[row['task']]} · {row['seed']}" for row in rows]
        ax.set_yticks(y_positions, labels)
    else:
        ax.tick_params(axis="y", which="both", labelleft=False)


def figure2(
    population: list[dict], aggregate: dict, output: Path, stats: list[dict]
) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(7.2, 5.25), sharey=True, constrained_layout=True
    )
    panels = [
        (
            "pred_donor",
            "pred_patch",
            "pred_loss",
            "Predicted action",
            "predicted_future_kv_mediation_action",
        ),
        (
            "exec_donor",
            "exec_patch",
            "exec_loss",
            "Executed action",
            "executed_future_kv_mediation_action",
        ),
        (
            "endpoint_donor",
            "endpoint_patch",
            "endpoint_loss",
            "Executed endpoint",
            "executed_future_kv_mediation_physical",
        ),
    ]
    for index, (before, after, loss, title, aggregate_key) in enumerate(panels):
        stat = aggregate["effects"][aggregate_key]
        kv_dumbbell_axis(
            axes[index],
            population,
            before=before,
            after=after,
            loss=loss,
            title=title,
            aggregate_stat=stat,
            show_labels=index == 0,
        )
        panel_label(axes[index], chr(ord("a") + index), x=-0.22 if index == 0 else -0.12)
        stats.append(
            {
                "figure": "fig2",
                "panel": chr(ord("a") + index),
                "measure": title,
                "condition": "donor minus self-future K/V patch",
                "n": len(population),
                "mean": float(stat["mean"]),
                "lower": float(stat["lower"]),
                "upper": float(stat["upper"]),
            }
        )
    y_positions = np.arange(len(population) - 1, -1, -1)
    axes[0].set_yticks(
        y_positions,
        [f"{TASK_SHORT[row['task']]} · {row['seed']}" for row in population],
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#555555",
                markeredgecolor="white",
                markersize=5,
                label="Donor future",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=5,
                label="After self-future K/V replacement",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        columnspacing=1.5,
    )
    save_figure(figure, output, "fig2_kv_pathway")


def factor_cell_axis(
    ax: plt.Axes,
    rows: Sequence[dict],
    *,
    prefix: str,
    title: str,
    seed: int,
) -> list[tuple[str, tuple[float, float, float]]]:
    conditions = [
        ("$O_0R_0$", f"{prefix}_o0r0"),
        ("$O_0R_1$", f"{prefix}_o0r1"),
        ("$O_1R_0$", f"{prefix}_o1r0"),
        ("$O_1R_1$", f"{prefix}_o1r1"),
        ("Coherent\ndonor", f"full_{prefix}"),
    ]
    positions = np.arange(len(conditions), dtype=float)
    offsets = np.random.default_rng(seed).uniform(-0.055, 0.055, len(rows))
    ax.axhspan(-0.10, 0.10, color=EQUIVALENCE, zorder=0)
    for row_index, row in enumerate(rows):
        row_values = [float(row[column]) for _, column in conditions]
        xs = positions + offsets[row_index]
        ax.plot(
            xs,
            row_values,
            color=TASK_COLOR[row["task"]],
            linewidth=0.6,
            alpha=0.16,
            zorder=1,
        )
        ax.scatter(
            xs,
            row_values,
            s=18,
            color=TASK_COLOR[row["task"]],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.85,
            zorder=3,
        )
    summaries = []
    for position, (label, column) in zip(positions, conditions):
        stat = add_mean_ci(
            ax,
            position,
            values(rows, column),
            seed=seed + 100 + int(position),
        )
        summaries.append((label, stat))
    ax.axhline(0, color="black", linewidth=0.65, zorder=1)
    ax.axhline(-0.10, color=REFERENCE, linestyle=":", linewidth=0.6, zorder=1)
    ax.axhline(0.10, color=REFERENCE, linestyle=":", linewidth=0.6, zorder=1)
    ax.axhline(1, color=REFERENCE, linestyle="--", linewidth=0.75, zorder=1)
    ax.set_xticks(positions, [label for label, _ in conditions])
    ax.set_ylim(-0.25, 1.35)
    ax.set_title(title, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    return summaries


def factor_effect_axis(
    ax: plt.Axes,
    rows: Sequence[dict],
    *,
    prefix: str,
    title: str,
    seed: int,
    registered: dict[str, dict],
) -> list[tuple[str, tuple[float, float, float]]]:
    effects = [
        ("Robot", f"{prefix}_robot"),
        ("Object", f"{prefix}_object"),
        ("Interaction", f"{prefix}_interaction"),
    ]
    positions = np.arange(len(effects), dtype=float)
    offsets = np.random.default_rng(seed).uniform(-0.055, 0.055, len(rows))
    ax.axhspan(-0.10, 0.10, color=EQUIVALENCE, zorder=0)
    for row_index, row in enumerate(rows):
        xs = positions + offsets[row_index]
        row_values = [float(row[column]) for _, column in effects]
        ax.scatter(
            xs,
            row_values,
            s=18,
            color=TASK_COLOR[row["task"]],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.85,
            zorder=3,
        )
    summaries = []
    for position, (label, column) in zip(positions, effects):
        registered_stat = registered[column]
        stat = add_mean_ci(
            ax,
            position,
            values(rows, column),
            seed=seed + 100 + int(position),
            statistic=(
                float(registered_stat["mean"]),
                float(registered_stat["lower"]),
                float(registered_stat["upper"]),
            ),
        )
        summaries.append((label, stat))
    ax.axhline(0, color="black", linewidth=0.65, zorder=1)
    ax.axhline(-0.10, color=REFERENCE, linestyle="--", linewidth=0.65, zorder=1)
    ax.axhline(0.10, color=REFERENCE, linestyle="--", linewidth=0.65, zorder=1)
    ax.set_xticks(positions, [label for label, _ in effects])
    ax.set_ylim(-0.12, 0.12)
    ax.set_title(title, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    return summaries


def figure3(
    factorization: list[dict], aggregate: dict, output: Path, stats: list[dict]
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.15), constrained_layout=True)
    panel_specs = [
        (axes[0, 0], "action", "Action projection", factor_cell_axis, 6301, "a"),
        (
            axes[0, 1],
            "endpoint",
            "Executed endpoint projection",
            factor_cell_axis,
            6302,
            "b",
        ),
        (axes[1, 0], "action", "Action factorial effects", factor_effect_axis, 6303, "c"),
        (
            axes[1, 1],
            "endpoint",
            "Executed endpoint factorial effects",
            factor_effect_axis,
            6304,
            "d",
        ),
    ]
    registered_factor = aggregate["factorization"]["effects"]
    registered_by_prefix = {
        "action": {
            "action_robot": registered_factor["action_robot_main_effect"],
            "action_object": registered_factor["action_object_main_effect"],
            "action_interaction": registered_factor["action_interaction"],
        },
        "endpoint": {
            "endpoint_robot": registered_factor["physical_all_robot_main_effect"],
            "endpoint_object": registered_factor["physical_all_object_main_effect"],
            "endpoint_interaction": registered_factor["physical_all_interaction"],
        },
    }
    for ax, prefix, title, function, seed, label in panel_specs:
        arguments = {"prefix": prefix, "title": title, "seed": seed}
        if function is factor_effect_axis:
            arguments["registered"] = registered_by_prefix[prefix]
        summaries = function(ax, factorization, **arguments)
        panel_label(ax, label)
        for condition, (mean, lower, upper) in summaries:
            stats.append(
                {
                    "figure": "fig3",
                    "panel": label,
                    "measure": title,
                    "condition": condition.replace("\n", " ").replace("$", ""),
                    "n": len(factorization),
                    "mean": mean,
                    "lower": lower,
                    "upper": upper,
                }
            )
    axes[0, 0].set_ylabel(r"Normalized donor projection, $\pi$")
    axes[0, 1].set_yticklabels([])
    axes[1, 0].set_ylabel(r"Factorial effect, $\Delta\pi$")
    axes[1, 1].set_yticklabels([])
    axes[0, 0].set_xlabel("Pixel condition")
    axes[0, 1].set_xlabel("Pixel condition")
    figure.legend(
        handles=task_legend(),
        loc="outside lower center",
        ncol=6,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    save_figure(figure, output, "fig3_pixel_factorization")


def select_policy(
    rows: Sequence[dict], *, contrast: str, prefix: int | None = None
) -> list[dict]:
    selected = [row for row in rows if row["contrast"] == contrast]
    if prefix is not None:
        selected = [row for row in selected if row["prefix"] == prefix]
    return selected


def policy_timing_axis(
    ax: plt.Axes,
    rows: Sequence[dict],
    *,
    metric: str,
    color: str,
    title: str,
    seed: int,
    registered_summary: dict,
) -> list[tuple[str, tuple[float, float, float]]]:
    contrast = "semantic_all_donor_minus_recipient"
    selected = select_policy(rows, contrast=contrast)
    prefixes = [0, 3, 6]
    by_task = defaultdict(dict)
    for row in selected:
        by_task[row["task"]][row["prefix"]] = row[metric]
    for task in sorted(by_task):
        task_values = [by_task[task][prefix] for prefix in prefixes]
        ax.plot(prefixes, task_values, color=color, alpha=0.17, linewidth=0.65, zorder=1)
        ax.scatter(
            prefixes,
            task_values,
            s=17,
            color=color,
            alpha=0.42,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
    summaries = []
    for index, prefix in enumerate(prefixes):
        sample = [
            row[metric]
            for row in selected
            if row["prefix"] == prefix
        ]
        metric_key = (
            "action_donor_steering"
            if metric == "action"
            else "physical_endpoint_donor_steering"
        )
        registered = registered_summary["semantic_by_prefix_chunks"][str(prefix)][
            contrast
        ][metric_key]
        stat = add_mean_ci(
            ax,
            prefix,
            sample,
            seed=seed + index,
            color=color,
            marker="D",
            size=4.2,
            statistic=(
                float(registered["mean"]),
                float(registered["lower"]),
                float(registered["upper"]),
            ),
        )
        summaries.append((str(prefix), stat))
    ax.axhline(0, color="black", linewidth=0.65)
    ax.axhline(0.10, color=REFERENCE, linestyle="--", linewidth=0.7)
    ax.set_xticks(prefixes)
    ax.set_xlim(-0.55, 6.55)
    ax.set_ylim(-0.20, 1.15)
    ax.set_xlabel("Open-loop chunks before query")
    ax.set_title(title, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    return summaries


def policy_modality_axis(
    ax: plt.Axes,
    rows: Sequence[dict],
    *,
    seed: int,
    registered_summary: dict,
) -> list[tuple[str, str, tuple[float, float, float]]]:
    categories = [
        ("All", "semantic_all_donor_minus_recipient"),
        ("Wrist", "semantic_wrist_donor_minus_recipient"),
        ("Primary", "semantic_primary_donor_minus_recipient"),
        ("Proprio.", "semantic_proprio_donor_minus_recipient"),
    ]
    outcomes = [
        ("Action", "action", ACTION, "o", -0.11, ACTION),
        ("Endpoint", "endpoint", ENDPOINT, "s", 0.11, "white"),
    ]
    summaries = []
    for outcome_index, (outcome, metric, color, marker, offset, face) in enumerate(outcomes):
        by_task = defaultdict(dict)
        for position, (_label, contrast) in enumerate(categories):
            selected = select_policy(rows, contrast=contrast, prefix=0)
            for row in selected:
                by_task[row["task"]][position] = row[metric]
        for task in sorted(by_task):
            task_values = [by_task[task][position] for position in range(len(categories))]
            xs = np.arange(len(categories), dtype=float) + offset
            ax.plot(xs, task_values, color=color, alpha=0.10, linewidth=0.55, zorder=1)
        for position, (label, contrast) in enumerate(categories):
            selected = select_policy(rows, contrast=contrast, prefix=0)
            sample = [row[metric] for row in selected]
            metric_key = (
                "action_donor_steering"
                if metric == "action"
                else "physical_endpoint_donor_steering"
            )
            registered = registered_summary["semantic_by_prefix_chunks"]["0"][
                contrast
            ][metric_key]
            jitter = np.random.default_rng(seed + outcome_index * 20 + position).uniform(
                -0.035, 0.035, len(sample)
            )
            ax.scatter(
                position + offset + jitter,
                sample,
                s=16,
                marker=marker,
                facecolor=face,
                edgecolor=color,
                linewidth=0.7 if face == "white" else 0.3,
                alpha=0.58,
                zorder=3,
            )
            stat = add_mean_ci(
                ax,
                position + offset,
                sample,
                seed=seed + 100 + outcome_index * 20 + position,
                color=color,
                marker=marker,
                facecolor=face,
                size=4.0,
                statistic=(
                    float(registered["mean"]),
                    float(registered["lower"]),
                    float(registered["upper"]),
                ),
            )
            summaries.append((outcome, label, stat))
    ax.axhline(0, color="black", linewidth=0.65)
    ax.axhline(0.10, color=REFERENCE, linestyle="--", linewidth=0.7)
    ax.set_xticks(range(len(categories)), [label for label, _ in categories])
    ax.set_xlim(-0.45, len(categories) - 0.55)
    ax.set_ylim(-0.20, 1.15)
    ax.set_ylabel(r"Donor $-$ recipient projection, $\Delta\pi$")
    ax.set_xlabel("Future component transplanted")
    ax.set_title("First-query modality effects", pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", color=ACTION, label="Action"
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=ENDPOINT,
                color=ENDPOINT,
                label="Executed endpoint",
            ),
        ],
        loc="upper right",
        frameon=False,
    )
    return summaries


def figure4(
    policy: list[dict], policy_summary: dict, output: Path, stats: list[dict]
) -> None:
    figure = plt.figure(figsize=(7.2, 5.05), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.04])
    action_ax = figure.add_subplot(grid[0, 0])
    endpoint_ax = figure.add_subplot(grid[0, 1], sharey=action_ax)
    modality_ax = figure.add_subplot(grid[1, :])
    action_stats = policy_timing_axis(
        action_ax,
        policy,
        metric="action",
        color=ACTION,
        title="Action across registered states",
        seed=6401,
        registered_summary=policy_summary,
    )
    endpoint_stats = policy_timing_axis(
        endpoint_ax,
        policy,
        metric="endpoint",
        color=ENDPOINT,
        title="Executed endpoint across registered states",
        seed=6402,
        registered_summary=policy_summary,
    )
    action_ax.set_ylabel(r"Donor $-$ recipient projection, $\Delta\pi$")
    endpoint_ax.tick_params(axis="y", which="both", labelleft=False)
    modality_stats = policy_modality_axis(
        modality_ax, policy, seed=6403, registered_summary=policy_summary
    )
    for label, ax in zip("abc", [action_ax, endpoint_ax, modality_ax]):
        panel_label(ax, label)
    for outcome, summaries, label in [
        ("Action timing", action_stats, "a"),
        ("Endpoint timing", endpoint_stats, "b"),
    ]:
        for condition, (mean, lower, upper) in summaries:
            stats.append(
                {
                    "figure": "fig4",
                    "panel": label,
                    "measure": outcome,
                    "condition": f"prefix {condition}",
                    "n": 10,
                    "mean": mean,
                    "lower": lower,
                    "upper": upper,
                }
            )
    for outcome, condition, (mean, lower, upper) in modality_stats:
        stats.append(
            {
                "figure": "fig4",
                "panel": "c",
                "measure": f"{outcome} modality",
                "condition": condition,
                "n": 10,
                "mean": mean,
                "lower": lower,
                "upper": upper,
            }
        )
    save_figure(figure, output, "fig4_cosmos_policy_scope")


def identity_axis(
    ax: plt.Axes,
    rows: Sequence[dict],
    *,
    x_column: str,
    y_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    x = values(rows, x_column)
    y = values(rows, y_column)
    lower = 0.0
    upper = float(max(x.max(), y.max()) * 1.06)
    for row in rows:
        ax.scatter(
            row[x_column],
            row[y_column],
            s=21,
            color=TASK_COLOR[row["task"]],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.85,
            zorder=3,
        )
    ax.plot([lower, upper], [lower, upper], color=REFERENCE, linestyle="--", linewidth=0.8)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n$r={np.corrcoef(x, y)[0, 1]:.3f}$", pad=4)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)


def supplement_metric_sanity(population: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    specifications = [
        (
            "native_action_l2",
            "pred_l2",
            "Predicted action",
            r"Native separation, $\|a_B-a_A\|_2$",
            r"Transplanted displacement, $\|a^*-a_A\|_2$",
        ),
        (
            "native_action_l2",
            "exec_l2",
            "Executed-future action",
            r"Native separation, $\|a_B-a_A\|_2$",
            r"Transplanted displacement, $\|a^*-a_A\|_2$",
        ),
        (
            "native_endpoint_l2",
            "endpoint_l2",
            "Physical endpoint",
            r"Native separation, $\|e_B-e_A\|_2$",
            r"Transplanted displacement, $\|e^*-e_A\|_2$",
        ),
    ]
    for label, ax, spec in zip("abc", axes, specifications):
        identity_axis(
            ax,
            population,
            x_column=spec[0],
            y_column=spec[1],
            title=spec[2],
            xlabel=spec[3],
            ylabel=spec[4],
        )
        panel_label(ax, label, x=-0.07, y=1.03)
    figure.legend(
        handles=task_legend(),
        loc="outside lower center",
        ncol=6,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    save_figure(figure, output, "supp1_metric_sanity")


def policy_control_axis(
    ax: plt.Axes,
    policy: Sequence[dict],
    *,
    metric: str,
    color: str,
    title: str,
    seed: int,
    registered_summary: dict,
) -> None:
    conditions = [
        ("Self", "semantic_all_donor_minus_recipient"),
        ("Gaussian", "semantic_all_donor_minus_gaussian"),
        ("Natural", "semantic_all_donor_minus_natural_control"),
        ("Shuffled", "semantic_all_donor_minus_shuffled"),
    ]
    for position, (_label, contrast) in enumerate(conditions):
        sample_rows = select_policy(policy, contrast=contrast, prefix=0)
        sample = [row[metric] for row in sample_rows]
        metric_key = (
            "action_donor_steering"
            if metric == "action"
            else "physical_endpoint_donor_steering"
        )
        registered = registered_summary["semantic_by_prefix_chunks"]["0"][
            contrast
        ][metric_key]
        jitter = np.random.default_rng(seed + position).uniform(-0.06, 0.06, len(sample))
        ax.scatter(
            position + jitter,
            sample,
            s=18,
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
        add_mean_ci(
            ax,
            position,
            sample,
            seed=seed + 100 + position,
            color=color,
            marker="D",
            statistic=(
                float(registered["mean"]),
                float(registered["lower"]),
                float(registered["upper"]),
            ),
        )
    ax.axhline(0, color="black", linewidth=0.65)
    ax.axhline(0.10, color=REFERENCE, linestyle="--", linewidth=0.7)
    ax.set_xticks(range(len(conditions)), [label for label, _ in conditions])
    ax.set_ylim(-0.20, 1.15)
    ax.set_title(title, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)


def supplement_policy_controls(
    policy: list[dict], policy_summary: dict, output: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    policy_control_axis(
        axes[0],
        policy,
        metric="action",
        color=ACTION,
        title="Action",
        seed=6601,
        registered_summary=policy_summary,
    )
    policy_control_axis(
        axes[1],
        policy,
        metric="endpoint",
        color=ENDPOINT,
        title="Executed endpoint",
        seed=6602,
        registered_summary=policy_summary,
    )
    axes[0].set_ylabel(r"Donor $-$ control projection, $\Delta\pi$")
    axes[1].tick_params(axis="y", which="both", labelleft=False)
    first = select_policy(
        policy, contrast="semantic_all_donor_minus_recipient", prefix=0
    )
    action_values = values(first, "action")
    endpoint_values = values(first, "endpoint")
    limits = (-0.05, 1.15)
    axes[2].scatter(
        action_values,
        endpoint_values,
        s=23,
        color=ACTION,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.82,
        zorder=3,
    )
    axes[2].plot(limits, limits, color=REFERENCE, linestyle="--", linewidth=0.8)
    axes[2].axhline(0, color="#AAAAAA", linewidth=0.55)
    axes[2].axvline(0, color="#AAAAAA", linewidth=0.55)
    axes[2].set_xlim(*limits)
    axes[2].set_ylim(*limits)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel(r"Action steering, $\Delta\pi_a$")
    axes[2].set_ylabel(r"Endpoint steering, $\Delta\pi_e$")
    axes[2].set_title(
        "First-query relation\n"
        + rf"$r={np.corrcoef(action_values, endpoint_values)[0, 1]:.3f}$",
        pad=4,
    )
    axes[2].grid(color=GRID, linewidth=0.5, zorder=0)
    for label, ax in zip("abc", axes):
        panel_label(ax, label)
    save_figure(figure, output, "supp2_policy_controls")


def write_statistics(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["figure", "panel", "measure", "condition", "n", "mean", "lower", "upper"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_condition_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("self", "recipient/native", "Recipient future or native recipient run"),
        (
            "natural C",
            "predicted",
            "Third model-generated future from the identical saved state",
        ),
        (
            "Gaussian G",
            "synthetic",
            "Latent matched to executed-donor norm and recipient distance",
        ),
        ("predicted donor B", "predicted", "Model-predicted future from donor branch B"),
        ("executed donor B", "executed", "Observed video produced by physically executing action B"),
        (
            "self-future K/V patch",
            "activation patch",
            "Donor-run future-video K/V replaced with recipient self-future K/V",
        ),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "source_type", "definition"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = parser.parse_args()

    configure_style()
    population = load_population(args.root)
    factorization = load_factorization(args.root)
    policy = load_policy_state_means(args.root)
    aggregate = json.loads(
        (
            args.root
            / "results"
            / "cosmos3_population_confirmatory_v1"
            / "aggregate_summary.json"
        ).read_text()
    )
    policy_summary = json.loads(
        (args.root / "results" / "confirmatory_v1" / "summary.json").read_text()
    )

    statistics_rows: list[dict] = []
    figure1(population, args.output_dir, statistics_rows)
    figure2(population, aggregate, args.output_dir, statistics_rows)
    figure3(factorization, aggregate, args.output_dir, statistics_rows)
    figure4(policy, policy_summary, args.output_dir, statistics_rows)
    supplement_metric_sanity(population, args.output_dir)
    supplement_policy_controls(policy, policy_summary, args.output_dir)
    write_statistics(args.tables_dir / "main_figure_statistics.csv", statistics_rows)
    write_condition_table(args.tables_dir / "condition_definitions.csv")
    print(
        f"Rendered 4 main and 2 supplementary figures to {args.output_dir}; "
        f"tables to {args.tables_dir}"
    )


if __name__ == "__main__":
    main()
