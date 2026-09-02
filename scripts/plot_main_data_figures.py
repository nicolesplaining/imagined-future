"""Render the four main empirical figures for the reorganized manuscript.

The figures intentionally use ordinary scientific plots: independent state
means, percentile bootstrap intervals, categorical experimental conditions,
and literal x-y relationships.  They avoid diagrammatic annotations and do not
treat process repeats, directions, or noise draws as independent observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output" / "main_data_figures"

TASK_COLORS = {
    "BananaInBowlTask": "#0072B2",
    "MarkerInMugTask": "#D55E00",
    "MustardInLeftBinTask": "#009E73",
    "RubiksCubeTask": "#CC79A7",
    "SmartphoneInBinTask": "#E69F00",
    "SpoonInMugTask": "#56B4E9",
}
TASK_LABELS = {
    "BananaInBowlTask": "banana → bowl",
    "MarkerInMugTask": "marker → mug",
    "MustardInLeftBinTask": "mustard → bin",
    "RubiksCubeTask": "Rubik's cube",
    "SmartphoneInBinTask": "phone → bin",
    "SpoonInMugTask": "spoon → mug",
}
ACTION_COLOR = "#245B8A"
ENDPOINT_COLOR = "#B45F3C"
POINT_GREY = "#707070"
GRID_GREY = "#E7E7E7"
REFERENCE_GREY = "#777777"


def bootstrap_mean(
    values: np.ndarray, *, seed: int, resamples: int = 10_000
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def nested(data: dict, path: str):
    value = data
    for key in path.split("."):
        value = value[key]
    return value


def load_cosmos3_population(root: Path) -> pd.DataFrame:
    records = []
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
    for path in sorted(folder.glob("*/summary.json")):
        data = json.loads(path.read_text())
        interventions = data["interventions"]
        record = {
            "unit": f"{data['task']}-seed-{data['environment_seed']}",
            "task": data["task"],
            "native_action_l2": data["native_action_l2"],
            "native_endpoint_l2": data["native_endpoint_l2"]["all"],
        }
        record.update({key: nested(interventions, field) for key, field in fields.items()})
        record["pred_steering"] = record["pred_donor"] - record["pred_self"]
        record["exec_steering"] = record["exec_donor"] - record["exec_self"]
        record["endpoint_steering"] = (
            record["endpoint_donor"] - record["endpoint_self"]
        )
        record["pred_patch_steering"] = record["pred_patch"] - record["pred_self"]
        record["exec_patch_steering"] = record["exec_patch"] - record["exec_self"]
        record["endpoint_patch_steering"] = (
            record["endpoint_patch"] - record["endpoint_self"]
        )
        record["pred_loss"] = record["pred_donor"] - record["pred_patch"]
        record["exec_loss"] = record["exec_donor"] - record["exec_patch"]
        record["endpoint_loss"] = record["endpoint_donor"] - record["endpoint_patch"]
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values(["task", "unit"])


def load_cosmos3_factorization(root: Path) -> pd.DataFrame:
    records = []
    folder = root / "results" / "cosmos3_population_confirmatory_v1"
    for path in sorted(folder.glob("*/summary.json")):
        data = json.loads(path.read_text())
        effects = (data.get("factorial_effects") or {}).get("composite")
        if not effects:
            continue
        action_effects = effects["action_donor_projection_effects"]
        endpoint_effects = effects["endpoint_donor_projection_effects"]["all"]
        record = {
            "unit": f"{data['task']}-seed-{data['environment_seed']}",
            "task": data["task"],
            "full_action": data["interventions"]["executed_donor"][
                "action_donor_projection"
            ],
            "full_endpoint": data["interventions"]["executed_donor"][
                "endpoint_donor_projection"
            ]["all"],
            "full_action_steering": data["interventions"]["executed_donor"][
                "action_donor_projection"
            ]
            - data["interventions"]["executed_self"]["action_donor_projection"],
            "full_endpoint_steering": data["interventions"]["executed_donor"][
                "endpoint_donor_projection"
            ]["all"]
            - data["interventions"]["executed_self"]["endpoint_donor_projection"][
                "all"
            ],
            "action_robot": action_effects["robot_main_effect"],
            "action_object": action_effects["object_main_effect"],
            "action_interaction": action_effects["interaction"],
            "endpoint_robot": endpoint_effects["robot_main_effect"],
            "endpoint_object": endpoint_effects["object_main_effect"],
            "endpoint_interaction": endpoint_effects["interaction"],
        }
        for cell, value in effects["action_donor_projection_cells"].items():
            record[f"action_{cell}"] = value
        for cell, value in effects["endpoint_donor_projection_cells"]["all"].items():
            record[f"endpoint_{cell}"] = value
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values(["task", "unit"])


def load_policy_state_means(root: Path) -> pd.DataFrame:
    path = root / "results" / "confirmatory_v1" / "semantic_state_repetitions.csv"
    repetitions = pd.read_csv(path)
    metrics = ["action_donor_steering", "physical_endpoint_donor_steering"]
    return (
        repetitions.groupby(
            ["unit_id", "task_id", "prefix_chunks", "contrast"], as_index=False
        )[metrics]
        .mean()
        .sort_values(["task_id", "prefix_chunks", "contrast"])
    )


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def horizontal_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.55, zorder=0)


def task_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=TASK_COLORS[task],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.8,
            label=TASK_LABELS[task],
        )
        for task in TASK_COLORS
    ]


def jitter(count: int, seed: int, width: float = 0.10) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, count)


def add_summary(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    *,
    seed: int,
    color: str = "black",
    marker: str = "D",
    facecolor: str | None = None,
    zorder: int = 5,
) -> None:
    mean, lower, upper = bootstrap_mean(values, seed=seed)
    ax.errorbar(
        x,
        mean,
        yerr=[[mean - lower], [upper - mean]],
        fmt=marker,
        color=color,
        markerfacecolor=facecolor if facecolor is not None else color,
        markeredgecolor=color,
        markeredgewidth=0.8,
        markersize=3.5,
        capsize=2,
        linewidth=1.0,
        zorder=zorder,
    )


def cosmos_condition_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    conditions: list[tuple[str, str]],
    *,
    title: str,
    seed: int,
) -> None:
    for position, (label, column) in enumerate(conditions):
        values = frame[column].to_numpy(dtype=float)
        xs = position + jitter(len(frame), seed + position)
        for x, value, task in zip(xs, values, frame["task"]):
            ax.scatter(
                x,
                value,
                s=17,
                color=TASK_COLORS[task],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
                zorder=3,
            )
        add_summary(ax, position, values, seed=seed + 100 + position)
    ax.axhline(0, color="black", linewidth=0.65, zorder=1)
    ax.axhline(1, color=REFERENCE_GREY, linestyle="--", linewidth=0.7, zorder=1)
    ax.set_xticks(range(len(conditions)), [label for label, _ in conditions])
    ax.set_ylim(-1.10, 1.35)
    ax.set_title(title, pad=4)
    horizontal_grid(ax)


def identity_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    limit: tuple[float, float],
    xlabel: str,
    ylabel: str,
) -> None:
    for task, selected in frame.groupby("task"):
        ax.scatter(
            selected[x_column],
            selected[y_column],
            s=20,
            color=TASK_COLORS[task],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )
    ax.plot(limit, limit, color=REFERENCE_GREY, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xlim(*limit)
    ax.set_ylim(*limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID_GREY, linewidth=0.55, zorder=0)


def effect_rows_axis(
    ax: plt.Axes,
    rows: list[tuple[str, np.ndarray, str]],
    *,
    title: str,
    xlabel: str,
    seed: int,
    criterion: float | None = None,
    donor_reference: bool = False,
) -> None:
    outcome_styles = {
        "action": (ACTION_COLOR, "o", ACTION_COLOR),
        "endpoint": (ENDPOINT_COLOR, "s", "white"),
    }
    positions = np.arange(len(rows))[::-1]
    for row_index, ((label, values, outcome), position) in enumerate(
        zip(rows, positions)
    ):
        values = np.asarray(values, dtype=float)
        color, marker, face = outcome_styles[outcome]
        ys = position + jitter(len(values), seed + row_index, width=0.11)
        ax.scatter(
            values,
            ys,
            s=13,
            marker=marker,
            facecolor=color if face != "white" else "white",
            edgecolor=color,
            linewidth=0.45,
            alpha=0.35,
            zorder=2,
        )
        mean, lower, upper = bootstrap_mean(values, seed=seed + 100 + row_index)
        ax.errorbar(
            mean,
            position,
            xerr=[[mean - lower], [upper - mean]],
            fmt=marker,
            color=color,
            markerfacecolor=face,
            markeredgecolor=color,
            markeredgewidth=0.9,
            markersize=5.0,
            capsize=2,
            linewidth=1.2,
            zorder=4,
        )
    ax.axvline(0, color="black", linewidth=0.7, zorder=1)
    if criterion is not None:
        ax.axvline(
            criterion, color=REFERENCE_GREY, linestyle="--", linewidth=0.75, zorder=1
        )
    if donor_reference:
        ax.axvline(1, color=REFERENCE_GREY, linestyle=":", linewidth=0.8, zorder=1)
    ax.set_yticks(positions, [label for label, _values, _outcome in rows])
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=5)
    ax.grid(axis="x", color=GRID_GREY, linewidth=0.55, zorder=0)


def effect_groups_axis(
    ax: plt.Axes,
    groups: list[tuple[str, list[tuple[np.ndarray, str]]]],
    *,
    title: str,
    xlabel: str,
    seed: int,
    criterion: float | None = None,
    donor_reference: bool = False,
) -> None:
    outcome_styles = {
        "action": (ACTION_COLOR, "o", ACTION_COLOR),
        "endpoint": (ENDPOINT_COLOR, "s", "white"),
    }
    positions = np.arange(len(groups))[::-1]
    series_index = 0
    for (label, series), position in zip(groups, positions):
        offsets = [0.0] if len(series) == 1 else [0.11, -0.11]
        for (values, outcome), offset in zip(series, offsets):
            values = np.asarray(values, dtype=float)
            color, marker, face = outcome_styles[outcome]
            ys = position + offset + jitter(
                len(values), seed + series_index, width=0.055
            )
            ax.scatter(
                values,
                ys,
                s=13,
                marker=marker,
                facecolor=color if face != "white" else "white",
                edgecolor=color,
                linewidth=0.45,
                alpha=0.48,
                zorder=2,
            )
            mean, lower, upper = bootstrap_mean(
                values, seed=seed + 100 + series_index
            )
            ax.errorbar(
                mean,
                position + offset,
                xerr=[[mean - lower], [upper - mean]],
                fmt=marker,
                color=color,
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=5.0,
                capsize=2,
                linewidth=1.2,
                zorder=4,
            )
            series_index += 1
    ax.axvline(0, color="black", linewidth=0.7, zorder=1)
    if criterion is not None:
        ax.axvline(
            criterion, color=REFERENCE_GREY, linestyle="--", linewidth=0.75, zorder=1
        )
    if donor_reference:
        ax.axvline(1, color=REFERENCE_GREY, linestyle=":", linewidth=0.8, zorder=1)
    ax.set_yticks(positions, [label for label, _series in groups])
    ax.set_ylim(-0.55, len(groups) - 0.45)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=5)
    ax.grid(axis="x", color=GRID_GREY, linewidth=0.55, zorder=0)


def figure2_steering(
    population: pd.DataFrame, policy: pd.DataFrame, output: Path
) -> None:
    first_query = policy.loc[policy["prefix_chunks"] == 0]

    def policy_values(contrast: str, metric: str) -> np.ndarray:
        return first_query.loc[first_query["contrast"] == contrast, metric].to_numpy(
            dtype=float
        )

    primary_rows = [
        (
            "Cosmos 3, predicted future\nAction",
            population["pred_steering"].to_numpy(dtype=float),
            "action",
        ),
        (
            "Cosmos 3, executed future\nAction",
            population["exec_steering"].to_numpy(dtype=float),
            "action",
        ),
        (
            "Cosmos 3, executed future\nEndpoint",
            population["endpoint_steering"].to_numpy(dtype=float),
            "endpoint",
        ),
        (
            "Cosmos Policy, first query\nAction",
            policy_values(
                "semantic_all_donor_minus_recipient", "action_donor_steering"
            ),
            "action",
        ),
        (
            "Cosmos Policy, first query\nEndpoint",
            policy_values(
                "semantic_all_donor_minus_recipient",
                "physical_endpoint_donor_steering",
            ),
            "endpoint",
        ),
    ]
    control_rows = [
        (
            "C3 predicted vs natural\nAction",
            (population["pred_donor"] - population["pred_natural"]).to_numpy(),
            "action",
        ),
        (
            "C3 executed vs Gaussian\nAction",
            (population["exec_donor"] - population["exec_gaussian"]).to_numpy(),
            "action",
        ),
        (
            "C3 executed vs natural\nAction",
            (population["exec_donor"] - population["exec_natural"]).to_numpy(),
            "action",
        ),
    ]
    for label, contrast in (
        ("Policy vs Gaussian", "semantic_all_donor_minus_gaussian"),
        ("Policy vs natural", "semantic_all_donor_minus_natural_control"),
        ("Policy vs shuffled", "semantic_all_donor_minus_shuffled"),
    ):
        control_rows.extend(
            [
                (
                    f"{label}\nAction",
                    policy_values(contrast, "action_donor_steering"),
                    "action",
                ),
                (
                    f"{label}\nEndpoint",
                    policy_values(contrast, "physical_endpoint_donor_steering"),
                    "endpoint",
                ),
            ]
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.725, 4.35),
        gridspec_kw={"width_ratios": [0.92, 1.25]},
        constrained_layout=True,
    )
    effect_rows_axis(
        axes[0],
        primary_rows,
        title="Primary steering contrasts",
        xlabel=r"Donor $-$ self steering, $\Delta_x$",
        seed=202620,
        criterion=0.10,
        donor_reference=True,
    )
    axes[0].set_xlim(-0.12, 1.18)
    effect_rows_axis(
        axes[1],
        control_rows,
        title="Specificity-control contrasts",
        xlabel=r"Donor advantage, $\pi_x(\mathrm{donor})-\pi_x(\mathrm{control})$",
        seed=202640,
    )
    axes[1].set_xlim(-0.08, 1.12)
    for label, panel in zip("ab", axes):
        panel.text(
            -0.04,
            1.02,
            label,
            transform=panel.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    figure.legend(
        handles=policy_legend_handles(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    save_figure(figure, output / "figure2_directional_tests")


def appendix_cosmos3_conditions(population: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.85), constrained_layout=True)
    cosmos_condition_axis(
        axes[0],
        population,
        [("Self", "pred_self"), ("Natural", "pred_natural"), ("Donor", "pred_donor")],
        title="Predicted future: action",
        seed=202620,
    )
    cosmos_condition_axis(
        axes[1],
        population,
        [
            ("Self", "exec_self"),
            ("Gaussian", "exec_gaussian"),
            ("Natural", "exec_natural"),
            ("Donor", "exec_donor"),
        ],
        title="Executed future: action",
        seed=202630,
    )
    cosmos_condition_axis(
        axes[2],
        population,
        [
            ("Self", "endpoint_self"),
            ("Gaussian", "endpoint_gaussian"),
            ("Natural", "endpoint_natural"),
            ("Donor", "endpoint_donor"),
        ],
        title="Executed future: endpoint",
        seed=202640,
    )
    axes[0].set_ylabel(r"Normalized donor projection, $\pi_x$")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    for label, ax in zip("abc", axes):
        ax.text(
            -0.02,
            1.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    figure.legend(
        handles=task_legend_handles(),
        loc="outside lower center",
        ncol=6,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    save_figure(figure, output / "appendix_cosmos3_condition_projections")


def appendix_magnitude_checks(population: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.85), constrained_layout=True)
    identity_axis(
        axes[0],
        population,
        "native_action_l2",
        "pred_l2",
        limit=(0, 9),
        xlabel=r"Native action separation, $\|a_B-a_A\|_2$",
        ylabel=r"Transplanted displacement, $\|\hat a-a_A\|_2$",
    )
    identity_axis(
        axes[1],
        population,
        "native_action_l2",
        "exec_l2",
        limit=(0, 9),
        xlabel=r"Native action separation, $\|a_B-a_A\|_2$",
        ylabel=r"Transplanted displacement, $\|\hat a-a_A\|_2$",
    )
    identity_axis(
        axes[2],
        population,
        "native_endpoint_l2",
        "endpoint_l2",
        limit=(0, 2.6),
        xlabel=r"Native endpoint separation, $\|e_B-e_A\|_2$",
        ylabel=r"Transplanted displacement, $\|\hat e-e_A\|_2$",
    )
    axes[0].set_title("Predicted future: action", pad=4)
    axes[1].set_title("Executed future: action", pad=4)
    axes[2].set_title("Executed future: endpoint", pad=4)
    for label, ax in zip("abc", axes):
        ax.text(
            -0.02,
            1.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    figure.legend(
        handles=task_legend_handles(),
        loc="outside lower center",
        ncol=6,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    save_figure(figure, output / "appendix_cosmos3_magnitude_checks")


def factor_cell_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    prefix: str,
    title: str,
    seed: int,
) -> None:
    cells = [
        ("Neither", f"{prefix}_o0r0"),
        ("Robot", f"{prefix}_o0r1"),
        ("Object", f"{prefix}_o1r0"),
        ("Both", f"{prefix}_o1r1"),
        ("Whole\nfuture", f"full_{prefix}"),
    ]
    for position, (_label, column) in enumerate(cells):
        values = frame[column].to_numpy(dtype=float)
        xs = position + jitter(len(frame), seed + position)
        for x, value, task in zip(xs, values, frame["task"]):
            ax.scatter(
                x,
                value,
                s=18,
                color=TASK_COLORS[task],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.85,
                zorder=3,
            )
        add_summary(ax, position, values, seed=seed + 100 + position)
    ax.axhline(0, color="black", linewidth=0.65)
    ax.axhline(1, color=REFERENCE_GREY, linestyle="--", linewidth=0.7)
    ax.set_xticks(range(len(cells)), [label for label, _ in cells])
    ax.set_xlabel("Future pixels transplanted")
    ax.set_ylim(-0.25, 1.35)
    ax.set_title(title, pad=4)
    horizontal_grid(ax)


def factor_effect_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    prefix: str,
    title: str,
    seed: int,
) -> None:
    effects = [
        ("Robot", f"{prefix}_robot"),
        ("Object", f"{prefix}_object"),
        ("Interaction", f"{prefix}_interaction"),
    ]
    ax.axhspan(-0.10, 0.10, color="#F0F0F0", zorder=0)
    for position, (_label, column) in enumerate(effects):
        values = frame[column].to_numpy(dtype=float)
        xs = position + jitter(len(frame), seed + position)
        for x, value, task in zip(xs, values, frame["task"]):
            ax.scatter(
                x,
                value,
                s=18,
                color=TASK_COLORS[task],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.85,
                zorder=3,
            )
        add_summary(ax, position, values, seed=seed + 100 + position)
    ax.axhline(0, color="black", linewidth=0.65, zorder=1)
    ax.axhline(-0.10, color=REFERENCE_GREY, linestyle="--", linewidth=0.6)
    ax.axhline(0.10, color=REFERENCE_GREY, linestyle="--", linewidth=0.6)
    ax.set_xticks(range(len(effects)), [label for label, _ in effects])
    ax.set_ylim(-0.12, 0.12)
    ax.set_title(title, pad=4)
    horizontal_grid(ax)


def factorial_cells_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    prefix: str,
    title: str,
    seed: int,
) -> None:
    object_conditions = [
        ("Recipient object pixels", "o0", POINT_GREY, "o", "white"),
        ("Donor object pixels", "o1", "#009E73", "o", "#009E73"),
    ]
    for object_index, (label, object_code, color, marker, face) in enumerate(
        object_conditions
    ):
        means = []
        for robot_index, robot_code in enumerate(("r0", "r1")):
            values = frame[f"{prefix}_{object_code}{robot_code}"].to_numpy(
                dtype=float
            )
            offset = -0.035 if object_index == 0 else 0.035
            xs = robot_index + offset + jitter(
                len(values), seed + object_index * 10 + robot_index, width=0.025
            )
            ax.scatter(
                xs,
                values,
                s=14,
                marker=marker,
                facecolor=face,
                edgecolor=color,
                linewidth=0.55,
                alpha=0.48,
                zorder=2,
            )
            mean, lower, upper = bootstrap_mean(
                values, seed=seed + 100 + object_index * 10 + robot_index
            )
            means.append(mean)
            ax.errorbar(
                robot_index + offset,
                mean,
                yerr=[[mean - lower], [upper - mean]],
                fmt=marker,
                color=color,
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=0.8,
                markersize=4.2,
                capsize=2,
                linewidth=1.0,
                zorder=4,
            )
        ax.plot(
            np.array([0, 1]) + offset,
            means,
            color=color,
            linewidth=1.15,
            label=label,
            zorder=3,
        )
    ax.axhline(0, color="black", linewidth=0.65, zorder=1)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.20, 0.36)
    ax.set_xticks([0, 1], ["Recipient", "Donor"])
    ax.set_xlabel("Robot pixels from")
    ax.set_title(title, pad=4)
    horizontal_grid(ax)


def factor_reference_axis(ax: plt.Axes, frame: pd.DataFrame, *, seed: int) -> None:
    effects = [
        (
            "Whole-future\nsteering",
            "full_action_steering",
            "full_endpoint_steering",
        ),
        ("Robot-pixel\nmain effect", "action_robot", "endpoint_robot"),
        ("Object-pixel\nmain effect", "action_object", "endpoint_object"),
    ]
    outcomes = [
        ("Action", 1, ACTION_COLOR, "o", ACTION_COLOR, -0.11),
        ("Executed endpoint", 2, ENDPOINT_COLOR, "s", "white", 0.11),
    ]
    for position, (_label, action_column, endpoint_column) in enumerate(effects):
        for outcome_index, (_name, _key, color, marker, face, offset) in enumerate(
            outcomes
        ):
            column = action_column if outcome_index == 0 else endpoint_column
            values = frame[column].to_numpy(dtype=float)
            xs = position + offset + jitter(
                len(values), seed + position * 10 + outcome_index, width=0.03
            )
            ax.scatter(
                xs,
                values,
                s=14,
                marker=marker,
                facecolor=face,
                edgecolor=color,
                linewidth=0.55,
                alpha=0.42,
                zorder=2,
            )
            add_summary(
                ax,
                position + offset,
                values,
                seed=seed + 100 + position * 10 + outcome_index,
                color=color,
                marker=marker,
                facecolor=face,
            )
    ax.axhspan(-0.10, 0.10, color="#F0F0F0", zorder=0)
    ax.axhline(0, color="black", linewidth=0.65, zorder=1)
    ax.axhline(1, color=REFERENCE_GREY, linestyle=":", linewidth=0.75, zorder=1)
    ax.axvline(0.5, color="#C8C8C8", linewidth=0.65, zorder=1)
    ax.set_xticks(range(len(effects)), [label for label, _a, _e in effects])
    ax.set_ylim(-0.18, 1.18)
    ax.set_ylabel("Effect on donor projection")
    ax.set_title("Whole future and pixel factors", pad=4)
    horizontal_grid(ax)


def figure3_factorization(factorization: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 3.0),
        gridspec_kw={"width_ratios": [0.9, 0.9, 1.35]},
        constrained_layout=True,
    )
    factorial_cells_axis(
        axes[0],
        factorization,
        prefix="action",
        title="2×2 hybrids: action",
        seed=202650,
    )
    factorial_cells_axis(
        axes[1],
        factorization,
        prefix="endpoint",
        title="2×2 hybrids: endpoint",
        seed=202660,
    )
    factor_reference_axis(axes[2], factorization, seed=202670)
    axes[0].set_ylabel(r"Normalized donor projection, $\pi_x$")
    axes[1].set_yticklabels([])
    axes[0].legend(
        loc="upper left",
        frameon=False,
        fontsize=6.4,
        handlelength=1.5,
        borderaxespad=0.15,
    )
    for label, ax in zip("abc", axes):
        panel_label(ax, label)
    figure.legend(
        handles=policy_legend_handles(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    save_figure(figure, output / "figure3_cosmos3_factorization")


def kv_paired_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    before: str,
    after: str,
    title: str,
) -> None:
    for row in frame.itertuples(index=False):
        left = float(getattr(row, before))
        right = float(getattr(row, after))
        color = TASK_COLORS[row.task]
        ax.plot([0, 1], [left, right], color=color, linewidth=0.75, alpha=0.48, zorder=1)
        ax.scatter(
            [0, 1],
            [left, right],
            s=16,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.82,
            zorder=2,
        )
    before_values = frame[before].to_numpy(dtype=float)
    after_values = frame[after].to_numpy(dtype=float)
    ax.plot(
        [0, 1],
        [before_values.mean(), after_values.mean()],
        color="black",
        linewidth=2.0,
        zorder=4,
    )
    add_summary(ax, 0, before_values, seed=202680)
    add_summary(ax, 1, after_values, seed=202681)
    ax.axhline(0, color="black", linewidth=0.65)
    ax.axhline(1, color=REFERENCE_GREY, linestyle="--", linewidth=0.7)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.20, 1.30)
    ax.set_xticks(
        [0, 1],
        ["Unpatched", "Self-future K/V\nreplacement"],
    )
    ax.set_title(title, pad=4)
    ax.set_xlabel("Future-token K/V condition")
    ax.set_ylabel(r"Donor $-$ self steering, $\Delta_x$")
    horizontal_grid(ax)


def figure4_kv(population: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(6.65, 5.35), constrained_layout=True)
    kv_paired_axis(
        axes[0, 0],
        population,
        before="pred_steering",
        after="pred_patch_steering",
        title="Predicted future → action",
    )
    kv_paired_axis(
        axes[0, 1],
        population,
        before="exec_steering",
        after="exec_patch_steering",
        title="Executed future → action",
    )
    kv_paired_axis(
        axes[1, 0],
        population,
        before="endpoint_steering",
        after="endpoint_patch_steering",
        title="Executed future → endpoint",
    )
    ax = axes[1, 1]
    losses = [
        ("Predicted future\n→ action", "pred_loss"),
        ("Executed future\n→ action", "exec_loss"),
        ("Executed future\n→ endpoint", "endpoint_loss"),
    ]
    for position, (_label, column) in enumerate(losses):
        values = population[column].to_numpy(dtype=float)
        xs = position + jitter(len(population), 202690 + position)
        for x, value, task in zip(xs, values, population["task"]):
            ax.scatter(
                x,
                value,
                s=18,
                color=TASK_COLORS[task],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.85,
                zorder=3,
            )
        add_summary(ax, position, values, seed=202790 + position)
    ax.axhline(0, color="black", linewidth=0.65)
    ax.set_xticks(range(len(losses)), [label for label, _ in losses])
    ax.set_ylim(-0.05, 1.25)
    ax.set_ylabel("Unpatched − replaced steering")
    ax.set_title("K/V replacement loss", pad=4)
    horizontal_grid(ax)
    for label, panel in zip("abcd", axes.flat):
        panel_label(panel, label)
    figure.legend(
        handles=task_legend_handles(),
        loc="outside lower center",
        ncol=6,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    save_figure(figure, output / "figure4_cosmos3_kv_pathway")


def policy_distribution_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    categories: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]],
    *,
    title: str,
    ylabel: str,
    seed: int,
    include_endpoint: bool = True,
    reference: float | None = 0.10,
) -> None:
    if include_endpoint:
        outcomes = [
            ("Action", "action_donor_steering", ACTION_COLOR, "o", -0.12, ACTION_COLOR),
            (
                "Endpoint",
                "physical_endpoint_donor_steering",
                ENDPOINT_COLOR,
                "s",
                0.12,
                "white",
            ),
        ]
    else:
        outcomes = [
            ("Action", "action_donor_steering", ACTION_COLOR, "o", 0.0, ACTION_COLOR)
        ]
    for position, (_label, selector) in enumerate(categories):
        selected = frame.loc[selector(frame)]
        for outcome_index, (_name, metric, color, marker, offset, face) in enumerate(outcomes):
            values = selected[metric].to_numpy(dtype=float)
            xs = position + offset + jitter(
                len(values), seed + position * 10 + outcome_index, width=0.035
            )
            ax.scatter(
                xs,
                values,
                s=17,
                marker=marker,
                facecolor=face,
                edgecolor=color,
                linewidth=0.75 if face == "white" else 0.35,
                alpha=0.80,
                zorder=3,
            )
            add_summary(
                ax,
                position + offset,
                values,
                seed=seed + 100 + position * 10 + outcome_index,
                color=color,
                marker=marker,
                facecolor=face,
            )
    ax.axhline(0, color="black", linewidth=0.65)
    if reference is not None:
        ax.axhline(
            reference, color=REFERENCE_GREY, linestyle="--", linewidth=0.7
        )
    ax.set_xticks(range(len(categories)), [label for label, _ in categories])
    ax.set_ylim(-0.20, 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    horizontal_grid(ax)


def policy_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ACTION_COLOR,
            markeredgecolor=ACTION_COLOR,
            markersize=5,
            label="Action",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=ENDPOINT_COLOR,
            markersize=5,
            label="Executed endpoint",
        ),
    ]


def figure5_policy(policy: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 3.0), constrained_layout=True)
    all_contrast = "semantic_all_donor_minus_recipient"
    policy_distribution_axis(
        axes[0],
        policy,
        [
            ("0", lambda f: (f["prefix_chunks"] == 0) & (f["contrast"] == all_contrast)),
            ("3", lambda f: (f["prefix_chunks"] == 3) & (f["contrast"] == all_contrast)),
            ("6", lambda f: (f["prefix_chunks"] == 6) & (f["contrast"] == all_contrast)),
        ],
        title="Registered state cohorts",
        ylabel=r"Donor $-$ self steering, $\Delta_x$",
        seed=202710,
    )
    axes[0].set_xlabel("Open-loop chunks before query")

    policy_distribution_axis(
        axes[1],
        policy,
        [
            (
                "All",
                lambda f: (f["prefix_chunks"] == 0)
                & (f["contrast"] == "semantic_all_donor_minus_recipient"),
            ),
            (
                "Wrist",
                lambda f: (f["prefix_chunks"] == 0)
                & (f["contrast"] == "semantic_wrist_donor_minus_recipient"),
            ),
            (
                "Primary",
                lambda f: (f["prefix_chunks"] == 0)
                & (f["contrast"] == "semantic_primary_donor_minus_recipient"),
            ),
            (
                "Proprio.",
                lambda f: (f["prefix_chunks"] == 0)
                & (f["contrast"] == "semantic_proprio_donor_minus_recipient"),
            ),
        ],
        title="First-query components",
        ylabel="Donor − self action steering",
        seed=202720,
        include_endpoint=False,
    )
    axes[1].set_xlabel("Future component transplanted")

    policy_distribution_axis(
        axes[2],
        policy,
        [
            (
                "Self",
                lambda f: (f["prefix_chunks"] == 0)
                & (f["contrast"] == "semantic_all_donor_minus_recipient"),
            ),
            (
                "Gauss.",
                lambda f: (f["prefix_chunks"] == 0)
                & (f["contrast"] == "semantic_all_donor_minus_gaussian"),
            ),
            (
                "Natural",
                lambda f: (f["prefix_chunks"] == 0)
                & (f["contrast"] == "semantic_all_donor_minus_natural_control"),
            ),
            (
                "Shuffled",
                lambda f: (f["prefix_chunks"] == 0)
                & (f["contrast"] == "semantic_all_donor_minus_shuffled"),
            ),
        ],
        title="First-query comparisons",
        ylabel="Donor advantage\n(donor − control projection)",
        seed=202730,
        reference=None,
    )
    axes[2].set_xlabel("Control future")

    for label, panel in zip("abc", axes):
        panel_label(panel, label)
    figure.legend(
        handles=policy_legend_handles(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    save_figure(figure, output / "figure5_cosmos_policy_scope")


def policy_identity_axis(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    limits = (-0.05, 1.15)
    ax.scatter(
        x,
        y,
        s=24,
        color=ACTION_COLOR,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.85,
        zorder=3,
    )
    ax.plot(limits, limits, color=REFERENCE_GREY, linestyle="--", linewidth=0.8)
    ax.axhline(0, color="#AAAAAA", linewidth=0.55)
    ax.axvline(0, color="#AAAAAA", linewidth=0.55)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.grid(color=GRID_GREY, linewidth=0.55, zorder=0)


def appendix_policy_relationships(policy: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(5.8, 2.9), constrained_layout=True)
    all_contrast = "semantic_all_donor_minus_recipient"
    first = policy.loc[
        (policy["prefix_chunks"] == 0) & (policy["contrast"] == all_contrast)
    ]
    policy_identity_axis(
        axes[0],
        first["action_donor_steering"].to_numpy(dtype=float),
        first["physical_endpoint_donor_steering"].to_numpy(dtype=float),
        xlabel="Donor − self action projection",
        ylabel="Donor − self endpoint projection",
        title="Action–endpoint relation",
    )
    wrist = policy.loc[
        (policy["prefix_chunks"] == 0)
        & (policy["contrast"] == "semantic_wrist_donor_minus_recipient"),
        ["unit_id", "action_donor_steering"],
    ].rename(columns={"action_donor_steering": "wrist"})
    joined = first[["unit_id", "action_donor_steering"]].merge(wrist, on="unit_id")
    policy_identity_axis(
        axes[1],
        joined["action_donor_steering"].to_numpy(dtype=float),
        joined["wrist"].to_numpy(dtype=float),
        xlabel="All-future action steering",
        ylabel="Wrist-future action steering",
        title="All future–wrist relation",
    )
    for label, panel in zip("ab", axes):
        panel_label(panel, label)
    save_figure(figure, output / "appendix_cosmos_policy_relationships")


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    style()
    population = load_cosmos3_population(args.root)
    factorization = load_cosmos3_factorization(args.root)
    policy = load_policy_state_means(args.root)
    figure2_steering(population, policy, args.output_dir)
    appendix_cosmos3_conditions(population, args.output_dir)
    appendix_magnitude_checks(population, args.output_dir)
    figure3_factorization(factorization, args.output_dir)
    figure4_kv(population, args.output_dir)
    figure5_policy(policy, args.output_dir)
    appendix_policy_relationships(policy, args.output_dir)
    print(f"Wrote main data figures to {args.output_dir}")


if __name__ == "__main__":
    main()
