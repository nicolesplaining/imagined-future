"""Generate plain exploratory plots for all manuscript-relevant result sets.

These plots are intentionally basic.  They expose raw state/task points, paired
conditions, and ordinary x-y relationships before any publication-figure
selection or visual design work.
"""

from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "output" / "exploratory_data_plots"

PLOTS: list[dict[str, str]] = []


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            parsed = number(value)
            if parsed is not None:
                row[key] = parsed
    return rows


def grouped_mean(
    rows: Iterable[dict[str, Any]], keys: tuple[str, ...], values: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        group_key = tuple(row.get(key) for key in keys)
        for value in values:
            parsed = number(row.get(value))
            if parsed is not None:
                groups[group_key][value].append(parsed)
    output = []
    for group_key, measured in groups.items():
        row = dict(zip(keys, group_key))
        for value in values:
            if measured[value]:
                row[value] = fmean(measured[value])
        output.append(row)
    return output


def task_colors(tasks: Iterable[str]) -> dict[str, Any]:
    unique = sorted(set(tasks))
    cmap = plt.get_cmap("tab10")
    return {task: cmap(i % 10) for i, task in enumerate(unique)}


def finish(fig: plt.Figure, group: str, name: str, title: str, source: str) -> None:
    folder = OUT / group
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    PLOTS.append(
        {
            "group": group,
            "name": name,
            "title": title,
            "source": source,
            "path": str(path.relative_to(OUT)),
        }
    )


def category_plot(
    data: dict[str, list[float]],
    *,
    group: str,
    name: str,
    title: str,
    ylabel: str,
    source: str,
    reference_lines: tuple[float, ...] = (),
) -> None:
    fig, ax = plt.subplots(figsize=(max(5.2, 1.05 * len(data)), 4.1))
    rng = np.random.default_rng(7)
    for x, (label, values) in enumerate(data.items()):
        values = [value for value in values if number(value) is not None]
        if not values:
            continue
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(x + jitter, values, s=23, alpha=0.65)
        ax.scatter(x, fmean(values), color="black", marker="_", s=220, linewidths=2)
    for value in reference_lines:
        ax.axhline(value, color="0.55", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(data)), list(data), rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, group, name, title, source)


def paired_plot(
    pairs: list[tuple[float, float]],
    *,
    labels: tuple[str, str],
    group: str,
    name: str,
    title: str,
    ylabel: str,
    source: str,
    reference_lines: tuple[float, ...] = (),
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    for left, right in pairs:
        ax.plot([0, 1], [left, right], color="0.75", linewidth=0.9, alpha=0.8)
        ax.scatter([0, 1], [left, right], s=20)
    if pairs:
        ax.plot(
            [0, 1],
            [fmean(pair[0] for pair in pairs), fmean(pair[1] for pair in pairs)],
            color="black",
            linewidth=2.5,
            marker="o",
        )
    for value in reference_lines:
        ax.axhline(value, color="0.55", linewidth=1, linestyle="--")
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    finish(fig, group, name, title, source)


def xy_plot(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    group: str,
    name: str,
    title: str,
    xlabel: str,
    ylabel: str,
    source: str,
    label_key: str | None = None,
    identity: bool = False,
    zero_lines: bool = False,
) -> None:
    points = [
        row
        for row in rows
        if number(row.get(x_key)) is not None and number(row.get(y_key)) is not None
    ]
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    if label_key and points:
        colors = task_colors(str(row[label_key]) for row in points)
        for label in colors:
            selected = [row for row in points if str(row[label_key]) == label]
            ax.scatter(
                [row[x_key] for row in selected],
                [row[y_key] for row in selected],
                s=30,
                alpha=0.75,
                label=label,
                color=colors[label],
            )
        if len(colors) <= 10:
            ax.legend(fontsize=6.5, frameon=False)
    else:
        ax.scatter(
            [row[x_key] for row in points],
            [row[y_key] for row in points],
            s=30,
            alpha=0.7,
        )
    if identity and points:
        values = [row[x_key] for row in points] + [row[y_key] for row in points]
        low, high = min(values), max(values)
        ax.plot([low, high], [low, high], color="0.45", linestyle="--", linewidth=1)
    if zero_lines:
        ax.axhline(0, color="0.65", linewidth=1)
        ax.axvline(0, color="0.65", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    finish(fig, group, name, title, source)


def line_by_unit(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    unit_key: str,
    *,
    group: str,
    name: str,
    title: str,
    xlabel: str,
    ylabel: str,
    source: str,
    reference_lines: tuple[float, ...] = (),
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if number(row.get(x_key)) is not None and number(row.get(y_key)) is not None:
            units[str(row[unit_key])].append(row)
    for values in units.values():
        values.sort(key=lambda row: row[x_key])
        ax.plot(
            [row[x_key] for row in values],
            [row[y_key] for row in values],
            color="0.7",
            alpha=0.6,
            linewidth=0.8,
            marker="o",
            markersize=3,
        )
    xs = sorted({row[x_key] for rows_for_unit in units.values() for row in rows_for_unit})
    means = []
    for x in xs:
        values = [
            row[y_key]
            for rows_for_unit in units.values()
            for row in rows_for_unit
            if row[x_key] == x
        ]
        means.append(fmean(values))
    if xs:
        ax.plot(xs, means, color="black", linewidth=2.5, marker="o")
    for value in reference_lines:
        ax.axhline(value, color="0.55", linewidth=1, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    finish(fig, group, name, title, source)


def nested(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        value = value[part]
    return value


def cosmos3_population() -> list[dict[str, Any]]:
    folder = RESULTS / "cosmos3_population_confirmatory_v1"
    summaries = []
    for path in sorted(folder.glob("*/summary.json")):
        data = json.loads(path.read_text())
        interventions = data["interventions"]
        row = {
            "unit": f"{data['task']}-seed-{data['environment_seed']}",
            "task": data["task"],
            "seed": data["environment_seed"],
            "action_separation": data["native_action_l2"],
            "endpoint_separation": data["native_endpoint_l2"]["all"],
        }
        condition_paths = {
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
            "endpoint_patch": "executed_donor_kv_patch_all_action.endpoint_donor_projection.all",
        }
        for key, field in condition_paths.items():
            row[key] = nested(interventions, field)
        row["pred_kv_loss"] = row["pred_donor"] - row["pred_patch"]
        row["exec_kv_loss"] = row["exec_donor"] - row["exec_patch"]
        row["endpoint_kv_loss"] = row["endpoint_donor"] - row["endpoint_patch"]
        summaries.append(row)

    source = "results/cosmos3_population_confirmatory_v1/*/summary.json"
    category_plot(
        {
            "Self": [row["pred_self"] for row in summaries],
            "Natural": [row["pred_natural"] for row in summaries],
            "Donor": [row["pred_donor"] for row in summaries],
        },
        group="cosmos3_population",
        name="01_predicted_action_by_condition",
        title="Predicted-future action projection by condition",
        ylabel="Action donor projection",
        source=source,
        reference_lines=(0, 1),
    )
    category_plot(
        {
            "Self": [row["exec_self"] for row in summaries],
            "Gaussian": [row["exec_gaussian"] for row in summaries],
            "Natural": [row["exec_natural"] for row in summaries],
            "Donor": [row["exec_donor"] for row in summaries],
        },
        group="cosmos3_population",
        name="02_executed_action_by_condition",
        title="Executed-future action projection by condition",
        ylabel="Action donor projection",
        source=source,
        reference_lines=(0, 1),
    )
    category_plot(
        {
            "Self": [row["endpoint_self"] for row in summaries],
            "Gaussian": [row["endpoint_gaussian"] for row in summaries],
            "Natural": [row["endpoint_natural"] for row in summaries],
            "Donor": [row["endpoint_donor"] for row in summaries],
        },
        group="cosmos3_population",
        name="03_executed_endpoint_by_condition",
        title="Executed endpoint projection by condition",
        ylabel="Endpoint donor projection",
        source=source,
        reference_lines=(0, 1),
    )

    xy_specs = [
        ("pred_donor", "exec_donor", "04_predicted_vs_executed_action", "Predicted action projection", "Executed action projection"),
        ("exec_donor", "endpoint_donor", "05_executed_action_vs_endpoint", "Executed action projection", "Endpoint projection"),
        ("pred_donor", "endpoint_donor", "06_predicted_action_vs_endpoint", "Predicted action projection", "Endpoint projection"),
        ("action_separation", "pred_donor", "07_action_separation_vs_predicted_projection", "Native action separation (L2)", "Predicted action projection"),
        ("action_separation", "exec_donor", "08_action_separation_vs_executed_projection", "Native action separation (L2)", "Executed action projection"),
        ("endpoint_separation", "endpoint_donor", "09_endpoint_separation_vs_projection", "Native endpoint separation (L2)", "Endpoint projection"),
        ("action_separation", "pred_l2", "10_native_vs_transplanted_predicted_action_l2", "Native action separation (L2)", "Transplanted action displacement (L2)"),
        ("action_separation", "exec_l2", "11_native_vs_transplanted_executed_action_l2", "Native action separation (L2)", "Transplanted action displacement (L2)"),
        ("endpoint_separation", "endpoint_l2", "12_native_vs_transplanted_endpoint_l2", "Native endpoint separation (L2)", "Transplanted endpoint displacement (L2)"),
        ("pred_natural", "pred_donor", "13_natural_vs_predicted_donor", "Natural-control projection", "Predicted-donor projection"),
        ("exec_gaussian", "exec_donor", "14_gaussian_vs_executed_donor", "Gaussian projection", "Executed-donor projection"),
        ("exec_natural", "exec_donor", "15_natural_vs_executed_donor", "Natural-control projection", "Executed-donor projection"),
    ]
    for x_key, y_key, name, xlabel, ylabel in xy_specs:
        xy_plot(
            summaries,
            x_key,
            y_key,
            group="cosmos3_population",
            name=name,
            title=f"{xlabel} vs. {ylabel}",
            xlabel=xlabel,
            ylabel=ylabel,
            source=source,
            label_key="task",
            identity=name.startswith("10_") or name.startswith("11_") or name.startswith("12_"),
        )
    return summaries


def cosmos3_pathway(rows: list[dict[str, Any]]) -> None:
    source = "results/cosmos3_population_confirmatory_v1/*/summary.json"
    specs = [
        ("pred_donor", "pred_patch", "01_predicted_kv_before_after", "Predicted action"),
        ("exec_donor", "exec_patch", "02_executed_kv_before_after", "Executed action"),
        ("endpoint_donor", "endpoint_patch", "03_endpoint_kv_before_after", "Endpoint"),
    ]
    for before, after, name, label in specs:
        paired_plot(
            [(row[before], row[after]) for row in rows],
            labels=("Donor K/V", "Self-future K/V"),
            group="cosmos3_pathway",
            name=name,
            title=f"{label}: projection before and after K/V replacement",
            ylabel="Donor projection",
            source=source,
            reference_lines=(0, 1),
        )
        xy_plot(
            rows,
            before,
            after,
            group="cosmos3_pathway",
            name=name.replace("before_after", "before_vs_after"),
            title=f"{label}: before vs. after K/V replacement",
            xlabel="Before replacement",
            ylabel="After replacement",
            source=source,
            label_key="task",
            identity=True,
        )
    for base, loss, name, label in [
        ("pred_donor", "pred_kv_loss", "07_predicted_effect_vs_kv_loss", "Predicted action"),
        ("exec_donor", "exec_kv_loss", "08_executed_effect_vs_kv_loss", "Executed action"),
        ("endpoint_donor", "endpoint_kv_loss", "09_endpoint_effect_vs_kv_loss", "Endpoint"),
    ]:
        xy_plot(
            rows,
            base,
            loss,
            group="cosmos3_pathway",
            name=name,
            title=f"{label}: donor projection vs. K/V loss",
            xlabel="Donor projection before replacement",
            ylabel="Reduction after replacement",
            source=source,
            label_key="task",
        )


def cosmos3_factorization() -> None:
    folder = RESULTS / "cosmos3_population_confirmatory_v1"
    rows = []
    for path in sorted(folder.glob("*/summary.json")):
        data = json.loads(path.read_text())
        effects = (data.get("factorial_effects") or {}).get("composite")
        if not effects:
            continue
        row = {
            "unit": f"{data['task']}-seed-{data['environment_seed']}",
            "task": data["task"],
            "full_action": data["interventions"]["executed_donor"]["action_donor_projection"],
            "full_endpoint": data["interventions"]["executed_donor"]["endpoint_donor_projection"]["all"],
            "robot_mask_fraction": data["factorization"]["robot_mask_future_pixel_fraction"],
            "object_mask_fraction": data["factorization"]["object_mask_future_pixel_fraction"],
            "action_robot_effect": effects["action_donor_projection_effects"]["robot_main_effect"],
            "action_object_effect": effects["action_donor_projection_effects"]["object_main_effect"],
            "action_interaction": effects["action_donor_projection_effects"]["interaction"],
            "endpoint_robot_effect": effects["endpoint_donor_projection_effects"]["all"]["robot_main_effect"],
            "endpoint_object_effect": effects["endpoint_donor_projection_effects"]["all"]["object_main_effect"],
            "endpoint_interaction": effects["endpoint_donor_projection_effects"]["all"]["interaction"],
        }
        for cell, value in effects["action_donor_projection_cells"].items():
            row[f"action_{cell}"] = value
        for cell, value in effects["endpoint_donor_projection_cells"]["all"].items():
            row[f"endpoint_{cell}"] = value
        rows.append(row)
    source = "results/cosmos3_population_confirmatory_v1/*/summary.json (factorial subset)"
    cells = ("o0r0", "o0r1", "o1r0", "o1r1")
    category_plot(
        {cell: [row[f"action_{cell}"] for row in rows] for cell in cells}
        | {"full donor": [row["full_action"] for row in rows]},
        group="cosmos3_factorization",
        name="01_action_projection_by_factorial_cell",
        title="Action projection by robot/object pixel cell",
        ylabel="Action donor projection",
        source=source,
        reference_lines=(0, 0.1, 1),
    )
    category_plot(
        {cell: [row[f"endpoint_{cell}"] for row in rows] for cell in cells}
        | {"full donor": [row["full_endpoint"] for row in rows]},
        group="cosmos3_factorization",
        name="02_endpoint_projection_by_factorial_cell",
        title="Endpoint projection by robot/object pixel cell",
        ylabel="Endpoint donor projection",
        source=source,
        reference_lines=(0, 0.1, 1),
    )
    category_plot(
        {
            "Robot": [row["action_robot_effect"] for row in rows],
            "Object": [row["action_object_effect"] for row in rows],
            "Interaction": [row["action_interaction"] for row in rows],
        },
        group="cosmos3_factorization",
        name="03_action_factor_effects",
        title="Action factorial effects",
        ylabel="Projection effect",
        source=source,
        reference_lines=(-0.1, 0, 0.1),
    )
    category_plot(
        {
            "Robot": [row["endpoint_robot_effect"] for row in rows],
            "Object": [row["endpoint_object_effect"] for row in rows],
            "Interaction": [row["endpoint_interaction"] for row in rows],
        },
        group="cosmos3_factorization",
        name="04_endpoint_factor_effects",
        title="Endpoint factorial effects",
        ylabel="Projection effect",
        source=source,
        reference_lines=(-0.1, 0, 0.1),
    )
    xy_specs = [
        ("action_robot_effect", "action_object_effect", "05_robot_vs_object_action_effect", "Robot action effect", "Object action effect"),
        ("endpoint_robot_effect", "endpoint_object_effect", "06_robot_vs_object_endpoint_effect", "Robot endpoint effect", "Object endpoint effect"),
        ("robot_mask_fraction", "action_robot_effect", "07_robot_mask_size_vs_action_effect", "Robot mask fraction", "Robot action effect"),
        ("object_mask_fraction", "action_object_effect", "08_object_mask_size_vs_action_effect", "Object mask fraction", "Object action effect"),
        ("action_o1r1", "full_action", "09_both_pixels_vs_full_donor_action", "Both-pixel cell projection", "Full donor projection"),
        ("endpoint_o1r1", "full_endpoint", "10_both_pixels_vs_full_donor_endpoint", "Both-pixel endpoint projection", "Full donor endpoint projection"),
    ]
    for x_key, y_key, name, xlabel, ylabel in xy_specs:
        xy_plot(
            rows,
            x_key,
            y_key,
            group="cosmos3_factorization",
            name=name,
            title=f"{xlabel} vs. {ylabel}",
            xlabel=xlabel,
            ylabel=ylabel,
            source=source,
            label_key="task",
            zero_lines=True,
        )


def cosmos_policy_semantic() -> None:
    path = RESULTS / "confirmatory_v1" / "semantic_state_repetitions.csv"
    rows = read_csv(path)
    means = grouped_mean(
        rows,
        ("unit_id", "task_id", "prefix_chunks", "contrast"),
        ("action_donor_steering", "physical_endpoint_donor_steering"),
    )
    source = str(path.relative_to(ROOT))
    all_self = [row for row in means if row["contrast"] == "semantic_all_donor_minus_recipient"]
    timing_action: dict[str, list[float]] = {}
    timing_endpoint: dict[str, list[float]] = {}
    for prefix in (0.0, 3.0, 6.0):
        timing_action[f"Prefix {int(prefix)}"] = [
            row["action_donor_steering"] for row in all_self if row["prefix_chunks"] == prefix
        ]
        timing_endpoint[f"Prefix {int(prefix)}"] = [
            row["physical_endpoint_donor_steering"] for row in all_self if row["prefix_chunks"] == prefix
        ]
    category_plot(
        timing_action,
        group="cosmos_policy",
        name="01_action_by_prefix",
        title="Cosmos Policy action steering by registered prefix",
        ylabel="Donor-minus-recipient projection",
        source=source,
        reference_lines=(0, 0.1),
    )
    category_plot(
        timing_endpoint,
        group="cosmos_policy",
        name="02_endpoint_by_prefix",
        title="Cosmos Policy endpoint steering by registered prefix",
        ylabel="Donor-minus-recipient projection",
        source=source,
        reference_lines=(0, 0.1),
    )
    first = [row for row in means if row["prefix_chunks"] == 0.0]
    modality_names = {
        "All": "semantic_all_donor_minus_recipient",
        "Wrist": "semantic_wrist_donor_minus_recipient",
        "Primary": "semantic_primary_donor_minus_recipient",
        "Proprio": "semantic_proprio_donor_minus_recipient",
    }
    for metric, suffix, ylabel in [
        ("action_donor_steering", "action", "Action projection"),
        ("physical_endpoint_donor_steering", "endpoint", "Endpoint projection"),
    ]:
        category_plot(
            {
                label: [row[metric] for row in first if row["contrast"] == contrast]
                for label, contrast in modality_names.items()
            },
            group="cosmos_policy",
            name=f"03_{suffix}_by_modality" if suffix == "action" else f"04_{suffix}_by_modality",
            title=f"First-query {suffix} steering by future modality",
            ylabel=ylabel,
            source=source,
            reference_lines=(0, 0.1),
        )
    control_names = {
        "Recipient": "semantic_all_donor_minus_recipient",
        "Gaussian": "semantic_all_donor_minus_gaussian",
        "Natural": "semantic_all_donor_minus_natural_control",
        "Shuffled": "semantic_all_donor_minus_shuffled",
    }
    for metric, name, ylabel in [
        ("action_donor_steering", "05_action_control_contrasts", "Action contrast"),
        ("physical_endpoint_donor_steering", "06_endpoint_control_contrasts", "Endpoint contrast"),
    ]:
        category_plot(
            {
                label: [row[metric] for row in first if row["contrast"] == contrast]
                for label, contrast in control_names.items()
            },
            group="cosmos_policy",
            name=name,
            title=f"First-query donor-minus-control {ylabel.lower()}",
            ylabel=ylabel,
            source=source,
            reference_lines=(0, 0.1),
        )
    xy_plot(
        all_self,
        "action_donor_steering",
        "physical_endpoint_donor_steering",
        group="cosmos_policy",
        name="07_action_vs_endpoint_all_prefixes",
        title="Action vs. endpoint steering across registered states",
        xlabel="Action steering",
        ylabel="Endpoint steering",
        source=source,
        label_key="task_id",
        identity=True,
        zero_lines=True,
    )
    xy_plot(
        [row for row in all_self if row["prefix_chunks"] == 0.0],
        "action_donor_steering",
        "physical_endpoint_donor_steering",
        group="cosmos_policy",
        name="08_action_vs_endpoint_first_query",
        title="First-query action vs. endpoint steering",
        xlabel="Action steering",
        ylabel="Endpoint steering",
        source=source,
        label_key="task_id",
        identity=True,
        zero_lines=True,
    )
    pivot: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in first:
        pivot[str(row["unit_id"])][str(row["contrast"])] = row
    for label, contrast in list(modality_names.items())[1:]:
        joined = []
        for unit, values in pivot.items():
            if modality_names["All"] in values and contrast in values:
                joined.append(
                    {
                        "unit": unit,
                        "task_id": values[contrast]["task_id"],
                        "all": values[modality_names["All"]]["action_donor_steering"],
                        "component": values[contrast]["action_donor_steering"],
                    }
                )
        xy_plot(
            joined,
            "all",
            "component",
            group="cosmos_policy",
            name=f"09_all_vs_{label.lower()}_action",
            title=f"All-future vs. {label.lower()}-future action steering",
            xlabel="All-future action steering",
            ylabel=f"{label} action steering",
            source=source,
            label_key="task_id",
            identity=True,
            zero_lines=True,
        )


def cosmos_policy_attention() -> None:
    path = RESULTS / "confirmatory_v1" / "attention_state_repetitions.csv"
    rows = read_csv(path)
    means = grouped_mean(
        rows,
        ("unit_id", "task_id", "prefix_chunks", "contrast", "gate"),
        (
            "action_l2_from_baseline",
            "max_abs_from_baseline",
            "action_donor_steering",
            "physical_endpoint_donor_steering",
        ),
    )
    source = str(path.relative_to(ROOT))
    dose = [
        row
        for row in means
        if str(row["contrast"]).startswith("block27_future_gate")
        and "minus" not in str(row["contrast"])
        and number(row.get("gate")) is not None
    ]
    for metric, name, ylabel in [
        ("action_l2_from_baseline", "01_gate_vs_action_l2", "Action L2 from baseline"),
        ("max_abs_from_baseline", "02_gate_vs_max_action_change", "Maximum action change"),
        ("action_donor_steering", "03_gate_vs_action_projection", "Action donor projection"),
        ("physical_endpoint_donor_steering", "04_gate_vs_endpoint_projection", "Endpoint donor projection"),
    ]:
        line_by_unit(
            dose,
            "gate",
            metric,
            "unit_id",
            group="cosmos_policy_attention",
            name=name,
            title=f"Future-key gate vs. {ylabel.lower()}",
            xlabel="Future-key removal gate",
            ylabel=ylabel,
            source=source,
            reference_lines=(0,),
        )
    contrasts = {
        "Block 27 future": "block27_future_gate1",
        "Block 27 random": "block27_random_gate1",
        "Block 27 current": "block27_current_gate1",
        "Block 0 future": "block0_future_gate1",
        "All-key control": "block27_all_key_control",
    }
    for metric, name, ylabel in [
        ("action_l2_from_baseline", "05_full_gate_control_action_l2", "Action L2 from baseline"),
        ("physical_endpoint_donor_steering", "06_full_gate_control_endpoint", "Endpoint donor projection"),
    ]:
        category_plot(
            {
                label: [row[metric] for row in means if row["contrast"] == contrast and metric in row]
                for label, contrast in contrasts.items()
            },
            group="cosmos_policy_attention",
            name=name,
            title=f"Full-gate controls: {ylabel.lower()}",
            ylabel=ylabel,
            source=source,
            reference_lines=(0,),
        )


def content_factorization() -> None:
    path = RESULTS / "content_factorization_v1" / "factorization_repetitions.csv"
    rows = read_csv(path)
    means = grouped_mean(
        rows,
        ("unit_id", "task_id", "pair_type", "contrast"),
        (
            "action_donor_steering",
            "goal_endpoint_donor_steering",
            "robot_endpoint_donor_steering",
        ),
    )
    source = str(path.relative_to(ROOT))
    action_contrasts = sorted({str(row["contrast"]) for row in means})
    category_plot(
        {
            contrast.replace("_donor_minus_", " − ").replace("_", " "): [
                row["action_donor_steering"]
                for row in means
                if row["contrast"] == contrast and "action_donor_steering" in row
            ]
            for contrast in action_contrasts
        },
        group="cosmos_policy_content_factorization",
        name="01_action_by_natural_pair_contrast",
        title="Natural-pair action steering contrasts",
        ylabel="Action projection contrast",
        source=source,
        reference_lines=(0, 0.1),
    )
    primary = [
        row
        for row in means
        if row["contrast"] in {"robot_all_donor_minus_recipient", "object_all_donor_minus_recipient"}
    ]
    category_plot(
        {
            "Robot pair": [
                row["action_donor_steering"]
                for row in primary
                if row["contrast"] == "robot_all_donor_minus_recipient"
            ],
            "Object pair": [
                row["action_donor_steering"]
                for row in primary
                if row["contrast"] == "object_all_donor_minus_recipient"
            ],
        },
        group="cosmos_policy_content_factorization",
        name="02_robot_vs_object_pair_action",
        title="Robot-pair and object-pair action steering",
        ylabel="Action donor projection",
        source=source,
        reference_lines=(0, 0.1),
    )
    robot_rows = [row for row in primary if row["contrast"] == "robot_all_donor_minus_recipient"]
    object_rows = [row for row in primary if row["contrast"] == "object_all_donor_minus_recipient"]
    xy_plot(
        robot_rows,
        "action_donor_steering",
        "robot_endpoint_donor_steering",
        group="cosmos_policy_content_factorization",
        name="03_robot_action_vs_robot_endpoint",
        title="Robot-pair action vs. robot endpoint steering",
        xlabel="Action steering",
        ylabel="Robot endpoint steering",
        source=source,
        label_key="task_id",
        identity=True,
        zero_lines=True,
    )
    xy_plot(
        object_rows,
        "action_donor_steering",
        "goal_endpoint_donor_steering",
        group="cosmos_policy_content_factorization",
        name="04_object_action_vs_goal_endpoint",
        title="Object-pair action vs. goal endpoint steering",
        xlabel="Action steering",
        ylabel="Goal endpoint steering",
        source=source,
        label_key="task_id",
        identity=True,
        zero_lines=True,
    )


def factorial_hybrid() -> None:
    cell_path = RESULTS / "factorial_hybrid_v1" / "factorial_cell_repetitions.csv"
    effect_path = RESULTS / "factorial_hybrid_v1" / "factorial_effect_repetitions.csv"
    cells = grouped_mean(
        read_csv(cell_path),
        ("unit_id", "task_id", "target_cell"),
        (
            "goal_endpoint_donor_steering",
            "robot_endpoint_donor_steering",
            "decoded_primary_target_margin",
            "decoded_wrist_target_margin",
        ),
    )
    source = str(cell_path.relative_to(ROOT))
    target_cells = ("o0r0", "o0r1", "o1r0", "o1r1")
    for metric, name, ylabel in [
        ("goal_endpoint_donor_steering", "01_goal_endpoint_by_cell", "Goal endpoint projection"),
        ("robot_endpoint_donor_steering", "02_robot_endpoint_by_cell", "Robot endpoint projection"),
        ("decoded_primary_target_margin", "03_primary_decode_margin_by_cell", "Primary decode margin"),
        ("decoded_wrist_target_margin", "04_wrist_decode_margin_by_cell", "Wrist decode margin"),
    ]:
        category_plot(
            {
                cell: [row[metric] for row in cells if row["target_cell"] == cell and metric in row]
                for cell in target_cells
            },
            group="cosmos_policy_rendered_factorial",
            name=name,
            title=f"{ylabel} by rendered factorial cell",
            ylabel=ylabel,
            source=source,
            reference_lines=(0,),
        )
    xy_plot(
        cells,
        "decoded_primary_target_margin",
        "goal_endpoint_donor_steering",
        group="cosmos_policy_rendered_factorial",
        name="05_primary_decode_margin_vs_goal_endpoint",
        title="Primary decode margin vs. goal endpoint projection",
        xlabel="Primary decode margin",
        ylabel="Goal endpoint projection",
        source=source,
        label_key="task_id",
        zero_lines=True,
    )
    effects = grouped_mean(
        read_csv(effect_path),
        ("unit_id", "task_id", "modality"),
        (
            "goal_endpoint_donor_steering__object_main_effect",
            "goal_endpoint_donor_steering__robot_main_effect",
            "goal_endpoint_donor_steering__interaction",
            "native_action_similarity__object_main_effect",
            "native_action_similarity__robot_main_effect",
            "native_action_similarity__interaction",
        ),
    )
    for modality in sorted({str(row["modality"]) for row in effects}):
        selected = [row for row in effects if row["modality"] == modality]
        category_plot(
            {
                "Object": [row["native_action_similarity__object_main_effect"] for row in selected if "native_action_similarity__object_main_effect" in row],
                "Robot": [row["native_action_similarity__robot_main_effect"] for row in selected if "native_action_similarity__robot_main_effect" in row],
                "Interaction": [row["native_action_similarity__interaction"] for row in selected if "native_action_similarity__interaction" in row],
            },
            group="cosmos_policy_rendered_factorial",
            name=f"06_{modality}_action_factor_effects",
            title=f"Rendered factorial action effects ({modality})",
            ylabel="Native-action similarity effect",
            source=str(effect_path.relative_to(ROOT)),
            reference_lines=(0, 0.1),
        )


def robocasa_replication() -> None:
    semantic_path = RESULTS / "robocasa_replication_v1" / "semantic_repetitions.csv"
    means = grouped_mean(
        read_csv(semantic_path),
        ("unit_id", "task_name", "contrast"),
        ("action_donor_steering", "physical_endpoint_donor_steering"),
    )
    source = str(semantic_path.relative_to(ROOT))
    contrasts = {
        "Recipient": "semantic_donor_minus_recipient",
        "Gaussian": "semantic_donor_minus_gaussian",
    }
    for metric, name, ylabel in [
        ("action_donor_steering", "01_action_contrasts", "Action contrast"),
        ("physical_endpoint_donor_steering", "02_endpoint_contrasts", "Endpoint contrast"),
    ]:
        category_plot(
            {
                label: [row[metric] for row in means if row["contrast"] == contrast]
                for label, contrast in contrasts.items()
            },
            group="robocasa_replication",
            name=name,
            title=f"RoboCasa {ylabel.lower()} by control",
            ylabel=ylabel,
            source=source,
            reference_lines=(0, 0.1),
        )
    primary = [row for row in means if row["contrast"] == "semantic_donor_minus_recipient"]
    xy_plot(
        primary,
        "action_donor_steering",
        "physical_endpoint_donor_steering",
        group="robocasa_replication",
        name="03_action_vs_endpoint",
        title="RoboCasa action vs. endpoint steering",
        xlabel="Action steering",
        ylabel="Endpoint steering",
        source=source,
        label_key="task_name",
        identity=True,
        zero_lines=True,
    )

    attention_path = RESULTS / "robocasa_replication_v1" / "attention_repetitions.csv"
    attention = grouped_mean(
        read_csv(attention_path),
        ("unit_id", "task_name", "contrast"),
        ("action_l2_from_baseline", "physical_endpoint_donor_steering"),
    )
    contrasts_attention = {
        "Future keys": "block27_future_gate1",
        "Current keys": "block27_current_gate1",
        "All-key control": "block27_all_key_control",
    }
    category_plot(
        {
            label: [
                row["action_l2_from_baseline"]
                for row in attention
                if row["contrast"] == contrast and "action_l2_from_baseline" in row
            ]
            for label, contrast in contrasts_attention.items()
        },
        group="robocasa_replication",
        name="04_attention_action_l2",
        title="RoboCasa attention-control action change",
        ylabel="Action L2 from baseline",
        source=str(attention_path.relative_to(ROOT)),
        reference_lines=(0,),
    )


def write_index() -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for plot in PLOTS:
        groups[plot["group"]].append(plot)
    cards = []
    for group in sorted(groups):
        cards.append(f"<h2>{html.escape(group.replace('_', ' ').title())}</h2><div class='grid'>")
        for plot in sorted(groups[group], key=lambda item: item["name"]):
            cards.append(
                "<figure>"
                f"<a href='{html.escape(plot['path'])}'><img src='{html.escape(plot['path'])}'></a>"
                f"<figcaption><strong>{html.escape(plot['title'])}</strong><br>"
                f"<code>{html.escape(plot['source'])}</code></figcaption>"
                "</figure>"
            )
        cards.append("</div>")
    document = """<!doctype html>
<html><head><meta charset="utf-8"><title>Exploratory data plots</title>
<style>
body { font: 15px system-ui, sans-serif; margin: 24px; color: #222; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 18px; }
figure { margin: 0; padding: 10px; border: 1px solid #ddd; }
img { width: 100%; height: auto; }
figcaption { margin-top: 8px; line-height: 1.35; }
code { font-size: 11px; color: #555; }
</style></head><body>
<h1>Exploratory data plots</h1>
<p>Plain diagnostic views only. Points are independent state/task means unless the title says otherwise; black marks denote simple means. These are not manuscript-ready figures.</p>
""" + "\n".join(cards) + "</body></html>"
    (OUT / "index.html").write_text(document)
    with (OUT / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("group", "name", "title", "source", "path"))
        writer.writeheader()
        writer.writerows(PLOTS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    population = cosmos3_population()
    cosmos3_pathway(population)
    cosmos3_factorization()
    cosmos_policy_semantic()
    cosmos_policy_attention()
    content_factorization()
    factorial_hybrid()
    robocasa_replication()
    write_index()
    print(f"Wrote {len(PLOTS)} plots to {OUT}")


if __name__ == "__main__":
    main()
