#!/usr/bin/env python3
"""Aggregate Cosmos 3 future-target x future-K/V factorial runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    import tomli as tomllib  # type: ignore[no-redef]

import numpy as np

from imagined_future.cosmos3_protocol import FROZEN_TASK_OBJECT_NAMES

from overnight_summary_common import (
    SummaryValidationError,
    atomic_write_text,
    collapse_to_states,
    discover_summaries,
    finite_or_none,
    format_estimate,
    latex_escape,
    leave_one_task_out,
    read_json,
    required,
    required_bool,
    required_finite_or_none,
    save_figure,
    state_id,
    summarize_metric,
    write_csv,
    write_json,
)


SOURCES = ("predicted", "executed")
DOMAINS = (
    "action",
    "endpoint_all",
    "endpoint_robot",
    "endpoint_object",
    "endpoint_target_object_position",
)
METRICS = (
    "normalized_projection",
    "distance_reduction",
    "cosine_alignment",
    "orthogonal_residual_normalized",
)
CELLS = (
    "recipient_future_recipient_kv",
    "donor_future_recipient_kv",
    "recipient_future_donor_kv",
    "donor_future_donor_kv",
)
CONTRASTS = (
    "suppression",
    "rescue",
    "interaction",
    "future_effect_with_recipient_kv",
    "future_effect_with_donor_kv",
)

MINIMAL_PHYSICAL_LABELS = frozenset(
    {
        "self",
        "predicted_donor",
        "executed_donor",
        "predicted_donor_kv_patch_all_action",
        "executed_donor_kv_patch_all_action",
        "self_with_predicted_donor_kv",
        "self_with_executed_donor_kv",
    }
)

MINIMAL_ACTION_ONLY_LABELS = frozenset(
    {
        "gaussian_executed",
        "executed_self",
        "self_kv_record",
        "self_kv_patch_all",
        "predicted_donor_kv_record",
        "predicted_donor_kv_replay",
        "executed_donor_kv_record",
        "executed_donor_kv_replay",
    }
)


def load_expected_manifest(
    path: Path, config_path: Path
) -> tuple[dict[str, dict[str, Any]], str]:
    manifest = read_json(path)
    source = str(path)
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    manifest_hash = str(required(manifest, "source_config_sha256", source=source))
    if manifest_hash != config_hash:
        raise SummaryValidationError(
            f"{source}: config hash {manifest_hash} != current frozen config {config_hash}"
        )
    config = tomllib.loads(config_bytes.decode())
    design = required(config, "cosmos3_kv_factorial", source=str(config_path))
    if required(design, "status", source=str(config_path)) != "frozen_before_rescue_outcomes":
        raise SummaryValidationError(f"{config_path}: K/V design is not frozen")
    expected_tasks = {str(task) for task in required(design, "tasks", source=str(config_path))}
    expected_seeds = {
        int(seed) for seed in required(design, "environment_seeds", source=str(config_path))
    }
    expected: dict[str, dict[str, Any]] = {}
    for candidate in required(manifest, "candidates", source=source):
        if not candidate.get("selected", False):
            continue
        if str(candidate["task"]) not in expected_tasks or int(
            candidate["environment_seed"]
        ) not in expected_seeds:
            continue
        identifier = str(required(candidate, "unit_id", source=source))
        expected[identifier] = dict(candidate)
    expected_units = int(required(design, "units", source=str(config_path)))
    if len(expected) != expected_units:
        raise SummaryValidationError(
            f"{source}: K/V subset has {len(expected)} states, expected {expected_units}"
        )
    return expected, config_hash


def expected_cell_labels(source: str) -> dict[str, str]:
    return {
        "recipient_future_recipient_kv": "self_kv_record",
        "donor_future_recipient_kv": f"{source}_donor_kv_patch_all_action",
        "recipient_future_donor_kv": f"self_with_{source}_donor_kv",
        "donor_future_donor_kv": f"{source}_donor_kv_record",
    }


def _grouped_metric(
    intervention: dict[str, Any], field: str, group: str, *, source_path: str
) -> float | None:
    mapping = required(intervention, field, source=source_path)
    if group not in mapping:
        raise SummaryValidationError(f"{source_path}: missing {field}.{group}")
    return finite_or_none(mapping[group], field=f"{field}.{group}", source=source_path)


def _cell_row(
    intervention: dict[str, Any],
    *,
    group: str | None,
    base: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    if group is None:
        return {
            **base,
            "domain": "action",
            "normalized_projection": required_finite_or_none(
                intervention, "action_donor_projection", source=source_path
            ),
            "distance_reduction": required_finite_or_none(
                intervention, "distance_reduction_to_target", source=source_path
            ),
            "cosine_alignment": required_finite_or_none(
                intervention, "cosine_alignment", source=source_path
            ),
            "orthogonal_residual_normalized": required_finite_or_none(
                intervention, "orthogonal_residual_normalized", source=source_path
            ),
        }
    return {
        **base,
        "domain": f"endpoint_{group}",
        "normalized_projection": _grouped_metric(
            intervention, "endpoint_donor_projection", group, source_path=source_path
        ),
        "distance_reduction": _grouped_metric(
            intervention,
            "endpoint_distance_reduction_to_target",
            group,
            source_path=source_path,
        ),
        "cosine_alignment": _grouped_metric(
            intervention, "endpoint_cosine_alignment", group, source_path=source_path
        ),
        "orthogonal_residual_normalized": _grouped_metric(
            intervention,
            "endpoint_orthogonal_residual_normalized",
            group,
            source_path=source_path,
        ),
    }


def _assert_exact_replay(
    report: dict[str, Any], interventions: dict[str, Any], *, source_path: str
) -> list[dict[str, Any]]:
    def exact_equal(left: Any, right: Any) -> bool:
        if isinstance(left, dict) and isinstance(right, dict):
            return set(left) == set(right) and all(
                exact_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(
                exact_equal(a, b) for a, b in zip(left, right)
            )
        if left is None or right is None:
            return left is right
        try:
            if np.isnan(left) and np.isnan(right):
                return True
        except TypeError:
            pass
        return bool(left == right)

    errors = required(
        report, "kv_patch_identity_action_maximum_errors", source=source_path
    )
    audit_rows: list[dict[str, Any]] = []
    labels = ["self_kv_patch_all"]
    for source in SOURCES:
        labels.extend((f"{source}_donor_kv_record", f"{source}_donor_kv_replay"))
    for label in labels:
        if label not in errors:
            raise SummaryValidationError(
                f"{source_path}: exact replay audit is missing {label}"
            )
        error = float(errors[label])
        if error != 0.0:
            raise SummaryValidationError(
                f"{source_path}: exact replay audit {label} is not exact; action error {error}"
            )
        audit_rows.append({"label": label, "action_maximum_absolute_error": error})

    scalar_fields = (
        "action_donor_projection",
        "action_l2_to_target_donor",
        "action_native_target_l2",
        "distance_reduction_to_target",
        "cosine_alignment",
        "orthogonal_residual_normalized",
        "endpoint_donor_projection",
        "endpoint_l2_to_target_donor",
        "endpoint_native_target_l2",
        "endpoint_distance_reduction_to_target",
        "endpoint_cosine_alignment",
        "endpoint_orthogonal_residual_normalized",
    )
    for source in SOURCES:
        record_label = f"{source}_donor_kv_record"
        replay_label = f"{source}_donor_kv_replay"
        record = required(interventions, record_label, source=source_path)
        replay = required(interventions, replay_label, source=source_path)
        for field in scalar_fields:
            left = required(record, field, source=f"{source_path}:{record_label}")
            right = required(replay, field, source=f"{source_path}:{replay_label}")
            if not exact_equal(left, right):
                raise SummaryValidationError(
                    f"{source_path}: {record_label}/{replay_label} differ at {field}"
                )
    return audit_rows


def extract_rows(
    report: dict[str, Any], path: Path, expected: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_path = str(path)
    identifier = state_id(report, source=source_path)
    task = str(required(report, "task", source=source_path))
    environment_seed = int(required(report, "environment_seed", source=source_path))
    recipient_seed = int(required(report, "recipient_seed", source=source_path))
    donor_seed = int(required(report, "donor_seed", source=source_path))
    patch_layers = [
        int(layer)
        for layer in required(report, "attention_kv_patch_layers", source=source_path)
    ]
    if recipient_seed != 211 or donor_seed != 223:
        raise SummaryValidationError(
            f"{source_path}: frozen K/V pair must be recipient 211 and donor 223; "
            f"got {recipient_seed}->{donor_seed}"
        )
    if patch_layers != list(range(36)):
        raise SummaryValidationError(
            f"{source_path}: frozen K/V layer list must be exactly 0..35"
        )
    target_object_name = str(
        required(report, "target_object_name", source=source_path)
    ).strip()
    if not target_object_name:
        raise SummaryValidationError(
            f"{source_path}: frozen K/V endpoints require target_object_name"
        )
    expected_object_name = FROZEN_TASK_OBJECT_NAMES.get(task)
    if expected_object_name is None or target_object_name != expected_object_name:
        raise SummaryValidationError(
            f"{source_path}: target_object_name {target_object_name!r} does not match "
            f"the frozen task mapping {expected_object_name!r}"
        )
    if expected is not None:
        for field, actual in (
            ("unit_id", identifier),
            ("task", task),
            ("environment_seed", environment_seed),
            ("branch_step", int(required(report, "branch_step", source=source_path))),
            ("target_object_name", target_object_name),
            ("recipient_seed", recipient_seed),
        ):
            if expected[field] != actual:
                raise SummaryValidationError(
                    f"{source_path}: {field} differs from population manifest"
                )
        expected_donors = [int(seed) for seed in expected["donor_seeds"]]
        if not expected_donors or donor_seed != expected_donors[0]:
            raise SummaryValidationError(
                f"{source_path}: donor_seed differs from the first frozen manifest donor"
            )
    if not required_bool(report, "attention_kv_factorial", source=source_path):
        raise SummaryValidationError(f"{source_path}: K/V factorial flag is false")
    if not required_bool(report, "minimal_kv_factorial", source=source_path):
        raise SummaryValidationError(
            f"{source_path}: overnight K/V result is not the frozen minimal factorial"
        )
    interventions = required(report, "interventions", source=source_path)
    physically_executed = {
        str(label)
        for label in required(
            report, "physically_executed_intervention_labels", source=source_path
        )
    }
    action_only = {
        str(label)
        for label in required(
            report, "action_only_intervention_labels", source=source_path
        )
    }
    if physically_executed != MINIMAL_PHYSICAL_LABELS:
        raise SummaryValidationError(
            f"{source_path}: minimal K/V physical labels differ from frozen set; "
            f"got {sorted(physically_executed)}"
        )
    if action_only != MINIMAL_ACTION_ONLY_LABELS:
        raise SummaryValidationError(
            f"{source_path}: minimal K/V action-only labels differ from frozen set; "
            f"got {sorted(action_only)}"
        )
    if physically_executed & action_only:
        raise SummaryValidationError(
            f"{source_path}: intervention labels cannot be both physical and action-only"
        )
    if set(interventions) != physically_executed | action_only:
        raise SummaryValidationError(
            f"{source_path}: intervention labels do not match execution metadata"
        )
    for label, intervention in interventions.items():
        executed = required_bool(
            intervention,
            "physically_executed",
            source=f"{source_path}:interventions.{label}",
        )
        if executed != (label in physically_executed):
            raise SummaryValidationError(
                f"{source_path}: interventions.{label}.physically_executed conflicts "
                "with execution metadata"
            )
    audit_rows = _assert_exact_replay(report, interventions, source_path=source_path)
    for row in audit_rows:
        row.update(
            {
                "summary_path": source_path,
                "state_id": identifier,
                "task": task,
                "environment_seed": environment_seed,
            }
        )

    reported_cells = required(report, "attention_kv_factorial_cells", source=source_path)
    reported_endpoint_cells = required(
        report, "attention_kv_factorial_endpoint_cells", source=source_path
    )
    rows: list[dict[str, Any]] = []
    donor_seed = int(required(report, "donor_seed", source=source_path))
    for future_source in SOURCES:
        labels = expected_cell_labels(future_source)
        if required(reported_cells, future_source, source=source_path) != labels:
            raise SummaryValidationError(
                f"{source_path}: {future_source} factorial cell map is not canonical"
            )
        endpoint_labels = required(
            reported_endpoint_cells, future_source, source=source_path
        )
        if set(endpoint_labels) != set(CELLS):
            raise SummaryValidationError(
                f"{source_path}: {future_source} endpoint cell map is incomplete"
            )
        for cell, label in labels.items():
            action_intervention = required(interventions, label, source=source_path)
            intervention_path = f"{source_path}:interventions.{label}"
            target = int(
                required(action_intervention, "target_donor_seed", source=intervention_path)
            )
            if target != donor_seed:
                raise SummaryValidationError(
                    f"{intervention_path}: target donor {target} != {donor_seed}"
                )
            base = {
                "summary_path": source_path,
                "study_id": str(required(report, "study_id", source=source_path)),
                "state_id": identifier,
                "task": task,
                "environment_seed": environment_seed,
                "branch_step": int(required(report, "branch_step", source=source_path)),
                "future_source": future_source,
                "cell": cell,
                "intervention_label": label,
                "target_donor_seed": donor_seed,
                "target_object_name": target_object_name,
            }
            rows.append(
                _cell_row(
                    action_intervention,
                    group=None,
                    base=base,
                    source_path=intervention_path,
                )
            )
            endpoint_label = str(endpoint_labels[cell])
            endpoint_intervention = required(
                interventions, endpoint_label, source=source_path
            )
            endpoint_path = f"{source_path}:interventions.{endpoint_label}"
            endpoint_target = int(
                required(
                    endpoint_intervention, "target_donor_seed", source=endpoint_path
                )
            )
            if endpoint_target != donor_seed:
                raise SummaryValidationError(
                    f"{endpoint_path}: endpoint target donor {endpoint_target} != {donor_seed}"
                )
            if not required_bool(
                endpoint_intervention, "physically_executed", source=endpoint_path
            ):
                raise SummaryValidationError(
                    f"{endpoint_path}: endpoint cell was not physically executed"
                )
            endpoint_base = {**base, "intervention_label": endpoint_label}
            for group in ("all", "robot", "object", "target_object_position"):
                rows.append(
                    _cell_row(
                        endpoint_intervention,
                        group=group,
                        base=endpoint_base,
                        source_path=endpoint_path,
                    )
                )
    return rows, audit_rows


def build_contrast_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contrast_rows: list[dict[str, Any]] = []
    state_keys = sorted(
        {
            (str(row["state_id"]), str(row["future_source"]), str(row["domain"]))
            for row in cell_rows
        }
    )
    for identifier, future_source, domain in state_keys:
        subset = [
            row
            for row in cell_rows
            if row["state_id"] == identifier
            and row["future_source"] == future_source
            and row["domain"] == domain
        ]
        by_cell = {str(row["cell"]): row for row in subset}
        missing = set(CELLS) - set(by_cell)
        if missing:
            raise SummaryValidationError(
                f"{identifier}/{future_source}/{domain}: missing cells {sorted(missing)}"
            )
        for metric in METRICS:
            values = {cell: by_cell[cell][metric] for cell in CELLS}
            if any(value is None for value in values.values()):
                contrasts = {contrast: None for contrast in CONTRASTS}
            else:
                aa = float(values["recipient_future_recipient_kv"])
                ba = float(values["donor_future_recipient_kv"])
                ab = float(values["recipient_future_donor_kv"])
                bb = float(values["donor_future_donor_kv"])
                contrasts = {
                    "suppression": bb - ba,
                    "rescue": ab - aa,
                    "interaction": bb - ba - ab + aa,
                    "future_effect_with_recipient_kv": ba - aa,
                    "future_effect_with_donor_kv": bb - ab,
                }
            for contrast, value in contrasts.items():
                contrast_rows.append(
                    {
                        "state_id": identifier,
                        "task": by_cell[CELLS[0]]["task"],
                        "environment_seed": by_cell[CELLS[0]]["environment_seed"],
                        "future_source": future_source,
                        "domain": domain,
                        "metric": metric,
                        "contrast": contrast,
                        "value": value,
                    }
                )
    return contrast_rows


def aggregate(
    cell_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    counter = 0
    for future_source in SOURCES:
        cells[future_source] = {}
        contrasts[future_source] = {}
        for domain in DOMAINS:
            cells[future_source][domain] = {}
            contrasts[future_source][domain] = {}
            domain_cells = [
                row
                for row in cell_rows
                if row["future_source"] == future_source and row["domain"] == domain
            ]
            domain_contrasts = [
                row
                for row in contrast_rows
                if row["future_source"] == future_source and row["domain"] == domain
            ]
            for metric in METRICS:
                cells[future_source][domain][metric] = {}
                contrasts[future_source][domain][metric] = {}
                for cell in CELLS:
                    counter += 1
                    subset = [row for row in domain_cells if row["cell"] == cell]
                    estimate = summarize_metric(
                        subset,
                        metric,
                        resamples=resamples,
                        seed=seed + counter,
                    )
                    estimate["leave_one_task_out"] = leave_one_task_out(subset, metric)
                    cells[future_source][domain][metric][cell] = estimate
                for contrast in CONTRASTS:
                    counter += 1
                    subset = [
                        row
                        for row in domain_contrasts
                        if row["metric"] == metric and row["contrast"] == contrast
                    ]
                    estimate = summarize_metric(
                        subset,
                        "value",
                        resamples=resamples,
                        seed=seed + counter,
                    )
                    estimate["leave_one_task_out"] = leave_one_task_out(subset, "value")
                    contrasts[future_source][domain][metric][contrast] = estimate
    return {"cell_estimates": cells, "contrast_estimates": contrasts}


def render_latex(summary: dict[str, Any]) -> str:
    domain_labels = {
        "action": "Action",
        "endpoint_all": "Full endpoint",
        "endpoint_robot": "Robot endpoint",
        "endpoint_object": "All-object state",
        "endpoint_target_object_position": "Task-object position",
    }
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Future source & Outcome & Suppression & Rescue & Interaction \\",
        r"\midrule",
    ]
    for future_source in SOURCES:
        for domain in DOMAINS:
            estimates = summary["contrast_estimates"][future_source][domain][
                "normalized_projection"
            ]
            lines.append(
                " & ".join(
                    (
                        latex_escape(future_source.title()),
                        domain_labels[domain],
                        format_estimate(estimates["suppression"]),
                        format_estimate(estimates["rescue"]),
                        format_estimate(estimates["interaction"]),
                    )
                )
                + r" \\"
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _mean_interval(estimate: dict[str, Any]) -> tuple[float, float, float]:
    if estimate["mean"] is None or estimate["ci95"] is None:
        return float("nan"), float("nan"), float("nan")
    mean = float(estimate["mean"])
    return mean, mean - float(estimate["ci95"][0]), float(estimate["ci95"][1]) - mean


def render_plots(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "recipient_future_recipient_kv": "#B8B8B8",
        "donor_future_recipient_kv": "#E45756",
        "recipient_future_donor_kv": "#54A24B",
        "donor_future_donor_kv": "#4C78A8",
    }
    figure, axes = plt.subplots(
        2,
        len(DOMAINS),
        figsize=(14.5, 5.8),
        sharey="row",
        constrained_layout=True,
    )
    for row_index, future_source in enumerate(SOURCES):
        for column, domain in enumerate(DOMAINS):
            axis = axes[row_index, column]
            estimates = summary["cell_estimates"][future_source][domain]["normalized_projection"]
            for position, cell in enumerate(CELLS):
                mean, lower, upper = _mean_interval(estimates[cell])
                axis.errorbar(
                    position,
                    mean,
                    yerr=np.asarray([[lower], [upper]]),
                    fmt="o",
                    capsize=3,
                    color=colors[cell],
                )
            axis.axhline(0.0, color="0.75", linewidth=0.8)
            axis.set_xticks(range(4), ("AA", "BA", "AB", "BB"))
            axis.set_title(f"{future_source.title()} · {domain.replace('_', ' ')}")
    axes[0, 0].set_ylabel("Action donor projection")
    axes[1, 0].set_ylabel("Donor projection")
    save_figure(figure, output_dir / "kv_factorial_cells")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharey=True, constrained_layout=True)
    for axis, future_source in zip(axes, SOURCES):
        x = np.arange(len(DOMAINS), dtype=float)
        for offset, contrast, color in (
            (-0.18, "suppression", "#E45756"),
            (0.0, "rescue", "#54A24B"),
            (0.18, "interaction", "#4C78A8"),
        ):
            estimates = [
                summary["contrast_estimates"][future_source][domain][
                    "normalized_projection"
                ][contrast]
                for domain in DOMAINS
            ]
            triples = [_mean_interval(estimate) for estimate in estimates]
            means = np.asarray([item[0] for item in triples])
            errors = np.asarray([[item[1] for item in triples], [item[2] for item in triples]])
            axis.errorbar(
                x + offset,
                means,
                yerr=errors,
                fmt="o",
                capsize=3,
                color=color,
                label=contrast.title(),
            )
        axis.axhline(0.0, color="0.65", linestyle="--", linewidth=0.9)
        axis.set_xticks(
            x,
            ("Action", "All end.", "Robot end.", "All objects", "Task object"),
            rotation=18,
            ha="right",
        )
        axis.set_title(future_source.title())
    axes[0].set_ylabel("Projection contrast")
    axes[1].legend(frameon=False)
    save_figure(figure, output_dir / "kv_factorial_contrasts")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    expected, config_hash = load_expected_manifest(args.manifest, args.config)
    paths = discover_summaries(args.inputs)
    reports: dict[str, tuple[dict[str, Any], Path]] = {}
    skipped: list[dict[str, str]] = []
    for path in paths:
        report = read_json(path)
        if not report.get("attention_kv_factorial", False):
            skipped.append({"path": str(path), "reason": "not_kv_factorial"})
            continue
        identifier = state_id(report, source=str(path))
        if identifier not in expected:
            raise SummaryValidationError(f"{path}: state {identifier} is not in manifest")
        if identifier in reports:
            raise SummaryValidationError(f"duplicate completed K/V summary for {identifier}")
        reports[identifier] = (report, path)
    if not reports:
        raise SummaryValidationError("no eligible completed K/V factorial summaries")

    cell_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for identifier, (report, path) in sorted(reports.items()):
        cells, audits = extract_rows(
            report, path, expected[identifier]
        )
        cell_rows.extend(cells)
        audit_rows.extend(audits)
    contrast_rows = build_contrast_rows(cell_rows)
    aggregation = aggregate(
        cell_rows,
        contrast_rows,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    missing = sorted(set(expected) - set(reports))
    summary = {
        "scope": "Cosmos 3 future-target by future-K/V factorial",
        "manifest": str(args.manifest),
        "config": str(args.config),
        "config_sha256": config_hash,
        "expected_states": len(expected),
        "completed_states": len(reports),
        "completed_state_ids": sorted(reports),
        "missing_state_ids": missing,
        "partial": bool(missing),
        "discovered_summary_files": len(paths),
        "skipped_summaries": skipped,
        "exact_replay_audits_passed": True,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        **aggregation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "kv_factorial_summary.json", summary)
    write_csv(args.output_dir / "kv_factorial_cell_rows.csv", cell_rows)
    write_csv(args.output_dir / "kv_factorial_contrast_rows.csv", contrast_rows)
    write_csv(args.output_dir / "kv_factorial_replay_audit.csv", audit_rows)
    atomic_write_text(args.output_dir / "kv_factorial_table.tex", render_latex(summary))
    if not args.no_plots:
        render_plots(summary, args.output_dir)
    print(
        json.dumps(
            {
                "completed_states": len(reports),
                "expected_states": len(expected),
                "partial": bool(missing),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
