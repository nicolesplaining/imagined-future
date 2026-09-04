#!/usr/bin/env python3
"""Aggregate the frozen selection-free four-way Cosmos 3 donor study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from imagined_future.cosmos3_protocol import FROZEN_TASK_OBJECT_NAMES

from overnight_summary_common import (
    SummaryValidationError,
    collapse_to_states,
    discover_summaries,
    exact_binomial_greater,
    finite_or_none,
    format_estimate,
    hierarchical_task_state_bootstrap,
    latex_escape,
    leave_one_task_out,
    read_json,
    required,
    required_bool,
    required_finite_or_none,
    save_figure,
    separation_quartiles,
    state_id,
    summarize_metric,
    write_csv,
    write_json,
    atomic_write_text,
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
    "top1",
    "distance_reduction",
    "cosine_alignment",
    "orthogonal_residual_normalized",
    "normalized_projection",
    "l2_to_target",
)


def load_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], float, dict[str, Any]]:
    manifest = read_json(path)
    source = str(path)
    if required(manifest, "status", source=source) != "frozen_before_outcomes":
        raise SummaryValidationError(f"{source}: manifest is not frozen before outcomes")
    if required(manifest, "selection_uses_native_or_intervention_outcomes", source=source):
        raise SummaryValidationError(f"{source}: selection-free manifest uses outcomes")
    chance = float(required(manifest, "primary_chance_rate", source=source))
    if chance != 0.25:
        raise SummaryValidationError(f"{source}: four-way chance rate must be 0.25")
    expected: dict[str, dict[str, Any]] = {}
    for candidate in required(manifest, "candidates", source=source):
        if not candidate.get("selected", False):
            continue
        identifier = str(required(candidate, "unit_id", source=source))
        if identifier in expected:
            raise SummaryValidationError(f"{source}: duplicate manifest unit {identifier}")
        expected[identifier] = dict(candidate)
    if not expected:
        raise SummaryValidationError(f"{source}: manifest has no selected states")
    return expected, chance, manifest


def _donor_label(source: str, target_seed: int, primary_seed: int) -> str:
    return (
        f"{source}_donor"
        if target_seed == primary_seed
        else f"{source}_donor_seed_{target_seed}"
    )


def _action_row(
    intervention: dict[str, Any], *, base: dict[str, Any], source_path: str
) -> dict[str, Any]:
    return {
        **base,
        "domain": "action",
        "nearest_native_seed": int(
            required(intervention, "nearest_native_action_seed", source=source_path)
        ),
        "top1": float(required_bool(intervention, "correct_action_donor_top1", source=source_path)),
        "normalized_projection": required_finite_or_none(
            intervention, "action_target_donor_projection", source=source_path
        ),
        "l2_to_target": required_finite_or_none(
            intervention, "action_l2_to_target_donor", source=source_path
        ),
        "native_target_l2": required_finite_or_none(
            intervention, "action_native_target_l2", source=source_path
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


def _endpoint_row(
    intervention: dict[str, Any],
    *,
    group: str,
    base: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    top1 = required(intervention, "correct_endpoint_donor_top1", source=source_path)
    nearest = required(intervention, "nearest_native_endpoint_seed", source=source_path)
    if group not in top1 or group not in nearest:
        raise SummaryValidationError(f"{source_path}: endpoint group {group!r} is missing")
    if not isinstance(top1[group], bool):
        raise SummaryValidationError(f"{source_path}: endpoint top-1 {group} must be Boolean")

    def grouped(field: str) -> float | None:
        mapping = required(intervention, field, source=source_path)
        if group not in mapping:
            raise SummaryValidationError(f"{source_path}: {field}.{group} is missing")
        return finite_or_none(mapping[group], field=f"{field}.{group}", source=source_path)

    return {
        **base,
        "domain": f"endpoint_{group}",
        "nearest_native_seed": int(nearest[group]),
        "top1": float(top1[group]),
        "normalized_projection": grouped("endpoint_target_donor_projection"),
        "l2_to_target": grouped("endpoint_l2_to_target_donor"),
        "native_target_l2": grouped("endpoint_native_target_l2"),
        "distance_reduction": grouped("endpoint_distance_reduction_to_target"),
        "cosine_alignment": grouped("endpoint_cosine_alignment"),
        "orthogonal_residual_normalized": grouped(
            "endpoint_orthogonal_residual_normalized"
        ),
    }


def extract_rows(
    report: dict[str, Any], path: Path, expected: dict[str, Any]
) -> list[dict[str, Any]]:
    source_path = str(path)
    identifier = state_id(report, source=source_path)
    task = str(required(report, "task", source=source_path))
    environment_seed = int(required(report, "environment_seed", source=source_path))
    branch_step = int(required(report, "branch_step", source=source_path))
    branch_seeds = [int(seed) for seed in required(report, "branch_seeds", source=source_path)]
    recipient_seed = int(required(report, "recipient_seed", source=source_path))
    primary_seed = int(required(report, "donor_seed", source=source_path))
    target_object_name = str(
        required(report, "target_object_name", source=source_path)
    ).strip()
    if not target_object_name:
        raise SummaryValidationError(
            f"{source_path}: frozen endpoint analysis requires target_object_name"
        )
    expected_object_name = FROZEN_TASK_OBJECT_NAMES.get(task)
    if expected_object_name is None or target_object_name != expected_object_name:
        raise SummaryValidationError(
            f"{source_path}: target_object_name {target_object_name!r} does not match "
            f"the frozen task mapping {expected_object_name!r}"
        )
    if identifier != str(expected["unit_id"]):
        raise SummaryValidationError(f"{source_path}: state ID differs from frozen manifest")
    for field, actual in (
        ("task", task),
        ("environment_seed", environment_seed),
        ("branch_step", branch_step),
        ("recipient_seed", recipient_seed),
        ("target_object_name", target_object_name),
    ):
        if actual != expected[field]:
            raise SummaryValidationError(
                f"{source_path}: {field}={actual!r} differs from manifest {expected[field]!r}"
            )
    expected_seeds = [int(seed) for seed in expected["branch_seeds"]]
    expected_donors = [int(seed) for seed in expected["donor_seeds"]]
    if branch_seeds != expected_seeds or len(branch_seeds) != 4:
        raise SummaryValidationError(f"{source_path}: branch seeds differ from frozen four-way set")
    if primary_seed != expected_donors[0]:
        raise SummaryValidationError(f"{source_path}: primary donor must be first frozen donor")
    if not required_bool(report, "multi_donor", source=source_path):
        raise SummaryValidationError(f"{source_path}: not a multi-donor run")
    if not required_bool(report, "frozen_pair_supplied", source=source_path):
        raise SummaryValidationError(f"{source_path}: pair was not frozen")
    if required_bool(report, "within_run_pair_selection", source=source_path):
        raise SummaryValidationError(f"{source_path}: pair was selected within the run")
    executed = [int(seed) for seed in required(report, "native_execution_seeds", source=source_path)]
    if sorted(executed) != sorted(branch_seeds):
        raise SummaryValidationError(f"{source_path}: not all four native branches were executed")
    reported_targets = [
        int(seed) for seed in required(report, "multi_donor_target_seeds", source=source_path)
    ]
    if reported_targets != expected_donors:
        raise SummaryValidationError(f"{source_path}: donor targets differ from frozen manifest")

    if not required_bool(report, "all_recipient_action_grid", source=source_path):
        raise SummaryValidationError(f"{source_path}: all-recipient action grid is missing")
    action_grid = required(report, "action_grid", source=source_path)
    if [int(seed) for seed in required(action_grid, "candidate_seeds", source=source_path)] != branch_seeds:
        raise SummaryValidationError(f"{source_path}: action-grid candidate seeds differ")
    expected_pairs = [
        (candidate_recipient, candidate_donor)
        for candidate_recipient in branch_seeds
        for candidate_donor in branch_seeds
        if candidate_recipient != candidate_donor
    ]
    manifest_pairs = [
        tuple(int(seed) for seed in pair)
        for pair in required(expected, "action_ordered_pairs", source=source_path)
    ]
    if manifest_pairs != expected_pairs:
        raise SummaryValidationError(
            f"{source_path}: manifest action pairs are not the canonical 12-pair grid"
        )
    reported_pairs = [
        tuple(int(seed) for seed in pair)
        for pair in required(action_grid, "ordered_pairs", source=source_path)
    ]
    if reported_pairs != expected_pairs or int(
        required(action_grid, "ordered_pair_count", source=source_path)
    ) != 12:
        raise SummaryValidationError(f"{source_path}: action grid is not the frozen 12-pair grid")
    if int(required(action_grid, "intervention_count", source=source_path)) != 24:
        raise SummaryValidationError(f"{source_path}: action grid must contain 24 interventions")
    if list(required(action_grid, "future_sources", source=source_path)) != list(SOURCES):
        raise SummaryValidationError(f"{source_path}: action-grid future sources differ")
    for audit_field in (
        "native_repeat_action_maximum_error",
        "clean_self_clamp_repeat_action_maximum_error",
    ):
        audit = required(action_grid, audit_field, source=source_path)
        if set(audit) != {str(seed) for seed in branch_seeds}:
            raise SummaryValidationError(f"{source_path}: {audit_field} has wrong seeds")
        for seed, error in audit.items():
            if float(error) != 0.0:
                raise SummaryValidationError(
                    f"{source_path}: {audit_field}[{seed}] is not exact: {error}"
                )
    for audit_field in ("native_repeat_exact", "clean_self_clamp_repeat_exact"):
        audit = required(action_grid, audit_field, source=source_path)
        if set(audit) != {str(seed) for seed in branch_seeds} or not all(
            value is True for value in audit.values()
        ):
            raise SummaryValidationError(f"{source_path}: {audit_field} did not pass")
    clean_self_error = required(
        action_grid, "clean_self_clamp_error_from_native", source=source_path
    )
    if set(clean_self_error) != {str(seed) for seed in branch_seeds}:
        raise SummaryValidationError(
            f"{source_path}: clean-self-clamp perturbation audit has wrong seeds"
        )
    for seed, errors in clean_self_error.items():
        for metric in ("maximum_absolute", "l2"):
            value = float(required(errors, metric, source=f"{source_path}:clean-self-{seed}"))
            if not np.isfinite(value) or value < 0.0:
                raise SummaryValidationError(
                    f"{source_path}: clean-self-clamp {seed} {metric} is invalid"
                )

    rows: list[dict[str, Any]] = []
    seen_grid: set[tuple[str, int, int]] = set()
    for index, intervention in enumerate(required(action_grid, "rows", source=source_path)):
        intervention_path = f"{source_path}:action_grid.rows[{index}]"
        future_source = str(required(intervention, "future_source", source=intervention_path))
        candidate_recipient = int(
            required(intervention, "recipient_seed", source=intervention_path)
        )
        target_seed = int(
            required(intervention, "target_donor_seed", source=intervention_path)
        )
        key = (future_source, candidate_recipient, target_seed)
        if future_source not in SOURCES or (candidate_recipient, target_seed) not in expected_pairs:
            raise SummaryValidationError(f"{intervention_path}: unexpected directed pair")
        if key in seen_grid:
            raise SummaryValidationError(f"{intervention_path}: duplicate action-grid row {key}")
        if required_bool(intervention, "physically_executed", source=intervention_path):
            raise SummaryValidationError(
                f"{intervention_path}: action-grid arm must not be physically executed"
            )
        seen_grid.add(key)
        base = {
            "summary_path": source_path,
            "study_id": str(required(report, "study_id", source=source_path)),
            "state_id": identifier,
            "task": task,
            "environment_seed": environment_seed,
            "branch_step": branch_step,
            "recipient_seed": candidate_recipient,
            "target_donor_seed": target_seed,
            "target_object_name": target_object_name,
            "future_source": future_source,
            "intervention_label": f"action_grid_{future_source}_{candidate_recipient}_{target_seed}",
        }
        rows.append(_action_row(intervention, base=base, source_path=intervention_path))
    expected_grid = {
        (future_source, candidate_recipient, target_seed)
        for future_source in SOURCES
        for candidate_recipient, target_seed in expected_pairs
    }
    if seen_grid != expected_grid:
        raise SummaryValidationError(
            f"{source_path}: action grid has {len(seen_grid)} rows, expected {len(expected_grid)}"
        )

    interventions = required(report, "interventions", source=source_path)
    for future_source in SOURCES:
        for target_seed in expected_donors:
            label = _donor_label(future_source, target_seed, primary_seed)
            intervention = required(interventions, label, source=source_path)
            intervention_path = f"{source_path}:interventions.{label}"
            actual_target = int(
                required(intervention, "target_donor_seed", source=intervention_path)
            )
            if actual_target != target_seed:
                raise SummaryValidationError(
                    f"{intervention_path}: target {actual_target} != {target_seed}"
                )
            if not required_bool(
                intervention, "physically_executed", source=intervention_path
            ):
                raise SummaryValidationError(
                    f"{intervention_path}: fixed-recipient endpoint arm was not executed"
                )
            base = {
                "summary_path": source_path,
                "study_id": str(required(report, "study_id", source=source_path)),
                "state_id": identifier,
                "task": task,
                "environment_seed": environment_seed,
                "branch_step": branch_step,
                "recipient_seed": recipient_seed,
                "target_donor_seed": target_seed,
                "target_object_name": target_object_name,
                "future_source": future_source,
                "intervention_label": label,
            }
            for group in ("all", "robot", "object", "target_object_position"):
                rows.append(
                    _endpoint_row(
                        intervention,
                        group=group,
                        base=base,
                        source_path=intervention_path,
                    )
                )
    return rows


def aggregate(
    rows: list[dict[str, Any]], *, chance: float, resamples: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    estimates: dict[str, Any] = {}
    state_rows_long: list[dict[str, Any]] = []
    counter = 0
    for future_source in SOURCES:
        estimates[future_source] = {}
        for domain in DOMAINS:
            subset = [
                row
                for row in rows
                if row["future_source"] == future_source and row["domain"] == domain
            ]
            domain_summary: dict[str, Any] = {}
            for metric in METRICS:
                counter += 1
                estimate = summarize_metric(
                    subset,
                    metric,
                    resamples=resamples,
                    seed=seed + counter,
                )
                estimate["leave_one_task_out"] = leave_one_task_out(subset, metric)
                if metric == "top1":
                    successes = int(sum(float(row[metric]) for row in subset))
                    trials = len(subset)
                    estimate.update(
                        {
                            "successes": successes,
                            "trials": trials,
                            "chance_rate": chance,
                            "exact_binomial_greater_p": exact_binomial_greater(
                                successes, trials, chance
                            ),
                            "binomial_note": (
                                "descriptive donor-level exact test; hierarchical confidence "
                                "interval treats saved state as the independent unit"
                            ),
                        }
                    )
                domain_summary[metric] = estimate
                for state_row in collapse_to_states(subset, metric):
                    state_rows_long.append(
                        {
                            "future_source": future_source,
                            "domain": domain,
                            "metric": metric,
                            **state_row,
                        }
                    )
            estimates[future_source][domain] = domain_summary

    quartiles: dict[str, Any] = {}
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["domain"] == domain]
        boundaries, assignments = separation_quartiles(domain_rows)
        quartiles[domain] = {"boundaries": boundaries, "sources": {}}
        for row in domain_rows:
            row["separation_quartile"] = assignments.get(
                (
                    str(row["state_id"]),
                    int(row["recipient_seed"]),
                    int(row["target_donor_seed"]),
                )
            )
        for future_source in SOURCES:
            quartiles[domain]["sources"][future_source] = {}
            for quartile in range(1, 5):
                subset = [
                    row
                    for row in domain_rows
                    if row["future_source"] == future_source
                    and row.get("separation_quartile") == quartile
                ]
                quartiles[domain]["sources"][future_source][str(quartile)] = {
                    metric: summarize_metric(
                        subset,
                        metric,
                        resamples=resamples,
                        seed=seed + 1000 + 100 * DOMAINS.index(domain) + 10 * quartile + METRICS.index(metric),
                    )
                    for metric in METRICS
                }
    return {"estimates": estimates, "separation_quartiles": quartiles}, state_rows_long


def render_latex(summary: dict[str, Any]) -> str:
    labels = {
        "action": "Action",
        "endpoint_all": "Full endpoint",
        "endpoint_robot": "Robot endpoint",
        "endpoint_object": "All-object state",
        "endpoint_target_object_position": "Task-object position",
    }
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Source & Outcome & Top-1 (\%) & Dist. reduction & Projection & Orth. residual \\",
        r"\midrule",
    ]
    for source in SOURCES:
        for domain in DOMAINS:
            values = summary["estimates"][source][domain]
            lines.append(
                " & ".join(
                    (
                        latex_escape(source.title()),
                        labels[domain],
                        format_estimate(values["top1"], percent=True),
                        format_estimate(values["distance_reduction"]),
                        format_estimate(values["normalized_projection"]),
                        format_estimate(values["orthogonal_residual_normalized"]),
                    )
                )
                + r" \\"
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def render_plots(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domain_labels = [
        "Action",
        "All end.",
        "Robot end.",
        "All objects",
        "Task object",
    ]
    colors = {"predicted": "#4C78A8", "executed": "#F58518"}

    def mean_interval(estimate: dict[str, Any]) -> tuple[float, float, float]:
        if estimate["mean"] is None or estimate["ci95"] is None:
            return float("nan"), float("nan"), float("nan")
        mean = float(estimate["mean"])
        return (
            mean,
            mean - float(estimate["ci95"][0]),
            float(estimate["ci95"][1]) - mean,
        )

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("top1", "distance_reduction"),
        ("Four-way correct donor", "Distance reduction to donor"),
    ):
        x = np.arange(len(DOMAINS), dtype=float)
        for offset, source in zip((-0.12, 0.12), SOURCES):
            estimates = [summary["estimates"][source][domain][metric] for domain in DOMAINS]
            triples = [mean_interval(estimate) for estimate in estimates]
            means = np.asarray([item[0] for item in triples], dtype=float)
            errors = np.asarray(
                [[item[1] for item in triples], [item[2] for item in triples]],
                dtype=float,
            )
            axis.errorbar(
                x + offset,
                means,
                yerr=errors,
                fmt="o",
                capsize=3,
                color=colors[source],
                label=source.title(),
            )
        if metric == "top1":
            axis.axhline(summary["chance_rate"], color="0.45", linestyle="--", linewidth=1)
            axis.set_ylim(-0.03, 1.03)
        axis.axhline(0.0, color="0.75", linewidth=0.8)
        axis.set_xticks(x, domain_labels, rotation=18, ha="right")
        axis.set_title(title)
    axes[0].legend(frameon=False)
    save_figure(figure, output_dir / "selection_free_metrics")
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        len(DOMAINS),
        figsize=(13.2, 3.0),
        sharey=True,
        constrained_layout=True,
    )
    for axis, domain, label in zip(axes, DOMAINS, domain_labels):
        for source in SOURCES:
            means = [
                summary["separation_quartiles"][domain]["sources"][source][str(quartile)][
                    "distance_reduction"
                ]["mean"]
                for quartile in range(1, 5)
            ]
            axis.plot(
                range(1, 5),
                [float("nan") if mean is None else float(mean) for mean in means],
                marker="o",
                color=colors[source],
                label=source.title(),
            )
        axis.axhline(0.0, color="0.75", linewidth=0.8)
        axis.set_xticks(range(1, 5))
        axis.set_xlabel("Native separation quartile")
        axis.set_title(label)
    axes[0].set_ylabel("Distance reduction")
    axes[-1].legend(frameon=False)
    save_figure(figure, output_dir / "selection_free_separation_quartiles")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    expected, chance, manifest = load_manifest(args.manifest)
    paths = discover_summaries(args.inputs)
    reports: dict[str, tuple[dict[str, Any], Path]] = {}
    skipped: list[dict[str, str]] = []
    for path in paths:
        report = read_json(path)
        if not report.get("multi_donor", False):
            skipped.append({"path": str(path), "reason": "not_multi_donor"})
            continue
        identifier = state_id(report, source=str(path))
        if identifier not in expected:
            raise SummaryValidationError(f"{path}: state {identifier} is not in frozen manifest")
        if identifier in reports:
            raise SummaryValidationError(f"duplicate completed summary for {identifier}")
        reports[identifier] = (report, path)
    if not reports:
        raise SummaryValidationError("no eligible completed multi-donor summaries")

    rows: list[dict[str, Any]] = []
    for identifier, (report, path) in sorted(reports.items()):
        rows.extend(extract_rows(report, path, expected[identifier]))
    aggregation, state_rows = aggregate(
        rows,
        chance=chance,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    completed = sorted(reports)
    missing = sorted(set(expected) - set(reports))
    summary = {
        "scope": "frozen selection-free four-way Cosmos 3 donor evaluation",
        "manifest": str(args.manifest),
        "manifest_study_id": manifest["study_id"],
        "chance_rate": chance,
        "expected_states": len(expected),
        "completed_states": len(completed),
        "completed_state_ids": completed,
        "missing_state_ids": missing,
        "partial": bool(missing),
        "discovered_summary_files": len(paths),
        "skipped_summaries": skipped,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        **aggregation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "selection_free_summary.json", summary)
    write_csv(args.output_dir / "selection_free_donor_rows.csv", rows)
    write_csv(args.output_dir / "selection_free_state_rows.csv", state_rows)
    atomic_write_text(args.output_dir / "selection_free_table.tex", render_latex(summary))
    if not args.no_plots:
        render_plots(summary, args.output_dir)
    print(
        json.dumps(
            {
                "completed_states": len(completed),
                "expected_states": len(expected),
                "partial": bool(missing),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
