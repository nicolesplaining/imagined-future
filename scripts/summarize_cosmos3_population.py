#!/usr/bin/env python3
"""Aggregate frozen Cosmos 3 population and robot/object-factorial results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from imagined_future.statistics import cluster_bootstrap_mean, exact_sign_test, holm_adjust


def unit_id(report: dict[str, Any]) -> str:
    return f"{report['task']}-seed-{int(report['environment_seed'])}"


def effect_summary(values: dict[str, float], *, seed: int) -> dict[str, Any]:
    ordered = {key: float(values[key]) for key in sorted(values)}
    return {
        **cluster_bootstrap_mean(ordered, resamples=10_000, seed=seed),
        "median": float(np.median(list(ordered.values()))),
        "positive_fraction": float(np.mean(np.asarray(list(ordered.values())) > 0.0)),
        "sign_test": exact_sign_test(list(ordered.values())),
        "unit_effects": ordered,
    }


def contrast(
    reports: dict[str, dict[str, Any]],
    left: str,
    right: str,
    field: str,
    *,
    endpoint_group: str | None = None,
) -> dict[str, float]:
    output = {}
    for key, report in reports.items():
        interventions = report["interventions"]
        if left not in interventions or right not in interventions:
            continue
        if endpoint_group is None:
            left_value = interventions[left][field]
            right_value = interventions[right][field]
        else:
            left_value = interventions[left][field][endpoint_group]
            right_value = interventions[right][field][endpoint_group]
        output[key] = float(left_value) - float(right_value)
    return output


def task_means(values: dict[str, float], manifest_rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for key, value in values.items():
        grouped[str(manifest_rows[key]["task"])].append(float(value))
    return {task: float(np.mean(items)) for task, items in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite population summary: {args.output}")

    manifest = json.loads(args.manifest.read_text())
    manifest_rows = {
        str(row["unit_id"]): row
        for row in manifest["candidates"]
        if row.get("selected", False)
    }
    reports: dict[str, dict[str, Any]] = {}
    for path in args.summaries:
        report = json.loads(path.read_text())
        key = unit_id(report)
        if key in reports:
            raise ValueError(f"duplicate result summary for {key}")
        if key not in manifest_rows:
            raise ValueError(f"result was not selected in the frozen manifest: {key}")
        frozen = manifest_rows[key]
        if (int(report["recipient_seed"]), int(report["donor_seed"])) != (
            int(frozen["recipient_seed"]),
            int(frozen["donor_seed"]),
        ):
            raise ValueError(f"result changed the frozen donor pair for {key}")
        if not report.get("frozen_pair_supplied", False):
            raise ValueError(f"result did not use the frozen pair interface for {key}")
        if int(report["natural_control_seed"]) != int(frozen["natural_control_seed"]):
            raise ValueError(f"result changed the frozen natural control for {key}")
        if float(report["prefix_maximum_state_error"]) != 0.0:
            raise ValueError(f"result failed exact prefix replay for {key}")
        if float(report["self_identity_action_maximum_error"]) != 0.0:
            raise ValueError(f"result failed exact native self recomputation for {key}")
        if not report["continuation_repeat_audit"]["exact"]:
            raise ValueError(f"result failed exact continuation repeat for {key}")
        if len(set(report["restore_state_digests"])) != 1:
            raise ValueError(f"result failed exact branch-state restoration for {key}")
        if any(float(value) != 0.0 for value in report["kv_patch_identity_action_maximum_errors"].values()):
            raise ValueError(f"result failed K/V identity recomputation for {key}")
        gaussian_server = report["interventions"]["gaussian_executed"]["server"]
        for metric in (
            "research_gaussian_norm_relative_error",
            "research_gaussian_distance_relative_error",
        ):
            if float(gaussian_server[metric]) > 1e-5:
                raise ValueError(f"result failed Gaussian matching for {key}: {metric}")
        reports[key] = report

    expected = set(manifest_rows)
    observed = set(reports)
    missing = sorted(expected - observed)

    definitions = {
        "predicted_donor_minus_self_action": (
            "predicted_donor",
            "self",
            "action_donor_projection",
            None,
        ),
        "executed_donor_minus_self_action": (
            "executed_donor",
            "self",
            "action_donor_projection",
            None,
        ),
        "executed_donor_minus_gaussian_action": (
            "executed_donor",
            "gaussian_executed",
            "action_donor_projection",
            None,
        ),
        "predicted_donor_minus_natural_control_action": (
            "predicted_donor",
            "natural_control",
            "action_donor_projection",
            None,
        ),
        "executed_donor_minus_natural_control_action": (
            "executed_donor",
            "natural_control",
            "action_donor_projection",
            None,
        ),
        "executed_donor_minus_executed_self_action": (
            "executed_donor",
            "executed_self",
            "action_donor_projection",
            None,
        ),
        "executed_donor_minus_executed_self_physical": (
            "executed_donor",
            "executed_self",
            "endpoint_donor_projection",
            "all",
        ),
        "predicted_future_kv_mediation_action": (
            "predicted_donor",
            "predicted_donor_kv_patch_all_action",
            "action_donor_projection",
            None,
        ),
        "executed_future_kv_mediation_action": (
            "executed_donor",
            "executed_donor_kv_patch_all_action",
            "action_donor_projection",
            None,
        ),
        "executed_future_kv_mediation_physical": (
            "executed_donor",
            "executed_donor_kv_patch_all_action",
            "endpoint_donor_projection",
            "all",
        ),
    }
    effects: dict[str, Any] = {}
    raw_sign_p = []
    names = []
    for offset, (name, (left, right, field, group)) in enumerate(definitions.items()):
        values = contrast(reports, left, right, field, endpoint_group=group)
        if not values:
            continue
        estimate = effect_summary(values, seed=args.bootstrap_seed + offset)
        estimate["task_means"] = task_means(values, manifest_rows)
        effects[name] = estimate
        names.append(name)
        raw_sign_p.append(float(estimate["sign_test"]["p_value"]))
    if raw_sign_p:
        for name, adjusted in zip(names, holm_adjust(raw_sign_p), strict=True):
            effects[name]["sign_test"]["holm_adjusted_p_value"] = adjusted

    factor_expected = set(manifest["factorization_selected_unit_ids"])
    unexpected_factor_reports = {
        key
        for key, report in reports.items()
        if report.get("factorize_selected_donor", False) and key not in factor_expected
    }
    if unexpected_factor_reports:
        raise ValueError(
            "factorization was run outside the frozen subset: "
            f"{sorted(unexpected_factor_reports)}"
        )
    factor_reports = {
        key: report
        for key, report in reports.items()
        if report.get("factorize_selected_donor", False) and key in factor_expected
    }
    for key, report in factor_reports.items():
        direction_rate = float(
            report["factorization"]["decoded_factor_edge_direction_rate"]["composite"]
        )
        if direction_rate < 0.90:
            raise ValueError(
                f"decoded composite factor-edge gate failed for {key}: {direction_rate}"
            )
    factor_effects: dict[str, Any] = {}
    for outcome, path in (
        ("action", ("action_donor_projection_effects", None)),
        ("physical_all", ("endpoint_donor_projection_effects", "all")),
        ("physical_robot", ("endpoint_donor_projection_effects", "robot")),
        ("physical_object", ("endpoint_donor_projection_effects", "object")),
    ):
        for factor in ("robot_main_effect", "object_main_effect", "interaction"):
            values = {}
            for key, report in factor_reports.items():
                primary = report["factorial_effects"]["composite"]
                branch = primary[path[0]] if path[1] is None else primary[path[0]][path[1]]
                values[key] = float(branch[factor])
            if values:
                name = f"{outcome}_{factor}"
                estimate = effect_summary(
                    values, seed=args.bootstrap_seed + 100 + len(factor_effects)
                )
                estimate["task_means"] = task_means(values, manifest_rows)
                factor_effects[name] = estimate

    factor_observed = set(factor_reports)
    report = {
        "scope": "frozen saved-state-clustered Cosmos 3 population estimates",
        "manifest_status": manifest["status"],
        "selection_uses_intervention_outcomes": False,
        "expected_units": len(expected),
        "observed_units": len(observed),
        "complete": not missing,
        "missing_unit_ids": missing,
        "effects": effects,
        "factorization": {
            "manifest_status": manifest["factorization_status"],
            "expected_units": len(factor_expected),
            "observed_units": len(factor_observed & factor_expected),
            "complete": factor_expected <= factor_observed,
            "missing_unit_ids": sorted(factor_expected - factor_observed),
            "effects": factor_effects,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
