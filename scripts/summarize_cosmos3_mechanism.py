#!/usr/bin/env python3
"""Aggregate Cosmos 3 mechanism-suite results over frozen saved-state units."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from imagined_future.statistics import cluster_bootstrap_mean, exact_sign_test

KV_PATCHES = {
    "all_layers": "predicted_donor_kv_patch_all_action",
    "layer_16": "predicted_donor_kv_patch_selected",
    "layers_00_11": "predicted_donor_kv_layers_00_11",
    "layers_12_23": "predicted_donor_kv_layers_12_23",
    "layers_24_35": "predicted_donor_kv_layers_24_35",
    "all_but_16": "predicted_donor_kv_all_but_16",
    "keys_only": "predicted_donor_k_patch_all",
    "values_only": "predicted_donor_v_patch_all",
    "current_video_and_action_queries": "predicted_donor_kv_patch_all_nonfuture",
    "natural_run_kv": "predicted_donor_natural_kv_patch_all",
}


def unit_metrics(report: dict[str, Any]) -> dict[str, float]:
    runs = report["interventions"]
    donor = report["donor_seed"]
    metrics: dict[str, float] = {}

    def action(label: str, field: str = "action_donor_projection") -> float | None:
        value = runs.get(label, {}).get(field)
        return None if value is None else float(value)

    def endpoint(label: str, group: str = "all") -> float | None:
        values = runs.get(label, {}).get("endpoint_donor_projection")
        return None if not values else float(values[group])

    baseline = action("predicted_donor")
    metrics["predicted_donor_action"] = baseline
    # E1: when during denoising the inserted future acts.
    for step in range(4):
        metrics[f"clamp_step_{step}_action"] = action(f"predicted_donor_clamp_step_{step}")
        metrics[f"state_insert_step_{step}_action"] = action(f"state_insert_step_{step}")
    for alpha in (25, 50, 75):
        metrics[f"interpolate_{alpha}_action"] = action(f"interpolate_alpha_{alpha}")
    executed_self_endpoint = endpoint("executed_self") or 0.0
    for label in ("state_insert_step_0", "interpolate_alpha_50", "predicted_donor"):
        value = endpoint(label)
        metrics[f"{label}_endpoint"] = None if value is None else value - executed_self_endpoint

    # E2: the action follows whichever future is inserted.
    pairs = [run for label, run in runs.items() if label.startswith("pair_")]
    if pairs:
        metrics["pair_top1_accuracy"] = float(np.mean([bool(run["correct_action_donor_top1"]) for run in pairs]))
        metrics["pair_target_projection"] = float(np.mean([run["action_target_donor_projection"] for run in pairs]))
        reciprocal = runs.get(f"pair_{donor}_from_{report['recipient_seed']}")
        if reciprocal is not None:
            metrics["reciprocal_target_projection"] = float(reciprocal["action_target_donor_projection"])
        natural = runs.get(f"pair_{report['recipient_seed']}_from_{report['natural_control_seed']}")
        if natural is not None:
            metrics["natural_own_target_projection"] = float(natural["action_target_donor_projection"])

    # E3: which K/V interfaces carry the donor effect.
    for name, label in KV_PATCHES.items():
        patched = action(label)
        if baseline is not None and patched is not None:
            metrics[f"kv_loss_{name}"] = baseline - patched
    natural_patch = runs.get(KV_PATCHES["natural_run_kv"])
    if natural_patch is not None:
        metrics["natural_kv_projection_onto_natural"] = float(natural_patch["action_target_donor_projection"])
        metrics["natural_kv_top1_natural"] = float(
            natural_patch["nearest_native_action_seed"] == report["natural_control_seed"]
        )

    # E4: incoherent, foreign, and executed natural futures.
    executed_self_action = action("executed_self") or 0.0
    for label in ("predicted_donor_scrambled", "predicted_donor_reversed"):
        metrics[f"{label}_action"] = action(label)
        value = endpoint(label)
        metrics[f"{label}_endpoint"] = None if value is None else value - executed_self_endpoint
    for label in ("foreign_executed", "executed_natural", "executed_donor", "gaussian_executed"):
        value = action(label)
        metrics[f"{label}_action"] = None if value is None else value - executed_self_action
        value = endpoint(label)
        metrics[f"{label}_endpoint"] = None if value is None else value - executed_self_endpoint
    if metrics.get("executed_donor_endpoint") is not None and metrics.get("executed_natural_endpoint") is not None:
        metrics["executed_donor_minus_executed_natural_endpoint"] = (
            metrics["executed_donor_endpoint"] - metrics["executed_natural_endpoint"]
        )
        metrics["executed_donor_minus_executed_natural_action"] = (
            metrics["executed_donor_action"] - metrics["executed_natural_action"]
        )
    natural_exec = runs.get("executed_natural")
    if natural_exec is not None:
        metrics["executed_natural_own_target_projection"] = float(natural_exec["action_target_donor_projection"])
    return {key: value for key, value in metrics.items() if value is not None}


def summarize(values: dict[str, float], seed: int) -> dict[str, Any]:
    estimate = cluster_bootstrap_mean(values, resamples=10_000, seed=seed)
    array = np.asarray(list(values.values()))
    estimate.update(
        median=float(np.median(array)),
        positive_units=int(np.sum(array > 0)),
        sign_test=exact_sign_test(list(values.values())),
    )
    return estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/cosmos3_mechanism_v1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = {
        path.parent.name: json.loads(path.read_text())
        for path in sorted(args.root.glob("*/summary.json"))
    }
    per_unit = {name: unit_metrics(report) for name, report in reports.items()}
    by_metric: dict[str, dict[str, float]] = defaultdict(dict)
    for name, metrics in per_unit.items():
        for metric, value in metrics.items():
            by_metric[metric][name] = value
    aggregate = {
        metric: summarize(values, seed=20261109 + index)
        for index, (metric, values) in enumerate(sorted(by_metric.items()))
    }
    output = {
        "units": sorted(reports),
        "tasks": sorted({report["task"] for report in reports.values()}),
        "metrics": aggregate,
        "per_unit": per_unit,
    }
    destination = args.output or args.root / "aggregate_summary.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for metric, estimate in aggregate.items():
        print(
            f"{metric:50s} n={estimate['units']:2d} mean={estimate['mean']:+.3f} "
            f"[{estimate['lower']:+.3f}, {estimate['upper']:+.3f}] pos={estimate['positive_units']}"
        )


if __name__ == "__main__":
    main()
