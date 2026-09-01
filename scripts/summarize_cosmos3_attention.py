#!/usr/bin/env python3
"""Aggregate excluded Cosmos 3 attention scans and a frozen held-out replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def physical_arm(summary: dict, source: str) -> dict:
    baseline = summary["interventions"][f"{source}_donor"]
    direct = summary["interventions"][f"{source}_donor_attention_action"]
    barrier = summary["interventions"][f"{source}_donor_attention_nonfuture"]
    return {
        "baseline_action_projection": baseline["action_donor_projection"],
        "direct_action_projection": direct["action_donor_projection"],
        "direct_mediation_loss": (
            baseline["action_donor_projection"] - direct["action_donor_projection"]
        ),
        "barrier_action_projection": barrier["action_donor_projection"],
        "barrier_mediation_loss": (
            baseline["action_donor_projection"] - barrier["action_donor_projection"]
        ),
        "baseline_endpoint_projection": baseline["endpoint_donor_projection"],
        "direct_endpoint_projection": direct["endpoint_donor_projection"],
        "barrier_endpoint_projection": barrier["endpoint_donor_projection"],
        "target_future_max_errors": {
            "baseline": baseline["server"]["research_target_future_max_error"],
            "direct": direct["server"]["research_target_future_max_error"],
            "barrier": barrier["server"]["research_target_future_max_error"],
        },
    }


def bootstrap_interval(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return [float(item) for item in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--heldout-summary", type=Path, required=True)
    parser.add_argument("--initial-calibration", type=Path)
    parser.add_argument("--initial-physical-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--bootstrap-replicates", type=int, default=100000)
    args = parser.parse_args()

    reports = []
    for path in sorted(args.calibration_root.glob("*/report.json")):
        report = json.loads(path.read_text())
        zero_errors = [
            report["implicit_vs_explicit_zero_gate_maximum_action_error"],
            report["empty_exclusion_maximum_action_error"],
            report["empty_nonfuture_exclusion_maximum_action_error"],
        ]
        if any(error != 0.0 for error in zero_errors):
            raise RuntimeError(f"nonzero identity error in {path}: {zero_errors}")
        reports.append((path.parent.name, report))
    if not reports:
        raise RuntimeError("no calibration reports found")

    layer_effects = np.asarray(
        [
            [row["mediation_loss_from_baseline"] for row in report["action_query_layers"]]
            for _, report in reports
        ],
        dtype=np.float64,
    )
    if layer_effects.shape[1] != 36:
        raise RuntimeError(f"expected 36 attention layers, got {layer_effects.shape}")
    positive_all = np.all(layer_effects > 0, axis=0)
    candidates = np.flatnonzero(positive_all)
    if not len(candidates):
        raise RuntimeError("no layer had positive mediation in every calibration task")
    selected_layer = int(candidates[np.argmax(layer_effects[:, candidates].mean(axis=0))])

    task_rows = []
    for task_index, (task, report) in enumerate(reports):
        task_rows.append(
            {
                "task": task,
                "baseline_action_projection": report[
                    "baseline_transplant_action_donor_projection"
                ],
                "all_direct_mediation_loss": report["all_layer_mediation_loss"],
                "all_barrier_mediation_loss": report[
                    "all_layer_nonfuture_barrier_mediation_loss"
                ],
                "selected_layer_mediation_loss": float(
                    layer_effects[task_index, selected_layer]
                ),
            }
        )

    selected = layer_effects[:, selected_layer]
    all_direct = np.asarray(
        [report["all_layer_mediation_loss"] for _, report in reports], dtype=np.float64
    )
    all_barrier = np.asarray(
        [
            report["all_layer_nonfuture_barrier_mediation_loss"]
            for _, report in reports
        ],
        dtype=np.float64,
    )
    heldout = json.loads(args.heldout_summary.read_text())
    failed_initial_layer34 = None
    if args.initial_calibration is not None and args.initial_physical_summary is not None:
        initial_calibration = json.loads(args.initial_calibration.read_text())
        initial_physical = json.loads(args.initial_physical_summary.read_text())
        failed_initial_layer34 = {
            "calibration": {
                "baseline_action_projection": initial_calibration[
                    "baseline_transplant_action_donor_projection"
                ],
                "direct_action_projection": initial_calibration[
                    "action_query_layers"
                ][34]["action_donor_projection"],
                "direct_mediation_loss": initial_calibration[
                    "action_query_layers"
                ][34]["mediation_loss_from_baseline"],
                "barrier_action_projection": initial_calibration[
                    "nonfuture_barrier_layers"
                ][34]["action_donor_projection"],
                "barrier_mediation_loss": initial_calibration[
                    "nonfuture_barrier_layers"
                ][34]["mediation_loss_from_baseline"],
            },
            "physical": {
                "attention_layers": initial_physical["attention_mediation_layers"],
                "predicted": physical_arm(initial_physical, "predicted"),
                "executed": physical_arm(initial_physical, "executed"),
            },
            "interpretation": "excluded calibration effect reversed sign in physical replay",
        }
    output = {
        "scope": (
            "excluded four-task attention calibration plus prospectively frozen "
            "held-out Banana physical replay"
        ),
        "calibration_tasks": [task for task, _ in reports],
        "selection_rule": (
            "among layers with positive mediation in every calibration task, select "
            "the largest task-mean single-layer mediation"
        ),
        "selected_layer": selected_layer,
        "task_rows": task_rows,
        "calibration_aggregate": {
            "selected_layer_mean_mediation_loss": float(selected.mean()),
            "selected_layer_positive_tasks": int((selected > 0).sum()),
            "selected_layer_task_bootstrap_95_interval": bootstrap_interval(
                selected,
                seed=args.bootstrap_seed,
                replicates=args.bootstrap_replicates,
            ),
            "all_direct_mean_mediation_loss": float(all_direct.mean()),
            "all_direct_positive_tasks": int((all_direct > 0).sum()),
            "all_direct_task_bootstrap_95_interval": bootstrap_interval(
                all_direct,
                seed=args.bootstrap_seed + 1,
                replicates=args.bootstrap_replicates,
            ),
            "all_barrier_mean_mediation_loss": float(all_barrier.mean()),
            "all_barrier_positive_tasks": int((all_barrier > 0).sum()),
            "all_barrier_task_bootstrap_95_interval": bootstrap_interval(
                all_barrier,
                seed=args.bootstrap_seed + 2,
                replicates=args.bootstrap_replicates,
            ),
        },
        "heldout": {
            "task": heldout["task"],
            "branch_step": heldout["branch_step"],
            "attention_layers": heldout["attention_mediation_layers"],
            "prefix_maximum_state_error": heldout["prefix_maximum_state_error"],
            "continuation_repeat_exact": heldout["continuation_repeat_audit"]["exact"],
            "self_action_projection": heldout["interventions"]["self"][
                "action_donor_projection"
            ],
            "gaussian_action_projection": heldout["interventions"]["gaussian_executed"][
                "action_donor_projection"
            ],
            "predicted": physical_arm(heldout, "predicted"),
            "executed": physical_arm(heldout, "executed"),
        },
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "failed_initial_layer34": failed_initial_layer34,
        "interpretation": (
            "the calibration-selected single layer failed with opposite-signed action "
            "mediation on the held-out physical state; no task-stable single-layer "
            "bottleneck is established"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
