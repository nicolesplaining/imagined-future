#!/usr/bin/env python3
"""Summarize frozen Cosmos 3 future-K/V mediation calibration and holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ("rubiks", "mustard", "spoon", "marker")
FROZEN_LAYER = 16
BOOTSTRAP_SEED = 20260901
BOOTSTRAP_REPETITIONS = 10_000


def interval(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(BOOTSTRAP_REPETITIONS, len(array)), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "task_bootstrap_95_interval": [float(low), float(high)],
        "positive_tasks": int((array > 0).sum()),
        "task_count": int(len(array)),
    }


def calibration_summary(root: Path) -> dict[str, Any]:
    reports = {
        task: json.loads((root / task / "report.json").read_text()) for task in TASKS
    }
    rows = []
    layer_losses: dict[int, list[float]] = {layer: [] for layer in range(36)}
    for task, report in reports.items():
        by_layer = {row["layer"]: row for row in report["layer_rows"]}
        rows.append(
            {
                "task": task,
                "study_id": report["study_id"],
                "baseline_projection": report[
                    "baseline_transplant_action_donor_projection"
                ],
                "all_direct_projection": report[
                    "all_direct_patch_action_donor_projection"
                ],
                "all_direct_mediation_loss": report[
                    "all_direct_patch_mediation_loss"
                ],
                "all_barrier_projection": report[
                    "all_barrier_patch_action_donor_projection"
                ],
                "all_barrier_mediation_loss": report[
                    "all_barrier_patch_mediation_loss"
                ],
                "layer16_projection": by_layer[FROZEN_LAYER][
                    "action_donor_projection"
                ],
                "layer16_mediation_loss": by_layer[FROZEN_LAYER][
                    "mediation_loss_from_baseline"
                ],
                "record_action_maximum_error": report[
                    "record_action_maximum_error"
                ],
                "self_patch_maximum_action_error": report[
                    "self_patch_maximum_action_error"
                ],
                "donor_repeat_maximum_action_error": report[
                    "donor_repeat_maximum_action_error"
                ],
                "cache_layers": len(report["cache_call_counts"]),
                "cache_calls_per_layer": sorted(
                    set(report["cache_call_counts"].values())
                ),
            }
        )
        for layer, row in by_layer.items():
            layer_losses[layer].append(row["mediation_loss_from_baseline"])

    eligible = [
        layer for layer, values in layer_losses.items() if all(value > 0 for value in values)
    ]
    selected = max(eligible, key=lambda layer: np.mean(layer_losses[layer]))
    if selected != FROZEN_LAYER:
        raise RuntimeError(
            f"frozen selection audit failed: expected layer {FROZEN_LAYER}, got {selected}"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = {}
    for key in (
        "baseline_projection",
        "all_direct_mediation_loss",
        "all_barrier_mediation_loss",
        "layer16_mediation_loss",
    ):
        metrics[key] = interval([float(row[key]) for row in rows], rng)

    return {
        "scope": "excluded four-task calibration",
        "tasks": list(TASKS),
        "rows": rows,
        "metrics": metrics,
        "layer_selection_rule": (
            "among layers with positive single-layer mediation loss in all four "
            "calibration tasks, select the largest task mean"
        ),
        "eligible_layers": eligible,
        "selected_layer": selected,
        "all_layer_means": {
            str(layer): float(np.mean(values)) for layer, values in layer_losses.items()
        },
        "bootstrap": {
            "cluster_unit": "task",
            "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPETITIONS,
        },
    }


def endpoint_losses(
    baseline: dict[str, Any], patched: dict[str, Any]
) -> dict[str, float]:
    return {
        group: float(value - patched["endpoint_donor_projection"][group])
        for group, value in baseline["endpoint_donor_projection"].items()
    }


def holdout_summary(
    path: Path,
    *,
    scope: str = "prospectively frozen held-out physical Banana state",
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    interventions = report["interventions"]
    arms = {}
    suffixes = {
        "selected": "kv_patch_selected",
        "all_direct": "kv_patch_all_action",
        "all_barrier": "kv_patch_all_nonfuture",
    }
    for source in ("predicted", "executed"):
        baseline = interventions[f"{source}_donor"]
        patches = {}
        for label, suffix in suffixes.items():
            patched = interventions[f"{source}_donor_{suffix}"]
            patches[label] = {
                "action_projection": patched["action_donor_projection"],
                "action_mediation_loss": float(
                    baseline["action_donor_projection"]
                    - patched["action_donor_projection"]
                ),
                "endpoint_projection": patched["endpoint_donor_projection"],
                "endpoint_mediation_loss": endpoint_losses(baseline, patched),
                "target_future_maximum_error": patched["server"][
                    "research_target_future_max_error"
                ],
            }
        arms[source] = {
            "baseline_action_projection": baseline["action_donor_projection"],
            "baseline_endpoint_projection": baseline["endpoint_donor_projection"],
            "patches": patches,
        }

    self_projection = interventions["self"]["action_donor_projection"]
    record_projection = interventions["self_kv_record"]["action_donor_projection"]
    self_patch_projection = interventions["self_kv_patch_all"][
        "action_donor_projection"
    ]
    cache_counts = interventions["self_kv_record"]["server"][
        "research_attention_interface"
    ]["cache_call_counts"]
    return {
        "scope": scope,
        "study_id": report["study_id"],
        "task": report["task"],
        "branch_step": report["branch_step"],
        "selected_layer": report["attention_kv_patch_layers"],
        "recipient_seed": report["recipient_seed"],
        "donor_seed": report["donor_seed"],
        "prefix_maximum_state_error": report["prefix_maximum_state_error"],
        "restore_image_maximum_absolute_errors": report[
            "restore_image_maximum_absolute_errors"
        ],
        "unique_restore_state_digests": len(set(report["restore_state_digests"])),
        "continuation_repeat_exact": report["continuation_repeat_audit"]["exact"],
        "server_state_hash": report["server_state_hash"],
        "controls": {
            "self_action_projection": self_projection,
            "gaussian_action_projection": interventions["gaussian_executed"][
                "action_donor_projection"
            ],
            "gaussian_endpoint_projection": interventions["gaussian_executed"][
                "endpoint_donor_projection"
            ],
        },
        "identity_gates": {
            "record_projection_error": float(record_projection - self_projection),
            "all_layer_self_patch_projection_error": float(
                self_patch_projection - self_projection
            ),
            "cache_layers": len(cache_counts),
            "cache_calls_per_layer": sorted(set(cache_counts.values())),
            "note": (
                "the runner required elementwise maximum action error exactly zero "
                "for both identity arms before executing donor patches"
            ),
        },
        "arms": arms,
    }


def physical_replication_summary(
    path: Path, reference: dict[str, Any]
) -> dict[str, Any]:
    replication = holdout_summary(
        path,
        scope=(
            "fresh-server exact-state physical process replication; repeated state is "
            "not an additional population unit"
        ),
    )
    comparison = {}
    for source in ("predicted", "executed"):
        reference_arm = reference["arms"][source]
        replication_arm = replication["arms"][source]
        patch_deltas = {}
        for label in ("selected", "all_direct", "all_barrier"):
            reference_patch = reference_arm["patches"][label]
            replication_patch = replication_arm["patches"][label]
            patch_deltas[label] = {
                "action_projection_delta": float(
                    replication_patch["action_projection"]
                    - reference_patch["action_projection"]
                ),
                "robot_endpoint_projection_delta": float(
                    replication_patch["endpoint_projection"]["robot"]
                    - reference_patch["endpoint_projection"]["robot"]
                ),
            }
        comparison[source] = {
            "baseline_action_projection_delta": float(
                replication_arm["baseline_action_projection"]
                - reference_arm["baseline_action_projection"]
            ),
            "baseline_robot_endpoint_projection_delta": float(
                replication_arm["baseline_endpoint_projection"]["robot"]
                - reference_arm["baseline_endpoint_projection"]["robot"]
            ),
            "patch_deltas": patch_deltas,
        }
    replication["comparison_to_holdout"] = comparison
    return replication


def process_block_summary(root: Path, calibration: dict[str, Any]) -> dict[str, Any]:
    reference = {row["task"]: row for row in calibration["rows"]}
    block_rows = []
    for block_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_rows = []
        for task in TASKS:
            report = json.loads((block_dir / task / "report.json").read_text())
            layer16 = next(
                row for row in report["layer_rows"] if row["layer"] == FROZEN_LAYER
            )
            row = {
                "task": task,
                "study_id": report["study_id"],
                "baseline_projection": report[
                    "baseline_transplant_action_donor_projection"
                ],
                "all_direct_mediation_loss": report[
                    "all_direct_patch_mediation_loss"
                ],
                "all_barrier_mediation_loss": report[
                    "all_barrier_patch_mediation_loss"
                ],
                "layer16_mediation_loss": layer16["mediation_loss_from_baseline"],
                "identity_errors": {
                    "record": report["record_action_maximum_error"],
                    "self_patch": report["self_patch_maximum_action_error"],
                    "donor_repeat": report["donor_repeat_maximum_action_error"],
                },
            }
            comparable = (
                "baseline_projection",
                "all_direct_mediation_loss",
                "all_barrier_mediation_loss",
                "layer16_mediation_loss",
            )
            row["exact_calibration_metric_match"] = all(
                row[key] == reference[task][key] for key in comparable
            )
            task_rows.append(row)
        block_rows.append(
            {
                "block": block_dir.name,
                "tasks": task_rows,
                "mean_all_direct_mediation_loss": float(
                    np.mean([row["all_direct_mediation_loss"] for row in task_rows])
                ),
                "mean_all_barrier_mediation_loss": float(
                    np.mean([row["all_barrier_mediation_loss"] for row in task_rows])
                ),
                "mean_layer16_mediation_loss": float(
                    np.mean([row["layer16_mediation_loss"] for row in task_rows])
                ),
                "all_identity_gates_exact": all(
                    value == 0.0
                    for row in task_rows
                    for value in row["identity_errors"].values()
                ),
                "all_metrics_exact_calibration_match": all(
                    row["exact_calibration_metric_match"] for row in task_rows
                ),
            }
        )

    return {
        "scope": (
            "fresh-process numerical replication on repeated excluded calibration states; "
            "blocks are not additional population units"
        ),
        "blocks": block_rows,
        "block_count": len(block_rows),
        "all_blocks_identity_exact": all(
            row["all_identity_gates_exact"] for row in block_rows
        ),
        "all_blocks_metrics_exact_calibration_match": all(
            row["all_metrics_exact_calibration_match"] for row in block_rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--physical-replication", type=Path)
    parser.add_argument("--process-block-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = calibration_summary(args.calibration_root)
    holdout = holdout_summary(args.holdout) if args.holdout else None
    if args.physical_replication and holdout is None:
        raise ValueError("--physical-replication requires --holdout")
    summary = {
        "scope": "Cosmos 3 token-count-preserving future-K/V content mediation",
        "calibration": calibration,
        "holdout": holdout,
        "physical_replication": (
            physical_replication_summary(args.physical_replication, holdout)
            if args.physical_replication and holdout is not None
            else None
        ),
        "process_blocks": (
            process_block_summary(args.process_block_root, calibration)
            if args.process_block_root
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
