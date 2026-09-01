#!/usr/bin/env python3
"""Aggregate frozen-state Cosmos 3 donor-transplant engineering pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def task_bootstrap(
    rows: list[dict[str, Any]], field: str, *, seed: int, replicates: int
) -> list[float] | None:
    """Bootstrap the task-clustered mean, retaining all donors within a task."""

    tasks = sorted({str(row["task"]) for row in rows})
    if len(tasks) < 2:
        return None
    by_task = {
        task: np.asarray([float(row[field]) for row in rows if row["task"] == task])
        for task in tasks
    }
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.choice(tasks, size=len(tasks), replace=True)
        estimates[index] = np.mean([by_task[str(task)].mean() for task in sampled])
    return percentile_interval(estimates)


def summarize_rows(rows: list[dict[str, Any]], *, seed: int, replicates: int) -> dict[str, Any]:
    action = np.asarray([row["action_projection"] for row in rows], dtype=np.float64)
    endpoint = np.asarray([row["endpoint_projection"] for row in rows], dtype=np.float64)
    return {
        "number_of_donors": len(rows),
        "number_of_tasks": len({row["task"] for row in rows}),
        "action_projection_mean": float(action.mean()),
        "action_projection_median": float(np.median(action)),
        "action_projection_positive_fraction": float(np.mean(action > 0.0)),
        "action_correct_donor_top1_fraction": float(
            np.mean([row["action_correct_donor_top1"] for row in rows])
        ),
        "endpoint_projection_mean": float(endpoint.mean()),
        "endpoint_projection_median": float(np.median(endpoint)),
        "endpoint_projection_positive_fraction": float(np.mean(endpoint > 0.0)),
        "endpoint_correct_donor_top1_fraction": float(
            np.mean([row["endpoint_correct_donor_top1"] for row in rows])
        ),
        "action_projection_task_bootstrap_95ci": task_bootstrap(
            rows, "action_projection", seed=seed, replicates=replicates
        ),
        "endpoint_projection_task_bootstrap_95ci": task_bootstrap(
            rows, "endpoint_projection", seed=seed + 1, replicates=replicates
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.summaries]
    if any(not report.get("multi_donor", False) for report in reports):
        raise ValueError("all summaries must come from --multi-donor runs")
    study_ids = [str(report["study_id"]) for report in reports]
    if len(study_ids) != len(set(study_ids)):
        raise ValueError("study IDs must be unique")

    rows: list[dict[str, Any]] = []
    selected_controls: list[dict[str, Any]] = []
    for report in reports:
        task = str(report["task"])
        recipient = int(report["recipient_seed"])
        selected_donor = int(report["donor_seed"])
        interventions = report["interventions"]
        for label, intervention in interventions.items():
            target = intervention.get("target_donor_seed")
            if target is None or not label.startswith(("predicted_donor", "executed_donor")):
                continue
            rows.append(
                {
                    "study_id": report["study_id"],
                    "task": task,
                    "branch_step": report["branch_step"],
                    "source": "predicted" if label.startswith("predicted") else "executed",
                    "label": label,
                    "recipient_seed": recipient,
                    "target_donor_seed": int(target),
                    "selected_maximum_separation_donor": int(target) == selected_donor,
                    "action_projection": intervention["action_target_donor_projection"],
                    "endpoint_projection": intervention["endpoint_target_donor_projection"]["all"],
                    "robot_endpoint_projection": intervention["endpoint_target_donor_projection"]["robot"],
                    "object_endpoint_projection": intervention["endpoint_target_donor_projection"]["object"],
                    "action_correct_donor_top1": intervention["correct_action_donor_top1"],
                    "endpoint_correct_donor_top1": intervention["correct_endpoint_donor_top1"]["all"],
                }
            )

        self_control = interventions["self"]
        gaussian = interventions["gaussian_executed"]
        selected_controls.append(
            {
                "study_id": report["study_id"],
                "task": task,
                "self_action_projection": self_control["action_donor_projection"],
                "gaussian_action_projection": gaussian["action_donor_projection"],
                "self_endpoint_projection": self_control["endpoint_donor_projection"]["all"],
                "gaussian_endpoint_projection": gaussian["endpoint_donor_projection"]["all"],
                "predicted_minus_self_action_projection": (
                    interventions["predicted_donor"]["action_donor_projection"]
                    - self_control["action_donor_projection"]
                ),
                "executed_minus_gaussian_action_projection": (
                    interventions["executed_donor"]["action_donor_projection"]
                    - gaussian["action_donor_projection"]
                ),
                "predicted_minus_self_endpoint_projection": (
                    interventions["predicted_donor"]["endpoint_donor_projection"]["all"]
                    - self_control["endpoint_donor_projection"]["all"]
                ),
                "executed_minus_gaussian_endpoint_projection": (
                    interventions["executed_donor"]["endpoint_donor_projection"]["all"]
                    - gaussian["endpoint_donor_projection"]["all"]
                ),
            }
        )

    source_summaries = {
        source: summarize_rows(
            [row for row in rows if row["source"] == source],
            seed=args.bootstrap_seed + offset,
            replicates=args.bootstrap_replicates,
        )
        for offset, source in enumerate(("predicted", "executed"))
    }
    report = {
        "scope": (
            "cross-task frozen-state engineering replication; one outcome-independent branch "
            "state per task; uncertainty resamples tasks"
        ),
        "study_ids": study_ids,
        "number_of_tasks": len(reports),
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "source_summaries": source_summaries,
        "selected_donor_controls": selected_controls,
        "donor_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
