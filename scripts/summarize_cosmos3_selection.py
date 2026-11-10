#!/usr/bin/env python3
"""Aggregate Cosmos 3 future-selection runs: selection value and outcome transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from imagined_future.statistics import cluster_bootstrap_mean


def rate(runs: list[dict[str, Any]]) -> float:
    return float(np.mean([run["success"] for run in runs]))


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def unit_rows(report: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, float]]]:
    outcomes = report["outcomes"]
    candidates = {int(seed): rate(outcomes[f"candidate_{seed}"]) for seed in report["candidate_seeds"]}
    distances = {int(seed): float(value) for seed, value in report["goal_distances"].items()}
    best, worst = int(report["best_seed"]), int(report["worst_seed"])
    metrics = {
        "random_candidate": float(np.mean(list(candidates.values()))),
        "best_candidate": candidates[best],
        "worst_candidate": candidates[worst],
        "best_minus_random": candidates[best] - float(np.mean(list(candidates.values()))),
        "varies": float(len(set(candidates.values())) > 1),
    }
    # Higher goal distance should mean lower success if the ranking is informative.
    rho = spearman([-distances[seed] for seed in candidates], [candidates[seed] for seed in candidates])
    if rho is not None:
        metrics["rank_success_spearman"] = rho
    insertions = []
    for label, runs in outcomes.items():
        if not label.startswith("inserted_"):
            continue
        source_name, recipient = label[len("inserted_"):].split("_into_")
        source = best if source_name == "best" else worst
        insertions.append(
            {
                "inserted": rate(runs),
                "source_native": candidates[source],
                "recipient_native": candidates[int(recipient)],
                "path_projection": float(report["insertion_projection_toward_source"][label[len("inserted_"):]]),
            }
        )
    return metrics, insertions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, nargs="+", default=[Path("results/cosmos3_selection_v1")])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Blocks are independent process runs on the same saved states; each state is
    # averaged across its blocks before resampling states.
    blocks: dict[str, list[dict[str, Any]]] = {}
    for root in args.root:
        for path in sorted(root.glob("*/summary.json")):
            blocks.setdefault(path.parent.name, []).append(json.loads(path.read_text()))
    reports = {name: items[0] for name, items in blocks.items()}
    per_unit, transfer = {}, {}
    # Route versus outcome: does the action follow the inserted path, and what
    # happens to success, when source and recipient share or differ in outcome?
    route_groups = {
        "same_outcome_success": lambda row: row["source_native"] == 1 and row["recipient_native"] == 1,
        "same_outcome_failure": lambda row: row["source_native"] == 0 and row["recipient_native"] == 0,
        "different_outcome": lambda row: row["source_native"] != row["recipient_native"],
    }
    route: dict[str, dict[str, dict[str, float]]] = {key: {"path": {}, "success": {}} for key in route_groups}
    for name, items in blocks.items():
        rows = [row for report in items for row in unit_rows(report)[1]]
        for key, predicate in route_groups.items():
            selected = [row for row in rows if predicate(row)]
            if selected:
                route[key]["path"][name] = float(np.mean([row["path_projection"] for row in selected]))
                route[key]["success"][name] = float(np.mean([row["inserted"] for row in selected]))
    for name, items in blocks.items():
        block_metrics, block_transfer = [], []
        for report in items:
            metrics, insertions = unit_rows(report)
            block_metrics.append(metrics)
            discordant = [row for row in insertions if row["source_native"] != row["recipient_native"]]
            if discordant:
                # Fraction of the source-recipient gap the inserted run moves toward the source.
                block_transfer.append(
                    float(
                        np.mean(
                            [
                                (row["inserted"] - row["recipient_native"])
                                / (row["source_native"] - row["recipient_native"])
                                for row in discordant
                            ]
                        )
                    )
                )
        keys = set().union(*block_metrics)
        per_unit[name] = {
            key: float(np.mean([metrics[key] for metrics in block_metrics if key in metrics])) for key in keys
        }
        per_unit[name]["varies"] = float(any(metrics["varies"] for metrics in block_metrics))
        per_unit[name]["blocks"] = float(len(items))
        if block_transfer:
            transfer[name] = float(np.mean(block_transfer))
    aggregate: dict[str, Any] = {}
    for metric in ("random_candidate", "best_candidate", "worst_candidate", "best_minus_random", "rank_success_spearman"):
        values = {name: row[metric] for name, row in per_unit.items() if metric in row}
        if values:
            aggregate[metric] = cluster_bootstrap_mean(values, seed=20261110)
    varying = {name: row for name, row in per_unit.items() if row["varies"]}
    for metric in ("best_minus_random",):
        values = {name: row[metric] for name, row in varying.items()}
        if values:
            aggregate[f"{metric}_varying_states"] = cluster_bootstrap_mean(values, seed=20261111)
    if transfer:
        aggregate["outcome_transfer_toward_source"] = cluster_bootstrap_mean(transfer, seed=20261112)
    for key, values in route.items():
        if values["path"]:
            aggregate[f"route_{key}_path_projection"] = cluster_bootstrap_mean(values["path"], seed=20261113)
            aggregate[f"route_{key}_inserted_success"] = cluster_bootstrap_mean(values["success"], seed=20261114)
    output = {
        "units": sorted(reports),
        "states_with_outcome_variation": sorted(varying),
        "aggregate": aggregate,
        "per_unit": per_unit,
        "transfer_per_unit": transfer,
    }
    destination = args.output or args.root[0] / "aggregate_summary.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"{len(reports)} states, {len(varying)} with outcome variation")
    for metric, estimate in aggregate.items():
        print(f"{metric:42s} n={estimate['units']:2d} mean={estimate['mean']:+.3f} [{estimate['lower']:+.3f}, {estimate['upper']:+.3f}]")


if __name__ == "__main__":
    main()
