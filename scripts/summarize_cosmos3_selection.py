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
            }
        )
    return metrics, insertions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/cosmos3_selection_v1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = {path.parent.name: json.loads(path.read_text()) for path in sorted(args.root.glob("*/summary.json"))}
    per_unit, transfer = {}, {}
    for name, report in reports.items():
        metrics, insertions = unit_rows(report)
        per_unit[name] = metrics
        discordant = [row for row in insertions if row["source_native"] != row["recipient_native"]]
        if discordant:
            # Fraction of the source-recipient gap the inserted run moves toward the source.
            transfer[name] = float(
                np.mean(
                    [
                        (row["inserted"] - row["recipient_native"]) / (row["source_native"] - row["recipient_native"])
                        for row in discordant
                    ]
                )
            )
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
    output = {
        "units": sorted(reports),
        "states_with_outcome_variation": sorted(varying),
        "aggregate": aggregate,
        "per_unit": per_unit,
        "transfer_per_unit": transfer,
    }
    destination = args.output or args.root / "aggregate_summary.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"{len(reports)} states, {len(varying)} with outcome variation")
    for metric, estimate in aggregate.items():
        print(f"{metric:42s} n={estimate['units']:2d} mean={estimate['mean']:+.3f} [{estimate['lower']:+.3f}, {estimate['upper']:+.3f}]")


if __name__ == "__main__":
    main()
