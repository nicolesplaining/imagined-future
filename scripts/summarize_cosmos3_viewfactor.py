#!/usr/bin/env python3
"""Aggregate the Cosmos 3 wrist/exterior camera-view factorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagined_future.statistics import cluster_bootstrap_mean

# (wrist from donor, exterior from donor) -> condition label
CELLS = {
    (0, 0): "executed_self",
    (1, 0): "view_wrist_from_donor",
    (0, 1): "view_exterior_from_donor",
    (1, 1): "executed_donor",
}


def cell_values(report: dict, outcome: str) -> dict[tuple[int, int], float]:
    runs = report["interventions"]
    values = {}
    for cell, label in CELLS.items():
        run = runs[label]
        values[cell] = (
            float(run["action_donor_projection"]) if outcome == "action" else float(run["endpoint_donor_projection"]["all"])
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/cosmos3_viewfactor_v1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = {path.parent.name: json.loads(path.read_text()) for path in sorted(args.root.glob("*/summary.json"))}
    metrics: dict[str, dict[str, float]] = {}
    for name, report in reports.items():
        for outcome in ("action", "endpoint"):
            cells = cell_values(report, outcome)
            full = cells[(1, 1)] - cells[(0, 0)]
            values = {
                "full_video": full,
                "wrist_only": cells[(1, 0)] - cells[(0, 0)],
                "exterior_only": cells[(0, 1)] - cells[(0, 0)],
                "wrist_main_effect": 0.5 * (cells[(1, 0)] + cells[(1, 1)] - cells[(0, 0)] - cells[(0, 1)]),
                "exterior_main_effect": 0.5 * (cells[(0, 1)] + cells[(1, 1)] - cells[(0, 0)] - cells[(1, 0)]),
                "interaction": cells[(1, 1)] - cells[(1, 0)] - cells[(0, 1)] + cells[(0, 0)],
            }
            wrist_nearest = report["interventions"]["view_wrist_from_donor"]["nearest_native_action_seed"]
            exterior_nearest = report["interventions"]["view_exterior_from_donor"]["nearest_native_action_seed"]
            if outcome == "action":
                values["wrist_only_top1_donor"] = float(wrist_nearest == report["donor_seed"])
                values["exterior_only_top1_donor"] = float(exterior_nearest == report["donor_seed"])
            for key, value in values.items():
                metrics.setdefault(f"{outcome}_{key}", {})[name] = value
    aggregate = {
        key: cluster_bootstrap_mean(values, seed=20261110 + index)
        for index, (key, values) in enumerate(sorted(metrics.items()))
    }
    output = {"units": sorted(reports), "aggregate": aggregate, "per_unit": metrics}
    destination = args.output or args.root / "aggregate_summary.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for key, estimate in aggregate.items():
        print(f"{key:36s} n={estimate['units']:2d} mean={estimate['mean']:+.3f} [{estimate['lower']:+.3f}, {estimate['upper']:+.3f}]")


if __name__ == "__main__":
    main()
