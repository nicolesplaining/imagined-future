"""Analyze two image-preserving rollouts from one task and initial state."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from imagined_future.paired_rollouts import paired_query_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-episode", type=Path, required=True)
    parser.add_argument("--right-episode", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    left_summary = json.loads((args.left_episode / "summary.json").read_text())
    right_summary = json.loads((args.right_episode / "summary.json").read_text())
    identity_fields = ("suite", "task_id", "initial_state_index", "initial_state_digest")
    mismatches = {
        field: (left_summary.get(field), right_summary.get(field))
        for field in identity_fields
        if left_summary.get(field) != right_summary.get(field)
    }
    if mismatches:
        raise ValueError(f"episodes are not matched on task and initial state: {mismatches}")

    left = np.load(args.left_episode / "rollout.npz", allow_pickle=False)
    right = np.load(args.right_episode / "rollout.npz", allow_pickle=False)
    if not np.array_equal(left["initial_state"], right["initial_state"]):
        raise ValueError("serialized initial simulator states are not bitwise equal")
    required_images = (
        "current_primary_images",
        "predicted_primary_images",
        "endpoint_primary_images",
    )
    missing = [name for name in required_images if name not in left.files or name not in right.files]
    if missing:
        raise ValueError(f"rollouts were not collected with --save-images; missing {missing}")

    common_queries = min(len(left["query_steps"]), len(right["query_steps"]))
    if not np.array_equal(left["query_steps"][:common_queries], right["query_steps"][:common_queries]):
        raise ValueError("paired query steps do not align")
    rows = [paired_query_metrics(left, right, index) for index in range(common_queries)]
    first = rows[0]
    if first["current_state_l2"] != 0.0 or first["current_primary_pixel_l1"] != 0.0:
        raise RuntimeError("the first paired query is not an exact current-state/observation match")

    payload = {
        "scope": "descriptive paired-rollout analysis; not a causal intervention",
        "task_id": left_summary["task_id"],
        "task_description": left_summary["task_description"],
        "initial_state_index": left_summary["initial_state_index"],
        "initial_state_digest": left_summary["initial_state_digest"],
        "left": {
            "seed": left_summary["model_seed"],
            "success": left_summary["success"],
            "success_step": left_summary["success_step"],
            "episode": str(args.left_episode.resolve()),
        },
        "right": {
            "seed": right_summary["model_seed"],
            "success": right_summary["success"],
            "success_step": right_summary["success_step"],
            "episode": str(args.right_episode.resolve()),
        },
        "common_queries": common_queries,
        "first_query": first,
        "maximum_predicted_divergence_query": max(rows, key=lambda row: row["predicted_primary_pixel_l1"]),
        "maximum_endpoint_divergence_query": max(rows, key=lambda row: row["endpoint_primary_pixel_l1"]),
        "queries": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "queries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: payload[key] for key in ("task_id", "common_queries", "first_query")}, indent=2))


if __name__ == "__main__":
    main()
