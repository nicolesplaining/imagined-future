"""Build a frozen donor-pair manifest from natural branch artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from imagined_future.libero_semantics import physical_endpoint_feature_vector
from imagined_future.study_design import (
    distance_matched_control_donor,
    pairwise_l2,
    select_primary_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-action-l2", type=float, default=0.01)
    parser.add_argument("--minimum-endpoint-l2", type=float, default=1e-8)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--cosmos-commit", default="18a2accadf4e7a3531e56754102af5a24d2316da")
    parser.add_argument("--checkpoint-revision", default="cb689ec0e3347c13667d70a78a3447388f5c3bb8")
    parser.add_argument(
        "--container-digest",
        default="sha256:2f5ff2badf82657c82e046bff84cc754acf4a2b3973828f12ab802c751f439e0",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {args.output}")

    units = []
    for run_dir in args.branch_run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        if not summary.get("deterministic_tokenizer") or not summary.get("replay_exact"):
            raise ValueError(f"{run_dir} is not a valid deterministic exact-replay unit")
        branch = np.load(run_dir / "branches.npz", allow_pickle=False)
        predicate_records = json.loads((run_dir / "endpoint_predicates.json").read_text())
        features = np.stack(
            [
                physical_endpoint_feature_vector(record["snapshot"], proprio)
                for record, proprio in zip(
                    predicate_records, branch["endpoint_proprios"], strict=True
                )
            ]
        )
        left, right, selection = select_primary_pair(
            branch["normalized_branch_actions"],
            features,
            minimum_action_l2=args.minimum_action_l2,
            minimum_endpoint_l2=args.minimum_endpoint_l2,
        )
        action_distances = pairwise_l2(branch["normalized_branch_actions"])
        endpoint_distances = pairwise_l2(features)
        distance_controls = {
            "forward": distance_matched_control_donor(
                recipient=left,
                primary_donor=right,
                action_distances=action_distances,
                endpoint_distances=endpoint_distances,
            ),
            "reverse": distance_matched_control_donor(
                recipient=right,
                primary_donor=left,
                action_distances=action_distances,
                endpoint_distances=endpoint_distances,
            ),
        }
        units.append(
            {
                "unit_id": run_dir.name,
                "branch_run_dir": str(run_dir),
                "task_id": int(summary["task_id"]),
                "initial_state_index": int(summary["initial_state_index"]),
                "prefix_chunks": int(summary["prefix_chunks"]),
                "primary_pair": [left, right],
                "distance_matched_control_by_direction": distance_controls,
                "selection": selection,
                "branch_seeds": branch["branch_seeds"].astype(int).tolist(),
                "artifact_sha256": {
                    "branches.npz": hashlib.sha256((run_dir / "branches.npz").read_bytes()).hexdigest(),
                    "summary.json": hashlib.sha256((run_dir / "summary.json").read_bytes()).hexdigest(),
                    "endpoint_predicates.json": hashlib.sha256(
                        (run_dir / "endpoint_predicates.json").read_bytes()
                    ).hexdigest(),
                },
                "action_distances": action_distances.tolist(),
                "physical_endpoint_distances": endpoint_distances.tolist(),
            }
        )

    units.sort(key=lambda unit: (unit["task_id"], unit["initial_state_index"], unit["prefix_chunks"]))
    by_task: dict[int, list[dict]] = {}
    for unit in units:
        by_task.setdefault(unit["task_id"], []).append(unit)
    for task_units in by_task.values():
        for index, unit in enumerate(task_units):
            shuffled = task_units[(index + 1) % len(task_units)]
            unit["within_task_shuffled_target"] = {
                "unit_id": shuffled["unit_id"],
                "branch_run_dir": shuffled["branch_run_dir"],
                "branch_index": shuffled["primary_pair"][1],
            }

    result = {
        "scope": "confirmatory pairs selected without intervention outcomes",
        "provenance": {
            "project_commit": args.project_commit,
            "cosmos_policy_commit": args.cosmos_commit,
            "checkpoint_revision": args.checkpoint_revision,
            "container_digest": args.container_digest,
        },
        "minimum_action_l2": args.minimum_action_l2,
        "minimum_endpoint_l2": args.minimum_endpoint_l2,
        "units": units,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "units": len(units)}, indent=2))


if __name__ == "__main__":
    main()
