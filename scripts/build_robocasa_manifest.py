"""Freeze RoboCasa donor pairs using natural actions and endpoints only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from imagined_future.study_design import pairwise_l2, select_primary_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-action-l2", type=float, default=0.01)
    parser.add_argument("--minimum-endpoint-l2", type=float, default=1e-8)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--container-digest", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {args.output}")

    units = []
    for run_dir in args.branch_run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        if not summary.get("deterministic_tokenizer") or not summary.get("replay_exact"):
            raise ValueError(f"{run_dir} is not a deterministic exact-replay RoboCasa unit")
        branch = np.load(run_dir / "branches.npz", allow_pickle=False)
        normalized_actions = branch["normalized_branch_actions"]
        physical_features = branch["endpoint_physical_features"]
        left, right, selection = select_primary_pair(
            normalized_actions,
            physical_features,
            minimum_action_l2=args.minimum_action_l2,
            minimum_endpoint_l2=args.minimum_endpoint_l2,
        )
        action_distances = pairwise_l2(normalized_actions)
        endpoint_distances = pairwise_l2(physical_features)
        units.append(
            {
                "unit_id": run_dir.name,
                "branch_run_dir": str(run_dir),
                "task_name": summary["task_name"],
                "episode_index": int(summary["episode_index"]),
                "prefix_chunks": int(summary["prefix_chunks"]),
                "primary_pair": [left, right],
                "selection": selection,
                "branch_seeds": branch["branch_seeds"].astype(int).tolist(),
                "action_distances": action_distances.tolist(),
                "physical_endpoint_distances": endpoint_distances.tolist(),
                "artifact_sha256": {
                    name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                    for name in ("branches.npz", "summary.json", "model.xml")
                },
            }
        )
    units.sort(key=lambda unit: (unit["task_name"], unit["episode_index"], unit["prefix_chunks"]))
    result = {
        "scope": "RoboCasa replication pairs selected without intervention outcomes",
        "provenance": {
            "project_commit": args.project_commit,
            "cosmos_policy_commit": "18a2accadf4e7a3531e56754102af5a24d2316da",
            "robocasa_commit": "edd9a328b3ec98050f42d194c1419307a79c4d87",
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
