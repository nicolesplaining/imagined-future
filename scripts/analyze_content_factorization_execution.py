"""Analyze executed task-world and robot endpoints for factorization interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from imagined_future.libero_semantics import goal_feature_vector
from imagined_future.metrics import donor_steering


def _steering(
    value: np.ndarray, recipient: np.ndarray, donor: np.ndarray
) -> float | None:
    recipient_array = np.asarray(recipient, dtype=np.float64)
    donor_array = np.asarray(donor, dtype=np.float64)
    if np.linalg.norm(donor_array - recipient_array) <= 1e-12:
        return None
    return float(
        donor_steering(
            torch.from_numpy(np.asarray(value, dtype=np.float64)).unsqueeze(0),
            torch.from_numpy(recipient_array).unsqueeze(0),
            torch.from_numpy(donor_array).unsqueeze(0),
        ).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--semantic-run-dir", type=Path, required=True)
    parser.add_argument("--execution-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")

    semantic = json.loads((args.semantic_run_dir / "summary.json").read_text())
    branch = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    predicates = json.loads(
        (args.branch_run_dir / "endpoint_predicates.json").read_text()
    )
    reference_goal = [goal_feature_vector(record["snapshot"]) for record in predicates]
    reference_robot = np.asarray(branch["endpoint_proprios"], dtype=np.float64)
    execution = json.loads(args.execution_artifact.read_text())
    endpoint_arrays = np.load(
        args.execution_artifact.with_name(
            f"{args.execution_artifact.stem}_endpoint_states.npz"
        ),
        allow_pickle=False,
    )
    proprios = {
        str(name): value
        for name, value in zip(
            endpoint_arrays["names"], endpoint_arrays["endpoint_proprios"], strict=True
        )
    }
    selection = semantic["selection"]
    anchor = int(selection["recipient"])
    object_donor = int(selection["object_donor"])
    robot_donor = int(selection["robot_donor"])
    pairs = {
        "object_forward": (anchor, object_donor),
        "object_reverse": (object_donor, anchor),
        "robot_forward": (anchor, robot_donor),
        "robot_reverse": (robot_donor, anchor),
    }
    row_metadata = {
        row["condition"]: row
        for row in semantic["rows"]
        if row.get("condition") is not None
    }
    rows = []
    for item in execution["executions"]:
        name = item["name"]
        metadata = row_metadata[name]
        recipient, donor = pairs[metadata["context"]]
        goal = goal_feature_vector(item["endpoint"])
        robot = proprios[name]
        row = {
            "condition": name,
            "context": metadata["context"],
            "pair_type": metadata["pair_type"],
            "direction": metadata["direction"],
            "future_noise_seed": metadata["future_noise_seed"],
            "target_role": metadata["target_role"],
            "recipient_branch": recipient,
            "donor_branch": donor,
            "goal_endpoint_donor_steering": _steering(
                goal, reference_goal[recipient], reference_goal[donor]
            ),
            "robot_endpoint_donor_steering": _steering(
                robot, reference_robot[recipient], reference_robot[donor]
            ),
            "endpoint_success": bool(item["endpoint"]["success"]),
            "first_success_step": item["first_success_step"],
        }
        rows.append(row)
    result = {
        "scope": "executed robot-versus-object factorization endpoints",
        "unit_id": semantic["unit_id"],
        "branch_run": str(args.branch_run_dir),
        "semantic_run": str(args.semantic_run_dir),
        "execution_artifact": str(args.execution_artifact),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
