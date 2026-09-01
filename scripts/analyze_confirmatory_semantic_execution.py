"""Compute task-grounded endpoints for a confirmatory semantic suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imagined_future.libero_semantics import goal_feature_vector, physical_endpoint_feature_vector
from imagined_future.metrics import donor_steering
from imagined_future.paired_rollouts import pixel_l1


def _steering(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> float:
    return float(
        donor_steering(
            torch.from_numpy(np.asarray(value, dtype=np.float64)).unsqueeze(0),
            torch.from_numpy(np.asarray(recipient, dtype=np.float64)).unsqueeze(0),
            torch.from_numpy(np.asarray(donor, dtype=np.float64)).unsqueeze(0),
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
        raise FileExistsError(f"refusing to overwrite existing analysis: {args.output}")

    semantic = json.loads((args.semantic_run_dir / "summary.json").read_text())
    branch = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    predicate_records = json.loads((args.branch_run_dir / "endpoint_predicates.json").read_text())
    reference_goal = [goal_feature_vector(record["snapshot"]) for record in predicate_records]
    reference_physical = [
        physical_endpoint_feature_vector(record["snapshot"], proprio)
        for record, proprio in zip(predicate_records, branch["endpoint_proprios"], strict=True)
    ]
    execution_json = json.loads(args.execution_artifact.read_text())
    execution_npz = np.load(
        args.execution_artifact.with_name(f"{args.execution_artifact.stem}_endpoint_states.npz"),
        allow_pickle=False,
    )
    execution_by_name = {item["name"]: item for item in execution_json["executions"]}
    proprios_by_name = {
        str(name): proprio
        for name, proprio in zip(execution_npz["names"], execution_npz["endpoint_proprios"], strict=True)
    }

    pair = [int(index) for index in semantic["primary_pair"]]
    direction_pairs = {"forward": (pair[0], pair[1]), "reverse": (pair[1], pair[0])}
    rows = []
    for name, execution in execution_by_name.items():
        direction = next((value for value in direction_pairs if name.startswith(f"{value}_")), None)
        if direction is None:
            continue
        recipient_index, donor_index = direction_pairs[direction]
        endpoint_goal = goal_feature_vector(execution["endpoint"])
        endpoint_proprio = proprios_by_name[name]
        endpoint_physical = physical_endpoint_feature_vector(execution["endpoint"], endpoint_proprio)
        endpoint_image = np.asarray(
            Image.open(args.semantic_run_dir / f"{args.execution_artifact.stem}_{name}_endpoint_primary.png")
        )
        recipient_image = branch["endpoint_primary_images"][recipient_index]
        donor_image = branch["endpoint_primary_images"][donor_index]
        row = {
            "condition": name,
            "direction": direction,
            "recipient_branch": recipient_index,
            "donor_branch": donor_index,
            "physical_endpoint_donor_steering": _steering(
                endpoint_physical,
                reference_physical[recipient_index],
                reference_physical[donor_index],
            ),
            "proprio_endpoint_donor_steering": _steering(
                endpoint_proprio,
                branch["endpoint_proprios"][recipient_index],
                branch["endpoint_proprios"][donor_index],
            ),
            "primary_pixel_l1_to_recipient": pixel_l1(endpoint_image, recipient_image),
            "primary_pixel_l1_to_donor": pixel_l1(endpoint_image, donor_image),
            "endpoint_success": bool(execution["endpoint"]["success"]),
            "first_success_step": execution["first_success_step"],
        }
        if np.linalg.norm(reference_goal[donor_index] - reference_goal[recipient_index]) > 1e-12:
            row["goal_endpoint_donor_steering"] = _steering(
                endpoint_goal, reference_goal[recipient_index], reference_goal[donor_index]
            )
        else:
            row["goal_endpoint_donor_steering"] = None
        row["primary_pixel_donor_preference"] = (
            row["primary_pixel_l1_to_recipient"] - row["primary_pixel_l1_to_donor"]
        )
        rows.append(row)

    result = {
        "scope": "task-grounded confirmatory semantic endpoint analysis",
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
