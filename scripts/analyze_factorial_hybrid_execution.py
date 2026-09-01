"""Analyze executed task-world and robot endpoints for 2x2 interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from imagined_future.libero_semantics import goal_feature_vector
from imagined_future.metrics import donor_steering

CELL_COORDINATES = {
    "o0r0": (0.0, 0.0),
    "o1r0": (1.0, 0.0),
    "o0r1": (0.0, 1.0),
    "o1r1": (1.0, 1.0),
}


def _steering(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> float:
    return float(
        donor_steering(
            torch.from_numpy(np.asarray(value, dtype=np.float64)).unsqueeze(0),
            torch.from_numpy(np.asarray(recipient, dtype=np.float64)).unsqueeze(0),
            torch.from_numpy(np.asarray(donor, dtype=np.float64)).unsqueeze(0),
        ).item()
    )


def _canonical_robot(proprio: np.ndarray, reference: np.ndarray) -> np.ndarray:
    value = np.asarray(proprio, dtype=np.float64).copy()
    reference_value = np.asarray(reference, dtype=np.float64)
    if np.dot(value[5:9], reference_value[5:9]) < 0:
        value[5:9] *= -1
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--semantic-run-dir", type=Path, required=True)
    parser.add_argument("--execution-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")

    semantic = json.loads((args.semantic_run_dir / "summary.json").read_text())
    target = np.load(args.target_dir / "branches.npz", allow_pickle=False)
    predicates = json.loads(
        (args.target_dir / "endpoint_predicates.json").read_text()
    )
    goal_by_cell = {
        record["cell"]: goal_feature_vector(record["snapshot"])
        for record in predicates
    }
    robot_by_cell = {
        str(name): value
        for name, value in zip(
            target["cell_names"], target["cell_proprios"], strict=True
        )
    }
    execution = json.loads(args.execution_artifact.read_text())
    endpoint_arrays = np.load(
        args.execution_artifact.with_name(
            f"{args.execution_artifact.stem}_endpoint_states.npz"
        ),
        allow_pickle=False,
    )
    executed_robot = {
        str(name): value
        for name, value in zip(
            endpoint_arrays["names"],
            endpoint_arrays["endpoint_proprios"],
            strict=True,
        )
    }
    metadata = {row["condition"]: row for row in semantic["rows"]}
    rows = []
    for item in execution["executions"]:
        name = item["name"]
        row_metadata = metadata[name]
        goal = goal_feature_vector(item["endpoint"])
        robot = executed_robot[name]
        goal_steering = _steering(goal, goal_by_cell["o0r0"], goal_by_cell["o1r1"])
        recipient_robot = robot_by_cell["o0r0"]
        robot_steering = _steering(
            _canonical_robot(robot, recipient_robot),
            recipient_robot,
            _canonical_robot(robot_by_cell["o1r1"], recipient_robot),
        )
        target_cell = row_metadata["target_cell"]
        target_identification = None
        target_coordinate_distance = None
        if target_cell in CELL_COORDINATES:
            distances = {
                cell: float(
                    np.hypot(goal_steering - coordinates[0], robot_steering - coordinates[1])
                )
                for cell, coordinates in CELL_COORDINATES.items()
            }
            minimum = min(distances.values())
            target_coordinate_distance = distances[target_cell]
            target_identification = float(
                np.isclose(distances[target_cell], minimum)
            )
        rows.append(
            {
                "condition": name,
                "future_noise_seed": row_metadata["future_noise_seed"],
                "modality": row_metadata["modality"],
                "target_cell": target_cell,
                "goal_endpoint_donor_steering": goal_steering,
                "robot_endpoint_donor_steering": robot_steering,
                "target_coordinate_distance": target_coordinate_distance,
                "correct_target_cell": target_identification,
                "endpoint_success": bool(item["endpoint"]["success"]),
                "first_success_step": item["first_success_step"],
            }
        )
    result = {
        "scope": "executed endpoints for prospective 2x2 rendered object/robot targets",
        "unit_id": semantic["unit_id"],
        "target_dir": str(args.target_dir),
        "semantic_run": str(args.semantic_run_dir),
        "execution_artifact": str(args.execution_artifact),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
