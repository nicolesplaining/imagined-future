"""Audit natural branches for robot/object endpoint disentanglement feasibility.

This script reads natural branch artifacts only.  It does not read semantic
intervention outputs and is intended for calibration sets that will be
excluded from subsequent confirmatory estimation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagined_future.libero_semantics import goal_feature_vector
from imagined_future.study_design import pairwise_l2


def _upper_triangle(values: np.ndarray) -> np.ndarray:
    rows, columns = np.triu_indices(values.shape[0], k=1)
    return np.asarray(values[rows, columns], dtype=np.float64)


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _pairwise_quaternion_angle(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("robot endpoint contains a zero quaternion")
    unit = values / norms
    absolute_cosine = np.clip(np.abs(unit @ unit.T), 0.0, 1.0)
    return 2.0 * np.arccos(absolute_cosine)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-action-l2", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")

    pooled: dict[str, list[float]] = {
        "action_l2": [],
        "robot_proprio_l2": [],
        "robot_position_l2_m": [],
        "robot_orientation_angle_rad": [],
        "robot_gripper_l2": [],
        "object_goal_l2": [],
    }
    units = []
    for run_dir in sorted(args.branch_run_dirs):
        summary = json.loads((run_dir / "summary.json").read_text())
        artifact = np.load(run_dir / "branches.npz", allow_pickle=False)
        predicate_records = json.loads(
            (run_dir / "endpoint_predicates.json").read_text()
        )
        object_features = np.stack(
            [goal_feature_vector(record["snapshot"]) for record in predicate_records]
        )
        robot_features = np.asarray(artifact["endpoint_proprios"], dtype=np.float64)
        action_distances = pairwise_l2(artifact["normalized_branch_actions"])
        robot_distances = pairwise_l2(robot_features)
        # Public Cosmos Policy ordering is gripper qpos (2), end-effector
        # position (3), then end-effector quaternion (4).
        robot_gripper_distances = pairwise_l2(robot_features[:, :2])
        robot_position_distances = pairwise_l2(robot_features[:, 2:5])
        robot_orientation_distances = _pairwise_quaternion_angle(robot_features[:, 5:9])
        object_distances = pairwise_l2(object_features)

        pair_rows = []
        for left in range(len(robot_features)):
            for right in range(left + 1, len(robot_features)):
                row = {
                    "left": left,
                    "right": right,
                    "left_seed": int(artifact["branch_seeds"][left]),
                    "right_seed": int(artifact["branch_seeds"][right]),
                    "action_l2": float(action_distances[left, right]),
                    "robot_proprio_l2": float(robot_distances[left, right]),
                    "robot_position_l2_m": float(robot_position_distances[left, right]),
                    "robot_orientation_angle_rad": float(
                        robot_orientation_distances[left, right]
                    ),
                    "robot_gripper_l2": float(robot_gripper_distances[left, right]),
                    "object_goal_l2": float(object_distances[left, right]),
                }
                if row["action_l2"] >= args.minimum_action_l2:
                    pair_rows.append(row)
                    for name, values in pooled.items():
                        values.append(row[name])

        eligible_object = [row["object_goal_l2"] for row in pair_rows]
        eligible_robot = [row["robot_proprio_l2"] for row in pair_rows]
        correlation = None
        if np.std(eligible_object) > 0 and np.std(eligible_robot) > 0:
            correlation = float(np.corrcoef(eligible_object, eligible_robot)[0, 1])
        units.append(
            {
                "unit_id": run_dir.name,
                "task_id": int(summary["task_id"]),
                "task_description": summary["task_description"],
                "initial_state_index": int(summary["initial_state_index"]),
                "prefix_chunks": int(summary["prefix_chunks"]),
                "branches": len(robot_features),
                "eligible_pairs": len(pair_rows),
                "robot_object_distance_correlation": correlation,
                "distance_summary": {
                    "action_l2": _quantiles([row["action_l2"] for row in pair_rows]),
                    "robot_proprio_l2": _quantiles(eligible_robot),
                    "robot_position_l2_m": _quantiles(
                        [row["robot_position_l2_m"] for row in pair_rows]
                    ),
                    "robot_orientation_angle_rad": _quantiles(
                        [row["robot_orientation_angle_rad"] for row in pair_rows]
                    ),
                    "robot_gripper_l2": _quantiles(
                        [row["robot_gripper_l2"] for row in pair_rows]
                    ),
                    "object_goal_l2": _quantiles(eligible_object),
                },
                "pairs": pair_rows,
            }
        )

    result = {
        "scope": "natural-branch calibration audit; no intervention outcomes read",
        "minimum_action_l2": args.minimum_action_l2,
        "units": units,
        "pooled_distance_summary": {
            name: _quantiles(values) for name, values in pooled.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "units": len(units)}, indent=2))


if __name__ == "__main__":
    main()
