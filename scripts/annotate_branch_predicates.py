"""Replay a branch-calibration artifact and add LIBERO predicate trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagined_future.branching import state_digest
from imagined_future.libero_semantics import goal_predicate_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    from libero.libero import benchmark

    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env

    artifact = np.load(args.run_dir / "branches.npz", allow_pickle=False)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    dummy_action = get_libero_dummy_action("cosmos")

    def env_factory():
        environment, _description = get_libero_env(task, "cosmos", resolution=256)
        return environment

    def restore_branch_point(environment):
        environment.reset()
        environment.set_init_state(artifact["initial_state"].copy())
        for _ in range(args.warmup_steps):
            environment.step(dummy_action)
        for action in artifact["prefix_actions"]:
            environment.step(action.tolist())

    probe = env_factory()
    try:
        restore_branch_point(probe)
        branch_point = goal_predicate_snapshot(probe)
        restored_digest = state_digest(probe.get_sim_state())
    finally:
        probe.close()
    expected_branch_digest = state_digest(artifact["branch_state"])
    if restored_digest != expected_branch_digest:
        raise RuntimeError(
            f"branch-point replay changed: expected {expected_branch_digest}, got {restored_digest}"
        )

    branches = []
    for index, (seed, actions, expected_endpoint) in enumerate(
        zip(artifact["branch_seeds"], artifact["branch_actions"], artifact["endpoint_states"], strict=True)
    ):
        environment = env_factory()
        try:
            restore_branch_point(environment)
            trajectory = []
            first_success_step = None
            for step, action in enumerate(actions, start=1):
                environment.step(action.tolist())
                snapshot = goal_predicate_snapshot(environment)
                snapshot["step"] = step
                trajectory.append(snapshot)
                if snapshot["success"] and first_success_step is None:
                    first_success_step = step
            observed_endpoint = np.asarray(environment.get_sim_state())
        finally:
            environment.close()
        maximum_error = float(np.max(np.abs(observed_endpoint - expected_endpoint)))
        if maximum_error != 0.0:
            raise RuntimeError(f"branch {index} endpoint replay changed (max abs error {maximum_error})")
        branches.append(
            {
                "index": index,
                "seed": int(seed),
                "first_success_step": first_success_step,
                "endpoint_success": trajectory[-1]["success"],
                "trajectory": trajectory,
            }
        )

    output = {
        "scope": "simulator-grounded replay annotation; no model inference",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_description": task.language,
        "branch_state_digest": restored_digest,
        "branch_point": branch_point,
        "branches": branches,
    }
    destination = args.run_dir / "predicate_annotations.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "destination": str(destination),
                "branch_point": branch_point,
                "branches": [
                    {
                        "index": branch["index"],
                        "seed": branch["seed"],
                        "first_success_step": branch["first_success_step"],
                        "endpoint_success": branch["endpoint_success"],
                    }
                    for branch in branches
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
