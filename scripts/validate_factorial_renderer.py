"""Smoke-test 2x2 MuJoCo factorization on two saved natural endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagined_future.libero_semantics import (
    goal_feature_vector,
    goal_predicate_snapshot,
)
from imagined_future.sim_state_factorization import (
    factorized_flat_state,
    robot_state_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke result: {args.output}")

    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        prepare_observation,
    )
    from libero.libero import benchmark

    artifact = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    left = np.asarray(artifact["endpoint_states"][args.left]).copy()
    right = np.asarray(artifact["endpoint_states"][args.right]).copy()
    task = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task_id)

    def environment_factory():
        environment, _description = get_libero_env(
            task, "cosmos", resolution=256
        )
        return environment

    environment = environment_factory()
    try:
        environment.reset()
        indices = robot_state_indices(environment.env.robots[0])
        nq = int(environment.env.sim.model.nq)
        nv = int(environment.env.sim.model.nv)
    finally:
        environment.close()
    states = {
        "o0r0": left,
        "o1r0": factorized_flat_state(
            right, left, nq=nq, nv=nv, robot_indices=indices
        ),
        "o0r1": factorized_flat_state(
            left, right, nq=nq, nv=nv, robot_indices=indices
        ),
        "o1r1": right,
    }
    records = {}
    for name, state in states.items():
        environment = environment_factory()
        try:
            environment.reset()
            raw = environment.set_init_state(state)
            observation = prepare_observation(raw, 256, True)
            records[name] = {
                "goal": goal_feature_vector(
                    goal_predicate_snapshot(environment)
                ),
                "proprio": np.asarray(observation["proprio"], dtype=np.float64),
                "contacts": int(environment.env.sim.data.ncon),
            }
        finally:
            environment.close()
    checks = {
        "o1r0_goal_to_o1r1": float(
            np.max(
                np.abs(records["o1r0"]["goal"] - records["o1r1"]["goal"]),
                initial=0.0,
            )
        ),
        "o0r1_goal_to_o0r0": float(
            np.max(
                np.abs(records["o0r1"]["goal"] - records["o0r0"]["goal"]),
                initial=0.0,
            )
        ),
        "o1r0_robot_to_o0r0": float(
            np.max(
                np.abs(
                    records["o1r0"]["proprio"] - records["o0r0"]["proprio"]
                ),
                initial=0.0,
            )
        ),
        "o0r1_robot_to_o1r1": float(
            np.max(
                np.abs(
                    records["o0r1"]["proprio"] - records["o1r1"]["proprio"]
                ),
                initial=0.0,
            )
        ),
    }
    result = {
        "scope": "renderer factorization smoke test; no intervention outcomes",
        "task_id": args.task_id,
        "branch_run": str(args.branch_run_dir),
        "left": args.left,
        "right": args.right,
        "nq": nq,
        "nv": nv,
        "robot_qpos_indices": indices.qpos,
        "robot_qvel_indices": indices.qvel,
        "contacts": {
            name: record["contacts"] for name, record in records.items()
        },
        "checks": checks,
        "valid": max(checks.values()) <= 1e-6,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
