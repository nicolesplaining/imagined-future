"""Execute saved action chunks from an exact LIBERO branch point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from imagined_future.branching import state_digest
from imagined_future.libero_semantics import goal_predicate_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--action-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")

    from libero.libero import benchmark

    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
    from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation

    branch = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    actions = np.load(args.action_artifact, allow_pickle=False)
    names = sorted(key.removeprefix("environment_") for key in actions.files if key.startswith("environment_"))
    if not names:
        raise ValueError("action artifact has no environment_* arrays")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    dummy_action = get_libero_dummy_action("cosmos")

    def env_factory():
        environment, _description = get_libero_env(task, "cosmos", resolution=256)
        return environment

    def restore(environment):
        environment.reset()
        raw_observation = environment.set_init_state(branch["initial_state"].copy())
        for _ in range(args.warmup_steps):
            raw_observation, _reward, _done, _info = environment.step(dummy_action)
        for action in branch["prefix_actions"]:
            raw_observation, _reward, _done, _info = environment.step(action.tolist())
        return raw_observation

    executions = []
    endpoint_states = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_digest = state_digest(branch["branch_state"])
    for name in names:
        environment = env_factory()
        try:
            restore(environment)
            restored_digest = state_digest(environment.get_sim_state())
            if restored_digest != expected_digest:
                raise RuntimeError(f"{name} branch-point replay changed: {restored_digest} != {expected_digest}")
            initial = goal_predicate_snapshot(environment)
            trajectory = []
            first_success_step = None
            for step, action in enumerate(actions[f"environment_{name}"], start=1):
                raw_observation, _reward, _done, _info = environment.step(action.tolist())
                snapshot = goal_predicate_snapshot(environment)
                snapshot["step"] = step
                trajectory.append(snapshot)
                if snapshot["success"] and first_success_step is None:
                    first_success_step = step
            endpoint_state = np.asarray(environment.get_sim_state()).copy()
            endpoint = prepare_observation(raw_observation, 256, True)
        finally:
            environment.close()
        endpoint_states.append(endpoint_state)
        Image.fromarray(np.asarray(endpoint["primary_image"], dtype=np.uint8)).save(
            args.output.parent / f"{args.output.stem}_{name}_endpoint_primary.png"
        )
        executions.append(
            {
                "name": name,
                "branch_state_digest": restored_digest,
                "first_success_step": first_success_step,
                "initial": initial,
                "trajectory": trajectory,
                "endpoint": trajectory[-1],
            }
        )

    result = {
        "scope": "exact-state execution of a saved action artifact",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_description": task.language,
        "branch_run": str(args.branch_run_dir),
        "action_artifact": str(args.action_artifact),
        "branch_state_digest": expected_digest,
        "executions": executions,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output.with_name(f"{args.output.stem}_endpoint_states.npz"),
        names=np.asarray(names),
        endpoint_states=np.stack(endpoint_states),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "executions": [
                    {
                        "name": execution["name"],
                        "first_success_step": execution["first_success_step"],
                        "endpoint_success": execution["endpoint"]["success"],
                    }
                    for execution in executions
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
