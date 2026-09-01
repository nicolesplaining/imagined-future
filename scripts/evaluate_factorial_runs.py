"""Batch-execute saved 2x2 action chunks from exact LIBERO branch states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from imagined_future.branching import state_digest
from imagined_future.libero_semantics import goal_predicate_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    from libero.libero import benchmark

    from cosmos_policy.experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
    )
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        prepare_observation,
    )

    units = []
    for manifest_path in args.manifests:
        units.extend(
            unit
            for unit in json.loads(manifest_path.read_text())["units"]
            if unit["valid"]
        )
    suite = benchmark.get_benchmark_dict()[args.suite]()
    dummy_action = get_libero_dummy_action("cosmos")
    completed = []
    for unit in units:
        run_dir = args.runs_root / unit["unit_id"]
        output = run_dir / "execution.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite execution: {output}")
        branch_dir = Path(unit["target_dir"])
        branch = np.load(branch_dir / "branches.npz", allow_pickle=False)
        actions = np.load(run_dir / "execution_actions.npz", allow_pickle=False)
        names = sorted(
            key.removeprefix("environment_")
            for key in actions.files
            if key.startswith("environment_")
        )
        if not names:
            raise ValueError(f"{unit['unit_id']} has no environment action arrays")
        task = suite.get_task(unit["task_id"])

        def environment_factory(_task=task):
            environment, _description = get_libero_env(
                _task, "cosmos", resolution=256
            )
            return environment

        def restore(environment, _branch=branch):
            environment.reset()
            raw = environment.set_init_state(_branch["initial_state"].copy())
            for _ in range(args.warmup_steps):
                raw, _reward, _done, _info = environment.step(dummy_action)
            for action in _branch["prefix_actions"]:
                raw, _reward, _done, _info = environment.step(action.tolist())
            return raw

        expected_digest = state_digest(branch["branch_state"])
        executions = []
        endpoint_states = []
        endpoint_proprios = []
        for name in names:
            environment = environment_factory()
            try:
                restore(environment)
                restored_digest = state_digest(environment.get_sim_state())
                if restored_digest != expected_digest:
                    raise RuntimeError(
                        f"{unit['unit_id']} {name} replay changed: "
                        f"{restored_digest} != {expected_digest}"
                    )
                initial = goal_predicate_snapshot(environment)
                trajectory = []
                first_success_step = None
                for step, action in enumerate(
                    actions[f"environment_{name}"], start=1
                ):
                    raw, _reward, _done, _info = environment.step(action.tolist())
                    snapshot = goal_predicate_snapshot(environment)
                    snapshot["step"] = step
                    trajectory.append(snapshot)
                    if snapshot["success"] and first_success_step is None:
                        first_success_step = step
                endpoint_states.append(
                    np.asarray(environment.get_sim_state()).copy()
                )
                endpoint = prepare_observation(raw, 256, True)
                endpoint_proprios.append(
                    np.asarray(endpoint["proprio"], dtype=np.float64)
                )
            finally:
                environment.close()
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
            "scope": "batch exact-state execution of 2x2 action chunks",
            "suite": args.suite,
            "task_id": unit["task_id"],
            "task_description": task.language,
            "branch_run": str(branch_dir),
            "action_artifact": str(run_dir / "execution_actions.npz"),
            "branch_state_digest": expected_digest,
            "executions": executions,
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        np.savez_compressed(
            output.with_name("execution_endpoint_states.npz"),
            names=np.asarray(names),
            endpoint_states=np.stack(endpoint_states),
            endpoint_proprios=np.stack(endpoint_proprios),
        )
        completed.append(unit["unit_id"])
    print(json.dumps({"runs_root": str(args.runs_root), "completed": completed}, indent=2))


if __name__ == "__main__":
    main()
