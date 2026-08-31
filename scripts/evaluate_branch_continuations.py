"""Roll exact-state action branches to termination under one shared continuation policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagined_future.branching import state_digest
from imagined_future.cosmos_config import libero_policy_config
from imagined_future.libero_semantics import goal_predicate_snapshot
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--continuation-seed", type=int, required=True)
    parser.add_argument("--max-policy-steps", type=int, default=520)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")

    from libero.libero import benchmark

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
        unnormalize_actions,
    )
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
    from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation

    branch = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    branch_summary = json.loads((args.branch_run_dir / "summary.json").read_text())
    if branch_summary["task_id"] != args.task_id:
        raise ValueError("task id differs from the branch artifact")
    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    dummy_action = get_libero_dummy_action("cosmos")
    expected_branch_digest = state_digest(branch["branch_state"])

    def env_factory():
        environment, _description = get_libero_env(task, "cosmos", resolution=256)
        return environment

    def restore(environment):
        environment.reset()
        raw = environment.set_init_state(branch["initial_state"].copy())
        for _ in range(args.warmup_steps):
            raw, _reward, _done, _info = environment.step(dummy_action)
        for action in branch["prefix_actions"]:
            raw, _reward, _done, _info = environment.step(action.tolist())
        digest = state_digest(environment.get_sim_state())
        if digest != expected_branch_digest:
            raise RuntimeError(f"branch-point replay changed: {digest} != {expected_branch_digest}")
        return raw

    outcomes = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, (branch_seed, initial_actions, expected_endpoint) in enumerate(
        zip(branch["branch_seeds"], branch["branch_actions"], branch["endpoint_states"], strict=True)
    ):
        environment = env_factory()
        try:
            raw_observation = restore(environment)
            policy_step = 0
            success = False
            success_step = None
            query_trajectory = []
            for action in initial_actions:
                raw_observation, _reward, done, _info = environment.step(action.tolist())
                policy_step += 1
                if done or environment.check_success():
                    success = True
                    success_step = policy_step
                    break
            observed_endpoint = np.asarray(environment.get_sim_state()).copy()
            endpoint_error = float(np.max(np.abs(observed_endpoint - expected_endpoint)))
            if endpoint_error != 0.0:
                raise RuntimeError(f"branch {index} immediate endpoint replay changed: {endpoint_error}")
            query_trajectory.append(
                {
                    "query_index": 0,
                    "query_step": 0,
                    "kind": "presaved_initial_branch",
                    "endpoint": goal_predicate_snapshot(environment),
                }
            )

            continuation_query = 0
            while policy_step < args.max_policy_steps and not success:
                observation = prepare_observation(raw_observation, 256, cfg.flip_images)
                result = get_action(
                    cfg=cfg,
                    model=model,
                    dataset_stats=dataset_stats,
                    obs=observation,
                    task_label_or_embedding=task.language,
                    seed=args.continuation_seed,
                    randomize_seed=False,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=True,
                )
                normalized_raw = np.asarray(result["actions"])
                actions = unnormalize_actions(normalized_raw.copy(), dataset_stats)
                del result
                query_start = policy_step
                for action in actions:
                    if policy_step >= args.max_policy_steps:
                        break
                    raw_observation, _reward, done, _info = environment.step(action.tolist())
                    policy_step += 1
                    if done or environment.check_success():
                        success = True
                        success_step = policy_step
                        break
                continuation_query += 1
                query_trajectory.append(
                    {
                        "query_index": continuation_query,
                        "query_step": query_start,
                        "kind": "shared_continuation",
                        "endpoint": goal_predicate_snapshot(environment),
                    }
                )
        finally:
            environment.close()
        outcome = {
            "branch_index": index,
            "branch_seed": int(branch_seed),
            "continuation_seed": args.continuation_seed,
            "success": success,
            "success_step": success_step,
            "policy_steps": policy_step,
            "queries": query_trajectory,
        }
        outcomes.append(outcome)
        print(json.dumps({key: outcome[key] for key in outcome if key != "queries"}), flush=True)

    result = {
        "scope": "exact-state branches with a shared deterministic continuation policy",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_description": task.language,
        "initial_state_index": branch_summary["initial_state_index"],
        "branch_run": str(args.branch_run_dir),
        "branch_state_digest": expected_branch_digest,
        "continuation_seed": args.continuation_seed,
        "max_policy_steps": args.max_policy_steps,
        "successes": sum(outcome["success"] for outcome in outcomes),
        "failures": sum(not outcome["success"] for outcome in outcomes),
        "outcomes": outcomes,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "successes": result["successes"], "failures": result["failures"]}))


if __name__ == "__main__":
    main()
