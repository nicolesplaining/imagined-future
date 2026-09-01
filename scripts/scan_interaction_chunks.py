"""Screen held-out LIBERO trajectories for natural task-world interaction chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagined_future.cosmos_config import (
    deterministic_tokenizer_enabled,
    libero_policy_config,
)
from imagined_future.libero_semantics import (
    goal_feature_vector,
    goal_predicate_snapshot,
)
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--initial-state-indices", type=int, nargs="+", required=True)
    parser.add_argument("--prefix-seed", type=int, default=503)
    parser.add_argument("--maximum-prefix-chunks", type=int, default=32)
    parser.add_argument("--candidate-chunks", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite screen: {args.output}")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
        unnormalize_actions,
    )
    from cosmos_policy.experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
    )
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        prepare_observation,
    )
    from libero.libero import benchmark

    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    dummy_action = get_libero_dummy_action("cosmos")

    units = []
    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        for initial_state_index in args.initial_state_indices:
            environment, _description = get_libero_env(task, "cosmos", resolution=256)
            try:
                environment.reset()
                raw_observation = environment.set_init_state(
                    np.asarray(initial_states[initial_state_index]).copy()
                )
                for _ in range(10):
                    raw_observation, _reward, done, _info = environment.step(
                        dummy_action
                    )
                    if done:
                        raise RuntimeError("task completed during warmup")
                chunks = []
                for chunk_index in range(args.maximum_prefix_chunks):
                    before = goal_predicate_snapshot(environment)
                    before_features = goal_feature_vector(before)
                    observation = prepare_observation(
                        raw_observation, 256, cfg.flip_images
                    )
                    result = get_action(
                        cfg=cfg,
                        model=model,
                        dataset_stats=dataset_stats,
                        obs=observation,
                        task_label_or_embedding=task.language,
                        seed=args.prefix_seed,
                        randomize_seed=False,
                        num_denoising_steps_action=cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=True,
                    )
                    actions = unnormalize_actions(
                        np.asarray(result["actions"]).copy(), dataset_stats
                    )
                    del result
                    first_success_step = None
                    for step, action in enumerate(actions, start=1):
                        raw_observation, _reward, done, _info = environment.step(
                            action.tolist()
                        )
                        if (
                            done or environment.check_success()
                        ) and first_success_step is None:
                            first_success_step = step
                    after = goal_predicate_snapshot(environment)
                    after_features = goal_feature_vector(after)
                    chunks.append(
                        {
                            "prefix_chunks": chunk_index,
                            "goal_feature_l2": float(
                                np.linalg.norm(after_features - before_features)
                            ),
                            "before_success": bool(before["success"]),
                            "after_success": bool(after["success"]),
                            "first_success_step": first_success_step,
                            "predicate_value_changes": [
                                {
                                    "index": index,
                                    "predicate": left["predicate"],
                                    "before": bool(left["value"]),
                                    "after": bool(right["value"]),
                                }
                                for index, (left, right) in enumerate(
                                    zip(
                                        before["predicates"],
                                        after["predicates"],
                                        strict=True,
                                    )
                                )
                                if bool(left["value"]) != bool(right["value"])
                            ],
                        }
                    )
                    if after["success"]:
                        break
                candidate_order = sorted(
                    chunks,
                    key=lambda item: (-item["goal_feature_l2"], item["prefix_chunks"]),
                )[: args.candidate_chunks]
                units.append(
                    {
                        "unit_id": f"task{task_id:02d}_state{initial_state_index:02d}",
                        "task_id": task_id,
                        "task_description": task.language,
                        "initial_state_index": initial_state_index,
                        "chunks_run": len(chunks),
                        "candidate_prefix_chunks": [
                            item["prefix_chunks"] for item in candidate_order
                        ],
                        "chunks": chunks,
                    }
                )
            finally:
                environment.close()

    result = {
        "scope": "held-out natural trajectory screen; no intervention outcomes",
        "suite": args.suite,
        "task_ids": args.task_ids,
        "initial_state_indices": args.initial_state_indices,
        "prefix_seed": args.prefix_seed,
        "maximum_prefix_chunks": args.maximum_prefix_chunks,
        "candidate_chunks": args.candidate_chunks,
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "units": units,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "units": len(units)}, indent=2))


if __name__ == "__main__":
    main()
