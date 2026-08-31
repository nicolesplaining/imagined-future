"""Screen LIBERO initial states for natural Cosmos Policy success/failure variation."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from imagined_future.branching import state_digest
from imagined_future.cosmos_config import deterministic_tokenizer_enabled, libero_policy_config
from imagined_future.libero_semantics import goal_predicate_snapshot
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution


def _stack(values: list[np.ndarray], *, empty_shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    return np.stack(values) if values else np.empty((0, *empty_shape), dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--initial-state-indices", type=int, nargs="+", required=True)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[195])
    parser.add_argument("--max-policy-steps", type=int, default=520)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

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

    if len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError("task ids must be unique")
    if len(set(args.initial_state_indices)) != len(args.initial_state_indices):
        raise ValueError("initial-state indices must be unique")
    if len(set(args.model_seeds)) != len(args.model_seeds):
        raise ValueError("model seeds must be unique")

    args.output_root.mkdir(parents=True, exist_ok=True)
    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    dummy_action = get_libero_dummy_action("cosmos")
    episode_summaries = []
    started = time.time()

    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        initial_states = np.asarray(suite.get_task_init_states(task_id))
        invalid = [index for index in args.initial_state_indices if not 0 <= index < len(initial_states)]
        if invalid:
            raise IndexError(f"task {task_id} has {len(initial_states)} initial states; invalid indices: {invalid}")

        def env_factory():
            environment, _description = get_libero_env(task, "cosmos", resolution=256)
            return environment

        for initial_state_index in args.initial_state_indices:
            for model_seed in args.model_seeds:
                episode_dir = (
                    args.output_root
                    / f"task_{task_id:02d}"
                    / f"state_{initial_state_index:02d}"
                    / f"seed_{model_seed}"
                )
                if (episode_dir / "summary.json").exists():
                    raise FileExistsError(f"refusing to overwrite completed episode: {episode_dir}")
                episode_dir.mkdir(parents=True, exist_ok=True)
                episode_started = time.time()
                environment = env_factory()
                try:
                    initial_state = initial_states[initial_state_index].copy()
                    environment.reset()
                    raw_observation = environment.set_init_state(initial_state.copy())
                    for _ in range(args.warmup_steps):
                        raw_observation, _reward, done, _info = environment.step(dummy_action)
                        if done:
                            raise RuntimeError("task completed during warmup")

                    warm_branch_state = np.asarray(environment.get_sim_state()).copy()
                    query_steps = []
                    current_states = []
                    endpoint_states = []
                    normalized_chunks = []
                    environment_chunks = []
                    executed_counts = []
                    predicted_values = []
                    current_proprios = []
                    endpoint_proprios = []
                    predicate_trajectory = []
                    current_primary_images = []
                    current_wrist_images = []
                    predicted_primary_images = []
                    predicted_wrist_images = []
                    endpoint_primary_images = []
                    endpoint_wrist_images = []
                    executed_actions = []
                    success = False
                    success_step = None
                    policy_step = 0

                    while policy_step < args.max_policy_steps and not success:
                        current = prepare_observation(raw_observation, 256, cfg.flip_images)
                        current_state = np.asarray(environment.get_sim_state()).copy()
                        current_predicates = goal_predicate_snapshot(environment)
                        result = get_action(
                            cfg=cfg,
                            model=model,
                            dataset_stats=dataset_stats,
                            obs=current,
                            task_label_or_embedding=task.language,
                            seed=model_seed,
                            randomize_seed=False,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=True,
                        )
                        # Preserve the official float32 -> unnormalize operation order.
                        # Casting before unnormalization changes actions by ~1e-8 and can
                        # produce different contact trajectories after many open-loop steps.
                        normalized_raw = np.asarray(result["actions"])
                        action_chunk = unnormalize_actions(normalized_raw.copy(), dataset_stats)
                        normalized = normalized_raw.astype(np.float64)
                        predictions = result["future_image_predictions"]

                        query_steps.append(policy_step)
                        current_states.append(current_state)
                        normalized_chunks.append(normalized)
                        environment_chunks.append(action_chunk)
                        predicted_values.append(float(result["value_prediction"]))
                        current_proprios.append(np.asarray(current["proprio"], dtype=np.float64))
                        if args.save_images:
                            current_primary_images.append(np.asarray(current["primary_image"], dtype=np.uint8))
                            current_wrist_images.append(np.asarray(current["wrist_image"], dtype=np.uint8))
                            predicted_primary_images.append(np.asarray(predictions["future_image"], dtype=np.uint8))
                            predicted_wrist_images.append(np.asarray(predictions["future_wrist_image"], dtype=np.uint8))
                        del result

                        count = 0
                        for action in action_chunk:
                            if policy_step >= args.max_policy_steps:
                                break
                            raw_observation, _reward, done, _info = environment.step(action.tolist())
                            executed_actions.append(action.copy())
                            policy_step += 1
                            count += 1
                            if done or environment.check_success():
                                success = True
                                success_step = policy_step
                                break
                        endpoint = prepare_observation(raw_observation, 256, cfg.flip_images)
                        endpoint_state = np.asarray(environment.get_sim_state()).copy()
                        endpoint_predicates = goal_predicate_snapshot(environment)
                        endpoint_states.append(endpoint_state)
                        endpoint_proprios.append(np.asarray(endpoint["proprio"], dtype=np.float64))
                        executed_counts.append(count)
                        predicate_trajectory.append(
                            {
                                "query_index": len(query_steps) - 1,
                                "query_step": query_steps[-1],
                                "executed_count": count,
                                "current": current_predicates,
                                "endpoint": endpoint_predicates,
                            }
                        )
                        if args.save_images:
                            endpoint_primary_images.append(np.asarray(endpoint["primary_image"], dtype=np.uint8))
                            endpoint_wrist_images.append(np.asarray(endpoint["wrist_image"], dtype=np.uint8))

                    state_dimension = warm_branch_state.shape[0]
                    arrays = {
                        "initial_state": initial_state,
                        "warm_branch_state": warm_branch_state,
                        "query_steps": np.asarray(query_steps, dtype=np.int64),
                        "current_states": _stack(
                            current_states, empty_shape=(state_dimension,), dtype=warm_branch_state.dtype
                        ),
                        "endpoint_states": _stack(
                            endpoint_states, empty_shape=(state_dimension,), dtype=warm_branch_state.dtype
                        ),
                        "normalized_action_chunks": _stack(
                            normalized_chunks, empty_shape=(cfg.chunk_size, 7), dtype=np.float64
                        ),
                        "environment_action_chunks": _stack(
                            environment_chunks, empty_shape=(cfg.chunk_size, 7), dtype=np.float64
                        ),
                        "executed_counts": np.asarray(executed_counts, dtype=np.int64),
                        "executed_actions": _stack(executed_actions, empty_shape=(7,), dtype=np.float64),
                        "predicted_values": np.asarray(predicted_values, dtype=np.float64),
                        "current_proprios": _stack(current_proprios, empty_shape=(9,), dtype=np.float64),
                        "endpoint_proprios": _stack(endpoint_proprios, empty_shape=(9,), dtype=np.float64),
                    }
                    if args.save_images:
                        arrays.update(
                            current_primary_images=np.stack(current_primary_images),
                            current_wrist_images=np.stack(current_wrist_images),
                            predicted_primary_images=np.stack(predicted_primary_images),
                            predicted_wrist_images=np.stack(predicted_wrist_images),
                            endpoint_primary_images=np.stack(endpoint_primary_images),
                            endpoint_wrist_images=np.stack(endpoint_wrist_images),
                        )
                    np.savez_compressed(episode_dir / "rollout.npz", **arrays)
                    (episode_dir / "predicates.json").write_text(
                        json.dumps(predicate_trajectory, indent=2, sort_keys=True) + "\n"
                    )
                    summary = {
                        "scope": "natural deterministic rollout screen",
                        "suite": args.suite,
                        "task_id": task_id,
                        "task_description": task.language,
                        "initial_state_index": initial_state_index,
                        "model_seed": model_seed,
                        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
                        "warmup_steps": args.warmup_steps,
                        "max_policy_steps": args.max_policy_steps,
                        "policy_steps": policy_step,
                        "num_queries": len(query_steps),
                        "success": success,
                        "success_step": success_step,
                        "initial_state_digest": state_digest(initial_state),
                        "warm_branch_state_digest": state_digest(warm_branch_state),
                        "save_images": args.save_images,
                        "elapsed_seconds": time.time() - episode_started,
                    }
                except Exception as error:
                    summary = {
                        "scope": "natural deterministic rollout screen",
                        "suite": args.suite,
                        "task_id": task_id,
                        "task_description": task.language,
                        "initial_state_index": initial_state_index,
                        "model_seed": model_seed,
                        "success": False,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                        "elapsed_seconds": time.time() - episode_started,
                    }
                finally:
                    environment.close()
                (episode_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                episode_summaries.append(summary)
                print(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "initial_state_index": initial_state_index,
                            "model_seed": model_seed,
                            "success": summary["success"],
                            "policy_steps": summary.get("policy_steps"),
                            "error": summary.get("error"),
                            "elapsed_seconds": summary["elapsed_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    completed = [summary for summary in episode_summaries if "error" not in summary]
    screen_summary = {
        "scope": "natural deterministic rollout screen",
        "suite": args.suite,
        "task_ids": args.task_ids,
        "initial_state_indices": args.initial_state_indices,
        "model_seeds": args.model_seeds,
        "save_images": args.save_images,
        "episodes_requested": len(args.task_ids) * len(args.initial_state_indices) * len(args.model_seeds),
        "episodes_completed": len(completed),
        "errors": len(episode_summaries) - len(completed),
        "successes": sum(int(summary["success"]) for summary in completed),
        "elapsed_seconds": time.time() - started,
        "episodes": episode_summaries,
    }
    (args.output_root / "screen_summary.json").write_text(
        json.dumps(screen_summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: screen_summary[key] for key in screen_summary if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
