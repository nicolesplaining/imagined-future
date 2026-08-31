"""Collect same-state LIBERO action branches for donor-pair calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from imagined_future.branching import BranchPoint, state_digest, tuple_actions, validate_replay_stability
from imagined_future.cosmos_config import libero_policy_config
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(values.shape[0], -1).astype(np.float64)
    return np.linalg.norm(flattened[:, None, :] - flattened[None, :, :], axis=-1)


def _off_diagonal_summary(distances: np.ndarray) -> dict[str, float]:
    values = distances[np.triu_indices(distances.shape[0], k=1)]
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }


def _execute_from_point(env_factory, point: BranchPoint, actions: np.ndarray):
    env = env_factory()
    try:
        env.reset()
        observation = env.set_init_state(point.initial_state.copy())
        for action in (*point.warmup_actions, *point.prefix_actions):
            observation, _reward, _done, _info = env.step(action)
        done_observed = False
        for action in actions:
            observation, _reward, done, _info = env.step(action.tolist())
            done_observed = done_observed or bool(done)
        return (
            np.asarray(env.get_sim_state()).copy(),
            observation,
            bool(done_observed or env.check_success()),
        )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--prefix-chunks", type=int, default=0)
    parser.add_argument("--prefix-seed", type=int, default=195)
    parser.add_argument("--branch-seeds", type=int, nargs="+", default=list(range(195, 203)))
    parser.add_argument("--replay-repeats", type=int, default=3)
    args = parser.parse_args()

    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {args.output_dir}")
    if len(set(args.branch_seeds)) != len(args.branch_seeds):
        raise ValueError("branch seeds must be unique")

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

    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    task_description = task.language
    initial_state = np.asarray(suite.get_task_init_states(args.task_id)[args.initial_state_index]).copy()

    def env_factory():
        environment, _description = get_libero_env(task, "cosmos", resolution=256)
        return environment

    dummy_action = get_libero_dummy_action("cosmos")
    warmup_actions = tuple_actions([dummy_action] * 10)
    prefix_actions: list[np.ndarray] = []
    prefix_env = env_factory()
    try:
        prefix_env.reset()
        raw_observation = prefix_env.set_init_state(initial_state.copy())
        for action in warmup_actions:
            raw_observation, _reward, done, _info = prefix_env.step(action)
            if done:
                raise RuntimeError("task completed during warmup")
        for _chunk_index in range(args.prefix_chunks):
            observation = prepare_observation(raw_observation, 256, cfg.flip_images)
            result = get_action(
                cfg=cfg,
                model=model,
                dataset_stats=dataset_stats,
                obs=observation,
                task_label_or_embedding=task_description,
                seed=args.prefix_seed,
                randomize_seed=False,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            normalized_raw = np.asarray(result["actions"])
            actions = unnormalize_actions(normalized_raw.copy(), dataset_stats)
            del result
            for action in actions:
                raw_observation, _reward, done, _info = prefix_env.step(action.tolist())
                prefix_actions.append(action.copy())
                if done:
                    raise RuntimeError("prefix reached task success; choose an earlier branch point")
    finally:
        prefix_env.close()

    point = BranchPoint(
        initial_state=initial_state,
        warmup_actions=warmup_actions,
        prefix_actions=tuple_actions(prefix_actions),
    )
    replay_results = validate_replay_stability(env_factory, point, repeats=args.replay_repeats, atol=0.0)
    branch_state = replay_results[0].state
    branch_observation = prepare_observation(replay_results[0].observation, 256, cfg.flip_images)

    branch_actions = []
    normalized_branch_actions = []
    endpoint_states = []
    endpoint_primary_images = []
    endpoint_wrist_images = []
    endpoint_proprios = []
    predicted_primary_images = []
    predicted_wrist_images = []
    predicted_values = []
    successes = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for branch_index, seed in enumerate(args.branch_seeds):
        result = get_action(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            obs=branch_observation,
            task_label_or_embedding=task_description,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
        )
        normalized_raw = np.asarray(result["actions"])
        actions = unnormalize_actions(normalized_raw.copy(), dataset_stats)
        predictions = result["future_image_predictions"]
        predicted_primary = np.asarray(predictions["future_image"], dtype=np.uint8)
        predicted_wrist = np.asarray(predictions["future_wrist_image"], dtype=np.uint8)
        predicted_value = float(result["value_prediction"])
        del result

        endpoint_state, raw_endpoint, success = _execute_from_point(env_factory, point, actions)
        endpoint = prepare_observation(raw_endpoint, 256, cfg.flip_images)
        endpoint_primary = np.asarray(endpoint["primary_image"], dtype=np.uint8)
        endpoint_wrist = np.asarray(endpoint["wrist_image"], dtype=np.uint8)
        endpoint_proprio = np.asarray(endpoint["proprio"], dtype=np.float64)

        branch_actions.append(actions)
        normalized_branch_actions.append(normalized_raw.astype(np.float64))
        endpoint_states.append(endpoint_state)
        endpoint_primary_images.append(endpoint_primary)
        endpoint_wrist_images.append(endpoint_wrist)
        endpoint_proprios.append(endpoint_proprio)
        predicted_primary_images.append(predicted_primary)
        predicted_wrist_images.append(predicted_wrist)
        predicted_values.append(predicted_value)
        successes.append(success)

        label = f"branch_{branch_index:02d}_seed_{seed}"
        Image.fromarray(endpoint_primary).save(args.output_dir / f"{label}_endpoint_primary.png")
        Image.fromarray(predicted_primary).save(args.output_dir / f"{label}_predicted_primary.png")

    branch_actions_array = np.stack(branch_actions)
    normalized_branch_actions_array = np.stack(normalized_branch_actions)
    endpoint_states_array = np.stack(endpoint_states)
    endpoint_primary_array = np.stack(endpoint_primary_images)
    endpoint_wrist_array = np.stack(endpoint_wrist_images)
    endpoint_proprio_array = np.stack(endpoint_proprios)
    predicted_primary_array = np.stack(predicted_primary_images)
    predicted_wrist_array = np.stack(predicted_wrist_images)
    predicted_values_array = np.asarray(predicted_values, dtype=np.float64)
    successes_array = np.asarray(successes, dtype=np.bool_)

    action_distances = _pairwise_distances(branch_actions_array)
    endpoint_proprio_distances = _pairwise_distances(endpoint_proprio_array)
    endpoint_primary_pixel_l1 = np.abs(
        endpoint_primary_array[:, None].astype(np.float64) - endpoint_primary_array[None].astype(np.float64)
    ).mean(axis=(2, 3, 4))
    predicted_primary_pixel_l1 = np.abs(
        predicted_primary_array[:, None].astype(np.float64) - predicted_primary_array[None].astype(np.float64)
    ).mean(axis=(2, 3, 4))

    np.savez_compressed(
        args.output_dir / "branches.npz",
        initial_state=initial_state,
        prefix_actions=np.asarray(prefix_actions, dtype=np.float64).reshape(-1, 7),
        branch_state=branch_state,
        current_primary_image=np.asarray(branch_observation["primary_image"], dtype=np.uint8),
        current_wrist_image=np.asarray(branch_observation["wrist_image"], dtype=np.uint8),
        current_proprio=np.asarray(branch_observation["proprio"], dtype=np.float64),
        branch_seeds=np.asarray(args.branch_seeds, dtype=np.int64),
        normalized_branch_actions=normalized_branch_actions_array,
        branch_actions=branch_actions_array,
        endpoint_states=endpoint_states_array,
        endpoint_primary_images=endpoint_primary_array,
        endpoint_wrist_images=endpoint_wrist_array,
        endpoint_proprios=endpoint_proprio_array,
        predicted_primary_images=predicted_primary_array,
        predicted_wrist_images=predicted_wrist_array,
        predicted_values=predicted_values_array,
        successes=successes_array,
        action_distances=action_distances,
        endpoint_proprio_distances=endpoint_proprio_distances,
        endpoint_primary_pixel_l1=endpoint_primary_pixel_l1,
        predicted_primary_pixel_l1=predicted_primary_pixel_l1,
    )
    summary: dict[str, Any] = {
        "scope": "same-state branch calibration; not a semantic intervention result",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_description": task_description,
        "initial_state_index": args.initial_state_index,
        "prefix_chunks": args.prefix_chunks,
        "prefix_seed": args.prefix_seed,
        "prefix_action_steps": len(prefix_actions),
        "branch_seeds": args.branch_seeds,
        "branch_state_digest": state_digest(branch_state),
        "replay_state_digests": [result.state_digest for result in replay_results],
        "replay_exact": len({result.state_digest for result in replay_results}) == 1,
        "successes": successes,
        "predicted_values": predicted_values,
        "pairwise_action_l2": _off_diagonal_summary(action_distances),
        "pairwise_endpoint_proprio_l2": _off_diagonal_summary(endpoint_proprio_distances),
        "pairwise_endpoint_primary_pixel_l1": _off_diagonal_summary(endpoint_primary_pixel_l1),
        "pairwise_predicted_primary_pixel_l1": _off_diagonal_summary(predicted_primary_pixel_l1),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
