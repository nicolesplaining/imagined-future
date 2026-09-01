"""Collect exact-replay RoboCasa branches through NVIDIA's released evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from imagined_future.branching import state_digest
from imagined_future.cosmos_config import deterministic_tokenizer_enabled, robocasa_policy_config
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution
from imagined_future.robocasa import environment_action, physical_state_vector
from imagined_future.study_design import pairwise_l2
from PIL import Image


def _bytes_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _off_diagonal_summary(distances: np.ndarray) -> dict[str, float]:
    values = distances[np.triu_indices(distances.shape[0], k=1)]
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--prefix-chunks", type=int, required=True)
    parser.add_argument("--environment-seed-base", type=int, default=195)
    parser.add_argument("--prefix-seed", type=int, default=307)
    parser.add_argument("--branch-seeds", type=int, nargs="+", default=[311, 313, 317, 319, 331, 337, 347, 349])
    parser.add_argument("--replay-repeats", type=int, default=3)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {args.output_dir}")
    if len(set(args.branch_seeds)) != len(args.branch_seeds):
        raise ValueError("branch seeds must be unique")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
        unnormalize_actions,
    )
    from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
        create_robocasa_env,
        prepare_observation,
    )
    from cosmos_policy.utils.utils import set_seed_everywhere

    cfg = robocasa_policy_config(args.task_name, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, cosmos_config = get_model(cfg)
    if cfg.chunk_size != cosmos_config.dataloader_train.dataset.chunk_size:
        raise RuntimeError("RoboCasa checkpoint action chunk does not match the released evaluator")

    environment_seed = args.environment_seed_base * args.episode_index * 256

    def recreate(prefix_actions: list[np.ndarray]):
        set_seed_everywhere(environment_seed)
        env, _kwargs = create_robocasa_env(cfg, seed=environment_seed, episode_idx=args.episode_index)
        raw = env.reset()
        initial_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        model_xml = env.sim.model.get_xml()
        for _ in range(10):
            raw, _reward, _done, _info = env.step(np.zeros(env.action_spec[0].shape))
        for action in prefix_actions:
            raw, _reward, _done, _info = env.step(environment_action(action, env.action_dim))
        return env, raw, initial_state, model_xml

    prefix_actions: list[np.ndarray] = []
    env, raw_observation, initial_state, model_xml = recreate(prefix_actions)
    try:
        task_description = env.get_ep_meta()["lang"]
        for _chunk_index in range(args.prefix_chunks):
            observation = prepare_observation(raw_observation, cfg.flip_images)
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
            normalized = np.asarray(result["actions"])[: cfg.num_open_loop_steps]
            actions = unnormalize_actions(normalized.copy(), dataset_stats)
            for action in actions:
                raw_observation, _reward, _done, _info = env.step(
                    environment_action(action, env.action_dim)
                )
                prefix_actions.append(action.copy())
        branch_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        branch_observation = prepare_observation(raw_observation, cfg.flip_images)
    finally:
        env.close()

    replay_state_digests = []
    replay_observation_digests = []
    for _repeat in range(args.replay_repeats):
        replay_env, replay_raw, replay_initial, replay_xml = recreate(prefix_actions)
        try:
            replay_state = np.asarray(replay_env.sim.get_state().flatten(), dtype=np.float64)
            replay_observation = prepare_observation(replay_raw, cfg.flip_images)
            if not np.array_equal(initial_state, replay_initial) or model_xml != replay_xml:
                raise RuntimeError("RoboCasa initial simulator reconstruction changed")
            replay_state_digests.append(state_digest(replay_state))
            replay_observation_digests.append(
                _bytes_digest(
                    replay_observation["primary_image"],
                    replay_observation["secondary_image"],
                    replay_observation["wrist_image"],
                    replay_observation["proprio"],
                )
            )
        finally:
            replay_env.close()
    replay_exact = len(set(replay_state_digests)) == 1 and len(set(replay_observation_digests)) == 1
    if not replay_exact or replay_state_digests[0] != state_digest(branch_state):
        raise RuntimeError("RoboCasa branch point does not replay exactly")

    branch_actions = []
    normalized_branch_actions = []
    endpoint_states = []
    endpoint_primary_images = []
    endpoint_secondary_images = []
    endpoint_wrist_images = []
    endpoint_proprios = []
    endpoint_physical_features = []
    predicted_primary_images = []
    predicted_secondary_images = []
    predicted_wrist_images = []
    predicted_values = []
    successes = []
    physical_schema = None
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
        normalized = np.asarray(result["actions"])[: cfg.num_open_loop_steps]
        actions = unnormalize_actions(normalized.copy(), dataset_stats)
        predictions = result["future_image_predictions"]
        predicted_primary = np.asarray(predictions["future_image"], dtype=np.uint8)
        predicted_secondary = np.asarray(predictions["future_image2"], dtype=np.uint8)
        predicted_wrist = np.asarray(predictions["future_wrist_image"], dtype=np.uint8)
        predicted_value = float(result["value_prediction"])

        endpoint_env, endpoint_raw, _endpoint_initial, _endpoint_xml = recreate(prefix_actions)
        try:
            for action in actions:
                endpoint_raw, _reward, _done, _info = endpoint_env.step(
                    environment_action(action, endpoint_env.action_dim)
                )
            endpoint_state = np.asarray(endpoint_env.sim.get_state().flatten(), dtype=np.float64)
            endpoint = prepare_observation(endpoint_raw, cfg.flip_images)
            physical, schema = physical_state_vector(endpoint_raw, endpoint_env.sim.data.qpos)
            if physical_schema is None:
                physical_schema = schema
            elif schema != physical_schema:
                raise RuntimeError("RoboCasa physical-observation schema changed between branches")
            success = bool(endpoint_env._check_success())
        finally:
            endpoint_env.close()

        branch_actions.append(actions)
        normalized_branch_actions.append(normalized.astype(np.float64))
        endpoint_states.append(endpoint_state)
        endpoint_primary_images.append(np.asarray(endpoint["primary_image"], dtype=np.uint8))
        endpoint_secondary_images.append(np.asarray(endpoint["secondary_image"], dtype=np.uint8))
        endpoint_wrist_images.append(np.asarray(endpoint["wrist_image"], dtype=np.uint8))
        endpoint_proprios.append(np.asarray(endpoint["proprio"], dtype=np.float64))
        endpoint_physical_features.append(physical)
        predicted_primary_images.append(predicted_primary)
        predicted_secondary_images.append(predicted_secondary)
        predicted_wrist_images.append(predicted_wrist)
        predicted_values.append(predicted_value)
        successes.append(success)
        label = f"branch_{branch_index:02d}_seed_{seed}"
        Image.fromarray(endpoint["primary_image"]).save(args.output_dir / f"{label}_endpoint_primary.png")
        Image.fromarray(predicted_primary).save(args.output_dir / f"{label}_predicted_primary.png")

    normalized_array = np.stack(normalized_branch_actions)
    physical_array = np.stack(endpoint_physical_features)
    action_distances = pairwise_l2(normalized_array)
    physical_distances = pairwise_l2(physical_array)
    np.savez_compressed(
        args.output_dir / "branches.npz",
        initial_state=initial_state,
        prefix_actions=np.asarray(prefix_actions, dtype=np.float64).reshape(-1, 7),
        branch_state=branch_state,
        current_primary_image=np.asarray(branch_observation["primary_image"], dtype=np.uint8),
        current_secondary_image=np.asarray(branch_observation["secondary_image"], dtype=np.uint8),
        current_wrist_image=np.asarray(branch_observation["wrist_image"], dtype=np.uint8),
        current_proprio=np.asarray(branch_observation["proprio"], dtype=np.float64),
        branch_seeds=np.asarray(args.branch_seeds, dtype=np.int64),
        normalized_branch_actions=normalized_array,
        branch_actions=np.stack(branch_actions),
        endpoint_states=np.stack(endpoint_states),
        endpoint_primary_images=np.stack(endpoint_primary_images),
        endpoint_secondary_images=np.stack(endpoint_secondary_images),
        endpoint_wrist_images=np.stack(endpoint_wrist_images),
        endpoint_proprios=np.stack(endpoint_proprios),
        endpoint_physical_features=physical_array,
        predicted_primary_images=np.stack(predicted_primary_images),
        predicted_secondary_images=np.stack(predicted_secondary_images),
        predicted_wrist_images=np.stack(predicted_wrist_images),
        predicted_values=np.asarray(predicted_values, dtype=np.float64),
        successes=np.asarray(successes, dtype=np.bool_),
        action_distances=action_distances,
        physical_endpoint_distances=physical_distances,
    )
    (args.output_dir / "model.xml").write_text(model_xml)
    summary = {
        "scope": "RoboCasa same-state natural branches; no intervention outcomes",
        "task_name": args.task_name,
        "episode_index": args.episode_index,
        "prefix_chunks": args.prefix_chunks,
        "prefix_seed": args.prefix_seed,
        "prefix_action_steps": len(prefix_actions),
        "environment_seed_base": args.environment_seed_base,
        "environment_seed": environment_seed,
        "task_description": task_description,
        "branch_seeds": args.branch_seeds,
        "branch_state_digest": state_digest(branch_state),
        "replay_state_digests": replay_state_digests,
        "replay_observation_digests": replay_observation_digests,
        "replay_exact": replay_exact,
        "model_xml_sha256": hashlib.sha256(model_xml.encode()).hexdigest(),
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "physical_observation_schema": physical_schema,
        "successes": successes,
        "predicted_values": predicted_values,
        "pairwise_action_l2": _off_diagonal_summary(action_distances),
        "pairwise_physical_endpoint_l2": _off_diagonal_summary(physical_distances),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
