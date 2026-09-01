"""Collect held-out natural branch pools for robot/object content factorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from imagined_future.branching import (
    BranchPoint,
    state_digest,
    tuple_actions,
    validate_replay_stability,
)
from imagined_future.content_factorization import (
    FactorizationThresholds,
    select_factorized_donors,
)
from imagined_future.cosmos_config import (
    deterministic_tokenizer_enabled,
    libero_policy_config,
)
from imagined_future.libero_semantics import (
    goal_feature_vector,
    goal_predicate_snapshot,
)
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution

STAGE_ONE_SEEDS = [
    521,
    523,
    541,
    547,
    557,
    563,
    569,
    571,
    577,
    587,
    593,
    599,
    601,
    607,
    613,
    617,
    619,
    631,
    641,
    643,
    647,
    653,
    659,
    661,
    673,
    677,
    683,
    691,
    701,
    709,
    719,
    727,
]
STAGE_TWO_SEEDS = [
    733,
    739,
    743,
    751,
    757,
    761,
    769,
    773,
    787,
    797,
    809,
    811,
    821,
    823,
    827,
    829,
    839,
    853,
    857,
    859,
    863,
    877,
    881,
    883,
    887,
    907,
    911,
    919,
    929,
    937,
    941,
    947,
]


def _execute_from_point(environment_factory, point: BranchPoint, actions: np.ndarray):
    environment = environment_factory()
    try:
        environment.reset()
        observation = environment.set_init_state(point.initial_state.copy())
        for action in (*point.warmup_actions, *point.prefix_actions):
            observation, _reward, _done, _info = environment.step(action)
        done_observed = False
        for action in actions:
            observation, _reward, done, _info = environment.step(action.tolist())
            done_observed = done_observed or bool(done)
        return (
            np.asarray(environment.get_sim_state()).copy(),
            observation,
            bool(done_observed or environment.check_success()),
            goal_predicate_snapshot(environment),
        )
    finally:
        environment.close()


def _collect_records(
    *,
    seeds: list[int],
    cfg,
    model,
    dataset_stats,
    task_description: str,
    branch_observation: dict,
    point: BranchPoint,
    environment_factory,
    get_action,
    unnormalize_actions,
) -> list[dict[str, Any]]:
    records = []
    for seed in seeds:
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
        normalized_actions = np.asarray(result["actions"], dtype=np.float64)
        actions = unnormalize_actions(normalized_actions.copy(), dataset_stats)
        predictions = result["future_image_predictions"]
        predicted_primary = np.asarray(predictions["future_image"], dtype=np.uint8)
        predicted_wrist = np.asarray(predictions["future_wrist_image"], dtype=np.uint8)
        predicted_value = float(result["value_prediction"])
        del result
        endpoint_state, raw_endpoint, success, predicates = _execute_from_point(
            environment_factory, point, actions
        )
        from cosmos_policy.experiments.robot.libero.run_libero_eval import (
            prepare_observation,
        )

        endpoint = prepare_observation(raw_endpoint, 256, cfg.flip_images)
        records.append(
            {
                "seed": seed,
                "normalized_actions": normalized_actions,
                "actions": actions,
                "endpoint_state": endpoint_state,
                "endpoint_primary_image": np.asarray(
                    endpoint["primary_image"], dtype=np.uint8
                ),
                "endpoint_wrist_image": np.asarray(
                    endpoint["wrist_image"], dtype=np.uint8
                ),
                "endpoint_proprio": np.asarray(endpoint["proprio"], dtype=np.float64),
                "predicted_primary_image": predicted_primary,
                "predicted_wrist_image": predicted_wrist,
                "predicted_value": predicted_value,
                "success": success,
                "predicates": predicates,
            }
        )
    return records


def _selection(records: list[dict[str, Any]]) -> dict:
    actions = np.stack([record["normalized_actions"] for record in records])
    robot = np.stack([record["endpoint_proprio"] for record in records])
    objects = np.stack(
        [goal_feature_vector(record["predicates"]) for record in records]
    )
    return select_factorized_donors(actions, robot, objects, FactorizationThresholds())


def _save_pool(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    point: BranchPoint,
    branch_state: np.ndarray,
    branch_observation: dict,
    replay_digests: list[str],
    unit: dict,
    prefix_chunks: int,
    stage: int,
    selection: dict | None,
    selection_error: str | None,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite branch pool: {output_dir}")
    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "branches.npz",
        initial_state=point.initial_state,
        prefix_actions=np.asarray(point.prefix_actions, dtype=np.float64).reshape(
            -1, 7
        ),
        branch_state=branch_state,
        current_primary_image=np.asarray(
            branch_observation["primary_image"], dtype=np.uint8
        ),
        current_wrist_image=np.asarray(
            branch_observation["wrist_image"], dtype=np.uint8
        ),
        current_proprio=np.asarray(branch_observation["proprio"], dtype=np.float64),
        branch_seeds=np.asarray([record["seed"] for record in records], dtype=np.int64),
        normalized_branch_actions=np.stack(
            [record["normalized_actions"] for record in records]
        ),
        branch_actions=np.stack([record["actions"] for record in records]),
        endpoint_states=np.stack([record["endpoint_state"] for record in records]),
        endpoint_primary_images=np.stack(
            [record["endpoint_primary_image"] for record in records]
        ),
        endpoint_wrist_images=np.stack(
            [record["endpoint_wrist_image"] for record in records]
        ),
        endpoint_proprios=np.stack([record["endpoint_proprio"] for record in records]),
        predicted_primary_images=np.stack(
            [record["predicted_primary_image"] for record in records]
        ),
        predicted_wrist_images=np.stack(
            [record["predicted_wrist_image"] for record in records]
        ),
        predicted_values=np.asarray(
            [record["predicted_value"] for record in records], dtype=np.float64
        ),
        successes=np.asarray([record["success"] for record in records], dtype=np.bool_),
    )
    (output_dir / "endpoint_predicates.json").write_text(
        json.dumps(
            [
                {
                    "branch_index": index,
                    "branch_seed": record["seed"],
                    "snapshot": record["predicates"],
                }
                for index, record in enumerate(records)
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "scope": "held-out natural branches for content factorization; no intervention outcomes",
        "unit_id": unit["unit_id"],
        "task_id": unit["task_id"],
        "task_description": unit["task_description"],
        "initial_state_index": unit["initial_state_index"],
        "prefix_chunks": prefix_chunks,
        "prefix_seed": 503,
        "stage": stage,
        "branches": len(records),
        "branch_seeds": [record["seed"] for record in records],
        "branch_state_digest": state_digest(branch_state),
        "replay_state_digests": replay_digests,
        "replay_exact": len(set(replay_digests)) == 1,
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "selection": selection,
        "selection_error": selection_error,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--suite", default="libero_10")
    args = parser.parse_args()
    if args.manifest_output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.manifest_output}")

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

    screen = json.loads(args.screen.read_text())
    units = [unit for unit in screen["units"] if unit["task_id"] in args.task_ids]
    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    dummy_action = get_libero_dummy_action("cosmos")
    warmup_actions = tuple_actions([dummy_action] * 10)
    manifest_units = []

    for unit in units:
        task = suite.get_task(unit["task_id"])
        initial_state = np.asarray(
            suite.get_task_init_states(unit["task_id"])[unit["initial_state_index"]]
        ).copy()

        def environment_factory(_task=task):
            environment, _description = get_libero_env(_task, "cosmos", resolution=256)
            return environment

        maximum_candidate = max(unit["candidate_prefix_chunks"])
        prefix_actions: list[np.ndarray] = []
        prefix_environment = environment_factory()
        try:
            prefix_environment.reset()
            raw_observation = prefix_environment.set_init_state(initial_state.copy())
            for action in warmup_actions:
                raw_observation, _reward, done, _info = prefix_environment.step(action)
                if done:
                    raise RuntimeError("task completed during warmup")
            for _chunk_index in range(maximum_candidate):
                observation = prepare_observation(raw_observation, 256, cfg.flip_images)
                result = get_action(
                    cfg=cfg,
                    model=model,
                    dataset_stats=dataset_stats,
                    obs=observation,
                    task_label_or_embedding=task.language,
                    seed=503,
                    randomize_seed=False,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=True,
                )
                actions = unnormalize_actions(
                    np.asarray(result["actions"]).copy(), dataset_stats
                )
                del result
                for action in actions:
                    raw_observation, _reward, done, _info = prefix_environment.step(
                        action.tolist()
                    )
                    prefix_actions.append(action.copy())
                    if done and len(prefix_actions) < maximum_candidate * 16:
                        raise RuntimeError("candidate prefix occurs after task success")
        finally:
            prefix_environment.close()

        selected = None
        candidate_records = []
        for candidate_rank, prefix_chunks in enumerate(
            unit["candidate_prefix_chunks"], start=1
        ):
            point = BranchPoint(
                initial_state=initial_state,
                warmup_actions=warmup_actions,
                prefix_actions=tuple_actions(prefix_actions[: prefix_chunks * 16]),
            )
            replays = validate_replay_stability(
                environment_factory, point, repeats=3, atol=0.0
            )
            branch_state = replays[0].state
            branch_observation = prepare_observation(
                replays[0].observation, 256, cfg.flip_images
            )
            records = _collect_records(
                seeds=STAGE_ONE_SEEDS,
                cfg=cfg,
                model=model,
                dataset_stats=dataset_stats,
                task_description=task.language,
                branch_observation=branch_observation,
                point=point,
                environment_factory=environment_factory,
                get_action=get_action,
                unnormalize_actions=unnormalize_actions,
            )
            stage_one_selection = None
            stage_one_error = None
            try:
                stage_one_selection = _selection(records)
            except ValueError as error:
                stage_one_error = str(error)
            stage_one_dir = (
                args.output_dir / f"{unit['unit_id']}_prefix{prefix_chunks:02d}_stage1"
            )
            _save_pool(
                output_dir=stage_one_dir,
                records=records,
                point=point,
                branch_state=branch_state,
                branch_observation=branch_observation,
                replay_digests=[replay.state_digest for replay in replays],
                unit=unit,
                prefix_chunks=prefix_chunks,
                stage=1,
                selection=stage_one_selection,
                selection_error=stage_one_error,
            )
            selected_dir = stage_one_dir
            final_selection = stage_one_selection
            final_error = stage_one_error
            if final_selection is None:
                records.extend(
                    _collect_records(
                        seeds=STAGE_TWO_SEEDS,
                        cfg=cfg,
                        model=model,
                        dataset_stats=dataset_stats,
                        task_description=task.language,
                        branch_observation=branch_observation,
                        point=point,
                        environment_factory=environment_factory,
                        get_action=get_action,
                        unnormalize_actions=unnormalize_actions,
                    )
                )
                try:
                    final_selection = _selection(records)
                    final_error = None
                except ValueError as error:
                    final_error = str(error)
                selected_dir = (
                    args.output_dir
                    / f"{unit['unit_id']}_prefix{prefix_chunks:02d}_stage2"
                )
                _save_pool(
                    output_dir=selected_dir,
                    records=records,
                    point=point,
                    branch_state=branch_state,
                    branch_observation=branch_observation,
                    replay_digests=[replay.state_digest for replay in replays],
                    unit=unit,
                    prefix_chunks=prefix_chunks,
                    stage=2,
                    selection=final_selection,
                    selection_error=final_error,
                )
            candidate_record = {
                "candidate_rank": candidate_rank,
                "prefix_chunks": prefix_chunks,
                "stage_one_dir": str(stage_one_dir),
                "final_dir": str(selected_dir),
                "eligible": final_selection is not None,
                "selection_error": final_error,
            }
            candidate_records.append(candidate_record)
            if final_selection is not None:
                selected = {
                    **candidate_record,
                    "branch_run_dir": str(selected_dir),
                    "selection": final_selection,
                    "artifact_sha256": {
                        name: hashlib.sha256(
                            (selected_dir / name).read_bytes()
                        ).hexdigest()
                        for name in (
                            "branches.npz",
                            "endpoint_predicates.json",
                            "summary.json",
                        )
                    },
                }
                break
        manifest_units.append(
            {
                **{
                    key: unit[key]
                    for key in (
                        "unit_id",
                        "task_id",
                        "task_description",
                        "initial_state_index",
                    )
                },
                "candidate_prefix_chunks": unit["candidate_prefix_chunks"],
                "candidates_evaluated": candidate_records,
                "selected": selected,
                "structurally_eligible": selected is not None,
            }
        )

    manifest = {
        "scope": "held-out natural factorization pairs selected without intervention outcomes",
        "screen": str(args.screen),
        "thresholds": FactorizationThresholds().__dict__,
        "units": manifest_units,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.manifest_output),
                "units": len(manifest_units),
                "eligible": sum(
                    unit["structurally_eligible"] for unit in manifest_units
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
