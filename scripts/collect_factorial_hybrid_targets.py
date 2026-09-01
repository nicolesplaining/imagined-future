"""Collect and render prospective 2x2 object/robot endpoint targets."""

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
from imagined_future.content_factorization import pairwise_quaternion_angle
from imagined_future.cosmos_config import (
    deterministic_tokenizer_enabled,
    libero_policy_config,
)
from imagined_future.libero_semantics import (
    goal_feature_vector,
    goal_predicate_snapshot,
)
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution
from imagined_future.sim_state_factorization import (
    factorized_flat_state,
    robot_state_indices,
)

CELL_NAMES = ("o0r0", "o1r0", "o0r1", "o1r1")


def _robot_distances(current: np.ndarray, future: np.ndarray) -> dict[str, float]:
    values = np.stack([current, future]).astype(np.float64)
    return {
        "robot_gripper_l2": float(np.linalg.norm(values[1, :2] - values[0, :2])),
        "robot_position_l2_m": float(
            np.linalg.norm(values[1, 2:5] - values[0, 2:5])
        ),
        "robot_orientation_angle_rad": float(
            pairwise_quaternion_angle(values[:, 5:9])[0, 1]
        ),
    }


def _candidate_valid(metrics: dict[str, float]) -> bool:
    return metrics["object_goal_l2"] >= 0.003 and (
        metrics["robot_position_l2_m"] >= 0.003
        or metrics["robot_orientation_angle_rad"] >= 0.03
        or metrics["robot_gripper_l2"] >= 0.003
    )


def _render_cells(
    environment_factory,
    states: dict[str, np.ndarray],
    *,
    cfg,
    prepare_observation,
) -> dict[str, dict[str, Any]]:
    records = {}
    for name, state in states.items():
        environment = environment_factory()
        try:
            environment.reset()
            raw = environment.set_init_state(state.copy())
            observation = prepare_observation(raw, 256, cfg.flip_images)
            records[name] = {
                "state": np.asarray(environment.get_sim_state()).copy(),
                "primary_image": np.asarray(
                    observation["primary_image"], dtype=np.uint8
                ),
                "wrist_image": np.asarray(
                    observation["wrist_image"], dtype=np.uint8
                ),
                "proprio": np.asarray(observation["proprio"], dtype=np.float64),
                "predicates": goal_predicate_snapshot(environment),
                "contacts": int(environment.env.sim.data.ncon),
            }
        finally:
            environment.close()
    return records


def _validate_cells(records: dict[str, dict[str, Any]]) -> dict[str, float]:
    features = {
        name: goal_feature_vector(record["predicates"])
        for name, record in records.items()
    }
    proprios = {name: record["proprio"] for name, record in records.items()}
    checks = {
        "o1r0_goal_to_o1r1": float(
            np.max(np.abs(features["o1r0"] - features["o1r1"]), initial=0.0)
        ),
        "o0r1_goal_to_o0r0": float(
            np.max(np.abs(features["o0r1"] - features["o0r0"]), initial=0.0)
        ),
        "o1r0_robot_to_o0r0": float(
            np.max(np.abs(proprios["o1r0"] - proprios["o0r0"]), initial=0.0)
        ),
        "o0r1_robot_to_o1r1": float(
            np.max(np.abs(proprios["o0r1"] - proprios["o1r1"]), initial=0.0)
        ),
    }
    if max(checks["o1r0_goal_to_o1r1"], checks["o0r1_goal_to_o0r0"]) > 1e-9:
        raise RuntimeError(f"hybrid object validation failed: {checks}")
    if max(
        checks["o1r0_robot_to_o0r0"], checks["o0r1_robot_to_o1r1"]
    ) > 1e-6:
        raise RuntimeError(f"hybrid robot validation failed: {checks}")
    for record in records.values():
        if not all(
            np.isfinite(np.asarray(record[key])).all()
            for key in ("state", "proprio")
        ):
            raise RuntimeError("hybrid target contains nonfinite state")
    return checks


def _save_target(
    output_dir: Path,
    *,
    unit: dict,
    candidate: dict,
    point: BranchPoint,
    replay_digests: list[str],
    records: dict[str, dict[str, Any]],
    validation: dict[str, float],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite target: {output_dir}")
    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "branches.npz",
        initial_state=point.initial_state,
        prefix_actions=np.asarray(point.prefix_actions, dtype=np.float64).reshape(
            -1, 7
        ),
        branch_state=records["o0r0"]["state"],
        current_primary_image=records["o0r0"]["primary_image"],
        current_wrist_image=records["o0r0"]["wrist_image"],
        current_proprio=records["o0r0"]["proprio"],
        cell_names=np.asarray(CELL_NAMES),
        cell_states=np.stack([records[name]["state"] for name in CELL_NAMES]),
        cell_primary_images=np.stack(
            [records[name]["primary_image"] for name in CELL_NAMES]
        ),
        cell_wrist_images=np.stack(
            [records[name]["wrist_image"] for name in CELL_NAMES]
        ),
        cell_proprios=np.stack(
            [records[name]["proprio"] for name in CELL_NAMES]
        ),
        normalized_native_actions=candidate["normalized_actions"],
        environment_native_actions=candidate["environment_actions"],
    )
    (output_dir / "endpoint_predicates.json").write_text(
        json.dumps(
            [
                {
                    "cell": name,
                    "snapshot": records[name]["predicates"],
                    "contacts": records[name]["contacts"],
                }
                for name in CELL_NAMES
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "scope": "prospective rendered 2x2 object/robot targets; no hybrid intervention outcomes",
        "unit_id": unit["unit_id"],
        "task_id": unit["task_id"],
        "task_description": unit["task_description"],
        "initial_state_index": unit["initial_state_index"],
        "prefix_chunks": candidate["prefix_chunks"],
        "prefix_seed": 503,
        "native_change_metrics": candidate["metrics"],
        "branch_state_digest": state_digest(records["o0r0"]["state"]),
        "replay_state_digests": replay_digests,
        "replay_exact": len(set(replay_digests)) == 1,
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "cell_contacts": {
            name: records[name]["contacts"] for name in CELL_NAMES
        },
        "validation": validation,
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
        raise FileExistsError(
            f"refusing to overwrite manifest: {args.manifest_output}"
        )

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
            environment, _description = get_libero_env(
                _task, "cosmos", resolution=256
            )
            return environment

        candidate_set = set(unit["candidate_prefix_chunks"])
        maximum_candidate = max(candidate_set)
        candidates: dict[int, dict[str, Any]] = {}
        prefix_actions: list[np.ndarray] = []
        environment = environment_factory()
        try:
            environment.reset()
            raw = environment.set_init_state(initial_state.copy())
            for action in warmup_actions:
                raw, _reward, done, _info = environment.step(action)
                if done:
                    raise RuntimeError("task completed during warmup")
            for chunk in range(maximum_candidate + 1):
                current_state = np.asarray(environment.get_sim_state()).copy()
                current_predicates = goal_predicate_snapshot(environment)
                current_observation = prepare_observation(raw, 256, cfg.flip_images)
                result = get_action(
                    cfg=cfg,
                    model=model,
                    dataset_stats=dataset_stats,
                    obs=current_observation,
                    task_label_or_embedding=task.language,
                    seed=503,
                    randomize_seed=False,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=True,
                )
                normalized_actions = np.asarray(result["actions"], dtype=np.float64)
                actions = unnormalize_actions(
                    normalized_actions.copy(), dataset_stats
                )
                del result
                for action in actions:
                    raw, _reward, done, _info = environment.step(action.tolist())
                    if done and chunk < maximum_candidate:
                        raise RuntimeError("candidate occurs after task success")
                if chunk in candidate_set:
                    future_state = np.asarray(environment.get_sim_state()).copy()
                    future_predicates = goal_predicate_snapshot(environment)
                    future_observation = prepare_observation(
                        raw, 256, cfg.flip_images
                    )
                    metrics = {
                        "object_goal_l2": float(
                            np.linalg.norm(
                                goal_feature_vector(future_predicates)
                                - goal_feature_vector(current_predicates)
                            )
                        ),
                        **_robot_distances(
                            np.asarray(
                                current_observation["proprio"], dtype=np.float64
                            ),
                            np.asarray(
                                future_observation["proprio"], dtype=np.float64
                            ),
                        ),
                    }
                    candidates[chunk] = {
                        "prefix_chunks": chunk,
                        "prefix_actions": tuple_actions(prefix_actions),
                        "current_state": current_state,
                        "future_state": future_state,
                        "normalized_actions": normalized_actions,
                        "environment_actions": actions,
                        "metrics": metrics,
                    }
                prefix_actions.extend(action.copy() for action in actions)
        finally:
            environment.close()

        selected = next(
            (
                candidates[chunk]
                for chunk in unit["candidate_prefix_chunks"]
                if _candidate_valid(candidates[chunk]["metrics"])
            ),
            None,
        )
        if selected is None:
            manifest_units.append(
                {
                    "unit_id": unit["unit_id"],
                    "task_id": unit["task_id"],
                    "initial_state_index": unit["initial_state_index"],
                    "valid": False,
                    "candidate_metrics": {
                        str(chunk): candidates[chunk]["metrics"]
                        for chunk in unit["candidate_prefix_chunks"]
                    },
                }
            )
            continue

        point = BranchPoint(
            initial_state=initial_state,
            warmup_actions=warmup_actions,
            prefix_actions=selected["prefix_actions"],
        )
        replays = validate_replay_stability(
            environment_factory, point, repeats=3, atol=0.0
        )
        replay_digests = [replay.state_digest for replay in replays]
        if replay_digests[0] != state_digest(selected["current_state"]):
            raise RuntimeError(f"{unit['unit_id']} recorded branch state changed")

        index_environment = environment_factory()
        try:
            index_environment.reset()
            robot = index_environment.env.robots[0]
            indices = robot_state_indices(robot)
            nq = int(index_environment.env.sim.model.nq)
            nv = int(index_environment.env.sim.model.nv)
        finally:
            index_environment.close()
        states = {
            "o0r0": selected["current_state"],
            "o1r0": factorized_flat_state(
                selected["future_state"],
                selected["current_state"],
                nq=nq,
                nv=nv,
                robot_indices=indices,
            ),
            "o0r1": factorized_flat_state(
                selected["current_state"],
                selected["future_state"],
                nq=nq,
                nv=nv,
                robot_indices=indices,
            ),
            "o1r1": selected["future_state"],
        }
        records = _render_cells(
            environment_factory,
            states,
            cfg=cfg,
            prepare_observation=prepare_observation,
        )
        validation = _validate_cells(records)
        output_dir = args.output_dir / unit["unit_id"]
        _save_target(
            output_dir,
            unit=unit,
            candidate=selected,
            point=point,
            replay_digests=replay_digests,
            records=records,
            validation=validation,
        )
        manifest_units.append(
            {
                "unit_id": unit["unit_id"],
                "task_id": unit["task_id"],
                "task_description": unit["task_description"],
                "initial_state_index": unit["initial_state_index"],
                "valid": True,
                "prefix_chunks": selected["prefix_chunks"],
                "target_dir": str(output_dir),
                "native_change_metrics": selected["metrics"],
                "artifact_sha256": {
                    name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                    for name in (
                        "branches.npz",
                        "endpoint_predicates.json",
                        "summary.json",
                    )
                },
            }
        )

    manifest = {
        "scope": "prospective 2x2 rendered targets selected without hybrid intervention outcomes",
        "screen": str(args.screen),
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
                "valid": sum(unit["valid"] for unit in manifest_units),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
