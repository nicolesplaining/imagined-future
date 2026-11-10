"""Does choosing a better imagined future change Cosmos Policy task success?

``screen`` rolls a few native seeds to completion at LIBERO first-decision
states and records which states have mixed outcomes. ``run`` then, at one such
state, draws K candidate (future, action) pairs, ranks them by the model's own
predicted value before anything is executed, and runs to completion under
shared continuation seeds:

* every candidate's own first chunk (the random-selection baseline),
* the first chunk recomputed after inserting the top-ranked candidate's future
  into other candidates' runs, and the same for the bottom-ranked future.

Executing the top-ranked candidate's own action is the direct-selection
comparison: insertion adds something only if it beats that, or transfers it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from imagined_future.branching import BranchPoint, state_digest, tuple_actions, validate_replay_stability
from imagined_future.cosmos_config import deterministic_tokenizer_enabled, libero_policy_config
from imagined_future.frames import LatentFrameGroups
from imagined_future.interventions import SemanticFutureClamp, replace_frames, resample_frames
from imagined_future.metrics import donor_steering
from imagined_future.model_patch import (
    defer_hf_config_checkpoint_resolution,
    transform_model_initial_noise,
    transform_model_x0_factory,
)
from imagined_future.semantic_latent import encode_semantic_future


class Policy:
    def __init__(self, suite: str) -> None:
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

        self.get_action = get_action
        self.unnormalize = unnormalize_actions
        self.prepare = prepare_observation
        self.get_env = get_libero_env
        self.cfg = libero_policy_config(suite, unnormalize_actions=False)
        self.stats = load_dataset_stats(self.cfg.dataset_stats_path)
        init_t5_text_embeddings_cache(self.cfg.t5_text_embeddings_path)
        with defer_hf_config_checkpoint_resolution():
            self.model, _ = get_model(self.cfg)
        self.suite = benchmark.get_benchmark_dict()[suite]()
        self.dummy = get_libero_dummy_action("cosmos")

    def observe(self, raw: Any) -> dict[str, Any]:
        return self.prepare(raw, 256, self.cfg.flip_images)

    def query(self, observation: dict[str, Any], task: str, seed: int) -> dict[str, Any]:
        return self.get_action(
            cfg=self.cfg,
            model=self.model,
            dataset_stats=self.stats,
            obs=observation,
            task_label_or_embedding=task,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=self.cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
        )

    def actions(self, result: dict[str, Any]) -> np.ndarray:
        return self.unnormalize(np.asarray(result["actions"]).copy(), self.stats)


def branch_point(policy: Policy, task_id: int, init_index: int):
    task = policy.suite.get_task(task_id)
    initial = np.asarray(policy.suite.get_task_init_states(task_id)[init_index]).copy()

    def env_factory():
        environment, _ = policy.get_env(task, "cosmos", resolution=256)
        return environment

    point = BranchPoint(initial_state=initial, warmup_actions=tuple_actions([policy.dummy] * 10), prefix_actions=())
    replays = validate_replay_stability(env_factory, point, repeats=2, atol=0.0)
    return task, env_factory, point, replays[0]


def rollout(
    policy: Policy,
    env_factory,
    point: BranchPoint,
    task: str,
    first_actions: np.ndarray,
    continuation_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Execute a first chunk from the branch point, then continue to termination."""

    environment = env_factory()
    try:
        environment.reset()
        raw = environment.set_init_state(point.initial_state.copy())
        for action in point.warmup_actions:
            raw, _reward, _done, _info = environment.step(action)
        start = state_digest(environment.get_sim_state())
        steps, success = 0, False
        queue = list(first_actions)
        endpoint = None
        while steps < max_steps and not success:
            if not queue:
                queue = list(policy.actions(policy.query(policy.observe(raw), task, continuation_seed)))
            raw, _reward, done, _info = environment.step(queue.pop(0).tolist())
            steps += 1
            success = bool(done or environment.check_success())
            if steps == len(first_actions):
                endpoint = policy.observe(raw)
        if endpoint is None:
            endpoint = policy.observe(raw)
        return {"success": success, "steps": steps, "start_digest": start, "endpoint": endpoint}
    finally:
        environment.close()


def screen(args: argparse.Namespace) -> None:
    policy = Policy(args.suite)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task_id in args.task_ids:
        for init_index in args.initial_state_indices:
            task, env_factory, point, replay = branch_point(policy, task_id, init_index)
            observation = policy.observe(replay.observation)
            outcomes = []
            for seed in args.seeds:
                first = policy.actions(policy.query(observation, task.language, seed))
                outcome = rollout(policy, env_factory, point, task.language, first, seed, args.max_policy_steps)
                outcomes.append({"seed": seed, "success": outcome["success"], "steps": outcome["steps"]})
            row = {
                "task_id": task_id,
                "initial_state_index": init_index,
                "successes": sum(item["success"] for item in outcomes),
                "seeds": len(outcomes),
                "outcomes": outcomes,
            }
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("task_id", "initial_state_index", "successes", "seeds")}), flush=True)
            args.output.write_text(json.dumps({"rows": rows}, indent=2) + "\n")


def run(args: argparse.Namespace) -> None:
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {args.output_dir}")
    policy = Policy(args.suite)
    task, env_factory, point, replay = branch_point(policy, args.task_id, args.initial_state_index)
    observation = policy.observe(replay.observation)
    seeds = list(args.candidate_seeds)

    candidates = []
    for seed in seeds:
        result = policy.query(observation, task.language, seed)
        candidates.append(
            {
                "seed": seed,
                "normalized": np.asarray(result["actions"]).astype(np.float64),
                "actions": policy.actions(result),
                "value": float(result["value_prediction"]),
            }
        )
        del result
    values = [item["value"] for item in candidates]
    best = int(np.argmax(values))
    worst = int(np.argmin(values))
    # Endpoint of each candidate's own first chunk: the insertion target.
    for item in candidates:
        item["endpoint"] = rollout(policy, env_factory, point, task.language, item["actions"], 0, len(item["actions"]))["endpoint"]

    # Recipients for insertion: lowest-ranked-index candidates other than the source, fixed before outcomes.
    def recipients(source: int) -> list[int]:
        return [index for index in range(len(seeds)) if index != source][: args.recipients]

    inserted: dict[str, dict[str, Any]] = {}
    noise = None
    for source_name, source in (("best", best), ("worst", worst)):
        for recipient in recipients(source):
            common_seed = seeds[recipient]
            baseline = policy.query(observation, task.language, common_seed)
            if float(np.max(np.abs(np.asarray(baseline["actions"]) - candidates[recipient]["normalized"]))) != 0.0:
                raise RuntimeError("recipient baseline does not reproduce its native action")
            target_batch = policy.query(candidates[source]["endpoint"], task.language, common_seed)
            clean = encode_semantic_future(
                policy.model,
                baseline["data_batch"],
                target_batch["data_batch"],
                baseline["orig_clean_latent_frames"],
                target_batch["proprio"],
            )
            groups = LatentFrameGroups.from_batch(baseline["data_batch"])
            frames = tuple(groups.future)
            if noise is None:
                noise = resample_frames(torch.zeros_like(clean), frames, seed=args.future_noise_seed, standard_deviation=1.0)
            clamp = SemanticFutureClamp(clean, noise, frames)

            def initial_transform(initial, batch, *, _clean=clean, _frames=frames, _groups=groups):
                if LatentFrameGroups.from_batch(batch) != _groups:
                    raise RuntimeError("latent frame layout changed during insertion")
                return replace_frames(initial, _clean + float(policy.model.sde.sigma_max) * noise, _frames)

            with transform_model_initial_noise(policy.model, initial_transform):
                with transform_model_x0_factory(policy.model, clamp.wrap):
                    result = policy.query(observation, task.language, common_seed)
            normalized = np.asarray(result["actions"]).astype(np.float64)
            steering = float(
                donor_steering(
                    torch.from_numpy(normalized).unsqueeze(0),
                    torch.from_numpy(candidates[recipient]["normalized"]).unsqueeze(0),
                    torch.from_numpy(candidates[source]["normalized"]).unsqueeze(0),
                ).item()
            )
            inserted[f"{source_name}_into_{recipient}"] = {
                "source": source,
                "recipient": recipient,
                "actions": policy.actions(result),
                "steering_toward_source": steering,
            }
            del baseline, target_batch, result

    arms: dict[str, np.ndarray] = {f"candidate_{index}": item["actions"] for index, item in enumerate(candidates)}
    arms.update({f"inserted_{name}": item["actions"] for name, item in inserted.items()})
    outcomes: dict[str, list[dict[str, Any]]] = {}
    for name, actions in arms.items():
        outcomes[name] = []
        for continuation in args.continuation_seeds:
            outcome = rollout(policy, env_factory, point, task.language, actions, continuation, args.max_policy_steps)
            outcomes[name].append({"continuation_seed": continuation, "success": outcome["success"], "steps": outcome["steps"]})
        print(json.dumps({"arm": name, "successes": sum(o["success"] for o in outcomes[name])}), flush=True)

    def rate(names: list[str]) -> float:
        flat = [o["success"] for name in names for o in outcomes[name]]
        return float(np.mean(flat))

    candidate_names = [f"candidate_{index}" for index in range(len(seeds))]
    summary = {
        "scope": "Cosmos Policy future selection and insertion; closed-loop LIBERO success",
        "task_id": args.task_id,
        "task_description": task.language,
        "initial_state_index": args.initial_state_index,
        "candidate_seeds": seeds,
        "continuation_seeds": args.continuation_seeds,
        "future_noise_seed": args.future_noise_seed,
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "ranking": "model-predicted value, computed before any execution",
        "predicted_values": values,
        "best_candidate": best,
        "worst_candidate": worst,
        "success_rates": {
            "random_candidate": rate(candidate_names),
            "best_candidate_direct": rate([f"candidate_{best}"]),
            "worst_candidate_direct": rate([f"candidate_{worst}"]),
            "best_inserted": rate([f"inserted_{name}" for name in inserted if name.startswith("best_")]),
            "worst_inserted": rate([f"inserted_{name}" for name in inserted if name.startswith("worst_")]),
            "recipients_of_best_native": rate([f"candidate_{inserted[name]['recipient']}" for name in inserted if name.startswith("best_")]),
            "recipients_of_worst_native": rate([f"candidate_{inserted[name]['recipient']}" for name in inserted if name.startswith("worst_")]),
        },
        "candidate_success_rates": [rate([name]) for name in candidate_names],
        "insertion_steering": {name: item["steering_toward_source"] for name, item in inserted.items()},
        "outcomes": outcomes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "actions.npz",
        **{name: actions for name, actions in arms.items()},
        predicted_values=np.asarray(values),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["success_rates"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    screen_parser = sub.add_parser("screen")
    screen_parser.add_argument("--output", type=Path, required=True)
    screen_parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    screen_parser.add_argument("--initial-state-indices", type=int, nargs="+", required=True)
    screen_parser.add_argument("--seeds", type=int, nargs="+", default=[601, 607, 613, 617])
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--task-id", type=int, required=True)
    run_parser.add_argument("--initial-state-index", type=int, required=True)
    run_parser.add_argument("--candidate-seeds", type=int, nargs="+", default=[701, 709, 719, 727, 733, 739, 743, 751])
    run_parser.add_argument("--continuation-seeds", type=int, nargs="+", default=[801, 809, 811])
    run_parser.add_argument("--recipients", type=int, default=3)
    run_parser.add_argument("--future-noise-seed", type=int, default=20261109)
    for sub_parser in (screen_parser, run_parser):
        sub_parser.add_argument("--suite", default="libero_10")
        sub_parser.add_argument("--max-policy-steps", type=int, default=520)
    args = parser.parse_args()
    if args.command == "screen":
        screen(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
