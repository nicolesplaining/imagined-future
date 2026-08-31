"""Validate fresh-environment replay for an official LIBERO initial state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from imagined_future.branching import BranchPoint, state_digest, tuple_actions, validate_replay_stability


def _numeric_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value)
        for key, value in observation.items()
        if np.asarray(value).dtype.kind in "biufc"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--state-atol", type=float, default=0.0)
    args = parser.parse_args()

    from libero.libero import benchmark

    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_state = np.asarray(suite.get_task_init_states(args.task_id)[args.initial_state_index]).copy()

    def env_factory():
        environment, _description = get_libero_env(task, "cosmos", resolution=256)
        return environment

    dummy_action = get_libero_dummy_action("cosmos")
    point = BranchPoint(
        initial_state=initial_state,
        warmup_actions=tuple_actions([dummy_action] * args.warmup_steps),
        prefix_actions=(),
    )
    results = validate_replay_stability(
        env_factory,
        point,
        repeats=args.repeats,
        atol=args.state_atol,
    )

    reference_state = results[0].state
    state_max_abs = [float(np.max(np.abs(result.state - reference_state))) for result in results]
    observations = [_numeric_observation(result.observation) for result in results]
    common_keys = sorted(set.intersection(*(set(observation) for observation in observations)))
    observation_record: dict[str, Any] = {}
    for key in common_keys:
        reference = observations[0][key]
        values = [observation[key] for observation in observations]
        if any(value.shape != reference.shape for value in values):
            raise RuntimeError(f"observation {key} changed shape across replays")
        maximum = max(float(np.max(np.abs(value.astype(np.float64) - reference))) for value in values)
        observation_record[key] = {
            "dtype": reference.dtype.str,
            "shape": list(reference.shape),
            "digests": [state_digest(value) for value in values],
            "max_abs_from_reference": maximum,
        }

    summary = {
        "scope": "libero fresh-environment replay calibration",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_description": task.language,
        "initial_state_index": args.initial_state_index,
        "warmup_steps": args.warmup_steps,
        "repeats": args.repeats,
        "state_atol": args.state_atol,
        "state_digests": [result.state_digest for result in results],
        "state_max_abs_from_reference": state_max_abs,
        "state_exact": len({result.state_digest for result in results}) == 1,
        "observations": observation_record,
        "observations_exact": all(
            len(set(record["digests"])) == 1 for record in observation_record.values()
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "initial_state.npy", initial_state, allow_pickle=False)
    np.save(args.output_dir / "replayed_state.npy", reference_state, allow_pickle=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
