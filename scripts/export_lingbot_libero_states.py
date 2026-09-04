#!/usr/bin/env python3
"""Export a frozen, outcome-blind LIBERO-Long cohort for LingBot-VA.

This script renders only predetermined simulator states.  It does not run a
policy, score a future, or filter an observation based on its contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


EVAL_INITIAL_STATE_INDICES = (10, 20, 30)
DEV_INITIAL_STATE_INDEX = 0
WAIT_STEPS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _extract_images(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    agentview = np.ascontiguousarray(observation["agentview_image"][::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1])
    if agentview.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise TypeError("LIBERO RGB observations must be uint8")
    return agentview, wrist


def _render_state(suite, task_id: int, initial_state_index: int):
    from experiments.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
    )
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    initial_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    initial_states = torch.load(initial_states_path, weights_only=False)
    if initial_state_index >= len(initial_states):
        raise IndexError(
            f"task {task_id} has {len(initial_states)} states, "
            f"cannot select {initial_state_index}"
        )
    env, task_description = get_libero_env(task, 128, seed=0)
    try:
        env.reset()
        observation = env.set_init_state(initial_states[initial_state_index])
        for _ in range(WAIT_STEPS):
            observation, _, _, _ = env.step(get_libero_dummy_action())
        agentview, wrist = _extract_images(observation)
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    return task_description, agentview, wrist, initial_states_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    from libero.libero import benchmark

    suite_name = "libero_10"
    suite = benchmark.get_benchmark_dict()[suite_name]()
    if suite.get_num_tasks() != 10:
        raise RuntimeError("expected exactly ten LIBERO-Long tasks")

    specs: list[tuple[str, int, int, str]] = []
    for task_id in range(10):
        for index in EVAL_INITIAL_STATE_INDICES:
            specs.append(("evaluation", task_id, index, f"task{task_id:02d}_state{index:03d}"))
    for task_id in (0, 1):
        specs.append(("development", task_id, DEV_INITIAL_STATE_INDEX, f"dev_task{task_id:02d}_state000"))

    records = []
    for admission, task_id, initial_state_index, state_id in specs:
        description, agentview, wrist, initial_states_path = _render_state(
            suite, task_id, initial_state_index
        )
        state_root = args.output_root / "states" / state_id
        artifact = state_root / "observation.npz"
        _atomic_npz(artifact, agentview=agentview, wrist=wrist)
        records.append(
            {
                "admission": admission,
                "initial_state_index": initial_state_index,
                "input_sha256": _sha256(artifact),
                "observation_path": str(artifact.resolve()),
                "prompt": description,
                "state_id": state_id,
                "suite": suite_name,
                "task_id": task_id,
                "wait_steps": WAIT_STEPS,
                "initial_states_path": str(initial_states_path),
            }
        )

    manifest = {
        "admission_rule": "all 30 predetermined evaluation states; no outcome or image filtering",
        "branch_ids": ["b0", "b1", "b2", "b3"],
        "video_seeds": [101, 211, 307, 401],
        "action_seeds": [1009, 2017, 3019, 4021],
        "evaluation_state_count": 30,
        "development_state_count": 2,
        "primary_outcome": "four_way_correct_future_source_action_identification",
        "chance_rate": 0.25,
        "repository_commit": "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb",
        "checkpoint": "robbyant/lingbot-va-posttrain-libero-long",
        "states": records,
    }
    _atomic_json(args.output_root / "manifest.json", manifest)
    print(args.output_root / "manifest.json")


if __name__ == "__main__":
    main()
