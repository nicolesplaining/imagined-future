#!/usr/bin/env python3
"""Freeze a FastWAM Optional-IDM study config into a content-addressed manifest."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, which the pinned FastWAM requires.
    import tomli as tomllib

from imagined_future.fastwam_optional_idm import (
    FastWAMStateSpec,
    build_manifest_body,
    freeze_manifest,
    make_branches,
    write_frozen_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    branches = make_branches(config["video_seeds"], config["action_seeds"])
    if "states" in config and "state_grid" in config:
        raise ValueError("config must use either explicit states or state_grid, not both")
    if "state_grid" in config:
        grid = config["state_grid"]
        state_values = [
            {
                "suite": suite,
                "task_id": task_id,
                "initial_state_index": state_index,
                "wait_steps": grid["wait_steps"],
            }
            for suite, task_id, state_index in itertools.product(
                grid["suites"], grid["task_ids"], grid["initial_state_indices"]
            )
        ]
    else:
        state_values = config.get("states", [])
    states = tuple(
        FastWAMStateSpec(
            suite=str(state["suite"]),
            task_id=int(state["task_id"]),
            initial_state_index=int(state["initial_state_index"]),
            wait_steps=int(state["wait_steps"]),
            branches=branches,
        )
        for state in state_values
    )
    body = build_manifest_body(
        study_name=str(config["study_name"]),
        states=states,
        inference=config["inference"],
    )
    if "design" in config:
        # Design metadata is hashed into the manifest alongside the explicitly
        # expanded state list, but is ignored by the GPU runner.
        body["design"] = config["design"]
    manifest = freeze_manifest(body)
    write_frozen_manifest(args.output, manifest)
    print(f"{manifest['manifest_id']} {args.output}")


if __name__ == "__main__":
    main()
