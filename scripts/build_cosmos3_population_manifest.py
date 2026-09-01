#!/usr/bin/env python3
"""Freeze eligible Cosmos 3 population units without intervention outcomes."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py


def read_success(path: Path) -> bool:
    with h5py.File(path, "r") as stream:
        return bool(stream["data/demo_0"].attrs["success"])


def ordered_pair(text: str) -> tuple[int, int]:
    left, right = text.split(":", 1)
    return int(left), int(right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cosmos3_replication.toml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")

    config = tomllib.loads(args.config.read_text())
    screen_cfg = config["screen"]
    expected_seeds = set(screen_cfg["confirmatory_environment_seeds"])
    expected_tasks = sorted(screen_cfg["branch_points"])
    expected_branches = list(screen_cfg["native_branch_seeds"])
    minimum_action = float(screen_cfg["minimum_normalized_action_l2"])
    minimum_endpoint = float(screen_cfg["minimum_endpoint_displacement_m"])
    maximum_per_task = int(screen_cfg["maximum_units_per_task"])

    candidates: list[dict[str, Any]] = []
    summaries_by_unit: dict[tuple[str, int], Path] = {}
    for path in sorted(args.screen_root.rglob("summary.json")):
        row = json.loads(path.read_text())
        if row.get("interventions_evaluated") is not False:
            continue
        task = str(row["task"])
        seed = int(row["environment_seed"])
        if task not in expected_tasks or seed not in expected_seeds:
            continue
        key = (task, seed)
        if key in summaries_by_unit:
            raise ValueError(
                f"multiple completed native screens for {task} seed {seed}: "
                f"{summaries_by_unit[key]} and {path}"
            )
        summaries_by_unit[key] = path

    for task in expected_tasks:
        for seed in sorted(expected_seeds):
            path = summaries_by_unit.get((task, seed))
            attempt_directories = sorted(
                candidate
                for candidate in args.screen_root.glob(f"{task}_seed_{seed}*")
                if candidate.is_dir()
            )
            if path is None:
                attempted = bool(attempt_directories)
                candidates.append(
                    {
                        "unit_id": f"{task}-seed-{seed}",
                        "task": task,
                        "environment_seed": seed,
                        "native_rollout_success": None,
                        "native_screen_attempted": attempted,
                        "native_screen_summary_present": False,
                        "screen_attempt_directories": [
                            str(candidate) for candidate in attempt_directories
                        ],
                        "selected": False,
                        "factorization_selected": False,
                        "eligible_before_task_rule": False,
                        "exclusion_reasons": [
                            "native_screen_failed" if attempted else "native_screen_missing"
                        ],
                    }
                )
                continue

            row = json.loads(path.read_text())
            recorded_hdf5 = Path(row["recorded_hdf5"])
            reasons: list[str] = []
            if list(row["branch_seeds"]) != expected_branches:
                reasons.append("native_branch_seed_census_changed")
            if float(row["prefix_maximum_state_error"]) != 0.0:
                reasons.append("prefix_replay_not_exact")
            if not row["continuation_repeat_audit"]["exact"]:
                reasons.append("continuation_repeat_not_exact")
            if len(set(row["restore_state_digests"])) != 1:
                reasons.append("restore_state_digest_mismatch")
            if float(row["native_action_l2"]) < minimum_action:
                reasons.append("native_action_separation_below_threshold")
            if float(row["native_endpoint_l2"]["all"]) < minimum_endpoint:
                reasons.append("native_endpoint_separation_below_threshold")

            selected_pair = (int(row["recipient_seed"]), int(row["donor_seed"]))
            action_pairs = {
                ordered_pair(key): float(value)
                for key, value in row["native_pairwise_action_l2"].items()
            }
            expected_pairs = {
                (left, right)
                for index, left in enumerate(expected_branches)
                for right in expected_branches[index + 1 :]
            }
            if set(action_pairs) != expected_pairs:
                reasons.append("native_action_pair_census_changed")
            frozen_pair = (
                max(
                    action_pairs,
                    key=lambda pair: (action_pairs[pair], -pair[0], -pair[1]),
                )
                if action_pairs
                else (-1, -1)
            )
            if selected_pair != frozen_pair:
                reasons.append("selected_pair_violates_frozen_rule")

            candidates.append(
                {
                    "unit_id": f"{task}-seed-{seed}",
                    "task": task,
                    "environment_seed": seed,
                    "native_rollout_success": read_success(recorded_hdf5),
                    "native_screen_attempted": True,
                    "native_screen_summary_present": True,
                    "screen_attempt_directories": [
                        str(candidate) for candidate in attempt_directories
                    ],
                    "branch_step": int(row["branch_step"]),
                    "recipient_seed": selected_pair[0],
                    "donor_seed": selected_pair[1],
                    "native_action_l2": float(row["native_action_l2"]),
                    "native_endpoint_l2": row["native_endpoint_l2"],
                    "target_object_name": str(row["target_object_name"]),
                    "native_target_object_position_l2": float(
                        row["native_target_object_position_l2"]
                    ),
                    "screen_summary": str(path),
                    "recorded_hdf5": str(recorded_hdf5),
                    "eligible_before_task_rule": not reasons,
                    "exclusion_reasons": reasons,
                }
            )

    successful_tasks = {
        row["task"] for row in candidates if row["native_rollout_success"]
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["task"] not in successful_tasks:
            row["exclusion_reasons"].append("task_has_no_successful_native_rollout")
        if not row["exclusion_reasons"]:
            by_task[row["task"]].append(row)

    selected: list[dict[str, Any]] = []
    for task in sorted(by_task, key=lambda name: (-len(by_task[name]), name)):
        ranked = sorted(
            by_task[task],
            key=lambda row: (-float(row["native_endpoint_l2"]["all"]), row["environment_seed"]),
        )
        for row in ranked[:maximum_per_task]:
            row["selected"] = True
            selected.append(row)
        for row in ranked[maximum_per_task:]:
            row["exclusion_reasons"].append("task_cap")

    for row in candidates:
        row.setdefault("selected", False)
    selected_tasks = sorted({row["task"] for row in selected})
    minimum_tasks = int(screen_cfg["minimum_confirmatory_tasks"])
    minimum_units = int(screen_cfg["minimum_confirmatory_units"])
    expected_unit_count = len(expected_tasks) * len(expected_seeds)
    census_complete = (
        len(candidates) == expected_unit_count
        and all(row["native_screen_attempted"] for row in candidates)
    )
    ready = (
        census_complete
        and len(selected_tasks) >= minimum_tasks
        and len(selected) >= minimum_units
    )

    factor_cfg = config["factorization"]
    factor_minimum_object = float(factor_cfg["minimum_object_goal_l2"])
    factor_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        if float(row["native_target_object_position_l2"]) >= factor_minimum_object:
            factor_by_task[row["task"]].append(row)
    factor_selected: list[dict[str, Any]] = []
    for task in sorted(factor_by_task):
        ranked = sorted(
            factor_by_task[task],
            key=lambda row: (
                -float(row["native_target_object_position_l2"]),
                row["environment_seed"],
            ),
        )
        for row in ranked[:2]:
            row["factorization_selected"] = True
            row["factorization_object_prim"] = factor_cfg["object_prims"][task]
            factor_selected.append(row)
    for row in candidates:
        row.setdefault("factorization_selected", False)
    factor_tasks = sorted({row["task"] for row in factor_selected})
    factor_ready = (
        len(factor_tasks) >= int(factor_cfg["confirmatory_minimum_tasks"])
        and len(factor_selected) >= int(factor_cfg["confirmatory_minimum_units"])
    )
    report = {
        "scope": "frozen Cosmos 3 population units selected without intervention outcomes",
        "status": "confirmatory_ready" if ready else "underpowered_feasibility",
        "selection_uses_intervention_outcomes": False,
        "registered_tasks": expected_tasks,
        "registered_environment_seeds": sorted(expected_seeds),
        "registered_native_branch_seeds": expected_branches,
        "expected_unit_count": expected_unit_count,
        "native_screen_attempted_count": sum(
            bool(row["native_screen_attempted"]) for row in candidates
        ),
        "native_screen_completed_count": sum(
            bool(row["native_screen_summary_present"]) for row in candidates
        ),
        "native_screen_census_complete": census_complete,
        "minimum_tasks": minimum_tasks,
        "minimum_units": minimum_units,
        "selected_task_count": len(selected_tasks),
        "selected_unit_count": len(selected),
        "selected_tasks": selected_tasks,
        "selected_unit_ids": [row["unit_id"] for row in selected],
        "factorization_status": (
            "confirmatory_ready" if factor_ready else "underpowered_feasibility"
        ),
        "factorization_selected_task_count": len(factor_tasks),
        "factorization_selected_unit_count": len(factor_selected),
        "factorization_selected_tasks": factor_tasks,
        "factorization_selected_unit_ids": [
            row["unit_id"] for row in factor_selected
        ],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
