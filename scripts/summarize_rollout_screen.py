"""Create deterministic machine-readable tables from rollout-screen artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from imagined_future.rollout_summary import compress_predicate_trajectory, first_true_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    episode_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for root in args.roots:
        for summary_path in sorted(root.glob("task_*/state_*/seed_*/summary.json")):
            summary = json.loads(summary_path.read_text())
            key = (
                int(summary["task_id"]),
                int(summary["initial_state_index"]),
                int(summary["model_seed"]),
            )
            if key in seen:
                raise ValueError(f"duplicate task/state/seed episode: {key}")
            seen.add(key)
            predicates_path = summary_path.with_name("predicates.json")
            predicates = json.loads(predicates_path.read_text()) if predicates_path.exists() else []
            first_steps = first_true_steps(predicates)
            episode_rows.append(
                {
                    "task_id": key[0],
                    "initial_state_index": key[1],
                    "model_seed": key[2],
                    "success": bool(summary["success"]),
                    "success_step": summary.get("success_step"),
                    "policy_steps": summary.get("policy_steps"),
                    "num_queries": summary.get("num_queries"),
                    "error": summary.get("error"),
                    "first_true_steps": first_steps,
                    "artifact_dir": str(summary_path.parent.resolve()),
                }
            )
            transition_rows.append(
                {
                    "task_id": key[0],
                    "initial_state_index": key[1],
                    "model_seed": key[2],
                    "transitions": compress_predicate_trajectory(predicates),
                }
            )

    by_state: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        by_state[(row["task_id"], row["initial_state_index"])].append(row)
    state_rows = []
    for (task_id, state_index), rows in sorted(by_state.items()):
        successes = sum(row["success"] for row in rows)
        state_rows.append(
            {
                "task_id": task_id,
                "initial_state_index": state_index,
                "episodes": len(rows),
                "successes": successes,
                "failures": len(rows) - successes,
                "mixed_outcome": 0 < successes < len(rows),
                "successful_seeds": [row["model_seed"] for row in rows if row["success"]],
                "failed_seeds": [row["model_seed"] for row in rows if not row["success"]],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": len(episode_rows),
        "successes": sum(row["success"] for row in episode_rows),
        "failures": sum(not row["success"] for row in episode_rows),
        "errors": sum(row["error"] is not None for row in episode_rows),
        "states": state_rows,
        "episode_rows": episode_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "predicate_transitions.json").write_text(
        json.dumps(transition_rows, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "episodes.csv").open("w", newline="") as handle:
        columns = [
            "task_id",
            "initial_state_index",
            "model_seed",
            "success",
            "success_step",
            "policy_steps",
            "num_queries",
            "error",
            "first_true_steps",
            "artifact_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow({**row, "first_true_steps": json.dumps(row["first_true_steps"], sort_keys=True)})
    print(json.dumps({key: payload[key] for key in ("episodes", "successes", "failures", "errors")}))


if __name__ == "__main__":
    main()
