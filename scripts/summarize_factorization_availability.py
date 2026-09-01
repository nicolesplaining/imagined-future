"""Summarize preregistered natural-pair availability without interventions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from imagined_future.content_factorization import (
    FactorizationThresholds,
    endpoint_distance_matrices,
    pair_class_masks,
)
from imagined_future.libero_semantics import goal_feature_vector


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pool_row(unit: dict, candidate: dict) -> dict:
    run_dir = Path(candidate["final_dir"])
    artifact = np.load(run_dir / "branches.npz", allow_pickle=False)
    predicates = json.loads((run_dir / "endpoint_predicates.json").read_text())
    objects = np.stack(
        [goal_feature_vector(record["snapshot"]) for record in predicates]
    )
    matrices = endpoint_distance_matrices(
        artifact["normalized_branch_actions"],
        artifact["endpoint_proprios"],
        objects,
    )
    masks = pair_class_masks(matrices, FactorizationThresholds())
    common_recipients = sum(
        bool(np.any(masks["object"][index]))
        and bool(np.any(masks["robot"][index]))
        for index in range(len(objects))
    )
    row = {
        "unit_id": unit["unit_id"],
        "task_id": unit["task_id"],
        "initial_state_index": unit["initial_state_index"],
        "candidate_rank": candidate["candidate_rank"],
        "prefix_chunks": candidate["prefix_chunks"],
        "branches": len(objects),
        "eligible": bool(candidate["eligible"]),
        "object_pairs": int(np.triu(masks["object"], 1).sum()),
        "robot_pairs": int(np.triu(masks["robot"], 1).sum()),
        "joint_pairs": int(np.triu(masks["joint"], 1).sum()),
        "common_recipients": common_recipients,
    }
    for name, matrix in matrices.items():
        upper = matrix[np.triu_indices(len(matrix), k=1)]
        row[f"maximum_{name}"] = float(np.max(upper, initial=0.0))
        row[f"median_{name}"] = float(np.median(upper)) if upper.size else 0.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite summary: {args.output_dir}")

    units = []
    rows = []
    for path in args.manifests:
        manifest = json.loads(path.read_text())
        for unit in manifest["units"]:
            units.append(unit)
            rows.extend(
                _pool_row(unit, candidate)
                for candidate in unit["candidates_evaluated"]
            )
    result = {
        "scope": "preregistered natural-pair structural availability; no intervention outcomes",
        "candidate_trajectories": len(units),
        "structurally_eligible_units": sum(
            unit["structurally_eligible"] for unit in units
        ),
        "units_with_any_object_pair": len(
            {row["unit_id"] for row in rows if row["object_pairs"] > 0}
        ),
        "units_with_any_robot_pair": len(
            {row["unit_id"] for row in rows if row["robot_pairs"] > 0}
        ),
        "units_with_a_common_recipient": len(
            {row["unit_id"] for row in rows if row["common_recipients"] > 0}
        ),
        "candidate_pools": len(rows),
        "thresholds": FactorizationThresholds().__dict__,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "candidate_pool_availability.csv", rows)
    (args.output_dir / "candidate_pool_availability.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir), **result}, indent=2))


if __name__ == "__main__":
    main()
