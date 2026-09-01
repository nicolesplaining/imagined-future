"""Attach robust natural continuation labels to a frozen pair manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagined_future.study_design import matched_same_label_donor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--continuation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--continuation-seeds", type=int, nargs="+", default=[353, 359, 367])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing labeled manifest: {args.output}")

    manifest = json.loads(args.manifest.read_text())
    records: dict[tuple[str, int], dict] = {}
    for path in args.continuation_files:
        result = json.loads(path.read_text())
        unit_id = Path(result["branch_run"]).name
        key = (unit_id, int(result["continuation_seed"]))
        if key in records:
            raise ValueError(f"duplicate continuation result for {key}")
        records[key] = result

    for unit in manifest["units"]:
        unit_id = unit["unit_id"]
        selected = set(int(index) for index in unit["primary_pair"])
        selected.update(int(index) for index in unit["distance_matched_control_by_direction"].values())
        outcomes_by_branch: dict[int, list[bool]] = {index: [] for index in sorted(selected)}
        for seed in args.continuation_seeds:
            result = records.get((unit_id, seed))
            if result is None:
                raise ValueError(f"missing continuation result for {unit_id}, seed {seed}")
            by_branch = {int(outcome["branch_index"]): bool(outcome["success"]) for outcome in result["outcomes"]}
            for branch_index in outcomes_by_branch:
                if branch_index not in by_branch:
                    raise ValueError(f"{unit_id}, seed {seed} omits branch {branch_index}")
                outcomes_by_branch[branch_index].append(by_branch[branch_index])

        robust_labels: list[bool | None] = [None] * len(unit["branch_seeds"])
        label_records = {}
        for branch_index, outcomes in outcomes_by_branch.items():
            robust = outcomes[0] if all(value == outcomes[0] for value in outcomes) else None
            robust_labels[branch_index] = robust
            label_records[str(branch_index)] = {"outcomes": outcomes, "robust_label": robust}
        unit["continuation_labels"] = label_records

        action_distances = np.asarray(unit["action_distances"], dtype=np.float64)
        endpoint_distances = np.asarray(unit["physical_endpoint_distances"], dtype=np.float64)
        left, right = (int(index) for index in unit["primary_pair"])
        unit["same_outcome_control_by_direction"] = {
            "forward": matched_same_label_donor(
                recipient=left,
                primary_donor=right,
                labels=robust_labels,
                action_distances=action_distances,
                endpoint_distances=endpoint_distances,
            ),
            "reverse": matched_same_label_donor(
                recipient=right,
                primary_donor=left,
                labels=robust_labels,
                action_distances=action_distances,
                endpoint_distances=endpoint_distances,
            ),
        }
        unit["robust_primary_outcome_contrast"] = (
            robust_labels[left] is not None
            and robust_labels[right] is not None
            and robust_labels[left] != robust_labels[right]
        )

    manifest["scope"] = "confirmatory natural pairs with robust shared-continuation labels"
    manifest["continuation_seeds"] = args.continuation_seeds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "units": len(manifest["units"]),
                "robust_outcome_contrasts": sum(
                    bool(unit["robust_primary_outcome_contrast"]) for unit in manifest["units"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
