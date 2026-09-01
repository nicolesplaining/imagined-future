"""Reconstruct a frozen natural-pair manifest from immutable branch pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from imagined_future.content_factorization import FactorizationThresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.output}")

    screen_units = []
    for path in args.screens:
        screen_units.extend(json.loads(path.read_text())["units"])
    manifest_units = []
    for unit in sorted(
        screen_units, key=lambda value: (value["task_id"], value["initial_state_index"])
    ):
        selected = None
        candidate_records = []
        for rank, prefix_chunks in enumerate(
            unit["candidate_prefix_chunks"], start=1
        ):
            stem = f"{unit['unit_id']}_prefix{prefix_chunks:02d}"
            stage_one_dir = args.branch_root / f"{stem}_stage1"
            if not (stage_one_dir / "summary.json").exists():
                raise FileNotFoundError(f"missing frozen stage-one pool: {stage_one_dir}")
            stage_one = json.loads((stage_one_dir / "summary.json").read_text())
            final_dir = stage_one_dir
            final_summary = stage_one
            if stage_one["selection"] is None:
                final_dir = args.branch_root / f"{stem}_stage2"
                if not (final_dir / "summary.json").exists():
                    raise FileNotFoundError(f"missing frozen stage-two pool: {final_dir}")
                final_summary = json.loads((final_dir / "summary.json").read_text())
            eligible = final_summary["selection"] is not None
            candidate_record = {
                "candidate_rank": rank,
                "prefix_chunks": prefix_chunks,
                "stage_one_dir": str(stage_one_dir),
                "final_dir": str(final_dir),
                "eligible": eligible,
                "selection_error": final_summary["selection_error"],
            }
            candidate_records.append(candidate_record)
            if eligible:
                selected = {
                    **candidate_record,
                    "branch_run_dir": str(final_dir),
                    "selection": final_summary["selection"],
                    "artifact_sha256": {
                        name: hashlib.sha256((final_dir / name).read_bytes()).hexdigest()
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
    result = {
        "scope": "reconstructed held-out natural factorization manifest from immutable pools",
        "screens": [str(path) for path in args.screens],
        "thresholds": FactorizationThresholds().__dict__,
        "units": manifest_units,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
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
