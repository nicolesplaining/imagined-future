#!/usr/bin/env python3
"""Freeze the identical 92-call dose matrix on one excluded Bagels state."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from imagined_future.cosmos3_archival import atomic_json, canonical_json, sha256
from imagined_future.cosmos3_dose_response import ALPHAS, frozen_request_specs


EVALUATION_TASKS = {
    "BananaInBowlTask",
    "RubiksCubeTask",
    "MustardInLeftBinTask",
    "SpoonInMugTask",
    "MarkerInMugTask",
    "SmartphoneInBinTask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-manifest", type=Path, required=True)
    parser.add_argument("--expected-main-manifest-sha256", required=True)
    parser.add_argument("--source-excluded-manifest", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke manifest: {args.output}")
    if sha256(args.main_manifest) != args.expected_main_manifest_sha256:
        raise ValueError("main dose manifest SHA differs from the frozen CLI value")
    main = json.loads(args.main_manifest.read_text(encoding="utf-8"))
    if (
        main.get("study_name") != "cosmos3-future-strength-dose-response-v2"
        or main.get("admission")
        != "prospective_action_level_future_strength_dose_response"
        or len(main.get("states", [])) != 30
    ):
        raise ValueError("main dose manifest is not the complete prospective cohort")
    snapshot_root = args.snapshot_root.resolve()
    if snapshot_root.as_posix() != main["runtime"].get("snapshot_root"):
        raise ValueError("snapshot root differs from main manifest")
    for relative, expected in main["runtime"].get("snapshot_file_sha256", {}).items():
        path = snapshot_root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"snapshot differs for {relative}")

    excluded = json.loads(args.source_excluded_manifest.read_text(encoding="utf-8"))
    if (
        excluded.get("admission") != "excluded_development_smoke"
        or excluded.get("status") != "frozen_before_model_outcomes"
        or excluded.get("selection_uses_model_or_intervention_outcomes") is not False
        or len(excluded.get("states", [])) != 1
    ):
        raise ValueError("excluded source is not the frozen one-state engineering manifest")
    unit = deepcopy(excluded["states"][0])
    if unit.get("task") in EVALUATION_TASKS or "Bagels" not in str(unit.get("task")):
        raise ValueError("dose smoke must use the excluded Bagels task")
    if any(unit["unit_id"] == row["unit_id"] for row in main["states"]):
        raise ValueError("excluded state overlaps the powered cohort")
    branch_seeds = tuple(int(seed) for seed in unit["branch_seeds"])
    specs = list(frozen_request_specs(branch_seeds))
    unit["phase"] = "middle"
    unit["phase_fraction"] = 0.5
    unit["alpha_grid"] = list(ALPHAS)
    unit["ordered_pairs"] = [
        [row["recipient_seed"], row["donor_seed"]] for row in specs[20:80:5]
    ]
    unit["request_sequence"] = specs

    body = deepcopy(main)
    body.pop("manifest_id", None)
    body["admission"] = "excluded_development_smoke"
    body["launch_authorization"] = "excluded_smoke_only_not_powered_evaluation"
    body["scope"]["excluded_development_smoke"] = True
    body["scope"]["admitted_to_evaluation"] = False
    body["source"]["main_dose_manifest_path"] = str(args.main_manifest.resolve())
    body["source"]["main_dose_manifest_id"] = main["manifest_id"]
    body["source"]["main_dose_manifest_sha256"] = args.expected_main_manifest_sha256
    body["source"]["excluded_source_manifest_path"] = str(
        args.source_excluded_manifest.resolve()
    )
    body["source"]["excluded_source_manifest_sha256"] = sha256(
        args.source_excluded_manifest
    )
    body["design"]["tasks"] = [str(unit["task"])]
    body["design"]["environment_seeds"] = [int(unit["environment_seed"])]
    body["states"] = [unit]
    manifest_id = "cosmos3-dose-excluded-smoke-" + hashlib.sha256(
        canonical_json(body)
    ).hexdigest()[:16]
    manifest = {"manifest_id": manifest_id, **body}
    atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "status": "frozen",
                "manifest_id": manifest_id,
                "manifest_sha256": sha256(args.output),
                "unit_id": unit["unit_id"],
                "request_count": 92,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
