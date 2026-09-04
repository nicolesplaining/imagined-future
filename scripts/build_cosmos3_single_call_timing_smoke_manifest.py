#!/usr/bin/env python3
"""Freeze one excluded-state smoke for the exact timing-v5 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from imagined_future.cosmos3_archival import atomic_json, canonical_json, sha256
from imagined_future.cosmos3_single_call_timing import (
    BRANCH_SEEDS,
    REQUESTS_PER_STATE,
    TASKS,
)


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
        raise FileExistsError(f"refusing to overwrite frozen smoke manifest: {args.output}")
    actual_main_hash = sha256(args.main_manifest)
    if actual_main_hash != args.expected_main_manifest_sha256:
        raise ValueError(
            f"main manifest SHA mismatch: {actual_main_hash} != "
            f"{args.expected_main_manifest_sha256}"
        )
    main_manifest = json.loads(args.main_manifest.read_text(encoding="utf-8"))
    if main_manifest.get("status") != "frozen_before_model_outcomes":
        raise ValueError("main manifest is not a pre-outcome freeze")
    if main_manifest.get("study_name") != "cosmos3-single-call-timing-v5":
        raise ValueError("main manifest is not timing v5")
    if main_manifest.get("admission") != "frozen_single_call_timing_evaluation":
        raise ValueError("main manifest is not the evaluation cohort")
    if len(main_manifest.get("states", [])) != 30:
        raise ValueError("main timing manifest does not contain exactly 30 states")

    snapshot_root = args.snapshot_root.resolve()
    if snapshot_root.as_posix() != main_manifest["runtime"]["snapshot_root"]:
        raise ValueError("smoke snapshot root differs from the main manifest")
    for relative, expected_hash in main_manifest["runtime"][
        "snapshot_file_sha256"
    ].items():
        path = snapshot_root / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"smoke snapshot differs for {relative}")

    excluded_hash = sha256(args.source_excluded_manifest)
    excluded = json.loads(args.source_excluded_manifest.read_text(encoding="utf-8"))
    if excluded.get("status") != "frozen_before_model_outcomes":
        raise ValueError("excluded source manifest is not pre-outcome")
    if excluded.get("selection_uses_model_or_intervention_outcomes") is not False:
        raise ValueError("excluded source manifest used model outcomes")
    if excluded.get("admission") != "excluded_development_smoke":
        raise ValueError("source manifest is not explicitly excluded development data")
    excluded_states = list(excluded.get("states", []))
    if len(excluded_states) != 1:
        raise ValueError("excluded source must contain exactly one state")
    unit = deepcopy(excluded_states[0])
    if str(unit.get("task")) in TASKS:
        raise ValueError("smoke state overlaps an evaluation task")
    if tuple(int(seed) for seed in unit.get("branch_seeds", [])) != BRANCH_SEEDS:
        raise ValueError("excluded state branch seeds differ from timing v5")
    if any(unit["unit_id"] == state["unit_id"] for state in main_manifest["states"]):
        raise ValueError("excluded state identifier overlaps the evaluation population")

    body = deepcopy(main_manifest)
    body.pop("manifest_id", None)
    body["admission"] = "excluded_development_smoke"
    body["launch_authorization"] = "excluded_smoke_only_not_evaluation"
    body["scope"]["excluded_development_smoke"] = True
    body["scope"]["admitted_to_evaluation"] = False
    body["source"]["main_timing_manifest"] = str(args.main_manifest.resolve())
    body["source"]["main_timing_manifest_id"] = main_manifest["manifest_id"]
    body["source"]["main_timing_manifest_sha256"] = actual_main_hash
    body["source"]["excluded_archival_manifest"] = str(
        args.source_excluded_manifest.resolve()
    )
    body["source"]["excluded_archival_manifest_sha256"] = excluded_hash
    body["source"]["excluded_state_copy_rule"] = (
        "the sole state from the pre-outcome excluded archival smoke manifest"
    )
    body["design"]["tasks"] = [str(unit["task"])]
    body["design"]["environment_seeds"] = [int(unit["environment_seed"])]
    body["design"]["state_count"] = 1
    body["design"]["requests_per_state"] = REQUESTS_PER_STATE
    body["design"]["total_request_count"] = REQUESTS_PER_STATE
    body["states"] = [unit]
    manifest_id = "cosmos3-timing-excluded-smoke-" + hashlib.sha256(
        canonical_json(body)
    ).hexdigest()[:16]
    manifest = {"manifest_id": manifest_id, **body}
    atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "status": "frozen",
                "admission": manifest["admission"],
                "manifest_id": manifest_id,
                "manifest_sha256": sha256(args.output),
                "unit_id": unit["unit_id"],
                "request_count": REQUESTS_PER_STATE,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
