#!/usr/bin/env python3
"""Materialize the outcome-blind overnight Cosmos 3 evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path


def build_manifest(config_path: Path, recording_root: Path) -> dict:
    raw = config_path.read_bytes()
    config = tomllib.loads(raw.decode())
    study = config["study"]
    design = config["cosmos3_selection_free"]
    if design["status"] != "frozen_before_outcomes":
        raise ValueError("selection-free design is not frozen")

    branch_seeds = [int(value) for value in design["branch_seeds"]]
    recipient_seed = int(design["recipient_seed"])
    donor_seeds = [int(value) for value in design["donor_seeds"]]
    if len(branch_seeds) != 4 or len(set(branch_seeds)) != 4:
        raise ValueError("the primary donor-identification set must contain four unique seeds")
    if branch_seeds != [recipient_seed, *donor_seeds]:
        raise ValueError("recipient and donor order must exactly match branch_seeds")

    candidates = []
    for task in design["tasks"]:
        branch_step = int(design["branch_points"][task])
        target_object_name = str(design["target_objects"][task])
        for environment_seed in design["environment_seeds"]:
            unit_id = f"{task}-seed-{int(environment_seed)}"
            unit_root = recording_root / f"seed_{int(environment_seed)}" / task
            candidates.append(
                {
                    "unit_id": unit_id,
                    "task": task,
                    "environment_seed": int(environment_seed),
                    "branch_step": branch_step,
                    "target_object_name": target_object_name,
                    "branch_seeds": branch_seeds,
                    "recipient_seed": recipient_seed,
                    "donor_seeds": donor_seeds,
                    "action_ordered_pairs": [
                        [source, target]
                        for source in branch_seeds
                        for target in branch_seeds
                        if source != target
                    ],
                    "physical_recipient_seed": recipient_seed,
                    "physical_donor_seeds": donor_seeds,
                    "recorded_hdf5": str(unit_root / "run_0.hdf5"),
                    "recorded_env_cfg": str(unit_root / "env_cfg.json"),
                    "selected": True,
                }
            )

    return {
        "study_id": study["study_id"],
        "status": design["status"],
        "frozen_at_utc": study["frozen_at_utc"],
        "source_config": str(config_path),
        "source_config_sha256": hashlib.sha256(raw).hexdigest(),
        "selection_uses_native_or_intervention_outcomes": False,
        "selection_rule": design["selection_rule"],
        "allowed_exclusions": design["allowed_exclusions"],
        "replacement_after_outcome": False,
        "candidate_count": len(candidates),
        "task_count": len(design["tasks"]),
        "states_per_task": len(design["environment_seeds"]),
        "primary_outcome": design["primary_outcome"],
        "primary_chance_rate": float(design["primary_chance_rate"]),
        "action_pair_design": design["action_pair_design"],
        "physical_endpoint_design": design["physical_endpoint_design"],
        "secondary_outcomes": design["secondary_outcomes"],
        "required_controls": design["required_controls"],
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
    manifest = build_manifest(args.config, args.recording_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: manifest[key] for key in ("study_id", "candidate_count", "source_config_sha256")}, indent=2))


if __name__ == "__main__":
    main()
