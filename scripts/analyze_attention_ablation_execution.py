"""Analyze physical endpoints produced by a future-attention ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imagined_future.metrics import donor_steering
from imagined_future.paired_rollouts import pixel_l1


def _steering(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> float:
    return float(
        donor_steering(
            torch.from_numpy(value.astype(np.float64)).unsqueeze(0),
            torch.from_numpy(recipient.astype(np.float64)).unsqueeze(0),
            torch.from_numpy(donor.astype(np.float64)).unsqueeze(0),
        ).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--ablation-run-dir", type=Path, required=True)
    parser.add_argument("--execution-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    summary = json.loads((args.ablation_run_dir / "summary.json").read_text())
    branch = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    execution = np.load(
        args.execution_artifact.with_name(f"{args.execution_artifact.stem}_endpoint_states.npz"),
        allow_pickle=False,
    )
    recipient_index = int(summary["recipient_branch"])
    donor_index = int(summary["donor_branch"])
    references = {
        "state": (branch["endpoint_states"][recipient_index], branch["endpoint_states"][donor_index]),
        "proprio": (
            branch["endpoint_proprios"][recipient_index],
            branch["endpoint_proprios"][donor_index],
        ),
        "image": (
            branch["endpoint_primary_images"][recipient_index],
            branch["endpoint_primary_images"][donor_index],
        ),
    }
    conditions = {}
    for index, name_raw in enumerate(execution["names"]):
        name = str(name_raw)
        state = execution["endpoint_states"][index]
        proprio = execution["endpoint_proprios"][index]
        image = np.asarray(
            Image.open(
                args.ablation_run_dir
                / f"{args.execution_artifact.stem}_{name}_endpoint_primary.png"
            )
        )
        image_to_recipient = pixel_l1(image, references["image"][0])
        image_to_donor = pixel_l1(image, references["image"][1])
        conditions[name] = {
            "state_donor_steering": _steering(state, *references["state"]),
            "proprio_donor_steering": _steering(proprio, *references["proprio"]),
            "primary_pixel_donor_preference": image_to_recipient - image_to_donor,
            "state_l2_from_recipient": float(np.linalg.norm(state - references["state"][0])),
            "proprio_l2_from_recipient": float(np.linalg.norm(proprio - references["proprio"][0])),
            "primary_pixel_l1_from_recipient": image_to_recipient,
        }
    for required in ("baseline", "all_key_control", "future_blocked"):
        if required not in conditions:
            raise ValueError(f"execution artifact is missing {required}")
    result = {
        "scope": "exact-state endpoint analysis for future-attention ablation",
        "branch_run": str(args.branch_run_dir),
        "ablation_run": str(args.ablation_run_dir),
        "conditions": conditions,
        "future_blocked_minus_baseline": {
            key: conditions["future_blocked"][key] - conditions["baseline"][key]
            for key in (
                "state_donor_steering",
                "proprio_donor_steering",
                "primary_pixel_donor_preference",
            )
        },
        "all_key_control_minus_baseline": {
            key: conditions["all_key_control"][key] - conditions["baseline"][key]
            for key in (
                "state_donor_steering",
                "proprio_donor_steering",
                "primary_pixel_donor_preference",
            )
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
