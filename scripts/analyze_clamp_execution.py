"""Compare executed clamp endpoints with matched recipient and donor branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imagined_future.metrics import donor_steering
from imagined_future.paired_rollouts import pixel_l1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--clamp-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    clamp = json.loads((args.clamp_run_dir / "summary.json").read_text())
    branch = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    execution = np.load(args.clamp_run_dir / "execution_endpoint_states.npz", allow_pickle=False)
    recipient_index = int(clamp["recipient_branch"])
    donor_index = int(clamp["donor_branch"])
    recipient_state = branch["endpoint_states"][recipient_index]
    donor_state = branch["endpoint_states"][donor_index]
    recipient_image = branch["endpoint_primary_images"][recipient_index]
    donor_image = branch["endpoint_primary_images"][donor_index]

    conditions = {}
    for name_raw, state in zip(execution["names"], execution["endpoint_states"], strict=True):
        name = str(name_raw)
        image = np.asarray(Image.open(args.clamp_run_dir / f"execution_{name}_endpoint_primary.png"))
        recipient_pixel_l1 = pixel_l1(image, recipient_image)
        donor_pixel_l1 = pixel_l1(image, donor_image)
        state_score = float(
            donor_steering(
                torch.from_numpy(state.astype(np.float64)).unsqueeze(0),
                torch.from_numpy(recipient_state.astype(np.float64)).unsqueeze(0),
                torch.from_numpy(donor_state.astype(np.float64)).unsqueeze(0),
            ).item()
        )
        conditions[name] = {
            "state_donor_steering": state_score,
            "state_l2_to_recipient": float(np.linalg.norm(state.astype(np.float64) - recipient_state)),
            "state_l2_to_donor": float(np.linalg.norm(state.astype(np.float64) - donor_state)),
            "primary_pixel_l1_to_recipient": recipient_pixel_l1,
            "primary_pixel_l1_to_donor": donor_pixel_l1,
            "primary_pixel_donor_preference": recipient_pixel_l1 - donor_pixel_l1,
        }
    for required in ("recipient_clamp", "donor_clamp"):
        if required not in conditions:
            raise ValueError(f"execution artifact is missing {required}")
    result = {
        "scope": "descriptive exact-state endpoint analysis for a semantic clamp",
        "branch_run": str(args.branch_run_dir),
        "clamp_run": str(args.clamp_run_dir),
        "recipient_branch": recipient_index,
        "donor_branch": donor_index,
        "reference_endpoint_state_l2": float(
            np.linalg.norm(donor_state.astype(np.float64) - recipient_state.astype(np.float64))
        ),
        "reference_endpoint_primary_pixel_l1": pixel_l1(recipient_image, donor_image),
        "conditions": conditions,
        "donor_minus_recipient_clamp": {
            "state_donor_steering": (
                conditions["donor_clamp"]["state_donor_steering"]
                - conditions["recipient_clamp"]["state_donor_steering"]
            ),
            "primary_pixel_donor_preference": (
                conditions["donor_clamp"]["primary_pixel_donor_preference"]
                - conditions["recipient_clamp"]["primary_pixel_donor_preference"]
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
