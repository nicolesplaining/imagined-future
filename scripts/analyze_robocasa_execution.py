"""Measure RoboCasa executed endpoints relative to natural recipient/donor branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _steering(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> float:
    direction = donor.astype(np.float64) - recipient.astype(np.float64)
    denominator = float(np.dot(direction.reshape(-1), direction.reshape(-1)))
    if denominator == 0.0:
        raise ValueError("recipient and donor endpoints are identical")
    return float(np.dot((value - recipient).reshape(-1), direction.reshape(-1)) / denominator)


def _pixel_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.float64) - right.astype(np.float64)).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    unit = next(item for item in manifest["units"] if item["unit_id"] == args.unit_id)
    branch = np.load(Path(unit["branch_run_dir"]) / "branches.npz", allow_pickle=False)
    execution = json.loads(args.execution.read_text())
    endpoint = np.load(execution["endpoint_artifact"], allow_pickle=False)
    by_condition = {str(name): index for index, name in enumerate(endpoint["conditions"].tolist())}
    left, right = (int(index) for index in unit["primary_pair"])
    direction_pairs = {"forward": (left, right), "reverse": (right, left)}
    rows = []
    execution_by_name = {row["condition"]: row for row in execution["rows"]}
    for condition, index in sorted(by_condition.items()):
        direction = condition.split("_", 1)[0]
        if direction not in direction_pairs:
            raise ValueError(f"condition has no registered direction: {condition}")
        recipient_index, donor_index = direction_pairs[direction]
        image = np.asarray(Image.open(execution_by_name[condition]["endpoint_primary_image"]))
        recipient_image = branch["endpoint_primary_images"][recipient_index]
        donor_image = branch["endpoint_primary_images"][donor_index]
        rows.append(
            {
                "condition": condition,
                "direction": direction,
                "recipient_branch": recipient_index,
                "donor_branch": donor_index,
                "endpoint_success": execution_by_name[condition]["success"],
                "physical_endpoint_donor_steering": _steering(
                    endpoint["physical_features"][index],
                    branch["endpoint_physical_features"][recipient_index],
                    branch["endpoint_physical_features"][donor_index],
                ),
                "proprio_endpoint_donor_steering": _steering(
                    endpoint["endpoint_proprios"][index],
                    branch["endpoint_proprios"][recipient_index],
                    branch["endpoint_proprios"][donor_index],
                ),
                "primary_pixel_donor_preference": _pixel_l1(image, recipient_image)
                - _pixel_l1(image, donor_image),
                "primary_pixel_l1_to_recipient": _pixel_l1(image, recipient_image),
                "primary_pixel_l1_to_donor": _pixel_l1(image, donor_image),
            }
        )
    result = {
        "scope": "RoboCasa executed endpoint donor steering",
        "unit_id": args.unit_id,
        "manifest": str(args.manifest),
        "execution": str(args.execution),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
