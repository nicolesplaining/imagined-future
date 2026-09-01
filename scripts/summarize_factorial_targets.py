"""Summarize validation and distribution-shift diagnostics for 2x2 targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite target audit: {args.output}")

    units = []
    for path in args.manifests:
        for unit in json.loads(path.read_text())["units"]:
            if not unit["valid"]:
                continue
            summary = json.loads((Path(unit["target_dir"]) / "summary.json").read_text())
            natural_contacts = [
                summary["cell_contacts"][name] for name in ("o0r0", "o1r1")
            ]
            hybrid_contacts = [
                summary["cell_contacts"][name] for name in ("o1r0", "o0r1")
            ]
            units.append(
                {
                    "unit_id": unit["unit_id"],
                    "task_id": unit["task_id"],
                    "initial_state_index": unit["initial_state_index"],
                    "prefix_chunks": unit["prefix_chunks"],
                    "native_change_metrics": summary["native_change_metrics"],
                    "validation": summary["validation"],
                    "live_current_vs_rerender": summary[
                        "live_current_vs_rerender"
                    ],
                    "cell_contacts": summary["cell_contacts"],
                    "maximum_hybrid_minus_natural_contacts": max(hybrid_contacts)
                    - max(natural_contacts),
                }
            )
    validation_values = [
        value for unit in units for value in unit["validation"].values()
    ]
    result = {
        "scope": "2x2 target validation and renderer distribution-shift audit",
        "validated_units": len(units),
        "maximum_factor_preservation_error": float(max(validation_values)),
        "native_object_goal_l2": _summary(
            [unit["native_change_metrics"]["object_goal_l2"] for unit in units]
        ),
        "native_robot_position_l2_m": _summary(
            [
                unit["native_change_metrics"]["robot_position_l2_m"]
                for unit in units
            ]
        ),
        "live_current_vs_rerender_primary_pixel_l1": _summary(
            [
                unit["live_current_vs_rerender"]["primary_pixel_l1"]
                for unit in units
            ]
        ),
        "live_current_vs_rerender_wrist_pixel_l1": _summary(
            [
                unit["live_current_vs_rerender"]["wrist_pixel_l1"]
                for unit in units
            ]
        ),
        "maximum_hybrid_minus_natural_contacts": _summary(
            [unit["maximum_hybrid_minus_natural_contacts"] for unit in units]
        ),
        "units": units,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validated_units": len(units),
                "maximum_factor_preservation_error": result[
                    "maximum_factor_preservation_error"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
