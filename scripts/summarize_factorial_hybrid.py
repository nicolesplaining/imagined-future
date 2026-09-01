"""Aggregate saved-state-clustered 2x2 object/robot effects."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from imagined_future.factorial import two_by_two_effects
from imagined_future.statistics import cluster_bootstrap_mean, exact_sign_test

CELL_NAMES = ("o0r0", "o1r0", "o0r1", "o1r1")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean_by_unit(rows: list[dict], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            grouped[row["unit_id"]].append(float(value))
    return {unit: float(np.mean(values)) for unit, values in grouped.items()}


def _estimate(rows: list[dict], key: str) -> dict | None:
    unit_values = _mean_by_unit(rows, key)
    if not unit_values:
        return None
    values = list(unit_values.values())
    return {
        **cluster_bootstrap_mean(unit_values),
        "median": float(np.median(values)),
        "sign_test": exact_sign_test(values),
        "unit_values": unit_values,
    }


def _run_rows(run_dir: Path) -> tuple[list[dict], list[dict]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    execution = json.loads((run_dir / "execution_analysis.json").read_text())
    action_by_name = {row["condition"]: row for row in summary["rows"]}
    endpoint_by_name = {row["condition"]: row for row in execution["rows"]}
    effect_rows = []
    cell_rows = []
    for noise_seed in summary["future_noise_seeds"]:
        for modality in ("all", "wrist", "primary", "proprio"):
            names = {
                cell: f"n{noise_seed}_{modality}_{cell}" for cell in CELL_NAMES
            }
            action_values = {
                cell: -float(action_by_name[name]["action_l2_to_native"])
                for cell, name in names.items()
            }
            metrics = {"native_action_similarity": action_values}
            if modality == "all":
                gaussian_name = f"n{noise_seed}_all_gaussian"
                metrics.update(
                    {
                        "goal_endpoint_donor_steering": {
                            cell: float(
                                endpoint_by_name[name][
                                    "goal_endpoint_donor_steering"
                                ]
                            )
                            for cell, name in names.items()
                        },
                        "robot_endpoint_donor_steering": {
                            cell: float(
                                endpoint_by_name[name][
                                    "robot_endpoint_donor_steering"
                                ]
                            )
                            for cell, name in names.items()
                        },
                    }
                )
            effects_by_metric = {
                metric: two_by_two_effects(values)
                for metric, values in metrics.items()
            }
            row = {
                "unit_id": summary["unit_id"],
                "task_id": summary["task_id"],
                "initial_state_index": summary["initial_state_index"],
                "prefix_chunks": summary["prefix_chunks"],
                "future_noise_seed": noise_seed,
                "modality": modality,
            }
            for metric, effects in effects_by_metric.items():
                for effect, value in effects.items():
                    row[f"{metric}__{effect}"] = value
            if modality == "all":
                row["native_action_similarity__o1r1_minus_gaussian"] = (
                    -float(action_by_name[names["o1r1"]]["action_l2_to_native"])
                    + float(action_by_name[gaussian_name]["action_l2_to_native"])
                )
                row["goal_endpoint_donor_steering__o1r1_minus_gaussian"] = (
                    float(
                        endpoint_by_name[names["o1r1"]][
                            "goal_endpoint_donor_steering"
                        ]
                    )
                    - float(
                        endpoint_by_name[gaussian_name][
                            "goal_endpoint_donor_steering"
                        ]
                    )
                )
                row["robot_endpoint_donor_steering__o1r1_minus_gaussian"] = (
                    float(
                        endpoint_by_name[names["o1r1"]][
                            "robot_endpoint_donor_steering"
                        ]
                    )
                    - float(
                        endpoint_by_name[gaussian_name][
                            "robot_endpoint_donor_steering"
                        ]
                    )
                )
            effect_rows.append(row)

            if modality == "all":
                for cell, name in names.items():
                    endpoint = endpoint_by_name[name]
                    cell_rows.append(
                        {
                            "unit_id": summary["unit_id"],
                            "task_id": summary["task_id"],
                            "initial_state_index": summary["initial_state_index"],
                            "prefix_chunks": summary["prefix_chunks"],
                            "future_noise_seed": noise_seed,
                            "target_cell": cell,
                            "goal_endpoint_donor_steering": endpoint[
                                "goal_endpoint_donor_steering"
                            ],
                            "robot_endpoint_donor_steering": endpoint[
                                "robot_endpoint_donor_steering"
                            ],
                            "correct_target_cell": endpoint[
                                "correct_target_cell"
                            ],
                            "target_coordinate_distance": endpoint[
                                "target_coordinate_distance"
                            ],
                            "decoded_primary_l1_to_target": action_by_name[name][
                                "decoded_primary_l1_to_target"
                            ],
                            "decoded_wrist_l1_to_target": action_by_name[name][
                                "decoded_wrist_l1_to_target"
                            ],
                            "decoded_primary_target_top1": action_by_name[name][
                                "decoded_primary_target_top1"
                            ],
                            "decoded_wrist_target_top1": action_by_name[name][
                                "decoded_wrist_target_top1"
                            ],
                            "decoded_primary_target_margin": action_by_name[name][
                                "decoded_primary_target_margin"
                            ],
                            "decoded_wrist_target_margin": action_by_name[name][
                                "decoded_wrist_target_margin"
                            ],
                        }
                    )
    return effect_rows, cell_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite summary: {args.output_dir}")

    manifest_units = []
    for path in args.manifests:
        manifest_units.extend(json.loads(path.read_text())["units"])
    effects = []
    cells = []
    for run_dir in args.run_dirs:
        run_effects, run_cells = _run_rows(run_dir)
        effects.extend(run_effects)
        cells.extend(run_cells)
    estimates = {
        key: estimate
        for key in sorted({key for row in effects for key in row if "__" in key})
        if (estimate := _estimate(effects, key)) is not None
    }
    cell_estimates = {
        key: estimate
        for key in (
            "correct_target_cell",
            "target_coordinate_distance",
            "decoded_primary_l1_to_target",
            "decoded_wrist_l1_to_target",
            "decoded_primary_target_top1",
            "decoded_wrist_target_top1",
            "decoded_primary_target_margin",
            "decoded_wrist_target_margin",
        )
        if (estimate := _estimate(cells, key)) is not None
    }
    result = {
        "scope": "held-out saved-state-clustered 2x2 rendered object/robot estimates",
        "candidate_trajectories": len(manifest_units),
        "validated_units": sum(unit["valid"] for unit in manifest_units),
        "analyzed_units": len({row["unit_id"] for row in effects}),
        "smallest_effect_of_interest": 0.10,
        "estimates": estimates,
        "target_cell_identification": cell_estimates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "factorial_effect_repetitions.csv", effects)
    _write_csv(args.output_dir / "factorial_cell_repetitions.csv", cells)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "validated": result["validated_units"],
                "analyzed": result["analyzed_units"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
