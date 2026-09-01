"""Aggregate held-out robot-versus-object content-factorization effects."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from imagined_future.metrics import donor_steering
from imagined_future.statistics import cluster_bootstrap_mean, exact_sign_test


def _steering(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> float:
    return float(
        donor_steering(
            torch.from_numpy(np.asarray(value, dtype=np.float64)).unsqueeze(0),
            torch.from_numpy(np.asarray(recipient, dtype=np.float64)).unsqueeze(0),
            torch.from_numpy(np.asarray(donor, dtype=np.float64)).unsqueeze(0),
        ).item()
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _difference(left: dict, right: dict, key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


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


def _run_rows(
    run_dir: Path, execution_analysis_name: str
) -> tuple[list[dict], list[dict]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    execution = json.loads((run_dir / execution_analysis_name).read_text())
    action_by_name = {
        row["condition"]: row
        for row in summary["rows"]
        if row.get("condition") is not None
    }
    endpoint_by_name = {row["condition"]: row for row in execution["rows"]}
    contrast_rows = []
    contexts = ("object_forward", "object_reverse", "robot_forward", "robot_reverse")
    for context in contexts:
        pair_type, direction = context.split("_")
        modalities = (
            ("all", "wrist", "primary", "proprio")
            if direction == "forward"
            else ("all",)
        )
        for noise_seed in summary["future_noise_seeds"]:
            for modality in modalities:
                prefix = f"{context}_n{noise_seed}_{modality}_"
                donor = action_by_name[f"{prefix}donor"]
                recipient = action_by_name[f"{prefix}recipient"]
                row = {
                    "unit_id": summary["unit_id"],
                    "task_id": summary["task_id"],
                    "initial_state_index": summary["initial_state_index"],
                    "prefix_chunks": summary["prefix_chunks"],
                    "pair_type": pair_type,
                    "direction": direction,
                    "future_noise_seed": noise_seed,
                    "contrast": f"{pair_type}_{modality}_donor_minus_recipient",
                    "action_donor_steering": donor["pair_donor_steering"]
                    - recipient["pair_donor_steering"],
                }
                if modality == "all":
                    donor_endpoint = endpoint_by_name[f"{prefix}donor"]
                    recipient_endpoint = endpoint_by_name[f"{prefix}recipient"]
                    endpoint_metrics = (
                        (
                            "goal_endpoint_donor_steering",
                            "goal_endpoint_projected_l2",
                        )
                        if pair_type == "object"
                        else (
                            "robot_endpoint_donor_steering",
                            "robot_endpoint_projected_l2",
                        )
                    )
                    for metric in endpoint_metrics:
                        row[metric] = _difference(
                            donor_endpoint, recipient_endpoint, metric
                        )
                contrast_rows.append(row)
            prefix = f"{context}_n{noise_seed}_all_"
            donor = action_by_name[f"{prefix}donor"]
            gaussian = action_by_name[f"{prefix}gaussian"]
            donor_endpoint = endpoint_by_name[f"{prefix}donor"]
            gaussian_endpoint = endpoint_by_name[f"{prefix}gaussian"]
            gaussian_row = {
                "unit_id": summary["unit_id"],
                "task_id": summary["task_id"],
                "initial_state_index": summary["initial_state_index"],
                "prefix_chunks": summary["prefix_chunks"],
                "pair_type": pair_type,
                "direction": direction,
                "future_noise_seed": noise_seed,
                "contrast": f"{pair_type}_all_donor_minus_gaussian",
                "action_donor_steering": donor["pair_donor_steering"]
                - gaussian["pair_donor_steering"],
            }
            endpoint_metrics = (
                ("goal_endpoint_donor_steering", "goal_endpoint_projected_l2")
                if pair_type == "object"
                else (
                    "robot_endpoint_donor_steering",
                    "robot_endpoint_projected_l2",
                )
            )
            for metric in endpoint_metrics:
                gaussian_row[metric] = _difference(
                    donor_endpoint, gaussian_endpoint, metric
                )
            contrast_rows.append(gaussian_row)
            natural_name = f"{context}_n{noise_seed}_all_natural_control"
            if natural_name in action_by_name:
                natural = action_by_name[natural_name]
                natural_endpoint = endpoint_by_name[natural_name]
                natural_row = {
                    "unit_id": summary["unit_id"],
                    "task_id": summary["task_id"],
                    "initial_state_index": summary["initial_state_index"],
                    "prefix_chunks": summary["prefix_chunks"],
                    "pair_type": pair_type,
                    "direction": direction,
                    "future_noise_seed": noise_seed,
                    "contrast": f"{pair_type}_all_donor_minus_natural_control",
                    "action_donor_steering": donor["pair_donor_steering"]
                    - natural["pair_donor_steering"],
                }
                for metric in endpoint_metrics:
                    natural_row[metric] = _difference(
                        donor_endpoint, natural_endpoint, metric
                    )
                contrast_rows.append(natural_row)

    selection = summary["selection"]
    branch = np.load(Path(summary["branch_run"]) / "branches.npz", allow_pickle=False)
    actions = np.load(run_dir / "actions.npz", allow_pickle=False)
    anchor = int(selection["recipient"])
    target_specs = {
        "object": (int(selection["object_donor"]), "object_forward", "donor"),
        "robot": (int(selection["robot_donor"]), "robot_forward", "donor"),
    }
    if selection["joint_donor"] is not None:
        target_specs["joint"] = (
            int(selection["joint_donor"]),
            "object_forward",
            "joint_donor",
        )
    if selection["natural_control"] is not None:
        target_specs["natural"] = (
            int(selection["natural_control"]),
            "object_forward",
            "natural_control",
        )
    recipient_action = branch["normalized_branch_actions"][anchor]
    multi_rows = []
    for noise_seed in summary["future_noise_seeds"]:
        matrix = {}
        for target_name, (_target_index, context, role) in target_specs.items():
            patched = actions[f"normalized_{context}_n{noise_seed}_all_{role}"]
            matrix[target_name] = {
                reference_name: _steering(
                    patched,
                    recipient_action,
                    branch["normalized_branch_actions"][reference_index],
                )
                for reference_name, (
                    reference_index,
                    _context,
                    _role,
                ) in target_specs.items()
            }
        for target_name, alignments in matrix.items():
            own = alignments[target_name]
            alternatives = [
                value for name, value in alignments.items() if name != target_name
            ]
            maximum = max(alignments.values())
            multi_rows.append(
                {
                    "unit_id": summary["unit_id"],
                    "task_id": summary["task_id"],
                    "initial_state_index": summary["initial_state_index"],
                    "prefix_chunks": summary["prefix_chunks"],
                    "future_noise_seed": noise_seed,
                    "target": target_name,
                    "own_alignment": own,
                    "mean_other_alignment": float(np.mean(alternatives)),
                    "diagonal_alignment_margin": own - float(np.mean(alternatives)),
                    "correct_donor_top1": float(np.isclose(own, maximum)),
                    "correct_donor_top1_minus_chance": float(
                        np.isclose(own, maximum)
                    )
                    - 1.0 / len(alignments),
                    "alignment_matrix_row": json.dumps(alignments, sort_keys=True),
                }
            )
    return contrast_rows, multi_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execution-analysis-name", default="execution_analysis.json"
    )
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite summary: {args.output_dir}")

    manifest_units = []
    for path in args.manifests:
        manifest_units.extend(json.loads(path.read_text())["units"])
    contrasts = []
    multi = []
    for run_dir in args.run_dirs:
        run_contrasts, run_multi = _run_rows(
            run_dir, args.execution_analysis_name
        )
        contrasts.extend(run_contrasts)
        multi.extend(run_multi)
    estimates = {}
    metrics = (
        "action_donor_steering",
        "goal_endpoint_donor_steering",
        "goal_endpoint_projected_l2",
        "robot_endpoint_donor_steering",
        "robot_endpoint_projected_l2",
    )
    for contrast in sorted({row["contrast"] for row in contrasts}):
        selected = [row for row in contrasts if row["contrast"] == contrast]
        estimates[contrast] = {
            metric: estimate
            for metric in metrics
            if (estimate := _estimate(selected, metric)) is not None
        }
    multi_estimates = {
        "diagonal_alignment_margin": _estimate(multi, "diagonal_alignment_margin"),
        "correct_donor_top1": _estimate(multi, "correct_donor_top1"),
        "correct_donor_top1_minus_chance": _estimate(
            multi, "correct_donor_top1_minus_chance"
        ),
    }
    result = {
        "scope": "held-out saved-state-clustered robot-versus-object estimates",
        "candidate_trajectories": len(manifest_units),
        "structurally_eligible_units": sum(
            unit["structurally_eligible"] for unit in manifest_units
        ),
        "analyzed_units": len({row["unit_id"] for row in contrasts}),
        "smallest_effect_of_interest": 0.10,
        "estimates": estimates,
        "multi_donor": multi_estimates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "factorization_repetitions.csv", contrasts)
    _write_csv(args.output_dir / "multi_donor_repetitions.csv", multi)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "eligible": result["structurally_eligible_units"],
                "analyzed": result["analyzed_units"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
