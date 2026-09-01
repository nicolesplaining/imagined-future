"""Aggregate held-out semantic and necessity effects at the saved-state level."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from imagined_future.statistics import cluster_bootstrap_mean, exact_sign_test

METRICS = (
    "action_donor_steering",
    "action_l2_from_baseline",
    "max_abs_from_baseline",
    "physical_endpoint_donor_steering",
    "proprio_endpoint_donor_steering",
    "primary_pixel_donor_preference",
)


def _mean_by_unit(rows: list[dict], value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is not None and np.isfinite(value):
            grouped[row["unit_id"]].append(float(value))
    return {unit: float(np.mean(values)) for unit, values in grouped.items()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _semantic_rows(run_dirs: list[Path]) -> list[dict]:
    output = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        endpoint = json.loads((run_dir / "execution_analysis.json").read_text())
        endpoint_by_name = {row["condition"]: row for row in endpoint["rows"]}
        action_by_name = {}
        for row in summary["rows"]:
            if row["condition"] == "latent_control_diagnostics":
                continue
            name = f'{row["direction"]}_n{row["future_noise_seed"]}_{row["condition"]}'
            action_by_name[name] = row
        for direction in summary["directions"]:
            for noise_seed in summary["future_noise_seeds"]:
                prefix = f"{direction}_n{noise_seed}_"
                for modality in ("all", "wrist", "primary", "proprio"):
                    recipient_name = f"{prefix}{modality}_recipient"
                    donor_name = f"{prefix}{modality}_donor"
                    recipient_action = action_by_name[recipient_name]
                    donor_action = action_by_name[donor_name]
                    recipient_endpoint = endpoint_by_name[recipient_name]
                    donor_endpoint = endpoint_by_name[donor_name]
                    output.append(
                        {
                            "unit_id": summary["unit_id"],
                            "task_id": summary["task_id"],
                            "initial_state_index": summary["initial_state_index"],
                            "prefix_chunks": summary["prefix_chunks"],
                            "direction": direction,
                            "future_noise_seed": noise_seed,
                            "contrast": f"semantic_{modality}_donor_minus_recipient",
                            "action_donor_steering": donor_action["donor_steering"]
                            - recipient_action["donor_steering"],
                            "physical_endpoint_donor_steering": donor_endpoint[
                                "physical_endpoint_donor_steering"
                            ]
                            - recipient_endpoint["physical_endpoint_donor_steering"],
                            "proprio_endpoint_donor_steering": donor_endpoint[
                                "proprio_endpoint_donor_steering"
                            ]
                            - recipient_endpoint["proprio_endpoint_donor_steering"],
                            "primary_pixel_donor_preference": donor_endpoint[
                                "primary_pixel_donor_preference"
                            ]
                            - recipient_endpoint["primary_pixel_donor_preference"],
                        }
                    )
                donor_name = f"{prefix}all_donor"
                for control in ("gaussian", "natural_control", "shuffled"):
                    control_name = f"{prefix}all_{control}"
                    donor_action = action_by_name[donor_name]
                    control_action = action_by_name[control_name]
                    donor_endpoint = endpoint_by_name[donor_name]
                    control_endpoint = endpoint_by_name[control_name]
                    output.append(
                        {
                            "unit_id": summary["unit_id"],
                            "task_id": summary["task_id"],
                            "initial_state_index": summary["initial_state_index"],
                            "prefix_chunks": summary["prefix_chunks"],
                            "direction": direction,
                            "future_noise_seed": noise_seed,
                            "contrast": f"semantic_all_donor_minus_{control}",
                            "action_donor_steering": donor_action["donor_steering"]
                            - control_action["donor_steering"],
                            "physical_endpoint_donor_steering": donor_endpoint[
                                "physical_endpoint_donor_steering"
                            ]
                            - control_endpoint["physical_endpoint_donor_steering"],
                            "proprio_endpoint_donor_steering": donor_endpoint[
                                "proprio_endpoint_donor_steering"
                            ]
                            - control_endpoint["proprio_endpoint_donor_steering"],
                            "primary_pixel_donor_preference": donor_endpoint[
                                "primary_pixel_donor_preference"
                            ]
                            - control_endpoint["primary_pixel_donor_preference"],
                        }
                    )
    return output


def _attention_rows(run_dirs: list[Path]) -> list[dict]:
    output = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        endpoint = json.loads((run_dir / "execution_analysis.json").read_text())
        endpoint_by_name = {row["condition"]: row for row in endpoint["rows"]}
        run_rows = []
        for row in summary["rows"]:
            name = f'{row["direction"]}_{row["condition"]}'
            endpoint_row = endpoint_by_name[name]
            baseline_endpoint = endpoint_by_name[f'{row["direction"]}_baseline']
            run_rows.append(
                {
                    "unit_id": summary["unit_id"],
                    "task_id": summary["task_id"],
                    "initial_state_index": summary["initial_state_index"],
                    "prefix_chunks": summary["prefix_chunks"],
                    "direction": row["direction"],
                    "contrast": row["condition"],
                    "gate": row["gate"],
                    "action_donor_steering": abs(row["donor_steering"]),
                    "physical_endpoint_donor_steering": abs(
                        baseline_endpoint["physical_endpoint_donor_steering"]
                        - endpoint_row["physical_endpoint_donor_steering"]
                    ),
                    "proprio_endpoint_donor_steering": abs(
                        baseline_endpoint["proprio_endpoint_donor_steering"]
                        - endpoint_row["proprio_endpoint_donor_steering"]
                    ),
                    "primary_pixel_donor_preference": abs(
                        baseline_endpoint["primary_pixel_donor_preference"]
                        - endpoint_row["primary_pixel_donor_preference"]
                    ),
                    "action_l2_from_baseline": row["action_l2_from_baseline"],
                    "max_abs_from_baseline": row["max_abs_from_baseline"],
                }
            )
        output.extend(run_rows)
        for direction in sorted({row["direction"] for row in run_rows}):
            by_condition = {
                row["contrast"]: row for row in run_rows if row["direction"] == direction
            }
            future = by_condition.get("block27_future_gate1")
            if future is None:
                continue
            for control in (
                "block27_current_gate1",
                "block27_random_gate1",
                "block27_all_key_control",
                "block0_future_gate1",
            ):
                control_row = by_condition.get(control)
                if control_row is None:
                    continue
                difference = {
                    key: future[key] - control_row[key]
                    for key in METRICS
                    if key in future and key in control_row
                }
                output.append(
                    {
                        **{
                            key: future[key]
                            for key in (
                                "unit_id",
                                "task_id",
                                "initial_state_index",
                                "prefix_chunks",
                                "direction",
                            )
                        },
                        "contrast": f"block27_future_gate1_minus_{control}",
                        "gate": 1.0,
                        **difference,
                    }
                )
    return output


def _estimates(rows: list[dict]) -> dict[str, dict[str, dict]]:
    estimates = {}
    contrasts = sorted({row["contrast"] for row in rows})
    for contrast in contrasts:
        selected = [row for row in rows if row["contrast"] == contrast]
        estimates[contrast] = {}
        for metric in METRICS:
            unit_values = _mean_by_unit(selected, metric)
            if unit_values:
                values = list(unit_values.values())
                estimates[contrast][metric] = {
                    **cluster_bootstrap_mean(unit_values),
                    "median": float(np.median(values)),
                    "sign_test": exact_sign_test(values),
                }
        task_estimates = {}
        for task_id in sorted({int(row["task_id"]) for row in selected}):
            task_rows = [row for row in selected if int(row["task_id"]) == task_id]
            values = _mean_by_unit(task_rows, "physical_endpoint_donor_steering")
            if values:
                task_estimates[str(task_id)] = {
                    "units": len(values),
                    "mean": float(np.mean(list(values.values()))),
                }
        estimates[contrast]["task_stratified_physical"] = task_estimates
    return estimates


def _prefix_estimates(rows: list[dict]) -> dict[str, dict[str, dict[str, dict]]]:
    return {
        str(prefix): _estimates([row for row in rows if int(row["prefix_chunks"]) == prefix])
        for prefix in sorted({int(row["prefix_chunks"]) for row in rows})
    }


def _paired_prefix_differences(rows: list[dict], contrast: str) -> dict[str, dict[str, dict]]:
    selected = [row for row in rows if row["contrast"] == contrast]
    task_prefix: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in selected:
        task_prefix[(int(row["task_id"]), int(row["prefix_chunks"]))].append(row)
    output = {}
    for later_prefix in sorted({prefix for _task, prefix in task_prefix if prefix != 0}):
        output[str(later_prefix)] = {}
        tasks = sorted(
            task
            for task, prefix in task_prefix
            if prefix == 0 and (task, later_prefix) in task_prefix
        )
        for metric in METRICS:
            differences = {}
            for task in tasks:
                early_values = [float(row[metric]) for row in task_prefix[(task, 0)] if row.get(metric) is not None]
                later_values = [
                    float(row[metric])
                    for row in task_prefix[(task, later_prefix)]
                    if row.get(metric) is not None
                ]
                if early_values and later_values:
                    differences[str(task)] = float(np.mean(early_values) - np.mean(later_values))
            if differences:
                values = list(differences.values())
                output[str(later_prefix)][metric] = {
                    **cluster_bootstrap_mean(differences),
                    "median": float(np.median(values)),
                    "sign_test": exact_sign_test(values),
                }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--attention-run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed summary: {args.output_dir}")

    semantic = _semantic_rows(args.semantic_run_dirs)
    attention = _attention_rows(args.attention_run_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "semantic_state_repetitions.csv", semantic)
    _write_csv(args.output_dir / "attention_state_repetitions.csv", attention)
    result = {
        "scope": "saved-state clustered confirmatory estimates",
        "semantic_units": len({row["unit_id"] for row in semantic}),
        "attention_units": len({row["unit_id"] for row in attention}),
        "smallest_effect_of_interest": 0.10,
        "semantic": _estimates(semantic),
        "semantic_by_prefix_chunks": _prefix_estimates(semantic),
        "semantic_early_minus_later": _paired_prefix_differences(
            semantic, "semantic_all_donor_minus_recipient"
        ),
        "attention": _estimates(attention),
        "attention_by_prefix_chunks": _prefix_estimates(attention),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir), "semantic_units": result["semantic_units"], "attention_units": result["attention_units"]}, indent=2))


if __name__ == "__main__":
    main()
