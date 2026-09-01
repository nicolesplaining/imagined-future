"""Aggregate the frozen RoboCasa replication at state and task levels."""

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
    "physical_endpoint_donor_steering",
    "proprio_endpoint_donor_steering",
    "primary_pixel_donor_preference",
    "decoded_primary_donor_preference",
)


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
        for direction in summary["directions"]:
            rows = {
                row["condition"]: row
                for row in summary["rows"]
                if row["direction"] == direction
            }
            endpoint_rows = {
                target: endpoint_by_name[
                    f"{direction}_n{summary['future_noise_seed']}_all_{target}"
                ]
                for target in ("recipient", "donor", "gaussian")
            }
            decoded = {
                target: rows[f"all_{target}"]["decoded_primary_l1_to_recipient"]
                - rows[f"all_{target}"]["decoded_primary_l1_to_donor"]
                for target in ("recipient", "donor", "gaussian")
            }
            for control in ("recipient", "gaussian"):
                donor_action = rows["all_donor"]
                control_action = rows[f"all_{control}"]
                donor_endpoint = endpoint_rows["donor"]
                control_endpoint = endpoint_rows[control]
                output.append(
                    {
                        "unit_id": summary["unit_id"],
                        "task_name": summary["task_name"],
                        "episode_index": summary["episode_index"],
                        "prefix_chunks": summary["prefix_chunks"],
                        "direction": direction,
                        "contrast": f"semantic_donor_minus_{control}",
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
                        "decoded_primary_donor_preference": decoded["donor"]
                        - decoded[control],
                    }
                )
    return output


def _attention_rows(run_dirs: list[Path]) -> list[dict]:
    output = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        endpoint = json.loads((run_dir / "execution_analysis.json").read_text())
        endpoint_by_name = {row["condition"]: row for row in endpoint["rows"]}
        unit_rows = []
        for row in summary["rows"]:
            direction = row["direction"]
            endpoint_row = endpoint_by_name[f"{direction}_{row['condition']}"]
            baseline = endpoint_by_name[f"{direction}_baseline"]
            unit_rows.append(
                {
                    "unit_id": summary["unit_id"],
                    "task_name": summary["task_name"],
                    "episode_index": summary["episode_index"],
                    "prefix_chunks": summary["prefix_chunks"],
                    "direction": direction,
                    "contrast": row["condition"],
                    "action_l2_from_baseline": row["action_l2_from_baseline"],
                    "physical_endpoint_donor_steering": abs(
                        endpoint_row["physical_endpoint_donor_steering"]
                        - baseline["physical_endpoint_donor_steering"]
                    ),
                    "proprio_endpoint_donor_steering": abs(
                        endpoint_row["proprio_endpoint_donor_steering"]
                        - baseline["proprio_endpoint_donor_steering"]
                    ),
                    "primary_pixel_donor_preference": abs(
                        endpoint_row["primary_pixel_donor_preference"]
                        - baseline["primary_pixel_donor_preference"]
                    ),
                }
            )
        output.extend(unit_rows)
        for direction in sorted({row["direction"] for row in unit_rows}):
            by_condition = {
                row["contrast"]: row
                for row in unit_rows
                if row["direction"] == direction
            }
            future = by_condition["block27_future_gate1"]
            for control in ("block27_current_gate1", "block27_all_key_control"):
                control_row = by_condition[control]
                output.append(
                    {
                        **{
                            key: future[key]
                            for key in (
                                "unit_id",
                                "task_name",
                                "episode_index",
                                "prefix_chunks",
                                "direction",
                            )
                        },
                        "contrast": f"block27_future_gate1_minus_{control}",
                        **{
                            metric: future[metric] - control_row[metric]
                            for metric in METRICS
                            if metric in future and metric in control_row
                        },
                    }
                )
    return output


def _cluster_means(rows: list[dict], cluster: str, metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            grouped[str(row[cluster])].append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _estimates(rows: list[dict]) -> dict[str, dict]:
    output = {}
    for contrast in sorted({row["contrast"] for row in rows}):
        selected = [row for row in rows if row["contrast"] == contrast]
        output[contrast] = {}
        for metric in METRICS:
            state_values = _cluster_means(selected, "unit_id", metric)
            if not state_values:
                continue
            task_values = _cluster_means(selected, "task_name", metric)
            state_array = list(state_values.values())
            task_array = list(task_values.values())
            output[contrast][metric] = {
                "state_cluster": {
                    **cluster_bootstrap_mean(state_values),
                    "median": float(np.median(state_array)),
                    "sign_test": exact_sign_test(state_array),
                },
                "task_cluster": {
                    **cluster_bootstrap_mean(task_values),
                    "median": float(np.median(task_array)),
                    "sign_test": exact_sign_test(task_array),
                },
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--attention-run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(
            f"refusing to overwrite completed summary: {args.output_dir}"
        )

    semantic = _semantic_rows(args.semantic_run_dirs)
    attention = _attention_rows(args.attention_run_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "semantic_repetitions.csv", semantic)
    _write_csv(args.output_dir / "attention_repetitions.csv", attention)
    result = {
        "scope": "exploratory RoboCasa state- and task-clustered replication estimates",
        "semantic_units": len({row["unit_id"] for row in semantic}),
        "attention_units": len({row["unit_id"] for row in attention}),
        "semantic": _estimates(semantic),
        "attention": _estimates(attention),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps({"output": str(args.output_dir), "units": result["semantic_units"]})
    )


if __name__ == "__main__":
    main()
