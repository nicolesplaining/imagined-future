#!/usr/bin/env python3
"""Analyze the complete frozen FastWAM future-latent x cache factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from imagined_future.fastwam_analysis import (
    DIRECTIONAL_METRICS,
    audit_fastwam_outputs,
    atomic_write_csv,
    atomic_write_text,
    hierarchical_bootstrap,
)
from imagined_future.fastwam_cache_factorial import (
    CACHE_FACTORIAL_CONDITIONS,
    expand_cache_factorial_runs,
    load_cache_factorial_manifest,
    states_from_factorial_manifest,
    validate_factorial_parent,
)
from imagined_future.fastwam_optional_idm import (
    FastWAMCondition,
    action_metrics,
    atomic_write_json,
    expand_run_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial-manifest", type=Path, required=True)
    parser.add_argument("--factorial-output-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-output-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--required-state-count", type=int, default=120)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_903)
    parser.add_argument("--exact-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def manifest_root(root: Path, manifest_id: str) -> Path:
    root = root.resolve()
    return root if root.name == manifest_id else root / manifest_id


def audit_outputs(manifest: dict[str, Any], output_root: Path, required: int):
    root = manifest_root(output_root, manifest["manifest_id"])
    rows: list[dict[str, Any]] = []
    for state in states_from_factorial_manifest(manifest):
        expected = {
            run.run_id: run
            for run in expand_cache_factorial_runs(manifest["manifest_id"], state)
        }
        run_dir = root / state.state_id / "runs"
        json_ids = {path.stem for path in run_dir.glob("run-*.json")}
        npz_ids = {path.stem for path in run_dir.glob("run-*.npz")}
        valid: set[str] = set()
        malformed: list[str] = []
        for run_id in sorted(set(expected) & json_ids & npz_ids):
            try:
                record = json.loads(
                    (run_dir / f"{run_id}.json").read_text(encoding="utf-8")
                )
                okay = (
                    record.get("status") == "complete"
                    and record.get("manifest_id") == manifest["manifest_id"]
                    and record.get("base_manifest_id") == manifest["base_manifest_id"]
                    and record.get("run") == expected[run_id].to_dict()
                    and record.get("array_file") == f"{run_id}.npz"
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                okay = False
            if okay:
                valid.add(run_id)
            else:
                malformed.append(run_id)
        summary_path = root / state.state_id / "summary.json"
        summary_valid = False
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary_valid = (
                    summary.get("status") == "complete"
                    and summary.get("manifest_id") == manifest["manifest_id"]
                    and summary.get("base_manifest_id") == manifest["base_manifest_id"]
                    and summary.get("state_id") == state.state_id
                    and summary.get("run_count") == len(expected)
                    and set(summary.get("completed_conditions", []))
                    == set(CACHE_FACTORIAL_CONDITIONS)
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                summary_valid = False
        missing_json = sorted(set(expected) - json_ids)
        missing_npz = sorted(set(expected) - npz_ids)
        unexpected_json = sorted(json_ids - set(expected))
        unexpected_npz = sorted(npz_ids - set(expected))
        rows.append(
            {
                "state_id": state.state_id,
                "expected_runs": len(expected),
                "valid_runs": len(valid),
                "missing_json": missing_json,
                "missing_npz": missing_npz,
                "malformed": malformed,
                "unexpected_json": unexpected_json,
                "unexpected_npz": unexpected_npz,
                "summary_valid": summary_valid,
                "state_complete": (
                    len(valid) == len(expected)
                    and summary_valid
                    and not malformed
                    and not unexpected_json
                    and not unexpected_npz
                ),
            }
        )
    complete = sum(bool(row["state_complete"]) for row in rows)
    expected_state_ids = {state.state_id for state in states_from_factorial_manifest(manifest)}
    actual_state_ids = (
        {
            path.name
            for path in root.iterdir()
            if path.is_dir() and path.name != "analysis"
        }
        if root.exists()
        else set()
    )
    unexpected_state_ids = sorted(actual_state_ids - expected_state_ids)
    audit = {
        "manifest_id": manifest["manifest_id"],
        "required_state_count": required,
        "manifest_state_count": len(rows),
        "complete_state_count": complete,
        "expected_run_count": sum(int(row["expected_runs"]) for row in rows),
        "valid_run_count": sum(int(row["valid_runs"]) for row in rows),
        "unexpected_state_ids": unexpected_state_ids,
        "all_outputs_complete": (
            len(rows) == required
            and complete == required
            and not unexpected_state_ids
        ),
    }
    return root, audit, rows


def finite_mean(values) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def load_native_actions(base: dict[str, Any], base_root: Path, state) -> dict[str, np.ndarray]:
    native = {}
    for spec in expand_run_specs(base["manifest_id"], state):
        if spec.condition != FastWAMCondition.NATIVE.value:
            continue
        with np.load(
            base_root / state.state_id / "runs" / f"{spec.run_id}.npz",
            allow_pickle=False,
        ) as arrays:
            native[spec.recipient_id] = np.asarray(arrays["action_model"], dtype=np.float64)
    return native


def aggregate_hierarchical(rows, metric: str, samples: int, seed: int, key: str):
    return hierarchical_bootstrap(
        rows, value_key=metric, samples=samples, seed=seed, key=key
    )


def main() -> None:
    args = parse_args()
    factorial = load_cache_factorial_manifest(args.factorial_manifest)
    base = validate_factorial_parent(factorial, args.base_manifest)
    summary_dir = args.summary_dir.resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    root, audit, missingness = audit_outputs(
        factorial, args.factorial_output_root, args.required_state_count
    )
    base_audit, _ = audit_fastwam_outputs(
        base,
        args.base_output_root,
        required_state_count=len(base["states"]),
    )
    atomic_write_json(summary_dir / "fastwam_cache_factorial_completeness.json", audit)
    missing_fields = (
        "state_id",
        "expected_runs",
        "valid_runs",
        "missing_json",
        "missing_npz",
        "malformed",
        "unexpected_json",
        "unexpected_npz",
        "summary_valid",
        "state_complete",
    )
    atomic_write_csv(
        summary_dir / "fastwam_cache_factorial_missingness.csv",
        missingness,
        missing_fields,
    )
    if not audit["all_outputs_complete"] or not base_audit["all_frozen_outputs_complete"]:
        atomic_write_json(
            summary_dir / "fastwam_cache_factorial_results.json",
            {
                "status": "incomplete",
                "audit": audit,
                "base_audit": base_audit,
                "evidence_gate": {
                    "passed": False,
                    "decision": "unavailable_incomplete_matrix",
                },
            },
        )
        raise SystemExit(2)

    base_root = manifest_root(args.base_output_root, base["manifest_id"])
    run_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    controls_by_state: dict[str, Any] = {}
    for state in states_from_factorial_manifest(factorial):
        native = load_native_actions(base, base_root, state)
        run_dir = root / state.state_id / "runs"
        current_state_runs: list[dict[str, Any]] = []
        for spec in expand_cache_factorial_runs(factorial["manifest_id"], state):
            record = json.loads(
                (run_dir / f"{spec.run_id}.json").read_text(encoding="utf-8")
            )
            with np.load(run_dir / f"{spec.run_id}.npz", allow_pickle=False) as arrays:
                action = np.asarray(arrays["action_model"], dtype=np.float64)
            finite = bool(np.isfinite(action).all())
            metrics = (
                action_metrics(
                    action,
                    native[spec.recipient_id],
                    native[spec.donor_id],
                    native,
                    donor_id=spec.donor_id,
                )
                if finite and all(np.isfinite(value).all() for value in native.values())
                else {}
            )
            row = {
                "state_id": state.state_id,
                "suite": state.suite,
                "task_id": state.task_id,
                "initial_state_index": state.initial_state_index,
                **spec.to_dict(),
                "array_finite": finite,
                "base_reference_max_abs_error": record.get(
                    "base_reference_max_abs_error"
                ),
                "recipient_cache_future_swap_max_abs_error": record.get(
                    "same_recipient_cache_future_swap_max_abs_error"
                ),
                "donor_cache_future_swap_max_abs_error": record.get(
                    "same_donor_cache_future_swap_max_abs_error"
                ),
                **{
                    key: metrics.get(key)
                    for key in (
                        "axis_degenerate",
                        "correct_donor_retrieval",
                        "nearest_branch_id",
                        "donor_distance_reduction",
                        "donor_projection",
                        "cosine_alignment",
                        "orthogonal_residual",
                        "orthogonal_residual_ratio",
                        "distance_to_recipient",
                        "distance_to_donor",
                        "native_recipient_to_donor_distance",
                    )
                },
            }
            run_rows.append(row)
            current_state_runs.append(row)

        for condition in CACHE_FACTORIAL_CONDITIONS:
            rows = [row for row in current_state_runs if row["condition"] == condition]
            valid_axes = [row for row in rows if row["axis_degenerate"] is False]
            state_rows.append(
                {
                    "state_id": state.state_id,
                    "suite": state.suite,
                    "task_id": state.task_id,
                    "initial_state_index": state.initial_state_index,
                    "condition": condition,
                    "n_rows": len(rows),
                    "valid_axes": len(valid_axes),
                    "degenerate_axes": sum(
                        row["axis_degenerate"] is True for row in rows
                    ),
                    "invalid_rows": sum(row["axis_degenerate"] is None for row in rows),
                    "correct_donor_retrieval_rate": finite_mean(
                        [
                            float(row["correct_donor_retrieval"])
                            for row in rows
                            if row["correct_donor_retrieval"] is not None
                        ]
                    ),
                    **{
                        metric: finite_mean(
                            [
                                row.get(metric)
                                for row in (
                                    rows if metric == "distance_to_donor" else valid_axes
                                )
                            ]
                        )
                        for metric in DIRECTIONAL_METRICS
                        if metric != "correct_donor_retrieval_rate"
                    },
                }
            )
        summary = json.loads(
            (root / state.state_id / "summary.json").read_text(encoding="utf-8")
        )
        controls_by_state[state.state_id] = {
            "native_action_regeneration_error": summary[
                "native_action_regeneration_max_abs_error"
            ],
            "stored_native_latent_regeneration_error": summary[
                "stored_native_latent_regeneration_max_abs_error"
            ],
            "base_reference_errors": summary["base_reference_max_abs_error"],
            "recipient_cache_future_swap_error": summary[
                "recipient_cache_future_swap_global_max_abs_error"
            ],
            "donor_cache_future_swap_error": summary[
                "donor_cache_future_swap_global_max_abs_error"
            ],
        }

    aggregate: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for condition in CACHE_FACTORIAL_CONDITIONS:
        rows = [row for row in state_rows if row["condition"] == condition]
        aggregate[condition] = {
            "n_states": len(rows),
            "n_rows": sum(int(row["n_rows"]) for row in rows),
            "valid_axes": sum(int(row["valid_axes"]) for row in rows),
            "degenerate_axes": sum(int(row["degenerate_axes"]) for row in rows),
            "invalid_rows": sum(int(row["invalid_rows"]) for row in rows),
        }
        for metric in DIRECTIONAL_METRICS:
            stats = aggregate_hierarchical(
                rows,
                metric,
                args.bootstrap_samples,
                args.bootstrap_seed,
                f"factorial:{condition}:{metric}",
            )
            aggregate[condition][metric] = stats
            aggregate_rows.append(
                {"kind": "condition", "label": condition, "metric": metric, **stats}
            )

    index = {
        (row["state_id"], row["condition"]): row for row in state_rows
    }
    templates = [
        row
        for row in state_rows
        if row["condition"] == "future_recipient_cache_recipient"
    ]
    contrasts: dict[str, Any] = {
        "cache_main_effect": {},
        "future_main_effect": {},
        "future_by_cache_interaction": {},
    }
    contrast_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rr = "future_recipient_cache_recipient"
    dr = "future_donor_cache_recipient"
    rd = "future_recipient_cache_donor"
    dd = "future_donor_cache_donor"
    for metric in DIRECTIONAL_METRICS:
        for template in templates:
            cells = {
                condition: index[(template["state_id"], condition)].get(metric)
                for condition in CACHE_FACTORIAL_CONDITIONS
            }
            values = list(cells.values())
            derived = {
                "cache_main_effect": (
                    None
                    if any(value is None for value in values)
                    else 0.5 * (cells[rd] + cells[dd] - cells[rr] - cells[dr])
                ),
                "future_main_effect": (
                    None
                    if any(value is None for value in values)
                    else 0.5 * (cells[dr] + cells[dd] - cells[rr] - cells[rd])
                ),
                "future_by_cache_interaction": (
                    None
                    if any(value is None for value in values)
                    else (cells[dd] - cells[rd]) - (cells[dr] - cells[rr])
                ),
            }
            for label, value in derived.items():
                contrast_rows.setdefault((label, metric), []).append(
                    {
                        "state_id": template["state_id"],
                        "suite": template["suite"],
                        "task_id": template["task_id"],
                        "value": value,
                    }
                )
        for label in contrasts:
            rows = contrast_rows[(label, metric)]
            stats = aggregate_hierarchical(
                rows,
                "value",
                args.bootstrap_samples,
                args.bootstrap_seed,
                f"factorial:{label}:{metric}",
            )
            contrasts[label][metric] = stats
            aggregate_rows.append(
                {"kind": "contrast", "label": label, "metric": metric, **stats}
            )

    def global_max(path):
        values = [path(value) for value in controls_by_state.values()]
        return max(float(value) for value in values)

    controls = {
        "native_action_regeneration_global_max_abs_error": global_max(
            lambda value: value["native_action_regeneration_error"]
        ),
        "stored_native_latent_regeneration_global_max_abs_error": global_max(
            lambda value: value["stored_native_latent_regeneration_error"]
        ),
        "recipient_cache_future_swap_global_max_abs_error": global_max(
            lambda value: value["recipient_cache_future_swap_error"]
        ),
        "donor_cache_future_swap_global_max_abs_error": global_max(
            lambda value: value["donor_cache_future_swap_error"]
        ),
        "base_reference_global_max_abs_error": {
            condition: global_max(
                lambda value, condition=condition: value["base_reference_errors"][
                    condition
                ]
            )
            for condition in (
                "future_recipient_cache_recipient",
                "future_recipient_cache_donor",
                "future_donor_cache_donor",
            )
        },
    }
    all_exact = [
        controls["native_action_regeneration_global_max_abs_error"],
        controls["stored_native_latent_regeneration_global_max_abs_error"],
        controls["recipient_cache_future_swap_global_max_abs_error"],
        controls["donor_cache_future_swap_global_max_abs_error"],
        *controls["base_reference_global_max_abs_error"].values(),
    ]
    cache_retrieval_lower = contrasts["cache_main_effect"][
        "correct_donor_retrieval_rate"
    ]["ci95_low"]
    cache_distance_lower = contrasts["cache_main_effect"][
        "donor_distance_reduction"
    ]["ci95_low"]
    criteria = {
        "all_120_states_and_5760_arms_complete": audit["all_outputs_complete"],
        "parent_120_states_and_8640_arms_complete": base_audit[
            "all_frozen_outputs_complete"
        ],
        "all_actions_finite": all(row["array_finite"] for row in run_rows),
        "exact_replay_and_future_swap_controls": max(all_exact) <= args.exact_tolerance,
        "cache_main_effect_retrieval_lower_above_zero": (
            cache_retrieval_lower is not None and cache_retrieval_lower > 0
        ),
        "cache_main_effect_distance_reduction_lower_above_zero": (
            cache_distance_lower is not None and cache_distance_lower > 0
        ),
    }
    gate = {
        "passed": all(criteria.values()),
        "decision": (
            "cache_dominance_criteria_met"
            if all(criteria.values())
            else "cache_dominance_criteria_not_met"
        ),
        "criteria": criteria,
        "exact_tolerance": args.exact_tolerance,
    }
    report = {
        "status": "complete",
        "manifest_id": factorial["manifest_id"],
        "base_manifest_id": factorial["base_manifest_id"],
        "audit": audit,
        "base_audit": base_audit,
        "analysis": {
            "independent_unit": "state",
            "bootstrap": "suite-task-state hierarchical",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "controls_by_state": controls_by_state,
        "controls": controls,
        "conditions": aggregate,
        "factorial_contrasts": contrasts,
        "evidence_gate": gate,
    }
    atomic_write_json(summary_dir / "fastwam_cache_factorial_results.json", report)
    run_fields = tuple(run_rows[0].keys())
    state_fields = tuple(state_rows[0].keys())
    aggregate_fields = (
        "kind",
        "label",
        "metric",
        "n_suites",
        "n_tasks",
        "n_states_total",
        "n_states_valid",
        "n_states_missing",
        "mean",
        "ci95_low",
        "ci95_high",
    )
    atomic_write_csv(
        summary_dir / "fastwam_cache_factorial_run_metrics.csv", run_rows, run_fields
    )
    atomic_write_csv(
        summary_dir / "fastwam_cache_factorial_state_metrics.csv",
        state_rows,
        state_fields,
    )
    atomic_write_csv(
        summary_dir / "fastwam_cache_factorial_aggregate_metrics.csv",
        aggregate_rows,
        aggregate_fields,
    )
    latex = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Future / cache & Retrieval & Distance reduction & Projection & Orth. ratio \\",
        r"\midrule",
    ]
    for condition in CACHE_FACTORIAL_CONDITIONS:
        result = aggregate[condition]
        values = []
        for metric in (
            "correct_donor_retrieval_rate",
            "donor_distance_reduction",
            "donor_projection",
            "orthogonal_residual_ratio",
        ):
            stats = result[metric]
            values.append(
                "--"
                if stats["mean"] is None
                else f"{stats['mean']:.3f} [{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}]"
            )
        latex.append(condition.replace("_", r"\_") + " & " + " & ".join(values) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", ""])
    atomic_write_text(
        summary_dir / "fastwam_cache_factorial_results.tex", "\n".join(latex)
    )
    print(f"status=complete decision={gate['decision']} summary_dir={summary_dir}")
    raise SystemExit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
