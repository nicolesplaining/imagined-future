#!/usr/bin/env python3
"""Summarize the frozen LingBot future-source intervention cohort.

All inferential summaries use saved simulator state as the independent unit.
Donor directions and diffusion paths are averaged within state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--admission", choices=("development", "evaluation"), required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50_000)
    parser.add_argument("--permutation-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=260904)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_actions(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value.reshape(value.shape[0], -1)


def bootstrap_ci(values: np.ndarray, repetitions: int, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [math.nan, math.nan]
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    estimates = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def permutation_pvalue(
    nearest: list[np.ndarray], observed: float, repetitions: int, rng: np.random.Generator
) -> float:
    """Within-state donor-label permutation for retrieval accuracy."""
    exceed = 0
    for _ in range(repetitions):
        state_scores = []
        for prediction in nearest:
            branch_count = prediction.shape[0]
            target = np.empty((branch_count, branch_count), dtype=np.int64)
            for recipient in range(branch_count):
                target[recipient] = rng.permutation(branch_count)
            state_scores.append(float(np.mean(prediction == target)))
        exceed += float(np.mean(state_scores)) >= observed - 1e-15
    return (exceed + 1.0) / (repetitions + 1.0)


def load_state(root: Path, record: dict) -> tuple[dict, dict[str, np.ndarray]]:
    state_root = root / record["state_id"]
    result_path = state_root / "result.json"
    arrays_path = state_root / "actions.npz"
    if not result_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(f"missing complete artifacts for {record['state_id']}")
    metadata = json.loads(result_path.read_text())
    if metadata.get("status") != "complete":
        raise RuntimeError(f"state is not complete: {record['state_id']}")
    if metadata.get("actions_sha256") != sha256_file(arrays_path):
        raise RuntimeError(f"actions hash mismatch: {record['state_id']}")
    arrays = dict(np.load(arrays_path, allow_pickle=False))
    return metadata, arrays


def state_metrics(record: dict, metadata: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, np.ndarray]:
    native_key = (
        "native_executed_actions"
        if "native_executed_actions" in arrays
        else "native_actions"
    )
    grid_key = (
        "latent_grid_executed_actions"
        if "latent_grid_executed_actions" in arrays
        else "latent_grid_actions"
    )
    native = flatten_actions(arrays[native_key])
    grid_raw = np.asarray(arrays[grid_key], dtype=np.float64)
    branch_count = native.shape[0]
    if branch_count != 4 or grid_raw.shape[:2] != (4, 4):
        raise ValueError(f"expected a 4x4 grid for {record['state_id']}, got {grid_raw.shape}")
    grid = grid_raw.reshape(branch_count, branch_count, -1)
    distance = np.linalg.norm(grid[:, :, None, :] - native[None, None, :, :], axis=-1)
    nearest = np.argmin(distance, axis=-1)
    targets = np.broadcast_to(np.arange(branch_count)[None, :], nearest.shape)
    diagonal = np.eye(branch_count, dtype=bool)

    projections = []
    distance_reductions = []
    cosines = []
    orthogonal_residuals = []
    normalized_final_distances = []
    for recipient in range(branch_count):
        baseline = native[recipient]
        for donor in range(branch_count):
            if donor == recipient:
                continue
            target = native[donor]
            patched = grid[recipient, donor]
            axis = target - baseline
            displacement = patched - baseline
            separation = float(np.linalg.norm(axis))
            if separation <= EPS:
                continue
            projection = float(np.dot(displacement, axis) / np.dot(axis, axis))
            residual = displacement - projection * axis
            disp_norm = float(np.linalg.norm(displacement))
            projections.append(projection)
            distance_reductions.append(1.0 - float(np.linalg.norm(patched - target)) / separation)
            normalized_final_distances.append(float(np.linalg.norm(patched - target)) / separation)
            orthogonal_residuals.append(float(np.linalg.norm(residual)) / separation)
            cosines.append(
                float(np.dot(displacement, axis) / (disp_norm * separation))
                if disp_norm > EPS
                else 0.0
            )

    row = {
        "state_id": record["state_id"],
        "task_id": int(record["task_id"]),
        "retrieval_accuracy_all": float(np.mean(nearest == targets)),
        "retrieval_accuracy_off_diagonal": float(np.mean((nearest == targets)[~diagonal])),
        "projection": float(np.mean(projections)),
        "distance_reduction": float(np.mean(distance_reductions)),
        "cosine_alignment": float(np.mean(cosines)),
        "orthogonal_residual": float(np.mean(orthogonal_residuals)),
        "normalized_final_distance": float(np.mean(normalized_final_distances)),
        "self_latent_max_abs_error": float(metadata["native_self_latent_max_abs_error"]),
        "self_cache_max_abs_error": float(metadata["native_self_cache_max_abs_error"]),
        "future_hashes_unique": int(metadata["native_future_hashes_unique"]),
        "cache_hashes_unique": int(metadata["native_cache_hashes_unique"]),
        "duration_seconds": float(metadata["duration_seconds"]),
    }

    if "gaussian_actions" in arrays:
        gaussian = flatten_actions(
            arrays.get("gaussian_executed_actions", arrays["gaussian_actions"])
        )
        native_scale = []
        for i in range(branch_count):
            other = [np.linalg.norm(native[i] - native[j]) for j in range(branch_count) if i != j]
            native_scale.append(float(np.mean(other)))
        row["gaussian_perturbation_normalized"] = float(
            np.mean(
                [
                    np.linalg.norm(gaussian[i] - native[i]) / max(native_scale[i], EPS)
                    for i in range(branch_count)
                ]
            )
        )

    if "donor_future_recipient_cache_actions" in arrays:
        dfrc = np.asarray(
            arrays.get(
                "donor_future_recipient_cache_executed_actions",
                arrays["donor_future_recipient_cache_actions"],
            ),
            dtype=np.float64,
        ).reshape(4, 4, -1)
        rfdb = np.asarray(
            arrays.get(
                "recipient_future_donor_cache_executed_actions",
                arrays["recipient_future_donor_cache_actions"],
            ),
            dtype=np.float64,
        ).reshape(4, 4, -1)
        dfrc_projection = []
        rfdb_projection = []
        for recipient in range(4):
            for donor in range(4):
                if donor == recipient:
                    continue
                axis = native[donor] - native[recipient]
                denom = float(np.dot(axis, axis))
                if denom <= EPS:
                    continue
                dfrc_projection.append(float(np.dot(dfrc[recipient, donor] - native[recipient], axis) / denom))
                rfdb_projection.append(float(np.dot(rfdb[recipient, donor] - native[recipient], axis) / denom))
        row["donor_future_recipient_cache_projection"] = float(np.mean(dfrc_projection))
        row["recipient_future_donor_cache_projection"] = float(np.mean(rfdb_projection))
    return row, nearest


def main() -> None:
    args = parse_args()
    manifest_path = args.result_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = [r for r in manifest["states"] if r["admission"] == args.admission]
    rows = []
    nearest = []
    failures = []
    for record in records:
        try:
            metadata, arrays = load_state(args.result_root, record)
            row, prediction = state_metrics(record, metadata, arrays)
            rows.append(row)
            nearest.append(prediction)
        except Exception as exc:
            failures.append({"state_id": record["state_id"], "error": repr(exc)})
    if not rows:
        raise RuntimeError(f"no complete states; failures={failures[:3]}")

    rng = np.random.default_rng(args.seed)
    metric_names = [
        "retrieval_accuracy_all",
        "retrieval_accuracy_off_diagonal",
        "projection",
        "distance_reduction",
        "cosine_alignment",
        "orthogonal_residual",
        "normalized_final_distance",
    ]
    optional = [
        "gaussian_perturbation_normalized",
        "donor_future_recipient_cache_projection",
        "recipient_future_donor_cache_projection",
    ]
    metric_names.extend(name for name in optional if all(name in row for row in rows))
    estimates = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        estimates[name] = {
            "mean": float(values.mean()),
            "state_bootstrap_95_ci": bootstrap_ci(values, args.bootstrap_repetitions, rng),
            "n_states": len(values),
        }

    observed = estimates["retrieval_accuracy_all"]["mean"]
    permutation_p = permutation_pvalue(
        nearest, observed, args.permutation_repetitions, rng
    )
    task_ids = sorted({row["task_id"] for row in rows})
    leave_one_task_out = {}
    for task_id in task_ids:
        kept = [row for row in rows if row["task_id"] != task_id]
        leave_one_task_out[str(task_id)] = {
            name: float(np.mean([row[name] for row in kept])) for name in metric_names
        }

    summary = {
        "admission": args.admission,
        "chance_rate": 0.25,
        "complete_state_count": len(rows),
        "expected_state_count": len(records),
        "failed_or_missing_states": failures,
        "state_is_independent_unit": True,
        "within_state_cells": 16,
        "estimates": estimates,
        "retrieval_within_state_label_permutation_p": permutation_p,
        "max_self_latent_abs_error": max(row["self_latent_max_abs_error"] for row in rows),
        "max_self_cache_abs_error": max(row["self_cache_max_abs_error"] for row in rows),
        "all_states_four_unique_futures": all(row["future_hashes_unique"] == 4 for row in rows),
        "all_states_four_unique_caches": all(row["cache_hashes_unique"] == 4 for row in rows),
        "leave_one_task_out": leave_one_task_out,
        "manifest_sha256": sha256_file(manifest_path),
        "result_root": str(args.result_root.resolve()),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "summary.json", summary)

    fieldnames = sorted({key for row in rows for key in row})
    csv_path = args.output_root / "state_metrics.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    lines = [
        f"# LingBot future transplant — {args.admission}",
        "",
        f"Complete states: {len(rows)}/{len(records)}",
        "",
        "| Estimand | Mean | State-bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for name in metric_names:
        item = estimates[name]
        lo, hi = item["state_bootstrap_95_ci"]
        lines.append(f"| {name.replace('_', ' ')} | {item['mean']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines.extend(
        [
            "",
            f"Within-state donor-label permutation p: {permutation_p:.6g}",
            f"Maximum native/self latent replay error: {summary['max_self_latent_abs_error']:.8g}",
            f"Maximum native/cache replay error: {summary['max_self_cache_abs_error']:.8g}",
            "",
            "The saved simulator state is the independent unit; all donor directions and diffusion paths are averaged within state.",
        ]
    )
    atomic_text(args.output_root / "results.md", "\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
