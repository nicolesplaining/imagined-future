#!/usr/bin/env python3
"""Deterministically summarize the preregistered LingBot latent-dose cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROTOCOL_SHA256 = "8b6b4103b5c172f28c896b9834fda114aa52684c53f8c570c78c346fda9d3eba"
CLARIFICATION_SHA256 = "2f3ca2211b66100c6d99d44e879b632bee1434da1f8d371fb2ed27f981ee7f8e"
CORE_RUNNER_SHA256 = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
DOSE_RUNNER_SHA256 = "2d8b419be882eb979ed58091f7d0b0cd4322f2503aac9e4a854c558834f21b2e"
ORACLE_SCRIPT_SHA256 = "893c2d9152575b583e1db0d8fafab79727c5038613099aa983fae1ad74f96afc"
PARITY_GATE_SHA256 = "6437f774088084b67e6aea001376304dfe01ee622358358dd40642d49d0a67d5"
UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"
EXPECTED_ALPHAS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--core-result-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--clarification", type=Path, required=True)
    parser.add_argument("--oracle-receipt", type=Path, required=True)
    parser.add_argument("--dose-runner", type=Path, required=True)
    parser.add_argument("--checkpoint-content-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50_000)
    parser.add_argument("--permutation-repetitions", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=260906)
    parser.add_argument("--permutation-seed", type=int, default=260907)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{time.time_ns()}{path.suffix}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{time.time_ns()}{path.suffix}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{time.time_ns()}{path.suffix}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def atomic_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{time.time_ns()}{path.suffix}")
    figure.savefig(
        temporary,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "summarize_lingbot_future_dose.py"},
    )
    plt.close(figure)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def state_order(record: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(record["task_id"]),
        int(record["initial_state_index"]),
        str(record["state_id"]),
    )


def slope(values: np.ndarray) -> float:
    x = EXPECTED_ALPHAS[1:-1]
    centered = x - x.mean()
    return float(np.dot(centered, values) / np.dot(centered, centered))


def load_and_validate(
    *,
    result_root: Path,
    core_root: Path,
    protocol: dict[str, Any],
    oracle_sha256: str,
    checkpoint_content_sha256: str,
    checkpoint_aggregate_sha256: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    manifest_path = result_root / "manifest.json"
    manifest = load_json(manifest_path)
    if sha256_file(manifest_path) != protocol["source"]["manifest_sha256"]:
        raise RuntimeError("dose manifest differs from preregistration")
    records = sorted(
        [item for item in manifest["states"] if item["admission"] == "evaluation"],
        key=state_order,
    )
    if [item["state_id"] for item in records] != protocol["cohort"]["state_ids"]:
        raise RuntimeError("dose cohort/order differs from preregistration")
    if len(records) != 30:
        raise RuntimeError(f"expected 30 states, found {len(records)}")
    observed_result_ids = sorted(
        path.parent.name for path in result_root.glob("*/result.json")
    )
    if observed_result_ids != sorted(protocol["cohort"]["state_ids"]):
        raise RuntimeError("dose output result set differs from frozen cohort")
    task_counts = Counter(int(record["task_id"]) for record in records)
    if task_counts != Counter({task_id: 3 for task_id in range(10)}):
        raise RuntimeError(f"task coverage differs: {task_counts}")

    rows: list[dict[str, Any]] = []
    projections: list[np.ndarray] = []
    dose_runner_hashes: set[str] = set()
    for record in records:
        state_id = str(record["state_id"])
        state_root = result_root / state_id
        result_path = state_root / "result.json"
        arrays_path = state_root / "actions.npz"
        if not result_path.is_file() or not arrays_path.is_file():
            raise RuntimeError(f"missing dose result: {state_id}")
        metadata = load_json(result_path)
        expected = {
            "status": "complete",
            "state_id": state_id,
            "admission": "evaluation",
            "core_result_root": str(core_root),
            "dose_result_root": str(result_root),
            "core_state_root": str((core_root / state_id).resolve()),
            "dose_state_root": str(state_root.resolve()),
            "result_path": str(result_path.resolve()),
            "actions_path": str(arrays_path.resolve()),
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": protocol["source"]["manifest_sha256"],
            "core_runner_sha256": CORE_RUNNER_SHA256,
            "dose_runner_sha256": DOSE_RUNNER_SHA256,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_content_manifest_sha256": checkpoint_content_sha256,
            "checkpoint_aggregate_sha256": checkpoint_aggregate_sha256,
            "oracle_receipt_sha256": oracle_sha256,
            "oracle_future_bitwise_equal": True,
            "oracle_action_bitwise_equal": True,
            "actions_sha256": sha256_file(arrays_path),
            "action_coordinate_intervention": "none",
            "interior_model_calls": 3,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"dose metadata mismatch {state_id}: {mismatches}")
        dose_runner_hashes.add(str(metadata["dose_runner_sha256"]))
        expected_semantics = {
            "ordered_pair": {"recipient": "b0", "donor": "b1"},
            "action_noise_source": "b0",
            "alphas": EXPECTED_ALPHAS.tolist(),
            "endpoint_source": [
                "core latent_grid_actions[0,0]",
                "core latent_grid_actions[0,1]",
            ],
            "interpolation_variable": "final normalized denoised future-video latent",
            "interpolation_dtype": "torch.bfloat16",
            "interpolation_compute_dtype": "torch.float32 before cast to source dtype",
            "present_frame_overwritten_from_frozen_input": True,
            "cache_installation": "official t=0 future forward recomputed at every interior alpha",
        }
        semantic_mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected_semantics.items()
            if metadata.get(key) != value
        }
        if semantic_mismatches:
            raise RuntimeError(
                f"dose intervention semantics mismatch {state_id}: {semantic_mismatches}"
            )
        cache_hashes = metadata.get("interior_cache_sha256", [])
        if (
            len(cache_hashes) != 5
            or cache_hashes[0] is not None
            or cache_hashes[4] is not None
            or any(not isinstance(cache_hashes[index], str) for index in (1, 2, 3))
            or len(set(cache_hashes[1:4])) != 3
        ):
            raise RuntimeError(f"interior cache receipt missing: {state_id}")

        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "alphas",
                "actions",
                "executed_actions",
                "endpoint_reused",
            }:
                raise RuntimeError(f"unexpected dose array keys: {state_id}")
            if (
                archive["alphas"].dtype != np.float64
                or archive["actions"].dtype != np.float32
                or archive["executed_actions"].dtype != np.float32
                or archive["endpoint_reused"].dtype != np.bool_
            ):
                raise RuntimeError(f"dose array dtype changed: {state_id}")
            alphas = np.asarray(archive["alphas"], dtype=np.float64)
            actions = np.asarray(archive["actions"], dtype=np.float32)
            executed_raw = np.asarray(archive["executed_actions"], dtype=np.float32)
            endpoint_reused = np.asarray(archive["endpoint_reused"], dtype=bool)
        executed = executed_raw.astype(np.float64)
        if not np.array_equal(alphas, EXPECTED_ALPHAS):
            raise RuntimeError(f"alpha schedule changed: {state_id}")
        if not np.array_equal(endpoint_reused, [True, False, False, False, True]):
            raise RuntimeError(f"endpoint reuse receipt changed: {state_id}")
        if actions.shape != (5, 7, 4, 4) or executed.shape != (5, 7, 3, 4):
            raise RuntimeError(f"dose array shape changed: {state_id}")
        if (
            not np.isfinite(actions).all()
            or not np.isfinite(executed).all()
            or not np.array_equal(executed_raw, actions[..., 1:, :])
        ):
            raise RuntimeError(f"non-finite dose action: {state_id}")

        core_path = core_root / state_id / "actions.npz"
        core_result_path = core_root / state_id / "result.json"
        core_metadata = load_json(core_result_path)
        expected_core = {
            "status": "complete",
            "state_id": state_id,
            "admission": "evaluation",
            "prompt": record["prompt"],
            "input_sha256": record["input_sha256"],
            "runner_sha256": CORE_RUNNER_SHA256,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "actions_sha256": sha256_file(core_path),
        }
        if any(core_metadata.get(key) != value for key, value in expected_core.items()):
            raise RuntimeError(f"core state identity mismatch: {state_id}")
        expected_input_hashes = {
            "core_actions": sha256_file(core_path),
            "core_frozen_inputs": sha256_file(core_root / state_id / "frozen_inputs.pt"),
            "future_b0": sha256_file(core_root / state_id / "future_b0.pt"),
            "future_b1": sha256_file(core_root / state_id / "future_b1.pt"),
        }
        if metadata.get("input_sha256") != expected_input_hashes:
            raise RuntimeError(f"dose source-file binding mismatch: {state_id}")

        frozen_path = core_root / state_id / "frozen_inputs.pt"
        frozen = torch.load(frozen_path, map_location="cpu")
        if (
            set(frozen) != {"init_latent", "action_noises", "action_noise_hashes"}
            or not isinstance(frozen["init_latent"], torch.Tensor)
            or tuple(frozen["init_latent"].shape) != (1, 48, 1, 8, 16)
            or frozen["init_latent"].dtype != torch.bfloat16
            or not isinstance(frozen["action_noises"], torch.Tensor)
            or tuple(frozen["action_noises"].shape) != (4, 1, 30, 4, 4, 1)
            or frozen["action_noises"].dtype != torch.bfloat16
        ):
            raise RuntimeError(f"core frozen-input schema changed: {state_id}")
        actual_noise_hashes = [tensor_hash(value) for value in frozen["action_noises"]]
        if (
            actual_noise_hashes != list(frozen["action_noise_hashes"])
            or actual_noise_hashes != list(core_metadata["action_noise_hashes"])
            or metadata.get("action_noise_sha256") != actual_noise_hashes[0]
        ):
            raise RuntimeError(f"dose action-noise binding mismatch: {state_id}")

        loaded_futures: list[torch.Tensor] = []
        for branch_index in (0, 1):
            future_path = core_root / state_id / f"future_b{branch_index}.pt"
            payload = torch.load(future_path, map_location="cpu")
            future = payload.get("future")
            if (
                set(payload) != {"future", "video_seed"}
                or not isinstance(future, torch.Tensor)
                or tuple(future.shape) != (1, 48, 4, 8, 16)
                or future.dtype != torch.bfloat16
                or payload["video_seed"] != manifest["video_seeds"][branch_index]
                or tensor_hash(future) != core_metadata["future_hashes"][branch_index]
                or not torch.equal(
                    future[:, :, 0:1], frozen["init_latent"][:, :, 0:1]
                )
            ):
                raise RuntimeError(
                    f"core future b{branch_index} binding failed: {state_id}"
                )
            loaded_futures.append(future)
        future0, future1 = loaded_futures
        expected_future_hashes = [tensor_hash(future0)]
        for alpha in EXPECTED_ALPHAS[1:-1]:
            interpolated = (
                (1.0 - float(alpha)) * future0.float()
                + float(alpha) * future1.float()
            ).to(future0.dtype)
            interpolated[:, :, 0:1] = frozen["init_latent"][:, :, 0:1].to(
                interpolated.dtype
            )
            expected_future_hashes.append(tensor_hash(interpolated))
        expected_future_hashes.append(tensor_hash(future1))
        future_hashes = metadata.get("future_sha256", [])
        if (
            future_hashes != expected_future_hashes
            or len(set(expected_future_hashes)) != 5
        ):
            raise RuntimeError(f"dose future hash receipt malformed: {state_id}")
        with np.load(core_path, allow_pickle=False) as core:
            core_full = np.asarray(core["latent_grid_actions"], dtype=np.float32)
            core_executed = np.asarray(core["latent_grid_executed_actions"], dtype=np.float64)
        if not np.array_equal(actions[0], core_full[0, 0]) or not np.array_equal(
            actions[-1], core_full[0, 1]
        ):
            raise RuntimeError(f"full endpoint reuse failed: {state_id}")
        if not np.array_equal(executed[0], core_executed[0, 0]) or not np.array_equal(
            executed[-1], core_executed[0, 1]
        ):
            raise RuntimeError(f"executed endpoint reuse failed: {state_id}")

        flattened = executed.reshape(5, -1)
        direction = flattened[-1] - flattened[0]
        denominator = float(np.dot(direction, direction))
        if denominator == 0.0:
            raise RuntimeError(f"undefined zero endpoint contrast: {state_id}")
        response = ((flattened - flattened[0]) @ direction) / denominator
        if abs(float(response[0])) > 1e-15 or abs(float(response[-1]) - 1.0) > 1e-12:
            raise RuntimeError(f"projection endpoint integrity failed: {state_id}")
        interior = response[1:-1]
        row = {
            "state_id": state_id,
            "task_id": int(record["task_id"]),
            "initial_state_index": int(record["initial_state_index"]),
            "endpoint_distance_l2": float(np.sqrt(denominator)),
            "interior_slope": slope(interior),
            "interior_nondecreasing": int(bool(np.all(np.diff(interior) >= 0.0))),
            "interior_mean_absolute_alpha_deviation": float(
                np.mean(np.abs(interior - EXPECTED_ALPHAS[1:-1]))
            ),
        }
        for alpha, value in zip(EXPECTED_ALPHAS, response):
            row[f"projection_alpha_{alpha:.2f}"] = float(value)
        rows.append(row)
        projections.append(response)
    if len(dose_runner_hashes) != 1:
        raise RuntimeError(f"mixed dose runners: {dose_runner_hashes}")
    return rows, np.stack(projections)


def bootstrap(
    rows: list[dict[str, Any]],
    projections: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    slopes = np.asarray([row["interior_slope"] for row in rows], dtype=np.float64)
    monotonic = np.asarray(
        [row["interior_nondecreasing"] for row in rows], dtype=np.float64
    )
    deviations = np.asarray(
        [row["interior_mean_absolute_alpha_deviation"] for row in rows],
        dtype=np.float64,
    )
    names = [
        "interior_slope",
        "interior_nondecreasing_fraction",
        "interior_mean_absolute_alpha_deviation",
        *[f"projection_alpha_{alpha:.2f}" for alpha in EXPECTED_ALPHAS],
    ]
    values = np.column_stack([slopes, monotonic, deviations, projections])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(repetitions, len(rows)), dtype=np.int16)
    draws = values[indices].mean(axis=1)
    summary = []
    for column, name in enumerate(names):
        low, high = np.quantile(draws[:, column], [0.025, 0.975])
        summary.append(
            {
                "metric": name,
                "estimate": float(values[:, column].mean()),
                "bootstrap_mean": float(draws[:, column].mean()),
                "bootstrap_95_ci_low": float(low),
                "bootstrap_95_ci_high": float(high),
                "n_states": len(rows),
                "repetitions": repetitions,
                "seed": seed,
            }
        )
    return draws, summary


def permutation(
    projections: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if repetitions <= 0:
        raise ValueError("permutation repetitions must be positive")
    interior = projections[:, 1:-1]
    observed_state_slopes = np.asarray([slope(row) for row in interior])
    observed = float(observed_state_slopes.mean())
    mappings = np.asarray(list(itertools.permutations(range(3))), dtype=np.int8)
    x = EXPECTED_ALPHAS[1:-1]
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    rng = np.random.default_rng(seed)
    null = np.empty(repetitions, dtype=np.float64)
    chunk_size = 2_000
    for start in range(0, repetitions, chunk_size):
        count = min(chunk_size, repetitions - start)
        choices = rng.integers(
            0, len(mappings), size=(count, len(interior)), dtype=np.int16
        )
        indices = mappings[choices]
        randomized = np.take_along_axis(interior[None], indices, axis=2)
        state_slopes = np.einsum("j,rsj->rs", centered, randomized) / denominator
        null[start : start + count] = state_slopes.mean(axis=1)
    tolerance = 1e-15
    return null, {
        "metric": "mean_interior_slope",
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_95_interval_low": float(np.quantile(null, 0.025)),
        "null_95_interval_high": float(np.quantile(null, 0.975)),
        "monte_carlo_p_greater_equal": float(
            (np.count_nonzero(null >= observed - tolerance) + 1) / (repetitions + 1)
        ),
        "monte_carlo_p_two_sided_absolute": float(
            (np.count_nonzero(np.abs(null) >= abs(observed) - tolerance) + 1)
            / (repetitions + 1)
        ),
        "permutation_unit": "three interior alpha labels independently within each frozen state; endpoints excluded",
        "repetitions": repetitions,
        "seed": seed,
    }


def plot_dose(projections: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for values in projections:
        axis.plot(EXPECTED_ALPHAS, values, color="#8da0cb", alpha=0.22, linewidth=0.9)
    axis.plot(
        EXPECTED_ALPHAS,
        projections.mean(axis=0),
        color="#d95f02",
        linewidth=2.8,
        marker="o",
        label="unweighted 30-state mean",
    )
    axis.plot(EXPECTED_ALPHAS, EXPECTED_ALPHAS, "--", color="#666666", label="linear reference")
    axis.axhline(0.0, color="#aaaaaa", linewidth=0.8)
    axis.set(
        xlabel="b0→b1 future-latent dose alpha",
        ylabel="projection toward b1 endpoint action",
        xticks=EXPECTED_ALPHAS,
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    figure.tight_layout()
    atomic_figure(output, figure)


def plot_slopes(rows: list[dict[str, Any]], output: Path) -> None:
    values = np.asarray([row["interior_slope"] for row in rows], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.hist(values, bins=12, color="#66c2a5", edgecolor="white")
    axis.axvline(0.0, linestyle="--", color="#666666", label="zero slope")
    axis.axvline(values.mean(), color="#d95f02", linewidth=2.2, label="30-state mean")
    axis.set(xlabel="state-level interior slope", ylabel="state count")
    axis.legend(frameon=False)
    figure.tight_layout()
    atomic_figure(output, figure)


def artifact_index(output_root: Path, metadata: dict[str, Any]) -> None:
    target = output_root / "artifact_index.json"
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != target and ".tmp." not in path.name:
            artifacts.append(
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_json(target, {**metadata, "artifacts": artifacts})


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    core_root = args.core_result_root.resolve()
    output_root = args.output_root.resolve()
    protocol_path = args.protocol.resolve()
    clarification_path = args.clarification.resolve()
    oracle_path = args.oracle_receipt.resolve()
    dose_runner_path = args.dose_runner.resolve()
    checkpoint_content_path = args.checkpoint_content_manifest.resolve()
    if output_root.exists():
        raise RuntimeError("analysis output root must be absent before generation")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("protocol hash differs from preregistration")
    if sha256_file(clarification_path) != CLARIFICATION_SHA256:
        raise RuntimeError("analysis clarification hash differs from outcome-blind freeze")
    protocol = load_json(protocol_path)
    if core_root != Path(protocol["source"]["core_result_root"]).resolve():
        raise RuntimeError("core result root differs from frozen protocol")
    if sha256_file(result_root / "protocol.json") != PROTOCOL_SHA256:
        raise RuntimeError("dose output protocol copy differs from preregistration")
    if not dose_runner_path.is_file() or sha256_file(dose_runner_path) != DOSE_RUNNER_SHA256:
        raise RuntimeError("dose runner differs from canonical executed version")
    checkpoint_content = load_json(checkpoint_content_path)
    checkpoint_content_sha256 = sha256_file(checkpoint_content_path)
    if checkpoint_content.get("huggingface_revisions") != [CHECKPOINT_REVISION]:
        raise RuntimeError("checkpoint content revision mismatch")
    oracle = load_json(oracle_path)
    oracle_sha256 = sha256_file(oracle_path)
    if (
        oracle.get("status") != "complete"
        or oracle.get("included_in_evaluation") is not False
        or oracle.get("parity_gate_passed") is not True
        or oracle.get("upstream_commit") != UPSTREAM_COMMIT
        or oracle.get("checkpoint_revision") != CHECKPOINT_REVISION
        or oracle.get("state_id") != "dev_task00_state000"
        or oracle.get("branch_index") != 0
        or oracle.get("parity_gate", {}).get("sha256") != PARITY_GATE_SHA256
        or oracle.get("audit_script", {}).get("sha256") != ORACLE_SCRIPT_SHA256
        or oracle.get("paths", {}).get("manifest_sha256")
        != protocol["source"]["manifest_sha256"]
        or oracle.get("paths", {}).get("controlled_runner_sha256")
        != CORE_RUNNER_SHA256
        or oracle.get("core_environment_provenance", {}).get(
            "checkpoint_content_manifest_sha256"
        )
        != checkpoint_content_sha256
        or oracle.get("core_environment_provenance", {}).get(
            "checkpoint_aggregate_sha256"
        )
        != checkpoint_content.get("aggregate_sha256")
    ):
        raise RuntimeError("oracle identity/exact-parity binding failed")
    comparison = oracle.get("comparison", {})
    for prefix in ("future", "action"):
        if not isinstance(comparison.get(f"{prefix}_bitwise_equal"), bool):
            raise RuntimeError(f"upstream {prefix} parity audit is absent")
        error = comparison.get(f"{prefix}_max_abs_error")
        if not isinstance(error, (int, float)) or not np.isfinite(error):
            raise RuntimeError(f"upstream {prefix} numerical audit is absent")
        if comparison[f"{prefix}_bitwise_equal"] is not True or float(error) != 0.0:
            raise RuntimeError(f"upstream {prefix} parity is not exact")
    if (
        oracle.get("frozen_input", {}).get("official_encoder_bitwise_equal") is not True
        or float(oracle.get("frozen_input", {}).get("official_encoder_max_abs_error", -1.0))
        != 0.0
    ):
        raise RuntimeError("upstream encoder parity is not exact")

    rows, projections = load_and_validate(
        result_root=result_root,
        core_root=core_root,
        protocol=protocol,
        oracle_sha256=oracle_sha256,
        checkpoint_content_sha256=checkpoint_content_sha256,
        checkpoint_aggregate_sha256=checkpoint_content["aggregate_sha256"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    bootstrap_draws, bootstrap_summary = bootstrap(
        rows, projections, args.bootstrap_repetitions, args.bootstrap_seed
    )
    permutation_null, permutation_summary = permutation(
        projections, args.permutation_repetitions, args.permutation_seed
    )
    state_fields = list(rows[0])
    atomic_csv(output_root / "state_metrics.csv", rows, state_fields)
    atomic_csv(
        output_root / "bootstrap_summary.csv",
        bootstrap_summary,
        list(bootstrap_summary[0]),
    )
    alpha_rows = [
        {
            "alpha": float(alpha),
            **{
                key: item[key]
                for key in (
                    "estimate",
                    "bootstrap_95_ci_low",
                    "bootstrap_95_ci_high",
                    "n_states",
                )
            },
        }
        for alpha, item in zip(EXPECTED_ALPHAS, bootstrap_summary[3:])
    ]
    atomic_csv(output_root / "alpha_summary.csv", alpha_rows, list(alpha_rows[0]))
    atomic_json(output_root / "permutation_summary.json", permutation_summary)
    atomic_npy(output_root / "bootstrap_draws.npy", bootstrap_draws)
    atomic_npy(output_root / "permutation_null.npy", permutation_null)
    atomic_json(
        output_root / "resampling_manifest.json",
        {
            "independent_unit": "frozen simulator state",
            "bootstrap": {
                "draws_file": "bootstrap_draws.npy",
                "columns": [item["metric"] for item in bootstrap_summary],
                "repetitions": args.bootstrap_repetitions,
                "seed": args.bootstrap_seed,
            },
            "permutation": {
                "draws_file": "permutation_null.npy",
                "unit": permutation_summary["permutation_unit"],
                "repetitions": args.permutation_repetitions,
                "seed": args.permutation_seed,
            },
        },
    )
    plot_dose(projections, output_root / "plots/all_state_dose_response.png")
    plot_slopes(rows, output_root / "plots/state_interior_slopes.png")
    primary_bootstrap = next(
        item for item in bootstrap_summary if item["metric"] == "interior_slope"
    )
    summary = {
        "status": "complete",
        "state_count": len(rows),
        "task_counts": dict(sorted(Counter(row["task_id"] for row in rows).items())),
        "ordered_pair": "b0_to_b1",
        "alphas": EXPECTED_ALPHAS.tolist(),
        "action_coordinate_intervention": "none",
        "primary_metric": "mean state-level interior projected-response slope",
        "primary_bootstrap": primary_bootstrap,
        "primary_permutation": permutation_summary,
        "interior_nondecreasing_state_count": int(
            sum(row["interior_nondecreasing"] for row in rows)
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "analysis_clarification_sha256": CLARIFICATION_SHA256,
        "oracle_receipt_path": str(oracle_path),
        "oracle_receipt_sha256": sha256_file(oracle_path),
        "oracle_future_bitwise_equal": comparison["future_bitwise_equal"],
        "oracle_future_max_abs_error": comparison["future_max_abs_error"],
        "oracle_action_bitwise_equal": comparison["action_bitwise_equal"],
        "oracle_action_max_abs_error": comparison["action_max_abs_error"],
        "upstream_parity_claim_scope": (
            "bitwise for the controlled official _infer audit"
            if comparison["future_bitwise_equal"]
            and comparison["action_bitwise_equal"]
            else "numerical audit only; do not claim bitwise upstream parity"
        ),
        "endpoint_cells_reused_from_core": True,
        "endpoint_cells_excluded_from_primary_inference": True,
        "outcome_selected_states_or_examples": False,
        "claim_scope": protocol["dose"]["claim_scope"],
    }
    atomic_json(output_root / "summary.json", summary)
    atomic_text(
        output_root / "README.md",
        "\n".join(
            [
                "# LingBot future-latent dose follow-up",
                "",
                "This is a separate, outcome-blind-preregistered follow-up over every one of the 30 frozen evaluation states.",
                "",
                "The three interior points use new model calls. Alpha 0 and 1 are exact core-grid endpoint reuses and are excluded from the primary interior-slope test.",
                "",
                "All thin lines in the dose-response plot are individual states in frozen order; no example or state is selected by outcome.",
                "",
                "Interpretation is limited to pathwise sensitivity along the fixed b0-to-b1 normalized-latent segment.",
                "",
            ]
        ),
    )
    artifact_index(
        output_root,
        {
            "status": "complete",
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": sha256_file(Path(__file__)),
            "source_result_root": str(result_root),
            "source_core_result_root": str(core_root),
            "protocol": {"path": str(protocol_path), "sha256": PROTOCOL_SHA256},
            "analysis_clarification": {
                "path": str(clarification_path),
                "sha256": CLARIFICATION_SHA256,
            },
            "oracle_receipt": {
                "path": str(oracle_path),
                "sha256": sha256_file(oracle_path),
            },
            "state_count": len(rows),
            "outcome_selected_states_or_examples": False,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
