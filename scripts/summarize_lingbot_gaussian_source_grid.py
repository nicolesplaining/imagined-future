#!/usr/bin/env python3
"""Verify and summarize the complete LingBot Gaussian-source control grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUNNER_SHA256 = "10360ed7e1c166cb7cefd224ce3c936b0b45b57392e5504c4b51e0bdc9ee3e2f"
MANIFEST_SHA256 = "9c8d7dd547955e4622d98c7eb42fffed157f6db983f41f02929b28a434be47b4"
CORE_RUNNER_SHA256 = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
CHECKPOINT_AGGREGATE_SHA256 = "bb895755e071bf5ab74494c07199a11c8e344b367971b4c6405321807e32b2e1"
ORACLE_RECEIPT_SHA256 = "f54eebb2d4525ec8d8c57ebdc3b1294041194e5d1e13fd0033380a09453719aa"
BRANCH_IDS = ("b0", "b1", "b2", "b3")
GAUSSIAN_SEEDS = (900000, 900001, 900002, 900003)
N_RESAMPLES = 100_000
RNG_SEED = 2_026_090_4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--shard-0", type=Path, required=True)
    parser.add_argument("--shard-1", type=Path, required=True)
    parser.add_argument("--initial-failure-log-0", type=Path, required=True)
    parser.add_argument("--initial-failure-log-1", type=Path, required=True)
    parser.add_argument("--successful-log-0", type=Path, required=True)
    parser.add_argument("--successful-log-1", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inventory(root: Path, exclude_index: bool = False) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden: {path}")
        if not path.is_file() or (exclude_index and path == root / "artifact_index.json"):
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode()); aggregate.update(b"\0")
        aggregate.update(str(size).encode()); aggregate.update(b"\0")
        aggregate.update(digest.encode()); aggregate.update(b"\n")
    return rows, aggregate.hexdigest()


def validate_index(root: Path) -> tuple[dict[str, Any], str, tuple[list[dict[str, Any]], str]]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"missing/unsafe shard: {root}")
    index = load_json(root / "artifact_index.json")
    if index.get("status") != "complete_mode_frozen_read_only_shard":
        raise RuntimeError(f"incomplete shard: {root}")
    snapshot = inventory(root, exclude_index=True)
    if snapshot != (index.get("files"), index.get("tree_aggregate_sha256")):
        raise RuntimeError(f"shard index mismatch: {root}")
    if stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise RuntimeError(f"shard root not 0555: {root}")
    if any(stat.S_IMODE(path.stat().st_mode) != 0o444 for path in root.rglob("*") if path.is_file()):
        raise RuntimeError(f"shard has mutable file: {root}")
    return index, sha256_file(root / "artifact_index.json"), snapshot


def projections(value: np.ndarray, templates: np.ndarray, recipient: int, source: int) -> tuple[int, float, float]:
    flat = value.astype(np.float64).reshape(-1)
    refs = templates.astype(np.float64).reshape(4, -1)
    distances = np.linalg.norm(refs - flat[None], axis=1)
    prediction = int(np.argmin(distances))
    axis = refs[source] - refs[recipient]
    denominator = float(axis @ axis)
    if recipient == source:
        projection = 0.0
    elif denominator <= 0:
        raise RuntimeError(f"degenerate recipient/source template axis: {recipient}/{source}")
    else:
        projection = float((flat - refs[recipient]) @ axis / denominator)
    reduction = float((distances[recipient] - distances[source]) / max(distances[recipient] + distances[source], 1e-12))
    return prediction, projection, reduction


def state_bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    n = len(values)
    means = np.empty(N_RESAMPLES, dtype=np.float64)
    for start in range(0, N_RESAMPLES, 5000):
        stop = min(start + 5000, N_RESAMPLES)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "resamples": N_RESAMPLES,
        "unit": "state",
    }


def shared_label_permutation(predictions: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    # One four-label permutation per state, shared across all recipient rows;
    # score only the original 12 off-diagonal cells.
    mask = ~np.eye(4, dtype=bool)[None]
    observed = float((predictions == np.arange(4)[None, None, :])[mask.repeat(len(predictions), axis=0)].mean())
    exceed = 0
    null_sum = 0.0
    for start in range(0, N_RESAMPLES, 1000):
        count = min(1000, N_RESAMPLES - start)
        permutations = np.argsort(rng.random((count, len(predictions), 4)), axis=2)
        correct = predictions[None] == permutations[:, :, None, :]
        scores = correct[:, mask.repeat(len(predictions), axis=0)].mean(axis=1)
        null_sum += float(scores.sum())
        exceed += int(np.count_nonzero(scores >= observed))
    return {
        "observed": observed,
        "null_mean": null_sum / N_RESAMPLES,
        "p_value_plus_one": (exceed + 1) / (N_RESAMPLES + 1),
        "permutations": N_RESAMPLES,
        "scheme": "one shared four-label source permutation per state; original 12 off-diagonal positions",
        "chance_rate": 0.25,
    }


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    if sha256_file(args.manifest) != MANIFEST_SHA256 or sha256_file(args.core_root / "manifest.json") != MANIFEST_SHA256:
        raise RuntimeError("manifest/core identity mismatch")
    manifest = load_json(args.manifest)
    states = [record for record in manifest["states"] if record["admission"] == "evaluation"]
    state_ids = [record["state_id"] for record in states]
    if len(state_ids) != 30 or tuple(manifest["branch_ids"]) != BRANCH_IDS:
        raise RuntimeError("cohort/branches changed")
    shards = (args.shard_0.resolve(), args.shard_1.resolve())
    if shards[0] == shards[1] or any(output == root or output.is_relative_to(root) for root in shards):
        raise RuntimeError("output must be disjoint from two source shards")
    source_indexes = []
    snapshots = []
    state_roots: dict[str, Path] = {}
    for shard_index, root in enumerate(shards):
        index, index_sha, snapshot = validate_index(root)
        provenance = load_json(root / "provenance.json")
        identities = provenance.get("identities", {})
        expected_ids = state_ids[shard_index::2]
        checks = {
            "status": "complete_mode_frozen_read_only_shard",
            "shard_index": shard_index, "shard_count": 2,
            "state_count": 15, "state_ids": expected_ids,
            "all_generated_control_latents_saved": True,
            "action_coordinate_intervention": "none",
        }
        identity_checks = {
            "runner_sha256": RUNNER_SHA256, "manifest_sha256": MANIFEST_SHA256,
            "core_runner_sha256": CORE_RUNNER_SHA256,
            "checkpoint_aggregate_sha256": CHECKPOINT_AGGREGATE_SHA256,
            "oracle_receipt_sha256": ORACLE_RECEIPT_SHA256,
        }
        bad = {key: (provenance.get(key), value) for key, value in checks.items() if provenance.get(key) != value}
        bad.update({f"identity.{key}": (identities.get(key), value) for key, value in identity_checks.items() if identities.get(key) != value})
        if bad:
            raise RuntimeError(f"shard provenance mismatch: {bad}")
        for state_id in expected_ids:
            state_roots[state_id] = root / state_id
        source_indexes.append({"root": str(root), "artifact_index_sha256": index_sha, "tree_aggregate_sha256": index["tree_aggregate_sha256"]})
        snapshots.append(snapshot)
    if set(state_roots) != set(state_ids):
        raise RuntimeError("combined shards are not exact 30-state cohort")

    initial_logs = (args.initial_failure_log_0.resolve(), args.initial_failure_log_1.resolve())
    successful_logs = (args.successful_log_0.resolve(), args.successful_log_1.resolve())
    log_snapshots: dict[Path, tuple[int, str]] = {}
    for shard_index, path in enumerate(initial_logs):
        if not path.is_file():
            raise RuntimeError(f"missing initial failure log: {path}")
        text = path.read_text(errors="replace")
        if (
            "MASTER_ADDR expected, but not set" not in text
            or "complete task" in text
            or "Loading checkpoint shards" in text
        ):
            raise RuntimeError(f"initial log is not the pre-model rendezvous failure: {path}")
        log_snapshots[path] = (path.stat().st_size, sha256_file(path))
    for shard_index, path in enumerate(successful_logs):
        if not path.is_file():
            raise RuntimeError(f"missing successful launch log: {path}")
        text = path.read_text(errors="replace")
        if text.count("complete task") != 15 or "Traceback (most recent call last)" in text:
            raise RuntimeError(f"successful log is incomplete/failed: {path}")
        log_snapshots[path] = (path.stat().st_size, sha256_file(path))

    cell_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    gaussian_predictions = np.empty((30, 4, 4), dtype=np.int64)
    native_predictions = np.empty_like(gaussian_predictions)
    all_control_hashes: list[str] = []
    fail_fast_gate_states: list[str] = []
    for state_index, record in enumerate(states):
        state_id = record["state_id"]
        root = state_roots[state_id]
        result = load_json(root / "result.json")
        actions_path = root / "actions.npz"
        controls_path = root / "gaussian_futures.pt"
        required = {
            "status": "complete", "state_id": state_id,
            "grid_axis_0": "recipient_action_noise_source",
            "grid_axis_1": "norm_matched_gaussian_source_from_native_future_branch",
            "branch_ids": list(BRANCH_IDS), "gaussian_seeds": list(GAUSSIAN_SEEDS),
            "cache_unique_by_source": 4, "cache_exact_across_recipients": True,
            "existing_gaussian_diagonal_bitwise_equal": True,
            "existing_gaussian_diagonal_max_abs_error": 0.0,
            "action_coordinate_intervention": "none",
        }
        bad = {key: (result.get(key), value) for key, value in required.items() if result.get(key) != value}
        if bad:
            raise RuntimeError(f"state result gate failed {state_id}: {bad}")
        if result["actions_sha256"] != sha256_file(actions_path) or result["gaussian_futures_sha256"] != sha256_file(controls_path):
            raise RuntimeError(f"state payload hash mismatch: {state_id}")
        if result.get("fail_fast_native_replay_executed") is True:
            if result.get("fail_fast_gaussian_replay_max_abs_error") != 0.0:
                raise RuntimeError(f"fail-fast replay mismatch: {state_id}")
            fail_fast_gate_states.append(state_id)
        elif result.get("fail_fast_gaussian_replay_max_abs_error") is not None:
            raise RuntimeError(f"unexpected replay value on nongate state: {state_id}")
        payload = torch.load(controls_path, map_location="cpu", weights_only=False)
        controls = payload["gaussian_futures"]
        if tuple(controls.shape) != (4, 1, 48, 4, 8, 16) or controls.dtype != torch.bfloat16:
            raise RuntimeError(f"saved Gaussian tensor schema mismatch: {state_id}")
        hashes = [tensor_hash(value) for value in controls]
        if hashes != payload["gaussian_future_tensor_sha256"] or hashes != result["gaussian_future_tensor_sha256"] or len(set(hashes)) != 4:
            raise RuntimeError(f"saved Gaussian tensor hashes mismatch: {state_id}")
        all_control_hashes.extend(hashes)
        if any(max(metric.values()) > result["norm_match_tolerance"] for metric in result["norm_match_metrics"]):
            raise RuntimeError(f"norm-match tolerance failed: {state_id}")
        cache_grid = result["grid_installed_cache_sha256"]
        if any(len({cache_grid[recipient][source] for recipient in range(4)}) != 1 for source in range(4)) or len(set(cache_grid[0])) != 4:
            raise RuntimeError(f"cache source identity failed: {state_id}")
        noise_hashes = result["action_noise_hashes"]
        if result["grid_action_noise_hashes"] != [[noise_hashes[recipient]] * 4 for recipient in range(4)]:
            raise RuntimeError(f"fixed recipient-noise grid failed: {state_id}")
        with np.load(actions_path, allow_pickle=False) as archive:
            if set(archive.files) != {"gaussian_grid_actions", "gaussian_grid_executed_actions", "existing_gaussian_diagonal_actions"}:
                raise RuntimeError(f"action array schema mismatch: {state_id}")
            actions = archive["gaussian_grid_actions"]
            executed = archive["gaussian_grid_executed_actions"]
            gaussian_templates_full = archive["existing_gaussian_diagonal_actions"]
        if actions.shape != (4, 4, 7, 4, 4) or actions.dtype != np.float32 or executed.shape != (4, 4, 7, 3, 4) or executed.dtype != np.float32 or not np.isfinite(actions).all() or not np.array_equal(executed, actions[..., 1:, :]):
            raise RuntimeError(f"action array shape/dtype/execution failed: {state_id}")
        if not np.array_equal(actions[np.arange(4), np.arange(4)], gaussian_templates_full):
            raise RuntimeError(f"Gaussian diagonal bytes failed: {state_id}")
        core_actions_path = args.core_root / state_id / "actions.npz"
        if sha256_file(core_actions_path) != result["core_actions_sha256"]:
            raise RuntimeError(f"current core actions changed: {state_id}")
        with np.load(core_actions_path, allow_pickle=False) as core:
            native_templates = core["native_executed_actions"]
            if not np.array_equal(core["gaussian_actions"], gaussian_templates_full):
                raise RuntimeError(f"existing Gaussian template binding failed: {state_id}")
        gaussian_templates = gaussian_templates_full[..., 1:, :]
        state_gaussian_correct = 0
        state_native_correct = 0
        state_gaussian_proj = []
        state_native_proj = []
        state_gaussian_reduction = []
        state_native_reduction = []
        state_native_recipient_retained = 0
        for recipient in range(4):
            for source in range(4):
                g_pred, g_proj, g_reduction = projections(executed[recipient, source], gaussian_templates, recipient, source)
                n_pred, n_proj, n_reduction = projections(executed[recipient, source], native_templates, recipient, source)
                gaussian_predictions[state_index, recipient, source] = g_pred
                native_predictions[state_index, recipient, source] = n_pred
                offdiag = recipient != source
                if offdiag:
                    state_gaussian_correct += int(g_pred == source)
                    state_native_correct += int(n_pred == source)
                    state_gaussian_proj.append(g_proj); state_native_proj.append(n_proj)
                    state_gaussian_reduction.append(g_reduction)
                    state_native_reduction.append(n_reduction)
                    state_native_recipient_retained += int(n_pred == recipient)
                cell_rows.append({
                    "state_id": state_id, "task_id": int(record["task_id"]),
                    "recipient": recipient, "source": source, "off_diagonal": offdiag,
                    "gaussian_template_prediction": g_pred,
                    "gaussian_source_correct": int(g_pred == source),
                    "gaussian_projection": g_proj, "gaussian_distance_reduction": g_reduction,
                    "native_template_prediction": n_pred,
                    "native_donor_correct": int(n_pred == source),
                    "native_projection": n_proj, "native_distance_reduction": n_reduction,
                })
        state_rows.append({
            "state_id": state_id, "task_id": int(record["task_id"]),
            "gaussian_source_retrieval_offdiag": state_gaussian_correct / 12,
            "native_donor_alignment_offdiag": state_native_correct / 12,
            "gaussian_projection_offdiag": float(np.mean(state_gaussian_proj)),
            "native_projection_offdiag": float(np.mean(state_native_proj)),
            "gaussian_distance_reduction_offdiag": float(np.mean(state_gaussian_reduction)),
            "native_distance_reduction_offdiag": float(np.mean(state_native_reduction)),
            "native_recipient_retention_offdiag": state_native_recipient_retained / 12,
        })
    if len(set(all_control_hashes)) != 120:
        raise RuntimeError("combined saved Gaussian controls are not globally unique")
    if fail_fast_gate_states != [state_ids[0], state_ids[1]]:
        raise RuntimeError(f"one fail-fast gate per shard did not pass: {fail_fast_gate_states}")

    rng = np.random.default_rng(RNG_SEED)
    gaussian_state = np.asarray([row["gaussian_source_retrieval_offdiag"] for row in state_rows])
    native_state = np.asarray([row["native_donor_alignment_offdiag"] for row in state_rows])
    gaussian_proj_state = np.asarray([row["gaussian_projection_offdiag"] for row in state_rows])
    native_proj_state = np.asarray([row["native_projection_offdiag"] for row in state_rows])
    gaussian_reduction_state = np.asarray([row["gaussian_distance_reduction_offdiag"] for row in state_rows])
    native_reduction_state = np.asarray([row["native_distance_reduction_offdiag"] for row in state_rows])
    native_retention_state = np.asarray([row["native_recipient_retention_offdiag"] for row in state_rows])
    summary = {
        "schema_version": 1, "status": "complete",
        "design": "complete 30-state x 4 recipient-action-noise x 4 branch-derived norm-matched Gaussian-source grid",
        "state_count": 30, "cell_count": 480, "off_diagonal_cell_count": 360,
        "all_exact_controls_pass": True, "all_120_control_latents_saved_and_unique": True,
        "timing": "postanalysis exploratory control requested after the core and dose outcomes; not publicly preregistered",
        "cohort_selection_uses_gaussian_outcomes": False,
        "gaussian_source_retrieval_off_diagonal": {
            "bootstrap": state_bootstrap(gaussian_state, rng),
            "permutation": shared_label_permutation(gaussian_predictions, rng),
            "template": "four diagonal existing Gaussian-control actions",
        },
        "native_donor_alignment_off_diagonal": {
            "bootstrap": state_bootstrap(native_state, rng),
            "permutation": shared_label_permutation(native_predictions, rng),
            "template": "four native branch actions",
        },
        "gaussian_template_projection_off_diagonal": state_bootstrap(gaussian_proj_state, rng),
        "native_template_projection_off_diagonal": state_bootstrap(native_proj_state, rng),
        "gaussian_template_distance_reduction_off_diagonal": state_bootstrap(gaussian_reduction_state, rng),
        "native_template_distance_reduction_off_diagonal": state_bootstrap(native_reduction_state, rng),
        "native_recipient_retention_off_diagonal": state_bootstrap(native_retention_state, rng),
        "interpretation": "Arbitrary-source/norm-matched Gaussian routing audit. Sources inherit branch-specific first/second moments and norm; this is not an equal-geometry null, semantic-content test, or natural-future intervention.",
        "rng_seed": RNG_SEED, "resamples": N_RESAMPLES,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        logs_root = staging / "logs"
        logs_root.mkdir()
        copied_logs: list[dict[str, Any]] = []
        for category, sources in (("initial_failure", initial_logs), ("successful", successful_logs)):
            for shard_index, source in enumerate(sources):
                destination = logs_root / f"{category}_shard{shard_index}.log"
                shutil.copy2(source, destination)
                if (destination.stat().st_size, sha256_file(destination)) != log_snapshots[source]:
                    raise RuntimeError(f"log changed while copying: {source}")
                copied_logs.append({
                    "category": category, "shard_index": shard_index,
                    "source_path": str(source), "package_path": destination.relative_to(staging).as_posix(),
                    "bytes": destination.stat().st_size, "sha256": sha256_file(destination),
                })
        common_argv = [
            "/home/ubuntu/if_external/envs/lingbot/bin/python",
            "/home/ubuntu/if_external/tools/run_lingbot_gaussian_source_grid.py",
            "--lingbot-root", "/home/ubuntu/if_external/lingbot-va",
            "--checkpoint", "/home/ubuntu/if_external/checkpoints/lingbot-va-posttrain-libero-long",
            "--checkpoint-manifest", "/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_eval_v1_artifacts_v2/checkpoint_content_manifest.json",
            "--oracle-receipt", "/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_upstream_native_parity_v4/upstream_native_parity.json",
            "--shim", "/home/ubuntu/if_external/compat/flash_attn/__init__.py",
            "--manifest", "/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_eval_v1/manifest.json",
            "--core-root", "/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_eval_v1",
            "--core-runner", "/home/ubuntu/if_external/tools/run_lingbot_future_transplants.py",
            "--dose-validator", "/home/ubuntu/if_external/tools/run_lingbot_future_dose.py",
        ]
        execution_receipt = {
            "schema_version": 1, "status": "complete",
            "initial_pre_model_failures": [
                {
                    "pid": 46739 + shard_index,
                    "shard_index": shard_index,
                    "failure": "torch.distributed rendezvous environment omitted MASTER_ADDR",
                    "disposition": "no scientific artifact produced",
                    "model_constructed": False,
                    "state_completed": False,
                    "installed_output_root_created": False,
                    "staging_cleanup_is_fail_closed_in_runner": True,
                    "environment": {"CUDA_VISIBLE_DEVICES": str(shard_index), "PYTHONPATH": "/home/ubuntu/if_external/compat", "MASTER_ADDR": None},
                    "argv": common_argv + ["--output-root", str(shards[shard_index]), "--shard-index", str(shard_index), "--shard-count", "2"],
                    "log": f"logs/initial_failure_shard{shard_index}.log",
                }
                for shard_index in range(2)
            ],
            "successful_launches": [
                {
                    "pid": 47067 + shard_index,
                    "shard_index": shard_index,
                    "physical_gpu": shard_index,
                    "master_addr": "127.0.0.1",
                    "master_port": 29920 + shard_index,
                    "rank": 0, "local_rank": 0, "world_size": 1,
                    "pythonpath": "/home/ubuntu/if_external/compat",
                    "argv": common_argv + ["--output-root", str(shards[shard_index]), "--shard-index", str(shard_index), "--shard-count", "2"],
                    "output_root": str(shards[shard_index]),
                    "log": f"logs/successful_shard{shard_index}.log",
                }
                for shard_index in range(2)
            ],
            "logs": copied_logs,
            "runner_path": "/home/ubuntu/if_external/tools/run_lingbot_gaussian_source_grid.py",
            "runner_sha256": RUNNER_SHA256,
            "note": "The first wrappers failed at distributed initialization before VA_Server construction and before any state result. The same still-absent target roots were then used by the successful, method-identical launches with the required rendezvous variables.",
        }
        write_json(staging / "execution_receipt.json", execution_receipt)
        write_json(staging / "summary.json", summary)
        for name, rows in (("cells.csv", cell_rows), ("state_metrics.csv", state_rows)):
            with (staging / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        shutil.copy2(Path(__file__).resolve(), staging / "summarize_lingbot_gaussian_source_grid.py")
        write_json(staging / "provenance.json", {
            "schema_version": 1, "status": "complete_mode_frozen_read_only_analysis",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "output_root": str(output),
            "source_shards": source_indexes,
            "analysis_policy": "All 30 frozen states and all 4x4 cells included. Metrics were fixed in this analyzer before completed Gaussian-grid outcomes were opened; the follow-up itself is postanalysis and not a public preregistration.",
            "execution_receipt_sha256": sha256_file(staging / "execution_receipt.json"),
            "identities": {"runner_sha256": RUNNER_SHA256, "analyzer_sha256": sha256_file(Path(__file__).resolve()), "manifest_sha256": MANIFEST_SHA256, "core_runner_sha256": CORE_RUNNER_SHA256, "checkpoint_aggregate_sha256": CHECKPOINT_AGGREGATE_SHA256, "oracle_receipt_sha256": ORACLE_RECEIPT_SHA256},
        })
        for root, snapshot in zip(shards, snapshots, strict=True):
            if inventory(root, exclude_index=True) != snapshot:
                raise RuntimeError(f"source shard changed during analysis: {root}")
        for path, snapshot in log_snapshots.items():
            if (path.stat().st_size, sha256_file(path)) != snapshot:
                raise RuntimeError(f"source log changed during analysis: {path}")
        rows, aggregate = inventory(staging, exclude_index=True)
        index = {"schema_version": 1, "status": "complete_mode_frozen_read_only_analysis", "file_count_excluding_index": len(rows), "tree_aggregate_sha256": aggregate, "files": rows}
        write_json(staging / "artifact_index.json", index)
        for path in staging.rglob("*"):
            if path.is_file(): path.chmod(0o444)
        for path in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True): path.chmod(0o555)
        staging.chmod(0o555)
        if inventory(staging, exclude_index=True) != (rows, aggregate):
            raise RuntimeError("analysis changed during freeze")
        if any(stat.S_IMODE(path.stat().st_mode) != 0o444 for path in staging.rglob("*") if path.is_file()):
            raise RuntimeError("analysis has mutable files")
        if stat.S_IMODE(staging.stat().st_mode) != 0o555 or any(stat.S_IMODE(path.stat().st_mode) != 0o555 for path in staging.rglob("*") if path.is_dir()):
            raise RuntimeError("analysis has mutable directories")
        os.replace(staging, output)
        print(json.dumps({"status": index["status"], "output_root": str(output), "artifact_index_sha256": sha256_file(output / "artifact_index.json"), "tree_aggregate_sha256": aggregate, "summary": summary}, sort_keys=True), flush=True)
    except BaseException:
        if staging.exists():
            for path in staging.rglob("*"):
                try: path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError: pass
            staging.chmod(0o755); shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
