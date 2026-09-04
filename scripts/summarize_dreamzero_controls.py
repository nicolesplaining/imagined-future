#!/usr/bin/env python3
"""Cryptographically audit and summarize DreamZero Gaussian/dose controls.

Every result is bound to the exact frozen core state, action grid, trace set,
cohort manifest, and runner identity before any scientific metric is computed.
The saved simulator state is the independent unit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import run_dreamzero_controls as gaussian_runner
import run_dreamzero_dose_response as dose_runner
import run_dreamzero_future_transplants as core


EPS = 1e-12
ALPHAS = np.asarray(dose_runner.ALPHAS, dtype=np.float64)
HEX = set("0123456789abcdef")
EXPECTED_MANIFEST_SHA256 = "d1ffc3111a10bed9ac8fdd17c631dc3a5d8eb3128ac4fa250d9398bcede12cfc"
EXPECTED_MANIFEST_ID = "dreamzero-droid-states-bef2b2e841db4dd3"
EXPECTED_FAMILIES = ("move", "open", "pick", "place", "press", "put", "remove", "take", "turn", "use")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian-root", type=Path, required=True)
    parser.add_argument("--dose-root", type=Path, required=True)
    parser.add_argument("--off-record-debug-root", type=Path, required=True)
    parser.add_argument("--core-result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core-server-trace-root", type=Path, required=True)
    parser.add_argument("--gaussian-server-trace-root", type=Path, required=True)
    parser.add_argument("--dose-server-trace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-state-count", type=int, default=30)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50_000)
    parser.add_argument("--permutation-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=260904)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - HEX)


def require_frozen(path: Path, expected_sha: str | None = None) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    digest = core.verify_sha_sidecar(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"SHA does not match owning artifact: {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise RuntimeError(f"completed artifact is not read-only: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.is_symlink() or stat.S_IMODE(sidecar.stat().st_mode) != 0o444:
        raise RuntimeError(f"artifact sidecar is not immutable: {sidecar}")
    return digest


def load_frozen_json(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    require_frozen(path, expected_sha)
    return core.load_json(path)


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def rng_for(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def estimate(values: Sequence[float], repetitions: int, rng: np.random.Generator) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("estimate contains no values or nonfinite values")
    draws = rng.integers(0, len(array), size=(repetitions, len(array)))
    bootstrap = array[draws].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "state_bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
        "n_states": int(len(array)),
    }


def sign_flip_pvalue(
    values: Sequence[float], repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    """One-sided state-level Monte Carlo sign-flip test of a positive mean."""
    array = np.asarray(values, dtype=np.float64)
    observed = float(array.mean())
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(repetitions, len(array)))
    null = (signs * array[None, :]).mean(axis=1)
    return (
        (float(np.sum(null >= observed - 1e-15)) + 1.0) / (repetitions + 1.0),
        float(null.mean()),
    )


def audit_off_record_debug(root: Path) -> dict[str, Any]:
    import torch

    result_path = root / "result.json"
    result_sha = require_frozen(result_path)
    result = core.load_json(result_path)
    runner_sha = sha256_file(Path(gaussian_runner.__file__).resolve())
    if (
        result.get("schema") != gaussian_runner.DEBUG_SCHEMA
        or result.get("status") != "complete"
        or result.get("admission") != "excluded_debug_smoke"
        or result.get("scientific_admission") is not False
        or result.get("mode_off_record_bit_exact") is not True
        or float(result.get("maximum_absolute_error", math.inf)) != 0.0
        or int(result.get("noise_seed", -1)) != 211
        or not valid_sha(result.get("input_fingerprint"))
        or result.get("runner_sha256") != runner_sha
    ):
        raise ValueError("mode-off/record debug control identity or parity gate failed")
    arrays_path = Path(str(result.get("actions_npz", {}).get("path", "")))
    if arrays_path.resolve() != (root / "actions.npz").resolve():
        raise ValueError("debug action artifact path differs")
    arrays_sha = require_frozen(arrays_path, result["actions_npz"]["sha256"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != {"mode_off_action", "mode_record_action"}:
            raise ValueError("debug action artifact schema differs")
        off = np.asarray(archive["mode_off_action"])
        record = np.asarray(archive["mode_record_action"])
    if not np.array_equal(off, record) or list(off.shape) != result.get("action_shape") or str(off.dtype) != result.get("action_dtype"):
        raise ValueError("mode-off and record actions are not bit exact")
    trace_meta = result.get("trace", {})
    client_trace = root / "traces" / "debug_record_seed_211.pt"
    if Path(str(trace_meta.get("client_path", ""))).resolve() != client_trace.resolve():
        raise ValueError("debug client trace path differs")
    trace_sha = require_frozen(client_trace, trace_meta.get("sha256"))
    trace = torch.load(client_trace, map_location="cpu", weights_only=True)
    if (
        trace.get("format_version") != 3
        or gaussian_runner.video_trace_sha256(trace) != trace.get("video_trace_sha256")
        or gaussian_runner.tensor_sha256(trace["initial_action_noise"]) != trace.get("initial_action_noise_sha256")
    ):
        raise ValueError("debug record trace content differs")
    gaussian_runner.validate_off_audit(
        {core.AUDIT_KEY: result.get("mode_off_audit")}, 211
    )
    core.validate_server_audit(
        {core.AUDIT_KEY: result.get("mode_record_audit")},
        label="mode_record_debug_gate",
        mode="record",
        noise_seed=211,
        trace_path=Path(str(trace_meta["server_path"])),
        action_reference_path=None,
        replay_start=0,
        replay_stop=core.EXPECTED_SOLVER_STEPS,
        expected_source_trace_hash=None,
        expected_source_action_hash=None,
        expected_recipient_action_hash=None,
    )
    audit = result["mode_record_audit"]
    if (
        audit.get("video_trace_sha256") != trace.get("video_trace_sha256")
        or audit.get("active_action_noise_sha256") != trace.get("initial_action_noise_sha256")
    ):
        raise ValueError("debug server audit does not bind saved record trace")
    return {
        "status": "passed",
        "scientific_admission": False,
        "result_sha256": result_sha,
        "actions_sha256": arrays_sha,
        "trace_sha256": trace_sha,
        "runner_sha256": runner_sha,
        "maximum_absolute_error": 0.0,
    }


def exact_state(result: Mapping[str, Any], manifest_state: Mapping[str, Any]) -> None:
    expected = {
        "state_id": str(manifest_state["state_id"]),
        "state_index": int(manifest_state["state_index"]),
        "episode_index": int(manifest_state["episode_index"]),
        "frame_index": int(manifest_state["frame_index"]),
        "task_family": str(manifest_state["task_family"]),
        "prompt": str(manifest_state["task"]),
    }
    observed = dict(result.get("state", {}))
    input_fingerprint = observed.pop("input_fingerprint", None)
    if observed != expected or not valid_sha(input_fingerprint):
        raise ValueError(f"state/manifest identity differs: {expected['state_id']}")


def audit_core_state(args: argparse.Namespace, manifest_state: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    state_id = str(manifest_state["state_id"])
    state_dir = args.core_result_root / "states" / state_id
    result_path = state_dir / "result.json"
    result_sha = require_frozen(result_path)
    result = core.load_json(result_path)
    if (
        result.get("schema") != core.SCHEMA
        or result.get("status") != "complete"
        or result.get("admission") != "evaluation"
        or result.get("scientific_admission") is not True
        or result_path.parent.name != state_id
        or result.get("runner_sha256") != sha256_file(Path(core.__file__).resolve())
        or result.get("provenance", {}).get("manifest_file_sha256") != manifest_sha
    ):
        raise ValueError(f"core result identity/status differs: {state_id}")
    exact_state(result, manifest_state)
    if int(result.get("call_count", -1)) != 20 or int(result.get("replay_grid_call_count", -1)) != 16:
        raise ValueError(f"core call grid incomplete: {state_id}")
    if result.get("self_replay_all_bit_exact") is not True or float(result.get("self_replay_maximum_error", math.inf)) != 0.0:
        raise ValueError(f"core self-replay gate failed: {state_id}")

    arrays_path = state_dir / result["artifacts"]["actions_npz"]["relative_path"]
    require_frozen(arrays_path, result["artifacts"]["actions_npz"]["sha256"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != {"branch_seeds", "native_actions", "replay_actions"}:
            raise ValueError(f"core NPZ schema differs: {state_id}")
        seeds = np.asarray(archive["branch_seeds"], dtype=np.int64)
        native = np.asarray(archive["native_actions"])
        replay = np.asarray(archive["replay_actions"])
    if tuple(seeds.tolist()) != tuple(core.BRANCH_SEEDS) or native.shape[0] != 4 or replay.shape[:2] != (4, 4):
        raise ValueError(f"core action shapes/seeds differ: {state_id}")

    inventory_path = state_dir / result["artifacts"]["trace_inventory"]["relative_path"]
    inventory = load_frozen_json(inventory_path, result["artifacts"]["trace_inventory"]["sha256"])
    trace_rows = inventory.get("traces")
    if (
        inventory.get("schema") != core.INVENTORY_SCHEMA
        or inventory.get("state_id") != state_id
        or int(inventory.get("trace_count", -1)) != 4
        or not isinstance(trace_rows, list)
        or len(trace_rows) != 4
    ):
        raise ValueError(f"core trace inventory differs: {state_id}")
    traces = {int(row["branch_seed"]): row for row in trace_rows}
    if len(traces) != 4 or set(traces) != set(core.BRANCH_SEEDS):
        raise ValueError(f"core trace seeds differ: {state_id}")
    for seed, row in traces.items():
        trace_path = args.core_result_root / "traces" / row["client_relative_path"]
        require_frozen(trace_path, row["sha256"])
        if row.get("server_path") != str(args.core_server_trace_root / state_id / f"native_seed_{seed}.pt"):
            raise ValueError(f"core server trace mapping differs: {state_id}/{seed}")
        if not valid_sha(row.get("video_trace_sha256")) or not valid_sha(row.get("action_noise_sha256")):
            raise ValueError(f"core trace content hash malformed: {state_id}/{seed}")

    grid_rows = result.get("grid")
    if not isinstance(grid_rows, list) or len(grid_rows) != 16:
        raise ValueError(f"core grid size differs: {state_id}")
    grid = {(int(row["recipient_seed"]), int(row["future_source_seed"])): row for row in grid_rows}
    expected_pairs = {(r, s) for r in core.BRANCH_SEEDS for s in core.BRANCH_SEEDS}
    if len(grid) != 16 or set(grid) != expected_pairs:
        raise ValueError(f"core grid pairs differ: {state_id}")
    for recipient, source in sorted(expected_pairs):
        ri = core.BRANCH_SEEDS.index(recipient)
        si = core.BRANCH_SEEDS.index(source)
        row = grid[(recipient, source)]
        if row.get("sha256") != core.array_sha256(np.ascontiguousarray(replay[ri, si])):
            raise ValueError(f"core grid action hash differs: {state_id}/{recipient}/{source}")
        core.validate_server_audit(
            {core.AUDIT_KEY: row.get("server_audit")},
            label=f"core_{state_id}_{recipient}_{source}",
            mode="replay",
            noise_seed=recipient,
            trace_path=args.core_server_trace_root / state_id / f"native_seed_{source}.pt",
            action_reference_path=args.core_server_trace_root / state_id / f"native_seed_{recipient}.pt",
            replay_start=0,
            replay_stop=core.EXPECTED_SOLVER_STEPS,
            expected_source_trace_hash=traces[source]["video_trace_sha256"],
            expected_source_action_hash=traces[source]["action_noise_sha256"],
            expected_recipient_action_hash=traces[recipient]["action_noise_sha256"],
        )
    if not all(np.array_equal(replay[i, i], native[i]) for i in range(4)):
        raise ValueError(f"core diagonal is not exact native replay: {state_id}")

    checkpoint_path = state_dir / "checkpoint.json"
    checkpoint_sha = require_frozen(checkpoint_path)
    checkpoint = core.load_json(checkpoint_path)
    expected_labels = {
        *(f"native_record_seed_{seed}" for seed in core.BRANCH_SEEDS),
        *(
            f"replay_recipient_{recipient}_source_{source}"
            for recipient in core.BRANCH_SEEDS
            for source in core.BRANCH_SEEDS
        ),
    }
    expected_intervention = {
        "control_key": core.CONTROL_KEY,
        "audit_key": core.AUDIT_KEY,
        "future_source": "source native video trace",
        "action_noise_source": "recipient native trace",
        "replay_start_inclusive": 0,
        "replay_stop_exclusive": core.EXPECTED_SOLVER_STEPS,
        "expected_solver_steps": core.EXPECTED_SOLVER_STEPS,
        "action_coordinates_written_by_client": False,
    }
    final = checkpoint.get("final_artifacts", {})
    if (
        checkpoint.get("schema") != core.CHECKPOINT_SCHEMA
        or checkpoint.get("status") != "complete"
        or checkpoint.get("admission") != "evaluation"
        or checkpoint.get("runner_sha256") != result["runner_sha256"]
        or checkpoint.get("provenance") != result["provenance"]
        or checkpoint.get("state") != result["state"]
        or checkpoint.get("branch_seeds") != list(core.BRANCH_SEEDS)
        or checkpoint.get("intervention") != expected_intervention
        or checkpoint.get("input_audit", {}).get("input_fingerprint")
            != result["state"]["input_fingerprint"]
        or set(checkpoint.get("completed", {})) != expected_labels
        or final.get("result", {}).get("sha256") != result_sha
        or final.get("actions", {}).get("sha256") != sha256_file(arrays_path)
        or final.get("traces", {}).get("sha256") != sha256_file(inventory_path)
    ):
        raise ValueError(f"core frozen checkpoint/header differs: {state_id}")
    completed = checkpoint["completed"]
    for seed in core.BRANCH_SEEDS:
        label = f"native_record_seed_{seed}"
        record = completed[label]
        index = core.BRANCH_SEEDS.index(seed)
        saved_action = core.action_from_record(record, label)
        client_trace = args.core_result_root / "traces" / state_id / f"native_seed_{seed}.pt"
        server_trace = args.core_server_trace_root / state_id / f"native_seed_{seed}.pt"
        if (
            not np.array_equal(saved_action, native[index])
            or record.get("label") != label
            or int(record.get("branch_seed", -1)) != seed
            or record.get("client_trace_relative_path") != f"{state_id}/native_seed_{seed}.pt"
            or record.get("server_trace_path") != str(server_trace)
            or record.get("trace_file_sha256") != traces[seed]["sha256"]
            or int(record.get("trace_size_bytes", -1)) != client_trace.stat().st_size
        ):
            raise ValueError(f"core native checkpoint record differs: {state_id}/{seed}")
        audit = core.validate_server_audit(
            {core.AUDIT_KEY: record.get("server_audit")},
            label=label,
            mode="record",
            noise_seed=seed,
            trace_path=server_trace,
            action_reference_path=None,
            replay_start=0,
            replay_stop=core.EXPECTED_SOLVER_STEPS,
            expected_source_trace_hash=None,
            expected_source_action_hash=None,
            expected_recipient_action_hash=None,
        )
        if (
            audit["video_trace_sha256"] != traces[seed]["video_trace_sha256"]
            or audit["active_action_noise_sha256"] != traces[seed]["action_noise_sha256"]
        ):
            raise ValueError(f"core native trace/server audit differs: {state_id}/{seed}")
    for recipient, source in sorted(expected_pairs):
        label = f"replay_recipient_{recipient}_source_{source}"
        record = completed[label]
        ri = core.BRANCH_SEEDS.index(recipient)
        si = core.BRANCH_SEEDS.index(source)
        if (
            not np.array_equal(core.action_from_record(record, label), replay[ri, si])
            or {key: value for key, value in record.items() if key != "values"}
                != grid[(recipient, source)]
        ):
            raise ValueError(f"core replay checkpoint/result grid differs: {state_id}/{recipient}/{source}")
    return {
        "state_id": state_id,
        "result": result,
        "result_path": result_path,
        "result_sha256": result_sha,
        "arrays_sha256": sha256_file(arrays_path),
        "native": native,
        "replay": replay,
        "traces": traces,
        "trace_paths": {
            seed: args.core_result_root / "traces" / state_id / f"native_seed_{seed}.pt"
            for seed in core.BRANCH_SEEDS
        },
        "checkpoint_sha256": checkpoint_sha,
    }


def audit_gaussian(args: argparse.Namespace, frozen: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    import torch

    state_id = str(frozen["state_id"])
    state_dir = args.gaussian_root / "states" / state_id
    result_path = state_dir / "result.json"
    result_sha = require_frozen(result_path)
    result = core.load_json(result_path)
    expected_runner = sha256_file(Path(gaussian_runner.__file__).resolve())
    if (
        result.get("schema") != gaussian_runner.GAUSSIAN_SCHEMA
        or result.get("status") != "complete"
        or result.get("admission") != "evaluation_control"
        or result.get("scientific_admission") is not True
        or result.get("control_class") != "incoherent_per_step_norm_matched_gaussian"
        or result_path.parent.name != state_id
        or result.get("state") != frozen["result"]["state"]
        or result.get("input_fingerprint") != frozen["result"]["state"]["input_fingerprint"]
        or result.get("branch_seeds") != list(core.BRANCH_SEEDS)
        or int(result.get("source_seed_for_norms", -1)) != 211
        or int(result.get("call_count", -1)) != 4
        or result.get("runner_sha256") != expected_runner
        or result.get("core_result", {}).get("sha256") != frozen["result_sha256"]
        or Path(str(result.get("core_result", {}).get("path", ""))).resolve() != frozen["result_path"].resolve()
    ):
        raise ValueError(f"Gaussian result/core identity differs: {state_id}")

    arrays_path = state_dir / result["actions_npz"]["relative_path"]
    arrays_sha = require_frozen(arrays_path, result["actions_npz"]["sha256"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != {"branch_seeds", "native_actions", "gaussian_actions"}:
            raise ValueError(f"Gaussian NPZ schema differs: {state_id}")
        seeds = np.asarray(archive["branch_seeds"], dtype=np.int64)
        native = np.asarray(archive["native_actions"])
        actions = np.asarray(archive["gaussian_actions"])
    if tuple(seeds.tolist()) != tuple(core.BRANCH_SEEDS) or not np.array_equal(native, frozen["native"]):
        raise ValueError(f"Gaussian native actions are not exact core actions: {state_id}")
    if actions.shape != native.shape or actions.dtype != native.dtype:
        raise ValueError(f"Gaussian action schema differs: {state_id}")

    trace_meta = result.get("gaussian_trace")
    if not isinstance(trace_meta, dict):
        raise ValueError(f"Gaussian trace metadata absent: {state_id}")
    expected_client = args.gaussian_root / "traces" / state_id / "norm_matched_gaussian.pt"
    expected_server = args.gaussian_server_trace_root / state_id / "norm_matched_gaussian.pt"
    if Path(str(trace_meta.get("client_path", ""))).resolve() != expected_client.resolve() or trace_meta.get("server_path") != str(expected_server):
        raise ValueError(f"Gaussian trace mapping differs: {state_id}")
    trace_sha = require_frozen(expected_client, trace_meta.get("trace_file_sha256"))
    trace = torch.load(expected_client, map_location="cpu", weights_only=True)
    source_trace = torch.load(
        frozen["trace_paths"][211], map_location="cpu", weights_only=True
    )
    if (
        trace.get("format_version") != 3
        or gaussian_runner.video_trace_sha256(trace) != trace_meta.get("video_trace_sha256")
        or trace.get("video_trace_sha256") != trace_meta.get("video_trace_sha256")
        or gaussian_runner.tensor_sha256(trace["initial_action_noise"]) != trace_meta.get("source_action_noise_sha256")
        or trace.get("initial_action_noise_sha256") != frozen["traces"][211]["action_noise_sha256"]
        or gaussian_runner.tensor_sha256(source_trace["initial_action_noise"])
            != frozen["traces"][211]["action_noise_sha256"]
        or not torch.equal(trace["initial_action_noise"], source_trace["initial_action_noise"])
    ):
        raise ValueError(f"Gaussian trace content differs: {state_id}")
    provenance = trace.get("synthetic_provenance", {})
    expected_source_path = args.core_result_root / "traces" / state_id / "native_seed_211.pt"
    if (
        provenance.get("kind") != "incoherent_per_step_norm_matched_gaussian"
        or provenance.get("state_id") != state_id
        or provenance.get("gaussian_salt") != gaussian_runner.GAUSSIAN_SALT
        or int(provenance.get("gaussian_rng_seed", -1)) != gaussian_runner.gaussian_seed(state_id)
        or Path(str(provenance.get("source_trace_path", ""))).resolve() != expected_source_path.resolve()
        or provenance.get("source_trace_file_sha256") != frozen["traces"][211]["sha256"]
        or provenance.get("source_video_trace_sha256") != frozen["traces"][211]["video_trace_sha256"]
        or provenance.get("source_action_noise_sha256") != frozen["traces"][211]["action_noise_sha256"]
    ):
        raise ValueError(f"Gaussian synthetic provenance differs: {state_id}")
    norm_rows = provenance.get("norm_audit")
    if not isinstance(norm_rows, list) or len(norm_rows) != 17 or [int(row["trace_index"]) for row in norm_rows] != list(range(17)):
        raise ValueError(f"Gaussian norm audit differs: {state_id}")
    source_values = [*source_trace["video_latents_pre_step"], source_trace["final_video_latent"]]
    gaussian_values = [*trace["video_latents_pre_step"], trace["final_video_latent"]]
    if len(source_values) != 17 or len(gaussian_values) != 17:
        raise ValueError(f"Gaussian/source trace length differs: {state_id}")
    generator = torch.Generator(device="cpu").manual_seed(
        gaussian_runner.gaussian_seed(state_id)
    )
    recomputed_errors: list[float] = []
    for index, (source_value, gaussian_value, saved_norm) in enumerate(
        zip(source_values, gaussian_values, norm_rows, strict=True)
    ):
        if source_value.shape != gaussian_value.shape or source_value.dtype != gaussian_value.dtype:
            raise ValueError(f"Gaussian/source latent schema differs: {state_id}/{index}")
        expected_gaussian, expected_norm = gaussian_runner.matched_gaussian(
            source_value, generator
        )
        if not torch.equal(expected_gaussian, gaussian_value):
            raise ValueError(f"Gaussian latent is not deterministic norm-matched draw: {state_id}/{index}")
        source_norm = float(torch.linalg.vector_norm(source_value.float()))
        gaussian_norm = float(torch.linalg.vector_norm(gaussian_value.float()))
        relative = abs(gaussian_norm - source_norm) / max(source_norm, gaussian_runner.EPS)
        values = (
            source_norm,
            gaussian_norm,
            relative,
            float(saved_norm.get("source_frobenius_norm", math.nan)),
            float(saved_norm.get("gaussian_frobenius_norm", math.nan)),
            float(saved_norm.get("relative_norm_error", math.nan)),
            float(expected_norm["relative_norm_error"]),
        )
        if not all(math.isfinite(value) for value in values) or relative > 5e-4:
            raise ValueError(f"Gaussian norm audit nonfinite/out of tolerance: {state_id}/{index}")
        if not (
            math.isclose(values[3], source_norm, rel_tol=0, abs_tol=1e-8)
            and math.isclose(values[4], gaussian_norm, rel_tol=0, abs_tol=1e-8)
            and math.isclose(values[5], relative, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(f"Gaussian saved/recomputed norm audit differs: {state_id}/{index}")
        recomputed_errors.append(relative)
    max_norm_error = max(recomputed_errors)

    calls = result.get("calls")
    if not isinstance(calls, list) or len(calls) != 4:
        raise ValueError(f"Gaussian calls incomplete: {state_id}")
    for index, (seed, record) in enumerate(zip(core.BRANCH_SEEDS, calls, strict=True)):
        if int(record.get("recipient_seed", -1)) != seed or record.get("control") != result["control_class"]:
            raise ValueError(f"Gaussian call identity differs: {state_id}/{seed}")
        saved_action = core.action_from_record(record, f"gaussian_{state_id}_{seed}")
        if not np.array_equal(saved_action, actions[index]):
            raise ValueError(f"Gaussian call/action artifact differs: {state_id}/{seed}")
        core.validate_server_audit(
            {core.AUDIT_KEY: record.get("server_audit")},
            label=f"gaussian_{state_id}_{seed}",
            mode="replay",
            noise_seed=seed,
            trace_path=expected_server,
            action_reference_path=args.core_server_trace_root / state_id / f"native_seed_{seed}.pt",
            replay_start=0,
            replay_stop=core.EXPECTED_SOLVER_STEPS,
            expected_source_trace_hash=trace_meta["video_trace_sha256"],
            expected_source_action_hash=trace_meta["source_action_noise_sha256"],
            expected_recipient_action_hash=frozen["traces"][seed]["action_noise_sha256"],
        )

    checkpoint_path = state_dir / "checkpoint.json"
    checkpoint = load_frozen_json(checkpoint_path)
    expected_labels = {f"gaussian_recipient_{seed}" for seed in core.BRANCH_SEEDS}
    if (
        checkpoint.get("status") != "complete"
        or checkpoint.get("runner_sha256") != expected_runner
        or checkpoint.get("input_fingerprint") != result["input_fingerprint"]
        or checkpoint.get("core_result_sha256") != frozen["result_sha256"]
        or checkpoint.get("gaussian_trace_file_sha256") != trace_sha
        or set(checkpoint.get("completed", {})) != expected_labels
        or checkpoint.get("result_sha256") != result_sha
        or checkpoint.get("actions_sha256") != arrays_sha
    ):
        raise ValueError(f"Gaussian frozen checkpoint differs: {state_id}")

    normalized = []
    nearest_recipient = []
    native_flat = native.astype(np.float64).reshape(4, -1)
    actions_flat = actions.astype(np.float64).reshape(4, -1)
    for recipient in range(4):
        scale = np.mean([np.linalg.norm(native_flat[recipient] - native_flat[other]) for other in range(4) if other != recipient])
        normalized.append(float(np.linalg.norm(actions_flat[recipient] - native_flat[recipient]) / max(scale, EPS)))
        nearest_recipient.append(int(np.argmin(np.linalg.norm(native_flat - actions_flat[recipient], axis=1)) == recipient))
    return ({
        "state_id": state_id,
        "state_index": int(result["state"]["state_index"]),
        "task_family": result["state"]["task_family"],
        "input_fingerprint": result["input_fingerprint"],
        "core_result_sha256": frozen["result_sha256"],
        "gaussian_perturbation_normalized": float(np.mean(normalized)),
        "gaussian_recipient_preservation": float(np.mean(nearest_recipient)),
        "max_trace_norm_relative_error": max_norm_error,
    }, {"result_sha256": result_sha, "actions_sha256": arrays_sha, "trace_sha256": trace_sha})


def audit_dose(args: argparse.Namespace, frozen: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    import torch

    state_id = str(frozen["state_id"])
    state_dir = args.dose_root / "states" / state_id
    result_path = state_dir / "result.json"
    result_sha = require_frozen(result_path)
    result = core.load_json(result_path)
    expected_runner = sha256_file(Path(dose_runner.__file__).resolve())
    if (
        result.get("schema") != dose_runner.SCHEMA
        or result.get("status") != "complete"
        or result.get("admission") != "evaluation_followup"
        or result.get("scientific_admission") is not True
        or result_path.parent.name != state_id
        or result.get("state") != frozen["result"]["state"]
        or int(result.get("recipient_seed", -1)) != dose_runner.RECIPIENT_SEED
        or int(result.get("donor_seed", -1)) != dose_runner.DONOR_SEED
        or result.get("alphas") != list(dose_runner.ALPHAS)
        or int(result.get("new_model_call_count", -1)) != 3
        or int(result.get("core_endpoint_call_count_reused", -1)) != 2
        or result.get("recipient_action_noise_fixed_for_new_calls") is not True
        or result.get("runner_sha256") != expected_runner
        or result.get("core_result", {}).get("sha256") != frozen["result_sha256"]
        or Path(str(result.get("core_result", {}).get("path", ""))).resolve() != frozen["result_path"].resolve()
    ):
        raise ValueError(f"dose result/core identity differs: {state_id}")

    arrays_path = state_dir / result["actions_npz"]["relative_path"]
    arrays_sha = require_frozen(arrays_path, result["actions_npz"]["sha256"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != {"alphas", "actions", "native_recipient_action", "native_donor_action"}:
            raise ValueError(f"dose NPZ schema differs: {state_id}")
        alphas = np.asarray(archive["alphas"], dtype=np.float64)
        actions = np.asarray(archive["actions"])
        recipient = np.asarray(archive["native_recipient_action"])
        donor = np.asarray(archive["native_donor_action"])
    ri = core.BRANCH_SEEDS.index(dose_runner.RECIPIENT_SEED)
    di = core.BRANCH_SEEDS.index(dose_runner.DONOR_SEED)
    if (
        not np.array_equal(alphas, ALPHAS)
        or not np.array_equal(recipient, frozen["native"][ri])
        or not np.array_equal(donor, frozen["native"][di])
        or not np.array_equal(actions[0], frozen["replay"][ri, ri])
        or not np.array_equal(actions[0], frozen["native"][ri])
        or not np.array_equal(actions[-1], frozen["replay"][ri, di])
    ):
        raise ValueError(f"dose endpoints/native actions are not exact frozen core cells: {state_id}")

    trace_manifest_path = state_dir / result["trace_manifest"]["relative_path"]
    trace_manifest_sha = require_frozen(trace_manifest_path, result["trace_manifest"]["sha256"])
    trace_manifest = core.load_json(trace_manifest_path)
    rows = trace_manifest.get("traces")
    if (
        trace_manifest.get("schema") != dose_runner.TRACE_MANIFEST_SCHEMA
        or trace_manifest.get("state_id") != state_id
        or int(trace_manifest.get("recipient_seed", -1)) != dose_runner.RECIPIENT_SEED
        or int(trace_manifest.get("donor_seed", -1)) != dose_runner.DONOR_SEED
        or trace_manifest.get("recipient_action_noise_fixed") is not True
        or not isinstance(rows, list)
        or len(rows) != 5
        or [float(row["alpha"]) for row in rows] != list(dose_runner.ALPHAS)
    ):
        raise ValueError(f"dose trace manifest differs: {state_id}")
    trace_by_alpha: dict[float, Mapping[str, Any]] = {}
    recipient_trace = dose_runner.load_trace(frozen["trace_paths"][211])
    donor_trace = dose_runner.load_trace(frozen["trace_paths"][223])
    dose_runner.assert_compatible(recipient_trace, donor_trace)
    recipient_values = [
        *recipient_trace["video_latents_pre_step"], recipient_trace["final_video_latent"]
    ]
    donor_values = [
        *donor_trace["video_latents_pre_step"], donor_trace["final_video_latent"]
    ]
    for row in rows:
        alpha = float(row["alpha"])
        trace_by_alpha[alpha] = row
        if alpha == 0.0:
            expected_path = args.core_result_root / "traces" / state_id / "native_seed_211.pt"
            expected_core = frozen["traces"][211]
        elif alpha == 1.0:
            expected_path = args.core_result_root / "traces" / state_id / "native_seed_223.pt"
            expected_core = frozen["traces"][223]
        else:
            expected_path = args.dose_root / "traces" / state_id / f"alpha_{int(round(alpha*100)):03d}.pt"
            expected_core = None
        if Path(str(row.get("path", ""))).resolve() != expected_path.resolve():
            raise ValueError(f"dose client trace mapping differs: {state_id}/{alpha}")
        trace_sha = require_frozen(expected_path, row.get("trace_file_sha256"))
        trace = dose_runner.load_trace(expected_path)
        if trace.get("video_trace_sha256") != row.get("video_trace_sha256") or trace.get("initial_action_noise_sha256") != row.get("initial_action_noise_sha256"):
            raise ValueError(f"dose trace content differs: {state_id}/{alpha}")
        if expected_core is not None:
            if trace_sha != expected_core["sha256"] or row.get("video_trace_sha256") != expected_core["video_trace_sha256"] or row.get("initial_action_noise_sha256") != expected_core["action_noise_sha256"]:
                raise ValueError(f"dose endpoint trace differs from core: {state_id}/{alpha}")
        else:
            provenance = trace.get("synthetic_provenance", {})
            if (
                provenance.get("kind") != "stepwise_linear_video_latent_interpolation"
                or provenance.get("state_id") != state_id
                or float(provenance.get("alpha", -1)) != alpha
                or int(provenance.get("recipient_seed", -1)) != 211
                or int(provenance.get("donor_seed", -1)) != 223
                or provenance.get("recipient_trace_file_sha256") != frozen["traces"][211]["sha256"]
                or provenance.get("donor_trace_file_sha256") != frozen["traces"][223]["sha256"]
                or provenance.get("recipient_video_trace_sha256") != frozen["traces"][211]["video_trace_sha256"]
                or provenance.get("donor_video_trace_sha256") != frozen["traces"][223]["video_trace_sha256"]
                or trace.get("initial_action_noise_sha256") != frozen["traces"][211]["action_noise_sha256"]
            ):
                raise ValueError(f"dose synthetic provenance differs: {state_id}/{alpha}")
            mixed_values = [*trace["video_latents_pre_step"], trace["final_video_latent"]]
            if len(mixed_values) != 17:
                raise ValueError(f"dose synthetic trace length differs: {state_id}/{alpha}")
            for index, (recipient_value, donor_value, mixed_value) in enumerate(
                zip(recipient_values, donor_values, mixed_values, strict=True)
            ):
                expected = (
                    (1.0 - alpha) * recipient_value.float()
                    + alpha * donor_value.float()
                ).to(dtype=recipient_value.dtype).contiguous()
                if not torch.equal(expected, mixed_value):
                    raise ValueError(
                        f"dose latent is not exact declared interpolation: "
                        f"{state_id}/{alpha}/{index}"
                    )
            if not torch.equal(trace["initial_action_noise"], recipient_trace["initial_action_noise"]):
                raise ValueError(f"dose action noise is not exact recipient noise: {state_id}/{alpha}")
    if len(trace_by_alpha) != 5:
        raise ValueError(f"duplicate dose alpha trace: {state_id}")

    calls = result.get("calls")
    if not isinstance(calls, list) or len(calls) != 3:
        raise ValueError(f"dose calls incomplete: {state_id}")
    for call_index, (alpha, record) in enumerate(zip(dose_runner.INTERIOR_ALPHAS, calls, strict=True), start=1):
        if float(record.get("alpha", -1)) != alpha or int(record.get("recipient_seed", -1)) != 211 or int(record.get("donor_seed", -1)) != 223:
            raise ValueError(f"dose call identity differs: {state_id}/{alpha}")
        saved_action = core.action_from_record(record, f"dose_{state_id}_{alpha}")
        if not np.array_equal(saved_action, actions[call_index]):
            raise ValueError(f"dose call/action artifact differs: {state_id}/{alpha}")
        trace_row = trace_by_alpha[alpha]
        core.validate_server_audit(
            {core.AUDIT_KEY: record.get("server_audit")},
            label=f"dose_{state_id}_{alpha}",
            mode="replay",
            noise_seed=211,
            trace_path=args.dose_server_trace_root / state_id / f"alpha_{int(round(alpha*100)):03d}.pt",
            action_reference_path=args.core_server_trace_root / state_id / "native_seed_211.pt",
            replay_start=0,
            replay_stop=core.EXPECTED_SOLVER_STEPS,
            expected_source_trace_hash=trace_row["video_trace_sha256"],
            expected_source_action_hash=trace_row["initial_action_noise_sha256"],
            expected_recipient_action_hash=frozen["traces"][211]["action_noise_sha256"],
        )

    checkpoint_path = state_dir / "checkpoint.json"
    checkpoint = load_frozen_json(checkpoint_path)
    expected_labels = {f"alpha_{int(round(alpha*100)):03d}" for alpha in dose_runner.INTERIOR_ALPHAS}
    if (
        checkpoint.get("status") != "complete"
        or checkpoint.get("runner_sha256") != expected_runner
        or checkpoint.get("input_fingerprint") != result["state"]["input_fingerprint"]
        or checkpoint.get("core_result_sha256") != frozen["result_sha256"]
        or checkpoint.get("trace_manifest_sha256") != trace_manifest_sha
        or set(checkpoint.get("completed", {})) != expected_labels
        or checkpoint.get("result_sha256") != result_sha
        or checkpoint.get("actions_sha256") != arrays_sha
    ):
        raise ValueError(f"dose frozen checkpoint differs: {state_id}")

    recipient_flat = recipient.astype(np.float64).reshape(-1)
    donor_flat = donor.astype(np.float64).reshape(-1)
    axis = donor_flat - recipient_flat
    denominator = float(np.dot(axis, axis))
    if denominator <= EPS:
        raise ValueError(f"zero dose endpoint separation: {state_id}")
    projections = np.asarray([
        float(np.dot(action.astype(np.float64).reshape(-1) - recipient_flat, axis) / denominator)
        for action in actions
    ])
    if not np.allclose(projections, np.asarray(result["normalized_projection_by_alpha"], dtype=np.float64), rtol=0, atol=1e-12):
        raise ValueError(f"dose saved/recomputed projections differ: {state_id}")
    alpha_rows = [{
        "state_id": state_id,
        "state_index": int(result["state"]["state_index"]),
        "task_family": result["state"]["task_family"],
        "alpha": float(alpha),
        "normalized_projection": float(projection),
    } for alpha, projection in zip(ALPHAS, projections, strict=True)]
    state_row = {
        "state_id": state_id,
        "state_index": int(result["state"]["state_index"]),
        "task_family": result["state"]["task_family"],
        "input_fingerprint": result["state"]["input_fingerprint"],
        "core_result_sha256": frozen["result_sha256"],
        "dose_interior_linear_slope": float(
            np.polyfit(ALPHAS[1:-1], projections[1:-1], 1)[0]
        ),
        "dose_interior_monotonic": int(np.all(np.diff(projections[1:-1]) >= -1e-8)),
        "dose_all_five_linear_slope_descriptive": float(np.polyfit(ALPHAS, projections, 1)[0]),
        "dose_all_five_monotonic_descriptive": int(np.all(np.diff(projections) >= -1e-8)),
        "dose_endpoint_zero_error": float(abs(projections[0])),
        "dose_endpoint_one_projection": float(projections[-1]),
    }
    return state_row, alpha_rows, {"result_sha256": result_sha, "actions_sha256": arrays_sha, "trace_manifest_sha256": trace_manifest_sha}


def audit_run_inventory(
    root: Path,
    schema: str,
    runner_sha: str,
    state_ids: list[str],
    result_hashes: Mapping[str, str],
    manifest_sha: str,
    core_result_root: Path,
    expected_fields: Mapping[str, Any],
) -> str:
    path = root / "run_inventory.json"
    digest = require_frozen(path)
    inventory = core.load_json(path)
    rows = inventory.get("results")
    observed: dict[str, str] = {}
    if isinstance(rows, list):
        for row in rows:
            result_path = Path(str(row.get("path", "")))
            state_id = result_path.parent.name
            if (
                state_id in observed
                or result_path.resolve() != (root / "states" / state_id / "result.json").resolve()
                or row.get("sha256") != require_frozen(result_path)
            ):
                raise ValueError(f"run-inventory result pointer differs: {result_path}")
            observed[state_id] = str(row["sha256"])
    if (
        inventory.get("schema") != schema
        or inventory.get("status") != "complete"
        or int(inventory.get("state_count", -1)) != len(state_ids)
        or inventory.get("state_ids") != state_ids
        or inventory.get("runner_sha256") != runner_sha
        or inventory.get("manifest_file_sha256") != manifest_sha
        or not str(inventory.get("core_result_root", ""))
        or Path(str(inventory["core_result_root"])).resolve() != core_result_root.resolve()
        or len(observed) != len(state_ids)
        or observed != dict(result_hashes)
        or any(inventory.get(key) != value for key, value in expected_fields.items())
    ):
        raise ValueError(f"run inventory differs: {path}")
    return digest


def atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    core.freeze_with_sidecar(path)


def write_json(path: Path, value: object) -> None:
    atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)
    core.freeze_with_sidecar(path)


def save_plot(output: Path, alpha_summary: Sequence[Mapping[str, Any]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    mean = np.asarray([row["mean"] for row in alpha_summary])
    low = np.asarray([row["state_bootstrap_95_ci"][0] for row in alpha_summary])
    high = np.asarray([row["state_bootstrap_95_ci"][1] for row in alpha_summary])
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.errorbar(ALPHAS, mean, yerr=np.vstack((mean - low, high - mean)), fmt="o-", color="#244A68", ecolor="#5B8DB8", capsize=3)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Future-latent interpolation alpha")
    ax.set_ylabel("Normalized action projection")
    ax.grid(axis="y", color="#D6E2EA", linewidth=0.6)
    fig.tight_layout()
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    if png.exists() or pdf.exists():
        raise FileExistsError(output)
    fig.savefig(png, dpi=220, metadata={"Software": "summarize_dreamzero_controls.py"})
    fig.savefig(pdf, metadata={"Creator": "summarize_dreamzero_controls.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    core.freeze_with_sidecar(png)
    core.freeze_with_sidecar(pdf)
    return [png, pdf]


def main() -> None:
    args = parse_args()
    if (
        args.expected_state_count != 30
        or args.bootstrap_repetitions < 100
        or args.permutation_repetitions < 100
    ):
        raise ValueError("confirmatory analysis requires 30 states and >=100 resampling draws")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"analysis output root is not empty: {args.output_root}")
    manifest_sha = require_frozen(args.manifest)
    manifest = core.load_json(args.manifest)
    if (
        manifest_sha != EXPECTED_MANIFEST_SHA256
        or manifest.get("manifest_id") != EXPECTED_MANIFEST_ID
        or manifest.get("schema") != core.EXPECTED_MANIFEST_SCHEMA
        or manifest.get("scope", {}).get("outcome_blind") is not True
        or manifest.get("scope", {}).get("model") != "DreamZero-DROID"
        or core.manifest_body_hash(manifest) != manifest.get("manifest_body_sha256")
    ):
        raise ValueError("manifest is not the canonical frozen outcome-blind cohort")
    states = manifest.get("states")
    if not isinstance(states, list) or len(states) != args.expected_state_count:
        raise ValueError("manifest state count differs")
    state_ids = [str(state["state_id"]) for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("manifest contains duplicate state IDs")
    family_counts = {
        family: sum(str(state["task_family"]) == family for state in states)
        for family in EXPECTED_FAMILIES
    }
    if set(str(state["task_family"]) for state in states) != set(EXPECTED_FAMILIES) or set(family_counts.values()) != {3}:
        raise ValueError(f"manifest does not contain the frozen 3x10 family quota: {family_counts}")
    debug_gate = audit_off_record_debug(args.off_record_debug_root)
    expected_sets = set(state_ids)
    for name, root in (("core", args.core_result_root), ("Gaussian", args.gaussian_root), ("dose", args.dose_root)):
        observed = [path.parent.name for path in sorted((root / "states").glob("*/result.json"))]
        if len(observed) != args.expected_state_count or len(set(observed)) != len(observed) or set(observed) != expected_sets:
            raise ValueError(f"{name} result cohort differs from exact frozen manifest")

    gaussian_rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    gaussian_hashes: dict[str, str] = {}
    dose_hashes: dict[str, str] = {}
    for state in states:
        frozen = audit_core_state(args, state, manifest_sha)
        gaussian, gaussian_artifacts = audit_gaussian(args, frozen)
        dose, dose_alpha, dose_artifacts = audit_dose(args, frozen)
        if gaussian["input_fingerprint"] != dose["input_fingerprint"] or gaussian["core_result_sha256"] != dose["core_result_sha256"]:
            raise ValueError(f"control products disagree on frozen core identity: {frozen['state_id']}")
        gaussian_rows.append(gaussian)
        dose_rows.append(dose)
        alpha_rows.extend(dose_alpha)
        gaussian_hashes[frozen["state_id"]] = gaussian_artifacts["result_sha256"]
        dose_hashes[frozen["state_id"]] = dose_artifacts["result_sha256"]
        source_inventory.append({
            "state_id": frozen["state_id"],
            "core_result_sha256": frozen["result_sha256"],
            "core_actions_sha256": frozen["arrays_sha256"],
            "gaussian": gaussian_artifacts,
            "dose": dose_artifacts,
        })
    gaussian_runner_sha = sha256_file(Path(gaussian_runner.__file__).resolve())
    dose_runner_sha = sha256_file(Path(dose_runner.__file__).resolve())
    gaussian_inventory_sha = audit_run_inventory(
        args.gaussian_root,
        "dreamzero-gaussian-control-run-inventory-v1",
        gaussian_runner_sha,
        state_ids,
        gaussian_hashes,
        manifest_sha,
        args.core_result_root,
        {
            "admission": "evaluation_control",
            "call_count": 4 * args.expected_state_count,
            "core_client_trace_root": str((args.core_result_root / "traces").resolve()),
            "core_server_trace_root": str(args.core_server_trace_root),
            "client_trace_root": str((args.gaussian_root / "traces").resolve()),
            "server_trace_root": str(args.gaussian_server_trace_root),
        },
    )
    dose_inventory_sha = audit_run_inventory(
        args.dose_root,
        "dreamzero-future-latent-dose-run-inventory-v1",
        dose_runner_sha,
        state_ids,
        dose_hashes,
        manifest_sha,
        args.core_result_root,
        {
            "recipient_seed": 211,
            "donor_seed": 223,
            "alphas": list(dose_runner.ALPHAS),
            "new_model_call_count": 3 * args.expected_state_count,
            "core_client_trace_root": str((args.core_result_root / "traces").resolve()),
            "core_server_trace_root": str(args.core_server_trace_root),
            "client_trace_root": str((args.dose_root / "traces").resolve()),
            "server_trace_root": str(args.dose_server_trace_root),
        },
    )

    estimates = {
        "gaussian_perturbation_normalized": estimate([row["gaussian_perturbation_normalized"] for row in gaussian_rows], args.bootstrap_repetitions, rng_for(args.seed, "gaussian_perturbation")),
        "gaussian_recipient_preservation": estimate([row["gaussian_recipient_preservation"] for row in gaussian_rows], args.bootstrap_repetitions, rng_for(args.seed, "gaussian_preservation")),
        "dose_interior_linear_slope": estimate([row["dose_interior_linear_slope"] for row in dose_rows], args.bootstrap_repetitions, rng_for(args.seed, "dose_interior_slope")),
        "dose_interior_monotonic_fraction": estimate([row["dose_interior_monotonic"] for row in dose_rows], args.bootstrap_repetitions, rng_for(args.seed, "dose_interior_monotonic")),
    }
    dose_sign_p, dose_sign_null = sign_flip_pvalue(
        [row["dose_interior_linear_slope"] for row in dose_rows],
        args.permutation_repetitions,
        rng_for(args.seed, "dose_interior_slope_sign_flip"),
    )
    alpha_summary = []
    for alpha in ALPHAS:
        values = [row["normalized_projection"] for row in alpha_rows if row["alpha"] == float(alpha)]
        alpha_summary.append({"alpha": float(alpha), **estimate(values, args.bootstrap_repetitions, rng_for(args.seed, f"dose_alpha:{alpha}"))})
    summary = {
        "schema": "dreamzero-control-analysis-v2",
        "audit_status": "passed",
        "state_is_independent_unit": True,
        "state_count": args.expected_state_count,
        "task_family_count": len({row["task_family"] for row in gaussian_rows}),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "permutation_repetitions": args.permutation_repetitions,
        "seed": args.seed,
        "estimates": estimates,
        "dose_interior_slope_sign_flip": {
            "method": "one-sided Monte Carlo state-level sign flip",
            "p_value": dose_sign_p,
            "null_mean": dose_sign_null,
            "endpoints_excluded": True,
        },
        "dose_by_alpha": alpha_summary,
        "dose_by_alpha_is_descriptive": True,
        "dose_reused_endpoint_alphas": [0.0, 1.0],
        "maximum_gaussian_trace_norm_relative_error": max(row["max_trace_norm_relative_error"] for row in gaussian_rows),
        "exact_core_endpoint_gates": {"alpha_0_equals_core_self_replay": True, "alpha_1_equals_core_211_to_223_replay": True},
        "patched_server_mode_off_record_gate": debug_gate,
        "canonical_family_counts": family_counts,
        "manifest_file_sha256": manifest_sha,
        "core_runner_sha256": sha256_file(Path(core.__file__).resolve()),
        "gaussian_runner_sha256": gaussian_runner_sha,
        "dose_runner_sha256": dose_runner_sha,
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "gaussian_run_inventory_sha256": gaussian_inventory_sha,
        "dose_run_inventory_sha256": dose_inventory_sha,
        "core_result_root": str(args.core_result_root.resolve()),
        "gaussian_root": str(args.gaussian_root.resolve()),
        "dose_root": str(args.dose_root.resolve()),
        "trace_root_mapping": {
            "core_client": str((args.core_result_root / "traces").resolve()),
            "core_server": str(args.core_server_trace_root),
            "gaussian_client": str((args.gaussian_root / "traces").resolve()),
            "gaussian_server": str(args.gaussian_server_trace_root),
            "dose_client": str((args.dose_root / "traces").resolve()),
            "dose_server": str(args.dose_server_trace_root),
            "contract": "identical state-relative trace path; content SHA revalidated from client artifact and server audit",
        },
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, rows in (
        ("gaussian_state_metrics.csv", gaussian_rows),
        ("dose_state_metrics.csv", dose_rows),
        ("dose_alpha_state_metrics.csv", alpha_rows),
        ("dose_alpha_summary.csv", alpha_summary),
        ("source_artifact_inventory.csv", [
            {
                "state_id": row["state_id"],
                "core_result_sha256": row["core_result_sha256"],
                "core_actions_sha256": row["core_actions_sha256"],
                "gaussian_result_sha256": row["gaussian"]["result_sha256"],
                "dose_result_sha256": row["dose"]["result_sha256"],
            }
            for row in source_inventory
        ]),
    ):
        path = args.output_root / filename
        write_csv(path, rows)
        written.append(path)
    summary_path = args.output_root / "summary.json"
    write_json(summary_path, summary)
    written.append(summary_path)
    plot_paths = save_plot(args.output_root / "dose_response", alpha_summary)
    written.extend(plot_paths)
    lines = [
        "# DreamZero controls",
        "",
        f"Audit: passed; exact frozen cohort of {args.expected_state_count} states.",
        "",
        "| Estimand | Mean | State-bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for name, item in estimates.items():
        low, high = item["state_bootstrap_95_ci"]
        lines.append(f"| {name.replace('_', ' ')} | {item['mean']:.4f} | [{low:.4f}, {high:.4f}] |")
    markdown_path = args.output_root / "results.md"
    atomic(markdown_path, ("\n".join(lines) + "\n").encode())
    written.append(markdown_path)
    artifact_inventory = {
        "schema": "dreamzero-control-analysis-artifact-inventory-v1",
        "analyzer_sha256": summary["analyzer_sha256"],
        "manifest_file_sha256": manifest_sha,
        "sources": source_inventory,
        "artifacts": [
            {"relative_path": str(path.relative_to(args.output_root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(written)
        ],
    }
    inventory_path = args.output_root / "artifact_inventory.json"
    write_json(inventory_path, artifact_inventory)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
