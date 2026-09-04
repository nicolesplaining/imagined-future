#!/usr/bin/env python3
"""Run a provenance-clean DreamZero video-latent dose response.

For every frozen state, the recipient is seed 211 and the donor is seed 223.
Alpha 0 and 1 reuse the exact frozen core-grid actions.  Only alpha 0.25, 0.5,
and 0.75 require new calls.  At every solver step the synthetic video state is
``(1-alpha) * recipient_trace + alpha * donor_trace``; the action-noise trace
is always the seed-211 recipient reference.  Core artifacts are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import run_dreamzero_future_transplants as core


SCHEMA = "dreamzero-future-latent-dose-state-v1"
TRACE_MANIFEST_SCHEMA = "dreamzero-future-latent-dose-trace-manifest-v1"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
INTERIOR_ALPHAS = (0.25, 0.5, 0.75)
RECIPIENT_SEED = 211
DONOR_SEED = 223


def require_frozen(path: Path) -> str:
    digest = core.verify_sha_sidecar(path)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise RuntimeError(f"completed artifact is not read-only: {path}")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--core-result-root", type=Path, required=True)
    parser.add_argument("--core-client-trace-root", type=Path, required=True)
    parser.add_argument("--core-server-trace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--client-trace-root", type=Path, required=True)
    parser.add_argument("--server-trace-root", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--response-timeout", type=float, default=3600.0)
    parser.add_argument("--state-id", action="append", default=[])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def tensor_sha256(value: Any) -> str:
    import torch

    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def video_trace_sha256(trace: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    values = [*trace["video_latents_pre_step"], trace["final_video_latent"]]
    for index, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise TypeError("trace contains non-tensor video state")
        value = value.detach().cpu().contiguous()
        digest.update(index.to_bytes(4, "little", signed=False))
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_trace(path: Path) -> dict[str, Any]:
    import torch

    core.verify_sha_sidecar(path)
    trace = torch.load(path, map_location="cpu", weights_only=True)
    if trace.get("format_version") != 3:
        raise ValueError(f"not a format-v3 trace: {path}")
    if video_trace_sha256(trace) != trace.get("video_trace_sha256"):
        raise ValueError(f"video-trace hash mismatch: {path}")
    if tensor_sha256(trace["initial_action_noise"]) != trace["initial_action_noise_sha256"]:
        raise ValueError(f"action-noise hash mismatch: {path}")
    return trace


def assert_compatible(recipient: Mapping[str, Any], donor: Mapping[str, Any]) -> None:
    import torch

    if recipient["current_start_frame"] != donor["current_start_frame"]:
        raise ValueError("source traces have different start frames")
    for key in ("video_timesteps", "video_sigmas", "action_timesteps", "action_sigmas"):
        if not torch.equal(recipient[key], donor[key]):
            raise ValueError(f"source traces have different {key}")
    left = [*recipient["video_latents_pre_step"], recipient["final_video_latent"]]
    right = [*donor["video_latents_pre_step"], donor["final_video_latent"]]
    if len(left) != core.EXPECTED_SOLVER_STEPS + 1 or len(right) != len(left):
        raise ValueError("source traces do not cover the complete solver trajectory")
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"source latent schema differs at trace index {index}")


def interpolate_trace(
    recipient_path: Path,
    donor_path: Path,
    output_path: Path,
    state_id: str,
    alpha: float,
) -> dict[str, Any]:
    import torch

    if alpha not in INTERIOR_ALPHAS:
        raise ValueError("only interior alpha traces are synthesized")
    recipient = load_trace(recipient_path)
    donor = load_trace(donor_path)
    assert_compatible(recipient, donor)
    recipient_values = [*recipient["video_latents_pre_step"], recipient["final_video_latent"]]
    donor_values = [*donor["video_latents_pre_step"], donor["final_video_latent"]]
    mixed = [
        ((1.0 - alpha) * left.float() + alpha * right.float()).to(dtype=left.dtype).contiguous()
        for left, right in zip(recipient_values, donor_values, strict=True)
    ]
    trace = {
        "format_version": 3,
        "current_start_frame": recipient["current_start_frame"],
        "noise_seed": RECIPIENT_SEED,
        "video_timesteps": recipient["video_timesteps"].clone(),
        "video_sigmas": recipient["video_sigmas"].clone(),
        "action_timesteps": recipient["action_timesteps"].clone(),
        "action_sigmas": recipient["action_sigmas"].clone(),
        "initial_action_noise": recipient["initial_action_noise"].clone(),
        "initial_action_noise_sha256": recipient["initial_action_noise_sha256"],
        "video_latents_pre_step": mixed[:-1],
        "final_video_latent": mixed[-1],
        "synthetic_provenance": {
            "kind": "stepwise_linear_video_latent_interpolation",
            "state_id": state_id,
            "alpha": alpha,
            "recipient_seed": RECIPIENT_SEED,
            "donor_seed": DONOR_SEED,
            "recipient_trace_path": str(recipient_path),
            "donor_trace_path": str(donor_path),
            "recipient_trace_file_sha256": core.sha256_file(recipient_path),
            "donor_trace_file_sha256": core.sha256_file(donor_path),
            "recipient_video_trace_sha256": recipient["video_trace_sha256"],
            "donor_video_trace_sha256": donor["video_trace_sha256"],
            "arithmetic": "cast((1-alpha)*recipient.float32 + alpha*donor.float32, source dtype)",
        },
    }
    trace["video_trace_sha256"] = video_trace_sha256(trace)
    if output_path.exists():
        trace_file_sha256 = require_frozen(output_path)
        existing = load_trace(output_path)
        if set(existing) != set(trace):
            raise ValueError(f"existing dose trace schema differs: {output_path}")
        scalar_keys = (
            "format_version",
            "current_start_frame",
            "noise_seed",
            "initial_action_noise_sha256",
            "video_trace_sha256",
            "synthetic_provenance",
        )
        if any(existing.get(key) != trace.get(key) for key in scalar_keys):
            raise ValueError(f"existing dose trace provenance differs: {output_path}")
        tensor_keys = (
            "video_timesteps",
            "video_sigmas",
            "action_timesteps",
            "action_sigmas",
            "initial_action_noise",
            "final_video_latent",
        )
        if any(not torch.equal(existing[key], trace[key]) for key in tensor_keys):
            raise ValueError(f"existing dose trace tensor differs: {output_path}")
        if len(existing["video_latents_pre_step"]) != len(trace["video_latents_pre_step"]) or any(
            not torch.equal(left, right)
            for left, right in zip(
                existing["video_latents_pre_step"], trace["video_latents_pre_step"], strict=True
            )
        ):
            raise ValueError(f"existing dose step trace differs: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        torch.save(trace, temporary)
        os.chmod(temporary, 0o444)
        os.replace(temporary, output_path)
        trace_file_sha256 = core.freeze_with_sidecar(output_path)["sha256"]
    return {
        "alpha": alpha,
        "trace_file_sha256": trace_file_sha256,
        "trace_size_bytes": output_path.stat().st_size,
        "video_trace_sha256": trace["video_trace_sha256"],
        "initial_action_noise_sha256": trace["initial_action_noise_sha256"],
        "path": str(output_path),
        "state_id": state_id,
        "recipient_seed": RECIPIENT_SEED,
        "donor_seed": DONOR_SEED,
        "recipient_trace_file_sha256": core.sha256_file(recipient_path),
        "donor_trace_file_sha256": core.sha256_file(donor_path),
        "recipient_video_trace_sha256": recipient["video_trace_sha256"],
        "donor_video_trace_sha256": donor["video_trace_sha256"],
        "reused_core_endpoint": False,
        "synthetic": True,
    }


def freeze_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    core.atomic_write_json(path, value, mode=0o444)
    core.freeze_with_sidecar(path)


def freeze_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    core.atomic_write_npz(path, arrays)
    core.freeze_with_sidecar(path)


def load_core(
    args: argparse.Namespace, inputs: core.FrozenInputs
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]], dict[str, Any]]:
    state_dir = args.core_result_root / "states" / inputs.state_id
    result_path = state_dir / "result.json"
    core.verify_sha_sidecar(result_path)
    result = core.load_json(result_path)
    if (
        result.get("schema") != core.SCHEMA
        or result.get("admission") != "evaluation"
        or result.get("status") != "complete"
        or result.get("scientific_admission") is not True
    ):
        raise ValueError(f"core result is not admitted and complete: {inputs.state_id}")
    expected_state = {
        "state_id": inputs.state_id,
        "state_index": inputs.state_index,
        "episode_index": inputs.episode_index,
        "frame_index": inputs.frame_index,
        "task_family": inputs.task_family,
        "prompt": inputs.prompt,
        "input_fingerprint": inputs.audit["input_fingerprint"],
    }
    if result.get("state") != expected_state or state_dir.name != inputs.state_id:
        raise ValueError(f"core state/input identity mismatch: {inputs.state_id}")
    if result.get("provenance", {}).get("manifest_file_sha256") != args.expected_manifest_sha256:
        raise ValueError(f"core result manifest mismatch: {inputs.state_id}")
    arrays_path = state_dir / result["artifacts"]["actions_npz"]["relative_path"]
    if core.verify_sha_sidecar(arrays_path) != result["artifacts"]["actions_npz"]["sha256"]:
        raise ValueError(f"core actions hash mismatch: {inputs.state_id}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        seeds = np.asarray(archive["branch_seeds"], dtype=np.int64)
        native = np.asarray(archive["native_actions"])
        replay = np.asarray(archive["replay_actions"])
    if tuple(seeds.tolist()) != tuple(core.BRANCH_SEEDS):
        raise ValueError(f"core branch order mismatch: {inputs.state_id}")
    inventory_path = state_dir / result["artifacts"]["trace_inventory"]["relative_path"]
    if core.verify_sha_sidecar(inventory_path) != result["artifacts"]["trace_inventory"]["sha256"]:
        raise ValueError(f"core trace inventory hash mismatch: {inputs.state_id}")
    inventory = core.load_json(inventory_path)
    if (
        inventory.get("schema") != core.INVENTORY_SCHEMA
        or inventory.get("state_id") != inputs.state_id
        or int(inventory.get("trace_count", -1)) != 4
        or not isinstance(inventory.get("traces"), list)
        or len(inventory["traces"]) != 4
    ):
        raise ValueError(f"core trace inventory identity/schema mismatch: {inputs.state_id}")
    traces = {int(row["branch_seed"]): row for row in inventory["traces"]}
    if len(traces) != 4 or set(traces) != set(core.BRANCH_SEEDS):
        raise ValueError(f"core trace seed mismatch: {inputs.state_id}")
    grid = {
        (int(row["recipient_seed"]), int(row["future_source_seed"])): row
        for row in result.get("grid", [])
    }
    for source_seed in (RECIPIENT_SEED, DONOR_SEED):
        pair = (RECIPIENT_SEED, source_seed)
        if pair not in grid:
            raise ValueError(f"core endpoint grid row missing {pair}: {inputs.state_id}")
        row = grid[pair]
        source_index = core.BRANCH_SEEDS.index(source_seed)
        action = np.ascontiguousarray(replay[core.BRANCH_SEEDS.index(RECIPIENT_SEED), source_index])
        if row.get("sha256") != core.array_sha256(action):
            raise ValueError(f"core endpoint action hash mismatch {pair}: {inputs.state_id}")
        core.validate_server_audit(
            {core.AUDIT_KEY: row.get("server_audit")},
            label=f"core_replay_recipient_{RECIPIENT_SEED}_source_{source_seed}",
            mode="replay",
            noise_seed=RECIPIENT_SEED,
            trace_path=(
                args.core_server_trace_root / inputs.state_id / f"native_seed_{source_seed}.pt"
            ),
            action_reference_path=(
                args.core_server_trace_root
                / inputs.state_id
                / f"native_seed_{RECIPIENT_SEED}.pt"
            ),
            replay_start=0,
            replay_stop=core.EXPECTED_SOLVER_STEPS,
            expected_source_trace_hash=traces[source_seed]["video_trace_sha256"],
            expected_source_action_hash=traces[source_seed]["action_noise_sha256"],
            expected_recipient_action_hash=traces[RECIPIENT_SEED]["action_noise_sha256"],
        )
    return native, replay, traces, result


def infer_alpha(
    args: argparse.Namespace,
    client: core.DreamZeroClient,
    inputs: core.FrozenInputs,
    alpha: float,
    trace_server: Path,
    trace_record: Mapping[str, Any],
    recipient_action_hash: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    action_reference = (
        args.core_server_trace_root / inputs.state_id / f"native_seed_{RECIPIENT_SEED}.pt"
    )
    client.reset()
    request = dict(inputs.request)
    request["session_id"] = f"dose:{inputs.state_id}:alpha_{alpha:.2f}"
    request[core.CONTROL_KEY] = core.request_control(
        mode="replay",
        noise_seed=RECIPIENT_SEED,
        trace_path=trace_server,
        action_noise_reference_path=action_reference,
        replay_start=0,
        replay_stop=core.EXPECTED_SOLVER_STEPS,
    )
    response = client.infer(request)
    action = core.action_from_response(response, f"alpha_{alpha:.2f}")
    audit = core.validate_server_audit(
        response,
        label=f"alpha_{alpha:.2f}",
        mode="replay",
        noise_seed=RECIPIENT_SEED,
        trace_path=trace_server,
        action_reference_path=action_reference,
        replay_start=0,
        replay_stop=core.EXPECTED_SOLVER_STEPS,
        expected_source_trace_hash=str(trace_record["video_trace_sha256"]),
        expected_source_action_hash=str(trace_record["initial_action_noise_sha256"]),
        expected_recipient_action_hash=recipient_action_hash,
    )
    return action, audit


def run_state(
    args: argparse.Namespace,
    client: core.DreamZeroClient,
    inputs: core.FrozenInputs,
) -> Path:
    state_dir = args.output_root / "states" / inputs.state_id
    result_path = state_dir / "result.json"
    native, replay, core_traces, core_result = load_core(args, inputs)
    recipient_index = core.BRANCH_SEEDS.index(RECIPIENT_SEED)
    donor_index = core.BRANCH_SEEDS.index(DONOR_SEED)
    recipient_client = (
        args.core_client_trace_root / inputs.state_id / f"native_seed_{RECIPIENT_SEED}.pt"
    )
    donor_client = args.core_client_trace_root / inputs.state_id / f"native_seed_{DONOR_SEED}.pt"
    if core.verify_sha_sidecar(recipient_client) != core_traces[RECIPIENT_SEED]["sha256"]:
        raise ValueError(f"recipient trace file mismatch: {inputs.state_id}")
    if core.verify_sha_sidecar(donor_client) != core_traces[DONOR_SEED]["sha256"]:
        raise ValueError(f"donor trace file mismatch: {inputs.state_id}")
    recipient_trace = load_trace(recipient_client)
    donor_trace = load_trace(donor_client)
    assert_compatible(recipient_trace, donor_trace)

    trace_manifest_rows: list[dict[str, Any]] = [
        {
            "alpha": 0.0,
            "path": str(recipient_client),
            "trace_file_sha256": core.sha256_file(recipient_client),
            "video_trace_sha256": recipient_trace["video_trace_sha256"],
            "initial_action_noise_sha256": recipient_trace["initial_action_noise_sha256"],
            "reused_core_endpoint": True,
            "synthetic": False,
        }
    ]
    interior: dict[float, dict[str, Any]] = {}
    for alpha in INTERIOR_ALPHAS:
        name = f"alpha_{int(round(alpha * 100)):03d}.pt"
        client_path = args.client_trace_root / inputs.state_id / name
        row = interpolate_trace(recipient_client, donor_client, client_path, inputs.state_id, alpha)
        interior[alpha] = row
        trace_manifest_rows.append(row)
    trace_manifest_rows.append(
        {
            "alpha": 1.0,
            "path": str(donor_client),
            "trace_file_sha256": core.sha256_file(donor_client),
            "video_trace_sha256": donor_trace["video_trace_sha256"],
            "initial_action_noise_sha256": donor_trace["initial_action_noise_sha256"],
            "reused_core_endpoint": True,
            "synthetic": False,
        }
    )
    trace_manifest_rows.sort(key=lambda row: float(row["alpha"]))
    trace_manifest = {
        "schema": TRACE_MANIFEST_SCHEMA,
        "state_id": inputs.state_id,
        "recipient_seed": RECIPIENT_SEED,
        "donor_seed": DONOR_SEED,
        "arithmetic_space": "matched-noise video latent at each of 16 solver steps",
        "recipient_action_noise_fixed": True,
        "traces": trace_manifest_rows,
    }
    trace_manifest_path = state_dir / "trace_manifest.json"
    if trace_manifest_path.exists():
        core.verify_sha_sidecar(trace_manifest_path)
        if core.load_json(trace_manifest_path) != trace_manifest:
            raise ValueError(f"existing dose trace manifest differs: {inputs.state_id}")
    else:
        freeze_json(trace_manifest_path, trace_manifest)

    checkpoint_path = state_dir / "checkpoint.json"
    header = {
        "schema": "dreamzero-future-latent-dose-checkpoint-v1",
        "status": "partial",
        "state_id": inputs.state_id,
        "input_fingerprint": inputs.audit["input_fingerprint"],
        "runner_sha256": core.sha256_file(Path(__file__).resolve()),
        "core_result_sha256": core.sha256_file(
            args.core_result_root / "states" / inputs.state_id / "result.json"
        ),
        "trace_manifest_sha256": core.sha256_file(trace_manifest_path),
        "recipient_seed": RECIPIENT_SEED,
        "donor_seed": DONOR_SEED,
        "alphas": list(ALPHAS),
        "completed": {},
    }
    if checkpoint_path.exists():
        checkpoint = core.load_json(checkpoint_path)
        for key, value in header.items():
            if key not in {"status", "completed"} and checkpoint.get(key) != value:
                raise ValueError(f"dose checkpoint mismatch at {key}: {inputs.state_id}")
        completed = checkpoint.get("completed")
        if not isinstance(completed, dict):
            raise ValueError("dose checkpoint lacks completed map")
    else:
        checkpoint = header
        completed = checkpoint["completed"]
        core.atomic_write_json(checkpoint_path, checkpoint)

    actions: dict[float, np.ndarray] = {
        0.0: np.asarray(replay[recipient_index, recipient_index]),
        1.0: np.asarray(replay[recipient_index, donor_index]),
    }
    if not np.array_equal(actions[0.0], native[recipient_index]):
        raise RuntimeError(f"core alpha=0 endpoint is not exact self replay: {inputs.state_id}")
    calls: dict[str, Any] = {}
    recipient_action_hash = core_traces[RECIPIENT_SEED]["action_noise_sha256"]
    for alpha in INTERIOR_ALPHAS:
        label = f"alpha_{int(round(alpha * 100)):03d}"
        if label in completed:
            record = completed[label]
            if (
                float(record.get("alpha", -1)) != alpha
                or int(record.get("recipient_seed", -1)) != RECIPIENT_SEED
                or int(record.get("donor_seed", -1)) != DONOR_SEED
            ):
                raise ValueError(f"{label}: saved call identity differs")
            name = f"alpha_{int(round(alpha * 100)):03d}.pt"
            core.validate_server_audit(
                {core.AUDIT_KEY: record.get("server_audit")},
                label=label,
                mode="replay",
                noise_seed=RECIPIENT_SEED,
                trace_path=args.server_trace_root / inputs.state_id / name,
                action_reference_path=(
                    args.core_server_trace_root
                    / inputs.state_id
                    / f"native_seed_{RECIPIENT_SEED}.pt"
                ),
                replay_start=0,
                replay_stop=core.EXPECTED_SOLVER_STEPS,
                expected_source_trace_hash=interior[alpha]["video_trace_sha256"],
                expected_source_action_hash=interior[alpha]["initial_action_noise_sha256"],
                expected_recipient_action_hash=recipient_action_hash,
            )
            action = core.action_from_record(record, label)
            actions[alpha] = action
            calls[label] = record
            continue
        name = f"alpha_{int(round(alpha * 100)):03d}.pt"
        server_path = args.server_trace_root / inputs.state_id / name
        action, audit = infer_alpha(
            args,
            client,
            inputs,
            alpha,
            server_path,
            interior[alpha],
            recipient_action_hash,
        )
        record = {
            **core.action_record(action, audit),
            "alpha": alpha,
            "recipient_seed": RECIPIENT_SEED,
            "donor_seed": DONOR_SEED,
        }
        completed[label] = record
        checkpoint["completed"] = completed
        core.atomic_write_json(checkpoint_path, checkpoint)
        actions[alpha] = action
        calls[label] = record

    expected_labels = {
        f"alpha_{int(round(alpha * 100)):03d}" for alpha in INTERIOR_ALPHAS
    }
    if set(completed) != expected_labels:
        raise ValueError(
            f"dose completed labels differ: missing={sorted(expected_labels-set(completed))}, "
            f"extra={sorted(set(completed)-expected_labels)}"
        )
    expected_shape = native[recipient_index].shape
    expected_dtype = native.dtype
    for alpha, action in actions.items():
        if action.shape != expected_shape or action.dtype != expected_dtype:
            raise ValueError(
                f"dose action schema differs at alpha={alpha}: "
                f"{action.shape}/{action.dtype} != {expected_shape}/{expected_dtype}"
            )
    action_array = np.stack([actions[alpha] for alpha in ALPHAS])
    arrays_path = state_dir / "actions.npz"
    expected_arrays = {
        "alphas": np.asarray(ALPHAS, dtype=np.float64),
        "actions": action_array,
        "native_recipient_action": native[recipient_index],
        "native_donor_action": native[donor_index],
    }
    if arrays_path.exists():
        core.verify_sha_sidecar(arrays_path)
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_arrays) or any(
                not np.array_equal(archive[key], value) for key, value in expected_arrays.items()
            ):
                raise ValueError(f"existing dose actions differ: {inputs.state_id}")
    else:
        freeze_npz(arrays_path, expected_arrays)
    axis = native[donor_index].astype(np.float64).reshape(-1) - native[recipient_index].astype(np.float64).reshape(-1)
    denominator = float(np.dot(axis, axis))
    projections = []
    for action in action_array:
        displacement = action.astype(np.float64).reshape(-1) - native[recipient_index].astype(np.float64).reshape(-1)
        projections.append(float(np.dot(displacement, axis) / denominator) if denominator > 0 else None)
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "admission": "evaluation_followup",
        "scientific_admission": True,
        "state": core_result["state"],
        "recipient_seed": RECIPIENT_SEED,
        "donor_seed": DONOR_SEED,
        "alphas": list(ALPHAS),
        "new_model_call_count": 3,
        "core_endpoint_call_count_reused": 2,
        "recipient_action_noise_fixed_for_new_calls": True,
        "action_coordinates_written_by_client": False,
        "normalized_projection_by_alpha": projections,
        "trace_manifest": {
            "relative_path": trace_manifest_path.name,
            "sha256": core.sha256_file(trace_manifest_path),
        },
        "actions_npz": {
            "relative_path": arrays_path.name,
            "sha256": core.sha256_file(arrays_path),
            "size_bytes": arrays_path.stat().st_size,
        },
        "core_result": {
            "path": str(args.core_result_root / "states" / inputs.state_id / "result.json"),
            "sha256": header["core_result_sha256"],
        },
        "runner_sha256": header["runner_sha256"],
        "calls": [calls[f"alpha_{int(round(alpha * 100)):03d}"] for alpha in INTERIOR_ALPHAS],
    }
    if result_path.exists():
        require_frozen(result_path)
        require_frozen(arrays_path)
        require_frozen(trace_manifest_path)
        require_frozen(checkpoint_path)
        if core.load_json(result_path) != result:
            raise ValueError(f"existing dose result identity differs: {inputs.state_id}")
        checkpoint_final = core.load_json(checkpoint_path)
        if (
            checkpoint_final.get("status") != "complete"
            or checkpoint_final.get("result_sha256") != core.sha256_file(result_path)
            or checkpoint_final.get("actions_sha256") != core.sha256_file(arrays_path)
        ):
            raise ValueError(f"existing dose checkpoint differs: {inputs.state_id}")
        return result_path
    freeze_json(result_path, result)
    checkpoint["status"] = "complete"
    checkpoint["result_sha256"] = core.sha256_file(result_path)
    checkpoint["actions_sha256"] = core.sha256_file(arrays_path)
    core.atomic_write_json(checkpoint_path, checkpoint, mode=0o444)
    core.freeze_with_sidecar(checkpoint_path)
    return result_path


def main() -> None:
    args = parse_args()
    if not args.server_trace_root.is_absolute() or not args.core_server_trace_root.is_absolute():
        raise ValueError("server trace roots must be absolute")
    manifest, resources, data_root, frozen = core.validate_manifest_and_receipt(args)
    receipt = core.load_json(data_root / "download_receipt.json")
    receipt_by_id = {str(item["resource_id"]): item for item in receipt["resources"]}
    states = core.selected_states(manifest["states"], args)
    inputs_list = [
        core.build_frozen_input(
            state,
            resource_by_id=resources,
            receipt_by_id=receipt_by_id,
            data_root=data_root,
            modality=frozen["modality"],
        )
        for state in states
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.client_trace_root.mkdir(parents=True, exist_ok=True)
    client = core.DreamZeroClient(
        args.host,
        args.port,
        connect_timeout=args.connect_timeout,
        response_timeout=args.response_timeout,
    )
    paths = []
    try:
        for index, inputs in enumerate(inputs_list, start=1):
            path = run_state(args, client, inputs)
            paths.append(path)
            print(
                json.dumps(
                    {
                        "event": "dose_state_complete",
                        "count": index,
                        "total": len(inputs_list),
                        "state_id": inputs.state_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        client.close()
    inventory_name = (
        "run_inventory.json"
        if args.shard_count == 1
        else f"run_inventory.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json"
    )
    inventory_path = args.output_root / inventory_name
    inventory = {
        "schema": "dreamzero-future-latent-dose-run-inventory-v1",
        "status": "complete",
        "state_count": len(paths),
        "state_ids": [inputs.state_id for inputs in inputs_list],
        "recipient_seed": RECIPIENT_SEED,
        "donor_seed": DONOR_SEED,
        "alphas": list(ALPHAS),
        "new_model_call_count": 3 * len(paths),
        "runner_sha256": core.sha256_file(Path(__file__).resolve()),
        "manifest_file_sha256": args.expected_manifest_sha256,
        "core_result_root": str(args.core_result_root.resolve()),
        "core_client_trace_root": str(args.core_client_trace_root.resolve()),
        "core_server_trace_root": str(args.core_server_trace_root),
        "client_trace_root": str(args.client_trace_root.resolve()),
        "server_trace_root": str(args.server_trace_root),
        "results": [{"path": str(path), "sha256": core.sha256_file(path)} for path in paths],
    }
    if inventory_path.exists():
        core.verify_sha_sidecar(inventory_path)
        if core.load_json(inventory_path) != inventory:
            raise ValueError("existing dose run inventory differs")
    else:
        freeze_json(inventory_path, inventory)


if __name__ == "__main__":
    main()
