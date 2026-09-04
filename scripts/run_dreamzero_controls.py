#!/usr/bin/env python3
"""Run provenance-separated DreamZero implementation and Gaussian controls.

This script never modifies the frozen core transplant outputs.  It provides:

* an excluded debug check that ``mode=record`` is exactly identical to the
  patched server's ``mode=off`` at the same seed; and
* one deterministic, per-step Frobenius-norm-matched Gaussian video trace per
  frozen evaluation state, replayed with each of the four recipient action
  noises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import run_dreamzero_future_transplants as core


GAUSSIAN_SCHEMA = "dreamzero-gaussian-future-control-state-v1"
DEBUG_SCHEMA = "dreamzero-off-record-control-v1"
GAUSSIAN_SALT = "dreamzero-per-state-norm-matched-gaussian-v1-20260904"
EPS = 1e-12


def common_client(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--client-trace-root", type=Path, required=True)
    parser.add_argument("--server-trace-root", type=Path, required=True)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--response-timeout", type=float, default=3600.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    debug = subparsers.add_parser("off-record-debug")
    common_client(debug)
    debug.add_argument("--debug-bundle-root", type=Path, required=True)
    debug.add_argument("--debug-prompt", default=core.DEBUG_PROMPT)
    debug.add_argument("--noise-seed", type=int, default=211)

    gaussian = subparsers.add_parser("gaussian-evaluation")
    common_client(gaussian)
    gaussian.add_argument("--manifest", type=Path, required=True)
    gaussian.add_argument("--expected-manifest-sha256", required=True)
    gaussian.add_argument("--data-root", type=Path, required=True)
    gaussian.add_argument("--metadata-root", type=Path, required=True)
    gaussian.add_argument("--core-result-root", type=Path, required=True)
    gaussian.add_argument("--core-client-trace-root", type=Path, required=True)
    gaussian.add_argument("--core-server-trace-root", type=Path, required=True)
    gaussian.add_argument("--gaussian-source-seed", type=int, default=211)
    gaussian.add_argument("--state-id", action="append", default=[])
    gaussian.add_argument("--shard-index", type=int, default=0)
    gaussian.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


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


def require_frozen(path: Path) -> str:
    """Verify a completed artifact and its immutable mode."""
    digest = core.verify_sha_sidecar(path)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise RuntimeError(f"completed artifact is not read-only: {path}")
    return digest


def infer_controlled(
    client: core.DreamZeroClient,
    inputs: core.FrozenInputs,
    label: str,
    control: Mapping[str, Any],
) -> dict[str, Any]:
    client.reset()
    request = dict(inputs.request)
    request["session_id"] = f"control:{inputs.state_id}:{label}"
    request[core.CONTROL_KEY] = dict(control)
    return client.infer(request)


def validate_off_audit(response: Mapping[str, Any], seed: int) -> dict[str, Any]:
    audit = response.get(core.AUDIT_KEY)
    if not isinstance(audit, dict):
        raise TypeError("mode=off response lacks intervention audit")
    if audit.get("mode") != "off" or audit.get("status") != "off":
        raise ValueError(f"mode=off audit mismatch: {audit}")
    if int(audit.get("noise_seed", -1)) != seed:
        raise ValueError("mode=off did not use requested noise seed")
    if int(audit.get("current_start_frame", -1)) != core.EXPECTED_FIRST_START_FRAME:
        raise ValueError("mode=off did not begin after exactly one conditioning frame")
    if int(audit.get("num_solver_steps", -1)) != core.EXPECTED_SOLVER_STEPS:
        raise ValueError("mode=off solver-step count mismatch")
    if audit.get("applied_video_steps") != []:
        raise ValueError("mode=off unexpectedly applied replay steps")
    if any(
        audit.get(key) is not None
        for key in (
            "trace_path",
            "action_noise_reference_path",
            "video_trace_sha256",
            "donor_action_noise_sha256",
            "recipient_reference_action_noise_sha256",
        )
    ):
        raise ValueError("mode=off unexpectedly reports trace provenance")
    return core.json_safe(audit)


def run_off_record_debug(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.client_trace_root.mkdir(parents=True, exist_ok=True)
    debug_args = argparse.Namespace(
        debug_bundle_root=args.debug_bundle_root,
        debug_prompt=args.debug_prompt,
    )
    inputs = core.build_debug_input(debug_args)
    trace_client = args.client_trace_root / f"debug_record_seed_{args.noise_seed}.pt"
    trace_server = args.server_trace_root / f"debug_record_seed_{args.noise_seed}.pt"
    result_path = args.output_root / "result.json"
    if result_path.exists():
        require_frozen(result_path)
        result = core.load_json(result_path)
        if (
            result.get("schema") != DEBUG_SCHEMA
            or result.get("status") != "complete"
            or result.get("admission") != "excluded_debug_smoke"
            or result.get("scientific_admission") is not False
            or int(result.get("noise_seed", -1)) != int(args.noise_seed)
            or result.get("input_fingerprint") != inputs.audit["input_fingerprint"]
            or result.get("runner_sha256") != core.sha256_file(Path(__file__).resolve())
            or result.get("trace", {}).get("server_path") != str(trace_server)
        ):
            raise ValueError("existing debug control result is invalid")
        trace_digest = require_frozen(trace_client)
        if trace_digest != result.get("trace", {}).get("sha256"):
            raise ValueError("existing debug trace differs")
        arrays_path = args.output_root / "actions.npz"
        require_frozen(arrays_path)
        if core.sha256_file(arrays_path) != result.get("actions_npz", {}).get("sha256"):
            raise ValueError("existing debug actions differ")
        with np.load(arrays_path, allow_pickle=False) as archive:
            off_action = np.asarray(archive["mode_off_action"])
            record_action = np.asarray(archive["mode_record_action"])
        if not np.array_equal(off_action, record_action):
            raise ValueError("existing debug mode-off/record parity failed")
        validate_off_audit({core.AUDIT_KEY: result.get("mode_off_audit")}, int(args.noise_seed))
        core.validate_server_audit(
            {core.AUDIT_KEY: result.get("mode_record_audit")},
            label="mode_record",
            mode="record",
            noise_seed=int(args.noise_seed),
            trace_path=trace_server,
            action_reference_path=None,
            replay_start=0,
            replay_stop=core.EXPECTED_SOLVER_STEPS,
            expected_source_trace_hash=None,
            expected_source_action_hash=None,
            expected_recipient_action_hash=None,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    client = core.DreamZeroClient(
        args.host,
        args.port,
        connect_timeout=args.connect_timeout,
        response_timeout=args.response_timeout,
    )
    try:
        off_response = infer_controlled(
            client,
            inputs,
            "mode_off",
            {"mode": "off", "noise_seed": int(args.noise_seed)},
        )
        off_action = core.action_from_response(off_response, "mode_off")
        off_audit = validate_off_audit(off_response, int(args.noise_seed))
        record_response = infer_controlled(
            client,
            inputs,
            "mode_record",
            core.request_control(
                mode="record",
                noise_seed=int(args.noise_seed),
                trace_path=trace_server,
                action_noise_reference_path=None,
                replay_start=0,
                replay_stop=core.EXPECTED_SOLVER_STEPS,
            ),
        )
        record_action = core.action_from_response(record_response, "mode_record")
        record_audit = core.validate_server_audit(
            record_response,
            label="mode_record",
            mode="record",
            noise_seed=int(args.noise_seed),
            trace_path=trace_server,
            action_reference_path=None,
            replay_start=0,
            replay_stop=core.EXPECTED_SOLVER_STEPS,
            expected_source_trace_hash=None,
            expected_source_action_hash=None,
            expected_recipient_action_hash=None,
        )
    finally:
        client.close()
    core.wait_for_trace(trace_client)
    if not np.array_equal(off_action, record_action):
        raise RuntimeError("patched mode=record differs from mode=off")
    trace_entry = core.freeze_with_sidecar(trace_client)
    arrays_path = args.output_root / "actions.npz"
    freeze_npz(
        arrays_path,
        {"mode_off_action": off_action, "mode_record_action": record_action},
    )
    result = {
        "schema": DEBUG_SCHEMA,
        "status": "complete",
        "admission": "excluded_debug_smoke",
        "scientific_admission": False,
        "input_fingerprint": inputs.audit["input_fingerprint"],
        "noise_seed": int(args.noise_seed),
        "mode_off_record_bit_exact": True,
        "maximum_absolute_error": 0.0,
        "action_shape": list(off_action.shape),
        "action_dtype": str(off_action.dtype),
        "mode_off_audit": off_audit,
        "mode_record_audit": record_audit,
        "trace": {**trace_entry, "client_path": str(trace_client), "server_path": str(trace_server)},
        "actions_npz": {
            "path": str(arrays_path),
            "sha256": core.sha256_file(arrays_path),
        },
        "runner_sha256": core.sha256_file(Path(__file__).resolve()),
    }
    freeze_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor expected")
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def video_trace_sha256(trace: Mapping[str, Any]) -> str:
    import torch

    values = [*trace["video_latents_pre_step"], trace["final_video_latent"]]
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise TypeError("video trace contains a non-tensor")
        value = value.detach().cpu().contiguous()
        digest.update(index.to_bytes(4, "little", signed=False))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gaussian_seed(state_id: str) -> int:
    digest = hashlib.sha256(f"{GAUSSIAN_SALT}:{state_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def matched_gaussian(target: Any, generator: Any) -> tuple[Any, dict[str, float]]:
    import torch

    target_float = target.detach().cpu().float()
    target_norm = float(torch.linalg.vector_norm(target_float))
    noise = torch.randn(target.shape, generator=generator, dtype=torch.float32)
    if target_norm == 0.0:
        matched = torch.zeros_like(target)
    else:
        noise = noise * (target_norm / float(torch.linalg.vector_norm(noise)))
        matched = noise.to(dtype=target.dtype)
        for _ in range(2):
            current = float(torch.linalg.vector_norm(matched.float()))
            matched = (matched.float() * (target_norm / current)).to(dtype=target.dtype)
    actual_norm = float(torch.linalg.vector_norm(matched.float()))
    relative_error = abs(actual_norm - target_norm) / max(target_norm, EPS)
    if relative_error > 5e-4:
        raise RuntimeError(f"Gaussian norm match failed: relative error {relative_error}")
    return matched.contiguous(), {
        "source_frobenius_norm": target_norm,
        "gaussian_frobenius_norm": actual_norm,
        "relative_norm_error": relative_error,
    }


def build_gaussian_trace(source_path: Path, output_path: Path, state_id: str) -> dict[str, Any]:
    import torch

    source_file_sha256 = core.verify_sha_sidecar(source_path)
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    if source.get("format_version") != 3:
        raise ValueError("source trace is not DreamZero format v3")
    if video_trace_sha256(source) != source.get("video_trace_sha256"):
        raise ValueError("source video trace hash is invalid")
    if tensor_sha256(source["initial_action_noise"]) != source["initial_action_noise_sha256"]:
        raise ValueError("source action-noise hash is invalid")
    seed = gaussian_seed(state_id)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    steps = []
    norm_audit = []
    source_values = [*source["video_latents_pre_step"], source["final_video_latent"]]
    for index, value in enumerate(source_values):
        matched, audit = matched_gaussian(value, generator)
        audit["trace_index"] = index
        norm_audit.append(audit)
        steps.append(matched)
    trace = {
        "format_version": 3,
        "current_start_frame": source["current_start_frame"],
        "noise_seed": source["noise_seed"],
        "video_timesteps": source["video_timesteps"].clone(),
        "video_sigmas": source["video_sigmas"].clone(),
        "action_timesteps": source["action_timesteps"].clone(),
        "action_sigmas": source["action_sigmas"].clone(),
        "initial_action_noise": source["initial_action_noise"].clone(),
        "initial_action_noise_sha256": source["initial_action_noise_sha256"],
        "video_latents_pre_step": steps[:-1],
        "final_video_latent": steps[-1],
        "synthetic_provenance": {
            "kind": "incoherent_per_step_norm_matched_gaussian",
            "state_id": state_id,
            "gaussian_salt": GAUSSIAN_SALT,
            "gaussian_rng_seed": seed,
            "source_trace_path": str(source_path),
            "source_trace_file_sha256": source_file_sha256,
            "source_video_trace_sha256": source["video_trace_sha256"],
            "source_action_noise_sha256": source["initial_action_noise_sha256"],
            "norm_audit": norm_audit,
        },
    }
    trace["video_trace_sha256"] = video_trace_sha256(trace)
    if output_path.exists():
        trace_file_sha256 = require_frozen(output_path)
        existing = torch.load(output_path, map_location="cpu", weights_only=True)
        if set(existing) != set(trace):
            raise ValueError(f"existing Gaussian trace schema differs: {output_path}")
        scalar_keys = (
            "format_version",
            "current_start_frame",
            "noise_seed",
            "initial_action_noise_sha256",
            "video_trace_sha256",
            "synthetic_provenance",
        )
        if any(existing.get(key) != trace.get(key) for key in scalar_keys):
            raise ValueError(f"existing Gaussian trace provenance differs: {output_path}")
        tensor_keys = (
            "video_timesteps",
            "video_sigmas",
            "action_timesteps",
            "action_sigmas",
            "initial_action_noise",
            "final_video_latent",
        )
        if any(not torch.equal(existing[key], trace[key]) for key in tensor_keys):
            raise ValueError(f"existing Gaussian trace tensor differs: {output_path}")
        if len(existing["video_latents_pre_step"]) != len(trace["video_latents_pre_step"]) or any(
            not torch.equal(left, right)
            for left, right in zip(
                existing["video_latents_pre_step"], trace["video_latents_pre_step"], strict=True
            )
        ):
            raise ValueError(f"existing Gaussian step trace differs: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        torch.save(trace, temporary)
        os.chmod(temporary, 0o444)
        os.replace(temporary, output_path)
        trace_file_sha256 = core.freeze_with_sidecar(output_path)["sha256"]
    return {
        "client_path": str(output_path),
        "trace_file_sha256": trace_file_sha256,
        "trace_size_bytes": output_path.stat().st_size,
        "video_trace_sha256": trace["video_trace_sha256"],
        "source_action_noise_sha256": trace["initial_action_noise_sha256"],
        "gaussian_rng_seed": seed,
        "gaussian_salt": GAUSSIAN_SALT,
        "source_trace_path": str(source_path),
        "source_trace_file_sha256": source_file_sha256,
        "source_video_trace_sha256": source["video_trace_sha256"],
        "norm_audit": norm_audit,
    }


def load_core_state(
    args: argparse.Namespace, inputs: core.FrozenInputs
) -> tuple[np.ndarray, dict[int, dict[str, Any]], dict[str, Any]]:
    state_dir = args.core_result_root / "states" / inputs.state_id
    result_path = state_dir / "result.json"
    core.verify_sha_sidecar(result_path)
    result = core.load_json(result_path)
    if (
        result.get("schema") != core.SCHEMA
        or result.get("admission") != "evaluation"
        or result.get("scientific_admission") is not True
        or result.get("status") != "complete"
    ):
        raise ValueError(f"core state is not an admitted complete result: {inputs.state_id}")
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
        raise ValueError(f"core state manifest mismatch: {inputs.state_id}")
    arrays_path = state_dir / result["artifacts"]["actions_npz"]["relative_path"]
    if core.verify_sha_sidecar(arrays_path) != result["artifacts"]["actions_npz"]["sha256"]:
        raise ValueError(f"core action artifact mismatch: {inputs.state_id}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        seeds = np.asarray(archive["branch_seeds"], dtype=np.int64)
        native = np.asarray(archive["native_actions"])
    if tuple(seeds.tolist()) != tuple(core.BRANCH_SEEDS):
        raise ValueError(f"core branch-seed order mismatch: {inputs.state_id}")
    inventory_path = state_dir / result["artifacts"]["trace_inventory"]["relative_path"]
    if core.verify_sha_sidecar(inventory_path) != result["artifacts"]["trace_inventory"]["sha256"]:
        raise ValueError(f"core trace inventory mismatch: {inputs.state_id}")
    inventory = core.load_json(inventory_path)
    if (
        inventory.get("schema") != core.INVENTORY_SCHEMA
        or inventory.get("state_id") != inputs.state_id
        or int(inventory.get("trace_count", -1)) != 4
        or not isinstance(inventory.get("traces"), list)
        or len(inventory["traces"]) != 4
    ):
        raise ValueError(f"core trace inventory identity/schema mismatch: {inputs.state_id}")
    trace_by_seed = {int(row["branch_seed"]): row for row in inventory["traces"]}
    if len(trace_by_seed) != 4 or set(trace_by_seed) != set(core.BRANCH_SEEDS):
        raise ValueError(f"core trace inventory seed mismatch: {inputs.state_id}")
    return native, trace_by_seed, result


def run_gaussian_state(
    args: argparse.Namespace,
    client: core.DreamZeroClient,
    inputs: core.FrozenInputs,
) -> Path:
    state_dir = args.output_root / "states" / inputs.state_id
    result_path = state_dir / "result.json"
    native, core_traces, core_result = load_core_state(args, inputs)
    source_seed = int(args.gaussian_source_seed)
    if source_seed not in core_traces:
        raise ValueError(f"Gaussian source seed is not a core branch: {source_seed}")
    source_client = args.core_client_trace_root / inputs.state_id / f"native_seed_{source_seed}.pt"
    if core.verify_sha_sidecar(source_client) != core_traces[source_seed]["sha256"]:
        raise ValueError(f"source trace file mismatch: {inputs.state_id}")
    gaussian_client = args.client_trace_root / inputs.state_id / "norm_matched_gaussian.pt"
    gaussian_server = args.server_trace_root / inputs.state_id / "norm_matched_gaussian.pt"
    gaussian = build_gaussian_trace(source_client, gaussian_client, inputs.state_id)
    gaussian["server_path"] = str(gaussian_server)

    checkpoint_path = state_dir / "checkpoint.json"
    header = {
        "schema": "dreamzero-gaussian-control-checkpoint-v1",
        "status": "partial",
        "state_id": inputs.state_id,
        "input_fingerprint": inputs.audit["input_fingerprint"],
        "core_result_sha256": core.sha256_file(
            args.core_result_root / "states" / inputs.state_id / "result.json"
        ),
        "runner_sha256": core.sha256_file(Path(__file__).resolve()),
        "gaussian_trace_file_sha256": gaussian["trace_file_sha256"],
        "gaussian_video_trace_sha256": gaussian["video_trace_sha256"],
        "recipient_seeds": list(core.BRANCH_SEEDS),
        "completed": {},
    }
    if checkpoint_path.exists():
        checkpoint = core.load_json(checkpoint_path)
        for key, value in header.items():
            if key not in {"status", "completed"} and checkpoint.get(key) != value:
                raise ValueError(f"Gaussian checkpoint mismatch at {key}: {inputs.state_id}")
        completed = checkpoint.get("completed")
        if not isinstance(completed, dict):
            raise ValueError("Gaussian checkpoint lacks completed map")
    else:
        checkpoint = header
        completed = checkpoint["completed"]
        core.atomic_write_json(checkpoint_path, checkpoint)

    gaussian_actions: dict[int, np.ndarray] = {}
    for recipient_index, recipient_seed in enumerate(core.BRANCH_SEEDS):
        label = f"gaussian_recipient_{recipient_seed}"
        if label in completed:
            record = completed[label]
            if (
                int(record.get("recipient_seed", -1)) != recipient_seed
                or record.get("control") != "incoherent_per_step_norm_matched_gaussian"
            ):
                raise ValueError(f"{label}: saved call identity differs")
            core.validate_server_audit(
                {core.AUDIT_KEY: record.get("server_audit")},
                label=label,
                mode="replay",
                noise_seed=recipient_seed,
                trace_path=gaussian_server,
                action_reference_path=(
                    args.core_server_trace_root
                    / inputs.state_id
                    / f"native_seed_{recipient_seed}.pt"
                ),
                replay_start=0,
                replay_stop=core.EXPECTED_SOLVER_STEPS,
                expected_source_trace_hash=gaussian["video_trace_sha256"],
                expected_source_action_hash=gaussian["source_action_noise_sha256"],
                expected_recipient_action_hash=core_traces[recipient_seed]["action_noise_sha256"],
            )
            gaussian_actions[recipient_seed] = core.action_from_record(record, label)
            continue
        recipient_server = (
            args.core_server_trace_root / inputs.state_id / f"native_seed_{recipient_seed}.pt"
        )
        control = core.request_control(
            mode="replay",
            noise_seed=recipient_seed,
            trace_path=gaussian_server,
            action_noise_reference_path=recipient_server,
            replay_start=0,
            replay_stop=core.EXPECTED_SOLVER_STEPS,
        )
        response = infer_controlled(client, inputs, label, control)
        action = core.action_from_response(response, label)
        audit = core.validate_server_audit(
            response,
            label=label,
            mode="replay",
            noise_seed=recipient_seed,
            trace_path=gaussian_server,
            action_reference_path=recipient_server,
            replay_start=0,
            replay_stop=core.EXPECTED_SOLVER_STEPS,
            expected_source_trace_hash=gaussian["video_trace_sha256"],
            expected_source_action_hash=gaussian["source_action_noise_sha256"],
            expected_recipient_action_hash=core_traces[recipient_seed]["action_noise_sha256"],
        )
        record = {
            **core.action_record(action, audit),
            "recipient_seed": recipient_seed,
            "control": "incoherent_per_step_norm_matched_gaussian",
        }
        completed[label] = record
        checkpoint["completed"] = completed
        core.atomic_write_json(checkpoint_path, checkpoint)
        gaussian_actions[recipient_seed] = action

    expected_labels = {f"gaussian_recipient_{seed}" for seed in core.BRANCH_SEEDS}
    if set(completed) != expected_labels:
        raise ValueError(
            f"Gaussian completed labels differ: missing={sorted(expected_labels-set(completed))}, "
            f"extra={sorted(set(completed)-expected_labels)}"
        )
    expected_shape = native[0].shape
    expected_dtype = native.dtype
    for seed, action in gaussian_actions.items():
        if action.shape != expected_shape or action.dtype != expected_dtype:
            raise ValueError(
                f"Gaussian action schema differs for recipient {seed}: "
                f"{action.shape}/{action.dtype} != {expected_shape}/{expected_dtype}"
            )
    gaussian_array = np.stack([gaussian_actions[seed] for seed in core.BRANCH_SEEDS])
    arrays_path = state_dir / "actions.npz"
    expected_arrays = {
        "branch_seeds": np.asarray(core.BRANCH_SEEDS, dtype=np.int64),
        "native_actions": native,
        "gaussian_actions": gaussian_array,
    }
    if arrays_path.exists():
        core.verify_sha_sidecar(arrays_path)
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_arrays) or any(
                not np.array_equal(archive[key], value)
                for key, value in expected_arrays.items()
            ):
                raise ValueError(f"existing Gaussian arrays differ: {inputs.state_id}")
    else:
        freeze_npz(arrays_path, expected_arrays)
    perturbation_l2 = np.linalg.norm(
        (gaussian_array.astype(np.float64) - native.astype(np.float64)).reshape(4, -1),
        axis=1,
    )
    result = {
        "schema": GAUSSIAN_SCHEMA,
        "status": "complete",
        "admission": "evaluation_control",
        "scientific_admission": True,
        "control_class": "incoherent_per_step_norm_matched_gaussian",
        "state": core_result["state"],
        "input_fingerprint": inputs.audit["input_fingerprint"],
        "branch_seeds": list(core.BRANCH_SEEDS),
        "source_seed_for_norms": source_seed,
        "call_count": 4,
        "action_noise_source": "recipient native trace for each row",
        "action_coordinates_written_by_client": False,
        "gaussian_trace": gaussian,
        "action_l2_from_native_by_seed": {
            str(seed): float(perturbation_l2[index])
            for index, seed in enumerate(core.BRANCH_SEEDS)
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
        "calls": [completed[f"gaussian_recipient_{seed}"] for seed in core.BRANCH_SEEDS],
    }
    if result_path.exists():
        require_frozen(result_path)
        require_frozen(arrays_path)
        require_frozen(checkpoint_path)
        if core.load_json(result_path) != result:
            raise ValueError(f"existing Gaussian result identity differs: {inputs.state_id}")
        checkpoint_final = core.load_json(checkpoint_path)
        if (
            checkpoint_final.get("status") != "complete"
            or checkpoint_final.get("result_sha256") != core.sha256_file(result_path)
            or checkpoint_final.get("actions_sha256") != core.sha256_file(arrays_path)
        ):
            raise ValueError(f"existing Gaussian checkpoint differs: {inputs.state_id}")
        return result_path
    freeze_json(result_path, result)
    checkpoint["status"] = "complete"
    checkpoint["result_sha256"] = core.sha256_file(result_path)
    checkpoint["actions_sha256"] = core.sha256_file(arrays_path)
    core.atomic_write_json(checkpoint_path, checkpoint, mode=0o444)
    core.freeze_with_sidecar(checkpoint_path)
    return result_path


def run_gaussian_evaluation(args: argparse.Namespace) -> None:
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
            path = run_gaussian_state(args, client, inputs)
            paths.append(path)
            print(
                json.dumps(
                    {
                        "event": "gaussian_state_complete",
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
        "schema": "dreamzero-gaussian-control-run-inventory-v1",
        "status": "complete",
        "admission": "evaluation_control",
        "state_count": len(paths),
        "state_ids": [inputs.state_id for inputs in inputs_list],
        "call_count": 4 * len(paths),
        "runner_sha256": core.sha256_file(Path(__file__).resolve()),
        "manifest_file_sha256": args.expected_manifest_sha256,
        "core_result_root": str(args.core_result_root.resolve()),
        "core_client_trace_root": str(args.core_client_trace_root.resolve()),
        "core_server_trace_root": str(args.core_server_trace_root),
        "client_trace_root": str(args.client_trace_root.resolve()),
        "server_trace_root": str(args.server_trace_root),
        "results": [
            {"path": str(path), "sha256": core.sha256_file(path)} for path in paths
        ],
    }
    if inventory_path.exists():
        core.verify_sha_sidecar(inventory_path)
        if core.load_json(inventory_path) != inventory:
            raise ValueError("existing Gaussian run inventory differs")
    else:
        freeze_json(inventory_path, inventory)


def main() -> None:
    args = parse_args()
    if args.command == "off-record-debug":
        run_off_record_debug(args)
    else:
        run_gaussian_evaluation(args)


if __name__ == "__main__":
    main()
