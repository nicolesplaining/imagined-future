#!/usr/bin/env python3
"""Run the preregistered LingBot b0-to-b1 latent dose follow-up.

The core 4x4 cohort is read-only.  Alpha endpoints are copied from its b0
action-noise row; only the three interior normalized future latents require new
model calls.  For each interior latent, the official t=0 future-cache forward
pass is recomputed before denoising actions from the unchanged frozen b0 noise.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL_SHA256 = "8b6b4103b5c172f28c896b9834fda114aa52684c53f8c570c78c346fda9d3eba"
CORE_RUNNER_SHA256 = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"
EXPECTED_CHECKPOINT_PATH = Path(
    "/home/ubuntu/if_external/checkpoints/lingbot-va-posttrain-libero-long"
)
PARITY_GATE_SHA256 = "6437f774088084b67e6aea001376304dfe01ee622358358dd40642d49d0a67d5"
ORACLE_SCRIPT_SHA256 = "893c2d9152575b583e1db0d8fafab79727c5038613099aa983fae1ad74f96afc"
OFFICIAL_SERVER_SHA256 = "9c2a427611db487fea5cf40f184b713bf2088e533990ee00fdcd020d2668b4bf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core-result-root", type=Path, required=True)
    parser.add_argument("--core-runner", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--oracle-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-content-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numpy_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def install_exact_copy(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"frozen output copy differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def import_core_runner(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_lingbot_core_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import core runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_checkpoint_content(checkpoint: Path, manifest_path: Path) -> dict[str, Any]:
    receipt = json.loads(manifest_path.read_text())
    if Path(receipt.get("checkpoint_root", "")).resolve() != checkpoint:
        raise RuntimeError("checkpoint content manifest root mismatch")
    if receipt.get("huggingface_revisions") != [CHECKPOINT_REVISION]:
        raise RuntimeError("checkpoint content manifest revision mismatch")
    aggregate = hashlib.sha256()
    expected_paths = []
    for item in receipt.get("files", []):
        relative = str(item["path"])
        expected_paths.append(relative)
        path = checkpoint / relative
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"checkpoint payload absent/size mismatch: {relative}")
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"checkpoint payload hash mismatch: {relative}")
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(item["bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    actual_paths = sorted(
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts
    )
    if actual_paths != expected_paths:
        raise RuntimeError("checkpoint payload set differs from content manifest")
    if aggregate.hexdigest() != receipt.get("aggregate_sha256"):
        raise RuntimeError("checkpoint aggregate hash mismatch")
    return receipt


def validate_oracle(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    comparison = receipt.get("comparison", {})
    if receipt.get("status") != "complete":
        raise RuntimeError("upstream native-oracle audit is not complete")
    if receipt.get("included_in_evaluation") is not False:
        raise RuntimeError("native-oracle receipt is not marked excluded")
    if receipt.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("native-oracle upstream commit mismatch")
    if receipt.get("checkpoint_revision") != CHECKPOINT_REVISION:
        raise RuntimeError("native-oracle checkpoint revision mismatch")
    if receipt.get("parity_gate", {}).get("sha256") != PARITY_GATE_SHA256:
        raise RuntimeError("native-oracle parity gate hash mismatch")
    if receipt.get("audit_script", {}).get("sha256") != ORACLE_SCRIPT_SHA256:
        raise RuntimeError("native-oracle audit script hash mismatch")
    audit_script_path = Path(receipt["audit_script"]["path"])
    if not audit_script_path.is_file() or sha256_file(audit_script_path) != ORACLE_SCRIPT_SHA256:
        raise RuntimeError("native-oracle audit script content changed")
    if receipt.get("official_entrypoint_source_sha256") != OFFICIAL_SERVER_SHA256:
        raise RuntimeError("native-oracle official source hash mismatch")
    if receipt.get("state_id") != "dev_task00_state000" or receipt.get("branch_index") != 0:
        raise RuntimeError("native-oracle state/branch differs from frozen gate")
    for prefix in ("future", "action"):
        if not isinstance(comparison.get(f"{prefix}_bitwise_equal"), bool):
            raise RuntimeError(f"native-oracle {prefix} parity result is absent")
        error = comparison.get(f"{prefix}_max_abs_error")
        if not isinstance(error, (int, float)) or not np.isfinite(error):
            raise RuntimeError(f"native-oracle {prefix} numerical error is absent")
    injection = receipt.get("controlled_rng_injection", {})
    if (
        injection.get("used") is not True
        or injection.get("upstream_loop_body_modified") is not False
        or injection.get("torch_randn_call_count") != 2
    ):
        raise RuntimeError("native-oracle RNG-injection scope is not documented")
    if receipt.get("parity_gate_passed") is not True:
        raise RuntimeError("native-oracle exact parity gate did not pass")
    if receipt["frozen_input"].get("official_encoder_bitwise_equal") is not True:
        raise RuntimeError("native-oracle encoder is not bitwise equal")
    if any(comparison.get(f"{prefix}_bitwise_equal") is not True for prefix in ("future", "action")):
        raise RuntimeError("native-oracle future/action is not bitwise equal")
    return receipt


def validate_core(
    *,
    records: list[dict[str, Any]],
    core_root: Path,
    manifest_sha256: str,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        state_id = str(record["state_id"])
        result_path = core_root / state_id / "result.json"
        actions_path = core_root / state_id / "actions.npz"
        if not result_path.is_file() or not actions_path.is_file():
            raise RuntimeError(f"core cohort incomplete: {state_id}")
        metadata = json.loads(result_path.read_text())
        expected = {
            "status": "complete",
            "state_id": state_id,
            "admission": "evaluation",
            "prompt": record["prompt"],
            "input_sha256": record["input_sha256"],
            "branch_ids": manifest["branch_ids"],
            "video_seeds": manifest["video_seeds"],
            "action_seeds": manifest["action_seeds"],
            "manifest_sha256": manifest_sha256,
            "runner_sha256": CORE_RUNNER_SHA256,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "actions_sha256": sha256_file(actions_path),
            "native_self_latent_max_abs_error": 0.0,
            "native_self_cache_max_abs_error": 0.0,
            "native_future_hashes_unique": 4,
            "native_cache_hashes_unique": 4,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"core gate failed for {state_id}: {mismatches}")
        observation_path = Path(record["observation_path"])
        if sha256_file(observation_path) != record["input_sha256"]:
            raise RuntimeError(f"core frozen observation mismatch: {state_id}")
        frozen_path = core_root / state_id / "frozen_inputs.pt"
        future_paths = [core_root / state_id / f"future_b{index}.pt" for index in (0, 1)]
        if not frozen_path.is_file() or any(not path.is_file() for path in future_paths):
            raise RuntimeError(f"core frozen tensors absent: {state_id}")
        frozen = torch.load(frozen_path, map_location="cpu")
        if set(frozen) != {"init_latent", "action_noises", "action_noise_hashes"}:
            raise RuntimeError(f"core frozen-input schema changed: {state_id}")
        if (
            tuple(frozen["init_latent"].shape) != (1, 48, 1, 8, 16)
            or frozen["init_latent"].dtype != torch.bfloat16
            or tuple(frozen["action_noises"].shape) != (4, 1, 30, 4, 4, 1)
            or frozen["action_noises"].dtype != torch.bfloat16
        ):
            raise RuntimeError(f"core frozen-input shape/dtype changed: {state_id}")
        actual_noise_hashes = [tensor_hash(value) for value in frozen["action_noises"]]
        if (
            actual_noise_hashes != list(frozen["action_noise_hashes"])
            or actual_noise_hashes != list(metadata["action_noise_hashes"])
        ):
            raise RuntimeError(f"core frozen action-noise hashes changed: {state_id}")
        for index, path in enumerate(future_paths):
            payload = torch.load(path, map_location="cpu")
            future = payload.get("future")
            if (
                set(payload) != {"future", "video_seed"}
                or not isinstance(future, torch.Tensor)
                or tuple(future.shape) != (1, 48, 4, 8, 16)
                or future.dtype != torch.bfloat16
                or payload["video_seed"] != manifest["video_seeds"][index]
                or tensor_hash(future) != metadata["future_hashes"][index]
                or not torch.equal(
                    future[:, :, 0:1], frozen["init_latent"][:, :, 0:1]
                )
            ):
                raise RuntimeError(f"core future b{index} binding failed: {state_id}")
        with np.load(actions_path, allow_pickle=False) as archive:
            required_arrays = {
                "native_actions",
                "latent_grid_actions",
                "cache_replay_actions",
                "native_executed_actions",
                "latent_grid_executed_actions",
                "cache_replay_executed_actions",
                "gaussian_actions",
                "gaussian_executed_actions",
                "donor_future_recipient_cache_actions",
                "recipient_future_donor_cache_actions",
                "donor_future_recipient_cache_executed_actions",
                "recipient_future_donor_cache_executed_actions",
            }
            missing = required_arrays - set(archive.files)
            if missing:
                raise RuntimeError(f"core arrays absent for {state_id}: {missing}")
            expected_shapes = {
                "native_actions": (4, 7, 4, 4),
                "latent_grid_actions": (4, 4, 7, 4, 4),
                "cache_replay_actions": (4, 7, 4, 4),
                "native_executed_actions": (4, 7, 3, 4),
                "latent_grid_executed_actions": (4, 4, 7, 3, 4),
                "cache_replay_executed_actions": (4, 7, 3, 4),
                "gaussian_actions": (4, 7, 4, 4),
                "gaussian_executed_actions": (4, 7, 3, 4),
                "donor_future_recipient_cache_actions": (4, 4, 7, 4, 4),
                "recipient_future_donor_cache_actions": (4, 4, 7, 4, 4),
                "donor_future_recipient_cache_executed_actions": (4, 4, 7, 3, 4),
                "recipient_future_donor_cache_executed_actions": (4, 4, 7, 3, 4),
            }
            for key, shape in expected_shapes.items():
                if (
                    tuple(archive[key].shape) != shape
                    or archive[key].dtype != np.float32
                    or not np.isfinite(archive[key]).all()
                ):
                    raise RuntimeError(f"core array shape/dtype/finite failed {state_id}:{key}")
            if not np.array_equal(
                archive["native_actions"][0], archive["latent_grid_actions"][0, 0]
            ):
                raise RuntimeError(f"core b0 self endpoint failed: {state_id}")
            native = np.asarray(archive["native_executed_actions"])
            grid = np.asarray(archive["latent_grid_executed_actions"])
            replay = np.asarray(archive["cache_replay_executed_actions"])
            donor_future_recipient_cache = np.asarray(
                archive["donor_future_recipient_cache_executed_actions"]
            )
            recipient_future_donor_cache = np.asarray(
                archive["recipient_future_donor_cache_executed_actions"]
            )
            if not np.array_equal(replay, native):
                raise RuntimeError(f"core cache replay failed: {state_id}")
            if not np.array_equal(
                donor_future_recipient_cache,
                np.broadcast_to(native[:, None], donor_future_recipient_cache.shape),
            ):
                raise RuntimeError(f"core recipient-cache routing failed: {state_id}")
            if not np.array_equal(recipient_future_donor_cache, grid):
                raise RuntimeError(f"core donor-cache routing failed: {state_id}")
            if not np.isfinite(archive["gaussian_executed_actions"]).all():
                raise RuntimeError(f"core gaussian control is non-finite: {state_id}")
        if metadata.get("grid_axis_0") != "recipient_action_noise_source":
            raise RuntimeError(f"core grid row identity changed: {state_id}")
        if metadata.get("grid_axis_1") != "future_source":
            raise RuntimeError(f"core grid column identity changed: {state_id}")
        if metadata.get("action_coordinate_intervention") != "none":
            raise RuntimeError(f"core wrote action coordinates: {state_id}")
        metadata_by_id[state_id] = metadata
    return metadata_by_id


def validate_completed_dose_state(
    *,
    state_id: str,
    metadata: dict[str, Any],
    arrays_path: Path,
    core_actions_path: Path,
) -> None:
    required_metadata = {
        "admission": "evaluation",
        "ordered_pair": {"recipient": "b0", "donor": "b1"},
        "action_noise_source": "b0",
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "interior_model_calls": 3,
        "endpoint_source": [
            "core latent_grid_actions[0,0]",
            "core latent_grid_actions[0,1]",
        ],
        "interpolation_variable": "final normalized denoised future-video latent",
        "interpolation_dtype": "torch.bfloat16",
        "interpolation_compute_dtype": "torch.float32 before cast to source dtype",
        "present_frame_overwritten_from_frozen_input": True,
        "cache_installation": "official t=0 future forward recomputed at every interior alpha",
        "action_coordinate_intervention": "none",
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in required_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"completed dose metadata invalid {state_id}: {mismatches}")
    if (
        len(metadata.get("future_sha256", [])) != 5
        or len(metadata.get("interior_cache_sha256", [])) != 5
        or metadata["interior_cache_sha256"][0] is not None
        or metadata["interior_cache_sha256"][-1] is not None
        or any(
            not isinstance(metadata["interior_cache_sha256"][index], str)
            for index in (1, 2, 3)
        )
    ):
        raise RuntimeError(f"completed dose future/cache receipt invalid: {state_id}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "alphas",
            "actions",
            "executed_actions",
            "endpoint_reused",
        }:
            raise RuntimeError(f"completed dose array schema invalid: {state_id}")
        alphas = archive["alphas"]
        actions = archive["actions"]
        executed = archive["executed_actions"]
        endpoint_reused = archive["endpoint_reused"]
    if (
        alphas.dtype != np.float64
        or not np.array_equal(alphas, [0.0, 0.25, 0.5, 0.75, 1.0])
        or actions.dtype != np.float32
        or tuple(actions.shape) != (5, 7, 4, 4)
        or executed.dtype != np.float32
        or tuple(executed.shape) != (5, 7, 3, 4)
        or endpoint_reused.dtype != np.bool_
        or not np.array_equal(endpoint_reused, [True, False, False, False, True])
        or not np.isfinite(actions).all()
        or not np.array_equal(executed, actions[..., 1:, :])
    ):
        raise RuntimeError(f"completed dose arrays invalid: {state_id}")
    with np.load(core_actions_path, allow_pickle=False) as core:
        if not np.array_equal(actions[0], core["latent_grid_actions"][0, 0]) or not np.array_equal(
            actions[-1], core["latent_grid_actions"][0, 1]
        ):
            raise RuntimeError(f"completed dose endpoint bytes invalid: {state_id}")


def run_state(
    *,
    server: Any,
    core_runner: Any,
    record: dict[str, Any],
    core_metadata: dict[str, Any],
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> None:
    state_id = str(record["state_id"])
    source_root = args.core_result_root / state_id
    output_root = args.output_root / state_id
    result_path = output_root / "result.json"
    arrays_path = output_root / "actions.npz"
    core_actions_path = source_root / "actions.npz"
    core_frozen_path = source_root / "frozen_inputs.pt"
    future0_path = source_root / "future_b0.pt"
    future1_path = source_root / "future_b1.pt"
    input_hashes = {
        "core_actions": sha256_file(core_actions_path),
        "core_frozen_inputs": sha256_file(core_frozen_path),
        "future_b0": sha256_file(future0_path),
        "future_b1": sha256_file(future1_path),
    }
    resume_identity = {
        "status": "complete",
        "state_id": state_id,
        "core_result_root": str(args.core_result_root),
        "dose_result_root": str(args.output_root),
        "core_state_root": str(source_root.resolve()),
        "dose_state_root": str(output_root.resolve()),
        "result_path": str(result_path.resolve()),
        "actions_path": str(arrays_path.resolve()),
        "protocol_sha256": args.protocol_sha256,
        "manifest_sha256": args.manifest_sha256,
        "core_runner_sha256": CORE_RUNNER_SHA256,
        "dose_runner_sha256": args.dose_runner_sha256,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_content_manifest_sha256": args.checkpoint_content_manifest_sha256,
        "checkpoint_aggregate_sha256": args.checkpoint_aggregate_sha256,
        "oracle_receipt_sha256": args.oracle_receipt_sha256,
        "oracle_future_bitwise_equal": args.oracle_future_bitwise_equal,
        "oracle_action_bitwise_equal": args.oracle_action_bitwise_equal,
        "action_noise_sha256": core_metadata["action_noise_hashes"][0],
        "input_sha256": input_hashes,
    }
    if result_path.exists() or arrays_path.exists():
        if not result_path.is_file() or not arrays_path.is_file():
            raise RuntimeError(f"partial dose output exists for {state_id}")
        existing = json.loads(result_path.read_text())
        expected = {**resume_identity, "actions_sha256": sha256_file(arrays_path)}
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"unsafe dose resume for {state_id}: {mismatches}")
        validate_completed_dose_state(
            state_id=state_id,
            metadata=existing,
            arrays_path=arrays_path,
            core_actions_path=core_actions_path,
        )
        print(f"skip verified complete {state_id}", flush=True)
        return

    frozen = torch.load(core_frozen_path, map_location="cpu")
    init_latent = frozen["init_latent"].detach().cpu()
    action_noise = frozen["action_noises"][0].detach().cpu()
    if core_runner.tensor_hash(action_noise) != core_metadata["action_noise_hashes"][0]:
        raise RuntimeError(f"frozen b0 action noise hash mismatch: {state_id}")
    future0 = torch.load(future0_path, map_location="cpu")["future"].detach().cpu()
    future1 = torch.load(future1_path, map_location="cpu")["future"].detach().cpu()
    if core_runner.tensor_hash(future0) != core_metadata["future_hashes"][0]:
        raise RuntimeError(f"core future b0 hash mismatch: {state_id}")
    if core_runner.tensor_hash(future1) != core_metadata["future_hashes"][1]:
        raise RuntimeError(f"core future b1 hash mismatch: {state_id}")
    frozen_present = init_latent[:, :, 0:1].to(future0.dtype)
    if (
        not torch.equal(future0[:, :, 0:1], frozen_present)
        or not torch.equal(future1[:, :, 0:1], frozen_present)
    ):
        raise RuntimeError(
            f"dose endpoint present differs from the frozen init latent: {state_id}"
        )

    with np.load(core_actions_path, allow_pickle=False) as archive:
        endpoint0 = np.asarray(archive["latent_grid_actions"][0, 0], dtype=np.float32)
        endpoint1 = np.asarray(archive["latent_grid_actions"][0, 1], dtype=np.float32)
    alphas = np.asarray(protocol["dose"]["alphas"], dtype=np.float64)
    if not np.array_equal(alphas, np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])):
        raise RuntimeError("frozen alpha schedule changed")
    actions = np.empty((len(alphas),) + endpoint0.shape, dtype=np.float32)
    actions[0] = endpoint0
    actions[-1] = endpoint1
    cache_hashes: list[str | None] = [None] * len(alphas)
    future_hashes: list[str] = [core_runner.tensor_hash(future0)]

    core_runner.reset(server, str(record["prompt"]))
    prompt_embeds = server.prompt_embeds.detach()
    negative_prompt_embeds = (
        server.negative_prompt_embeds.detach()
        if server.negative_prompt_embeds is not None
        else None
    )
    durations: dict[str, float] = {}
    for alpha_index, alpha in enumerate(alphas[1:-1], start=1):
        started = time.time()
        interpolated = (
            (1.0 - float(alpha)) * future0.float()
            + float(alpha) * future1.float()
        ).to(future0.dtype)
        interpolated[:, :, 0:1] = init_latent[:, :, 0:1].to(interpolated.dtype)
        future_hashes.append(core_runner.tensor_hash(interpolated))
        core_runner.reset_with_frozen_prompt(
            server, prompt_embeds, negative_prompt_embeds
        )
        core_runner.install_future(server, init_latent, interpolated)
        installed_cache = core_runner.snapshot_cache(server)
        cache_hashes[alpha_index] = core_runner.cache_hash(installed_cache)
        actions[alpha_index] = core_runner.generate_action(server, action_noise)
        durations[f"alpha_{float(alpha):.2f}"] = time.time() - started
    future_hashes.append(core_runner.tensor_hash(future1))

    executed = core_runner.executed_action_view(actions)
    if not np.array_equal(executed[0], core_runner.executed_action_view(endpoint0)):
        raise RuntimeError(f"alpha=0 executed endpoint changed: {state_id}")
    if not np.array_equal(executed[-1], core_runner.executed_action_view(endpoint1)):
        raise RuntimeError(f"alpha=1 executed endpoint changed: {state_id}")
    if not np.isfinite(actions).all():
        raise RuntimeError(f"dose produced non-finite actions: {state_id}")

    atomic_npz(
        arrays_path,
        alphas=alphas,
        actions=actions,
        executed_actions=executed,
        endpoint_reused=np.asarray([True, False, False, False, True]),
    )
    metadata = {
        **resume_identity,
        "admission": "evaluation",
        "status": "complete",
        "ordered_pair": {"recipient": "b0", "donor": "b1"},
        "action_noise_source": "b0",
        "alphas": alphas.tolist(),
        "interior_model_calls": 3,
        "endpoint_source": [
            "core latent_grid_actions[0,0]",
            "core latent_grid_actions[0,1]",
        ],
        "interpolation_variable": "final normalized denoised future-video latent",
        "interpolation_dtype": str(future0.dtype),
        "interpolation_compute_dtype": "torch.float32 before cast to source dtype",
        "present_frame_overwritten_from_frozen_input": True,
        "cache_installation": "official t=0 future forward recomputed at every interior alpha",
        "action_coordinate_intervention": "none",
        "future_sha256": future_hashes,
        "interior_cache_sha256": cache_hashes,
        "endpoint_action_sha256": [numpy_hash(endpoint0), numpy_hash(endpoint1)],
        "duration_seconds_by_interior_alpha": durations,
        "actions_sha256": sha256_file(arrays_path),
    }
    atomic_json(result_path, metadata)
    print(f"complete {state_id}", flush=True)


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    args.lingbot_root = args.lingbot_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.manifest = args.manifest.resolve()
    args.core_result_root = args.core_result_root.resolve()
    args.core_runner = args.core_runner.resolve()
    args.protocol = args.protocol.resolve()
    args.oracle_receipt = args.oracle_receipt.resolve()
    args.checkpoint_content_manifest = args.checkpoint_content_manifest.resolve()
    args.output_root = args.output_root.resolve()
    if (
        args.output_root == args.core_result_root
        or args.output_root.is_relative_to(args.core_result_root)
        or args.core_result_root.is_relative_to(args.output_root)
    ):
        raise RuntimeError("dose output must be disjoint from the read-only core result root")
    args.manifest_sha256 = sha256_file(args.manifest)
    args.protocol_sha256 = sha256_file(args.protocol)
    args.dose_runner_sha256 = sha256_file(Path(__file__))
    if args.protocol_sha256 != PROTOCOL_SHA256:
        raise RuntimeError("dose protocol differs from the outcome-blind frozen version")
    if args.manifest_sha256 != "9c8d7dd547955e4622d98c7eb42fffed157f6db983f41f02929b28a434be47b4":
        raise RuntimeError("dose manifest differs from the frozen core cohort")
    if args.checkpoint != EXPECTED_CHECKPOINT_PATH.resolve():
        raise RuntimeError("dose checkpoint path differs from the frozen deployment")
    if sha256_file(args.core_runner) != CORE_RUNNER_SHA256:
        raise RuntimeError("dose core-runner dependency changed")
    commit = subprocess.run(
        ["git", "-C", str(args.lingbot_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"LingBot checkout {commit} != {UPSTREAM_COMMIT}")

    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    protocol_sources = protocol["source"]
    for label, actual in (
        ("core_result_root", args.core_result_root),
        ("core_runner_path", args.core_runner),
        ("manifest_path", args.manifest),
    ):
        if Path(protocol_sources[label]).resolve() != actual:
            raise RuntimeError(f"runtime {label} differs from preregistration")
    ordered_pair = protocol["ordered_pair"]
    if (
        ordered_pair.get("recipient_branch_index") != 0
        or ordered_pair.get("donor_branch_index") != 1
        or ordered_pair.get("action_noise_branch_index") != 0
    ):
        raise RuntimeError("frozen b0-to-b1 ordered pair changed")
    records = [
        record for record in manifest["states"] if record["admission"] == "evaluation"
    ]
    state_ids = [str(record["state_id"]) for record in records]
    if state_ids != protocol["cohort"]["state_ids"] or len(records) != 30:
        raise RuntimeError("manifest evaluation cohort differs from preregistration")
    core_metadata = validate_core(
        records=records,
        core_root=args.core_result_root,
        manifest_sha256=args.manifest_sha256,
        manifest=manifest,
    )
    checkpoint_content = verify_checkpoint_content(
        args.checkpoint, args.checkpoint_content_manifest
    )
    oracle_receipt = validate_oracle(args.oracle_receipt)
    oracle_checkpoint = oracle_receipt["core_environment_provenance"]
    if (
        oracle_checkpoint.get("checkpoint_content_manifest_sha256")
        != sha256_file(args.checkpoint_content_manifest)
        or oracle_checkpoint.get("checkpoint_aggregate_sha256")
        != checkpoint_content["aggregate_sha256"]
        or Path(oracle_receipt["paths"]["checkpoint"]).resolve() != args.checkpoint
    ):
        raise RuntimeError("dose checkpoint differs from the exact oracle/core content")
    args.oracle_receipt_sha256 = sha256_file(args.oracle_receipt)
    args.checkpoint_content_manifest_sha256 = sha256_file(
        args.checkpoint_content_manifest
    )
    args.checkpoint_aggregate_sha256 = checkpoint_content["aggregate_sha256"]
    args.oracle_future_bitwise_equal = bool(
        oracle_receipt["comparison"]["future_bitwise_equal"]
    )
    args.oracle_action_bitwise_equal = bool(
        oracle_receipt["comparison"]["action_bitwise_equal"]
    )
    if not (
        args.oracle_future_bitwise_equal and args.oracle_action_bitwise_equal
    ):
        print(
            "WARNING: upstream native audit is numerical-only/non-exact; "
            "dose outputs must not be described as bitwise upstream parity",
            flush=True,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    install_exact_copy(args.manifest, args.output_root / "manifest.json")
    install_exact_copy(args.protocol, args.output_root / "protocol.json")

    sys.path.insert(0, str(args.lingbot_root))
    sys.path.insert(0, str(args.lingbot_root / "wan_va"))
    core_runner = import_core_runner(args.core_runner)
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.wan_va_server import VA_Server

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    init_distributed(world_size, local_rank, rank)
    config = copy.deepcopy(VA_CONFIGS["libero"])
    if config.video_exec_step != -1:
        raise RuntimeError("dose requires full video denoising and t=0 cache install")
    config.wan22_pretrained_model_name_or_path = str(args.checkpoint)
    config.local_rank = local_rank
    config.rank = rank
    config.world_size = world_size
    config.enable_offload = False
    config.save_root = str(args.output_root / "upstream_debug")
    server = VA_Server(config)
    for record in records[args.shard_index :: args.shard_count]:
        run_state(
            server=server,
            core_runner=core_runner,
            record=record,
            core_metadata=core_metadata[str(record["state_id"])],
            args=args,
            protocol=protocol,
        )


if __name__ == "__main__":
    main()
