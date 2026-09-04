#!/usr/bin/env python3
"""Run a complete 4x4 LingBot norm-matched Gaussian-source control grid."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


MANIFEST_SHA256 = "9c8d7dd547955e4622d98c7eb42fffed157f6db983f41f02929b28a434be47b4"
CORE_RUNNER_SHA256 = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
DOSE_VALIDATOR_SHA256 = "2d8b419be882eb979ed58091f7d0b0cd4322f2503aac9e4a854c558834f21b2e"
UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"
CHECKPOINT_MANIFEST_SHA256 = "b1c8722becbb4a77a840cb5716cbd57e68c5bd77cb4a0af3d19a6e9ee1de00fd"
CHECKPOINT_AGGREGATE_SHA256 = "bb895755e071bf5ab74494c07199a11c8e344b367971b4c6405321807e32b2e1"
ORACLE_RECEIPT_SHA256 = "f54eebb2d4525ec8d8c57ebdc3b1294041194e5d1e13fd0033380a09453719aa"
SHIM_SHA256 = "7f1448bdeae5f4991112d78131688d417836c91fee79624929cda5d2f135bec8"
OFFICIAL_SERVER_SHA256 = "9c2a427611db487fea5cf40f184b713bf2088e533990ee00fdcd020d2668b4bf"
EXPECTED_CHECKPOINT = Path("/home/ubuntu/if_external/checkpoints/lingbot-va-posttrain-libero-long")
BRANCH_IDS = ("b0", "b1", "b2", "b3")
GAUSSIAN_SEEDS = (900000, 900001, 900002, 900003)
MATCH_TOLERANCE = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--oracle-receipt", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--core-runner", type=Path, required=True)
    parser.add_argument("--dose-validator", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def validate_all_sources(
    *, records: list[dict[str, Any]], manifest: dict[str, Any], core_root: Path,
    core_metadata: dict[str, dict[str, Any]], core_runner: Any,
) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    all_source_hashes: list[str] = []
    all_control_hashes: list[str] = []
    for record in records:
        state_id = record["state_id"]
        root = core_root / state_id
        metadata = core_metadata[state_id]
        frozen = torch.load(root / "frozen_inputs.pt", map_location="cpu", weights_only=False)
        init_latent = frozen["init_latent"].detach().cpu()
        action_noises = frozen["action_noises"].detach().cpu()
        if tuple(action_noises.shape) != (4, 1, 30, 4, 4, 1) or action_noises.dtype != torch.bfloat16:
            raise RuntimeError(f"action-noise schema changed: {state_id}")
        action_noise_hashes = [tensor_hash(value) for value in action_noises]
        if action_noise_hashes != metadata["action_noise_hashes"] or action_noise_hashes != frozen["action_noise_hashes"]:
            raise RuntimeError(f"action-noise hash binding failed: {state_id}")
        futures: list[torch.Tensor] = []
        controls: list[torch.Tensor] = []
        metrics: list[dict[str, float]] = []
        future_file_hashes: list[str] = []
        for source in range(4):
            path = root / f"future_b{source}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            future = payload.get("future")
            if (
                set(payload) != {"future", "video_seed"}
                or not isinstance(future, torch.Tensor)
                or tuple(future.shape) != (1, 48, 4, 8, 16)
                or future.dtype != torch.bfloat16
                or payload["video_seed"] != manifest["video_seeds"][source]
                or tensor_hash(future) != metadata["future_hashes"][source]
                or not torch.equal(future[:, :, 0:1], init_latent[:, :, 0:1])
            ):
                raise RuntimeError(f"native future binding failed: {state_id}/b{source}")
            control = core_runner.gaussian_future(future, init_latent, GAUSSIAN_SEEDS[source])
            replay = core_runner.gaussian_future(future, init_latent, GAUSSIAN_SEEDS[source])
            if not torch.equal(control, replay):
                raise RuntimeError(f"Gaussian construction is nondeterministic: {state_id}/b{source}")
            if (
                tuple(control.shape) != tuple(future.shape)
                or control.dtype != future.dtype
                or not bool(torch.isfinite(control.float()).all().item())
                or not torch.equal(control[:, :, 0:1], init_latent[:, :, 0:1])
            ):
                raise RuntimeError(f"Gaussian control schema/present failed: {state_id}/b{source}")
            ref = future[:, :, 1:].float()
            ctrl = control[:, :, 1:].float()
            mean_error = abs(float(ctrl.mean() - ref.mean()))
            std_relative_error = abs(float(ctrl.std() - ref.std())) / max(float(ref.std()), 1e-12)
            norm_relative_error = abs(float(torch.linalg.vector_norm(ctrl) - torch.linalg.vector_norm(ref))) / max(float(torch.linalg.vector_norm(ref)), 1e-12)
            if max(mean_error, std_relative_error, norm_relative_error) > MATCH_TOLERANCE:
                raise RuntimeError(f"Gaussian norm-match tolerance failed: {state_id}/b{source}")
            futures.append(future)
            controls.append(control)
            future_file_hashes.append(sha256_file(path))
            source_hash = tensor_hash(future)
            control_hash = tensor_hash(control)
            all_source_hashes.append(source_hash)
            all_control_hashes.append(control_hash)
            metrics.append({
                "mean_absolute_error": mean_error,
                "std_relative_error": std_relative_error,
                "l2_norm_relative_error": norm_relative_error,
            })
        control_hashes = [tensor_hash(value) for value in controls]
        if len(set(control_hashes)) != 4:
            raise RuntimeError(f"Gaussian source controls are not unique: {state_id}")
        prepared[state_id] = {
            "init_latent": init_latent, "action_noises": action_noises,
            "action_noise_hashes": action_noise_hashes, "futures": futures,
            "controls": controls, "control_hashes": control_hashes,
            "future_file_hashes": future_file_hashes, "norm_match_metrics": metrics,
        }
    if len(set(all_source_hashes)) != 120 or len(set(all_control_hashes)) != 120:
        raise RuntimeError("source or Gaussian-control tensors are not globally 120-way unique")
    return prepared


def run_state(
    server: Any, core_runner: Any, record: dict[str, Any], source: dict[str, Any],
    core_root: Path, staging: Path, run_replay_gate: bool,
) -> dict[str, Any]:
    state_id = record["state_id"]
    state_started = time.time()
    core_actions_path = core_root / state_id / "actions.npz"
    with np.load(core_actions_path, allow_pickle=False) as archive:
        native = np.asarray(archive["latent_grid_actions"], dtype=np.float32)
        prior_gaussian = np.asarray(archive["gaussian_actions"], dtype=np.float32)
    core_runner.reset(server, record["prompt"])
    prompt_embeds = server.prompt_embeds.detach()
    negative_prompt_embeds = server.negative_prompt_embeds.detach() if server.negative_prompt_embeds is not None else None

    if run_replay_gate:
        core_runner.reset_with_frozen_prompt(server, prompt_embeds, negative_prompt_embeds)
        core_runner.install_future(server, source["init_latent"], source["futures"][0])
        native_replay = core_runner.generate_action(server, source["action_noises"][0])
        if not np.array_equal(native_replay, native[0, 0]):
            raise RuntimeError("fail-fast native core replay is not bitwise exact")

    actions = np.empty((4, 4, 7, 4, 4), dtype=np.float32)
    cache_hashes: list[list[str]] = [[""] * 4 for _ in range(4)]
    first_cell_replay_error: float | None = None
    for recipient in range(4):
        for source_index in range(4):
            core_runner.reset_with_frozen_prompt(server, prompt_embeds, negative_prompt_embeds)
            core_runner.install_future(server, source["init_latent"], source["controls"][source_index])
            cache_hashes[recipient][source_index] = core_runner.cache_hash(core_runner.snapshot_cache(server))
            actions[recipient, source_index] = core_runner.generate_action(server, source["action_noises"][recipient])
            if recipient == source_index and not np.array_equal(actions[recipient, source_index], prior_gaussian[recipient]):
                raise RuntimeError(f"existing Gaussian diagonal replay failed: {state_id}/b{recipient}")
            if run_replay_gate and recipient == 0 and source_index == 0:
                core_runner.reset_with_frozen_prompt(server, prompt_embeds, negative_prompt_embeds)
                core_runner.install_future(server, source["init_latent"], source["controls"][0])
                replay = core_runner.generate_action(server, source["action_noises"][0])
                first_cell_replay_error = float(np.max(np.abs(replay - actions[0, 0])))
                if first_cell_replay_error != 0.0:
                    raise RuntimeError("fail-fast Gaussian cell replay is not bitwise exact")
    if not np.isfinite(actions).all():
        raise RuntimeError(f"Gaussian grid produced nonfinite actions: {state_id}")
    for source_index in range(4):
        column = [cache_hashes[recipient][source_index] for recipient in range(4)]
        if len(set(column)) != 1:
            raise RuntimeError(f"Gaussian installed cache changed across recipients: {state_id}/b{source_index}")
    if len({cache_hashes[0][source] for source in range(4)}) != 4:
        raise RuntimeError(f"Gaussian sources did not make four unique caches: {state_id}")
    diagonal_error = float(np.max(np.abs(actions[np.arange(4), np.arange(4)] - prior_gaussian)))
    if diagonal_error != 0.0:
        raise RuntimeError(f"Gaussian diagonal aggregate replay failed: {state_id}")

    state_staging = staging / f".{state_id}.tmp"
    state_output = staging / state_id
    state_staging.mkdir(parents=True, exist_ok=False)
    torch.save({
        "gaussian_futures": torch.stack(source["controls"]),
        "gaussian_seeds": list(GAUSSIAN_SEEDS),
        "source_branch_ids": list(BRANCH_IDS),
        "source_future_file_sha256": source["future_file_hashes"],
        "source_future_tensor_sha256": [tensor_hash(value) for value in source["futures"]],
        "gaussian_future_tensor_sha256": source["control_hashes"],
        "norm_match_metrics": source["norm_match_metrics"],
    }, state_staging / "gaussian_futures.pt")
    with (state_staging / "actions.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            gaussian_grid_actions=actions,
            gaussian_grid_executed_actions=core_runner.executed_action_view(actions),
            existing_gaussian_diagonal_actions=prior_gaussian,
        )
    result = {
        "schema_version": 1, "status": "complete", "state_id": state_id,
        "task_id": int(record["task_id"]), "initial_state_index": int(record["initial_state_index"]),
        "grid_axis_0": "recipient_action_noise_source",
        "grid_axis_1": "norm_matched_gaussian_source_from_native_future_branch",
        "branch_ids": list(BRANCH_IDS), "gaussian_seeds": list(GAUSSIAN_SEEDS),
        "action_noise_hashes": source["action_noise_hashes"],
        "grid_action_noise_hashes": [[source["action_noise_hashes"][recipient]] * 4 for recipient in range(4)],
        "source_future_tensor_sha256": [tensor_hash(value) for value in source["futures"]],
        "gaussian_future_tensor_sha256": source["control_hashes"],
        "norm_match_tolerance": MATCH_TOLERANCE,
        "norm_match_metrics": source["norm_match_metrics"],
        "grid_installed_cache_sha256": cache_hashes,
        "cache_unique_by_source": 4,
        "cache_exact_across_recipients": True,
        "existing_gaussian_diagonal_bitwise_equal": True,
        "existing_gaussian_diagonal_max_abs_error": diagonal_error,
        "fail_fast_native_replay_executed": run_replay_gate,
        "fail_fast_gaussian_replay_max_abs_error": first_cell_replay_error,
        "action_coordinate_intervention": "none",
        "intervention_target": "predicted-video latent/cache only",
        "core_actions_sha256": sha256_file(core_actions_path),
        "gaussian_futures_sha256": sha256_file(state_staging / "gaussian_futures.pt"),
        "actions_sha256": sha256_file(state_staging / "actions.npz"),
        "duration_seconds": time.time() - state_started,
    }
    write_json(state_staging / "result.json", result)
    os.replace(state_staging, state_output)
    print(f"complete {state_id} {result['duration_seconds']:.1f}s", flush=True)
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.shard_count != 2 or args.shard_index not in (0, 1):
        raise RuntimeError("confirmatory Gaussian grid requires exactly two 15-state shards")
    for name in ("lingbot_root", "checkpoint", "checkpoint_manifest", "oracle_receipt", "shim", "manifest", "core_root", "core_runner", "dose_validator", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_root.exists():
        raise FileExistsError(f"refusing overwrite: {args.output_root}")
    if args.output_root == args.core_root or args.output_root.is_relative_to(args.core_root) or args.core_root.is_relative_to(args.output_root):
        raise RuntimeError("Gaussian output and core root must be disjoint")
    exact = {
        args.manifest: MANIFEST_SHA256, args.core_root / "manifest.json": MANIFEST_SHA256,
        args.core_runner: CORE_RUNNER_SHA256, args.dose_validator: DOSE_VALIDATOR_SHA256,
        args.checkpoint_manifest: CHECKPOINT_MANIFEST_SHA256,
        args.oracle_receipt: ORACLE_RECEIPT_SHA256, args.shim: SHIM_SHA256,
    }
    for path, expected in exact.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"identity mismatch: {path}")
    if args.checkpoint != EXPECTED_CHECKPOINT.resolve():
        raise RuntimeError("checkpoint path differs from canonical deployment")
    commit = subprocess.run(["git", "-C", str(args.lingbot_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError("upstream commit mismatch")
    tracked_status = subprocess.run(
        ["git", "-C", str(args.lingbot_root), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if tracked_status:
        raise RuntimeError(f"upstream tracked worktree is dirty: {tracked_status}")
    if sha256_file(args.lingbot_root / "wan_va/wan_va_server.py") != OFFICIAL_SERVER_SHA256:
        raise RuntimeError("official VA server source changed")
    manifest = load_json(args.manifest)
    records = [record for record in manifest["states"] if record["admission"] == "evaluation"]
    if len(records) != 30 or tuple(manifest["branch_ids"]) != BRANCH_IDS:
        raise RuntimeError("frozen cohort/branches changed")
    core_before = inventory(args.core_root)

    dose_validator = load_module("canonical_dose_validator", args.dose_validator)
    core_metadata = dose_validator.validate_core(
        records=records, core_root=args.core_root, manifest_sha256=MANIFEST_SHA256, manifest=manifest,
    )
    checkpoint = dose_validator.verify_checkpoint_content(args.checkpoint, args.checkpoint_manifest)
    if checkpoint.get("aggregate_sha256") != CHECKPOINT_AGGREGATE_SHA256:
        raise RuntimeError("checkpoint aggregate changed")
    oracle = dose_validator.validate_oracle(args.oracle_receipt)
    if oracle.get("parity_gate_passed") is not True:
        raise RuntimeError("exact upstream parity gate absent")
    core_runner = load_module("canonical_core_runner", args.core_runner)
    prepared = validate_all_sources(
        records=records, manifest=manifest, core_root=args.core_root,
        core_metadata=core_metadata, core_runner=core_runner,
    )
    assigned = records[args.shard_index :: args.shard_count]
    if len(assigned) != 15:
        raise RuntimeError("Gaussian shard is not exactly 15 states")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent))
    try:
        sys.path.insert(0, str(args.lingbot_root))
        sys.path.insert(0, str(args.lingbot_root / "wan_va"))
        from wan_va.configs import VA_CONFIGS
        from wan_va.distributed.util import init_distributed
        from wan_va.wan_va_server import VA_Server
        import flash_attn
        if Path(flash_attn.__file__).resolve() != args.shim or sha256_file(Path(flash_attn.__file__)) != SHIM_SHA256:
            raise RuntimeError("actual imported flash_attn shim mismatch")
        rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", "0")); world_size = int(os.environ.get("WORLD_SIZE", "1"))
        init_distributed(world_size, local_rank, rank)
        config = copy.deepcopy(VA_CONFIGS["libero"])
        if config.video_exec_step != -1:
            raise RuntimeError("full future-cache install required")
        config.wan22_pretrained_model_name_or_path = str(args.checkpoint)
        config.local_rank = local_rank; config.rank = rank; config.world_size = world_size
        config.enable_offload = False; config.save_root = str(staging / "upstream_debug")
        server = VA_Server(config)
        results = []
        for index, record in enumerate(assigned):
            results.append(run_state(
                server, core_runner, record, prepared[record["state_id"]],
                args.core_root, staging, run_replay_gate=index == 0,
            ))
        if inventory(args.core_root) != core_before:
            raise RuntimeError("core raw tree changed during Gaussian run")
        shutil.copy2(Path(__file__).resolve(), staging / "run_lingbot_gaussian_source_grid.py")
        shutil.copy2(args.manifest, staging / "manifest.json")
        provenance = {
            "schema_version": 1, "status": "complete_mode_frozen_read_only_shard",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(args.output_root), "shard_index": args.shard_index,
            "shard_count": args.shard_count, "state_count": len(results),
            "state_ids": [record["state_id"] for record in assigned],
            "duration_seconds": time.time() - started,
            "design": "complete 4 recipient action-noise sources x 4 norm-matched Gaussian future sources per state",
            "intervention_cells_per_state": 16,
            "extra_fail_fast_action_trajectories_first_state": 2,
            "all_generated_control_latents_saved": True,
            "action_coordinate_intervention": "none",
            "identities": {
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "manifest_sha256": MANIFEST_SHA256, "core_runner_sha256": CORE_RUNNER_SHA256,
                "dose_validator_sha256": DOSE_VALIDATOR_SHA256, "upstream_commit": UPSTREAM_COMMIT,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
                "checkpoint_aggregate_sha256": CHECKPOINT_AGGREGATE_SHA256,
                "oracle_receipt_sha256": ORACLE_RECEIPT_SHA256,
                "shim_sha256": SHIM_SHA256, "pythonpath": os.environ.get("PYTHONPATH"),
                "core_tree_aggregate_sha256": core_before[1],
            },
        }
        write_json(staging / "provenance.json", provenance)
        rows, aggregate = inventory(staging, exclude_index=True)
        index = {"schema_version": 1, "status": "complete_mode_frozen_read_only_shard", "file_count_excluding_index": len(rows), "tree_aggregate_sha256": aggregate, "files": rows}
        write_json(staging / "artifact_index.json", index)
        for path in staging.rglob("*"):
            if path.is_file(): path.chmod(0o444)
        for path in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True): path.chmod(0o555)
        staging.chmod(0o555)
        if inventory(staging, exclude_index=True) != (rows, aggregate):
            raise RuntimeError("Gaussian shard changed during freeze")
        if any(stat.S_IMODE(p.stat().st_mode) != 0o444 for p in staging.rglob("*") if p.is_file()):
            raise RuntimeError("Gaussian shard has mutable files")
        if stat.S_IMODE(staging.stat().st_mode) != 0o555 or any(stat.S_IMODE(p.stat().st_mode) != 0o555 for p in staging.rglob("*") if p.is_dir()):
            raise RuntimeError("Gaussian shard has mutable directories")
        os.replace(staging, args.output_root)
        print(json.dumps({"status": index["status"], "output_root": str(args.output_root), "state_count": len(results), "artifact_index_sha256": sha256_file(args.output_root / "artifact_index.json"), "tree_aggregate_sha256": aggregate, "duration_seconds": time.time() - started}, sort_keys=True), flush=True)
    except BaseException:
        if staging.exists():
            for path in staging.rglob("*"):
                try: path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError: pass
            staging.chmod(0o755); shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
