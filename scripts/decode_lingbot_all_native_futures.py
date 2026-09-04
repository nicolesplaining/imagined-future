#!/usr/bin/env python3
"""Exhaustively decode frozen LingBot native futures with the official VAE path."""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
MANIFEST_SHA256 = "9c8d7dd547955e4622d98c7eb42fffed157f6db983f41f02929b28a434be47b4"
CORE_RUNNER_SHA256 = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"
CHECKPOINT_MANIFEST_SHA256 = "b1c8722becbb4a77a840cb5716cbd57e68c5bd77cb4a0af3d19a6e9ee1de00fd"
CHECKPOINT_AGGREGATE_SHA256 = "bb895755e071bf5ab74494c07199a11c8e344b367971b4c6405321807e32b2e1"
VAE_CONFIG_SHA256 = "d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360"
VAE_PAYLOAD_SHA256 = "62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b"
SERVER_SOURCE_SHA256 = "9c2a427611db487fea5cf40f184b713bf2088e533990ee00fdcd020d2668b4bf"
MODULE_UTILS_SHA256 = "563499ae1aa9fdbe990963c8c133423241f1a63d3f59743b778a1eb4c47757f4"
SHIM_SHA256 = "7f1448bdeae5f4991112d78131688d417836c91fee79624929cda5d2f135bec8"
BRANCH_IDS = ("b0", "b1", "b2", "b3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--engineering-smoke", action="store_true")
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


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"source symlink forbidden: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    return rows, aggregate.hexdigest()


def output_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"output symlink forbidden: {path}")
        if not path.is_file() or path == root / "artifact_index.json":
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    return rows, aggregate.hexdigest()


def verify_environment(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    commit = subprocess.run(
        ["git", "-C", str(args.lingbot_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {commit}")
    server_source = args.lingbot_root / "wan_va/wan_va_server.py"
    module_utils = args.lingbot_root / "wan_va/modules/utils.py"
    exact_files = {
        server_source: SERVER_SOURCE_SHA256,
        module_utils: MODULE_UTILS_SHA256,
        args.shim: SHIM_SHA256,
        args.checkpoint_manifest: CHECKPOINT_MANIFEST_SHA256,
        args.checkpoint / "vae/config.json": VAE_CONFIG_SHA256,
        args.checkpoint / "vae/diffusion_pytorch_model.safetensors": VAE_PAYLOAD_SHA256,
    }
    for path, expected in exact_files.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"identity mismatch for {path}: {actual} != {expected}")
    checkpoint_manifest = load_json(args.checkpoint_manifest)
    if checkpoint_manifest.get("aggregate_sha256") != CHECKPOINT_AGGREGATE_SHA256:
        raise RuntimeError("checkpoint aggregate identity changed")
    expected_vae = {
        "vae/config.json": VAE_CONFIG_SHA256,
        "vae/diffusion_pytorch_model.safetensors": VAE_PAYLOAD_SHA256,
    }
    manifest_rows = {row["path"]: row for row in checkpoint_manifest["files"]}
    for relative, expected in expected_vae.items():
        if manifest_rows.get(relative, {}).get("sha256") != expected:
            raise RuntimeError(f"VAE payload not bound by checkpoint manifest: {relative}")
        path = args.checkpoint / relative
        if manifest_rows[relative]["bytes"] != path.stat().st_size:
            raise RuntimeError(f"VAE payload size changed: {relative}")
    core_rows, core_aggregate = source_inventory(args.core_root)
    if len(core_rows) != 211:
        raise RuntimeError(f"core raw census {len(core_rows)} != 211")
    return checkpoint_manifest, core_rows, core_aggregate


def validate_and_enumerate(args: argparse.Namespace) -> tuple[dict[str, Any], list[tuple[dict[str, Any], int]]]:
    if sha256_file(args.manifest) != MANIFEST_SHA256:
        raise RuntimeError("frozen manifest SHA mismatch")
    if sha256_file(args.core_root / "manifest.json") != MANIFEST_SHA256:
        raise RuntimeError("core manifest SHA mismatch")
    manifest = load_json(args.manifest)
    if tuple(manifest.get("branch_ids", [])) != BRANCH_IDS:
        raise RuntimeError("branch identity changed")
    states = [record for record in manifest["states"] if record["admission"] == "evaluation"]
    if len(states) != 30 or len({r["state_id"] for r in states}) != 30:
        raise RuntimeError("expected exactly 30 unique evaluation states")
    items = [(record, branch) for record in states for branch in range(4)]
    items = items[args.shard_index :: args.shard_count]
    if args.max_items is not None:
        items = items[: args.max_items]
    return manifest, items


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("max-items must be positive")
    if not args.engineering_smoke and args.shard_count not in (1, 2):
        raise RuntimeError("exhaustive decode permits only one or two shards")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    core = args.core_root.resolve()
    if output == core or output.is_relative_to(core) or core.is_relative_to(output):
        raise RuntimeError("output and core raw root must be disjoint")
    checkpoint_manifest, core_rows_before, core_aggregate_before = verify_environment(args)
    manifest, items = validate_and_enumerate(args)
    expected_count = len(items)
    if not args.engineering_smoke and args.max_items is not None:
        raise RuntimeError("max-items is allowed only for excluded engineering smoke")
    if not args.engineering_smoke and args.shard_count == 2 and expected_count != 60:
        raise RuntimeError(f"two-way exhaustive shard must contain 60 items, got {expected_count}")
    if not args.engineering_smoke and args.shard_count == 1 and expected_count != 120:
        raise RuntimeError(f"unsharded exhaustive run must contain 120 items, got {expected_count}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        sys.path.insert(0, str(args.lingbot_root / "wan_va"))
        from diffusers.utils import export_to_video
        from diffusers.video_processor import VideoProcessor
        from modules.utils import load_vae
        from wan_va_server import VA_Server
        import flash_attn

        imported_shim = Path(flash_attn.__file__).resolve()
        if imported_shim != args.shim.resolve():
            raise RuntimeError(
                f"unexpected flash_attn import {imported_shim}; expected {args.shim.resolve()}"
            )
        if sha256_file(imported_shim) != SHIM_SHA256:
            raise RuntimeError("imported flash_attn shim changed")

        decode_source = inspect.getsource(VA_Server.decode_one_video)
        if "self.vae.decode(latents, return_dict=False)[0]" not in decode_source:
            raise RuntimeError("official decode method body changed")
        device = torch.device("cuda", torch.cuda.current_device())
        vae = load_vae(
            str(args.checkpoint / "vae"),
            torch_dtype=torch.bfloat16,
            torch_device=device,
        )
        vae.eval()
        holder = SimpleNamespace(vae=vae, video_processor=VideoProcessor(vae_scale_factor=1))
        records: list[dict[str, Any]] = []
        for item_index, (record, branch) in enumerate(items):
            state_id = record["state_id"]
            branch_id = BRANCH_IDS[branch]
            state_root = core / state_id
            result = load_json(state_root / "result.json")
            checks = {
                "status": "complete",
                "state_id": state_id,
                "admission": "evaluation",
                "input_sha256": record["input_sha256"],
                "prompt": record["prompt"],
                "branch_ids": list(BRANCH_IDS),
                "video_seeds": manifest["video_seeds"],
                "action_seeds": manifest["action_seeds"],
                "manifest_sha256": MANIFEST_SHA256,
                "runner_sha256": CORE_RUNNER_SHA256,
                "upstream_commit": UPSTREAM_COMMIT,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "action_coordinate_intervention": "none",
            }
            bad = {key: (result.get(key), value) for key, value in checks.items() if result.get(key) != value}
            if bad:
                raise RuntimeError(f"core identity mismatch {state_id}: {bad}")
            actions_path = state_root / "actions.npz"
            if result.get("actions_sha256") != sha256_file(actions_path):
                raise RuntimeError(f"core action payload changed: {state_id}")
            future_path = state_root / f"future_{branch_id}.pt"
            payload = torch.load(future_path, map_location="cpu", weights_only=False)
            future = payload.get("future")
            expected_shape = (1, 48, 4, 8, 16)
            if not isinstance(future, torch.Tensor) or tuple(future.shape) != expected_shape or future.dtype != torch.bfloat16:
                raise RuntimeError(f"invalid future tensor {state_id}/{branch_id}")
            if not bool(torch.isfinite(future.float()).all().item()):
                raise RuntimeError(f"nonfinite future tensor {state_id}/{branch_id}")
            if payload.get("video_seed") != manifest["video_seeds"][branch]:
                raise RuntimeError(f"future seed mismatch {state_id}/{branch_id}")
            future_tensor_sha = tensor_hash(future)
            if future_tensor_sha != result["future_hashes"][branch]:
                raise RuntimeError(f"future tensor hash mismatch {state_id}/{branch_id}")
            item_started = time.time()
            # Upstream calls this method from its @torch.no_grad generate path;
            # preserve that official call context when invoking it unbound.
            with torch.no_grad():
                decoded = VA_Server.decode_one_video(holder, future.to(device), "np")
            if not isinstance(decoded, np.ndarray) or decoded.shape[0] != 1:
                raise RuntimeError(f"unexpected official decode output {type(decoded)} {getattr(decoded, 'shape', None)}")
            frames = np.ascontiguousarray(decoded[0])
            if (
                tuple(frames.shape) != (13, 128, 256, 3)
                or frames.dtype != np.float32
                or not np.isfinite(frames).all()
                or float(frames.min()) < 0.0
                or float(frames.max()) > 1.0
            ):
                raise RuntimeError(f"invalid decoded frames {state_id}/{branch_id}: {frames.shape}")
            item_root = staging / "decoded" / state_id / branch_id
            item_root.mkdir(parents=True, exist_ok=False)
            raw_path = item_root / "decoded_frames.npz"
            with raw_path.open("wb") as handle:
                np.savez_compressed(handle, frames=frames)
            mp4_path = item_root / "decoded_video.mp4"
            export_to_video(frames, str(mp4_path), fps=10)
            if not mp4_path.is_file() or mp4_path.stat().st_size == 0:
                raise RuntimeError(f"video export failed {state_id}/{branch_id}")
            item_receipt = {
                "status": "complete",
                "state_id": state_id,
                "task_id": int(record["task_id"]),
                "initial_state_index": int(record["initial_state_index"]),
                "branch_id": branch_id,
                "branch_index": branch,
                "video_seed": int(payload["video_seed"]),
                "core_future_relative_path": future_path.relative_to(core).as_posix(),
                "core_future_file_sha256": sha256_file(future_path),
                "core_future_tensor_sha256": future_tensor_sha,
                "core_actions_sha256": result["actions_sha256"],
                "decoded_array_shape": list(frames.shape),
                "decoded_array_dtype": str(frames.dtype),
                "decoded_array_sha256": array_hash(frames),
                "decoded_npz_sha256": sha256_file(raw_path),
                "decoded_mp4_sha256": sha256_file(mp4_path),
                "fps": 10,
                "official_decode_entrypoint": "VA_Server.decode_one_video(latents, 'np')[0]",
                "official_server_source_sha256": SERVER_SOURCE_SHA256,
                "action_coordinate_intervention": "none",
                "included_in_core_inference": False,
                "duration_seconds": time.time() - item_started,
            }
            receipt_path = item_root / "receipt.json"
            write_json(receipt_path, item_receipt)
            records.append(item_receipt)
            print(f"decoded {item_index + 1}/{expected_count} {state_id}/{branch_id} {item_receipt['duration_seconds']:.2f}s", flush=True)

        core_rows_after, core_aggregate_after = source_inventory(core)
        if core_rows_after != core_rows_before or core_aggregate_after != core_aggregate_before:
            raise RuntimeError("core raw tree changed during postanalysis decoding")
        policy = {
            "schema_version": 1,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "engineering_smoke": bool(args.engineering_smoke),
            "included_in_core_inference": False,
            "timing": "postanalysis visualization after the complete core and dose analyses",
            "selection_rule": "Cartesian product of all 30 frozen evaluation states and all four native future branches, in frozen manifest order; no effect, appearance, semantic, or outcome selection",
            "selection_uses_outcomes": False,
            "decode_scope": "exhaustive native futures" if not args.engineering_smoke else "excluded engineering smoke",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "expected_item_count": expected_count,
            "decoded_item_count": len(records),
            "state_ids": sorted({record["state_id"] for record in records}),
            "branch_ids": list(BRANCH_IDS),
            "action_coordinate_intervention": "none",
        }
        write_json(staging / "selection_policy.json", policy)
        provenance = {
            "schema_version": 1,
            "status": "complete_mode_frozen_read_only_shard",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output),
            "duration_seconds": time.time() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "identities": {
                "manifest_sha256": MANIFEST_SHA256,
                "core_runner_sha256": CORE_RUNNER_SHA256,
                "upstream_commit": UPSTREAM_COMMIT,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "checkpoint_content_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
                "checkpoint_aggregate_sha256": CHECKPOINT_AGGREGATE_SHA256,
                "vae_config_sha256": VAE_CONFIG_SHA256,
                "vae_payload_sha256": VAE_PAYLOAD_SHA256,
                "official_server_source_sha256": SERVER_SOURCE_SHA256,
                "official_module_utils_sha256": MODULE_UTILS_SHA256,
                "flash_attn_import_shim_sha256": SHIM_SHA256,
                "flash_attn_imported_path": str(imported_shim),
                "pythonpath": os.environ.get("PYTHONPATH"),
                "decoder_script_sha256": sha256_file(Path(__file__).resolve()),
                "core_source_tree_aggregate_sha256": core_aggregate_before,
            },
            "official_decode_entrypoint": "VA_Server.decode_one_video(latents, 'np')[0]",
            "decoded_item_count": len(records),
            "records": records,
        }
        write_json(staging / "provenance.json", provenance)
        shutil.copy2(Path(__file__).resolve(), staging / "decode_lingbot_all_native_futures.py")
        rows, aggregate = output_inventory(staging)
        index = {
            "schema_version": 1,
            "status": "complete_mode_frozen_read_only_shard",
            "file_count_excluding_index": len(rows),
            "tree_aggregate_sha256": aggregate,
            "files": rows,
        }
        write_json(staging / "artifact_index.json", index)
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            path.chmod(0o555)
        staging.chmod(0o555)
        frozen_rows, frozen_aggregate = output_inventory(staging)
        if frozen_rows != rows or frozen_aggregate != aggregate:
            raise RuntimeError("decode output changed during freeze")
        if any(stat.S_IMODE(p.stat().st_mode) != 0o444 for p in staging.rglob("*") if p.is_file()):
            raise RuntimeError("not every decode artifact is 0444")
        os.replace(staging, output)
        print(json.dumps({
            "status": index["status"],
            "output_root": str(output),
            "decoded_item_count": len(records),
            "artifact_index_sha256": sha256_file(output / "artifact_index.json"),
            "tree_aggregate_sha256": aggregate,
            "duration_seconds": time.time() - started,
        }, sort_keys=True), flush=True)
    except BaseException:
        if staging.exists():
            for path in staging.rglob("*"):
                try:
                    path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
            staging.chmod(0o755)
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
