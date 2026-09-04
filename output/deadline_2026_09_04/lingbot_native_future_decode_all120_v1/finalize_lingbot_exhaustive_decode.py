#!/usr/bin/env python3
"""Verify two LingBot decode shards and build exhaustive, selection-neutral media."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DECODER_SHA256 = "4a1335ac0392f1dd86af13d067476eaae41588915a9a64b1d87eb3468dbad0be"
MANIFEST_SHA256 = "9c8d7dd547955e4622d98c7eb42fffed157f6db983f41f02929b28a434be47b4"
CORE_RUNNER_SHA256 = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
SERVER_SOURCE_SHA256 = "9c2a427611db487fea5cf40f184b713bf2088e533990ee00fdcd020d2668b4bf"
UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"
CHECKPOINT_MANIFEST_SHA256 = "b1c8722becbb4a77a840cb5716cbd57e68c5bd77cb4a0af3d19a6e9ee1de00fd"
CHECKPOINT_AGGREGATE_SHA256 = "bb895755e071bf5ab74494c07199a11c8e344b367971b4c6405321807e32b2e1"
VAE_CONFIG_SHA256 = "d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360"
VAE_PAYLOAD_SHA256 = "62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b"
MODULE_UTILS_SHA256 = "563499ae1aa9fdbe990963c8c133423241f1a63d3f59743b778a1eb4c47757f4"
SHIM_SHA256 = "7f1448bdeae5f4991112d78131688d417836c91fee79624929cda5d2f135bec8"
BRANCH_IDS = ("b0", "b1", "b2", "b3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--shard-0", type=Path, required=True)
    parser.add_argument("--shard-1", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
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
        row = {"path": relative, "bytes": size, "sha256": digest}
        rows.append(row)
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    return rows, aggregate.hexdigest()


def validate_index(root: Path) -> tuple[dict[str, Any], str]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"missing/unsafe decode shard: {root}")
    index_path = root / "artifact_index.json"
    index = load_json(index_path)
    if index.get("status") != "complete_mode_frozen_read_only_shard":
        raise RuntimeError(f"incomplete decode shard: {root}")
    for row in index.get("files", []):
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"changed shard artifact: {path}")
    rows, aggregate = inventory(root, exclude_index=True)
    if rows != index["files"] or aggregate != index["tree_aggregate_sha256"]:
        raise RuntimeError(f"shard recursive inventory mismatch: {root}")
    if stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise RuntimeError(f"decode shard root is not mode 0555: {root}")
    if any(stat.S_IMODE(path.stat().st_mode) != 0o444 for path in root.rglob("*") if path.is_file()):
        raise RuntimeError(f"decode shard has a mutable file: {root}")
    if any(stat.S_IMODE(path.stat().st_mode) != 0o555 for path in root.rglob("*") if path.is_dir()):
        raise RuntimeError(f"decode shard has a mutable directory: {root}")
    return index, sha256_file(index_path)


def render_label(frame: np.ndarray, text: str) -> np.ndarray:
    value = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(value)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 8
    height = box[3] - box[1] + 6
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    shards = (args.shard_0.resolve(), args.shard_1.resolve())
    if shards[0] == shards[1] or output in shards or any(output.is_relative_to(root) for root in shards):
        raise RuntimeError("output must be disjoint from two distinct source shards")
    if sha256_file(args.manifest) != MANIFEST_SHA256:
        raise RuntimeError("manifest identity mismatch")
    manifest = load_json(args.manifest)
    core = args.core_root.resolve()
    if sha256_file(core / "manifest.json") != MANIFEST_SHA256:
        raise RuntimeError("current core root manifest identity mismatch")
    states = [record for record in manifest["states"] if record["admission"] == "evaluation"]
    state_ids = [record["state_id"] for record in states]
    if len(state_ids) != 30 or len(set(state_ids)) != 30 or tuple(manifest["branch_ids"]) != BRANCH_IDS:
        raise RuntimeError("frozen cohort/branch schema mismatch")
    expected = [(state_id, branch) for state_id in state_ids for branch in BRANCH_IDS]

    source_indexes = []
    source_snapshots = []
    combined: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for shard_index, root in enumerate(shards):
        index, index_sha = validate_index(root)
        provenance = load_json(root / "provenance.json")
        policy = load_json(root / "selection_policy.json")
        identities = provenance.get("identities", {})
        checks = {
            "status": "complete_mode_frozen_read_only_shard",
            "decoded_item_count": 60,
        }
        bad = {key: (provenance.get(key), value) for key, value in checks.items() if provenance.get(key) != value}
        identity_checks = {
            "manifest_sha256": MANIFEST_SHA256,
            "core_runner_sha256": CORE_RUNNER_SHA256,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_content_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "checkpoint_aggregate_sha256": CHECKPOINT_AGGREGATE_SHA256,
            "vae_config_sha256": VAE_CONFIG_SHA256,
            "official_server_source_sha256": SERVER_SOURCE_SHA256,
            "official_module_utils_sha256": MODULE_UTILS_SHA256,
            "vae_payload_sha256": VAE_PAYLOAD_SHA256,
            "flash_attn_import_shim_sha256": SHIM_SHA256,
            "decoder_script_sha256": DECODER_SHA256,
        }
        bad.update({f"identity.{key}": (identities.get(key), value) for key, value in identity_checks.items() if identities.get(key) != value})
        if bad:
            raise RuntimeError(f"decode provenance mismatch shard {shard_index}: {bad}")
        if (
            policy.get("engineering_smoke") is not False
            or policy.get("selection_uses_outcomes") is not False
            or policy.get("shard_index") != shard_index
            or policy.get("shard_count") != 2
            or policy.get("decoded_item_count") != 60
        ):
            raise RuntimeError(f"decode policy mismatch shard {shard_index}")
        records = provenance.get("records", [])
        if len(records) != 60:
            raise RuntimeError(f"decode record count mismatch shard {shard_index}")
        for record in records:
            key = (record["state_id"], record["branch_id"])
            if key in combined or key not in expected:
                raise RuntimeError(f"duplicate/unexpected decode key: {key}")
            item_root = root / "decoded" / key[0] / key[1]
            item_receipt = load_json(item_root / "receipt.json")
            if item_receipt != record:
                raise RuntimeError(f"item receipt/provenance mismatch: {key}")
            raw_path = item_root / "decoded_frames.npz"
            mp4_path = item_root / "decoded_video.mp4"
            if sha256_file(raw_path) != record["decoded_npz_sha256"] or sha256_file(mp4_path) != record["decoded_mp4_sha256"]:
                raise RuntimeError(f"decoded payload mismatch: {key}")
            frames = np.load(raw_path, allow_pickle=False)["frames"]
            if tuple(frames.shape) != (13, 128, 256, 3) or frames.dtype != np.float32 or not np.isfinite(frames).all():
                raise RuntimeError(f"decoded array schema mismatch: {key}")
            if array_hash(frames) != record["decoded_array_sha256"]:
                raise RuntimeError(f"decoded array hash mismatch: {key}")
            core_future_path = core / record["core_future_relative_path"]
            core_result = load_json(core / key[0] / "result.json")
            if (
                sha256_file(core_future_path) != record["core_future_file_sha256"]
                or core_result.get("actions_sha256") != record["core_actions_sha256"]
                or sha256_file(core / key[0] / "actions.npz") != record["core_actions_sha256"]
            ):
                raise RuntimeError(f"current core future/action binding mismatch: {key}")
            combined[key] = (record, raw_path)
        source_indexes.append({"root": str(root), "artifact_index_sha256": index_sha, "tree_aggregate_sha256": index["tree_aggregate_sha256"]})
        source_snapshots.append(inventory(root))
    if list(sorted(combined, key=lambda key: expected.index(key))) != expected:
        raise RuntimeError("combined shards are not exactly the frozen 30x4 Cartesian product")
    future_hashes = [combined[key][0]["core_future_tensor_sha256"] for key in expected]
    if len(set(future_hashes)) != 120:
        raise RuntimeError("120 native future tensors are not globally unique")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        from diffusers.utils import export_to_video

        media = staging / "media"
        media.mkdir(parents=True)
        all_video_frames: list[np.ndarray] = []
        terminal_rows: list[np.ndarray] = []
        rows_json: list[dict[str, Any]] = []
        for state_index, state_id in enumerate(state_ids):
            branch_frames = []
            terminal_tiles = []
            for branch in BRANCH_IDS:
                record, raw_path = combined[(state_id, branch)]
                frames = np.load(raw_path, allow_pickle=False)["frames"]
                labeled = np.stack([render_label(frame, f"{state_id} {branch}") for frame in frames])
                branch_frames.append(labeled)
                terminal_tiles.append(labeled[-1])
                rows_json.append({**record, "source_decoded_npz": str(raw_path), "source_decoded_mp4": str(raw_path.with_name("decoded_video.mp4"))})
            top = np.concatenate((branch_frames[0], branch_frames[1]), axis=2)
            bottom = np.concatenate((branch_frames[2], branch_frames[3]), axis=2)
            state_video = np.concatenate((top, bottom), axis=1)
            all_video_frames.extend(state_video)
            terminal_rows.append(np.concatenate(terminal_tiles, axis=1))
        global_video_path = media / "all_30_states_all_4_branches.mp4"
        export_to_video(np.stack(all_video_frames), str(global_video_path), fps=10)
        if not global_video_path.is_file() or global_video_path.stat().st_size == 0:
            raise RuntimeError("overview video export failed")
        contact = np.concatenate(terminal_rows, axis=0)
        contact_path = media / "all_120_terminal_frames_contact_sheet.png"
        Image.fromarray(contact).save(contact_path, format="PNG", optimize=False)
        with Image.open(contact_path) as reopened:
            reopened.verify()
        with Image.open(contact_path) as reopened:
            if reopened.size != (1024, 3840) or reopened.mode != "RGB":
                raise RuntimeError(f"contact-sheet schema changed: {reopened.size}/{reopened.mode}")
        write_json(staging / "all_120_decodes.json", {
            "schema_version": 1,
            "status": "complete",
            "item_count": 120,
            "records": rows_json,
        })
        csv_path = staging / "all_120_decodes.csv"
        fields = [
            "state_id", "task_id", "initial_state_index", "branch_id", "branch_index", "video_seed",
            "core_future_file_sha256", "core_future_tensor_sha256", "decoded_array_sha256",
            "decoded_npz_sha256", "decoded_mp4_sha256", "source_decoded_npz", "source_decoded_mp4",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_json)
        policy = {
            "schema_version": 1,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "timing": "postanalysis visualization after the complete core and dose analyses",
            "selection_rule": "all 30 frozen evaluation states x all four native branches, in frozen manifest order; every one of 13 decoded frames is included in the overview video; contact sheet uses the predetermined terminal decoded frame for every item",
            "selection_uses_outcomes": False,
            "selection_uses_visual_appearance": False,
            "selection_uses_semantics": False,
            "included_in_core_inference": False,
            "item_count": 120,
            "overview_layout": "2x2 branches b0,b1 / b2,b3; 30 states concatenated in manifest order; 390 frames at 10 fps",
            "contact_sheet_layout": "30 manifest-order state rows x four branch columns; terminal decoded frame",
            "interpretation_limit": "Decoded images are postanalysis visualization of latent tensors, not an intervention, selection criterion, or proof of semantic distinctness.",
        }
        write_json(staging / "selection_policy.json", policy)
        receipt = {
            "schema_version": 1,
            "status": "complete_mode_frozen_read_only_umbrella",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output),
            "item_count": 120,
            "state_count": 30,
            "branch_count": 4,
            "source_shards": source_indexes,
            "bindings": {
                "manifest_sha256": MANIFEST_SHA256,
                "core_runner_sha256": CORE_RUNNER_SHA256,
                "decoder_sha256": DECODER_SHA256,
                "official_server_source_sha256": SERVER_SOURCE_SHA256,
                "vae_payload_sha256": VAE_PAYLOAD_SHA256,
                "finalizer_sha256": sha256_file(Path(__file__).resolve()),
            },
            "media": {
                "overview_video": "media/all_30_states_all_4_branches.mp4",
                "overview_video_sha256": sha256_file(global_video_path),
                "contact_sheet": "media/all_120_terminal_frames_contact_sheet.png",
                "contact_sheet_sha256": sha256_file(contact_path),
            },
        }
        write_json(staging / "provenance.json", receipt)
        shutil.copy2(Path(__file__).resolve(), staging / "finalize_lingbot_exhaustive_decode.py")
        for root, before in zip(shards, source_snapshots, strict=True):
            if inventory(root) != before:
                raise RuntimeError(f"decode shard changed during finalization: {root}")
        rows, aggregate = inventory(staging, exclude_index=True)
        index = {
            "schema_version": 1,
            "status": "complete_mode_frozen_read_only_umbrella",
            "file_count_excluding_index": len(rows),
            "tree_aggregate_sha256": aggregate,
            "files": rows,
        }
        write_json(staging / "artifact_index.json", index)
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            path.chmod(0o555)
        staging.chmod(0o555)
        frozen_rows, frozen_aggregate = inventory(staging, exclude_index=True)
        if frozen_rows != rows or frozen_aggregate != aggregate:
            raise RuntimeError("umbrella changed during freeze")
        if any(stat.S_IMODE(p.stat().st_mode) != 0o444 for p in staging.rglob("*") if p.is_file()):
            raise RuntimeError("not every umbrella file is 0444")
        if stat.S_IMODE(staging.stat().st_mode) != 0o555 or any(stat.S_IMODE(p.stat().st_mode) != 0o555 for p in staging.rglob("*") if p.is_dir()):
            raise RuntimeError("not every umbrella directory is 0555")
        os.replace(staging, output)
        print(json.dumps({
            "status": index["status"], "output_root": str(output), "item_count": 120,
            "artifact_index_sha256": sha256_file(output / "artifact_index.json"),
            "tree_aggregate_sha256": aggregate,
            "overview_video_sha256": receipt["media"]["overview_video_sha256"],
            "contact_sheet_sha256": receipt["media"]["contact_sheet_sha256"],
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
