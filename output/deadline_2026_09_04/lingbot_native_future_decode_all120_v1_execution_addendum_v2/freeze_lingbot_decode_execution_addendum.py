#!/usr/bin/env python3
"""Freeze the LingBot exhaustive-decode engineering and execution history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DECODER_SHA256 = "4a1335ac0392f1dd86af13d067476eaae41588915a9a64b1d87eb3468dbad0be"
SMOKE_V2_DECODER_SHA256 = "1785efb84aae3d7bf9de1b2b7ee3d30dfb18f2c0cff1da19f4b1e887c483fc63"
SMOKE_V1_DECODER_SHA256 = "45cb3b14f92aef451e10d434e5e1c025c43e3dc9afc82be73a439f1d8fe51548"
SMOKE_V2_INDEX_SHA256 = "30a5c2c41f1d026a2d7fcb29d848279ce4aef90d3e273b08229489088a0b820f"
SHARD_INDEX_SHA256 = (
    "d577b669f146e0a68b28809fec894ce73754e14e830611bd4fe7f6ce8aa3c8c0",
    "956efec68a8d7f9bba9376084bef14d60e04f4ba64a62f734b95750903d29b94",
)
FINAL_INDEX_SHA256 = "f89a96b3b12c35e25cc121284c84dc83de1edb2c8cfc28b3e0faadcaa6c3b332"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
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


def validate_package(root: Path, expected_index_sha: str, count: int, decoder_sha: str = DECODER_SHA256) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"missing/unsafe package: {root}")
    index_path = root / "artifact_index.json"
    if sha256_file(index_path) != expected_index_sha:
        raise RuntimeError(f"package index identity changed: {root}")
    index = load_json(index_path)
    rows, aggregate = inventory(root, exclude_index=True)
    if rows != index.get("files") or aggregate != index.get("tree_aggregate_sha256"):
        raise RuntimeError(f"package index content mismatch: {root}")
    if stat.S_IMODE(root.stat().st_mode) != 0o555 or any(
        stat.S_IMODE(path.stat().st_mode) != 0o444 for path in root.rglob("*") if path.is_file()
    ):
        raise RuntimeError(f"package is not immutable: {root}")
    provenance = load_json(root / "provenance.json")
    identities = provenance.get("identities", {})
    if identities.get("decoder_script_sha256") != decoder_sha:
        raise RuntimeError(f"decoder identity mismatch: {root}")
    if provenance.get("decoded_item_count") != count:
        raise RuntimeError(f"decoded item count mismatch: {root}")
    return index


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    smoke1_output = base / "lingbot_native_future_decode_smoke_v1"
    if smoke1_output.exists():
        raise RuntimeError("failed smoke unexpectedly installed an output root")
    smoke2 = base / "lingbot_native_future_decode_smoke_v2"
    shard_roots = tuple(base / f"lingbot_native_future_decode_all120_v1_shard{i}" for i in range(2))
    final = base / "lingbot_native_future_decode_all120_v1"
    validate_package(smoke2, SMOKE_V2_INDEX_SHA256, 1, SMOKE_V2_DECODER_SHA256)
    shard_indexes = [validate_package(root, SHARD_INDEX_SHA256[i], 60) for i, root in enumerate(shard_roots)]
    if sha256_file(final / "artifact_index.json") != FINAL_INDEX_SHA256:
        raise RuntimeError("exhaustive umbrella index identity changed")
    final_index = load_json(final / "artifact_index.json")
    if inventory(final, exclude_index=True) != (final_index["files"], final_index["tree_aggregate_sha256"]):
        raise RuntimeError("exhaustive umbrella index content mismatch")

    log_sources = {
        "smoke_v1_failure.log": base / "lingbot_native_future_decode_smoke_v1.log",
        "smoke_v2_success.log": base / "lingbot_native_future_decode_smoke_v2.log",
        "exhaustive_shard0_success.log": base / "lingbot_native_future_decode_all120_v1_shard0.log",
        "exhaustive_shard1_success.log": base / "lingbot_native_future_decode_all120_v1_shard1.log",
        "finalize_success.log": base / "lingbot_native_future_decode_all120_v1.finalize.log",
    }
    snapshots: dict[Path, tuple[int, str]] = {}
    texts: dict[str, str] = {}
    for name, path in log_sources.items():
        if not path.is_file():
            raise RuntimeError(f"missing decode execution log: {path}")
        snapshots[path] = (path.stat().st_size, sha256_file(path))
        texts[name] = path.read_text(errors="replace")
    if "Can't call numpy() on Tensor that requires grad" not in texts["smoke_v1_failure.log"] or "decoded 1/1" in texts["smoke_v1_failure.log"]:
        raise RuntimeError("smoke-v1 failure attribution changed")
    if texts["smoke_v2_success.log"].count("decoded 1/1") != 1 or "Traceback" in texts["smoke_v2_success.log"]:
        raise RuntimeError("smoke-v2 log is not exact success")
    for shard_index in range(2):
        text = texts[f"exhaustive_shard{shard_index}_success.log"]
        if text.count("decoded ") != 60 or "Traceback" in text:
            raise RuntimeError(f"exhaustive shard log is not exact success: {shard_index}")
    if "complete_mode_frozen_read_only_umbrella" not in texts["finalize_success.log"] or "Traceback" in texts["finalize_success.log"]:
        raise RuntimeError("finalize log is not exact success")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        logs = staging / "logs"
        logs.mkdir()
        log_rows = []
        for name, source in log_sources.items():
            destination = logs / name
            shutil.copy2(source, destination)
            if (destination.stat().st_size, sha256_file(destination)) != snapshots[source]:
                raise RuntimeError(f"log changed while copying: {source}")
            log_rows.append({"path": f"logs/{name}", "source_path": str(source), "bytes": destination.stat().st_size, "sha256": sha256_file(destination)})
        receipt = {
            "schema_version": 1, "status": "complete_mode_frozen_read_only_execution_addendum",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "output_root": str(output),
            "engineering_smokes": [
                {
                    "pid": 39275, "version": "v1", "included_in_inference": False,
                    "status": "failed_engineering_smoke", "failure": "unbound official decode call omitted generate() no_grad context",
                    "disposition": "no scientific artifact produced; staging removed; no installed output root",
                    "method_change_after_failure": "wrapped the same official VA_Server.decode_one_video call in torch.no_grad, matching upstream generate() context",
                    "decoder_sha256": SMOKE_V1_DECODER_SHA256,
                    "log": "logs/smoke_v1_failure.log",
                },
                {
                    "pid": 39526, "version": "v2", "included_in_inference": False,
                    "status": "complete_excluded_engineering_smoke", "decoded_item_count": 1,
                    "artifact_index_sha256": SMOKE_V2_INDEX_SHA256, "log": "logs/smoke_v2_success.log",
                    "decoder_sha256": SMOKE_V2_DECODER_SHA256,
                },
            ],
            "smoke_v2_to_exhaustive_decoder_delta": {
                "from_sha256": SMOKE_V2_DECODER_SHA256,
                "to_sha256": DECODER_SHA256,
                "decode_computation_changed": False,
                "official_entrypoint_changed": False,
                "changes": [
                    "require non-smoke shard_count to be exactly one or two",
                    "prove the imported flash_attn module is the exact pinned shim and record its path/PYTHONPATH",
                    "fail closed on the smoke-established decoded shape (13,128,256,3), float32 dtype, finiteness, and [0,1] range",
                ],
                "interpretation": "The excluded smoke validates the same no_grad-wrapped official VA_Server.decode_one_video computation used exhaustively; the final delta only adds fail-closed identity/schema/provenance gates.",
            },
            "exhaustive_launches": [
                {"pid": 40003 + i, "shard_index": i, "physical_gpu": i, "decoded_item_count": 60, "runtime_seconds": load_json(shard_roots[i] / "provenance.json")["duration_seconds"], "artifact_index_sha256": SHARD_INDEX_SHA256[i], "tree_aggregate_sha256": shard_indexes[i]["tree_aggregate_sha256"], "log": f"logs/exhaustive_shard{i}_success.log"}
                for i in range(2)
            ],
            "final_umbrella": {"root": str(final), "artifact_index_sha256": FINAL_INDEX_SHA256, "tree_aggregate_sha256": final_index["tree_aggregate_sha256"], "item_count": 120, "log": "logs/finalize_success.log"},
            "decoder_sha256": DECODER_SHA256, "logs": log_rows,
        }
        write_json(staging / "execution_receipt.json", receipt)
        shutil.copy2(Path(__file__).resolve(), staging / "freeze_lingbot_decode_execution_addendum.py")
        for path, snapshot in snapshots.items():
            if (path.stat().st_size, sha256_file(path)) != snapshot:
                raise RuntimeError(f"source log changed during freeze: {path}")
        rows, aggregate = inventory(staging, exclude_index=True)
        index = {"schema_version": 1, "status": receipt["status"], "file_count_excluding_index": len(rows), "tree_aggregate_sha256": aggregate, "files": rows}
        write_json(staging / "artifact_index.json", index)
        for path in staging.rglob("*"):
            if path.is_file(): path.chmod(0o444)
        for path in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True): path.chmod(0o555)
        staging.chmod(0o555)
        if inventory(staging, exclude_index=True) != (rows, aggregate):
            raise RuntimeError("execution addendum changed during freeze")
        os.replace(staging, output)
        print(json.dumps({"status": index["status"], "output_root": str(output), "artifact_index_sha256": sha256_file(output / "artifact_index.json"), "tree_aggregate_sha256": aggregate, "file_count_excluding_index": len(rows)}, sort_keys=True))
    except BaseException:
        if staging.exists():
            for path in staging.rglob("*"):
                try: path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError: pass
            staging.chmod(0o755); shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
