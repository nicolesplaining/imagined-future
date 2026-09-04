#!/usr/bin/env python3
"""Fail-closed verification of a checkpoint against a content manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "checkpoint-content-manifest-v1"


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def verify_checkpoint(
    checkpoint_root: Path,
    content_manifest: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    checkpoint_root = checkpoint_root.resolve(strict=True)
    content_manifest = content_manifest.resolve(strict=True)
    if not checkpoint_root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {checkpoint_root}")
    actual_manifest_sha256 = sha256_file(content_manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "checkpoint content-manifest hash mismatch: "
            f"{actual_manifest_sha256} != {expected_manifest_sha256}"
        )
    payload = json.loads(content_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("unsupported checkpoint content-manifest schema")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("checkpoint content manifest contains no file entries")
    relative_paths = [str(row["relative_path"]) for row in entries]
    if relative_paths != sorted(relative_paths) or len(set(relative_paths)) != len(relative_paths):
        raise ValueError("checkpoint content-manifest paths are not sorted and unique")
    actual_paths = sorted(
        (
            path.relative_to(checkpoint_root).as_posix()
            for path in checkpoint_root.rglob("*")
            if path.is_file()
        )
    )
    if actual_paths != relative_paths:
        missing = sorted(set(relative_paths) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(relative_paths))
        raise ValueError(
            f"checkpoint file-set mismatch: missing={missing[:10]} extra={extra[:10]}"
        )
    total_size = 0
    for row in entries:
        path = checkpoint_root / str(row["relative_path"])
        if path.is_symlink():
            raise ValueError(f"symlinked checkpoint file is not allowed: {path}")
        stat_before = path.stat()
        expected_size = int(row["size_bytes"])
        if stat_before.st_size != expected_size:
            raise ValueError(f"checkpoint file size mismatch: {path}")
        actual_sha256 = sha256_file(path)
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            or stat_before.st_ino != stat_after.st_ino
        ):
            raise RuntimeError(f"checkpoint file changed while verifying: {path}")
        if actual_sha256 != str(row["sha256"]):
            raise ValueError(f"checkpoint file digest mismatch: {path}")
        total_size += stat_after.st_size
    if int(payload.get("file_count", -1)) != len(entries):
        raise ValueError("checkpoint file-count metadata mismatch")
    if int(payload.get("total_size_bytes", -1)) != total_size:
        raise ValueError("checkpoint total-size metadata mismatch")
    return {
        "schema_version": "checkpoint-content-verification-v1",
        "status": "pass",
        "checkpoint_root": checkpoint_root.as_posix(),
        "content_manifest_path": content_manifest.as_posix(),
        "content_manifest_sha256": actual_manifest_sha256,
        "file_count": len(entries),
        "total_size_bytes": total_size,
        "zero_symlinks": True,
        "exact_file_set": True,
        "all_file_sizes_match": True,
        "all_file_sha256_match": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--content-manifest", type=Path, required=True)
    parser.add_argument("--expected-content-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    result = verify_checkpoint(
        args.checkpoint_root,
        args.content_manifest,
        args.expected_content_manifest_sha256,
    )
    result["verification_elapsed_seconds"] = time.monotonic() - started
    encoded = canonical_json_bytes(result)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite verification receipt: {args.output}")
        atomic_write(args.output, encoded)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
