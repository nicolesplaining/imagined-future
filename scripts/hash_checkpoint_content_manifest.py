#!/usr/bin/env python3
"""Create a deterministic, file-by-file checkpoint content manifest.

The content-manifest JSON is deterministic: it contains sorted relative paths,
byte sizes, and SHA-256 digests, but no wall-clock metadata.  A separate audit
JSON records the one-time I/O duration and the manifest digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "checkpoint-content-manifest-v1"


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


def build_manifest(checkpoint_root: Path) -> dict[str, Any]:
    checkpoint_root = checkpoint_root.resolve(strict=True)
    if not checkpoint_root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {checkpoint_root}")

    candidates = sorted(
        (path for path in checkpoint_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(checkpoint_root).as_posix(),
    )
    entries: list[dict[str, Any]] = []
    for path in candidates:
        if path.is_symlink():
            raise ValueError(f"symlinked checkpoint file is not allowed: {path}")
        stat_before = path.stat()
        digest = sha256_file(path)
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            or stat_before.st_ino != stat_after.st_ino
        ):
            raise RuntimeError(f"checkpoint file changed while hashing: {path}")
        entries.append(
            {
                "relative_path": path.relative_to(checkpoint_root).as_posix(),
                "size_bytes": stat_after.st_size,
                "sha256": digest,
            }
        )

    if not entries:
        raise ValueError(f"checkpoint root contains no files: {checkpoint_root}")
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_root": checkpoint_root.as_posix(),
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() == args.audit_output.resolve():
        raise ValueError("content manifest and audit output must be different files")
    started_wall = time.time()
    started_monotonic = time.monotonic()
    manifest = build_manifest(args.checkpoint_root)
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    atomic_write(args.output, manifest_bytes)
    elapsed_seconds = time.monotonic() - started_monotonic
    audit = {
        "schema_version": "checkpoint-content-manifest-audit-v1",
        "checkpoint_root": manifest["checkpoint_root"],
        "content_manifest_path": args.output.resolve().as_posix(),
        "content_manifest_sha256": manifest_sha256,
        "file_count": manifest["file_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "hashing_started_unix_seconds": started_wall,
        "hashing_elapsed_seconds": elapsed_seconds,
    }
    atomic_write(args.audit_output, canonical_json_bytes(audit))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
