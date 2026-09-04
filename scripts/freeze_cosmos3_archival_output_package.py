#!/usr/bin/env python3
"""Verify, hash, and make a complete Cosmos archival output package read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
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
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inventory_output.exists():
        raise FileExistsError(f"refusing to overwrite inventory: {args.inventory_output}")
    if args.output_root.is_symlink() or not args.output_root.is_dir():
        raise ValueError("output root must be an existing nonsymlink directory")
    actual_manifest_sha256 = sha256(args.manifest)
    if actual_manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError("manifest hash differs from the frozen CLI value")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "frozen_before_model_outcomes"
        or manifest.get("admission")
        != "frozen_archival_selection_free_action_level_evaluation"
    ):
        raise ValueError("manifest is not the frozen archival evaluation")
    states = manifest.get("states", [])
    if len(states) != 90:
        raise ValueError("manifest does not contain exactly 90 states")
    expected_names = sorted(f"{state['unit_id']}.json" for state in states)
    if len(set(expected_names)) != 90:
        raise ValueError("manifest unit filenames are not unique")
    actual_entries = sorted(args.output_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in actual_entries):
        raise ValueError("output root contains a symlink or non-file entry")
    actual_names = [path.name for path in actual_entries]
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise ValueError(
            f"output file set is not exact 90/90: missing={missing} extra={extra}"
        )

    rows: list[dict[str, Any]] = []
    for path in actual_entries:
        unit_id = path.stem
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_metadata = {
            "status": "complete",
            "admission": "frozen_archival_selection_free_action_level_evaluation",
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": actual_manifest_sha256,
            "unit_id": unit_id,
        }
        actual_metadata = {key: payload.get(key) for key in expected_metadata}
        if actual_metadata != expected_metadata:
            raise ValueError(f"output metadata mismatch for {path.name}")
        rows.append(
            {
                "filename": path.name,
                "unit_id": unit_id,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    for path in actual_entries:
        path.chmod(0o444)
    if any(stat.S_IMODE(path.stat().st_mode) != 0o444 for path in actual_entries):
        raise RuntimeError("not every output was frozen to mode 0444")
    for path, row in zip(actual_entries, rows, strict=True):
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"output changed between hash and mode freeze: {path.name}")

    inventory = {
        "schema_version": "cosmos3-archival-output-inventory-v1",
        "status": "complete_mode_frozen_read_only_package",
        "manifest_id": manifest["manifest_id"],
        "manifest_path": args.manifest.resolve().as_posix(),
        "manifest_sha256": actual_manifest_sha256,
        "output_root": args.output_root.resolve().as_posix(),
        "file_count": len(rows),
        "expected_file_count": 90,
        "exact_manifest_file_set": True,
        "output_mode": "0444",
        "files": rows,
    }
    atomic_write(args.inventory_output, canonical_json_bytes(inventory))
    args.inventory_output.chmod(0o444)
    print(
        json.dumps(
            {
                "status": inventory["status"],
                "file_count": inventory["file_count"],
                "inventory_output": args.inventory_output.as_posix(),
                "inventory_sha256": sha256(args.inventory_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
