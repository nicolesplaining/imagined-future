#!/usr/bin/env python3
"""Create the completion inventory required by the independent timing-v5 audit.

The utility first proves the directory is the exact 30-file manifest-derived set;
only then does it open JSON to verify non-scientific identity/completion metadata.
It never reads or summarizes timing-effect fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
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
    actual_manifest_hash = sha256(args.manifest)
    if actual_manifest_hash != args.expected_manifest_sha256:
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_model_outcomes" \
            or manifest.get("study_name") != "cosmos3-single-call-timing-v5":
        raise ValueError("manifest is not frozen timing v5")
    states = manifest.get("states", [])
    if len(states) != 30:
        raise ValueError("manifest does not contain exactly 30 states")
    expected = sorted(f"{state['unit_id']}.json" for state in states)
    entries = sorted(args.output_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("run root contains a symlink or non-file entry")
    if [path.name for path in entries] != expected:
        raise ValueError("refusing partial/extra package: run root is not exact 30/30")

    unit_by_id = {str(unit["unit_id"]): unit for unit in states}
    rows: list[dict[str, Any]] = []
    for path in entries:
        # This is reached only after exact set completeness.  Read identity/control
        # metadata only; never inspect the timing rows or derived outcomes.
        report = json.loads(path.read_text(encoding="utf-8"))
        unit = unit_by_id[path.stem]
        exact = {
            "status": "complete", "manifest_id": manifest["manifest_id"],
            "manifest_sha256": actual_manifest_hash, "unit_id": unit["unit_id"],
            "task": unit["task"], "environment_seed": unit["environment_seed"],
            "phase": "middle", "request_count": 108,
            "action_shape": [32, 8], "action_coordinate_count": 256,
            "shape_valid_response_action_count": 108, "action_shape_failure_count": 0,
        }
        if any(report.get(key) != value for key, value in exact.items()):
            raise ValueError(f"identity/completion metadata differs for {path.name}")
        rows.append({
            "filename": path.name, "unit_id": path.stem,
            "size_bytes": path.stat().st_size, "sha256": sha256(path),
        })
    for path in entries: path.chmod(0o444)
    for path, row in zip(entries, rows, strict=True):
        if stat.S_IMODE(path.stat().st_mode) != 0o444 or sha256(path) != row["sha256"]:
            raise RuntimeError(f"post-freeze mode/hash verification failed: {path.name}")
    args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
    inventory = {
        "schema_version": "cosmos3-timing-output-inventory-v1",
        "status": "complete_mode_frozen_read_only_package",
        "manifest_id": manifest["manifest_id"],
        "manifest_path": args.manifest.resolve().as_posix(),
        "manifest_sha256": actual_manifest_hash,
        "output_root": args.output_root.resolve().as_posix(),
        "file_count": 30, "expected_file_count": 30,
        "exact_manifest_file_set": True, "output_mode": "0444", "files": rows,
    }
    atomic(args.inventory_output, inventory); args.inventory_output.chmod(0o444)
    print(json.dumps({
        "status": inventory["status"], "file_count": 30,
        "inventory_output": str(args.inventory_output),
        "inventory_sha256": sha256(args.inventory_output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
