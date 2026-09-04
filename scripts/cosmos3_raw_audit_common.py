#!/usr/bin/env python3
"""Shared fail-closed I/O helpers for independent Cosmos raw-output audits.

This module deliberately has no dependency on any study runner, scientific helper,
or frozen analyzer.  It opens outcome payloads only after a separate inventory has
proved that the manifest-derived file set is complete, hash-stable, and mode 0444.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: Boolean is not a numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: nonfinite value")
    return number


def require_sha(path: Path, expected: str, label: str) -> str:
    if len(expected) != 64:
        raise ValueError(f"{label}: expected SHA-256 is not 64 hexadecimal characters")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label}: SHA-256 mismatch: {actual} != {expected}")
    return actual


def verify_frozen_package(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    output_root: Path,
    inventory_path: Path,
    expected_inventory_sha256: str,
    expected_count: int,
    expected_inventory_schema: str,
) -> tuple[dict[str, Any], list[Path], dict[str, Any]]:
    """Verify a frozen package without opening any scientific outcome payload.

    The order here is intentional.  Only the manifest and hash inventory are parsed
    first.  Outcome JSON is not parsed by this function at all.  Every expected path,
    mode, size, and digest must agree before callers receive paths to open.
    """

    require_sha(manifest_path, expected_manifest_sha256, "manifest")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "frozen_before_model_outcomes":
        raise ValueError("manifest is not frozen-before-outcomes")
    states = manifest.get("states")
    if not isinstance(states, list) or len(states) != expected_count:
        raise ValueError(f"manifest must contain exactly {expected_count} states")
    expected_names = sorted(f"{state['unit_id']}.json" for state in states)
    if len(set(expected_names)) != expected_count:
        raise ValueError("manifest-derived output filenames are not unique")

    require_sha(inventory_path, expected_inventory_sha256, "output inventory")
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise ValueError("output inventory must be a regular nonsymlink file")
    if stat.S_IMODE(inventory_path.stat().st_mode) != 0o444:
        raise ValueError("output inventory is not mode 0444")
    inventory = load_json(inventory_path)
    expected_header = {
        "schema_version": expected_inventory_schema,
        "status": "complete_mode_frozen_read_only_package",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": expected_manifest_sha256,
        "file_count": expected_count,
        "expected_file_count": expected_count,
        "exact_manifest_file_set": True,
        "output_mode": "0444",
    }
    for key, expected in expected_header.items():
        if inventory.get(key) != expected:
            raise ValueError(f"inventory {key} differs: {inventory.get(key)!r} != {expected!r}")
    rows = inventory.get("files")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("inventory file rows are incomplete")
    row_names = [row.get("filename") for row in rows]
    if sorted(row_names) != expected_names or len(set(row_names)) != expected_count:
        raise ValueError("inventory does not contain the exact manifest-derived file set")
    by_name = {str(row["filename"]): row for row in rows}

    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("output root must be an existing nonsymlink directory")
    entries = sorted(output_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("output root contains a symlink or non-file entry")
    if [path.name for path in entries] != expected_names:
        raise ValueError("live output directory is not the exact manifest-derived file set")

    # Hashing bytes is deliberately completed before any outcome JSON is interpreted.
    for path in entries:
        row = by_name[path.name]
        if row.get("unit_id") != path.stem:
            raise ValueError(f"inventory unit ID differs for {path.name}")
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise ValueError(f"outcome is not mode 0444: {path.name}")
        if path.stat().st_size != int(row.get("size_bytes", -1)):
            raise ValueError(f"outcome size differs from inventory: {path.name}")
        if sha256(path) != row.get("sha256"):
            raise ValueError(f"outcome hash differs from inventory: {path.name}")
    return manifest, entries, inventory


def compare_tree(
    computed: Any,
    reference: Any,
    *,
    path: str,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> list[str]:
    """Return all discrepancies, requiring the computed tree's exact key topology."""

    problems: list[str] = []
    if isinstance(computed, Mapping):
        if not isinstance(reference, Mapping):
            return [f"{path}: reference type {type(reference).__name__}, expected mapping"]
        missing = sorted(set(computed) - set(reference))
        extra = sorted(set(reference) - set(computed))
        if missing:
            problems.append(f"{path}: reference missing keys {missing}")
        if extra:
            problems.append(f"{path}: reference has extra keys {extra}")
        for key in computed:
            if key in reference:
                problems.extend(
                    compare_tree(
                        computed[key], reference[key], path=f"{path}.{key}", atol=atol, rtol=rtol
                    )
                )
        return problems
    if isinstance(computed, (list, tuple)):
        if not isinstance(reference, (list, tuple)) or len(computed) != len(reference):
            return [f"{path}: sequence shape differs"]
        for index, (left, right) in enumerate(zip(computed, reference, strict=True)):
            problems.extend(
                compare_tree(left, right, path=f"{path}[{index}]", atol=atol, rtol=rtol)
            )
        return problems
    if isinstance(computed, bool) or isinstance(reference, bool):
        if computed is not reference:
            problems.append(f"{path}: {computed!r} != {reference!r}")
        return problems
    if isinstance(computed, (int, float)) and isinstance(reference, (int, float)):
        left = finite(computed, path)
        right = finite(reference, f"{path} reference")
        if abs(left - right) > atol + rtol * abs(right):
            problems.append(f"{path}: {left:.17g} != {right:.17g}")
        return problems
    if computed != reference:
        problems.append(f"{path}: {computed!r} != {reference!r}")
    return problems


def atomic_json_no_overwrite(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_reference_after_hash(path: Path, expected_sha256: str) -> dict[str, Any]:
    require_sha(path, expected_sha256, "frozen summary")
    value = load_json(path)
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise ValueError("frozen summary is not a complete JSON object")
    return value


def require_exact_strings(value: Any, expected: Sequence[str], label: str) -> None:
    if value != list(expected):
        raise ValueError(f"{label}: exact ordered labels differ")
