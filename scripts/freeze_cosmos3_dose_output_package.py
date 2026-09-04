#!/usr/bin/env python3
"""Verify, hash, and mode-freeze the complete Cosmos dose output package.

The utility refuses to open any state payload until the output directory is
the exact 30-file manifest-derived set. It then checks identity, request/schema,
and manipulation-control fields only; it never summarizes dose outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_MANIFEST_SHA256 = (
    "1a37bdba796b580c905970c441aededccf4193c51fe370f0f3557eccd5a9ee7d"
)
EXPECTED_STATE_COUNT = 30
EXPECTED_REQUEST_COUNT = 92
EXPECTED_ACTION_SHAPE = (32, 8)
EXPECTED_ACTION_COORDINATES = 256
EXPECTED_SITE_TOLERANCE = 1e-7


def sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
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
    parser.add_argument(
        "--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    return parser.parse_args()


def finite_action_count(response_actions: Any) -> int:
    if not isinstance(response_actions, dict) or len(response_actions) != 92:
        raise ValueError("response action census is not exactly 92")
    validated = 0
    for label, action in response_actions.items():
        if not isinstance(label, str) or not isinstance(action, list) or len(action) != 32:
            raise ValueError("response action has malformed label or timestep count")
        coordinate_count = 0
        for step in action:
            if not isinstance(step, list) or len(step) != 8:
                raise ValueError("response action does not have exact [32,8] shape")
            for coordinate in step:
                value = float(coordinate)
                if not math.isfinite(value):
                    raise ValueError("response action contains NaN or infinity")
                coordinate_count += 1
        if coordinate_count != EXPECTED_ACTION_COORDINATES:
            raise ValueError("response action coordinate count differs from 256")
        validated += 1
    return validated


def main() -> None:
    args = parse_args()
    if args.expected_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest SHA differs from the packager's frozen dose manifest")
    if args.inventory_output.exists():
        raise FileExistsError(f"refusing to overwrite inventory: {args.inventory_output}")
    if args.output_root.is_symlink() or not args.output_root.is_dir():
        raise ValueError("output root must be an existing nonsymlink directory")
    manifest_sha256 = sha256(args.manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest hash differs from the frozen CLI value")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "frozen_before_model_outcomes"
        or manifest.get("study_name")
        != "cosmos3-future-strength-dose-response-v2"
        or manifest.get("admission")
        != "prospective_action_level_future_strength_dose_response"
        or manifest.get("freeze_stage") != "evaluation_ready"
    ):
        raise ValueError("manifest is not the frozen dose-v2 evaluation")
    states = manifest.get("states", [])
    if len(states) != EXPECTED_STATE_COUNT:
        raise ValueError("manifest does not contain exactly 30 states")
    expected_names = sorted(f"{state['unit_id']}.json" for state in states)
    if len(set(expected_names)) != EXPECTED_STATE_COUNT:
        raise ValueError("manifest-derived output filenames are not unique")

    entries = sorted(args.output_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("output root contains a symlink or non-file entry")
    actual_names = [path.name for path in entries]
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise ValueError(
            f"refusing partial/extra package: missing={missing}, extra={extra}"
        )

    unit_by_id = {str(state["unit_id"]): state for state in states}
    expected_census = {
        "native": 4,
        "native_replay": 4,
        "none": 4,
        "self": 4,
        "self_replay": 4,
        "dose": 60,
        "midpoint_replay": 12,
    }
    rows: list[dict[str, Any]] = []
    for path in entries:
        report = json.loads(path.read_text(encoding="utf-8"))
        unit = unit_by_id[path.stem]
        exact = {
            "status": "complete",
            "admission": "prospective_action_level_future_strength_dose_response",
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest_sha256,
            "unit_id": unit["unit_id"],
            "task": unit["task"],
            "environment_seed": unit["environment_seed"],
            "phase": "middle",
            "request_count": EXPECTED_REQUEST_COUNT,
            "wire_schema_validated_response_count": EXPECTED_REQUEST_COUNT,
            "request_class_census": expected_census,
            "input_fingerprint_count": 1,
            "parameter_probe_hash_count": 1,
            "degenerate_native_action_axis_count": 0,
            "intervention_response_count": 84,
            "active_intervention_response_count": 80,
            "active_intervention_site_count": 320,
        }
        if any(report.get(key) != value for key, value in exact.items()):
            raise ValueError(f"identity/completion/control metadata differs for {path.name}")
        if finite_action_count(report.get("response_actions")) != EXPECTED_REQUEST_COUNT:
            raise ValueError(f"action shape census failed for {path.name}")
        action_errors = report.get("action_coordinate_errors")
        if (
            not isinstance(action_errors, dict)
            or len(action_errors) != 84
            or any(row != {"input": 0.0, "output": 0.0} for row in action_errors.values())
        ):
            raise ValueError(f"action-coordinate nonwrite controls failed for {path.name}")
        for key in (
            "model_input_future_clamp_max_error",
            "returned_future_velocity_overwrite_max_error",
        ):
            value = float(report.get(key, float("nan")))
            if not math.isfinite(value) or value > EXPECTED_SITE_TOLERANCE:
                raise ValueError(f"intervention-site control {key} failed for {path.name}")
        rows.append(
            {
                "filename": path.name,
                "unit_id": path.stem,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    for path in entries:
        path.chmod(0o444)
    for path, row in zip(entries, rows, strict=True):
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise RuntimeError(f"output was not mode-frozen: {path.name}")
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"output changed between hash and mode freeze: {path.name}")

    inventory = {
        "schema_version": "cosmos3-dose-output-inventory-v1",
        "status": "complete_mode_frozen_read_only_package",
        "manifest_id": manifest["manifest_id"],
        "manifest_path": args.manifest.resolve().as_posix(),
        "manifest_sha256": manifest_sha256,
        "output_root": args.output_root.resolve().as_posix(),
        "file_count": EXPECTED_STATE_COUNT,
        "expected_file_count": EXPECTED_STATE_COUNT,
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
                "file_count": EXPECTED_STATE_COUNT,
                "inventory_output": args.inventory_output.as_posix(),
                "inventory_sha256": sha256(args.inventory_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
