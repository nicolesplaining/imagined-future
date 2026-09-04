#!/usr/bin/env python3
"""Fail-closed sequential launcher for the frozen Cosmos 3 timing audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_closure(
    manifest: dict[str, Any], snapshot_root: Path, launcher: Path
) -> None:
    runtime = manifest["runtime"]
    if sha256(launcher) != runtime["launcher_sha256"]:
        raise ValueError("launcher differs from frozen manifest")
    closure = runtime["snapshot_file_sha256"]
    for relative, expected in closure.items():
        path = snapshot_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"snapshot hash mismatch for {relative}: {actual} != {expected}")


def validate_full_checkpoint_content(manifest: dict[str, Any]) -> dict[str, Any]:
    """Rehash every checkpoint file before the first evaluation model call."""

    runtime = manifest["runtime"]
    if runtime.get("checkpoint_identity_kind") != (
        "sha256_of_canonical_full_file_content_manifest"
    ):
        raise ValueError("checkpoint identity is not a full file-content manifest")
    content_manifest_path = Path(str(runtime["checkpoint_content_manifest"]))
    expected_manifest_hash = str(runtime["checkpoint_content_manifest_sha256"])
    if not content_manifest_path.is_file():
        raise FileNotFoundError(content_manifest_path)
    if sha256(content_manifest_path) != expected_manifest_hash:
        raise ValueError("checkpoint content-manifest artifact hash mismatch")
    content = json.loads(content_manifest_path.read_text(encoding="utf-8"))
    canonical_bytes = (
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if content_manifest_path.read_bytes() != canonical_bytes:
        raise ValueError("checkpoint content manifest is not canonical JSON bytes")
    if Path(str(runtime["checkpoint_root"])).resolve() != Path(
        str(content["checkpoint_root"])
    ).resolve():
        raise ValueError("provenance checkpoint root differs from content manifest")
    checkpoint_root = Path(str(runtime["checkpoint_verification_root"])).resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(checkpoint_root)
    paths = sorted(checkpoint_root.rglob("*"))
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise ValueError(f"checkpoint tree contains symlinks: {symlinks[:3]}")
    actual_files = [path for path in paths if path.is_file()]
    expected_entries = list(content["files"])
    expected_relatives = [str(entry["relative_path"]) for entry in expected_entries]
    actual_relatives = [
        path.relative_to(checkpoint_root).as_posix() for path in actual_files
    ]
    if actual_relatives != expected_relatives:
        raise ValueError("checkpoint file set differs from full content manifest")
    if int(content["file_count"]) != int(runtime["checkpoint_content_manifest_file_count"]):
        raise ValueError("checkpoint file-count provenance differs")
    if int(content["total_size_bytes"]) != int(
        runtime["checkpoint_content_manifest_total_size_bytes"]
    ):
        raise ValueError("checkpoint byte-count provenance differs")
    started = time.monotonic()
    total_size = 0
    for path, entry in zip(actual_files, expected_entries, strict=True):
        stat_before = path.stat()
        if stat_before.st_size != int(entry["size_bytes"]):
            raise ValueError(f"checkpoint file size differs: {path}")
        actual_hash = sha256(path)
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            or stat_before.st_ino != stat_after.st_ino
        ):
            raise RuntimeError(f"checkpoint file changed during verification: {path}")
        if actual_hash != str(entry["sha256"]):
            raise ValueError(f"checkpoint file SHA-256 differs: {path}")
        total_size += stat_after.st_size
    elapsed = time.monotonic() - started
    if total_size != int(content["total_size_bytes"]):
        raise ValueError("verified checkpoint byte total differs")
    return {
        "event": "full_checkpoint_content_verified",
        "checkpoint_content_manifest_sha256": expected_manifest_hash,
        "file_count": len(actual_files),
        "total_size_bytes": total_size,
        "verification_elapsed_seconds": elapsed,
    }


def validate_state_output(
    path: Path, unit: dict[str, Any], manifest: dict[str, Any], manifest_hash: str
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "complete",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_hash,
        "unit_id": unit["unit_id"],
        "task": unit["task"],
        "environment_seed": unit["environment_seed"],
        "phase": "middle",
        "request_count": 108,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"completed unit failed {key}: {payload.get(key)!r} != {expected!r}"
            )
    if payload.get("runtime_gate", {}).get("passed") is not True:
        raise RuntimeError(f"completed unit failed runtime gate: {path}")
    for key, expected in (
        ("structural_projection_null_count", 28),
        ("finite_off_diagonal_projection_count", 72),
        ("native_projection_absent_count", 8),
        ("shape_valid_response_action_count", 108),
        ("action_shape_failure_count", 0),
    ):
        if payload.get(key) != expected:
            raise RuntimeError(f"completed unit failed frozen {key} census: {path}")
    if payload.get("runtime_gate", {}).get(
        "exact_projection_applicability_census"
    ) is not True:
        raise RuntimeError(f"completed unit failed projection applicability gate: {path}")
    if payload.get("action_shape") != [32, 8] or payload.get(
        "action_coordinate_count"
    ) != 256:
        raise RuntimeError(f"completed unit failed action shape/count gate: {path}")
    if payload.get("runtime_gate", {}).get("exact_action_shape_and_count") is not True:
        raise RuntimeError(f"completed unit failed exact action-shape gate: {path}")


def validate_independent_go(
    audit_path: Path,
    expected_audit_hash: str,
    manifest: dict[str, Any],
    manifest_hash: str,
) -> None:
    actual = sha256(audit_path)
    if actual != expected_audit_hash:
        raise ValueError(f"audit-report SHA mismatch: {actual} != {expected_audit_hash}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required = {
        "verdict": "GO",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_hash,
        "protocol_sha256": manifest["source"]["protocol_sha256"],
        "checklist_sha256": manifest["source"]["outcome_blind_checklist_sha256"],
        "finiteness_amendment_sha256": manifest["source"][
            "finiteness_amendment_sha256"
        ],
        "timing_server_sha256": manifest["runtime"]["timing_server_sha256"],
        "action_shape_amendment_sha256": manifest["source"][
            "action_shape_amendment_sha256"
        ],
        "snapshot_closure_sha256": manifest["runtime"]["snapshot_closure_sha256"],
        "checkpoint_content_manifest_sha256": manifest["runtime"][
            "checkpoint_content_manifest_sha256"
        ],
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise ValueError(
                f"independent audit {key}={audit.get(key)!r}, expected {expected!r}"
            )
    if not isinstance(audit.get("auditor"), str) or not audit["auditor"].strip():
        raise ValueError("independent audit does not identify its auditor")
    if audit.get("all_checklist_items_passed") is not True:
        raise ValueError("independent audit did not pass every checklist item")
    if audit.get("registry_empty_before_smoke") is not True:
        raise ValueError("independent audit lacks empty timing-server registry proof")
    if audit.get("registry_empty_before_evaluation") is not True:
        raise ValueError("independent audit lacks post-smoke empty-registry proof")
    if audit.get("dedicated_server_port") != manifest["runtime"]["server_port"]:
        raise ValueError("independent audit names the wrong dedicated server port")
    if audit.get("dedicated_server_container") != manifest["runtime"][
        "server_container"
    ]:
        raise ValueError("independent audit names the wrong dedicated server container")
    smoke_path = Path(str(audit.get("excluded_smoke_path", "")))
    smoke_hash = str(audit.get("excluded_smoke_sha256", ""))
    if not smoke_path.is_file() or sha256(smoke_path) != smoke_hash:
        raise ValueError("excluded-smoke artifact differs from the independent audit")


def main() -> None:
    args = parse_args()
    manifest_hash = sha256(args.manifest)
    if manifest_hash != args.expected_manifest_sha256:
        raise ValueError(f"manifest SHA mismatch: {manifest_hash}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_model_outcomes":
        raise ValueError("manifest is not frozen before outcomes")
    if manifest.get("admission") != "frozen_single_call_timing_evaluation":
        raise ValueError("launcher refuses a non-evaluation manifest")
    if manifest.get("launch_authorization") != "independent_outcome_blind_go_required":
        raise ValueError("manifest lacks independent launch-authorization requirement")
    if args.port != int(manifest["runtime"]["server_port"]):
        raise ValueError("CLI port differs from frozen timing server port")
    if len(manifest.get("states", [])) != 30:
        raise ValueError("timing launcher requires exactly 30 states")
    validate_closure(manifest, args.snapshot_root, Path(__file__).resolve())
    checkpoint_audit = validate_full_checkpoint_content(manifest)
    print(json.dumps(checkpoint_audit, sort_keys=True), flush=True)
    validate_independent_go(
        args.audit_report,
        args.expected_audit_sha256,
        manifest,
        manifest_hash,
    )
    if args.summary_dir.exists():
        raise FileExistsError(f"summary directory already exists: {args.summary_dir}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            "timing output root is nonempty; frozen no-resume rule requires a new version"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    runner = args.snapshot_root / manifest["runtime"]["runner_relative_path"]
    analyzer = args.snapshot_root / manifest["runtime"]["analyzer_relative_path"]
    for index, unit in enumerate(manifest["states"]):
        output = args.output_root / f"{unit['unit_id']}.json"
        command = [
            sys.executable,
            str(runner),
            "--manifest",
            str(args.manifest),
            "--expected-manifest-sha256",
            manifest_hash,
            "--unit-id",
            str(unit["unit_id"]),
            "--screen-root",
            str(args.screen_root),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--output",
            str(output),
        ]
        print(
            json.dumps(
                {
                    "event": "state_start",
                    "index": index,
                    "count": 30,
                    "unit_id": unit["unit_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        subprocess.run(command, check=True)
        validate_state_output(output, unit, manifest, manifest_hash)
        print(
            json.dumps(
                {
                    "event": "state_complete",
                    "index": index,
                    "unit_id": unit["unit_id"],
                    "output_sha256": sha256(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    actual = {path.name for path in args.output_root.glob("*.json")}
    expected = {f"{unit['unit_id']}.json" for unit in manifest["states"]}
    if actual != expected:
        raise RuntimeError("complete cohort filename audit failed")
    analyze = [
        sys.executable,
        str(analyzer),
        "--manifest",
        str(args.manifest),
        "--expected-manifest-sha256",
        manifest_hash,
        "--output-root",
        str(args.output_root),
        "--summary-dir",
        str(args.summary_dir),
        "--bootstrap-samples",
        "10000",
        "--bootstrap-seed",
        "20260903",
    ]
    subprocess.run(analyze, check=True)
    print(
        json.dumps(
            {
                "event": "cohort_complete",
                "manifest_id": manifest["manifest_id"],
                "state_count": 30,
                "request_count": 3240,
                "summary_dir": str(args.summary_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
