#!/usr/bin/env python3
"""Freeze the excluded-state DreamZero clean-upstream parity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--patched-root", type=Path, required=True)
    parser.add_argument("--clean-log", type=Path, required=True)
    parser.add_argument("--patched-log", type=Path, required=True)
    parser.add_argument("--capture-runner", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    args = parser.parse_args()

    root = args.output_root
    upstream_receipt_path = root / "upstream_receipt.json"
    patched_receipt_path = root / "patched_receipt.json"
    upstream_receipt = json.loads(upstream_receipt_path.read_text())
    patched_receipt = json.loads(patched_receipt_path.read_text())
    if upstream_receipt["input_fingerprint"] != patched_receipt["input_fingerprint"]:
        raise RuntimeError("input fingerprints differ")
    if upstream_receipt["runner_sha256"] != patched_receipt["runner_sha256"]:
        raise RuntimeError("capture runner differs between arms")
    if patched_receipt["comparison"]["bitwise_exact"] is not True:
        raise RuntimeError("patched/upstream parity did not pass")
    if float(patched_receipt["comparison"]["maximum_absolute_error"]) != 0.0:
        raise RuntimeError("patched/upstream maximum absolute error is nonzero")
    with np.load(root / "upstream_actions.npz", allow_pickle=False) as archive:
        upstream = np.ascontiguousarray(archive["actions"])
    with np.load(root / "patched_actions.npz", allow_pickle=False) as archive:
        patched = np.ascontiguousarray(archive["actions"])
    if not np.array_equal(upstream, patched):
        raise RuntimeError("raw action arrays differ")

    for source, name in (
        (args.clean_log, "clean_upstream_server.log"),
        (args.patched_log, "patched_mode_off_server.log"),
        (args.capture_runner, "audit_dreamzero_upstream_native_parity.py"),
    ):
        destination = root / name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o444)

    clean_commit = git(args.clean_root, "rev-parse", "HEAD")
    clean_status = git(args.clean_root, "status", "--porcelain")
    if clean_status:
        raise RuntimeError(f"clean checkout is dirty: {clean_status}")
    patched_commit = git(args.patched_root, "rev-parse", "HEAD")
    if clean_commit != patched_commit:
        raise RuntimeError("clean and patched repositories have different base commits")
    patched_diff = subprocess.run(
        ["git", "-C", str(args.patched_root), "diff", "--binary"],
        check=True,
        capture_output=True,
    ).stdout

    receipt_path = root / "execution_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    receipt = {
        "schema": "dreamzero-upstream-native-parity-execution-v1",
        "status": "complete",
        "admission": "excluded_development_control",
        "scientific_admission": False,
        "official_commit": clean_commit,
        "checkpoint_revision": args.checkpoint_revision,
        "clean_checkout_status": "clean",
        "clean_source_sha256": {
            str(relative): sha256(args.clean_root / relative)
            for relative in (
                Path("groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py"),
                Path("groot/vla/model/n1_5/sim_policy.py"),
                Path("socket_test_optimized_AR.py"),
            )
        },
        "patched_diff_sha256": hashlib.sha256(patched_diff).hexdigest(),
        "patched_modified_files": git(args.patched_root, "status", "--short").splitlines(),
        "input_fingerprint": upstream_receipt["input_fingerprint"],
        "noise_seed": 1140,
        "action_shape": list(upstream.shape),
        "action_dtype": str(upstream.dtype),
        "action_array_sha256": upstream_receipt["action_array_sha256"],
        "bitwise_exact": True,
        "maximum_absolute_error": 0.0,
        "capture_runner_sha256": upstream_receipt["runner_sha256"],
        "worker_shutdown_after_comparison": True,
        "failed_attempts": [
            {
                "stage": "first clean-client capture wrapper",
                "disposition": "failed after inference but before any artifact write",
                "error": "TypeError: controlled DreamZero response expected a mapping but official upstream returned its native ndarray",
                "resolution": "capture runner was updated to admit the native ndarray response; both successful arms used the same final runner hash",
                "stderr_was_not_persisted": True,
            }
        ],
    }
    write_json(receipt_path, receipt)

    indexed = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name == "artifact_index.json":
            continue
        os.chmod(path, 0o444)
        indexed.append(
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "mode": stat.S_IMODE(path.stat().st_mode),
                "sha256": sha256(path),
            }
        )
    index_path = root / "artifact_index.json"
    write_json(
        index_path,
        {
            "schema": "dreamzero-upstream-native-parity-artifact-index-v1",
            "status": "complete",
            "artifact_count": len(indexed),
            "artifacts": indexed,
        },
    )
    print(json.dumps({"receipt": str(receipt_path), "index_sha256": sha256(index_path)}, indent=2))


if __name__ == "__main__":
    main()
