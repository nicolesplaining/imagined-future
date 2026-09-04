#!/usr/bin/env python3
"""Verify every runtime checkpoint file against the frozen content manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_cosmos3_single_call_timing_manifest import SNAPSHOT_FILES, closure_hash
from imagined_future.cosmos3_archival import atomic_json, sha256
from launch_cosmos3_single_call_timing import validate_full_checkpoint_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-manifest", type=Path, required=True)
    parser.add_argument("--expected-content-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint receipt: {args.output}")
    if sha256(args.content_manifest) != args.expected_content_manifest_sha256:
        raise ValueError("checkpoint content-manifest SHA mismatch")
    content = json.loads(args.content_manifest.read_text(encoding="utf-8"))
    snapshot_root = args.snapshot_root.resolve()
    snapshot_hashes = {}
    for relative in SNAPSHOT_FILES:
        path = snapshot_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        snapshot_hashes[relative] = sha256(path)
    snapshot_closure = closure_hash(snapshot_hashes)
    runtime = {
        "checkpoint_identity_kind": (
            "sha256_of_canonical_full_file_content_manifest"
        ),
        "checkpoint_content_manifest": str(args.content_manifest.resolve()),
        "checkpoint_content_manifest_sha256": args.expected_content_manifest_sha256,
        "checkpoint_content_manifest_file_count": int(content["file_count"]),
        "checkpoint_content_manifest_total_size_bytes": int(
            content["total_size_bytes"]
        ),
        "checkpoint_root": str(content["checkpoint_root"]),
        "checkpoint_verification_root": str(args.checkpoint_root.resolve()),
    }
    verification = validate_full_checkpoint_content({"runtime": runtime})
    receipt = {
        "status": "pass",
        "scope": "full_runtime_checkpoint_content_pre_timing_calls",
        "checkpoint_content_manifest": str(args.content_manifest.resolve()),
        "checkpoint_content_manifest_sha256": args.expected_content_manifest_sha256,
        "checkpoint_provenance_root": str(content["checkpoint_root"]),
        "checkpoint_verification_root": str(args.checkpoint_root.resolve()),
        "file_count": verification["file_count"],
        "total_size_bytes": verification["total_size_bytes"],
        "verification_elapsed_seconds": verification[
            "verification_elapsed_seconds"
        ],
        "symlink_count": 0,
        "snapshot_root": str(snapshot_root),
        "snapshot_file_sha256": snapshot_hashes,
        "snapshot_closure_sha256": snapshot_closure,
    }
    atomic_json(args.output, receipt)
    print(
        json.dumps(
            {
                "status": "pass",
                "checkpoint_content_manifest_sha256": (
                    args.expected_content_manifest_sha256
                ),
                "snapshot_closure_sha256": snapshot_closure,
                "verification_elapsed_seconds": verification[
                    "verification_elapsed_seconds"
                ],
                "output": str(args.output),
                "output_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
