#!/usr/bin/env python3
"""Freeze a predicted-future x K/V factorial on the existing Cosmos cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_STATES = 22
DEFAULT_DEVELOPMENT_STATE = "BananaInBowlTask_seed_103"
COSMOS_COMMIT = "d4599e2e43fbd06168e9884205b9b66c3902d8f6"
COSMOS_SERVER_IMAGE = (
    "sha256:95a8933deb69f2fc73697d846f7e17066a4d4e05dc9ef85718a7ee12cfea2a8c"
)
RUNNER_SHA256 = "83285e99b993e7f996a40189332643338e33805fe03e00b931f0c214e32179de"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-state", default=DEFAULT_DEVELOPMENT_STATE)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def resolve_recording(recorded: str, recording_root: Path) -> Path:
    marker = "/output/"
    if marker not in recorded:
        raise ValueError(f"unexpected recorded-HDF5 path: {recorded}")
    relative = recorded.split(marker, 1)[1]
    return recording_root / relative


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
    cohort_root = args.cohort_root.resolve()
    recording_root = args.recording_root.resolve()
    source_dirs = sorted(
        path
        for path in cohort_root.iterdir()
        if path.is_dir()
        and "attempt" not in path.name
        and (path / "summary.json").is_file()
        and (path / "self.npz").is_file()
    )
    if len(source_dirs) != EXPECTED_SOURCE_STATES:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_STATES} canonical source states, got {len(source_dirs)}"
        )
    if args.development_state not in {path.name for path in source_dirs}:
        raise ValueError("development-state exclusion is absent from the source cohort")

    states: list[dict[str, Any]] = []
    for state_dir in source_dirs:
        if state_dir.name == args.development_state:
            continue
        summary_path = state_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        required = {
            "task",
            "environment_seed",
            "branch_step",
            "instruction",
            "recipient_seed",
            "donor_seed",
            "recorded_hdf5",
        }
        missing = sorted(required - set(summary))
        if missing:
            raise ValueError(f"{state_dir.name} is missing design fields: {missing}")
        recorded_hdf5 = resolve_recording(
            str(summary["recorded_hdf5"]), recording_root
        )
        if not recorded_hdf5.is_file():
            raise FileNotFoundError(recorded_hdf5)
        states.append(
            {
                "state_id": state_dir.name,
                "task": str(summary["task"]),
                "environment_seed": int(summary["environment_seed"]),
                "branch_step": int(summary["branch_step"]),
                "instruction": str(summary["instruction"]),
                "recipient_seed": int(summary["recipient_seed"]),
                "donor_seed": int(summary["donor_seed"]),
                "asset_video": str((state_dir / "self.npz").resolve()),
                "recorded_hdf5": str(recorded_hdf5.resolve()),
                "branch_summary": str(summary_path.resolve()),
                "input_sha256": {
                    "asset_video": sha256(state_dir / "self.npz"),
                    "recorded_hdf5": sha256(recorded_hdf5),
                    "branch_summary": sha256(summary_path),
                },
            }
        )

    if len(states) != EXPECTED_SOURCE_STATES - 1:
        raise RuntimeError("development exclusion did not yield exactly 21 evaluation states")
    body = {
        "schema_version": 1,
        "study_name": "cosmos3-existing-cohort-predicted-future-kv-factorial-v3",
        "scope": (
            "population-level suppression/rescue extension on the existing selected-pair "
            "cohort; not selection-free and not a replacement for the fresh held-out study"
        ),
        "source_cohort": str(cohort_root),
        "source_state_count": EXPECTED_SOURCE_STATES,
        "excluded_development_state": args.development_state,
        "evaluation_state_count": len(states),
        "future_source": "model-predicted",
        "factorial_cells": [
            "recipient_future_recipient_kv",
            "donor_future_recipient_kv",
            "donor_future_donor_kv",
            "recipient_future_donor_kv",
        ],
        "attention_scope": {
            "layers": list(range(36)),
            "queries": "action",
            "keys_values": "future_video",
            "token_count_preserved": True,
        },
        "runtime": {
            "cosmos_commit": COSMOS_COMMIT,
            "server_image": COSMOS_SERVER_IMAGE,
            "single_state_runner_sha256": RUNNER_SHA256,
        },
        "controls": [
            "native recipient exact repeat",
            "recipient K/V record versus uninstrumented clean recipient-future clamp",
            "recipient K/V exact replay",
            "donor K/V record versus uninstrumented donor-future clamp",
            "donor K/V exact replay",
            "fixed visible-future signature within each K/V crossing",
            "no direct action-coordinate mutation",
            "one model-state fingerprint across arms",
        ],
        "analysis": {
            "independent_unit": "saved simulator state",
            "within_state_measurements": "four factorial cells",
            "bootstrap": "task-to-state hierarchical bootstrap",
            "bootstrap_samples": 10000,
            "bootstrap_seed": 20260903,
            "report": [
                "cellwise donor projection",
                "cellwise distance to recipient and donor",
                "K/V effect at fixed recipient future",
                "K/V effect at fixed donor future",
                "future x K/V interaction",
                "per-task and leave-one-task-out estimates",
                "number of states following K/V rather than visible-future identity",
            ],
        },
        "states": states,
    }
    manifest_id = "cosmos3-kv-existing-" + hashlib.sha256(canonical(body)).hexdigest()[:16]
    manifest = {"manifest_id": manifest_id, **body}
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
