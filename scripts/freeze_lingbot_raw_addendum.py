#!/usr/bin/env python3
"""Copy, recursively inventory, and mode-freeze LingBot core and dose raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_SHA256 = "9c8d7dd547955e4622d98c7eb42fffed157f6db983f41f02929b28a434be47b4"
CORE_RUNNER_SHA256 = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
DOSE_RUNNER_SHA256 = "2d8b419be882eb979ed58091f7d0b0cd4322f2503aac9e4a854c558834f21b2e"
DOSE_PROTOCOL_SHA256 = "8b6b4103b5c172f28c896b9834fda114aa52684c53f8c570c78c346fda9d3eba"
UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"
FINAL_V1_INDEX_SHA256 = "698a818bce0f8dbeda22aee7df76752dce70f1d70089c6dd79cedf3e3faf273e"
CORE_ANALYSIS_INDEX_SHA256 = "0cc2ab978a157496018f1f43514b190630b1074ddd03381299718295bb51bab9"
DOSE_ANALYSIS_INDEX_SHA256 = "52211b2f463ed907468f0749783c67500a7a20d6699687cf0749392659d1dd93"
ORACLE_RECEIPT_SHA256 = "f54eebb2d4525ec8d8c57ebdc3b1294041194e5d1e13fd0033380a09453719aa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--dose-root", type=Path, required=True)
    parser.add_argument("--existing-final-root", type=Path, required=True)
    parser.add_argument("--core-analysis-root", type=Path, required=True)
    parser.add_argument("--dose-analysis-root", type=Path, required=True)
    parser.add_argument("--oracle-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")


def source_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"missing or unsafe source directory: {root}")
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlinks forbidden: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    return rows, aggregate.hexdigest()


def validate_core(root: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows, aggregate = source_inventory(root)
    if len(rows) != 211:
        raise RuntimeError(f"core raw census is {len(rows)}, expected exactly 211")
    require_sha(root / "manifest.json", MANIFEST_SHA256)
    manifest = load_json(root / "manifest.json")
    records = [item for item in manifest["states"] if item["admission"] == "evaluation"]
    expected = [item["state_id"] for item in records]
    actual = sorted(path.parent.name for path in root.glob("task*/result.json"))
    if actual != sorted(expected) or len(actual) != 30:
        raise RuntimeError("core state set is not the frozen 30-state cohort")
    task_counts: Counter[int] = Counter()
    for record in records:
        state_id = record["state_id"]
        state = root / state_id
        expected_names = {
            "actions.npz", "frozen_inputs.pt", "result.json",
            "future_b0.pt", "future_b1.pt", "future_b2.pt", "future_b3.pt",
        }
        if {p.name for p in state.iterdir()} != expected_names:
            raise RuntimeError(f"unexpected core state schema: {state}")
        result = load_json(state / "result.json")
        task_counts[int(state_id[4:6])] += 1
        checks = {
            "status": "complete",
            "state_id": state_id,
            "admission": "evaluation",
            "input_sha256": record["input_sha256"],
            "prompt": record["prompt"],
            "branch_ids": manifest["branch_ids"],
            "video_seeds": manifest["video_seeds"],
            "action_seeds": manifest["action_seeds"],
            "manifest_sha256": MANIFEST_SHA256,
            "runner_sha256": CORE_RUNNER_SHA256,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "action_coordinate_intervention": "none",
        }
        bad = {k: (result.get(k), v) for k, v in checks.items() if result.get(k) != v}
        if bad:
            raise RuntimeError(f"core identity/control mismatch {state_id}: {bad}")
        if result.get("native_self_latent_max_abs_error") != 0.0:
            raise RuntimeError(f"nonexact self-latent replay: {state_id}")
        if result.get("native_self_cache_max_abs_error") != 0.0:
            raise RuntimeError(f"nonexact self-cache replay: {state_id}")
        if result.get("native_future_hashes_unique") != 4 or result.get("native_cache_hashes_unique") != 4:
            raise RuntimeError(f"nonunique future/cache controls: {state_id}")
        if result.get("actions_sha256") != sha256_file(state / "actions.npz"):
            raise RuntimeError(f"action payload hash mismatch: {state_id}")
        if len(result.get("future_hashes", [])) != 4 or len(result.get("action_noise_hashes", [])) != 4:
            raise RuntimeError(f"incomplete future/noise hashes: {state_id}")
    if task_counts != Counter({task: 3 for task in range(10)}):
        raise RuntimeError(f"core task coverage changed: {task_counts}")
    return rows, aggregate, {"state_count": 30, "task_counts": dict(task_counts)}


def validate_dose(root: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows, aggregate = source_inventory(root)
    require_sha(root / "manifest.json", MANIFEST_SHA256)
    manifest = load_json(root / "manifest.json")
    expected = sorted(
        item["state_id"] for item in manifest["states"] if item["admission"] == "evaluation"
    )
    actual = sorted(path.parent.name for path in root.glob("task*/result.json"))
    if actual != expected or len(actual) != 30:
        raise RuntimeError("dose state set is not the frozen 30-state cohort")
    task_counts: Counter[int] = Counter()
    for state_id in expected:
        state = root / state_id
        if {p.name for p in state.iterdir()} != {"actions.npz", "result.json"}:
            raise RuntimeError(f"unexpected dose state schema: {state}")
        result = load_json(state / "result.json")
        task_counts[int(state_id[4:6])] += 1
        checks = {
            "status": "complete",
            "state_id": state_id,
            "admission": "evaluation",
            "manifest_sha256": MANIFEST_SHA256,
            "core_runner_sha256": CORE_RUNNER_SHA256,
            "dose_runner_sha256": DOSE_RUNNER_SHA256,
            "protocol_sha256": DOSE_PROTOCOL_SHA256,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "action_coordinate_intervention": "none",
            "interior_model_calls": 3,
        }
        bad = {k: (result.get(k), v) for k, v in checks.items() if result.get(k) != v}
        if bad:
            raise RuntimeError(f"dose identity/control mismatch {state_id}: {bad}")
        if result.get("actions_sha256") != sha256_file(state / "actions.npz"):
            raise RuntimeError(f"dose action payload hash mismatch: {state_id}")
    if task_counts != Counter({task: 3 for task in range(10)}):
        raise RuntimeError(f"dose task coverage changed: {task_counts}")
    return rows, aggregate, {"state_count": 30, "task_counts": dict(task_counts), "file_count": len(rows)}


def recursive_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in addendum: {path}")
        if not path.is_file() or path == root / "artifact_index.json":
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    return rows, aggregate.hexdigest()


def main() -> None:
    args = parse_args()
    core = args.core_root.resolve()
    dose = args.dose_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    if output in (core, dose) or output.is_relative_to(core) or output.is_relative_to(dose):
        raise RuntimeError("output must be disjoint from original raw roots")
    require_sha(args.existing_final_root / "artifact_index.json", FINAL_V1_INDEX_SHA256)
    require_sha(args.core_analysis_root / "artifact_index.json", CORE_ANALYSIS_INDEX_SHA256)
    require_sha(args.dose_analysis_root / "artifact_index.json", DOSE_ANALYSIS_INDEX_SHA256)
    require_sha(args.oracle_receipt, ORACLE_RECEIPT_SHA256)
    core_rows, core_aggregate, core_summary = validate_core(core)
    dose_rows, dose_aggregate, dose_summary = validate_dose(dose)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(core, staging / "core_raw")
        shutil.copytree(dose, staging / "dose_raw")
        shutil.copy2(Path(__file__).resolve(), staging / "freeze_lingbot_raw_addendum.py")
        copied_core_rows, copied_core_aggregate = source_inventory(staging / "core_raw")
        copied_dose_rows, copied_dose_aggregate = source_inventory(staging / "dose_raw")
        final_source_core_rows, final_source_core_aggregate = source_inventory(core)
        final_source_dose_rows, final_source_dose_aggregate = source_inventory(dose)
        if (final_source_core_rows, final_source_core_aggregate) != (core_rows, core_aggregate):
            raise RuntimeError("original core tree changed during copy")
        if (final_source_dose_rows, final_source_dose_aggregate) != (dose_rows, dose_aggregate):
            raise RuntimeError("original dose tree changed during copy")
        if (copied_core_rows, copied_core_aggregate) != (core_rows, core_aggregate):
            raise RuntimeError("copied core tree differs bytewise from original")
        if (copied_dose_rows, copied_dose_aggregate) != (dose_rows, dose_aggregate):
            raise RuntimeError("copied dose tree differs bytewise from original")
        receipt = {
            "schema_version": 1,
            "status": "complete_mode_frozen_read_only_addendum",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output),
            "policy": {
                "original_roots_preserved": True,
                "copy_is_bytewise_verified": True,
                "output_root_required_absent": True,
                "symlinks_allowed": False,
                "files_mode": "0444",
                "directories_mode": "0555",
            },
            "scope": {
                "core_raw": {**core_summary, "source_root": str(core), "file_count": len(core_rows), "source_tree_aggregate_sha256": core_aggregate},
                "dose_raw": {**dose_summary, "source_root": str(dose), "source_tree_aggregate_sha256": dose_aggregate},
            },
            "bindings": {
                "manifest_sha256": MANIFEST_SHA256,
                "core_runner_sha256": CORE_RUNNER_SHA256,
                "dose_runner_sha256": DOSE_RUNNER_SHA256,
                "dose_protocol_sha256": DOSE_PROTOCOL_SHA256,
                "upstream_commit": UPSTREAM_COMMIT,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "existing_final_v1_index_sha256": FINAL_V1_INDEX_SHA256,
                "core_analysis_index_sha256": CORE_ANALYSIS_INDEX_SHA256,
                "dose_analysis_index_sha256": DOSE_ANALYSIS_INDEX_SHA256,
                "official_oracle_receipt_sha256": ORACLE_RECEIPT_SHA256,
                "freezer_sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        write_json(staging / "raw_addendum_receipt.json", receipt)
        rows, aggregate = recursive_inventory(staging)
        index = {
            "schema_version": 1,
            "status": "complete_mode_frozen_read_only_addendum",
            "file_count_excluding_index": len(rows),
            "tree_aggregate_sha256": aggregate,
            "files": rows,
        }
        write_json(staging / "artifact_index.json", index)
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            path.chmod(0o555)
        staging.chmod(0o555)
        frozen_rows, frozen_aggregate = recursive_inventory(staging)
        if frozen_rows != rows or frozen_aggregate != aggregate:
            raise RuntimeError("addendum changed during freeze")
        if any(stat.S_IMODE(p.stat().st_mode) != 0o444 for p in staging.rglob("*") if p.is_file()):
            raise RuntimeError("not all files are 0444")
        if stat.S_IMODE(staging.stat().st_mode) != 0o555 or any(stat.S_IMODE(p.stat().st_mode) != 0o555 for p in staging.rglob("*") if p.is_dir()):
            raise RuntimeError("not all directories are 0555")
        os.replace(staging, output)
        print(json.dumps({
            "status": index["status"],
            "output_root": str(output),
            "artifact_index_sha256": sha256_file(output / "artifact_index.json"),
            "file_count_excluding_index": len(rows),
            "tree_aggregate_sha256": aggregate,
            "core_source_tree_aggregate_sha256": core_aggregate,
            "dose_source_tree_aggregate_sha256": dose_aggregate,
        }, sort_keys=True), flush=True)
    except BaseException:
        if staging.exists():
            for path in staging.rglob("*"):
                try:
                    path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
            staging.chmod(0o755)
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
