#!/usr/bin/env python3
"""Validate and freeze the final LingBot evaluation bundle without overwriting."""

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


CORE_INDEX_SHA256 = "0cc2ab978a157496018f1f43514b190630b1074ddd03381299718295bb51bab9"
CORE_BUILDER_SHA256 = "efdaaaf1236936bc9398f19d38a151905302b3dc10bb8d9eae02810a818ca441"
CORE_RUNNER_SHA256 = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
ORACLE_RECEIPT_SHA256 = "f54eebb2d4525ec8d8c57ebdc3b1294041194e5d1e13fd0033380a09453719aa"
ORACLE_SCRIPT_SHA256 = "893c2d9152575b583e1db0d8fafab79727c5038613099aa983fae1ad74f96afc"
DOSE_INDEX_SHA256 = "52211b2f463ed907468f0749783c67500a7a20d6699687cf0749392659d1dd93"
DOSE_RUNNER_SHA256 = "2d8b419be882eb979ed58091f7d0b0cd4322f2503aac9e4a854c558834f21b2e"
DOSE_ANALYZER_SHA256 = "03b899c5755f52023c094a1347760423f1f4c5d114757b3cc23b6e66ac367ac2"
DOSE_PROTOCOL_SHA256 = "8b6b4103b5c172f28c896b9834fda114aa52684c53f8c570c78c346fda9d3eba"
DOSE_CLARIFICATION_SHA256 = "2f3ca2211b66100c6d99d44e879b632bee1434da1f8d371fb2ed27f981ee7f8e"
PARITY_GATE_SHA256 = "6437f774088084b67e6aea001376304dfe01ee622358358dd40642d49d0a67d5"
ENVIRONMENT_ADDENDUM_SHA256 = "339b1c50892445ea2a8869bb5e96c3ac51d7e11f6187682bd2266cfc118b8bda"
SHIM_SHA256 = "7f1448bdeae5f4991112d78131688d417836c91fee79624929cda5d2f135bec8"
MANIFEST_SHA256 = "9c8d7dd547955e4622d98c7eb42fffed157f6db983f41f02929b28a434be47b4"
UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"

SOURCE_DIRECTORIES = (
    "core_artifacts",
    "upstream_parity",
    "dose_results",
    "dose_analysis",
    "canonical_scripts",
    "canonical_protocols",
    "logs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
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


def validate_index(root: Path, expected_index_sha256: str) -> dict[str, Any]:
    index_path = root / "artifact_index.json"
    require_sha(index_path, expected_index_sha256)
    index = load_json(index_path)
    if index.get("status") != "complete":
        raise RuntimeError(f"artifact index is not complete: {index_path}")
    for item in index.get("artifacts", []):
        path = root / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(f"indexed artifact changed: {path}")
    return index


def validate_source(source: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name in SOURCE_DIRECTORIES:
        path = source / name
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"required source directory missing or unsafe: {path}")

    core_index = validate_index(source / "core_artifacts", CORE_INDEX_SHA256)
    if core_index.get("generator_sha256") != CORE_BUILDER_SHA256:
        raise RuntimeError("core artifact builder identity changed")
    core_summary = load_json(source / "core_artifacts/summary.json")
    if (
        core_summary.get("status") != "complete"
        or core_summary.get("state_count") != 30
        or core_summary.get("all_exact_controls_pass") is not True
        or core_summary.get("primary_inferential_metric")
        != "retrieval_accuracy_off_diagonal"
        or core_summary.get("runner_sha256") != CORE_RUNNER_SHA256
        or core_summary.get("manifest_sha256") != MANIFEST_SHA256
        or core_summary.get("upstream_commit") != UPSTREAM_COMMIT
        or core_summary.get("checkpoint_revision") != CHECKPOINT_REVISION
    ):
        raise RuntimeError("core summary identity/control gate failed")

    oracle_path = source / "upstream_parity/upstream_native_parity.json"
    require_sha(oracle_path, ORACLE_RECEIPT_SHA256)
    oracle = load_json(oracle_path)
    comparison = oracle.get("comparison", {})
    if (
        oracle.get("status") != "complete"
        or oracle.get("included_in_evaluation") is not False
        or oracle.get("parity_gate_passed") is not True
        or oracle.get("audit_script", {}).get("sha256") != ORACLE_SCRIPT_SHA256
        or oracle.get("upstream_commit") != UPSTREAM_COMMIT
        or oracle.get("checkpoint_revision") != CHECKPOINT_REVISION
        or oracle.get("frozen_input", {}).get("official_encoder_bitwise_equal") is not True
        or oracle.get("frozen_input", {}).get("official_encoder_max_abs_error") != 0.0
        or comparison.get("future_bitwise_equal") is not True
        or comparison.get("future_max_abs_error") != 0.0
        or comparison.get("action_bitwise_equal") is not True
        or comparison.get("action_max_abs_error") != 0.0
        or oracle.get("controlled_rng_injection", {}).get("torch_randn_call_count") != 2
        or oracle.get("environment_addendum", {}).get("shim_call_count") != 0
        or oracle.get("environment_addendum", {}).get(
            "all_attention_modules_use_custom_torch_sdpa"
        )
        is not True
    ):
        raise RuntimeError("official upstream parity gate failed")

    dose_index = validate_index(source / "dose_analysis", DOSE_INDEX_SHA256)
    if dose_index.get("generator_sha256") != DOSE_ANALYZER_SHA256:
        raise RuntimeError("dose analyzer identity changed")
    dose_summary = load_json(source / "dose_analysis/summary.json")
    if (
        dose_summary.get("status") != "complete"
        or dose_summary.get("state_count") != 30
        or dose_summary.get("protocol_sha256") != DOSE_PROTOCOL_SHA256
        or dose_summary.get("analysis_clarification_sha256")
        != DOSE_CLARIFICATION_SHA256
        or dose_summary.get("oracle_receipt_sha256") != ORACLE_RECEIPT_SHA256
        or dose_summary.get("oracle_future_bitwise_equal") is not True
        or dose_summary.get("oracle_action_bitwise_equal") is not True
        or dose_summary.get("endpoint_cells_reused_from_core") is not True
        or dose_summary.get("endpoint_cells_excluded_from_primary_inference") is not True
        or dose_summary.get("outcome_selected_states_or_examples") is not False
    ):
        raise RuntimeError("dose analysis identity/control gate failed")

    raw_root = source / "dose_results"
    result_paths = sorted(raw_root.glob("task*/result.json"))
    action_paths = sorted(raw_root.glob("task*/actions.npz"))
    if len(result_paths) != 30 or len(action_paths) != 30:
        raise RuntimeError("dose raw result census is not exactly 30 states")
    state_ids = [path.parent.name for path in result_paths]
    if state_ids != [path.parent.name for path in action_paths] or len(set(state_ids)) != 30:
        raise RuntimeError("dose raw state/action pairing failed")
    task_counts: Counter[int] = Counter()
    for result_path, action_path in zip(result_paths, action_paths, strict=True):
        result = load_json(result_path)
        state_id = result_path.parent.name
        task_counts[int(state_id[4:6])] += 1
        if (
            result.get("status") != "complete"
            or result.get("state_id") != state_id
            or result.get("dose_runner_sha256") != DOSE_RUNNER_SHA256
            or result.get("protocol_sha256") != DOSE_PROTOCOL_SHA256
            or result.get("manifest_sha256") != MANIFEST_SHA256
            or result.get("oracle_receipt_sha256") != ORACLE_RECEIPT_SHA256
            or result.get("actions_sha256") != sha256_file(action_path)
            or result.get("action_coordinate_intervention") != "none"
            or result.get("interior_model_calls") != 3
        ):
            raise RuntimeError(f"dose raw identity/control gate failed: {state_id}")
    if task_counts != Counter({task_id: 3 for task_id in range(10)}):
        raise RuntimeError(f"dose task coverage changed: {task_counts}")

    exact_local_files = {
        "canonical_scripts/run_lingbot_future_transplants.py": CORE_RUNNER_SHA256,
        "canonical_scripts/build_lingbot_postrun_artifacts.py": CORE_BUILDER_SHA256,
        "canonical_scripts/audit_lingbot_upstream_native_parity.py": ORACLE_SCRIPT_SHA256,
        "canonical_scripts/run_lingbot_future_dose.py": DOSE_RUNNER_SHA256,
        "canonical_scripts/summarize_lingbot_future_dose.py": DOSE_ANALYZER_SHA256,
        "canonical_protocols/lingbot_future_dose_v1.json": DOSE_PROTOCOL_SHA256,
        "canonical_protocols/lingbot_future_dose_v1_analysis_clarification.json": DOSE_CLARIFICATION_SHA256,
        "canonical_protocols/lingbot_upstream_native_parity_gate_v1.json": PARITY_GATE_SHA256,
        "canonical_protocols/lingbot_upstream_native_parity_environment_addendum_v1.json": ENVIRONMENT_ADDENDUM_SHA256,
        "canonical_protocols/flash_attn/__init__.py": SHIM_SHA256,
    }
    for relative, expected in exact_local_files.items():
        require_sha(source / relative, expected)
    return core_summary, oracle, dose_summary


def readme_text() -> str:
    return """# Frozen LingBot evaluation package

This read-only package contains the complete 30-state, 10-task LingBot core evaluation, the excluded official-upstream parity audit, the outcome-blind prespecified b0-to-b1 dose follow-up, raw dose outputs, exact executed scripts/protocols, logs, and recursive SHA-256 inventory.

The claim-facing core statistic is off-diagonal future-source retrieval over the full executed three-frame action chunk. LingBot's conditioned first action frame, which LIBERO does not execute, is excluded. Post hoc slice diagnostics are materially weaker at the earliest executed boundary, so the whole-chunk result must not be described as uniformly strong or immediate.

The factorial arrays are cache-routing/interface controls. Holding the recipient cache fixes the action near the recipient even when the raw future label changes; installing the donor future-derived cache redirects the action toward the donor. Because the released action stage reads that cache, this is not evidence for an additional independently identified raw-future pathway.

Distinct future hashes establish distinct latent tensors only. They do not establish visually or semantically distinct imagined content.

The dose result supports pathwise sensitivity along the fixed b0-to-b1 normalized-latent segment. It is not a semantic intervention over future concepts or evidence about naturalness. The protocol was frozen before any dose execution and without outcome inspection while the separate core cohort was still completing; call it an outcome-blind prespecified/frozen follow-up, not a public preregistration.

Representative media are selection-neutral: `core_artifacts/media/all_states_overview.mp4` and `core_artifacts/media/all_states_contact_sheet.png` include all frozen states in predetermined order. No example was chosen for effect size or appearance.
"""


def make_provenance(
    output_root: Path,
    core_summary: dict[str, Any],
    oracle: dict[str, Any],
    dose_summary: dict[str, Any],
    packager_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete_pending_recursive_inventory",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(output_root),
        "package_policy": {
            "output_root_required_absent": True,
            "regular_file_mode_after_freeze": "0444",
            "directory_mode_after_freeze": "0555",
            "symlinks_allowed": False,
            "recursive_sha256_inventory": "artifact_index.json",
        },
        "identities": {
            "manifest_sha256": MANIFEST_SHA256,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "core_runner_sha256": CORE_RUNNER_SHA256,
            "core_builder_sha256": CORE_BUILDER_SHA256,
            "oracle_script_sha256": ORACLE_SCRIPT_SHA256,
            "oracle_receipt_sha256": ORACLE_RECEIPT_SHA256,
            "dose_protocol_sha256": DOSE_PROTOCOL_SHA256,
            "dose_runner_sha256": DOSE_RUNNER_SHA256,
            "dose_analyzer_sha256": DOSE_ANALYZER_SHA256,
            "packager_sha256": packager_sha256,
        },
        "core": {
            "state_count": core_summary["state_count"],
            "task_counts": core_summary["task_counts"],
            "primary_inferential_metric": core_summary["primary_inferential_metric"],
            "bootstrap": core_summary["bootstrap"],
            "permutation": core_summary["permutation"],
            "all_exact_controls_pass": core_summary["all_exact_controls_pass"],
            "source_artifact_index_sha256": CORE_INDEX_SHA256,
        },
        "official_upstream_oracle": {
            "included_in_evaluation": False,
            "parity_gate_passed": oracle["parity_gate_passed"],
            "frozen_input": oracle["frozen_input"],
            "comparison": oracle["comparison"],
            "controlled_rng_injection": oracle["controlled_rng_injection"],
            "environment_addendum": oracle["environment_addendum"],
        },
        "dose": {
            **dose_summary,
            "source_artifact_index_sha256": DOSE_INDEX_SHA256,
            "freeze_description": "outcome-blind prespecified/frozen follow-up; not a public preregistration",
        },
        "interpretation_limits": {
            "core_primary_estimand": "full executed three-frame action chunk; conditioned first frame excluded",
            "earliest_boundary": "post hoc slice diagnostics are materially weaker at the earliest executed boundary",
            "factorial": "cache-routing/interface control; action follows installed future-derived cache identity",
            "future_hashes": "tensor distinctness only; not visual or semantic distinctness",
            "dose": "pathwise sensitivity on one fixed b0-to-b1 segment only",
        },
        "media": {
            "selection_uses_outcomes": False,
            "selection_rule": "all 30 frozen states in predetermined order",
            "overview_video": "core_artifacts/media/all_states_overview.mp4",
            "contact_sheet": "core_artifacts/media/all_states_contact_sheet.png",
            "policy": "core_artifacts/media/selection_policy.json",
        },
        "provenance_limits": {
            "core_launch_receipt_captured_pythonpath": False,
            "core_pythonpath_inference": "Import success plus prelaunch shim mtime/hash; exact path independently required by oracle",
            "launch_time_checkpoint_digest_available": False,
            "checkpoint_binding": "Postrun full payload hashes bound one-to-one to prelaunch Hugging Face metadata/LFS etags and revision; payload/metadata mtimes predate launch",
        },
    }


def recursive_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in frozen package: {path}")
        if not path.is_file() or path == root / "artifact_index.json":
            continue
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return rows, aggregate.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen package: {output}")
    if output == source or output.is_relative_to(source):
        raise RuntimeError("output package must be disjoint from source root")
    core_summary, oracle, dose_summary = validate_source(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name in SOURCE_DIRECTORIES:
            shutil.copytree(source / name, staging / name)
        packager_copy = staging / "canonical_scripts" / Path(__file__).name
        shutil.copy2(Path(__file__).resolve(), packager_copy)
        packager_sha256 = sha256_file(packager_copy)
        (staging / "README.md").write_text(readme_text(), encoding="utf-8")
        provenance = make_provenance(
            output, core_summary, oracle, dose_summary, packager_sha256
        )
        provenance["status"] = "complete_mode_frozen_read_only_package"
        write_json(staging / "final_provenance.json", provenance)
        rows, aggregate = recursive_inventory(staging)
        index = {
            "schema_version": 1,
            "status": "complete_mode_frozen_read_only_package",
            "package_root": str(output),
            "file_count_excluding_index": len(rows),
            "tree_aggregate_sha256": aggregate,
            "files": rows,
        }
        write_json(staging / "artifact_index.json", index)

        for path in sorted(staging.rglob("*")):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            path.chmod(0o555)
        staging.chmod(0o555)
        frozen_rows, frozen_aggregate = recursive_inventory(staging)
        if frozen_rows != rows or frozen_aggregate != aggregate:
            raise RuntimeError("package changed during mode freeze/install")
        if any(
            stat.S_IMODE(path.stat().st_mode) != 0o444
            for path in staging.rglob("*")
            if path.is_file()
        ):
            raise RuntimeError("not every package file is mode 0444")
        if stat.S_IMODE(staging.stat().st_mode) != 0o555 or any(
            stat.S_IMODE(path.stat().st_mode) != 0o555
            for path in staging.rglob("*")
            if path.is_dir()
        ):
            raise RuntimeError("not every package directory is mode 0555")
        os.replace(staging, output)
        print(
            json.dumps(
                {
                    "status": index["status"],
                    "output_root": str(output),
                    "file_count_excluding_index": len(rows),
                    "tree_aggregate_sha256": aggregate,
                    "artifact_index_sha256": sha256_file(output / "artifact_index.json"),
                },
                sort_keys=True,
            )
        )
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
