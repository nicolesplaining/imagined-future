#!/usr/bin/env python3
"""Fail-closed consistency checks for the 2026-09-04 deadline handoff."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> dict:
    value = json.loads((ROOT / relative).read_text())
    if not isinstance(value, dict):
        raise TypeError(relative)
    return value


def close(left: str | float, right: float, tolerance: float = 5e-7) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    rows = {
        row["model"]: row
        for row in csv.DictReader(
            (ROOT / "output/deadline_2026_09_04/cross_model_results_table.csv").open()
        )
    }
    dream = load_json(
        "output/deadline_2026_09_04/dreamzero/core_analysis_final/summary.json"
    )
    ling = load_json(
        "output/deadline_2026_09_04/lingbot/core_artifacts/summary.json"
    )
    expected = {
        "DreamZero": {
            "correct_source_rate": dream["estimates"][
                "retrieval_accuracy_off_diagonal"
            ]["mean"],
            "distance_reduction": dream["estimates"]["distance_reduction"]["mean"],
            "normalized_projection": dream["estimates"]["normalized_projection"]["mean"],
            "cosine_alignment": dream["estimates"]["cosine_alignment"]["mean"],
            "orthogonal_residual": dream["estimates"]["orthogonal_residual"]["mean"],
        }
    }
    ling_metrics = {item["metric"]: item["estimate"] for item in ling["bootstrap"]}
    expected["LingBot-VA"] = {
        "correct_source_rate": ling_metrics["retrieval_accuracy_off_diagonal"],
        "distance_reduction": ling_metrics["distance_reduction"],
        "normalized_projection": ling_metrics["projection"],
        "cosine_alignment": ling_metrics["cosine_alignment"],
        "orthogonal_residual": ling_metrics["orthogonal_residual"],
    }
    for model, metrics in expected.items():
        for key, value in metrics.items():
            assert close(rows[model][key], value), (model, key)
    print("CORE_TABLE_VS_SUMMARIES PASS")

    dose = list(
        csv.DictReader(
            (ROOT / "output/deadline_2026_09_04/cross_model_dose_table.csv").open()
        )
    )
    assert len(dose) == 2
    assert dose[0]["projection_definition"] == (
        "native_recipient_action_to_native_donor_action_axis"
    )
    assert "endpoints_zero_and_one_by_construction" in dose[1]["projection_definition"]
    assert all("interior_nondecreasing_states" in row for row in dose)
    print("DOSE_TABLE_SCHEMA_AND_DEFINITIONS PASS")

    gaussian_rows = list(
        csv.DictReader(
            (
                ROOT
                / "output/deadline_2026_09_04/lingbot_gaussian_routing_table.csv"
            ).open()
        )
    )
    assert len(gaussian_rows) == 3 and all(None not in row for row in gaussian_rows)
    gaussian = load_json(
        "output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final/summary.json"
    )
    assert close(
        gaussian_rows[0]["rate"],
        gaussian["gaussian_source_retrieval_off_diagonal"]["bootstrap"]["estimate"],
    )
    assert close(
        gaussian_rows[1]["rate"],
        gaussian["native_donor_alignment_off_diagonal"]["bootstrap"]["estimate"],
    )
    assert close(
        gaussian_rows[2]["rate"],
        gaussian["native_recipient_retention_off_diagonal"]["estimate"],
    )
    print("GAUSSIAN_TABLE_VS_SUMMARY PASS")

    for relative in (
        "docs/deadline_2026_09_04/noon_final_results.md",
        "docs/deadline_2026_09_04/README.md",
        "docs/deadline_2026_09_04/goal_requirement_matrix.md",
        "docs/deadline_2026_09_04/claim_safe_synthesis.md",
        "docs/deadline_2026_09_04/final_science_adversarial_audit.md",
        "docs/deadline_2026_09_04/dreamzero_audit_results.md",
        "docs/deadline_2026_09_04/dreamzero_provenance_reconciliation.md",
        "docs/deadline_2026_09_04/dreamzero_kv_factorial_feasibility.md",
        "docs/deadline_2026_09_04/dreamzero_future_kv_factorial_protocol.md",
        "docs/deadline_2026_09_04/lambda_storage.md",
    ):
        path = ROOT / relative
        missing = []
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            if not (path.parent / target).resolve().exists():
                missing.append(target)
        assert not missing, (relative, missing)
        print(f"LOCAL_LINKS PASS {relative}")

    immutable_hashes = {
        "output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final/artifact_index.json":
            "6b53379df8c5a1f0030e7429df8527df33f68529482d144e60557ef8121af0ad",
        "output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1_execution_addendum_v2/artifact_index.json":
            "052f4c41f132ae1a42ed6eb5a715e8f744887e64c166a7febea235949a313ec3",
        "output/deadline_2026_09_04/dreamzero/all_native_media/receipt.json":
            "89c7f4742b42698d2a2cb122f63c5792d6e53da070280c019ad18a1fb3acacdb",
        "output/deadline_2026_09_04/dreamzero/all_native_media_derived/receipt.json":
            "80d0180c3df8fa88e715c7869c1345661c2fd4d85dbd7d062dd9d326cf6b533b",
        "output/deadline_2026_09_04/dreamzero/upstream_native_parity/artifact_index.json":
            "5d2e3c71b85c9e2327337139062715aabbe9c6beae4b7d4fc2a5139c74da270f",
    }
    for relative, expected_hash in immutable_hashes.items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected_hash, (relative, observed, expected_hash)
    print("KEY_FROZEN_HASHES PASS")

    dream_parity = load_json(
        "output/deadline_2026_09_04/dreamzero/upstream_native_parity/execution_receipt.json"
    )
    assert dream_parity["official_commit"] == "ab790c198fbce33503358efbbd4187ce9a89adf3"
    assert dream_parity["bitwise_exact"] is True
    assert dream_parity["maximum_absolute_error"] == 0.0
    print("DREAMZERO_CLEAN_UPSTREAM_PARITY PASS")

    closure = load_json("output/deadline_2026_09_04/compute_closure.json")
    assert closure["instances_stopped_or_deleted"] is False
    assert all(node["experimental_processes_remaining"] == 0 for node in closure["nodes"])
    assert all(node["gpu_memory_mib"] == [0, 0] for node in closure["nodes"])
    print("COMPUTE_CLOSURE_RECEIPT PASS")
    print("FINAL_HANDOFF_VERIFICATION PASS")


if __name__ == "__main__":
    main()
