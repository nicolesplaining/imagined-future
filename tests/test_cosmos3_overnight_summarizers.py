from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from overnight_summary_common import SummaryValidationError  # noqa: E402
from summarize_cosmos3_kv_factorial import (  # noqa: E402
    aggregate as aggregate_kv,
    build_contrast_rows,
    expected_cell_labels,
    extract_rows as extract_kv_rows,
)
from summarize_cosmos3_selection_free import (  # noqa: E402
    aggregate as aggregate_selection,
    extract_rows as extract_selection_rows,
)


SEEDS = [211, 223, 227, 229]


def target_intervention(
    value: float, target: int | None, *, physically_executed: bool = True
) -> dict:
    endpoint_value = {
        group: value
        for group in ("all", "robot", "object", "target_object_position")
    }
    endpoint_l2 = {group: 1.0 - value for group in endpoint_value}
    endpoint_native = {group: 1.0 for group in endpoint_value}
    if not physically_executed:
        endpoint_value = {group: None for group in endpoint_value}
        endpoint_l2 = {group: None for group in endpoint_l2}
        endpoint_native = {group: None for group in endpoint_native}
    return {
        "physically_executed": physically_executed,
        "target_donor_seed": target,
        "action_donor_projection": value,
        "action_target_donor_projection": value,
        "action_l2_to_target_donor": 1.0 - value,
        "action_native_target_l2": 1.0,
        "distance_reduction_to_target": value,
        "cosine_alignment": 1.0,
        "orthogonal_residual_normalized": 0.1,
        "nearest_native_action_seed": target,
        "correct_action_donor_top1": target is not None,
        "endpoint_donor_projection": endpoint_value,
        "endpoint_target_donor_projection": endpoint_value,
        "endpoint_l2_to_target_donor": endpoint_l2,
        "endpoint_native_target_l2": endpoint_native,
        "endpoint_distance_reduction_to_target": endpoint_value,
        "endpoint_cosine_alignment": {
            group: (1.0 if physically_executed else None) for group in endpoint_value
        },
        "endpoint_orthogonal_residual_normalized": {
            group: (0.1 if physically_executed else None) for group in endpoint_value
        },
        "nearest_native_endpoint_seed": {group: target for group in endpoint_value},
        "correct_endpoint_donor_top1": {
            group: (target is not None if physically_executed else None)
            for group in endpoint_value
        },
    }


def selection_expected() -> dict:
    return {
        "unit_id": "BananaInBowlTask-seed-3554",
        "task": "BananaInBowlTask",
        "environment_seed": 3554,
        "branch_step": 64,
        "branch_seeds": SEEDS,
        "recipient_seed": 211,
        "donor_seeds": [223, 227, 229],
        "target_object_name": "banana",
        "action_ordered_pairs": [
            [recipient, donor]
            for recipient in SEEDS
            for donor in SEEDS
            if recipient != donor
        ],
    }


def selection_report() -> dict:
    action_rows = []
    for source in ("predicted", "executed"):
        for recipient in SEEDS:
            for donor in SEEDS:
                if donor == recipient:
                    continue
                row = target_intervention(0.8, donor)
                row.update(
                    {
                        "future_source": source,
                        "recipient_seed": recipient,
                        "target_donor_seed": donor,
                        "physically_executed": False,
                    }
                )
                action_rows.append(row)
    interventions = {}
    for source in ("predicted", "executed"):
        for donor in (223, 227, 229):
            label = f"{source}_donor" if donor == 223 else f"{source}_donor_seed_{donor}"
            interventions[label] = target_intervention(0.75, donor)
    return {
        "study_id": "selection-free-banana-3554",
        "task": "BananaInBowlTask",
        "environment_seed": 3554,
        "branch_step": 64,
        "branch_seeds": SEEDS,
        "recipient_seed": 211,
        "donor_seed": 223,
        "target_object_name": "banana",
        "multi_donor": True,
        "frozen_pair_supplied": True,
        "within_run_pair_selection": False,
        "native_execution_seeds": SEEDS,
        "multi_donor_target_seeds": [223, 227, 229],
        "all_recipient_action_grid": True,
        "action_grid": {
            "candidate_seeds": SEEDS,
            "ordered_pairs": [
                [recipient, donor]
                for recipient in SEEDS
                for donor in SEEDS
                if donor != recipient
            ],
            "ordered_pair_count": 12,
            "future_sources": ["predicted", "executed"],
            "intervention_count": 24,
            "native_repeat_exact": {str(seed): True for seed in SEEDS},
            "native_repeat_action_maximum_error": {str(seed): 0.0 for seed in SEEDS},
            "clean_self_clamp_repeat_exact": {str(seed): True for seed in SEEDS},
            "clean_self_clamp_repeat_action_maximum_error": {
                str(seed): 0.0 for seed in SEEDS
            },
            "clean_self_clamp_error_from_native": {
                str(seed): {"maximum_absolute": 0.05, "l2": 0.2} for seed in SEEDS
            },
            "rows": action_rows,
        },
        "interventions": interventions,
    }


def test_selection_free_prefers_12_pair_action_grid_and_fixed_recipient_endpoints() -> None:
    rows = extract_selection_rows(
        selection_report(), Path("state/summary.json"), selection_expected()
    )

    action = [row for row in rows if row["domain"] == "action"]
    endpoints = [row for row in rows if row["domain"].startswith("endpoint_")]
    assert len(action) == 24
    assert len(endpoints) == 24
    assert {row["recipient_seed"] for row in action} == set(SEEDS)
    assert {row["recipient_seed"] for row in endpoints} == {211}
    assert all(row["top1"] == 1.0 for row in rows)

    summary, state_rows = aggregate_selection(
        rows, chance=0.25, resamples=100, seed=7
    )
    assert summary["estimates"]["predicted"]["action"]["top1"]["mean"] == 1.0
    assert summary["estimates"]["executed"]["endpoint_robot"][
        "distance_reduction"
    ]["mean"] == 0.75
    assert state_rows


def test_selection_free_fails_loudly_on_missing_frozen_metric() -> None:
    report = selection_report()
    del report["action_grid"]["rows"][0]["cosine_alignment"]

    with pytest.raises(SummaryValidationError, match="cosine_alignment"):
        extract_selection_rows(report, Path("state/summary.json"), selection_expected())


def kv_report() -> dict:
    values = {
        "recipient_future_recipient_kv": 0.0,
        "donor_future_recipient_kv": 0.2,
        "recipient_future_donor_kv": 0.6,
        "donor_future_donor_kv": 1.0,
    }
    interventions = {
        "self": target_intervention(0.0, 223),
        "predicted_donor": target_intervention(1.0, 223),
        "executed_donor": target_intervention(1.0, 223),
        "gaussian_executed": target_intervention(
            0.0, None, physically_executed=False
        ),
        "executed_self": target_intervention(0.0, None, physically_executed=False),
    }
    action_cell_maps = {}
    endpoint_cell_maps = {}
    for source in ("predicted", "executed"):
        labels = expected_cell_labels(source)
        action_cell_maps[source] = labels
        endpoint_cell_maps[source] = {
            "recipient_future_recipient_kv": "self",
            "donor_future_recipient_kv": f"{source}_donor_kv_patch_all_action",
            "recipient_future_donor_kv": f"self_with_{source}_donor_kv",
            "donor_future_donor_kv": f"{source}_donor",
        }
        for cell, label in labels.items():
            interventions[label] = target_intervention(
                values[cell],
                223,
                physically_executed=(
                    label.endswith("_kv_patch_all_action")
                    or label.startswith("self_with_")
                ),
            )
        interventions[f"{source}_donor_kv_replay"] = copy.deepcopy(
            interventions[f"{source}_donor_kv_record"]
        )
    interventions["self_kv_patch_all"] = copy.deepcopy(interventions["self_kv_record"])
    physical_labels = sorted(
        label for label, arm in interventions.items() if arm["physically_executed"]
    )
    action_only_labels = sorted(set(interventions) - set(physical_labels))
    return {
        "study_id": "kv-banana-103",
        "task": "BananaInBowlTask",
        "environment_seed": 103,
        "branch_step": 64,
        "recipient_seed": 211,
        "donor_seed": 223,
        "target_object_name": "banana",
        "attention_kv_patch_layers": list(range(36)),
        "attention_kv_factorial": True,
        "minimal_kv_factorial": True,
        "attention_kv_factorial_cells": action_cell_maps,
        "attention_kv_factorial_endpoint_cells": endpoint_cell_maps,
        "physically_executed_intervention_labels": physical_labels,
        "action_only_intervention_labels": action_only_labels,
        "kv_patch_identity_action_maximum_errors": {
            "self_kv_patch_all": 0.0,
            "predicted_donor_kv_record": 0.0,
            "predicted_donor_kv_replay": 0.0,
            "executed_donor_kv_record": 0.0,
            "executed_donor_kv_replay": 0.0,
        },
        "interventions": interventions,
    }


def test_kv_factorial_extracts_cells_and_correct_contrasts() -> None:
    cells, audits = extract_kv_rows(kv_report(), Path("kv/summary.json"), None)
    contrasts = build_contrast_rows(cells)

    assert len(cells) == 40
    assert len(audits) == 5
    action_projection = {
        row["contrast"]: row["value"]
        for row in contrasts
        if row["future_source"] == "predicted"
        and row["domain"] == "action"
        and row["metric"] == "normalized_projection"
    }
    assert action_projection["suppression"] == pytest.approx(0.8)
    assert action_projection["rescue"] == pytest.approx(0.6)
    assert action_projection["interaction"] == pytest.approx(0.2)

    summary = aggregate_kv(cells, contrasts, resamples=100, seed=11)
    suppression = summary["contrast_estimates"]["predicted"]["action"][
        "normalized_projection"
    ]["suppression"]
    assert suppression["mean"] == pytest.approx(0.8)


def test_kv_factorial_rejects_nonexact_replay() -> None:
    report = kv_report()
    report["kv_patch_identity_action_maximum_errors"]["predicted_donor_kv_replay"] = 1e-7

    with pytest.raises(SummaryValidationError, match="not exact"):
        extract_kv_rows(report, Path("kv/summary.json"), None)


def test_kv_factorial_rejects_nonminimal_execution_metadata() -> None:
    report = kv_report()
    report["interventions"]["predicted_donor_kv_record"]["physically_executed"] = True

    with pytest.raises(SummaryValidationError, match="physically_executed conflicts"):
        extract_kv_rows(report, Path("kv/summary.json"), None)


def test_kv_factorial_rejects_wrong_pair_or_layer_scope() -> None:
    wrong_pair = kv_report()
    wrong_pair["donor_seed"] = 227
    with pytest.raises(SummaryValidationError, match="recipient 211 and donor 223"):
        extract_kv_rows(wrong_pair, Path("kv/summary.json"), None)

    wrong_layers = kv_report()
    wrong_layers["attention_kv_patch_layers"] = [16]
    with pytest.raises(SummaryValidationError, match="exactly 0..35"):
        extract_kv_rows(wrong_layers, Path("kv/summary.json"), None)


def test_kv_exact_replay_accepts_matching_undefined_projection() -> None:
    report = kv_report()
    for source in ("predicted", "executed"):
        for label in (f"{source}_donor_kv_record", f"{source}_donor_kv_replay"):
            report["interventions"][label]["action_donor_projection"] = float("nan")

    cells, _audits = extract_kv_rows(report, Path("kv/summary.json"), None)
    assert len(cells) == 40


def test_selection_and_kv_require_named_task_object() -> None:
    selection = selection_report()
    del selection["target_object_name"]
    with pytest.raises(SummaryValidationError, match="target_object_name"):
        extract_selection_rows(
            selection, Path("state/summary.json"), selection_expected()
        )

    kv = kv_report()
    del kv["target_object_name"]
    with pytest.raises(SummaryValidationError, match="target_object_name"):
        extract_kv_rows(kv, Path("kv/summary.json"), None)
