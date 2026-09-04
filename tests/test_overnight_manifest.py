from pathlib import Path

import pytest

from scripts.build_overnight_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_overnight_manifest_is_complete_and_outcome_blind(tmp_path: Path) -> None:
    manifest = build_manifest(
        ROOT / "configs" / "overnight_2026-09-03.toml",
        tmp_path / "recordings",
    )

    assert manifest["candidate_count"] == 48
    assert manifest["task_count"] == 6
    assert manifest["states_per_task"] == 8
    assert not manifest["selection_uses_native_or_intervention_outcomes"]
    assert not manifest["replacement_after_outcome"]
    assert len({row["unit_id"] for row in manifest["candidates"]}) == 48
    assert all(row["selected"] for row in manifest["candidates"])
    assert all(row["target_object_name"] for row in manifest["candidates"])
    assert all(row["branch_seeds"] == [211, 223, 227, 229] for row in manifest["candidates"])
    assert all(row["recipient_seed"] == 211 for row in manifest["candidates"])
    assert all(row["donor_seeds"] == [223, 227, 229] for row in manifest["candidates"])
    assert manifest["action_pair_design"] == "all_12_ordered_recipient_to_donor_pairs"
    assert manifest["physical_endpoint_design"] == "fixed_recipient_211_to_three_donors"
    assert all(len(row["action_ordered_pairs"]) == 12 for row in manifest["candidates"])
    assert all(
        len({tuple(pair) for pair in row["action_ordered_pairs"]}) == 12
        for row in manifest["candidates"]
    )


def test_unfrozen_config_is_rejected(tmp_path: Path) -> None:
    original = (ROOT / "configs" / "overnight_2026-09-03.toml").read_text()
    config = tmp_path / "unfrozen.toml"
    config.write_text(original.replace('status = "frozen_before_outcomes"', 'status = "development"', 1))

    with pytest.raises(ValueError, match="not frozen"):
        build_manifest(config, tmp_path / "recordings")
