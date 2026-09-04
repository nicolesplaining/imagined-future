from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imagined_future.cosmos3_archival import (
    BRANCH_SEEDS,
    admissible_branch_steps,
    atomic_json,
    compose_cosmos_observation,
    deterministic_shuffled_source,
    deterministic_wrong_donor,
    phase_branch_steps,
    recorded_proprio,
)


def test_admissible_steps_are_strictly_valid_and_16_mod_32() -> None:
    assert admissible_branch_steps(113) == (16, 48, 80, 112)
    assert all(step % 32 == 16 for step in admissible_branch_steps(500))
    assert admissible_branch_steps(16) == ()


def test_phase_mapping_uses_fixed_quantiles_and_distinct_steps() -> None:
    rows = phase_branch_steps(247)
    assert [row["phase"] for row in rows] == ["early", "middle", "late"]
    assert [row["branch_step"] for row in rows] == [48, 112, 208]
    assert [row["mp4_frame_index"] for row in rows] == [47, 111, 207]
    assert all(int(row["branch_step"]) % 32 == 16 for row in rows)


def test_phase_mapping_breaks_equal_distance_tie_downward() -> None:
    # For length 129, targets 32/64/96 lie midway between admissible steps.
    rows = phase_branch_steps(
        129, phases=(("one", 0.25), ("two", 0.50), ("three", 0.75))
    )
    assert [row["branch_step"] for row in rows] == [16, 48, 80]


def test_phase_mapping_stops_instead_of_adapting_short_episode() -> None:
    with pytest.raises(ValueError, match="three distinct"):
        phase_branch_steps(80)


def test_compose_uses_wrist_above_left_and_right() -> None:
    frame = np.zeros((360, 2560, 3), dtype=np.uint8)
    frame[:, 0:640] = 11
    frame[:, 640:1280] = 22
    frame[:, 1280:1920] = 33
    frame[:, 1920:2560] = 44
    composed = compose_cosmos_observation(frame)
    assert composed.shape == (540, 640, 3)
    assert composed.dtype == np.uint8
    assert np.all(composed[:360] == 44)
    assert np.all(composed[360:, :320] == 22)
    assert np.all(composed[360:, 320:] == 33)


def test_recorded_proprio_matches_policy_normalization() -> None:
    raw = np.asarray([1, 2, 3, 4, 5, 6, 7, np.pi / 8, 99], dtype=np.float32)
    joints, gripper = recorded_proprio(raw)
    assert joints.tolist() == pytest.approx([1, 2, 3, 4, 5, 6, 7])
    assert gripper.tolist() == pytest.approx([0.5])


def test_wrong_donor_is_neither_recipient_nor_true_donor() -> None:
    for recipient in BRANCH_SEEDS:
        mapped = []
        for donor in BRANCH_SEEDS:
            if recipient == donor:
                continue
            wrong = deterministic_wrong_donor(recipient, donor)
            mapped.append(wrong)
            assert wrong in BRANCH_SEEDS
            assert wrong not in {recipient, donor}
        assert sorted(mapped) == sorted(seed for seed in BRANCH_SEEDS if seed != recipient)


def test_four_way_source_shuffle_is_a_derangement_and_bijection() -> None:
    mapped = [deterministic_shuffled_source(seed) for seed in BRANCH_SEEDS]
    assert sorted(mapped) == sorted(BRANCH_SEEDS)
    assert all(source != shuffled for source, shuffled in zip(BRANCH_SEEDS, mapped))


def test_atomic_json_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    atomic_json(output, {"finite": 1.0})
    assert json.loads(output.read_text()) == {"finite": 1.0}
    assert output.stat().st_mode & 0o777 == 0o644
    with pytest.raises(FileExistsError):
        atomic_json(output, {"finite": 2.0})
