from __future__ import annotations

import pytest

from imagined_future.factorial import two_by_two_effects


def test_two_by_two_effects_recovers_additive_terms() -> None:
    effects = two_by_two_effects(
        {"o0r0": 3.0, "o1r0": 5.0, "o0r1": 7.0, "o1r1": 9.0}
    )
    assert effects == {
        "object_main_effect": 2.0,
        "robot_main_effect": 4.0,
        "interaction": 0.0,
    }


def test_two_by_two_effects_reports_interaction() -> None:
    effects = two_by_two_effects(
        {"o0r0": 0.0, "o1r0": 1.0, "o0r1": 2.0, "o1r1": 7.0}
    )
    assert effects["object_main_effect"] == pytest.approx(3.0)
    assert effects["robot_main_effect"] == pytest.approx(4.0)
    assert effects["interaction"] == pytest.approx(4.0)


def test_two_by_two_effects_requires_all_cells() -> None:
    with pytest.raises(ValueError, match="o1r1"):
        two_by_two_effects({"o0r0": 0.0, "o1r0": 1.0, "o0r1": 2.0})
