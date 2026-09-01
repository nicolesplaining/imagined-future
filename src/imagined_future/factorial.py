"""Small, audited helpers for 2x2 object/robot estimands."""

from __future__ import annotations

from collections.abc import Mapping


def two_by_two_effects(values: Mapping[str, float]) -> dict[str, float]:
    """Return equal-weight main effects and the difference of differences."""

    missing = {"o0r0", "o1r0", "o0r1", "o1r1"} - values.keys()
    if missing:
        raise ValueError(f"missing factorial cells: {sorted(missing)}")
    o0r0 = float(values["o0r0"])
    o1r0 = float(values["o1r0"])
    o0r1 = float(values["o0r1"])
    o1r1 = float(values["o1r1"])
    return {
        "object_main_effect": 0.5 * ((o1r0 - o0r0) + (o1r1 - o0r1)),
        "robot_main_effect": 0.5 * ((o0r1 - o0r0) + (o1r1 - o1r0)),
        "interaction": o1r1 - o1r0 - o0r1 + o0r0,
    }
