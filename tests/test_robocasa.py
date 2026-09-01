from __future__ import annotations

import numpy as np
import pytest
from imagined_future.robocasa import (
    environment_action,
    physical_observation_vector,
    physical_state_vector,
)


def test_environment_action_matches_public_fixed_base_adapter() -> None:
    mapped = environment_action(np.arange(7), 12)
    assert mapped.tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, -1]
    with pytest.raises(ValueError, match="cannot map"):
        environment_action(np.arange(6), 12)


def test_physical_observation_vector_is_stable_and_excludes_images() -> None:
    vector, schema = physical_observation_vector(
        {"z": np.asarray([3.0]), "camera_image": np.zeros((2, 2, 3)), "a": np.asarray([1.0, 2.0])}
    )
    assert vector.tolist() == [1.0, 2.0, 3.0]
    assert [item["key"] for item in schema] == ["a", "z"]


def test_physical_state_vector_appends_articulated_generalized_positions() -> None:
    vector, schema = physical_state_vector({"a": np.asarray([1.0])}, np.asarray([2.0, 3.0]))
    assert vector.tolist() == [1.0, 2.0, 3.0]
    assert schema[-1] == {"key": "sim.data.qpos", "start": 1, "stop": 3}
