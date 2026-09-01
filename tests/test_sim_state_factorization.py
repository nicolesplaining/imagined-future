from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from imagined_future.sim_state_factorization import (
    RobotStateIndices,
    factorized_flat_state,
    robot_state_indices,
)


def test_robot_state_indices_includes_arm_and_gripper() -> None:
    model = SimpleNamespace(
        get_joint_qpos_addr=lambda name: {"arm": 1, "finger": (4, 6)}[name],
        get_joint_qvel_addr=lambda name: {"arm": 0, "finger": (3, 5)}[name],
    )
    robot = SimpleNamespace(
        robot_joints=["arm"],
        gripper_joints=["finger"],
        sim=SimpleNamespace(model=model),
    )
    indices = robot_state_indices(robot)
    assert indices == RobotStateIndices((1, 4, 5), (0, 3, 4))


def test_factorized_state_copies_only_robot_coordinates() -> None:
    # Flattened state layout: [time, qpos(4), qvel(3)].
    objects = np.arange(8, dtype=np.float64)
    robot = np.arange(100, 108, dtype=np.float64)
    hybrid = factorized_flat_state(
        objects,
        robot,
        nq=4,
        nv=3,
        robot_indices=RobotStateIndices((1, 3), (0, 2)),
    )
    np.testing.assert_array_equal(
        hybrid,
        np.asarray([0, 1, 102, 3, 104, 105, 6, 107], dtype=np.float64),
    )


def test_factorized_state_rejects_invalid_layout() -> None:
    with pytest.raises(ValueError, match="length 6"):
        factorized_flat_state(
            np.zeros(5),
            np.zeros(6),
            nq=3,
            nv=2,
            robot_indices=RobotStateIndices((0,), (0,)),
        )


def test_indices_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        RobotStateIndices((1, 1), (0,))
