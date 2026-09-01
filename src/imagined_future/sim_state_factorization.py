"""Factor robot and non-robot coordinates in flattened MuJoCo states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotStateIndices:
    """Robot generalized-coordinate indices in MuJoCo qpos and qvel arrays."""

    qpos: tuple[int, ...]
    qvel: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(set(self.qpos)) != len(self.qpos):
            raise ValueError("robot qpos indices must be unique")
        if len(set(self.qvel)) != len(self.qvel):
            raise ValueError("robot qvel indices must be unique")
        if any(index < 0 for index in (*self.qpos, *self.qvel)):
            raise ValueError("robot state indices must be non-negative")


def robot_state_indices(robot: object) -> RobotStateIndices:
    """Resolve robosuite arm and gripper joints through the MuJoCo model."""

    def expand(address: int | tuple[int, int]) -> list[int]:
        if isinstance(address, tuple):
            return list(range(int(address[0]), int(address[1])))
        return [int(address)]

    joint_names = list(robot.robot_joints)
    if robot.gripper_joints is not None:
        joint_names.extend(robot.gripper_joints)
    qpos = []
    qvel = []
    for name in joint_names:
        qpos.extend(expand(robot.sim.model.get_joint_qpos_addr(name)))
        qvel.extend(expand(robot.sim.model.get_joint_qvel_addr(name)))
    return RobotStateIndices(tuple(qpos), tuple(qvel))


def factorized_flat_state(
    object_source: np.ndarray,
    robot_source: np.ndarray,
    *,
    nq: int,
    nv: int,
    robot_indices: RobotStateIndices,
) -> np.ndarray:
    """Combine non-robot state from one endpoint with robot state from another.

    robosuite's public ``MjSimState.flatten`` layout is time, qpos, then qvel.
    The returned time follows the object source; only registered robot arm and
    gripper generalized coordinates are transplanted.
    """

    object_array = np.asarray(object_source, dtype=np.float64).reshape(-1)
    robot_array = np.asarray(robot_source, dtype=np.float64).reshape(-1)
    expected = 1 + nq + nv
    if object_array.size != expected or robot_array.size != expected:
        raise ValueError(
            f"flattened states must have length {expected}; got "
            f"{object_array.size} and {robot_array.size}"
        )
    if any(index >= nq for index in robot_indices.qpos):
        raise ValueError("robot qpos index exceeds MuJoCo nq")
    if any(index >= nv for index in robot_indices.qvel):
        raise ValueError("robot qvel index exceeds MuJoCo nv")

    hybrid = object_array.copy()
    qpos_offset = 1
    qvel_offset = 1 + nq
    qpos = np.asarray(robot_indices.qpos, dtype=np.int64)
    qvel = np.asarray(robot_indices.qvel, dtype=np.int64)
    hybrid[qpos_offset + qpos] = robot_array[qpos_offset + qpos]
    hybrid[qvel_offset + qvel] = robot_array[qvel_offset + qvel]
    return hybrid
