import numpy as np
import pytest

from imagined_future.content_factorization import (
    FactorizationThresholds,
    endpoint_distance_matrices,
    pair_class_masks,
    pairwise_quaternion_angle,
    select_factorized_donors,
)


def _proprio(position, *, angle=0.0, gripper=0.0):
    quaternion = [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]
    return [gripper, gripper, *position, *quaternion]


def test_quaternion_angle_is_sign_invariant():
    values = np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])
    assert pairwise_quaternion_angle(values)[0, 1] == pytest.approx(0.0)


def test_pair_class_masks_separate_robot_and_object_changes():
    actions = np.asarray([np.zeros((2, 1)), [[0.02], [0.0]], [[0.0], [0.02]]])
    robot = np.asarray(
        [
            _proprio([0.0, 0.0, 0.0]),
            _proprio([0.001, 0.0, 0.0]),
            _proprio([0.010, 0.0, 0.0]),
        ]
    )
    objects = np.asarray([[0.0], [0.010], [0.0001]])
    matrices = endpoint_distance_matrices(actions, robot, objects)
    masks = pair_class_masks(matrices, FactorizationThresholds())
    assert masks["object"][0, 1]
    assert masks["robot"][0, 2]
    assert not masks["object"][0, 2]
    assert not masks["robot"][0, 1]


def test_select_factorized_donors_uses_common_recipient():
    actions = np.zeros((5, 2, 1))
    actions[:, 0, 0] = [0.0, 0.02, -0.02, 0.04, -0.04]
    robot = np.asarray(
        [
            _proprio([0.0, 0.0, 0.0]),
            _proprio([0.001, 0.0, 0.0]),
            _proprio([0.010, 0.0, 0.0]),
            _proprio([0.012, 0.0, 0.0]),
            _proprio([0.0005, 0.0, 0.0]),
        ]
    )
    objects = np.asarray([[0.0], [0.010], [0.0001], [0.012], [0.009]])
    result = select_factorized_donors(actions, robot, objects)
    matrices = endpoint_distance_matrices(actions, robot, objects)
    masks = pair_class_masks(matrices, FactorizationThresholds())
    recipient = result["recipient"]
    assert masks["object"][recipient, result["object_donor"]]
    assert masks["robot"][recipient, result["robot_donor"]]
    assert masks["joint"][recipient, result["joint_donor"]]
    assert (
        len(
            {
                recipient,
                result["object_donor"],
                result["robot_donor"],
                result["joint_donor"],
            }
        )
        == 4
    )


def test_select_factorized_donors_rejects_missing_object_pair():
    actions = np.zeros((3, 2, 1))
    actions[:, 0, 0] = [0.0, 0.02, -0.02]
    robot = np.asarray(
        [
            _proprio([0.0, 0.0, 0.0]),
            _proprio([0.01, 0.0, 0.0]),
            _proprio([0.02, 0.0, 0.0]),
        ]
    )
    objects = np.zeros((3, 1))
    with pytest.raises(ValueError, match="no common recipient"):
        select_factorized_donors(actions, robot, objects)
