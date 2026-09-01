"""Natural-pair construction for separating robot and task-world future content."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from imagined_future.study_design import _average_ranks, pairwise_l2


@dataclass(frozen=True)
class FactorizationThresholds:
    minimum_action_l2: float = 0.01
    robot_position_match_m: float = 0.003
    robot_orientation_match_rad: float = 0.03
    robot_gripper_match_l2: float = 0.005
    minimum_object_goal_l2: float = 0.003
    maximum_object_match_l2: float = 0.0005
    minimum_robot_position_l2_m: float = 0.003
    minimum_robot_orientation_angle_rad: float = 0.03
    minimum_robot_gripper_l2: float = 0.003


def pairwise_quaternion_angle(quaternions: np.ndarray) -> np.ndarray:
    """Return sign-invariant pairwise quaternion geodesic angles."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("quaternions must have shape [branches, 4]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("quaternions must be nonzero")
    unit = values / norms
    absolute_cosine = np.clip(np.abs(unit @ unit.T), 0.0, 1.0)
    return 2.0 * np.arccos(absolute_cosine)


def endpoint_distance_matrices(
    normalized_actions: np.ndarray,
    robot_proprios: np.ndarray,
    object_goal_features: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute registered action, robot-component, and task-world distances."""

    robot = np.asarray(robot_proprios, dtype=np.float64)
    if robot.ndim != 2 or robot.shape[1] != 9:
        raise ValueError("LIBERO robot proprioception must have shape [branches, 9]")
    if len(normalized_actions) != len(robot) or len(object_goal_features) != len(robot):
        raise ValueError(
            "action, robot, and object arrays must contain the same branches"
        )
    return {
        "action_l2": pairwise_l2(normalized_actions),
        "robot_gripper_l2": pairwise_l2(robot[:, :2]),
        "robot_position_l2_m": pairwise_l2(robot[:, 2:5]),
        "robot_orientation_angle_rad": pairwise_quaternion_angle(robot[:, 5:9]),
        "object_goal_l2": pairwise_l2(object_goal_features),
    }


def _robot_signal(
    matrices: dict[str, np.ndarray], thresholds: FactorizationThresholds
) -> np.ndarray:
    return np.maximum.reduce(
        [
            matrices["robot_position_l2_m"] / thresholds.minimum_robot_position_l2_m,
            matrices["robot_orientation_angle_rad"]
            / thresholds.minimum_robot_orientation_angle_rad,
            matrices["robot_gripper_l2"] / thresholds.minimum_robot_gripper_l2,
        ]
    )


def pair_class_masks(
    matrices: dict[str, np.ndarray], thresholds: FactorizationThresholds
) -> dict[str, np.ndarray]:
    """Return symmetric eligibility masks for object-, robot-, and joint-divergent pairs."""

    action_ok = matrices["action_l2"] >= thresholds.minimum_action_l2
    robot_matched = (
        (matrices["robot_position_l2_m"] <= thresholds.robot_position_match_m)
        & (
            matrices["robot_orientation_angle_rad"]
            <= thresholds.robot_orientation_match_rad
        )
        & (matrices["robot_gripper_l2"] <= thresholds.robot_gripper_match_l2)
    )
    robot_divergent = (
        (matrices["robot_position_l2_m"] >= thresholds.minimum_robot_position_l2_m)
        | (
            matrices["robot_orientation_angle_rad"]
            >= thresholds.minimum_robot_orientation_angle_rad
        )
        | (matrices["robot_gripper_l2"] >= thresholds.minimum_robot_gripper_l2)
    )
    object_divergent = matrices["object_goal_l2"] >= thresholds.minimum_object_goal_l2
    object_matched = matrices["object_goal_l2"] <= thresholds.maximum_object_match_l2
    not_self = ~np.eye(action_ok.shape[0], dtype=np.bool_)
    return {
        "object": action_ok & robot_matched & object_divergent & not_self,
        "robot": action_ok & object_matched & robot_divergent & not_self,
        "joint": action_ok & object_divergent & robot_divergent & not_self,
    }


def _ranked_choice(
    candidates: list[int],
    values: list[np.ndarray],
    *,
    recipient: int,
    maximize: list[bool],
) -> int:
    if not candidates:
        raise ValueError("cannot rank an empty candidate set")
    score = np.zeros(len(candidates), dtype=np.float64)
    for matrix, high_is_good in zip(values, maximize, strict=True):
        column = np.asarray(
            [matrix[recipient, donor] for donor in candidates], dtype=np.float64
        )
        score += _average_ranks(column if high_is_good else -column)
    return max(
        range(len(candidates)),
        key=lambda index: (score[index], -candidates[index]),
    )


def select_factorized_donors(
    normalized_actions: np.ndarray,
    robot_proprios: np.ndarray,
    object_goal_features: np.ndarray,
    thresholds: FactorizationThresholds | None = None,
) -> dict:
    """Select a common recipient and natural object/robot/joint donors.

    Selection depends only on natural branch actions and simulator endpoints.
    It raises when no recipient has both registered pair classes.
    """

    if thresholds is None:
        thresholds = FactorizationThresholds()
    matrices = endpoint_distance_matrices(
        normalized_actions, robot_proprios, object_goal_features
    )
    masks = pair_class_masks(matrices, thresholds)
    robot_signal = _robot_signal(matrices, thresholds)
    triples: list[tuple[int, int, int]] = []
    for recipient in range(len(robot_proprios)):
        object_donors = np.flatnonzero(masks["object"][recipient]).tolist()
        robot_donors = np.flatnonzero(masks["robot"][recipient]).tolist()
        triples.extend(
            (recipient, object_donor, robot_donor)
            for object_donor in object_donors
            for robot_donor in robot_donors
            if object_donor != robot_donor
        )
    if not triples:
        raise ValueError("no common recipient has both registered natural pair classes")

    criteria: list[tuple[np.ndarray, bool]] = [
        (matrices["action_l2"], True),
        (matrices["object_goal_l2"], True),
        (matrices["robot_position_l2_m"], False),
        (matrices["robot_orientation_angle_rad"], False),
        (matrices["robot_gripper_l2"], False),
        (matrices["action_l2"], True),
        (robot_signal, True),
        (matrices["object_goal_l2"], False),
    ]
    triple_values = []
    for recipient, object_donor, robot_donor in triples:
        triple_values.append(
            [
                matrix[recipient, object_donor]
                if index < 5
                else matrix[recipient, robot_donor]
                for index, (matrix, _maximize) in enumerate(criteria)
            ]
        )
    values = np.asarray(triple_values, dtype=np.float64)
    scores = np.zeros(len(triples), dtype=np.float64)
    for column_index, (_matrix, maximize) in enumerate(criteria):
        column = values[:, column_index]
        scores += _average_ranks(column if maximize else -column)
    selected_index = max(
        range(len(triples)),
        key=lambda index: (scores[index], tuple(-value for value in triples[index])),
    )
    recipient, object_donor, robot_donor = triples[selected_index]

    excluded = {recipient, object_donor, robot_donor}
    joint_candidates = [
        donor
        for donor in np.flatnonzero(masks["joint"][recipient]).tolist()
        if donor not in excluded
    ]
    joint_donor = None
    if joint_candidates:
        choice = _ranked_choice(
            joint_candidates,
            [matrices["action_l2"], matrices["object_goal_l2"], robot_signal],
            recipient=recipient,
            maximize=[True, True, True],
        )
        joint_donor = joint_candidates[choice]
        excluded.add(joint_donor)

    remaining = [index for index in range(len(robot_proprios)) if index not in excluded]
    natural_control = None
    if remaining:
        action_target = matrices["action_l2"][recipient, object_donor]
        object_target = matrices["object_goal_l2"][recipient, object_donor]
        robot_target = robot_signal[recipient, object_donor]
        scales = [
            max(action_target, 1e-12),
            max(object_target, 1e-12),
            max(robot_target, 1e-12),
        ]
        natural_control = min(
            remaining,
            key=lambda donor: (
                abs(matrices["action_l2"][recipient, donor] - action_target) / scales[0]
                + abs(matrices["object_goal_l2"][recipient, donor] - object_target)
                / scales[1]
                + abs(robot_signal[recipient, donor] - robot_target) / scales[2],
                donor,
            ),
        )

    def pair_metrics(donor: int | None) -> dict[str, float] | None:
        if donor is None:
            return None
        return {
            name: float(matrix[recipient, donor]) for name, matrix in matrices.items()
        }

    return {
        "recipient": recipient,
        "object_donor": object_donor,
        "robot_donor": robot_donor,
        "joint_donor": joint_donor,
        "natural_control": natural_control,
        "selection_score": float(scores[selected_index]),
        "eligible_common_recipient_triples": len(triples),
        "thresholds": asdict(thresholds),
        "pair_metrics": {
            "object": pair_metrics(object_donor),
            "robot": pair_metrics(robot_donor),
            "joint": pair_metrics(joint_donor),
            "natural_control": pair_metrics(natural_control),
        },
    }
