"""Small adapters around NVIDIA's public RoboCasa evaluator."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

IMAGE_KEY_SUFFIX = "_image"


def environment_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    """Apply the released evaluator's fixed-base action conversion."""

    action = np.asarray(action, dtype=np.float64)
    if action.shape[-1] == 7 and action_dim == 12:
        return np.concatenate((action, np.asarray([0.0, 0.0, 0.0, 0.0, -1.0])))
    if action.shape[-1] != action_dim:
        raise ValueError(f"cannot map {action.shape[-1]} policy dimensions to {action_dim}")
    return action


def physical_observation_vector(observation: Mapping[str, object]) -> tuple[np.ndarray, list[dict]]:
    """Flatten every finite non-image simulator observation in stable key order."""

    values = []
    schema = []
    offset = 0
    for key in sorted(observation):
        if key.endswith(IMAGE_KEY_SUFFIX):
            continue
        array = np.asarray(observation[key])
        if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
            continue
        flattened = array.astype(np.float64).reshape(-1)
        if flattened.size == 0:
            continue
        values.append(flattened)
        schema.append({"key": key, "start": offset, "stop": offset + flattened.size})
        offset += flattened.size
    if not values:
        raise ValueError("RoboCasa observation contains no finite non-image features")
    return np.concatenate(values), schema


def physical_state_vector(
    observation: Mapping[str, object], generalized_positions: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    """Add simulator qpos so articulated fixtures remain in the endpoint metric."""

    observation_vector, schema = physical_observation_vector(observation)
    qpos = np.asarray(generalized_positions, dtype=np.float64).reshape(-1)
    if qpos.size == 0 or not np.all(np.isfinite(qpos)):
        raise ValueError("RoboCasa generalized positions must be finite and non-empty")
    start = observation_vector.size
    return np.concatenate((observation_vector, qpos)), [
        *schema,
        {"key": "sim.data.qpos", "start": start, "stop": start + qpos.size},
    ]
