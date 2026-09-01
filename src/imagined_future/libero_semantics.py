"""Simulator-grounded semantic measurements for LIBERO goal predicates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _numbers(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _joint_snapshot(environment: Any, object_state: Any) -> dict[str, Any] | None:
    """Return joint positions and declared articulation thresholds when available."""

    simulator_environment = environment.env
    if getattr(object_state, "object_state_type", None) == "site":
        site = simulator_environment.object_sites_dict[object_state.object_name]
        joint_names = list(site.joints or ())
        articulated_object = simulator_environment.get_object(object_state.parent_name)
    elif getattr(object_state, "object_state_type", None) == "object":
        articulated_object = simulator_environment.get_object(object_state.object_name)
        joint_names = list(getattr(articulated_object, "joints", ()) or ())
    else:
        return None

    if not joint_names:
        return None
    qpos = []
    for name in joint_names:
        address = simulator_environment.sim.model.get_joint_qpos_addr(name)
        if isinstance(address, tuple):
            value = simulator_environment.sim.data.qpos[slice(*address)]
        else:
            value = simulator_environment.sim.data.qpos[address]
        qpos.append(_numbers(value))
    articulation = getattr(articulated_object, "object_properties", {}).get("articulation", {})
    return {
        "names": joint_names,
        "qpos": qpos,
        "default_open_ranges": _numbers(articulation.get("default_open_ranges", ())),
        "default_close_ranges": _numbers(articulation.get("default_close_ranges", ())),
    }


def _argument_snapshot(environment: Any, name: str) -> dict[str, Any]:
    object_state = environment.env.object_states_dict[name]
    record: dict[str, Any] = {
        "name": name,
        "state_type": getattr(object_state, "object_state_type", None),
    }
    try:
        geom = object_state.get_geom_state()
    except (AttributeError, NotImplementedError):
        geom = None
    if geom is not None:
        record["position"] = _numbers(geom["pos"])
        record["quaternion"] = _numbers(geom["quat"])
    joints = _joint_snapshot(environment, object_state)
    if joints is not None:
        record["joints"] = joints
    return record


def goal_predicate_snapshot(environment: Any) -> dict[str, Any]:
    """Evaluate each LIBERO goal predicate and record its physical state.

    ``environment`` is a LIBERO ``ControlEnv`` wrapper. The function deliberately
    uses LIBERO's own parsed goal and predicate evaluator so labels exactly match
    the benchmark's success definition.
    """

    simulator_environment = environment.env
    predicates = []
    for state in simulator_environment.parsed_problem["goal_state"]:
        state = list(state)
        name, *arguments = state
        predicates.append(
            {
                "predicate": name,
                "arguments": [_argument_snapshot(environment, argument) for argument in arguments],
                "value": bool(simulator_environment._eval_predicate(state)),
            }
        )
    return {
        "success": bool(all(predicate["value"] for predicate in predicates)),
        "predicates": predicates,
    }


def goal_feature_vector(snapshot: dict[str, Any]) -> np.ndarray:
    """Flatten task-relevant physical quantities from a predicate snapshot.

    Positions and articulated-joint coordinates cover the common LIBERO
    spatial, containment, and articulation predicates. Quaternion values are
    added only when the predicate explicitly describes orientation. Traversal
    follows parsed goal and argument order, so vectors are stable across
    branches of the same task.
    """

    values: list[float] = []
    orientation_terms = ("upright", "orient", "align", "horizontal", "vertical")
    for predicate in snapshot.get("predicates", ()):
        predicate_name = str(predicate.get("predicate", "")).lower()
        include_quaternion = any(term in predicate_name for term in orientation_terms)
        for argument in predicate.get("arguments", ()):
            values.extend(float(value) for value in argument.get("position", ()))
            if include_quaternion:
                values.extend(float(value) for value in argument.get("quaternion", ()))
            joints = argument.get("joints")
            if joints is not None:
                for joint in joints.get("qpos", ()):
                    # Movable objects expose a seven-dimensional free joint
                    # that duplicates position and quaternion. Articulated
                    # hinge/slide joints are low-dimensional and retained.
                    if len(joint) <= 3:
                        values.extend(float(value) for value in joint)
    return np.asarray(values, dtype=np.float64)


def physical_endpoint_feature_vector(snapshot: dict[str, Any], proprio: Any) -> np.ndarray:
    """Combine goal-relevant world coordinates with official robot proprioception."""

    return np.concatenate(
        [goal_feature_vector(snapshot), np.asarray(proprio, dtype=np.float64).reshape(-1)]
    )
