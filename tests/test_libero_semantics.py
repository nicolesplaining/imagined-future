from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from imagined_future.libero_semantics import goal_feature_vector, goal_predicate_snapshot


class FakeState:
    object_state_type = "site"
    object_name = "drawer_region"
    parent_name = "cabinet"

    def get_geom_state(self):
        return {"pos": np.array([1.0, 2.0, 3.0]), "quat": np.array([1.0, 0.0, 0.0, 0.0])}


def test_goal_predicate_snapshot_uses_libero_goal_and_records_joint_state() -> None:
    articulated = SimpleNamespace(
        object_properties={
            "articulation": {
                "default_open_ranges": [-0.16, -0.14],
                "default_close_ranges": [0.0, 0.005],
            }
        }
    )
    simulator_environment = SimpleNamespace(
        parsed_problem={"goal_state": [["Close", "drawer_region"]]},
        object_states_dict={"drawer_region": FakeState()},
        object_sites_dict={"drawer_region": SimpleNamespace(joints=["drawer_joint"])},
        sim=SimpleNamespace(
            model=SimpleNamespace(get_joint_qpos_addr=lambda _name: 2),
            data=SimpleNamespace(qpos=np.array([9.0, 9.0, 0.001])),
        ),
        get_object=lambda _name: articulated,
        _eval_predicate=lambda state: state[0] == "Close",
    )
    snapshot = goal_predicate_snapshot(SimpleNamespace(env=simulator_environment))

    assert snapshot["success"] is True
    predicate = snapshot["predicates"][0]
    assert predicate["value"] is True
    argument = predicate["arguments"][0]
    assert argument["position"] == [1.0, 2.0, 3.0]
    assert argument["joints"]["qpos"] == [[0.001]]
    assert argument["joints"]["default_close_ranges"] == [0.0, 0.005]


def test_goal_predicate_snapshot_handles_free_joint_address_ranges() -> None:
    state = FakeState()
    state.object_state_type = "object"
    state.object_name = "bowl"
    simulator_environment = SimpleNamespace(
        parsed_problem={"goal_state": [["Held", "bowl"]]},
        object_states_dict={"bowl": state},
        sim=SimpleNamespace(
            model=SimpleNamespace(get_joint_qpos_addr=lambda _name: (1, 4)),
            data=SimpleNamespace(qpos=np.array([9.0, 1.0, 2.0, 3.0, 9.0])),
        ),
        get_object=lambda _name: SimpleNamespace(joints=["free_joint"], object_properties={}),
        _eval_predicate=lambda _state: False,
    )
    snapshot = goal_predicate_snapshot(SimpleNamespace(env=simulator_environment))

    assert snapshot["success"] is False
    assert snapshot["predicates"][0]["arguments"][0]["joints"]["qpos"] == [[1.0, 2.0, 3.0]]


def test_goal_predicate_snapshot_handles_objects_without_joints() -> None:
    state = FakeState()
    state.object_state_type = "object"
    state.object_name = "fixed_object"
    simulator_environment = SimpleNamespace(
        parsed_problem={"goal_state": [["Visible", "fixed_object"]]},
        object_states_dict={"fixed_object": state},
        get_object=lambda _name: SimpleNamespace(joints=None, object_properties={}),
        _eval_predicate=lambda _state: True,
    )
    snapshot = goal_predicate_snapshot(SimpleNamespace(env=simulator_environment))

    assert snapshot["success"] is True
    assert "joints" not in snapshot["predicates"][0]["arguments"][0]


def test_goal_feature_vector_uses_positions_and_joint_coordinates() -> None:
    snapshot = {
        "predicates": [
            {
                "predicate": "Open",
                "arguments": [
                    {
                        "position": [1, 2, 3],
                        "quaternion": [0, 0, 0, 1],
                        "joints": {"qpos": [[0.4], [0.5]]},
                    }
                ],
            }
        ]
    }
    np.testing.assert_allclose(goal_feature_vector(snapshot), [1, 2, 3, 0.4, 0.5])
