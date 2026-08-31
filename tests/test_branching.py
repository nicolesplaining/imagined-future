from __future__ import annotations

import numpy as np
import pytest

from imagined_future.branching import BranchPoint, state_digest, tuple_actions, validate_replay_stability


class FakeEnv:
    def __init__(self, drift: float = 0.0) -> None:
        self.state = np.zeros(2, dtype=np.float64)
        self.drift = drift
        self.closed = False

    def reset(self):
        self.state[:] = 0

    def set_init_state(self, state):
        self.state = state.copy()
        return self.state.copy()

    def step(self, action):
        self.state += np.asarray(action) + self.drift
        return self.state.copy(), 0.0, False, {}

    def get_sim_state(self):
        return self.state.copy()

    def close(self):
        self.closed = True


def test_stable_fresh_replay_reaches_identical_branch_state() -> None:
    point = BranchPoint(
        initial_state=np.array([1.0, 2.0]),
        warmup_actions=tuple_actions([[0.0, 0.0]]),
        prefix_actions=tuple_actions([[1.0, -1.0], [0.5, 0.5]]),
    )

    results = validate_replay_stability(FakeEnv, point, repeats=3)

    assert len({result.state_digest for result in results}) == 1
    assert np.array_equal(results[0].state, np.array([2.5, 1.5]))


def test_replay_validation_rejects_hidden_drift() -> None:
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return FakeEnv(drift=float(calls) * 1e-3)

    point = BranchPoint(np.zeros(2), (), tuple_actions([[1.0, 1.0]]))
    with pytest.raises(RuntimeError, match="diverged"):
        validate_replay_stability(factory, point, repeats=2)


def test_state_digest_includes_dtype() -> None:
    assert state_digest(np.array([1], dtype=np.float32)) != state_digest(np.array([1], dtype=np.float64))
