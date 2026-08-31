"""Deterministic replay utilities for same-state counterfactual branches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np


class ReplayEnvironment(Protocol):
    def reset(self) -> Any: ...

    def set_init_state(self, state: np.ndarray) -> Any: ...

    def step(self, action: Sequence[float]) -> tuple[Any, float, bool, dict[str, Any]]: ...

    def get_sim_state(self) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class BranchPoint:
    """Replayable branch defined by an initial state and exact action prefix."""

    initial_state: np.ndarray
    warmup_actions: tuple[tuple[float, ...], ...]
    prefix_actions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ReplayResult:
    state: np.ndarray
    observation: Any
    state_digest: str


def state_digest(state: np.ndarray) -> str:
    """Hash dtype, shape, and bytes so restore checks cannot silently coerce."""

    array = np.ascontiguousarray(state)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def replay_branch_point(env_factory: Callable[[], ReplayEnvironment], point: BranchPoint) -> ReplayResult:
    """Construct a fresh environment and deterministically replay to a branch.

    Fresh replay is preferred to setting a mid-episode MuJoCo state because the
    flattened simulator state omits controller, observable, and RNG internals.
    """

    env = env_factory()
    try:
        env.reset()
        observation = env.set_init_state(point.initial_state.copy())
        for action in (*point.warmup_actions, *point.prefix_actions):
            observation, _reward, _done, _info = env.step(action)
        state = np.asarray(env.get_sim_state()).copy()
        return ReplayResult(state=state, observation=observation, state_digest=state_digest(state))
    finally:
        env.close()


def validate_replay_stability(
    env_factory: Callable[[], ReplayEnvironment],
    point: BranchPoint,
    *,
    repeats: int = 3,
    atol: float = 0.0,
) -> tuple[ReplayResult, ...]:
    """Replay repeatedly and reject branch points with hidden-state drift."""

    if repeats < 2:
        raise ValueError("stability validation requires at least two replays")
    results = tuple(replay_branch_point(env_factory, point) for _ in range(repeats))
    reference = results[0].state
    for index, result in enumerate(results[1:], start=1):
        if not np.allclose(reference, result.state, rtol=0.0, atol=atol, equal_nan=False):
            maximum = float(np.max(np.abs(reference - result.state)))
            raise RuntimeError(f"branch replay {index} diverged from reference; max_abs={maximum:.3e}")
    return results


def tuple_actions(actions: Iterable[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Freeze mutable action arrays before storing a branch definition."""

    return tuple(tuple(float(value) for value in action) for action in actions)
