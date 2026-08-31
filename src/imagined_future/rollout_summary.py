"""Pure helpers for summarizing natural LIBERO rollout screens."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def predicate_name(predicate: Mapping[str, Any]) -> str:
    """Return a stable human-readable label for a serialized goal predicate."""

    arguments = ",".join(str(argument["name"]) for argument in predicate.get("arguments", ()))
    return f"{predicate['predicate']}({arguments})"


def predicate_values(snapshot: Mapping[str, Any]) -> dict[str, bool]:
    """Map serialized goal-predicate labels to their Boolean values."""

    return {predicate_name(item): bool(item["value"]) for item in snapshot["predicates"]}


def compress_predicate_trajectory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only query points where current or chunk-end goal values change."""

    transitions: list[dict[str, Any]] = []
    previous: tuple[tuple[tuple[str, bool], ...], tuple[tuple[str, bool], ...]] | None = None
    for row in rows:
        current = predicate_values(row["current"])
        endpoint = predicate_values(row["endpoint"])
        signature = (tuple(current.items()), tuple(endpoint.items()))
        if signature == previous:
            continue
        transitions.append(
            {
                "query_index": int(row["query_index"]),
                "query_step": int(row["query_step"]),
                "current": current,
                "endpoint": endpoint,
            }
        )
        previous = signature
    return transitions


def first_true_steps(rows: Iterable[Mapping[str, Any]]) -> dict[str, int | None]:
    """Return the first policy step at which each predicate is true at chunk end."""

    result: dict[str, int | None] = {}
    for row in rows:
        for name, value in predicate_values(row["endpoint"]).items():
            result.setdefault(name, None)
            if value and result[name] is None:
                result[name] = int(row["query_step"]) + int(row["executed_count"])
    return result
