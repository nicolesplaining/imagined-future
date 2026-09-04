"""Dependency-light validation and state-clustered summaries for overnight studies."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class SummaryValidationError(ValueError):
    """A completed per-state summary violates the frozen analysis contract."""


def discover_summaries(inputs: Sequence[Path]) -> list[Path]:
    """Find completed atomic summaries while naturally ignoring unfinished directories."""

    found: set[Path] = set()
    missing_inputs: list[str] = []
    for supplied in inputs:
        path = supplied.expanduser()
        if not path.exists():
            missing_inputs.append(str(path))
            continue
        if path.is_file():
            found.add(path.resolve())
            continue
        direct = path / "summary.json"
        if direct.is_file():
            found.add(direct.resolve())
        found.update(candidate.resolve() for candidate in path.rglob("summary.json"))
    if not found:
        detail = f"; missing inputs: {missing_inputs}" if missing_inputs else ""
        raise FileNotFoundError(f"no completed summary.json files found{detail}")
    return sorted(found)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SummaryValidationError(f"{path}: root must be a JSON object")
    return value


def required(mapping: Mapping[str, Any], path: str, *, source: str) -> Any:
    value: Any = mapping
    traversed: list[str] = []
    for component in path.split("."):
        traversed.append(component)
        if not isinstance(value, Mapping) or component not in value:
            raise SummaryValidationError(
                f"{source}: missing mandatory field {'.'.join(traversed)}"
            )
        value = value[component]
    return value


def finite_or_none(value: Any, *, field: str, source: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SummaryValidationError(f"{source}: {field} must be numeric or null") from error
    if not math.isfinite(result):
        return None
    return result


def required_finite_or_none(
    mapping: Mapping[str, Any], path: str, *, source: str
) -> float | None:
    return finite_or_none(required(mapping, path, source=source), field=path, source=source)


def required_bool(mapping: Mapping[str, Any], path: str, *, source: str) -> bool:
    value = required(mapping, path, source=source)
    if not isinstance(value, bool):
        raise SummaryValidationError(f"{source}: {path} must be Boolean")
    return value


def state_id(report: Mapping[str, Any], *, source: str) -> str:
    task = str(required(report, "task", source=source))
    environment_seed = int(required(report, "environment_seed", source=source))
    return f"{task}-seed-{environment_seed}"


def collapse_to_states(
    rows: Sequence[Mapping[str, Any]], value_key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    state_tasks: dict[str, str] = {}
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        identifier = str(row["state_id"])
        task = str(row["task"])
        if identifier in state_tasks and state_tasks[identifier] != task:
            raise SummaryValidationError(f"state {identifier} appears under multiple tasks")
        state_tasks[identifier] = task
        grouped[identifier].append(numeric)
    return [
        {
            "state_id": identifier,
            "task": state_tasks[identifier],
            "value": float(np.mean(values)),
            "within_state_observations": len(values),
        }
        for identifier, values in sorted(grouped.items())
    ]


def hierarchical_task_state_bootstrap(
    state_rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int = 10_000,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Equal-task hierarchical bootstrap, resampling states inside sampled tasks."""

    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in state_rows:
        value = float(row["value"])
        if not math.isfinite(value):
            raise SummaryValidationError("bootstrap state values must be finite")
        by_task[str(row["task"])].append(value)
    if not by_task:
        return {
            "tasks": 0,
            "states": 0,
            "mean": None,
            "state_weighted_mean": None,
            "ci95": None,
            "resamples": resamples,
            "seed": seed,
        }

    tasks = sorted(by_task)
    arrays = [np.asarray(by_task[task], dtype=np.float64) for task in tasks]
    task_means = np.asarray([values.mean() for values in arrays], dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled_tasks = generator.integers(0, len(tasks), size=(resamples, len(tasks)))
    estimates = np.zeros(resamples, dtype=np.float64)
    for slot in range(len(tasks)):
        selected = sampled_tasks[:, slot]
        for task_index, values in enumerate(arrays):
            mask = selected == task_index
            count = int(mask.sum())
            if count == 0:
                continue
            state_indices = generator.integers(
                0, len(values), size=(count, len(values))
            )
            estimates[mask] += values[state_indices].mean(axis=1)
    estimates /= len(tasks)
    return {
        "tasks": len(tasks),
        "states": len(state_rows),
        "mean": float(task_means.mean()),
        "state_weighted_mean": float(
            np.mean([float(row["value"]) for row in state_rows])
        ),
        "ci95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "resamples": resamples,
        "seed": seed,
    }


def summarize_metric(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    state_rows = collapse_to_states(rows, value_key)
    result = hierarchical_task_state_bootstrap(
        state_rows, resamples=resamples, seed=seed
    )
    values = [float(row[value_key]) for row in rows if row.get(value_key) is not None]
    result.update(
        {
            "donor_observations": len(values),
            "median_state_mean": (
                float(np.median([row["value"] for row in state_rows]))
                if state_rows
                else None
            ),
        }
    )
    return result


def exact_binomial_greater(successes: int, trials: int, chance: float) -> float | None:
    if trials == 0:
        return None
    if not 0.0 < chance < 1.0:
        raise ValueError("chance must be in (0,1)")
    return float(
        min(
            1.0,
            sum(
                math.comb(trials, count)
                * chance**count
                * (1.0 - chance) ** (trials - count)
                for count in range(successes, trials + 1)
            ),
        )
    )


def leave_one_task_out(
    rows: Sequence[Mapping[str, Any]], value_key: str
) -> dict[str, float | None]:
    tasks = sorted({str(row["task"]) for row in rows})
    output: dict[str, float | None] = {}
    for held_out in tasks:
        state_rows = collapse_to_states(
            [row for row in rows if str(row["task"]) != held_out], value_key
        )
        if not state_rows:
            output[held_out] = None
            continue
        by_task: dict[str, list[float]] = defaultdict(list)
        for row in state_rows:
            by_task[str(row["task"])].append(float(row["value"]))
        output[held_out] = float(
            np.mean([np.mean(values) for values in by_task.values()])
        )
    return output


def separation_quartiles(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[float], dict[tuple[str, int, int], int]]:
    """Assign quartiles once per directed state pair, independent of future source."""

    unique: dict[tuple[str, int, int], float] = {}
    for row in rows:
        separation = row.get("native_target_l2")
        if separation is None:
            continue
        key = (
            str(row["state_id"]),
            int(row["recipient_seed"]),
            int(row["target_donor_seed"]),
        )
        numeric = float(separation)
        previous = unique.get(key)
        if previous is not None and not math.isclose(previous, numeric, rel_tol=0.0, abs_tol=1e-12):
            raise SummaryValidationError(
                f"native separation differs across sources for {key}: {previous} versus {numeric}"
            )
        unique[key] = numeric
    if not unique:
        return [], {}
    values = np.asarray(list(unique.values()), dtype=np.float64)
    boundaries = [float(value) for value in np.quantile(values, [0.25, 0.5, 0.75])]
    assignments = {
        key: int(np.searchsorted(boundaries, separation, side="right")) + 1
        for key, separation in unique.items()
    }
    return boundaries, assignments


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def format_estimate(estimate: Mapping[str, Any], *, percent: bool = False) -> str:
    mean = estimate.get("mean")
    interval = estimate.get("ci95")
    if mean is None or interval is None:
        return "--"
    scale = 100.0 if percent else 1.0
    return f"{scale * float(mean):.2f} [{scale * float(interval[0]):.2f}, {scale * float(interval[1]):.2f}]"


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    return "".join(replacements.get(character, character) for character in value)


def save_figure(figure: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
