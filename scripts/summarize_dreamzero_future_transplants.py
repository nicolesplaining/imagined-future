#!/usr/bin/env python3
"""Audit and summarize the frozen DreamZero future-latent transplant cohort.

The saved DROID state is the independent unit.  All twelve off-diagonal
recipient-noise x future-source cells are averaged within state.  Exact self
replays are implementation controls and are excluded from the primary
four-source retrieval estimand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


STATE_SCHEMA = "dreamzero-future-transplant-state-v1"
TRACE_INVENTORY_SCHEMA = "dreamzero-future-transplant-trace-inventory-v1"
EXPECTED_BRANCH_COUNT = 4
EXPECTED_GRID_COUNT = 16
EXPECTED_SOLVER_STEPS = 16
EPS = 1e-12
METRICS = (
    "retrieval_accuracy_off_diagonal",
    "distance_reduction",
    "normalized_projection",
    "cosine_alignment",
    "orthogonal_residual",
)
CELL_METRIC = {
    "retrieval_accuracy_off_diagonal": "correct_source_top1",
    "distance_reduction": "distance_reduction",
    "normalized_projection": "normalized_projection",
    "cosine_alignment": "cosine_alignment",
    "orthogonal_residual": "orthogonal_residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Frozen cohort manifest; required for evaluation admission.",
    )
    parser.add_argument(
        "--admission",
        choices=("evaluation", "excluded_debug_smoke"),
        default="evaluation",
    )
    parser.add_argument("--expected-state-count", type=int, default=30)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=50_000)
    parser.add_argument("--permutation-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=260904)
    parser.add_argument(
        "--provenance-receipt",
        type=Path,
        help="Verified runtime receipt to copy into the analysis package.",
    )
    return parser.parse_args()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise FileNotFoundError(sidecar)
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    actual = sha256_file(path)
    if not fields or fields[0] != actual:
        raise ValueError(f"SHA sidecar mismatch for {path}")
    return actual


def atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, canonical_json(json_safe(value)) + b"\n")


def freeze_analysis_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    os.chmod(path, 0o444)
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    atomic_text(sidecar, f"{digest}  {path.name}\n")
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "sidecar_sha256": sha256_file(sidecar),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object at {path}")
    return value


def rng_for(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def bootstrap_ci(
    values: Sequence[float], repetitions: int, rng: np.random.Generator
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return [math.nan, math.nan]
    draws = rng.integers(0, len(array), size=(repetitions, len(array)))
    estimates = array[draws].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def summarize_values(
    values: Sequence[float], repetitions: int, rng: np.random.Generator
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": math.nan, "state_bootstrap_95_ci": [math.nan, math.nan], "n_states": 0}
    return {
        "mean": float(array.mean()),
        "state_bootstrap_95_ci": bootstrap_ci(array, repetitions, rng),
        "n_states": int(len(array)),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: "" if isinstance(value, float) and not math.isfinite(value) else value
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def discover_results(root: Path, admission: str) -> list[Path]:
    paths = sorted((root / "states").glob("*/result.json"))
    selected = []
    for path in paths:
        result = load_json(path)
        if result.get("admission") == admission:
            selected.append(path)
    return selected


def load_and_audit_state(
    result_path: Path, expected_admission: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    result_sha = verify_sidecar(result_path)
    del result_sha
    result = load_json(result_path)
    if result.get("schema") != STATE_SCHEMA or result.get("status") != "complete":
        raise ValueError(f"invalid result schema/status: {result_path}")
    if result.get("admission") != expected_admission:
        raise ValueError(f"admission mismatch: {result_path}")
    expected_scientific = expected_admission == "evaluation"
    if result.get("scientific_admission") is not expected_scientific:
        raise ValueError(f"scientific-admission flag mismatch: {result_path}")
    if result_path.parent.name != result.get("state", {}).get("state_id"):
        raise ValueError(f"result directory/state identity mismatch: {result_path}")
    if int(result.get("call_count", -1)) != 20:
        raise ValueError(f"expected 20 calls: {result_path}")
    if int(result.get("native_record_call_count", -1)) != 4:
        raise ValueError(f"expected four native records: {result_path}")
    if int(result.get("replay_grid_call_count", -1)) != EXPECTED_GRID_COUNT:
        raise ValueError(f"expected complete 4x4 replay grid: {result_path}")
    if result.get("self_replay_all_bit_exact") is not True:
        raise ValueError(f"self replay is not bit exact: {result_path}")
    if float(result.get("self_replay_maximum_error", math.inf)) != 0.0:
        raise ValueError(f"nonzero self replay error: {result_path}")

    seeds = np.asarray(result.get("branch_seeds"), dtype=np.int64)
    if seeds.shape != (EXPECTED_BRANCH_COUNT,) or len(set(seeds.tolist())) != 4:
        raise ValueError(f"invalid branch seeds: {result_path}")
    seed_to_index = {int(seed): index for index, seed in enumerate(seeds)}

    state_dir = result_path.parent
    arrays_path = state_dir / str(result["artifacts"]["actions_npz"]["relative_path"])
    arrays_sha = verify_sidecar(arrays_path)
    if arrays_sha != result["artifacts"]["actions_npz"]["sha256"]:
        raise ValueError(f"actions digest mismatch: {result_path}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        array_seeds = np.asarray(archive["branch_seeds"], dtype=np.int64)
        native = np.asarray(archive["native_actions"])
        replay = np.asarray(archive["replay_actions"])
    if not np.array_equal(array_seeds, seeds):
        raise ValueError(f"array/result seed mismatch: {result_path}")
    if native.ndim != 3 or native.shape[0] != 4:
        raise ValueError(f"invalid native action shape {native.shape}: {result_path}")
    if replay.shape != (4, 4, *native.shape[1:]):
        raise ValueError(f"invalid replay grid shape {replay.shape}: {result_path}")
    if not np.issubdtype(native.dtype, np.floating) or native.dtype != replay.dtype:
        raise ValueError(f"invalid action dtypes: {result_path}")
    if not np.isfinite(native).all() or not np.isfinite(replay).all():
        raise ValueError(f"nonfinite action values: {result_path}")
    if list(native.shape[1:]) != result.get("action_shape"):
        raise ValueError(f"action-shape metadata mismatch: {result_path}")
    if str(native.dtype) != result.get("action_dtype"):
        raise ValueError(f"action-dtype metadata mismatch: {result_path}")
    native_hashes = result["native_action_sha256_by_seed"]
    for seed, index in seed_to_index.items():
        if array_sha256(native[index]) != native_hashes[str(seed)]:
            raise ValueError(f"native action hash mismatch for seed {seed}: {result_path}")
        if not np.array_equal(replay[index, index], native[index]):
            raise ValueError(f"diagonal replay differs for seed {seed}: {result_path}")
    computed_action_distinct = len({array_sha256(native[index]) for index in range(4)})
    if computed_action_distinct != int(result.get("native_action_hash_distinct_count", -1)):
        raise ValueError(f"stored native-action distinct count is wrong: {result_path}")

    inventory_path = state_dir / str(result["artifacts"]["trace_inventory"]["relative_path"])
    inventory_sha = verify_sidecar(inventory_path)
    if inventory_sha != result["artifacts"]["trace_inventory"]["sha256"]:
        raise ValueError(f"trace inventory digest mismatch: {result_path}")
    inventory = load_json(inventory_path)
    if inventory.get("schema") != TRACE_INVENTORY_SCHEMA or inventory.get("trace_count") != 4:
        raise ValueError(f"invalid trace inventory: {result_path}")
    trace_by_seed = {int(row["branch_seed"]): row for row in inventory["traces"]}
    if set(trace_by_seed) != set(seed_to_index):
        raise ValueError(f"trace seed set mismatch: {result_path}")
    computed_video_distinct = len(
        {str(trace_by_seed[seed]["video_trace_sha256"]) for seed in seed_to_index}
    )
    if computed_video_distinct != int(result.get("native_video_trace_hash_distinct_count", -1)):
        raise ValueError(f"stored video-trace distinct count is wrong: {result_path}")

    rows = result.get("grid")
    if not isinstance(rows, list) or len(rows) != EXPECTED_GRID_COUNT:
        raise ValueError(f"invalid result grid: {result_path}")
    grid_by_pair: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        recipient = int(row["recipient_seed"])
        source = int(row["future_source_seed"])
        pair = (recipient, source)
        if pair in grid_by_pair or recipient not in seed_to_index or source not in seed_to_index:
            raise ValueError(f"invalid/duplicate grid pair {pair}: {result_path}")
        grid_by_pair[pair] = row
        recipient_index = seed_to_index[recipient]
        source_index = seed_to_index[source]
        if array_sha256(replay[recipient_index, source_index]) != row["sha256"]:
            raise ValueError(f"replay action hash mismatch for {pair}: {result_path}")
        audit = row.get("server_audit")
        if not isinstance(audit, dict):
            raise ValueError(f"missing server audit for {pair}: {result_path}")
        recipient_noise = str(trace_by_seed[recipient]["action_noise_sha256"])
        source_noise = str(trace_by_seed[source]["action_noise_sha256"])
        source_video = str(trace_by_seed[source]["video_trace_sha256"])
        if audit.get("mode") != "replay" or audit.get("status") != "replayed":
            raise ValueError(f"bad replay mode/status for {pair}: {result_path}")
        if int(audit.get("noise_seed", -1)) != recipient:
            raise ValueError(f"wrong recipient seed for {pair}: {result_path}")
        if audit.get("active_action_noise_sha256") != recipient_noise:
            raise ValueError(f"active action noise is not recipient noise for {pair}")
        if audit.get("recipient_reference_action_noise_sha256") != recipient_noise:
            raise ValueError(f"recipient reference is wrong for {pair}")
        if audit.get("donor_action_noise_sha256") != source_noise:
            raise ValueError(f"source noise provenance is wrong for {pair}")
        if audit.get("video_trace_sha256") != source_video:
            raise ValueError(f"wrong source video trace for {pair}")
        if audit.get("applied_video_steps") != list(range(EXPECTED_SOLVER_STEPS)):
            raise ValueError(f"incomplete video intervention for {pair}")
        if int(audit.get("replay_start", -1)) != 0 or int(audit.get("replay_stop", -1)) != 16:
            raise ValueError(f"wrong replay interval for {pair}")
    expected_pairs = set(itertools.product(seed_to_index, repeat=2))
    if set(grid_by_pair) != expected_pairs:
        raise ValueError(f"incomplete pair set: {result_path}")

    return result, seeds, native.astype(np.float64), replay.astype(np.float64)


def compute_state_metrics(
    result: Mapping[str, Any], seeds: np.ndarray, native: np.ndarray, replay: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    native_flat = native.reshape(4, -1)
    replay_flat = replay.reshape(4, 4, -1)
    distances = np.linalg.norm(
        replay_flat[:, :, None, :] - native_flat[None, None, :, :], axis=-1
    )
    nearest = np.argmin(distances, axis=-1)
    offdiag = ~np.eye(4, dtype=bool)
    target = np.broadcast_to(np.arange(4)[None, :], (4, 4))
    cells: list[dict[str, Any]] = []
    for recipient in range(4):
        for source in range(4):
            if recipient == source:
                continue
            baseline = native_flat[recipient]
            donor = native_flat[source]
            patched = replay_flat[recipient, source]
            axis = donor - baseline
            displacement = patched - baseline
            separation = float(np.linalg.norm(axis))
            displacement_norm = float(np.linalg.norm(displacement))
            distance_to_donor = float(np.linalg.norm(patched - donor))
            nearest_distance = float(distances[recipient, source, nearest[recipient, source]])
            tied = np.isclose(
                distances[recipient, source], nearest_distance, rtol=1e-12, atol=1e-12
            )
            if separation <= EPS:
                projection = math.nan
                distance_reduction = math.nan
                cosine = math.nan
                orthogonal = math.nan
                normalized_final_distance = math.nan
            else:
                denominator = float(np.dot(axis, axis))
                projection = float(np.dot(displacement, axis) / denominator)
                residual = displacement - projection * axis
                distance_reduction = 1.0 - distance_to_donor / separation
                normalized_final_distance = distance_to_donor / separation
                orthogonal = float(np.linalg.norm(residual) / separation)
                cosine = (
                    float(np.dot(displacement, axis) / (displacement_norm * separation))
                    if displacement_norm > EPS
                    else 0.0
                )
            cells.append(
                {
                    "state_id": str(result["state"]["state_id"]),
                    "task_family": str(result["state"]["task_family"]),
                    "recipient_seed": int(seeds[recipient]),
                    "future_source_seed": int(seeds[source]),
                    "recipient_index": recipient,
                    "future_source_index": source,
                    "nearest_native_seed": int(seeds[nearest[recipient, source]]),
                    "correct_source_top1": int(nearest[recipient, source] == source),
                    "nearest_tie_count": int(np.sum(tied)),
                    "correct_source_tie_fraction": (
                        float(1.0 / np.sum(tied)) if tied[source] else 0.0
                    ),
                    "native_separation_l2": separation,
                    "distance_to_donor_l2": distance_to_donor,
                    "normalized_final_distance": normalized_final_distance,
                    "distance_reduction": distance_reduction,
                    "normalized_projection": projection,
                    "cosine_alignment": cosine,
                    "orthogonal_residual": orthogonal,
                }
            )
    state_row: dict[str, Any] = {
        "state_id": str(result["state"]["state_id"]),
        "state_index": int(result["state"]["state_index"]),
        "episode_index": result["state"]["episode_index"],
        "frame_index": int(result["state"]["frame_index"]),
        "task_family": str(result["state"]["task_family"]),
        "prompt": str(result["state"]["prompt"]),
        "retrieval_accuracy_off_diagonal": float(np.mean((nearest == target)[offdiag])),
        "retrieval_accuracy_all_secondary": float(np.mean(nearest == target)),
        "retrieval_accuracy_tie_fraction": float(
            np.mean([cell["correct_source_tie_fraction"] for cell in cells])
        ),
        "distance_reduction": float(np.nanmean([cell["distance_reduction"] for cell in cells])),
        "normalized_projection": float(np.nanmean([cell["normalized_projection"] for cell in cells])),
        "cosine_alignment": float(np.nanmean([cell["cosine_alignment"] for cell in cells])),
        "orthogonal_residual": float(np.nanmean([cell["orthogonal_residual"] for cell in cells])),
        "normalized_final_distance": float(
            np.nanmean([cell["normalized_final_distance"] for cell in cells])
        ),
        "mean_native_separation_l2": float(
            np.mean([cell["native_separation_l2"] for cell in cells])
        ),
        "native_action_hash_distinct_count": int(result["native_action_hash_distinct_count"]),
        "native_video_trace_hash_distinct_count": int(
            result["native_video_trace_hash_distinct_count"]
        ),
        "self_replay_maximum_error": float(result["self_replay_maximum_error"]),
    }
    return state_row, cells, nearest


def permutation_pvalue(
    predictions: Sequence[np.ndarray],
    observed: float,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Shuffle all four future-source identities independently within state."""
    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
    offdiag = ~np.eye(4, dtype=bool)
    score_tables = []
    for prediction in predictions:
        scores = []
        for permutation in permutations:
            permuted_target = np.broadcast_to(permutation[None, :], (4, 4))
            scores.append(float(np.mean((prediction == permuted_target)[offdiag])))
        score_tables.append(scores)
    score_tables_array = np.asarray(score_tables, dtype=np.float64)
    draws = rng.integers(0, len(permutations), size=(repetitions, len(predictions)))
    state_indices = np.arange(len(predictions))[None, :]
    null = score_tables_array[state_indices, draws].mean(axis=1)
    pvalue = (float(np.sum(null >= observed - 1e-15)) + 1.0) / (repetitions + 1.0)
    return pvalue, float(null.mean())


def assign_separation_quartiles(cells: list[dict[str, Any]]) -> tuple[list[float], list[str]]:
    separations = np.asarray([cell["native_separation_l2"] for cell in cells], dtype=np.float64)
    boundaries = np.quantile(separations, (0.25, 0.50, 0.75)).tolist()
    labels = ("Q1 (smallest)", "Q2", "Q3", "Q4 (largest)")
    for cell in cells:
        index = int(np.searchsorted(boundaries, cell["native_separation_l2"], side="right"))
        cell["separation_quartile"] = index + 1
        cell["separation_quartile_label"] = labels[index]
    return [float(value) for value in boundaries], list(labels)


def quartile_summaries(
    cells: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    state_ids = sorted({str(cell["state_id"]) for cell in cells})
    for quartile, label in enumerate(labels, start=1):
        selected = [cell for cell in cells if int(cell["separation_quartile"]) == quartile]
        state_metric: dict[str, dict[str, float]] = {}
        for state_id in state_ids:
            rows = [cell for cell in selected if cell["state_id"] == state_id]
            if not rows:
                continue
            state_metric[state_id] = {
                metric: float(
                    np.nanmean([float(row[CELL_METRIC[metric]]) for row in rows])
                )
                for metric in METRICS
            }
        row: dict[str, Any] = {
            "separation_quartile": quartile,
            "label": label,
            "cell_count": len(selected),
            "state_count": len(state_metric),
            "separation_min": float(min(cell["native_separation_l2"] for cell in selected)),
            "separation_max": float(max(cell["native_separation_l2"] for cell in selected)),
        }
        for metric in METRICS:
            estimate = summarize_values(
                [values[metric] for values in state_metric.values()],
                repetitions,
                rng_for(seed, f"quartile:{quartile}:{metric}"),
            )
            row[f"{metric}_mean"] = estimate["mean"]
            row[f"{metric}_ci_low"] = estimate["state_bootstrap_95_ci"][0]
            row[f"{metric}_ci_high"] = estimate["state_bootstrap_95_ci"][1]
        output.append(row)
    return output


def leave_one_family_out(
    states: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    families = sorted({str(row["task_family"]) for row in states})
    output = []
    for family in families:
        kept = [row for row in states if row["task_family"] != family]
        value: dict[str, Any] = {
            "held_out_family": family,
            "n_states": len(kept),
        }
        for metric in metrics:
            array = np.asarray([row[metric] for row in kept], dtype=np.float64)
            finite = array[np.isfinite(array)]
            value[metric] = float(finite.mean()) if len(finite) else math.nan
        output.append(value)
    return output


def save_figure(fig: Any, stem: Path) -> None:
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "summarize_dreamzero_future_transplants.py"},
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "summarize_dreamzero_future_transplants.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )


def make_plots(
    output_root: Path,
    states: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
    quartiles: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    navy = "#244A68"
    blue = "#5B8DB8"
    gray = "#777777"
    light = "#D6E2EA"

    ordered = sorted(states, key=lambda row: (row["task_family"], row["state_id"]))
    families = sorted({str(row["task_family"]) for row in ordered})
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    positions = []
    labels = []
    cursor = 0
    for family in families:
        group = [row for row in ordered if row["task_family"] == family]
        x = np.arange(cursor, cursor + len(group))
        y = [row["retrieval_accuracy_off_diagonal"] for row in group]
        ax.scatter(x, y, s=27, color=blue, edgecolor="white", linewidth=0.5, zorder=3)
        ax.hlines(float(np.mean(y)), x[0] - 0.3, x[-1] + 0.3, color=navy, linewidth=2)
        positions.append(float(np.mean(x)))
        labels.append(family)
        cursor += len(group) + 1
    ax.axhline(0.25, color=gray, linestyle="--", linewidth=1, label="chance = 0.25")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("Correct future-source retrieval")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=light, linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, output_root / "retrieval_by_state")
    plt.close(fig)

    labels_for_metric = {
        "retrieval_accuracy_off_diagonal": "4-way source retrieval",
        "distance_reduction": "Distance reduction",
        "normalized_projection": "Normalized projection",
        "cosine_alignment": "Cosine alignment",
        "orthogonal_residual": "Orthogonal residual",
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(METRICS))[::-1]
    means = np.asarray([estimates[name]["mean"] for name in METRICS])
    cis = np.asarray([estimates[name]["state_bootstrap_95_ci"] for name in METRICS])
    errors = np.vstack((means - cis[:, 0], cis[:, 1] - means))
    ax.errorbar(means, y, xerr=errors, fmt="o", color=navy, ecolor=blue, capsize=3)
    ax.axvline(0.0, color=gray, linewidth=0.8)
    ax.scatter([0.25], [y[0]], marker="|", s=90, color=gray, zorder=4)
    ax.set_yticks(y, [labels_for_metric[name] for name in METRICS])
    ax.set_xlabel("State-level mean (95% bootstrap interval)")
    ax.grid(axis="x", color=light, linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, output_root / "metric_summary")
    plt.close(fig)

    x = np.arange(1, 5)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for ax, metric, ylabel, reference in (
        (axes[0], "retrieval_accuracy_off_diagonal", "4-way source retrieval", 0.25),
        (axes[1], "normalized_projection", "Normalized projection", 0.0),
    ):
        means = np.asarray([row[f"{metric}_mean"] for row in quartiles])
        low = np.asarray([row[f"{metric}_ci_low"] for row in quartiles])
        high = np.asarray([row[f"{metric}_ci_high"] for row in quartiles])
        axes_error = np.vstack((means - low, high - means))
        ax.errorbar(x, means, yerr=axes_error, fmt="o-", color=navy, ecolor=blue, capsize=3)
        ax.axhline(reference, color=gray, linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("Native action separation")
        ax.grid(axis="y", color=light, linewidth=0.6)
    axes[0].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    save_figure(fig, output_root / "separation_quartiles")
    plt.close(fig)


def render_markdown(
    summary: Mapping[str, Any], estimates: Mapping[str, Mapping[str, Any]]
) -> str:
    lines = [
        f"# DreamZero future-latent transplant — {summary['admission']}",
        "",
        f"Complete states: {summary['complete_state_count']}/{summary['expected_state_count']}",
        "",
        "| Estimand | Mean | State-bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        estimate = estimates[metric]
        low, high = estimate["state_bootstrap_95_ci"]
        lines.append(
            f"| {metric.replace('_', ' ')} | {estimate['mean']:.4f} | "
            f"[{low:.4f}, {high:.4f}] |"
        )
    lines.extend(
        [
            "",
            f"Off-diagonal within-state future-label permutation p: "
            f"{summary['retrieval_within_state_label_permutation_p']:.6g}",
            f"Permutation-null mean: {summary['retrieval_permutation_null_mean']:.4f}",
            f"Chance rate: {summary['chance_rate']:.2f}",
            f"Maximum self-replay error: {summary['maximum_self_replay_error']:.8g}",
            "",
            "The saved DROID state is the independent unit. The primary retrieval "
            "estimand excludes the four diagonal self-replays and averages the "
            "twelve donor cells within each state.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_latex(estimates: Mapping[str, Mapping[str, Any]]) -> str:
    labels = {
        "retrieval_accuracy_off_diagonal": "Correct source retrieval",
        "distance_reduction": "Distance reduction",
        "normalized_projection": "Normalized projection",
        "cosine_alignment": "Cosine alignment",
        "orthogonal_residual": "Orthogonal residual",
    }
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Estimand & Mean & 95\% state-bootstrap CI \\",
        r"\midrule",
    ]
    for metric in METRICS:
        item = estimates[metric]
        low, high = item["state_bootstrap_95_ci"]
        lines.append(f"{labels[metric]} & {item['mean']:.3f} & [{low:.3f}, {high:.3f}] \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.expected_state_count < 1:
        raise ValueError("--expected-state-count must be positive")
    if args.bootstrap_repetitions < 100 or args.permutation_repetitions < 100:
        raise ValueError("resampling repetitions must be at least 100")
    paths = discover_results(args.result_root, args.admission)
    if not args.allow_incomplete and len(paths) != args.expected_state_count:
        raise RuntimeError(
            f"refusing incomplete analysis: found {len(paths)} complete "
            f"{args.admission} states, expected {args.expected_state_count}"
        )
    if not paths:
        raise RuntimeError("no complete states")

    expected_states: dict[str, Mapping[str, Any]] | None = None
    manifest_sha: str | None = None
    if args.admission == "evaluation":
        if args.manifest is None:
            raise ValueError("evaluation analysis requires --manifest")
        manifest_sha = verify_sidecar(args.manifest)
        manifest = load_json(args.manifest)
        manifest_states = manifest.get("states")
        if not isinstance(manifest_states, list) or len(manifest_states) != args.expected_state_count:
            raise ValueError("frozen manifest state count differs from analysis contract")
        expected_states = {str(row["state_id"]): row for row in manifest_states}
        if len(expected_states) != len(manifest_states):
            raise ValueError("frozen manifest contains duplicate state IDs")
        observed_ids = {path.parent.name for path in paths}
        if observed_ids != set(expected_states):
            raise ValueError(
                f"result/manifest state membership differs: missing={sorted(set(expected_states)-observed_ids)}, "
                f"extra={sorted(observed_ids-set(expected_states))}"
            )
    elif args.manifest is not None:
        raise ValueError("excluded debug analysis must not use an evaluation manifest")

    state_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    runner_hashes: set[str] = set()
    manifest_hashes: set[str] = set()
    for path in paths:
        result, seeds, native, replay = load_and_audit_state(path, args.admission)
        state, cells, nearest = compute_state_metrics(result, seeds, native, replay)
        if expected_states is not None:
            expected = expected_states[state["state_id"]]
            for key in ("state_index", "episode_index", "frame_index", "task_family", "task"):
                observed_key = "prompt" if key == "task" else key
                if state[observed_key] != expected[key]:
                    raise ValueError(f"result/manifest state field differs at {state['state_id']}:{key}")
        state_rows.append(state)
        cell_rows.extend(cells)
        predictions.append(nearest)
        runner_hashes.add(str(result["runner_sha256"]))
        manifest_hashes.add(str(result["provenance"].get("manifest_file_sha256")))
    if len(runner_hashes) != 1:
        raise ValueError(f"multiple runner identities in cohort: {sorted(runner_hashes)}")
    if args.admission == "evaluation" and len(manifest_hashes) != 1:
        raise ValueError(f"multiple frozen manifest identities: {sorted(manifest_hashes)}")
    hex_characters = set("0123456789abcdef")
    runner_identity = next(iter(runner_hashes))
    if len(runner_identity) != 64 or set(runner_identity) - hex_characters:
        raise ValueError("runner identity is not a lowercase SHA-256")
    manifest_identity = next(iter(manifest_hashes))
    if args.admission == "evaluation" and (
        len(manifest_identity) != 64
        or set(manifest_identity) - hex_characters
        or manifest_identity != manifest_sha
    ):
        raise ValueError("result provenance does not match the frozen manifest SHA-256")

    state_rows.sort(key=lambda row: int(row["state_index"]))
    cell_rows.sort(key=lambda row: (row["state_id"], row["recipient_seed"], row["future_source_seed"]))
    boundaries, quartile_labels = assign_separation_quartiles(cell_rows)
    estimates = {
        metric: summarize_values(
            [float(row[metric]) for row in state_rows],
            args.bootstrap_repetitions,
            rng_for(args.seed, f"overall:{metric}"),
        )
        for metric in METRICS
    }
    secondary_estimates = {
        "retrieval_accuracy_all": summarize_values(
            [float(row["retrieval_accuracy_all_secondary"]) for row in state_rows],
            args.bootstrap_repetitions,
            rng_for(args.seed, "secondary:retrieval_accuracy_all"),
        )
    }
    observed = float(estimates["retrieval_accuracy_off_diagonal"]["mean"])
    permutation_p, null_mean = permutation_pvalue(
        predictions,
        observed,
        args.permutation_repetitions,
        rng_for(args.seed, "offdiagonal-four-source-permutation"),
    )
    quartiles = quartile_summaries(
        cell_rows,
        quartile_labels,
        args.bootstrap_repetitions,
        args.seed,
    )
    leave_out = leave_one_family_out(state_rows, METRICS)
    families = sorted({str(row["task_family"]) for row in state_rows})
    summary = {
        "schema": "dreamzero-future-transplant-analysis-v1",
        "admission": args.admission,
        "result_root": str(args.result_root.resolve()),
        "complete_state_count": len(state_rows),
        "expected_state_count": args.expected_state_count,
        "allow_incomplete": bool(args.allow_incomplete),
        "state_is_independent_unit": True,
        "primary_cells_per_state": 12,
        "diagonal_self_cells_excluded_from_primary": True,
        "chance_rate": 0.25,
        "retrieval_target_count": 4,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "permutation_repetitions": args.permutation_repetitions,
        "permutation_method": "Monte Carlo, independent global four-label permutation within each state",
        "resampling_seed": args.seed,
        "estimates": estimates,
        "secondary_audits": {
            "retrieval_accuracy_all_including_deterministic_self_controls":
                secondary_estimates["retrieval_accuracy_all"],
            "used_for_primary_inference": False,
        },
        "retrieval_within_state_label_permutation_p": permutation_p,
        "retrieval_permutation_null_mean": null_mean,
        "permutation_unit": "independent four-label permutation within each state",
        "separation_quartile_boundaries_l2": boundaries,
        "separation_quartiles": quartiles,
        "leave_one_family_out": leave_out,
        "task_families": families,
        "task_family_count": len(families),
        "runner_sha256": runner_identity,
        "manifest_file_sha256": manifest_identity,
        "all_states_four_distinct_native_actions": all(
            row["native_action_hash_distinct_count"] == 4 for row in state_rows
        ),
        "all_states_four_distinct_video_traces": all(
            row["native_video_trace_hash_distinct_count"] == 4 for row in state_rows
        ),
        "maximum_self_replay_error": max(row["self_replay_maximum_error"] for row in state_rows),
        "tie_affected_offdiagonal_cell_count": sum(
            int(row["nearest_tie_count"] > 1) for row in cell_rows
        ),
    }

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"analysis output root is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.provenance_receipt is not None:
        receipt_sha = verify_sidecar(args.provenance_receipt)
        receipt = load_json(args.provenance_receipt)
        if receipt.get("schema") != "dreamzero-runtime-provenance-receipt-v1":
            raise ValueError("unexpected DreamZero provenance receipt schema")
        if receipt.get("experiment", {}).get("runner", {}).get("sha256") != summary["runner_sha256"]:
            raise ValueError("provenance receipt runner differs from analyzed cohort")
        if receipt.get("experiment", {}).get("manifest", {}).get("sha256") != summary["manifest_file_sha256"]:
            raise ValueError("provenance receipt manifest differs from analyzed cohort")
        summary["provenance_receipt_sha256"] = receipt_sha
        atomic_bytes(
            args.output_root / "provenance_receipt.json",
            args.provenance_receipt.read_bytes(),
        )
    atomic_json(args.output_root / "summary.json", summary)
    write_csv(args.output_root / "state_metrics.csv", state_rows)
    write_csv(args.output_root / "cell_metrics.csv", cell_rows)
    write_csv(args.output_root / "separation_quartiles.csv", quartiles)
    write_csv(args.output_root / "leave_one_family_out.csv", leave_out)
    atomic_text(args.output_root / "results.md", render_markdown(summary, estimates))
    atomic_text(args.output_root / "table_main.tex", render_latex(estimates))
    make_plots(args.output_root, state_rows, estimates, quartiles)

    inventory_rows = []
    for path in sorted(args.output_root.iterdir()):
        if path.is_file() and not path.name.endswith(".sha256") and path.name != "artifact_inventory.json":
            inventory_rows.append(freeze_analysis_file(path))
    inventory = {
        "schema": "dreamzero-future-transplant-analysis-artifacts-v1",
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "artifacts": inventory_rows,
    }
    inventory_path = args.output_root / "artifact_inventory.json"
    atomic_json(inventory_path, inventory)
    freeze_analysis_file(inventory_path)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
