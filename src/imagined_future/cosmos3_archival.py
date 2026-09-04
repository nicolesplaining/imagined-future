"""Deterministic input helpers for the archival Cosmos 3 action-only study."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PAPER_TASKS = (
    "BananaInBowlTask",
    "RubiksCubeTask",
    "MustardInLeftBinTask",
    "SpoonInMugTask",
    "MarkerInMugTask",
    "SmartphoneInBinTask",
)
ENVIRONMENT_SEEDS = (101, 103, 107, 109, 113)
BRANCH_SEEDS = (211, 223, 227, 229)
PHASES = (("early", 0.20), ("middle", 0.50), ("late", 0.80))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    """Write a new immutable JSON artifact without exposing partial output."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def admissible_branch_steps(episode_length: int) -> tuple[int, ...]:
    """Return valid timesteps congruent to 16 modulo 32."""

    length = int(episode_length)
    if length < 2:
        return ()
    return tuple(step for step in range(16, length, 32) if 1 <= step <= length - 1)


def phase_branch_steps(
    episode_length: int,
    phases: Iterable[tuple[str, float]] = PHASES,
) -> tuple[dict[str, float | int | str], ...]:
    """Map fixed episode-relative phases to nearest admissible steps.

    The continuous target is ``q * (episode_length - 1)``.  Equal-distance
    ties select the lower timestep.  The caller must not adapt a failed grid.
    """

    length = int(episode_length)
    candidates = admissible_branch_steps(length)
    if not candidates:
        raise ValueError(f"episode length {length} has no valid 16 mod 32 timestep")
    rows: list[dict[str, float | int | str]] = []
    for phase, quantile in phases:
        q = float(quantile)
        if not 0.0 < q < 1.0:
            raise ValueError(f"phase quantile must be inside (0,1), got {q}")
        target = q * (length - 1)
        step = min(candidates, key=lambda candidate: (abs(candidate - target), candidate))
        rows.append(
            {
                "phase": str(phase),
                "phase_fraction": q,
                "continuous_target_step": target,
                "branch_step": step,
                "mp4_frame_index": step - 1,
                "hdf5_state_index": step - 1,
            }
        )
    steps = [int(row["branch_step"]) for row in rows]
    if len(steps) != 3 or len(set(steps)) != 3:
        raise ValueError(
            f"episode length {length} does not yield three distinct frozen phase steps: {steps}"
        )
    return tuple(rows)


def half_size_bilinear(image: np.ndarray) -> np.ndarray:
    """Match RoboLab's torch bilinear 360x640 -> 180x320 composition."""

    import torch
    import torch.nn.functional as functional

    value = np.asarray(image)
    if value.shape != (360, 640, 3) or value.dtype != np.uint8:
        raise ValueError(f"panel must be uint8 (360,640,3), got {value.shape} {value.dtype}")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)
    resized = functional.interpolate(
        tensor.unsqueeze(0).float(), size=(180, 320), mode="bilinear"
    )
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)


def compose_cosmos_observation(frame: np.ndarray) -> np.ndarray:
    """Compose wrist + left/right views from an archived four-panel MP4 frame."""

    value = np.asarray(frame)
    if value.shape != (360, 2560, 3) or value.dtype != np.uint8:
        raise ValueError(
            f"archived frame must be uint8 (360,2560,3), got {value.shape} {value.dtype}"
        )
    # Recorded panel order is head, over-shoulder-left, over-shoulder-right, wrist.
    head, left, right, wrist = np.split(value, 4, axis=1)
    del head
    bottom = np.concatenate((half_size_bilinear(left), half_size_bilinear(right)), axis=1)
    return np.concatenate((wrist, bottom), axis=0)


def recorded_proprio(joint_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(joint_position, dtype=np.float32).reshape(-1)
    if value.size < 8:
        raise ValueError(f"recorded joint vector must have at least 8 entries, got {value.size}")
    joints = value[:7].copy()
    gripper = np.clip(value[7:8] / (np.pi / 4), 0, 1).astype(np.float32)
    return joints, gripper


def deterministic_wrong_donor(
    recipient_seed: int, donor_seed: int, branch_seeds: Iterable[int] = BRANCH_SEEDS
) -> int:
    """Assign a nonrecipient, nondonor label without looking at model output."""

    seeds = tuple(int(seed) for seed in branch_seeds)
    donor_cycle = [seed for seed in seeds if seed != recipient_seed]
    if donor_seed not in donor_cycle or len(set(seeds)) != len(seeds):
        raise ValueError("recipient/donor must be distinct members of the unique branch-seed set")
    # A one-place cyclic rotation is a balanced derangement of the three donor
    # labels within each recipient; it depends only on frozen seed order.
    donor_index = donor_cycle.index(donor_seed)
    return donor_cycle[(donor_index + 1) % len(donor_cycle)]


def deterministic_shuffled_source(
    source_seed: int, branch_seeds: Iterable[int] = BRANCH_SEEDS
) -> int:
    """Apply a balanced cyclic derangement to all four future-source labels."""

    seeds = tuple(int(seed) for seed in branch_seeds)
    if len(set(seeds)) != len(seeds) or source_seed not in seeds:
        raise ValueError("source must be a member of the unique branch-seed set")
    index = seeds.index(int(source_seed))
    return seeds[(index + 1) % len(seeds)]
