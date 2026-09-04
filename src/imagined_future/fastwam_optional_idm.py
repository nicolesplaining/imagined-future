"""Frozen design and artifact utilities for the FastWAM Optional-IDM study.

The GPU-facing runner lives in ``scripts/run_fastwam_optional_idm.py`` so this
module remains importable without FastWAM, Hydra, or LIBERO installed.  Keeping
the design code dependency-light also makes the manifest and metric tests
runnable before a cloud instance is provisioned.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


FASTWAM_UPSTREAM_REPOSITORY = "https://github.com/yuantianyuan01/FastWAM.git"
FASTWAM_UPSTREAM_COMMIT = "7faa71108368fbb3b6885649f112af607427a2d4"
FASTWAM_CHECKPOINT_REPOSITORY = "yuanty/fastwam"
FASTWAM_CHECKPOINT_FILENAME = "libero_optional_idm_2cam224.pt"
FASTWAM_STATS_FILENAME = "libero_optional_idm_2cam224_dataset_stats.json"
FASTWAM_CHECKPOINT_SHA256 = "26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4"
FASTWAM_STATS_SHA256 = "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
MANIFEST_SCHEMA_VERSION = 2


class FastWAMCondition(str, Enum):
    """Predeclared conditions in the Optional-IDM intervention matrix."""

    NATIVE = "native"
    SELF_LATENT = "self_latent"
    SELF_CACHE = "self_cache"
    DONOR_LATENT = "donor_latent"
    DONOR_CACHE = "donor_cache"
    WRONG_LATENT = "wrong_latent"
    SHUFFLED_CACHE = "shuffled_cache"
    FIRST_FRAME = "first_frame"


CORE_CONDITIONS = tuple(condition.value for condition in FastWAMCondition)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _short_digest(value: Any, *, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()[:length]


@dataclass(frozen=True)
class FastWAMBranchSpec:
    branch_id: str
    video_seed: int
    action_seed: int

    def __post_init__(self) -> None:
        if not self.branch_id or any(char.isspace() for char in self.branch_id):
            raise ValueError("branch_id must be nonempty and contain no whitespace")
        if self.video_seed < 0 or self.action_seed < 0:
            raise ValueError("video and action seeds must be nonnegative")

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class FastWAMStateSpec:
    suite: str
    task_id: int
    initial_state_index: int
    wait_steps: int
    branches: tuple[FastWAMBranchSpec, ...]

    def __post_init__(self) -> None:
        if not self.suite.startswith("libero_"):
            raise ValueError("suite must be a LIBERO benchmark name")
        if min(self.task_id, self.initial_state_index, self.wait_steps) < 0:
            raise ValueError("task, state, and wait indices must be nonnegative")
        if len(self.branches) < 3:
            raise ValueError("each state needs at least three branches for a wrong-donor control")
        branch_ids = [branch.branch_id for branch in self.branches]
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("branch IDs must be unique within a state")
        video_seeds = [branch.video_seed for branch in self.branches]
        if len(set(video_seeds)) != len(video_seeds):
            raise ValueError("video seeds must be unique within a state")

    @property
    def state_id(self) -> str:
        return (
            f"{self.suite}_task{self.task_id:02d}_"
            f"state{self.initial_state_index:03d}_wait{self.wait_steps:02d}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "suite": self.suite,
            "task_id": self.task_id,
            "initial_state_index": self.initial_state_index,
            "wait_steps": self.wait_steps,
            "branches": [branch.to_dict() for branch in self.branches],
        }


@dataclass(frozen=True)
class FastWAMRunSpec:
    run_id: str
    state_id: str
    condition: str
    recipient_id: str
    donor_id: str | None
    source_id: str | None
    action_seed: int
    shuffle_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_branches(
    video_seeds: Sequence[int], action_seeds: Sequence[int]
) -> tuple[FastWAMBranchSpec, ...]:
    if len(video_seeds) != len(action_seeds):
        raise ValueError("video_seeds and action_seeds must have equal lengths")
    return tuple(
        FastWAMBranchSpec(f"b{index:02d}", int(video_seed), int(action_seed))
        for index, (video_seed, action_seed) in enumerate(
            zip(video_seeds, action_seeds, strict=True)
        )
    )


def build_manifest_body(
    *,
    study_name: str,
    states: Sequence[FastWAMStateSpec],
    inference: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct the immutable portion of a manifest.

    Paths to local checkouts, checkpoints, and output directories are excluded
    deliberately: moving an identical registered study between workers must not
    change its identity.
    """

    if not study_name.strip():
        raise ValueError("study_name must be nonempty")
    if not states:
        raise ValueError("manifest must contain at least one state")
    state_ids = [state.state_id for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("manifest contains duplicate state IDs")
    required_inference = {
        "num_inference_steps",
        "sigma_shift",
        "num_video_frames",
        "action_horizon",
        "rand_device",
    }
    missing = required_inference - set(inference)
    if missing:
        raise ValueError(f"inference config missing keys: {sorted(missing)}")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study_name": study_name,
        "upstream": {
            "repository": FASTWAM_UPSTREAM_REPOSITORY,
            "commit": FASTWAM_UPSTREAM_COMMIT,
        },
        "checkpoint": {
            "repository": FASTWAM_CHECKPOINT_REPOSITORY,
            "filename": FASTWAM_CHECKPOINT_FILENAME,
            "stats_filename": FASTWAM_STATS_FILENAME,
        },
        "conditions": list(CORE_CONDITIONS),
        "inference": dict(inference),
        "states": [state.to_dict() for state in states],
    }


def freeze_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a content-derived ID and reject malformed upstream metadata."""

    normalized = json.loads(_canonical_json(dict(body)).decode("utf-8"))
    if "manifest_id" in normalized:
        raise ValueError("freeze_manifest expects an unfrozen body")
    if normalized.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema version")
    upstream = normalized.get("upstream", {})
    if upstream.get("repository") != FASTWAM_UPSTREAM_REPOSITORY:
        raise ValueError("manifest uses an unexpected FastWAM repository")
    if upstream.get("commit") != FASTWAM_UPSTREAM_COMMIT:
        raise ValueError("manifest uses an unexpected FastWAM commit")
    return {"manifest_id": f"fastwam-{_short_digest(normalized)}", **normalized}


def validate_frozen_manifest(manifest: Mapping[str, Any]) -> None:
    manifest_dict = dict(manifest)
    manifest_id = manifest_dict.pop("manifest_id", None)
    if not isinstance(manifest_id, str):
        raise ValueError("manifest_id is missing")
    expected = freeze_manifest(manifest_dict)["manifest_id"]
    if manifest_id != expected:
        raise ValueError(
            f"manifest content hash mismatch: recorded {manifest_id}, expected {expected}"
        )


def load_frozen_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_frozen_manifest(manifest)
    return manifest


def state_from_dict(value: Mapping[str, Any]) -> FastWAMStateSpec:
    branches = tuple(
        FastWAMBranchSpec(
            branch_id=str(branch["branch_id"]),
            video_seed=int(branch["video_seed"]),
            action_seed=int(branch["action_seed"]),
        )
        for branch in value["branches"]
    )
    state = FastWAMStateSpec(
        suite=str(value["suite"]),
        task_id=int(value["task_id"]),
        initial_state_index=int(value["initial_state_index"]),
        wait_steps=int(value["wait_steps"]),
        branches=branches,
    )
    if value.get("state_id") != state.state_id:
        raise ValueError("state_id does not match its frozen state fields")
    return state


def _run_spec(
    *,
    manifest_id: str,
    state_id: str,
    condition: FastWAMCondition,
    recipient: FastWAMBranchSpec,
    donor_id: str | None,
    source_id: str | None,
    shuffle_seed: int | None = None,
) -> FastWAMRunSpec:
    identity = {
        "manifest_id": manifest_id,
        "state_id": state_id,
        "condition": condition.value,
        "recipient_id": recipient.branch_id,
        "donor_id": donor_id,
        "source_id": source_id,
        "action_seed": recipient.action_seed,
        "shuffle_seed": shuffle_seed,
    }
    return FastWAMRunSpec(
        run_id=f"run-{_short_digest(identity, length=20)}",
        state_id=state_id,
        condition=condition.value,
        recipient_id=recipient.branch_id,
        donor_id=donor_id,
        source_id=source_id,
        action_seed=recipient.action_seed,
        shuffle_seed=shuffle_seed,
    )


def expand_run_specs(
    manifest_id: str, state: FastWAMStateSpec
) -> tuple[FastWAMRunSpec, ...]:
    """Expand one state into the complete preregistered condition matrix."""

    runs: list[FastWAMRunSpec] = []
    for recipient in state.branches:
        for condition in (
            FastWAMCondition.NATIVE,
            FastWAMCondition.SELF_LATENT,
            FastWAMCondition.SELF_CACHE,
        ):
            runs.append(
                _run_spec(
                    manifest_id=manifest_id,
                    state_id=state.state_id,
                    condition=condition,
                    recipient=recipient,
                    donor_id=None,
                    source_id=recipient.branch_id,
                )
            )

        for donor in state.branches:
            if donor.branch_id == recipient.branch_id:
                continue
            # The first-frame path rejects future latent/cache overrides. We
            # nevertheless run one donor-labelled arm per ordered pair with the
            # donor's registered video seed and the recipient's action seed.
            # Exact agreement across donors verifies that donor identity is not
            # consumed by the no-future route.
            runs.append(
                _run_spec(
                    manifest_id=manifest_id,
                    state_id=state.state_id,
                    condition=FastWAMCondition.FIRST_FRAME,
                    recipient=recipient,
                    donor_id=donor.branch_id,
                    source_id=donor.branch_id,
                )
            )
            wrong = next(
                branch
                for branch in state.branches
                if branch.branch_id not in {recipient.branch_id, donor.branch_id}
            )
            for condition, source in (
                (FastWAMCondition.DONOR_LATENT, donor),
                (FastWAMCondition.DONOR_CACHE, donor),
                (FastWAMCondition.WRONG_LATENT, wrong),
            ):
                runs.append(
                    _run_spec(
                        manifest_id=manifest_id,
                        state_id=state.state_id,
                        condition=condition,
                        recipient=recipient,
                        donor_id=donor.branch_id,
                        source_id=source.branch_id,
                    )
                )
            shuffle_seed = int(
                hashlib.sha256(
                    f"{manifest_id}:{state.state_id}:{recipient.branch_id}:{donor.branch_id}".encode()
                ).hexdigest()[:8],
                16,
            )
            runs.append(
                _run_spec(
                    manifest_id=manifest_id,
                    state_id=state.state_id,
                    condition=FastWAMCondition.SHUFFLED_CACHE,
                    recipient=recipient,
                    donor_id=donor.branch_id,
                    source_id=donor.branch_id,
                    shuffle_seed=shuffle_seed,
                )
            )
    run_ids = [run.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("run ID collision in expanded manifest")
    return tuple(runs)


def action_metrics(
    action: np.ndarray,
    recipient_action: np.ndarray,
    donor_action: np.ndarray,
    candidate_actions: Mapping[str, np.ndarray],
    *,
    donor_id: str,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compute directional and donor-retrieval metrics for one action chunk."""

    action_flat = np.asarray(action, dtype=np.float64).reshape(-1)
    recipient_flat = np.asarray(recipient_action, dtype=np.float64).reshape(-1)
    donor_flat = np.asarray(donor_action, dtype=np.float64).reshape(-1)
    if action_flat.shape != recipient_flat.shape or donor_flat.shape != recipient_flat.shape:
        raise ValueError("action chunks must have identical flattened shapes")
    donor_delta = donor_flat - recipient_flat
    patched_delta = action_flat - recipient_flat
    donor_norm_sq = float(np.dot(donor_delta, donor_delta))
    patched_norm = float(np.linalg.norm(patched_delta))
    distances = {
        branch_id: float(
            np.linalg.norm(action_flat - np.asarray(candidate, dtype=np.float64).reshape(-1))
        )
        for branch_id, candidate in candidate_actions.items()
    }
    if donor_id not in distances:
        raise ValueError("donor_id is absent from candidate_actions")
    nearest_id = min(distances, key=lambda key: (distances[key], key))
    distance_to_recipient = float(np.linalg.norm(action_flat - recipient_flat))
    distance_to_donor = float(np.linalg.norm(action_flat - donor_flat))
    if donor_norm_sq <= eps:
        # Selection-free cohorts can contain genuinely indistinguishable native
        # actions. Preserve the row and mark its directional axis undefined;
        # never silently exclude or invent a projection for it.
        return {
            "axis_degenerate": True,
            "donor_projection": None,
            "cosine_alignment": None,
            "distance_to_recipient": distance_to_recipient,
            "distance_to_donor": distance_to_donor,
            "native_recipient_to_donor_distance": float(np.sqrt(donor_norm_sq)),
            "donor_distance_reduction": None,
            "orthogonal_residual": patched_norm,
            "orthogonal_residual_ratio": None,
            "nearest_branch_id": nearest_id,
            "correct_donor_retrieval": nearest_id == donor_id,
            "candidate_distances": distances,
        }
    donor_distance = float(np.sqrt(donor_norm_sq))
    projection = float(np.dot(patched_delta, donor_delta) / donor_norm_sq)
    aligned = projection * donor_delta
    orthogonal = patched_delta - aligned
    cosine = (
        float(np.dot(patched_delta, donor_delta) / (patched_norm * donor_distance))
        if patched_norm > eps
        else 0.0
    )
    return {
        "axis_degenerate": False,
        "donor_projection": projection,
        "cosine_alignment": cosine,
        "distance_to_recipient": distance_to_recipient,
        "distance_to_donor": distance_to_donor,
        "native_recipient_to_donor_distance": donor_distance,
        "donor_distance_reduction": 1.0 - distance_to_donor / donor_distance,
        "orthogonal_residual": float(np.linalg.norm(orthogonal)),
        "orthogonal_residual_ratio": float(np.linalg.norm(orthogonal) / donor_distance),
        "nearest_branch_id": nearest_id,
        "correct_donor_retrieval": nearest_id == donor_id,
        "candidate_distances": distances,
    }


def shuffled_kv_cache(
    cache_k: Sequence[torch.Tensor],
    cache_v: Sequence[torch.Tensor],
    *,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Break K/V correspondence while preserving each tensor's marginal values.

    Paired token permutation would be attention-invariant when action queries can
    attend all future tokens.  This control therefore leaves keys fixed and
    deterministically permutes values along the token axis independently at each
    layer.
    """

    if len(cache_k) != len(cache_v) or not cache_k:
        raise ValueError("cache K and V must have the same nonzero layer count")
    shuffled_k = [tensor.detach().clone() for tensor in cache_k]
    shuffled_v: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for layer, (key, value) in enumerate(zip(cache_k, cache_v, strict=True)):
        if key.shape != value.shape or value.ndim < 2:
            raise ValueError(f"cache layer {layer} has incompatible K/V shapes")
        permutation = torch.randperm(value.shape[1], generator=generator)
        shuffled_v.append(value.detach().clone().index_select(1, permutation.to(value.device)))
    return shuffled_k, shuffled_v


def cache_descriptor(
    cache_k: Sequence[torch.Tensor], cache_v: Sequence[torch.Tensor]
) -> dict[str, Any]:
    if len(cache_k) != len(cache_v):
        raise ValueError("cache K/V layer counts differ")
    return {
        "layers": len(cache_k),
        "k_shapes": [list(tensor.shape) for tensor in cache_k],
        "v_shapes": [list(tensor.shape) for tensor in cache_v],
        "k_dtypes": [str(tensor.dtype) for tensor in cache_k],
        "v_dtypes": [str(tensor.dtype) for tensor in cache_v],
        "k_l2": [float(tensor.float().norm().item()) for tensor in cache_k],
        "v_l2": [float(tensor.float().norm().item()) for tensor in cache_v],
    }


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_npz(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    compressed: bool = False,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp.npz",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_frozen_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write once, or verify that an existing manifest is byte-equivalent."""

    validate_frozen_manifest(manifest)
    destination = Path(path)
    if destination.exists():
        existing = load_frozen_manifest(destination)
        if _canonical_json(existing) != _canonical_json(dict(manifest)):
            raise FileExistsError(f"refusing to replace a different manifest at {destination}")
        return
    atomic_write_json(destination, manifest)
