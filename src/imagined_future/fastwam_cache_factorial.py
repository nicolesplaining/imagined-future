"""Frozen design utilities for the FastWAM future x video-cache factorial.

This is intentionally separate from the eight-condition population manifest so
the already-running powered study and its global condition enum remain intact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fastwam_optional_idm import (
    FASTWAM_CHECKPOINT_SHA256,
    FASTWAM_STATS_SHA256,
    FASTWAM_UPSTREAM_COMMIT,
    FastWAMStateSpec,
    atomic_write_json,
    load_frozen_manifest,
    state_from_dict,
)


CACHE_FACTORIAL_SCHEMA_VERSION = 1
CACHE_FACTORIAL_CONDITIONS = (
    "future_recipient_cache_recipient",
    "future_donor_cache_recipient",
    "future_recipient_cache_donor",
    "future_donor_cache_donor",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()[:length]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheFactorialRunSpec:
    run_id: str
    state_id: str
    condition: str
    recipient_id: str
    donor_id: str
    future_source_id: str
    cache_source_id: str
    action_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_cache_factorial_body(
    *,
    study_name: str,
    base_manifest: Mapping[str, Any],
    base_manifest_sha256: str,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    if not study_name.strip():
        raise ValueError("study_name must be nonempty")
    if len(base_manifest_sha256) != 64:
        raise ValueError("base_manifest_sha256 must be a full SHA-256 digest")
    return {
        "schema_version": CACHE_FACTORIAL_SCHEMA_VERSION,
        "study_name": study_name,
        "base_manifest_id": str(base_manifest["manifest_id"]),
        "base_manifest_sha256": base_manifest_sha256,
        "upstream": dict(base_manifest["upstream"]),
        "checkpoint": {
            **dict(base_manifest["checkpoint"]),
            "checkpoint_sha256": FASTWAM_CHECKPOINT_SHA256,
            "stats_sha256": FASTWAM_STATS_SHA256,
        },
        "inference": dict(base_manifest["inference"]),
        "states": list(base_manifest["states"]),
        "conditions": list(CACHE_FACTORIAL_CONDITIONS),
        "design": dict(design),
    }


def freeze_cache_factorial_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_canonical_json(dict(body)).decode("utf-8"))
    if "manifest_id" in normalized:
        raise ValueError("freeze expects an unfrozen factorial body")
    if normalized.get("schema_version") != CACHE_FACTORIAL_SCHEMA_VERSION:
        raise ValueError("unsupported cache-factorial schema")
    if normalized.get("conditions") != list(CACHE_FACTORIAL_CONDITIONS):
        raise ValueError("cache-factorial conditions differ from the frozen 2x2")
    if normalized.get("upstream", {}).get("commit") != FASTWAM_UPSTREAM_COMMIT:
        raise ValueError("cache-factorial manifest uses an unexpected FastWAM commit")
    return {"manifest_id": f"fastwam-kvfact-{_digest(normalized)}", **normalized}


def validate_cache_factorial_manifest(manifest: Mapping[str, Any]) -> None:
    body = dict(manifest)
    recorded = body.pop("manifest_id", None)
    if not isinstance(recorded, str):
        raise ValueError("factorial manifest_id is missing")
    expected = freeze_cache_factorial_manifest(body)["manifest_id"]
    if recorded != expected:
        raise ValueError(
            f"factorial manifest content hash mismatch: {recorded} != {expected}"
        )


def load_cache_factorial_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_cache_factorial_manifest(manifest)
    return manifest


def write_cache_factorial_manifest(
    path: str | Path, manifest: Mapping[str, Any]
) -> None:
    validate_cache_factorial_manifest(manifest)
    destination = Path(path)
    if destination.exists():
        existing = load_cache_factorial_manifest(destination)
        if _canonical_json(existing) != _canonical_json(dict(manifest)):
            raise FileExistsError(f"refusing to replace different manifest: {destination}")
        return
    atomic_write_json(destination, manifest)


def validate_factorial_parent(
    factorial_manifest: Mapping[str, Any], base_manifest_path: str | Path
) -> dict[str, Any]:
    base_manifest = load_frozen_manifest(base_manifest_path)
    if base_manifest["manifest_id"] != factorial_manifest["base_manifest_id"]:
        raise ValueError("factorial manifest points to a different base manifest ID")
    actual_sha = file_sha256(base_manifest_path)
    if actual_sha != factorial_manifest["base_manifest_sha256"]:
        raise ValueError("base manifest file SHA-256 differs from the frozen factorial")
    if base_manifest["states"] != factorial_manifest["states"]:
        raise ValueError("factorial state population differs from its base manifest")
    if base_manifest["inference"] != factorial_manifest["inference"]:
        raise ValueError("factorial inference settings differ from its base manifest")
    return base_manifest


def expand_cache_factorial_runs(
    manifest_id: str, state: FastWAMStateSpec
) -> tuple[CacheFactorialRunSpec, ...]:
    source_by_condition = {
        "future_recipient_cache_recipient": ("recipient", "recipient"),
        "future_donor_cache_recipient": ("donor", "recipient"),
        "future_recipient_cache_donor": ("recipient", "donor"),
        "future_donor_cache_donor": ("donor", "donor"),
    }
    runs: list[CacheFactorialRunSpec] = []
    for recipient in state.branches:
        for donor in state.branches:
            if donor.branch_id == recipient.branch_id:
                continue
            for condition in CACHE_FACTORIAL_CONDITIONS:
                future_side, cache_side = source_by_condition[condition]
                future_source_id = (
                    recipient.branch_id if future_side == "recipient" else donor.branch_id
                )
                cache_source_id = (
                    recipient.branch_id if cache_side == "recipient" else donor.branch_id
                )
                identity = {
                    "manifest_id": manifest_id,
                    "state_id": state.state_id,
                    "condition": condition,
                    "recipient_id": recipient.branch_id,
                    "donor_id": donor.branch_id,
                    "future_source_id": future_source_id,
                    "cache_source_id": cache_source_id,
                    "action_seed": recipient.action_seed,
                }
                runs.append(
                    CacheFactorialRunSpec(
                        run_id=f"run-{_digest(identity, 20)}",
                        state_id=state.state_id,
                        condition=condition,
                        recipient_id=recipient.branch_id,
                        donor_id=donor.branch_id,
                        future_source_id=future_source_id,
                        cache_source_id=cache_source_id,
                        action_seed=recipient.action_seed,
                    )
                )
    if len(runs) != len(state.branches) * (len(state.branches) - 1) * 4:
        raise RuntimeError("factorial expansion produced the wrong run count")
    if len({run.run_id for run in runs}) != len(runs):
        raise RuntimeError("factorial run ID collision")
    return tuple(runs)


def states_from_factorial_manifest(
    manifest: Mapping[str, Any]
) -> tuple[FastWAMStateSpec, ...]:
    return tuple(state_from_dict(value) for value in manifest["states"])

