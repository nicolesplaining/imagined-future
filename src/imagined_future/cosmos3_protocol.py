"""Pure protocol helpers for the Cosmos 3 RoboLab study runner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


FROZEN_TASK_OBJECT_NAMES = {
    "BananaInBowlTask": "banana",
    "RubiksCubeTask": "rubiks_cube",
    "MustardInLeftBinTask": "mustard",
    "SpoonInMugTask": "spoon_big",
    "MarkerInMugTask": "marker",
    "SmartphoneInBinTask": "smartphone",
}


MINIMAL_KV_PHYSICAL_LABELS = frozenset(
    {
        "self",
        "predicted_donor",
        "executed_donor",
        "predicted_donor_kv_patch_all_action",
        "executed_donor_kv_patch_all_action",
        "self_with_predicted_donor_kv",
        "self_with_executed_donor_kv",
    }
)


def should_execute_intervention(label: str, *, minimal_kv_factorial: bool) -> bool:
    """Return whether an intervention action should be stepped in the simulator."""

    return not minimal_kv_factorial or label in MINIMAL_KV_PHYSICAL_LABELS


def ordered_recipient_donor_pairs(
    branch_seeds: Sequence[int],
) -> list[tuple[int, int]]:
    """Enumerate every directed non-self pair in frozen seed order."""

    seeds = [int(seed) for seed in branch_seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("branch seeds must be unique")
    return [
        (recipient, donor)
        for recipient in seeds
        for donor in seeds
        if donor != recipient
    ]


def directional_target_metrics(
    value: np.ndarray,
    recipient: np.ndarray,
    target: np.ndarray | None,
    *,
    eps: float = 1e-12,
) -> dict[str, float | None]:
    """Measure target attraction without assuming the result lies on the donor axis."""

    names = (
        "l2_to_target",
        "native_target_l2",
        "distance_reduction_to_target",
        "cosine_alignment",
        "orthogonal_residual_normalized",
    )
    if target is None:
        return {name: None for name in names}

    value_flat = np.asarray(value, dtype=np.float64).reshape(-1)
    recipient_flat = np.asarray(recipient, dtype=np.float64).reshape(-1)
    target_flat = np.asarray(target, dtype=np.float64).reshape(-1)
    if value_flat.shape != recipient_flat.shape or target_flat.shape != recipient_flat.shape:
        raise ValueError("value, recipient, and target must have identical flattened shapes")

    target_direction = target_flat - recipient_flat
    result_direction = value_flat - recipient_flat
    native_target_l2 = float(np.linalg.norm(target_direction))
    l2_to_target = float(np.linalg.norm(value_flat - target_flat))
    if native_target_l2 <= eps:
        return {
            "l2_to_target": l2_to_target,
            "native_target_l2": native_target_l2,
            "distance_reduction_to_target": None,
            "cosine_alignment": None,
            "orthogonal_residual_normalized": None,
        }

    result_l2 = float(np.linalg.norm(result_direction))
    projection_coefficient = float(
        np.dot(result_direction, target_direction) / np.dot(target_direction, target_direction)
    )
    orthogonal = result_direction - projection_coefficient * target_direction
    return {
        "l2_to_target": l2_to_target,
        "native_target_l2": native_target_l2,
        "distance_reduction_to_target": 1.0 - l2_to_target / native_target_l2,
        "cosine_alignment": (
            float(np.dot(result_direction, target_direction) / (result_l2 * native_target_l2))
            if result_l2 > eps
            else None
        ),
        "orthogonal_residual_normalized": float(
            np.linalg.norm(orthogonal) / native_target_l2
        ),
    }


def native_execution_seeds(
    branch_seeds: Sequence[int],
    frozen_pair: tuple[int, int] | None,
    *,
    multi_donor: bool,
) -> set[int]:
    """Return branches that require physical execution and endpoint registration."""

    if multi_donor or frozen_pair is None:
        return {int(seed) for seed in branch_seeds}
    return {int(seed) for seed in frozen_pair}


def donor_selection_description(
    frozen_pair: tuple[int, int] | None,
    *,
    multi_donor: bool,
) -> str:
    """Describe pair selection without inferring how an external manifest was built."""

    if frozen_pair is None:
        return "maximum physically executed native endpoint separation"
    if multi_donor:
        return (
            "externally supplied frozen recipient/donor pair; all other branch seeds are "
            "prespecified additional donors; no within-run pair selection"
        )
    return "externally supplied frozen recipient/donor pair; no within-run pair selection"


def donor_kv_factorial_interventions(
    *,
    study_id: str,
    recipient_seed: int,
    donor_seed: int,
    layers: Sequence[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Build the donor-cache arms that complete the future x K/V factorial.

    These conditions must be appended after every consumer of the recipient K/V
    cache. Recording a cache currently clears the server's prior cache, so the
    returned insertion order is record donor, exact donor replay, then rescue of
    the recipient future with donor K/V.
    """

    all_layers = [int(layer) for layer in layers]
    recipient_id = f"{study_id}-native-{recipient_seed}"
    specs: dict[str, dict[str, Any]] = {}
    for source, donor_id in (
        ("predicted", f"{study_id}-native-{donor_seed}"),
        ("executed", f"{study_id}-executed-{donor_seed}"),
    ):
        cache_id = f"{study_id}-{source}-donor-future-kv"
        common = {
            "research_attention_cache_id": cache_id,
            "research_attention_exclude_layers": all_layers,
            "research_attention_exclude_scope": "action",
        }
        specs.update(
            {
                f"{source}_donor_kv_record": {
                    "research_mode": "donor",
                    "research_donor_id": donor_id,
                    "research_attention_mode": "record",
                    **common,
                },
                f"{source}_donor_kv_replay": {
                    "research_mode": "donor",
                    "research_donor_id": donor_id,
                    "research_attention_mode": "patch",
                    **common,
                },
                f"self_with_{source}_donor_kv": {
                    "research_mode": "self",
                    "research_donor_id": recipient_id,
                    "research_attention_mode": "patch",
                    **common,
                },
            }
        )
    targets = {label: donor_seed for label in specs}
    return specs, targets
