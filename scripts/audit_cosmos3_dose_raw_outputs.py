#!/usr/bin/env python3
"""Independent, inventory-gated audit of complete Cosmos dose-v2 outputs.

No frozen dose module, runner, analyzer, or analysis helper is imported. Outcome
JSON is not parsed until the external inventory proves an exact, immutable 30-file
package. The frozen summary is loaded only after every raw estimate is recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import cosmos3_raw_audit_common as audit_common

from cosmos3_raw_audit_common import (
    atomic_json_no_overwrite,
    compare_tree,
    finite,
    load_json,
    load_reference_after_hash,
    require_sha,
    sha256,
    verify_frozen_package,
)


SEEDS = (211, 223, 227, 229)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260903
PRIMARY_ADMISSION = "prospective_action_level_future_strength_dose_response"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--authorization-report", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def action(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (32, 8) or result.size != 256 or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected finite [32,8]/256 action")
    return result


def request_specs(branch_seeds: list[int]) -> list[dict[str, Any]]:
    seeds = tuple(int(value) for value in branch_seeds)
    if seeds != SEEDS:
        raise ValueError(f"branch seeds differ from frozen order: {seeds}")
    pairs = [(left, right) for left in seeds for right in seeds if left != right]
    rows: list[dict[str, Any]] = []
    rows.extend({"label": f"native-{seed}", "kind": "native", "recipient_seed": seed} for seed in seeds)
    rows.extend({"label": f"native-repeat-{seed}", "kind": "native_repeat", "recipient_seed": seed} for seed in seeds)
    rows.extend({"label": f"none-{seed}", "kind": "none", "recipient_seed": seed} for seed in seeds)
    rows.extend({"label": f"self-{seed}", "kind": "self", "recipient_seed": seed} for seed in seeds)
    rows.extend({"label": f"self-repeat-{seed}", "kind": "self_repeat", "recipient_seed": seed} for seed in seeds)
    for recipient, donor in pairs:
        for alpha in ALPHAS:
            rows.append({
                "label": dose_label(recipient, donor, alpha), "kind": "dose",
                "recipient_seed": recipient, "donor_seed": donor, "alpha": alpha,
            })
    rows.extend({
        "label": dose_label(recipient, donor, 0.5) + "-repeat",
        "kind": "dose_midpoint_repeat", "recipient_seed": recipient,
        "donor_seed": donor, "alpha": 0.5,
    } for recipient, donor in pairs)
    if len(rows) != 92 or len({row["label"] for row in rows}) != 92:
        raise AssertionError("frozen dose request sequence is not 92 unique calls")
    return rows


def dose_label(recipient: int, donor: int, alpha: float) -> str:
    tokens = {0.0: "000", 0.25: "025", 0.5: "050", 0.75: "075", 1.0: "100"}
    if float(alpha) not in tokens:
        raise ValueError(f"alpha outside frozen grid: {alpha}")
    return f"recipient-{int(recipient)}-donor-{int(donor)}-alpha-{tokens[float(alpha)]}"


def metadata_signature(metadata: Mapping[str, Any]) -> str:
    excluded = {"research_id", "research_infer_ms", "server_timing"}
    payload = {key: value for key, value in metadata.items() if key not in excluded}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def behavior_signature(value: Any, metadata: Mapping[str, Any]) -> str:
    """Independently reproduce the frozen alpha-zero routing signature."""

    keys = (
        "research_target_hash", "research_recipient_future_hash",
        "research_recipient_path_noise_hash", "research_initial_state_hash",
        "research_output_future_hash", "research_final_sampler_target_max_abs_error",
        "research_final_sampler_target_l2", "research_sigmas", "research_x0_sigmas",
        "research_x0_vision_hashes", "research_x0_action_hashes", "research_vision_shape",
        "research_future_frame_indices", "research_vision_coordinate_count",
        "research_future_mask_coordinate_count", "research_future_mask_index_hash",
        "research_requested_active_call_indices", "research_observed_active_call_indices",
        "research_clamped_call_indices", "research_inactive_call_indices",
        "research_requested_active_sigmas", "research_observed_active_sigmas",
        "research_model_input_future_clamp_errors",
        "research_returned_future_velocity_overwrite_errors",
        "research_maximum_action_input_error", "research_maximum_action_output_error",
        "research_action_input_errors", "research_action_output_errors",
        "research_inactive_wrapper_write_count",
    )
    missing = [key for key in keys if key not in metadata]
    if missing:
        raise ValueError(f"behavior signature missing fields: {missing}")
    payload = {"action": value, **{key: metadata[key] for key in keys}}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def null_paths(value: Any, prefix: str = "") -> tuple[set[str], set[str]]:
    nulls: set[str] = set()
    nonfinite: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            child_nulls, child_nonfinite = null_paths(child, child_prefix)
            nulls.update(child_nulls); nonfinite.update(child_nonfinite)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_nulls, child_nonfinite = null_paths(child, f"{prefix}[{index}]")
            nulls.update(child_nulls); nonfinite.update(child_nonfinite)
    elif value is None:
        nulls.add(prefix)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) \
            and not math.isfinite(float(value)):
        nonfinite.add(prefix)
    return nulls, nonfinite


def bool_mask_digest(shape: tuple[int, ...], frames: tuple[int, ...]) -> tuple[str, int]:
    mask = np.zeros(shape, dtype=np.bool_)
    index: list[Any] = [slice(None)] * len(shape)
    index[len(shape) - 3] = list(frames)
    mask[tuple(index)] = True
    flat = np.ascontiguousarray(mask.reshape(-1))
    digest = hashlib.sha256()
    digest.update(b"torch.bool")
    digest.update(np.asarray(flat.shape, dtype=np.int64).tobytes())
    digest.update(flat.view(np.uint8).tobytes())
    return digest.hexdigest(), int(mask.sum())


def metrics(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray,
            natives: Mapping[int, np.ndarray], donor_seed: int) -> dict[str, Any]:
    candidate = value.astype(np.float64).reshape(-1)
    base = recipient.astype(np.float64).reshape(-1)
    target = donor.astype(np.float64).reshape(-1)
    direction = target - base; displacement = candidate - base
    separation = float(np.linalg.norm(direction))
    if not math.isfinite(separation) or separation <= 1e-12:
        raise ValueError("native donor axis is degenerate")
    distance = float(np.linalg.norm(candidate - target))
    projection = float(np.dot(displacement, direction) / np.dot(direction, direction))
    displacement_norm = float(np.linalg.norm(displacement))
    cosine = float(np.dot(displacement, direction) / (displacement_norm * separation)) \
        if displacement_norm > 1e-12 else 0.0
    distances = {
        str(seed): float(np.linalg.norm(candidate - natives[seed].astype(np.float64).reshape(-1)))
        for seed in SEEDS
    }
    ordered = sorted(SEEDS, key=lambda seed: (distances[str(seed)], SEEDS.index(seed)))
    minimum = distances[str(ordered[0])]
    tied = [seed for seed in SEEDS if distances[str(seed)] == minimum]
    return {
        "native_donor_l2": separation,
        "l2_to_donor": distance,
        "distance_reduction_to_donor": float(1.0 - distance / separation),
        "normalized_projection": projection,
        "cosine_alignment": cosine,
        "orthogonal_residual_normalized": float(
            np.linalg.norm(displacement - projection * direction) / separation
        ),
        "nearest_native_seed": int(ordered[0]),
        "correct_donor_top1": bool(ordered[0] == donor_seed),
        "nearest_native_exact_tie": len(tied) > 1,
        "nearest_native_tie_count": len(tied),
        "nearest_native_tied_seeds": tied,
        "nearest_native_top_two_margin": float(distances[str(ordered[1])] - minimum),
        "distances_to_native_actions": distances,
    }


def ols(values: list[float]) -> float:
    if len(values) != 5 or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("OLS requires one finite value at each frozen alpha")
    return float(np.dot(np.asarray(ALPHAS) - 0.5, np.asarray(values, dtype=np.float64)) / 0.625)


def adjacent(values: Mapping[float, float]) -> dict[str, float]:
    return {
        f"{left:.2f}_to_{right:.2f}": float(values[right] - values[left])
        for left, right in zip(ALPHAS[:-1], ALPHAS[1:])
    }


def derive_state(report: dict[str, Any], unit: dict[str, Any], manifest: dict[str, Any],
                 manifest_hash: str, authorization: dict[str, Any],
                 authorization_sha: str) -> tuple[dict[str, Any], dict[str, int | float | str]]:
    unit_id = str(unit["unit_id"])
    exact = {
        "status": "complete", "admission": manifest["admission"],
        "manifest_id": manifest["manifest_id"], "manifest_sha256": manifest_hash,
        "unit_id": unit_id, "task": unit["task"], "episode_id": unit["episode_id"],
        "environment_seed": unit["environment_seed"], "phase": "middle",
        "request_count": 92, "wire_schema_validated_response_count": 92,
        "request_class_census": {
            "native": 4, "native_replay": 4, "none": 4, "self": 4,
            "self_replay": 4, "dose": 60, "midpoint_replay": 12,
        },
        "intervention_response_count": 84,
        "active_intervention_response_count": 80,
        "active_intervention_site_count": 320,
        "input_fingerprint_count": 1, "parameter_probe_hash_count": 1,
        "degenerate_native_action_axis_count": 0, "fixed_recipient_noise": True,
        "recipient_schedule_identity_count": 84,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise ValueError(f"{unit_id}: header/control {key} differs")
    if report.get("branch_seeds") != unit["branch_seeds"] \
            or report.get("branch_step") != unit["branch_step"] \
            or report.get("phase_fraction") != unit["phase_fraction"]:
        raise ValueError(f"{unit_id}: frozen state/branch binding differs")

    if report.get("authorization_audit_sha256") != authorization_sha \
            or not str(report.get("authorization_audit_path", "")).endswith(".json"):
        raise ValueError(f"{unit_id}: authorization report binding differs")
    powered = manifest["admission"] == PRIMARY_ADMISSION
    expected_authorization = {
        "status": "pass", "verdict": "GO" if powered else "GO_SMOKE",
        "scope": (
            "outcome_blind_prelaunch_audit"
            if powered else "outcome_blind_excluded_smoke_prelaunch_audit"
        ),
        "manifest_id": manifest["manifest_id"], "manifest_sha256": manifest_hash,
        "snapshot_checksum_list_sha256": manifest["runtime"]["snapshot_checksum_list_sha256"],
        "authorized_state_count": 30 if powered else 1,
        "authorized_call_count": 2760 if powered else 92,
    }
    if any(authorization.get(key) != value for key, value in expected_authorization.items()):
        raise ValueError(f"{unit_id}: authorization does not bind powered cohort")

    specs = request_specs(unit["branch_seeds"])
    labels = [str(spec["label"]) for spec in specs]
    if report.get("request_sequence") != labels or report.get("alpha_grid") != list(ALPHAS) \
            or report.get("ordered_pairs") != unit["ordered_pairs"]:
        raise ValueError(f"{unit_id}: request grid/order differs")
    actions = report.get("response_actions", {})
    metadata = report.get("response_metadata", {})
    if set(actions) != set(labels) or set(metadata) != set(labels) or len(actions) != 92:
        raise ValueError(f"{unit_id}: response census differs")
    parsed = {label: action(value, f"{unit_id}:{label}") for label, value in actions.items()}
    spec_by_label = {str(spec["label"]): spec for spec in specs}

    expected_probe = manifest["runtime"]["expected_parameter_probe_hash"]
    fingerprints = {row.get("research_state_hash") for row in metadata.values()}
    probes = {row.get("research_parameter_probe_hash") for row in metadata.values()}
    if None in fingerprints or report.get("input_fingerprints") != sorted(fingerprints) or len(fingerprints) != 1 \
            or report.get("parameter_probe_hashes") != [expected_probe] \
            or report.get("expected_parameter_probe_hash") != expected_probe or probes != {expected_probe}:
        raise ValueError(f"{unit_id}: singleton input/model fingerprint gate failed")

    null_count = finite_count = absent_count = 0
    for label, row in metadata.items():
        kind = str(spec_by_label[label]["kind"])
        nulls, nonfinite = null_paths(row)
        allowed = {"research_attention_interface.cache_id"}
        if kind in {"none", "self", "self_repeat"}:
            allowed.add("research_action_donor_projection")
        if kind in {"dose", "dose_midpoint_repeat"}:
            alpha = float(spec_by_label[label]["alpha"])
            if alpha == 0.0:
                allowed.update({"research_alpha_one_donor_future_max_abs_error",
                                "research_alpha_one_target_hash_matches_donor"})
            elif alpha == 1.0:
                allowed.update({"research_alpha_zero_recipient_future_max_abs_error",
                                "research_alpha_zero_target_hash_matches_recipient"})
            else:
                allowed.update({"research_alpha_zero_recipient_future_max_abs_error",
                                "research_alpha_one_donor_future_max_abs_error",
                                "research_alpha_zero_target_hash_matches_recipient",
                                "research_alpha_one_target_hash_matches_donor"})
        if nonfinite or nulls != allowed:
            raise ValueError(f"{unit_id}:{label}: null/nonfinite topology differs")
        present = "research_action_donor_projection" in row
        applicable = row.get("research_action_donor_projection_applicable")
        if kind in {"native", "native_repeat"}:
            if present or "research_action_donor_projection_applicable" in row:
                raise ValueError(f"{unit_id}:{label}: native projection must be absent")
            absent_count += 1
        elif kind in {"none", "self", "self_repeat"}:
            if not present or row["research_action_donor_projection"] is not None or applicable is not False:
                raise ValueError(f"{unit_id}:{label}: diagonal projection schema differs")
            null_count += 1
        else:
            if not present or applicable is not True:
                raise ValueError(f"{unit_id}:{label}: dose projection schema differs")
            finite(row["research_action_donor_projection"], f"{unit_id}:{label}: projection")
            finite_count += 1
    if (null_count, finite_count, absent_count) != (12, 72, 8):
        raise ValueError(f"{unit_id}: projection census {null_count}/{finite_count}/{absent_count}")
    expected_null_schema = {
        "research_action_donor_projection": (
            "null iff recipient and donor are identical in none/self controls; "
            "finite for all off-diagonal dose responses"
        ),
        "expected_null_count": 12,
        "expected_finite_count": 72,
        "expected_absent_native_count": 8,
    }
    if report.get("structural_null_schema") != expected_null_schema:
        raise ValueError(f"{unit_id}: projection structural-null declaration differs")

    natives = {seed: parsed[f"native-{seed}"] for seed in SEEDS}
    native_server = report.get("native_server", {})
    if set(native_server) != {str(seed) for seed in SEEDS}:
        raise ValueError(f"{unit_id}: native metadata grid differs")
    for seed in SEEDS:
        base = f"native-{seed}"; repeat = f"native-repeat-{seed}"
        expected_native_id = f"{manifest['manifest_id']}-{unit_id}-{base}"
        if metadata[base].get("research_id") != expected_native_id \
                or metadata[base].get("research_seed") != seed \
                or metadata[base].get("research_mode") != "native" \
                or metadata[repeat].get("research_id") \
                != f"{manifest['manifest_id']}-{unit_id}-{repeat}" \
                or metadata[repeat].get("research_seed") != seed \
                or metadata[repeat].get("research_mode") != "native":
            raise ValueError(f"{unit_id}:{seed}: native record identity differs")
        if not np.array_equal(parsed[base], parsed[repeat]) \
                or metadata_signature(metadata[base]) != metadata_signature(metadata[repeat]) \
                or native_server[str(seed)] != metadata[base]:
            raise ValueError(f"{unit_id}:{seed}: native replay/copy differs")
    if report.get("native_actions") is None \
            or set(report["native_actions"]) != {str(seed) for seed in SEEDS} \
            or any(
                not np.array_equal(
                    action(report["native_actions"][str(seed)], f"{unit_id}: native copy {seed}"),
                    natives[seed],
                )
                for seed in SEEDS
            ):
        raise ValueError(f"{unit_id}: native action copies differ")
    expected_future_hashes = {
        str(seed): metadata[f"native-{seed}"].get("research_future_hash") for seed in SEEDS
    }
    if report.get("native_future_hashes") != expected_future_hashes \
            or report.get("native_future_hashes_distinct") is not True \
            or len(set(expected_future_hashes.values())) != 4 \
            or None in expected_future_hashes.values():
        raise ValueError(f"{unit_id}: native future-hash identity/distinctness differs")
    gates = report.get("native_repeat_gates", {})
    expected_native_gate = {"action_max_abs_error": 0.0,
                            "deterministic_metadata_exact": True, "future_trace_exact": True}
    if set(gates) != {str(seed) for seed in SEEDS} \
            or any(value != expected_native_gate for value in gates.values()):
        raise ValueError(f"{unit_id}: native replay gates differ")

    none_controls = report.get("none_controls", [])
    self_controls = report.get("self_controls", [])
    alpha_zero = report.get("alpha_zero_routing_controls", [])
    midpoint = report.get("midpoint_replay_controls", [])
    if len(none_controls) != 4 or any(
        row.get("action_max_abs_error_vs_native") != 0.0
        or row.get("native_trace_exact") is not True
        or row.get("projection_structural_null") is not True for row in none_controls
    ):
        raise ValueError(f"{unit_id}: none gate differs")
    if [int(row.get("recipient_seed", -1)) for row in none_controls] != list(SEEDS):
        raise ValueError(f"{unit_id}: none-control order differs")
    if len(self_controls) != 4 or any(
        row.get("repeat_action_max_abs_error") != 0.0
        or row.get("repeat_signature_exact") is not True
        or row.get("projection_structural_null") is not True for row in self_controls
    ):
        raise ValueError(f"{unit_id}: self gate differs")
    if [int(row.get("recipient_seed", -1)) for row in self_controls] != list(SEEDS):
        raise ValueError(f"{unit_id}: self-control order differs")
    if len(alpha_zero) != 12 or any(
        row.get("action_max_abs_error_vs_self") != 0.0
        or row.get("behavior_signature_exact_vs_self") is not True for row in alpha_zero
    ):
        raise ValueError(f"{unit_id}: alpha-zero routing gate differs")
    if len(midpoint) != 12 or any(
        row.get("alpha") != 0.5 or row.get("action_max_abs_error") != 0.0
        or row.get("deterministic_metadata_exact") is not True for row in midpoint
    ):
        raise ValueError(f"{unit_id}: midpoint replay gate differs")
    expected_pairs = [tuple(int(value) for value in pair) for pair in unit["ordered_pairs"]]
    if [(int(row.get("recipient_seed", -1)), int(row.get("donor_seed", -1)))
            for row in alpha_zero] != expected_pairs \
            or [(int(row.get("recipient_seed", -1)), int(row.get("donor_seed", -1)))
                for row in midpoint] != expected_pairs:
        raise ValueError(f"{unit_id}: alpha-zero/midpoint control order differs")
    for seed in SEEDS:
        if not np.array_equal(parsed[f"none-{seed}"], natives[seed]) \
                or not np.array_equal(parsed[f"self-{seed}"], parsed[f"self-repeat-{seed}"]):
            raise ValueError(f"{unit_id}:{seed}: none/self action control differs")
        none_row = none_controls[SEEDS.index(seed)]
        none_metadata = metadata[f"none-{seed}"]
        native_metadata = metadata[f"native-{seed}"]
        if none_row.get("server") != none_metadata \
                or none_metadata.get("research_output_future_hash") != native_metadata.get("research_future_hash") \
                or any(
                    none_metadata.get(key) != native_metadata.get(key)
                    for key in ("research_x0_vision_hashes", "research_x0_action_hashes",
                                "research_sigmas", "research_x0_sigmas")
                ):
            raise ValueError(f"{unit_id}:{seed}: none raw no-op trace differs")
        self_row = self_controls[SEEDS.index(seed)]
        self_action = parsed[f"self-{seed}"]
        self_metadata = metadata[f"self-{seed}"]
        if self_row.get("server") != self_metadata \
                or self_row.get("action") != actions[f"self-{seed}"] \
                or not math.isclose(
                    finite(self_row.get("clean_clamp_vs_native_max_abs_error"),
                           f"{unit_id}:{seed}: clean-clamp max"),
                    float(np.max(np.abs(self_action.astype(np.float64) - natives[seed].astype(np.float64)))),
                    rel_tol=1e-12, abs_tol=1e-12,
                ) \
                or not math.isclose(
                    finite(self_row.get("clean_clamp_vs_native_l2"),
                           f"{unit_id}:{seed}: clean-clamp l2"),
                    float(np.linalg.norm(self_action.astype(np.float64) - natives[seed].astype(np.float64))),
                    rel_tol=1e-12, abs_tol=1e-12,
                ):
            raise ValueError(f"{unit_id}:{seed}: self raw replay/diagnostic differs")
        for donor in SEEDS:
            if donor != seed and not np.array_equal(
                parsed[f"self-{seed}"], parsed[dose_label(seed, donor, 0.0)]
            ):
                raise ValueError(f"{unit_id}:{seed}:{donor}: alpha-zero action differs")
    for recipient, donor in ((int(a), int(b)) for a, b in unit["ordered_pairs"]):
        if not np.array_equal(parsed[dose_label(recipient, donor, 0.5)],
                              parsed[dose_label(recipient, donor, 0.5) + "-repeat"]) \
                or metadata_signature(metadata[dose_label(recipient, donor, 0.5)]) \
                != metadata_signature(metadata[dose_label(recipient, donor, 0.5) + "-repeat"]):
            raise ValueError(f"{unit_id}:{recipient}:{donor}: midpoint replay differs")
    for seed in SEEDS:
        self_label = f"self-{seed}"
        expected_signature = behavior_signature(actions[self_label], metadata[self_label])
        signatures: set[str] = set()
        for row in alpha_zero:
            if int(row["recipient_seed"]) != seed:
                continue
            label = dose_label(seed, int(row["donor_seed"]), 0.0)
            actual_signature = behavior_signature(actions[label], metadata[label])
            if actual_signature != expected_signature or row.get("behavior_signature") != actual_signature:
                raise ValueError(f"{unit_id}:{label}: alpha-zero behavior signature differs")
            signatures.add(actual_signature)
        if len(signatures) != 1:
            raise ValueError(f"{unit_id}:{seed}: alpha-zero donor-label invariance differs")

    intervention_labels = {
        label for label in labels if spec_by_label[label]["kind"]
        not in {"native", "native_repeat"}
    }
    audits = report.get("intervention_site_audits", {})
    action_errors = report.get("action_coordinate_errors", {})
    if set(audits) != intervention_labels or set(action_errors) != intervention_labels \
            or any(value != {"input": 0.0, "output": 0.0} for value in action_errors.values()):
        raise ValueError(f"{unit_id}: intervention/nonwrite audit census differs")
    tolerance = float(manifest["runtime"]["intervention_site_error_tolerance"])
    active_sites = 0
    site_input_maxima: list[float] = []
    site_velocity_maxima: list[float] = []
    recipient_paths: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    recipient_initial: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    for label, audit in audits.items():
        raw = metadata[label]; kind = str(spec_by_label[label]["kind"])
        mode = "none" if kind == "none" else "self" if kind in {"self", "self_repeat"} else "dose"
        if audit.get("mode") != mode:
            raise ValueError(f"{unit_id}:{label}: mode differs")
        active = mode != "none"; active_indices = [0, 1, 2, 3] if active else []
        inactive_indices = [] if active else [0, 1, 2, 3]
        sigmas = [finite(value, f"{unit_id}:{label}: sigma") for value in audit.get("sigmas", [])]
        if len(sigmas) != 4 or audit.get("requested_active_call_indices") != active_indices \
                or audit.get("observed_active_call_indices") != active_indices \
                or audit.get("clamped_call_indices") != active_indices \
                or audit.get("inactive_call_indices") != inactive_indices \
                or audit.get("requested_active_sigmas") != [sigmas[index] for index in active_indices] \
                or audit.get("observed_active_sigmas") != [sigmas[index] for index in active_indices] \
                or int(audit.get("active_site_count", -1)) != len(active_indices):
            raise ValueError(f"{unit_id}:{label}: schedule/site audit differs")
        active_sites += len(active_indices)
        site_error_vectors: dict[str, list[float]] = {}
        for key in ("model_input_future_clamp_errors", "returned_future_velocity_overwrite_errors"):
            values = [finite(value, f"{unit_id}:{label}:{key}") for value in audit.get(key, [])]
            if len(values) != len(active_indices) or any(value > tolerance for value in values):
                raise ValueError(f"{unit_id}:{label}: {key} failed")
            site_error_vectors[key] = values
        model_input_max = finite(audit.get("model_input_max_error"),
                                 f"{unit_id}:{label}: model input maximum")
        velocity_max = finite(audit.get("returned_velocity_max_error"),
                              f"{unit_id}:{label}: velocity maximum")
        if model_input_max != max(site_error_vectors["model_input_future_clamp_errors"], default=0.0) \
                or velocity_max != max(
                    site_error_vectors["returned_future_velocity_overwrite_errors"], default=0.0
                ):
            raise ValueError(f"{unit_id}:{label}: site-error maxima do not match raw vectors")
        site_input_maxima.append(model_input_max); site_velocity_maxima.append(velocity_max)
        for key in ("action_input_errors", "action_output_errors"):
            values = [finite(value, f"{unit_id}:{label}:{key}") for value in audit.get(key, [])]
            if len(values) != 4 or any(value != 0.0 for value in values):
                raise ValueError(f"{unit_id}:{label}: {key} failed")
        if audit.get("inactive_wrapper_write_count") != 0 \
                or audit.get("maximum_action_input_error") != 0.0 \
                or audit.get("maximum_action_output_error") != 0.0:
            raise ValueError(f"{unit_id}:{label}: action/inactive write observed")
        shape = tuple(int(value) for value in audit.get("vision_shape", []))
        frames = tuple(int(value) for value in audit.get("future_frame_indices", []))
        if len(shape) not in {4, 5} or frames != tuple(range(1, 9)):
            raise ValueError(f"{unit_id}:{label}: vision/frame schema differs")
        mask_hash, mask_count = bool_mask_digest(shape, frames)
        if audit.get("vision_coordinate_count") != int(np.prod(shape)) \
                or audit.get("target_coordinate_count") != int(np.prod(shape)) \
                or audit.get("target_finite_coordinate_count") != int(np.prod(shape)) \
                or audit.get("mask_coordinate_count") != mask_count \
                or audit.get("mask_coordinate_count") \
                != raw.get("research_future_mask_coordinate_count") \
                or audit.get("future_mask_index_hash") != mask_hash:
            raise ValueError(f"{unit_id}:{label}: coordinate/mask audit differs")
        direct = (
            "mode", "target_source", "target_hash", "target_source_record_ids",
            "recipient_future_hash", "donor_future_hash", "recipient_path_noise_hash",
            "initial_state_hash", "sigmas", "requested_active_call_indices",
            "observed_active_call_indices", "clamped_call_indices", "inactive_call_indices",
            "requested_active_sigmas", "observed_active_sigmas", "future_frame_indices",
            "vision_shape", "future_mask_index_hash", "model_input_future_clamp_errors",
            "returned_future_velocity_overwrite_errors", "final_sampler_target_max_abs_error",
            "final_sampler_target_l2", "vision_coordinate_count", "target_coordinate_count",
            "target_finite_coordinate_count", "maximum_action_input_error",
            "maximum_action_output_error", "action_input_errors", "action_output_errors",
            "inactive_wrapper_write_count", "alpha", "interpolation_formula_max_abs_error",
            "nonfuture_recipient_target_max_abs_error", "alpha_zero_recipient_future_max_abs_error",
            "alpha_one_donor_future_max_abs_error", "alpha_zero_target_hash_matches_recipient",
            "alpha_one_target_hash_matches_donor", "interpolated_future_hash",
            "recipient_future_mask_hash", "donor_future_mask_hash", "current_frame_hash",
            "recipient_current_frame_hash", "future_mask_hash",
        )
        if any(audit.get(key) != raw.get(f"research_{key}") for key in direct):
            raise ValueError(f"{unit_id}:{label}: retained/raw audit copies differ")
        recipient = int(spec_by_label[label]["recipient_seed"])
        donor = int(spec_by_label[label].get("donor_seed", recipient))
        recipient_native = metadata[f"native-{recipient}"]
        donor_native = metadata[f"native-{donor}"]
        if raw.get("research_id") != f"{manifest['manifest_id']}-{unit_id}-{label}" \
                or raw.get("research_seed") != recipient \
                or audit.get("recipient_id") != recipient_native.get("research_id") \
                or audit.get("donor_id") != donor_native.get("research_id") \
                or raw.get("research_recipient_id") != recipient_native.get("research_id") \
                or raw.get("research_donor_id") != donor_native.get("research_id"):
            raise ValueError(f"{unit_id}:{label}: recipient/donor record identity differs")
        if raw.get("research_sigmas") != recipient_native.get("research_sigmas") \
                or raw.get("research_x0_sigmas") != recipient_native.get("research_x0_sigmas"):
            raise ValueError(f"{unit_id}:{label}: recipient denoising schedule differs")
        expected_source = "recipient" if mode in {"none", "self"} else "recipient_donor_linear_interpolation"
        expected_ids = [audit.get("recipient_id")] if expected_source == "recipient" \
            else [audit.get("recipient_id"), audit.get("donor_id")]
        if audit.get("target_source") != expected_source or audit.get("target_source_record_ids") != expected_ids \
                or audit.get("recipient_future_hash") != recipient_native.get("research_future_hash") \
                or audit.get("donor_future_hash") != donor_native.get("research_future_hash") \
                or audit.get("recipient_path_noise_hash") != recipient_native.get("research_path_noise_hash") \
                or audit.get("initial_state_hash") != recipient_native.get("research_initial_state_hash"):
            raise ValueError(f"{unit_id}:{label}: source/RNG provenance differs")
        recipient_paths[recipient].add(str(audit["recipient_path_noise_hash"]))
        recipient_initial[recipient].add(str(audit["initial_state_hash"]))
        if mode in {"none", "self"} and audit.get("target_hash") != recipient_native.get("research_future_hash"):
            raise ValueError(f"{unit_id}:{label}: recipient target hash differs")
        if mode == "dose":
            alpha = float(spec_by_label[label]["alpha"])
            if audit.get("alpha") != alpha or raw.get("research_alpha_grid") != list(ALPHAS) \
                    or raw.get("research_interpolation_formula") != "F_A + alpha * (F_B - F_A)" \
                    or audit.get("interpolation_formula_max_abs_error") != 0.0 \
                    or audit.get("nonfuture_recipient_target_max_abs_error") != 0.0 \
                    or audit.get("current_frame_hash") != audit.get("recipient_current_frame_hash") \
                    or audit.get("future_mask_hash") != audit.get("future_mask_index_hash"):
                raise ValueError(f"{unit_id}:{label}: interpolation audit differs")
            if alpha == 0.0 and not (
                audit.get("alpha_zero_recipient_future_max_abs_error") == 0.0
                and audit.get("alpha_zero_target_hash_matches_recipient") is True
                and audit.get("target_hash") == recipient_native.get("research_future_hash")
                and audit.get("interpolated_future_hash") == audit.get("recipient_future_mask_hash")
            ):
                raise ValueError(f"{unit_id}:{label}: alpha-zero identity differs")
            if alpha == 1.0 and not (
                audit.get("alpha_one_donor_future_max_abs_error") == 0.0
                and audit.get("alpha_one_target_hash_matches_donor") is True
                and audit.get("target_hash") == donor_native.get("research_future_hash")
                and audit.get("interpolated_future_hash") == audit.get("donor_future_mask_hash")
            ):
                raise ValueError(f"{unit_id}:{label}: alpha-one identity differs")
        interface = raw.get("research_attention_interface", {})
        if interface.get("instrumented_server") is not False \
                or interface.get("intervention_requested") is not False \
                or interface.get("mode") != "exclude":
            raise ValueError(f"{unit_id}:{label}: attention metadata differs")
    if active_sites != 320 or any(len(values) != 1 for values in recipient_paths.values()) \
            or any(len(values) != 1 for values in recipient_initial.values()):
        raise ValueError(f"{unit_id}: active-site/RNG identity census differs")
    if report.get("model_input_future_clamp_max_error") != max(site_input_maxima) \
            or report.get("returned_future_velocity_overwrite_max_error") != max(site_velocity_maxima) \
            or max(site_input_maxima) > tolerance or max(site_velocity_maxima) > tolerance:
        raise ValueError(f"{unit_id}: aggregate intervention-site error differs")

    native_pair_l2 = {
        f"{int(recipient)}:{int(donor)}": float(np.linalg.norm(
            natives[int(donor)].astype(np.float64).reshape(-1)
            - natives[int(recipient)].astype(np.float64).reshape(-1)
        )) for recipient, donor in unit["ordered_pairs"]
    }
    reported_native_pair_l2 = report.get("native_action_pair_l2")
    pair_axes_match = isinstance(reported_native_pair_l2, dict) \
        and set(reported_native_pair_l2) == set(native_pair_l2) \
        and all(
            math.isclose(
                finite(reported_native_pair_l2[key], f"{unit_id}: native pair {key}"),
                value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key, value in native_pair_l2.items()
        )
    if not pair_axes_match or any(value <= 1e-12 for value in native_pair_l2.values()):
        raise ValueError(f"{unit_id}: native action axes differ")

    dose_rows = report.get("dose_rows", [])
    expected_cells = [(int(recipient), int(donor), alpha)
                      for recipient, donor in unit["ordered_pairs"] for alpha in ALPHAS]
    actual_cells = [(int(row.get("recipient_seed", -1)), int(row.get("donor_seed", -1)),
                     float(row.get("alpha", -1))) for row in dose_rows]
    if len(dose_rows) != 60 or actual_cells != expected_cells or len(set(actual_cells)) != 60:
        raise ValueError(f"{unit_id}: dose grid/order differs")
    validated: list[dict[str, Any]] = []
    for row in dose_rows:
        recipient = int(row["recipient_seed"]); donor = int(row["donor_seed"]); alpha = float(row["alpha"])
        label = dose_label(recipient, donor, alpha)
        value = action(row["action"], f"{unit_id}:{label}: row action")
        if not np.array_equal(value, parsed[label]):
            raise ValueError(f"{unit_id}:{label}: action copy differs")
        recomputed = metrics(value, natives[recipient], natives[donor], natives, donor)
        for key in (
            "native_donor_l2", "l2_to_donor", "distance_reduction_to_donor",
            "normalized_projection", "cosine_alignment", "orthogonal_residual_normalized",
            "nearest_native_top_two_margin",
        ):
            if not math.isclose(
                finite(row.get(key), f"{unit_id}:{label}:{key}"),
                finite(recomputed[key], f"{unit_id}:{label}: recomputed {key}"),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{unit_id}:{label}: {key} differs")
        for key in ("nearest_native_seed", "correct_donor_top1", "nearest_native_exact_tie",
                    "nearest_native_tie_count", "nearest_native_tied_seeds"):
            if row.get(key) != recomputed[key]:
                raise ValueError(f"{unit_id}:{label}: {key} differs")
        reported_distances = row.get("distances_to_native_actions")
        recomputed_distances = recomputed["distances_to_native_actions"]
        if not isinstance(reported_distances, dict) \
                or set(reported_distances) != set(recomputed_distances) \
                or any(
                    not math.isclose(
                        finite(reported_distances[key], f"{unit_id}:{label}: native distance {key}"),
                        value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for key, value in recomputed_distances.items()
                ):
            raise ValueError(f"{unit_id}:{label}: distances_to_native_actions differs")
        raw = metadata[label]
        server_projection = finite(raw.get("research_action_donor_projection"),
                                   f"{unit_id}:{label}: server projection")
        direction = natives[donor].astype(np.float64) - natives[recipient].astype(np.float64)
        server_recomputed = float(
            ((value - natives[recipient]) * direction).sum() / np.square(direction).sum()
        )
        if server_projection != server_recomputed \
                or finite(row.get("server_action_donor_projection"), f"{unit_id}:{label}: stored") != server_projection \
                or row.get("target_hash") != raw.get("research_target_hash") \
                or row.get("final_sampler_target_max_abs_error") != raw.get("research_final_sampler_target_max_abs_error") \
                or row.get("final_sampler_target_l2") != raw.get("research_final_sampler_target_l2"):
            raise ValueError(f"{unit_id}:{label}: raw server/copy diagnostic differs")
        canonical_row = dict(row)
        canonical_row.update(recomputed)
        validated.append(canonical_row)

    by_pair: dict[tuple[int, int], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in validated:
        by_pair[(int(row["recipient_seed"]), int(row["donor_seed"]))][float(row["alpha"])] = row
    alpha_distance = {alpha: float(np.mean([
        finite(grid[alpha]["distance_reduction_to_donor"], "distance") for grid in by_pair.values()
    ])) for alpha in ALPHAS}
    alpha_projection = {alpha: float(np.mean([
        finite(grid[alpha]["normalized_projection"], "projection") for grid in by_pair.values()
    ])) for alpha in ALPHAS}
    alpha_identification = {alpha: float(np.mean([
        bool(grid[alpha]["correct_donor_top1"]) for grid in by_pair.values()
    ])) for alpha in ALPHAS}
    alpha_cosine = {alpha: float(np.mean([
        finite(grid[alpha]["cosine_alignment"], "cosine") for grid in by_pair.values()
    ])) for alpha in ALPHAS}
    alpha_orthogonal = {alpha: float(np.mean([
        finite(grid[alpha]["orthogonal_residual_normalized"], "orthogonal") for grid in by_pair.values()
    ])) for alpha in ALPHAS}
    distance_slopes: list[float] = []; projection_slopes: list[float] = []
    nondecreasing: list[bool] = []; profiles: dict[str, dict[str, Any]] = {}
    for pair, grid in by_pair.items():
        distance_profile = {alpha: finite(grid[alpha]["distance_reduction_to_donor"], "distance") for alpha in ALPHAS}
        projection_profile = {alpha: finite(grid[alpha]["normalized_projection"], "projection") for alpha in ALPHAS}
        l2_profile = {alpha: finite(grid[alpha]["l2_to_donor"], "l2") for alpha in ALPHAS}
        distance_slope = ols([distance_profile[alpha] for alpha in ALPHAS])
        projection_slope = ols([projection_profile[alpha] for alpha in ALPHAS])
        monotone = all(l2_profile[right] <= l2_profile[left]
                       for left, right in zip(ALPHAS[:-1], ALPHAS[1:]))
        distance_slopes.append(distance_slope); projection_slopes.append(projection_slope)
        nondecreasing.append(monotone)
        profiles[f"{pair[0]}:{pair[1]}"] = {
            "recipient_seed": pair[0], "donor_seed": pair[1],
            "distance_reduction_slope": distance_slope, "projection_slope": projection_slope,
            "distance_reduction_by_alpha": {str(alpha): distance_profile[alpha] for alpha in ALPHAS},
            "projection_by_alpha": {str(alpha): projection_profile[alpha] for alpha in ALPHAS},
            "l2_to_donor_by_alpha": {str(alpha): l2_profile[alpha] for alpha in ALPHAS},
            "correct_donor_top1_by_alpha": {
                str(alpha): bool(grid[alpha]["correct_donor_top1"]) for alpha in ALPHAS
            },
            "donor_proximity_nondecreasing": monotone,
        }
    distance_slope = float(np.mean(distance_slopes)); projection_slope = float(np.mean(projection_slopes))
    distance_profile_slope = ols([alpha_distance[alpha] for alpha in ALPHAS])
    projection_profile_slope = ols([alpha_projection[alpha] for alpha in ALPHAS])
    state = {
        "unit_id": unit_id, "task": unit["task"], "episode_id": unit["episode_id"],
        "environment_seed": unit["environment_seed"],
        "distance_reduction_slope": distance_slope,
        "distance_reduction_profile_slope": distance_profile_slope,
        "distance_slope_equivalence_abs_error": abs(distance_slope - distance_profile_slope),
        "projection_slope": projection_slope,
        "projection_profile_slope": projection_profile_slope,
        "projection_slope_equivalence_abs_error": abs(projection_slope - projection_profile_slope),
        "distance_reduction_endpoint_contrast": float(alpha_distance[1.0] - alpha_distance[0.0]),
        "projection_endpoint_contrast": float(alpha_projection[1.0] - alpha_projection[0.0]),
        "distance_reduction_by_alpha": {str(alpha): alpha_distance[alpha] for alpha in ALPHAS},
        "projection_by_alpha": {str(alpha): alpha_projection[alpha] for alpha in ALPHAS},
        "donor_identification_by_alpha": {str(alpha): alpha_identification[alpha] for alpha in ALPHAS},
        "cosine_alignment_by_alpha": {str(alpha): alpha_cosine[alpha] for alpha in ALPHAS},
        "orthogonal_residual_by_alpha": {str(alpha): alpha_orthogonal[alpha] for alpha in ALPHAS},
        "distance_reduction_adjacent_contrasts": adjacent(alpha_distance),
        "projection_adjacent_contrasts": adjacent(alpha_projection),
        "nondecreasing_pair_fraction": float(np.mean(nondecreasing)),
        "ordered_pair_profiles": profiles,
    }
    residual_max_map = report.get("final_sampler_target_max_abs_errors", {})
    residual_l2_map = report.get("final_sampler_target_l2_errors", {})
    expected_residual_labels = {
        label for label, spec in spec_by_label.items()
        if spec["kind"] in {"self", "self_repeat", "dose", "dose_midpoint_repeat"}
    }
    if not isinstance(residual_max_map, dict) or not isinstance(residual_l2_map, dict) \
            or set(residual_max_map) != expected_residual_labels \
            or set(residual_l2_map) != expected_residual_labels:
        raise ValueError(f"{unit_id}: descriptive residual census differs")
    for label in expected_residual_labels:
        if residual_max_map[label] != metadata[label].get("research_final_sampler_target_max_abs_error") \
                or residual_l2_map[label] != metadata[label].get("research_final_sampler_target_l2"):
            raise ValueError(f"{unit_id}:{label}: descriptive residual copy differs")
    residual_max = [finite(value, f"{unit_id}: residual max") for value in residual_max_map.values()]
    residual_l2 = [finite(value, f"{unit_id}: residual l2") for value in residual_l2_map.values()]
    audit_counts: dict[str, int | float | str] = {
        "authorization_sha256": authorization_sha,
        "shape_valid_actions": 92, "intervention_responses": 84,
        "active_responses": 80, "active_sites": 320,
        "projection_null": null_count, "projection_finite": finite_count,
        "projection_absent": absent_count,
        "site_input_max_error": max(site_input_maxima),
        "site_velocity_max_error": max(site_velocity_maxima),
    }
    state["_residual_max"] = residual_max
    state["_residual_l2"] = residual_l2
    return state, audit_counts


def draw_table(states: list[dict[str, Any]]) -> tuple[tuple[str, ...], dict[str, tuple[dict[str, Any], ...]], np.ndarray, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        grouped[str(state["task"])].append(state)
    tasks = tuple(sorted(grouped))
    if len(tasks) != 6 or any(len(grouped[task]) != 5 for task in tasks):
        raise ValueError("bootstrap requires exactly six tasks by five states")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    task_indices = np.empty((BOOTSTRAPS, 6), dtype=np.int64)
    state_indices = np.empty((BOOTSTRAPS, 6, 5), dtype=np.int64)
    for draw in range(BOOTSTRAPS):
        task_indices[draw] = rng.integers(0, 6, size=6)
        for occurrence in range(6):
            state_indices[draw, occurrence] = rng.integers(0, 5, size=5)
    return tasks, {task: tuple(grouped[task]) for task in tasks}, task_indices, state_indices


def estimate(table: tuple[tuple[str, ...], dict[str, tuple[dict[str, Any], ...]], np.ndarray, np.ndarray],
             getter: Callable[[dict[str, Any]], float]) -> dict[str, float]:
    tasks, grouped, task_indices, state_indices = table
    point = float(np.mean([np.mean([getter(row) for row in grouped[task]]) for task in tasks]))
    draws = np.empty(BOOTSTRAPS, dtype=np.float64)
    for draw in range(BOOTSTRAPS):
        occurrence_means = []
        for occurrence, task_index in enumerate(task_indices[draw]):
            rows = grouped[tasks[int(task_index)]]
            occurrence_means.append(float(np.mean([
                getter(rows[int(index)]) for index in state_indices[draw, occurrence]
            ])))
        draws[draw] = float(np.mean(occurrence_means))
    return {
        "estimate": point,
        "lower": float(np.quantile(draws, 0.025, method="linear")),
        "upper": float(np.quantile(draws, 0.975, method="linear")),
    }


def build_summary(states: list[dict[str, Any]], manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
    residual_max = [value for state in states for value in state.pop("_residual_max")]
    residual_l2 = [value for state in states for value in state.pop("_residual_l2")]
    table = draw_table(states)
    get = lambda key: (lambda row: finite(row[key], key))
    primary = estimate(table, get("distance_reduction_slope"))
    distance_endpoint = estimate(table, get("distance_reduction_endpoint_contrast"))
    projection_slope = estimate(table, get("projection_slope"))
    projection_endpoint = estimate(table, get("projection_endpoint_contrast"))
    monotonic_fraction = estimate(table, get("nondecreasing_pair_fraction"))
    contrast_names = ("0.00_to_0.25", "0.25_to_0.50", "0.50_to_0.75", "0.75_to_1.00")
    distance_adjacent = {name: estimate(table, lambda row, name=name: finite(
        row["distance_reduction_adjacent_contrasts"][name], name
    )) for name in contrast_names}
    by_alpha: dict[str, dict[str, dict[str, float]]] = {
        name: {} for name in ("distance", "projection", "identification", "cosine", "orthogonal")
    }
    mappings = {
        "distance": "distance_reduction_by_alpha", "projection": "projection_by_alpha",
        "identification": "donor_identification_by_alpha", "cosine": "cosine_alignment_by_alpha",
        "orthogonal": "orthogonal_residual_by_alpha",
    }
    for alpha in ALPHAS:
        token = str(alpha)
        for name, key in mappings.items():
            by_alpha[name][token] = estimate(
                table, lambda row, token=token, key=key: finite(row[key][token], token)
            )
    extractors: dict[str, Callable[[dict[str, Any]], float]] = {
        "distance_reduction_slope": get("distance_reduction_slope"),
        "distance_reduction_endpoint_contrast": get("distance_reduction_endpoint_contrast"),
        "projection_slope": get("projection_slope"),
        "projection_endpoint_contrast": get("projection_endpoint_contrast"),
        "nondecreasing_pair_fraction": get("nondecreasing_pair_fraction"),
    }
    for alpha in ALPHAS:
        token = str(alpha)
        for prefix, key in (
            ("distance_reduction", "distance_reduction_by_alpha"),
            ("projection", "projection_by_alpha"),
            ("donor_identification", "donor_identification_by_alpha"),
            ("cosine", "cosine_alignment_by_alpha"),
            ("orthogonal", "orthogonal_residual_by_alpha"),
        ):
            extractors[f"{prefix}_alpha_{token}"] = (
                lambda row, token=token, key=key: finite(row[key][token], token)
            )
    for name in contrast_names:
        extractors[f"distance_adjacent_{name}"] = (
            lambda row, name=name: finite(row["distance_reduction_adjacent_contrasts"][name], name)
        )
    tasks = sorted({str(state["task"]) for state in states})
    per_task: dict[str, dict[str, float]] = {}
    loto: dict[str, dict[str, float]] = {}
    for name, extractor in extractors.items():
        task_means = {task: float(np.mean([extractor(state) for state in states if state["task"] == task]))
                      for task in tasks}
        per_task[name] = task_means
        loto[name] = {held: float(np.mean([value for task, value in task_means.items() if task != held]))
                      for held in tasks}
    pooled = float(np.mean([finite(state["distance_reduction_slope"], "slope") for state in states]))
    pooled_error = abs(pooled - primary["estimate"])
    def residual_summary(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values), "maximum": float(array.max()),
            "quantiles": {str(q): float(np.quantile(array, q)) for q in (0.5, 0.9, 0.95, 0.99)},
            "count_gt_0_03": int(np.count_nonzero(array > 0.03)),
        }
    adjacent_lower = all(value["lower"] > 0.0 for value in distance_adjacent.values())
    adjacent_point = all(value["estimate"] >= 0.0 for value in distance_adjacent.values())
    return {
        "status": "complete", "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_hash,
        "analyzer_sha256": manifest["runtime"]["analyzer_sha256"],
        "state_count": 30, "expected_state_count": 30, "missing_state_count": 0,
        "extra_state_count": 0, "exclusion_count": 0, "call_count": 2760,
        "shape_valid_response_action_count": 2760,
        "released_action_coordinate_census": 2760 * 256,
        "action_shape": [32, 8], "action_coordinate_count": 256,
        "primary_distance_reduction_slope": primary,
        "primary_criterion_pass": primary["lower"] > 0.0,
        "permitted_wording": {
            "positive_linear_dose_trend_in_donor_directed_action_distance_under_imposed_all_call_future_interpolation": primary["lower"] > 0.0,
            "task_weighted_mean_profile_increased_strictly_at_every_adjacent_alpha_step": adjacent_lower,
            "numerically_nondecreasing_sample_mean_profile": adjacent_point,
            "monotonic": False,
        },
        "distance_reduction_endpoint_contrast": distance_endpoint,
        "distance_reduction_adjacent_contrasts": distance_adjacent,
        "projection_slope": projection_slope,
        "projection_endpoint_contrast": projection_endpoint,
        "distance_reduction_by_alpha": by_alpha["distance"],
        "projection_by_alpha": by_alpha["projection"],
        "donor_identification_by_alpha": by_alpha["identification"],
        "cosine_alignment_by_alpha": by_alpha["cosine"],
        "orthogonal_residual_by_alpha": by_alpha["orthogonal"],
        "nondecreasing_pair_fraction": monotonic_fraction,
        "pooled_30_state_primary_point_sensitivity": pooled,
        "pooled_equal_task_primary_point_abs_error": pooled_error,
        "pooled_equals_equal_task_point_within_1e_15": pooled_error <= 1e-15,
        "per_task_primary": per_task["distance_reduction_slope"],
        "leave_one_task_out_primary": loto["distance_reduction_slope"],
        "per_task_all_estimands": per_task,
        "leave_one_task_out_all_estimands": loto,
        "final_sampler_target_residual": {
            "max_abs": residual_summary(residual_max), "l2": residual_summary(residual_l2),
            "role": "descriptive_only_not_an_admission_or_evidence_criterion",
        },
        "bootstrap": {
            "samples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED,
            "hierarchy": "task -> episode/state", "bit_generator": "PCG64",
            "shared_draw_table_reused_for_every_estimand": True,
            "quantile_method": "linear",
        },
        "state_rows": states,
    }


def main() -> None:
    args = parse_args()
    manifest, paths, _inventory = verify_frozen_package(
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        output_root=args.run_root,
        inventory_path=args.inventory,
        expected_inventory_sha256=args.expected_inventory_sha256,
        expected_count=30,
        expected_inventory_schema="cosmos3-dose-output-inventory-v1",
    )
    if manifest.get("study_name") != "cosmos3-future-strength-dose-response-v2" \
            or manifest.get("admission") != PRIMARY_ADMISSION \
            or manifest.get("freeze_stage") != "evaluation_ready" \
            or manifest.get("launch_authorization") != "powered_evaluation_after_independent_go" \
            or manifest.get("design", {}).get("request_count_per_state") != 92 \
            or manifest.get("design", {}).get("released_action_shape") != [32, 8] \
            or manifest.get("design", {}).get("released_action_coordinate_count") != 256 \
            or manifest.get("analysis", {}).get("bootstrap_samples") != BOOTSTRAPS \
            or manifest.get("analysis", {}).get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise ValueError("manifest design/analysis constants differ from dose v2")
    require_sha(
        args.authorization_report,
        args.expected_authorization_sha256,
        "independent powered authorization",
    )
    authorization = load_json(args.authorization_report)
    unit_by_name = {f"{unit['unit_id']}.json": unit for unit in manifest["states"]}
    states: list[dict[str, Any]] = []
    counters: list[dict[str, int | float | str]] = []
    for path in paths:
        state, audit = derive_state(
            load_json(path), unit_by_name[path.name], manifest,
            args.expected_manifest_sha256, authorization,
            args.expected_authorization_sha256,
        )
        states.append(state); counters.append(audit)
    # Preserve manifest state order rather than filesystem order for byte-level comparison.
    by_id = {str(state["unit_id"]): state for state in states}
    states = [by_id[str(unit["unit_id"])] for unit in manifest["states"]]
    computed = build_summary(states, manifest, args.expected_manifest_sha256)
    reference = load_reference_after_hash(args.summary_json, args.expected_summary_sha256)
    problems: list[str] = []
    if set(computed) != set(reference):
        problems.append("summary: top-level key topology differs")
    problems.extend(compare_tree(computed, reference, path="summary", atol=1e-12, rtol=1e-12))
    if problems:
        raise ValueError("independent dose audit differs from frozen analysis: " + "; ".join(problems[:20]))
    authorization_hashes = sorted({str(row["authorization_sha256"]) for row in counters})
    site_input_max = max(float(row["site_input_max_error"]) for row in counters)
    site_velocity_max = max(float(row["site_velocity_max_error"]) for row in counters)
    site_tolerance = float(manifest["runtime"]["intervention_site_error_tolerance"])
    audit = {
        "audit": "independent-cosmos3-dose-v2-raw-output-audit-v1",
        "status": "pass", "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.expected_manifest_sha256,
        "inventory_sha256": args.expected_inventory_sha256,
        "summary_sha256": args.expected_summary_sha256,
        "auditor_sha256": sha256(Path(__file__)),
        "io_helper_sha256": sha256(Path(audit_common.__file__)),
        "outcome_file_count": 30, "request_count": 2760,
        "dose_cell_count": 1800, "state_row_count": 30,
        "raw_action_metrics_recomputed": True,
        "frozen_analyzer_or_helpers_imported": False,
        "authorization_sha256_values": authorization_hashes,
        "controls": {
            "shape_valid_response_action_count": int(sum(int(row["shape_valid_actions"]) for row in counters)),
            "intervention_response_count": int(sum(int(row["intervention_responses"]) for row in counters)),
            "active_response_count": int(sum(int(row["active_responses"]) for row in counters)),
            "active_site_count": int(sum(int(row["active_sites"]) for row in counters)),
            "structural_projection_null_count": int(sum(int(row["projection_null"]) for row in counters)),
            "finite_projection_count": int(sum(int(row["projection_finite"]) for row in counters)),
            "native_projection_absent_count": int(sum(int(row["projection_absent"]) for row in counters)),
            "all_action_coordinate_errors_zero": True,
            "site_errors_within_tolerance": (
                site_input_max <= site_tolerance and site_velocity_max <= site_tolerance
            ),
            "all_site_errors_zero": site_input_max == 0.0 and site_velocity_max == 0.0,
            "maximum_model_input_future_clamp_error": site_input_max,
            "maximum_returned_future_velocity_overwrite_error": site_velocity_max,
            "intervention_site_error_tolerance": site_tolerance,
            "rng_source_target_and_replay_gates_exact": True,
        },
        "comparison_discrepancies": problems,
        "computed": computed,
    }
    atomic_json_no_overwrite(args.output, audit)
    print(json.dumps({"status": "pass", "discrepancy_count": 0,
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
