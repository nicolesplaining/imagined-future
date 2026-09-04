#!/usr/bin/env python3
"""Run one prospective Cosmos 3 future-strength dose-response state.

This worker reconstructs a policy input from an archived lossy RoboLab MP4 and
noise-free recorded proprioception.  It never restores or steps the simulator,
so its admissible scope is action-level evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from imagined_future.cosmos3_archival import (
    atomic_json,
    compose_cosmos_observation,
    recorded_proprio,
    sha256,
)
from imagined_future.cosmos3_dose_response import (
    ALPHAS,
    EXPECTED_ACTIVE_RESPONSES_PER_STATE,
    EXPECTED_ACTIVE_SITES_PER_STATE,
    EXPECTED_CALLS_PER_STATE,
    EXPECTED_DENOISING_CALLS,
    EXPECTED_FUTURE_FRAME_INDICES,
    dose_action_metrics,
    dose_label,
    frozen_request_specs,
    validate_released_action,
)


INTERVENTION_SITE_ERROR_TOLERANCE = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open archived MP4: {path}")
    try:
        for index in range(frame_index + 1):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"MP4 ended before frozen frame {frame_index}: {path}")
        if index != frame_index:
            raise AssertionError("sequential decoder stopped at the wrong frame")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def torch_bool_tensor_digest(value: np.ndarray) -> str:
    """Reproduce the server's tensor digest for a contiguous flat torch.bool mask."""

    flat = np.ascontiguousarray(np.asarray(value, dtype=np.bool_).reshape(-1))
    digest = hashlib.sha256()
    digest.update(b"torch.bool")
    digest.update(np.asarray(flat.shape, dtype=np.int64).tobytes())
    digest.update(flat.view(np.uint8).tobytes())
    return digest.hexdigest()


def normalized_projection(
    value: np.ndarray, recipient: np.ndarray, donor: np.ndarray, *, eps: float = 1e-12
) -> float | None:
    direction = donor.astype(np.float64).reshape(-1) - recipient.astype(np.float64).reshape(-1)
    denominator = float(np.dot(direction, direction))
    if denominator <= eps:
        return None
    displacement = value.astype(np.float64).reshape(-1) - recipient.astype(np.float64).reshape(-1)
    return float(np.dot(displacement, direction) / denominator)


def response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in response.items():
        if key in {"action", "video"}:
            continue
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def response_null_and_nonfinite_paths(
    value: Any, prefix: str = ""
) -> tuple[set[str], set[str]]:
    """Enumerate structural nulls and forbidden NaN/Inf values on the wire."""

    nulls: set[str] = set()
    nonfinite: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            child_nulls, child_nonfinite = response_null_and_nonfinite_paths(item, path)
            nulls.update(child_nulls)
            nonfinite.update(child_nonfinite)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_nulls, child_nonfinite = response_null_and_nonfinite_paths(
                item, f"{prefix}[{index}]"
            )
            nulls.update(child_nulls)
            nonfinite.update(child_nonfinite)
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            nonfinite.add(prefix)
    elif value is None:
        nulls.add(prefix)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        nonfinite.add(prefix)
    return nulls, nonfinite


def _canonical_signature_value(value: Any) -> Any:
    """Convert deterministic response metadata into strict canonical JSON values."""

    if isinstance(value, dict):
        return {
            str(key): _canonical_signature_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_signature_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical_signature_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return {"__nonfinite_float__": "nan"}
        if math.isinf(number):
            return {"__nonfinite_float__": "inf" if number > 0 else "-inf"}
        return number
    return value


def signature(response: dict[str, Any]) -> str:
    """Hash all deterministic server metadata, excluding only request ID and timing."""

    excluded = {
        "action",
        "video",
        "research_id",
        "research_infer_ms",
        "server_timing",
    }
    payload = {
        key: _canonical_signature_value(value)
        for key, value in response.items()
        if key not in excluded
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def behavior_signature(response: dict[str, Any]) -> str:
    """Hash model behavior while excluding intentional donor-routing metadata."""

    keys = (
        "action",
        "research_target_hash",
        "research_recipient_future_hash",
        "research_recipient_path_noise_hash",
        "research_initial_state_hash",
        "research_output_future_hash",
        "research_final_sampler_target_max_abs_error",
        "research_final_sampler_target_l2",
        "research_sigmas",
        "research_x0_sigmas",
        "research_x0_vision_hashes",
        "research_x0_action_hashes",
        "research_vision_shape",
        "research_future_frame_indices",
        "research_vision_coordinate_count",
        "research_future_mask_coordinate_count",
        "research_future_mask_index_hash",
        "research_requested_active_call_indices",
        "research_observed_active_call_indices",
        "research_clamped_call_indices",
        "research_inactive_call_indices",
        "research_requested_active_sigmas",
        "research_observed_active_sigmas",
        "research_model_input_future_clamp_errors",
        "research_returned_future_velocity_overwrite_errors",
        "research_maximum_action_input_error",
        "research_maximum_action_output_error",
        "research_action_input_errors",
        "research_action_output_errors",
        "research_inactive_wrapper_write_count",
    )
    missing = [key for key in keys if key not in response]
    if missing:
        raise RuntimeError(f"behavior signature is missing fields: {missing}")
    encoded = json.dumps(
        {key: _canonical_signature_value(response[key]) for key in keys},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_intervention_site_audit(
    response: dict[str, Any],
    *,
    label: str,
    active: bool,
    target_source: str,
    recipient_id: str,
    donor_id: str,
    recipient_future_hash: str,
    donor_future_hash: str,
    recipient_path_noise_hash: str,
    recipient_initial_state_hash: str,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Validate the actual model-input and returned-velocity intervention sites."""

    sigmas = np.asarray(response["research_sigmas"], dtype=np.float64).reshape(-1)
    if len(sigmas) != EXPECTED_DENOISING_CALLS or not np.isfinite(sigmas).all():
        raise RuntimeError(f"{label}: denoising-call sigma audit failed: {sigmas}")
    expected_indices = (
        np.arange(EXPECTED_DENOISING_CALLS, dtype=np.int64)
        if active
        else np.asarray([], dtype=np.int64)
    )
    requested_indices = np.asarray(
        response["research_requested_active_call_indices"], dtype=np.int64
    ).reshape(-1)
    observed_indices = np.asarray(
        response["research_observed_active_call_indices"], dtype=np.int64
    ).reshape(-1)
    if not (
        np.array_equal(requested_indices, expected_indices)
        and np.array_equal(observed_indices, expected_indices)
        and np.array_equal(
            np.asarray(response["research_clamped_call_indices"], dtype=np.int64),
            expected_indices,
        )
    ):
        raise RuntimeError(
            f"{label}: requested/observed active calls differ: "
            f"{requested_indices}, {observed_indices}, expected {expected_indices}"
        )
    expected_inactive_indices = np.asarray(
        [
            index
            for index in range(EXPECTED_DENOISING_CALLS)
            if index not in set(expected_indices.tolist())
        ],
        dtype=np.int64,
    )
    inactive_indices = np.asarray(
        response["research_inactive_call_indices"], dtype=np.int64
    ).reshape(-1)
    if not np.array_equal(inactive_indices, expected_inactive_indices):
        raise RuntimeError(
            f"{label}: inactive calls {inactive_indices} differ from "
            f"{expected_inactive_indices}"
        )
    requested_sigmas = np.asarray(
        response["research_requested_active_sigmas"], dtype=np.float64
    ).reshape(-1)
    observed_sigmas = np.asarray(
        response["research_observed_active_sigmas"], dtype=np.float64
    ).reshape(-1)
    expected_sigmas = sigmas[expected_indices]
    if not (
        np.array_equal(requested_sigmas, expected_sigmas)
        and np.array_equal(observed_sigmas, expected_sigmas)
    ):
        raise RuntimeError(f"{label}: requested/observed active sigmas differ")

    frames = tuple(
        int(item)
        for item in np.asarray(
            response["research_future_frame_indices"], dtype=np.int64
        ).reshape(-1)
    )
    vision_coordinates = int(response["research_vision_coordinate_count"])
    mask_coordinates = int(response["research_future_mask_coordinate_count"])
    vision_shape = tuple(
        int(item)
        for item in np.asarray(
            response["research_vision_shape"], dtype=np.int64
        ).reshape(-1)
    )
    if len(vision_shape) not in (4, 5):
        raise RuntimeError(f"{label}: invalid vision shape {vision_shape}")
    temporal_axis = len(vision_shape) - 3
    expected_mask = np.zeros(vision_shape, dtype=np.bool_)
    mask_index = [slice(None)] * len(vision_shape)
    mask_index[temporal_axis] = list(frames)
    expected_mask[tuple(mask_index)] = True
    expected_mask_hash = torch_bool_tensor_digest(expected_mask)
    if (
        frames != EXPECTED_FUTURE_FRAME_INDICES
        or vision_coordinates <= 0
        or int(np.prod(vision_shape)) != vision_coordinates
        or vision_coordinates % 9 != 0
        or mask_coordinates != (vision_coordinates // 9) * 8
        or response.get("research_future_mask_index_hash") != expected_mask_hash
    ):
        raise RuntimeError(
            f"{label}: future mask/frame audit failed: frames={frames}, "
            f"mask={mask_coordinates}, vision={vision_coordinates}"
        )
    if (
        int(response.get("research_target_coordinate_count", -1))
        != vision_coordinates
        or int(response.get("research_target_finite_coordinate_count", -1))
        != vision_coordinates
    ):
        raise RuntimeError(f"{label}: target shape/finiteness gate failed")

    model_input_errors = np.asarray(
        response["research_model_input_future_clamp_errors"], dtype=np.float64
    ).reshape(-1)
    velocity_errors = np.asarray(
        response["research_returned_future_velocity_overwrite_errors"],
        dtype=np.float64,
    ).reshape(-1)
    expected_site_count = EXPECTED_DENOISING_CALLS if active else 0
    if (
        len(model_input_errors) != expected_site_count
        or len(velocity_errors) != expected_site_count
        or not np.isfinite(model_input_errors).all()
        or not np.isfinite(velocity_errors).all()
        or (len(model_input_errors) and model_input_errors.max() > INTERVENTION_SITE_ERROR_TOLERANCE)
        or (len(velocity_errors) and velocity_errors.max() > INTERVENTION_SITE_ERROR_TOLERANCE)
    ):
        raise RuntimeError(
            f"{label}: intervention-site fidelity failed: input={model_input_errors}, "
            f"velocity={velocity_errors}"
        )

    expected_source_ids = (
        [recipient_id]
        if target_source == "recipient"
        else [donor_id]
        if target_source == "donor"
        else [recipient_id, donor_id]
    )
    if (
        response.get("research_target_source") != target_source
        or response.get("research_target_source_record_ids") != expected_source_ids
        or response.get("research_recipient_future_hash") != recipient_future_hash
        or response.get("research_donor_future_hash") != donor_future_hash
        or response.get("research_recipient_path_noise_hash")
        != recipient_path_noise_hash
        or response.get("research_initial_state_hash") != recipient_initial_state_hash
    ):
        raise RuntimeError(f"{label}: target-source provenance audit failed")
    if target_source == "recipient" and response.get("research_target_hash") != recipient_future_hash:
        raise RuntimeError(f"{label}: recipient target hash differs")
    if target_source == "donor" and response.get("research_target_hash") != donor_future_hash:
        raise RuntimeError(f"{label}: donor target hash differs")
    if target_source == "recipient_donor_linear_interpolation":
        if alpha is None or float(response.get("research_alpha")) != float(alpha):
            raise RuntimeError(f"{label}: returned interpolation alpha differs")
        alpha_grid = tuple(
            float(item)
            for item in np.asarray(response.get("research_alpha_grid"), dtype=np.float64)
            .reshape(-1)
            .tolist()
        )
        if alpha_grid != ALPHAS:
            raise RuntimeError(f"{label}: server alpha grid differs: {alpha_grid}")
        formula_error = float(response["research_interpolation_formula_max_abs_error"])
        nonfuture_error = float(
            response["research_nonfuture_recipient_target_max_abs_error"]
        )
        if formula_error != 0.0 or nonfuture_error != 0.0:
            raise RuntimeError(
                f"{label}: interpolation formula/nonfuture gate failed: "
                f"{formula_error}, {nonfuture_error}"
            )
        for key in (
            "research_interpolated_future_hash",
            "research_recipient_future_mask_hash",
            "research_donor_future_mask_hash",
            "research_current_frame_hash",
            "research_recipient_current_frame_hash",
            "research_future_mask_hash",
        ):
            value = response.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError(f"{label}: missing interpolation hash {key}")
        if (
            response["research_current_frame_hash"]
            != response["research_recipient_current_frame_hash"]
            or response["research_future_mask_hash"]
            != response["research_future_mask_index_hash"]
        ):
            raise RuntimeError(f"{label}: current-frame/future-mask identity failed")
        alpha_zero_error = response.get(
            "research_alpha_zero_recipient_future_max_abs_error"
        )
        alpha_one_error = response.get("research_alpha_one_donor_future_max_abs_error")
        if alpha == 0.0:
            if (
                alpha_zero_error != 0.0
                or alpha_one_error is not None
                or response.get("research_target_hash") != recipient_future_hash
                or response.get("research_alpha_zero_target_hash_matches_recipient")
                is not True
                or response.get("research_alpha_one_target_hash_matches_donor")
                is not None
                or response["research_interpolated_future_hash"]
                != response["research_recipient_future_mask_hash"]
            ):
                raise RuntimeError(f"{label}: alpha=0 endpoint identity failed")
        elif alpha == 1.0:
            if (
                alpha_one_error != 0.0
                or alpha_zero_error is not None
                or response.get("research_target_hash") != donor_future_hash
                or response.get("research_alpha_one_target_hash_matches_donor")
                is not True
                or response.get("research_alpha_zero_target_hash_matches_recipient")
                is not None
                or response["research_interpolated_future_hash"]
                != response["research_donor_future_mask_hash"]
            ):
                raise RuntimeError(f"{label}: alpha=1 endpoint identity failed")
        elif (
            alpha_zero_error is not None
            or alpha_one_error is not None
            or response.get("research_alpha_zero_target_hash_matches_recipient")
            is not None
            or response.get("research_alpha_one_target_hash_matches_donor")
            is not None
        ):
            raise RuntimeError(f"{label}: nonendpoint identity fields must be null")
    final_residual_max_abs = float(
        response["research_final_sampler_target_max_abs_error"]
    )
    final_residual_l2 = float(response["research_final_sampler_target_l2"])
    if not math.isfinite(final_residual_max_abs) or not math.isfinite(final_residual_l2):
        raise RuntimeError(f"{label}: final sampler target residual is not finite")
    action_errors = (
        float(response["research_maximum_action_input_error"]),
        float(response["research_maximum_action_output_error"]),
    )
    if action_errors != (0.0, 0.0):
        raise RuntimeError(f"{label}: action coordinates were written: {action_errors}")
    action_input_errors = np.asarray(
        response["research_action_input_errors"], dtype=np.float64
    ).reshape(-1)
    action_output_errors = np.asarray(
        response["research_action_output_errors"], dtype=np.float64
    ).reshape(-1)
    if (
        len(action_input_errors) != EXPECTED_DENOISING_CALLS
        or len(action_output_errors) != EXPECTED_DENOISING_CALLS
        or not np.isfinite(action_input_errors).all()
        or not np.isfinite(action_output_errors).all()
        or np.any(action_input_errors != 0.0)
        or np.any(action_output_errors != 0.0)
        or int(response["research_inactive_wrapper_write_count"]) != 0
    ):
        raise RuntimeError(f"{label}: per-call action/inactive write audit failed")
    return {
        "mode": str(response["research_mode"]),
        "target_source": target_source,
        "target_hash": str(response["research_target_hash"]),
        "recipient_id": recipient_id,
        "donor_id": donor_id,
        "target_source_record_ids": list(response["research_target_source_record_ids"]),
        "recipient_future_hash": recipient_future_hash,
        "donor_future_hash": donor_future_hash,
        "recipient_path_noise_hash": recipient_path_noise_hash,
        "initial_state_hash": recipient_initial_state_hash,
        "active_site_count": expected_site_count,
        "sigmas": sigmas.tolist(),
        "requested_active_call_indices": requested_indices.tolist(),
        "observed_active_call_indices": observed_indices.tolist(),
        "clamped_call_indices": observed_indices.tolist(),
        "inactive_call_indices": inactive_indices.tolist(),
        "requested_active_sigmas": requested_sigmas.tolist(),
        "observed_active_sigmas": observed_sigmas.tolist(),
        "future_frame_indices": list(frames),
        "vision_shape": list(vision_shape),
        "future_mask_index_hash": expected_mask_hash,
        "model_input_future_clamp_errors": model_input_errors.tolist(),
        "returned_future_velocity_overwrite_errors": velocity_errors.tolist(),
        "model_input_max_error": float(model_input_errors.max())
        if len(model_input_errors)
        else 0.0,
        "returned_velocity_max_error": float(velocity_errors.max())
        if len(velocity_errors)
        else 0.0,
        "final_sampler_target_max_abs_error": final_residual_max_abs,
        "final_sampler_target_l2": final_residual_l2,
        "mask_coordinate_count": mask_coordinates,
        "vision_coordinate_count": vision_coordinates,
        "target_coordinate_count": vision_coordinates,
        "target_finite_coordinate_count": vision_coordinates,
        "maximum_action_input_error": action_errors[0],
        "maximum_action_output_error": action_errors[1],
        "action_input_errors": action_input_errors.tolist(),
        "action_output_errors": action_output_errors.tolist(),
        "inactive_wrapper_write_count": 0,
        "alpha": alpha,
        "interpolation_formula_max_abs_error": (
            float(response["research_interpolation_formula_max_abs_error"])
            if alpha is not None
            else None
        ),
        "nonfuture_recipient_target_max_abs_error": (
            float(response["research_nonfuture_recipient_target_max_abs_error"])
            if alpha is not None
            else None
        ),
        "alpha_zero_recipient_future_max_abs_error": (
            response.get("research_alpha_zero_recipient_future_max_abs_error")
            if alpha is not None
            else None
        ),
        "alpha_one_donor_future_max_abs_error": (
            response.get("research_alpha_one_donor_future_max_abs_error")
            if alpha is not None
            else None
        ),
        "alpha_zero_target_hash_matches_recipient": (
            response.get("research_alpha_zero_target_hash_matches_recipient")
            if alpha is not None
            else None
        ),
        "alpha_one_target_hash_matches_donor": (
            response.get("research_alpha_one_target_hash_matches_donor")
            if alpha is not None
            else None
        ),
        "interpolated_future_hash": (
            response.get("research_interpolated_future_hash")
            if alpha is not None
            else None
        ),
        "recipient_future_mask_hash": (
            response.get("research_recipient_future_mask_hash")
            if alpha is not None
            else None
        ),
        "donor_future_mask_hash": (
            response.get("research_donor_future_mask_hash")
            if alpha is not None
            else None
        ),
        "current_frame_hash": (
            response.get("research_current_frame_hash")
            if alpha is not None
            else None
        ),
        "recipient_current_frame_hash": (
            response.get("research_recipient_current_frame_hash")
            if alpha is not None
            else None
        ),
        "future_mask_hash": (
            response.get("research_future_mask_hash")
            if alpha is not None
            else None
        ),
    }


def validate_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_hash = sha256(args.manifest)
    if actual_hash != args.expected_manifest_sha256:
        raise ValueError(
            f"manifest SHA mismatch: expected {args.expected_manifest_sha256}, got {actual_hash}"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_model_outcomes":
        raise ValueError("manifest is not frozen before model outcomes")
    if manifest.get("study_name") != "cosmos3-future-strength-dose-response-v2":
        raise ValueError("manifest is not the prospective dose-response study")
    if manifest.get("selection_uses_model_or_intervention_outcomes") is not False:
        raise ValueError("manifest does not explicitly prohibit outcome-based selection")
    if manifest.get("scope", {}).get("physical_endpoint_evidence") is not False:
        raise ValueError("dose-response runner cannot make a physical-endpoint claim")
    admission = manifest.get("admission")
    if admission == "prospective_action_level_future_strength_dose_response":
        if (
            manifest.get("freeze_stage") != "evaluation_ready"
            or manifest.get("launch_authorization")
            != "powered_evaluation_after_independent_go"
            or len(manifest.get("states", [])) != 30
        ):
            raise ValueError("runner refuses a non-final powered-evaluation manifest")
    elif admission == "excluded_development_smoke":
        if (
            manifest.get("freeze_stage") != "pre_smoke"
            or manifest.get("launch_authorization")
            != "excluded_smoke_only_not_powered_evaluation"
            or len(manifest.get("states", [])) != 1
        ):
            raise ValueError("runner refuses a malformed excluded-smoke manifest")
    else:
        raise ValueError(f"unsupported dose admission: {admission!r}")
    actual_audit_sha256 = sha256(args.audit_report)
    if actual_audit_sha256 != args.expected_audit_sha256:
        raise ValueError("independent authorization audit SHA differs from frozen CLI")
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    expected_audit = (
        {
            "status": "pass",
            "verdict": "GO",
            "scope": "outcome_blind_prelaunch_audit",
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": actual_hash,
            "snapshot_checksum_list_sha256": manifest["runtime"][
                "snapshot_checksum_list_sha256"
            ],
            "authorized_state_count": 30,
            "authorized_call_count": 2760,
        }
        if admission == "prospective_action_level_future_strength_dose_response"
        else {
            "status": "pass",
            "verdict": "GO_SMOKE",
            "scope": "outcome_blind_excluded_smoke_prelaunch_audit",
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": actual_hash,
            "snapshot_checksum_list_sha256": manifest["runtime"][
                "snapshot_checksum_list_sha256"
            ],
            "authorized_state_count": 1,
            "authorized_call_count": EXPECTED_CALLS_PER_STATE,
        }
    )
    if {key: audit.get(key) for key in expected_audit} != expected_audit:
        raise ValueError("independent audit does not authorize this exact manifest")
    runner_hash = sha256(Path(__file__).resolve())
    expected_runner = manifest.get("runtime", {}).get("runner_sha256")
    if runner_hash != expected_runner:
        raise ValueError(f"runner SHA mismatch: manifest {expected_runner}, current {runner_hash}")
    matches = [unit for unit in manifest.get("states", []) if unit.get("unit_id") == args.unit_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest unit {args.unit_id!r}, got {len(matches)}")
    return manifest, matches[0]


def build_request(
    unit: dict[str, Any], screen_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    assets = unit["assets"]
    episode_dir = screen_root / assets["relative_episode_directory"]
    mp4 = episode_dir / assets["mp4_filename"]
    hdf5 = episode_dir / assets["hdf5_filename"]
    env_cfg = episode_dir / assets["env_cfg_filename"]
    for label, path in (("mp4", mp4), ("hdf5", hdf5), ("env_cfg", env_cfg)):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        expected = unit["input_sha256"][label]
        if actual != expected:
            raise ValueError(f"{label} SHA mismatch for {unit['unit_id']}: {actual} != {expected}")

    frame = read_frame(mp4, int(unit["mp4_frame_index"]))
    image = compose_cosmos_observation(frame)
    with h5py.File(hdf5, "r") as stream:
        dataset = stream["data/demo_0/states/articulation/robot/joint_position"]
        index = int(unit["hdf5_state_index"])
        if index < 0 or index >= len(dataset):
            raise IndexError(f"frozen HDF5 state index {index} is out of range")
        raw_proprio = np.asarray(dataset[index], dtype=np.float32)
    joints, gripper = recorded_proprio(raw_proprio)
    request = {
        "observation/image": image,
        "observation/joint_position": joints,
        "observation/gripper_position": gripper,
        "prompt": str(unit["instruction"]),
    }
    audit = {
        "reconstruction": (
            "lossy H.264 MP4 frame branch_step-1; panel order head,left,right,wrist; "
            "Cosmos input is wrist above bilinear half-scale left/right"
        ),
        "proprio": "noise-free recorded post-step joint state at branch_step-1",
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "joint_position_sha256": hashlib.sha256(joints.tobytes()).hexdigest(),
        "gripper_position_sha256": hashlib.sha256(gripper.tobytes()).hexdigest(),
    }
    return request, audit


def dose_main() -> None:
    """Execute the frozen 92-call dose matrix for exactly one manifest state."""

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite completed unit: {args.output}")
    manifest, unit = validate_manifest(args)
    base, input_audit = build_request(unit, args.screen_root)
    branch_seeds = tuple(int(seed) for seed in unit["branch_seeds"])
    request_specs = frozen_request_specs(branch_seeds)
    if unit.get("request_sequence") != list(request_specs):
        raise ValueError("unit request sequence differs from the frozen 92-call matrix")
    if unit.get("phase") != "middle" or float(unit.get("phase_fraction")) != 0.5:
        raise ValueError("dose response accepts only the 30 frozen middle-phase states")

    client = WebsocketClientPolicy(args.host, args.port)
    study_id = f"{manifest['manifest_id']}-{unit['unit_id']}"
    responses: dict[str, dict[str, Any]] = {}
    request_trace: list[str] = []
    intervention_site_audits: dict[str, dict[str, Any]] = {}

    def infer(label: str, **research: Any) -> dict[str, Any]:
        if label in responses:
            raise RuntimeError(f"duplicate response label: {label}")
        request_id = f"{study_id}-{label}"
        request = {**base, "research_id": request_id, **research}
        response = client.infer(request)
        null_paths, nonfinite_paths = response_null_and_nonfinite_paths(response)
        allowed_null_paths = {"research_attention_interface.cache_id"}
        requested_mode = str(request.get("research_mode", "native"))
        if requested_mode in {"none", "self"}:
            allowed_null_paths.add("research_action_donor_projection")
        if requested_mode == "dose":
            alpha = float(request["research_alpha"])
            if alpha == 0.0:
                allowed_null_paths.update(
                    {
                        "research_alpha_one_donor_future_max_abs_error",
                        "research_alpha_one_target_hash_matches_donor",
                    }
                )
            elif alpha == 1.0:
                allowed_null_paths.update(
                    {
                        "research_alpha_zero_recipient_future_max_abs_error",
                        "research_alpha_zero_target_hash_matches_recipient",
                    }
                )
            else:
                allowed_null_paths.update(
                    {
                        "research_alpha_zero_recipient_future_max_abs_error",
                        "research_alpha_one_donor_future_max_abs_error",
                        "research_alpha_zero_target_hash_matches_recipient",
                        "research_alpha_one_target_hash_matches_donor",
                    }
                )
        if nonfinite_paths or null_paths != allowed_null_paths:
            raise RuntimeError(
                f"{label}: strict wire finiteness/null schema failed: "
                f"nonfinite={sorted(nonfinite_paths)}, nulls={sorted(null_paths)}, "
                f"allowed={sorted(allowed_null_paths)}"
            )
        for key in (
            "research_id",
            "research_mode",
            "research_seed",
            "research_recipient_id",
            "research_donor_id",
        ):
            if key in request and response.get(key) != request[key]:
                raise RuntimeError(
                    f"{label}: returned {key}={response.get(key)!r}, expected "
                    f"{request[key]!r}"
                )
        try:
            validate_released_action(response["action"], label=label)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        responses[label] = response
        request_trace.append(label)
        return response

    native: dict[int, dict[str, Any]] = {}
    native_repeat: dict[int, dict[str, Any]] = {}
    for seed in branch_seeds:
        native[seed] = infer(
            f"native-{seed}", research_mode="native", research_seed=seed
        )
    for seed in branch_seeds:
        native_repeat[seed] = infer(
            f"native-repeat-{seed}", research_mode="native", research_seed=seed
        )

    native_actions = {
        seed: np.asarray(response["action"], dtype=np.float32)
        for seed, response in native.items()
    }
    native_repeat_gates: dict[str, dict[str, Any]] = {}
    for seed in branch_seeds:
        action_error = maximum_error(
            native_actions[seed], np.asarray(native_repeat[seed]["action"], dtype=np.float32)
        )
        metadata_exact = signature(native[seed]) == signature(native_repeat[seed])
        future_trace_exact = all(
            _canonical_signature_value(native[seed][key])
            == _canonical_signature_value(native_repeat[seed][key])
            for key in (
                "research_future_hash",
                "research_x0_vision_hashes",
                "research_x0_action_hashes",
                "research_path_noise_hash",
                "research_initial_state_hash",
                "research_sigmas",
                "research_x0_sigmas",
            )
        )
        native_repeat_gates[str(seed)] = {
            "action_max_abs_error": action_error,
            "deterministic_metadata_exact": metadata_exact,
            "future_trace_exact": future_trace_exact,
        }
        if action_error != 0.0 or not metadata_exact or not future_trace_exact:
            raise RuntimeError(f"native replay gate failed for seed {seed}")

    none_rows: list[dict[str, Any]] = []
    for seed in branch_seeds:
        native_id = f"{study_id}-native-{seed}"
        label = f"none-{seed}"
        response = infer(
            label,
            research_mode="none",
            research_seed=seed,
            research_recipient_id=native_id,
            research_donor_id=native_id,
            research_timing_steps=[],
        )
        intervention_site_audits[label] = validate_intervention_site_audit(
            response,
            label=label,
            active=False,
            target_source="recipient",
            recipient_id=native_id,
            donor_id=native_id,
            recipient_future_hash=str(native[seed]["research_future_hash"]),
            donor_future_hash=str(native[seed]["research_future_hash"]),
            recipient_path_noise_hash=str(native[seed]["research_path_noise_hash"]),
            recipient_initial_state_hash=str(native[seed]["research_initial_state_hash"]),
        )
        action_error = maximum_error(
            np.asarray(response["action"], dtype=np.float32), native_actions[seed]
        )
        exact_trace = (
            response["research_output_future_hash"] == native[seed]["research_future_hash"]
            and _canonical_signature_value(response["research_x0_vision_hashes"])
            == _canonical_signature_value(native[seed]["research_x0_vision_hashes"])
            and _canonical_signature_value(response["research_x0_action_hashes"])
            == _canonical_signature_value(native[seed]["research_x0_action_hashes"])
            and _canonical_signature_value(response["research_sigmas"])
            == _canonical_signature_value(native[seed]["research_sigmas"])
            and _canonical_signature_value(response["research_x0_sigmas"])
            == _canonical_signature_value(native[seed]["research_x0_sigmas"])
        )
        projection_is_structural_null = (
            response.get("research_action_donor_projection_applicable") is False
            and response.get("research_action_donor_projection") is None
        )
        if action_error != 0.0 or not exact_trace or not projection_is_structural_null:
            raise RuntimeError(f"zero-active-site no-op gate failed for seed {seed}")
        none_rows.append(
            {
                "recipient_seed": seed,
                "action_max_abs_error_vs_native": action_error,
                "native_trace_exact": exact_trace,
                "projection_structural_null": projection_is_structural_null,
                "server": response_metadata(response),
            }
        )

    self_first: dict[int, dict[str, Any]] = {}
    self_repeat: dict[int, dict[str, Any]] = {}
    for seed in branch_seeds:
        native_id = f"{study_id}-native-{seed}"
        self_first[seed] = infer(
            f"self-{seed}",
            research_mode="self",
            research_seed=seed,
            research_recipient_id=native_id,
            research_donor_id=native_id,
        )
    for seed in branch_seeds:
        native_id = f"{study_id}-native-{seed}"
        self_repeat[seed] = infer(
            f"self-repeat-{seed}",
            research_mode="self",
            research_seed=seed,
            research_recipient_id=native_id,
            research_donor_id=native_id,
        )

    self_rows: list[dict[str, Any]] = []
    for seed in branch_seeds:
        native_id = f"{study_id}-native-{seed}"
        for label, response in (
            (f"self-{seed}", self_first[seed]),
            (f"self-repeat-{seed}", self_repeat[seed]),
        ):
            intervention_site_audits[label] = validate_intervention_site_audit(
                response,
                label=label,
                active=True,
                target_source="recipient",
                recipient_id=native_id,
                donor_id=native_id,
                recipient_future_hash=str(native[seed]["research_future_hash"]),
                donor_future_hash=str(native[seed]["research_future_hash"]),
                recipient_path_noise_hash=str(native[seed]["research_path_noise_hash"]),
                recipient_initial_state_hash=str(native[seed]["research_initial_state_hash"]),
            )
        first_action = np.asarray(self_first[seed]["action"], dtype=np.float32)
        repeat_action = np.asarray(self_repeat[seed]["action"], dtype=np.float32)
        repeat_error = maximum_error(first_action, repeat_action)
        repeat_exact = signature(self_first[seed]) == signature(self_repeat[seed])
        projection_is_structural_null = all(
            response.get("research_action_donor_projection_applicable") is False
            and response.get("research_action_donor_projection") is None
            for response in (self_first[seed], self_repeat[seed])
        )
        if repeat_error != 0.0 or not repeat_exact or not projection_is_structural_null:
            raise RuntimeError(f"self-clamp replay gate failed for seed {seed}")
        self_rows.append(
            {
                "recipient_seed": seed,
                "repeat_action_max_abs_error": repeat_error,
                "repeat_signature_exact": repeat_exact,
                "projection_structural_null": projection_is_structural_null,
                "clean_clamp_vs_native_max_abs_error": maximum_error(
                    first_action, native_actions[seed]
                ),
                "clean_clamp_vs_native_l2": float(
                    np.linalg.norm(first_action.astype(np.float64) - native_actions[seed])
                ),
                "action": first_action.tolist(),
                "server": response_metadata(self_first[seed]),
            }
        )

    dose_responses: dict[tuple[int, int, float], dict[str, Any]] = {}
    dose_rows: list[dict[str, Any]] = []
    pairs = [
        (int(pair[0]), int(pair[1])) for pair in unit["ordered_pairs"]
    ]
    if pairs != [
        (int(row["recipient_seed"]), int(row["donor_seed"]))
        for row in request_specs[20:80:5]
    ]:
        raise ValueError("manifest ordered-pair grid differs from request sequence")
    native_axis_l2 = {
        f"{recipient}:{donor}": float(
            np.linalg.norm(
                native_actions[donor].astype(np.float64).reshape(-1)
                - native_actions[recipient].astype(np.float64).reshape(-1)
            )
        )
        for recipient, donor in pairs
    }
    if len(native_axis_l2) != 12 or any(
        not math.isfinite(value) or value <= 1e-12
        for value in native_axis_l2.values()
    ):
        raise RuntimeError(f"native donor-axis gate failed: {native_axis_l2}")
    for recipient_seed, donor_seed in pairs:
        recipient_id = f"{study_id}-native-{recipient_seed}"
        donor_id = f"{study_id}-native-{donor_seed}"
        for alpha in ALPHAS:
            label = dose_label(recipient_seed, donor_seed, alpha)
            response = infer(
                label,
                research_mode="dose",
                research_seed=recipient_seed,
                research_recipient_id=recipient_id,
                research_donor_id=donor_id,
                research_alpha=alpha,
            )
            dose_responses[(recipient_seed, donor_seed, alpha)] = response
            intervention_site_audits[label] = validate_intervention_site_audit(
                response,
                label=label,
                active=True,
                target_source="recipient_donor_linear_interpolation",
                recipient_id=recipient_id,
                donor_id=donor_id,
                recipient_future_hash=str(native[recipient_seed]["research_future_hash"]),
                donor_future_hash=str(native[donor_seed]["research_future_hash"]),
                recipient_path_noise_hash=str(
                    native[recipient_seed]["research_path_noise_hash"]
                ),
                recipient_initial_state_hash=str(
                    native[recipient_seed]["research_initial_state_hash"]
                ),
                alpha=alpha,
            )
            if response.get("research_action_donor_projection_applicable") is not True:
                raise RuntimeError(f"{label}: donor axis is degenerate")
            server_projection = float(response["research_action_donor_projection"])
            if not math.isfinite(server_projection):
                raise RuntimeError(f"{label}: server donor projection is nonfinite")
            action = np.asarray(response["action"], dtype=np.float32)
            metrics = dose_action_metrics(
                action,
                native_actions[recipient_seed],
                native_actions[donor_seed],
                native_actions,
                donor_seed,
            )
            directional_values = [
                metrics[key]
                for key in (
                    "distance_reduction_to_donor",
                    "normalized_projection",
                    "cosine_alignment",
                    "orthogonal_residual_normalized",
                )
            ]
            if any(value is None or not math.isfinite(float(value)) for value in directional_values):
                raise RuntimeError(f"{label}: a required directional metric is undefined")
            if abs(float(metrics["normalized_projection"]) - server_projection) > 1e-6:
                raise RuntimeError(f"{label}: server/client projection audit differs")
            dose_rows.append(
                {
                    "recipient_seed": recipient_seed,
                    "donor_seed": donor_seed,
                    "alpha": alpha,
                    **metrics,
                    "server_action_donor_projection": server_projection,
                    "target_hash": str(response["research_target_hash"]),
                    "final_sampler_target_max_abs_error": float(
                        response["research_final_sampler_target_max_abs_error"]
                    ),
                    "final_sampler_target_l2": float(
                        response["research_final_sampler_target_l2"]
                    ),
                    "action": action.tolist(),
                    "server": response_metadata(response),
                }
            )

    alpha_zero_control_rows: list[dict[str, Any]] = []
    for recipient_seed in branch_seeds:
        self_response = self_first[recipient_seed]
        self_action = np.asarray(self_response["action"], dtype=np.float32)
        self_behavior_signature = behavior_signature(self_response)
        recipient_rows = []
        for donor_seed in branch_seeds:
            if donor_seed == recipient_seed:
                continue
            response = dose_responses[(recipient_seed, donor_seed, 0.0)]
            action_error = maximum_error(
                np.asarray(response["action"], dtype=np.float32), self_action
            )
            signature_exact = behavior_signature(response) == self_behavior_signature
            row = {
                "recipient_seed": recipient_seed,
                "donor_seed": donor_seed,
                "action_max_abs_error_vs_self": action_error,
                "behavior_signature_exact_vs_self": signature_exact,
                "behavior_signature": behavior_signature(response),
            }
            recipient_rows.append(row)
            alpha_zero_control_rows.append(row)
            if action_error != 0.0 or not signature_exact:
                raise RuntimeError(
                    f"alpha=0 donor-routing invariance failed for "
                    f"{(recipient_seed, donor_seed)}"
                )
        if len({row["behavior_signature"] for row in recipient_rows}) != 1:
            raise RuntimeError(
                f"alpha=0 responses differ across donor labels for {recipient_seed}"
            )

    midpoint_replay_rows: list[dict[str, Any]] = []
    for recipient_seed, donor_seed in pairs:
        alpha = 0.5
        base_label = dose_label(recipient_seed, donor_seed, alpha)
        replay_label = base_label + "-repeat"
        recipient_id = f"{study_id}-native-{recipient_seed}"
        donor_id = f"{study_id}-native-{donor_seed}"
        replay = infer(
            replay_label,
            research_mode="dose",
            research_seed=recipient_seed,
            research_recipient_id=recipient_id,
            research_donor_id=donor_id,
            research_alpha=alpha,
        )
        intervention_site_audits[replay_label] = validate_intervention_site_audit(
            replay,
            label=replay_label,
            active=True,
            target_source="recipient_donor_linear_interpolation",
            recipient_id=recipient_id,
            donor_id=donor_id,
            recipient_future_hash=str(native[recipient_seed]["research_future_hash"]),
            donor_future_hash=str(native[donor_seed]["research_future_hash"]),
            recipient_path_noise_hash=str(native[recipient_seed]["research_path_noise_hash"]),
            recipient_initial_state_hash=str(native[recipient_seed]["research_initial_state_hash"]),
            alpha=alpha,
        )
        original = dose_responses[(recipient_seed, donor_seed, alpha)]
        action_error = maximum_error(
            np.asarray(original["action"], dtype=np.float32),
            np.asarray(replay["action"], dtype=np.float32),
        )
        metadata_exact = signature(original) == signature(replay)
        if action_error != 0.0 or not metadata_exact:
            raise RuntimeError(
                f"midpoint replay gate failed for {(recipient_seed, donor_seed)}"
            )
        midpoint_replay_rows.append(
            {
                "recipient_seed": recipient_seed,
                "donor_seed": donor_seed,
                "alpha": alpha,
                "action_max_abs_error": action_error,
                "deterministic_metadata_exact": metadata_exact,
            }
        )

    expected_trace = [str(row["label"]) for row in request_specs]
    if request_trace != expected_trace or len(responses) != EXPECTED_CALLS_PER_STATE:
        raise RuntimeError(
            f"request order/cardinality differs: {len(responses)}, trace={request_trace}"
        )
    input_fingerprints = {
        str(response["research_state_hash"]) for response in responses.values()
    }
    parameter_probe_hashes = {
        str(response["research_parameter_probe_hash"]) for response in responses.values()
    }
    expected_parameter_probe = str(manifest["runtime"]["expected_parameter_probe_hash"])
    if len(input_fingerprints) != 1:
        raise RuntimeError(f"transformed input fingerprint diverged: {input_fingerprints}")
    if parameter_probe_hashes != {expected_parameter_probe}:
        raise RuntimeError(
            f"checkpoint parameter probe differs from frozen value: {parameter_probe_hashes}"
        )

    if len(intervention_site_audits) != 84:
        raise RuntimeError(
            f"expected 84 intervention response audits, got {len(intervention_site_audits)}"
        )
    active_response_count = sum(
        int(audit["active_site_count"] > 0)
        for audit in intervention_site_audits.values()
    )
    active_site_count = sum(
        int(audit["active_site_count"])
        for audit in intervention_site_audits.values()
    )
    if active_response_count != EXPECTED_ACTIVE_RESPONSES_PER_STATE:
        raise RuntimeError(
            f"expected {EXPECTED_ACTIVE_RESPONSES_PER_STATE} active responses, got "
            f"{active_response_count}"
        )
    if active_site_count != EXPECTED_ACTIVE_SITES_PER_STATE:
        raise RuntimeError(
            f"expected {EXPECTED_ACTIVE_SITES_PER_STATE} active sites, got {active_site_count}"
        )
    site_input_max = max(
        float(audit["model_input_max_error"])
        for audit in intervention_site_audits.values()
    )
    site_velocity_max = max(
        float(audit["returned_velocity_max_error"])
        for audit in intervention_site_audits.values()
    )
    if (
        site_input_max > INTERVENTION_SITE_ERROR_TOLERANCE
        or site_velocity_max > INTERVENTION_SITE_ERROR_TOLERANCE
    ):
        raise RuntimeError("aggregate intervention-site fidelity gate failed")

    active_residual_max_abs = {
        label: float(response["research_final_sampler_target_max_abs_error"])
        for label, response in responses.items()
        if response.get("research_mode") in {"self", "dose"}
    }
    active_residual_l2 = {
        label: float(response["research_final_sampler_target_l2"])
        for label, response in responses.items()
        if response.get("research_mode") in {"self", "dose"}
    }
    if (
        len(active_residual_max_abs) != EXPECTED_ACTIVE_RESPONSES_PER_STATE
        or len(active_residual_l2) != EXPECTED_ACTIVE_RESPONSES_PER_STATE
        or not np.isfinite(list(active_residual_max_abs.values())).all()
        or not np.isfinite(list(active_residual_l2.values())).all()
    ):
        raise RuntimeError("final sampler residual census/finiteness failed")

    action_coordinate_errors = {
        label: {
            "input": float(response["research_maximum_action_input_error"]),
            "output": float(response["research_maximum_action_output_error"]),
        }
        for label, response in responses.items()
        if "research_maximum_action_input_error" in response
    }
    if len(action_coordinate_errors) != 84 or any(
        row != {"input": 0.0, "output": 0.0}
        for row in action_coordinate_errors.values()
    ):
        raise RuntimeError("action-coordinate audit census/nonwrite gate failed")

    native_by_id = {
        f"{study_id}-native-{seed}": native[seed] for seed in branch_seeds
    }
    schedule_identity_count = 0
    for label, response in responses.items():
        if response.get("research_mode") not in {"none", "self", "dose"}:
            continue
        recipient_id = str(response.get("research_recipient_id"))
        if recipient_id not in native_by_id:
            raise RuntimeError(f"{label}: recipient ID does not resolve to a native record")
        recipient_native = native_by_id[recipient_id]
        if (
            _canonical_signature_value(response.get("research_sigmas"))
            != _canonical_signature_value(recipient_native.get("research_sigmas"))
            or _canonical_signature_value(response.get("research_x0_sigmas"))
            != _canonical_signature_value(recipient_native.get("research_x0_sigmas"))
        ):
            raise RuntimeError(f"{label}: recipient solver schedule differs")
        schedule_identity_count += 1
    if schedule_identity_count != 84:
        raise RuntimeError(
            f"recipient schedule-identity census is {schedule_identity_count}, expected 84"
        )

    report = {
        "status": "complete",
        "admission": manifest["admission"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.expected_manifest_sha256,
        "authorization_audit_path": str(args.audit_report.resolve()),
        "authorization_audit_sha256": args.expected_audit_sha256,
        "unit_id": unit["unit_id"],
        "task": unit["task"],
        "environment_seed": unit["environment_seed"],
        "episode_id": unit["episode_id"],
        "phase": unit["phase"],
        "phase_fraction": unit["phase_fraction"],
        "branch_step": unit["branch_step"],
        "scope": manifest["scope"],
        "input": input_audit,
        "branch_seeds": list(branch_seeds),
        "ordered_pairs": [list(pair) for pair in pairs],
        "native_action_pair_l2": native_axis_l2,
        "degenerate_native_action_axis_count": 0,
        "alpha_grid": list(ALPHAS),
        "request_count": len(responses),
        "request_sequence": request_trace,
        "request_class_census": {
            "native": 4,
            "native_replay": 4,
            "none": 4,
            "self": 4,
            "self_replay": 4,
            "dose": 60,
            "midpoint_replay": 12,
        },
        "response_actions": {
            label: np.asarray(response["action"], dtype=np.float32).tolist()
            for label, response in responses.items()
        },
        "response_metadata": {
            label: response_metadata(response) for label, response in responses.items()
        },
        "wire_schema_validated_response_count": len(responses),
        "recipient_schedule_identity_count": schedule_identity_count,
        "fixed_recipient_noise": True,
        "native_actions": {
            str(seed): native_actions[seed].tolist() for seed in branch_seeds
        },
        "native_server": {
            str(seed): response_metadata(native[seed]) for seed in branch_seeds
        },
        "native_future_hashes": {
            str(seed): str(native[seed]["research_future_hash"])
            for seed in branch_seeds
        },
        "native_future_hashes_distinct": len(
            {str(native[seed]["research_future_hash"]) for seed in branch_seeds}
        )
        == len(branch_seeds),
        "native_repeat_gates": native_repeat_gates,
        "none_controls": none_rows,
        "self_controls": self_rows,
        "dose_rows": dose_rows,
        "alpha_zero_routing_controls": alpha_zero_control_rows,
        "midpoint_replay_controls": midpoint_replay_rows,
        "input_fingerprint_count": len(input_fingerprints),
        "input_fingerprints": sorted(input_fingerprints),
        "parameter_probe_hash_count": len(parameter_probe_hashes),
        "parameter_probe_hashes": sorted(parameter_probe_hashes),
        "expected_parameter_probe_hash": expected_parameter_probe,
        "intervention_response_count": len(intervention_site_audits),
        "active_intervention_response_count": active_response_count,
        "active_intervention_site_count": active_site_count,
        "intervention_site_error_tolerance": INTERVENTION_SITE_ERROR_TOLERANCE,
        "model_input_future_clamp_max_error": site_input_max,
        "returned_future_velocity_overwrite_max_error": site_velocity_max,
        "intervention_site_audits": intervention_site_audits,
        "action_coordinate_errors": action_coordinate_errors,
        "final_sampler_target_max_abs_errors": active_residual_max_abs,
        "final_sampler_target_l2_errors": active_residual_l2,
        "structural_null_schema": {
            "research_action_donor_projection": (
                "null iff recipient and donor are identical in none/self controls; "
                "finite for all off-diagonal dose responses"
            ),
            "expected_null_count": 12,
            "expected_finite_count": 72,
            "expected_absent_native_count": 8,
        },
    }
    atomic_json(args.output, json_safe(report))
    print(
        json.dumps(
            {
                "status": "complete",
                "unit_id": unit["unit_id"],
                "output": str(args.output),
                "request_count": len(responses),
                "active_intervention_site_count": active_site_count,
            },
            sort_keys=True,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    dose_main()
