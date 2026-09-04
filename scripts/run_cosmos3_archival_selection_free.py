#!/usr/bin/env python3
"""Run one frozen archival, selection-free Cosmos 3 action-only state.

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
    deterministic_shuffled_source,
    deterministic_wrong_donor,
    recorded_proprio,
    sha256,
)
from imagined_future.cosmos3_protocol import (
    directional_target_metrics,
    ordered_recipient_donor_pairs,
)


GAUSSIAN_SEED = 1223
GAUSSIAN_TOLERANCE = 1e-5
INTERVENTION_SITE_ERROR_TOLERANCE = 1e-7
EXPECTED_DENOISING_CALLS = 4
EXPECTED_FUTURE_FRAME_INDICES = tuple(range(1, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
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
        "maximum_action_input_error": action_errors[0],
        "maximum_action_output_error": action_errors[1],
        "action_input_errors": action_input_errors.tolist(),
        "action_output_errors": action_output_errors.tolist(),
        "inactive_wrapper_write_count": 0,
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
    if manifest.get("selection_uses_model_or_intervention_outcomes") is not False:
        raise ValueError("manifest does not explicitly prohibit outcome-based selection")
    if manifest.get("scope", {}).get("physical_endpoint_evidence") is not False:
        raise ValueError("archival runner cannot make a physical-endpoint claim")
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


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite completed unit: {args.output}")
    manifest, unit = validate_manifest(args)
    base, input_audit = build_request(unit, args.screen_root)
    branch_seeds = [int(seed) for seed in unit["branch_seeds"]]
    expected_pairs = ordered_recipient_donor_pairs(branch_seeds)
    manifest_pairs = [tuple(int(value) for value in pair) for pair in unit["ordered_pairs"]]
    if manifest_pairs != expected_pairs or len(expected_pairs) != 12:
        raise ValueError("unit does not contain the canonical 12 ordered donor pairs")
    expected_retrieval_cells = [
        (recipient, source) for recipient in branch_seeds for source in branch_seeds
    ]
    manifest_retrieval_cells = [
        tuple(int(value) for value in cell)
        for cell in unit["future_source_retrieval_cells"]
    ]
    if manifest_retrieval_cells != expected_retrieval_cells:
        raise ValueError("unit does not contain the canonical 4x4 future-source grid")

    client = WebsocketClientPolicy(args.host, args.port)
    study_id = f"{manifest['manifest_id']}-{unit['unit_id']}"
    responses: dict[str, dict[str, Any]] = {}
    intervention_site_audits: dict[str, dict[str, Any]] = {}

    def infer(label: str, **research: Any) -> dict[str, Any]:
        if label in responses:
            raise RuntimeError(f"duplicate response label: {label}")
        request = {**base, "research_id": f"{study_id}-{label}", **research}
        response = client.infer(request)
        for key in (
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
        responses[label] = response
        return response

    native: dict[int, dict[str, Any]] = {}
    native_repeat: dict[int, dict[str, Any]] = {}
    for seed in branch_seeds:
        native[seed] = infer(
            f"native-{seed}",
            research_mode="native",
            research_seed=seed,
            research_id=f"{study_id}-native-{seed}",
        )
        native_repeat[seed] = infer(
            f"native-repeat-{seed}", research_mode="native", research_seed=seed
        )

    native_actions = {
        seed: np.asarray(response["action"], dtype=np.float32) for seed, response in native.items()
    }
    native_repeat_errors: dict[str, float] = {}
    native_future_replay_exact: dict[str, bool] = {}
    native_deterministic_metadata_replay_exact: dict[str, bool] = {}
    for seed in branch_seeds:
        error = maximum_error(native_actions[seed], np.asarray(native_repeat[seed]["action"]))
        native_repeat_errors[str(seed)] = error
        native_future_replay_exact[str(seed)] = all(
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
        native_deterministic_metadata_replay_exact[str(seed)] = (
            signature(native[seed]) == signature(native_repeat[seed])
        )
        if (
            error != 0.0
            or not native_future_replay_exact[str(seed)]
            or not native_deterministic_metadata_replay_exact[str(seed)]
        ):
            raise RuntimeError(f"native repeat gate failed for seed {seed}")

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
            recipient_initial_state_hash=str(
                native[seed]["research_initial_state_hash"]
            ),
        )
        action_error = maximum_error(
            np.asarray(response["action"], dtype=np.float32), native_actions[seed]
        )
        future_exact = (
            response["research_output_future_hash"]
            == native[seed]["research_future_hash"]
        )
        x0_exact = (
            _canonical_signature_value(response["research_x0_vision_hashes"])
            == _canonical_signature_value(native[seed]["research_x0_vision_hashes"])
            and _canonical_signature_value(response["research_x0_action_hashes"])
            == _canonical_signature_value(native[seed]["research_x0_action_hashes"])
        )
        sigma_exact = (
            _canonical_signature_value(response["research_sigmas"])
            == _canonical_signature_value(native[seed]["research_sigmas"])
            and _canonical_signature_value(response["research_x0_sigmas"])
            == _canonical_signature_value(native[seed]["research_x0_sigmas"])
        )
        trace_signature = {
            "action": np.asarray(response["action"], dtype=np.float32),
            "future_hash": response["research_output_future_hash"],
            "x0_vision_hashes": response["research_x0_vision_hashes"],
            "x0_action_hashes": response["research_x0_action_hashes"],
            "sigmas": response["research_sigmas"],
            "x0_sigmas": response["research_x0_sigmas"],
        }
        native_trace_signature = {
            "action": native_actions[seed],
            "future_hash": native[seed]["research_future_hash"],
            "x0_vision_hashes": native[seed]["research_x0_vision_hashes"],
            "x0_action_hashes": native[seed]["research_x0_action_hashes"],
            "sigmas": native[seed]["research_sigmas"],
            "x0_sigmas": native[seed]["research_x0_sigmas"],
        }
        trace_signature_exact = (
            _canonical_signature_value(trace_signature)
            == _canonical_signature_value(native_trace_signature)
        )
        none_rows.append(
            {
                "recipient_seed": seed,
                "active_call_count": 0,
                "action_maximum_error_vs_native": action_error,
                "future_exact_vs_native": future_exact,
                "x0_exact_vs_native": x0_exact,
                "sigma_exact_vs_native": sigma_exact,
                "trace_signature_exact_vs_native": trace_signature_exact,
                "server": response_metadata(response),
            }
        )
        if not (
            action_error == 0.0
            and future_exact
            and x0_exact
            and sigma_exact
            and trace_signature_exact
        ):
            raise RuntimeError(f"zero-active-site no-op gate failed for seed {seed}")

    self_rows: list[dict[str, Any]] = []
    for seed in branch_seeds:
        common = {
            "research_mode": "self",
            "research_seed": seed,
            "research_recipient_id": f"{study_id}-native-{seed}",
            "research_donor_id": f"{study_id}-native-{seed}",
        }
        first_label = f"self-{seed}"
        repeat_label = f"self-repeat-{seed}"
        first = infer(first_label, **common)
        repeat = infer(repeat_label, **common)
        for label, response in ((first_label, first), (repeat_label, repeat)):
            intervention_site_audits[label] = validate_intervention_site_audit(
                response,
                label=label,
                active=True,
                target_source="recipient",
                recipient_id=common["research_recipient_id"],
                donor_id=common["research_donor_id"],
                recipient_future_hash=str(native[seed]["research_future_hash"]),
                donor_future_hash=str(native[seed]["research_future_hash"]),
                recipient_path_noise_hash=str(native[seed]["research_path_noise_hash"]),
                recipient_initial_state_hash=str(
                    native[seed]["research_initial_state_hash"]
                ),
            )
        first_action = np.asarray(first["action"], dtype=np.float32)
        distances = {
            str(candidate): float(np.linalg.norm(first_action - native_actions[candidate]))
            for candidate in branch_seeds
        }
        nearest_seed = min(branch_seeds, key=lambda candidate: (distances[str(candidate)], candidate))
        shuffled_source = deterministic_shuffled_source(seed, branch_seeds)
        repeat_error = maximum_error(first_action, np.asarray(repeat["action"]))
        signature_exact = signature(first) == signature(repeat)
        target_matches_native = first["research_target_hash"] == native[seed]["research_future_hash"]
        self_rows.append(
            {
                "recipient_seed": seed,
                "future_source_seed": seed,
                "source_relation": "self",
                "distances_to_native_actions": distances,
                "nearest_native_seed": nearest_seed,
                "correct_future_source_top1": nearest_seed == seed,
                "shuffled_source_seed": shuffled_source,
                "shuffled_source_top1": nearest_seed == shuffled_source,
                "repeat_action_maximum_error": repeat_error,
                "repeat_signature_exact": signature_exact,
                "target_matches_native_future": target_matches_native,
                "clean_clamp_vs_unconstrained_native_maximum_error": maximum_error(
                    first_action, native_actions[seed]
                ),
                "clean_clamp_vs_unconstrained_native_l2": float(
                    np.linalg.norm(first_action - native_actions[seed])
                ),
                "final_sampler_target_max_abs_error": float(
                    first["research_final_sampler_target_max_abs_error"]
                ),
                "final_sampler_target_l2": float(
                    first["research_final_sampler_target_l2"]
                ),
                "action": first_action.tolist(),
                "server": response_metadata(first),
            }
        )
        if repeat_error != 0.0 or not signature_exact or not target_matches_native:
            raise RuntimeError(f"clean self-clamp replay gate failed for seed {seed}")

    donor_rows: list[dict[str, Any]] = []
    gaussian_rows: list[dict[str, Any]] = []
    for recipient_seed, donor_seed in expected_pairs:
        common = {
            "research_seed": recipient_seed,
            "research_recipient_id": f"{study_id}-native-{recipient_seed}",
            "research_donor_id": f"{study_id}-native-{donor_seed}",
        }
        label = f"recipient-{recipient_seed}-donor-{donor_seed}"
        donor_response = infer(label, research_mode="donor", **common)
        repeat_label = f"{label}-repeat"
        donor_repeat = infer(repeat_label, research_mode="donor", **common)
        for audit_label, response in (
            (label, donor_response),
            (repeat_label, donor_repeat),
        ):
            intervention_site_audits[audit_label] = validate_intervention_site_audit(
                response,
                label=audit_label,
                active=True,
                target_source="donor",
                recipient_id=common["research_recipient_id"],
                donor_id=common["research_donor_id"],
                recipient_future_hash=str(native[recipient_seed]["research_future_hash"]),
                donor_future_hash=str(native[donor_seed]["research_future_hash"]),
                recipient_path_noise_hash=str(
                    native[recipient_seed]["research_path_noise_hash"]
                ),
                recipient_initial_state_hash=str(
                    native[recipient_seed]["research_initial_state_hash"]
                ),
            )
        action = np.asarray(donor_response["action"], dtype=np.float32)
        repeat_error = maximum_error(action, np.asarray(donor_repeat["action"]))
        signature_exact = signature(donor_response) == signature(donor_repeat)
        target_matches_native = (
            donor_response["research_target_hash"] == native[donor_seed]["research_future_hash"]
        )
        metrics = directional_target_metrics(
            action, native_actions[recipient_seed], native_actions[donor_seed]
        )
        distances = {
            str(seed): float(np.linalg.norm(action - native_actions[seed]))
            for seed in branch_seeds
        }
        nearest_seed = min(branch_seeds, key=lambda seed: (distances[str(seed)], seed))
        wrong_seed = deterministic_wrong_donor(recipient_seed, donor_seed, branch_seeds)
        shuffled_source = deterministic_shuffled_source(donor_seed, branch_seeds)
        donor_rows.append(
            {
                "recipient_seed": recipient_seed,
                "future_source_seed": donor_seed,
                "source_relation": "donor",
                "target_donor_seed": donor_seed,
                "frozen_wrong_donor_seed": wrong_seed,
                "shuffled_source_seed": shuffled_source,
                "normalized_projection": normalized_projection(
                    action, native_actions[recipient_seed], native_actions[donor_seed]
                ),
                **metrics,
                "l2_from_recipient": float(
                    np.linalg.norm(action - native_actions[recipient_seed])
                ),
                "distances_to_native_actions": distances,
                "nearest_native_seed": nearest_seed,
                "correct_donor_top1": nearest_seed == donor_seed,
                "correct_future_source_top1": nearest_seed == donor_seed,
                "wrong_donor_top1": nearest_seed == wrong_seed,
                "shuffled_source_top1": nearest_seed == shuffled_source,
                "repeat_action_maximum_error": repeat_error,
                "repeat_signature_exact": signature_exact,
                "target_matches_native_future": target_matches_native,
                "final_sampler_target_max_abs_error": float(
                    donor_response["research_final_sampler_target_max_abs_error"]
                ),
                "final_sampler_target_l2": float(
                    donor_response["research_final_sampler_target_l2"]
                ),
                "action": action.tolist(),
                "server": response_metadata(donor_response),
            }
        )
        if repeat_error != 0.0 or not signature_exact or not target_matches_native:
            raise RuntimeError(f"donor replay gate failed for pair {(recipient_seed, donor_seed)}")

        gaussian_label = f"gaussian-{label}"
        gaussian = infer(
            gaussian_label,
            research_mode="gaussian",
            research_gaussian_seed=GAUSSIAN_SEED,
            **common,
        )
        intervention_site_audits[gaussian_label] = validate_intervention_site_audit(
            gaussian,
            label=gaussian_label,
            active=True,
            target_source="gaussian_geometry",
            recipient_id=common["research_recipient_id"],
            donor_id=common["research_donor_id"],
            recipient_future_hash=str(native[recipient_seed]["research_future_hash"]),
            donor_future_hash=str(native[donor_seed]["research_future_hash"]),
            recipient_path_noise_hash=str(
                native[recipient_seed]["research_path_noise_hash"]
            ),
            recipient_initial_state_hash=str(
                native[recipient_seed]["research_initial_state_hash"]
            ),
        )
        gaussian_action = np.asarray(gaussian["action"], dtype=np.float32)
        gaussian_metrics = directional_target_metrics(
            gaussian_action, native_actions[recipient_seed], native_actions[donor_seed]
        )
        gaussian_distances = {
            str(seed): float(np.linalg.norm(gaussian_action - native_actions[seed]))
            for seed in branch_seeds
        }
        gaussian_nearest = min(
            branch_seeds, key=lambda seed: (gaussian_distances[str(seed)], seed)
        )
        norm_error = float(gaussian["research_gaussian_norm_relative_error"])
        distance_error = float(gaussian["research_gaussian_distance_relative_error"])
        gaussian_rows.append(
            {
                "recipient_seed": recipient_seed,
                "target_donor_seed": donor_seed,
                "gaussian_seed": GAUSSIAN_SEED,
                "normalized_projection": normalized_projection(
                    gaussian_action,
                    native_actions[recipient_seed],
                    native_actions[donor_seed],
                ),
                **gaussian_metrics,
                "l2_from_recipient": float(
                    np.linalg.norm(gaussian_action - native_actions[recipient_seed])
                ),
                "distances_to_native_actions": gaussian_distances,
                "nearest_native_seed": gaussian_nearest,
                "correct_donor_top1": gaussian_nearest == donor_seed,
                "norm_relative_error": norm_error,
                "distance_relative_error": distance_error,
                "final_sampler_target_max_abs_error": float(
                    gaussian["research_final_sampler_target_max_abs_error"]
                ),
                "final_sampler_target_l2": float(
                    gaussian["research_final_sampler_target_l2"]
                ),
                "action": gaussian_action.tolist(),
                "server": response_metadata(gaussian),
            }
        )
        if (
            not math.isfinite(norm_error)
            or not math.isfinite(distance_error)
            or norm_error > GAUSSIAN_TOLERANCE
            or distance_error > GAUSSIAN_TOLERANCE
        ):
            raise RuntimeError(
                f"Gaussian geometry gate failed for pair {(recipient_seed, donor_seed)}: "
                f"{norm_error}, {distance_error}"
            )

    all_responses = list(responses.values())
    input_fingerprints = {str(response["research_state_hash"]) for response in all_responses}
    if len(input_fingerprints) != 1:
        raise RuntimeError(f"transformed input fingerprint diverged: {input_fingerprints}")
    parameter_probe_hashes = {
        str(response["research_parameter_probe_hash"]) for response in all_responses
    }
    expected_parameter_probe = str(manifest["runtime"]["expected_parameter_probe_hash"])
    if parameter_probe_hashes != {expected_parameter_probe}:
        raise RuntimeError(
            f"checkpoint parameter probe differs from frozen value: {parameter_probe_hashes}"
        )
    action_coordinate_errors: dict[str, dict[str, float]] = {}
    final_sampler_target_max_abs_residuals: dict[str, float] = {}
    final_sampler_target_l2_residuals: dict[str, float] = {}
    for label, response in responses.items():
        if response.get("research_mode") in {"self", "donor", "gaussian"}:
            max_abs = float(response["research_final_sampler_target_max_abs_error"])
            l2 = float(response["research_final_sampler_target_l2"])
            if not math.isfinite(max_abs) or not math.isfinite(l2):
                raise RuntimeError(f"{label} final sampler residual is not finite")
            final_sampler_target_max_abs_residuals[label] = max_abs
            final_sampler_target_l2_residuals[label] = l2
        if "research_maximum_action_input_error" not in response:
            continue
        errors = {
            "input": float(response["research_maximum_action_input_error"]),
            "output": float(response["research_maximum_action_output_error"]),
        }
        action_coordinate_errors[label] = errors
        if errors != {"input": 0.0, "output": 0.0}:
            raise RuntimeError(f"{label} directly mutated action coordinates: {errors}")
    if len(action_coordinate_errors) != 48:
        raise RuntimeError(
            f"expected coordinate audits for exactly 48 interventional calls, got "
            f"{len(action_coordinate_errors)}"
        )
    if (
        len(final_sampler_target_max_abs_residuals) != 44
        or len(final_sampler_target_l2_residuals) != 44
    ):
        raise RuntimeError(
            f"expected final residuals for exactly 44 active-clamp calls, got "
            f"{len(final_sampler_target_max_abs_residuals)} max-abs and "
            f"{len(final_sampler_target_l2_residuals)} l2"
        )
    if len(intervention_site_audits) != 48:
        raise RuntimeError(
            f"expected intervention-site audits for exactly 48 calls, got "
            f"{len(intervention_site_audits)}"
        )
    active_site_count = sum(
        int(audit["active_site_count"])
        for audit in intervention_site_audits.values()
    )
    if active_site_count != 44 * EXPECTED_DENOISING_CALLS:
        raise RuntimeError(
            f"expected {44 * EXPECTED_DENOISING_CALLS} active clamp sites, got "
            f"{active_site_count}"
        )
    input_site_max_error = max(
        float(audit["model_input_max_error"])
        for audit in intervention_site_audits.values()
    )
    velocity_site_max_error = max(
        float(audit["returned_velocity_max_error"])
        for audit in intervention_site_audits.values()
    )
    def describe_residuals(values: dict[str, float]) -> dict[str, Any]:
        array = np.asarray(list(values.values()), dtype=np.float64)
        return {
            "count": int(len(array)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "quantiles": {
                str(q): float(np.quantile(array, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
        }

    max_abs_residual_summary = {
        **describe_residuals(final_sampler_target_max_abs_residuals),
        "count_gt_0_03": int(
            np.count_nonzero(
                np.asarray(
                    list(final_sampler_target_max_abs_residuals.values()),
                    dtype=np.float64,
                )
                > 0.03
            )
        ),
    }
    l2_residual_summary = {
        **describe_residuals(final_sampler_target_l2_residuals),
        "count_gt_0_03": int(
            np.count_nonzero(
                np.asarray(
                    list(final_sampler_target_l2_residuals.values()),
                    dtype=np.float64,
                )
                > 0.03
            )
        ),
    }

    native_pair_l2 = {
        f"{recipient}:{donor}": float(
            np.linalg.norm(native_actions[donor] - native_actions[recipient])
        )
        for recipient, donor in expected_pairs
    }
    self_by_cell = {
        (int(row["recipient_seed"]), int(row["future_source_seed"])): row
        for row in self_rows
    }
    donor_by_cell = {
        (int(row["recipient_seed"]), int(row["future_source_seed"])): row
        for row in donor_rows
    }
    retrieval_rows = []
    for cell in expected_retrieval_cells:
        source_row = self_by_cell.get(cell) or donor_by_cell.get(cell)
        if source_row is None:
            raise RuntimeError(f"missing future-source retrieval cell {cell}")
        retrieval_rows.append(
            {
                key: source_row[key]
                for key in (
                    "recipient_seed",
                    "future_source_seed",
                    "source_relation",
                    "distances_to_native_actions",
                    "nearest_native_seed",
                    "correct_future_source_top1",
                    "shuffled_source_seed",
                    "shuffled_source_top1",
                )
            }
        )
    if len(retrieval_rows) != 16:
        raise RuntimeError("future-source retrieval grid is not 4x4")
    report = {
        "status": "complete",
        "admission": manifest["admission"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.expected_manifest_sha256,
        "unit_id": unit["unit_id"],
        "task": unit["task"],
        "environment_seed": unit["environment_seed"],
        "episode_id": unit["episode_id"],
        "phase": unit["phase"],
        "phase_fraction": unit["phase_fraction"],
        "branch_step": unit["branch_step"],
        "scope": manifest["scope"],
        "input": input_audit,
        "branch_seeds": branch_seeds,
        "ordered_pairs": [list(pair) for pair in expected_pairs],
        "future_source_retrieval_cells": [
            list(cell) for cell in expected_retrieval_cells
        ],
        "future_source_retrieval_rows": retrieval_rows,
        "fixed_recipient_noise": True,
        "request_count": len(responses),
        "native_actions": {str(seed): native_actions[seed].tolist() for seed in branch_seeds},
        "native_action_pair_l2": native_pair_l2,
        "native_future_hashes_distinct": len(
            {str(native[seed]["research_future_hash"]) for seed in branch_seeds}
        )
        == len(branch_seeds),
        "native_future_hashes": {
            str(seed): str(native[seed]["research_future_hash"]) for seed in branch_seeds
        },
        "native_repeat_action_maximum_error": native_repeat_errors,
        "native_future_replay_exact": native_future_replay_exact,
        "native_deterministic_metadata_replay_exact": (
            native_deterministic_metadata_replay_exact
        ),
        "none_controls": none_rows,
        "self_controls": self_rows,
        "donor_rows": donor_rows,
        "gaussian_rows": gaussian_rows,
        "shuffle_control": {
            "construction": (
                "balanced cyclic derangement across all four future-source labels; "
                "mapping uses only frozen branch-seed order"
            ),
            "mapping": [
                {
                    "source_seed": source,
                    "shuffled_source_seed": deterministic_shuffled_source(
                        source, branch_seeds
                    ),
                }
                for source in branch_seeds
            ],
        },
        "input_fingerprint_count": len(input_fingerprints),
        "input_fingerprints": sorted(input_fingerprints),
        "parameter_probe_hash_count": len(parameter_probe_hashes),
        "parameter_probe_hashes": sorted(parameter_probe_hashes),
        "expected_parameter_probe_hash": expected_parameter_probe,
        "intervention_site_error_tolerance": INTERVENTION_SITE_ERROR_TOLERANCE,
        "intervention_site_audits": intervention_site_audits,
        "active_intervention_response_count": 44,
        "active_intervention_site_count": active_site_count,
        "model_input_future_clamp_max_error": input_site_max_error,
        "returned_future_velocity_overwrite_max_error": velocity_site_max_error,
        "final_sampler_target_max_abs_errors": final_sampler_target_max_abs_residuals,
        "final_sampler_target_l2_errors": final_sampler_target_l2_residuals,
        "final_sampler_target_residual_summary": {
            "max_abs": max_abs_residual_summary,
            "l2": l2_residual_summary,
        },
        "action_coordinate_errors": action_coordinate_errors,
        "native_server": {
            str(seed): response_metadata(native[seed]) for seed in branch_seeds
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
                "input_fingerprint_count": len(input_fingerprints),
                "parameter_probe_hash_count": len(parameter_probe_hashes),
            },
            sort_keys=True,
        )
    )
    # WebsocketClientPolicy owns a receive thread without a public close method.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
