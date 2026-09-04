#!/usr/bin/env python3
"""Run one frozen archival Cosmos 3 single-call timing state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

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
from imagined_future.cosmos3_single_call_timing import (
    ACTION_COORDINATE_COUNT,
    ACTION_SHAPE,
    BRANCH_SEEDS,
    RESEARCH_SIGMAS,
    REQUESTS_PER_STATE,
    TIMING_CONDITIONS,
    all_finite,
    expected_request_labels,
    maximum_error,
    nearest_native_seed,
    ordered_source_cells,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
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


def build_request(
    unit: Mapping[str, Any], screen_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    assets = unit["assets"]
    episode_dir = screen_root / assets["relative_episode_directory"]
    paths = {
        "mp4": episode_dir / assets["mp4_filename"],
        "hdf5": episode_dir / assets["hdf5_filename"],
        "env_cfg": episode_dir / assets["env_cfg_filename"],
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != unit["input_sha256"][label]:
            raise ValueError(f"{label} SHA mismatch for {unit['unit_id']}")
    frame = read_frame(paths["mp4"], int(unit["mp4_frame_index"]))
    image = compose_cosmos_observation(frame)
    with h5py.File(paths["hdf5"], "r") as stream:
        dataset = stream["data/demo_0/states/articulation/robot/joint_position"]
        raw_proprio = np.asarray(dataset[int(unit["hdf5_state_index"])], dtype=np.float32)
    joints, gripper = recorded_proprio(raw_proprio)
    request = {
        "observation/image": image,
        "observation/joint_position": joints,
        "observation/gripper_position": gripper,
        "prompt": str(unit["instruction"]),
    }
    audit = {
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "joint_position_sha256": hashlib.sha256(joints.tobytes()).hexdigest(),
        "gripper_position_sha256": hashlib.sha256(gripper.tobytes()).hexdigest(),
        "reconstruction": (
            "lossy H.264 MP4 frame branch_step-1; wrist above bilinear half-scale left/right"
        ),
        "proprio": "noise-free recorded post-step joint state at branch_step-1",
    }
    return request, audit


def canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return canonical_value(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("signature contains a nonfinite float")
        return value
    return value


def deterministic_signature(
    response: Mapping[str, Any], *, exclude: set[str] | None = None
) -> str:
    excluded = {
        "action",
        "video",
        "research_id",
        "research_infer_ms",
        "server_timing",
    }
    if exclude:
        excluded.update(exclude)
    payload = {
        key: canonical_value(value)
        for key, value in response.items()
        if key not in excluded
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def behavior_signature(response: Mapping[str, Any]) -> str:
    """Hash no-op outputs while excluding intentionally varying source metadata."""

    keys = (
        "research_output_future_hash",
        "research_sigmas",
        "research_x0_sigmas",
        "research_x0_vision_hashes",
        "research_x0_action_hashes",
        "research_recipient_path_noise_hash",
        "research_initial_state_hash",
        "research_state_hash",
    )
    payload = {key: canonical_value(response[key]) for key in keys}
    payload["action_sha256"] = hashlib.sha256(
        np.asarray(response["action"], dtype=np.float32).tobytes()
    ).hexdigest()
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: canonical_value(value)
        for key, value in response.items()
        if key not in {"action", "video"}
    }


def null_paths(value: Any, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if value is None:
        return {path}
    if isinstance(value, Mapping):
        output: set[tuple[str, ...]] = set()
        for key, item in value.items():
            output.update(null_paths(item, path + (str(key),)))
        return output
    if isinstance(value, (list, tuple)):
        output = set()
        for index, item in enumerate(value):
            output.update(null_paths(item, path + (str(index),)))
        return output
    return set()


def allowed_routing_null_paths(response: Mapping[str, Any]) -> set[tuple[str, ...]]:
    """Validate and return the sole nonnumeric routing-null path."""

    path = ("research_attention_interface", "cache_id")
    interface = response.get("research_attention_interface")
    if not isinstance(interface, Mapping):
        raise RuntimeError("response lacks research_attention_interface mapping")
    if interface.get("cache_id") is not None:
        return set()
    if (
        interface.get("instrumented_server") is not False
        or interface.get("intervention_requested") is not False
        or interface.get("mode") != "exclude"
    ):
        raise RuntimeError("attention cache_id is null outside the inactive routing case")
    return {path}


def validate_manifest(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = sha256(args.manifest)
    if actual != args.expected_manifest_sha256:
        raise ValueError(f"manifest SHA mismatch: {actual} != {args.expected_manifest_sha256}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_model_outcomes":
        raise ValueError("manifest is not frozen before outcomes")
    if manifest.get("study_name") != "cosmos3-single-call-timing-v5":
        raise ValueError("manifest is not the timing v5 study")
    if manifest.get("selection_uses_model_or_intervention_outcomes") is not False:
        raise ValueError("manifest does not prohibit outcome-based selection")
    if int(manifest["runtime"]["server_port"]) != args.port:
        raise ValueError("CLI port differs from frozen server port")
    if sha256(Path(__file__).resolve()) != manifest["runtime"]["runner_sha256"]:
        raise ValueError("runner differs from frozen manifest")
    matches = [unit for unit in manifest["states"] if unit["unit_id"] == args.unit_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest state {args.unit_id!r}, got {len(matches)}")
    return manifest, matches[0]


def require_exact_array(
    response: Mapping[str, Any], key: str, expected: np.ndarray, label: str
) -> None:
    if key not in response:
        raise ValueError(f"{label}: missing {key}")
    actual = np.asarray(response[key], dtype=expected.dtype)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"{label}: {key} differs: {actual.tolist()} != {expected.tolist()}")


def require_exact_action(response: Mapping[str, Any], label: str) -> np.ndarray:
    if "action" not in response or response["action"] is None:
        raise RuntimeError(f"{label}: missing action")
    try:
        action = np.asarray(response["action"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label}: action is not a rectangular numeric array") from error
    if action.shape != ACTION_SHAPE or action.size != ACTION_COORDINATE_COUNT:
        raise RuntimeError(
            f"{label}: action shape/count {action.shape}/{action.size} differs from "
            f"{ACTION_SHAPE}/{ACTION_COORDINATE_COUNT}"
        )
    if not np.all(np.isfinite(action)):
        raise RuntimeError(f"{label}: action contains NaN or infinity")
    return action


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {args.output}")
    manifest, unit = validate_manifest(args)
    base, input_audit = build_request(unit, args.screen_root)
    seeds = tuple(int(seed) for seed in unit["branch_seeds"])
    if seeds != BRANCH_SEEDS:
        raise ValueError(f"branch order differs from frozen protocol: {seeds}")
    request_labels = expected_request_labels(seeds)
    if list(request_labels) != manifest["design"]["request_labels"]:
        raise ValueError("manifest request order differs from frozen construction")
    if (
        tuple(int(value) for value in manifest["design"].get("action_shape", []))
        != ACTION_SHAPE
        or int(manifest["design"].get("action_coordinate_count", -1))
        != ACTION_COORDINATE_COUNT
    ):
        raise ValueError("manifest action shape/count differs from frozen 32x8/256 schema")
    expected_sigmas = np.asarray(RESEARCH_SIGMAS, dtype=np.float32)
    expected_frames = np.asarray(
        manifest["design"]["future_frame_indices"], dtype=np.int64
    )
    expected_vision_count = int(manifest["design"]["vision_coordinate_count"])
    expected_mask_count = int(manifest["design"]["future_mask_coordinate_count"])
    expected_vision_shape = np.asarray(
        manifest["design"]["vision_shape"], dtype=np.int64
    )
    expected_mask_hash = str(manifest["design"]["future_mask_index_hash"])
    client = WebsocketClientPolicy(args.host, args.port)
    study_id = f"{manifest['manifest_id']}-{unit['unit_id']}"
    responses: dict[str, dict[str, Any]] = {}
    structural_projection_null_count = 0
    finite_off_diagonal_projection_count = 0
    native_projection_absent_count = 0
    shape_valid_response_action_count = 0

    def infer(
        label: str,
        *,
        expected_projection_applicable: bool | None,
        **research: Any,
    ) -> dict[str, Any]:
        nonlocal structural_projection_null_count
        nonlocal finite_off_diagonal_projection_count
        nonlocal native_projection_absent_count
        nonlocal shape_valid_response_action_count
        if label in responses:
            raise RuntimeError(f"duplicate request label: {label}")
        if label != request_labels[len(responses)]:
            raise RuntimeError(
                f"request-order violation: {label!r} != {request_labels[len(responses)]!r}"
            )
        request = {**base, "research_id": f"{study_id}-{label}", **research}
        response = client.infer(request)
        require_exact_action(response, label)
        shape_valid_response_action_count += 1
        for key in (
            "research_mode",
            "research_seed",
            "research_recipient_id",
            "research_donor_id",
        ):
            if key in request and response.get(key) != request[key]:
                raise RuntimeError(
                    f"{label}: response {key}={response.get(key)!r}, expected {request[key]!r}"
                )
        if response.get("research_state_hash") is None:
            raise RuntimeError(f"{label}: missing transformed-input fingerprint")
        if response.get("research_parameter_probe_hash") is None:
            raise RuntimeError(f"{label}: missing parameter-probe hash")
        require_exact_array(response, "research_sigmas", expected_sigmas, label)
        require_exact_array(response, "research_x0_sigmas", expected_sigmas, label)
        projection_key = "research_action_donor_projection"
        applicability_key = "research_action_donor_projection_applicable"
        routing_nulls = allowed_routing_null_paths(response)
        expected_raw_nulls = set(routing_nulls)
        if expected_projection_applicable is False:
            expected_raw_nulls.add((projection_key,))
        if null_paths(response) != expected_raw_nulls:
            raise RuntimeError(f"{label}: raw response contains an unexpected null")
        if expected_projection_applicable is None:
            if projection_key in response or applicability_key in response:
                raise RuntimeError(
                    f"{label}: native response unexpectedly has projection metadata"
                )
            native_projection_absent_count += 1
        else:
            if projection_key not in response or applicability_key not in response:
                raise RuntimeError(f"{label}: intervention response lacks projection metadata")
            if expected_projection_applicable:
                if response[applicability_key] is not True:
                    raise RuntimeError(f"{label}: off-diagonal projection is not applicable")
                raw_projection = float(response[projection_key])
                if not np.isfinite(raw_projection):
                    raise RuntimeError(f"{label}: off-diagonal projection is nonfinite")
                finite_off_diagonal_projection_count += 1
            else:
                if response[applicability_key] is not False:
                    raise RuntimeError(f"{label}: diagonal projection marked applicable")
                recipient_id = str(research.get("research_recipient_id", ""))
                donor_id = str(research.get("research_donor_id", ""))
                if not recipient_id or recipient_id != donor_id:
                    raise RuntimeError(f"{label}: structural null lacks identical record IDs")
                seed = int(research.get("research_seed", -1))
                expected_record_id = f"{study_id}-native-{seed}"
                if recipient_id != expected_record_id:
                    raise RuntimeError(
                        f"{label}: diagonal identity is not the frozen native branch"
                    )
                recipient_hash = response.get("research_recipient_future_hash")
                donor_hash = response.get("research_donor_future_hash")
                if (
                    not isinstance(recipient_hash, str)
                    or not recipient_hash
                    or recipient_hash != donor_hash
                ):
                    raise RuntimeError(
                        f"{label}: diagonal recipient/donor future identities differ"
                    )
                if response[projection_key] is not None:
                    raise RuntimeError(f"{label}: diagonal projection is not raw null")
                structural_projection_null_count += 1
        allowed_nulls = set(routing_nulls)
        if expected_projection_applicable is False:
            allowed_nulls.add((projection_key,))
        if null_paths(response) != allowed_nulls:
            raise RuntimeError(f"{label}: normalized response null-path census differs")
        if not all_finite(response):
            raise RuntimeError(f"{label}: response contains missing/nonfinite numeric data")
        responses[label] = response
        return response

    native: dict[int, dict[str, Any]] = {}
    native_replay: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        native[seed] = infer(
            f"native-{seed}",
            expected_projection_applicable=None,
            research_mode="native",
            research_seed=seed,
            research_id=f"{study_id}-native-{seed}",
        )
    for seed in seeds:
        native_replay[seed] = infer(
            f"native-replay-{seed}",
            expected_projection_applicable=None,
            research_mode="native",
            research_seed=seed,
        )

    native_actions = {
        seed: np.asarray(response["action"], dtype=np.float32)
        for seed, response in native.items()
    }
    native_shape = native_actions[seeds[0]].shape
    if native_shape != ACTION_SHAPE:
        raise RuntimeError(f"native action shape {native_shape} differs from {ACTION_SHAPE}")
    for seed in seeds:
        response = native[seed]
        action = native_actions[seed]
        if action.shape != native_shape or not np.all(np.isfinite(action)):
            raise RuntimeError(f"native action for seed {seed} is nonfinite or shape-mismatched")
        for key in (
            "research_future_hash",
            "research_path_noise_hash",
            "research_initial_state_hash",
            "research_state_hash",
            "research_parameter_probe_hash",
        ):
            if not isinstance(response.get(key), str) or not response[key]:
                raise RuntimeError(f"native seed {seed} is missing required {key}")
        for key in ("research_x0_vision_hashes", "research_x0_action_hashes"):
            values = response.get(key)
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise RuntimeError(f"native seed {seed} has invalid {key}")
            if any(not isinstance(value, str) or not value for value in values):
                raise RuntimeError(f"native seed {seed} has empty {key}")
    native_pair_l2 = {
        f"{recipient}:{source}": float(
            np.linalg.norm(
                native_actions[recipient].astype(np.float64).reshape(-1)
                - native_actions[source].astype(np.float64).reshape(-1)
            )
        )
        for recipient, source in ordered_source_cells(seeds)
        if recipient != source
    }
    if any(not np.isfinite(value) or value <= 1e-12 for value in native_pair_l2.values()):
        raise RuntimeError("native action grid contains a degenerate off-diagonal axis")
    native_replay_errors: dict[str, float] = {}
    native_replay_signatures: dict[str, bool] = {}
    for seed in seeds:
        native_replay_errors[str(seed)] = maximum_error(
            native_actions[seed], native_replay[seed]["action"]
        )
        native_replay_signatures[str(seed)] = (
            deterministic_signature(native[seed])
            == deterministic_signature(native_replay[seed])
        )
        if native_replay_errors[str(seed)] != 0.0 or not native_replay_signatures[str(seed)]:
            raise RuntimeError(f"native replay failed for seed {seed}")

    timing_rows: list[dict[str, Any]] = []
    timing_response_index: dict[tuple[str, int, int], dict[str, Any]] = {}
    maximum_action_input_error = 0.0
    maximum_action_output_error = 0.0
    maximum_model_input_error = 0.0
    maximum_velocity_error = 0.0
    inactive_wrapper_write_count = 0
    target_hash_gate_exact = True
    rng_hash_gate_exact = True
    schedule_and_index_gate_exact = True
    for timing, active_indices in TIMING_CONDITIONS:
        expected_active = np.asarray(active_indices, dtype=np.int64)
        expected_inactive = np.asarray(
            [index for index in range(4) if index not in active_indices], dtype=np.int64
        )
        expected_active_sigmas = expected_sigmas[expected_active]
        for recipient, source in ordered_source_cells(seeds):
            label = f"timing-{timing}-recipient-{recipient}-source-{source}"
            mode = "none" if timing == "none" else "self" if recipient == source else "donor"
            response = infer(
                label,
                expected_projection_applicable=recipient != source,
                research_mode=mode,
                research_seed=recipient,
                research_recipient_id=f"{study_id}-native-{recipient}",
                research_donor_id=f"{study_id}-native-{source}",
                research_timing_steps=list(active_indices),
            )
            projection = response["research_action_donor_projection"]
            if timing == "none" and recipient != source and projection != 0.0:
                raise RuntimeError(
                    f"{label}: off-diagonal none projection is not exactly zero"
                )
            timing_response_index[(timing, recipient, source)] = response
            for key in (
                "research_requested_active_call_indices",
                "research_observed_active_call_indices",
                "research_clamped_call_indices",
            ):
                require_exact_array(response, key, expected_active, label)
            require_exact_array(
                response, "research_inactive_call_indices", expected_inactive, label
            )
            require_exact_array(
                response, "research_requested_active_sigmas", expected_active_sigmas, label
            )
            require_exact_array(
                response, "research_observed_active_sigmas", expected_active_sigmas, label
            )
            require_exact_array(
                response, "research_future_frame_indices", expected_frames, label
            )
            require_exact_array(
                response, "research_vision_shape", expected_vision_shape, label
            )
            if int(response["research_vision_coordinate_count"]) != expected_vision_count:
                raise RuntimeError(f"{label}: vision coordinate count changed")
            if int(response["research_future_mask_coordinate_count"]) != expected_mask_count:
                raise RuntimeError(f"{label}: future-mask coordinate count changed")
            if response.get("research_future_mask_index_hash") != expected_mask_hash:
                raise RuntimeError(f"{label}: future-mask coordinate indices changed")
            input_errors = np.asarray(
                response["research_model_input_future_clamp_errors"], dtype=np.float64
            )
            velocity_errors = np.asarray(
                response["research_returned_future_velocity_overwrite_errors"],
                dtype=np.float64,
            )
            if input_errors.shape != expected_active.shape or velocity_errors.shape != expected_active.shape:
                raise RuntimeError(f"{label}: live intervention capture cardinality mismatch")
            if np.any(input_errors != 0.0) or np.any(velocity_errors != 0.0):
                raise RuntimeError(f"{label}: live intervention-site replacement is not exact")
            action_input_errors = np.asarray(
                response["research_action_input_errors"], dtype=np.float64
            )
            action_output_errors = np.asarray(
                response["research_action_output_errors"], dtype=np.float64
            )
            if (
                action_input_errors.shape != expected_sigmas.shape
                or action_output_errors.shape != expected_sigmas.shape
            ):
                raise RuntimeError(f"{label}: per-call action audit is not four calls")
            if (
                np.any(~np.isfinite(action_input_errors))
                or np.any(~np.isfinite(action_output_errors))
                or np.any(action_input_errors != 0.0)
                or np.any(action_output_errors != 0.0)
            ):
                raise RuntimeError(f"{label}: intervention wrote action coordinates")
            action_input_error = float(response["research_maximum_action_input_error"])
            action_output_error = float(response["research_maximum_action_output_error"])
            if action_input_error != 0.0 or action_output_error != 0.0:
                raise RuntimeError(f"{label}: intervention wrote action coordinates")
            observed_inactive_writes = int(response["research_inactive_wrapper_write_count"])
            if observed_inactive_writes != 0:
                raise RuntimeError(f"{label}: inactive call performed a wrapper write")
            expected_recipient_hash = native[recipient]["research_future_hash"]
            expected_donor_hash = native[source]["research_future_hash"]
            expected_target_hash = (
                expected_recipient_hash
                if timing == "none" or recipient == source
                else expected_donor_hash
            )
            expected_target_source = (
                "recipient" if timing == "none" or recipient == source else "donor"
            )
            expected_target_ids = [
                f"{study_id}-native-{recipient if expected_target_source == 'recipient' else source}"
            ]
            target_exact = all(
                (
                    response["research_recipient_future_hash"] == expected_recipient_hash,
                    response["research_donor_future_hash"] == expected_donor_hash,
                    response["research_target_hash"] == expected_target_hash,
                    response["research_target_source"] == expected_target_source,
                    response["research_target_source_record_ids"] == expected_target_ids,
                )
            )
            rng_exact = all(
                (
                    response["research_recipient_path_noise_hash"]
                    == native[recipient]["research_path_noise_hash"],
                    response["research_initial_state_hash"]
                    == native[recipient]["research_initial_state_hash"],
                )
            )
            if not target_exact:
                raise RuntimeError(f"{label}: target/source hashes do not match native records")
            if not rng_exact:
                raise RuntimeError(f"{label}: recipient RNG state/path hashes changed")
            action = np.asarray(response["action"], dtype=np.float32)
            final_target_max_abs = float(
                response["research_final_sampler_target_max_abs_error"]
            )
            final_target_l2 = float(response["research_final_sampler_target_l2"])
            if (
                not np.isfinite(final_target_max_abs)
                or not np.isfinite(final_target_l2)
                or final_target_max_abs < 0.0
                or final_target_l2 < 0.0
            ):
                raise RuntimeError(f"{label}: invalid descriptive final-target residual")
            nearest, distances, tied, margin = nearest_native_seed(
                action, native_actions, seeds
            )
            timing_rows.append(
                {
                    "timing_condition": timing,
                    "active_call_indices": list(active_indices),
                    "recipient_seed": recipient,
                    "source_seed": source,
                    "source_relation": "self" if recipient == source else "donor",
                    "action": action.tolist(),
                    "nearest_native_seed": nearest,
                    "distances_to_native_actions": {
                        str(seed): distance for seed, distance in distances.items()
                    },
                    "top1_tie": tied,
                    "top2_margin": margin,
                    "correct_source_top1": nearest == source,
                    "final_sampler_target_max_abs_error": final_target_max_abs,
                    "final_sampler_target_l2": final_target_l2,
                    "server": response_metadata(response),
                }
            )
            maximum_action_input_error = max(maximum_action_input_error, action_input_error)
            maximum_action_output_error = max(maximum_action_output_error, action_output_error)
            maximum_model_input_error = max(
                maximum_model_input_error,
                float(np.max(input_errors, initial=0.0)),
            )
            maximum_velocity_error = max(
                maximum_velocity_error,
                float(np.max(velocity_errors, initial=0.0)),
            )
            inactive_wrapper_write_count += observed_inactive_writes
            target_hash_gate_exact = target_hash_gate_exact and target_exact
            rng_hash_gate_exact = rng_hash_gate_exact and rng_exact

    all_calls_replays: dict[int, dict[str, Any]] = {}
    replay_errors: dict[str, float] = {}
    replay_signatures: dict[str, bool] = {}
    for seed in seeds:
        label = f"all-calls-diagonal-replay-{seed}"
        response = infer(
            label,
            expected_projection_applicable=False,
            research_mode="self",
            research_seed=seed,
            research_recipient_id=f"{study_id}-native-{seed}",
            research_donor_id=f"{study_id}-native-{seed}",
            research_timing_steps=[0, 1, 2, 3],
        )
        all_calls_replays[seed] = response
        original = timing_response_index[("all_calls", seed, seed)]
        replay_errors[str(seed)] = maximum_error(response["action"], original["action"])
        replay_signatures[str(seed)] = (
            deterministic_signature(response) == deterministic_signature(original)
        )
        if replay_errors[str(seed)] != 0.0 or not replay_signatures[str(seed)]:
            raise RuntimeError(f"all-calls diagonal replay failed for seed {seed}")

    none_noop_errors: dict[str, float] = {}
    none_source_invariance: dict[str, bool] = {}
    none_source_action_errors: dict[str, float] = {}
    for recipient in seeds:
        rows = [timing_response_index[("none", recipient, source)] for source in seeds]
        behavior = [behavior_signature(response) for response in rows]
        none_source_invariance[str(recipient)] = len(set(behavior)) == 1
        none_noop_errors[str(recipient)] = max(
            maximum_error(response["action"], native_actions[recipient])
            for response in rows
        )
        none_source_action_errors[str(recipient)] = max(
            maximum_error(rows[0]["action"], response["action"]) for response in rows[1:]
        )
        native_behavior = {
            "action": native[recipient]["action"],
            "research_output_future_hash": native[recipient]["research_future_hash"],
            "research_sigmas": native[recipient]["research_sigmas"],
            "research_x0_sigmas": native[recipient]["research_sigmas"],
            "research_x0_vision_hashes": native[recipient]["research_x0_vision_hashes"],
            "research_x0_action_hashes": native[recipient]["research_x0_action_hashes"],
            "research_recipient_path_noise_hash": native[recipient]["research_path_noise_hash"],
            "research_initial_state_hash": native[recipient]["research_initial_state_hash"],
            "research_state_hash": native[recipient]["research_state_hash"],
        }
        if (
            none_noop_errors[str(recipient)] != 0.0
            or not none_source_invariance[str(recipient)]
            or behavior[0] != behavior_signature(native_behavior)
        ):
            raise RuntimeError(f"none condition is not a full no-op for seed {recipient}")

    input_fingerprints = sorted(
        {str(response["research_state_hash"]) for response in responses.values()}
    )
    parameter_probe_hashes = sorted(
        {str(response["research_parameter_probe_hash"]) for response in responses.values()}
    )
    if input_fingerprints == [""] or len(input_fingerprints) != 1:
        raise RuntimeError("transformed-input fingerprint changed within state")
    expected_probe = manifest["runtime"]["expected_parameter_probe_hash"]
    if parameter_probe_hashes != [expected_probe]:
        raise RuntimeError("parameter-probe hash changed or differs from manifest")
    if len(responses) != REQUESTS_PER_STATE or tuple(responses) != request_labels:
        raise RuntimeError("runner did not execute exactly the frozen 108 requests")
    if (
        structural_projection_null_count != 28
        or finite_off_diagonal_projection_count != 72
        or native_projection_absent_count != 8
        or shape_valid_response_action_count != REQUESTS_PER_STATE
    ):
        raise RuntimeError(
            "projection/action-shape census differs from frozen 28/72/8/108 design"
        )
    report: dict[str, Any] = {
        "status": "complete",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": args.expected_manifest_sha256,
        "unit_id": unit["unit_id"],
        "episode_id": unit["episode_id"],
        "task": unit["task"],
        "environment_seed": unit["environment_seed"],
        "phase": unit["phase"],
        "branch_step": unit["branch_step"],
        "branch_seeds": list(seeds),
        "action_shape": list(ACTION_SHAPE),
        "action_coordinate_count": ACTION_COORDINATE_COUNT,
        "shape_valid_response_action_count": shape_valid_response_action_count,
        "action_shape_failure_count": 0,
        "request_count": len(responses),
        "request_labels": list(responses),
        "input": input_audit,
        "native_actions": {
            str(seed): action.tolist() for seed, action in native_actions.items()
        },
        "native_future_hashes": {
            str(seed): response["research_future_hash"] for seed, response in native.items()
        },
        "native_path_noise_hashes": {
            str(seed): response["research_path_noise_hash"]
            for seed, response in native.items()
        },
        "native_initial_state_hashes": {
            str(seed): response["research_initial_state_hash"]
            for seed, response in native.items()
        },
        "native_pair_l2": native_pair_l2,
        "timing_rows": timing_rows,
        "native_replay_action_errors": native_replay_errors,
        "native_replay_signature_exact": native_replay_signatures,
        "all_calls_diagonal_replay_action_errors": replay_errors,
        "all_calls_diagonal_replay_signature_exact": replay_signatures,
        "none_noop_action_errors": none_noop_errors,
        "none_source_invariance_exact": none_source_invariance,
        "none_source_action_errors": none_source_action_errors,
        "native_replay_max_action_error": max(native_replay_errors.values()),
        "all_calls_diagonal_replay_max_action_error": max(replay_errors.values()),
        "none_noop_max_action_error": max(none_noop_errors.values()),
        "none_source_invariance_max_action_error": max(none_source_action_errors.values()),
        "maximum_action_input_error": maximum_action_input_error,
        "maximum_action_output_error": maximum_action_output_error,
        "maximum_active_model_input_future_clamp_error": maximum_model_input_error,
        "maximum_active_returned_future_velocity_error": maximum_velocity_error,
        "inactive_wrapper_write_count": inactive_wrapper_write_count,
        "schedule_and_index_gate_exact": schedule_and_index_gate_exact,
        "target_hash_gate_exact": target_hash_gate_exact,
        "rng_hash_gate_exact": rng_hash_gate_exact,
        "replay_signature_gate_exact": all(native_replay_signatures.values())
        and all(replay_signatures.values()),
        "input_fingerprint_count": len(input_fingerprints),
        "input_fingerprints": input_fingerprints,
        "parameter_probe_hash_count": len(parameter_probe_hashes),
        "parameter_probe_hashes": parameter_probe_hashes,
        "structural_projection_null_count": structural_projection_null_count,
        "finite_off_diagonal_projection_count": finite_off_diagonal_projection_count,
        "native_projection_absent_count": native_projection_absent_count,
        "runtime_gate": {
            "passed": True,
            "exact_schedule": True,
            "exact_active_site_captures": True,
            "exact_mask": True,
            "zero_action_coordinate_writes": True,
            "zero_inactive_wrapper_writes": True,
            "exact_none_noop": True,
            "exact_replays": True,
            "exact_rng_and_target_hashes": True,
            "all_finite": True,
            "required_numeric_fields_finite": True,
            "structural_null_census_exact": True,
            "exact_projection_applicability_census": True,
            "exact_action_shape_and_count": True,
        },
        "scope": {
            "action_only": True,
            "imposed_intervention_timing_strength": True,
            "natural_mediation": False,
            "physical_endpoint_evidence": False,
        },
    }
    expected_report_nulls: set[tuple[str, ...]] = set()
    for index, row in enumerate(timing_rows):
        server = row["server"]
        if server["research_attention_interface"]["cache_id"] is None:
            expected_report_nulls.add(
                (
                    "timing_rows",
                    str(index),
                    "server",
                    "research_attention_interface",
                    "cache_id",
                )
            )
        if row["recipient_seed"] == row["source_seed"]:
            expected_report_nulls.add(
                (
                    "timing_rows",
                    str(index),
                    "server",
                    "research_action_donor_projection",
                )
            )
    if null_paths(report) != expected_report_nulls:
        raise RuntimeError("final report contains a null outside frozen structural paths")
    if not all_finite(report):
        raise RuntimeError("final state report contains missing/nonfinite numeric data")
    atomic_json(args.output, canonical_value(report))
    print(
        json.dumps(
            {
                "status": "complete",
                "unit_id": unit["unit_id"],
                "request_count": len(responses),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
