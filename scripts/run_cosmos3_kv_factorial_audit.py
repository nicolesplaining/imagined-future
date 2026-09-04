#!/usr/bin/env python3
"""Run the complete Cosmos 3 future x future-token-K/V factorial.

This model-only worker does not restore or execute a simulator state. Its
caller must explicitly label the study scope so excluded engineering audits
cannot be confused with evaluation-cohort results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

ALL_LAYERS = list(range(36))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--asset-video", type=Path, required=True)
    parser.add_argument("--recorded-hdf5", type=Path)
    parser.add_argument("--branch-summary", type=Path)
    parser.add_argument("--branch-step", type=int)
    parser.add_argument("--prompt", default="Pick up the banana and place it in the bowl.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--recipient-seed", type=int, required=True)
    parser.add_argument("--donor-seed", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projection(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> float:
    direction = donor.astype(np.float64) - recipient.astype(np.float64)
    denominator = float(np.square(direction).sum())
    if denominator == 0.0:
        raise ValueError("recipient and donor actions define a degenerate axis")
    return float(
        ((value.astype(np.float64) - recipient.astype(np.float64)) * direction).sum()
        / denominator
    )


def current_request(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the model request without importing the Isaac/AV stack."""

    if args.asset_video.suffix != ".npz":
        raise ValueError("this model-only audit requires an NPZ video artifact")
    with np.load(args.asset_video, allow_pickle=False) as payload:
        video = np.asarray(payload["video"])
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
        raise ValueError(f"asset video must be uint8 [T,H,W,3], got {video.shape} {video.dtype}")

    prompt = args.prompt
    branch_step = args.branch_step
    if args.branch_summary is not None:
        branch_summary = json.loads(args.branch_summary.read_text(encoding="utf-8"))
        prompt = str(branch_summary["instruction"])
        if branch_step is None:
            branch_step = int(branch_summary["branch_step"])

    joint_position = np.zeros(7, dtype=np.float32)
    gripper_position = np.zeros(1, dtype=np.float32)
    proprio_source = "zeros"
    if args.recorded_hdf5 is not None:
        if branch_step is None or branch_step <= 0:
            raise ValueError("--recorded-hdf5 requires a positive branch step")
        import h5py

        with h5py.File(args.recorded_hdf5, "r") as stream:
            recorded = np.asarray(
                stream["data/demo_0/states/articulation/robot/joint_position"][
                    branch_step - 1
                ],
                dtype=np.float32,
            )
        joint_position = recorded[:7].copy()
        gripper_position = np.clip(recorded[7:8] / (np.pi / 4), 0, 1).astype(
            np.float32
        )
        proprio_source = "noise-free recorded post-step simulator state"

    request = {
        "observation/image": video[0].copy(),
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }
    audit = {
        "asset_video": str(args.asset_video),
        "recorded_hdf5": str(args.recorded_hdf5) if args.recorded_hdf5 else None,
        "branch_summary": str(args.branch_summary) if args.branch_summary else None,
        "branch_step": branch_step,
        "prompt": prompt,
        "proprio_source": proprio_source,
    }
    return request, audit


def metadata(response: dict[str, Any]) -> dict[str, Any]:
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
    """Replace non-finite diagnostics with null while preserving all finite data."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def atomic_json(path: Path, value: Any) -> None:
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


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite completed audit: {args.output}")
    if args.recipient_seed == args.donor_seed:
        raise ValueError("recipient and donor seeds must differ")

    base, current_audit = current_request(args)
    client = WebsocketClientPolicy(args.host, args.port)
    recipient_id = f"{args.study_id}-recipient"
    donor_id = f"{args.study_id}-donor"
    recipient_cache_id = f"{args.study_id}-recipient-kv"
    donor_cache_id = f"{args.study_id}-donor-kv"

    responses: dict[str, dict[str, Any]] = {}

    def infer(label: str, **research: Any) -> dict[str, Any]:
        response = client.infer(
            {
                **base,
                "research_id": f"{args.study_id}-{label}",
                **research,
            }
        )
        responses[label] = response
        return response

    recipient_native = infer(
        "recipient-native",
        research_mode="native",
        research_seed=args.recipient_seed,
        research_id=recipient_id,
    )
    recipient_repeat = infer(
        "recipient-repeat",
        research_mode="native",
        research_seed=args.recipient_seed,
    )
    donor_native = infer(
        "donor-native",
        research_mode="native",
        research_seed=args.donor_seed,
        research_id=donor_id,
    )

    common = {
        "research_seed": args.recipient_seed,
        "research_recipient_id": recipient_id,
    }

    def future(
        label: str,
        *,
        source: str,
        attention_mode: str | None = None,
        cache_id: str | None = None,
    ) -> dict[str, Any]:
        donor = recipient_id if source == "recipient" else donor_id
        attention: dict[str, Any] = {}
        if attention_mode is not None:
            if cache_id is None:
                raise ValueError("attention record/patch requires a cache ID")
            attention = {
                "research_attention_mode": attention_mode,
                "research_attention_cache_id": cache_id,
                "research_attention_exclude_layers": ALL_LAYERS,
                "research_attention_exclude_scope": "action",
            }
        return infer(
            label,
            research_mode="self" if source == "recipient" else "donor",
            research_donor_id=donor,
            **common,
            **attention,
        )

    recipient_baseline = future("recipient-baseline", source="recipient")
    donor_baseline = future("donor-baseline", source="donor")

    # R-future/R-KV: record and exact replay the recipient-future K/V.
    recipient_record = future(
        "recipient-kv-record",
        source="recipient",
        attention_mode="record",
        cache_id=recipient_cache_id,
    )
    recipient_replay = future(
        "recipient-kv-replay",
        source="recipient",
        attention_mode="patch",
        cache_id=recipient_cache_id,
    )

    # D-future/R-KV: the necessity/suppression cell.
    donor_with_recipient_kv = future(
        "donor-future-recipient-kv",
        source="donor",
        attention_mode="patch",
        cache_id=recipient_cache_id,
    )

    # Recording a new cache replaces the server's prior cache.  Record donor
    # K/V only after every recipient-cache consumer has finished.
    donor_record = future(
        "donor-kv-record",
        source="donor",
        attention_mode="record",
        cache_id=donor_cache_id,
    )
    donor_replay = future(
        "donor-kv-replay",
        source="donor",
        attention_mode="patch",
        cache_id=donor_cache_id,
    )

    # R-future/D-KV: the missing rescue/sufficiency cell.
    recipient_with_donor_kv = future(
        "recipient-future-donor-kv",
        source="recipient",
        attention_mode="patch",
        cache_id=donor_cache_id,
    )

    recipient_action = np.asarray(recipient_native["action"], dtype=np.float32)
    donor_action = np.asarray(donor_native["action"], dtype=np.float32)
    actions = {
        label: np.asarray(response["action"], dtype=np.float32)
        for label, response in responses.items()
    }
    exact_errors = {
        "recipient_native_repeat": maximum_error(
            actions["recipient-native"], actions["recipient-repeat"]
        ),
        "recipient_record_vs_baseline": maximum_error(
            actions["recipient-kv-record"], actions["recipient-baseline"]
        ),
        "recipient_replay_vs_record": maximum_error(
            actions["recipient-kv-replay"], actions["recipient-kv-record"]
        ),
        "donor_record_vs_baseline": maximum_error(
            actions["donor-kv-record"], actions["donor-baseline"]
        ),
        "donor_replay_vs_record": maximum_error(
            actions["donor-kv-replay"], actions["donor-kv-record"]
        ),
    }
    diagnostic_errors = {
        # A clean recipient-future clamp is a real intervention and need not
        # equal unconstrained joint denoising.  Report, but do not treat, this
        # contrast as an identity control.
        "recipient_clean_clamp_vs_native": maximum_error(
            actions["recipient-baseline"], recipient_action
        ),
    }

    factorial_labels = {
        "recipient_future_recipient_kv": "recipient-kv-record",
        "donor_future_recipient_kv": "donor-future-recipient-kv",
        "donor_future_donor_kv": "donor-kv-record",
        "recipient_future_donor_kv": "recipient-future-donor-kv",
    }
    factorial = {
        cell: {
            "response_label": label,
            "donor_projection": projection(
                actions[label], recipient_action, donor_action
            ),
            "distance_to_recipient": float(
                np.linalg.norm(actions[label] - recipient_action)
            ),
            "distance_to_donor": float(np.linalg.norm(actions[label] - donor_action)),
            "target_future_max_error": float(
                responses[label]["research_target_future_max_error"]
            ),
        }
        for cell, label in factorial_labels.items()
    }

    state_hashes = {
        str(response["research_state_hash"]) for response in responses.values()
    }
    action_coordinate_errors = {
        label: {
            "input": float(response["research_maximum_action_input_error"]),
            "output": float(response["research_maximum_action_output_error"]),
        }
        for label, response in responses.items()
        if "research_maximum_action_input_error" in response
    }
    cache_interfaces = {
        label: metadata(response).get("research_attention_interface")
        for label, response in responses.items()
        if "research_attention_interface" in response
    }
    future_signatures = {
        label: {
            "target_hash": str(response["research_target_hash"]),
            "output_hash": str(response["research_output_future_hash"]),
            "target_max_error": float(response["research_target_future_max_error"]),
        }
        for label, response in responses.items()
        if "research_target_hash" in response
    }
    recipient_future_labels = (
        "recipient-baseline",
        "recipient-kv-record",
        "recipient-kv-replay",
        "recipient-future-donor-kv",
    )
    donor_future_labels = (
        "donor-baseline",
        "donor-future-recipient-kv",
        "donor-kv-record",
        "donor-kv-replay",
    )

    def signatures_identical(labels: tuple[str, ...]) -> bool:
        return len(
            {
                (
                    future_signatures[label]["target_hash"],
                    future_signatures[label]["output_hash"],
                    future_signatures[label]["target_max_error"],
                )
                for label in labels
            }
        ) == 1

    future_target_consistency = {
        "recipient_future_arms_identical": signatures_identical(
            recipient_future_labels
        ),
        "donor_future_arms_identical": signatures_identical(donor_future_labels),
        "recipient_and_donor_targets_distinct": (
            future_signatures[recipient_future_labels[0]]["target_hash"]
            != future_signatures[donor_future_labels[0]]["target_hash"]
        ),
    }
    report = {
        "status": "complete",
        "scope": args.scope,
        "study_id": args.study_id,
        "recipient_seed": args.recipient_seed,
        "donor_seed": args.donor_seed,
        "layers": ALL_LAYERS,
        "current_request": current_audit,
        "input_sha256": {
            "asset_video": sha256(args.asset_video),
            "recorded_hdf5": sha256(args.recorded_hdf5)
            if args.recorded_hdf5
            else None,
            "branch_summary": sha256(args.branch_summary)
            if args.branch_summary
            else None,
        },
        "native_action_l2": float(np.linalg.norm(donor_action - recipient_action)),
        "exact_errors": exact_errors,
        "diagnostic_errors": diagnostic_errors,
        "factorial": factorial,
        "state_hash_count": len(state_hashes),
        "state_hashes": sorted(state_hashes),
        "action_coordinate_errors": action_coordinate_errors,
        "attention_interfaces": cache_interfaces,
        "future_signatures": future_signatures,
        "future_target_consistency": future_target_consistency,
        "responses": {label: metadata(response) for label, response in responses.items()},
    }

    if len(state_hashes) != 1:
        raise RuntimeError(f"state fingerprints diverged: {state_hashes}")
    if any(error != 0.0 for error in exact_errors.values()):
        raise RuntimeError(f"identity/replay gate failed: {exact_errors}")
    for label, errors in action_coordinate_errors.items():
        if errors["input"] != 0.0 or errors["output"] != 0.0:
            raise RuntimeError(f"{label} directly mutated action coordinates: {errors}")
    if not all(future_target_consistency.values()):
        raise RuntimeError(
            f"future targets changed across K/V arms: {future_target_consistency}"
        )

    report = json_safe(report)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    # WebsocketClientPolicy owns a background receive thread without a public
    # shutdown method.  The result is already atomically written and fsynced;
    # explicitly end this one-shot worker so batch launchers do not hang after
    # successful completion.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
