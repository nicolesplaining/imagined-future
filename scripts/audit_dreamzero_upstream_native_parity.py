#!/usr/bin/env python3
"""Capture and compare one excluded DreamZero upstream/native parity case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dreamzero_future_transplants as core


SCHEMA = "dreamzero-upstream-native-parity-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--debug-bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", choices=("upstream", "patched"), required=True)
    parser.add_argument("--noise-seed", type=int, default=1140)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--compare-upstream", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    actions_path = args.output_root / f"{args.label}_actions.npz"
    receipt_path = args.output_root / f"{args.label}_receipt.json"
    for path in (actions_path, receipt_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    inputs = core.build_debug_input(
        argparse.Namespace(
            debug_bundle_root=args.debug_bundle_root,
            debug_prompt=core.DEBUG_PROMPT,
        )
    )
    request = dict(inputs.request)
    request["session_id"] = f"excluded-upstream-parity-{args.label}"
    if args.label == "patched":
        request[core.CONTROL_KEY] = {
            "mode": "off",
            "noise_seed": int(args.noise_seed),
        }

    client = core.DreamZeroClient(
        args.host,
        args.port,
        connect_timeout=60,
        response_timeout=3600,
    )
    try:
        client.reset()
        payload = dict(request)
        payload["endpoint"] = "infer"
        payload["dreamzero_skip_video_save"] = True
        client._connection.send(client._packer.pack(payload))
        raw_response = client._connection.recv(timeout=client._response_timeout)
        if isinstance(raw_response, str):
            raise RuntimeError(f"DreamZero server error:\n{raw_response}")
        response = client._unpack(raw_response)
    finally:
        client.close()
    if isinstance(response, dict):
        action = core.action_from_response(response, args.label)
        response_keys = sorted(response)
        intervention_audit = core.json_safe(response.get(core.AUDIT_KEY))
    else:
        action = np.ascontiguousarray(np.asarray(response))
        if (
            action.ndim != 2
            or action.shape[0] < 1
            or action.shape[1] != core.EXPECTED_ACTION_WIDTH
            or not np.issubdtype(action.dtype, np.floating)
            or not np.isfinite(action).all()
        ):
            raise ValueError(
                f"{args.label}: invalid raw upstream action {action.dtype} {action.shape}"
            )
        response_keys = None
        intervention_audit = None

    parity = None
    maximum_absolute_error = None
    upstream_sha256 = None
    if args.compare_upstream is not None:
        with np.load(args.compare_upstream, allow_pickle=False) as archive:
            upstream = np.ascontiguousarray(archive["actions"])
        upstream_sha256 = core.array_sha256(upstream)
        if upstream.shape != action.shape or upstream.dtype != action.dtype:
            parity = False
        else:
            parity = bool(np.array_equal(upstream, action))
            maximum_absolute_error = float(np.max(np.abs(upstream - action)))
        if not parity:
            raise RuntimeError(
                "patched mode-off output differs from clean upstream output: "
                f"max_abs={maximum_absolute_error}"
            )

    core.atomic_write_npz(actions_path, {"actions": action})
    actions_entry = core.freeze_with_sidecar(actions_path)
    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "admission": "excluded_development_control",
        "scientific_admission": False,
        "label": args.label,
        "source_commit": args.source_commit,
        "checkpoint_revision": args.checkpoint_revision,
        "noise_seed": int(args.noise_seed),
        "input_fingerprint": inputs.audit["input_fingerprint"],
        "action_shape": list(action.shape),
        "action_dtype": str(action.dtype),
        "action_array_sha256": core.array_sha256(action),
        "actions_artifact": actions_entry,
        "response_keys": response_keys,
        "intervention_audit": intervention_audit,
        "comparison": {
            "upstream_actions_path": (
                str(args.compare_upstream) if args.compare_upstream is not None else None
            ),
            "upstream_action_array_sha256": upstream_sha256,
            "bitwise_exact": parity,
            "maximum_absolute_error": maximum_absolute_error,
        },
        "runner_sha256": core.sha256_file(Path(__file__).resolve()),
    }
    core.atomic_write_json(receipt_path, receipt, mode=0o444)
    core.freeze_with_sidecar(receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
