#!/usr/bin/env python3
"""Decode the four native futures for the frozen DreamZero representative.

This is a post-analysis media export.  It re-runs each registered native seed,
checks the returned action bit-for-bit against the immutable core artifact, and
asks the already-running official server to decode the generated future latent.
The script never writes into the confirmatory result or trace directories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BRANCH_SEEDS = (211, 223, 227, 229)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o444)


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("dreamzero_frozen_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reset_and_maybe_save(client, *, save_video: bool) -> None:
    client._connection.send(
        client._packer.pack(
            {
                "endpoint": "reset",
                "dreamzero_skip_video_save": not save_video,
            }
        )
    )
    response = client._connection.recv(timeout=client._response_timeout)
    if response != "reset successful":
        raise RuntimeError(f"DreamZero reset failed: {response!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--selection-rule", type=Path, required=True)
    parser.add_argument("--server-video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite media root: {args.output_root}")
    selection = json.loads(args.selection_rule.read_text())
    state_id = str(selection["selected_state_id"])

    runner = load_runner(args.runner.resolve())
    frozen_args = SimpleNamespace(
        manifest=args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
        data_root=args.data_root,
        metadata_root=args.metadata_root,
    )
    manifest, resource_by_id, data_root, frozen = runner.validate_manifest_and_receipt(
        frozen_args
    )
    state_matches = [item for item in manifest["states"] if item["state_id"] == state_id]
    if len(state_matches) != 1:
        raise ValueError(f"representative state match count is {len(state_matches)}")
    receipt = json.loads((data_root / "download_receipt.json").read_text())
    receipt_by_id = {str(item["resource_id"]): item for item in receipt["resources"]}
    inputs = runner.build_frozen_input(
        state_matches[0],
        resource_by_id=resource_by_id,
        receipt_by_id=receipt_by_id,
        data_root=data_root,
        modality=frozen["modality"],
    )

    core_state = args.core_root / "states" / state_id
    core_arrays_path = core_state / "actions.npz"
    core_arrays = np.load(core_arrays_path)
    registered_seeds = tuple(int(value) for value in core_arrays["branch_seeds"])
    if registered_seeds != BRANCH_SEEDS:
        raise ValueError(f"unexpected core seeds: {registered_seeds}")
    native_actions = np.asarray(core_arrays["native_actions"])

    existing_videos = {path.resolve() for path in args.server_video_root.glob("*.mp4")}
    temp_trace_root = args.output_root.with_name(args.output_root.name + "_traces")
    if temp_trace_root.exists():
        raise FileExistsError(temp_trace_root)
    temp_trace_root.mkdir(parents=True)
    args.output_root.mkdir(parents=True)

    records: list[dict[str, object]] = []
    client = runner.DreamZeroClient(
        args.host,
        args.port,
        connect_timeout=30.0,
        response_timeout=3600.0,
    )
    try:
        for index, seed in enumerate(BRANCH_SEEDS):
            reset_and_maybe_save(client, save_video=False)
            trace_path = temp_trace_root / f"native_seed_{seed}.pt"
            request = dict(inputs.request)
            request["session_id"] = f"postanalysis-media:{state_id}:seed-{seed}"
            request[runner.CONTROL_KEY] = runner.request_control(
                mode="record",
                noise_seed=seed,
                trace_path=trace_path,
                action_noise_reference_path=None,
                replay_start=0,
                replay_stop=runner.EXPECTED_SOLVER_STEPS,
            )
            response = client.infer(request)
            action = runner.action_from_response(response, f"media_seed_{seed}")
            if not np.array_equal(action, native_actions[index]):
                error = runner.maximum_error(action, native_actions[index])
                raise RuntimeError(
                    f"seed {seed} action does not bitwise match frozen core; max error={error}"
                )

            before = {path.resolve() for path in args.server_video_root.glob("*.mp4")}
            reset_and_maybe_save(client, save_video=True)
            after = {path.resolve() for path in args.server_video_root.glob("*.mp4")}
            created = sorted(after - before)
            if len(created) != 1:
                raise RuntimeError(f"seed {seed} created {len(created)} videos: {created}")
            source_video = created[0]
            destination = args.output_root / f"native_future_seed_{seed}.mp4"
            shutil.copyfile(source_video, destination)
            destination.chmod(0o444)
            trace_path.chmod(0o444)
            records.append(
                {
                    "seed": seed,
                    "video": destination.name,
                    "video_sha256": sha256_file(destination),
                    "video_size_bytes": destination.stat().st_size,
                    "source_server_video": str(source_video),
                    "trace_sha256": sha256_file(trace_path),
                    "action_sha256": runner.array_sha256(action),
                    "frozen_core_action_sha256": runner.array_sha256(native_actions[index]),
                    "action_bit_exact_to_frozen_core": True,
                }
            )
    finally:
        client.close()

    current_videos = {path.resolve() for path in args.server_video_root.glob("*.mp4")}
    if len(current_videos - existing_videos) != len(BRANCH_SEEDS):
        raise RuntimeError("server media-file creation count does not match four branches")

    receipt_value = {
        "schema": "dreamzero-postanalysis-representative-video-export-v1",
        "status": "complete",
        "purpose": "descriptive media only; excluded from all state selection and inference",
        "state_id": state_id,
        "selection_rule_sha256": sha256_file(args.selection_rule),
        "runner_path": str(args.runner.resolve()),
        "runner_sha256": sha256_file(args.runner.resolve()),
        "exporter_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "core_actions_npz_sha256": sha256_file(core_arrays_path),
        "input_fingerprint": inputs.audit["input_fingerprint"],
        "records": records,
    }
    receipt_path = args.output_root / "receipt.json"
    write_immutable(receipt_path, canonical_bytes(receipt_value) + b"\n")
    write_immutable(
        receipt_path.with_suffix(".json.sha256"),
        f"{sha256_file(receipt_path)}  {receipt_path.name}\n".encode("ascii"),
    )
    temp_trace_root.chmod(stat.S_IREAD | stat.S_IEXEC)
    args.output_root.chmod(stat.S_IREAD | stat.S_IEXEC)
    print(json.dumps(receipt_value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
