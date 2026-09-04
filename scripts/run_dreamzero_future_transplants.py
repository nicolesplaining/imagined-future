#!/usr/bin/env python3
"""Run a frozen DreamZero-DROID donor-future transplant grid over WebSocket.

For each state this runner records four native DreamZero branches, then replays
the complete 4 x 4 recipient-noise x future-source matrix.  The intervention is
requested per call through ``dreamzero_intervention``; no process-global
environment variables are mutated by the client.

The canonical input path reads one exact DROID row from parquet and the matching
frame from each of the three episode MP4s.  A separate, explicitly excluded
mode supports DreamZero's bundled ``debug_image`` videos for integration smoke
tests.  Completed states and their trace files are hash-inventoried and made
read-only.  A mutable atomic checkpoint makes interrupted states resumable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "dreamzero-future-transplant-state-v1"
CHECKPOINT_SCHEMA = "dreamzero-future-transplant-checkpoint-v1"
INVENTORY_SCHEMA = "dreamzero-future-transplant-trace-inventory-v1"
EXPECTED_MANIFEST_SCHEMA = "dreamzero-droid-evaluation-state-manifest-v1"
EXPECTED_RECEIPT_SCHEMA = "dreamzero-droid-download-receipt-v1"
CONTROL_KEY = "dreamzero_intervention"
AUDIT_KEY = "dreamzero_intervention_audit"
BRANCH_SEEDS = (211, 223, 227, 229)
EXPECTED_SOLVER_STEPS = 16
EXPECTED_ACTION_WIDTH = 8
EXPECTED_FIRST_START_FRAME = 1

CAMERA_RESOURCE_TO_REQUEST = {
    "observation.images.exterior_image_1_left":
        "observation/exterior_image_0_left",
    "observation.images.exterior_image_2_left":
        "observation/exterior_image_1_left",
    "observation.images.wrist_image_left":
        "observation/wrist_image_left",
}
DEBUG_CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}
DEBUG_PROMPT = (
    "Move the pan forward and use the brush in the middle of the plates to "
    "brush the inside of the pan"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--admission",
        choices=("evaluation", "excluded_debug_smoke"),
        required=True,
        help="Scientific admission class; bundled debug input is always excluded.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Downloaded-state root containing files/ and download_receipt.json; "
        "defaults to the manifest parent.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help="Directory containing the frozen DROID modality.json metadata.",
    )
    parser.add_argument(
        "--debug-bundle-root",
        type=Path,
        help="DreamZero repo debug_image directory (excluded smoke only).",
    )
    parser.add_argument("--debug-prompt", default=DEBUG_PROMPT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--client-trace-root",
        type=Path,
        required=True,
        help="Client-visible trace directory, normally on shared storage.",
    )
    parser.add_argument(
        "--server-trace-root",
        type=Path,
        required=True,
        help="Absolute path to the same trace directory in the server namespace.",
    )
    parser.add_argument("--replay-start", type=int, default=0)
    parser.add_argument("--replay-stop", type=int, default=EXPECTED_SOLVER_STEPS)
    parser.add_argument("--state-id", action="append", default=[])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
        help="Validate and decode selected inputs without opening a WebSocket.",
    )
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--response-timeout", type=float, default=3600.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value is not admissible JSON: {value}")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: object, *, mode: int = 0o644) -> None:
    atomic_write_bytes(path, canonical_json_bytes(json_safe(value)) + b"\n", mode=mode)


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def freeze_with_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"cannot freeze non-regular file: {path}")
    os.chmod(path, 0o444)
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    atomic_write_bytes(sidecar, f"{digest}  {path.name}\n".encode(), mode=0o444)
    if sha256_file(path) != digest:
        raise RuntimeError(f"post-freeze hash changed: {path}")
    return {
        "relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "mode": stat.S_IMODE(path.stat().st_mode),
        "sidecar_sha256": sha256_file(sidecar),
    }


def verify_sha_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields:
        raise ValueError(f"empty SHA sidecar: {sidecar}")
    actual = sha256_file(path)
    if fields[0] != actual:
        raise ValueError(f"SHA sidecar mismatch for {path}: {fields[0]} != {actual}")
    return actual


def manifest_body_hash(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("manifest_id", None)
    body.pop("manifest_body_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


@dataclasses.dataclass(frozen=True)
class FrozenInputs:
    state_id: str
    state_index: int
    episode_index: int | None
    frame_index: int
    prompt: str
    task_family: str
    request: dict[str, Any]
    audit: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def validate_manifest_and_receipt(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path, dict[str, Any]]:
    if args.manifest is None or args.expected_manifest_sha256 is None:
        raise ValueError("evaluation requires --manifest and --expected-manifest-sha256")
    if args.metadata_root is None:
        raise ValueError("evaluation requires --metadata-root")
    manifest_sha = verify_sha_sidecar(args.manifest)
    if manifest_sha != args.expected_manifest_sha256:
        raise ValueError(
            f"manifest SHA mismatch: {manifest_sha} != {args.expected_manifest_sha256}"
        )
    manifest = load_json(args.manifest)
    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        raise ValueError(f"unexpected manifest schema: {manifest.get('schema')!r}")
    if manifest.get("scope", {}).get("outcome_blind") is not True:
        raise ValueError("manifest is not explicitly outcome-blind")
    if manifest.get("scope", {}).get("model") != "DreamZero-DROID":
        raise ValueError("manifest model scope is not DreamZero-DROID")
    body_hash = manifest_body_hash(manifest)
    if body_hash != manifest.get("manifest_body_sha256"):
        raise ValueError("manifest internal body hash mismatch")
    if manifest.get("manifest_id") != f"dreamzero-droid-states-{body_hash[:16]}":
        raise ValueError("manifest ID is not content-addressed to its body")
    states = manifest.get("states")
    if not isinstance(states, list) or len(states) != 30:
        raise ValueError(f"canonical manifest must contain exactly 30 states, got {len(states or [])}")

    data_root = args.data_root or args.manifest.parent
    receipt_path = data_root / "download_receipt.json"
    receipt_sha = verify_sha_sidecar(receipt_path)
    receipt = load_json(receipt_path)
    if receipt.get("schema") != EXPECTED_RECEIPT_SCHEMA:
        raise ValueError("unexpected download receipt schema")
    if receipt.get("status") != "complete":
        raise ValueError("download receipt is not complete")
    if receipt.get("manifest_id") != manifest["manifest_id"]:
        raise ValueError("download receipt manifest ID mismatch")
    if receipt.get("manifest_file_sha256") != manifest_sha:
        raise ValueError("download receipt manifest SHA mismatch")
    if int(receipt.get("resource_count", -1)) != len(manifest.get("resources", [])):
        raise ValueError("download receipt resource count mismatch")

    metadata_by_name = {
        str(item["relative_path"]): item
        for item in manifest["dataset"]["metadata_inventory"]
    }
    modality_path = args.metadata_root / "modality.json"
    modality_key = "meta/modality.json"
    if modality_key not in metadata_by_name:
        raise ValueError("manifest does not pin meta/modality.json")
    modality_sha = sha256_file(modality_path)
    if modality_sha != metadata_by_name[modality_key]["sha256"]:
        raise ValueError("modality.json differs from frozen manifest")
    modality = load_json(modality_path)

    receipt_resources = receipt.get("resources", [])
    receipt_by_id = {str(item["resource_id"]): item for item in receipt_resources}
    if len(receipt_by_id) != len(receipt_resources):
        raise ValueError("duplicate resource IDs in download receipt")
    resources = manifest.get("resources", [])
    resource_by_id: dict[str, dict[str, Any]] = {}
    for item in resources:
        resource_id = str(item["resource_id"])
        if resource_id in resource_by_id:
            raise ValueError(f"duplicate manifest resource ID: {resource_id}")
        resource_by_id[resource_id] = item
        received = receipt_by_id.get(resource_id)
        if received is None:
            raise ValueError(f"resource absent from receipt: {resource_id}")
        if received.get("relative_path") != item.get("relative_path"):
            raise ValueError(f"resource path mismatch in receipt: {resource_id}")
    provenance = {
        "manifest_id": manifest["manifest_id"],
        "manifest_file_sha256": manifest_sha,
        "manifest_body_sha256": body_hash,
        "download_receipt_sha256": receipt_sha,
        "modality_sha256": modality_sha,
        "dataset_revision": manifest["dataset"]["revision"],
    }
    return manifest, resource_by_id, data_root, {"modality": modality, **provenance}


def validate_resource(
    resource: Mapping[str, Any], data_root: Path, receipt: Mapping[str, Any]
) -> Path:
    path = data_root / str(resource["destination_relative_path"])
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    expected_size = int(receipt["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(f"resource size mismatch: {path}")
    digest = sha256_file(path)
    if digest != receipt["sha256"]:
        raise ValueError(f"resource SHA mismatch: {path}")
    return path


def decode_exact_frame(path: Path, frame_index: int) -> tuple[np.ndarray, int]:
    try:
        import decord
    except ImportError as error:  # pragma: no cover - runtime dependency gate
        raise RuntimeError("decord is required to decode frozen MP4 frames") from error
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    frame_count = len(reader)
    if not 0 <= frame_index < frame_count:
        raise IndexError(f"frame {frame_index} outside [0,{frame_count}) for {path}")
    frame = np.asarray(reader[frame_index].asnumpy())
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"unexpected decoded frame {frame.dtype} {frame.shape}: {path}")
    return np.ascontiguousarray(frame), frame_count


def parquet_row(path: Path, row_index: int) -> tuple[dict[str, Any], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - runtime dependency gate
        raise RuntimeError("pyarrow is required to read frozen DROID parquet") from error
    table = pq.ParquetFile(path).read(
        columns=(
            "observation.state",
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
        )
    )
    if not 0 <= row_index < table.num_rows:
        raise IndexError(f"row {row_index} outside parquet with {table.num_rows} rows")
    return {name: table[name][row_index].as_py() for name in table.column_names}, table.num_rows


def split_state(raw_state: Sequence[float], modality: Mapping[str, Any]) -> dict[str, np.ndarray]:
    array = np.asarray(raw_state, dtype=np.float64)
    if array.shape != (14,) or not np.isfinite(array).all():
        raise ValueError(f"DROID observation.state must be finite shape (14,), got {array.shape}")
    expected = {
        "cartesian_position": (0, 6),
        "gripper_position": (6, 7),
        "joint_position": (7, 14),
    }
    state_meta = modality.get("state", {})
    values: dict[str, np.ndarray] = {}
    for name, (start, stop) in expected.items():
        entry = state_meta.get(name)
        if entry != {"start": start, "end": stop}:
            raise ValueError(f"unexpected modality slice for state.{name}: {entry!r}")
        values[name] = np.ascontiguousarray(array[start:stop])
    return values


def input_fingerprint(request: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(request):
        if key in {"session_id", CONTROL_KEY}:
            continue
        digest.update(key.encode("utf-8"))
        value = request[key]
        if isinstance(value, np.ndarray):
            digest.update(array_sha256(value).encode("ascii"))
        else:
            digest.update(canonical_json_bytes(json_safe(value)))
    return digest.hexdigest()


def build_frozen_input(
    state: Mapping[str, Any],
    *,
    resource_by_id: Mapping[str, Mapping[str, Any]],
    receipt_by_id: Mapping[str, Mapping[str, Any]],
    data_root: Path,
    modality: Mapping[str, Any],
) -> FrozenInputs:
    resources = [resource_by_id[str(item)] for item in state["resource_ids"]]
    parquet_resources = [item for item in resources if item["kind"] == "parquet"]
    video_resources = [item for item in resources if item["kind"] == "video"]
    if len(parquet_resources) != 1 or len(video_resources) != 3:
        raise ValueError(f"state {state['state_id']} does not have one parquet and three videos")
    parquet_resource = parquet_resources[0]
    parquet_path = validate_resource(
        parquet_resource, data_root, receipt_by_id[str(parquet_resource["resource_id"])]
    )
    row, row_count = parquet_row(parquet_path, int(state["frame_index"]))
    if int(row["episode_index"]) != int(state["episode_index"]):
        raise ValueError("parquet episode index differs from manifest")
    if int(row["frame_index"]) != int(state["frame_index"]):
        raise ValueError("parquet frame index differs from manifest")
    if row_count != int(state["episode_length"]):
        raise ValueError("parquet row count differs from frozen episode length")
    if int(row["task_index"]) not in [int(v) for v in state["task_indices_for_exact_prompt"]]:
        raise ValueError("parquet task index does not identify the frozen prompt")
    expected_timestamp = float(state["timestamp_seconds_from_fps"])
    if not math.isclose(float(row["timestamp"]), expected_timestamp, abs_tol=1e-9):
        raise ValueError("parquet timestamp differs from frozen manifest")
    state_parts = split_state(row["observation.state"], modality)

    request: dict[str, Any] = {
        "observation/cartesian_position": state_parts["cartesian_position"],
        "observation/gripper_position": state_parts["gripper_position"],
        "observation/joint_position": state_parts["joint_position"],
        "prompt": str(state["task"]),
    }
    camera_audit: dict[str, Any] = {}
    for resource in video_resources:
        video_key = str(resource["video_key"])
        if video_key not in CAMERA_RESOURCE_TO_REQUEST:
            raise ValueError(f"unexpected video key: {video_key}")
        path = validate_resource(
            resource, data_root, receipt_by_id[str(resource["resource_id"])]
        )
        frame, frame_count = decode_exact_frame(path, int(state["frame_index"]))
        request_key = CAMERA_RESOURCE_TO_REQUEST[video_key]
        request[request_key] = frame
        camera_audit[request_key] = {
            "dataset_video_key": video_key,
            "resource_id": resource["resource_id"],
            "relative_path": resource["relative_path"],
            "file_sha256": receipt_by_id[str(resource["resource_id"])]["sha256"],
            "frame_count": frame_count,
            "frame_index": int(state["frame_index"]),
            "frame_shape": list(frame.shape),
            "frame_dtype": str(frame.dtype),
            "frame_sha256": array_sha256(frame),
        }
    audit = {
        "source": "frozen_dreamzero_droid_manifest",
        "parquet": {
            "resource_id": parquet_resource["resource_id"],
            "relative_path": parquet_resource["relative_path"],
            "file_sha256": receipt_by_id[str(parquet_resource["resource_id"])]["sha256"],
            "row_index": int(state["frame_index"]),
            "row_count": row_count,
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
            "timestamp": float(row["timestamp"]),
            "task_index": int(row["task_index"]),
        },
        "cameras": camera_audit,
        "state": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": array_sha256(value),
                "values": value.tolist(),
            }
            for key, value in state_parts.items()
        },
    }
    audit["input_fingerprint"] = input_fingerprint(request)
    return FrozenInputs(
        state_id=str(state["state_id"]),
        state_index=int(state["state_index"]),
        episode_index=int(state["episode_index"]),
        frame_index=int(state["frame_index"]),
        prompt=str(state["task"]),
        task_family=str(state["task_family"]),
        request=request,
        audit=audit,
    )


def build_debug_input(args: argparse.Namespace) -> FrozenInputs:
    if args.debug_bundle_root is None:
        raise ValueError("excluded_debug_smoke requires --debug-bundle-root")
    request: dict[str, Any] = {
        "observation/cartesian_position": np.zeros(6, dtype=np.float64),
        "observation/gripper_position": np.zeros(1, dtype=np.float64),
        "observation/joint_position": np.zeros(7, dtype=np.float64),
        "prompt": str(args.debug_prompt),
    }
    cameras: dict[str, Any] = {}
    for request_key, filename in DEBUG_CAMERA_FILES.items():
        path = args.debug_bundle_root / filename
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        frame, frame_count = decode_exact_frame(path, 0)
        request[request_key] = frame
        cameras[request_key] = {
            "relative_path": filename,
            "file_sha256": sha256_file(path),
            "frame_count": frame_count,
            "frame_index": 0,
            "frame_shape": list(frame.shape),
            "frame_dtype": str(frame.dtype),
            "frame_sha256": array_sha256(frame),
        }
    audit = {
        "source": "excluded_upstream_bundled_debug_image",
        "scientific_admission": False,
        "cameras": cameras,
        "state": {
            "cartesian_position": request["observation/cartesian_position"].tolist(),
            "gripper_position": request["observation/gripper_position"].tolist(),
            "joint_position": request["observation/joint_position"].tolist(),
        },
        "input_fingerprint": input_fingerprint(request),
    }
    return FrozenInputs(
        state_id="excluded_bundled_debug_frame_000000",
        state_index=0,
        episode_index=None,
        frame_index=0,
        prompt=str(args.debug_prompt),
        task_family="excluded_debug",
        request=request,
        audit=audit,
    )


class DreamZeroClient:
    def __init__(
        self, host: str, port: int, *, connect_timeout: float, response_timeout: float
    ) -> None:
        try:
            import websockets.sync.client
            from openpi_client import msgpack_numpy
        except ImportError as error:  # pragma: no cover - runtime dependency gate
            raise RuntimeError("websockets and openpi_client are required") from error
        self._packer = msgpack_numpy.Packer()
        self._unpack = msgpack_numpy.unpackb
        self._response_timeout = response_timeout
        self._connection = websockets.sync.client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=connect_timeout,
            ping_interval=60,
            ping_timeout=600,
        )
        metadata_message = self._connection.recv(timeout=connect_timeout)
        if isinstance(metadata_message, str):
            raise RuntimeError(f"server returned text instead of metadata: {metadata_message}")
        self.metadata = self._unpack(metadata_message)
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        metadata = self.metadata
        if not isinstance(metadata, dict):
            raise TypeError("server metadata is not a mapping")
        expected = {
            "needs_wrist_camera": True,
            "n_external_cameras": 2,
            "needs_session_id": True,
            "action_space": "joint_position",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"unexpected server metadata {key}={metadata.get(key)!r}")

    def reset(self) -> None:
        self._connection.send(
            self._packer.pack(
                {"endpoint": "reset", "dreamzero_skip_video_save": True}
            )
        )
        response = self._connection.recv(timeout=self._response_timeout)
        if response != "reset successful":
            raise RuntimeError(f"DreamZero reset failed: {response!r}")

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        payload["endpoint"] = "infer"
        payload["dreamzero_skip_video_save"] = True
        self._connection.send(self._packer.pack(payload))
        response = self._connection.recv(timeout=self._response_timeout)
        if isinstance(response, str):
            raise RuntimeError(f"DreamZero server error:\n{response}")
        value = self._unpack(response)
        if not isinstance(value, dict):
            raise TypeError(
                "controlled DreamZero response must be a dict containing actions "
                f"and {AUDIT_KEY}; got {type(value).__name__}"
            )
        return value

    def close(self) -> None:
        self._connection.close()


def wait_for_trace(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.1)
    raise FileNotFoundError(f"server trace did not become visible: {path}")


def action_from_response(response: Mapping[str, Any], label: str) -> np.ndarray:
    if "actions" not in response:
        raise KeyError(f"{label}: controlled response lacks actions")
    action = np.asarray(response["actions"])
    if (
        action.ndim != 2
        or action.shape[0] < 1
        or action.shape[1] != EXPECTED_ACTION_WIDTH
        or not np.issubdtype(action.dtype, np.floating)
        or not np.isfinite(action).all()
    ):
        raise ValueError(f"{label}: invalid action array {action.dtype} {action.shape}")
    return np.ascontiguousarray(action)


def canonical_path(path: Path) -> str:
    return os.path.normpath(str(path))


def validate_server_audit(
    response: Mapping[str, Any],
    *,
    label: str,
    mode: str,
    noise_seed: int,
    trace_path: Path,
    action_reference_path: Path | None,
    replay_start: int,
    replay_stop: int,
    expected_source_trace_hash: str | None,
    expected_source_action_hash: str | None,
    expected_recipient_action_hash: str | None,
) -> dict[str, Any]:
    audit = response.get(AUDIT_KEY)
    if not isinstance(audit, dict):
        raise TypeError(f"{label}: response lacks mapping {AUDIT_KEY}")
    required = {
        "schema_version",
        "trace_format_version",
        "mode",
        "status",
        "noise_seed",
        "trace_path",
        "action_noise_reference_path",
        "current_start_frame",
        "replay_start",
        "replay_stop",
        "num_solver_steps",
        "video_trace_sha256",
        "donor_action_noise_sha256",
        "recipient_reference_action_noise_sha256",
        "active_action_noise_sha256",
        "applied_video_steps",
    }
    if set(audit) != required:
        raise KeyError(
            f"{label}: audit schema differs; missing={sorted(required-set(audit))}, "
            f"extra={sorted(set(audit)-required)}"
        )
    expected_status = "recorded" if mode == "record" else "replayed"
    if (
        audit["schema_version"] != 1
        or audit["trace_format_version"] != 3
        or audit["mode"] != mode
        or audit["status"] != expected_status
    ):
        raise ValueError(f"{label}: server audit version/mode/status mismatch: {audit}")
    if int(audit["noise_seed"]) != noise_seed:
        raise ValueError(f"{label}: server noise seed differs from request")
    if canonical_path(Path(str(audit["trace_path"]))) != canonical_path(trace_path):
        raise ValueError(f"{label}: server trace path differs from request")
    observed_reference = audit["action_noise_reference_path"]
    if action_reference_path is None:
        if observed_reference is not None:
            raise ValueError(f"{label}: record unexpectedly used an action reference")
    elif canonical_path(Path(str(observed_reference))) != canonical_path(action_reference_path):
        raise ValueError(f"{label}: server action-reference path differs from request")
    # DreamZero consumes the single conditioning frame before entering joint
    # video/action denoising, so the first generated future chunk starts at 1.
    if int(audit["current_start_frame"]) != EXPECTED_FIRST_START_FRAME:
        raise ValueError(
            f"{label}: intervention did not run at the first generated chunk "
            f"({EXPECTED_FIRST_START_FRAME})"
        )
    if int(audit["num_solver_steps"]) != EXPECTED_SOLVER_STEPS:
        raise ValueError(f"{label}: solver-step count differs")
    if mode == "record":
        interval_matches = (
            audit["replay_start"] is None
            and audit["replay_stop"] is None
            and audit["applied_video_steps"] == []
        )
    else:
        interval_matches = (
            int(audit["replay_start"]) == replay_start
            and int(audit["replay_stop"]) == replay_stop
            and audit["applied_video_steps"] == list(range(replay_start, replay_stop))
        )
    if not interval_matches:
        raise ValueError(f"{label}: solver interval metadata differs from request")
    hash_fields = (
        "video_trace_sha256",
        "donor_action_noise_sha256",
        "recipient_reference_action_noise_sha256",
        "active_action_noise_sha256",
    )
    for field in hash_fields:
        value = audit[field]
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label}: invalid {field}: {value!r}")
    if mode == "record":
        if audit["video_trace_sha256"] is None:
            raise ValueError(f"{label}: record response omitted video trace hash")
        if (
            audit["donor_action_noise_sha256"] is not None
            or audit["recipient_reference_action_noise_sha256"] is not None
        ):
            raise ValueError(f"{label}: record unexpectedly reports replay-only hashes")
    else:
        if (
            expected_source_trace_hash is None
            or expected_source_action_hash is None
            or expected_recipient_action_hash is None
        ):
            raise AssertionError("replay audit requires frozen source/reference hashes")
        if audit["video_trace_sha256"] != expected_source_trace_hash:
            raise ValueError(f"{label}: replay used the wrong donor-video trace")
        if audit["donor_action_noise_sha256"] != expected_source_action_hash:
            raise ValueError(f"{label}: replay donor action-noise provenance differs")
        if audit["recipient_reference_action_noise_sha256"] != expected_recipient_action_hash:
            raise ValueError(f"{label}: replay used the wrong recipient action reference")
        if audit["active_action_noise_sha256"] != expected_recipient_action_hash:
            raise ValueError(f"{label}: replay active action noise is not recipient-native")
    return json_safe(audit)


def action_record(action: np.ndarray, audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shape": list(action.shape),
        "dtype": str(action.dtype),
        "sha256": array_sha256(action),
        "values": action.tolist(),
        "server_audit": json_safe(audit),
    }


def action_from_record(record: Mapping[str, Any], label: str) -> np.ndarray:
    action = np.asarray(record["values"], dtype=np.dtype(str(record["dtype"])))
    if list(action.shape) != list(record["shape"]):
        raise ValueError(f"{label}: checkpoint action shape mismatch")
    if array_sha256(action) != record["sha256"]:
        raise ValueError(f"{label}: checkpoint action hash mismatch")
    if not np.isfinite(action).all():
        raise ValueError(f"{label}: checkpoint action is non-finite")
    return np.ascontiguousarray(action)


def expected_checkpoint_header(
    *, args: argparse.Namespace, inputs: FrozenInputs, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "status": "partial",
        "admission": args.admission,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "provenance": json_safe(provenance),
        "state": {
            "state_id": inputs.state_id,
            "state_index": inputs.state_index,
            "episode_index": inputs.episode_index,
            "frame_index": inputs.frame_index,
            "task_family": inputs.task_family,
            "prompt": inputs.prompt,
            "input_fingerprint": inputs.audit["input_fingerprint"],
        },
        "branch_seeds": list(BRANCH_SEEDS),
        "grid_order": [
            {"recipient_seed": recipient, "future_source_seed": source}
            for recipient in BRANCH_SEEDS
            for source in BRANCH_SEEDS
        ],
        "intervention": {
            "control_key": CONTROL_KEY,
            "audit_key": AUDIT_KEY,
            "replay_start_inclusive": args.replay_start,
            "replay_stop_exclusive": args.replay_stop,
            "expected_solver_steps": EXPECTED_SOLVER_STEPS,
            "action_noise_source": "recipient native trace",
            "future_source": "source native video trace",
            "action_coordinates_written_by_client": False,
        },
        "input_audit": inputs.audit,
    }


def load_or_create_checkpoint(
    path: Path, expected_header: Mapping[str, Any]
) -> dict[str, Any]:
    if path.exists():
        checkpoint = load_json(path)
        for key, expected in expected_header.items():
            if key == "status":
                continue
            if checkpoint.get(key) != expected:
                raise ValueError(f"checkpoint header mismatch at {key}: {path}")
        if checkpoint.get("status") not in {"partial", "complete"}:
            raise ValueError(f"invalid checkpoint status: {path}")
        if not isinstance(checkpoint.get("completed"), dict):
            raise ValueError(f"checkpoint completed map missing: {path}")
        return checkpoint
    checkpoint = {**expected_header, "completed": {}}
    atomic_write_json(path, checkpoint)
    return checkpoint


def trace_paths(
    *, args: argparse.Namespace, state_id: str, seed: int
) -> tuple[Path, Path]:
    name = f"native_seed_{seed}.pt"
    return (
        args.client_trace_root / state_id / name,
        args.server_trace_root / state_id / name,
    )


def request_control(
    *,
    mode: str,
    noise_seed: int,
    trace_path: Path,
    action_noise_reference_path: Path | None,
    replay_start: int,
    replay_stop: int,
) -> dict[str, Any]:
    result = {
        "mode": mode,
        "trace_path": str(trace_path),
        "noise_seed": int(noise_seed),
        "replay_start": int(replay_start),
        "replay_stop": int(replay_stop),
    }
    if action_noise_reference_path is not None:
        result["action_noise_reference_path"] = str(action_noise_reference_path)
    return result


def maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return math.inf
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def directional_metrics(
    action: np.ndarray, recipient: np.ndarray, source: np.ndarray
) -> dict[str, Any]:
    value = action.astype(np.float64).reshape(-1)
    start = recipient.astype(np.float64).reshape(-1)
    target = source.astype(np.float64).reshape(-1)
    axis = target - start
    displacement = value - start
    axis_squared = float(np.dot(axis, axis))
    recipient_l2 = float(np.linalg.norm(displacement))
    source_l2 = float(np.linalg.norm(value - target))
    separation_l2 = float(np.sqrt(axis_squared))
    if axis_squared == 0.0:
        projection = None
        cosine = None
        orthogonal_l2 = None
    else:
        projection = float(np.dot(displacement, axis) / axis_squared)
        parallel = projection * axis
        orthogonal_l2 = float(np.linalg.norm(displacement - parallel))
        displacement_norm = float(np.linalg.norm(displacement))
        cosine = (
            float(np.dot(displacement, axis) / (displacement_norm * separation_l2))
            if displacement_norm > 0.0
            else None
        )
    return {
        "native_axis_sha256": array_sha256(axis),
        "native_separation_l2": separation_l2,
        "axis_squared_norm": axis_squared,
        "replay_l2_from_recipient": recipient_l2,
        "replay_l2_to_future_source": source_l2,
        "normalized_projection": projection,
        "cosine_alignment": cosine,
        "orthogonal_residual_l2": orthogonal_l2,
    }


def run_state(
    *,
    args: argparse.Namespace,
    client: DreamZeroClient,
    inputs: FrozenInputs,
    provenance: Mapping[str, Any],
) -> Path:
    state_dir = args.output_root / "states" / inputs.state_id
    result_path = state_dir / "result.json"
    arrays_path = state_dir / "actions.npz"
    inventory_path = state_dir / "trace_inventory.json"
    checkpoint_path = state_dir / "checkpoint.json"
    if result_path.exists() or arrays_path.exists() or inventory_path.exists():
        required = (result_path, arrays_path, inventory_path, checkpoint_path)
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"incomplete final artifact set in {state_dir}")
        for path in required:
            verify_sha_sidecar(path)
            if stat.S_IMODE(path.stat().st_mode) != 0o444:
                raise RuntimeError(f"completed output is not read-only: {path}")
        result = load_json(result_path)
        if result.get("schema") != SCHEMA or result.get("state", {}).get("state_id") != inputs.state_id:
            raise ValueError(f"completed result identity mismatch: {result_path}")
        return result_path

    header = expected_checkpoint_header(args=args, inputs=inputs, provenance=provenance)
    checkpoint = load_or_create_checkpoint(checkpoint_path, header)
    completed: dict[str, Any] = checkpoint["completed"]

    def save_checkpoint() -> None:
        checkpoint["completed"] = completed
        atomic_write_json(checkpoint_path, checkpoint)

    def controlled_infer(label: str, control: dict[str, Any]) -> dict[str, Any]:
        client.reset()
        request = dict(inputs.request)
        request["session_id"] = f"{inputs.state_id}:{label}"
        request[CONTROL_KEY] = control
        return client.infer(request)

    native_actions: dict[int, np.ndarray] = {}
    native_records: dict[int, dict[str, Any]] = {}
    for seed in BRANCH_SEEDS:
        label = f"native_record_seed_{seed}"
        client_trace, server_trace = trace_paths(args=args, state_id=inputs.state_id, seed=seed)
        if label in completed:
            record = completed[label]
            action = action_from_record(record, label)
            wait_for_trace(client_trace)
            if sha256_file(client_trace) != record["trace_file_sha256"]:
                raise ValueError(f"{label}: native trace file hash changed")
        else:
            client_trace.parent.mkdir(parents=True, exist_ok=True)
            control = request_control(
                mode="record",
                noise_seed=seed,
                trace_path=server_trace,
                action_noise_reference_path=None,
                replay_start=args.replay_start,
                replay_stop=args.replay_stop,
            )
            response = controlled_infer(label, control)
            action = action_from_response(response, label)
            audit = validate_server_audit(
                response,
                label=label,
                mode="record",
                noise_seed=seed,
                trace_path=server_trace,
                action_reference_path=None,
                replay_start=args.replay_start,
                replay_stop=args.replay_stop,
                expected_source_trace_hash=None,
                expected_source_action_hash=None,
                expected_recipient_action_hash=None,
            )
            wait_for_trace(client_trace)
            record = {
                **action_record(action, audit),
                "label": label,
                "branch_seed": seed,
                "client_trace_relative_path": str(
                    client_trace.relative_to(args.client_trace_root)
                ),
                "server_trace_path": str(server_trace),
                "trace_file_sha256": sha256_file(client_trace),
                "trace_size_bytes": client_trace.stat().st_size,
            }
            completed[label] = record
            save_checkpoint()
        native_actions[seed] = action
        native_records[seed] = record

    replay_actions: dict[tuple[int, int], np.ndarray] = {}
    replay_records: dict[tuple[int, int], dict[str, Any]] = {}
    for recipient_seed in BRANCH_SEEDS:
        recipient_client_trace, recipient_server_trace = trace_paths(
            args=args, state_id=inputs.state_id, seed=recipient_seed
        )
        del recipient_client_trace
        recipient_action_hash = str(
            native_records[recipient_seed]["server_audit"]["active_action_noise_sha256"]
        )
        for source_seed in BRANCH_SEEDS:
            label = f"replay_recipient_{recipient_seed}_source_{source_seed}"
            _, source_server_trace = trace_paths(
                args=args, state_id=inputs.state_id, seed=source_seed
            )
            if label in completed:
                record = completed[label]
                action = action_from_record(record, label)
            else:
                control = request_control(
                    mode="replay",
                    noise_seed=recipient_seed,
                    trace_path=source_server_trace,
                    action_noise_reference_path=recipient_server_trace,
                    replay_start=args.replay_start,
                    replay_stop=args.replay_stop,
                )
                response = controlled_infer(label, control)
                action = action_from_response(response, label)
                audit = validate_server_audit(
                    response,
                    label=label,
                    mode="replay",
                    noise_seed=recipient_seed,
                    trace_path=source_server_trace,
                    action_reference_path=recipient_server_trace,
                    replay_start=args.replay_start,
                    replay_stop=args.replay_stop,
                    expected_source_trace_hash=str(
                        native_records[source_seed]["server_audit"]["video_trace_sha256"]
                    ),
                    expected_source_action_hash=str(
                        native_records[source_seed]["server_audit"]["active_action_noise_sha256"]
                    ),
                    expected_recipient_action_hash=recipient_action_hash,
                )
                metrics = directional_metrics(
                    action, native_actions[recipient_seed], native_actions[source_seed]
                )
                distances = {
                    str(seed): float(np.linalg.norm(
                        action.astype(np.float64) - native_actions[seed].astype(np.float64)
                    ))
                    for seed in BRANCH_SEEDS
                }
                nearest_seed = min(BRANCH_SEEDS, key=lambda seed: (distances[str(seed)], seed))
                self_error = maximum_error(action, native_actions[recipient_seed])
                record = {
                    **action_record(action, audit),
                    "label": label,
                    "recipient_seed": recipient_seed,
                    "future_source_seed": source_seed,
                    "source_relation": (
                        "self_future_source"
                        if recipient_seed == source_seed
                        else "donor_future_source"
                    ),
                    "distances_to_native_actions": distances,
                    "nearest_native_seed": nearest_seed,
                    "correct_future_source_top1": nearest_seed == source_seed,
                    "self_replay_maximum_error": (
                        self_error if recipient_seed == source_seed else None
                    ),
                    "self_replay_bit_exact": (
                        bool(np.array_equal(action, native_actions[recipient_seed]))
                        if recipient_seed == source_seed
                        else None
                    ),
                    **metrics,
                }
                if recipient_seed == source_seed and (
                    record["sha256"] != native_records[recipient_seed]["sha256"]
                    or not record["self_replay_bit_exact"]
                    or self_error != 0.0
                ):
                    raise RuntimeError(f"{label}: strict native/self replay gate failed")
                completed[label] = record
                save_checkpoint()
            replay_actions[(recipient_seed, source_seed)] = action
            replay_records[(recipient_seed, source_seed)] = record

    expected_labels = {
        *(f"native_record_seed_{seed}" for seed in BRANCH_SEEDS),
        *(
            f"replay_recipient_{recipient}_source_{source}"
            for recipient in BRANCH_SEEDS
            for source in BRANCH_SEEDS
        ),
    }
    if set(completed) != expected_labels:
        raise RuntimeError(
            f"completed call labels differ: missing={sorted(expected_labels-set(completed))}, "
            f"extra={sorted(set(completed)-expected_labels)}"
        )
    for seed in BRANCH_SEEDS:
        replay = replay_actions[(seed, seed)]
        if not np.array_equal(replay, native_actions[seed]):
            raise RuntimeError(f"final strict self replay gate failed for seed {seed}")

    action_shape = native_actions[BRANCH_SEEDS[0]].shape
    action_dtype = native_actions[BRANCH_SEEDS[0]].dtype
    for label, record in completed.items():
        action = action_from_record(record, label)
        if action.shape != action_shape or action.dtype != action_dtype:
            raise RuntimeError(f"action schema differs at {label}")
    native_array = np.stack([native_actions[seed] for seed in BRANCH_SEEDS])
    replay_array = np.stack(
        [
            np.stack([replay_actions[(recipient, source)] for source in BRANCH_SEEDS])
            for recipient in BRANCH_SEEDS
        ]
    )
    atomic_write_npz(
        arrays_path,
        {
            "branch_seeds": np.asarray(BRANCH_SEEDS, dtype=np.int64),
            "native_actions": native_array,
            "replay_actions": replay_array,
        },
    )

    trace_rows: list[dict[str, Any]] = []
    for seed in BRANCH_SEEDS:
        client_trace, server_trace = trace_paths(args=args, state_id=inputs.state_id, seed=seed)
        expected = native_records[seed]["trace_file_sha256"]
        if sha256_file(client_trace) != expected:
            raise RuntimeError(f"trace changed before freeze for seed {seed}")
        frozen_trace = freeze_with_sidecar(client_trace)
        post_hash = frozen_trace["sha256"]
        if post_hash != expected:
            raise RuntimeError(f"trace changed while freezing for seed {seed}")
        trace_rows.append(
            {
                "branch_seed": seed,
                "client_relative_path": str(client_trace.relative_to(args.client_trace_root)),
                "server_path": str(server_trace),
                "sha256": post_hash,
                "size_bytes": client_trace.stat().st_size,
                "mode": stat.S_IMODE(client_trace.stat().st_mode),
                "sidecar_sha256": frozen_trace["sidecar_sha256"],
                "video_trace_sha256": native_records[seed]["server_audit"]["video_trace_sha256"],
                "action_noise_sha256": native_records[seed]["server_audit"]["active_action_noise_sha256"],
            }
        )
    trace_inventory = {
        "schema": INVENTORY_SCHEMA,
        "state_id": inputs.state_id,
        "trace_count": len(trace_rows),
        "traces": trace_rows,
    }
    atomic_write_json(inventory_path, trace_inventory, mode=0o444)

    native_hashes = {str(seed): native_records[seed]["sha256"] for seed in BRANCH_SEEDS}
    video_trace_hashes = {
        str(seed): native_records[seed]["server_audit"]["video_trace_sha256"]
        for seed in BRANCH_SEEDS
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "admission": args.admission,
        "scientific_admission": args.admission == "evaluation",
        "runner_sha256": header["runner_sha256"],
        "provenance": json_safe(provenance),
        "state": header["state"],
        "branch_seeds": list(BRANCH_SEEDS),
        "call_count": len(completed),
        "native_record_call_count": 4,
        "replay_grid_call_count": 16,
        "action_shape": list(action_shape),
        "action_dtype": str(action_dtype),
        "native_action_sha256_by_seed": native_hashes,
        "native_video_trace_sha256_by_seed": video_trace_hashes,
        "native_action_hash_distinct_count": len(set(native_hashes.values())),
        "native_video_trace_hash_distinct_count": len(set(video_trace_hashes.values())),
        "self_replay_all_bit_exact": all(
            replay_records[(seed, seed)]["self_replay_bit_exact"] for seed in BRANCH_SEEDS
        ),
        "self_replay_maximum_error": max(
            float(replay_records[(seed, seed)]["self_replay_maximum_error"])
            for seed in BRANCH_SEEDS
        ),
        "grid": [
            {
                key: value
                for key, value in replay_records[(recipient, source)].items()
                if key not in {"values"}
            }
            for recipient in BRANCH_SEEDS
            for source in BRANCH_SEEDS
        ],
        "artifacts": {
            "actions_npz": {
                "relative_path": arrays_path.name,
                "sha256": sha256_file(arrays_path),
                "size_bytes": arrays_path.stat().st_size,
            },
            "trace_inventory": {
                "relative_path": inventory_path.name,
                "sha256": sha256_file(inventory_path),
                "size_bytes": inventory_path.stat().st_size,
            },
            "checkpoint": {"relative_path": checkpoint_path.name},
        },
    }
    atomic_write_json(result_path, result, mode=0o444)
    result_inventory = freeze_with_sidecar(result_path)
    arrays_inventory = freeze_with_sidecar(arrays_path)
    traces_inventory = freeze_with_sidecar(inventory_path)
    checkpoint["status"] = "complete"
    checkpoint["final_artifacts"] = {
        "result": result_inventory,
        "actions": arrays_inventory,
        "traces": traces_inventory,
    }
    atomic_write_json(checkpoint_path, checkpoint, mode=0o444)
    freeze_with_sidecar(checkpoint_path)
    return result_path


def selected_states(states: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[Mapping[str, Any]]:
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard arguments must satisfy 0 <= index < count")
    requested = set(args.state_id)
    known = {str(state["state_id"]) for state in states}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown requested state IDs: {sorted(unknown)}")
    selected = [
        state
        for index, state in enumerate(states)
        if index % args.shard_count == args.shard_index
        and (not requested or str(state["state_id"]) in requested)
    ]
    if not selected:
        raise ValueError("state selection is empty")
    return selected


def main() -> None:
    args = parse_args()
    if args.replay_start != 0 or args.replay_stop != EXPECTED_SOLVER_STEPS:
        raise ValueError("this runner is frozen to the complete matched 16-step interval [0,16)")
    if not args.server_trace_root.is_absolute():
        raise ValueError("--server-trace-root must be an absolute server-visible path")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.client_trace_root.mkdir(parents=True, exist_ok=True)

    if args.admission == "evaluation":
        if args.debug_bundle_root is not None:
            raise ValueError("evaluation cannot use --debug-bundle-root")
        manifest, resource_by_id, data_root, frozen = validate_manifest_and_receipt(args)
        receipt = load_json(data_root / "download_receipt.json")
        receipt_by_id = {str(item["resource_id"]): item for item in receipt["resources"]}
        states = selected_states(manifest["states"], args)
        inputs = [
            build_frozen_input(
                state,
                resource_by_id=resource_by_id,
                receipt_by_id=receipt_by_id,
                data_root=data_root,
                modality=frozen["modality"],
            )
            for state in states
        ]
        provenance = {key: value for key, value in frozen.items() if key != "modality"}
    else:
        if args.manifest is not None or args.expected_manifest_sha256 is not None:
            raise ValueError("excluded debug smoke must not be linked to the evaluation manifest")
        if args.state_id or args.shard_count != 1 or args.shard_index != 0:
            raise ValueError("excluded debug smoke is exactly one bundled state")
        inputs = [build_debug_input(args)]
        provenance = {
            "manifest_id": None,
            "manifest_file_sha256": None,
            "dataset_revision": None,
            "excluded_debug_bundle_sha256": hashlib.sha256(
                canonical_json_bytes(inputs[0].audit["cameras"])
            ).hexdigest(),
        }

    preflight = {
        "schema": "dreamzero-future-transplant-input-preflight-v1",
        "status": "validated",
        "admission": args.admission,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "selected_state_count": len(inputs),
        "selected_state_ids": [item.state_id for item in inputs],
        "branch_seeds": list(BRANCH_SEEDS),
        "call_count_per_state": 20,
        "planned_call_count": 20 * len(inputs),
        "replay_interval": [args.replay_start, args.replay_stop],
        "provenance": provenance,
        "input_fingerprints": {
            item.state_id: item.audit["input_fingerprint"] for item in inputs
        },
    }
    preflight_name = (
        "input_preflight.json"
        if args.shard_count == 1
        else f"input_preflight.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json"
    )
    preflight_path = args.output_root / preflight_name
    if preflight_path.exists():
        existing = load_json(preflight_path)
        if existing != json_safe(preflight):
            raise ValueError("existing input preflight differs from current invocation")
    else:
        atomic_write_json(preflight_path, preflight, mode=0o444)
        freeze_with_sidecar(preflight_path)
    if args.validate_inputs_only:
        print(json.dumps(preflight, sort_keys=True, allow_nan=False))
        return

    client = DreamZeroClient(
        args.host,
        args.port,
        connect_timeout=args.connect_timeout,
        response_timeout=args.response_timeout,
    )
    runtime_provenance = {
        **provenance,
        "server": {
            "host": args.host,
            "port": args.port,
            "metadata": json_safe(client.metadata),
            "metadata_sha256": hashlib.sha256(
                canonical_json_bytes(json_safe(client.metadata))
            ).hexdigest(),
        },
    }
    result_paths: list[Path] = []
    try:
        for index, inputs_for_state in enumerate(inputs, start=1):
            path = run_state(
                args=args,
                client=client,
                inputs=inputs_for_state,
                provenance=runtime_provenance,
            )
            result_paths.append(path)
            print(
                json.dumps(
                    {
                        "event": "state_complete",
                        "count": index,
                        "total": len(inputs),
                        "state_id": inputs_for_state.state_id,
                        "result": str(path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        client.close()

    inventory_name = (
        "run_inventory.json"
        if args.shard_count == 1
        else f"run_inventory.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json"
    )
    run_inventory_path = args.output_root / inventory_name
    run_inventory = {
        "schema": "dreamzero-future-transplant-run-inventory-v1",
        "status": "complete",
        "admission": args.admission,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "state_count": len(result_paths),
        "state_ids": [item.state_id for item in inputs],
        "call_count": 20 * len(result_paths),
        "results": [
            {
                "state_id": inputs_for_path.state_id,
                "relative_path": str(path.relative_to(args.output_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "mode": stat.S_IMODE(path.stat().st_mode),
            }
            for inputs_for_path, path in zip(inputs, result_paths, strict=True)
        ],
    }
    if run_inventory_path.exists():
        if load_json(run_inventory_path) != run_inventory:
            raise ValueError("existing run inventory differs from completed run")
        verify_sha_sidecar(run_inventory_path)
    else:
        atomic_write_json(run_inventory_path, run_inventory, mode=0o444)
        freeze_with_sidecar(run_inventory_path)


if __name__ == "__main__":
    main()
