#!/usr/bin/env python3
"""Build deterministic, publication-neutral artifacts for the LingBot cohort.

The script is intentionally post-run only.  It requires a complete cohort and a
completed output from ``summarize_lingbot_future_transplants.py``.  Every plot,
table, and media frame includes all admitted states or a fixed aggregate over
all admitted states; no state is selected using an outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PRIMARY_METRICS = (
    "retrieval_accuracy_off_diagonal",
    "retrieval_accuracy_all",
    "projection",
    "distance_reduction",
    "cosine_alignment",
    "orthogonal_residual",
    "normalized_final_distance",
)
CANONICAL_MANIFEST_SHA256 = "9c8d7dd547955e4622d98c7eb42fffed157f6db983f41f02929b28a434be47b4"
CANONICAL_RUNNER_SHA256 = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
CANONICAL_UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
CANONICAL_CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"
CANONICAL_COMPATIBILITY_SHIM = Path(
    "/home/ubuntu/if_external/compat/flash_attn/__init__.py"
)
CANONICAL_COMPATIBILITY_SHIM_SHA256 = "7f1448bdeae5f4991112d78131688d417836c91fee79624929cda5d2f135bec8"
OPTIONAL_METRICS = (
    "gaussian_perturbation_normalized",
    "donor_future_recipient_cache_projection",
    "recipient_future_donor_cache_projection",
)
CONTROL_ARRAY_KEYS = (
    "native_actions",
    "latent_grid_actions",
    "cache_replay_actions",
    "native_executed_actions",
    "latent_grid_executed_actions",
    "cache_replay_executed_actions",
    "gaussian_actions",
    "gaussian_executed_actions",
    "donor_future_recipient_cache_actions",
    "recipient_future_donor_cache_actions",
    "donor_future_recipient_cache_executed_actions",
    "recipient_future_donor_cache_executed_actions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--live-runner", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument(
        "--admission", choices=("development", "evaluation"), required=True
    )
    parser.add_argument("--expected-states", type=int, default=30)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=50_000)
    parser.add_argument("--permutation-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=260904)
    parser.add_argument("--video-fps", type=float, default=2.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary(path: Path) -> Path:
    return path.with_name(
        f"{path.stem}.tmp.{os.getpid()}.{time.time_ns()}{path.suffix}"
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def atomic_image(path: Path, value: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    value.save(temporary, format="PNG", optimize=False, compress_level=9)
    os.replace(temporary, path)


def atomic_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    figure.savefig(
        temporary,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "build_lingbot_postrun_artifacts.py"},
    )
    plt.close(figure)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def command_output(arguments: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checkpoint_inventory(
    checkpoint: Path, *, launch_epoch_seconds: float
) -> dict[str, Any]:
    payload_paths = sorted(
        path
        for path in checkpoint.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts
    )
    entries = []
    aggregate = hashlib.sha256()
    for path in payload_paths:
        relative = path.relative_to(checkpoint).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        entries.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": digest,
                "mtime_ns": path.stat().st_mtime_ns,
                "mtime_predates_launch": path.stat().st_mtime < launch_epoch_seconds,
            }
        )
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")

    metadata_entries = []
    metadata_root = checkpoint / ".cache" / "huggingface" / "download"
    for path in sorted(metadata_root.rglob("*.metadata")) if metadata_root.is_dir() else []:
        lines = path.read_text(encoding="utf-8").splitlines()
        metadata_entries.append(
            {
                "path": path.relative_to(metadata_root).as_posix(),
                "payload_path": path.relative_to(metadata_root).as_posix()[
                    : -len(".metadata")
                ],
                "revision": lines[0] if lines else None,
                "etag": lines[1] if len(lines) > 1 else None,
                "mtime_ns": path.stat().st_mtime_ns,
                "mtime_predates_launch": path.stat().st_mtime
                < launch_epoch_seconds,
            }
        )
    payload_by_path = {item["path"]: item for item in entries}
    if {item["payload_path"] for item in metadata_entries} != set(payload_by_path):
        raise RuntimeError("checkpoint payloads and Hugging Face metadata do not pair 1:1")
    for item in metadata_entries:
        payload = payload_by_path[item["payload_path"]]
        etag = str(item["etag"])
        if len(etag) == 64:
            observed_etag = payload["sha256"]
            etag_kind = "lfs_sha256"
        elif len(etag) == 40:
            value = (checkpoint / item["payload_path"]).read_bytes()
            git_blob = hashlib.sha1(usedforsecurity=False)
            git_blob.update(f"blob {len(value)}\0".encode("ascii"))
            git_blob.update(value)
            observed_etag = git_blob.hexdigest()
            etag_kind = "git_blob_sha1"
        else:
            raise RuntimeError(f"unrecognized Hugging Face etag: {etag!r}")
        item["etag_kind"] = etag_kind
        item["etag_matches_payload"] = observed_etag == etag
        if not item["etag_matches_payload"]:
            raise RuntimeError(
                f"checkpoint payload does not match Hugging Face etag: {item['payload_path']}"
            )
    if not all(
        item["mtime_predates_launch"] for item in [*entries, *metadata_entries]
    ):
        raise RuntimeError("a checkpoint payload/metadata mtime does not predate launch")
    return {
        "checkpoint_root": str(checkpoint.resolve()),
        "payload_file_count": len(entries),
        "payload_total_bytes": sum(item["bytes"] for item in entries),
        "aggregate_sha256": aggregate.hexdigest(),
        "launch_epoch_seconds": launch_epoch_seconds,
        "all_payload_and_metadata_mtimes_predate_launch": True,
        "all_huggingface_etags_match_payload": True,
        "files": entries,
        "huggingface_download_metadata": metadata_entries,
        "huggingface_revisions": sorted(
            {item["revision"] for item in metadata_entries if item["revision"]}
        ),
    }


def package_versions(names: list[str]) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output


def collect_provenance(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_sha256: str,
    cohort_provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    lingbot_root = args.lingbot_root.resolve()
    checkpoint = args.checkpoint.resolve()
    live_runner = args.live_runner.resolve()
    launch_receipt_path = args.launch_receipt.resolve()
    launch_receipt = load_json(launch_receipt_path)
    if launch_receipt.get("capture_is_outcome_blind") is not True:
        raise RuntimeError("launch receipt was not captured outcome-blind")
    if launch_receipt.get("sha256", {}).get("manifest") != manifest_sha256:
        raise RuntimeError("launch receipt manifest hash mismatch")
    runner_sha256 = sha256_file(live_runner)
    if launch_receipt.get("sha256", {}).get("runner") != runner_sha256:
        raise RuntimeError("launch receipt runner hash mismatch")
    if cohort_provenance["runner_sha256"] != [runner_sha256]:
        raise RuntimeError("live runner differs from the completed cohort runner")
    canonical_observed = {
        "manifest_sha256": manifest_sha256,
        "runner_sha256": runner_sha256,
        "upstream_commit": cohort_provenance["upstream_commit"][0],
        "checkpoint_revision": cohort_provenance["checkpoint_revision"][0],
    }
    canonical_expected = {
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "runner_sha256": CANONICAL_RUNNER_SHA256,
        "upstream_commit": CANONICAL_UPSTREAM_COMMIT,
        "checkpoint_revision": CANONICAL_CHECKPOINT_REVISION,
    }
    if canonical_observed != canonical_expected:
        raise RuntimeError(
            f"cohort differs from canonical frozen identity: {canonical_observed}"
        )

    launch_paths = launch_receipt.get("paths", {})
    for key, expected in (
        ("repo", lingbot_root),
        ("checkpoint", checkpoint),
        ("runner", live_runner),
        ("output_root", args.result_root.resolve()),
    ):
        raw = launch_paths.get(key)
        if not raw or Path(raw).resolve() != expected:
            raise RuntimeError(f"launch receipt {key} path mismatch: {raw!r}")
    launched_manifest = Path(launch_paths.get("manifest", "")).resolve()
    if not launched_manifest.is_file() or sha256_file(launched_manifest) != manifest_sha256:
        raise RuntimeError("launched manifest is absent or differs from the cohort manifest")

    parsed_commands = [
        shlex.split(str(process.get("command", "")))
        for process in launch_receipt.get("processes", [])
    ]
    runner_commands = [parts for parts in parsed_commands if str(live_runner) in parts]
    if not runner_commands:
        raise RuntimeError("launch receipt contains no command for the live runner")
    shard_indices: set[int] = set()
    for parts in runner_commands:
        for flag in ("--gaussian-controls", "--run-factorial"):
            if flag not in parts:
                raise RuntimeError(f"live command omitted required flag {flag}")
        for option, expected in (
            ("--admission", args.admission),
            ("--shard-count", "2"),
            ("--output-root", str(args.result_root.resolve())),
            ("--lingbot-root", str(lingbot_root)),
            ("--checkpoint", str(checkpoint)),
            ("--manifest", str(launched_manifest)),
        ):
            if option not in parts or parts[parts.index(option) + 1] != expected:
                raise RuntimeError(f"live command has incorrect {option}")
        if "--shard-index" not in parts:
            raise RuntimeError("live command omitted --shard-index")
        shard_indices.add(int(parts[parts.index("--shard-index") + 1]))
    if shard_indices != {0, 1}:
        raise RuntimeError(f"launch receipt does not cover both shards: {shard_indices}")

    launch_start_times = [
        datetime.strptime(str(process["start_time"]), "%a %b %d %H:%M:%S %Y")
        .replace(tzinfo=UTC)
        .timestamp()
        for process in launch_receipt.get("processes", [])
        if process.get("start_time")
    ]
    if not launch_start_times:
        raise RuntimeError("launch receipt contains no parseable process start time")
    checkpoint_content = checkpoint_inventory(
        checkpoint, launch_epoch_seconds=min(launch_start_times)
    )
    expected_revision = cohort_provenance["checkpoint_revision"][0]
    if checkpoint_content["huggingface_revisions"] != [expected_revision]:
        raise RuntimeError(
            "checkpoint Hugging Face metadata revisions do not match cohort: "
            f"{checkpoint_content['huggingface_revisions']} != {[expected_revision]}"
        )

    repo_commit = command_output(["git", "rev-parse", "HEAD"], cwd=lingbot_root)
    if repo_commit != cohort_provenance["upstream_commit"][0]:
        raise RuntimeError("LingBot repo commit differs from completed cohort")
    repo_status = command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=lingbot_root,
    )
    remote_url = command_output(
        ["git", "remote", "get-url", "origin"], cwd=lingbot_root
    )
    source_files = (
        "wan_va/wan_va_server.py",
        "wan_va/modules/model.py",
        "wan_va/configs/va_libero_cfg.py",
        "evaluation/libero/client.py",
    )
    source_hashes = {
        name: sha256_file(lingbot_root / name) for name in source_files
    }
    compatibility_shim = CANONICAL_COMPATIBILITY_SHIM.resolve()
    if (
        not compatibility_shim.is_file()
        or sha256_file(compatibility_shim)
        != CANONICAL_COMPATIBILITY_SHIM_SHA256
    ):
        raise RuntimeError("LingBot import-only FlashAttention compatibility shim changed")
    if compatibility_shim.stat().st_mtime >= min(launch_start_times):
        raise RuntimeError("compatibility shim mtime does not predate core launch")

    try:
        import torch

        torch_receipt: dict[str, Any] = {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn": int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None,
            "gpus": [],
        }
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                torch_receipt["gpus"].append(
                    {
                        "visible_index": index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "compute_capability": [properties.major, properties.minor],
                    }
                )
    except Exception as exc:
        torch_receipt = {"inspection_error": repr(exc)}

    try:
        nvidia_smi = command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        nvidia_smi = f"unavailable: {exc!r}"

    log_receipts = []
    for raw_path in launch_receipt.get("paths", {}).get("logs", []):
        path = Path(raw_path)
        log_receipts.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    receipt = {
        "schema_version": 1,
        "repository": {
            "path": str(lingbot_root),
            "origin": remote_url,
            "commit": repo_commit,
            "tracked_files_clean": not bool(
                command_output(["git", "diff", "--name-only"], cwd=lingbot_root)
                or command_output(
                    ["git", "diff", "--cached", "--name-only"], cwd=lingbot_root
                )
            ),
            "porcelain_status": repo_status.splitlines() if repo_status else [],
            "source_sha256": source_hashes,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "huggingface_repo": "robbyant/lingbot-va-posttrain-libero-long",
            "huggingface_revision": expected_revision,
            "content_inventory_file": "checkpoint_content_manifest.json",
            "payload_file_count": checkpoint_content["payload_file_count"],
            "payload_total_bytes": checkpoint_content["payload_total_bytes"],
            "aggregate_sha256": checkpoint_content["aggregate_sha256"],
            "launch_time_checkpoint_digest_available": False,
            "digest_timing": "postrun",
            "all_payload_and_metadata_mtimes_predate_launch": checkpoint_content[
                "all_payload_and_metadata_mtimes_predate_launch"
            ],
            "all_huggingface_etags_match_payload": checkpoint_content[
                "all_huggingface_etags_match_payload"
            ],
            "temporal_evidence_limit": "No checkpoint aggregate was captured at launch; postrun content is bound to prelaunch Hugging Face metadata/etags and file mtimes, which cannot exclude a mutate-and-restore scenario.",
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(
                [
                    "numpy",
                    "torch",
                    "torchvision",
                    "transformers",
                    "diffusers",
                    "flash-attn",
                    "matplotlib",
                    "Pillow",
                    "imageio",
                ]
            ),
            "torch_cuda": torch_receipt,
            "nvidia_smi_gpu_inventory": nvidia_smi.splitlines(),
            "import_compatibility": {
                "pythonpath_used_by_core": "/home/ubuntu/if_external/compat",
                "launch_receipt_captured_pythonpath": False,
                "inference_basis": "Core execution could import LingBot without installed flash_attn only through this staged path; launch receipt captured selected environment variables but omitted PYTHONPATH.",
                "shim_path": str(compatibility_shim),
                "shim_sha256": sha256_file(compatibility_shim),
                "shim_mtime_ns": compatibility_shim.stat().st_mtime_ns,
                "shim_mtime_predates_launch": True,
                "shim_contract": "import-only; raises if flash_attn_func is called; released server selects attn_mode='torch'",
            },
        },
        "frozen_design": {
            "manifest_path": str((args.result_root / "manifest.json").resolve()),
            "manifest_sha256": manifest_sha256,
            "branch_ids": manifest["branch_ids"],
            "video_seeds": manifest["video_seeds"],
            "action_seeds": manifest["action_seeds"],
            "state_count": len(
                [
                    record
                    for record in manifest["states"]
                    if record["admission"] == args.admission
                ]
            ),
        },
        "intervention": {
            "live_runner_path": str(live_runner),
            "live_runner_sha256": runner_sha256,
            "runner_functions": [
                "generate_future",
                "install_future",
                "generate_action",
                "snapshot_cache",
                "restore_cache",
            ],
            "upstream_video_action_boundary": "wan_va/wan_va_server.py:_infer, after video loop and before action loop",
            "upstream_cache_location": "wan_va/modules/model.py:WanAttention.attn_caches['pos']",
            "cache_fields": ["k", "v", "id", "mask", "is_pred"],
            "action_coordinates_written_by_intervention": False,
        },
        "artifact_generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "launch": launch_receipt,
        "final_log_receipts": log_receipts,
        "paths": {
            "result_root": str(args.result_root.resolve()),
            "analysis_root": str(args.analysis_root.resolve()),
            "artifact_root": str(args.output_root.resolve()),
            "launch_receipt": str(launch_receipt_path),
        },
    }
    return receipt, checkpoint_content


def read_state_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for source_row in source:
        row: dict[str, Any] = {"state_id": source_row["state_id"]}
        for key, raw in source_row.items():
            if key == "state_id":
                continue
            if raw is None or raw == "":
                continue
            row[key] = int(raw) if key in {"task_id", "future_hashes_unique", "cache_hashes_unique"} else float(raw)
        rows.append(row)
    return rows


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _flatten_action(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value.reshape(value.shape[0], -1)


def _state_order(record: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(record["task_id"]),
        int(record.get("initial_state_index", -1)),
        str(record["state_id"]),
    )


def _projection_matrix(native_raw: np.ndarray, grid_raw: np.ndarray) -> np.ndarray:
    native = _flatten_action(native_raw)
    grid = np.asarray(grid_raw, dtype=np.float64).reshape(4, 4, -1)
    result = np.zeros((4, 4), dtype=np.float64)
    for recipient in range(4):
        for source in range(4):
            if recipient == source:
                result[recipient, source] = 0.0
                continue
            axis = native[source] - native[recipient]
            denominator = float(np.dot(axis, axis))
            result[recipient, source] = (
                float(np.dot(grid[recipient, source] - native[recipient], axis))
                / denominator
                if denominator > 1e-12
                else math.nan
            )
    return result


def validate_cohort(
    *,
    result_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    controls: list[dict[str, Any]] = []
    nearest_by_state: dict[str, np.ndarray] = {}
    media: list[dict[str, Any]] = []
    runner_hashes: set[str] = set()
    upstream_commits: set[str] = set()
    checkpoint_revisions: set[str] = set()

    for record in records:
        state_id = str(record["state_id"])
        state_root = result_root / state_id
        result_path = state_root / "result.json"
        arrays_path = state_root / "actions.npz"
        if not result_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(f"missing completed state artifacts: {state_id}")
        metadata = load_json(result_path)
        if metadata.get("status") != "complete":
            raise RuntimeError(f"state {state_id} status is {metadata.get('status')!r}")
        if metadata.get("state_id") != state_id:
            raise RuntimeError(f"state ID mismatch in {result_path}")
        if metadata.get("admission") != record["admission"]:
            raise RuntimeError(f"admission mismatch for {state_id}")
        if metadata.get("prompt") != record["prompt"]:
            raise RuntimeError(f"prompt mismatch for {state_id}")
        for key in ("branch_ids", "video_seeds", "action_seeds"):
            if metadata.get(key) != manifest[key]:
                raise RuntimeError(f"{key} mismatch for {state_id}")
        if metadata.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError(f"manifest hash mismatch for {state_id}")
        if metadata.get("input_sha256") != record["input_sha256"]:
            raise RuntimeError(f"recorded input hash mismatch for {state_id}")
        observation_path = Path(record["observation_path"])
        if sha256_file(observation_path) != record["input_sha256"]:
            raise RuntimeError(f"current frozen input hash mismatch for {state_id}")
        if metadata.get("actions_sha256") != sha256_file(arrays_path):
            raise RuntimeError(f"actions hash mismatch for {state_id}")
        runner_hashes.add(str(metadata.get("runner_sha256")))
        upstream_commits.add(str(metadata.get("upstream_commit")))
        checkpoint_revisions.add(str(metadata.get("checkpoint_revision")))

        with np.load(arrays_path, allow_pickle=False) as archive:
            missing = sorted(set(CONTROL_ARRAY_KEYS) - set(archive.files))
            if missing:
                raise RuntimeError(f"state {state_id} lacks required arrays: {missing}")
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError(f"state {state_id} contains a non-finite action")

        expected_shapes = {
            "native_actions": (4, 7, 4, 4),
            "latent_grid_actions": (4, 4, 7, 4, 4),
            "cache_replay_actions": (4, 7, 4, 4),
            "native_executed_actions": (4, 7, 3, 4),
            "latent_grid_executed_actions": (4, 4, 7, 3, 4),
            "cache_replay_executed_actions": (4, 7, 3, 4),
        }
        for key, expected in expected_shapes.items():
            if tuple(arrays[key].shape) != expected:
                raise RuntimeError(
                    f"state {state_id} {key} shape {arrays[key].shape} != {expected}"
                )

        execution_errors = {
            "native_execution_slice": _max_abs(
                arrays["native_executed_actions"], arrays["native_actions"][..., 1:, :]
            ),
            "grid_execution_slice": _max_abs(
                arrays["latent_grid_executed_actions"],
                arrays["latent_grid_actions"][..., 1:, :],
            ),
            "cache_execution_slice": _max_abs(
                arrays["cache_replay_executed_actions"],
                arrays["cache_replay_actions"][..., 1:, :],
            ),
        }
        native = arrays["native_executed_actions"]
        grid = arrays["latent_grid_executed_actions"]
        cache_replay = arrays["cache_replay_executed_actions"]
        diagonal = np.stack([grid[index, index] for index in range(4)])
        self_latent_error = _max_abs(diagonal, native)
        self_cache_error = _max_abs(cache_replay, native)

        donor_future_recipient_cache = arrays[
            "donor_future_recipient_cache_executed_actions"
        ]
        recipient_future_donor_cache = arrays[
            "recipient_future_donor_cache_executed_actions"
        ]
        routing_recipient_error = _max_abs(
            donor_future_recipient_cache,
            np.broadcast_to(native[:, None], donor_future_recipient_cache.shape),
        )
        routing_donor_error = _max_abs(recipient_future_donor_cache, grid)

        failures = {
            **execution_errors,
            "self_latent": self_latent_error,
            "self_cache": self_cache_error,
            "recipient_cache_routing": routing_recipient_error,
            "donor_cache_routing": routing_donor_error,
        }
        if any(value != 0.0 for value in failures.values()):
            raise RuntimeError(f"state {state_id} failed exact controls: {failures}")
        if float(metadata.get("native_self_latent_max_abs_error", math.nan)) != 0.0:
            raise RuntimeError(f"metadata self-latent control failed for {state_id}")
        if float(metadata.get("native_self_cache_max_abs_error", math.nan)) != 0.0:
            raise RuntimeError(f"metadata self-cache control failed for {state_id}")
        if int(metadata.get("native_future_hashes_unique", -1)) != 4:
            raise RuntimeError(f"future diversity control failed for {state_id}")
        if int(metadata.get("native_cache_hashes_unique", -1)) != 4:
            raise RuntimeError(f"cache diversity control failed for {state_id}")
        action_noise_hashes = list(metadata.get("action_noise_hashes", []))
        grid_noise_hashes = metadata.get("grid_action_noise_hashes")
        expected_grid_hashes = [[action_noise_hashes[row]] * 4 for row in range(4)] if len(action_noise_hashes) == 4 else None
        if len(set(action_noise_hashes)) != 4 or grid_noise_hashes != expected_grid_hashes:
            raise RuntimeError(f"action-noise binding control failed for {state_id}")
        if metadata.get("grid_axis_0") != "recipient_action_noise_source":
            raise RuntimeError(f"grid axis 0 mismatch for {state_id}")
        if metadata.get("grid_axis_1") != "future_source":
            raise RuntimeError(f"grid axis 1 mismatch for {state_id}")
        if metadata.get("action_coordinate_intervention") != "none":
            raise RuntimeError(f"action intervention declaration mismatch for {state_id}")

        native_flat = _flatten_action(native)
        grid_flat = np.asarray(grid, dtype=np.float64).reshape(4, 4, -1)
        distances = np.linalg.norm(
            grid_flat[:, :, None, :] - native_flat[None, None, :, :], axis=-1
        )
        nearest_by_state[state_id] = np.argmin(distances, axis=-1).astype(np.int8)
        controls.append(
            {
                "state_id": state_id,
                "task_id": int(record["task_id"]),
                "self_latent_max_abs_error": self_latent_error,
                "self_cache_max_abs_error": self_cache_error,
                "recipient_cache_routing_max_abs_error": routing_recipient_error,
                "donor_cache_routing_max_abs_error": routing_donor_error,
                "future_hashes_unique": 4,
                "cache_hashes_unique": 4,
                "action_noise_hashes_unique": 4,
            }
        )
        media.append(
            {
                "record": record,
                "observation_path": observation_path,
                "projection_matrix": _projection_matrix(native, grid),
            }
        )

    provenance = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "runner_sha256": sorted(runner_hashes),
        "upstream_commit": sorted(upstream_commits),
        "checkpoint_revision": sorted(checkpoint_revisions),
    }
    for label in ("runner_sha256", "upstream_commit", "checkpoint_revision"):
        if len(provenance[label]) != 1:
            raise RuntimeError(f"cohort has mixed {label}: {provenance[label]}")
    nearest = np.stack([nearest_by_state[str(record["state_id"])] for record in records])
    return controls, {"nearest": nearest, "provenance": provenance}, media


def validate_analysis(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    manifest_sha256: str,
    nearest: np.ndarray,
) -> list[str]:
    expected_ids = [str(record["state_id"]) for record in records]
    if [str(row["state_id"]) for row in rows] != expected_ids:
        raise RuntimeError("analyzer state rows are absent or not in frozen manifest order")
    if int(summary.get("complete_state_count", -1)) != len(records):
        raise RuntimeError("analyzer did not include every admitted state")
    if int(summary.get("expected_state_count", -1)) != len(records):
        raise RuntimeError("analyzer expected-state count differs from manifest")
    if summary.get("failed_or_missing_states"):
        raise RuntimeError("analyzer reports failed or missing states")
    if summary.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("analyzer manifest hash differs from result manifest")
    if float(summary.get("max_self_latent_abs_error", math.nan)) != 0.0:
        raise RuntimeError("analyzer reports a failed self-latent control")
    if float(summary.get("max_self_cache_abs_error", math.nan)) != 0.0:
        raise RuntimeError("analyzer reports a failed self-cache control")
    if summary.get("all_states_four_unique_futures") is not True:
        raise RuntimeError("analyzer reports a future-diversity failure")
    if summary.get("all_states_four_unique_caches") is not True:
        raise RuntimeError("analyzer reports a cache-diversity failure")

    metric_names = [name for name in PRIMARY_METRICS if all(name in row for row in rows)]
    metric_names.extend(
        name for name in OPTIONAL_METRICS if all(name in row for row in rows)
    )
    if tuple(metric_names[: len(PRIMARY_METRICS)]) != PRIMARY_METRICS:
        raise RuntimeError("analyzer is missing a primary metric")
    if not all(
        math.isfinite(float(row[name])) for row in rows for name in metric_names
    ):
        raise RuntimeError("analyzer contains a non-finite inferential metric")
    targets = np.broadcast_to(np.arange(4)[None, :], (4, 4))
    diagonal = np.eye(4, dtype=bool)
    for index, row in enumerate(rows):
        all_accuracy = float(np.mean(nearest[index] == targets))
        off_accuracy = float(np.mean((nearest[index] == targets)[~diagonal]))
        if abs(float(row["retrieval_accuracy_all"]) - all_accuracy) > 1e-12:
            raise RuntimeError(f"retrieval recomputation mismatch: {row['state_id']}")
        if abs(float(row["retrieval_accuracy_off_diagonal"]) - off_accuracy) > 1e-12:
            raise RuntimeError(f"off-diagonal retrieval mismatch: {row['state_id']}")
    return metric_names


def bootstrap_outputs(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    values = np.asarray(
        [[float(row[name]) for name in metric_names] for row in rows], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(repetitions, len(rows)), dtype=np.int32)
    draws = np.empty((repetitions, len(metric_names)), dtype=np.float64)
    for column in range(len(metric_names)):
        draws[:, column] = values[indices, column].mean(axis=1)
    summaries = []
    for column, name in enumerate(metric_names):
        low, high = np.quantile(draws[:, column], [0.025, 0.975])
        summaries.append(
            {
                "metric": name,
                "estimate": float(values[:, column].mean()),
                "bootstrap_mean": float(draws[:, column].mean()),
                "bootstrap_95_ci_low": float(low),
                "bootstrap_95_ci_high": float(high),
                "n_states": len(rows),
                "repetitions": repetitions,
                "seed": seed,
            }
        )
    return draws, summaries


def permutation_outputs(
    nearest: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if repetitions <= 0:
        raise ValueError("permutation repetitions must be positive")
    if tuple(nearest.shape[1:]) != (4, 4):
        raise ValueError(f"expected nearest array [state,4,4], got {nearest.shape}")
    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.int8)
    rng = np.random.default_rng(seed)
    # Retain float64 so null draws that equal the observed rational accuracy are
    # counted as ties rather than moving across the tail threshold on a float32
    # round trip.
    null = np.empty((repetitions, 2), dtype=np.float64)
    diagonal = np.eye(4, dtype=bool)
    targets = np.broadcast_to(np.arange(4)[None, :], (4, 4))
    observed = np.asarray(
        [
            float(np.mean(nearest == targets[None])),
            float(np.mean((nearest == targets[None])[:, ~diagonal])),
        ],
        dtype=np.float64,
    )
    chunk_size = 1_000
    for start in range(0, repetitions, chunk_size):
        count = min(chunk_size, repetitions - start)
        choices = rng.integers(
            0, len(permutations), size=(count, nearest.shape[0]), dtype=np.int16
        )
        # A future branch has one identity across all four action-noise rows.
        # Therefore each state receives one shared four-label permutation, not
        # four independently randomized row-level mappings.
        randomized_targets = permutations[choices]
        matches = nearest[None] == randomized_targets[:, :, None, :]
        null[start : start + count, 0] = matches.mean(axis=(1, 2, 3))
        null[start : start + count, 1] = matches[..., ~diagonal].mean(axis=(1, 2))
    names = ("retrieval_accuracy_all", "retrieval_accuracy_off_diagonal")
    summaries = []
    for column, name in enumerate(names):
        low, high = np.quantile(null[:, column], [0.025, 0.975])
        summaries.append(
            {
                "metric": name,
                "observed": float(observed[column]),
                "null_mean": float(null[:, column].mean()),
                "null_95_interval_low": float(low),
                "null_95_interval_high": float(high),
                "monte_carlo_p_greater_equal": float(
                    (np.count_nonzero(null[:, column] >= observed[column] - 1e-15) + 1)
                    / (repetitions + 1)
                ),
                "repetitions": repetitions,
                "seed": seed,
                "permutation_unit": "one shared permutation of all four future-source labels over all four native-action labels per state; the mapping is shared across recipient rows and off-diagonal scoring retains the original 12 cells",
            }
        )
    return null, summaries


def task_rows(
    rows: list[dict[str, Any]], metric_names: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["task_id"])].append(row)
    output = []
    for task_id in sorted(grouped):
        group = grouped[task_id]
        item: dict[str, Any] = {"task_id": task_id, "n_states": len(group)}
        for name in metric_names:
            item[name] = float(np.mean([float(row[name]) for row in group]))
        output.append(item)
    return output


def plot_retrieval_by_task(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    task_ids = sorted({int(row["task_id"]) for row in rows})
    for task_id in task_ids:
        values = [
            float(row["retrieval_accuracy_off_diagonal"])
            for row in rows
            if int(row["task_id"]) == task_id
        ]
        offsets = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            np.asarray([task_id] * len(values)) + offsets,
            values,
            color="#386cb0",
            alpha=0.75,
            s=28,
        )
        axis.plot(task_id, np.mean(values), marker="D", color="#1b1b1b", ms=5)
    axis.axhline(0.25, color="#777777", linestyle="--", linewidth=1, label="chance (0.25)")
    axis.set(
        xlabel="LIBERO-Long task ID",
        ylabel="Off-diagonal four-way future-source retrieval",
        ylim=(0, 1.02),
    )
    axis.set_xticks(task_ids)
    axis.legend(frameon=False, loc="best")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    atomic_figure(output, figure)


def plot_state_metrics(rows: list[dict[str, Any]], output: Path) -> None:
    names = ("projection", "distance_reduction", "cosine_alignment")
    figure, axes = plt.subplots(1, len(names), figsize=(11.0, 3.8), sharex=True)
    task_ids = np.asarray([int(row["task_id"]) for row in rows])
    state_offsets = np.tile(np.linspace(-0.11, 0.11, 3), math.ceil(len(rows) / 3))[: len(rows)]
    for axis, name in zip(axes, names):
        values = np.asarray([float(row[name]) for row in rows])
        axis.scatter(task_ids + state_offsets, values, s=22, alpha=0.72, color="#7a5195")
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("task ID")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("state estimate")
    figure.tight_layout()
    atomic_figure(output, figure)


def plot_bootstrap_intervals(
    summaries: list[dict[str, Any]], output: Path
) -> None:
    selected = [
        item
        for item in summaries
        if item["metric"]
        in {
            "retrieval_accuracy_all",
            "retrieval_accuracy_off_diagonal",
            "projection",
            "distance_reduction",
            "cosine_alignment",
        }
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    y = np.arange(len(selected))
    estimates = np.asarray([item["estimate"] for item in selected])
    low = np.asarray([item["bootstrap_95_ci_low"] for item in selected])
    high = np.asarray([item["bootstrap_95_ci_high"] for item in selected])
    axis.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - low, high - estimates]),
        fmt="o",
        color="#1b9e77",
        capsize=3,
    )
    axis.axvline(0.0, color="#888888", linestyle="--", linewidth=1)
    axis.set_yticks(y, [item["metric"].replace("_", " ") for item in selected])
    axis.invert_yaxis()
    axis.set_xlabel("state-level mean and percentile bootstrap interval")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    atomic_figure(output, figure)


def plot_permutation_null(
    null: np.ndarray, summaries: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for column, (axis, summary) in enumerate(zip(axes, summaries)):
        axis.hist(null[:, column], bins=35, color="#80b1d3", edgecolor="white")
        axis.axvline(summary["observed"], color="#d95f02", linewidth=2, label="observed")
        axis.set_title(summary["metric"].replace("_", " "))
        axis.set_xlabel("permuted cohort mean")
        axis.set_ylabel("draw count")
        axis.legend(frameon=False)
    figure.tight_layout()
    atomic_figure(output, figure)


def _load_observation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        agent = np.asarray(archive["agentview"])
        wrist = np.asarray(archive["wrist"])
    for name, value in (("agentview", agent), ("wrist", wrist)):
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise RuntimeError(f"{path} {name} is not uint8 HWC RGB: {value.shape} {value.dtype}")
    return agent, wrist


def build_contact_sheet(media: list[dict[str, Any]], output: Path) -> None:
    font = ImageFont.load_default()
    tile_width, image_height, label_height = 260, 128, 26
    columns = 5
    rows = math.ceil(len(media) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(media):
        record = item["record"]
        agent, wrist = _load_observation(item["observation_path"])
        left = Image.fromarray(agent).resize((128, 128), Image.Resampling.BILINEAR)
        right = Image.fromarray(wrist).resize((128, 128), Image.Resampling.BILINEAR)
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + label_height)
        sheet.paste(left, (x, y))
        sheet.paste(right, (x + 130, y))
        draw.text(
            (x + 3, y + image_height + 4),
            f"{record['state_id']}  task={record['task_id']}",
            fill="black",
            font=font,
        )
    atomic_image(output, sheet)


def _heat_color(value: float) -> tuple[int, int, int]:
    if not math.isfinite(value):
        return (190, 190, 190)
    clipped = max(-1.0, min(1.0, value))
    midpoint = np.asarray((247, 247, 247), dtype=np.float64)
    endpoint = np.asarray(
        (33, 102, 172) if clipped < 0 else (178, 24, 43), dtype=np.float64
    )
    color = midpoint + abs(clipped) * (endpoint - midpoint)
    return tuple(int(round(channel)) for channel in color)


def build_video_frame(item: dict[str, Any]) -> np.ndarray:
    record = item["record"]
    agent, wrist = _load_observation(item["observation_path"])
    canvas = Image.new("RGB", (768, 432), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    canvas.paste(
        Image.fromarray(agent).resize((256, 256), Image.Resampling.BILINEAR),
        (16, 40),
    )
    canvas.paste(
        Image.fromarray(wrist).resize((256, 256), Image.Resampling.BILINEAR),
        (280, 40),
    )
    draw.text((16, 17), "agent view", fill="black", font=font)
    draw.text((280, 17), "wrist view", fill="black", font=font)
    draw.text((560, 17), "future-source projection", fill="black", font=font)
    matrix = np.asarray(item["projection_matrix"], dtype=np.float64)
    cell = 42
    origin_x, origin_y = 577, 65
    for row in range(4):
        draw.text((origin_x - 20, origin_y + row * cell + 15), f"r{row}", fill="black", font=font)
        for column in range(4):
            x0 = origin_x + column * cell
            y0 = origin_y + row * cell
            draw.rectangle(
                (x0, y0, x0 + cell - 2, y0 + cell - 2),
                fill=_heat_color(float(matrix[row, column])),
                outline="white",
            )
            draw.text(
                (x0 + 7, y0 + 14),
                f"{matrix[row, column]:.2f}" if math.isfinite(matrix[row, column]) else "NA",
                fill="black",
                font=font,
            )
    for column in range(4):
        draw.text((origin_x + column * cell + 13, origin_y - 18), f"s{column}", fill="black", font=font)
    draw.text((16, 315), f"{record['state_id']} | task {record['task_id']} | initial state {record.get('initial_state_index')}", fill="black", font=font)
    prompt_lines = textwrap.wrap(str(record.get("prompt", "")), width=105)[:4]
    for line_index, line in enumerate(prompt_lines):
        draw.text((16, 338 + line_index * 17), line, fill=(40, 40, 40), font=font)
    draw.text(
        (560, 255),
        "Rows: action-noise source\nColumns: future source\nFixed display range: [-1, 1]",
        fill=(40, 40, 40),
        font=font,
        spacing=5,
    )
    return np.asarray(canvas, dtype=np.uint8)


def build_all_state_video(media: list[dict[str, Any]], output: Path, fps: float) -> None:
    if fps <= 0:
        raise ValueError("video fps must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(output)
    writer = imageio.get_writer(
        temporary,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
        output_params=["-pix_fmt", "yuv420p", "-map_metadata", "-1"],
    )
    try:
        for item in media:
            writer.append_data(build_video_frame(item))
    finally:
        writer.close()
    os.replace(temporary, output)


def build_artifact_index(output_root: Path, metadata: dict[str, Any]) -> None:
    index_path = output_root / "artifact_index.json"
    files = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path == index_path or ".tmp." in path.name:
            continue
        files.append(
            {
                "path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_json(index_path, {**metadata, "artifacts": files})


def main() -> None:
    args = parse_args()
    if args.expected_states <= 0 or args.expected_tasks <= 0:
        raise ValueError("expected state/task counts must be positive")
    result_root = args.result_root.resolve()
    analysis_root = args.analysis_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = result_root / "manifest.json"
    manifest = load_json(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    records = sorted(
        [
            record
            for record in manifest["states"]
            if record["admission"] == args.admission
        ],
        key=_state_order,
    )
    if len(records) != args.expected_states:
        raise RuntimeError(
            f"expected {args.expected_states} admitted states, found {len(records)}"
        )
    task_counts = Counter(int(record["task_id"]) for record in records)
    if len(task_counts) != args.expected_tasks or len(set(task_counts.values())) != 1:
        raise RuntimeError(f"unexpected task coverage: {dict(sorted(task_counts.items()))}")
    if args.admission == "evaluation" and sorted(task_counts) != list(
        range(args.expected_tasks)
    ):
        raise RuntimeError(f"evaluation task IDs are not complete: {sorted(task_counts)}")
    if args.admission == "evaluation":
        state_indices_by_task = {
            task_id: sorted(
                int(record["initial_state_index"])
                for record in records
                if int(record["task_id"]) == task_id
            )
            for task_id in sorted(task_counts)
        }
        if any(indices != [10, 20, 30] for indices in state_indices_by_task.values()):
            raise RuntimeError(
                f"evaluation initial-state coverage changed: {state_indices_by_task}"
            )
    prompts_by_task = {
        task_id: {str(record["prompt"]) for record in records if int(record["task_id"]) == task_id}
        for task_id in sorted(task_counts)
    }
    if any(len(prompts) != 1 for prompts in prompts_by_task.values()):
        raise RuntimeError("a task maps to multiple prompts in the frozen manifest")

    analysis_summary_path = analysis_root / "summary.json"
    analysis_csv_path = analysis_root / "state_metrics.csv"
    analysis_summary = load_json(analysis_summary_path)
    rows_by_id = {row["state_id"]: row for row in read_state_metrics(analysis_csv_path)}
    rows = [rows_by_id[str(record["state_id"])] for record in records if str(record["state_id"]) in rows_by_id]

    controls, validation, media = validate_cohort(
        result_root=result_root,
        manifest_path=manifest_path,
        manifest=manifest,
        records=records,
        manifest_sha256=manifest_sha256,
    )
    nearest = validation["nearest"]
    metric_names = validate_analysis(
        summary=analysis_summary,
        rows=rows,
        records=records,
        manifest_sha256=manifest_sha256,
        nearest=nearest,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    provenance_receipt, checkpoint_content = collect_provenance(
        args=args,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        cohort_provenance=validation["provenance"],
    )
    atomic_json(
        output_root / "checkpoint_content_manifest.json", checkpoint_content
    )
    atomic_json(output_root / "environment_provenance.json", provenance_receipt)
    provenance_receipt_sha256 = sha256_file(
        output_root / "environment_provenance.json"
    )
    checkpoint_content_sha256 = sha256_file(
        output_root / "checkpoint_content_manifest.json"
    )

    bootstrap_seed = args.seed
    permutation_seed = args.seed + 1
    bootstrap_draws, bootstrap_summary = bootstrap_outputs(
        rows, metric_names, args.bootstrap_repetitions, bootstrap_seed
    )
    permutation_null, permutation_summary = permutation_outputs(
        nearest, args.permutation_repetitions, permutation_seed
    )

    state_fields = ["state_id"] + sorted(
        {key for row in rows for key in row if key != "state_id"}
    )
    atomic_csv(output_root / "state_metrics.csv", rows, state_fields)
    per_task = task_rows(rows, metric_names)
    atomic_csv(
        output_root / "task_metrics.csv",
        per_task,
        ["task_id", "n_states", *metric_names],
    )
    atomic_csv(
        output_root / "bootstrap_summary.csv",
        bootstrap_summary,
        list(bootstrap_summary[0]),
    )
    atomic_csv(
        output_root / "permutation_summary.csv",
        permutation_summary,
        list(permutation_summary[0]),
    )
    atomic_npy(output_root / "bootstrap_draws.npy", bootstrap_draws)
    atomic_npy(output_root / "permutation_null.npy", permutation_null)
    atomic_json(
        output_root / "resampling_manifest.json",
        {
            "bootstrap": {
                "columns": metric_names,
                "draws_file": "bootstrap_draws.npy",
                "independent_unit": "frozen simulator state",
                "repetitions": args.bootstrap_repetitions,
                "seed": bootstrap_seed,
            },
            "permutation": {
                "columns": [item["metric"] for item in permutation_summary],
                "draws_file": "permutation_null.npy",
                "primary_metric": "retrieval_accuracy_off_diagonal",
                "repetitions": args.permutation_repetitions,
                "seed": permutation_seed,
                "unit": "one shared permutation of all four future-source labels over all four native-action labels per state; the mapping is shared across recipient rows and off-diagonal scoring retains the original 12 cells",
                "off_diagonal_null_expected_accuracy": 0.25,
            },
        },
    )
    atomic_json(
        output_root / "control_audit.json",
        {
            "all_exact_controls_pass": True,
            "state_count": len(records),
            "task_counts": dict(sorted(task_counts.items())),
            "states": controls,
            "provenance": validation["provenance"],
        },
    )

    plot_root = output_root / "plots"
    plot_retrieval_by_task(rows, plot_root / "retrieval_by_task.png")
    plot_state_metrics(rows, plot_root / "state_directional_metrics.png")
    plot_bootstrap_intervals(
        bootstrap_summary, plot_root / "bootstrap_intervals.png"
    )
    plot_permutation_null(
        permutation_null, permutation_summary, plot_root / "permutation_null.png"
    )

    media_root = output_root / "media"
    build_contact_sheet(media, media_root / "all_states_contact_sheet.png")
    build_all_state_video(media, media_root / "all_states_overview.mp4", args.video_fps)
    atomic_json(
        media_root / "selection_policy.json",
        {
            "policy": "include every admitted state exactly once in frozen task/state order",
            "selected_by_outcome": False,
            "state_ids": [str(record["state_id"]) for record in records],
            "contact_sheet_content": "two frozen input camera views for every state",
            "video_content": "two frozen input views plus the 4x4 future-source projection matrix for every state",
            "projection_display_range": [-1.0, 1.0],
            "video_fps": args.video_fps,
        },
    )

    summary = {
        "status": "complete",
        "admission": args.admission,
        "state_count": len(records),
        "task_counts": dict(sorted(task_counts.items())),
        "all_exact_controls_pass": True,
        "analysis_summary_path": str(analysis_summary_path),
        "analysis_summary_sha256": sha256_file(analysis_summary_path),
        "analysis_state_metrics_sha256": sha256_file(analysis_csv_path),
        "manifest_sha256": manifest_sha256,
        "runner_sha256": validation["provenance"]["runner_sha256"][0],
        "upstream_commit": validation["provenance"]["upstream_commit"][0],
        "checkpoint_revision": validation["provenance"]["checkpoint_revision"][0],
        "environment_provenance_path": "environment_provenance.json",
        "environment_provenance_sha256": provenance_receipt_sha256,
        "checkpoint_content_manifest_path": "checkpoint_content_manifest.json",
        "checkpoint_content_manifest_sha256": checkpoint_content_sha256,
        "metric_names": metric_names,
        "primary_inferential_metric": "retrieval_accuracy_off_diagonal",
        "retrieval_accuracy_all_includes_exact_self_controls": True,
        "bootstrap": bootstrap_summary,
        "permutation": permutation_summary,
        "media_selection_uses_outcomes": False,
    }
    atomic_json(output_root / "summary.json", summary)
    atomic_text(
        output_root / "README.md",
        "\n".join(
            [
                "# LingBot post-run artifacts",
                "",
                f"Cohort: {len(records)} {args.admission} states across {len(task_counts)} tasks.",
                "",
                "All summaries use frozen simulator state as the independent unit. Bootstrap and permutation draws are retained as `.npy` files with column order and seeds in `resampling_manifest.json`.",
                "",
                "Off-diagonal retrieval is the claim-facing retrieval statistic. The all-cell statistic is retained descriptively but includes four exact self-replay controls per state.",
                "",
                "The contact sheet includes every frozen state. The overview video uses the same complete, predetermined ordering; no example was chosen or omitted using an outcome.",
                "",
                "This package reports descriptive estimates and resampling results without manuscript claims.",
                "",
            ]
        ),
    )
    build_artifact_index(
        output_root,
        {
            "status": "complete",
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": sha256_file(Path(__file__)),
            "source_result_root": str(result_root),
            "source_analysis_root": str(analysis_root),
            "state_count": len(records),
            "media_selection_uses_outcomes": False,
            "environment_provenance": {
                "path": "environment_provenance.json",
                "sha256": provenance_receipt_sha256,
            },
            "checkpoint_content_manifest": {
                "path": "checkpoint_content_manifest.json",
                "sha256": checkpoint_content_sha256,
            },
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
