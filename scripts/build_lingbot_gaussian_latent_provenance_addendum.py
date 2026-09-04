#!/usr/bin/env python3
"""Build an immutable, offline provenance archive of LingBot Gaussian futures.

This script does not load a model or generate actions.  It extracts and executes
the exact ``gaussian_future`` function from a pinned experiment runner, applies
it to each archived native future, verifies its invariants, and saves the
resulting tensors with complete source and output hashes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import socket
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


SCHEMA_VERSION = "lingbot-gaussian-latent-provenance-v1"
BRANCH_COUNT = 4
EXPECTED_STATE_COUNT = 30
EXPECTED_TENSOR_COUNT = EXPECTED_STATE_COUNT * BRANCH_COUNT

# The runner performs float32 normalization followed by a cast back to the
# native bfloat16 dtype.  These thresholds only accommodate that final cast.
MEAN_ABS_TOLERANCE = 1.0e-4
STD_ABS_TOLERANCE = 1.0e-4
NORM_ABS_TOLERANCE = 1.0e-2
NORM_REL_TOLERANCE = 1.0e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: torch.Tensor) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def extract_pinned_functions(
    runner_path: Path,
) -> tuple[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor], Callable, dict]:
    """Compile only gaussian_future and tensor_hash from the pinned runner."""
    source = runner_path.read_text()
    parsed = ast.parse(source, filename=str(runner_path))
    requested_names = ("tensor_hash", "gaussian_future")
    nodes = {
        node.name: node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in requested_names
    }
    if set(nodes) != set(requested_names):
        raise RuntimeError(
            f"pinned runner is missing functions: {set(requested_names) - set(nodes)}"
        )
    ordered_nodes = [nodes[name] for name in requested_names]
    module = ast.Module(body=ordered_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"torch": torch, "hashlib": hashlib}
    exec(compile(module, str(runner_path), "exec"), namespace)

    lines = source.splitlines(keepends=True)
    function_metadata = {}
    for name in requested_names:
        node = nodes[name]
        exact_source = "".join(lines[node.lineno - 1 : node.end_lineno])
        function_metadata[name] = {
            "line_start": node.lineno,
            "line_end": node.end_lineno,
            "source_sha256": sha256_bytes(exact_source.encode("utf-8")),
        }
    return namespace["gaussian_future"], namespace["tensor_hash"], function_metadata


def safe_torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def mode_string(path: Path) -> str:
    return format(stat.S_IMODE(path.stat().st_mode), "04o")


def build_artifact_index(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact_index.json" and path.parent == root:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    aggregate_payload = b"".join(
        f"{record['path']}\0{record['sha256']}\n".encode("utf-8")
        for record in files
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "index_excludes_itself": True,
        "file_count_excluding_index": len(files),
        "total_bytes_excluding_index": sum(record["bytes"] for record in files),
        "tree_aggregate_sha256": sha256_bytes(aggregate_payload),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-artifact-index", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runner-package-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-index-sha256", required=True)
    parser.add_argument("--expected-runner-package-index-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    source_index_path = args.source_artifact_index.resolve()
    runner_path = args.runner.resolve()
    runner_package_index_path = args.runner_package_index.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root.with_name(f".{output_root.name}.building.{os.getpid()}")
    if temporary_root.exists():
        raise FileExistsError(f"temporary path already exists: {temporary_root}")
    temporary_root.mkdir(mode=0o755)

    runner_sha256 = sha256_file(runner_path)
    if runner_sha256 != args.expected_runner_sha256:
        raise RuntimeError(
            f"runner hash mismatch: {runner_sha256} != {args.expected_runner_sha256}"
        )
    source_index_sha256 = sha256_file(source_index_path)
    if source_index_sha256 != args.expected_source_index_sha256:
        raise RuntimeError(
            "source artifact-index hash mismatch: "
            f"{source_index_sha256} != {args.expected_source_index_sha256}"
        )
    runner_package_index_sha256 = sha256_file(runner_package_index_path)
    if runner_package_index_sha256 != args.expected_runner_package_index_sha256:
        raise RuntimeError(
            "runner-package artifact-index hash mismatch: "
            f"{runner_package_index_sha256} != "
            f"{args.expected_runner_package_index_sha256}"
        )
    runner_package_index = json.loads(runner_package_index_path.read_text())
    runner_package_root = runner_package_index_path.parent
    runner_relative_path = runner_path.relative_to(runner_package_root).as_posix()
    runner_index_record = next(
        (
            record
            for record in runner_package_index["files"]
            if record["path"] == runner_relative_path
        ),
        None,
    )
    if runner_index_record is None or runner_index_record["sha256"] != runner_sha256:
        raise RuntimeError("runner is not identically bound by its frozen package index")
    manifest_path = source_root / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != args.expected_manifest_sha256:
        raise RuntimeError(
            f"manifest hash mismatch: {manifest_sha256} != "
            f"{args.expected_manifest_sha256}"
        )

    source_index = json.loads(source_index_path.read_text())
    source_index_root = source_index_path.parent
    indexed_hashes = {
        record["path"]: record["sha256"] for record in source_index["files"]
    }
    gaussian_future, pinned_tensor_hash, function_metadata = extract_pinned_functions(
        runner_path
    )
    manifest = json.loads(manifest_path.read_text())
    records = [
        record for record in manifest["states"] if record.get("admission") == "evaluation"
    ]
    if len(records) != EXPECTED_STATE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_STATE_COUNT} evaluation states, found {len(records)}"
        )
    state_ids = [record["state_id"] for record in records]
    if len(set(state_ids)) != len(state_ids):
        raise RuntimeError("manifest contains duplicate evaluation state IDs")

    builder_copy_path = temporary_root / "builder_script.py"
    shutil.copy2(Path(__file__).resolve(), builder_copy_path)
    tensors_root = temporary_root / "tensors"
    tensors_root.mkdir()

    tensor_records: list[dict[str, Any]] = []
    source_action_records: list[dict[str, Any]] = []
    source_input_records: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []

    for state_number, manifest_record in enumerate(records, start=1):
        state_id = manifest_record["state_id"]
        state_root = source_root / state_id
        if not state_root.is_dir():
            raise FileNotFoundError(f"missing state directory: {state_root}")
        result_path = state_root / "result.json"
        actions_path = state_root / "actions.npz"
        frozen_path = state_root / "frozen_inputs.pt"

        source_paths = [result_path, actions_path, frozen_path] + [
            state_root / f"future_b{branch}.pt" for branch in range(BRANCH_COUNT)
        ]
        observed_source_hashes: dict[str, str] = {}
        for source_path in source_paths:
            if not source_path.is_file():
                raise FileNotFoundError(f"missing source artifact: {source_path}")
            indexed_relative_path = source_path.relative_to(source_index_root).as_posix()
            observed_hash = sha256_file(source_path)
            expected_hash = indexed_hashes.get(indexed_relative_path)
            if expected_hash is None:
                raise RuntimeError(
                    f"source artifact absent from frozen index: {indexed_relative_path}"
                )
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"source artifact hash mismatch for {indexed_relative_path}: "
                    f"{observed_hash} != {expected_hash}"
                )
            observed_source_hashes[source_path.name] = observed_hash

        result = json.loads(result_path.read_text())
        if result.get("status") != "complete":
            raise RuntimeError(f"source state is not complete: {state_id}")
        if result.get("runner_sha256") != runner_sha256:
            raise RuntimeError(f"source runner binding mismatch for {state_id}")
        if result.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError(f"source manifest binding mismatch for {state_id}")
        if result.get("actions_sha256") != observed_source_hashes["actions.npz"]:
            raise RuntimeError(f"source action binding mismatch for {state_id}")

        source_action_records.append(
            {
                "state_id": state_id,
                "path": str(actions_path),
                "bytes": actions_path.stat().st_size,
                "sha256": observed_source_hashes["actions.npz"],
                "result_recorded_sha256": result["actions_sha256"],
                "result_binding_equal": True,
            }
        )
        source_input_records.append(
            {
                "state_id": state_id,
                "frozen_inputs_path": str(frozen_path),
                "frozen_inputs_file_sha256": observed_source_hashes["frozen_inputs.pt"],
                "result_path": str(result_path),
                "result_file_sha256": observed_source_hashes["result.json"],
            }
        )

        frozen_inputs = safe_torch_load(frozen_path)
        if not isinstance(frozen_inputs, dict) or "init_latent" not in frozen_inputs:
            raise RuntimeError(f"invalid frozen_inputs.pt for {state_id}")
        init_latent = frozen_inputs["init_latent"].detach().cpu()
        if not torch.is_tensor(init_latent):
            raise TypeError(f"init_latent is not a tensor for {state_id}")

        state_output_root = tensors_root / state_id
        state_output_root.mkdir()
        for branch in range(BRANCH_COUNT):
            seed = 900_000 + branch
            reference_path = state_root / f"future_b{branch}.pt"
            future_record = safe_torch_load(reference_path)
            if not isinstance(future_record, dict) or "future" not in future_record:
                raise RuntimeError(f"invalid future archive: {reference_path}")
            reference = future_record["future"].detach().cpu()
            if reference.ndim < 3 or reference.shape[2] < 2:
                raise RuntimeError(
                    f"future tensor has no future region: {state_id} branch {branch}"
                )
            if tuple(init_latent.shape[:2]) != tuple(reference.shape[:2]):
                raise RuntimeError(
                    f"present/future shape mismatch: {state_id} branch {branch}"
                )

            gaussian = gaussian_future(reference, init_latent, seed=seed)
            repeated = gaussian_future(reference, init_latent, seed=seed)
            reference_tensor_sha256 = pinned_tensor_hash(reference)
            if reference_tensor_sha256 != result["future_hashes"][branch]:
                raise RuntimeError(
                    f"source future tensor binding mismatch: {state_id} b{branch}"
                )
            if int(future_record["video_seed"]) != int(result["video_seeds"][branch]):
                raise RuntimeError(
                    f"source future video-seed mismatch: {state_id} b{branch}"
                )
            if int(future_record["video_seed"]) != int(manifest["video_seeds"][branch]):
                raise RuntimeError(
                    f"manifest future video-seed mismatch: {state_id} b{branch}"
                )
            gaussian_tensor_sha256 = pinned_tensor_hash(gaussian)
            repeated_tensor_sha256 = pinned_tensor_hash(repeated)
            deterministic_repeat_equal = bool(torch.equal(gaussian, repeated))
            if not deterministic_repeat_equal:
                raise RuntimeError(
                    f"Gaussian regeneration is not deterministic: {state_id} b{branch}"
                )

            present_reference = init_latent[:, :, 0:1]
            present_output = gaussian[:, :, 0:1]
            present_exact = bool(torch.equal(present_reference, present_output))
            present_max_abs_error = float(
                (present_reference.float() - present_output.float()).abs().max().item()
            )
            native_region = reference[:, :, 1:].float()
            gaussian_region = gaussian[:, :, 1:].float()
            native_mean = float(native_region.mean().item())
            gaussian_mean = float(gaussian_region.mean().item())
            native_std = float(native_region.std().item())
            gaussian_std = float(gaussian_region.std().item())
            native_norm = float(torch.linalg.vector_norm(native_region).item())
            gaussian_norm = float(torch.linalg.vector_norm(gaussian_region).item())
            mean_abs_error = abs(gaussian_mean - native_mean)
            std_abs_error = abs(gaussian_std - native_std)
            norm_abs_error = abs(gaussian_norm - native_norm)
            norm_rel_error = norm_abs_error / max(native_norm, 1.0e-12)
            checks = {
                "present_bitwise_equal": present_exact,
                "mean_within_tolerance": mean_abs_error <= MEAN_ABS_TOLERANCE,
                "std_within_tolerance": std_abs_error <= STD_ABS_TOLERANCE,
                "norm_within_tolerance": (
                    norm_abs_error <= NORM_ABS_TOLERANCE
                    and norm_rel_error <= NORM_REL_TOLERANCE
                ),
                "deterministic_repeat_bitwise_equal": deterministic_repeat_equal,
            }

            output_path = state_output_root / (
                f"gaussian_b{branch}_seed{seed}.pt"
            )
            atomic_torch(output_path, gaussian)
            reloaded = safe_torch_load(output_path)
            saved_tensor_sha256 = pinned_tensor_hash(reloaded)
            saved_tensor_equal = bool(torch.equal(reloaded, gaussian))
            checks["saved_tensor_bitwise_equal"] = saved_tensor_equal
            if not all(checks.values()):
                discrepancies.append(
                    {"state_id": state_id, "branch": branch, "checks": checks}
                )

            tensor_records.append(
                {
                    "state_id": state_id,
                    "state_number": state_number,
                    "branch": branch,
                    "seed": seed,
                    "call": (
                        "gaussian_future(reference=future_bR, "
                        "present=init_latent, seed=900000+R)"
                    ),
                    "source_reference_path": str(reference_path),
                    "source_reference_file_sha256": observed_source_hashes[
                        f"future_b{branch}.pt"
                    ],
                    "source_reference_tensor_sha256": reference_tensor_sha256,
                    "source_video_seed": int(future_record["video_seed"]),
                    "output_path": output_path.relative_to(temporary_root).as_posix(),
                    "output_file_sha256": sha256_file(output_path),
                    "output_tensor_sha256": gaussian_tensor_sha256,
                    "saved_tensor_sha256": saved_tensor_sha256,
                    "repeat_tensor_sha256": repeated_tensor_sha256,
                    "shape": list(gaussian.shape),
                    "dtype": str(gaussian.dtype),
                    "future_region_element_count": native_region.numel(),
                    "statistics_computed_after_native_dtype_roundtrip_in_float32": {
                        "native_mean": native_mean,
                        "gaussian_mean": gaussian_mean,
                        "mean_abs_error": mean_abs_error,
                        "native_sample_std": native_std,
                        "gaussian_sample_std": gaussian_std,
                        "std_abs_error": std_abs_error,
                        "native_l2_norm": native_norm,
                        "gaussian_l2_norm": gaussian_norm,
                        "norm_abs_error": norm_abs_error,
                        "norm_rel_error": norm_rel_error,
                        "present_max_abs_error": present_max_abs_error,
                    },
                    "checks": checks,
                }
            )

    if len(tensor_records) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_TENSOR_COUNT} tensors, built {len(tensor_records)}"
        )
    if discrepancies:
        raise RuntimeError(
            f"{len(discrepancies)} tensor validation discrepancies: "
            f"{discrepancies[:3]}"
        )

    source_action_aggregate_sha256 = canonical_json_sha256(source_action_records)
    source_bindings = {
        "schema_version": SCHEMA_VERSION,
        "runner": {
            "path": str(runner_path),
            "sha256": runner_sha256,
            "functions": function_metadata,
        },
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "source_artifact_index": {
            "path": str(source_index_path),
            "sha256": source_index_sha256,
        },
        "runner_package_artifact_index": {
            "path": str(runner_package_index_path),
            "sha256": runner_package_index_sha256,
            "runner_relative_path": runner_relative_path,
            "runner_index_record": runner_index_record,
        },
        "core_actions": {
            "count": len(source_action_records),
            "canonical_records_sha256": source_action_aggregate_sha256,
            "records": source_action_records,
        },
        "source_inputs": source_input_records,
    }
    atomic_json(temporary_root / "source_bindings.json", source_bindings)
    atomic_json(temporary_root / "tensor_records.json", tensor_records)

    maxima = {
        "present_max_abs_error": max(
            item["statistics_computed_after_native_dtype_roundtrip_in_float32"][
                "present_max_abs_error"
            ]
            for item in tensor_records
        ),
        "mean_abs_error": max(
            item["statistics_computed_after_native_dtype_roundtrip_in_float32"][
                "mean_abs_error"
            ]
            for item in tensor_records
        ),
        "std_abs_error": max(
            item["statistics_computed_after_native_dtype_roundtrip_in_float32"][
                "std_abs_error"
            ]
            for item in tensor_records
        ),
        "norm_abs_error": max(
            item["statistics_computed_after_native_dtype_roundtrip_in_float32"][
                "norm_abs_error"
            ]
            for item in tensor_records
        ),
        "norm_rel_error": max(
            item["statistics_computed_after_native_dtype_roundtrip_in_float32"][
                "norm_rel_error"
            ]
            for item in tensor_records
        ),
    }
    unique_output_tensor_hashes = len(
        {item["output_tensor_sha256"] for item in tensor_records}
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "offline deterministic reconstruction and archival of Gaussian "
            "future latents only; no model loading, action generation, or core "
            "artifact modification"
        ),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "state_count": len(records),
        "branches_per_state": BRANCH_COUNT,
        "tensor_count": len(tensor_records),
        "unique_output_tensor_hash_count": unique_output_tensor_hashes,
        "seed_rule": "seed = 900000 + branch_index",
        "branch_seeds": {str(branch): 900_000 + branch for branch in range(4)},
        "exact_call": (
            "gaussian_future(reference=future_bR, present=init_latent, "
            "seed=900000+R)"
        ),
        "source_bindings": {
            "runner_sha256": runner_sha256,
            "manifest_sha256": manifest_sha256,
            "source_artifact_index_sha256": source_index_sha256,
            "runner_package_artifact_index_sha256": (
                runner_package_index_sha256
            ),
            "core_actions_canonical_records_sha256": (
                source_action_aggregate_sha256
            ),
            "core_action_file_count": len(source_action_records),
        },
        "validation": {
            "all_checks_pass": True,
            "discrepancy_count": len(discrepancies),
            "present_bitwise_equal_count": sum(
                item["checks"]["present_bitwise_equal"] for item in tensor_records
            ),
            "saved_tensor_bitwise_equal_count": sum(
                item["checks"]["saved_tensor_bitwise_equal"]
                for item in tensor_records
            ),
            "deterministic_repeat_bitwise_equal_count": sum(
                item["checks"]["deterministic_repeat_bitwise_equal"]
                for item in tensor_records
            ),
            "tolerances": {
                "mean_abs": MEAN_ABS_TOLERANCE,
                "std_abs": STD_ABS_TOLERANCE,
                "norm_abs": NORM_ABS_TOLERANCE,
                "norm_rel": NORM_REL_TOLERANCE,
            },
            "observed_maxima": maxima,
            "statistical_definition": (
                "global mean, unbiased sample standard deviation, and L2 norm "
                "over reference[:, :, 1:].float() and saved control[:, :, "
                "1:].float(); small nonzero errors arise only from the exact "
                "runner's final cast back to native bfloat16"
            ),
        },
        "immutability": {
            "file_mode": "0444",
            "directory_mode": "0555",
            "applied_recursively_after_index_write": True,
            "verified_by_builder_after_application": True,
        },
    }
    atomic_json(temporary_root / "receipt.json", receipt)
    readme = f"""# LingBot Gaussian latent provenance addendum

This immutable addendum archives all {EXPECTED_TENSOR_COUNT} norm-matched Gaussian
future tensors for the {EXPECTED_STATE_COUNT} frozen evaluation states and four
native future sources per state. Each tensor was produced offline with the exact
`gaussian_future` implementation extracted from the pinned core runner:

`gaussian_future(reference=future_bR, present=init_latent, seed=900000+R)`

No model was loaded, no action was generated, and no existing core artifact was
modified. `receipt.json` summarizes the audit, `source_bindings.json` binds the
pinned runner, manifest, source package, and all 30 core action files, and
`tensor_records.json` records source/output file and logical tensor hashes plus
all validation statistics for every tensor.

All files are mode 0444 and all directories are mode 0555.
"""
    atomic_text(temporary_root / "README.md", readme)

    artifact_index = build_artifact_index(temporary_root)
    atomic_json(temporary_root / "artifact_index.json", artifact_index)

    for path in sorted(
        (item for item in temporary_root.rglob("*") if item.is_file()), reverse=True
    ):
        path.chmod(0o444)
    directories = sorted(
        (item for item in temporary_root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in directories:
        path.chmod(0o555)
    temporary_root.chmod(0o555)

    wrong_file_modes = [
        str(path)
        for path in temporary_root.rglob("*")
        if path.is_file() and mode_string(path) != "0444"
    ]
    wrong_directory_modes = [
        str(path)
        for path in [temporary_root, *temporary_root.rglob("*")]
        if path.is_dir() and mode_string(path) != "0555"
    ]
    if wrong_file_modes or wrong_directory_modes:
        raise RuntimeError(
            f"immutability verification failed: files={wrong_file_modes[:3]}, "
            f"directories={wrong_directory_modes[:3]}"
        )

    os.replace(temporary_root, output_root)
    final_index_path = output_root / "artifact_index.json"
    final_receipt_path = output_root / "receipt.json"
    result = {
        "status": "complete",
        "output_root": str(output_root),
        "state_count": len(records),
        "tensor_count": len(tensor_records),
        "discrepancy_count": len(discrepancies),
        "artifact_index_sha256": sha256_file(final_index_path),
        "receipt_sha256": sha256_file(final_receipt_path),
        "tree_aggregate_sha256": artifact_index["tree_aggregate_sha256"],
        "indexed_file_count_excluding_index": artifact_index[
            "file_count_excluding_index"
        ],
        "observed_maxima": maxima,
        "all_file_modes_0444": True,
        "all_directory_modes_0555": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
