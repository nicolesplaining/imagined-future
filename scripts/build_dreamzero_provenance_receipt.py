#!/usr/bin/env python3
"""Build a machine-readable DreamZero runtime and intervention receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_REPO_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
EXPECTED_PATCH_SHA256 = "7c601e25d335a348056ef674065e445279482e91098f28493b3f218882b3d25a"
EXPECTED_CHECKPOINT_REVISION = "96ad344138c66e82536422432ad742f015784942"
EXPECTED_MANIFEST_SHA256 = "d1ffc3111a10bed9ac8fdd17c631dc3a5d8eb3128ac4fa250d9398bcede12cfc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-hf-revision", required=True)
    parser.add_argument("--checkpoint-content-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-verification-receipt", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--related-script", type=Path, action="append", default=[])
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gaussian-run-root", type=Path, required=True)
    parser.add_argument("--dose-run-root", type=Path, required=True)
    parser.add_argument("--control-analysis-root", type=Path, required=True)
    parser.add_argument("--off-record-debug-root", type=Path, required=True)
    parser.add_argument("--core-client-trace-root", type=Path, required=True)
    parser.add_argument("--core-server-trace-root", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--server-log-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-patch-sha256", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(arguments, cwd=cwd, text=True).strip()


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def snapshot_file(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    payload = source.read_bytes()
    atomic_write(destination, payload)
    record = file_record(destination)
    if stat.S_IMODE(destination.stat().st_mode) != 0o444:
        raise RuntimeError(f"snapshot is not immutable: {destination}")
    record["source_path"] = str(source.resolve())
    record["semantics"] = "immutable point-in-time snapshot taken after all experiment calls"
    return record


def verify_checkpoint_content(
    checkpoint_root: Path,
    manifest_path: Path,
    audit_path: Path,
    expected_revision: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = checkpoint_root.resolve(strict=True)
    manifest_record = file_record(manifest_path)
    audit_record = file_record(audit_path)
    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = content.get("files")
    if (
        content.get("schema_version") != "checkpoint-content-manifest-v1"
        or Path(str(content.get("checkpoint_root", ""))).resolve() != root
        or not isinstance(rows, list)
        or int(content.get("file_count", -1)) != len(rows)
    ):
        raise ValueError("checkpoint content manifest schema/root/count differs")
    expected: dict[str, dict[str, Any]] = {}
    total = 0
    for row in rows:
        relative = str(row["relative_path"])
        if relative in expected or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"invalid/duplicate checkpoint manifest path: {relative}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"checkpoint manifest path is not a regular file: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(row["size_bytes"]) or digest != row["sha256"]:
            raise ValueError(f"checkpoint content differs: {relative}")
        expected[relative] = row
        total += size
    observed = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if observed != set(expected):
        raise ValueError(
            f"checkpoint file set differs: missing={sorted(set(expected)-observed)}, "
            f"extra={sorted(observed-set(expected))}"
        )
    if total != int(content.get("total_size_bytes", -1)):
        raise ValueError("checkpoint manifest total byte count differs")
    if (
        audit.get("schema_version") != "checkpoint-content-manifest-audit-v1"
        or Path(str(audit.get("checkpoint_root", ""))).resolve() != root
        or audit.get("content_manifest_sha256") != manifest_record["sha256"]
        or int(audit.get("file_count", -1)) != len(rows)
        or int(audit.get("total_size_bytes", -1)) != total
    ):
        raise ValueError("checkpoint verification receipt does not bind the content manifest")
    metadata = sorted((root / ".cache" / "huggingface" / "download").rglob("*.metadata"))
    if not metadata:
        raise ValueError("checkpoint has no Hugging Face snapshot metadata")
    revisions = {
        path.read_text(encoding="utf-8").splitlines()[0].strip()
        for path in metadata
    }
    if revisions != {expected_revision}:
        raise ValueError(f"checkpoint metadata revision differs: {sorted(revisions)}")
    return (
        manifest_record,
        audit_record,
        {
            "file_count": len(rows),
            "total_size_bytes": total,
            "verified_against_checkpoint_root": True,
            "exact_file_set_verified": True,
            "all_file_sizes_and_sha256_verified": True,
            "hugging_face_metadata_file_count": len(metadata),
            "hugging_face_revision_verified_from_local_metadata": expected_revision,
        },
    )


def source_location(repo: Path, relative: str, needle: str, function: str) -> dict[str, Any]:
    path = repo / relative
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if needle in line]
    if len(matches) != 1:
        raise ValueError(f"expected one {needle!r} in {path}, got {matches}")
    return {
        "file": relative,
        "function": function,
        "line": matches[0],
        "anchor": needle,
        "file_sha256": sha256_file(path),
    }


def process_record(pid: int) -> dict[str, Any]:
    root = Path(f"/proc/{pid}")
    if not root.is_dir():
        raise ProcessLookupError(pid)
    command_line = (root / "cmdline").read_bytes().rstrip(b"\0").replace(b"\0", b" ").decode()
    environment = {}
    allowed = {
        "CUDA_VISIBLE_DEVICES",
        "DYNAMIC_CACHE_SCHEDULE",
        "NUM_DIT_STEPS",
        "PYTHONPATH",
    }
    for item in (root / "environ").read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        name = key.decode(errors="replace")
        if name in allowed:
            environment[name] = value.decode(errors="replace")
    return {
        "pid": pid,
        "command": command_line,
        "cwd": os.readlink(root / "cwd"),
        "environment_allowlist": dict(sorted(environment.items())),
    }


def verify_run_inventory(
    root: Path,
    expected_schema: str,
    expected_runner_sha: str,
    expected_state_ids: list[str],
    expected_manifest_sha: str | None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    inventory_path = root / "run_inventory.json"
    inventory_record = file_record(inventory_path)
    sidecar = inventory_path.with_name(inventory_path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip().split()[0] != inventory_record["sha256"]:
        raise ValueError(f"run inventory sidecar differs: {inventory_path}")
    if stat.S_IMODE(inventory_path.stat().st_mode) != 0o444:
        raise ValueError(f"run inventory is not immutable: {inventory_path}")
    inventory = json.loads(inventory_path.read_text())
    rows = inventory.get("results")
    if (
        inventory.get("schema") != expected_schema
        or inventory.get("status") != "complete"
        or int(inventory.get("state_count", -1)) != 30
        or inventory.get("state_ids") != expected_state_ids
        or inventory.get("runner_sha256") != expected_runner_sha
        or not isinstance(rows, list)
        or len(rows) != 30
    ):
        raise ValueError(f"run inventory identity/cohort differs: {inventory_path}")
    if expected_manifest_sha is not None and inventory.get("manifest_file_sha256") != expected_manifest_sha:
        raise ValueError(f"run inventory manifest differs: {inventory_path}")
    result_rows = []
    result_sha256_by_state: dict[str, str] = {}
    observed_ids = set()
    for row in rows:
        state_id = str(row.get("state_id", ""))
        if "path" in row:
            path = Path(str(row["path"]))
            if not state_id:
                state_id = path.parent.name
        else:
            path = root / str(row.get("relative_path", ""))
        if (
            state_id in observed_ids
            or path.resolve() != (root / "states" / state_id / "result.json").resolve()
        ):
            raise ValueError(f"run inventory result path differs: {path}")
        observed_ids.add(state_id)
        record = file_record(path)
        if record["sha256"] != row["sha256"]:
            raise ValueError(f"run inventory result SHA differs: {path}")
        result_sidecar = path.with_name(path.name + ".sha256")
        if result_sidecar.read_text().strip().split()[0] != record["sha256"] or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise ValueError(f"run result is not frozen: {path}")
        result_rows.append(record)
        result_sha256_by_state[state_id] = record["sha256"]
    if observed_ids != set(expected_state_ids):
        raise ValueError(f"run result membership differs: {root}")
    return {
        "root": str(root),
        "inventory": inventory_record,
        "runner_sha256": expected_runner_sha,
        "state_count": 30,
        "result_set_digest": hashlib.sha256(
            json.dumps(result_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "result_sha256_by_state": result_sha256_by_state,
    }


def verify_control_analysis(root: Path, expected_manifest_sha: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    inventory_path = root / "artifact_inventory.json"
    inventory_record = file_record(inventory_path)
    sidecar = inventory_path.with_name(inventory_path.name + ".sha256")
    if sidecar.read_text().strip().split()[0] != inventory_record["sha256"] or stat.S_IMODE(inventory_path.stat().st_mode) != 0o444:
        raise ValueError("control-analysis inventory is not frozen")
    inventory = json.loads(inventory_path.read_text())
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        inventory.get("schema") != "dreamzero-control-analysis-artifact-inventory-v1"
        or inventory.get("manifest_file_sha256") != expected_manifest_sha
        or summary.get("audit_status") != "passed"
        or int(summary.get("state_count", -1)) != 30
        or summary.get("manifest_file_sha256") != expected_manifest_sha
        or summary.get("analyzer_sha256") != inventory.get("analyzer_sha256")
    ):
        raise ValueError("control analysis identity/audit differs")
    for row in inventory.get("artifacts", []):
        path = root / row["relative_path"]
        record = file_record(path)
        if record["sha256"] != row["sha256"] or record["size_bytes"] != int(row["size_bytes"]):
            raise ValueError(f"control-analysis artifact differs: {path}")
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise ValueError(f"control-analysis artifact is not immutable: {path}")
    sources = inventory.get("sources")
    if not isinstance(sources, list) or len(sources) != 30:
        raise ValueError("control-analysis source inventory is incomplete")
    source_sha256_by_state: dict[str, dict[str, str]] = {}
    for row in sources:
        state_id = str(row.get("state_id", ""))
        if state_id in source_sha256_by_state:
            raise ValueError(f"duplicate control-analysis source state: {state_id}")
        source_sha256_by_state[state_id] = {
            "core": str(row.get("core_result_sha256", "")),
            "gaussian": str(row.get("gaussian", {}).get("result_sha256", "")),
            "dose": str(row.get("dose", {}).get("result_sha256", "")),
        }
    return {
        "root": str(root),
        "artifact_inventory": inventory_record,
        "summary": file_record(summary_path),
        "analyzer_sha256": summary["analyzer_sha256"],
        "audit_status": "passed",
        "source_sha256_by_state": source_sha256_by_state,
        "patched_server_mode_off_record_gate": summary.get(
            "patched_server_mode_off_record_gate"
        ),
    }


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve(strict=True)
    patch = file_record(args.patch)
    runner = file_record(args.runner)
    commit = command(["git", "rev-parse", "HEAD"], repo)
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=repo)
    diff_sha = hashlib.sha256(diff).hexdigest()
    untracked = command(
        ["git", "ls-files", "--others", "--exclude-standard"], repo
    ).splitlines()
    if untracked:
        raise ValueError(f"untracked repository files are not allowed in runtime provenance: {untracked}")
    if args.expected_repo_commit != EXPECTED_REPO_COMMIT or commit != EXPECTED_REPO_COMMIT:
        raise ValueError(f"repo commit mismatch: {commit}")
    if args.expected_patch_sha256 != EXPECTED_PATCH_SHA256 or patch["sha256"] != EXPECTED_PATCH_SHA256:
        raise ValueError(f"patch SHA mismatch: {patch['sha256']}")
    if diff_sha != patch["sha256"]:
        raise ValueError(f"applied git diff does not match patch: {diff_sha} != {patch['sha256']}")

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("receipt must run in the DreamZero environment") from error
    gpu_query = command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if args.checkpoint_hf_revision != EXPECTED_CHECKPOINT_REVISION:
        raise ValueError("checkpoint revision is not the canonical frozen revision")
    manifest_record, checkpoint_audit_record, manifest_summary = verify_checkpoint_content(
        args.checkpoint_root,
        args.checkpoint_content_manifest,
        args.checkpoint_verification_receipt,
        args.checkpoint_hf_revision,
    )
    server_log_prefix = file_record(args.server_log)
    experiment_manifest = file_record(args.manifest)
    if experiment_manifest["sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("experiment manifest is not the canonical frozen manifest")
    manifest_json = json.loads(args.manifest.read_text())
    state_ids = [str(row["state_id"]) for row in manifest_json.get("states", [])]
    if len(state_ids) != 30 or len(set(state_ids)) != 30:
        raise ValueError("experiment manifest does not contain 30 unique states")
    related_scripts = [file_record(path) for path in args.related_script]
    script_by_name = {Path(row["path"]).name: row for row in related_scripts}
    required_scripts = {
        "run_dreamzero_controls.py",
        "run_dreamzero_dose_response.py",
        "summarize_dreamzero_controls.py",
    }
    if not required_scripts.issubset(script_by_name):
        raise ValueError(f"related script inventory is incomplete: {required_scripts-set(script_by_name)}")
    core_run = verify_run_inventory(
        args.run_root,
        "dreamzero-future-transplant-run-inventory-v1",
        runner["sha256"],
        state_ids,
        None,
    )
    gaussian_run = verify_run_inventory(
        args.gaussian_run_root,
        "dreamzero-gaussian-control-run-inventory-v1",
        script_by_name["run_dreamzero_controls.py"]["sha256"],
        state_ids,
        EXPECTED_MANIFEST_SHA256,
    )
    dose_run = verify_run_inventory(
        args.dose_run_root,
        "dreamzero-future-latent-dose-run-inventory-v1",
        script_by_name["run_dreamzero_dose_response.py"]["sha256"],
        state_ids,
        EXPECTED_MANIFEST_SHA256,
    )
    control_analysis = verify_control_analysis(
        args.control_analysis_root, EXPECTED_MANIFEST_SHA256
    )
    expected_sources = {
        state_id: {
            "core": core_run["result_sha256_by_state"][state_id],
            "gaussian": gaussian_run["result_sha256_by_state"][state_id],
            "dose": dose_run["result_sha256_by_state"][state_id],
        }
        for state_id in state_ids
    }
    if control_analysis["source_sha256_by_state"] != expected_sources:
        raise ValueError("control analysis is not bound to the verified run result sets")
    debug_result = file_record(args.off_record_debug_root / "result.json")
    debug_sidecar = args.off_record_debug_root / "result.json.sha256"
    if (
        not debug_sidecar.is_file()
        or debug_sidecar.read_text().strip().split()[0] != debug_result["sha256"]
        or stat.S_IMODE((args.off_record_debug_root / "result.json").stat().st_mode) != 0o444
    ):
        raise ValueError("mode-off/record debug result is not immutable")
    debug_json = json.loads((args.off_record_debug_root / "result.json").read_text())
    if (
        debug_json.get("mode_off_record_bit_exact") is not True
        or float(debug_json.get("maximum_absolute_error", math.inf)) != 0.0
        or debug_json.get("runner_sha256")
            != script_by_name["run_dreamzero_controls.py"]["sha256"]
        or control_analysis["patched_server_mode_off_record_gate"].get("result_sha256")
            != debug_result["sha256"]
    ):
        raise ValueError("mode-off/record debug control is not the canonical passing gate")
    server_log_snapshot = snapshot_file(args.server_log, args.server_log_snapshot)
    server = process_record(args.server_pid)
    command_tokens = shlex.split(server["command"])
    expected_tail = [
        "socket_test_optimized_AR.py",
        "--port",
        "5000",
        "--model-path",
        str(args.checkpoint_root.resolve()),
    ]
    if (
        Path(server["cwd"]).resolve() != repo
        or command_tokens[-len(expected_tail):] != expected_tail
        or "torchrun" not in Path(command_tokens[1]).name
        or "--standalone" not in command_tokens
        or "--nproc_per_node=2" not in command_tokens
        or server["environment_allowlist"].get("CUDA_VISIBLE_DEVICES") != "0,1"
        or server["environment_allowlist"].get("DYNAMIC_CACHE_SCHEDULE") != "False"
        or server["environment_allowlist"].get("NUM_DIT_STEPS") != "16"
    ):
        raise ValueError(f"server launch command/cwd/environment differs: {server}")
    receipt = {
        "schema": "dreamzero-runtime-provenance-receipt-v1",
        "model": "DreamZero-DROID",
        "official_repository": {
            "root": str(repo),
            "origin": command(["git", "remote", "get-url", "origin"], repo),
            "commit": commit,
            "status_porcelain": command(["git", "status", "--porcelain=v1"], repo).splitlines(),
            "untracked_files": untracked,
            "diff_scope": "git diff HEAD --binary (staged and unstaged tracked changes)",
            "applied_diff_sha256": diff_sha,
            "intervention_patch": patch,
        },
        "checkpoint": {
            "root": str(args.checkpoint_root.resolve(strict=True)),
            "hugging_face_revision": EXPECTED_CHECKPOINT_REVISION,
            "content_manifest": manifest_record,
            "content_manifest_verification_receipt": checkpoint_audit_record,
            "content_manifest_summary": manifest_summary,
        },
        "runtime": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "gpus": gpu_query,
            "server": server,
            "server_log_prefix_at_receipt": server_log_prefix,
            "server_log_post_call_snapshot": server_log_snapshot,
        },
        "experiment": {
            "runner": runner,
            "related_scripts": related_scripts,
            "branch_seeds": [211, 223, 227, 229],
            "solver_steps": 16,
            "future_replay_interval": [0, 16],
            "manifest": experiment_manifest,
            "download_receipt": file_record(args.download_receipt),
            "run_root": str(args.run_root.resolve()),
            "verified_runs": {
                "core": core_run,
                "gaussian": gaussian_run,
                "dose": dose_run,
                "control_analysis": control_analysis,
                "mode_off_record_debug_result": debug_result,
            },
            "trace_root_mapping": {
                "client_root": str(args.core_client_trace_root.resolve()),
                "server_root": str(args.core_server_trace_root),
                "relative_path_contract": "state_id/native_seed_<seed>.pt is identical across client and server roots",
                "verification": "client files are SHA-verified; server responses bind requested path and content hashes",
            },
            "action_noise_source": "recipient native trace",
            "future_source": "source native video latent trajectory",
            "action_coordinates_written": False,
            "server_log_snapshot_taken_after_verified_complete_run_inventories": True,
        },
        "intervention_locations": [
            source_location(
                repo,
                "groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py",
                "request_intervention = self._dreamzero_intervention",
                "WANPolicyHead.lazy_joint_video_action",
            ),
            source_location(
                repo,
                "groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py",
                "noisy_input = donor_video_latent.to(",
                "WANPolicyHead.lazy_joint_video_action",
            ),
            source_location(
                repo,
                "groot/vla/model/n1_5/sim_policy.py",
                'raw_intervention = batch.obs.pop("dreamzero_intervention", None)',
                "GrootSimPolicy.lazy_joint_forward_causal",
            ),
            source_location(
                repo,
                "socket_test_optimized_AR.py",
                'converted["dreamzero_intervention"] = dict(intervention)',
                "ARDroidRoboarenaPolicy._convert_observation",
            ),
        ],
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    atomic_write(args.output, payload)
    atomic_write(
        args.output.with_name(args.output.name + ".sha256"),
        f"{hashlib.sha256(payload).hexdigest()}  {args.output.name}\n".encode(),
    )
    if stat.S_IMODE(args.output.stat().st_mode) != 0o444:
        raise RuntimeError("receipt output is not immutable")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
