#!/usr/bin/env python3
"""Sequential launcher for the frozen 21-state Cosmos future x K/V study.

The attention K/V cache is process-global in the research server.  This
launcher therefore deliberately runs exactly one single-state worker at a
time, validates its complete control surface, and only then advances to the
next frozen state.  It never replaces an existing scientific artifact.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


HOST_NFS_ROOT = Path("/lambda/nfs/imagined-future")
CONTAINER_NFS_ROOT = Path("/research")
FROZEN_MANIFEST_ID = "cosmos3-kv-existing-bb8591311eda8a59"
FROZEN_MANIFEST_SHA256 = (
    "972bcab77e2b999c703250f9e7ab17f3d854c858d99712bb7873d0bbf589714f"
)
FROZEN_RUNNER_SHA256 = (
    "83285e99b993e7f996a40189332643338e33805fe03e00b931f0c214e32179de"
)
EXPECTED_SERVER_IMAGE = (
    "sha256:95a8933deb69f2fc73697d846f7e17066a4d4e05dc9ef85718a7ee12cfea2a8c"
)
EXPECTED_CELLS = (
    "recipient_future_recipient_kv",
    "donor_future_recipient_kv",
    "donor_future_donor_kv",
    "recipient_future_donor_kv",
)
EXPECTED_LAYERS = tuple(range(36))
EXPECTED_RESPONSE_LABELS = (
    "recipient-native",
    "recipient-repeat",
    "donor-native",
    "recipient-baseline",
    "donor-baseline",
    "recipient-kv-record",
    "recipient-kv-replay",
    "donor-future-recipient-kv",
    "donor-kv-record",
    "donor-kv-replay",
    "recipient-future-donor-kv",
)
INTERVENTION_LABELS = (
    "recipient-baseline",
    "donor-baseline",
    "recipient-kv-record",
    "recipient-kv-replay",
    "donor-future-recipient-kv",
    "donor-kv-record",
    "donor-kv-replay",
    "recipient-future-donor-kv",
)
ATTENTION_LABEL_MODES = {
    "recipient-kv-record": "record",
    "recipient-kv-replay": "patch",
    "donor-future-recipient-kv": "patch",
    "donor-kv-record": "record",
    "donor-kv-replay": "patch",
    "recipient-future-donor-kv": "patch",
}
FACTORIAL_RESPONSE_LABELS = {
    "recipient_future_recipient_kv": "recipient-kv-record",
    "donor_future_recipient_kv": "donor-future-recipient-kv",
    "donor_future_donor_kv": "donor-kv-record",
    "recipient_future_donor_kv": "recipient-future-donor-kv",
}
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class ValidationError(ValueError):
    """A frozen input or completed state violates the study contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Frozen v3 manifest (its exact ID and SHA-256 are enforced).",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=HOST_NFS_ROOT / "scripts/run_cosmos3_kv_factorial_audit.py",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--container-python", default="/workspace/.venv/bin/python")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument(
        "--sudo",
        action="store_true",
        help="Prefix Docker inspection and execution with sudo.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip valid states and retry states having only failed-attempt logs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest, inputs, runner, container image, and mount only.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, label: str | None = None) -> dict[str, Any]:
    source = label or str(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"{source}: JSON root must be an object")
    return value


def require(mapping: Mapping[str, Any], key: str, *, source: str) -> Any:
    if key not in mapping:
        raise ValidationError(f"{source}: missing required field {key}")
    return mapping[key]


def require_mapping(mapping: Mapping[str, Any], key: str, *, source: str) -> Mapping[str, Any]:
    value = require(mapping, key, source=source)
    if not isinstance(value, Mapping):
        raise ValidationError(f"{source}: {key} must be an object")
    return value


def require_sequence(mapping: Mapping[str, Any], key: str, *, source: str) -> Sequence[Any]:
    value = require(mapping, key, source=source)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"{source}: {key} must be an array")
    return value


def require_finite(mapping: Mapping[str, Any], key: str, *, source: str) -> float:
    value = require(mapping, key, source=source)
    if isinstance(value, bool):
        raise ValidationError(f"{source}: {key} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{source}: {key} must be a finite number") from error
    if not math.isfinite(result):
        raise ValidationError(f"{source}: {key} must be finite")
    return result


def require_exact_keys(mapping: Mapping[str, Any], expected: Sequence[str], *, source: str) -> None:
    actual = set(mapping)
    wanted = set(expected)
    if actual != wanted:
        raise ValidationError(
            f"{source}: key mismatch; missing={sorted(wanted - actual)}, "
            f"unexpected={sorted(actual - wanted)}"
        )


def require_sha256(value: Any, *, source: str) -> str:
    text = str(value)
    if HEX256.fullmatch(text) is None:
        raise ValidationError(f"{source}: expected a lowercase SHA-256 digest")
    return text


def _resolved_under(path: Path, root: Path, *, source: str, must_exist: bool) -> tuple[Path, Path]:
    if not path.is_absolute():
        raise ValidationError(f"{source}: path must be absolute: {path}")
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as error:
        raise ValidationError(f"{source}: cannot resolve {path}: {error}") from error
    root_resolved = root.resolve(strict=True)
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValidationError(f"{source}: {path} resolves outside {root_resolved}") from error
    return resolved, relative


def host_to_container(path: Path, *, must_exist: bool = True, source: str = "path") -> Path:
    _, relative = _resolved_under(
        path, HOST_NFS_ROOT, source=source, must_exist=must_exist
    )
    return CONTAINER_NFS_ROOT / relative


def validate_manifest(path: Path, *, verify_inputs: bool = True) -> tuple[dict[str, Any], str]:
    manifest_path, _ = _resolved_under(
        path, HOST_NFS_ROOT, source="manifest", must_exist=True
    )
    raw = manifest_path.read_bytes()
    file_sha = sha256_bytes(raw)
    if file_sha != FROZEN_MANIFEST_SHA256:
        raise ValidationError(
            "manifest file SHA-256 is not the frozen v3 digest: "
            f"expected {FROZEN_MANIFEST_SHA256}, got {file_sha}"
        )
    manifest = read_json(manifest_path, label="manifest")
    if manifest.get("manifest_id") != FROZEN_MANIFEST_ID:
        raise ValidationError(
            f"manifest_id must be {FROZEN_MANIFEST_ID!r}, got "
            f"{manifest.get('manifest_id')!r}"
        )
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    derived_id = "cosmos3-kv-existing-" + sha256_bytes(canonical_json(body))[:16]
    if derived_id != FROZEN_MANIFEST_ID:
        raise ValidationError(
            f"manifest self-hash mismatch: expected {FROZEN_MANIFEST_ID}, got {derived_id}"
        )
    exact_fields = {
        "schema_version": 1,
        "study_name": "cosmos3-existing-cohort-predicted-future-kv-factorial-v3",
        "source_state_count": 22,
        "evaluation_state_count": 21,
        "excluded_development_state": "BananaInBowlTask_seed_103",
        "future_source": "model-predicted",
    }
    for key, expected in exact_fields.items():
        if manifest.get(key) != expected:
            raise ValidationError(
                f"manifest {key} must equal {expected!r}, got {manifest.get(key)!r}"
            )
    if tuple(require_sequence(manifest, "factorial_cells", source="manifest")) != EXPECTED_CELLS:
        raise ValidationError("manifest factorial cell order/content changed")
    attention = require_mapping(manifest, "attention_scope", source="manifest")
    expected_attention = {
        "layers": list(EXPECTED_LAYERS),
        "queries": "action",
        "keys_values": "future_video",
        "token_count_preserved": True,
    }
    if dict(attention) != expected_attention:
        raise ValidationError(f"manifest attention scope changed: {dict(attention)!r}")
    analysis = require_mapping(manifest, "analysis", source="manifest")
    analysis_contract = {
        "independent_unit": "saved simulator state",
        "within_state_measurements": "four factorial cells",
        "bootstrap": "task-to-state hierarchical bootstrap",
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20_260_903,
    }
    for key, expected in analysis_contract.items():
        if analysis.get(key) != expected:
            raise ValidationError(
                f"manifest analysis.{key} must equal {expected!r}, got {analysis.get(key)!r}"
            )
    runtime = require_mapping(manifest, "runtime", source="manifest")
    if runtime.get("single_state_runner_sha256") != FROZEN_RUNNER_SHA256:
        raise ValidationError("manifest does not bind the frozen process-exit-fixed runner")
    if runtime.get("server_image") != EXPECTED_SERVER_IMAGE:
        raise ValidationError("manifest server image digest changed")
    if HEX_GIT_COMMIT.fullmatch(str(runtime.get("cosmos_commit"))) is None:
        raise ValidationError("manifest runtime.cosmos_commit must be a lowercase 40-digit commit")
    source_cohort = Path(str(require(manifest, "source_cohort", source="manifest")))
    _resolved_under(
        source_cohort,
        HOST_NFS_ROOT,
        source="manifest source_cohort",
        must_exist=verify_inputs,
    )

    states = require_sequence(manifest, "states", source="manifest")
    if len(states) != 21:
        raise ValidationError(f"manifest must contain exactly 21 states, got {len(states)}")
    state_ids: set[str] = set()
    design_keys: set[tuple[str, int]] = set()
    for index, untyped_state in enumerate(states):
        source = f"manifest states[{index}]"
        if not isinstance(untyped_state, Mapping):
            raise ValidationError(f"{source}: state must be an object")
        state = untyped_state
        state_id = str(require(state, "state_id", source=source))
        if SAFE_ID.fullmatch(state_id) is None or state_id in {".", ".."}:
            raise ValidationError(f"{source}: unsafe state_id {state_id!r}")
        if state_id in state_ids:
            raise ValidationError(f"manifest repeats state_id {state_id!r}")
        state_ids.add(state_id)
        task = str(require(state, "task", source=source))
        if not task or SAFE_ID.fullmatch(task) is None:
            raise ValidationError(f"{source}: unsafe/empty task {task!r}")
        environment_seed = int(require(state, "environment_seed", source=source))
        key = (task, environment_seed)
        if key in design_keys:
            raise ValidationError(f"manifest repeats task/environment seed {key!r}")
        design_keys.add(key)
        branch_step = int(require(state, "branch_step", source=source))
        recipient_seed = int(require(state, "recipient_seed", source=source))
        donor_seed = int(require(state, "donor_seed", source=source))
        instruction = str(require(state, "instruction", source=source))
        if branch_step <= 0 or recipient_seed < 0 or donor_seed < 0 or not instruction:
            raise ValidationError(f"{source}: invalid branch step, seed, or instruction")
        if recipient_seed == donor_seed:
            raise ValidationError(f"{source}: recipient and donor seeds must differ")
        expected_hashes = require_mapping(state, "input_sha256", source=source)
        require_exact_keys(
            expected_hashes,
            ("asset_video", "recorded_hdf5", "branch_summary"),
            source=f"{source}.input_sha256",
        )
        resolved_inputs: dict[str, Path] = {}
        for input_name in ("asset_video", "recorded_hdf5", "branch_summary"):
            supplied = Path(str(require(state, input_name, source=source)))
            resolved, _ = _resolved_under(
                supplied,
                HOST_NFS_ROOT,
                source=f"{source}.{input_name}",
                must_exist=verify_inputs,
            )
            resolved_inputs[input_name] = resolved
            expected_hash = require_sha256(
                expected_hashes[input_name], source=f"{source}.input_sha256.{input_name}"
            )
            if verify_inputs:
                if not resolved.is_file():
                    raise ValidationError(f"{source}.{input_name}: not a regular file")
                observed_hash = sha256_file(resolved)
                if observed_hash != expected_hash:
                    raise ValidationError(
                        f"{source}.{input_name}: SHA-256 mismatch; "
                        f"expected {expected_hash}, got {observed_hash}"
                    )
        if verify_inputs:
            summary = read_json(resolved_inputs["branch_summary"], label=f"{source}.branch_summary")
            summary_contract = {
                "task": task,
                "environment_seed": environment_seed,
                "branch_step": branch_step,
                "instruction": instruction,
                "recipient_seed": recipient_seed,
                "donor_seed": donor_seed,
            }
            for key_name, expected in summary_contract.items():
                if summary.get(key_name) != expected:
                    raise ValidationError(
                        f"{source}.branch_summary {key_name} changed: "
                        f"expected {expected!r}, got {summary.get(key_name)!r}"
                    )
    if manifest["excluded_development_state"] in state_ids:
        raise ValidationError("excluded development state appears in evaluation states")
    return manifest, file_sha


def validate_runner(path: Path, manifest: Mapping[str, Any]) -> tuple[Path, str]:
    runner, _ = _resolved_under(path, HOST_NFS_ROOT, source="runner", must_exist=True)
    if not runner.is_file():
        raise ValidationError(f"runner is not a regular file: {runner}")
    observed = sha256_file(runner)
    expected = str(require_mapping(manifest, "runtime", source="manifest")["single_state_runner_sha256"])
    if observed != expected or observed != FROZEN_RUNNER_SHA256:
        raise ValidationError(
            f"runner SHA-256 mismatch: expected {FROZEN_RUNNER_SHA256}, got {observed}"
        )
    return runner, observed


def docker_prefix(args: argparse.Namespace) -> list[str]:
    prefix = [args.docker_bin]
    if args.sudo:
        prefix.insert(0, "sudo")
    return prefix


def inspect_container(args: argparse.Namespace, manifest: Mapping[str, Any]) -> dict[str, Any]:
    command = [*docker_prefix(args), "inspect", args.container]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ValidationError(
            f"cannot inspect container {args.container!r}: {completed.stderr.strip()}"
        )
    try:
        records = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValidationError(f"docker inspect returned invalid JSON: {error}") from error
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValidationError("docker inspect must return exactly one container record")
    record = records[0]
    expected_image = str(require_mapping(manifest, "runtime", source="manifest")["server_image"])
    if record.get("Image") != expected_image:
        raise ValidationError(
            f"container image mismatch: expected {expected_image}, got {record.get('Image')!r}"
        )
    state = record.get("State")
    if not isinstance(state, Mapping) or state.get("Running") is not True:
        raise ValidationError(f"container {args.container!r} is not running")
    mounts = record.get("Mounts")
    if not isinstance(mounts, list):
        raise ValidationError("docker inspect Mounts must be an array")
    matches = []
    expected_source = str(HOST_NFS_ROOT.resolve(strict=True))
    for mount in mounts:
        if not isinstance(mount, Mapping):
            continue
        if mount.get("Destination") != str(CONTAINER_NFS_ROOT):
            continue
        source = Path(str(mount.get("Source", "")))
        try:
            source_resolved = str(source.resolve(strict=True))
        except OSError:
            source_resolved = str(source)
        if source_resolved == expected_source:
            matches.append(mount)
    if len(matches) != 1:
        raise ValidationError(
            f"container must map host {expected_source} exactly once to {CONTAINER_NFS_ROOT}"
        )
    if matches[0].get("RW") is not True:
        raise ValidationError(f"container mount {CONTAINER_NFS_ROOT} must be writable")
    return record


def _expect_equal(actual: Any, expected: Any, *, source: str) -> None:
    if actual != expected:
        raise ValidationError(f"{source}: expected {expected!r}, got {actual!r}")


def _validate_cache_counts(interface: Mapping[str, Any], *, source: str) -> None:
    counts = require_mapping(interface, "cache_call_counts", source=source)
    normalized: dict[int, int] = {}
    for key, value in counts.items():
        try:
            layer = int(key)
            count = int(value)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{source}: invalid cache census entry {key!r}: {value!r}") from error
        if str(layer) != str(key) and key != layer:
            raise ValidationError(f"{source}: noncanonical cache layer key {key!r}")
        if layer in normalized:
            raise ValidationError(f"{source}: duplicate normalized cache layer {layer}")
        normalized[layer] = count
    if normalized != {layer: 8 for layer in EXPECTED_LAYERS}:
        raise ValidationError(
            f"{source}: cache census must be exactly 8 calls in each of layers 0--35"
        )


def _validate_response_metadata(
    responses: Mapping[str, Any], *, state: Mapping[str, Any], study_id: str, source: str
) -> None:
    require_exact_keys(responses, EXPECTED_RESPONSE_LABELS, source=f"{source}.responses")
    recipient_seed = int(state["recipient_seed"])
    donor_seed = int(state["donor_seed"])
    recipient_id = f"{study_id}-recipient"
    donor_id = f"{study_id}-donor"
    parameter_hashes: set[str] = set()
    state_hashes: set[str] = set()
    typed: dict[str, Mapping[str, Any]] = {}
    for label in EXPECTED_RESPONSE_LABELS:
        response = responses[label]
        if not isinstance(response, Mapping):
            raise ValidationError(f"{source}.responses.{label}: must be an object")
        typed[label] = response
        state_hashes.add(str(require(response, "research_state_hash", source=f"{source}.responses.{label}")))
        parameter_hashes.add(
            str(require(response, "research_parameter_probe_hash", source=f"{source}.responses.{label}"))
        )
    if len(state_hashes) != 1 or "" in state_hashes:
        raise ValidationError(f"{source}: response-level model-state fingerprints diverged")
    if len(parameter_hashes) != 1 or "" in parameter_hashes:
        raise ValidationError(f"{source}: model parameter-probe fingerprints diverged")

    native_contract = {
        "recipient-native": (recipient_seed, "native", recipient_id),
        "recipient-repeat": (recipient_seed, "native", f"{study_id}-recipient-repeat"),
        "donor-native": (donor_seed, "native", donor_id),
    }
    for label, (seed, mode, research_id) in native_contract.items():
        response = typed[label]
        _expect_equal(response.get("research_seed"), seed, source=f"{source}.responses.{label}.research_seed")
        _expect_equal(response.get("research_mode"), mode, source=f"{source}.responses.{label}.research_mode")
        _expect_equal(response.get("research_id"), research_id, source=f"{source}.responses.{label}.research_id")
    recipient_path_hash = typed["recipient-native"].get("research_path_noise_hash")
    _expect_equal(
        typed["recipient-repeat"].get("research_path_noise_hash"),
        recipient_path_hash,
        source=f"{source}.responses.recipient-repeat.research_path_noise_hash",
    )
    if not recipient_path_hash or typed["donor-native"].get("research_path_noise_hash") == recipient_path_hash:
        raise ValidationError(f"{source}: recipient and donor native RNG paths are not distinct")

    recipient_source = {
        "recipient-baseline",
        "recipient-kv-record",
        "recipient-kv-replay",
        "recipient-future-donor-kv",
    }
    for label in INTERVENTION_LABELS:
        response = typed[label]
        expected_mode = "self" if label in recipient_source else "donor"
        expected_donor = recipient_id if label in recipient_source else donor_id
        _expect_equal(
            response.get("research_seed"), recipient_seed, source=f"{source}.responses.{label}.research_seed"
        )
        _expect_equal(
            response.get("research_mode"), expected_mode, source=f"{source}.responses.{label}.research_mode"
        )
        _expect_equal(
            response.get("research_recipient_id"),
            recipient_id,
            source=f"{source}.responses.{label}.research_recipient_id",
        )
        _expect_equal(
            response.get("research_donor_id"),
            expected_donor,
            source=f"{source}.responses.{label}.research_donor_id",
        )
        _expect_equal(
            response.get("research_id"),
            f"{study_id}-{label}",
            source=f"{source}.responses.{label}.research_id",
        )

    for label, expected_mode in ATTENTION_LABEL_MODES.items():
        response = typed[label]
        _expect_equal(
            response.get("research_attention_exclude_layers"),
            list(EXPECTED_LAYERS),
            source=f"{source}.responses.{label}.research_attention_exclude_layers",
        )
        _expect_equal(
            response.get("research_attention_exclude_scope"),
            "action",
            source=f"{source}.responses.{label}.research_attention_exclude_scope",
        )
        interface = response.get("research_attention_interface")
        if not isinstance(interface, Mapping):
            raise ValidationError(f"{source}.responses.{label}.research_attention_interface missing")
        _expect_equal(interface.get("layers"), 36, source=f"{source}.responses.{label}.interface.layers")
        _expect_equal(
            interface.get("excluded_keys_values"),
            "future_video",
            source=f"{source}.responses.{label}.interface.excluded_keys_values",
        )
        _expect_equal(
            interface.get("instrumented_server"), True, source=f"{source}.responses.{label}.interface.instrumented_server"
        )
        _expect_equal(
            interface.get("intervention_requested"), True, source=f"{source}.responses.{label}.interface.intervention_requested"
        )
        _expect_equal(interface.get("mode"), expected_mode, source=f"{source}.responses.{label}.interface.mode")
        expected_cache = (
            f"{study_id}-recipient-kv" if label.startswith("recipient-kv") or label == "donor-future-recipient-kv"
            else f"{study_id}-donor-kv"
        )
        _expect_equal(interface.get("cache_id"), expected_cache, source=f"{source}.responses.{label}.interface.cache_id")
        _validate_cache_counts(interface, source=f"{source}.responses.{label}.interface")


def validate_report(
    path: Path,
    *,
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    report_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    source = f"report[{state['state_id']}]"
    if not path.is_file():
        raise ValidationError(f"{source}: missing {path}")
    observed_sha = sha256_file(path)
    if report_sha256 is not None and observed_sha != report_sha256:
        raise ValidationError(
            f"{source}: receipt/report SHA-256 mismatch; expected {report_sha256}, got {observed_sha}"
        )
    report = read_json(path, label=source)
    study_id = f"{manifest['manifest_id']}-{state['state_id']}"
    _expect_equal(report.get("status"), "complete", source=f"{source}.status")
    _expect_equal(report.get("scope"), manifest["scope"], source=f"{source}.scope")
    _expect_equal(report.get("study_id"), study_id, source=f"{source}.study_id")
    _expect_equal(report.get("recipient_seed"), state["recipient_seed"], source=f"{source}.recipient_seed")
    _expect_equal(report.get("donor_seed"), state["donor_seed"], source=f"{source}.donor_seed")
    _expect_equal(report.get("layers"), list(EXPECTED_LAYERS), source=f"{source}.layers")

    current = require_mapping(report, "current_request", source=source)
    request_contract = {
        "asset_video": str(host_to_container(Path(str(state["asset_video"])), source=f"{source}.asset_video")),
        "recorded_hdf5": str(
            host_to_container(Path(str(state["recorded_hdf5"])), source=f"{source}.recorded_hdf5")
        ),
        "branch_summary": str(
            host_to_container(Path(str(state["branch_summary"])), source=f"{source}.branch_summary")
        ),
        "branch_step": state["branch_step"],
        "prompt": state["instruction"],
        "proprio_source": "noise-free recorded post-step simulator state",
    }
    for key, expected in request_contract.items():
        _expect_equal(current.get(key), expected, source=f"{source}.current_request.{key}")
    report_hashes = require_mapping(report, "input_sha256", source=source)
    require_exact_keys(
        report_hashes,
        ("asset_video", "recorded_hdf5", "branch_summary"),
        source=f"{source}.input_sha256",
    )
    _expect_equal(dict(report_hashes), dict(state["input_sha256"]), source=f"{source}.input_sha256")

    native_action_l2 = require_finite(report, "native_action_l2", source=source)
    if native_action_l2 <= 0.0:
        raise ValidationError(f"{source}: recipient/donor action axis is degenerate")
    exact_errors = require_mapping(report, "exact_errors", source=source)
    exact_error_names = (
        "recipient_native_repeat",
        "recipient_record_vs_baseline",
        "recipient_replay_vs_record",
        "donor_record_vs_baseline",
        "donor_replay_vs_record",
    )
    require_exact_keys(exact_errors, exact_error_names, source=f"{source}.exact_errors")
    for name in exact_error_names:
        if require_finite(exact_errors, name, source=f"{source}.exact_errors") != 0.0:
            raise ValidationError(f"{source}: exact identity/replay control {name} failed")
    _expect_equal(report.get("state_hash_count"), 1, source=f"{source}.state_hash_count")
    state_hashes = require_sequence(report, "state_hashes", source=source)
    if len(state_hashes) != 1 or not str(state_hashes[0]):
        raise ValidationError(f"{source}: expected one nonempty model-state fingerprint")

    coordinate = require_mapping(report, "action_coordinate_errors", source=source)
    require_exact_keys(coordinate, INTERVENTION_LABELS, source=f"{source}.action_coordinate_errors")
    for label in INTERVENTION_LABELS:
        errors = coordinate[label]
        if not isinstance(errors, Mapping):
            raise ValidationError(f"{source}.action_coordinate_errors.{label} must be an object")
        require_exact_keys(errors, ("input", "output"), source=f"{source}.action_coordinate_errors.{label}")
        for kind in ("input", "output"):
            if require_finite(errors, kind, source=f"{source}.action_coordinate_errors.{label}") != 0.0:
                raise ValidationError(f"{source}: {label} directly mutated {kind} action coordinates")

    consistency = require_mapping(report, "future_target_consistency", source=source)
    consistency_names = (
        "recipient_future_arms_identical",
        "donor_future_arms_identical",
        "recipient_and_donor_targets_distinct",
    )
    require_exact_keys(consistency, consistency_names, source=f"{source}.future_target_consistency")
    for name in consistency_names:
        _expect_equal(consistency[name], True, source=f"{source}.future_target_consistency.{name}")
    signatures = require_mapping(report, "future_signatures", source=source)
    require_exact_keys(signatures, INTERVENTION_LABELS, source=f"{source}.future_signatures")
    for label in INTERVENTION_LABELS:
        signature = signatures[label]
        if not isinstance(signature, Mapping):
            raise ValidationError(f"{source}.future_signatures.{label} must be an object")
        for hash_name in ("target_hash", "output_hash"):
            require_sha256(signature.get(hash_name), source=f"{source}.future_signatures.{label}.{hash_name}")
        error = require_finite(signature, "target_max_error", source=f"{source}.future_signatures.{label}")
        if error < 0.0:
            raise ValidationError(f"{source}.future_signatures.{label}: negative target error")
    recipient_future = (
        "recipient-baseline",
        "recipient-kv-record",
        "recipient-kv-replay",
        "recipient-future-donor-kv",
    )
    donor_future = (
        "donor-baseline",
        "donor-future-recipient-kv",
        "donor-kv-record",
        "donor-kv-replay",
    )
    for labels in (recipient_future, donor_future):
        signature_tuples = {
            (
                signatures[label]["target_hash"],
                signatures[label]["output_hash"],
                float(signatures[label]["target_max_error"]),
            )
            for label in labels
        }
        if len(signature_tuples) != 1:
            raise ValidationError(f"{source}: visible-future signature changed within a future source")
    if signatures[recipient_future[0]]["target_hash"] == signatures[donor_future[0]]["target_hash"]:
        raise ValidationError(f"{source}: recipient and donor future targets are identical")

    factorial = require_mapping(report, "factorial", source=source)
    require_exact_keys(factorial, EXPECTED_CELLS, source=f"{source}.factorial")
    for cell, response_label in FACTORIAL_RESPONSE_LABELS.items():
        payload = factorial[cell]
        if not isinstance(payload, Mapping):
            raise ValidationError(f"{source}.factorial.{cell}: must be an object")
        require_exact_keys(
            payload,
            ("response_label", "donor_projection", "distance_to_recipient", "distance_to_donor", "target_future_max_error"),
            source=f"{source}.factorial.{cell}",
        )
        _expect_equal(payload.get("response_label"), response_label, source=f"{source}.factorial.{cell}.response_label")
        require_finite(payload, "donor_projection", source=f"{source}.factorial.{cell}")
        for key in ("distance_to_recipient", "distance_to_donor", "target_future_max_error"):
            if require_finite(payload, key, source=f"{source}.factorial.{cell}") < 0.0:
                raise ValidationError(f"{source}.factorial.{cell}.{key}: must be nonnegative")
        _expect_equal(
            float(payload["target_future_max_error"]),
            float(signatures[response_label]["target_max_error"]),
            source=f"{source}.factorial.{cell}.target_future_max_error",
        )

    attention_interfaces = require_mapping(report, "attention_interfaces", source=source)
    require_exact_keys(attention_interfaces, EXPECTED_RESPONSE_LABELS, source=f"{source}.attention_interfaces")
    responses = require_mapping(report, "responses", source=source)
    _validate_response_metadata(responses, state=state, study_id=study_id, source=source)
    for label in EXPECTED_RESPONSE_LABELS:
        _expect_equal(
            attention_interfaces[label],
            responses[label].get("research_attention_interface"),
            source=f"{source}.attention_interfaces.{label}",
        )
    return report, observed_sha


def receipt_payload(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    state: Mapping[str, Any],
    runner_sha256: str,
    report_sha256: str,
    container: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "accepted",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha256,
        "state_id": state["state_id"],
        "state_spec_sha256": sha256_bytes(canonical_json(state)),
        "study_id": f"{manifest['manifest_id']}-{state['state_id']}",
        "runner_sha256": runner_sha256,
        "server_image": require_mapping(manifest, "runtime", source="manifest")["server_image"],
        "container": container,
        "report_sha256": report_sha256,
        "completed_at_utc": utc_now(),
        "validated_controls": {
            "exact_identity_and_replay": True,
            "single_state_fingerprint": True,
            "single_parameter_probe_fingerprint": True,
            "recipient_rng_repeat_exact_and_donor_rng_distinct": True,
            "fixed_recipient_rng_for_all_interventions": True,
            "future_signatures_fixed_within_source_and_distinct_between_sources": True,
            "no_action_coordinate_mutation": True,
            "cache_layers": 36,
            "cache_calls_per_layer": 8,
        },
    }


def validate_receipt(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    state: Mapping[str, Any],
    runner_sha256: str,
    report_sha256: str,
) -> dict[str, Any]:
    receipt = read_json(path, label=f"receipt[{state['state_id']}]")
    expected = {
        "schema_version": 1,
        "status": "accepted",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha256,
        "state_id": state["state_id"],
        "state_spec_sha256": sha256_bytes(canonical_json(state)),
        "study_id": f"{manifest['manifest_id']}-{state['state_id']}",
        "runner_sha256": runner_sha256,
        "server_image": require_mapping(manifest, "runtime", source="manifest")["server_image"],
        "report_sha256": report_sha256,
    }
    for key, value in expected.items():
        _expect_equal(receipt.get(key), value, source=f"receipt[{state['state_id']}].{key}")
    controls = require_mapping(receipt, "validated_controls", source=f"receipt[{state['state_id']}]")
    for key, value in controls.items():
        if key in {"cache_layers", "cache_calls_per_layer"}:
            continue
        _expect_equal(value, True, source=f"receipt[{state['state_id']}].validated_controls.{key}")
    _expect_equal(controls.get("cache_layers"), 36, source=f"receipt[{state['state_id']}].validated_controls.cache_layers")
    _expect_equal(
        controls.get("cache_calls_per_layer"),
        8,
        source=f"receipt[{state['state_id']}].validated_controls.cache_calls_per_layer",
    )
    return receipt


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_create_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    """Atomically create ``path`` while refusing an existing destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {path}") from error
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_create_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    atomic_create_bytes(path, payload)


def acquire_lock(output_root: Path) -> BinaryIO:
    lock_path = output_root.parent / f".{output_root.name}.launcher.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"another launcher holds {lock_path}") from error
    return handle


def prepare_output_root(
    output_root: Path, *, manifest_path: Path, manifest_sha256: str, resume: bool
) -> Path:
    resolved_parent, relative_parent = _resolved_under(
        output_root.parent,
        HOST_NFS_ROOT,
        source="output-root parent",
        must_exist=True,
    )
    root = resolved_parent / output_root.name
    if root.exists() and not root.is_dir():
        raise ValidationError(f"output root exists but is not a directory: {root}")
    if root.exists() and not resume and any(root.iterdir()):
        raise FileExistsError(
            f"output root is nonempty; use --resume only for this exact frozen run: {root}"
        )
    root.mkdir(mode=0o755, exist_ok=True)
    snapshot = root / "manifest.json"
    if snapshot.exists():
        if sha256_file(snapshot) != manifest_sha256:
            raise ValidationError(f"output manifest snapshot differs from frozen manifest: {snapshot}")
    else:
        if resume and any(root.iterdir()):
            raise ValidationError("resume output has artifacts but no manifest snapshot")
        atomic_create_bytes(snapshot, manifest_path.read_bytes())
    # Assert that the eventual report path maps through the requested mount.
    host_to_container(root / "states" / "placeholder" / "report.json", must_exist=False, source="output report")
    del relative_parent
    return root


def run_command(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    runner: Path,
    report_path: Path,
) -> list[str]:
    return [
        *docker_prefix(args),
        "exec",
        "--env",
        "LD_LIBRARY_PATH=",
        "--env",
        "PYTHONPATH=/research/src",
        args.container,
        args.container_python,
        str(host_to_container(runner, source="runner")),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--asset-video",
        str(host_to_container(Path(str(state["asset_video"])), source="asset_video")),
        "--recorded-hdf5",
        str(host_to_container(Path(str(state["recorded_hdf5"])), source="recorded_hdf5")),
        "--branch-summary",
        str(host_to_container(Path(str(state["branch_summary"])), source="branch_summary")),
        "--branch-step",
        str(state["branch_step"]),
        "--prompt",
        str(state["instruction"]),
        "--output",
        str(host_to_container(report_path, must_exist=False, source="output report")),
        "--study-id",
        f"{manifest['manifest_id']}-{state['state_id']}",
        "--scope",
        str(manifest["scope"]),
        "--recipient-seed",
        str(state["recipient_seed"]),
        "--donor-seed",
        str(state["donor_seed"]),
    ]


def _failed_log_name(state_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return state_dir / f"attempt-{stamp}-{uuid.uuid4().hex[:8]}.failed.log"


def launch_state(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    state: Mapping[str, Any],
    runner: Path,
    runner_sha256: str,
    output_root: Path,
) -> str:
    state_dir = output_root / "states" / str(state["state_id"])
    state_dir.mkdir(parents=True, exist_ok=True)
    report_path = state_dir / "report.json"
    receipt_path = state_dir / "receipt.json"
    run_log = state_dir / "run.log"

    if receipt_path.exists():
        if not report_path.exists() or not run_log.exists():
            raise ValidationError(f"{state['state_id']}: receipt exists without report and run log")
        _, report_sha = validate_report(report_path, state=state, manifest=manifest)
        validate_receipt(
            receipt_path,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            state=state,
            runner_sha256=runner_sha256,
            report_sha256=report_sha,
        )
        return "skipped-valid"

    if report_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing pre-existing report without --resume: {report_path}")
        if not run_log.exists():
            raise ValidationError(
                f"{state['state_id']}: report lacks run.log proof of a zero-exit launcher attempt"
            )
        _, report_sha = validate_report(report_path, state=state, manifest=manifest)
        atomic_create_json(
            receipt_path,
            receipt_payload(
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                state=state,
                runner_sha256=runner_sha256,
                report_sha256=report_sha,
                container=args.container,
            ),
        )
        return "reconciled-valid"

    if run_log.exists():
        raise ValidationError(f"{state['state_id']}: run.log exists without a completed report")
    if not args.resume:
        unexpected = list(state_dir.iterdir())
        if unexpected:
            raise FileExistsError(
                f"refusing nonempty state directory without --resume: {state_dir}"
            )
    else:
        unexpected = [
            path
            for path in state_dir.iterdir()
            if not (path.is_file() and path.name.startswith("attempt-") and path.name.endswith(".failed.log"))
        ]
        if unexpected:
            raise ValidationError(
                f"{state['state_id']}: resume found unrecognized partial artifacts: "
                f"{[path.name for path in unexpected]}"
            )

    command = run_command(
        args=args,
        manifest=manifest,
        state=state,
        runner=runner,
        report_path=report_path,
    )
    temporary_log = state_dir / f".attempt-{uuid.uuid4().hex}.log"
    started = time.monotonic()
    return_code: int | None = None
    try:
        with temporary_log.open("xb") as log:
            header = {
                "event": "state_start",
                "state_id": state["state_id"],
                "started_at_utc": utc_now(),
                "command_argv": command,
            }
            log.write(canonical_json(header) + b"\n")
            log.flush()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            return_code = process.wait()
            footer = {
                "event": "state_exit",
                "state_id": state["state_id"],
                "return_code": return_code,
                "elapsed_seconds": time.monotonic() - started,
                "ended_at_utc": utc_now(),
            }
            log.write(b"\n" + canonical_json(footer) + b"\n")
            log.flush()
            os.fsync(log.fileno())
    except BaseException:
        if temporary_log.exists():
            os.rename(temporary_log, _failed_log_name(state_dir))
            _fsync_directory(state_dir)
        raise
    if return_code != 0:
        failed_log = _failed_log_name(state_dir)
        os.rename(temporary_log, failed_log)
        _fsync_directory(state_dir)
        raise RuntimeError(
            f"{state['state_id']}: worker exited {return_code}; retained log at {failed_log}"
        )
    if not report_path.exists():
        failed_log = _failed_log_name(state_dir)
        os.rename(temporary_log, failed_log)
        _fsync_directory(state_dir)
        raise RuntimeError(
            f"{state['state_id']}: worker exited zero without an atomic report; log at {failed_log}"
        )
    if run_log.exists():
        raise FileExistsError(f"refusing to replace {run_log}")
    os.rename(temporary_log, run_log)
    _fsync_directory(state_dir)
    _, report_sha = validate_report(report_path, state=state, manifest=manifest)
    atomic_create_json(
        receipt_path,
        receipt_payload(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            state=state,
            runner_sha256=runner_sha256,
            report_sha256=report_sha,
            container=args.container,
        ),
    )
    return "completed"


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValidationError("--port must be in [1, 65535]")
    manifest_path = args.manifest.resolve(strict=True)
    manifest, manifest_sha = validate_manifest(manifest_path, verify_inputs=True)
    runner, runner_sha = validate_runner(args.runner, manifest)
    inspect_container(args, manifest)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "manifest_id": manifest["manifest_id"],
                    "manifest_sha256": manifest_sha,
                    "runner_sha256": runner_sha,
                    "states": len(manifest["states"]),
                    "container": args.container,
                    "server_image": manifest["runtime"]["server_image"],
                    "host_mount": str(HOST_NFS_ROOT),
                    "container_mount": str(CONTAINER_NFS_ROOT),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    output_candidate = args.output_root
    if not output_candidate.is_absolute():
        raise ValidationError("--output-root must be absolute")
    _resolved_under(
        output_candidate.parent,
        HOST_NFS_ROOT,
        source="output-root parent",
        must_exist=False,
    )
    lock = acquire_lock(output_candidate)
    try:
        output_root = prepare_output_root(
            output_candidate,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            resume=args.resume,
        )
        states = manifest["states"]
        counts = {"completed": 0, "skipped-valid": 0, "reconciled-valid": 0}
        for index, state in enumerate(states, start=1):
            print(
                f"[{index:02d}/{len(states):02d}] {state['state_id']}: validating/running",
                flush=True,
            )
            status = launch_state(
                args=args,
                manifest=manifest,
                manifest_sha256=manifest_sha,
                state=state,
                runner=runner,
                runner_sha256=runner_sha,
                output_root=output_root,
            )
            counts[status] += 1
            print(f"[{index:02d}/{len(states):02d}] {state['state_id']}: {status}", flush=True)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "manifest_id": manifest["manifest_id"],
                    "output_root": str(output_root),
                    "states": len(states),
                    **counts,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, FileExistsError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
