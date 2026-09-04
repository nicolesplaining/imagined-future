#!/usr/bin/env python3
"""Freeze and materialize an outcome-blind DreamZero-DROID state cohort.

The utility deliberately separates state selection from file transfer:

* ``freeze`` reads only LeRobot metadata, deterministically selects states, and
  writes a read-only, content-addressed manifest.
* ``download`` verifies that frozen manifest and fetches exactly its parquet
  and video resources from immutable Hugging Face ``resolve`` URLs.
* ``all`` performs those two phases in order.

No parquet row, image, video frame, native rollout, or intervention outcome is
read during selection.  A failed or missing download never causes replacement
of a frozen state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "dreamzero-droid-evaluation-state-manifest-v1"
RECEIPT_SCHEMA = "dreamzero-droid-download-receipt-v1"
DATASET_ID = "GEAR-Dreams/DreamZero-DROID-Data"
DATASET_REVISION = "2abc197ca7f14f53a6bf464bf80018ce998f18cc"
SELECTION_SALT = "dreamzero-droid-30-state-v1-20260904"
DEFAULT_STATE_COUNT = 30
DEFAULT_FAMILY_COUNT = 10
DEFAULT_FRAME_FRACTION = 0.50
DEFAULT_FRAME_MARGIN = 48
DEFAULT_MIN_EPISODE_LENGTH = 97
BLOCK_BYTES = 8 * 1024 * 1024
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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
        for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes, mode: int = 0o444) -> None:
    """Create ``path`` atomically and refuse to replace an existing artifact."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
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
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any, mode: int = 0o444) -> None:
    rendered = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    atomic_bytes(path, rendered, mode=mode)


def write_sha256_sidecar(path: Path) -> Path:
    sidecar = path.with_name(f"{path.name}.sha256")
    atomic_bytes(sidecar, f"{sha256_file(path)}  {path.name}\n".encode("ascii"))
    return sidecar


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object in {path}:{line_number}")
            yield value


def normalized_task(task: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(task.casefold()))


def task_family(task: str) -> str:
    tokens = TOKEN_PATTERN.findall(task.casefold())
    return tokens[0] if tokens else "unknown"


def stable_rank(salt: str, *fields: object) -> str:
    value = "\x1f".join((salt, *(str(field) for field in fields)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_tasks(raw_tasks: object) -> tuple[str, ...]:
    if not isinstance(raw_tasks, list):
        return ()
    values = []
    for value in raw_tasks:
        if not isinstance(value, str):
            continue
        task = " ".join(value.split())
        if task and task.casefold() != "not provided":
            values.append(task)
    return tuple(values)


def state_frame_index(length: int, fraction: float, margin: int) -> int:
    if length < 2 * margin + 1:
        raise ValueError(
            f"episode length {length} cannot support a {margin}-frame margin"
        )
    continuous = fraction * (length - 1)
    # Half-up rounding is specified here rather than relying on Python's
    # ties-to-even round(), which is less obvious in a frozen protocol.
    nearest = int(math.floor(continuous + 0.5))
    return min(length - 1 - margin, max(margin, nearest))


def metadata_inventory(metadata_root: Path) -> list[dict[str, Any]]:
    names = (
        "episodes.jsonl",
        "tasks.jsonl",
        "info.json",
        "modality.json",
        "relative_stats_dreamzero.json",
        "relative_horizon_stats_dreamzero.json",
    )
    rows = []
    for name in names:
        path = metadata_root / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required regular metadata file is missing: {path}")
        rows.append(
            {
                "relative_path": f"meta/{name}",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def load_task_indices(path: Path) -> dict[str, tuple[int, ...]]:
    values: dict[str, list[int]] = defaultdict(list)
    seen_indices: set[int] = set()
    for row in iter_jsonl(path):
        index = int(row["task_index"])
        task = str(row["task"])
        if index in seen_indices:
            raise ValueError(f"duplicate task_index {index} in {path}")
        seen_indices.add(index)
        values[task].append(index)
    return {task: tuple(indices) for task, indices in values.items()}


def eligible_candidates(
    episodes_path: Path,
    *,
    minimum_length: int,
    frame_fraction: float,
    frame_margin: int,
    salt: str,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    candidates: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen_episodes: set[int] = set()
    total_rows = 0
    for row in iter_jsonl(episodes_path):
        total_rows += 1
        episode_index = int(row["episode_index"])
        if episode_index in seen_episodes:
            raise ValueError(f"duplicate episode_index {episode_index}")
        seen_episodes.add(episode_index)
        length = int(row["length"])
        tasks = valid_tasks(row.get("tasks"))
        if not tasks:
            exclusions["missing_task"] += 1
            continue
        if length < minimum_length or length < 2 * frame_margin + 1:
            exclusions["episode_too_short"] += 1
            continue
        prompt = tasks[0]
        normalized = normalized_task(prompt)
        family = task_family(prompt)
        if not normalized or family == "unknown":
            exclusions["unparseable_primary_task"] += 1
            continue
        frame_index = state_frame_index(length, frame_fraction, frame_margin)
        candidates.append(
            {
                "episode_index": episode_index,
                "episode_length": length,
                "success_metadata": bool(row.get("success", False)),
                "task": prompt,
                "task_normalized": normalized,
                "task_family": family,
                "all_episode_tasks": list(tasks),
                "frame_index": frame_index,
                "rank_sha256": stable_rank(salt, family, episode_index, normalized),
            }
        )
    return candidates, dict(sorted(exclusions.items())), total_rows


def family_quotas(
    candidates: Sequence[Mapping[str, Any]], state_count: int, family_count: int
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    counts = Counter(str(candidate["task_family"]) for candidate in candidates)
    if not counts:
        raise ValueError("no eligible task families")
    selected_families = [
        family
        for family, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            : min(family_count, state_count, len(counts))
        ]
    ]
    base, extra = divmod(state_count, len(selected_families))
    quotas = {
        family: base + (1 if index < extra else 0)
        for index, family in enumerate(selected_families)
    }
    return selected_families, quotas, dict(sorted(counts.items()))


def select_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    state_count: int,
    family_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families, quotas, population = family_quotas(candidates, state_count, family_count)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["task_family"] in quotas:
            by_family[str(candidate["task_family"])].append(candidate)
    for values in by_family.values():
        values.sort(key=lambda value: (value["rank_sha256"], value["episode_index"]))

    selected: list[dict[str, Any]] = []
    used_episodes: set[int] = set()
    used_tasks: set[str] = set()
    shortfalls: dict[str, int] = {}
    for family in families:
        before = len(selected)
        for candidate in by_family[family]:
            if len(selected) - before >= quotas[family]:
                break
            episode = int(candidate["episode_index"])
            task = str(candidate["task_normalized"])
            if episode in used_episodes or task in used_tasks:
                continue
            selected.append(candidate)
            used_episodes.add(episode)
            used_tasks.add(task)
        missing = quotas[family] - (len(selected) - before)
        if missing:
            shortfalls[family] = missing

    # A deterministic global fill is allowed only when a selected family lacks
    # enough distinct task strings.  It never examines downloadable content.
    if len(selected) < state_count:
        remaining = sorted(
            candidates,
            key=lambda value: (
                stable_rank(
                    "dreamzero-droid-global-fill-v1",
                    value["rank_sha256"],
                    value["episode_index"],
                ),
                value["episode_index"],
            ),
        )
        for candidate in remaining:
            episode = int(candidate["episode_index"])
            task = str(candidate["task_normalized"])
            if episode in used_episodes or task in used_tasks:
                continue
            selected.append(candidate)
            used_episodes.add(episode)
            used_tasks.add(task)
            if len(selected) == state_count:
                break
    if len(selected) != state_count:
        raise ValueError(
            f"could select only {len(selected)} of {state_count} requested unique-task states"
        )

    selected.sort(
        key=lambda value: (
            families.index(value["task_family"])
            if value["task_family"] in families
            else len(families),
            value["rank_sha256"],
            value["episode_index"],
        )
    )
    realized = Counter(str(candidate["task_family"]) for candidate in selected)
    audit = {
        "eligible_family_population": population,
        "selected_families_by_population": families,
        "planned_family_quotas": quotas,
        "realized_family_counts": dict(sorted(realized.items())),
        "quota_shortfalls_before_global_fill": shortfalls,
        "unique_episode_count": len(used_episodes),
        "unique_normalized_task_count": len(used_tasks),
    }
    return selected, audit


def direct_url(dataset_id: str, revision: str, relative_path: str) -> str:
    repository = urllib.parse.quote(dataset_id, safe="/")
    pinned_revision = urllib.parse.quote(revision, safe="")
    resource = urllib.parse.quote(relative_path, safe="/")
    return f"https://huggingface.co/datasets/{repository}/resolve/{pinned_revision}/{resource}"


def format_resource_path(template: str, *, episode_index: int, video_key: str | None) -> str:
    values = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // 1000,
        "video_key": video_key,
    }
    try:
        return template.format(**values)
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported dataset path template {template!r}") from error


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if not HEX_40.fullmatch(args.revision):
        raise ValueError("--revision must be an immutable 40-character commit SHA")
    if args.state_count < 1 or args.family_count < 1:
        raise ValueError("state and family counts must be positive")
    if not 0.0 < args.frame_fraction < 1.0:
        raise ValueError("--frame-fraction must lie strictly between 0 and 1")
    if args.frame_margin < 0:
        raise ValueError("--frame-margin must be nonnegative")

    metadata_root = args.metadata_root.resolve()
    inventory = metadata_inventory(metadata_root)
    info = load_json(metadata_root / "info.json")
    if not isinstance(info, dict):
        raise ValueError("info.json must contain an object")
    if str(info.get("robot_type")) != "droid":
        raise ValueError(f"expected DROID metadata, got robot_type={info.get('robot_type')!r}")
    chunks_size = int(info["chunks_size"])
    if chunks_size != 1000:
        raise ValueError(f"expected chunks_size=1000, got {chunks_size}")
    fps = float(info["fps"])
    video_keys = sorted(
        key
        for key, feature in dict(info["features"]).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )
    expected_video_keys = [
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
        "observation.images.wrist_image_left",
    ]
    if video_keys != expected_video_keys:
        raise ValueError(f"unexpected DROID video keys: {video_keys}")

    minimum_length = max(args.min_episode_length, 2 * args.frame_margin + 1)
    candidates, exclusions, episode_rows = eligible_candidates(
        metadata_root / "episodes.jsonl",
        minimum_length=minimum_length,
        frame_fraction=args.frame_fraction,
        frame_margin=args.frame_margin,
        salt=args.selection_salt,
    )
    if episode_rows != int(info["total_episodes"]):
        raise ValueError(
            f"episodes row count {episode_rows} != info total {info['total_episodes']}"
        )
    selected, selection_audit = select_candidates(
        candidates,
        state_count=args.state_count,
        family_count=args.family_count,
    )
    task_indices = load_task_indices(metadata_root / "tasks.jsonl")

    resources_by_path: dict[str, dict[str, Any]] = {}
    states: list[dict[str, Any]] = []
    for state_index, candidate in enumerate(selected):
        episode_index = int(candidate["episode_index"])
        episode_chunk = episode_index // chunks_size
        resource_paths = [
            (
                "parquet",
                None,
                format_resource_path(
                    str(info["data_path"]),
                    episode_index=episode_index,
                    video_key=None,
                ),
            )
        ]
        for video_key in video_keys:
            resource_paths.append(
                (
                    "video",
                    video_key,
                    format_resource_path(
                        str(info["video_path"]),
                        episode_index=episode_index,
                        video_key=video_key,
                    ),
                )
            )
        resource_ids = []
        for kind, video_key, relative_path in resource_paths:
            resource_id = sha256_bytes(relative_path.encode("utf-8"))[:20]
            resource_ids.append(resource_id)
            value = {
                "resource_id": resource_id,
                "kind": kind,
                "episode_index": episode_index,
                "relative_path": relative_path,
                "destination_relative_path": f"files/{relative_path}",
                "url": direct_url(args.dataset_id, args.revision, relative_path),
            }
            if video_key is not None:
                value["video_key"] = video_key
            existing = resources_by_path.setdefault(relative_path, value)
            if existing != value:
                raise AssertionError(f"resource collision for {relative_path}")

        prompt = str(candidate["task"])
        matching_task_indices = list(task_indices.get(prompt, ()))
        if not matching_task_indices:
            raise ValueError(f"episode task is absent from tasks.jsonl: {prompt!r}")
        frame_index = int(candidate["frame_index"])
        states.append(
            {
                "state_index": state_index,
                "state_id": f"droid_episode_{episode_index:06d}_frame_{frame_index:06d}",
                "episode_index": episode_index,
                "episode_chunk": episode_chunk,
                "episode_length": int(candidate["episode_length"]),
                "frame_index": frame_index,
                "timestamp_seconds_from_fps": frame_index / fps,
                "frame_fraction_target": args.frame_fraction,
                "frame_margin": args.frame_margin,
                "task": prompt,
                "task_normalized": str(candidate["task_normalized"]),
                "task_family": str(candidate["task_family"]),
                "task_indices_for_exact_prompt": matching_task_indices,
                "all_episode_tasks": list(candidate["all_episode_tasks"]),
                "success_metadata": bool(candidate["success_metadata"]),
                "selection_rank_sha256": str(candidate["rank_sha256"]),
                "resource_ids": resource_ids,
            }
        )

    resources = sorted(resources_by_path.values(), key=lambda value: value["relative_path"])
    if len(resources) != args.state_count * (1 + len(video_keys)):
        raise AssertionError("each selected episode must contribute one parquet and three videos")
    body = {
        "schema": SCHEMA,
        "preparer": {
            "script_name": Path(__file__).name,
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "scope": {
            "model": "DreamZero-DROID",
            "domain": "DROID",
            "purpose": "external-WAM donor-future causal evaluation state preparation",
            "outcome_blind": True,
            "selection_inputs": ["meta/episodes.jsonl", "meta/tasks.jsonl", "meta/info.json"],
            "selection_forbidden_inputs": [
                "parquet rows",
                "video pixels",
                "model outputs",
                "native rollout actions",
                "intervention outcomes",
            ],
            "one_state_per_episode": True,
            "missing_resource_policy": "fail without replacement",
        },
        "dataset": {
            "dataset_id": args.dataset_id,
            "revision": args.revision,
            "robot_type": info["robot_type"],
            "codebase_version": info["codebase_version"],
            "fps": fps,
            "chunks_size": chunks_size,
            "video_keys": video_keys,
            "metadata_inventory": inventory,
        },
        "selection": {
            "algorithm": "top-frequency first-token families, equal quotas, SHA256-ranked unique tasks",
            "algorithm_version": "dreamzero-droid-balanced-task-family-v1",
            "selection_salt": args.selection_salt,
            "state_count": args.state_count,
            "family_count": args.family_count,
            "frame_fraction": args.frame_fraction,
            "frame_rounding": "nearest integer, exact half rounds upward",
            "frame_margin": args.frame_margin,
            "minimum_episode_length": minimum_length,
            "episode_metadata_rows": episode_rows,
            "eligible_candidate_count": len(candidates),
            "technical_exclusions": exclusions,
            **selection_audit,
        },
        "states": states,
        "resources": resources,
    }
    body_hash = sha256_bytes(canonical_json(body))
    return {
        **body,
        "manifest_id": f"dreamzero-droid-states-{body_hash[:16]}",
        "manifest_body_sha256": body_hash,
    }


def verify_manifest(manifest: Mapping[str, Any], metadata_root: Path) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unexpected manifest schema: {manifest.get('schema')!r}")
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_id", "manifest_body_sha256"}
    }
    body_hash = sha256_bytes(canonical_json(body))
    if manifest.get("manifest_body_sha256") != body_hash:
        raise ValueError("manifest body hash mismatch")
    if manifest.get("manifest_id") != f"dreamzero-droid-states-{body_hash[:16]}":
        raise ValueError("manifest ID does not match its body hash")
    expected_inventory = manifest["dataset"]["metadata_inventory"]
    if metadata_inventory(metadata_root.resolve()) != expected_inventory:
        raise ValueError("current metadata files do not match the frozen inventory")
    states = list(manifest["states"])
    resources = list(manifest["resources"])
    expected_count = int(manifest["selection"]["state_count"])
    if len(states) != expected_count or len(resources) != expected_count * 4:
        raise ValueError("manifest state/resource cardinality mismatch")
    if len({state["episode_index"] for state in states}) != expected_count:
        raise ValueError("manifest violates one-state-per-episode scope")
    if len({state["task_normalized"] for state in states}) != expected_count:
        raise ValueError("manifest does not contain unique normalized tasks")
    resource_ids = [resource["resource_id"] for resource in resources]
    paths = [resource["destination_relative_path"] for resource in resources]
    if len(set(resource_ids)) != len(resource_ids) or len(set(paths)) != len(paths):
        raise ValueError("manifest contains duplicate resource IDs or destinations")


def freeze_manifest(args: argparse.Namespace, manifest_path: Path) -> dict[str, Any]:
    manifest = build_manifest(args)
    atomic_json(manifest_path, manifest)
    write_sha256_sidecar(manifest_path)
    return manifest


class RecordingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Record immutable-revision/checksum headers while following HF redirects."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict[str, str]] = []

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> urllib.request.Request | None:
        self.history.append({str(key).casefold(): str(value) for key, value in headers.items()})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def clean_etag(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().removeprefix("W/").strip('"').casefold()
    return cleaned or None


def redirect_metadata(history: Sequence[Mapping[str, str]]) -> tuple[str | None, str | None]:
    revision = None
    linked_etag = None
    for headers in history:
        revision = revision or headers.get("x-repo-commit")
        linked_etag = linked_etag or clean_etag(headers.get("x-linked-etag"))
    return revision, linked_etag


def request_opener() -> tuple[urllib.request.OpenerDirector, RecordingRedirectHandler]:
    redirects = RecordingRedirectHandler()
    return urllib.request.build_opener(redirects), redirects


def make_request(url: str, token: str | None, *, start: int | None = None) -> urllib.request.Request:
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "imagined-future-dreamzero-state-preparer/1",
    }
    if start is not None:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers, method="GET")
    # Unredirected headers are not forwarded to the signed storage hostname.
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def content_total(headers: Mapping[str, str], start: int, status: int) -> int | None:
    content_range = headers.get("Content-Range") or headers.get("content-range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        if total.isdigit():
            return int(total)
    content_length = headers.get("Content-Length") or headers.get("content-length")
    if content_length and str(content_length).isdigit():
        length = int(content_length)
        return start + length if status == 206 else length
    return None


def verify_existing_file(
    destination: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
) -> tuple[int, str]:
    size = destination.stat().st_size
    digest = sha256_file(destination)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"existing file has wrong size: {destination}: {size} != {expected_size}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"existing file has wrong SHA-256: {destination}")
    return size, digest


def download_resource(
    output_root: Path,
    resource: Mapping[str, Any],
    *,
    revision: str,
    token: str | None,
    retries: int,
    timeout: float,
) -> dict[str, Any]:
    destination_relative = str(resource["destination_relative_path"])
    destination = (output_root / destination_relative).resolve()
    files_root = (output_root / "files").resolve()
    if files_root != destination and files_root not in destination.parents:
        raise ValueError(f"resource escapes files root: {destination_relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    if destination.is_symlink() or partial.is_symlink():
        raise ValueError(f"refusing symlink resource path: {destination}")

    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            opener, redirects = request_opener()
            request = make_request(str(resource["url"]), token, start=start if start else None)
            with opener.open(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                observed_revision, linked_etag = redirect_metadata(redirects.history)
                if observed_revision and observed_revision != revision:
                    raise ValueError(
                        f"resolved repository commit {observed_revision} != frozen {revision}"
                    )
                expected_size = content_total(response.headers, start, status)
                expected_sha256 = linked_etag if linked_etag and HEX_64.fullmatch(linked_etag) else None

                if destination.exists():
                    size, digest = verify_existing_file(
                        destination,
                        expected_size=expected_size,
                        expected_sha256=expected_sha256,
                    )
                    os.chmod(destination, 0o444)
                    return {
                        **dict(resource),
                        "size_bytes": size,
                        "sha256": digest,
                        "remote_linked_etag": linked_etag,
                        "observed_repo_commit": observed_revision,
                        "download_disposition": "verified_existing",
                    }

                if start and status != 206:
                    # The endpoint ignored Range; restart this dedicated partial
                    # rather than concatenating a complete response.
                    partial.unlink()
                    start = 0
                    mode = "wb"
                else:
                    mode = "ab" if start else "wb"
                with partial.open(mode) as handle:
                    for block in iter(lambda: response.read(BLOCK_BYTES), b""):
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            size = partial.stat().st_size
            digest = sha256_file(partial)
            if expected_size is not None and size != expected_size:
                raise IOError(f"short resource {resource['relative_path']}: {size} != {expected_size}")
            if expected_sha256 is not None and digest != expected_sha256:
                raise IOError(f"remote checksum mismatch for {resource['relative_path']}")
            os.chmod(partial, 0o444)
            os.replace(partial, destination)
            fsync_directory(destination.parent)
            return {
                **dict(resource),
                "size_bytes": size,
                "sha256": digest,
                "remote_linked_etag": linked_etag,
                "observed_repo_commit": observed_revision,
                "download_disposition": "downloaded",
            }
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 416 and partial.exists():
                # A stale/full partial cannot be resumed beyond EOF.  It has
                # never been admitted as a completed resource, so restart the
                # dedicated partial on the next bounded retry.
                partial.unlink()
                if attempt + 1 < retries:
                    time.sleep(min(2**attempt, 30))
                continue
            if error.code in {400, 401, 403, 404}:
                break
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
            if isinstance(error, ValueError):
                break
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"failed to materialize {resource['relative_path']} after {retries} attempts: {last_error}"
    ) from last_error


def verify_completed_receipt(
    output_root: Path, manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("unexpected receipt schema")
    if receipt.get("manifest_id") != manifest.get("manifest_id"):
        raise ValueError("receipt manifest ID mismatch")
    rows = list(receipt.get("resources", []))
    if len(rows) != len(manifest["resources"]):
        raise ValueError("receipt resource count mismatch")
    manifest_paths = {resource["destination_relative_path"] for resource in manifest["resources"]}
    receipt_paths = {resource["destination_relative_path"] for resource in rows}
    if receipt_paths != manifest_paths:
        raise ValueError("receipt resource set mismatch")
    for row in rows:
        path = output_root / row["destination_relative_path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"receipt resource is missing or not regular: {path}")
        verify_existing_file(
            path,
            expected_size=int(row["size_bytes"]),
            expected_sha256=str(row["sha256"]),
        )


def download_manifest(
    args: argparse.Namespace, manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    receipt_path = output_root / "download_receipt.json"
    manifest_file_hash = sha256_file(manifest_path)
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if receipt.get("manifest_file_sha256") != manifest_file_hash:
            raise ValueError("existing receipt refers to a different manifest file")
        verify_completed_receipt(output_root, manifest, receipt)
        return receipt

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    resources = list(manifest["resources"])
    revision = str(manifest["dataset"]["revision"])
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_resource,
                output_root,
                resource,
                revision=revision,
                token=token,
                retries=args.retries,
                timeout=args.timeout,
            ): resource
            for resource in resources
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    results.sort(key=lambda value: value["relative_path"])
    observed_commits = sorted(
        {row["observed_repo_commit"] for row in results if row["observed_repo_commit"]}
    )
    if observed_commits and observed_commits != [revision]:
        raise ValueError(f"download resolved unexpected revisions: {observed_commits}")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "manifest_id": manifest["manifest_id"],
        "manifest_file_sha256": manifest_file_hash,
        "dataset_revision": revision,
        "resource_count": len(results),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in results),
        "resources": results,
    }
    atomic_json(receipt_path, receipt)
    write_sha256_sidecar(receipt_path)
    verify_completed_receipt(output_root, manifest, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("freeze", "download", "all"),
        default="freeze",
        help="freeze only (default), download an existing freeze, or perform both in order",
    )
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--state-count", type=int, default=DEFAULT_STATE_COUNT)
    parser.add_argument("--family-count", type=int, default=DEFAULT_FAMILY_COUNT)
    parser.add_argument("--selection-salt", default=SELECTION_SALT)
    parser.add_argument("--frame-fraction", type=float, default=DEFAULT_FRAME_FRACTION)
    parser.add_argument("--frame-margin", type=int, default=DEFAULT_FRAME_MARGIN)
    parser.add_argument(
        "--min-episode-length", type=int, default=DEFAULT_MIN_EPISODE_LENGTH
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("workers, retries, and timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.json"
    if args.mode in {"freeze", "all"}:
        manifest = freeze_manifest(args, manifest_path)
    else:
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise FileNotFoundError(f"frozen manifest is missing: {manifest_path}")
        manifest = load_json(manifest_path)
    verify_manifest(manifest, args.metadata_root)

    print(
        f"manifest_id={manifest['manifest_id']} "
        f"states={len(manifest['states'])} resources={len(manifest['resources'])} "
        f"manifest_sha256={sha256_file(manifest_path)} path={manifest_path}"
    )
    if args.mode in {"download", "all"}:
        receipt = download_manifest(args, manifest_path, manifest)
        print(
            f"download_status={receipt['status']} resources={receipt['resource_count']} "
            f"bytes={receipt['total_size_bytes']} "
            f"receipt_sha256={sha256_file(output_root / 'download_receipt.json')}"
        )


if __name__ == "__main__":
    main()
