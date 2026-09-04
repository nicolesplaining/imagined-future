#!/usr/bin/env python3
"""Exhaustively export all frozen DreamZero native futures as post-analysis media.

This wrapper invokes the already-audited representative-video exporter for the
Cartesian product of all 30 frozen states and four registered seeds.  It adds
an umbrella census, requires each regenerated trace to byte-match its frozen
core trace, checks every MP4 mechanically, and installs a read-only package in
a fresh output directory.  It never writes into the frozen core result tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_SHA256 = "d1ffc3111a10bed9ac8fdd17c631dc3a5d8eb3128ac4fa250d9398bcede12cfc"
CORE_INVENTORY_SHA256 = "42e3b916612c92bd4732e7765d478f6c0a7ef3a2032b85458f139b51039cb60c"
RUNNER_SHA256 = "e627132e037679717512faac2f7bc46ddda8898f1e7bfe5637445a99e8163019"
EXPORTER_SHA256 = "e7efd87a06f6a2e0d11a9a6bc1a6c28336346fc3060bb19f73953eadad92ff4c"
BRANCH_SEEDS = (211, 223, 227, 229)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--server-video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def inventory(root: Path, *, exclude_index: bool = False) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden: {path}")
        if not path.is_file() or (exclude_index and path == root / "artifact_index.json"):
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    return rows, aggregate.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    streams = value.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream: {path}")
    stream = streams[0]
    observed = {
        "codec_name": stream.get("codec_name"),
        "width": int(stream.get("width", -1)),
        "height": int(stream.get("height", -1)),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": int(stream.get("nb_frames", -1)),
        "duration": float(value.get("format", {}).get("duration", "nan")),
    }
    expected = {
        "codec_name": "h264", "width": 640, "height": 352,
        "avg_frame_rate": "5/1", "nb_frames": 9,
    }
    bad = {key: (observed[key], target) for key, target in expected.items() if observed[key] != target}
    if bad or not 1.79 <= observed["duration"] <= 1.81:
        raise RuntimeError(f"invalid decoded video {path}: {observed}; mismatches={bad}")
    return observed


def require_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch {path}: {actual} != {expected}")


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_root.resolve()
    core = args.core_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if output == core or output.is_relative_to(core) or core.is_relative_to(output):
        raise RuntimeError("output and core roots must be disjoint")
    require_sha(args.exporter, EXPORTER_SHA256)
    require_sha(args.runner, RUNNER_SHA256)
    require_sha(args.manifest, MANIFEST_SHA256)
    require_sha(core / "run_inventory.json", CORE_INVENTORY_SHA256)
    manifest = load_json(args.manifest)
    states = manifest.get("states")
    if not isinstance(states, list) or len(states) != 30:
        raise RuntimeError("manifest must contain exactly 30 states")
    state_ids = [str(record["state_id"]) for record in states]
    if len(set(state_ids)) != 30:
        raise RuntimeError("manifest state IDs are not unique")
    core_rows_before, core_aggregate_before = inventory(core)
    server_videos_before = {
        path.resolve() for path in args.server_video_root.glob("*.mp4")
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    records: list[dict[str, Any]] = []
    for state_index, (record, state_id) in enumerate(zip(states, state_ids, strict=True)):
        if str(record.get("admission", "evaluation")) not in ("evaluation", ""):
            raise RuntimeError(f"unexpected state admission: {state_id}")
        core_state = core / "states" / state_id
        core_result = load_json(core_state / "result.json")
        core_traces = load_json(core_state / "trace_inventory.json")
        trace_by_seed = {int(item["branch_seed"]): item for item in core_traces["traces"]}
        if set(trace_by_seed) != set(BRANCH_SEEDS):
            raise RuntimeError(f"frozen trace seed mismatch: {state_id}")
        state_root = staging / "states" / state_id
        state_root.mkdir(parents=True)
        selection = state_root / "selection_rule.json"
        write_json(
            selection,
            {
                "schema": "dreamzero-exhaustive-postanalysis-media-selection-v1",
                "selected_state_id": state_id,
                "state_index": state_index,
                "selection": "all frozen manifest states in manifest order",
                "uses_outcomes_or_visuals": False,
            },
        )
        media_root = state_root / "media"
        command = [
            str(args.python), str(args.exporter),
            "--runner", str(args.runner),
            "--manifest", str(args.manifest),
            "--manifest-sha256", MANIFEST_SHA256,
            "--data-root", str(args.data_root),
            "--metadata-root", str(args.metadata_root),
            "--core-root", str(core),
            "--selection-rule", str(selection),
            "--server-video-root", str(args.server_video_root),
            "--output-root", str(media_root),
            "--host", args.host, "--port", str(args.port),
        ]
        call_started = time.time()
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        (state_root / "exporter_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (state_root / "exporter_stderr.log").write_text(completed.stderr, encoding="utf-8")
        receipt = load_json(media_root / "receipt.json")
        if receipt.get("status") != "complete" or receipt.get("state_id") != state_id:
            raise RuntimeError(f"invalid exporter receipt: {state_id}")
        if tuple(int(item["seed"]) for item in receipt["records"]) != BRANCH_SEEDS:
            raise RuntimeError(f"exported seeds differ: {state_id}")
        state_records: list[dict[str, Any]] = []
        trace_root = state_root / "media_traces"
        for item in receipt["records"]:
            seed = int(item["seed"])
            frozen_trace = trace_by_seed[seed]
            rerun_trace = trace_root / f"native_seed_{seed}.pt"
            rerun_trace_sha = sha256_file(rerun_trace)
            if rerun_trace_sha != frozen_trace["sha256"] or item["trace_sha256"] != frozen_trace["sha256"]:
                raise RuntimeError(f"rerun trace differs bytewise from core: {state_id}/{seed}")
            video_path = media_root / str(item["video"])
            if sha256_file(video_path) != item["video_sha256"]:
                raise RuntimeError(f"video hash mismatch: {state_id}/{seed}")
            if item.get("action_bit_exact_to_frozen_core") is not True:
                raise RuntimeError(f"action mismatch: {state_id}/{seed}")
            state_records.append(
                {
                    "seed": seed,
                    "action_sha256": item["action_sha256"],
                    "frozen_core_action_sha256": item["frozen_core_action_sha256"],
                    "action_bit_exact_to_frozen_core": True,
                    "rerun_trace_sha256": rerun_trace_sha,
                    "frozen_core_trace_sha256": frozen_trace["sha256"],
                    "frozen_video_trace_sha256": frozen_trace["video_trace_sha256"],
                    "video_relative_path": video_path.relative_to(staging).as_posix(),
                    "video_sha256": item["video_sha256"],
                    "video_probe": ffprobe(video_path),
                }
            )
        records.append(
            {
                "state_id": state_id,
                "state_index": state_index,
                "core_result_sha256": sha256_file(core_state / "result.json"),
                "core_actions_sha256": core_result["artifacts"]["actions_npz"]["sha256"],
                "input_fingerprint": receipt["input_fingerprint"],
                "duration_seconds": time.time() - call_started,
                "branches": state_records,
            }
        )
        print(f"complete {state_index + 1}/30 {state_id}", flush=True)

    core_rows_after, core_aggregate_after = inventory(core)
    if core_rows_after != core_rows_before or core_aggregate_after != core_aggregate_before:
        raise RuntimeError("frozen core tree changed during media export")
    server_videos_after = {
        path.resolve() for path in args.server_video_root.glob("*.mp4")
    }
    created_server_videos = sorted(str(path) for path in server_videos_after - server_videos_before)
    if len(created_server_videos) != 120:
        raise RuntimeError(f"expected 120 new server MP4s, got {len(created_server_videos)}")
    receipt = {
        "schema": "dreamzero-exhaustive-postanalysis-native-media-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "exhaustive descriptive post-analysis media; no outcome or appearance selection",
        "selection": "all 30 frozen manifest states x all four registered native seeds, in manifest order",
        "state_count": len(records),
        "branch_count_per_state": 4,
        "video_count": sum(len(item["branches"]) for item in records),
        "all_actions_bit_exact_to_frozen_core": True,
        "all_rerun_traces_byte_exact_to_frozen_core": True,
        "manifest_sha256": MANIFEST_SHA256,
        "core_run_inventory_sha256": CORE_INVENTORY_SHA256,
        "core_tree_aggregate_sha256": core_aggregate_before,
        "runner_sha256": RUNNER_SHA256,
        "representative_exporter_sha256": EXPORTER_SHA256,
        "umbrella_exporter_sha256": sha256_file(Path(__file__).resolve()),
        "server_video_root": str(args.server_video_root.resolve()),
        "new_server_video_count": len(created_server_videos),
        "duration_seconds": time.time() - started,
        "records": records,
    }
    write_json(staging / "receipt.json", receipt)
    (staging / "export_dreamzero_all_native_videos.py").write_bytes(Path(__file__).read_bytes())
    rows, aggregate = inventory(staging, exclude_index=True)
    write_json(
        staging / "artifact_index.json",
        {
            "schema": "dreamzero-exhaustive-postanalysis-media-index-v1",
            "status": "complete_read_only",
            "file_count_excluding_index": len(rows),
            "tree_aggregate_sha256": aggregate,
            "files": rows,
        },
    )
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        path.chmod(0o555)
    staging.chmod(0o555)
    frozen_rows, frozen_aggregate = inventory(staging, exclude_index=True)
    if frozen_rows != rows or frozen_aggregate != aggregate:
        raise RuntimeError("media package changed during freeze")
    if any(stat.S_IMODE(p.stat().st_mode) != 0o444 for p in staging.rglob("*") if p.is_file()):
        raise RuntimeError("not all package files are 0444")
    os.replace(staging, output)
    print(json.dumps({"status": "complete", "output_root": str(output), "video_count": 120, "tree_aggregate_sha256": aggregate}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
