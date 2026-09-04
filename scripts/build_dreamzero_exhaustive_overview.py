#!/usr/bin/env python3
"""Build selection-neutral overview media from the frozen DreamZero all-120 export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EXPECTED_RAW_RECEIPT_SHA256 = "89c7f4742b42698d2a2cb122f63c5792d6e53da070280c019ad18a1fb3acacdb"
EXPECTED_RAW_INDEX_SHA256 = "8907b7f854f7ea5217cfdce842bb56d2cc86649fd2fa53eed11480966f5f5aa6"
EXPECTED_RAW_TREE_AGGREGATE = "e6e34e5b60ae6dd3233cd95e614a8911227d4c6f869222d6195dd0ebcf3f56c5"
EXPECTED_SEEDS = (211, 223, 227, 229)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def probe(path: Path) -> dict[str, object]:
    output = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
            "-of", "json", str(path),
        ],
        text=True,
    )
    parsed = json.loads(output)
    stream = parsed["streams"][0]
    return {
        "codec_name": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "nb_frames": int(stream["nb_frames"]),
        "duration": float(parsed["format"]["duration"]),
    }


def quote_drawtext(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"Unexpected symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--script-path", type=Path, required=True)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    script_path = args.script_path.resolve()
    if output_root.exists() or output_root.with_name(output_root.name + ".staging").exists():
        raise RuntimeError("Output and staging roots must both be absent")

    raw_receipt_path = raw_root / "receipt.json"
    raw_index_path = raw_root / "artifact_index.json"
    if sha256_file(raw_receipt_path) != EXPECTED_RAW_RECEIPT_SHA256:
        raise RuntimeError("Unexpected raw receipt hash")
    if sha256_file(raw_index_path) != EXPECTED_RAW_INDEX_SHA256:
        raise RuntimeError("Unexpected raw index hash")
    raw_receipt = json.loads(raw_receipt_path.read_text())
    if raw_receipt.get("status") != "complete" or raw_receipt.get("state_count") != 30:
        raise RuntimeError("Raw export is not the complete 30-state cohort")
    if raw_receipt.get("video_count") != 120 or len(raw_receipt.get("records", [])) != 30:
        raise RuntimeError("Raw export is not the complete 120-video cohort")
    if not raw_receipt.get("all_actions_bit_exact_to_frozen_core"):
        raise RuntimeError("Raw export action parity gate failed")
    if not raw_receipt.get("all_rerun_traces_byte_exact_to_frozen_core"):
        raise RuntimeError("Raw export trace parity gate failed")

    records: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for expected_index, state in enumerate(raw_receipt["records"]):
        if state["state_index"] != expected_index:
            raise RuntimeError("Raw records are not in manifest order")
        branches = state["branches"]
        if tuple(branch["seed"] for branch in branches) != EXPECTED_SEEDS:
            raise RuntimeError(f"Unexpected seed order in state {expected_index}")
        for branch_index, branch in enumerate(branches):
            key = (state["state_id"], branch["seed"])
            if key in seen:
                raise RuntimeError(f"Duplicate branch {key}")
            seen.add(key)
            video = raw_root / branch["video_relative_path"]
            if sha256_file(video) != branch["video_sha256"]:
                raise RuntimeError(f"Raw video hash mismatch: {video}")
            records.append(
                {
                    "state_index": expected_index,
                    "state_id": state["state_id"],
                    "branch_index": branch_index,
                    "seed": branch["seed"],
                    "video": video,
                    "video_relative_path": branch["video_relative_path"],
                    "video_sha256": branch["video_sha256"],
                }
            )
    if len(records) != 120 or len(seen) != 120:
        raise RuntimeError("Expected exactly 120 unique manifest-ordered branches")

    staging = output_root.with_name(output_root.name + ".staging")
    staging.mkdir(parents=True)
    temp_root = Path(tempfile.mkdtemp(prefix="dreamzero-overview-"))
    try:
        script_copy = staging / script_path.name
        shutil.copy2(script_path, script_copy)
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        font_bold_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        font = ImageFont.truetype(str(font_path), 18)
        font_small = ImageFont.truetype(str(font_path), 13)
        font_bold = ImageFont.truetype(str(font_bold_path), 16)

        terminal_paths: list[Path] = []
        state_panel_paths: list[Path] = []
        mapping_path = staging / "manifest_order.csv"
        with mapping_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["state_index", "state_id", "branch_index", "seed", "video_relative_path", "video_sha256"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow({key: record[key] for key in writer.fieldnames})

        for record in records:
            frame_path = temp_root / f"terminal_{record['state_index']:02d}_{record['branch_index']}.png"
            run([
                "ffmpeg", "-v", "error", "-i", str(record["video"]),
                "-vf", "select=eq(n\\,8)", "-vsync", "0", "-frames:v", "1", str(frame_path),
            ])
            terminal_paths.append(frame_path)

        tile_width, tile_height, label_height = 320, 176, 38
        sheet = Image.new("RGB", (tile_width * 4, (tile_height + label_height) * 30), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, (record, frame_path) in enumerate(zip(records, terminal_paths, strict=True)):
            row = int(record["state_index"])
            col = int(record["branch_index"])
            image = Image.open(frame_path).convert("RGB").resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            x, y = col * tile_width, row * (tile_height + label_height)
            sheet.paste(image, (x, y + label_height))
            label = f"state {row + 1:02d} · seed {record['seed']}"
            draw.text((x + 7, y + 8), label, fill="black", font=font_bold)
        contact_path = staging / "all_30x4_terminal_contact_sheet.png"
        sheet.save(contact_path, format="PNG", optimize=True)

        for state_index in range(30):
            state_records = records[state_index * 4 : (state_index + 1) * 4]
            panel_path = temp_root / f"state_{state_index:02d}.mp4"
            inputs: list[str] = []
            filters: list[str] = []
            for branch_index, record in enumerate(state_records):
                inputs.extend(["-i", str(record["video"])])
                state_id = quote_drawtext(str(record["state_id"]))
                label = quote_drawtext(
                    f"state {state_index + 1:02d}/30 | seed {record['seed']} | {state_id}"
                )
                filters.append(
                    f"[{branch_index}:v]scale=640:352,"
                    f"drawtext=fontfile={font_path}:text='{label}':x=12:y=12:"
                    "fontsize=18:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6"
                    f"[v{branch_index}]"
                )
            filters.extend(["[v0][v1]hstack=inputs=2[top]", "[v2][v3]hstack=inputs=2[bottom]", "[top][bottom]vstack=inputs=2[out]"])
            run([
                "ffmpeg", "-v", "error", *inputs,
                "-filter_complex", ";".join(filters), "-map", "[out]",
                "-an", "-r", "5", "-frames:v", "9", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18", str(panel_path),
            ])
            state_panel_paths.append(panel_path)

        concat_list = temp_root / "concat.txt"
        concat_list.write_text("".join(f"file '{path}'\n" for path in state_panel_paths))
        overview_path = staging / "all_30_states_4_branches_overview.mp4"
        run([
            "ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", "-movflags", "+faststart", str(overview_path),
        ])
        overview_probe = probe(overview_path)
        if overview_probe["width"] != 1280 or overview_probe["height"] != 704:
            raise RuntimeError(f"Unexpected overview dimensions: {overview_probe}")
        if overview_probe["nb_frames"] != 270 or overview_probe["avg_frame_rate"] != "5/1":
            raise RuntimeError(f"Unexpected overview timing: {overview_probe}")
        with Image.open(contact_path) as check:
            if check.size != (1280, 6420):
                raise RuntimeError(f"Unexpected contact-sheet size: {check.size}")

        # Re-validate all source videos and the immutable umbrella identifiers after derivation.
        if sha256_file(raw_receipt_path) != EXPECTED_RAW_RECEIPT_SHA256:
            raise RuntimeError("Raw receipt changed during derivation")
        if sha256_file(raw_index_path) != EXPECTED_RAW_INDEX_SHA256:
            raise RuntimeError("Raw index changed during derivation")
        for record in records:
            if sha256_file(record["video"]) != record["video_sha256"]:
                raise RuntimeError(f"Raw video changed during derivation: {record['video']}")

        receipt = {
            "schema": "dreamzero-exhaustive-derived-media-v1",
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "selection-neutral descriptive media; no outcome or appearance selection",
            "selection": "all 30 frozen manifest states x all four registered branches, in raw receipt order",
            "source_raw_root": str(raw_root),
            "source_raw_receipt_sha256": EXPECTED_RAW_RECEIPT_SHA256,
            "source_raw_artifact_index_sha256": EXPECTED_RAW_INDEX_SHA256,
            "source_raw_tree_aggregate_sha256": EXPECTED_RAW_TREE_AGGREGATE,
            "source_state_count": 30,
            "source_video_count": 120,
            "source_videos_rehashed_before_and_after": True,
            "overview": {
                "relative_path": overview_path.name,
                "sha256": sha256_file(overview_path),
                "probe": overview_probe,
                "state_order": "raw receipt / frozen manifest order",
                "branch_order": list(EXPECTED_SEEDS),
            },
            "terminal_contact_sheet": {
                "relative_path": contact_path.name,
                "sha256": sha256_file(contact_path),
                "width": 1280,
                "height": 6420,
                "rows": 30,
                "columns": 4,
                "frame_index_zero_based": 8,
            },
            "mapping_csv_sha256": sha256_file(mapping_path),
            "script_sha256": sha256_file(script_copy),
        }
        receipt_path = staging / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

        rows = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                rows.append({
                    "relative_path": str(path.relative_to(staging)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        index = {
            "schema": "dreamzero-exhaustive-derived-media-index-v1",
            "artifact_count_excluding_index": len(rows),
            "artifacts": rows,
        }
        index_path = staging / "artifact_index.json"
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        freeze_tree(staging)
        os.replace(staging, output_root)
        print(json.dumps({
            "output_root": str(output_root),
            "receipt_sha256": sha256_file(output_root / "receipt.json"),
            "artifact_index_sha256": sha256_file(output_root / "artifact_index.json"),
            "overview_sha256": receipt["overview"]["sha256"],
            "contact_sheet_sha256": receipt["terminal_contact_sheet"]["sha256"],
        }, sort_keys=True))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
