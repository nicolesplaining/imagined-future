#!/usr/bin/env python3
"""Rank same-state native rollout pairs by visible and action divergence.

This is a figure-selection utility, not a scientific estimator.  It compares
only native rollouts that begin from the same saved state, removes any residual
first-frame offset, and writes both a CSV ranking and compact inspection sheets.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FRAME_INDICES = (0, 8, 16, 24, 32)
PRIMARY_CAMERA_HEIGHT = 360


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source: str
    left_label: str
    right_label: str
    left_path: Path
    right_path: Path


def discover_candidates(gallery: Path) -> list[Candidate]:
    candidates: list[Candidate] = []

    for state_dir in sorted((gallery / "raw").glob("*")):
        recipient = state_dir / "native_recipient.npz"
        donor = state_dir / "native_donor.npz"
        if recipient.is_file() and donor.is_file():
            candidates.append(
                Candidate(
                    candidate_id=state_dir.name,
                    source="confirmatory_pair",
                    left_label="Recipient native future",
                    right_label="Donor native future",
                    left_path=recipient,
                    right_path=donor,
                )
            )

    multidonor_root = gallery / "multidonor" / "raw"
    for state_dir in sorted(multidonor_root.glob("*")):
        native = sorted(state_dir.glob("native_*.npz"))
        for left, right in combinations(native, 2):
            left_seed = left.stem.removeprefix("native_")
            right_seed = right.stem.removeprefix("native_")
            candidates.append(
                Candidate(
                    candidate_id=f"{state_dir.name}__{left_seed}_vs_{right_seed}",
                    source="multidonor_native_pair",
                    left_label=f"Native future, seed {left_seed}",
                    right_label=f"Native future, seed {right_seed}",
                    left_path=left,
                    right_path=right,
                )
            )

    if not candidates:
        raise RuntimeError(f"no native rollout pairs found beneath {gallery}")
    return candidates


def load_rollout(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        video = np.asarray(payload["video"], dtype=np.uint8)
        action = np.asarray(payload["action"], dtype=np.float32) if "action" in payload else None
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"unexpected video shape {video.shape}: {path}")
    return video, action


def primary_camera(video: np.ndarray) -> np.ndarray:
    if video.shape[1] >= PRIMARY_CAMERA_HEIGHT:
        return video[:, :PRIMARY_CAMERA_HEIGHT]
    return video


def score_candidate(candidate: Candidate) -> dict[str, float | str]:
    left_video, left_action = load_rollout(candidate.left_path)
    right_video, right_action = load_rollout(candidate.right_path)
    if left_video.shape != right_video.shape:
        raise ValueError(
            f"paired videos differ in shape: {left_video.shape} != {right_video.shape}"
        )

    left = primary_camera(left_video)[::4, ::4, ::4].astype(np.int16)
    right = primary_camera(right_video)[::4, ::4, ::4].astype(np.int16)
    direct_delta = left - right
    initial_delta = direct_delta[0]
    motion_delta = direct_delta[1:] - initial_delta[None]

    pixel_l1 = float(np.mean(np.abs(direct_delta[1:])) / 255.0)
    initial_l1 = float(np.mean(np.abs(initial_delta)) / 255.0)
    motion_l1 = float(np.mean(np.abs(motion_delta)) / 255.0)
    final_l1 = float(np.mean(np.abs(direct_delta[-1] - initial_delta)) / 255.0)

    action_rms = float("nan")
    if left_action is not None and right_action is not None and left_action.shape == right_action.shape:
        action_rms = float(np.sqrt(np.mean(np.square(left_action - right_action))))

    native_action_l2 = float("nan")
    robot_endpoint_l2 = float("nan")
    object_endpoint_l2 = float("nan")
    summary_path = (
        Path("results")
        / "cosmos3_population_confirmatory_v1"
        / candidate.candidate_id
        / "summary.json"
    )
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        native_action_l2 = float(summary.get("native_action_l2", float("nan")))
        endpoints = summary.get("native_endpoint_l2", {})
        robot_endpoint_l2 = float(endpoints.get("robot", float("nan")))
        object_endpoint_l2 = float(endpoints.get("object_position", float("nan")))

    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "left_label": candidate.left_label,
        "right_label": candidate.right_label,
        "left_path": str(candidate.left_path),
        "right_path": str(candidate.right_path),
        "pixel_l1": pixel_l1,
        "initial_l1": initial_l1,
        "motion_l1": motion_l1,
        "final_l1": final_l1,
        "action_rms": action_rms,
        "native_action_l2": native_action_l2,
        "robot_endpoint_l2": robot_endpoint_l2,
        "object_endpoint_l2": object_endpoint_l2,
    }


def minmax(rows: list[dict[str, float | str]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    lo, hi = float(values[finite].min()), float(values[finite].max())
    normalized = np.zeros_like(values)
    if hi > lo:
        normalized[finite] = (values[finite] - lo) / (hi - lo)
    return normalized


def add_composite_scores(rows: list[dict[str, float | str]]) -> None:
    motion = minmax(rows, "motion_l1")
    final = minmax(rows, "final_l1")
    action = minmax(rows, "action_rms")
    native_action = minmax(rows, "native_action_l2")
    robot_endpoint = minmax(rows, "robot_endpoint_l2")
    object_endpoint = minmax(rows, "object_endpoint_l2")
    for index, row in enumerate(rows):
        visual = 0.65 * motion[index] + 0.35 * final[index]
        row["visual_score"] = float(visual)
        row["behavior_score"] = float(0.65 * visual + 0.35 * action[index])
        has_state_metrics = np.isfinite(float(row["robot_endpoint_l2"]))
        if has_state_metrics:
            state_space = (
                0.50 * object_endpoint[index]
                + 0.30 * robot_endpoint[index]
                + 0.20 * native_action[index]
            )
            row["state_space_score"] = float(state_space)
            row["figure_score"] = float(0.50 * visual + 0.50 * state_space)
        else:
            row["state_space_score"] = float("nan")
            row["figure_score"] = float(row["behavior_score"])


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{suffix}", size)


def render_pair_sheet(
    candidate: Candidate,
    row: dict[str, float | str],
    rank: int,
    destination: Path,
) -> None:
    left_video, _ = load_rollout(candidate.left_path)
    right_video, _ = load_rollout(candidate.right_path)
    videos = (primary_camera(left_video), primary_camera(right_video))

    label_width = 250
    cell_width, cell_height = 256, 144
    header_height, row_pitch = 54, 160
    canvas = Image.new("RGB", (label_width + 5 * cell_width, header_height + 2 * row_pitch), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"#{rank}  {candidate.candidate_id}    "
        f"visual={float(row['visual_score']):.3f}  "
        f"action RMS={float(row['action_rms']):.3f}"
    )
    draw.text((10, 8), title, fill="#151515", font=font(16, bold=True))
    for column, frame_index in enumerate(FRAME_INDICES):
        draw.text(
            (label_width + column * cell_width + 8, 34),
            f"t={frame_index}",
            fill="#333333",
            font=font(12),
        )

    for row_index, (video, label) in enumerate(
        ((videos[0], candidate.left_label), (videos[1], candidate.right_label))
    ):
        y = header_height + row_index * row_pitch
        draw.text((10, y + 62), label, fill="#222222", font=font(14))
        for column, frame_index in enumerate(FRAME_INDICES):
            frame = Image.fromarray(video[frame_index], mode="RGB")
            frame = frame.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            canvas.paste(frame, (label_width + column * cell_width, y))

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)


def write_html(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    cards = []
    for row in rows:
        rank = int(row["rank"])
        image_name = f"rank_{rank:02d}__{row['candidate_id']}.png"
        cards.append(
            "<article><h2>"
            + html.escape(f"#{rank} {row['candidate_id']}")
            + "</h2><p>"
            + html.escape(
                f"visual={float(row['visual_score']):.3f}; "
                f"motion={float(row['motion_l1']):.4f}; "
                f"final={float(row['final_l1']):.4f}; "
                f"action RMS={float(row['action_rms']):.3f}; "
                f"robot endpoint={float(row['robot_endpoint_l2']):.3f}; "
                f"object endpoint={float(row['object_endpoint_l2']):.3f}"
            )
            + f'</p><a href="{html.escape(image_name)}"><img src="{html.escape(image_name)}"></a></article>'
        )
    document = """<!doctype html>
<meta charset="utf-8">
<title>Most dissimilar native futures</title>
<style>
body{font:15px Arial,sans-serif;margin:24px;background:#f3f5f8;color:#181818}
h1{margin-bottom:4px} .note{color:#555;margin-top:0} article{background:white;padding:16px;
margin:18px 0;border-radius:12px;box-shadow:0 2px 10px #0001} h2{margin:0 0 4px}
p{margin:0 0 12px;color:#444} img{display:block;width:100%;height:auto;border:1px solid #ddd}
</style>
<h1>Most dissimilar same-state native futures</h1>
<p class="note">Ranked for visual triage using motion-relative and final-frame divergence plus action and, where available, simulator robot/object endpoint separation. Large camera drift can still score highly, so inspect the shortlist before publication.</p>
""" + "\n".join(cards)
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gallery",
        type=Path,
        default=Path("output/cover_figure_candidates"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cover_figure_candidates/ranked_differences"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    candidates = discover_candidates(args.gallery)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    rows = [score_candidate(candidate) for candidate in candidates]
    add_composite_scores(rows)
    rows.sort(key=lambda row: float(row["figure_score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Rank numbers can change when new rollouts arrive. Remove only sheets made by
    # this script so stale rank-numbered images do not survive a rerun.
    for old_sheet in args.output_dir.glob("rank_*.png"):
        old_sheet.unlink()
    fieldnames = list(rows[0].keys())
    with (args.output_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected = rows[: min(args.top_k, len(rows))]
    for row in selected:
        rank = int(row["rank"])
        candidate = candidate_by_id[str(row["candidate_id"])]
        render_pair_sheet(
            candidate,
            row,
            rank,
            args.output_dir / f"rank_{rank:02d}__{candidate.candidate_id}.png",
        )
    write_html(selected, args.output_dir)
    print(f"ranked {len(rows)} same-state pairs; rendered top {len(selected)}")
    for row in selected[:10]:
        print(
            f"{int(row['rank']):2d}  {row['candidate_id']:<45} "
            f"visual={float(row['visual_score']):.3f} "
            f"motion={float(row['motion_l1']):.4f} "
            f"final={float(row['final_l1']):.4f} "
            f"action={float(row['action_rms']):.3f}"
        )


if __name__ == "__main__":
    main()
