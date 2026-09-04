#!/usr/bin/env python3
"""Render a non-cherry-picked DreamZero state selected by a frozen median rule."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import run_dreamzero_future_transplants as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--core-result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(args.output_root)
    metrics = list(csv.DictReader(args.state_metrics.open(newline="")))
    if len(metrics) != 30 or len({row["state_id"] for row in metrics}) != 30:
        raise ValueError("representative selection requires the complete 30-state analysis")
    values = np.asarray([float(row["distance_reduction"]) for row in metrics])
    median = float(np.median(values))
    ranked = sorted(
        metrics,
        key=lambda row: (abs(float(row["distance_reduction"]) - median), row["state_id"]),
    )
    selected = ranked[0]
    state_id = selected["state_id"]

    manifest, resources, data_root, frozen = core.validate_manifest_and_receipt(args)
    manifest_sha = args.expected_manifest_sha256
    state_by_id = {str(row["state_id"]): row for row in manifest["states"]}
    if set(state_by_id) != {row["state_id"] for row in metrics}:
        raise ValueError("analysis/manifest state cohorts differ")
    receipt = core.load_json(data_root / "download_receipt.json")
    receipt_by_id = {str(row["resource_id"]): row for row in receipt["resources"]}
    inputs = core.build_frozen_input(
        state_by_id[state_id],
        resource_by_id=resources,
        receipt_by_id=receipt_by_id,
        data_root=data_root,
        modality=frozen["modality"],
    )
    state_dir = args.core_result_root / "states" / state_id
    result_path = state_dir / "result.json"
    core.verify_sha_sidecar(result_path)
    result = core.load_json(result_path)
    arrays_path = state_dir / result["artifacts"]["actions_npz"]["relative_path"]
    core.verify_sha_sidecar(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        seeds = np.asarray(archive["branch_seeds"], dtype=np.int64)
        native = np.asarray(archive["native_actions"], dtype=np.float64)
        replay = np.asarray(archive["replay_actions"], dtype=np.float64)
    if tuple(seeds.tolist()) != tuple(core.BRANCH_SEEDS):
        raise ValueError("branch seed order differs")

    native_flat = native.reshape(4, -1)
    projections = np.full((4, 4), np.nan, dtype=np.float64)
    for recipient in range(4):
        for source in range(4):
            if recipient == source:
                continue
            axis = native_flat[source] - native_flat[recipient]
            displacement = replay[recipient, source].reshape(-1) - native_flat[recipient]
            projections[recipient, source] = float(np.dot(displacement, axis) / np.dot(axis, axis))
    pairwise = np.linalg.norm(native_flat[:, None, :] - native_flat[None, :, :], axis=-1)

    trajectories = native.reshape(4 * native.shape[1], -1)
    centered = trajectories - trajectories.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pca = (centered @ vt[:2].T).reshape(4, native.shape[1], 2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(11.2, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.15))
    camera_keys = (
        ("observation/exterior_image_0_left", "Exterior camera 1"),
        ("observation/exterior_image_1_left", "Exterior camera 2"),
        ("observation/wrist_image_left", "Wrist camera"),
    )
    for column, (key, label) in enumerate(camera_keys):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(inputs.request[key])
        ax.set_title(label)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, 0])
    projection_cmap = plt.get_cmap("viridis").copy()
    projection_cmap.set_bad("#d9d9d9")
    image = ax.imshow(projections, vmin=0, vmax=1, cmap=projection_cmap)
    ax.set_title("Future-source action projection")
    ax.set_xlabel("Future source seed")
    ax.set_ylabel("Recipient action-noise seed")
    ax.set_xticks(range(4), seeds)
    ax.set_yticks(range(4), seeds)
    for row in range(4):
        for column in range(4):
            value = projections[row, column]
            ax.text(
                column,
                row,
                "self" if not np.isfinite(value) else f"{value:.2f}",
                ha="center",
                va="center",
                color="black" if not np.isfinite(value) or value >= .55 else "white",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    ax = fig.add_subplot(grid[1, 1])
    image = ax.imshow(pairwise, cmap="magma")
    ax.set_title("Native action separation (L2)")
    ax.set_xlabel("Branch seed")
    ax.set_ylabel("Branch seed")
    ax.set_xticks(range(4), seeds)
    ax.set_yticks(range(4), seeds)
    for row in range(4):
        for column in range(4):
            ax.text(column, row, f"{pairwise[row, column]:.2f}", ha="center", va="center", color="white" if pairwise[row, column] < pairwise.max() * .55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    ax = fig.add_subplot(grid[1, 2])
    colors = plt.get_cmap("tab10").colors
    for index, seed in enumerate(seeds):
        ax.plot(pca[index, :, 0], pca[index, :, 1], marker="o", markersize=2.2, linewidth=1.4, color=colors[index], label=f"seed {seed}")
        ax.scatter(pca[index, 0, 0], pca[index, 0, 1], marker="s", s=28, color=colors[index])
        ax.scatter(pca[index, -1, 0], pca[index, -1, 1], marker="X", s=36, color=colors[index])
    ax.set_title("Native action chunks (PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Median-effect state: {state_id}\n{inputs.prompt}", fontsize=11)

    args.output_root.mkdir(parents=True, exist_ok=True)
    png = args.output_root / "representative_state.png"
    pdf = args.output_root / "representative_state.pdf"
    fig.savefig(png, dpi=220, metadata={"Software": "render_dreamzero_representative_state.py"})
    fig.savefig(pdf, metadata={"Creator": "render_dreamzero_representative_state.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    selection = {
        "schema": "dreamzero-representative-selection-v1",
        "selection_rule": "minimum absolute deviation from the across-state median primary distance_reduction; lexicographic state_id tie-break",
        "selection_rule_frozen_before_visual_inspection": True,
        "candidate_state_count": 30,
        "median_distance_reduction": median,
        "selected_state_id": state_id,
        "selected_distance_reduction": float(selected["distance_reduction"]),
        "absolute_deviation_from_median": abs(float(selected["distance_reduction"]) - median),
        "state_index": int(selected["state_index"]),
        "task_family": selected["task_family"],
        "prompt": inputs.prompt,
        "manifest_sha256": manifest_sha,
        "core_result_sha256": core.sha256_file(result_path),
        "core_actions_sha256": core.sha256_file(arrays_path),
        "rendered_content": "three exact frozen input-camera frames, native action separation, transplant projection grid, and native action PCA trajectories",
        "future_video_note": "native future traces are retained as VAE latents; this panel does not claim to decode them into RGB video",
    }
    selection_path = args.output_root / "selection_rule.json"
    core.atomic_write_json(selection_path, selection, mode=0o444)
    for path in (png, pdf, selection_path):
        core.freeze_with_sidecar(path)
    inventory = {
        "schema": "dreamzero-representative-artifacts-v1",
        "renderer_sha256": core.sha256_file(Path(__file__).resolve()),
        "artifacts": [
            {"name": path.name, "sha256": core.sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (png, pdf, selection_path)
        ],
    }
    inventory_path = args.output_root / "artifact_inventory.json"
    core.atomic_write_json(inventory_path, inventory, mode=0o444)
    core.freeze_with_sidecar(inventory_path)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
