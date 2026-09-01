"""Run the frozen future-to-action necessity matrix with one model load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from imagined_future.attention_ablation import restrict_future_to_action_attention
from imagined_future.cosmos_config import deterministic_tokenizer_enabled, libero_policy_config
from imagined_future.frames import LatentFrameGroups
from imagined_future.metrics import donor_steering
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution
from imagined_future.paired_rollouts import pixel_l1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--primary-layer", type=int, default=27)
    parser.add_argument("--secondary-layer", type=int, default=0)
    parser.add_argument("--gates", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--random-frame-seed", type=int, default=431)
    parser.add_argument("--directions", choices=("forward", "reverse"), nargs="+", default=["forward", "reverse"])
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {args.output_dir}")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
        unnormalize_actions,
    )

    manifest = json.loads(args.manifest.read_text())
    unit = next((item for item in manifest["units"] if item["unit_id"] == args.unit_id), None)
    if unit is None:
        raise KeyError(f"unit is absent from manifest: {args.unit_id}")
    branch_dir = Path(unit["branch_run_dir"])
    branch = np.load(branch_dir / "branches.npz", allow_pickle=False)
    branch_summary = json.loads((branch_dir / "summary.json").read_text())
    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)

    observation = {
        "primary_image": branch["current_primary_image"],
        "wrist_image": branch["current_wrist_image"],
        "proprio": branch["current_proprio"],
    }
    left, right = (int(index) for index in unit["primary_pair"])
    direction_specs = {"forward": (left, right), "reverse": (right, left)}
    action_arrays: dict[str, np.ndarray] = {}
    rows = []

    for direction_name in args.directions:
        recipient_index, donor_index = direction_specs[direction_name]
        model_seed = int(branch["branch_seeds"][recipient_index])
        common = dict(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            obs=observation,
            task_label_or_embedding=branch_summary["task_description"],
            seed=model_seed,
            randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
        )
        baseline = get_action(**common)
        baseline_action = np.asarray(baseline["actions"])
        recipient_reference = np.asarray(branch["normalized_branch_actions"][recipient_index])
        donor_reference = np.asarray(branch["normalized_branch_actions"][donor_index])
        baseline_error = float(np.max(np.abs(baseline_action.astype(np.float64) - recipient_reference)))
        if baseline_error != 0.0:
            raise RuntimeError(f"{direction_name} baseline does not exactly reproduce recipient: {baseline_error}")
        groups = LatentFrameGroups.from_batch(baseline["data_batch"])
        frame_count = int(baseline["orig_clean_latent_frames"].shape[2])
        random_candidates = sorted(set(range(frame_count)) - set(groups.action) - set(groups.future))
        if len(random_candidates) < len(groups.future):
            raise ValueError("not enough non-action, non-future frames for equal-count random control")
        generator = np.random.default_rng(args.random_frame_seed)
        random_frames = tuple(
            sorted(generator.choice(random_candidates, len(groups.future), replace=False).tolist())
        )
        condition_specs = [
            (f"block{args.primary_layer}_future_gate{gate:g}", args.primary_layer, groups.future, gate, True)
            for gate in args.gates
        ]
        condition_specs.extend(
            [
                (f"block{args.primary_layer}_current_gate1", args.primary_layer, groups.current, 1.0, True),
                (f"block{args.primary_layer}_random_gate1", args.primary_layer, random_frames, 1.0, True),
                (f"block{args.secondary_layer}_future_gate1", args.secondary_layer, groups.future, 1.0, True),
                (f"block{args.primary_layer}_all_key_control", args.primary_layer, groups.future, 1.0, False),
                (f"block{args.secondary_layer}_all_key_control", args.secondary_layer, groups.future, 1.0, False),
            ]
        )
        baseline_name = f"{direction_name}_baseline"
        action_arrays[f"normalized_{baseline_name}"] = baseline_action
        action_arrays[f"environment_{baseline_name}"] = unnormalize_actions(baseline_action.copy(), dataset_stats)

        for condition_name, layer, excluded_frames, gate, exclude_keys in condition_specs:
            with restrict_future_to_action_attention(
                [model.net.blocks[layer]],
                action_frames=groups.action,
                future_frames=excluded_frames,
                exclude_future_keys=exclude_keys,
                gate=gate,
                block_ids=[layer],
            ) as stats:
                result = get_action(**common)
            normalized_action = np.asarray(result["actions"])
            name = f"{direction_name}_{condition_name}"
            action_arrays[f"normalized_{name}"] = normalized_action
            action_arrays[f"environment_{name}"] = unnormalize_actions(normalized_action.copy(), dataset_stats)
            steering = float(
                donor_steering(
                    torch.from_numpy(normalized_action.astype(np.float64)).unsqueeze(0),
                    torch.from_numpy(recipient_reference.astype(np.float64)).unsqueeze(0),
                    torch.from_numpy(donor_reference.astype(np.float64)).unsqueeze(0),
                ).item()
            )
            rows.append(
                {
                    "direction": direction_name,
                    "recipient_branch": recipient_index,
                    "donor_branch": donor_index,
                    "recipient_seed": model_seed,
                    "donor_seed": int(branch["branch_seeds"][donor_index]),
                    "condition": condition_name,
                    "layer": layer,
                    "excluded_frames": list(excluded_frames),
                    "gate": gate,
                    "exclude_keys": exclude_keys,
                    "donor_steering": steering,
                    "action_l2_from_baseline": float(np.linalg.norm(normalized_action - baseline_action)),
                    "max_abs_from_baseline": float(np.max(np.abs(normalized_action - baseline_action))),
                    "future_primary_l1_from_baseline": pixel_l1(
                        result["future_image_predictions"]["future_image"],
                        baseline["future_image_predictions"]["future_image"],
                    ),
                    "attention_calls_by_block": stats.calls_by_block,
                    "attention_l2_by_block": stats.selected_output_l2_by_block,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "actions.npz", **action_arrays)
    summary = {
        "scope": "frozen confirmatory attention necessity suite",
        "study_manifest": str(args.manifest),
        "unit_id": args.unit_id,
        "branch_run": str(branch_dir),
        "branch_state_digest": branch_summary["branch_state_digest"],
        "task_id": int(branch_summary["task_id"]),
        "initial_state_index": int(branch_summary["initial_state_index"]),
        "prefix_chunks": int(branch_summary["prefix_chunks"]),
        "primary_pair": [left, right],
        "primary_layer": args.primary_layer,
        "secondary_layer": args.secondary_layer,
        "gates": args.gates,
        "random_frame_seed": args.random_frame_seed,
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir), "conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
