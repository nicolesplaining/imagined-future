"""Block future-token keys from action queries throughout Cosmos Policy's DiT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagined_future.attention_ablation import restrict_future_to_action_attention
from imagined_future.cosmos_config import deterministic_tokenizer_enabled, libero_policy_config
from imagined_future.frames import LatentFrameGroups
from imagined_future.metrics import donor_steering
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution
from imagined_future.paired_rollouts import pixel_l1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--recipient-branch", type=int, required=True)
    parser.add_argument("--donor-branch", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--excluded-key-group", choices=("future", "current"), default="future")
    args = parser.parse_args()

    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {args.output_dir}")

    import torch

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
        unnormalize_actions,
    )

    branch = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    branch_summary = json.loads((args.branch_run_dir / "summary.json").read_text())
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
    common = dict(
        cfg=cfg,
        model=model,
        dataset_stats=dataset_stats,
        obs=observation,
        task_label_or_embedding=branch_summary["task_description"],
        seed=args.model_seed,
        randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=True,
    )
    baseline = get_action(**common)
    groups = LatentFrameGroups.from_batch(baseline["data_batch"])
    layer_ids = args.layers or list(range(len(model.net.blocks)))
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("layers must be unique")
    if any(index < 0 or index >= len(model.net.blocks) for index in layer_ids):
        raise IndexError(f"layers must be within 0..{len(model.net.blocks) - 1}")
    selected_blocks = [model.net.blocks[index] for index in layer_ids]
    excluded_frames = groups.future if args.excluded_key_group == "future" else groups.current
    if args.excluded_key_group == "current" and len(excluded_frames) != len(groups.future):
        raise ValueError(
            "equal-count current-key control requires current and future groups to have the same size"
        )
    recipient_reference = branch["normalized_branch_actions"][args.recipient_branch]
    donor_reference = branch["normalized_branch_actions"][args.donor_branch]
    baseline_action = np.asarray(baseline["actions"])
    baseline_reference_error = float(
        np.max(np.abs(baseline_action.astype(np.float64) - recipient_reference.astype(np.float64)))
    )
    if baseline_reference_error != 0.0:
        raise RuntimeError(f"baseline does not exactly reproduce recipient: {baseline_reference_error}")

    with restrict_future_to_action_attention(
        selected_blocks,
        action_frames=groups.action,
        future_frames=excluded_frames,
        exclude_future_keys=False,
        block_ids=layer_ids,
    ) as control_stats:
        all_key_control = get_action(**common)
    with restrict_future_to_action_attention(
        selected_blocks,
        action_frames=groups.action,
        future_frames=excluded_frames,
        exclude_future_keys=True,
        block_ids=layer_ids,
    ) as blocked_stats:
        future_blocked = get_action(**common)

    normalized_actions = {
        "baseline": baseline_action,
        "all_key_control": np.asarray(all_key_control["actions"]),
        "future_blocked": np.asarray(future_blocked["actions"]),
    }
    environment_actions = {
        name: unnormalize_actions(action.copy(), dataset_stats)
        for name, action in normalized_actions.items()
    }
    recipient_tensor = torch.from_numpy(recipient_reference.astype(np.float64)).unsqueeze(0)
    donor_tensor = torch.from_numpy(donor_reference.astype(np.float64)).unsqueeze(0)
    steering = {
        name: float(
            donor_steering(
                torch.from_numpy(action.astype(np.float64)).unsqueeze(0),
                recipient_tensor,
                donor_tensor,
            ).item()
        )
        for name, action in normalized_actions.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "actions.npz",
        **{f"normalized_{name}": value for name, value in normalized_actions.items()},
        **{f"environment_{name}": value for name, value in environment_actions.items()},
    )
    control_action_error = float(np.max(np.abs(normalized_actions["all_key_control"] - baseline_action)))
    summary = {
        "scope": "future-to-action self-attention necessity ablation"
        if args.excluded_key_group == "future"
        else "equal-count current-key attention ablation control",
        "branch_run": str(args.branch_run_dir),
        "branch_state_digest": branch_summary["branch_state_digest"],
        "task_id": branch_summary["task_id"],
        "initial_state_index": branch_summary["initial_state_index"],
        "recipient_branch": args.recipient_branch,
        "recipient_seed": int(branch["branch_seeds"][args.recipient_branch]),
        "donor_branch": args.donor_branch,
        "donor_seed": int(branch["branch_seeds"][args.donor_branch]),
        "model_seed": args.model_seed,
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "action_frame_indices": list(groups.action),
        "future_frame_indices": list(groups.future),
        "current_frame_indices": list(groups.current),
        "excluded_key_group": args.excluded_key_group,
        "excluded_frame_indices": list(excluded_frames),
        "ablated_layers": layer_ids,
        "baseline_reference_max_abs_error": baseline_reference_error,
        "all_key_control_action_max_abs_error": control_action_error,
        "future_blocked_action_l2_from_baseline": float(
            np.linalg.norm(normalized_actions["future_blocked"] - baseline_action)
        ),
        "donor_steering": steering,
        "primary_future_pixel_l1": {
            "all_key_control_from_baseline": pixel_l1(
                all_key_control["future_image_predictions"]["future_image"],
                baseline["future_image_predictions"]["future_image"],
            ),
            "future_blocked_from_baseline": pixel_l1(
                future_blocked["future_image_predictions"]["future_image"],
                baseline["future_image_predictions"]["future_image"],
            ),
        },
        "predicted_values": {
            "baseline": float(baseline["value_prediction"]),
            "all_key_control": float(all_key_control["value_prediction"]),
            "future_blocked": float(future_blocked["value_prediction"]),
        },
        "attention_output_diagnostics": {
            "all_key_control": {
                "calls_by_block": control_stats.calls_by_block,
                "max_abs_by_block": control_stats.selected_output_max_abs_by_block,
            },
            "future_blocked": {
                "calls_by_block": blocked_stats.calls_by_block,
                "l2_by_block": blocked_stats.selected_output_l2_by_block,
            },
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
