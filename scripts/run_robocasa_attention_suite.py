"""Run the registered future-to-action attention test on RoboCasa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from imagined_future.attention_ablation import restrict_future_to_action_attention
from imagined_future.cosmos_config import deterministic_tokenizer_enabled, robocasa_policy_config
from imagined_future.frames import LatentFrameGroups
from imagined_future.metrics import donor_steering
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=27)
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
        raise KeyError(f"unit is absent from RoboCasa manifest: {args.unit_id}")
    branch_dir = Path(unit["branch_run_dir"])
    branch = np.load(branch_dir / "branches.npz", allow_pickle=False)
    branch_summary = json.loads((branch_dir / "summary.json").read_text())
    cfg = robocasa_policy_config(branch_summary["task_name"], unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)
    observation = {
        "primary_image": branch["current_primary_image"],
        "secondary_image": branch["current_secondary_image"],
        "wrist_image": branch["current_wrist_image"],
        "proprio": branch["current_proprio"],
    }
    left, right = (int(index) for index in unit["primary_pair"])
    directions = {"forward": (left, right), "reverse": (right, left)}
    rows = []
    actions = {}
    for direction, (recipient_index, donor_index) in directions.items():
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
        baseline_action = np.asarray(baseline["actions"])[: cfg.num_open_loop_steps]
        recipient_reference = np.asarray(branch["normalized_branch_actions"][recipient_index])
        donor_reference = np.asarray(branch["normalized_branch_actions"][donor_index])
        error = float(np.max(np.abs(baseline_action.astype(np.float64) - recipient_reference)))
        if error != 0.0:
            raise RuntimeError(f"{direction} baseline does not exactly reproduce recipient: {error}")
        groups = LatentFrameGroups.from_batch(baseline["data_batch"])
        if len(groups.current) != len(groups.future):
            raise RuntimeError("registered equal-count current-key control is not equal count")
        actions[f"normalized_{direction}_baseline"] = baseline_action
        actions[f"environment_{direction}_baseline"] = unnormalize_actions(
            baseline_action.copy(), dataset_stats
        )
        conditions = (
            ("block27_future_gate1", groups.future, True),
            ("block27_current_gate1", groups.current, True),
            ("block27_all_key_control", groups.future, False),
        )
        for condition, excluded_frames, exclude_keys in conditions:
            with restrict_future_to_action_attention(
                [model.net.blocks[args.layer]],
                action_frames=groups.action,
                future_frames=excluded_frames,
                exclude_future_keys=exclude_keys,
                gate=1.0,
                block_ids=[args.layer],
            ) as stats:
                result = get_action(**common)
            normalized = np.asarray(result["actions"])[: cfg.num_open_loop_steps]
            name = f"{direction}_{condition}"
            actions[f"normalized_{name}"] = normalized
            actions[f"environment_{name}"] = unnormalize_actions(normalized.copy(), dataset_stats)
            rows.append(
                {
                    "direction": direction,
                    "recipient_branch": recipient_index,
                    "donor_branch": donor_index,
                    "condition": condition,
                    "layer": args.layer,
                    "excluded_frames": list(excluded_frames),
                    "gate": 1.0,
                    "exclude_keys": exclude_keys,
                    "donor_steering": float(
                        donor_steering(
                            torch.from_numpy(normalized.astype(np.float64)).unsqueeze(0),
                            torch.from_numpy(recipient_reference.astype(np.float64)).unsqueeze(0),
                            torch.from_numpy(donor_reference.astype(np.float64)).unsqueeze(0),
                        ).item()
                    ),
                    "action_l2_from_baseline": float(np.linalg.norm(normalized - baseline_action)),
                    "max_abs_from_baseline": float(np.max(np.abs(normalized - baseline_action))),
                    "attention_calls_by_block": stats.calls_by_block,
                    "attention_l2_by_block": stats.selected_output_l2_by_block,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "actions.npz", **actions)
    summary = {
        "scope": "frozen RoboCasa future-to-action attention replication",
        "study_manifest": str(args.manifest),
        "unit_id": args.unit_id,
        "branch_run": str(branch_dir),
        "task_name": branch_summary["task_name"],
        "episode_index": branch_summary["episode_index"],
        "prefix_chunks": branch_summary["prefix_chunks"],
        "primary_pair": [left, right],
        "layer": args.layer,
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir), "conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
