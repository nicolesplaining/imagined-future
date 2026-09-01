"""Run the frozen semantic intervention on a RoboCasa replication unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from imagined_future.cosmos_config import deterministic_tokenizer_enabled, robocasa_policy_config
from imagined_future.frames import LatentFrameGroups
from imagined_future.interventions import (
    SemanticFutureClamp,
    norm_distance_matched_random_target,
    replace_frames,
    resample_frames,
)
from imagined_future.metrics import donor_steering
from imagined_future.model_patch import (
    defer_hf_config_checkpoint_resolution,
    transform_model_initial_noise,
    transform_model_x0_factory,
)
from imagined_future.paired_rollouts import pixel_l1, resize_uint8_image
from imagined_future.semantic_latent import encode_semantic_future


def _observation(artifact: np.lib.npyio.NpzFile, prefix: str, branch_index: int | None = None) -> dict:
    if branch_index is None:
        return {
            "primary_image": artifact[f"{prefix}primary_image"],
            "secondary_image": artifact[f"{prefix}secondary_image"],
            "wrist_image": artifact[f"{prefix}wrist_image"],
            "proprio": artifact[f"{prefix}proprio"],
        }
    return {
        "primary_image": artifact["endpoint_primary_images"][branch_index],
        "secondary_image": artifact["endpoint_secondary_images"][branch_index],
        "wrist_image": artifact["endpoint_wrist_images"][branch_index],
        "proprio": artifact["endpoint_proprios"][branch_index],
    }


def _matched_random_target(
    recipient_clean: torch.Tensor,
    donor_clean: torch.Tensor,
    frames: tuple[int, ...],
    seed: int,
) -> torch.Tensor:
    index = torch.as_tensor(frames, device=recipient_clean.device)
    recipient = torch.index_select(recipient_clean, 2, index)
    donor = torch.index_select(donor_clean, 2, index)
    matched = norm_distance_matched_random_target(recipient, donor, seed=seed)
    output = recipient_clean.clone()
    output[:, :, index, :, :] = matched
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-noise-seed", type=int, default=401)
    parser.add_argument("--gaussian-seed", type=int, default=433)
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

    current_observation = _observation(branch, "current_")
    left, right = (int(index) for index in unit["primary_pair"])
    directions = {"forward": (left, right), "reverse": (right, left)}
    action_arrays = {}
    decoded_primary = {}
    rows = []
    for direction, (recipient_index, donor_index) in directions.items():
        model_seed = int(branch["branch_seeds"][recipient_index])
        common = dict(
            cfg=cfg,
            model=model,
            dataset_stats=dataset_stats,
            task_label_or_embedding=branch_summary["task_description"],
            seed=model_seed,
            randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
        )
        baseline = get_action(obs=current_observation, **common)
        baseline_action = np.asarray(baseline["actions"])[: cfg.num_open_loop_steps]
        recipient_reference = np.asarray(branch["normalized_branch_actions"][recipient_index])
        donor_reference = np.asarray(branch["normalized_branch_actions"][donor_index])
        baseline_error = float(np.max(np.abs(baseline_action.astype(np.float64) - recipient_reference)))
        if baseline_error != 0.0:
            raise RuntimeError(f"{direction} baseline does not exactly reproduce recipient: {baseline_error}")
        target_observations = {
            "recipient": _observation(branch, "", recipient_index),
            "donor": _observation(branch, "", donor_index),
        }
        target_results = {
            name: get_action(obs=observation, **common)
            for name, observation in target_observations.items()
        }
        groups = LatentFrameGroups.from_batch(baseline["data_batch"])
        frames = tuple(groups.future)
        clean_targets = {
            name: encode_semantic_future(
                model,
                baseline["data_batch"],
                result["data_batch"],
                baseline["orig_clean_latent_frames"],
                result["proprio"],
            )
            for name, result in target_results.items()
        }
        clean_targets["gaussian"] = _matched_random_target(
            clean_targets["recipient"],
            clean_targets["donor"],
            frames,
            args.gaussian_seed + recipient_index,
        )
        shared_noise = resample_frames(
            torch.zeros_like(clean_targets["donor"]),
            frames,
            seed=args.future_noise_seed,
            standard_deviation=1.0,
        )
        baseline_name = f"{direction}_baseline"
        action_arrays[f"normalized_{baseline_name}"] = baseline_action
        action_arrays[f"environment_{baseline_name}"] = unnormalize_actions(
            baseline_action.copy(), dataset_stats
        )
        action_arrays[f"normalized_{direction}_action_transplant"] = donor_reference
        action_arrays[f"environment_{direction}_action_transplant"] = branch["branch_actions"][donor_index]

        for target_name in ("recipient", "donor", "gaussian"):
            clamp = SemanticFutureClamp(clean_targets[target_name], shared_noise, frames)

            def initial_transform(initial: torch.Tensor, batch: dict, *, _target=target_name) -> torch.Tensor:
                if LatentFrameGroups.from_batch(batch) != groups:
                    raise RuntimeError("RoboCasa latent frame layout changed during intervention")
                target_at_maximum_noise = clean_targets[_target] + float(model.sde.sigma_max) * shared_noise
                return replace_frames(initial, target_at_maximum_noise, frames)

            with transform_model_initial_noise(model, initial_transform):
                with transform_model_x0_factory(model, clamp.wrap):
                    result = get_action(obs=current_observation, **common)
            normalized = np.asarray(result["actions"])[: cfg.num_open_loop_steps]
            name = f"{direction}_n{args.future_noise_seed}_all_{target_name}"
            action_arrays[f"normalized_{name}"] = normalized
            action_arrays[f"environment_{name}"] = unnormalize_actions(normalized.copy(), dataset_stats)
            prediction = np.asarray(result["future_image_predictions"]["future_image"], dtype=np.uint8)
            decoded_primary[name] = prediction
            target_shape = prediction.shape[:2]
            rows.append(
                {
                    "direction": direction,
                    "recipient_branch": recipient_index,
                    "donor_branch": donor_index,
                    "recipient_seed": model_seed,
                    "donor_seed": int(branch["branch_seeds"][donor_index]),
                    "future_noise_seed": args.future_noise_seed,
                    "condition": f"all_{target_name}",
                    "target_kind": target_name,
                    "clamp_frames": list(frames),
                    "donor_steering": float(
                        donor_steering(
                            torch.from_numpy(normalized.astype(np.float64)).unsqueeze(0),
                            torch.from_numpy(recipient_reference.astype(np.float64)).unsqueeze(0),
                            torch.from_numpy(donor_reference.astype(np.float64)).unsqueeze(0),
                        ).item()
                    ),
                    "action_l2_from_baseline": float(np.linalg.norm(normalized - baseline_action)),
                    "decoded_primary_l1_to_recipient": pixel_l1(
                        prediction,
                        resize_uint8_image(target_observations["recipient"]["primary_image"], target_shape),
                    ),
                    "decoded_primary_l1_to_donor": pixel_l1(
                        prediction,
                        resize_uint8_image(target_observations["donor"]["primary_image"], target_shape),
                    ),
                    "denoiser_sigmas": clamp.calls,
                }
            )
        selected_index = torch.as_tensor(frames, device=clean_targets["donor"].device)
        selected = {
            name: torch.index_select(value, 2, selected_index).float()
            for name, value in clean_targets.items()
        }
        rows.append(
            {
                "direction": direction,
                "condition": "latent_control_diagnostics",
                "recipient_norm": float(torch.linalg.vector_norm(selected["recipient"]).item()),
                "donor_norm": float(torch.linalg.vector_norm(selected["donor"]).item()),
                "gaussian_norm": float(torch.linalg.vector_norm(selected["gaussian"]).item()),
                "donor_distance_from_recipient": float(
                    torch.linalg.vector_norm(selected["donor"] - selected["recipient"]).item()
                ),
                "gaussian_distance_from_recipient": float(
                    torch.linalg.vector_norm(selected["gaussian"] - selected["recipient"]).item()
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "actions.npz", **action_arrays)
    np.savez_compressed(args.output_dir / "decoded_primary.npz", **decoded_primary)
    summary = {
        "scope": "frozen RoboCasa cross-domain semantic replication",
        "study_manifest": str(args.manifest),
        "unit_id": args.unit_id,
        "branch_run": str(branch_dir),
        "branch_state_digest": branch_summary["branch_state_digest"],
        "task_name": branch_summary["task_name"],
        "episode_index": branch_summary["episode_index"],
        "prefix_chunks": branch_summary["prefix_chunks"],
        "primary_pair": [left, right],
        "directions": list(directions),
        "future_noise_seed": args.future_noise_seed,
        "gaussian_seed": args.gaussian_seed,
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_dir), "conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
