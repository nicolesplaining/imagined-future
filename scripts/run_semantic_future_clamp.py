"""Exploratory closed-vs-open semantic future clamp at one LIBERO state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imagined_future.cosmos_config import libero_policy_config
from imagined_future.frames import LatentFrameGroups
from imagined_future.interventions import SemanticFutureClamp, replace_frames, resample_frames
from imagined_future.model_patch import (
    defer_hf_config_checkpoint_resolution,
    transform_model_initial_noise,
    transform_model_x0_factory,
)
from imagined_future.semantic_latent import encode_semantic_future


def _observation(artifact: np.lib.npyio.NpzFile, *, endpoint_branch: int | None = None) -> dict:
    if endpoint_branch is None:
        return {
            "primary_image": artifact["current_primary_image"],
            "wrist_image": artifact["current_wrist_image"],
            "proprio": artifact["current_proprio"],
        }
    return {
        "primary_image": artifact["endpoint_primary_images"][endpoint_branch],
        "wrist_image": artifact["endpoint_wrist_images"][endpoint_branch],
        "proprio": artifact["endpoint_proprios"][endpoint_branch],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipient-run-dir", type=Path, required=True)
    parser.add_argument("--closed-target-run-dir", type=Path, required=True)
    parser.add_argument("--closed-target-branch", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--model-seed", type=int, default=195)
    parser.add_argument("--future-noise-seed", type=int, default=20195)
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

    recipient_artifact = np.load(args.recipient_run_dir / "branches.npz", allow_pickle=False)
    target_artifact = np.load(args.closed_target_run_dir / "branches.npz", allow_pickle=False)
    recipient_summary = json.loads((args.recipient_run_dir / "summary.json").read_text())
    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)

    common = dict(
        cfg=cfg,
        model=model,
        dataset_stats=dataset_stats,
        task_label_or_embedding=recipient_summary["task_description"],
        seed=args.model_seed,
        randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=True,
    )
    recipient_observation = _observation(recipient_artifact)
    closed_observation = _observation(target_artifact, endpoint_branch=args.closed_target_branch)
    recipient = get_action(obs=recipient_observation, **common)
    closed_target = get_action(obs=closed_observation, **common)
    groups = LatentFrameGroups.from_batch(recipient["data_batch"])
    if LatentFrameGroups.from_batch(closed_target["data_batch"]) != groups:
        raise RuntimeError("closed target uses a different latent frame layout")

    open_clean = encode_semantic_future(
        model,
        recipient["data_batch"],
        recipient["data_batch"],
        recipient["orig_clean_latent_frames"],
        recipient["proprio"],
    )
    closed_clean = encode_semantic_future(
        model,
        recipient["data_batch"],
        closed_target["data_batch"],
        recipient["orig_clean_latent_frames"],
        closed_target["proprio"],
    )
    donor_noise = resample_frames(
        torch.zeros_like(closed_clean), groups.future, seed=args.future_noise_seed, standard_deviation=1.0
    )

    def run_clamp(clean: torch.Tensor):
        clamp = SemanticFutureClamp(clean, donor_noise, groups.future)

        def initial_transform(initial: torch.Tensor, batch: dict) -> torch.Tensor:
            if LatentFrameGroups.from_batch(batch) != groups:
                raise RuntimeError("latent frame layout changed during clamp")
            donor_at_max = clean + float(model.sde.sigma_max) * donor_noise
            return replace_frames(initial, donor_at_max, groups.future)

        with transform_model_initial_noise(model, initial_transform):
            with transform_model_x0_factory(model, clamp.wrap):
                result = get_action(obs=recipient_observation, **common)
        return result, clamp

    open_result, open_clamp = run_clamp(open_clean)
    closed_result, closed_clamp = run_clamp(closed_clean)

    normalized_actions = {
        "baseline": np.asarray(recipient["actions"]),
        "open_clamp": np.asarray(open_result["actions"]),
        "closed_clamp": np.asarray(closed_result["actions"]),
    }
    environment_actions = {
        name: unnormalize_actions(action.copy(), dataset_stats) for name, action in normalized_actions.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, result in (("baseline", recipient), ("open_clamp", open_result), ("closed_clamp", closed_result)):
        for modality, image in result["future_image_predictions"].items():
            Image.fromarray(image).save(args.output_dir / f"{label}_{modality}.png")
    Image.fromarray(recipient["all_camera_images"][1]).save(args.output_dir / "open_target_primary.png")
    Image.fromarray(closed_target["all_camera_images"][1]).save(args.output_dir / "closed_target_primary.png")
    np.savez_compressed(
        args.output_dir / "actions.npz",
        **{f"normalized_{name}": value for name, value in normalized_actions.items()},
        **{f"environment_{name}": value for name, value in environment_actions.items()},
    )
    summary = {
        "scope": "exploratory semantic future clamp; target pair is temporally matched, not a natural success/failure pair",
        "recipient_run": str(args.recipient_run_dir),
        "closed_target_run": str(args.closed_target_run_dir),
        "closed_target_branch": args.closed_target_branch,
        "model_seed": args.model_seed,
        "future_noise_seed": args.future_noise_seed,
        "future_frame_indices": list(groups.future),
        "denoiser_sigmas": {"open": open_clamp.calls, "closed": closed_clamp.calls},
        "normalized_action_l2": {
            "open_from_baseline": float(
                np.linalg.norm(
                    normalized_actions["open_clamp"].astype(np.float64)
                    - normalized_actions["baseline"].astype(np.float64)
                )
            ),
            "closed_from_baseline": float(
                np.linalg.norm(
                    normalized_actions["closed_clamp"].astype(np.float64)
                    - normalized_actions["baseline"].astype(np.float64)
                )
            ),
            "closed_from_open": float(
                np.linalg.norm(
                    normalized_actions["closed_clamp"].astype(np.float64)
                    - normalized_actions["open_clamp"].astype(np.float64)
                )
            ),
        },
        "predicted_values": {
            "baseline": float(recipient["value_prediction"]),
            "open_clamp": float(open_result["value_prediction"]),
            "closed_clamp": float(closed_result["value_prediction"]),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
