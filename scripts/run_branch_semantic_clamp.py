"""Clamp a policy future to matched endpoints from one exact-state branch set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imagined_future.cosmos_config import deterministic_tokenizer_enabled, libero_policy_config
from imagined_future.frames import LatentFrameGroups, future_frame_modalities
from imagined_future.interventions import SemanticFutureClamp, replace_frames, resample_frames
from imagined_future.metrics import donor_steering
from imagined_future.model_patch import (
    defer_hf_config_checkpoint_resolution,
    transform_model_initial_noise,
    transform_model_x0_factory,
)
from imagined_future.paired_rollouts import pixel_l1, resize_uint8_image
from imagined_future.semantic_latent import encode_semantic_future


def _current_observation(artifact: np.lib.npyio.NpzFile) -> dict:
    return {
        "primary_image": artifact["current_primary_image"],
        "wrist_image": artifact["current_wrist_image"],
        "proprio": artifact["current_proprio"],
    }


def _endpoint_observation(artifact: np.lib.npyio.NpzFile, branch: int) -> dict:
    return {
        "primary_image": artifact["endpoint_primary_images"][branch],
        "wrist_image": artifact["endpoint_wrist_images"][branch],
        "proprio": artifact["endpoint_proprios"][branch],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--recipient-branch", type=int, required=True)
    parser.add_argument("--donor-branch", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--future-noise-seed", type=int, default=20195)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("proprio", "wrist", "primary"),
        default=["proprio", "wrist", "primary"],
    )
    args = parser.parse_args()

    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {args.output_dir}")
    if args.recipient_branch == args.donor_branch:
        raise ValueError("recipient and donor branches must differ")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
        unnormalize_actions,
    )

    artifact = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    branch_summary = json.loads((args.branch_run_dir / "summary.json").read_text())
    branch_count = len(artifact["branch_seeds"])
    for name, index in (("recipient", args.recipient_branch), ("donor", args.donor_branch)):
        if not 0 <= index < branch_count:
            raise IndexError(f"{name} branch {index} is outside 0..{branch_count - 1}")
    if "normalized_branch_actions" not in artifact.files:
        raise ValueError("branch artifact predates exact normalized-action recording; recollect it with the current script")

    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)

    common = dict(
        cfg=cfg,
        model=model,
        dataset_stats=dataset_stats,
        task_label_or_embedding=branch_summary["task_description"],
        seed=args.model_seed,
        randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=True,
    )
    current_observation = _current_observation(artifact)
    recipient_target_observation = _endpoint_observation(artifact, args.recipient_branch)
    donor_target_observation = _endpoint_observation(artifact, args.donor_branch)
    baseline = get_action(obs=current_observation, **common)
    recipient_action = artifact["normalized_branch_actions"][args.recipient_branch]
    donor_action = artifact["normalized_branch_actions"][args.donor_branch]
    baseline_reference_error = float(
        np.max(np.abs(np.asarray(baseline["actions"]).astype(np.float64) - recipient_action.astype(np.float64)))
    )
    if baseline_reference_error != 0.0:
        raise RuntimeError(
            "baseline generation does not reproduce the saved recipient action exactly; "
            f"maximum absolute error is {baseline_reference_error}"
        )
    recipient_target_batch = get_action(obs=recipient_target_observation, **common)
    donor_target_batch = get_action(obs=donor_target_observation, **common)
    groups = LatentFrameGroups.from_batch(baseline["data_batch"])
    modality_frames = future_frame_modalities(baseline["data_batch"])
    if len(set(args.modalities)) != len(args.modalities):
        raise ValueError("modalities must be unique")
    clamp_frames = tuple(index for name in args.modalities for index in modality_frames[name])
    if not clamp_frames:
        raise ValueError("selected modalities have no future frames in this batch")
    if LatentFrameGroups.from_batch(recipient_target_batch["data_batch"]) != groups:
        raise RuntimeError("recipient target uses a different latent frame layout")
    if LatentFrameGroups.from_batch(donor_target_batch["data_batch"]) != groups:
        raise RuntimeError("donor target uses a different latent frame layout")

    recipient_clean = encode_semantic_future(
        model,
        baseline["data_batch"],
        recipient_target_batch["data_batch"],
        baseline["orig_clean_latent_frames"],
        recipient_target_batch["proprio"],
    )
    donor_clean = encode_semantic_future(
        model,
        baseline["data_batch"],
        donor_target_batch["data_batch"],
        baseline["orig_clean_latent_frames"],
        donor_target_batch["proprio"],
    )
    shared_noise = resample_frames(
        torch.zeros_like(donor_clean), clamp_frames, seed=args.future_noise_seed, standard_deviation=1.0
    )

    def run_clamp(clean: torch.Tensor):
        clamp = SemanticFutureClamp(clean, shared_noise, clamp_frames)

        def initial_transform(initial: torch.Tensor, batch: dict) -> torch.Tensor:
            if LatentFrameGroups.from_batch(batch) != groups:
                raise RuntimeError("latent frame layout changed during clamp")
            target_at_maximum_noise = clean + float(model.sde.sigma_max) * shared_noise
            return replace_frames(initial, target_at_maximum_noise, clamp_frames)

        with transform_model_initial_noise(model, initial_transform):
            with transform_model_x0_factory(model, clamp.wrap):
                result = get_action(obs=current_observation, **common)
        return result, clamp

    recipient_result, recipient_clamp = run_clamp(recipient_clean)
    donor_result, donor_clamp = run_clamp(donor_clean)
    normalized_actions = {
        "baseline": np.asarray(baseline["actions"]),
        "recipient_clamp": np.asarray(recipient_result["actions"]),
        "donor_clamp": np.asarray(donor_result["actions"]),
    }
    environment_actions = {
        name: unnormalize_actions(action.copy(), dataset_stats) for name, action in normalized_actions.items()
    }
    steering = {
        name: float(
            donor_steering(
                torch.from_numpy(action.astype(np.float64)).unsqueeze(0),
                torch.from_numpy(recipient_action.astype(np.float64)).unsqueeze(0),
                torch.from_numpy(donor_action.astype(np.float64)).unsqueeze(0),
            ).item()
        )
        for name, action in normalized_actions.items()
    }

    def sliced_steering(action: np.ndarray, *, transpose: bool = False) -> list[float]:
        tensors = [
            torch.from_numpy(value.astype(np.float64))
            for value in (action, recipient_action, donor_action)
        ]
        if transpose:
            tensors = [value.T for value in tensors]
        return donor_steering(*tensors).tolist()

    steering_by_timestep = {
        name: sliced_steering(action) for name, action in normalized_actions.items()
    }
    steering_by_dimension = {
        name: sliced_steering(action, transpose=True) for name, action in normalized_actions.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "baseline": baseline,
        "recipient_clamp": recipient_result,
        "donor_clamp": donor_result,
    }
    for label, result in results.items():
        for modality, image in result["future_image_predictions"].items():
            Image.fromarray(image).save(args.output_dir / f"{label}_{modality}.png")
    target_images = {
        "primary": {
            "recipient": np.asarray(recipient_target_batch["all_camera_images"][1], dtype=np.uint8),
            "donor": np.asarray(donor_target_batch["all_camera_images"][1], dtype=np.uint8),
        },
        "wrist": {
            "recipient": np.asarray(recipient_target_batch["all_camera_images"][0], dtype=np.uint8),
            "donor": np.asarray(donor_target_batch["all_camera_images"][0], dtype=np.uint8),
        },
    }
    for modality, targets in target_images.items():
        for name, image in targets.items():
            Image.fromarray(image).save(args.output_dir / f"{name}_target_{modality}.png")
    np.savez_compressed(
        args.output_dir / "actions.npz",
        recipient_reference_normalized=recipient_action,
        donor_reference_normalized=donor_action,
        **{f"normalized_{name}": value for name, value in normalized_actions.items()},
        **{f"environment_{name}": value for name, value in environment_actions.items()},
    )

    decoded_recipient = recipient_result["future_image_predictions"]["future_image"]
    decoded_donor = donor_result["future_image_predictions"]["future_image"]
    target_shape = decoded_recipient.shape[:2]
    recipient_target = resize_uint8_image(target_images["primary"]["recipient"], target_shape)
    donor_target = resize_uint8_image(target_images["primary"]["donor"], target_shape)
    decoded_recipient_wrist = recipient_result["future_image_predictions"]["future_wrist_image"]
    decoded_donor_wrist = donor_result["future_image_predictions"]["future_wrist_image"]
    wrist_shape = decoded_recipient_wrist.shape[:2]
    recipient_target_wrist = resize_uint8_image(target_images["wrist"]["recipient"], wrist_shape)
    donor_target_wrist = resize_uint8_image(target_images["wrist"]["donor"], wrist_shape)
    frame_index = torch.as_tensor(clamp_frames, device=donor_clean.device)
    recipient_selected = torch.index_select(recipient_clean, 2, frame_index).float()
    donor_selected = torch.index_select(donor_clean, 2, frame_index).float()
    summary = {
        "scope": "matched exact-state semantic future clamp",
        "branch_run": str(args.branch_run_dir),
        "branch_state_digest": branch_summary["branch_state_digest"],
        "task_id": branch_summary["task_id"],
        "task_description": branch_summary["task_description"],
        "initial_state_index": branch_summary["initial_state_index"],
        "recipient_branch": args.recipient_branch,
        "recipient_seed": int(artifact["branch_seeds"][args.recipient_branch]),
        "donor_branch": args.donor_branch,
        "donor_seed": int(artifact["branch_seeds"][args.donor_branch]),
        "model_seed": args.model_seed,
        "future_noise_seed": args.future_noise_seed,
        "deterministic_tokenizer": deterministic_tokenizer_enabled(),
        "modalities": args.modalities,
        "future_frame_indices": list(clamp_frames),
        "denoiser_sigmas": {"recipient": recipient_clamp.calls, "donor": donor_clamp.calls},
        "reference_action_l2": float(
            np.linalg.norm(donor_action.astype(np.float64) - recipient_action.astype(np.float64))
        ),
        "baseline_reference_max_abs_error": baseline_reference_error,
        "donor_steering": steering,
        "donor_steering_effect": steering["donor_clamp"] - steering["recipient_clamp"],
        "donor_steering_effect_by_timestep": [
            donor - recipient
            for donor, recipient in zip(
                steering_by_timestep["donor_clamp"],
                steering_by_timestep["recipient_clamp"],
                strict=True,
            )
        ],
        "donor_steering_effect_by_action_dimension": [
            donor - recipient
            for donor, recipient in zip(
                steering_by_dimension["donor_clamp"],
                steering_by_dimension["recipient_clamp"],
                strict=True,
            )
        ],
        "normalized_action_l2": {
            "recipient_clamp_from_baseline": float(
                np.linalg.norm(normalized_actions["recipient_clamp"] - normalized_actions["baseline"])
            ),
            "donor_clamp_from_baseline": float(
                np.linalg.norm(normalized_actions["donor_clamp"] - normalized_actions["baseline"])
            ),
            "donor_from_recipient_clamp": float(
                np.linalg.norm(normalized_actions["donor_clamp"] - normalized_actions["recipient_clamp"])
            ),
        },
        "primary_pixel_l1": {
            "recipient_decoded_to_recipient_target": pixel_l1(decoded_recipient, recipient_target),
            "donor_decoded_to_donor_target": pixel_l1(decoded_donor, donor_target),
            "recipient_target_to_donor_target": pixel_l1(recipient_target, donor_target),
            "recipient_decoded_to_donor_decoded": pixel_l1(decoded_recipient, decoded_donor),
        },
        "wrist_pixel_l1": {
            "recipient_decoded_to_recipient_target": pixel_l1(
                decoded_recipient_wrist, recipient_target_wrist
            ),
            "donor_decoded_to_donor_target": pixel_l1(decoded_donor_wrist, donor_target_wrist),
            "recipient_target_to_donor_target": pixel_l1(recipient_target_wrist, donor_target_wrist),
            "recipient_decoded_to_donor_decoded": pixel_l1(decoded_recipient_wrist, decoded_donor_wrist),
        },
        "selected_clean_latent_l2": {
            "recipient_norm": float(torch.linalg.vector_norm(recipient_selected).item()),
            "donor_norm": float(torch.linalg.vector_norm(donor_selected).item()),
            "donor_minus_recipient": float(
                torch.linalg.vector_norm(donor_selected - recipient_selected).item()
            ),
        },
        "predicted_values": {name: float(result["value_prediction"]) for name, result in results.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
