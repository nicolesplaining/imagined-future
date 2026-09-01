"""Run frozen robot-versus-object reachable-future interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from imagined_future.cosmos_config import (
    deterministic_tokenizer_enabled,
    libero_policy_config,
)
from imagined_future.frames import LatentFrameGroups, future_frame_modalities
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


def _observation(artifact: np.lib.npyio.NpzFile, branch_index: int | None) -> dict:
    if branch_index is None:
        return {
            "primary_image": artifact["current_primary_image"],
            "wrist_image": artifact["current_wrist_image"],
            "proprio": artifact["current_proprio"],
        }
    return {
        "primary_image": artifact["endpoint_primary_images"][branch_index],
        "wrist_image": artifact["endpoint_wrist_images"][branch_index],
        "proprio": artifact["endpoint_proprios"][branch_index],
    }


def _matched_random_target(
    recipient_clean: torch.Tensor,
    donor_clean: torch.Tensor,
    frames: tuple[int, ...],
    *,
    seed: int,
) -> torch.Tensor:
    index = torch.as_tensor(frames, device=recipient_clean.device)
    recipient = torch.index_select(recipient_clean, 2, index)
    donor = torch.index_select(donor_clean, 2, index)
    matched = norm_distance_matched_random_target(recipient, donor, seed=seed)
    output = recipient_clean.clone()
    output[:, :, index, :, :] = matched
    return output


def _steering(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> float:
    return float(
        donor_steering(
            torch.from_numpy(np.asarray(value, dtype=np.float64)).unsqueeze(0),
            torch.from_numpy(np.asarray(recipient, dtype=np.float64)).unsqueeze(0),
            torch.from_numpy(np.asarray(donor, dtype=np.float64)).unsqueeze(0),
        ).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument(
        "--future-noise-seeds", type=int, nargs="+", default=[1201, 1213, 1217]
    )
    parser.add_argument("--gaussian-seed", type=int, default=1223)
    args = parser.parse_args()

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
        unnormalize_actions,
    )

    manifest = json.loads(args.manifest.read_text())
    units = [
        unit
        for unit in manifest["units"]
        if unit["structurally_eligible"] and unit["task_id"] in args.task_ids
    ]
    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)

    completed = []
    for unit in units:
        output_dir = args.output_dir / unit["unit_id"]
        if (output_dir / "summary.json").exists():
            raise FileExistsError(f"refusing to overwrite completed run: {output_dir}")
        selected = unit["selected"]
        selection = selected["selection"]
        branch_dir = Path(selected["branch_run_dir"])
        branch = np.load(branch_dir / "branches.npz", allow_pickle=False)
        branch_summary = json.loads((branch_dir / "summary.json").read_text())
        current_observation = _observation(branch, None)
        anchor = int(selection["recipient"])
        object_donor = int(selection["object_donor"])
        robot_donor = int(selection["robot_donor"])
        joint_donor = selection["joint_donor"]
        natural_control = selection["natural_control"]
        contexts = [
            {
                "name": "object_forward",
                "pair_type": "object",
                "recipient": anchor,
                "donor": object_donor,
                "modalities": ("all", "wrist", "primary", "proprio"),
                "extras": {
                    "joint_donor": joint_donor,
                    "natural_control": natural_control,
                },
            },
            {
                "name": "object_reverse",
                "pair_type": "object",
                "recipient": object_donor,
                "donor": anchor,
                "modalities": ("all",),
                "extras": {},
            },
            {
                "name": "robot_forward",
                "pair_type": "robot",
                "recipient": anchor,
                "donor": robot_donor,
                "modalities": ("all", "wrist", "primary", "proprio"),
                "extras": {},
            },
            {
                "name": "robot_reverse",
                "pair_type": "robot",
                "recipient": robot_donor,
                "donor": anchor,
                "modalities": ("all",),
                "extras": {},
            },
        ]
        action_arrays: dict[str, np.ndarray] = {}
        execution_arrays: dict[str, np.ndarray] = {}
        rows = []
        for context in contexts:
            recipient_index = context["recipient"]
            donor_index = context["donor"]
            model_seed = int(branch["branch_seeds"][recipient_index])
            common = {
                "cfg": cfg,
                "model": model,
                "dataset_stats": dataset_stats,
                "task_label_or_embedding": branch_summary["task_description"],
                "seed": model_seed,
                "randomize_seed": False,
                "num_denoising_steps_action": cfg.num_denoising_steps_action,
                "generate_future_state_and_value_in_parallel": True,
            }
            baseline = get_action(obs=current_observation, **common)
            baseline_action = np.asarray(baseline["actions"])
            recipient_reference = np.asarray(
                branch["normalized_branch_actions"][recipient_index]
            )
            donor_reference = np.asarray(
                branch["normalized_branch_actions"][donor_index]
            )
            baseline_error = float(
                np.max(np.abs(baseline_action.astype(np.float64) - recipient_reference))
            )
            if baseline_error != 0.0:
                raise RuntimeError(
                    f"{unit['unit_id']} {context['name']} baseline mismatch: {baseline_error}"
                )
            target_indices = {
                "recipient": recipient_index,
                "donor": donor_index,
                **{
                    name: int(index)
                    for name, index in context["extras"].items()
                    if index is not None
                },
            }
            target_observations = {
                name: _observation(branch, index)
                for name, index in target_indices.items()
            }
            target_batches = {
                name: get_action(obs=observation, **common)
                for name, observation in target_observations.items()
            }
            groups = LatentFrameGroups.from_batch(baseline["data_batch"])
            modality_groups = future_frame_modalities(baseline["data_batch"])
            all_frames = tuple(groups.future)
            clean_targets = {
                name: encode_semantic_future(
                    model,
                    baseline["data_batch"],
                    result["data_batch"],
                    baseline["orig_clean_latent_frames"],
                    result["proprio"],
                )
                for name, result in target_batches.items()
            }
            clean_targets["gaussian"] = _matched_random_target(
                clean_targets["recipient"],
                clean_targets["donor"],
                all_frames,
                seed=args.gaussian_seed + recipient_index,
            )
            for noise_seed in args.future_noise_seeds:
                shared_noise = resample_frames(
                    torch.zeros_like(clean_targets["donor"]),
                    all_frames,
                    seed=noise_seed,
                    standard_deviation=1.0,
                )
                conditions = []
                for modality in context["modalities"]:
                    frames = (
                        all_frames
                        if modality == "all"
                        else tuple(modality_groups[modality])
                    )
                    conditions.extend(
                        [
                            (modality, "recipient", frames),
                            (modality, "donor", frames),
                        ]
                    )
                conditions.append(("all", "gaussian", all_frames))
                if context["name"] == "object_forward":
                    conditions.extend(
                        ("all", name, all_frames)
                        for name in context["extras"]
                        if name in clean_targets
                    )
                for modality, target_role, clamp_frames in conditions:
                    clean_target = clean_targets[target_role]
                    clamp = SemanticFutureClamp(
                        clean_target, shared_noise, clamp_frames
                    )

                    def initial_transform(
                        initial: torch.Tensor,
                        batch: dict,
                        *,
                        _clean=clean_target,
                        _frames=clamp_frames,
                        _groups=groups,
                        _shared_noise=shared_noise,
                    ) -> torch.Tensor:
                        if LatentFrameGroups.from_batch(batch) != _groups:
                            raise RuntimeError("latent frame layout changed")
                        target_at_maximum_noise = (
                            _clean + float(model.sde.sigma_max) * _shared_noise
                        )
                        return replace_frames(initial, target_at_maximum_noise, _frames)

                    with (
                        transform_model_initial_noise(model, initial_transform),
                        transform_model_x0_factory(model, clamp.wrap),
                    ):
                        result = get_action(obs=current_observation, **common)
                    name = f"{context['name']}_n{noise_seed}_{modality}_{target_role}"
                    normalized_action = np.asarray(result["actions"])
                    environment_action = unnormalize_actions(
                        normalized_action.copy(), dataset_stats
                    )
                    action_arrays[f"normalized_{name}"] = normalized_action
                    action_arrays[f"environment_{name}"] = environment_action
                    if modality == "all":
                        execution_arrays[f"environment_{name}"] = environment_action
                    decoded_primary = np.asarray(
                        result["future_image_predictions"]["future_image"],
                        dtype=np.uint8,
                    )
                    target_shape = decoded_primary.shape[:2]
                    recipient_image = resize_uint8_image(
                        target_observations["recipient"]["primary_image"], target_shape
                    )
                    donor_image = resize_uint8_image(
                        target_observations["donor"]["primary_image"], target_shape
                    )
                    own_index = target_indices.get(target_role)
                    own_steering = None
                    if own_index is not None and own_index != recipient_index:
                        own_steering = _steering(
                            normalized_action,
                            recipient_reference,
                            branch["normalized_branch_actions"][own_index],
                        )
                    rows.append(
                        {
                            "condition": name,
                            "context": context["name"],
                            "pair_type": context["pair_type"],
                            "direction": (
                                "forward"
                                if context["name"].endswith("forward")
                                else "reverse"
                            ),
                            "recipient_branch": recipient_index,
                            "donor_branch": donor_index,
                            "future_noise_seed": noise_seed,
                            "modality": modality,
                            "target_role": target_role,
                            "target_branch": own_index,
                            "pair_donor_steering": _steering(
                                normalized_action, recipient_reference, donor_reference
                            ),
                            "own_target_action_steering": own_steering,
                            "action_l2_from_baseline": float(
                                np.linalg.norm(normalized_action - baseline_action)
                            ),
                            "decoded_primary_l1_to_recipient": pixel_l1(
                                decoded_primary, recipient_image
                            ),
                            "decoded_primary_l1_to_donor": pixel_l1(
                                decoded_primary, donor_image
                            ),
                            "denoiser_sigmas": clamp.calls,
                        }
                    )

            index = torch.as_tensor(all_frames, device=clean_targets["donor"].device)
            selected_latents = {
                name: torch.index_select(value, 2, index).float()
                for name, value in clean_targets.items()
            }
            rows.append(
                {
                    "context": context["name"],
                    "target_role": "latent_control_diagnostics",
                    "recipient_norm": float(
                        torch.linalg.vector_norm(selected_latents["recipient"]).item()
                    ),
                    "donor_norm": float(
                        torch.linalg.vector_norm(selected_latents["donor"]).item()
                    ),
                    "gaussian_norm": float(
                        torch.linalg.vector_norm(selected_latents["gaussian"]).item()
                    ),
                    "donor_distance_from_recipient": float(
                        torch.linalg.vector_norm(
                            selected_latents["donor"] - selected_latents["recipient"]
                        ).item()
                    ),
                    "gaussian_distance_from_recipient": float(
                        torch.linalg.vector_norm(
                            selected_latents["gaussian"] - selected_latents["recipient"]
                        ).item()
                    ),
                }
            )

        output_dir.mkdir(parents=True)
        np.savez_compressed(output_dir / "actions.npz", **action_arrays)
        np.savez_compressed(output_dir / "execution_actions.npz", **execution_arrays)
        summary = {
            "scope": "frozen held-out robot-versus-object semantic intervention suite",
            "study_manifest": str(args.manifest),
            "unit_id": unit["unit_id"],
            "task_id": unit["task_id"],
            "task_description": unit["task_description"],
            "initial_state_index": unit["initial_state_index"],
            "prefix_chunks": selected["prefix_chunks"],
            "branch_run": str(branch_dir),
            "branch_state_digest": branch_summary["branch_state_digest"],
            "selection": selection,
            "future_noise_seeds": args.future_noise_seeds,
            "gaussian_seed": args.gaussian_seed,
            "deterministic_tokenizer": deterministic_tokenizer_enabled(),
            "rows": rows,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        completed.append(unit["unit_id"])
    print(
        json.dumps({"output": str(args.output_dir), "completed": completed}, indent=2)
    )


if __name__ == "__main__":
    main()
