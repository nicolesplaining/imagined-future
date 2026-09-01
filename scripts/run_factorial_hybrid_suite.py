"""Run frozen 2x2 rendered object/robot future interventions."""

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
from imagined_future.model_patch import (
    defer_hf_config_checkpoint_resolution,
    transform_model_initial_noise,
    transform_model_x0_factory,
)
from imagined_future.paired_rollouts import pixel_l1, resize_uint8_image
from imagined_future.semantic_latent import encode_semantic_future

CELL_NAMES = ("o0r0", "o1r0", "o0r1", "o1r1")


def _cell_observation(artifact: np.lib.npyio.NpzFile, name: str) -> dict:
    names = [str(value) for value in artifact["cell_names"]]
    index = names.index(name)
    return {
        "primary_image": artifact["cell_primary_images"][index],
        "wrist_image": artifact["cell_wrist_images"][index],
        "proprio": artifact["cell_proprios"][index],
    }


def _current_observation(artifact: np.lib.npyio.NpzFile) -> dict:
    return {
        "primary_image": artifact["current_primary_image"],
        "wrist_image": artifact["current_wrist_image"],
        "proprio": artifact["current_proprio"],
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
        if unit["valid"] and unit["task_id"] in args.task_ids
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
        target_dir = Path(unit["target_dir"])
        target_summary = json.loads((target_dir / "summary.json").read_text())
        artifact = np.load(target_dir / "branches.npz", allow_pickle=False)
        observations = {
            name: _cell_observation(artifact, name) for name in CELL_NAMES
        }
        current_observation = _current_observation(artifact)
        common = {
            "cfg": cfg,
            "model": model,
            "dataset_stats": dataset_stats,
            "task_label_or_embedding": target_summary["task_description"],
            "seed": 503,
            "randomize_seed": False,
            "num_denoising_steps_action": cfg.num_denoising_steps_action,
            "generate_future_state_and_value_in_parallel": True,
        }
        baseline = get_action(obs=current_observation, **common)
        baseline_action = np.asarray(baseline["actions"])
        native_action = np.asarray(artifact["normalized_native_actions"])
        baseline_error = float(
            np.max(np.abs(baseline_action.astype(np.float64) - native_action))
        )
        if baseline_error != 0.0:
            raise RuntimeError(
                f"{unit['unit_id']} native baseline mismatch: {baseline_error}"
            )
        target_batches = {
            name: get_action(obs=observations[name], **common)
            for name in CELL_NAMES
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
            clean_targets["o0r0"],
            clean_targets["o1r1"],
            all_frames,
            seed=args.gaussian_seed,
        )
        action_arrays: dict[str, np.ndarray] = {
            "normalized_native": native_action,
            "environment_native": np.asarray(
                artifact["environment_native_actions"]
            ),
        }
        execution_arrays: dict[str, np.ndarray] = {
            "environment_native": np.asarray(
                artifact["environment_native_actions"]
            )
        }
        rows = [
            {
                "condition": "native",
                "future_noise_seed": None,
                "modality": "unclamped",
                "target_cell": "native",
                "action_l2_to_native": 0.0,
            }
        ]
        for noise_seed in args.future_noise_seeds:
            shared_noise = resample_frames(
                torch.zeros_like(clean_targets["o1r1"]),
                all_frames,
                seed=noise_seed,
                standard_deviation=1.0,
            )
            conditions = [
                (modality, cell)
                for modality in ("all", "wrist", "primary", "proprio")
                for cell in CELL_NAMES
            ]
            conditions.append(("all", "gaussian"))
            for modality, target_cell in conditions:
                clamp_frames = (
                    all_frames
                    if modality == "all"
                    else tuple(modality_groups[modality])
                )
                clean_target = clean_targets[target_cell]
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
                    return replace_frames(
                        initial, target_at_maximum_noise, _frames
                    )

                with (
                    transform_model_initial_noise(model, initial_transform),
                    transform_model_x0_factory(model, clamp.wrap),
                ):
                    result = get_action(obs=current_observation, **common)
                name = f"n{noise_seed}_{modality}_{target_cell}"
                normalized_action = np.asarray(result["actions"])
                environment_action = unnormalize_actions(
                    normalized_action.copy(), dataset_stats
                )
                action_arrays[f"normalized_{name}"] = normalized_action
                action_arrays[f"environment_{name}"] = environment_action
                if modality == "all":
                    execution_arrays[f"environment_{name}"] = environment_action
                decoded = result["future_image_predictions"]
                decoded_primary = np.asarray(decoded["future_image"], dtype=np.uint8)
                decoded_wrist = np.asarray(
                    decoded["future_wrist_image"], dtype=np.uint8
                )
                primary_shape = decoded_primary.shape[:2]
                wrist_shape = decoded_wrist.shape[:2]
                target_primary_l1 = None
                target_wrist_l1 = None
                primary_target_top1 = None
                wrist_target_top1 = None
                primary_target_margin = None
                wrist_target_margin = None
                primary_distances = None
                wrist_distances = None
                if target_cell in observations:
                    primary_distances = {
                        cell: pixel_l1(
                            decoded_primary,
                            resize_uint8_image(
                                observations[cell]["primary_image"],
                                primary_shape,
                            ),
                        )
                        for cell in CELL_NAMES
                    }
                    wrist_distances = {
                        cell: pixel_l1(
                            decoded_wrist,
                            resize_uint8_image(
                                observations[cell]["wrist_image"], wrist_shape
                            ),
                        )
                        for cell in CELL_NAMES
                    }
                    target_primary_l1 = primary_distances[target_cell]
                    target_wrist_l1 = wrist_distances[target_cell]
                    primary_other = min(
                        value
                        for cell, value in primary_distances.items()
                        if cell != target_cell
                    )
                    wrist_other = min(
                        value
                        for cell, value in wrist_distances.items()
                        if cell != target_cell
                    )
                    primary_target_margin = primary_other - target_primary_l1
                    wrist_target_margin = wrist_other - target_wrist_l1
                    primary_target_top1 = float(
                        np.isclose(target_primary_l1, min(primary_distances.values()))
                    )
                    wrist_target_top1 = float(
                        np.isclose(target_wrist_l1, min(wrist_distances.values()))
                    )
                rows.append(
                    {
                        "condition": name,
                        "future_noise_seed": noise_seed,
                        "modality": modality,
                        "target_cell": target_cell,
                        "action_l2_to_native": float(
                            np.linalg.norm(normalized_action - native_action)
                        ),
                        "action_l2_from_unclamped": float(
                            np.linalg.norm(normalized_action - baseline_action)
                        ),
                        "decoded_primary_l1_to_target": target_primary_l1,
                        "decoded_wrist_l1_to_target": target_wrist_l1,
                        "decoded_primary_target_top1": primary_target_top1,
                        "decoded_wrist_target_top1": wrist_target_top1,
                        "decoded_primary_target_margin": primary_target_margin,
                        "decoded_wrist_target_margin": wrist_target_margin,
                        "decoded_primary_l1_by_cell": primary_distances,
                        "decoded_wrist_l1_by_cell": wrist_distances,
                        "denoiser_sigmas": clamp.calls,
                    }
                )

        index = torch.as_tensor(all_frames, device=clean_targets["o0r0"].device)
        selected_latents = {
            name: torch.index_select(value, 2, index).float()
            for name, value in clean_targets.items()
        }
        latent_diagnostics = {
            name: {
                "norm": float(torch.linalg.vector_norm(value).item()),
                "distance_from_o0r0": float(
                    torch.linalg.vector_norm(
                        value - selected_latents["o0r0"]
                    ).item()
                ),
            }
            for name, value in selected_latents.items()
        }
        output_dir.mkdir(parents=True)
        np.savez_compressed(output_dir / "actions.npz", **action_arrays)
        np.savez_compressed(
            output_dir / "execution_actions.npz", **execution_arrays
        )
        summary = {
            "scope": "frozen prospective 2x2 rendered object/robot intervention suite",
            "study_manifest": str(args.manifest),
            "unit_id": unit["unit_id"],
            "task_id": unit["task_id"],
            "task_description": unit["task_description"],
            "initial_state_index": unit["initial_state_index"],
            "prefix_chunks": unit["prefix_chunks"],
            "target_dir": str(target_dir),
            "branch_state_digest": target_summary["branch_state_digest"],
            "future_noise_seeds": args.future_noise_seeds,
            "gaussian_seed": args.gaussian_seed,
            "deterministic_tokenizer": deterministic_tokenizer_enabled(),
            "latent_diagnostics": latent_diagnostics,
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
