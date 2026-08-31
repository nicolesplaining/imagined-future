"""Run a controlled future-noise smoke test on Cosmos's public LIBERO sample."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from imagined_future.frames import LatentFrameGroups
from imagined_future.interventions import resample_frames
from imagined_future.model_patch import transform_model_initial_noise


def _actions(result: dict) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=195)
    parser.add_argument("--future-noise-seed", type=int, default=10195)
    args = parser.parse_args()

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

    cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
        t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=False,
        chunk_size=16,
        num_open_loop_steps=16,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        deterministic=True,
        randomize_seed=False,
        use_variance_scale=False,
    )
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, _ = get_model(cfg)

    sample_path = Path("cosmos_policy/experiments/robot/libero/sample_libero_10_observation.pkl")
    with sample_path.open("rb") as handle:
        observation = pickle.load(handle)
    task = "put both the alphabet soup and the tomato sauce in the basket"

    common = dict(
        cfg=cfg,
        model=model,
        dataset_stats=dataset_stats,
        obs=observation,
        task_label_or_embedding=task,
        seed=args.model_seed,
        randomize_seed=False,
        num_denoising_steps_action=5,
        generate_future_state_and_value_in_parallel=True,
    )
    baseline = get_action(**common)

    seen_groups: list[LatentFrameGroups] = []

    def resample_future(initial, batch):
        groups = LatentFrameGroups.from_batch(batch)
        seen_groups.append(groups)
        return resample_frames(
            initial,
            groups.future,
            seed=args.future_noise_seed,
            standard_deviation=float(model.sde.sigma_max),
        )

    with transform_model_initial_noise(model, resample_future):
        intervened = get_action(**common)

    baseline_action = _actions(baseline)
    intervened_action = _actions(intervened)
    delta = intervened_action - baseline_action
    future_pixel_l1 = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, baseline_image in baseline["future_image_predictions"].items():
        intervention_image = intervened["future_image_predictions"][key]
        future_pixel_l1[key] = float(
            np.abs(intervention_image.astype(np.float64) - baseline_image.astype(np.float64)).mean()
        )
        Image.fromarray(baseline_image).save(args.output_dir / f"baseline_{key}.png")
        Image.fromarray(intervention_image).save(args.output_dir / f"future_noise_{key}.png")

    summary = {
        "scope": "public-observation coupling smoke test; not a confirmatory semantic-use result",
        "model_seed": args.model_seed,
        "future_noise_seed": args.future_noise_seed,
        "future_frame_indices": list(seen_groups[0].future),
        "action_l2": float(np.linalg.norm(delta)),
        "action_max_abs": float(np.abs(delta).max()),
        "baseline_action_l2": float(np.linalg.norm(baseline_action)),
        "future_pixel_l1": future_pixel_l1,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "actions.npz",
        baseline=baseline_action,
        future_noise=intervened_action,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
