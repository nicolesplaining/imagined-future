"""Exploratory layer scan for future-noise-to-action residual-stream coupling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imagined_future.activation_patch import capture_module_outputs, transplant_module_output
from imagined_future.cosmos_config import libero_policy_config
from imagined_future.frames import LatentFrameGroups
from imagined_future.interventions import resample_frames
from imagined_future.model_patch import defer_hf_config_checkpoint_resolution, transform_model_initial_noise


def _actions(result: dict) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float64)


def _steering(recipient: np.ndarray, donor: np.ndarray, patched: np.ndarray) -> float:
    displacement = donor - recipient
    denominator = float(np.square(displacement).sum())
    if denominator == 0.0:
        raise RuntimeError("donor and recipient actions are identical")
    return float(((patched - recipient) * displacement).sum() / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--model-seed", type=int, default=195)
    parser.add_argument("--future-noise-seed", type=int, default=10195)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--run-self-control", action="store_true")
    args = parser.parse_args()

    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {args.output_dir}")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    artifact = np.load(args.run_dir / "branches.npz", allow_pickle=False)
    calibration = json.loads((args.run_dir / "summary.json").read_text())
    observation = {
        "primary_image": artifact["current_primary_image"],
        "wrist_image": artifact["current_wrist_image"],
        "proprio": artifact["current_proprio"],
    }
    cfg = libero_policy_config(args.suite, unnormalize_actions=False)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    with defer_hf_config_checkpoint_resolution():
        model, _cosmos_config = get_model(cfg)

    blocks = {str(index): block for index, block in enumerate(model.net.blocks)}
    requested = [str(index) for index in args.layers]
    missing = sorted(set(requested) - set(blocks))
    if missing:
        raise ValueError(f"invalid block indices: {missing}; model has {len(blocks)} blocks")
    if any(not 0.0 <= scale <= 1.0 for scale in args.scales):
        raise ValueError("all transplant scales must be in [0, 1]")

    common = dict(
        cfg=cfg,
        model=model,
        dataset_stats=dataset_stats,
        obs=observation,
        task_label_or_embedding=calibration["task_description"],
        seed=args.model_seed,
        randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=True,
    )
    recipient = get_action(**common)
    groups = LatentFrameGroups.from_batch(recipient["data_batch"])

    def future_noise(initial, batch):
        observed = LatentFrameGroups.from_batch(batch)
        if observed != groups:
            raise RuntimeError(f"latent frame layout changed: {observed} != {groups}")
        return resample_frames(
            initial,
            groups.future,
            seed=args.future_noise_seed,
            standard_deviation=float(model.sde.sigma_max),
        )

    with capture_module_outputs(blocks, groups.future) as bank:
        with transform_model_initial_noise(model, future_noise):
            donor = get_action(**common)

    recipient_action = _actions(recipient)
    donor_action = _actions(donor)
    self_control = None
    if args.run_self_control:
        requested_blocks = {name: blocks[name] for name in requested}
        with capture_module_outputs(requested_blocks, groups.future) as recipient_bank:
            recipient_capture = get_action(**common)
        recipient_capture_action = _actions(recipient_capture)
        control_layer = requested[0]
        with transplant_module_output(
            blocks[control_layer], recipient_bank.calls[control_layer], groups.future
        ) as control_transplant:
            recipient_self_patch = get_action(**common)
            control_transplant.validate_complete()
        recipient_self_patch_action = _actions(recipient_self_patch)
        self_control = {
            "layer": int(control_layer),
            "deterministic_rerun_bitwise_equal": bool(np.array_equal(recipient_capture_action, recipient_action)),
            "deterministic_rerun_max_abs": float(np.abs(recipient_capture_action - recipient_action).max()),
            "self_transplant_bitwise_equal": bool(np.array_equal(recipient_self_patch_action, recipient_action)),
            "self_transplant_max_abs": float(np.abs(recipient_self_patch_action - recipient_action).max()),
            "denoiser_calls": control_transplant.calls,
        }
    records = []
    patched_actions = []
    record_layers = []
    record_scales = []
    for name in requested:
        for scale in args.scales:
            with transplant_module_output(
                blocks[name], bank.calls[name], groups.future, scale=scale
            ) as transplant:
                patched = get_action(**common)
                transplant.validate_complete()
            patched_action = _actions(patched)
            patched_actions.append(patched_action)
            record_layers.append(int(name))
            record_scales.append(scale)
            records.append(
                {
                    "layer": int(name),
                    "scale": scale,
                    "denoiser_calls": transplant.calls,
                    "donor_steering": _steering(recipient_action, donor_action, patched_action),
                    "action_l2_from_recipient": float(np.linalg.norm(patched_action - recipient_action)),
                    "action_l2_to_donor": float(np.linalg.norm(patched_action - donor_action)),
                    "patch_l2_by_call": transplant.patch_l2,
                    "donor_recipient_activation_l2_by_call": transplant.donor_recipient_l2,
                    "recipient_activation_l2_by_call": transplant.recipient_l2,
                    "donor_activation_l2_by_call": transplant.donor_l2,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "actions.npz",
        recipient=recipient_action,
        donor=donor_action,
        layers=np.asarray(record_layers, dtype=np.int64),
        scales=np.asarray(record_scales, dtype=np.float64),
        patched=np.stack(patched_actions),
    )
    summary = {
        "scope": "exploratory future-noise residual-stream coupling scan; not semantic mediation evidence",
        "source_run": str(args.run_dir),
        "model_seed": args.model_seed,
        "future_noise_seed": args.future_noise_seed,
        "future_frame_indices": list(groups.future),
        "recipient_to_donor_action_l2": float(np.linalg.norm(donor_action - recipient_action)),
        "self_control": self_control,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
