#!/usr/bin/env python3
"""Run the additive FastWAM future-latent x video-cache 2x2 factorial."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from imagined_future.fastwam_analysis import audit_fastwam_outputs
from imagined_future.fastwam_cache_factorial import (
    CACHE_FACTORIAL_CONDITIONS,
    CacheFactorialRunSpec,
    expand_cache_factorial_runs,
    load_cache_factorial_manifest,
    states_from_factorial_manifest,
    validate_factorial_parent,
    write_cache_factorial_manifest,
)
from imagined_future.fastwam_optional_idm import (
    FastWAMCondition,
    action_metrics,
    atomic_write_json,
    atomic_write_npz,
    cache_descriptor,
    expand_run_specs,
)

# This script lives beside the frozen population runner. Importing its setup
# helpers avoids modifying the already-running base implementation.
from run_fastwam_optional_idm import (
    _infer_idm,
    _load_fastwam_stack,
    _prepare_state_input,
    _to_env_action,
    _validate_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--factorial-manifest", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--exact-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _manifest_root(root: Path, manifest_id: str) -> Path:
    root = root.resolve()
    return root if root.name == manifest_id else root / manifest_id


def _load_base_arrays(base_manifest: dict[str, Any], state, base_root: Path):
    run_dir = base_root / state.state_id / "runs"
    specs = expand_run_specs(base_manifest["manifest_id"], state)
    actions: dict[tuple[str, str, str | None], np.ndarray] = {}
    stored_native_latents: dict[str, np.ndarray] = {}
    for spec in specs:
        key = (spec.condition, spec.recipient_id, spec.donor_id)
        if key in actions:
            raise RuntimeError(f"duplicate base reference key: {key}")
        with np.load(run_dir / f"{spec.run_id}.npz", allow_pickle=False) as arrays:
            action = np.asarray(arrays["action_model"], dtype=np.float32)
            if not np.isfinite(action).all():
                raise ValueError(f"nonfinite base action: {spec.run_id}")
            actions[key] = action
            if spec.condition == FastWAMCondition.NATIVE.value:
                latent = np.asarray(arrays["video_latent"], dtype=np.float32)
                if not np.isfinite(latent).all():
                    raise ValueError(f"nonfinite base video latent: {spec.run_id}")
                stored_native_latents[spec.recipient_id] = latent
    native_actions = {
        branch.branch_id: actions[(FastWAMCondition.NATIVE.value, branch.branch_id, None)]
        for branch in state.branches
    }
    return actions, native_actions, stored_native_latents


def _write_factorial_run(
    *,
    run_dir: Path,
    run: CacheFactorialRunSpec,
    manifest: dict[str, Any],
    action_model: np.ndarray,
    action_env: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    json_path = run_dir / f"{run.run_id}.json"
    npz_path = run_dir / f"{run.run_id}.npz"
    if json_path.exists() and npz_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing.get("run") != run.to_dict():
            raise RuntimeError(f"factorial run ID collision: {json_path}")
        return
    atomic_write_npz(
        npz_path,
        {
            "action_model": np.asarray(action_model, dtype=np.float32),
            "action_env": np.asarray(action_env, dtype=np.float32),
        },
    )
    atomic_write_json(
        json_path,
        {
            "status": "complete",
            "manifest_id": manifest["manifest_id"],
            "base_manifest_id": manifest["base_manifest_id"],
            "run": run.to_dict(),
            "array_file": npz_path.name,
            **metadata,
        },
    )


def _run_state(
    *,
    state,
    factorial_manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    cfg,
    model,
    processor,
    obs_to_model_input,
    denormalize_action,
    base_root: Path,
    output_root: Path,
    exact_tolerance: float,
) -> None:
    state_root = output_root / factorial_manifest["manifest_id"] / state.state_id
    run_dir = state_root / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    base_actions, native_actions, stored_native_latents = _load_base_arrays(
        base_manifest, state, base_root
    )
    image, proprio, context, context_mask, task_description = _prepare_state_input(
        state=state,
        cfg=cfg,
        model=model,
        processor=processor,
        obs_to_model_input=obs_to_model_input,
    )
    inference = factorial_manifest["inference"]
    common = {
        "prompt": None,
        "context": context,
        "context_mask": context_mask,
        "input_image": image,
        "proprio": proprio,
        "action_horizon": int(inference["action_horizon"]),
        "num_video_frames": int(inference["num_video_frames"]),
        "num_inference_steps": int(inference["num_inference_steps"]),
        "sigma_shift": float(inference["sigma_shift"]),
        "negative_prompt": "",
        "text_cfg_scale": 1.0,
        "rand_device": str(inference["rand_device"]),
        "tiled": False,
        "compile_action_infer": False,
    }

    cache_by_branch: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    native_latents: dict[str, torch.Tensor] = {}
    cache_meta: dict[str, Any] = {}
    native_action_replay_errors: dict[str, float] = {}
    stored_latent_replay_errors: dict[str, float] = {}
    for branch in state.branches:
        # Regenerate the native trajectory from its frozen seed. The parent
        # result intentionally stores latents as FP16, which is sufficient for
        # analysis but not a sound source for a bitwise K/V reconstruction
        # control. Regeneration preserves the original model dtype; the two
        # checks below bind it to both the parent action and stored artifact.
        output = _infer_idm(
            model=model,
            common=common,
            video_seed=branch.video_seed,
            action_seed=branch.action_seed,
            return_intermediates=True,
        )
        reconstructed = output["action"].detach().cpu().numpy().astype(np.float32)
        reference = base_actions[
            (FastWAMCondition.NATIVE.value, branch.branch_id, None)
        ]
        error = float(np.max(np.abs(reconstructed - reference)))
        native_action_replay_errors[branch.branch_id] = error
        if error > exact_tolerance:
            raise RuntimeError(
                f"{state.state_id} {branch.branch_id} native action regeneration "
                f"error {error} exceeds {exact_tolerance}"
            )
        latent = output["video_latents"].detach().cpu()
        stored_replay = latent.float().numpy().astype(np.float16).astype(np.float32)
        stored_error = float(
            np.max(np.abs(stored_replay - stored_native_latents[branch.branch_id]))
        )
        stored_latent_replay_errors[branch.branch_id] = stored_error
        if stored_error > exact_tolerance:
            raise RuntimeError(
                f"{state.state_id} {branch.branch_id} stored native latent "
                f"regeneration error {stored_error} exceeds {exact_tolerance}"
            )
        native_latents[branch.branch_id] = latent
        cache_k = [tensor.detach().cpu() for tensor in output["video_cache_k"]]
        cache_v = [tensor.detach().cpu() for tensor in output["video_cache_v"]]
        cache_by_branch[branch.branch_id] = (cache_k, cache_v)
        cache_meta[branch.branch_id] = cache_descriptor(cache_k, cache_v)
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    branch_by_id = {branch.branch_id: branch for branch in state.branches}
    all_runs = expand_cache_factorial_runs(factorial_manifest["manifest_id"], state)
    run_index = {
        (run.recipient_id, run.donor_id, run.condition): run for run in all_runs
    }
    base_reference_condition = {
        "future_recipient_cache_recipient": FastWAMCondition.SELF_CACHE.value,
        "future_recipient_cache_donor": FastWAMCondition.DONOR_CACHE.value,
        "future_donor_cache_donor": FastWAMCondition.DONOR_LATENT.value,
    }
    state_base_errors = {condition: [] for condition in base_reference_condition}
    recipient_cache_invariance: list[float] = []
    donor_cache_invariance: list[float] = []
    for recipient in state.branches:
        for donor in state.branches:
            if donor.branch_id == recipient.branch_id:
                continue
            cell_actions: dict[str, np.ndarray] = {}
            cell_env_actions: dict[str, np.ndarray] = {}
            cell_durations: dict[str, float] = {}
            for condition in CACHE_FACTORIAL_CONDITIONS:
                run = run_index[(recipient.branch_id, donor.branch_id, condition)]
                started = time.time()
                output = _infer_idm(
                    model=model,
                    common=common,
                    video_seed=branch_by_id[run.future_source_id].video_seed,
                    action_seed=recipient.action_seed,
                    video_latent=native_latents[run.future_source_id],
                    cache=cache_by_branch[run.cache_source_id],
                )
                action = output["action"].detach().cpu().numpy().astype(np.float32)
                cell_actions[condition] = action
                cell_env_actions[condition] = _to_env_action(
                    action, denormalize_action, processor
                )
                cell_durations[condition] = time.time() - started
                del output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            recipient_invariance = float(
                np.max(
                    np.abs(
                        cell_actions["future_donor_cache_recipient"]
                        - cell_actions["future_recipient_cache_recipient"]
                    )
                )
            )
            donor_invariance = float(
                np.max(
                    np.abs(
                        cell_actions["future_donor_cache_donor"]
                        - cell_actions["future_recipient_cache_donor"]
                    )
                )
            )
            recipient_cache_invariance.append(recipient_invariance)
            donor_cache_invariance.append(donor_invariance)
            for condition in CACHE_FACTORIAL_CONDITIONS:
                run = run_index[(recipient.branch_id, donor.branch_id, condition)]
                base_error = None
                reference_condition = base_reference_condition.get(condition)
                if reference_condition is not None:
                    reference_donor = (
                        None
                        if reference_condition == FastWAMCondition.SELF_CACHE.value
                        else donor.branch_id
                    )
                    reference = base_actions[
                        (reference_condition, recipient.branch_id, reference_donor)
                    ]
                    base_error = float(
                        np.max(np.abs(cell_actions[condition] - reference))
                    )
                    state_base_errors[condition].append(base_error)
                _write_factorial_run(
                    run_dir=run_dir,
                    run=run,
                    manifest=factorial_manifest,
                    action_model=cell_actions[condition],
                    action_env=cell_env_actions[condition],
                    metadata={
                        "task_description": task_description,
                        "duration_seconds": cell_durations[condition],
                        "base_reference_condition": reference_condition,
                        "base_reference_max_abs_error": base_error,
                        "same_recipient_cache_future_swap_max_abs_error": recipient_invariance,
                        "same_donor_cache_future_swap_max_abs_error": donor_invariance,
                        "metrics_model_space": action_metrics(
                            cell_actions[condition],
                            native_actions[recipient.branch_id],
                            native_actions[donor.branch_id],
                            native_actions,
                            donor_id=donor.branch_id,
                        ),
                    },
                )

    summary = {
        "status": "complete",
        "manifest_id": factorial_manifest["manifest_id"],
        "base_manifest_id": factorial_manifest["base_manifest_id"],
        "state_id": state.state_id,
        "task_description": task_description,
        "completed_conditions": list(CACHE_FACTORIAL_CONDITIONS),
        "run_count": len(all_runs),
        "native_action_regeneration_max_abs_error": max(
            native_action_replay_errors.values()
        ),
        "stored_native_latent_regeneration_max_abs_error": max(
            stored_latent_replay_errors.values()
        ),
        "base_reference_max_abs_error": {
            condition: max(values) if values else None
            for condition, values in state_base_errors.items()
        },
        "recipient_cache_future_swap_global_max_abs_error": max(
            recipient_cache_invariance
        ),
        "donor_cache_future_swap_global_max_abs_error": max(donor_cache_invariance),
        "cache_descriptors": cache_meta,
    }
    atomic_write_json(state_root / "summary.json", summary)


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.exact_tolerance < 0:
        raise ValueError("exact-tolerance must be nonnegative")
    factorial = load_cache_factorial_manifest(args.factorial_manifest)
    base = validate_factorial_parent(factorial, args.base_manifest)
    validation_args = SimpleNamespace(
        fastwam_root=args.fastwam_root,
        manifest=args.base_manifest,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        device=args.device,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    _validate_paths(validation_args)
    base_audit, _ = audit_fastwam_outputs(
        base,
        args.base_output_root,
        required_state_count=len(base["states"]),
    )
    if not base_audit["all_frozen_outputs_complete"]:
        raise RuntimeError(
            "refusing factorial launch until every parent powered arm and state summary completes"
        )
    output_root = args.output_root.resolve()
    write_cache_factorial_manifest(
        output_root / factorial["manifest_id"] / "manifest.json", factorial
    )
    cfg, model, processor, obs_to_model_input, denormalize_action, _ = _load_fastwam_stack(
        validation_args, base
    )
    base_root = _manifest_root(args.base_output_root, base["manifest_id"])
    states = states_from_factorial_manifest(factorial)
    for state in states[args.shard_index :: args.shard_count]:
        _run_state(
            state=state,
            factorial_manifest=factorial,
            base_manifest=base,
            cfg=cfg,
            model=model,
            processor=processor,
            obs_to_model_input=obs_to_model_input,
            denormalize_action=denormalize_action,
            base_root=base_root,
            output_root=output_root,
            exact_tolerance=args.exact_tolerance,
        )


if __name__ == "__main__":
    main()
