#!/usr/bin/env python3
"""Run the frozen FastWAM Optional-IDM intervention matrix on LIBERO states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from imagined_future.fastwam_optional_idm import (
    CORE_CONDITIONS,
    FASTWAM_CHECKPOINT_SHA256,
    FASTWAM_STATS_SHA256,
    FASTWAM_UPSTREAM_COMMIT,
    FastWAMCondition,
    FastWAMRunSpec,
    FastWAMStateSpec,
    action_metrics,
    atomic_write_json,
    atomic_write_npz,
    cache_descriptor,
    expand_run_specs,
    load_frozen_manifest,
    shuffled_kv_cache,
    state_from_dict,
    write_frozen_manifest,
)


@dataclass
class RecordedBranch:
    branch_id: str
    video_seed: int
    action_seed: int
    action_model: np.ndarray
    action_env: np.ndarray
    video_latent: torch.Tensor
    cache_k: list[torch.Tensor]
    cache_v: list[torch.Tensor]
    cache_meta: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--conditions",
        default=",".join(condition.value for condition in FastWAMCondition),
        help="Comma-separated subset; native branches are always computed for references.",
    )
    parser.add_argument(
        "--persist-native-kv",
        action="store_true",
        help="Persist full native K/V tensors. This can require >1 GB per branch.",
    )
    return parser.parse_args()


def _validate_paths(args: argparse.Namespace) -> None:
    for label, path in (
        ("FastWAM checkout", args.fastwam_root),
        ("manifest", args.manifest),
        ("checkpoint", args.checkpoint),
        ("dataset stats", args.dataset_stats),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not (args.fastwam_root / ".git").is_dir():
        raise ValueError(f"not a FastWAM git checkout: {args.fastwam_root}")
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(args.fastwam_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != FASTWAM_UPSTREAM_COMMIT:
        raise ValueError(
            f"FastWAM checkout is {commit}; frozen manifest requires {FASTWAM_UPSTREAM_COMMIT}"
        )
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    for label, path, expected in (
        ("checkpoint", args.checkpoint, FASTWAM_CHECKPOINT_SHA256),
        ("dataset stats", args.dataset_stats, FASTWAM_STATS_SHA256),
    ):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
            )


def _load_fastwam_stack(args: argparse.Namespace, manifest: dict[str, Any]):
    sys.path.insert(0, str(args.fastwam_root))
    # The pinned official evaluator imports its sibling action_ensembler module
    # as a top-level module, matching direct script execution.
    sys.path.insert(0, str(args.fastwam_root / "experiments" / "libero"))
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str((args.fastwam_root / "configs").resolve()),
    ):
        cfg = compose(
            config_name="sim_libero.yaml",
            overrides=[
                "task=libero_optional_idm_2cam224_1e-4",
                f"ckpt={args.checkpoint.resolve()}",
                f"EVALUATION.dataset_stats_path={args.dataset_stats.resolve()}",
                f"EVALUATION.device={args.device}",
                "EVALUATION.compile_action_infer=false",
                f"EVALUATION.sigma_shift={float(manifest['inference']['sigma_shift'])}",
            ],
        )
    from experiments.libero.eval_libero_single import (
        _denormalize_action,
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _obs_to_model_input,
    )
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    _load_model_checkpoint(model, str(args.checkpoint))
    model = model.to(args.device).eval()
    stats = load_dataset_stats_from_json(str(args.dataset_stats))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    return cfg, model, processor, _obs_to_model_input, _denormalize_action, OmegaConf


def _prepare_state_input(
    *,
    state: FastWAMStateSpec,
    cfg,
    model,
    processor,
    obs_to_model_input,
):
    from experiments.libero.libero_utils import (
        LIBERO_ENV_RESOLUTION,
        get_libero_dummy_action,
        get_libero_env,
    )
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[state.suite]()
    task = suite.get_task(state.task_id)
    initial_states_path = (
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    )
    initial_states = torch.load(initial_states_path, weights_only=False)
    if state.initial_state_index >= len(initial_states):
        raise IndexError(
            f"{state.state_id} requests initial state {state.initial_state_index}, "
            f"but {initial_states_path} contains {len(initial_states)}"
        )
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=0)
    try:
        env.reset()
        observation = env.set_init_state(initial_states[state.initial_state_index])
        for _ in range(state.wait_steps):
            observation, _, _, _ = env.step(get_libero_dummy_action())
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()

    video_size = cfg.data.train.get("video_size", [224, 448])
    input_h, input_w = int(video_size[0]), int(video_size[1])
    image, proprio, _ = obs_to_model_input(
        observation,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=str(model.device),
        dtype=model.torch_dtype,
    )
    prompt = DEFAULT_PROMPT.format(task=task_description)
    with torch.no_grad():
        context, context_mask = model.encode_prompt(prompt)
    return image, proprio, context, context_mask, task_description


def _to_env_action(action: np.ndarray, denormalize_action, processor) -> np.ndarray:
    from experiments.libero.libero_utils import invert_gripper_action

    env_action = denormalize_action(torch.from_numpy(action), processor)[0]
    env_action[..., -1] = env_action[..., -1] * 2 - 1
    env_action = invert_gripper_action(env_action)
    return np.asarray(env_action, dtype=np.float32)


def _infer_idm(
    *,
    model,
    common: dict[str, Any],
    video_seed: int,
    action_seed: int,
    video_latent: torch.Tensor | None = None,
    cache: tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]] | None = None,
    return_intermediates: bool = False,
) -> dict[str, Any]:
    with torch.no_grad():
        return model.infer_action(
            **common,
            action_infer_mode="idm",
            video_seed=int(video_seed),
            action_seed=int(action_seed),
            video_latents_override=video_latent,
            video_cache_override=cache,
            return_intermediates=return_intermediates,
        )


def _infer_first_frame(
    *, model, common: dict[str, Any], video_seed: int, action_seed: int
) -> dict[str, Any]:
    with torch.no_grad():
        return model.infer_action(
            **common,
            action_infer_mode="first_frame",
            video_seed=int(video_seed),
            action_seed=int(action_seed),
            return_intermediates=True,
        )


def _write_cache(path: Path, branch: RecordedBranch) -> None:
    arrays: dict[str, np.ndarray] = {}
    for index, tensor in enumerate(branch.cache_k):
        arrays[f"k_{index:02d}"] = tensor.float().numpy().astype(np.float16)
    for index, tensor in enumerate(branch.cache_v):
        arrays[f"v_{index:02d}"] = tensor.float().numpy().astype(np.float16)
    atomic_write_npz(path, arrays)


def _write_run(
    *,
    run_dir: Path,
    run: FastWAMRunSpec,
    manifest_id: str,
    action_model: np.ndarray,
    action_env: np.ndarray,
    metadata: dict[str, Any],
    video_latent: torch.Tensor | None = None,
) -> None:
    json_path = run_dir / f"{run.run_id}.json"
    npz_path = run_dir / f"{run.run_id}.npz"
    if json_path.exists() and npz_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing.get("run", {}).get("run_id") != run.run_id:
            raise RuntimeError(f"run ID collision at {json_path}")
        return
    arrays: dict[str, np.ndarray] = {
        "action_model": np.asarray(action_model, dtype=np.float32),
        "action_env": np.asarray(action_env, dtype=np.float32),
    }
    if video_latent is not None:
        arrays["video_latent"] = video_latent.float().numpy().astype(np.float16)
    atomic_write_npz(npz_path, arrays)
    atomic_write_json(
        json_path,
        {
            "status": "complete",
            "manifest_id": manifest_id,
            "upstream_commit": FASTWAM_UPSTREAM_COMMIT,
            "run": run.to_dict(),
            "array_file": npz_path.name,
            **metadata,
        },
    )


def _run_state(
    *,
    state: FastWAMStateSpec,
    manifest: dict[str, Any],
    cfg,
    model,
    processor,
    obs_to_model_input,
    denormalize_action,
    output_root: Path,
    selected_conditions: set[str],
    persist_native_kv: bool,
) -> None:
    state_root = output_root / manifest["manifest_id"] / state.state_id
    run_dir = state_root / "runs"
    artifact_dir = state_root / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    image, proprio, context, context_mask, task_description = _prepare_state_input(
        state=state,
        cfg=cfg,
        model=model,
        processor=processor,
        obs_to_model_input=obs_to_model_input,
    )
    inference = manifest["inference"]
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
    atomic_write_npz(
        artifact_dir / "state_input.npz",
        {
            "input_image": image.detach().cpu().float().numpy().astype(np.float16),
            "proprio": proprio.detach().cpu().float().numpy(),
        },
    )

    all_runs = expand_run_specs(manifest["manifest_id"], state)
    run_by_native = {
        run.recipient_id: run
        for run in all_runs
        if run.condition == FastWAMCondition.NATIVE.value
    }
    recorded: dict[str, RecordedBranch] = {}
    for branch in state.branches:
        started = time.time()
        output = _infer_idm(
            model=model,
            common=common,
            video_seed=branch.video_seed,
            action_seed=branch.action_seed,
            return_intermediates=True,
        )
        action_model = output["action"].detach().cpu().numpy().astype(np.float32)
        action_env = _to_env_action(action_model, denormalize_action, processor)
        video_latent = output["video_latents"].detach().cpu()
        cache_k = [tensor.detach().cpu() for tensor in output["video_cache_k"]]
        cache_v = [tensor.detach().cpu() for tensor in output["video_cache_v"]]
        cache_meta = cache_descriptor(cache_k, cache_v)
        record = RecordedBranch(
            branch_id=branch.branch_id,
            video_seed=branch.video_seed,
            action_seed=branch.action_seed,
            action_model=action_model,
            action_env=action_env,
            video_latent=video_latent,
            cache_k=cache_k,
            cache_v=cache_v,
            cache_meta=cache_meta,
        )
        recorded[branch.branch_id] = record
        if persist_native_kv:
            _write_cache(artifact_dir / f"{branch.branch_id}_kv.npz", record)
        native_run = run_by_native[branch.branch_id]
        if FastWAMCondition.NATIVE.value in selected_conditions:
            _write_run(
                run_dir=run_dir,
                run=native_run,
                manifest_id=manifest["manifest_id"],
                action_model=action_model,
                action_env=action_env,
                video_latent=video_latent,
                metadata={
                    "task_description": task_description,
                    "duration_seconds": time.time() - started,
                    "cache": cache_meta,
                },
            )
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    native_actions = {key: branch.action_model for key, branch in recorded.items()}
    intervention_runs = [
        run
        for run in all_runs
        if run.condition != FastWAMCondition.NATIVE.value
        and run.condition in selected_conditions
    ]
    first_frame_actions: dict[str, list[np.ndarray]] = {}
    for run in intervention_runs:
        started = time.time()
        recipient = recorded[run.recipient_id]
        source = recorded[run.source_id] if run.source_id is not None else None
        if run.condition == FastWAMCondition.FIRST_FRAME.value:
            assert source is not None
            output = _infer_first_frame(
                model=model,
                common=common,
                video_seed=source.video_seed,
                action_seed=recipient.action_seed,
            )
        elif run.condition == FastWAMCondition.SELF_LATENT.value:
            output = _infer_idm(
                model=model,
                common=common,
                video_seed=recipient.video_seed,
                action_seed=recipient.action_seed,
                video_latent=recipient.video_latent,
            )
        elif run.condition == FastWAMCondition.SELF_CACHE.value:
            output = _infer_idm(
                model=model,
                common=common,
                video_seed=recipient.video_seed,
                action_seed=recipient.action_seed,
                video_latent=recipient.video_latent,
                cache=(recipient.cache_k, recipient.cache_v),
            )
        elif run.condition in {
            FastWAMCondition.DONOR_LATENT.value,
            FastWAMCondition.WRONG_LATENT.value,
        }:
            assert source is not None
            output = _infer_idm(
                model=model,
                common=common,
                video_seed=source.video_seed,
                action_seed=recipient.action_seed,
                video_latent=source.video_latent,
            )
        elif run.condition == FastWAMCondition.DONOR_CACHE.value:
            assert source is not None
            output = _infer_idm(
                model=model,
                common=common,
                video_seed=recipient.video_seed,
                action_seed=recipient.action_seed,
                video_latent=recipient.video_latent,
                cache=(source.cache_k, source.cache_v),
            )
        elif run.condition == FastWAMCondition.SHUFFLED_CACHE.value:
            assert source is not None and run.shuffle_seed is not None
            shuffled = shuffled_kv_cache(
                source.cache_k,
                source.cache_v,
                seed=run.shuffle_seed,
            )
            output = _infer_idm(
                model=model,
                common=common,
                video_seed=recipient.video_seed,
                action_seed=recipient.action_seed,
                video_latent=recipient.video_latent,
                cache=shuffled,
            )
            del shuffled
        else:
            raise ValueError(f"unimplemented condition: {run.condition}")

        action_model = output["action"].detach().cpu().numpy().astype(np.float32)
        action_env = _to_env_action(action_model, denormalize_action, processor)
        metadata: dict[str, Any] = {
            "task_description": task_description,
            "duration_seconds": time.time() - started,
            "native_recipient_max_abs_error": float(
                np.max(np.abs(action_model - recipient.action_model))
            ),
        }
        if run.condition in {
            FastWAMCondition.SELF_LATENT.value,
            FastWAMCondition.SELF_CACHE.value,
        }:
            metadata["self_replay_max_abs_error"] = metadata[
                "native_recipient_max_abs_error"
            ]
        if run.condition == FastWAMCondition.FIRST_FRAME.value:
            first_frame_actions.setdefault(run.recipient_id, []).append(action_model)
        if run.donor_id is not None:
            metadata["metrics_model_space"] = action_metrics(
                action_model,
                recipient.action_model,
                recorded[run.donor_id].action_model,
                native_actions,
                donor_id=run.donor_id,
            )
        _write_run(
            run_dir=run_dir,
            run=run,
            manifest_id=manifest["manifest_id"],
            action_model=action_model,
            action_env=action_env,
            metadata=metadata,
        )
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    latent_matrix = np.stack(
        [branch.video_latent.float().numpy().reshape(-1) for branch in recorded.values()]
    )
    action_matrix = np.stack([branch.action_model.reshape(-1) for branch in recorded.values()])
    latent_distances = np.linalg.norm(latent_matrix[:, None] - latent_matrix[None, :], axis=-1)
    action_distances = np.linalg.norm(action_matrix[:, None] - action_matrix[None, :], axis=-1)
    first_frame_invariance = {}
    for recipient_id, actions in first_frame_actions.items():
        stack = np.stack(actions)
        first_frame_invariance[recipient_id] = float(
            np.max(np.abs(stack - stack[0:1]))
        )
    summary_path = state_root / "summary.json"
    previous_summary: dict[str, Any] = {}
    if summary_path.exists():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous_summary.get("manifest_id") != manifest["manifest_id"]:
            raise RuntimeError(f"summary manifest mismatch at {summary_path}")
    previous_conditions = set(previous_summary.get("completed_conditions", []))
    previous_invariance = previous_summary.get(
        "first_frame_donor_seed_max_abs_error", {}
    )
    if not first_frame_invariance and isinstance(previous_invariance, dict):
        first_frame_invariance = {
            str(key): float(value) for key, value in previous_invariance.items()
        }
    completed_conditions = previous_conditions | selected_conditions
    summary = {
        "status": (
            "complete"
            if set(CORE_CONDITIONS).issubset(completed_conditions)
            else "partial"
        ),
        "manifest_id": manifest["manifest_id"],
        "state_id": state.state_id,
        "task_description": task_description,
        "branch_ids": list(recorded),
        "native_video_latent_pairwise_l2": latent_distances.tolist(),
        "native_action_pairwise_l2": action_distances.tolist(),
        "first_frame_donor_seed_max_abs_error": first_frame_invariance,
        "first_frame_donor_seed_global_max_abs_error": (
            max(first_frame_invariance.values()) if first_frame_invariance else None
        ),
        "completed_conditions": sorted(completed_conditions),
    }
    atomic_write_json(summary_path, summary)


def main() -> None:
    args = parse_args()
    _validate_paths(args)
    manifest = load_frozen_manifest(args.manifest)
    selected_conditions = {
        item.strip() for item in args.conditions.split(",") if item.strip()
    }
    unknown = selected_conditions - {condition.value for condition in FastWAMCondition}
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    output_root = args.output_root.resolve()
    write_frozen_manifest(output_root / manifest["manifest_id"] / "manifest.json", manifest)
    cfg, model, processor, obs_to_model_input, denormalize_action, _ = _load_fastwam_stack(
        args, manifest
    )
    states = tuple(state_from_dict(value) for value in manifest["states"])
    selected_states = states[args.shard_index :: args.shard_count]
    for state in selected_states:
        _run_state(
            state=state,
            manifest=manifest,
            cfg=cfg,
            model=model,
            processor=processor,
            obs_to_model_input=obs_to_model_input,
            denormalize_action=denormalize_action,
            output_root=output_root,
            selected_conditions=selected_conditions,
            persist_native_kv=args.persist_native_kv,
        )


if __name__ == "__main__":
    main()
