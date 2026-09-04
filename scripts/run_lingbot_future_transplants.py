#!/usr/bin/env python3
"""Run controlled donor-future and future-cache interventions on LingBot-VA.

The runner uses only the first autoregressive chunk.  Observation, instruction,
and recipient action noise are held fixed while the generated future source is
crossed with the action-noise source.  The action coordinates are never written
by an intervention.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange


UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--admission", choices=("development", "evaluation"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--run-factorial", action="store_true")
    parser.add_argument("--gaussian-controls", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    torch.save(value, temporary)
    os.replace(temporary, path)


def _cache_blocks(server) -> list[Any]:
    return [block.attn1 for block in server.transformer.blocks]


def snapshot_cache(server) -> list[dict[str, torch.Tensor]]:
    snapshot = []
    patch_t, patch_h, patch_w = server.job_config.patch_size
    expected_slots = (
        server.job_config.frame_chunk_size
        * server.latent_height
        * server.latent_width
    ) // (patch_t * patch_h * patch_w)
    reference_slots = None
    reference_ids = None
    for attention in _cache_blocks(server):
        cache = attention.attn_caches[server.cache_name]
        valid = cache["mask"].nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() != expected_slots:
            raise RuntimeError(
                f"predicted cache has {valid.numel()} valid slots; expected {expected_slots}"
            )
        if not bool(cache["is_pred"][valid].all().item()):
            raise RuntimeError("predicted cache includes slots not marked as predicted")
        if reference_slots is None:
            reference_slots = valid.detach().cpu()
            reference_ids = cache["id"][valid].detach().cpu()
        elif not torch.equal(reference_slots, valid.detach().cpu()) or not torch.equal(
            reference_ids, cache["id"][valid].detach().cpu()
        ):
            raise RuntimeError("predicted-cache slot/id topology differs across layers")
        snapshot.append(
            {
                "slots": valid.detach().cpu(),
                "k": cache["k"][:, valid].detach().cpu(),
                "v": cache["v"][:, valid].detach().cpu(),
                "id": cache["id"][valid].detach().cpu(),
                "is_pred": cache["is_pred"][valid].detach().cpu(),
            }
        )
    return snapshot


def cache_hash(snapshot: list[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for layer, item in enumerate(snapshot):
        digest.update(str(layer).encode())
        for field in ("slots", "k", "v", "id", "is_pred"):
            digest.update(field.encode())
            digest.update(tensor_hash(item[field]).encode())
    return digest.hexdigest()


def restore_cache(server, snapshot: list[dict[str, torch.Tensor]]) -> None:
    blocks = _cache_blocks(server)
    if len(blocks) != len(snapshot):
        raise ValueError("cache layer count mismatch")
    for attention, source in zip(blocks, snapshot):
        cache = attention.attn_caches[server.cache_name]
        cache["mask"].fill_(False)
        cache["id"].fill_(-1)
        cache["is_pred"].fill_(False)
        slots = source["slots"].to(device=cache["mask"].device)
        cache["k"][:, slots] = source["k"].to(cache["k"])
        cache["v"][:, slots] = source["v"].to(cache["v"])
        cache["id"][slots] = source["id"].to(cache["id"])
        cache["mask"][slots] = True
        cache["is_pred"][slots] = source["is_pred"].to(cache["is_pred"])


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def sample_action_noise(server, seed: int) -> torch.Tensor:
    return torch.randn(
        1,
        server.job_config.action_dim,
        server.job_config.frame_chunk_size,
        server.action_per_frame,
        1,
        device=server.device,
        dtype=server.dtype,
        generator=_generator(server.device, seed),
    ).detach().cpu()


@torch.no_grad()
def generate_future(server, init_latent_cpu: torch.Tensor, video_seed: int):
    from wan_va.utils import data_seq_to_patch

    frame_st_id = 0
    frame_chunk_size = server.job_config.frame_chunk_size
    init_latent = init_latent_cpu.to(device=server.device, dtype=server.dtype)
    server.init_latent = init_latent
    latents = torch.randn(
        1,
        48,
        frame_chunk_size,
        server.latent_height,
        server.latent_width,
        device=server.device,
        dtype=server.dtype,
        generator=_generator(server.device, video_seed),
    )
    server.scheduler.set_timesteps(server.job_config.num_inference_steps)
    # Match VA_Server._infer exactly: the extra t=0 forward pass does not take
    # a scheduler step; it materializes the final predicted-future K/V cache.
    timesteps = torch.nn.functional.pad(
        server.scheduler.timesteps, (0, 1), mode="constant", value=0
    )
    video_step = server.job_config.video_exec_step
    if video_step != -1:
        timesteps = timesteps[:video_step]

    server.transformer.clear_pred_cache(server.cache_name)
    for index, timestep in enumerate(timesteps):
        last_step = index == len(timesteps) - 1
        latent_cond = init_latent[:, :, 0:1].to(server.dtype)
        model_input = server._prepare_latent_input(
            latents,
            None,
            timestep,
            timestep,
            latent_cond,
            None,
            frame_st_id=frame_st_id,
        )["latent_res_lst"]
        prediction = server.transformer(
            server._repeat_input_for_cfg(model_input),
            update_cache=1 if last_step else 0,
            cache_name=server.cache_name,
            action_mode=False,
        )
        if not last_step or video_step != -1:
            prediction = data_seq_to_patch(
                server.job_config.patch_size,
                prediction,
                frame_chunk_size,
                server.latent_height,
                server.latent_width,
                batch_size=2 if server.use_cfg else 1,
            )
            if server.job_config.guidance_scale > 1:
                prediction = prediction[1:] + server.job_config.guidance_scale * (
                    prediction[:1] - prediction[1:]
                )
            else:
                prediction = prediction[:1]
            latents = server.scheduler.step(
                prediction, timestep, latents, return_dict=False
            )
        latents[:, :, 0:1] = latent_cond
    return latents.detach().cpu(), snapshot_cache(server), init_latent.detach().cpu()


@torch.no_grad()
def install_future(
    server, init_latent_cpu: torch.Tensor, future: torch.Tensor
) -> torch.Tensor:
    """Recompute the future K/V cache from a recorded clean future latent."""
    init_latent = init_latent_cpu.to(device=server.device, dtype=server.dtype)
    server.init_latent = init_latent
    value = future.to(device=server.device, dtype=server.dtype).clone()
    if value.shape[2] != server.job_config.frame_chunk_size:
        raise ValueError("future frame count mismatch")
    present_error = float(
        (value[:, :, 0:1] - init_latent[:, :, 0:1].to(server.dtype))
        .abs()
        .max()
        .item()
    )
    if present_error != 0.0:
        raise RuntimeError(
            f"recorded future present frame differs from fixed observation: {present_error}"
        )
    value[:, :, 0:1] = init_latent[:, :, 0:1].to(server.dtype)
    server.transformer.clear_pred_cache(server.cache_name)
    model_input = server._prepare_latent_input(
        value,
        None,
        0,
        0,
        init_latent[:, :, 0:1].to(server.dtype),
        None,
        frame_st_id=0,
    )["latent_res_lst"]
    server.transformer(
        server._repeat_input_for_cfg(model_input),
        update_cache=1,
        cache_name=server.cache_name,
        action_mode=False,
    )
    return init_latent.detach().cpu()


@torch.no_grad()
def generate_action(server, action_noise_cpu: torch.Tensor) -> np.ndarray:
    frame_chunk_size = server.job_config.frame_chunk_size
    expected_shape = (
        1,
        server.job_config.action_dim,
        frame_chunk_size,
        server.action_per_frame,
        1,
    )
    if tuple(action_noise_cpu.shape) != expected_shape:
        raise ValueError(
            f"action noise shape {tuple(action_noise_cpu.shape)} != {expected_shape}"
        )
    actions = action_noise_cpu.to(device=server.device, dtype=server.dtype).clone()
    server.action_scheduler.set_timesteps(server.job_config.action_num_inference_steps)
    # Match VA_Server._infer exactly.  The final t=0 call records action K/V and
    # intentionally skips a scheduler update.
    timesteps = torch.nn.functional.pad(
        server.action_scheduler.timesteps,
        (0, 1),
        mode="constant",
        value=0,
    )
    action_cond = torch.zeros(
        [1, server.job_config.action_dim, 1, server.action_per_frame, 1],
        device=server.device,
        dtype=server.dtype,
    )
    for index, timestep in enumerate(timesteps):
        last_step = index == len(timesteps) - 1
        model_input = server._prepare_latent_input(
            None,
            actions,
            timestep,
            timestep,
            None,
            action_cond,
            frame_st_id=0,
        )["action_res_lst"]
        prediction = server.transformer(
            server._repeat_input_for_cfg(model_input),
            update_cache=1 if last_step else 0,
            cache_name=server.cache_name,
            action_mode=True,
        )
        if not last_step:
            prediction = rearrange(
                prediction,
                "b (f n) c -> b c f n 1",
                f=frame_chunk_size,
            )
            if server.job_config.action_guidance_scale > 1:
                prediction = prediction[1:] + server.job_config.action_guidance_scale * (
                    prediction[:1] - prediction[1:]
                )
            else:
                prediction = prediction[:1]
            actions = server.action_scheduler.step(
                prediction, timestep, actions, return_dict=False
            )
        actions[:, :, 0:1] = action_cond
    actions[:, ~server.action_mask] *= 0
    return np.asarray(server.postprocess_action(actions), dtype=np.float32)


def reset(server, prompt: str) -> None:
    server._reset(prompt=prompt)


def reset_with_frozen_prompt(
    server,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
) -> None:
    """Reset model/cache state without repeatedly re-running the text encoder."""
    server._reset(prompt=None)
    server.prompt_embeds = prompt_embeds
    server.negative_prompt_embeds = negative_prompt_embeds


def gaussian_future(reference: torch.Tensor, present: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    future = reference.clone().float()
    region = reference[:, :, 1:].float()
    noise = torch.randn(region.shape, generator=generator)
    noise = (noise - noise.mean()) / noise.std().clamp_min(1e-8)
    noise = noise * region.std() + region.mean()
    future[:, :, 1:] = noise
    future[:, :, 0:1] = present[:, :, 0:1].float()
    return future.to(reference.dtype)


def executed_action_view(actions: np.ndarray) -> np.ndarray:
    """Drop LingBot's conditioned first action frame, which LIBERO never executes."""
    value = np.asarray(actions)
    if value.shape[-2] != 4:
        raise ValueError(f"expected four action frames, got shape {value.shape}")
    return value[..., 1:, :]


def run_state(server, record: dict[str, Any], manifest: dict[str, Any], args) -> None:
    state_id = record["state_id"]
    state_root = args.output_root / state_id
    complete_path = state_root / "result.json"
    arrays_path = state_root / "actions.npz"
    if complete_path.exists() and arrays_path.exists():
        existing = json.loads(complete_path.read_text())
        if existing.get("status") == "complete":
            resume_checks = {
                "input_sha256": record["input_sha256"],
                "manifest_sha256": args.manifest_sha256,
                "runner_sha256": args.runner_sha256,
                "upstream_commit": UPSTREAM_COMMIT,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "gaussian_controls": bool(args.gaussian_controls),
                "run_factorial": bool(args.run_factorial),
                "actions_sha256": sha256_file(arrays_path),
            }
            mismatches = {
                key: (existing.get(key), expected)
                for key, expected in resume_checks.items()
                if existing.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    f"refusing unsafe resume for {state_id}: {mismatches}"
                )
            print(f"skip verified complete {state_id}", flush=True)
            return
    observation_path = Path(record["observation_path"])
    observed_input_sha256 = sha256_file(observation_path)
    if observed_input_sha256 != record["input_sha256"]:
        raise RuntimeError(
            f"frozen input hash mismatch for {state_id}: "
            f"{observed_input_sha256} != {record['input_sha256']}"
        )
    observation = np.load(observation_path, allow_pickle=False)
    obs = {
        "obs": [
            {
                "observation.images.agentview_rgb": observation["agentview"],
                "observation.images.eye_in_hand_rgb": observation["wrist"],
            }
        ]
    }
    prompt = str(record["prompt"])
    video_seeds = [int(value) for value in manifest["video_seeds"]]
    action_seeds = [int(value) for value in manifest["action_seeds"]]
    branch_ids = list(manifest["branch_ids"])
    if not (
        len(branch_ids) == len(video_seeds) == len(action_seeds) == 4
        and len(set(branch_ids)) == len(set(video_seeds)) == len(set(action_seeds)) == 4
    ):
        raise RuntimeError(
            "confirmatory contract requires four unique branch IDs, video seeds, "
            "and action seeds"
        )
    branch_count = 4
    futures: list[torch.Tensor] = []
    caches: list[list[dict[str, torch.Tensor]]] = []
    native_actions: list[np.ndarray] = []
    metadata: dict[str, Any] = {
        "status": "running",
        "state_id": state_id,
        "admission": record["admission"],
        "input_sha256": record["input_sha256"],
        "prompt": prompt,
        "video_seeds": video_seeds,
        "action_seeds": action_seeds,
        "branch_ids": branch_ids,
        "grid_axis_0": "recipient_action_noise_source",
        "grid_axis_1": "future_source",
        "future_hashes": [],
        "cache_hashes": [],
        "action_noise_hashes": [],
        "durations_seconds": {},
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "manifest_sha256": args.manifest_sha256,
        "runner_sha256": args.runner_sha256,
        "gaussian_controls": bool(args.gaussian_controls),
        "run_factorial": bool(args.run_factorial),
        "intervention_target": "predicted-video latent/cache only",
        "action_coordinate_intervention": "none",
        "first_action_frame_executed_by_libero": False,
    }
    state_started = time.time()
    state_root.mkdir(parents=True, exist_ok=True)

    # Freeze state and language conditioning once.  This avoids mutating the
    # streaming VAE context across cells and removes repeated encoder noise.
    reset(server, prompt)
    init_latent = server._encode_obs(obs).detach().cpu()
    prompt_embeds = server.prompt_embeds.detach()
    negative_prompt_embeds = (
        server.negative_prompt_embeds.detach()
        if server.negative_prompt_embeds is not None
        else None
    )
    action_noises = [sample_action_noise(server, seed) for seed in action_seeds]
    action_noise_hashes = [tensor_hash(value) for value in action_noises]
    metadata["action_noise_hashes"] = action_noise_hashes
    metadata["grid_action_noise_hashes"] = [
        [action_noise_hashes[recipient]] * branch_count
        for recipient in range(branch_count)
    ]
    atomic_torch(
        state_root / "frozen_inputs.pt",
        {
            "init_latent": init_latent,
            "action_noises": torch.stack(action_noises),
            "action_noise_hashes": action_noise_hashes,
        },
    )

    for branch_index in range(branch_count):
        video_seed = video_seeds[branch_index]
        started = time.time()
        reset_with_frozen_prompt(server, prompt_embeds, negative_prompt_embeds)
        future, cache, present = generate_future(server, init_latent, video_seed)
        action = generate_action(server, action_noises[branch_index])
        futures.append(future)
        caches.append(cache)
        native_actions.append(action)
        metadata["future_hashes"].append(tensor_hash(future))
        metadata["cache_hashes"].append(cache_hash(cache))
        metadata["durations_seconds"][f"native_b{branch_index}"] = time.time() - started
        atomic_torch(
            state_root / f"future_b{branch_index}.pt",
            {"future": future, "video_seed": video_seed},
        )

    native = np.stack(native_actions)
    latent_grid = np.empty((branch_count, branch_count) + native.shape[1:], dtype=np.float32)
    cache_replay = np.empty_like(native)
    for recipient in range(branch_count):
        for source in range(branch_count):
            started = time.time()
            reset_with_frozen_prompt(server, prompt_embeds, negative_prompt_embeds)
            install_future(server, init_latent, futures[source])
            latent_grid[recipient, source] = generate_action(
                server, action_noises[recipient]
            )
            metadata["durations_seconds"][f"latent_r{recipient}_s{source}"] = (
                time.time() - started
            )
        reset_with_frozen_prompt(server, prompt_embeds, negative_prompt_embeds)
        install_future(server, init_latent, futures[recipient])
        restore_cache(server, caches[recipient])
        cache_replay[recipient] = generate_action(server, action_noises[recipient])

    arrays: dict[str, np.ndarray] = {
        "native_actions": native,
        "latent_grid_actions": latent_grid,
        "cache_replay_actions": cache_replay,
        "native_executed_actions": executed_action_view(native),
        "latent_grid_executed_actions": executed_action_view(latent_grid),
        "cache_replay_executed_actions": executed_action_view(cache_replay),
    }
    self_latent = np.stack(
        [latent_grid[index, index] for index in range(branch_count)]
    )
    metadata["native_self_latent_max_abs_error"] = float(
        np.max(np.abs(self_latent - native))
    )
    metadata["native_self_cache_max_abs_error"] = float(
        np.max(np.abs(cache_replay - native))
    )
    metadata["native_future_hashes_unique"] = len(set(metadata["future_hashes"]))
    metadata["native_cache_hashes_unique"] = len(set(metadata["cache_hashes"]))
    native_flat = native.reshape(branch_count, -1).astype(np.float64)
    native_separations = [
        float(np.linalg.norm(native_flat[i] - native_flat[j]))
        for i in range(branch_count)
        for j in range(i + 1, branch_count)
    ]
    metadata["native_action_pairwise_distance_min"] = min(native_separations)
    metadata["native_action_pairwise_distance_median"] = float(
        np.median(native_separations)
    )

    control_failures = []
    if not all(np.isfinite(value).all() for value in arrays.values()):
        control_failures.append("non-finite action output")
    if metadata["native_self_latent_max_abs_error"] != 0.0:
        control_failures.append("native/self future replay is not bitwise exact")
    if metadata["native_self_cache_max_abs_error"] != 0.0:
        control_failures.append("native/cache replay is not bitwise exact")
    if metadata["native_future_hashes_unique"] != 4:
        control_failures.append("four video seeds did not produce four unique futures")
    if metadata["native_cache_hashes_unique"] != 4:
        control_failures.append("four futures did not produce four unique caches")
    if latent_grid.shape[:2] != (4, 4):
        control_failures.append("future-source grid is not 4x4")
    if control_failures:
        metadata["status"] = "failed_control"
        metadata["control_failures"] = control_failures
        metadata["duration_seconds"] = time.time() - state_started
        atomic_npz(arrays_path, **arrays)
        metadata["actions_sha256"] = sha256_file(arrays_path)
        atomic_json(complete_path, metadata)
        raise RuntimeError(f"{state_id} failed controls: {control_failures}")

    if args.gaussian_controls:
        gaussian_actions = np.empty_like(native)
        for recipient in range(branch_count):
            reset_with_frozen_prompt(server, prompt_embeds, negative_prompt_embeds)
            control = gaussian_future(
                futures[recipient], init_latent, seed=900_000 + recipient
            )
            install_future(server, init_latent, control)
            gaussian_actions[recipient] = generate_action(
                server, action_noises[recipient]
            )
        arrays["gaussian_actions"] = gaussian_actions
        arrays["gaussian_executed_actions"] = executed_action_view(gaussian_actions)

    if args.run_factorial:
        # LingBot's action stage consumes the predicted-video cache, not a raw
        # video latent.  These cross cells are therefore cache-routing controls:
        # they test whether replacing the cache redirects/rescues the action,
        # but are not an independently identifiable raw-latent x cache factorial.
        donor_future_recipient_cache = np.full_like(latent_grid, np.nan)
        recipient_future_donor_cache = np.full_like(latent_grid, np.nan)
        for recipient in range(branch_count):
            for donor in range(branch_count):
                reset_with_frozen_prompt(server, prompt_embeds, negative_prompt_embeds)
                install_future(server, init_latent, futures[donor])
                restore_cache(server, caches[recipient])
                donor_future_recipient_cache[recipient, donor] = generate_action(
                    server, action_noises[recipient]
                )
                reset_with_frozen_prompt(server, prompt_embeds, negative_prompt_embeds)
                install_future(server, init_latent, futures[recipient])
                restore_cache(server, caches[donor])
                recipient_future_donor_cache[recipient, donor] = generate_action(
                    server, action_noises[recipient]
                )
        arrays["donor_future_recipient_cache_actions"] = donor_future_recipient_cache
        arrays["recipient_future_donor_cache_actions"] = recipient_future_donor_cache
        arrays["donor_future_recipient_cache_executed_actions"] = executed_action_view(
            donor_future_recipient_cache
        )
        arrays["recipient_future_donor_cache_executed_actions"] = executed_action_view(
            recipient_future_donor_cache
        )
        metadata["cache_routing_design"] = (
            "Action generation reads only the installed video K/V cache; raw future "
            "latent and cache source are not independently identifiable at this stage."
        )

    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError(f"{state_id} produced non-finite optional-control output")
    metadata["duration_seconds"] = time.time() - state_started
    metadata["status"] = "complete"
    atomic_npz(arrays_path, **arrays)
    metadata["actions_sha256"] = sha256_file(arrays_path)
    atomic_json(complete_path, metadata)
    print(f"complete {state_id} {metadata['duration_seconds']:.1f}s", flush=True)


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    commit = subprocess.run(
        ["git", "-C", str(args.lingbot_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"LingBot checkout {commit} != {UPSTREAM_COMMIT}")
    args.runner_sha256 = sha256_file(Path(__file__))
    args.manifest_sha256 = sha256_file(args.manifest)
    sys.path.insert(0, str(args.lingbot_root))
    sys.path.insert(0, str(args.lingbot_root / "wan_va"))
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.wan_va_server import VA_Server

    manifest = json.loads(args.manifest.read_text())
    states = [
        record
        for record in manifest["states"]
        if record["admission"] == args.admission
    ]
    states = states[args.shard_index :: args.shard_count]
    if args.max_states is not None:
        states = states[: args.max_states]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_copy = args.output_root / "manifest.json"
    if not manifest_copy.exists():
        atomic_json(manifest_copy, manifest)
    elif sha256_file(manifest_copy) != args.manifest_sha256:
        raise RuntimeError(
            f"output manifest differs from frozen input manifest: {manifest_copy}"
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    init_distributed(world_size, local_rank, rank)
    config = copy.deepcopy(VA_CONFIGS["libero"])
    if config.video_exec_step != -1:
        raise RuntimeError(
            "future reinstall parity is defined only for full video denoising "
            "(video_exec_step == -1)"
        )
    config.wan22_pretrained_model_name_or_path = str(args.checkpoint.resolve())
    config.local_rank = local_rank
    config.rank = rank
    config.world_size = world_size
    config.enable_offload = False
    config.save_root = str((args.output_root / "upstream_debug").resolve())
    server = VA_Server(config)
    if len(_cache_blocks(server)) != 30:
        raise RuntimeError(
            f"expected 30 LingBot transformer blocks, got {len(_cache_blocks(server))}"
        )
    for record in states:
        run_state(server, record, manifest, args)


if __name__ == "__main__":
    main()
