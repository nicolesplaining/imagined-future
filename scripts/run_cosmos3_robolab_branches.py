#!/usr/bin/env python3
"""Run same-state reachable-donor Cosmos 3 branches in public RoboLab."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2  # Must precede Isaac Lab imports.
import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="BananaInBowlTask")
parser.add_argument("--remote-host", default="localhost")
parser.add_argument("--remote-port", type=int, default=8001)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--recorded-hdf5", type=Path, required=True)
parser.add_argument("--branch-seeds", type=int, nargs="+", default=[211, 223, 227, 229])
parser.add_argument("--gaussian-seed", type=int, default=1223)
parser.add_argument("--branch-step", type=int, default=0)
parser.add_argument("--study-id")
parser.add_argument(
    "--target-object-name",
    help="RoboLab rigid-object key used for target-object endpoint separation.",
)
parser.add_argument("--fixed-current-video", type=Path)
parser.add_argument("--timing-sweep", action="store_true")
parser.add_argument(
    "--attention-mediation-layers",
    type=int,
    nargs="+",
    help="Add direct-action and total-nonfuture attention mediation conditions.",
)
parser.add_argument(
    "--attention-kv-patch-layers",
    type=int,
    nargs="+",
    help=(
        "Record self-future K/V and add token-count-preserving selected-layer "
        "and all-layer content-patch conditions."
    ),
)
parser.add_argument(
    "--factorize-selected-donor",
    action="store_true",
    help="Render robot/object visibility masks and transplant each content factor separately.",
)
parser.add_argument(
    "--factorization-object-prim",
    help="Absolute USD path of the task object, required with --factorize-selected-donor.",
)
parser.add_argument("--factorization-mask-threshold", type=int, default=8)
parser.add_argument("--factorization-mask-dilation", type=int, default=2)
parser.add_argument(
    "--multi-donor",
    action="store_true",
    help="Transplant every non-recipient native future and audit donor identification.",
)
parser.add_argument(
    "--native-screen-only",
    action="store_true",
    help=(
        "Stop after native branch collection, exact replay auditing, and the "
        "outcome-independent maximum-endpoint-separation donor selection."
    ),
)
parser.add_argument(
    "--restore-strategy",
    choices=("replay", "snapshot", "fresh_replay"),
    default="fresh_replay",
    help=(
        "How to recover the audited branch point before each condition. "
        "'snapshot' restores the exact saved RoboLab scene state; 'replay' "
        "reruns the recorded prefix after a full environment reset; "
        "'fresh_replay' recreates the environment and validates the recorded "
        "prefix before every condition."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
args.enable_cameras = True
args.save_videos = False
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch  # noqa: E402

from policies.cosmos3.client import Cosmos3Client  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.replay.scene_state import restore_recorded_initial_state  # noqa: E402
from robolab.core.replay.scene_state import StateValidator  # noqa: E402
from robolab.core.utils.file_utils import load_hdf5_episode_data  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402

import robolab.constants  # noqa: E402


def clone_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


def flatten_tree(value: dict[str, Any], prefix: str = "") -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, item in value.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            output.update(flatten_tree(item, path))
        elif isinstance(item, torch.Tensor):
            output[path] = item.detach().cpu()
    return output


def state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path, tensor in sorted(flatten_tree(state).items()):
        if not (path.startswith("articulation/") or path.startswith("rigid_object/")):
            continue
        value = tensor.contiguous()
        digest.update(path.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_vector(state: dict[str, Any], group: str) -> np.ndarray:
    selected: list[np.ndarray] = []
    for path, tensor in sorted(flatten_tree(state).items()):
        if not (path.endswith("joint_position") or path.endswith("root_pose")):
            continue
        if group == "robot" and not path.startswith("articulation/robot/"):
            continue
        if group == "object" and not path.startswith("rigid_object/"):
            continue
        if group in {"object_position", "object_orientation"}:
            if not (path.startswith("rigid_object/") and path.endswith("root_pose")):
                continue
            root_pose = tensor[0].double().reshape(-1).numpy()
            selected.append(root_pose[:3] if group == "object_position" else root_pose[3:7])
            continue
        if group == "all" and not (
            path.startswith("articulation/robot/") or path.startswith("rigid_object/")
        ):
            continue
        selected.append(tensor[0].double().reshape(-1).numpy())
    if not selected:
        raise ValueError(f"state group has no leaves: {group}")
    return np.concatenate(selected)


def target_object_position(state: dict[str, Any], object_name: str) -> np.ndarray:
    try:
        root_pose = state["rigid_object"][object_name]["root_pose"]
    except KeyError as error:
        raise ValueError(f"target object {object_name!r} is absent from scene state") from error
    return root_pose[0].detach().cpu().double().reshape(-1).numpy()[:3]


def projection(value: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> float:
    direction = donor.astype(np.float64) - recipient.astype(np.float64)
    denominator = float(np.square(direction).sum())
    if denominator == 0.0:
        return float("nan")
    return float(((value.astype(np.float64) - recipient.astype(np.float64)) * direction).sum() / denominator)


def response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in response.items():
        if key in {"action", "video"}:
            continue
        if isinstance(value, np.ndarray):
            metadata[key] = value.tolist()
        elif isinstance(value, np.generic):
            metadata[key] = value.item()
        else:
            metadata[key] = value
    return metadata


def main() -> None:
    if len(set(args.branch_seeds)) != len(args.branch_seeds):
        raise ValueError("branch seeds must be unique")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {args.output_dir}")
    if args.branch_step < 0:
        raise ValueError("branch_step must be nonnegative")
    if args.factorize_selected_donor and not args.factorization_object_prim:
        raise ValueError("--factorization-object-prim is required for donor factorization")
    if args.native_screen_only and not args.target_object_name:
        raise ValueError("--target-object-name is required for a native population screen")
    study_id = args.study_id or f"{args.task.lower()}-step{args.branch_step}-{args.output_dir.parent.name}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recorded_env_cfg_path = args.recorded_hdf5.parent / "env_cfg.json"
    if not recorded_env_cfg_path.exists():
        raise FileNotFoundError(
            f"recorded environment config is missing: {recorded_env_cfg_path}"
        )
    recorded_env_cfg = json.loads(recorded_env_cfg_path.read_text())
    recorded_environment_seed = int(recorded_env_cfg["seed"])

    def progress(phase: str, **details: Any) -> None:
        record = {"phase": phase, **details}
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"BRANCH_PROGRESS {json.dumps(record, sort_keys=True)}", flush=True)

    set_output_dir(str(args.output_dir / "robolab_output"))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False

    auto_register_droid_envs(task=args.task, cameras=WRIST_LEFT_RIGHT_HEAD)
    task_envs = get_envs(task=args.task)
    if task_envs != [args.task]:
        raise ValueError(f"expected one exact registered task, got {task_envs}")
    env, env_cfg = create_env(args.task, device=args.device, num_envs=1, use_fabric=True)
    progress("environment_created")
    client = Cosmos3Client(remote_host=args.remote_host, remote_port=args.remote_port)
    progress("server_connected")

    base_initial_state: dict[str, Any] | None = None
    prefix_actions = np.empty((0, 8), dtype=np.float32)
    expected_branch_digest: str | None = None

    def pack(observation: dict[str, Any]) -> dict[str, Any]:
        extracted = client._extract_observation(observation, env_id=0)
        return client._pack_request(extracted, env_cfg.instruction)

    def restore(branch_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal env, env_cfg
        if args.restore_strategy == "fresh_replay":
            env.close()
            env, fresh_env_cfg = create_env(
                args.task,
                device=args.device,
                num_envs=1,
                use_fabric=True,
            )
            if fresh_env_cfg.instruction != env_cfg.instruction:
                raise RuntimeError("fresh environment changed the task instruction")
            env_cfg = fresh_env_cfg
            observation, _ = env.reset()
            restore_recorded_initial_state(env, str(args.recorded_hdf5), 0)
            validator = StateValidator(str(args.recorded_hdf5), 0, tolerance=0.0)
            for step, action_step in enumerate(prefix_actions):
                tensor = torch.from_numpy(action_step).unsqueeze(0).to(
                    device=env.device,
                    dtype=torch.float32,
                )
                observation, _reward, term, trunc, _info = env.step(tensor)
                validator.check_step(env, step)
                if bool(term[0].item() or trunc[0].item()):
                    raise RuntimeError("recorded prefix terminated before the requested branch point")
            if validator.max_drift != 0.0:
                raise RuntimeError(
                    f"fresh recorded-prefix replay drifted by {validator.max_drift} at "
                    f"{validator.max_drift_path} step {validator.max_drift_step}"
                )
            restored = env.scene.get_state(is_relative=True)
            if expected_branch_digest is not None and state_digest(restored) != expected_branch_digest:
                raise RuntimeError("fresh recorded-prefix replay changed the branch state")
            return observation, restored

        env.reset_eval_state()
        env.reset()
        ids = torch.tensor([0], device=env.device, dtype=torch.long)
        get_world(env).reset_predicate_state(ids)
        use_prefix_replay = args.restore_strategy == "replay" and len(prefix_actions) > 0
        replay_state = base_initial_state if use_prefix_replay else branch_state
        if replay_state is None:
            raise RuntimeError("base initial state is not initialized")
        env.reset_to(clone_tree(replay_state), env_ids=None, is_relative=True)
        observation = env.observation_manager.compute()
        for action_step in prefix_actions if use_prefix_replay else ():
            tensor = torch.from_numpy(action_step).unsqueeze(0).to(device=env.device, dtype=torch.float32)
            observation, _reward, term, trunc, _info = env.step(tensor)
            if bool(term[0].item() or trunc[0].item()):
                raise RuntimeError("recorded prefix terminated before the requested branch point")
        restored = env.scene.get_state(is_relative=True)
        if expected_branch_digest is not None and state_digest(restored) != expected_branch_digest:
            raise RuntimeError(f"{args.restore_strategy} did not recover the exact branch state")
        return observation, restored

    def execute(
        branch_state: dict[str, Any],
        fixed_request: dict[str, Any],
        response: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        _observation, restored = restore(branch_state)
        raw_action = np.asarray(response["action"], dtype=np.float32)
        action = client._postprocess_chunk(raw_action)
        frames = [np.asarray(fixed_request["observation/image"], dtype=np.uint8)]
        states = [clone_tree(restored)] if args.factorize_selected_donor else None
        terminated = False
        for action_step in action:
            tensor = torch.from_numpy(action_step).unsqueeze(0).to(device=env.device, dtype=torch.float32)
            observation, _reward, term, trunc, _info = env.step(tensor)
            frames.append(np.asarray(pack(observation)["observation/image"], dtype=np.uint8))
            if states is not None:
                states.append(clone_tree(env.scene.get_state(is_relative=True)))
            terminated = bool(term[0].item() or trunc[0].item())
            if terminated:
                break
        if len(frames) != 33:
            raise RuntimeError(f"{label} terminated after {len(frames) - 1} actions; full donor video required")
        endpoint = env.scene.get_state(is_relative=True)
        donor_path = args.output_dir / f"{label}.npz"
        np.savez_compressed(
            donor_path,
            video=np.stack(frames),
            action=raw_action,
            executed_action=action,
        )
        return {
            "label": label,
            "raw_action": raw_action,
            "executed_action": action,
            "video_path": donor_path,
            "restored_state": restored,
            "endpoint_state": clone_tree(endpoint),
            "states": states,
            "terminated": terminated,
        }

    try:
        initial_observation, _ = env.reset()
        restore_recorded_initial_state(env, str(args.recorded_hdf5), 0)
        base_initial_state = clone_tree(env.scene.get_state(is_relative=True))
        progress("recorded_state_restored")
        recorded_actions = np.asarray(
            load_hdf5_episode_data(str(args.recorded_hdf5), 0, "actions"),
            dtype=np.float32,
        )
        if args.branch_step > len(recorded_actions):
            raise ValueError(
                f"branch_step {args.branch_step} exceeds recorded trajectory length {len(recorded_actions)}"
            )
        prefix_actions = recorded_actions[: args.branch_step].copy()
        prefix_validator = StateValidator(str(args.recorded_hdf5), 0, tolerance=0.0)
        for step, action_step in enumerate(prefix_actions):
            tensor = torch.from_numpy(action_step).unsqueeze(0).to(device=env.device, dtype=torch.float32)
            initial_observation, _reward, term, trunc, _info = env.step(tensor)
            prefix_validator.check_step(env, step)
            if bool(term[0].item() or trunc[0].item()):
                raise RuntimeError("recorded prefix terminated before the requested branch point")
        branch_state = clone_tree(env.scene.get_state(is_relative=True))
        expected_branch_digest = state_digest(branch_state)
        if prefix_validator.max_drift != 0.0:
            raise RuntimeError(
                f"recorded prefix replay drifted by {prefix_validator.max_drift} at "
                f"{prefix_validator.max_drift_path} step {prefix_validator.max_drift_step}"
            )
        progress("branch_point_reached", branch_step=args.branch_step, prefix_maximum_state_error=0.0)
        initial_observation = env.observation_manager.compute()
        initial_request = pack(initial_observation)
        if args.fixed_current_video is not None:
            with np.load(args.fixed_current_video, allow_pickle=False) as payload:
                fixed_video = np.asarray(payload["video"])
            if fixed_video.shape != (33, 540, 640, 3) or fixed_video.dtype != np.uint8:
                raise ValueError(
                    "fixed current video must contain uint8 video with shape (33, 540, 640, 3)"
                )
            initial_request["observation/image"] = fixed_video[0].copy()
        initial_image = np.asarray(initial_request["observation/image"], dtype=np.uint8)

        restore_digests = []
        restore_image_errors = []
        for _ in range(3):
            replay_observation, replay_state = restore(branch_state)
            restore_digests.append(state_digest(replay_state))
            replay_image = np.asarray(pack(replay_observation)["observation/image"], dtype=np.uint8)
            restore_image_errors.append(int(np.abs(replay_image.astype(np.int16) - initial_image).max()))
        if len(set(restore_digests)) != 1:
            raise RuntimeError(f"branch state restore is not exact: {restore_digests}")
        progress(
            "restore_audit_passed",
            restore_strategy=args.restore_strategy,
            state_digest=restore_digests[0],
            rerender_image_errors=restore_image_errors,
            model_current_observation="cached_original",
        )

        native_responses: dict[int, dict[str, Any]] = {}
        native_runs: dict[int, dict[str, Any]] = {}
        server_state_hashes = set()
        for seed in args.branch_seeds:
            request = dict(initial_request)
            request.update(
                research_mode="native",
                research_seed=seed,
                research_id=f"{study_id}-native-{seed}",
            )
            response = client.client.infer(request)
            native_responses[seed] = response
            server_state_hashes.add(response["research_state_hash"])
            if not args.native_screen_only:
                native_runs[seed] = execute(
                    branch_state, initial_request, response, f"native_{seed}"
                )
                progress("native_branch_completed", seed=seed)
            else:
                progress("native_response_completed", seed=seed)
        if len(server_state_hashes) != 1:
            raise RuntimeError(f"native server state hashes differ: {server_state_hashes}")

        native_action_pair_distances = {
            (left, right): float(
                np.linalg.norm(
                    np.asarray(native_responses[left]["action"], dtype=np.float32)
                    - np.asarray(native_responses[right]["action"], dtype=np.float32)
                )
            )
            for left, right in combinations(args.branch_seeds, 2)
        }
        if args.native_screen_only:
            recipient_seed, donor_seed = max(
                native_action_pair_distances,
                key=lambda pair: (native_action_pair_distances[pair], -pair[0], -pair[1]),
            )
            for seed in (recipient_seed, donor_seed):
                native_runs[seed] = execute(
                    branch_state,
                    initial_request,
                    native_responses[seed],
                    f"native_{seed}",
                )
                progress("selected_native_branch_completed", seed=seed)

        repeat_seed = recipient_seed if args.native_screen_only else args.branch_seeds[0]
        repeat_run = execute(
            branch_state,
            initial_request,
            native_responses[repeat_seed],
            f"native_{repeat_seed}_repeat",
        )
        repeat_reference_digest = state_digest(native_runs[repeat_seed]["endpoint_state"])
        repeat_endpoint_digest = state_digest(repeat_run["endpoint_state"])
        if repeat_endpoint_digest != repeat_reference_digest:
            raise RuntimeError(
                "identical continuation from the restored branch state was not deterministic: "
                f"{repeat_endpoint_digest} != {repeat_reference_digest}"
            )
        progress("continuation_repeat_audit_passed", seed=repeat_seed)

        endpoint_vectors = {seed: state_vector(run["endpoint_state"], "all") for seed, run in native_runs.items()}
        pair_distances = {
            (left, right): float(np.linalg.norm(endpoint_vectors[left] - endpoint_vectors[right]))
            for left, right in combinations(native_runs, 2)
        }
        if not args.native_screen_only:
            recipient_seed, donor_seed = max(pair_distances, key=pair_distances.get)
        progress("donor_pair_selected", recipient_seed=recipient_seed, donor_seed=donor_seed)

        if args.native_screen_only:
            groups = ("all", "robot", "object", "object_position", "object_orientation")
            recipient_action = native_runs[recipient_seed]["raw_action"]
            donor_action = native_runs[donor_seed]["raw_action"]
            recipient_endpoints = {
                group: state_vector(native_runs[recipient_seed]["endpoint_state"], group)
                for group in groups
            }
            donor_endpoints = {
                group: state_vector(native_runs[donor_seed]["endpoint_state"], group)
                for group in groups
            }
            summary = {
                "scope": (
                    "native-only population screen; no future transplant or attention "
                    "intervention was evaluated"
                ),
                "task": args.task,
                "study_id": study_id,
                "branch_step": args.branch_step,
                "restore_strategy": args.restore_strategy,
                "prefix_maximum_state_error": prefix_validator.max_drift,
                "instruction": env_cfg.instruction,
                "recorded_hdf5": str(args.recorded_hdf5),
                "recorded_env_cfg": str(recorded_env_cfg_path),
                "environment_seed": recorded_environment_seed,
                "branch_seeds": args.branch_seeds,
                "selection_rule": (
                    "maximum native action L2 among all frozen seeds; ties use the lower "
                    "ordered seed pair; only that pair is physically executed"
                ),
                "recipient_seed": recipient_seed,
                "donor_seed": donor_seed,
                "native_pairwise_action_l2": {
                    f"{left}:{right}": distance
                    for (left, right), distance in native_action_pair_distances.items()
                },
                "native_pairwise_endpoint_l2": {
                    f"{left}:{right}": distance
                    for (left, right), distance in pair_distances.items()
                },
                "native_action_l2": float(np.linalg.norm(donor_action - recipient_action)),
                "native_endpoint_l2": {
                    group: float(
                        np.linalg.norm(donor_endpoints[group] - recipient_endpoints[group])
                    )
                    for group in groups
                },
                "native_target_object_position_l2": float(
                    np.linalg.norm(
                        target_object_position(
                            native_runs[donor_seed]["endpoint_state"],
                            args.target_object_name,
                        )
                        - target_object_position(
                            native_runs[recipient_seed]["endpoint_state"],
                            args.target_object_name,
                        )
                    )
                ),
                "target_object_name": args.target_object_name,
                "restore_state_digests": restore_digests,
                "restore_image_maximum_absolute_errors": restore_image_errors,
                "continuation_repeat_audit": {
                    "seed": repeat_seed,
                    "reference_endpoint_state_digest": repeat_reference_digest,
                    "repeat_endpoint_state_digest": repeat_endpoint_digest,
                    "exact": True,
                },
                "server_state_hash": next(iter(server_state_hashes)),
                "native": {
                    str(seed): {
                        "selected_for_physical_execution": seed in native_runs,
                        "endpoint_state_digest": (
                            state_digest(native_runs[seed]["endpoint_state"])
                            if seed in native_runs
                            else None
                        ),
                        "video_path": (
                            str(native_runs[seed]["video_path"])
                            if seed in native_runs
                            else None
                        ),
                        "server": response_metadata(native_responses[seed]),
                    }
                    for seed in args.branch_seeds
                },
                "interventions_evaluated": False,
            }
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            progress("completed")
            return

        for seed, run in native_runs.items():
            register_request = dict(initial_request)
            register_request.update(
                research_mode="register_executed",
                research_id=f"{study_id}-executed-{seed}",
                research_donor_path=str(run["video_path"]),
            )
            registered = client.client.infer(register_request)
            if registered["research_state_hash"] not in server_state_hashes:
                raise RuntimeError("executed donor registration changed the state fingerprint")
            progress("executed_donor_registered", seed=seed)

        factorization_report = None
        state_cell_donor_ids: dict[str, str] = {}
        composite_cell_donor_ids: dict[str, str] = {}
        if args.factorize_selected_donor:
            recipient_states = native_runs[recipient_seed]["states"]
            donor_states = native_runs[donor_seed]["states"]
            if recipient_states is None or donor_states is None:
                raise RuntimeError("factorization requires saved native trajectory states")
            object_name = Path(args.factorization_object_prim).name
            if object_name not in recipient_states[0].get("rigid_object", {}):
                raise ValueError(
                    f"factorization object {object_name!r} is absent from the RoboLab scene state"
                )
            env.close()
            env, render_env_cfg = create_env(
                args.task,
                device=args.device,
                num_envs=1,
                use_fabric=False,
            )
            if render_env_cfg.instruction != env_cfg.instruction:
                raise RuntimeError("non-Fabric render environment changed the task instruction")
            state_cell_audits = {}
            state_cell_videos = {}
            for cell, use_robot, use_object in (
                ("o0r0", False, False),
                ("o0r1", True, False),
                ("o1r0", False, True),
                ("o1r1", True, True),
            ):
                frames = [initial_image.copy()]
                state_errors = []
                for recipient_state, donor_state in zip(
                    recipient_states[1:], donor_states[1:], strict=True
                ):
                    hybrid_state = clone_tree(recipient_state)
                    if use_robot:
                        hybrid_state["articulation"]["robot"] = clone_tree(
                            donor_state["articulation"]["robot"]
                        )
                    if use_object:
                        hybrid_state["rigid_object"][object_name] = clone_tree(
                            donor_state["rigid_object"][object_name]
                        )
                    observation, _ = env.reset_to(
                        hybrid_state, env_ids=None, is_relative=True
                    )
                    realized_state = env.scene.get_state(is_relative=True)
                    exact = state_digest(realized_state) == state_digest(hybrid_state)
                    state_errors.append(0.0 if exact else float("inf"))
                    if not exact:
                        raise RuntimeError(f"state-factorized cell {cell} did not restore exactly")
                    frames.append(np.asarray(pack(observation)["observation/image"], dtype=np.uint8))
                video = np.stack(frames)
                if video.shape != (33, 540, 640, 3):
                    raise RuntimeError(f"state-factorized cell {cell} has shape {video.shape}")
                state_cell_videos[cell] = video
                donor_path = args.output_dir / f"target_state_cell_{cell}.npz"
                np.savez_compressed(
                    donor_path,
                    video=video,
                    recipient_seed=np.asarray(recipient_seed),
                    donor_seed=np.asarray(donor_seed),
                    object_name=np.asarray(object_name),
                )
                state_cell_audits[cell] = {
                    "robot_from_donor": use_robot,
                    "object_from_donor": use_object,
                    "maximum_state_restore_error": max(state_errors),
                    "video_path": str(donor_path),
                }

            video_contrasts = {}
            for left, right in combinations(state_cell_videos, 2):
                difference = np.abs(
                    state_cell_videos[left].astype(np.int16)
                    - state_cell_videos[right].astype(np.int16)
                )
                video_contrasts[f"{left}:{right}"] = {
                    "maximum_absolute_rgb_difference": int(difference.max()),
                    "mean_absolute_rgb_difference": float(difference.mean()),
                }
            for left, right in (("o0r0", "o0r1"), ("o0r0", "o1r0")):
                if video_contrasts[f"{left}:{right}"]["maximum_absolute_rgb_difference"] == 0:
                    raise RuntimeError(f"state-factorized target videos {left} and {right} are identical")

            recipient_video = np.load(
                native_runs[recipient_seed]["video_path"], allow_pickle=False
            )["video"]
            donor_video = np.load(
                native_runs[donor_seed]["video_path"], allow_pickle=False
            )["video"]
            if recipient_video.shape != donor_video.shape or recipient_video.shape != (
                33,
                540,
                640,
                3,
            ):
                raise RuntimeError("native videos have incompatible factorization shapes")
            kernel_size = 2 * args.factorization_mask_dilation + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

            def difference_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
                raw = np.max(
                    np.abs(left.astype(np.int16) - right.astype(np.int16)), axis=-1
                ) >= args.factorization_mask_threshold
                return np.stack(
                    [cv2.dilate(frame.astype(np.uint8), kernel).astype(bool) for frame in raw]
                )

            robot_mask = difference_mask(state_cell_videos["o0r0"], state_cell_videos["o0r1"])
            object_mask = difference_mask(state_cell_videos["o0r0"], state_cell_videos["o1r0"])
            robot_mask[0] = False
            object_mask[0] = False
            composite_cell_videos = {}
            composite_cell_audits = {}
            for cell, use_robot, use_object in (
                ("o0r0", False, False),
                ("o0r1", True, False),
                ("o1r0", False, True),
                ("o1r1", True, True),
            ):
                mask = np.zeros(robot_mask.shape, dtype=bool)
                if use_robot:
                    mask |= robot_mask
                if use_object:
                    mask |= object_mask
                video = recipient_video.copy()
                video[mask] = donor_video[mask]
                if not np.array_equal(video[0], recipient_video[0]):
                    raise RuntimeError("masked factorization changed the current frame")
                donor_path = args.output_dir / f"target_composite_cell_{cell}.npz"
                np.savez_compressed(
                    donor_path,
                    video=video,
                    recipient_seed=np.asarray(recipient_seed),
                    donor_seed=np.asarray(donor_seed),
                    object_name=np.asarray(object_name),
                )
                composite_cell_videos[cell] = video
                composite_cell_audits[cell] = {
                    "robot_from_donor": use_robot,
                    "object_from_donor": use_object,
                    "replaced_pixel_fraction": float(mask[1:].mean()),
                    "video_path": str(donor_path),
                }

            target_designs = {
                "state": (state_cell_videos, state_cell_audits, state_cell_donor_ids),
                "composite": (
                    composite_cell_videos,
                    composite_cell_audits,
                    composite_cell_donor_ids,
                ),
            }
            decoded_target_audits = {}
            for design, (videos, audits, donor_ids) in target_designs.items():
                roundtrip_videos = {}
                for cell, audit in audits.items():
                    record_id = f"{study_id}-{design}-cell-{cell}"
                    register_request = dict(initial_request)
                    register_request.update(
                        research_mode="register_executed",
                        research_id=record_id,
                        research_donor_path=audit["video_path"],
                        research_return_video=True,
                    )
                    registered = client.client.infer(register_request)
                    if registered["research_state_hash"] not in server_state_hashes:
                        raise RuntimeError(
                            f"{design}-factorized donor registration changed the state fingerprint"
                        )
                    donor_ids[cell] = record_id
                    roundtrip_videos[cell] = np.asarray(registered["video"], dtype=np.uint8)
                    progress("factorized_donor_registered", design=design, cell=cell)
                design_audits = {}
                for cell, decoded in roundtrip_videos.items():
                    if decoded.shape != videos[cell].shape:
                        raise RuntimeError(
                            f"decoded {design} cell {cell} has shape {decoded.shape}, "
                            f"expected {videos[cell].shape}"
                        )
                    distances = {
                        candidate: float(
                            np.square(
                                decoded[1:].astype(np.float32)
                                - candidate_video[1:].astype(np.float32)
                            ).mean()
                        )
                        for candidate, candidate_video in videos.items()
                    }
                    nearest = min(
                        distances, key=lambda candidate: (distances[candidate], candidate)
                    )
                    design_audits[cell] = {
                        "nearest_raw_target_cell": nearest,
                        "correct_target_top1": nearest == cell,
                        "raw_target_mean_squared_rgb_distances": distances,
                    }
                decoded_target_audits[design] = design_audits

            factorization_report = {
                "recipient_seed": recipient_seed,
                "donor_seed": donor_seed,
                "object_name": object_name,
                "state_cells": state_cell_audits,
                "composite_cells": composite_cell_audits,
                "target_video_contrasts": video_contrasts,
                "decoded_target_audits": decoded_target_audits,
                "decoded_target_top1_rate": {
                    design: float(
                        np.mean([audit["correct_target_top1"] for audit in audits.values()])
                    )
                    for design, audits in decoded_target_audits.items()
                },
                "mask_threshold_rgb": args.factorization_mask_threshold,
                "mask_dilation_pixels": args.factorization_mask_dilation,
                "robot_mask_future_pixel_fraction": float(robot_mask[1:].mean()),
                "object_mask_future_pixel_fraction": float(object_mask[1:].mean()),
                "state_composition": (
                    "recipient simulator state with robot and target-object subtrees replaced "
                    "factorially from the donor at every future frame; current frame held exactly fixed"
                ),
            }

        intervention_specs = {
            "self": {
                "research_mode": "native",
            },
            "predicted_donor": {
                "research_mode": "donor",
                "research_donor_id": f"{study_id}-native-{donor_seed}",
            },
            "executed_donor": {
                "research_mode": "donor",
                "research_donor_id": f"{study_id}-executed-{donor_seed}",
            },
            "gaussian_executed": {
                "research_mode": "gaussian",
                "research_donor_id": f"{study_id}-executed-{donor_seed}",
                "research_gaussian_seed": args.gaussian_seed,
            },
        }
        intervention_target_seeds: dict[str, int | None] = {
            "self": None,
            "predicted_donor": donor_seed,
            "executed_donor": donor_seed,
            "gaussian_executed": None,
        }
        if args.factorize_selected_donor:
            intervention_specs.update(
                {
                    "executed_self": {
                        "research_mode": "donor",
                        "research_donor_id": f"{study_id}-executed-{recipient_seed}",
                    },
                    **{
                        f"state_cell_{cell}": {
                            "research_mode": "donor",
                            "research_donor_id": record_id,
                        }
                        for cell, record_id in state_cell_donor_ids.items()
                    },
                    **{
                        f"composite_cell_{cell}": {
                            "research_mode": "donor",
                            "research_donor_id": record_id,
                        }
                        for cell, record_id in composite_cell_donor_ids.items()
                    },
                }
            )
            intervention_target_seeds.update(
                {
                    "executed_self": None,
                    "state_cell_o0r0": None,
                    "state_cell_o0r1": donor_seed,
                    "state_cell_o1r0": donor_seed,
                    "state_cell_o1r1": donor_seed,
                    "composite_cell_o0r0": None,
                    "composite_cell_o0r1": donor_seed,
                    "composite_cell_o1r0": donor_seed,
                    "composite_cell_o1r1": donor_seed,
                }
            )
        if args.multi_donor:
            for candidate_seed in args.branch_seeds:
                if candidate_seed in {recipient_seed, donor_seed}:
                    continue
                for source, donor_id in (
                    ("predicted", f"{study_id}-native-{candidate_seed}"),
                    ("executed", f"{study_id}-executed-{candidate_seed}"),
                ):
                    label = f"{source}_donor_seed_{candidate_seed}"
                    intervention_specs[label] = {
                        "research_mode": "donor",
                        "research_donor_id": donor_id,
                    }
                    intervention_target_seeds[label] = candidate_seed
        if args.timing_sweep:
            intervention_specs.update(
                {
                    f"predicted_donor_step_{step}": {
                        "research_mode": "donor",
                        "research_donor_id": f"{study_id}-native-{donor_seed}",
                        "research_timing_steps": [step],
                    }
                    for step in range(4)
                }
            )
            intervention_target_seeds.update(
                {f"predicted_donor_step_{step}": donor_seed for step in range(4)}
            )
        if args.attention_mediation_layers is not None:
            if len(set(args.attention_mediation_layers)) != len(args.attention_mediation_layers):
                raise ValueError("attention mediation layers must be unique")
            for source, donor_id in (
                ("predicted", f"{study_id}-native-{donor_seed}"),
                ("executed", f"{study_id}-executed-{donor_seed}"),
            ):
                for scope in ("action", "nonfuture"):
                    label = f"{source}_donor_attention_{scope}"
                    intervention_specs[label] = {
                        "research_mode": "donor",
                        "research_donor_id": donor_id,
                        "research_attention_exclude_layers": args.attention_mediation_layers,
                        "research_attention_exclude_scope": scope,
                    }
                    intervention_target_seeds[label] = donor_seed
        if args.attention_kv_patch_layers is not None:
            selected_layers = args.attention_kv_patch_layers
            if len(set(selected_layers)) != len(selected_layers):
                raise ValueError("attention K/V patch layers must be unique")
            if any(layer < 0 or layer >= 36 for layer in selected_layers):
                raise ValueError("attention K/V patch layers must be in [0,36)")
            all_layers = list(range(36))
            cache_id = f"{study_id}-self-future-kv"
            intervention_specs.update(
                {
                    "self_kv_record": {
                        "research_mode": "self",
                        "research_donor_id": f"{study_id}-native-{recipient_seed}",
                        "research_attention_mode": "record",
                        "research_attention_cache_id": cache_id,
                        "research_attention_exclude_layers": all_layers,
                    },
                    "self_kv_patch_all": {
                        "research_mode": "self",
                        "research_donor_id": f"{study_id}-native-{recipient_seed}",
                        "research_attention_mode": "patch",
                        "research_attention_cache_id": cache_id,
                        "research_attention_exclude_layers": all_layers,
                    },
                }
            )
            intervention_target_seeds.update(
                {"self_kv_record": None, "self_kv_patch_all": None}
            )
            for source, donor_id in (
                ("predicted", f"{study_id}-native-{donor_seed}"),
                ("executed", f"{study_id}-executed-{donor_seed}"),
            ):
                conditions = {
                    f"{source}_donor_kv_patch_selected": selected_layers,
                    f"{source}_donor_kv_patch_all_action": all_layers,
                    f"{source}_donor_kv_patch_all_nonfuture": all_layers,
                }
                for label, layers in conditions.items():
                    intervention_specs[label] = {
                        "research_mode": "donor",
                        "research_donor_id": donor_id,
                        "research_attention_mode": "patch",
                        "research_attention_cache_id": cache_id,
                        "research_attention_exclude_layers": layers,
                        "research_attention_exclude_scope": (
                            "nonfuture" if label.endswith("nonfuture") else "action"
                        ),
                    }
                    intervention_target_seeds[label] = donor_seed
        intervention_responses = {}
        intervention_runs = {}
        kv_patch_identity_action_errors: dict[str, float] = {}
        self_identity_action_maximum_error: float | None = None
        self_clamp_action_maximum_error: float | None = None
        for label, spec in intervention_specs.items():
            request = dict(initial_request)
            request.update(
                research_id=f"{study_id}-intervention-{label}",
                research_seed=recipient_seed,
                research_recipient_id=f"{study_id}-native-{recipient_seed}",
                **spec,
            )
            response = client.client.infer(request)
            if label == "self":
                self_identity_action_maximum_error = float(
                    np.abs(
                        np.asarray(response["action"], dtype=np.float32)
                        - np.asarray(native_responses[recipient_seed]["action"], dtype=np.float32)
                    ).max()
                )
                if self_identity_action_maximum_error != 0.0:
                    raise RuntimeError(
                        "self-target recomputation changed native action by "
                        f"{self_identity_action_maximum_error}"
                    )
            if label == "gaussian_executed":
                for metric in (
                    "research_gaussian_norm_relative_error",
                    "research_gaussian_distance_relative_error",
                ):
                    if float(response[metric]) > 1e-5:
                        raise RuntimeError(f"Gaussian matching failed: {metric}={response[metric]}")
            if label == "self_kv_record":
                reference_action = np.asarray(intervention_responses["self"]["action"], dtype=np.float32)
                self_clamp_action_maximum_error = float(
                    np.abs(np.asarray(response["action"], dtype=np.float32) - reference_action).max()
                )
            if label == "self_kv_patch_all":
                reference_action = np.asarray(
                    intervention_responses["self_kv_record"]["action"], dtype=np.float32
                )
                action_error = float(
                    np.abs(np.asarray(response["action"], dtype=np.float32) - reference_action).max()
                )
                kv_patch_identity_action_errors[label] = action_error
                if action_error != 0.0:
                    raise RuntimeError(f"{label} changed self action by {action_error}")
            intervention_responses[label] = response
            intervention_runs[label] = execute(branch_state, initial_request, response, label)
            progress("intervention_completed", label=label)

        recipient_run = native_runs[recipient_seed]
        donor_run = native_runs[donor_seed]
        recipient_action = recipient_run["raw_action"]
        donor_action = donor_run["raw_action"]
        groups = ("all", "robot", "object", "object_position", "object_orientation")
        recipient_endpoints = {group: state_vector(recipient_run["endpoint_state"], group) for group in groups}
        donor_endpoints = {group: state_vector(donor_run["endpoint_state"], group) for group in groups}
        interventions = {}
        for label, run in intervention_runs.items():
            response = intervention_responses[label]
            target_seed = intervention_target_seeds[label]
            target_action = native_runs[target_seed]["raw_action"] if target_seed is not None else None
            target_endpoints = (
                {
                    group: state_vector(native_runs[target_seed]["endpoint_state"], group)
                    for group in groups
                }
                if target_seed is not None
                else None
            )
            nearest_action_seed = min(
                args.branch_seeds,
                key=lambda seed: float(
                    np.linalg.norm(run["raw_action"] - native_runs[seed]["raw_action"])
                ),
            )
            nearest_endpoint_seed = {
                group: min(
                    args.branch_seeds,
                    key=lambda seed: float(
                        np.linalg.norm(
                            state_vector(run["endpoint_state"], group)
                            - state_vector(native_runs[seed]["endpoint_state"], group)
                        )
                    ),
                )
                for group in groups
            }
            interventions[label] = {
                "target_donor_seed": target_seed,
                "action_donor_projection": projection(run["raw_action"], recipient_action, donor_action),
                "action_target_donor_projection": (
                    projection(run["raw_action"], recipient_action, target_action)
                    if target_action is not None
                    else None
                ),
                "action_l2_from_recipient": float(np.linalg.norm(run["raw_action"] - recipient_action)),
                "nearest_native_action_seed": nearest_action_seed,
                "correct_action_donor_top1": (
                    nearest_action_seed == target_seed if target_seed is not None else None
                ),
                "endpoint_donor_projection": {
                    group: projection(
                        state_vector(run["endpoint_state"], group),
                        recipient_endpoints[group],
                        donor_endpoints[group],
                    )
                    for group in groups
                },
                "endpoint_target_donor_projection": (
                    {
                        group: projection(
                            state_vector(run["endpoint_state"], group),
                            recipient_endpoints[group],
                            target_endpoints[group],
                        )
                        for group in groups
                    }
                    if target_endpoints is not None
                    else None
                ),
                "nearest_native_endpoint_seed": nearest_endpoint_seed,
                "correct_endpoint_donor_top1": (
                    {
                        group: nearest_endpoint_seed[group] == target_seed for group in groups
                    }
                    if target_seed is not None
                    else None
                ),
                "endpoint_l2_from_recipient": {
                    group: float(
                        np.linalg.norm(
                            state_vector(run["endpoint_state"], group) - recipient_endpoints[group]
                        )
                    )
                    for group in groups
                },
                "server": response_metadata(response),
                "video_path": str(run["video_path"]),
            }

        factorial_effects = None
        if args.factorize_selected_donor:
            designs = {
                "state": {
                    "o0r0": "state_cell_o0r0",
                    "o0r1": "state_cell_o0r1",
                    "o1r0": "state_cell_o1r0",
                    "o1r1": "state_cell_o1r1",
                },
                "composite": {
                    "o0r0": "composite_cell_o0r0",
                    "o0r1": "composite_cell_o0r1",
                    "o1r0": "composite_cell_o1r0",
                    "o1r1": "composite_cell_o1r1",
                },
            }

            def contrast(cells: dict[str, float]) -> dict[str, float]:
                return {
                    "object_main_effect": 0.5
                    * (cells["o1r0"] + cells["o1r1"] - cells["o0r0"] - cells["o0r1"]),
                    "robot_main_effect": 0.5
                    * (cells["o0r1"] + cells["o1r1"] - cells["o0r0"] - cells["o1r0"]),
                    "interaction": cells["o1r1"] - cells["o1r0"] - cells["o0r1"] + cells["o0r0"],
                }

            factorial_effects = {}
            for design, labels in designs.items():
                action_cells = {
                    cell: interventions[label]["action_donor_projection"]
                    for cell, label in labels.items()
                }
                endpoint_cells = {
                    group: {
                        cell: interventions[label]["endpoint_donor_projection"][group]
                        for cell, label in labels.items()
                    }
                    for group in groups
                }
                factorial_effects[design] = {
                    "action_donor_projection_cells": action_cells,
                    "action_donor_projection_effects": contrast(action_cells),
                    "endpoint_donor_projection_cells": endpoint_cells,
                    "endpoint_donor_projection_effects": {
                        group: contrast(cells) for group, cells in endpoint_cells.items()
                    },
                }

        summary = {
            "scope": "same-state RoboLab engineering pilot; donor pair selected only by native endpoint separation",
            "task": args.task,
            "study_id": study_id,
            "branch_step": args.branch_step,
            "restore_strategy": args.restore_strategy,
            "fixed_current_video": str(args.fixed_current_video) if args.fixed_current_video else None,
            "timing_sweep": args.timing_sweep,
            "attention_mediation_layers": args.attention_mediation_layers,
            "attention_kv_patch_layers": args.attention_kv_patch_layers,
            "kv_patch_identity_action_maximum_errors": kv_patch_identity_action_errors,
            "self_identity_action_maximum_error": self_identity_action_maximum_error,
            "self_clamp_action_maximum_error": self_clamp_action_maximum_error,
            "multi_donor": args.multi_donor,
            "factorize_selected_donor": args.factorize_selected_donor,
            "factorization": factorization_report,
            "factorial_effects": factorial_effects,
            "prefix_maximum_state_error": prefix_validator.max_drift,
            "instruction": env_cfg.instruction,
            "recorded_hdf5": str(args.recorded_hdf5),
            "recorded_env_cfg": str(recorded_env_cfg_path),
            "environment_seed": recorded_environment_seed,
            "branch_seeds": args.branch_seeds,
            "recipient_seed": recipient_seed,
            "donor_seed": donor_seed,
            "native_pairwise_endpoint_l2": {
                f"{left}:{right}": distance for (left, right), distance in pair_distances.items()
            },
            "native_action_l2": float(np.linalg.norm(donor_action - recipient_action)),
            "native_endpoint_l2": {
                group: float(np.linalg.norm(donor_endpoints[group] - recipient_endpoints[group]))
                for group in groups
            },
            "native_target_object_position_l2": (
                float(
                    np.linalg.norm(
                        target_object_position(
                            donor_run["endpoint_state"], args.target_object_name
                        )
                        - target_object_position(
                            recipient_run["endpoint_state"], args.target_object_name
                        )
                    )
                )
                if args.target_object_name
                else None
            ),
            "target_object_name": args.target_object_name,
            "restore_state_digests": restore_digests,
            "restore_image_maximum_absolute_errors": restore_image_errors,
            "continuation_repeat_audit": {
                "seed": repeat_seed,
                "reference_endpoint_state_digest": repeat_reference_digest,
                "repeat_endpoint_state_digest": repeat_endpoint_digest,
                "exact": True,
            },
            "server_state_hash": next(iter(server_state_hashes)),
            "native": {
                str(seed): {
                    "endpoint_state_digest": state_digest(run["endpoint_state"]),
                    "video_path": str(run["video_path"]),
                    "server": response_metadata(native_responses[seed]),
                }
                for seed, run in native_runs.items()
            },
            "interventions": interventions,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        progress("completed")
    finally:
        env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"RoboLab branch pilot failed: {error}", flush=True)
        traceback.print_exc()
        raise
