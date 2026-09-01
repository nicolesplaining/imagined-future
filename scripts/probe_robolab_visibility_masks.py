#!/usr/bin/env python3
"""Probe paired visible/invisible RoboLab renders for causal pixel masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2  # Must precede Isaac Lab imports.
import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="BananaInBowlTask")
parser.add_argument("--object", default="banana")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--difference-threshold", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
args.enable_cameras = True
args.save_videos = False
launcher = AppLauncher(args)
simulation_app = launcher.app

from pxr import UsdGeom  # noqa: E402

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402


SENSORS = ("wrist_cam", "over_shoulder_left_camera", "over_shoulder_right_camera")


def capture(env) -> dict[str, np.ndarray]:
    env.sim.render()
    observation = env.observation_manager.compute()["image_obs"]
    return {name: observation[name][0].detach().cpu().numpy() for name in SENSORS}


def resolve_target(env, requested_path: str):
    leaf = Path(requested_path).name
    asset = env.scene.articulations.get(leaf) or env.scene.rigid_objects.get(leaf)
    if asset is not None:
        return asset.set_visibility, str(asset.cfg.prim_path)
    prim = env.sim.stage.GetPrimAtPath(requested_path)
    if not prim.IsValid():
        candidates = [
            candidate
            for candidate in env.sim.stage.Traverse()
            if candidate.GetName() == leaf and "/env_0/" in str(candidate.GetPath())
        ]
        if len(candidates) != 1:
            paths = [str(candidate.GetPath()) for candidate in candidates]
            raise ValueError(f"visibility target {requested_path!r} resolved to {paths}")
        prim = candidates[0]
    imageable = UsdGeom.Imageable(prim)

    def set_visible(visible: bool) -> None:
        imageable.MakeVisible() if visible else imageable.MakeInvisible()

    return set_visible, str(prim.GetPath())


def main() -> None:
    auto_register_droid_envs(task=args.task, cameras=WRIST_LEFT_RIGHT_HEAD)
    env, _env_cfg = create_env(args.task, device=args.device, num_envs=1, use_fabric=True)
    try:
        env.reset()
        native = capture(env)
        masks = {}
        report = {"threshold": args.difference_threshold, "targets": {}}
        for label, path in (
            ("robot", "/World/envs/env_0/robot"),
            ("object", f"/World/envs/env_0/{args.object}"),
        ):
            set_visible, resolved_path = resolve_target(env, path)
            set_visible(False)
            try:
                hidden = capture(env)
            finally:
                set_visible(True)
            restored = capture(env)
            target_report = {}
            for sensor in SENSORS:
                hidden_before = np.abs(
                    native[sensor].astype(np.int16) - hidden[sensor].astype(np.int16)
                )
                hidden_after = np.abs(
                    restored[sensor].astype(np.int16) - hidden[sensor].astype(np.int16)
                )
                restoration = np.abs(
                    native[sensor].astype(np.int16) - restored[sensor].astype(np.int16)
                )
                replicated_signal = np.minimum(
                    hidden_before.max(axis=-1), hidden_after.max(axis=-1)
                )
                mask = (replicated_signal > args.difference_threshold) & (
                    restoration.max(axis=-1) <= args.difference_threshold
                )
                masks[f"{label}_{sensor}"] = mask
                target_report[sensor] = {
                    "mask_pixels": int(mask.sum()),
                    "mask_fraction": float(mask.mean()),
                    "hidden_maximum_replicated_rgb_difference": int(replicated_signal.max()),
                    "restored_maximum_rgb_error": int(restoration.max()),
                    "unstable_visible_pixel_fraction": float(
                        (restoration.max(axis=-1) > args.difference_threshold).mean()
                    ),
                }
            report["targets"][label] = target_report
            report["targets"][label]["_resolved_prim_path"] = resolved_path
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_suffix(".npz"), **masks)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
