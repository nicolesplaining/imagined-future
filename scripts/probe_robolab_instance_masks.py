#!/usr/bin/env python3
"""Probe RoboLab camera instance-ID outputs for causal pixel factorization."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2  # Must precede Isaac Lab imports.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="BananaInBowlTask")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--include-wrist", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
args.enable_cameras = True
args.save_videos = False
launcher = AppLauncher(args)
simulation_app = launcher.app

from isaaclab.sensors import CameraCfg  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402


def add_instance_ids(camera_cfg_cls) -> None:
    """Add raw instance IDs to an isolated process's public camera config."""

    instance = camera_cfg_cls()
    for name in dir(instance):
        if name.startswith("_"):
            continue
        value = getattr(instance, name)
        if isinstance(value, CameraCfg):
            camera = copy.deepcopy(value)
            if "instance_id_segmentation_fast" not in camera.data_types:
                camera.data_types = [*camera.data_types, "instance_id_segmentation_fast"]
            camera.colorize_instance_id_segmentation = False
            setattr(camera_cfg_cls, name, camera)


def jsonify(value):
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> None:
    augmented_names = []
    for camera in WRIST_LEFT_RIGHT_HEAD:
        if camera.__name__ == "WristCameraCfg" and not args.include_wrist:
            continue
        add_instance_ids(camera)
        augmented_names.append(camera.__name__)
    auto_register_droid_envs(task=args.task, cameras=WRIST_LEFT_RIGHT_HEAD)
    env, _env_cfg = create_env(args.task, device=args.device, num_envs=1, use_fabric=True)
    try:
        env.reset()
        env.sim.render()
        env.observation_manager.compute()
        report = {}
        sensor_names = ["over_shoulder_left_camera", "over_shoulder_right_camera"]
        if args.include_wrist:
            sensor_names.insert(0, "wrist_cam")
        for name in sensor_names:
            sensor = env.scene.sensors[name]
            ids = sensor.data.output["instance_id_segmentation_fast"][0, ..., 0]
            values, counts = ids.unique(return_counts=True)
            report[name] = {
                "shape": list(ids.shape),
                "dtype": str(ids.dtype),
                "ids": [
                    {"id": int(value.item()), "pixels": int(count.item())}
                    for value, count in zip(values, counts, strict=True)
                ],
                "info": jsonify(sensor.data.info.get("instance_id_segmentation_fast", {})),
            }
        report["_audit"] = {"augmented_camera_configs": augmented_names}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
