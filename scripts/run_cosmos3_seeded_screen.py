#!/usr/bin/env python3
"""Run the public RoboLab Cosmos 3 evaluator with an explicit environment seed.

RoboLab's public runner fixes ``create_env`` to its default seed and does not
expose that argument on the command line.  This wrapper preserves the public
evaluation loop and Cosmos 3 client, changing only the seed passed to
``robolab.core.environments.runtime.create_env``.  The resolved seed is then
recorded by RoboLab in ``env_cfg.json``.
"""

from __future__ import annotations

import argparse
import functools
import sys
import traceback

import cv2  # Must precede Isaac Lab imports.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--environment-seed", type=int, required=True)
parser.add_argument("--remote-host", default="localhost")
parser.add_argument("--remote-port", default=8000, type=int)

from robolab.eval.runner import add_common_eval_args, run_evaluation


add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("SEEDED_SCREEN_PROGRESS app_launched", flush=True)

from policies.cosmos3.client import Cosmos3Client  # noqa: E402
from robolab.core.environments import runtime  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import (  # noqa: E402
    WRIST_LEFT_RIGHT_HEAD,
)
print("SEEDED_SCREEN_PROGRESS imports_completed", flush=True)


_public_create_env = runtime.create_env
runtime.create_env = functools.partial(_public_create_env, seed=args.environment_seed)
auto_register_droid_envs(task=args.task, cameras=WRIST_LEFT_RIGHT_HEAD)
print("SEEDED_SCREEN_PROGRESS tasks_registered", flush=True)


def make_client(_args: argparse.Namespace) -> Cosmos3Client:
    return Cosmos3Client(remote_host=_args.remote_host, remote_port=_args.remote_port)


def main() -> None:
    print("SEEDED_SCREEN_PROGRESS evaluation_started", flush=True)
    run_evaluation(args, policy="cosmos3", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[RoboLab] Terminated with error: {error}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
