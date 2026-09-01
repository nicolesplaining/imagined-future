"""Execute every saved RoboCasa action condition from one exact branch point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from imagined_future.cosmos_config import robocasa_policy_config
from imagined_future.robocasa import environment_action, physical_state_vector
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run-dir", type=Path, required=True)
    parser.add_argument("--action-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite execution: {args.output}")

    from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
        create_robocasa_env,
        prepare_observation,
    )
    from cosmos_policy.utils.utils import set_seed_everywhere

    summary = json.loads((args.branch_run_dir / "summary.json").read_text())
    branch = np.load(args.branch_run_dir / "branches.npz", allow_pickle=False)
    actions = np.load(args.action_artifact, allow_pickle=False)
    cfg = robocasa_policy_config(summary["task_name"], unnormalize_actions=True)
    environment_seed = int(summary["environment_seed"])
    prefix_actions = branch["prefix_actions"]
    rows = []
    physical_features = []
    endpoint_proprios = []
    endpoint_states = []
    schema = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for key in sorted(actions.files):
        if not key.startswith("environment_"):
            continue
        condition = key.removeprefix("environment_")
        set_seed_everywhere(environment_seed)
        env, _kwargs = create_robocasa_env(
            cfg, seed=environment_seed, episode_idx=int(summary["episode_index"])
        )
        try:
            raw = env.reset()
            for _ in range(10):
                raw, _reward, _done, _info = env.step(np.zeros(env.action_spec[0].shape))
            for action in prefix_actions:
                raw, _reward, _done, _info = env.step(environment_action(action, env.action_dim))
            for action in actions[key]:
                raw, _reward, _done, _info = env.step(environment_action(action, env.action_dim))
            endpoint = prepare_observation(raw, cfg.flip_images)
            physical, current_schema = physical_state_vector(raw, env.sim.data.qpos)
            if schema is None:
                schema = current_schema
            elif schema != current_schema:
                raise RuntimeError("RoboCasa endpoint schema changed between action conditions")
            success = bool(env._check_success())
            endpoint_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        finally:
            env.close()
        image_path = args.output.parent / f"execution_{condition}_endpoint_primary.png"
        Image.fromarray(endpoint["primary_image"]).save(image_path)
        rows.append(
            {
                "condition": condition,
                "success": success,
                "endpoint_primary_image": str(image_path),
            }
        )
        physical_features.append(physical)
        endpoint_proprios.append(np.asarray(endpoint["proprio"], dtype=np.float64))
        endpoint_states.append(endpoint_state)
    endpoint_artifact = args.output.with_name(f"{args.output.stem}_endpoint_states.npz")
    np.savez_compressed(
        endpoint_artifact,
        conditions=np.asarray([row["condition"] for row in rows]),
        physical_features=np.stack(physical_features),
        endpoint_proprios=np.stack(endpoint_proprios),
        endpoint_states=np.stack(endpoint_states),
    )
    result = {
        "scope": "RoboCasa action-condition endpoint execution",
        "branch_run": str(args.branch_run_dir),
        "action_artifact": str(args.action_artifact),
        "endpoint_artifact": str(endpoint_artifact),
        "physical_observation_schema": schema,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
