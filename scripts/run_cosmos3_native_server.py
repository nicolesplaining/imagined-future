#!/usr/bin/env python3
"""Launch the pinned public RoboLab policy service without auxiliary guardrails."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic-seed", action="store_true")
    args = parser.parse_args()

    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
        serve,
    )

    native_build_setup_args = RobolabPolicyService._build_setup_args

    def build_without_guardrails(self, server_args):
        return native_build_setup_args(self, server_args).model_copy(update={"guardrails": False})

    RobolabPolicyService._build_setup_args = build_without_guardrails
    serve(
        RobolabServerArgs(
            checkpoint_path=str(args.checkpoint),
            port=args.port,
            seed=args.seed,
            deterministic_seed=args.deterministic_seed,
        )
    )


if __name__ == "__main__":
    main()
