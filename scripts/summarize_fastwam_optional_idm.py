#!/usr/bin/env python3
"""Summarize the complete, frozen FastWAM Optional-IDM smoke matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from imagined_future.fastwam_analysis import analyze_fastwam_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Runner output root, either above or at the manifest-ID directory.",
    )
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_903)
    parser.add_argument("--replay-tolerance", type=float, default=1e-6)
    parser.add_argument("--latent-distinct-tolerance", type=float, default=1e-6)
    parser.add_argument("--required-state-count", type=int)
    parser.add_argument(
        "--bootstrap-mode", choices=("state", "hierarchical"), default="state"
    )
    parser.add_argument(
        "--gate-mode",
        choices=("smoke_scale", "powered_evidence"),
        default="smoke_scale",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples must be nonnegative")
    if args.replay_tolerance < 0 or args.latent_distinct_tolerance < 0:
        raise ValueError("tolerances must be nonnegative")
    report, exit_code = analyze_fastwam_smoke(
        manifest_path=args.manifest,
        output_root=args.output_root,
        summary_dir=args.summary_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        replay_tolerance=args.replay_tolerance,
        latent_distinct_tolerance=args.latent_distinct_tolerance,
        required_state_count=args.required_state_count,
        bootstrap_mode=args.bootstrap_mode,
        gate_mode=args.gate_mode,
        make_plot=not args.no_plot,
    )
    print(
        f"status={report['status']} "
        f"decision={report['gate']['decision']} "
        f"summary_dir={args.summary_dir.resolve()}"
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
