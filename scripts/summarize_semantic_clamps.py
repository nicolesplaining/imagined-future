"""Create a validated JSON/CSV table from semantic-clamp run directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from imagined_future.clamp_summary import summarize_clamp_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-deterministic", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing summary: {args.output_dir}")
    rows = summarize_clamp_runs(args.run_dirs, require_deterministic=args.require_deterministic)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
