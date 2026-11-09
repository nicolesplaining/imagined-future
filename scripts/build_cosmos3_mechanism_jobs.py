#!/usr/bin/env python3
"""Build mechanism-suite jobs for the frozen Cosmos 3 population units.

Every job reuses a completed unit's frozen recipient, donor, and natural seeds.
Two extra donors are the lowest registered native seeds outside that trio, and
the foreign (shuffled) future is the recipient video of the next completed unit
of the same task. Both rules are fixed before any mechanism outcome exists.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-root", type=Path, default=Path("results/cosmos3_population_confirmatory_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/cosmos3_replication.toml"))
    parser.add_argument("--output-root", default="/research/results/cosmos3_mechanism_v1")
    parser.add_argument("--extra-donors", type=int, default=2)
    parser.add_argument("--study-prefix", default="cosmos3-mechanism-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registered = sorted(tomllib.loads(args.config.read_text())["screen"]["native_branch_seeds"])
    units = {}
    for path in sorted(args.population_root.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        units[path.parent.name] = summary
    by_task: dict[str, list[str]] = {}
    for name, summary in units.items():
        by_task.setdefault(summary["task"], []).append(name)
    for names in by_task.values():
        names.sort(key=lambda name: units[name]["environment_seed"])

    jobs = []
    for name, summary in units.items():
        recipient, donor = int(summary["recipient_seed"]), int(summary["donor_seed"])
        natural = int(summary["natural_control_seed"])
        trio = [recipient, donor, natural]
        extras = [seed for seed in registered if seed not in trio][: args.extra_donors]
        siblings = by_task[summary["task"]]
        foreign_unit = siblings[(siblings.index(name) + 1) % len(siblings)]
        foreign = None
        if foreign_unit != name:
            other = units[foreign_unit]
            foreign = other["native"][str(other["recipient_seed"])]["video_path"]
        jobs.append(
            {
                "unit": name,
                "task": summary["task"],
                "recorded_hdf5": summary["recorded_hdf5"],
                "branch_step": int(summary["branch_step"]),
                "target_object_name": summary["target_object_name"],
                "recipient_seed": recipient,
                "donor_seed": donor,
                "natural_control_seed": natural,
                "branch_seeds": trio + extras,
                "foreign_unit": foreign_unit if foreign else None,
                "foreign_donor_video": foreign,
                "output_dir": f"{args.output_root}/{name}",
                "study_id": f"{args.study_prefix}-{summary['task']}-seed-{summary['environment_seed']}",
            }
        )
    args.output.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs))
    print(f"wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()
