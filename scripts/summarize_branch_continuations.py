"""Aggregate exact-state branch outcomes across shared continuation seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.inputs]
    identity = (payloads[0]["branch_run"], payloads[0]["branch_state_digest"])
    if any((payload["branch_run"], payload["branch_state_digest"]) != identity for payload in payloads[1:]):
        raise ValueError("continuation files do not describe the same exact-state branch set")
    continuation_seeds = [int(payload["continuation_seed"]) for payload in payloads]
    if len(set(continuation_seeds)) != len(continuation_seeds):
        raise ValueError("continuation seeds must be unique")

    by_branch = defaultdict(list)
    for payload in payloads:
        for outcome in payload["outcomes"]:
            by_branch[(int(outcome["branch_index"]), int(outcome["branch_seed"]))].append(
                {
                    "continuation_seed": int(payload["continuation_seed"]),
                    "success": bool(outcome["success"]),
                    "success_step": outcome["success_step"],
                }
            )
    rows = []
    for (index, branch_seed), outcomes in sorted(by_branch.items()):
        outcomes.sort(key=lambda outcome: outcome["continuation_seed"])
        successes = sum(outcome["success"] for outcome in outcomes)
        rows.append(
            {
                "branch_index": index,
                "branch_seed": branch_seed,
                "replications": len(outcomes),
                "successes": successes,
                "failures": len(outcomes) - successes,
                "robust_label": "success" if successes == len(outcomes) else "failure" if successes == 0 else "mixed",
                "outcomes": outcomes,
            }
        )

    result = {
        "scope": "descriptive robustness across shared continuation-policy seeds",
        "branch_run": identity[0],
        "branch_state_digest": identity[1],
        "continuation_seeds": sorted(continuation_seeds),
        "branches": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "branches.csv").open("w", newline="") as handle:
        fieldnames = ("branch_index", "branch_seed", "replications", "successes", "failures", "robust_label")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    print(json.dumps({"continuation_seeds": sorted(continuation_seeds), "branches": rows}, indent=2))


if __name__ == "__main__":
    main()
