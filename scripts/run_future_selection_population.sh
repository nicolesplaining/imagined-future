#!/usr/bin/env bash
# Cosmos Policy future selection: screen first-decision states, then run every
# state with mixed native outcomes (1-3 of 4 screen seeds succeed).
# Usage: run_future_selection_population.sh OUTPUT_ROOT [INIT_STATE_INDICES...]
set -euo pipefail

root=${1:?output root is required}
shift
indices=("${@:-0 1 2 3 4}")
script=/work/imagined-future/scripts/run_future_selection_success.py
mkdir -p "${root}"

if [[ ! -s "${root}/screen.json" ]]; then
    PYTHONPATH=/work/imagined-future/src uv run --no-sync python "${script}" screen \
        --output "${root}/screen.json" --initial-state-indices ${indices[*]} \
        > "${root}/screen.log" 2>&1
fi

uv run --no-sync python - "${root}/screen.json" > "${root}/selected.txt" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))["rows"]
for row in rows:
    if 0 < row["successes"] < row["seeds"]:
        print(row["task_id"], row["initial_state_index"])
EOF
echo "selected $(wc -l < "${root}/selected.txt") mixed-outcome states"

while read -r task init; do
    out="${root}/task${task}_init${init}"
    [[ -f "${out}/summary.json" ]] && continue
    echo "run task ${task} init ${init} $(date -u +%FT%TZ)"
    PYTHONPATH=/work/imagined-future/src uv run --no-sync python "${script}" run \
        --output-dir "${out}" --task-id "${task}" --initial-state-index "${init}" \
        > "${root}/task${task}_init${init}.log" 2>&1 || echo "FAILED task ${task} init ${init}"
done < "${root}/selected.txt"
echo "ALL_DONE"
