#!/usr/bin/env bash
# Run camera-view factorization jobs inside the RoboLab container against one research server.
# Usage: run_cosmos3_viewfactor_suite.sh JOBS.jsonl SHARD NUM_SHARDS [HOST] [PORT]
set -euo pipefail

jobs=${1:?jobs file is required}
shard=${2:?shard index is required}
num_shards=${3:?shard count is required}
server_host=${4:-localhost}
server_port=${5:-8001}

python=$(command -v python3 || echo /isaac-sim/python.sh)

cd /workspace/robolab
line_number=0
while IFS= read -r job; do
    if (( line_number++ % num_shards != shard )); then
        continue
    fi
    field() { "${python}" -c "import json,sys; v=json.loads(sys.argv[1])[sys.argv[2]]; print(' '.join(map(str,v)) if isinstance(v,list) else ('' if v is None else v))" "${job}" "$1"; }
    output_dir=$(field output_dir)
    if [[ -f "${output_dir}/summary.json" ]]; then
        echo "skip $(field unit): complete"
        continue
    fi
    mkdir -p "${output_dir}"
    echo "start $(field unit) $(date -u +%FT%TZ)"
    # shellcheck disable=SC2046
    /workspace/isaaclab/isaaclab.sh -p /research/scripts/run_cosmos3_robolab_branches.py \
        --headless \
        --device cuda:0 \
        --task "$(field task)" \
        --remote-host "${server_host}" \
        --remote-port "${server_port}" \
        --recorded-hdf5 "$(field recorded_hdf5)" \
        --output-dir "${output_dir}" \
        --branch-step "$(field branch_step)" \
        --branch-seeds $(field branch_seeds) \
        --frozen-recipient-seed "$(field recipient_seed)" \
        --frozen-donor-seed "$(field donor_seed)" \
        --natural-control-seed "$(field natural_control_seed)" \
        --target-object-name "$(field target_object_name)" \
        --gaussian-seed 1223 \
        --restore-strategy fresh_replay \
        --view-factorization \
        --study-id "$(field study_id)-$(date +%s)" \
        > "${output_dir}/run.log" 2>&1 || echo "FAILED $(field unit)"
    echo "done $(field unit) $(date -u +%FT%TZ)"
done < "${jobs}"
