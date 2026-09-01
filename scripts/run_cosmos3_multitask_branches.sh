#!/usr/bin/env bash
set -euo pipefail

screen_root=${1:?screen root is required}
output_root=${2:?output root is required}
server_host=${3:-localhost}
server_port=${4:-8001}
study_version=${5:-v2}

run_task() {
    task=$1
    branch_step=$2
    slug=$3
    output_dir="${output_root}/${slug}"
    mkdir -p "${output_dir}"
    /workspace/isaaclab/isaaclab.sh -p scripts/run_cosmos3_robolab_branches.py \
        --headless \
        --device cuda:0 \
        --task "${task}" \
        --remote-host "${server_host}" \
        --remote-port "${server_port}" \
        --recorded-hdf5 "${screen_root}/${task}/run_0.hdf5" \
        --output-dir "${output_dir}/pilot" \
        --branch-step "${branch_step}" \
        --branch-seeds 211 223 227 229 \
        --gaussian-seed 1223 \
        --restore-strategy fresh_replay \
        --multi-donor \
        --study-id "cosmos3-cross-task-${study_version}-${slug}" \
        > "${output_dir}/run.log" 2>&1
}

cd /workspace/robolab
run_task RubiksCubeTask 64 rubiks_step64
run_task MustardInLeftBinTask 96 mustard_step96
run_task SpoonInMugTask 128 spoon_step128
run_task MarkerInMugTask 320 marker_step320
