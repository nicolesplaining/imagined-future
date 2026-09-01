#!/usr/bin/env bash
set -euo pipefail

screen_root=${1:?screen root is required}
output_dir=${2:?output directory is required}
fixed_current_video=${3:?fixed current video is required}
server_host=${4:-localhost}
server_port=${5:-8001}
study_id=${6:-cosmos3-kv-patch-v1-banana-step64-layer16}

mkdir -p "${output_dir}"
cd /workspace/robolab
/workspace/isaaclab/isaaclab.sh -p scripts/run_cosmos3_robolab_branches.py \
    --headless \
    --device cuda:0 \
    --task BananaInBowlTask \
    --remote-host "${server_host}" \
    --remote-port "${server_port}" \
    --recorded-hdf5 "${screen_root}/BananaInBowlTask/run_0.hdf5" \
    --output-dir "${output_dir}/pilot" \
    --branch-step 64 \
    --branch-seeds 307 311 313 317 \
    --gaussian-seed 1223 \
    --restore-strategy fresh_replay \
    --fixed-current-video "${fixed_current_video}" \
    --attention-kv-patch-layers 16 \
    --study-id "${study_id}" \
    > "${output_dir}/run.log" 2>&1
