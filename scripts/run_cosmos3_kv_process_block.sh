#!/usr/bin/env bash
set -euo pipefail

container=${1:?server container is required}
port=${2:?server port is required}
block=${3:?process block id is required}
output_root=${4:-/lambda/nfs/imagined-future/results/cosmos3_kv_patch_process_blocks_v1}
container_output_root=${output_root/#\/lambda\/nfs\/imagined-future/\/research}
if [[ "${container_output_root}" == "${output_root}" ]]; then
    echo "output root must be under /lambda/nfs/imagined-future" >&2
    exit 2
fi

for task in rubiks mustard spoon marker; do
    case "${task}" in
        rubiks)
            asset=/research/results/cosmos3_multitask_branches/rubiks_step64/pilot/native_223.npz
            hdf5=/research/results/cosmos3_multitask_screen/cosmos3_screen_v1/RubiksCubeTask/run_0.hdf5
            branch=/research/results/cosmos3_multitask_branches/rubiks_step64/pilot/summary.json
            recipient_seed=601
            donor_seed=607
            ;;
        mustard)
            asset=/research/results/cosmos3_multitask_branches/mustard_step96/pilot/native_223.npz
            hdf5=/research/results/cosmos3_multitask_screen/cosmos3_screen_v1/MustardInLeftBinTask/run_0.hdf5
            branch=/research/results/cosmos3_multitask_branches/mustard_step96/pilot/summary.json
            recipient_seed=613
            donor_seed=617
            ;;
        spoon)
            asset=/research/results/cosmos3_multitask_branches/spoon_step128/pilot/native_227.npz
            hdf5=/research/results/cosmos3_multitask_screen/cosmos3_screen_v1/SpoonInMugTask/run_0.hdf5
            branch=/research/results/cosmos3_multitask_branches/spoon_step128/pilot/summary.json
            recipient_seed=619
            donor_seed=631
            ;;
        marker)
            asset=/research/results/cosmos3_multitask_branches/marker_step320/pilot/native_211.npz
            hdf5=/research/results/cosmos3_multitask_screen/cosmos3_screen_v1/MarkerInMugTask/run_0.hdf5
            branch=/research/results/cosmos3_multitask_branches/marker_step320/pilot/summary.json
            recipient_seed=641
            donor_seed=643
            ;;
    esac
    output_dir=${output_root}/${block}/${task}
    mkdir -p "${output_dir}"
    sudo docker exec "${container}" bash -lc \
        "LD_LIBRARY_PATH= PYTHONPATH=/research/src /workspace/.venv/bin/python \
        /research/scripts/run_cosmos3_kv_patch_scan.py \
        --host localhost --port ${port} \
        --asset-video ${asset} --recorded-hdf5 ${hdf5} \
        --branch-summary ${branch} \
        --output ${container_output_root}/${block}/${task}/report.json \
        --study-id cosmos3-kv-patch-process-v1-${block}-${task} \
        --recipient-seed ${recipient_seed} --donor-seed ${donor_seed}" \
        > "${output_dir}/run.log" 2>&1
done
