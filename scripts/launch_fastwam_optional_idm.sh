#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

: "${FASTWAM_ROOT:?Set FASTWAM_ROOT to the pinned, patched FastWAM checkout}"
: "${FASTWAM_LIBERO_ROOT:?Set FASTWAM_LIBERO_ROOT to the pinned LIBERO checkout}"
: "${FASTWAM_CHECKPOINT:?Set FASTWAM_CHECKPOINT to the Optional-IDM .pt file}"
: "${FASTWAM_DATASET_STATS:?Set FASTWAM_DATASET_STATS to the matching JSON file}"
: "${LIBERO_CONFIG_PATH:?Set LIBERO_CONFIG_PATH to an initialized, isolated LIBERO config directory}"

manifest="${FASTWAM_MANIFEST:-${project_root}/results/fastwam_optional_idm_smoke_v1/manifest.json}"
output_root="${FASTWAM_OUTPUT_ROOT:-${project_root}/results/fastwam_optional_idm}"
python_bin="${FASTWAM_PYTHON:-${project_root}/.venv-fastwam/bin/python}"
gpu_list="${FASTWAM_GPUS:-0}"
conditions="${FASTWAM_CONDITIONS:-native,self_latent,self_cache,donor_latent,donor_cache,wrong_latent,shuffled_cache,first_frame}"

IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
shard_count="${#gpu_ids[@]}"
if [[ "${shard_count}" -eq 0 ]]; then
  echo "FASTWAM_GPUS did not contain a GPU ID." >&2
  exit 1
fi

pids=()
for shard_index in "${!gpu_ids[@]}"; do
  gpu_id="${gpu_ids[${shard_index}]}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHONPATH="${FASTWAM_LIBERO_ROOT}:${FASTWAM_ROOT}:${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
    MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
    "${python_bin}" \
    "${project_root}/scripts/run_fastwam_optional_idm.py" \
    --fastwam-root "${FASTWAM_ROOT}" \
    --manifest "${manifest}" \
    --checkpoint "${FASTWAM_CHECKPOINT}" \
    --dataset-stats "${FASTWAM_DATASET_STATS}" \
    --output-root "${output_root}" \
    --device cuda \
    --shard-index "${shard_index}" \
    --shard-count "${shard_count}" \
    --conditions "${conditions}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
