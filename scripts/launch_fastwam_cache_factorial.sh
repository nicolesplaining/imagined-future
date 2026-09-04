#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

: "${FASTWAM_ROOT:?Set FASTWAM_ROOT to the pinned, patched FastWAM checkout}"
: "${FASTWAM_LIBERO_ROOT:?Set FASTWAM_LIBERO_ROOT to the pinned LIBERO checkout}"
: "${FASTWAM_CHECKPOINT:?Set FASTWAM_CHECKPOINT to the Optional-IDM checkpoint}"
: "${FASTWAM_DATASET_STATS:?Set FASTWAM_DATASET_STATS to the matching statistics}"
: "${FASTWAM_FACTORIAL_MANIFEST:?Set the frozen cache-factorial manifest}"
: "${FASTWAM_BASE_MANIFEST:?Set the frozen powered parent manifest}"
: "${FASTWAM_BASE_OUTPUT_ROOT:?Set the completed powered parent output root}"
: "${FASTWAM_FACTORIAL_OUTPUT_ROOT:?Set the cache-factorial output root}"
: "${LIBERO_CONFIG_PATH:?Set the isolated LIBERO config directory}"

python_bin="${FASTWAM_PYTHON:-${project_root}/.venv-fastwam/bin/python}"
gpu_list="${FASTWAM_GPUS:-0}"
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
    "${project_root}/scripts/run_fastwam_cache_factorial.py" \
    --fastwam-root "${FASTWAM_ROOT}" \
    --factorial-manifest "${FASTWAM_FACTORIAL_MANIFEST}" \
    --base-manifest "${FASTWAM_BASE_MANIFEST}" \
    --base-output-root "${FASTWAM_BASE_OUTPUT_ROOT}" \
    --checkpoint "${FASTWAM_CHECKPOINT}" \
    --dataset-stats "${FASTWAM_DATASET_STATS}" \
    --output-root "${FASTWAM_FACTORIAL_OUTPUT_ROOT}" \
    --device cuda \
    --shard-index "${shard_index}" \
    --shard-count "${shard_count}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"

