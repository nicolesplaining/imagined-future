#!/usr/bin/env bash
set -euo pipefail

project_dir="${1:?usage: bootstrap_remote.sh PROJECT_DIR EXTERNAL_DIR}"
external_dir="${2:?usage: bootstrap_remote.sh PROJECT_DIR EXTERNAL_DIR}"
cosmos_dir="${external_dir}/cosmos-policy"
cosmos_commit="18a2accadf4e7a3531e56754102af5a24d2316da"

mkdir -p "${external_dir}"
if [[ ! -d "${cosmos_dir}/.git" ]]; then
  git clone https://github.com/NVlabs/cosmos-policy.git "${cosmos_dir}"
fi
git -C "${cosmos_dir}" fetch origin
git -C "${cosmos_dir}" checkout --detach "${cosmos_commit}"

echo "project=${project_dir}"
echo "cosmos=${cosmos_dir}"
echo "cosmos_commit=$(git -C "${cosmos_dir}" rev-parse HEAD)"
echo "Follow ${cosmos_dir}/SETUP.md to build the official Docker image."
echo "Inside the container, install this project with: uv pip install -e ${project_dir}"
