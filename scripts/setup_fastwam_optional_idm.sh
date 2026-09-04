#!/usr/bin/env bash
set -euo pipefail

readonly FASTWAM_REPOSITORY="https://github.com/yuantianyuan01/FastWAM.git"
readonly FASTWAM_COMMIT="7faa71108368fbb3b6885649f112af607427a2d4"
readonly LIBERO_REPOSITORY="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
readonly LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
fastwam_dir="${FASTWAM_DIR:-${project_root}/external/FastWAM}"
libero_dir="${LIBERO_DIR:-${project_root}/external/LIBERO}"
venv_dir="${FASTWAM_VENV:-${project_root}/.venv-fastwam}"
python_bin="${PYTHON_BIN:-python3.10}"
patch_file="${project_root}/third_party/patches/fastwam_optional_idm_interventions.patch"

clone_at_commit() {
  local repository="$1"
  local commit="$2"
  local destination="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    mkdir -p "$(dirname "${destination}")"
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "${repository}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  local current
  current="$(git -C "${destination}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${current}" != "${commit}" ]]; then
    if [[ -n "$(git -C "${destination}" status --short)" ]]; then
      echo "Refusing to change a dirty checkout: ${destination}" >&2
      exit 1
    fi
    git -C "${destination}" checkout --detach "${commit}"
  fi
}

clone_at_commit "${FASTWAM_REPOSITORY}" "${FASTWAM_COMMIT}" "${fastwam_dir}"

if git -C "${fastwam_dir}" apply --check "${patch_file}"; then
  git -C "${fastwam_dir}" apply "${patch_file}"
elif git -C "${fastwam_dir}" apply --reverse --check "${patch_file}"; then
  echo "FastWAM intervention patch is already applied."
else
  echo "FastWAM checkout does not match the pinned patch." >&2
  exit 1
fi

clone_at_commit "${LIBERO_REPOSITORY}" "${LIBERO_COMMIT}" "${libero_dir}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_dir}"
fi
"${venv_dir}/bin/python" -m ensurepip --upgrade
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
"${venv_dir}/bin/python" -m pip install -e "${fastwam_dir}"
"${venv_dir}/bin/python" -m pip install \
  robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 easydict \
  opencv-python robomimic==0.2.0 thop future cloudpickle matplotlib
"${venv_dir}/bin/python" -m pip install mujoco==3.3.2
# LIBERO's pinned setup.py does not expose its namespace package correctly to
# modern editable installers. Install its metadata, while the launcher adds the
# pinned checkout root to PYTHONPATH explicitly.
"${venv_dir}/bin/python" -m pip install --no-deps -e "${libero_dir}"
"${venv_dir}/bin/python" -m pip install -e "${project_root}[test]"

actual_commit="$(git -C "${fastwam_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${FASTWAM_COMMIT}" ]]; then
  echo "FastWAM commit verification failed: ${actual_commit}" >&2
  exit 1
fi

echo "FastWAM checkout: ${fastwam_dir}"
echo "FastWAM commit:   ${actual_commit}"
echo "LIBERO checkout: ${libero_dir}"
echo "Python:           ${venv_dir}/bin/python"
echo "Checkpoint weights were not downloaded by this script."
