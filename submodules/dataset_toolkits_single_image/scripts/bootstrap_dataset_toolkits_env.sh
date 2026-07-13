#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="dataset_toolkits"
ENV_FILE="${REPO_ROOT}/envs/dataset_toolkits.yaml"
CONDA_SH="${CONDA_SH:-/home/cfy/anaconda3/etc/profile.d/conda.sh}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
elif ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Set CONDA_SH=/path/to/conda.sh or initialize conda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[env] ${ENV_NAME} already exists; skipping conda env create"
else
  echo "[env] creating ${ENV_NAME} from ${ENV_FILE}"
  CONDA_NO_PLUGINS=true conda env create --solver classic -f "${ENV_FILE}"
fi

echo "[env] installing flash-attn after torch is present"
CONDA_NO_PLUGINS=true conda run -n "${ENV_NAME}" \
  python -m pip install --no-build-isolation "flash-attn==2.7.0.post2"

echo "[env] ready: conda activate ${ENV_NAME}"
