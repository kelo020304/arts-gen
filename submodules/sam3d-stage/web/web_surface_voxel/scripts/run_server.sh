#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects-process

# Required: point at sam3d checkpoints
export SAM3D_CONFIG_PATH="${SAM3D_CONFIG_PATH:-$HOME/cfy/arts/arts-recon/submodules/sam-3d-objects/checkpoints/hf/pipeline.yaml}"
export WEB_DATA_DIR="${WEB_DATA_DIR:-$PWD/data}"

# Proxy for HF downloads (DINO etc.)
export http_proxy="${http_proxy:-http://127.0.0.1:10808}"
export https_proxy="${https_proxy:-http://127.0.0.1:10808}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

exec uvicorn backend.server:app --host 0.0.0.0 --port "${PORT:-8001}" --log-level info
