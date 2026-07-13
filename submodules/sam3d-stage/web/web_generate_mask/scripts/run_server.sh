#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3-process

export SAM3_CKPT_PATH="${SAM3_CKPT_PATH:-$HOME/cfy/arts/arts-recon/submodules/sam3/ckpt/sam3.pt}"
export WEB_DATA_DIR="${WEB_DATA_DIR:-$PWD/data}"

export http_proxy="${http_proxy:-http://127.0.0.1:10808}"
export https_proxy="${https_proxy:-http://127.0.0.1:10808}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

exec uvicorn backend.server:app --host 0.0.0.0 --port "${PORT:-8003}" --log-level info
