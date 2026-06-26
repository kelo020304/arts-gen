#!/usr/bin/env bash
# Package this repo and upload code to TOS for a cloud dev instance.
# Submodule working trees are included by default so offline dev machines do
# not need public network access. Git metadata, caches, checkpoints, and large
# generated data are excluded.
#
# Usage:
#   bash scripts/ops/tos/tos_push_code.sh
#
# Optional:
#   TOS_URI=tos://robot-data-lab/arts-reconstruction/code/my_snapshot.tar.gz bash scripts/ops/tos/tos_push_code.sh
#   ARCHIVE=/tmp/arts_reconstruction_code.tar.gz TOSUTIL=tosutil bash scripts/ops/tos/tos_push_code.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ARCHIVE="${ARCHIVE:-/tmp/arts_reconstruction_code.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/code/latest.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

mkdir -p "$(dirname "$ARCHIVE")"
rm -f "$ARCHIVE"

excludes=(
  --exclude='./.git'
  --exclude='./.gsd'
  --exclude='./.pytest_cache'
  --exclude='./.mypy_cache'
  --exclude='./.ruff_cache'
  --exclude='./.cache'
  --exclude='./.venv'
  --exclude='./**/.venv'
  --exclude='./__pycache__'
  --exclude='./**/__pycache__'
  --exclude='./code_update'
  --exclude='./docs'
  --exclude='./data'
  --exclude='./sam3d_cu118_deps'
  --exclude='./pretrained'
  --exclude='./runs'
  --exclude='./outputs'
  --exclude='./tmp'
  --exclude='./.tmp'
  --exclude='./software'
  --exclude='./wandb'
  --exclude='./*.tar.gz'
  --exclude='./submodules/*/.git'
  --exclude='./submodules/*/.cache'
  --exclude='./submodules/*/.venv'
  --exclude='./submodules/*/ckpts'
  --exclude='./submodules/*/checkpoints'
  --exclude='./submodules/sam3d-stage/submodules/sam-3d-objects/.venv'
  --exclude='./submodules/TRELLIS.1/.git'
  --exclude='./submodules/TRELLIS.1/.cache'
  --exclude='./submodules/TRELLIS.1/ckpts'
  --exclude='./submodules/TRELLIS.1/**/*.egg-info'
  --exclude='./submodules/dataset_toolkits/.git'
)

echo "[submodules] including local submodule working trees under ./submodules"
echo "[pack] $REPO_ROOT -> $ARCHIVE"
tar -C "$REPO_ROOT" -czf "$ARCHIVE" "${excludes[@]}" .

echo "[size] $(du -h "$ARCHIVE" | awk '{print $1}')"
echo "[upload] $ARCHIVE -> $TOS_URI"
"$TOSUTIL" cp "$ARCHIVE" "$TOS_URI"
echo "[done] uploaded code archive to $TOS_URI"
