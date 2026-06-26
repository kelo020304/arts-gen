#!/usr/bin/env bash
# Pull the offline sam3d cu118 dependency bundle from TOS onto the dev box.
# Run on the OFFLINE H20 + CUDA 11.8 dev box BEFORE setup_sam3d_env_cu118.sh.
#
# Bundle contents (built/fetched on an online machine, see commit log):
#   kaolin-0.17.0-cp311-cu118.whl  (prebuilt, torch 2.5.1+cu118)
#   pytorch3d/ gsplat/ MoGe/       (source at sam3d's pinned commits; gsplat has glm submodule)
# These are the deps an offline box CANNOT fetch (NVIDIA-S3 / github).
#
# Usage:
#   bash scripts/ops/tos/tos_pull_sam3d_cu118_deps.sh
# Optional:
#   SAM3D_DEPS_DIR=/path/to/extract bash scripts/ops/tos/tos_pull_sam3d_cu118_deps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SAM3D_DEPS_DIR="${SAM3D_DEPS_DIR:-$REPO_ROOT/sam3d_cu118_deps}"
ARCHIVE="${ARCHIVE:-/tmp/sam3d_cu118_deps.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/weights/sam3d_cu118_deps.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH." >&2; exit 127; }

mkdir -p "$SAM3D_DEPS_DIR" "$(dirname "$ARCHIVE")"
echo "[download] $TOS_URI -> $ARCHIVE"
"$TOSUTIL" cp "$TOS_URI" "$ARCHIVE"
echo "[extract] $ARCHIVE -> $SAM3D_DEPS_DIR/"
tar -xzf "$ARCHIVE" -C "$SAM3D_DEPS_DIR"

# Verify the 4 pieces landed (fail loudly, no half-populated dir).
for item in kaolin-0.17.0-cp311-cu118.whl pytorch3d gsplat MoGe; do
  if [ ! -e "$SAM3D_DEPS_DIR/$item" ]; then
    echo "ERROR: missing $item under $SAM3D_DEPS_DIR after extract" >&2; exit 4
  fi
done
ls -la "$SAM3D_DEPS_DIR"
echo "[done] sam3d cu118 deps at $SAM3D_DEPS_DIR"
echo "[next] bash scripts/ops/setup/setup_sam3d_env_cu118.sh   (it reads SAM3D_DEPS_DIR=$SAM3D_DEPS_DIR)"
