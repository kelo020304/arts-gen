#!/usr/bin/env bash
# Download a code archive from TOS and deploy it on a cloud dev instance.
#
# Run this inside the cloud dev-instance terminal.
#
# Usage:
#   bash scripts/ops/tos/tos_pull_code.sh
#
# Optional:
#   TOS_URI=tos://robot-data-lab/arts-reconstruction/code/my_snapshot.tar.gz bash scripts/ops/tos/tos_pull_code.sh
#   REMOTE_DIR="$HOME/code/arts-reconstruction" VEPFS_DIR=/robot/data-lab/arts-reconstruction bash scripts/ops/tos/tos_pull_code.sh
#   USE_LOCAL_STORAGE=1 bash scripts/ops/tos/tos_pull_code.sh
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/root/code/arts-gen}"
VEPFS_DIR="${VEPFS_DIR:-/robot/data-lab/arts-reconstruction}"
USE_LOCAL_STORAGE="${USE_LOCAL_STORAGE:-0}"
ARCHIVE="${ARCHIVE:-/tmp/arts_reconstruction_code.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/code/latest.tar.gz}"

is_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

mkdir -p "$(dirname "$ARCHIVE")" "$REMOTE_DIR"

echo "[download] $TOS_URI -> $ARCHIVE"
"$TOSUTIL" cp "$TOS_URI" "$ARCHIVE"

echo "[extract] $ARCHIVE -> $REMOTE_DIR"
tar -xzf "$ARCHIVE" -C "$REMOTE_DIR"

REMOTE_DIR="$REMOTE_DIR" VEPFS_DIR="$VEPFS_DIR" USE_LOCAL_STORAGE="$USE_LOCAL_STORAGE" bash "$REMOTE_DIR/scripts/ops/setup/setup_cloud_storage.sh"

echo "[done] code deployed to $REMOTE_DIR"
if is_enabled "$USE_LOCAL_STORAGE"; then
  echo "[data] large-file dirs are local under $REMOTE_DIR"
else
  echo "[data] large-file dirs are linked to $VEPFS_DIR"
fi
echo "Next:"
echo "  cd $REMOTE_DIR"
echo "  PYTHONPATH=TRELLIS-arts python3 -m pytest TRELLIS-arts/tests/arts/part_ss_latent_flow -q"
