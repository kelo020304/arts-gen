#!/usr/bin/env bash
# Package the local pretrained/ directory (DINOv2 + TRELLIS SS encoder/decoder
# + dinov2 repo) and upload it to TOS.
#
# Usage:
#   bash scripts/ops/tos/tos_push_weights.sh
#
# Optional:
#   WEIGHTS_DIR=/path/to/pretrained bash scripts/ops/tos/tos_push_weights.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO_ROOT/pretrained}"
ARCHIVE="${ARCHIVE:-/tmp/arts_pretrained.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/weights/arts_pretrained.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

if [ ! -d "$WEIGHTS_DIR" ]; then
  echo "ERROR: weights dir not found: $WEIGHTS_DIR" >&2
  echo "Set WEIGHTS_DIR=/path/to/pretrained if your weights live elsewhere." >&2
  exit 2
fi

mkdir -p "$(dirname "$ARCHIVE")"
rm -f "$ARCHIVE"

WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"
echo "[pack] $WEIGHTS_DIR -> $ARCHIVE"
tar -C "$(dirname "$WEIGHTS_DIR")" -czf "$ARCHIVE" "$(basename "$WEIGHTS_DIR")"
echo "[size] $(du -h "$ARCHIVE" | awk '{print $1}')"
echo "[upload] $ARCHIVE -> $TOS_URI"
"$TOSUTIL" cp "$ARCHIVE" "$TOS_URI"
echo "[done] uploaded weights archive to $TOS_URI"
