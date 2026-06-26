#!/usr/bin/env bash
# Download the arts-reconstruction pretrained weights bundle from TOS.
# Run on the cloud dev instance.
#
# Usage:
#   bash scripts/ops/tos/tos_pull_weights.sh
#
# Optional:
#   WEIGHTS_PARENT=/path/where/to/extract bash scripts/ops/tos/tos_pull_weights.sh
#       (the archive contains a top-level "pretrained/" directory)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
# TRELLIS pretrained is big -> default it onto VePFS (shared), NOT the local code
# dir. The platform scans this via THIRD_PARTY_WEIGHTS_DIR. Archive top-level is
# 'pretrained/', so it lands at <WEIGHTS_PARENT>/pretrained/...
WEIGHTS_PARENT="${WEIGHTS_PARENT:-/robot/data-lab/jzh/arts-gen/third-party-weights}"
ARCHIVE="${ARCHIVE:-/tmp/arts_pretrained.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/weights/arts_pretrained.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

mkdir -p "$WEIGHTS_PARENT" "$(dirname "$ARCHIVE")"

echo "[download] $TOS_URI -> $ARCHIVE"
"$TOSUTIL" cp "$TOS_URI" "$ARCHIVE"

echo "[extract] $ARCHIVE -> $WEIGHTS_PARENT/pretrained/"
tar -xzf "$ARCHIVE" -C "$WEIGHTS_PARENT"

ls -la "$WEIGHTS_PARENT/pretrained/" | head
echo "[done] weights extracted to $WEIGHTS_PARENT/pretrained/"
