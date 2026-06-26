#!/usr/bin/env bash
# Download the arts-reconstruction software/ bundle (Blender) from TOS.
# Run on the cloud dev instance.
#
# Usage:
#   bash scripts/ops/tos/tos_pull_software.sh
#
# Optional:
#   SOFTWARE_PARENT=/path/where/to/extract bash scripts/ops/tos/tos_pull_software.sh
#       (the archive contains a top-level "software/" directory)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SOFTWARE_PARENT="${SOFTWARE_PARENT:-$REPO_ROOT}"
ARCHIVE="${ARCHIVE:-/tmp/arts_software.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/software/arts_software.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

mkdir -p "$SOFTWARE_PARENT" "$(dirname "$ARCHIVE")"

echo "[download] $TOS_URI -> $ARCHIVE"
"$TOSUTIL" cp "$TOS_URI" "$ARCHIVE"

echo "[extract] $ARCHIVE -> $SOFTWARE_PARENT/software/"
tar -xzf "$ARCHIVE" -C "$SOFTWARE_PARENT"

if [ -x "$SOFTWARE_PARENT/software/blender-4.4.0-linux-x64/blender" ]; then
  echo "[verify] Blender:"
  "$SOFTWARE_PARENT/software/blender-4.4.0-linux-x64/blender" --version | head -1
fi
echo "[done] software extracted to $SOFTWARE_PARENT/software/"
