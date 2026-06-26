#!/usr/bin/env bash
# Package the local software/ directory (Blender 4.4.0 linux-x64) and upload
# it to TOS. Blender is required by pipeline step 2 (multi-view rendering).
# The same x86_64 binary works on any modern headless Linux distro.
#
# Usage:
#   bash scripts/ops/tos/tos_push_software.sh
#
# Optional:
#   SOFTWARE_DIR=/path/to/software bash scripts/ops/tos/tos_push_software.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SOFTWARE_DIR="${SOFTWARE_DIR:-$REPO_ROOT/software}"
ARCHIVE="${ARCHIVE:-/tmp/arts_software.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/software/arts_software.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

if [ ! -d "$SOFTWARE_DIR" ]; then
  echo "ERROR: software dir not found: $SOFTWARE_DIR" >&2
  echo "Run scripts/ops/setup/setup_arts_gen.sh or download Blender first." >&2
  exit 2
fi

mkdir -p "$(dirname "$ARCHIVE")"
rm -f "$ARCHIVE"

SOFTWARE_DIR="$(cd "$SOFTWARE_DIR" && pwd)"
echo "[pack] $SOFTWARE_DIR -> $ARCHIVE"
# Skip the .tar.xz blender installer if still present; only ship the unpacked dir.
tar -C "$(dirname "$SOFTWARE_DIR")" \
  --exclude='*.tar.xz' --exclude='*.tar.gz' \
  -czf "$ARCHIVE" "$(basename "$SOFTWARE_DIR")"
echo "[size] $(du -h "$ARCHIVE" | awk '{print $1}')"
echo "[upload] $ARCHIVE -> $TOS_URI"
"$TOSUTIL" cp "$ARCHIVE" "$TOS_URI"
echo "[done] uploaded software archive to $TOS_URI"
