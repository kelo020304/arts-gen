#!/usr/bin/env bash
# Download a data archive from TOS and extract it on the dev machine.
#
# Default downloads `data/smoke_test.tar.gz` and extracts so the result lands
# at `<repo>/data/smoke_test/` (matches the local-machine path so YAML configs
# referencing `data/smoke_test/...` keep working).
#
# Usage:
#   bash scripts/ops/tos/tos_pull_data.sh
#
# Optional:
#   DATA_PARENT=/abs/path/where/to/extract bash scripts/ops/tos/tos_pull_data.sh
#   TOS_URI=tos://.../other_data.tar.gz bash scripts/ops/tos/tos_pull_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
# Archive's top-level entry is `smoke_test/`, so DATA_PARENT must be the
# directory that should contain it (i.e. `<repo>/data/`).
DATA_PARENT="${DATA_PARENT:-$REPO_ROOT/data}"
ARCHIVE="${ARCHIVE:-/tmp/arts_smoke_test.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/data/smoke_test.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

mkdir -p "$DATA_PARENT" "$(dirname "$ARCHIVE")"

echo "[download] $TOS_URI -> $ARCHIVE"
"$TOSUTIL" cp "$TOS_URI" "$ARCHIVE"

echo "[extract] $ARCHIVE -> $DATA_PARENT/"
tar -xzf "$ARCHIVE" -C "$DATA_PARENT"

ls -la "$DATA_PARENT/" | head
echo "[done] data extracted under $DATA_PARENT/"
