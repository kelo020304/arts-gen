#!/usr/bin/env bash
# Package the smoke-test data directory and upload it to TOS so the dev
# machine can run pipeline / inference end-to-end on a known-good input.
#
# Default target is `data/smoke_test/` (PartNet-Mobility processed smoke set
# with renders + voxels + DINOv2 tokens + SS latents). Override DATA_DIR /
# TOS_URI to push other data subsets (e.g. Xiaomi raw).
#
# Usage:
#   bash scripts/ops/tos/tos_push_data.sh
#
# Optional:
#   DATA_DIR=/abs/path/to/data_dir TOS_URI=tos://.../mydata.tar.gz \
#     bash scripts/ops/tos/tos_push_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/smoke_test}"
ARCHIVE="${ARCHIVE:-/tmp/arts_smoke_test.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/data/smoke_test.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: data dir not found: $DATA_DIR" >&2
  echo "Set DATA_DIR=/path/to/your/data if it lives elsewhere." >&2
  exit 2
fi

mkdir -p "$(dirname "$ARCHIVE")"
rm -f "$ARCHIVE"

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
echo "[pack] $DATA_DIR -> $ARCHIVE"
tar -C "$(dirname "$DATA_DIR")" -czf "$ARCHIVE" "$(basename "$DATA_DIR")"
echo "[size] $(du -h "$ARCHIVE" | awk '{print $1}')"
echo "[upload] $ARCHIVE -> $TOS_URI"
"$TOSUTIL" cp "$ARCHIVE" "$TOS_URI"
echo "[done] uploaded data archive to $TOS_URI"
