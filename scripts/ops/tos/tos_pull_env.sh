#!/usr/bin/env bash
# Download the arts-gen Python environment from TOS into this repo.
# Run on the cloud dev instance.
#
# Usage:
#   bash scripts/ops/tos/tos_pull_env.sh
#
# Optional:
#   ENV_DIR=/path/where/to/extract bash scripts/ops/tos/tos_pull_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ENV_DIR="${ENV_DIR:-/opt/venvs/arts-gen}"
ARCHIVE="${ARCHIVE:-/tmp/arts_gen_env.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/env/arts_gen_env.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

mkdir -p "$(dirname "$ENV_DIR")" "$(dirname "$ARCHIVE")"
rm -rf "$ENV_DIR"

echo "[download] $TOS_URI -> $ARCHIVE"
"$TOSUTIL" cp "$TOS_URI" "$ARCHIVE"

echo "[extract] $ARCHIVE -> $ENV_DIR"
mkdir -p "$ENV_DIR"
tar -xzf "$ARCHIVE" -C "$ENV_DIR"

if [ -x "$ENV_DIR/bin/conda-unpack" ]; then
  echo "[unpack] fixing relocated env paths"
  "$ENV_DIR/bin/conda-unpack"
fi

if [ -x "$ENV_DIR/bin/python" ]; then
  "$ENV_DIR/bin/python" -V
fi
echo "[done] env extracted to $ENV_DIR"
echo "Activate with:"
echo "  source $ENV_DIR/bin/activate"
