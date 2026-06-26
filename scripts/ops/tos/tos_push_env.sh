#!/usr/bin/env bash
# Package the local arts-gen conda env and upload it to TOS.
#
# Usage:
#   bash scripts/ops/tos/tos_push_env.sh
#
# Optional:
#   ENV_DIR=/path/to/env bash scripts/ops/tos/tos_push_env.sh
set -euo pipefail

ENV_DIR="${ENV_DIR:-/home/mi/anaconda3/envs/arts-gen}"
ARCHIVE="${ARCHIVE:-/tmp/arts_gen_env.tar.gz}"
CONDAPACK="${CONDAPACK:-/home/mi/anaconda3/bin/conda-pack}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/env/arts_gen_env.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

if [ ! -d "$ENV_DIR" ]; then
  echo "ERROR: env dir not found: $ENV_DIR" >&2
  echo "Set ENV_DIR=/path/to/env if your environment lives elsewhere." >&2
  exit 2
fi

command -v "$CONDAPACK" >/dev/null 2>&1 || {
  echo "ERROR: '$CONDAPACK' not found in PATH. Install conda-pack before packaging an env." >&2
  echo "Try: /home/mi/anaconda3/bin/python -m pip install conda-pack" >&2
  exit 127
}

mkdir -p "$(dirname "$ARCHIVE")"
rm -f "$ARCHIVE"

ENV_DIR="$(cd "$ENV_DIR" && pwd)"
echo "[pack] $ENV_DIR -> $ARCHIVE"
"$CONDAPACK" -p "$ENV_DIR" -o "$ARCHIVE" --force --ignore-editable-packages
echo "[size] $(du -h "$ARCHIVE" | awk '{print $1}')"
echo "[upload] $ARCHIVE -> $TOS_URI"
"$TOSUTIL" cp "$ARCHIVE" "$TOS_URI"
echo "[done] uploaded env archive to $TOS_URI"
