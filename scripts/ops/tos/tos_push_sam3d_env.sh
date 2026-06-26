#!/usr/bin/env bash
# Package the repo-local sam3d cu118 venv plus its Python 3.11 runtime and
# upload it to TOS. This is a venv archive, not a conda/conda-pack archive.
#
# Usage:
#   bash scripts/ops/tos/tos_push_sam3d_env.sh
#
# Optional:
#   VENV_DIR=/path/to/sam3d-cu118 bash scripts/ops/tos/tos_push_sam3d_env.sh
#   TOS_URI=tos://robot-data-lab/arts-reconstruction/env/sam3d_cu118_venv.tar.gz bash scripts/ops/tos/tos_push_sam3d_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DEFAULT_VENV_DIR="$REPO_ROOT/submodules/sam3d-stage/submodules/sam-3d-objects/.venv/sam3d-cu118"

VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"
ARCHIVE="${ARCHIVE:-/tmp/sam3d_cu118_venv.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/env/sam3d_cu118_venv.tar.gz}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

if [ ! -d "$VENV_DIR" ]; then
  echo "ERROR: sam3d venv dir not found: $VENV_DIR" >&2
  echo "Build it first with: PYTHON_BIN=/path/to/python3.11 bash scripts/ops/setup/setup_sam3d_env_cu118.sh" >&2
  exit 2
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "ERROR: venv python not executable: $VENV_DIR/bin/python" >&2
  exit 2
fi

VENV_DIR="$(cd "$VENV_DIR" && pwd)"
PYTHON_REAL="$(readlink -f "$VENV_DIR/bin/python")"
PYTHON_RUNTIME_DIR="$(cd "$(dirname "$PYTHON_REAL")/.." && pwd)"
PYTHON_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

if [ ! -x "$PYTHON_RUNTIME_DIR/bin/python3.11" ] || [ ! -d "$PYTHON_RUNTIME_DIR/lib/python3.11" ]; then
  echo "ERROR: could not locate a complete Python 3.11 runtime for $VENV_DIR" >&2
  echo "       resolved python: $PYTHON_REAL" >&2
  echo "       runtime dir    : $PYTHON_RUNTIME_DIR" >&2
  exit 2
fi

PYVENV_HOME=""
if [ -f "$VENV_DIR/pyvenv.cfg" ]; then
  PYVENV_HOME="$(awk -F'= ' '$1 == "home " || $1 == "home" {print $2; exit}' "$VENV_DIR/pyvenv.cfg" || true)"
fi

mkdir -p "$(dirname "$ARCHIVE")"
rm -f "$ARCHIVE"
STAGE="$(mktemp -d -t sam3d_venv_pack.XXXXXX)"
cleanup() {
  rm -rf "$STAGE"
}
trap cleanup EXIT

mkdir -p "$STAGE/venv" "$STAGE/python-runtime"

echo "[pack] venv runtime: $PYTHON_RUNTIME_DIR"
tar -C "$PYTHON_RUNTIME_DIR" -cf - . | tar -C "$STAGE/python-runtime" -xf -

echo "[pack] venv: $VENV_DIR"
tar -C "$VENV_DIR" --exclude='./.python-runtime' -cf - . | tar -C "$STAGE/venv" -xf -

{
  printf 'archive_version=%q\n' "1"
  printf 'python_version=%q\n' "$PYTHON_VERSION"
  printf 'source_repo_root=%q\n' "$REPO_ROOT"
  printf 'source_venv_dir=%q\n' "$VENV_DIR"
  printf 'source_python_runtime_dir=%q\n' "$PYTHON_RUNTIME_DIR"
  printf 'source_python_real=%q\n' "$PYTHON_REAL"
  printf 'source_pyvenv_home=%q\n' "$PYVENV_HOME"
} > "$STAGE/metadata.env"

echo "[archive] $STAGE -> $ARCHIVE"
tar -C "$STAGE" -czf "$ARCHIVE" metadata.env python-runtime venv

echo "[size] $(du -h "$ARCHIVE" | awk '{print $1}')"
echo "[upload] $ARCHIVE -> $TOS_URI"
"$TOSUTIL" cp "$ARCHIVE" "$TOS_URI"
echo "[done] uploaded sam3d venv archive to $TOS_URI"
