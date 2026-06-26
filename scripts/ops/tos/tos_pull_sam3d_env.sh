#!/usr/bin/env bash
# Download the sam3d cu118 venv archive from TOS and relocate it into this repo.
# The archive includes a Python 3.11 runtime, so the cloud box does not need
# python3.11 on PATH and does not need conda.
#
# Usage on the cloud dev instance:
#   bash scripts/ops/tos/tos_pull_sam3d_env.sh
#
# Optional:
#   VENV_DIR=/path/to/sam3d-cu118 bash scripts/ops/tos/tos_pull_sam3d_env.sh
#   FORCE=0 bash scripts/ops/tos/tos_pull_sam3d_env.sh
#   SKIP_IMPORT_CHECK=1 bash scripts/ops/tos/tos_pull_sam3d_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DEFAULT_VENV_DIR="$REPO_ROOT/submodules/sam3d-stage/submodules/sam-3d-objects/.venv/sam3d-cu118"

VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"
ARCHIVE="${ARCHIVE:-/tmp/sam3d_cu118_venv.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/env/sam3d_cu118_venv.tar.gz}"
FORCE="${FORCE:-1}"
SKIP_IMPORT_CHECK="${SKIP_IMPORT_CHECK:-0}"

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

if [ -e "$VENV_DIR" ] && ! is_enabled "$FORCE"; then
  echo "ERROR: target venv already exists: $VENV_DIR" >&2
  echo "Set FORCE=1 to replace it." >&2
  exit 3
fi

mkdir -p "$(dirname "$ARCHIVE")"
STAGE="$(mktemp -d -t sam3d_venv_pull.XXXXXX)"
cleanup() {
  rm -rf "$STAGE"
}
trap cleanup EXIT

echo "[download] $TOS_URI -> $ARCHIVE"
"$TOSUTIL" cp "$TOS_URI" "$ARCHIVE"

echo "[extract] $ARCHIVE -> $STAGE"
tar -xzf "$ARCHIVE" -C "$STAGE"

if [ ! -d "$STAGE/venv" ] || [ ! -d "$STAGE/python-runtime" ]; then
  echo "ERROR: archive layout is invalid; expected venv/ and python-runtime/" >&2
  exit 2
fi

archive_version=""
python_version="3.11"
source_repo_root=""
source_venv_dir=""
source_python_runtime_dir=""
source_python_real=""
source_pyvenv_home=""
if [ -f "$STAGE/metadata.env" ]; then
  # shellcheck disable=SC1090
  source "$STAGE/metadata.env"
fi

VENV_DIR="$(mkdir -p "$(dirname "$VENV_DIR")" && cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"
RUNTIME_DIR="$VENV_DIR/.python-runtime/cpython-3.11"

echo "[install] replacing venv at $VENV_DIR"
rm -rf "$VENV_DIR"
mkdir -p "$VENV_DIR"
tar -C "$STAGE/venv" -cf - . | tar -C "$VENV_DIR" -xf -

rm -rf "$VENV_DIR/.python-runtime"
mkdir -p "$VENV_DIR/.python-runtime"
mv "$STAGE/python-runtime" "$RUNTIME_DIR"

rm -f "$VENV_DIR/bin/python" "$VENV_DIR/bin/python3" "$VENV_DIR/bin/python3.11"
ln -s "../.python-runtime/cpython-3.11/bin/python3.11" "$VENV_DIR/bin/python3.11"
ln -s "python3.11" "$VENV_DIR/bin/python3"
ln -s "python3.11" "$VENV_DIR/bin/python"

python_version="$("$VENV_DIR/bin/python3.11" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
cat > "$VENV_DIR/pyvenv.cfg" <<EOF
home = $RUNTIME_DIR/bin
include-system-site-packages = false
version = $python_version
executable = $RUNTIME_DIR/bin/python3.11
command = $RUNTIME_DIR/bin/python3.11 -m venv $VENV_DIR
EOF

export REPO_ROOT VENV_DIR RUNTIME_DIR
export OLD_REPO_ROOT="$source_repo_root"
export OLD_VENV_DIR="$source_venv_dir"
export OLD_RUNTIME_DIR="$source_python_runtime_dir"
export OLD_PYTHON_REAL="$source_python_real"
export OLD_PYVENV_HOME="$source_pyvenv_home"

"$VENV_DIR/bin/python3.11" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["VENV_DIR"])
replacements = [
    (os.environ.get("OLD_REPO_ROOT", ""), os.environ["REPO_ROOT"]),
    (os.environ.get("OLD_VENV_DIR", ""), os.environ["VENV_DIR"]),
    (os.environ.get("OLD_RUNTIME_DIR", ""), os.environ["RUNTIME_DIR"]),
    (os.environ.get("OLD_PYTHON_REAL", ""), f"{os.environ['RUNTIME_DIR']}/bin/python3.11"),
    (os.environ.get("OLD_PYVENV_HOME", ""), f"{os.environ['RUNTIME_DIR']}/bin"),
]
replacements = [(old.encode(), new.encode()) for old, new in replacements if old and old != new]

changed = 0
for path in root.rglob("*"):
    if path.is_symlink() or not path.is_file():
        continue
    try:
        data = path.read_bytes()
    except OSError:
        continue
    if b"\0" in data[:4096]:
        continue
    new_data = data
    for old, new in replacements:
        new_data = new_data.replace(old, new)
    if new_data != data:
        path.write_bytes(new_data)
        changed += 1

print(f"[relocate] text files updated: {changed}")
PY

"$VENV_DIR/bin/python" -V

if ! is_enabled "$SKIP_IMPORT_CHECK"; then
  echo "[check] sam3d + pytorch3d imports"
  LIDRA_SKIP_INIT=true PYTHONPATH="$REPO_ROOT/TRELLIS-arts" "$VENV_DIR/bin/python" - <<'PY'
import pytorch3d
import sam3d_objects
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
print("sam3d import OK")
PY
fi

echo "[done] sam3d cu118 venv extracted to $VENV_DIR"
echo "Use it with:"
echo "  export SAM3D_VENV_PYTHON=$VENV_DIR/bin/python"
