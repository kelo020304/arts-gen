#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_OBJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_PYTHON="$GEN_OBJ_ROOT/.post_process/bin/python"

discover_conda_sh() {
  local candidate=""
  local conda_exe=""

  if [[ -n "${CONDA_SH:-}" ]]; then
    if [[ -f "$CONDA_SH" ]]; then
      printf '%s\n' "$CONDA_SH"
      return 0
    fi
    echo "Error: CONDA_SH is set but not found: $CONDA_SH" >&2
    return 1
  fi

  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE:-}" ]]; then
    candidate="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)/etc/profile.d/conda.sh"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  conda_exe="$(command -v conda 2>/dev/null || true)"
  if [[ -n "$conda_exe" && -x "$conda_exe" ]]; then
    candidate="$(cd "$(dirname "$conda_exe")/.." && pwd)/etc/profile.d/conda.sh"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "Error: conda.sh not found." >&2
  echo "Install Miniconda/Anaconda, or run with CONDA_SH=/path/to/conda.sh $0 ..." >&2
  return 1
}

CONDA_SH="$(discover_conda_sh)"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Error: conda.sh not found at $CONDA_SH" >&2
  exit 1
fi

if [[ ! -x "$WEB_PYTHON" ]]; then
  echo "Error: web editor virtualenv python not found at $WEB_PYTHON" >&2
  echo "Run 'uv venv --python 3.10 .post_process && uv pip install --python .post_process/bin/python -e .' from $GEN_OBJ_ROOT first." >&2
  exit 1
fi

if ! command -v setsid >/dev/null 2>&1; then
  echo "Error: setsid is required but was not found in PATH" >&2
  exit 1
fi

source "$CONDA_SH"

WEB_PORT="8080"
ISAAC_PORT_VALUE="8081"
WEB_ARGS=()
ISAAC_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      if [[ $# -lt 2 ]]; then
        echo "Error: --port requires a value" >&2
        exit 2
      fi
      WEB_PORT="$2"
      shift 2
      ;;
    --port=*)
      WEB_PORT="${1#*=}"
      shift
      ;;
    --isaac-port)
      if [[ $# -lt 2 ]]; then
        echo "Error: --isaac-port requires a value" >&2
        exit 2
      fi
      ISAAC_PORT_VALUE="$2"
      shift 2
      ;;
    --isaac-port=*)
      ISAAC_PORT_VALUE="${1#*=}"
      shift
      ;;
    *)
      WEB_ARGS+=("$1")
      shift
      ;;
  esac
done

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$ISAAC_PID" ]] && kill -0 "$ISAAC_PID" 2>/dev/null; then
    kill -TERM -- "-$ISAAC_PID" 2>/dev/null || true
    wait "$ISAAC_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

export ISAAC_PORT="$ISAAC_PORT_VALUE"

activate_conda_env() {
  local env_name="$1"
  local activate_rc=0
  local python_path=""

  conda activate "$env_name" || activate_rc=$?
  python_path="$(which python 2>/dev/null || true)"

  if [[ "${CONDA_DEFAULT_ENV:-}" != "$env_name" || -z "${CONDA_PREFIX:-}" || "$python_path" != "$CONDA_PREFIX/bin/python" ]]; then
    echo "Error: failed to activate conda env '$env_name' (python=${python_path:-not found}, CONDA_PREFIX=${CONDA_PREFIX:-unset})" >&2
    return 1
  fi

  if (( activate_rc != 0 )); then
    echo "Warning: conda activate '$env_name' returned $activate_rc, but the expected python is active: $python_path" >&2
  fi
}

activate_conda_env env_isaaclab
setsid python "$GEN_OBJ_ROOT/utils/isaac_export_service.py" --host 127.0.0.1 --port "$ISAAC_PORT_VALUE" &
ISAAC_PID=$!

"$WEB_PYTHON" "$GEN_OBJ_ROOT/object_post_process/web_editor.py" --no-browser --port "$WEB_PORT" "${WEB_ARGS[@]}"
