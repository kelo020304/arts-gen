#!/usr/bin/env bash
# Prepare large-file project directories on the cloud dev instance.
# By default, link them from the local code checkout to vePFS. If vePFS is not
# writable or not available, set USE_LOCAL_STORAGE=1 to keep them on local disk.
#
# Run inside the cloud dev-instance terminal.
#
# Usage:
#   bash scripts/ops/setup/setup_cloud_storage.sh
#
# Optional:
#   REMOTE_DIR="$HOME/code/arts-reconstruction" VEPFS_DIR=/robot/data-lab/arts-reconstruction bash scripts/ops/setup/setup_cloud_storage.sh
#   USE_LOCAL_STORAGE=1 bash scripts/ops/setup/setup_cloud_storage.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
REMOTE_DIR="${REMOTE_DIR:-$REPO_ROOT}"
VEPFS_DIR="${VEPFS_DIR:-/robot/data-lab/arts-reconstruction}"
USE_LOCAL_STORAGE="${USE_LOCAL_STORAGE:-0}"

is_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_local_dir() {
  local path="$1"

  mkdir -p "$(dirname "$path")"
  if [ -L "$path" ]; then
    rm "$path"
  fi
  if [ -e "$path" ] && [ ! -d "$path" ]; then
    echo "WARN: $path exists and is not a directory; leaving it unchanged." >&2
    return
  fi
  mkdir -p "$path"
}

if is_enabled "$USE_LOCAL_STORAGE"; then
  ensure_local_dir "$REMOTE_DIR/data"
  ensure_local_dir "$REMOTE_DIR/runs"
  ensure_local_dir "$REMOTE_DIR/outputs"
  ensure_local_dir "$REMOTE_DIR/checkpoints"

  echo "[done] using local disk for large-file dirs under $REMOTE_DIR"
  exit 0
fi

if ! mkdir -p \
  "$VEPFS_DIR/data" \
  "$VEPFS_DIR/runs" \
  "$VEPFS_DIR/outputs" \
  "$VEPFS_DIR/checkpoints"; then
  echo "WARN: cannot write large-file dirs under $VEPFS_DIR." >&2
  echo "      Code is still available in $REMOTE_DIR, but data/output workflows need writable vePFS." >&2
  exit 0
fi

link_dir() {
  local target="$1"
  local link="$2"

  mkdir -p "$(dirname "$link")"
  if [ -L "$link" ]; then
    ln -sfn "$target" "$link"
    return
  fi
  if [ -d "$link" ]; then
    if [ -z "$(find "$link" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
      rmdir "$link"
    else
      echo "WARN: $link exists and is not empty; leaving it unchanged." >&2
      echo "      Expected large files under $target." >&2
      return
    fi
  elif [ -e "$link" ]; then
    echo "WARN: $link exists and is not a directory; leaving it unchanged." >&2
    return
  fi
  ln -s "$target" "$link"
}

link_dir "$VEPFS_DIR/data" "$REMOTE_DIR/data"
link_dir "$VEPFS_DIR/runs" "$REMOTE_DIR/runs"
link_dir "$VEPFS_DIR/outputs" "$REMOTE_DIR/outputs"
link_dir "$VEPFS_DIR/checkpoints" "$REMOTE_DIR/checkpoints"

echo "[done] linked large-file dirs from $REMOTE_DIR to $VEPFS_DIR"
