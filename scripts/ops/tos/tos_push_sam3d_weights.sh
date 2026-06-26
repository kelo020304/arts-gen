#!/usr/bin/env bash
# Package the SAM 3D Objects inference checkpoints and upload them to TOS.
#
# Source (default): /home/mi/下载/hf  — the sam3d weight drop, containing
#   pipeline.yaml, ss_generator.ckpt, slat_generator.ckpt, slat_decoder_*.ckpt,
#   ss_decoder.ckpt and the matching *.yaml configs (~12 GB total).
#
# The archive stores the *contents* of the source dir at the archive root
# (./pipeline.yaml, ./ss_generator.ckpt, ...) so the dev-machine puller can
# extract straight into /robot/data-lab/jzh/art-gen/weights with the files
# landing directly there (no nested hf/ dir). That directory is what the sam3d
# pipeline.yaml's workspace_dir points at.
#
# This is a SEPARATE TOS object from tos_push_weights.sh (which packs the
# TRELLIS pretrained/ bundle into weights/arts_pretrained.tar.gz). They do not
# clobber each other.
#
# Usage:
#   bash scripts/ops/tos/tos_push_sam3d_weights.sh
#
# Optional:
#   WEIGHTS_DIR=/path/to/hf bash scripts/ops/tos/tos_push_sam3d_weights.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/home/mi/下载/hf}"
ARCHIVE="${ARCHIVE:-/tmp/sam3d_hf.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_ROOT="${TOS_ROOT:-tos://robot-data-lab/arts-reconstruction}"
TOS_URI="${TOS_URI:-$TOS_ROOT/weights/sam3d_hf.tar.gz}"
# Hard floor on the packed archive size. The ckpts are ~12 GB and largely
# incompressible, so a healthy archive is many GB. Anything under this means
# the pack silently lost files — refuse to upload an empty/partial tarball.
MIN_ARCHIVE_BYTES="${MIN_ARCHIVE_BYTES:-5000000000}"  # 5 GB

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

if [ ! -d "$WEIGHTS_DIR" ]; then
  echo "ERROR: sam3d weights dir not found: $WEIGHTS_DIR" >&2
  echo "Set WEIGHTS_DIR=/path/to/hf if your weights live elsewhere." >&2
  exit 2
fi

# Sanity-check the key checkpoints exist before packing 12 GB.
for f in pipeline.yaml ss_generator.ckpt slat_generator.ckpt ss_decoder.ckpt; do
  if [ ! -f "$WEIGHTS_DIR/$f" ]; then
    echo "ERROR: expected sam3d weight file missing: $WEIGHTS_DIR/$f" >&2
    exit 3
  fi
done

mkdir -p "$(dirname "$ARCHIVE")"
rm -f "$ARCHIVE"

WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"
echo "[pack] contents of $WEIGHTS_DIR -> $ARCHIVE (this is ~12 GB, takes a while)"
tar -C "$WEIGHTS_DIR" -czf "$ARCHIVE" .

ARCHIVE_BYTES="$(stat -c %s "$ARCHIVE")"
echo "[size] $(du -h "$ARCHIVE" | awk '{print $1}') ($ARCHIVE_BYTES bytes)"
if [ "$ARCHIVE_BYTES" -lt "$MIN_ARCHIVE_BYTES" ]; then
  echo "ERROR: archive is only $ARCHIVE_BYTES bytes (< $MIN_ARCHIVE_BYTES floor)." >&2
  echo "Pack likely lost files; refusing to upload. Inspect $ARCHIVE." >&2
  exit 4
fi

echo "[upload] $ARCHIVE -> $TOS_URI"
"$TOSUTIL" cp "$ARCHIVE" "$TOS_URI"
echo "[done] uploaded sam3d weights archive to $TOS_URI"
echo "[next]  on dev: bash scripts/ops/tos/tos_pull_sam3d_weights.sh"
