#!/usr/bin/env bash
# Download the SAM 3D Objects inference checkpoints from TOS onto the dev box.
# Run on the cloud dev instance.
#
# The direct TOS folder holds the weight files at its root (pipeline.yaml,
# ss_generator.ckpt, ...). They download DIRECTLY into the target dir, which
# defaults to /robot/data-lab/jzh/art-gen/weights - the path the sam3d
# pipeline.yaml's workspace_dir expects (a.k.a. sam-3d-objects/checkpoints/hf).
# A legacy .tar.gz archive URI is still supported for old runs.
#
# Usage:
#   bash scripts/ops/tos/tos_pull_sam3d_weights.sh
#
# Optional:
#   WEIGHTS_DIR=/path/to/extract bash scripts/ops/tos/tos_pull_sam3d_weights.sh
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-/robot/data-lab/jzh/art-gen/weights}"
ARCHIVE="${ARCHIVE:-/tmp/sam3d_hf.tar.gz}"
TOSUTIL="${TOSUTIL:-tosutil}"
TOS_URI="${TOS_URI:-tos://robot-data-lab/jzh/sam3d_weights/}"
REQUIRE_ENCODER="${REQUIRE_ENCODER:-0}"

command -v "$TOSUTIL" >/dev/null 2>&1 || {
  echo "ERROR: '$TOSUTIL' not found in PATH. Install/configure tosutil first." >&2
  exit 127
}

mkdir -p "$WEIGHTS_DIR" "$(dirname "$ARCHIVE")"

check_nonempty() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "ERROR: expected non-empty weight file missing or empty: $path" >&2
    exit 4
  fi
}

verify_core_weights() {
  for f in pipeline.yaml ss_generator.ckpt slat_generator.ckpt ss_decoder.ckpt; do
    check_nonempty "$WEIGHTS_DIR/$f"
  done
  if [ "$REQUIRE_ENCODER" = "1" ]; then
    check_nonempty "$WEIGHTS_DIR/ss_encoder.safetensors"
    check_nonempty "$WEIGHTS_DIR/ss_encoder.yaml"
  fi
}

case "$TOS_URI" in
  *.tar.gz)
    echo "[download] $TOS_URI -> $ARCHIVE (~12 GB)"
    "$TOSUTIL" cp "$TOS_URI" "$ARCHIVE"

    echo "[extract] $ARCHIVE -> $WEIGHTS_DIR/"
    tar -xzf "$ARCHIVE" -C "$WEIGHTS_DIR"
    ;;
  */)
    echo "[download] $TOS_URI -> $WEIGHTS_DIR/ (~12 GB if empty)"
    files=(
      pipeline.yaml
      ss_generator.ckpt
      ss_generator.yaml
      ss_decoder.ckpt
      ss_decoder.yaml
      slat_generator.ckpt
      slat_generator.yaml
      slat_decoder_mesh.ckpt
      slat_decoder_mesh.pt
      slat_decoder_mesh.yaml
      slat_decoder_gs.ckpt
      slat_decoder_gs.yaml
      slat_decoder_gs_4.ckpt
      slat_decoder_gs_4.yaml
    )
    if [ "$REQUIRE_ENCODER" = "1" ]; then
      files+=(ss_encoder.safetensors ss_encoder.yaml)
    fi
    for f in "${files[@]}"; do
      echo "[download] $TOS_URI$f -> $WEIGHTS_DIR/$f"
      "$TOSUTIL" cp "$TOS_URI$f" "$WEIGHTS_DIR/$f"
    done
    ;;
  *)
    echo "ERROR: TOS_URI must be a .tar.gz archive or a folder URI ending with /: $TOS_URI" >&2
    exit 2
    ;;
esac

verify_core_weights

ls -la "$WEIGHTS_DIR" | head
echo "[done] sam3d weights extracted to $WEIGHTS_DIR/"
echo "[note] point the inference config at $WEIGHTS_DIR/pipeline.yaml"
