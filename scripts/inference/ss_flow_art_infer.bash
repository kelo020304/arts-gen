#!/usr/bin/env bash
# ============================================================
# SS Flow Art Inference Launcher (Phase 10 D-24/D-25/D-26)
# 串联 pipeline/01_ss_flow_mv.py + pipeline/01_ss_decode.py
# image (multi-view) -> ss_latent.npz -> occupancy.npz
#
# Usage:
#   bash scripts/inference/ss_flow_art_infer.bash
#   OBJECT_ID=100013 bash scripts/inference/ss_flow_art_infer.bash
#   bash scripts/inference/ss_flow_art_infer.bash --ckpt-dir pretrained/ckpts \
#        --input-dir data/smoke_test/test_data/renders/100013/angle_0/rgb \
#        --output-dir outputs/ss_flow_art_smoke
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# Defaults (overridable via env or CLI)
OBJECT_ID="${OBJECT_ID:-100013}"
ANGLE_IDX="${ANGLE_IDX:-0}"
CKPT_DIR="${CKPT_DIR:-pretrained/ckpts}"
INPUT_DIR="${INPUT_DIR:-data/smoke_test/test_data/renders/${OBJECT_ID}/angle_${ANGLE_IDX}/rgb}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ss_flow_art/${OBJECT_ID}_angle${ANGLE_IDX}}"
NUM_STEPS="${NUM_STEPS:-25}"
THRESHOLD="${THRESHOLD:-0.0}"
SS_FLOW_CKPT_NAME="${SS_FLOW_CKPT_NAME:-ss_flow_img_dit_L_16l8_fp16.safetensors}"
SS_DEC_CKPT_NAME="${SS_DEC_CKPT_NAME:-ss_dec_conv3d_16l8_fp16.safetensors}"

# CLI parse (override env)
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-dir)   CKPT_DIR=$2; shift 2 ;;
        --input-dir)  INPUT_DIR=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --num-steps)  NUM_STEPS=$2; shift 2 ;;
        --threshold)  THRESHOLD=$2; shift 2 ;;
        *) echo "[ERROR] unknown arg: $1"; exit 1 ;;
    esac
done

SS_FLOW_CKPT="$CKPT_DIR/$SS_FLOW_CKPT_NAME"
SS_DEC_CKPT="$CKPT_DIR/$SS_DEC_CKPT_NAME"

echo "============================================================"
echo "SS Flow Art Inference"
echo "  OBJECT_ID:   $OBJECT_ID  (angle $ANGLE_IDX)"
echo "  INPUT_DIR:   $INPUT_DIR"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  SS_FLOW:     $SS_FLOW_CKPT"
echo "  SS_DEC:      $SS_DEC_CKPT"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"
shopt -s nullglob
IMAGES=( "$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpg )
if [ "${#IMAGES[@]}" -eq 0 ]; then
    echo "[ERROR] no images under $INPUT_DIR"; exit 1
fi

# Step 1: SS Flow (multi-view) -> ss_latent.npz
python "$ROOT/pipeline/01_ss_flow_mv.py" \
    --images "${IMAGES[@]}" \
    --ckpt "$SS_FLOW_CKPT" \
    --num_steps "$NUM_STEPS" \
    --output_dir "$OUTPUT_DIR"

# Step 2: SS decode -> occupancy.npz
python "$ROOT/pipeline/01_ss_decode.py" \
    --ss_latent "$OUTPUT_DIR/ss_latent.npz" \
    --ckpt "$SS_DEC_CKPT" \
    --threshold "$THRESHOLD" \
    --output_dir "$OUTPUT_DIR"

echo "[ss_flow_art_infer] done -> $OUTPUT_DIR"
