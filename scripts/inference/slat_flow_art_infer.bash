#!/usr/bin/env bash
# ============================================================
# SLat Flow Art Inference Launcher (Phase 10 D-24/D-25/D-26)
# 串联 pipeline/03_slat_flow.py + pipeline/03_final_decode.py
# image (multi-view) + occupancy.npz -> slat.pt -> mesh.obj + gaussians.ply
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

OBJECT_ID="${OBJECT_ID:-100013}"
ANGLE_IDX="${ANGLE_IDX:-0}"
CKPT_DIR="${CKPT_DIR:-pretrained/ckpts}"
INPUT_DIR="${INPUT_DIR:-data/smoke_test/test_data/renders/${OBJECT_ID}/angle_${ANGLE_IDX}/rgb}"
OCCUPANCY="${OCCUPANCY:-outputs/ss_flow_art/${OBJECT_ID}_angle${ANGLE_IDX}/occupancy.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/slat_flow_art/${OBJECT_ID}_angle${ANGLE_IDX}}"
NUM_STEPS="${NUM_STEPS:-25}"
FORMATS="${FORMATS:-mesh,gaussian}"
SLAT_FLOW_CKPT_NAME="${SLAT_FLOW_CKPT_NAME:-slat_flow_img_dit_L_64l8p2_fp16.safetensors}"
SLAT_DEC_CKPT_NAME="${SLAT_DEC_CKPT_NAME:-slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-dir)   CKPT_DIR=$2; shift 2 ;;
        --input-dir)  INPUT_DIR=$2; shift 2 ;;
        --occupancy)  OCCUPANCY=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --num-steps)  NUM_STEPS=$2; shift 2 ;;
        --formats)    FORMATS=$2; shift 2 ;;
        *) echo "[ERROR] unknown arg: $1"; exit 1 ;;
    esac
done

SLAT_FLOW_CKPT="$CKPT_DIR/$SLAT_FLOW_CKPT_NAME"
SLAT_DEC_CKPT="$CKPT_DIR/$SLAT_DEC_CKPT_NAME"

echo "============================================================"
echo "SLat Flow Art Inference"
echo "  OBJECT_ID:   $OBJECT_ID  (angle $ANGLE_IDX)"
echo "  INPUT_DIR:   $INPUT_DIR"
echo "  OCCUPANCY:   $OCCUPANCY"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  SLAT_FLOW:   $SLAT_FLOW_CKPT"
echo "  SLAT_DEC:    $SLAT_DEC_CKPT"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"
shopt -s nullglob
IMAGES=( "$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpg )
if [ "${#IMAGES[@]}" -eq 0 ]; then echo "[ERROR] no images under $INPUT_DIR"; exit 1; fi

python "$ROOT/pipeline/03_slat_flow.py" \
    --images "${IMAGES[@]}" \
    --occupancy "$OCCUPANCY" \
    --ckpt "$SLAT_FLOW_CKPT" \
    --num_steps "$NUM_STEPS" \
    --output_dir "$OUTPUT_DIR"

python "$ROOT/pipeline/03_final_decode.py" \
    --slat "$OUTPUT_DIR/slat.pt" \
    --ckpt "$SLAT_DEC_CKPT" \
    --formats "$FORMATS" \
    --output_dir "$OUTPUT_DIR"

echo "[slat_flow_art_infer] done -> $OUTPUT_DIR"
