#!/usr/bin/env bash
# ============================================================
# Part Flow Inference Launcher (Phase 10 D-24/D-25/D-26)
# 调 pipeline/02_part_flow.py: occupancy.npz + tokens.npz + mask labels -> labels
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

OBJECT_ID="${OBJECT_ID:-100013}"
ANGLE_IDX="${ANGLE_IDX:-0}"
CKPT="${CKPT:-runs/partflow_smoke_4090/ckpts/step_5.pt}"
OCCUPANCY="${OCCUPANCY:-outputs/ss_flow_art/${OBJECT_ID}_angle${ANGLE_IDX}/occupancy.npz}"
TOKENS="${TOKENS:-data/smoke_test/reconstruction/dinov2_tokens/${OBJECT_ID}/angle_${ANGLE_IDX}/tokens.npz}"
MASK_TOKEN_LABELS="${MASK_TOKEN_LABELS:-}"
NUM_PARTS="${NUM_PARTS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/part_flow/${OBJECT_ID}_angle${ANGLE_IDX}}"
NUM_STEPS="${NUM_STEPS:-25}"
POSTPROCESS_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)        CKPT=$2; shift 2 ;;
        --occupancy)   OCCUPANCY=$2; shift 2 ;;
        --tokens)      TOKENS=$2; shift 2 ;;
        --mask-token-labels) MASK_TOKEN_LABELS=$2; shift 2 ;;
        --num-parts)   NUM_PARTS=$2; shift 2 ;;
        --output-dir)  OUTPUT_DIR=$2; shift 2 ;;
        --num-steps)   NUM_STEPS=$2; shift 2 ;;
        --postprocess) POSTPROCESS_FLAG="--postprocess"; shift ;;
        *) echo "[ERROR] unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$MASK_TOKEN_LABELS" ]] && { echo "[ERROR] --mask-token-labels is required"; exit 2; }
[[ -z "$NUM_PARTS" ]] && { echo "[ERROR] --num-parts K+1 including empty is required"; exit 2; }

echo "============================================================"
echo "Part Flow Inference"
echo "  OBJECT_ID:   $OBJECT_ID  (angle $ANGLE_IDX)"
echo "  CKPT:        $CKPT"
echo "  OCCUPANCY:   $OCCUPANCY"
echo "  TOKENS:      $TOKENS"
echo "  MASK_LABELS: $MASK_TOKEN_LABELS"
echo "  NUM_PARTS:   $NUM_PARTS"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  NUM_STEPS:   $NUM_STEPS"
[ -n "$POSTPROCESS_FLAG" ] && echo "  POSTPROCESS: on"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"
python "$ROOT/pipeline/02_part_flow.py" \
    --occupancy "$OCCUPANCY" \
    --tokens "$TOKENS" \
    --mask_token_labels "$MASK_TOKEN_LABELS" \
    --num_parts "$NUM_PARTS" \
    --ckpt "$CKPT" \
    --num_steps "$NUM_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    $POSTPROCESS_FLAG

echo "[part_flow_infer] done -> $OUTPUT_DIR"
