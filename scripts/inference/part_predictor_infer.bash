#!/usr/bin/env bash
# ============================================================
# Part Predictor Inference Launcher (Phase 10 D-24/D-25)
# 调 scripts/eval/part_predictor/infer.py: 单样本/批量推理 + 着色体素可视化
# part_predictor 不在 pipeline/ 内（独立模型，非 5 步串联的一部分）
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

OBJECT_ID="${OBJECT_ID:-100015}"
ANGLE_IDX="${ANGLE_IDX:-0}"
CKPT="${CKPT:-output/part_predictor_smoke/ckpts/step_5.pt}"
DATA_ROOT="${DATA_ROOT:-data/smoke_test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/part_predictor/${OBJECT_ID}_angle${ANGLE_IDX}}"
MANIFEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)        CKPT=$2; shift 2 ;;
        --data-root)   DATA_ROOT=$2; shift 2 ;;
        --obj-id)      OBJECT_ID=$2; shift 2 ;;
        --angle-idx)   ANGLE_IDX=$2; shift 2 ;;
        --output-dir)  OUTPUT_DIR=$2; shift 2 ;;
        --manifest)    MANIFEST=$2; shift 2 ;;
        *) echo "[ERROR] unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Part Predictor Inference"
echo "  CKPT:        $CKPT"
echo "  DATA_ROOT:   $DATA_ROOT"
[ -n "$MANIFEST" ] && echo "  MANIFEST:    $MANIFEST (batch mode)"
[ -z "$MANIFEST" ] && echo "  OBJECT:      $OBJECT_ID  (angle $ANGLE_IDX)"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"
ARGS=( --ckpt "$CKPT" --data_root "$DATA_ROOT" --output "$OUTPUT_DIR" )
if [ -n "$MANIFEST" ]; then
    ARGS+=( --manifest "$MANIFEST" )
else
    ARGS+=( --obj_id "$OBJECT_ID" --angle_idx "$ANGLE_IDX" )
fi

python "$ROOT/scripts/eval/part_predictor/infer.py" "${ARGS[@]}"

echo "[part_predictor_infer] done -> $OUTPUT_DIR"
