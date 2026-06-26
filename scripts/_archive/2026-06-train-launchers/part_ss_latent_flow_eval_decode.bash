#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-TRELLIS-arts/configs/arts/part_ss_latent_flow/part_ss_latent_flow.yaml}"
SCRIPT="TRELLIS-arts/eval_part_ss_latent_flow.py"

GPU_ID="${GPU_ID:-0}"
CHECKPOINT="${CHECKPOINT:-}"
LOAD_DIR="${LOAD_DIR:-}"
STEP="${STEP:-}"
DATA_ROOT="${DATA_ROOT:-}"
RECON_SUBDIR="${RECON_SUBDIR:-}"
MASK_SUBDIR="${MASK_SUBDIR:-}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
OBJECT_IDS="${OBJECT_IDS:-}"
NUM_STEPS="${NUM_STEPS:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/robot/data-lab/arts-gen-data/output}"
RUN_ID="${RUN_ID:-manual_eval}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/part_ss_latent_flow/$RUN_ID}"
INSPECTION_ROOT="${INSPECTION_ROOT:-$RUN_DIR/inspections_eval}"
EVAL_LOG="${EVAL_LOG:-$RUN_DIR/eval_decode.log}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$RUN_DIR" "$INSPECTION_ROOT"
exec > >(tee -a "$EVAL_LOG") 2>&1

CKPT_ARGS=()
if [ -n "$CHECKPOINT" ]; then
    CKPT_ARGS+=(--ckpt "$CHECKPOINT")
else
    if [ -z "$LOAD_DIR" ] || [ -z "$STEP" ]; then
        echo "[ERROR] Set CHECKPOINT=/path/step_N.pt or set LOAD_DIR=/path/to/run and STEP=N" >&2
        exit 1
    fi
    CKPT_ARGS+=(--load-dir "$LOAD_DIR" --step "$STEP")
fi

OVERRIDES=()
[ -n "$DATA_ROOT" ]        && OVERRIDES+=("data.data_root=$DATA_ROOT")
[ -n "$RECON_SUBDIR" ]     && OVERRIDES+=("data.recon_subdir=$RECON_SUBDIR")
[ -n "$MASK_SUBDIR" ]      && OVERRIDES+=("data.mask_subdir=$MASK_SUBDIR")
[ -n "$MANIFEST_PATH" ]    && OVERRIDES+=("data.manifest_path=$MANIFEST_PATH")

OVERRIDES+=("$@")

echo "============================================================"
echo "Part SS Latent Flow Standalone Eval/Decode Launcher"
echo "  CONFIG:          $CONFIG"
echo "  GPU_ID:          $GPU_ID"
echo "  CHECKPOINT:      ${CHECKPOINT:-<from LOAD_DIR/STEP>}"
echo "  LOAD_DIR:        ${LOAD_DIR:-<none>}"
echo "  STEP:            ${STEP:-<none>}"
echo "  RUN_DIR:         $RUN_DIR"
echo "  INSPECTION_ROOT: $INSPECTION_ROOT"
echo "  EVAL_LOG:        $EVAL_LOG"
echo "  MAX_SAMPLES:     $MAX_SAMPLES"
echo "  SAMPLE_MODE:     $SAMPLE_MODE"
echo "  OBJECT_IDS:      ${OBJECT_IDS:-<none>}"
echo "  NUM_STEPS:       $NUM_STEPS"
echo "  EXTRA OVERRIDES: ${OVERRIDES[*]:-<none>}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python "$SCRIPT" \
    --config "$CONFIG" \
    "${CKPT_ARGS[@]}" \
    --inspection-root "$INSPECTION_ROOT" \
    --max-samples "$MAX_SAMPLES" \
    --sample-mode "$SAMPLE_MODE" \
    ${OBJECT_IDS:+--object-ids "$OBJECT_IDS"} \
    --num-steps "$NUM_STEPS" \
    --device "$DEVICE" \
    "${OVERRIDES[@]}"
