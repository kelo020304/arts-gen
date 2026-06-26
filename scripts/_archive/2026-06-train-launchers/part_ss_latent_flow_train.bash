#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/train/_ddp_common.sh
source "$HERE/_ddp_common.sh"

# Reduce CUDA allocator fragmentation by default (train_arts.py also sets this as
# a fallback for non-shell launches). Honors any externally-provided value.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-TRELLIS-arts/configs/arts/part_ss_latent_flow/part_ss_latent_flow.yaml}"
SCRIPT="TRELLIS-arts/train_arts.py"

GPU_IDS="${GPU_IDS:-}"
NUM_GPUS="${NUM_GPUS:-1}"
if [ -n "$GPU_IDS" ]; then
    IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS"
    NUM_GPUS="${#GPU_IDS_ARR[@]}"
fi

DATA_ROOT="${DATA_ROOT:-}"
RECON_SUBDIR="${RECON_SUBDIR:-}"
MASK_SUBDIR="${MASK_SUBDIR:-}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/robot/data-lab/arts-gen-data/output}"
RUN_ID="${RUN_ID:-part-ss-latent-flow-4view-$(date +%m%d%H%M)}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-}"
SS_DECODER_CKPT="${SS_DECODER_CKPT:-}"
INSPECTION_ROOT="${INSPECTION_ROOT:-$OUTPUT_DIR/inspections}"
TRAIN_LOG="${TRAIN_LOG:-$RUN_DIR/train.log}"
LOAD_DIR="${LOAD_DIR:-}"
RESUME_STEP="${RESUME_STEP:-}"

mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$INSPECTION_ROOT"
exec > >(tee -a "$TRAIN_LOG") 2>&1

EXTRA_ARGS=()
if [ -n "$LOAD_DIR" ] || [ -n "$RESUME_STEP" ]; then
    if [ -z "$LOAD_DIR" ] || [ -z "$RESUME_STEP" ]; then
        echo "[ERROR] LOAD_DIR and RESUME_STEP must be set together" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--load-dir "$LOAD_DIR" --resume-step "$RESUME_STEP")
fi

OVERRIDES=()
[ -n "$DATA_ROOT" ]        && OVERRIDES+=("data.data_root=$DATA_ROOT")
[ -n "$RECON_SUBDIR" ]     && OVERRIDES+=("data.recon_subdir=$RECON_SUBDIR")
[ -n "$MASK_SUBDIR" ]      && OVERRIDES+=("data.mask_subdir=$MASK_SUBDIR")
[ -n "$MANIFEST_PATH" ]    && OVERRIDES+=("data.manifest_path=$MANIFEST_PATH")
[ -n "$OUTPUT_DIR" ]       && OVERRIDES+=("training.output_dir=$OUTPUT_DIR")
[ -n "$MAX_STEPS" ]        && OVERRIDES+=("training.max_steps=$MAX_STEPS")
[ -n "$LR" ]               && OVERRIDES+=("training.lr=$LR")
[ -n "$SS_DECODER_CKPT" ]  && OVERRIDES+=("eval.ss_decoder_ckpt=$SS_DECODER_CKPT")
[ -n "$INSPECTION_ROOT" ]  && OVERRIDES+=("eval.inspection_root=$INSPECTION_ROOT")

OVERRIDES+=("$@")

echo "============================================================"
echo "Part SS Latent Flow Training"
echo "  CONFIG:          $CONFIG"
echo "  NUM_GPUS:        $NUM_GPUS  GPU_IDS='$GPU_IDS'"
echo "  RUN_ID:          $RUN_ID"
echo "  RUN_DIR:         $RUN_DIR"
echo "  TRAIN_LOG:       $TRAIN_LOG"
[ -n "$MAX_STEPS" ] && echo "  MAX_STEPS:       $MAX_STEPS (override)"
echo "  OUTPUT_DIR:      $OUTPUT_DIR"
echo "  INSPECTION_ROOT: $INSPECTION_ROOT"
[ -n "$LOAD_DIR" ] && echo "  LOAD_DIR:        $LOAD_DIR"
[ -n "$RESUME_STEP" ] && echo "  RESUME_STEP:     $RESUME_STEP"
echo "  EXTRA OVERRIDES: ${OVERRIDES[*]:-<none>}"
echo "============================================================"

launch_ddp "$NUM_GPUS" "$GPU_IDS" "$SCRIPT" \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}" \
    "${OVERRIDES[@]}"
