#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/train/_ddp_common.sh
source "$HERE/_ddp_common.sh"

CONFIG="${CONFIG:-TRELLIS-arts/configs/arts/ss_flow_global_z/full_train.yaml}"
SCRIPT="TRELLIS-arts/train_arts.py"

GPU_IDS="${GPU_IDS:-}"
NUM_GPUS="${NUM_GPUS:-1}"
if [ -n "$GPU_IDS" ]; then
    IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS"
    NUM_GPUS="${#GPU_IDS_ARR[@]}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/robot/data-lab/jzh/art-gen-output}"
RUN_ID="${RUN_ID:-tre_ss_flow_manifest_4view_$(date +%m%d%H%M)}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR}"
TRAIN_LOG="${TRAIN_LOG:-$RUN_DIR/train.log}"

MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
SS_DECODER_CKPT="${SS_DECODER_CKPT:-}"
SNAPSHOT_ENABLED="${SNAPSHOT_ENABLED:-}"
SNAPSHOT_NUM_SAMPLES="${SNAPSHOT_NUM_SAMPLES:-}"
SNAPSHOT_NUM_STEPS="${SNAPSHOT_NUM_STEPS:-}"
I_SAMPLE="${I_SAMPLE:-}"
I_SAVE="${I_SAVE:-}"
LOAD_DIR="${LOAD_DIR:-}"
RESUME_STEP="${RESUME_STEP:-}"

mkdir -p "$RUN_DIR" "$OUTPUT_DIR"
exec > >(tee -a "$TRAIN_LOG") 2>&1

EXTRA_ARGS=()
if [ -n "$LOAD_DIR" ] || [ -n "$RESUME_STEP" ]; then
    if [ -z "$LOAD_DIR" ] || [ -z "$RESUME_STEP" ]; then
        echo "[ERROR] LOAD_DIR and RESUME_STEP must be set together" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--load-dir "$LOAD_DIR" --resume-step "$RESUME_STEP")
fi

OVERRIDES=(
    "training.output_dir=$OUTPUT_DIR"
    "wandb.enabled=false"
    "wandb.mode=disabled"
)
[ -n "$MAX_STEPS" ]            && OVERRIDES+=("training.max_steps=$MAX_STEPS")
[ -n "$LR" ]                   && OVERRIDES+=("training.optimizer.args.lr=$LR")
[ -n "$PRETRAINED_CKPT" ]      && OVERRIDES+=("training.pretrained_ckpt=$PRETRAINED_CKPT")
[ -n "$SS_DECODER_CKPT" ]      && OVERRIDES+=("snapshot.ss_decoder_ckpt=$SS_DECODER_CKPT")
[ -n "$SNAPSHOT_ENABLED" ]     && OVERRIDES+=("snapshot.enabled=$SNAPSHOT_ENABLED")
[ -n "$SNAPSHOT_NUM_SAMPLES" ] && OVERRIDES+=("snapshot.num_samples=$SNAPSHOT_NUM_SAMPLES")
[ -n "$SNAPSHOT_NUM_STEPS" ]   && OVERRIDES+=("snapshot.num_steps=$SNAPSHOT_NUM_STEPS")
[ -n "$I_SAMPLE" ]             && OVERRIDES+=("training.i_sample=$I_SAMPLE")
[ -n "$I_SAVE" ]               && OVERRIDES+=("training.i_save=$I_SAVE")

OVERRIDES+=("$@")

echo "============================================================"
echo "SS Flow Global-Z Training"
echo "  CONFIG:      $CONFIG"
echo "  NUM_GPUS:    $NUM_GPUS  GPU_IDS='${GPU_IDS:-}'"
echo "  RUN_ID:      $RUN_ID"
echo "  RUN_DIR:     $RUN_DIR"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  TRAIN_LOG:   $TRAIN_LOG"
echo "  WANDB:       disabled"
echo "  OVERRIDES:   ${OVERRIDES[*]}"
echo "============================================================"

launch_ddp "$NUM_GPUS" "$GPU_IDS" "$SCRIPT" \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}" \
    "${OVERRIDES[@]}"
