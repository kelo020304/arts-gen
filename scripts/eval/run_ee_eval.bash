#!/usr/bin/env bash
# Standard launcher for the accepted 0617 end-to-end eval path.
#
# Usage:
#   bash scripts/eval/run_ee_eval.bash
#
# One-object smoke:
#   SMOKE=1 GPUS=0 bash scripts/eval/run_ee_eval.bash
#
# 1024-object run:
#   OUT_DIR=/mnt/robot-data-lab/jzh/art-gen/ee-eval/0626-1024-1 \
#   LIMIT=1024 TRAIN_COUNT=1024 HELD_COUNT=0 GPUS=0,1,2,3 \
#   bash scripts/eval/run_ee_eval.bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

ARTS_GEN_ENV_DIR="${ARTS_GEN_ENV_DIR:-/opt/venvs/arts-gen}"
PYTHON="${PYTHON:-$ARTS_GEN_ENV_DIR/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "[run_ee_eval] python not found: $PYTHON" >&2
    exit 2
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/TRELLIS-arts:${PYTHONPATH:-}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export SS_FLOW_FUSION_MODE="${SS_FLOW_FUSION_MODE:-concat}"

SMOKE="${SMOKE:-0}"
SLAT_TOKEN_SOURCE="${SLAT_TOKEN_SOURCE:-live}"
SELECTION_MODE="${SELECTION_MODE:-samples}"
SAMPLE_SELECTION_UNIT="${SAMPLE_SELECTION_UNIT:-objects}"
ALLOWED_DATASETS="${ALLOWED_DATASETS:-phyx-verse,realappliance}"
FORCE="${FORCE:-1}"
OVERWRITE_SELECTION="${OVERWRITE_SELECTION:-1}"
PART_SEG_CKPT="${PART_SEG_CKPT:-}"
PART_JOINT_CANDIDATE_MODE="${PART_JOINT_CANDIDATE_MODE:-proposal}"
PART_JOINT_REFINE="${PART_JOINT_REFINE:-0}"
PART_JOINT_REFINE_ITERS="${PART_JOINT_REFINE_ITERS:-1}"
PART_JOINT_REFINE_PAIRWISE="${PART_JOINT_REFINE_PAIRWISE:-3.0}"
PART_JOINT_REFINE_MARGIN="${PART_JOINT_REFINE_MARGIN:-0.0}"
PART_JOINT_REFINE_MARGIN_QUANTILE="${PART_JOINT_REFINE_MARGIN_QUANTILE:-0.01}"
PART_JOINT_REFINE_NEIGHBORHOOD="${PART_JOINT_REFINE_NEIGHBORHOOD:-6}"
PART_JOINT_REFINE_MIN_VOTE_GAIN="${PART_JOINT_REFINE_MIN_VOTE_GAIN:-0.0}"
PART_JOINT_REFINE_PRESERVE_SMALL_CLASSES="${PART_JOINT_REFINE_PRESERVE_SMALL_CLASSES:-32}"
PART_JOINT_SAVE_LOGITS="${PART_JOINT_SAVE_LOGITS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ "$SMOKE" = "1" ]; then
    LIMIT="${LIMIT:-1}"
    TRAIN_COUNT="${TRAIN_COUNT:-1}"
    HELD_COUNT="${HELD_COUNT:-0}"
    GPUS="${GPUS:-0}"
    OUT_DIR="${OUT_DIR:-/mnt/robot-data-lab/jzh/art-gen/ee-eval/run-ee-eval-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
else
    LIMIT="${LIMIT:-128}"
    TRAIN_COUNT="${TRAIN_COUNT:-85}"
    HELD_COUNT="${HELD_COUNT:-43}"
    GPUS="${GPUS:-0,1,2,3}"
    OUT_DIR="${OUT_DIR:-/mnt/robot-data-lab/jzh/art-gen/ee-eval/ee_0617_$(date -u +%Y%m%dT%H%M%SZ)}"
fi

ARGS=(
    ee_0617
    --out-dir "$OUT_DIR"
    --limit "$LIMIT"
    --train-count "$TRAIN_COUNT"
    --held-count "$HELD_COUNT"
    --gpus "$GPUS"
    --allowed-datasets "$ALLOWED_DATASETS"
    --selection-mode "$SELECTION_MODE"
    --sample-selection-unit "$SAMPLE_SELECTION_UNIT"
    --slat-token-source "$SLAT_TOKEN_SOURCE"
    --part-joint-candidate-mode "$PART_JOINT_CANDIDATE_MODE"
    --part-joint-refine-iters "$PART_JOINT_REFINE_ITERS"
    --part-joint-refine-pairwise "$PART_JOINT_REFINE_PAIRWISE"
    --part-joint-refine-margin "$PART_JOINT_REFINE_MARGIN"
    --part-joint-refine-margin-quantile "$PART_JOINT_REFINE_MARGIN_QUANTILE"
    --part-joint-refine-neighborhood "$PART_JOINT_REFINE_NEIGHBORHOOD"
    --part-joint-refine-min-vote-gain "$PART_JOINT_REFINE_MIN_VOTE_GAIN"
    --part-joint-refine-preserve-small-classes "$PART_JOINT_REFINE_PRESERVE_SMALL_CLASSES"
)

[ -n "$PART_SEG_CKPT" ] && ARGS+=(--part-seg-ckpt "$PART_SEG_CKPT")
[ "$PART_JOINT_REFINE" = "1" ] && ARGS+=(--part-joint-refine) || ARGS+=(--no-part-joint-refine)
[ "$PART_JOINT_SAVE_LOGITS" = "1" ] && ARGS+=(--part-joint-save-logits) || ARGS+=(--no-part-joint-save-logits)
[ "$FORCE" = "1" ] && ARGS+=(--force)
[ "$OVERWRITE_SELECTION" = "1" ] && ARGS+=(--overwrite-selection)

if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)
    ARGS+=("${EXTRA_ARR[@]}")
fi

echo "[run_ee_eval] repo=$REPO_ROOT"
echo "[run_ee_eval] out_dir=$OUT_DIR"
echo "[run_ee_eval] limit=$LIMIT train=$TRAIN_COUNT held=$HELD_COUNT gpus=$GPUS token_source=$SLAT_TOKEN_SOURCE"
echo "[run_ee_eval] part_seg_ckpt=${PART_SEG_CKPT:-<default>} joint_candidate=$PART_JOINT_CANDIDATE_MODE joint_refine=$PART_JOINT_REFINE save_logits=$PART_JOINT_SAVE_LOGITS"

cd "$REPO_ROOT"
"$PYTHON" "$REPO_ROOT/scripts/eval/run_eval.py" "${ARGS[@]}"
