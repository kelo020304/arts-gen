#!/usr/bin/env bash
# Standard launcher for PromptablePartLatentSegNet training.
#
# Usage:
#   bash scripts/train/part_promptable_seg/run_train.bash
#
# Single-GPU smoke, writes to a temporary OUT_DIR and runs 5 steps:
#   SMOKE=1 GPU_IDS=0 bash scripts/train/part_promptable_seg/run_train.bash
#
# 8-GPU training:
#   MODEL_SIZE=S NUM_GPUS=8 GPU_IDS=0,1,2,3,4,5,6,7 \
#   OUT_DIR=/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/my_run \
#   STEPS=100000 BATCH=16 bash scripts/train/part_promptable_seg/run_train.bash
#
# Important env:
#   MODEL_SIZE=S|M
#   PACKED_DIR=/mnt/robot-data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5
#   SELECTION_JSON=/path/to/selection.json   # optional; smoke auto-generates one
#   OUT_DIR=/path/to/output
#   STEPS=100000
#   BATCH=16
#   NUM_GPUS=8
#   GPU_IDS=0,1,2,3,4,5,6,7
#   EXTRA_ARGS="--warm-start /path/to/ckpt.pt"
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
# shellcheck source=scripts/train/_ddp_common.sh
source "$REPO_ROOT/scripts/train/_ddp_common.sh"

ARTS_GEN_ENV_DIR="${ARTS_GEN_ENV_DIR:-/opt/venvs/arts-gen}"
ARTS_GEN_PYTHON="${ARTS_GEN_PYTHON:-$ARTS_GEN_ENV_DIR/bin/python}"
if [ -x "$ARTS_GEN_PYTHON" ]; then
    export PATH="$(dirname "$ARTS_GEN_PYTHON"):$PATH"
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/TRELLIS-arts:${PYTHONPATH:-}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"

SMOKE="${SMOKE:-0}"
MODEL_SIZE="${MODEL_SIZE:-S}"
PACKED_DIR="${PACKED_DIR:-/mnt/robot-data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5}"
SELECTION_JSON="${SELECTION_JSON:-}"
GPU_IDS="${GPU_IDS:-}"
ROUTE="${ROUTE:-voxel}"
MASK_ENCODER="${MASK_ENCODER:-fg_points}"
USE_PACKED_WHOLE_OCC="${USE_PACKED_WHOLE_OCC:-1}"
COMPILE="${COMPILE:-0}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$MODEL_SIZE" in
    S)
        DIM="${DIM:-256}"
        DEPTH="${DEPTH:-6}"
        HEADS="${HEADS:-8}"
        ;;
    M)
        DIM="${DIM:-384}"
        DEPTH="${DEPTH:-8}"
        HEADS="${HEADS:-8}"
        ;;
    *)
        echo "[run_train] MODEL_SIZE must be S or M, got: $MODEL_SIZE" >&2
        exit 2
        ;;
esac

if [ "$SMOKE" = "1" ]; then
    NUM_GPUS="${NUM_GPUS:-1}"
    GPU_IDS="${GPU_IDS:-0}"
    STEPS="${STEPS:-5}"
    BATCH="${BATCH:-1}"
    EVAL_BATCH="${EVAL_BATCH:-1}"
    LR="${LR:-1e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-1}"
    CKPT_EVERY="${CKPT_EVERY:-5}"
    EVAL_EVERY="${EVAL_EVERY:-999}"
    LOG_EVERY="${LOG_EVERY:-1}"
    FP16="${FP16:-0}"
    FINAL_FULL_EVAL="${FINAL_FULL_EVAL:-0}"
    FILTER_UNDETECTABLE="${FILTER_UNDETECTABLE:-0}"
    OUT_DIR="${OUT_DIR:-/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
    if [ -z "$SELECTION_JSON" ]; then
        SELECTION_JSON="/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_selection.json"
        "$ARTS_GEN_PYTHON" - "$PACKED_DIR" "$SELECTION_JSON" <<'PY'
import json
import sys
from pathlib import Path

packed_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
index = json.loads((packed_dir / "index.json").read_text(encoding="utf-8"))
selection = [
    {
        "dataset_id": entry.get("dataset_id", ""),
        "obj_id": entry["obj_id"],
        "angle_idx": int(entry["angle_idx"]),
        "part_name": entry["part_name"],
    }
    for entry in index["entries"][:8]
]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out)
PY
    fi
else
    RUN_NAME="${RUN_NAME:-part_promptable_seg_${MODEL_SIZE}_$(date -u +%Y%m%dT%H%M%SZ)}"
    OUT_DIR="${OUT_DIR:-/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/${RUN_NAME}}"
    NUM_GPUS="${NUM_GPUS:-8}"
    STEPS="${STEPS:-100000}"
    BATCH="${BATCH:-16}"
    EVAL_BATCH="${EVAL_BATCH:-$BATCH}"
    LR="${LR:-1e-4}"
    WARMUP_STEPS="${WARMUP_STEPS:-200}"
    CKPT_EVERY="${CKPT_EVERY:-2000}"
    EVAL_EVERY="${EVAL_EVERY:-1000}"
    LOG_EVERY="${LOG_EVERY:-50}"
    FP16="${FP16:-1}"
    FINAL_FULL_EVAL="${FINAL_FULL_EVAL:-1}"
    FILTER_UNDETECTABLE="${FILTER_UNDETECTABLE:-1}"
fi

ARGS=(
    --mode gate1
    --out-dir "$OUT_DIR"
    --steps "$STEPS"
    --batch-size "$BATCH"
    --eval-batch-size "$EVAL_BATCH"
    --num-workers "${NUM_WORKERS:-0}"
    --lr "$LR"
    --warmup-steps "$WARMUP_STEPS"
    --dim "$DIM"
    --depth "$DEPTH"
    --heads "$HEADS"
    --route "$ROUTE"
    --mask-encoder "$MASK_ENCODER"
    --point-resample-points
    --voxel-embedding-dim "${VOXEL_EMBEDDING_DIM:-0}"
    --packed-dir "$PACKED_DIR"
    --no-auto-pack
    --log-every "$LOG_EVERY"
    --eval-every "$EVAL_EVERY"
    --ckpt-every "$CKPT_EVERY"
    --eval-max-rows "${EVAL_MAX_ROWS:-1}"
    --train-eval-max-rows "${TRAIN_EVAL_MAX_ROWS:-1}"
    --heldout-eval-max-rows "${HELDOUT_EVAL_MAX_ROWS:-1}"
    --full-eval-every "${FULL_EVAL_EVERY:-0}"
)

[ -n "$SELECTION_JSON" ] && ARGS+=(--selection-json "$SELECTION_JSON")
[ "$USE_PACKED_WHOLE_OCC" = "1" ] && ARGS+=(--use-packed-whole-occ)
[ "$FP16" = "1" ] || ARGS+=(--no-fp16)
[ "$COMPILE" = "1" ] && ARGS+=(--compile) || ARGS+=(--no-compile)
[ "$FINAL_FULL_EVAL" = "1" ] && ARGS+=(--final-full-eval) || ARGS+=(--no-final-full-eval)
[ "$FILTER_UNDETECTABLE" = "1" ] && ARGS+=(--filter-undetectable) || ARGS+=(--no-filter-undetectable)

if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)
    ARGS+=("${EXTRA_ARR[@]}")
fi

echo "[run_train] repo=$REPO_ROOT"
echo "[run_train] model_size=$MODEL_SIZE dim=$DIM depth=$DEPTH heads=$HEADS"
echo "[run_train] num_gpus=$NUM_GPUS gpu_ids='${GPU_IDS:-}' smoke=$SMOKE"
echo "[run_train] packed_dir=$PACKED_DIR"
echo "[run_train] selection_json=${SELECTION_JSON:-<none>}"
echo "[run_train] out_dir=$OUT_DIR"

cd "$REPO_ROOT"
export WANDB_DISABLED
launch_ddp "$NUM_GPUS" "$GPU_IDS" "$REPO_ROOT/scripts/train/part_promptable_seg/train_part_promptable_seg.py" "${ARGS[@]}"
