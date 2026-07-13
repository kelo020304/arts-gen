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
#   OUT_DIR=/robot/data-lab/jzh/art-gen-output/part_promptable_seg/my_run \
#   STEPS=100000 BATCH=16 bash scripts/train/part_promptable_seg/run_train.bash
#
# Important env:
#   MODEL_SIZE=S|M|L
#   PACKED_DIR=/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v6
#   DATA_ROOT=/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v6  # alias for PACKED_DIR
#   SELECTION_JSON=/path/to/selection.json   # optional; smoke auto-generates one
#   OUT_DIR=/path/to/output
#   STEPS=100000
#   BATCH=16
#   PRECISION=bf16
#   NUM_GPUS=8
#   GPU_IDS=0,1,2,3,4,5,6,7
#   RESUME=/path/to/ckpts/step_14000.pt
#   AUTO_RESUME=1
#   WARM_START=/path/to/ckpts/step_50000.pt
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
export PROMPTSEG_SS_ENCODER_CKPT="${PROMPTSEG_SS_ENCODER_CKPT:-/robot/data-lab/jzh/art-gen/third-party-weights/trellis/pretrained/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.safetensors}"
export PROMPTSEG_SS_DECODER_CKPT="${PROMPTSEG_SS_DECODER_CKPT:-/robot/data-lab/jzh/art-gen/third-party-weights/trellis/pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors}"

SMOKE="${SMOKE:-0}"
MODE="${MODE:-gate1}"
MODEL_SIZE="${MODEL_SIZE:-S}"
PACKED_DIR="${PACKED_DIR:-${DATA_ROOT:-${DATA:-/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v6}}}"
SELECTION_JSON="${SELECTION_JSON:-}"
SPLIT_JSON="${SPLIT_JSON:-}"
PROXY_JSON="${PROXY_JSON:-}"
GPU_IDS="${GPU_IDS:-}"
ROUTE="${ROUTE:-voxel}"
MASK_ENCODER="${MASK_ENCODER:-fg_points}"
USE_PACKED_WHOLE_OCC="${USE_PACKED_WHOLE_OCC:-1}"
COMPILE="${COMPILE:-0}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"
PRECISION="${PRECISION:-}"
SEG_DISCRIMINATIVE="${SEG_DISCRIMINATIVE:-${JOINT_SEG:-0}}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-0}"
WARM_START="${WARM_START:-}"
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
    L)
        DIM="${DIM:-512}"
        DEPTH="${DEPTH:-12}"
        HEADS="${HEADS:-8}"
        ;;
    *)
        echo "[run_train] MODEL_SIZE must be S, M, or L, got: $MODEL_SIZE" >&2
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
    OUT_DIR="${OUT_DIR:-/robot/data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
    if [ -z "$SELECTION_JSON" ]; then
        SELECTION_JSON="/robot/data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_selection.json"
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
    OUT_DIR="${OUT_DIR:-/robot/data-lab/jzh/art-gen-output/part_promptable_seg/${RUN_NAME}}"
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
    --mode "$MODE"
    --out-dir "$OUT_DIR"
    --steps "$STEPS"
    --batch-size "$BATCH"
    --eval-batch-size "$EVAL_BATCH"
    --num-workers "${NUM_WORKERS:-8}"
    --prefetch-factor "${PREFETCH_FACTOR:-4}"
    --lr "$LR"
    --warmup-steps "$WARMUP_STEPS"
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --grad-clip "${GRAD_CLIP:-1.0}"
    --dim "$DIM"
    --depth "$DEPTH"
    --head-depth "${HEAD_DEPTH:-2}"
    --heads "$HEADS"
    --voxel-depth "${VOXEL_DEPTH:-3}"
    --route "$ROUTE"
    --mask-encoder "$MASK_ENCODER"
    --point-resample-points
    --voxel-embedding-dim "${VOXEL_EMBEDDING_DIM:-0}"
    --voxel-max-tokens "${VOXEL_MAX_TOKENS:-0}"
    --decode-dice-weight "${DECODE_DICE_WEIGHT:-0.5}"
    --latent-part-weight "${LATENT_PART_WEIGHT:-8.0}"
    --latent-loss-mode "${LATENT_LOSS_MODE:-weighted}"
    --voxel-loss-weight "${VOXEL_LOSS_WEIGHT:-1.0}"
    --spconv-depth "${SPCONV_DEPTH:-4}"
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
[ -n "$SPLIT_JSON" ] && ARGS+=(--split-json "$SPLIT_JSON")
[ -n "$PROXY_JSON" ] && ARGS+=(--proxy-json "$PROXY_JSON")
[ "$USE_PACKED_WHOLE_OCC" = "1" ] && ARGS+=(--use-packed-whole-occ)
[ "$SEG_DISCRIMINATIVE" = "1" ] && ARGS+=(--joint-seg)
[ -n "${BODY_CLASS_WEIGHT:-}" ] && ARGS+=(--body-class-weight "$BODY_CLASS_WEIGHT")
[ -n "${JOINT_KMAX:-}" ] && ARGS+=(--joint-kmax "$JOINT_KMAX")
[ -n "${JOINT_SMALL_PART_THRESHOLD:-}" ] && ARGS+=(--joint-small-part-threshold "$JOINT_SMALL_PART_THRESHOLD")
[ -n "${JOINT_SMALL_PART_WEIGHT:-}" ] && ARGS+=(--joint-small-part-weight "$JOINT_SMALL_PART_WEIGHT")
[ -n "${JOINT_SMOOTH_WEIGHT:-}" ] && ARGS+=(--joint-smooth-weight "$JOINT_SMOOTH_WEIGHT")
[ -n "${JOINT_SMOOTH_SAME_LABEL_WEIGHT:-}" ] && ARGS+=(--joint-smooth-same-label-weight "$JOINT_SMOOTH_SAME_LABEL_WEIGHT")
[ -n "${JOINT_SMOOTH_ALL_LABEL_WEIGHT:-}" ] && ARGS+=(--joint-smooth-all-label-weight "$JOINT_SMOOTH_ALL_LABEL_WEIGHT")
[ -n "${JOINT_SMOOTH_CROSS_LABEL_WEIGHT:-}" ] && ARGS+=(--joint-smooth-cross-label-weight "$JOINT_SMOOTH_CROSS_LABEL_WEIGHT")
[ -n "${JOINT_SMOOTH_NEIGHBORHOOD:-}" ] && ARGS+=(--joint-smooth-neighborhood "$JOINT_SMOOTH_NEIGHBORHOOD")
[ "${JOINT_CRF_EVAL:-0}" = "1" ] && ARGS+=(--joint-crf-eval)
[ -n "${JOINT_CRF_ITERS:-}" ] && ARGS+=(--joint-crf-iters "$JOINT_CRF_ITERS")
[ -n "${JOINT_CRF_PAIRWISE:-}" ] && ARGS+=(--joint-crf-pairwise "$JOINT_CRF_PAIRWISE")
[ -n "${JOINT_CRF_NEIGHBORHOOD:-}" ] && ARGS+=(--joint-crf-neighborhood "$JOINT_CRF_NEIGHBORHOOD")
[ "${PERSISTENT_WORKERS:-1}" = "1" ] && ARGS+=(--persistent-workers) || ARGS+=(--no-persistent-workers)
[ "${PIN_MEMORY:-1}" = "1" ] && ARGS+=(--pin-memory) || ARGS+=(--no-pin-memory)
[ -n "${GROUP_COST_BUDGET:-}" ] && ARGS+=(--group-cost-budget "$GROUP_COST_BUDGET")
[ "${USE_CHECKPOINT:-0}" = "1" ] && ARGS+=(--use-checkpoint) || ARGS+=(--no-use-checkpoint)
[ -n "$PRECISION" ] && ARGS+=(--precision "$PRECISION")
[ "${TF32:-}" = "1" ] && ARGS+=(--tf32)
[ "${TF32:-}" = "0" ] && ARGS+=(--no-tf32)
[ "$FP16" = "1" ] || ARGS+=(--no-fp16)
[ "$COMPILE" = "1" ] && ARGS+=(--compile) || ARGS+=(--no-compile)
[ "$FINAL_FULL_EVAL" = "1" ] && ARGS+=(--final-full-eval) || ARGS+=(--no-final-full-eval)
[ "$FILTER_UNDETECTABLE" = "1" ] && ARGS+=(--filter-undetectable) || ARGS+=(--no-filter-undetectable)
[ -n "$RESUME" ] && ARGS+=(--resume "$RESUME")
[ "$AUTO_RESUME" = "1" ] && ARGS+=(--auto-resume)
[ -n "$WARM_START" ] && ARGS+=(--warm-start "$WARM_START")
[ -n "${BOUNDARY_BAND_RADIUS:-}" ] && ARGS+=(--boundary-band-radius "$BOUNDARY_BAND_RADIUS")
[ "${BOUNDARY_HARD_MINING:-0}" = "1" ] && ARGS+=(--boundary-hard-mining)
[ -n "${BOUNDARY_HARD_MINING_TOPK:-}" ] && ARGS+=(--boundary-hard-mining-topk "$BOUNDARY_HARD_MINING_TOPK")
[ -n "${BOUNDARY_HARD_MINING_WEIGHT:-}" ] && ARGS+=(--boundary-hard-mining-weight "$BOUNDARY_HARD_MINING_WEIGHT")
[ "${NEGATIVE_PROMPT_CHANNEL:-0}" = "1" ] && ARGS+=(--negative-prompt-channel)
[ "${NEGATIVE_PROMPT_EQUIV_CHECK:-1}" = "0" ] && ARGS+=(--no-negative-prompt-equivalence-check)
[ "${VOXEL_CORRUPT:-0}" = "1" ] && ARGS+=(--voxel-corrupt)
[ -n "${VOXEL_CORRUPT_DROP_PROB:-}" ] && ARGS+=(--voxel-corrupt-drop-prob "$VOXEL_CORRUPT_DROP_PROB")
[ -n "${VOXEL_CORRUPT_SHELL_PROB:-}" ] && ARGS+=(--voxel-corrupt-shell-prob "$VOXEL_CORRUPT_SHELL_PROB")
[ -n "${VOXEL_CORRUPT_SPECKLE_PROB:-}" ] && ARGS+=(--voxel-corrupt-speckle-prob "$VOXEL_CORRUPT_SPECKLE_PROB")
[ -n "${VOXEL_CORRUPT_VISUALIZE_DIR:-}" ] && ARGS+=(--voxel-corrupt-visualize-dir "$VOXEL_CORRUPT_VISUALIZE_DIR")
[ -n "${VOXEL_CORRUPT_VISUALIZE_COUNT:-}" ] && ARGS+=(--voxel-corrupt-visualize-count "$VOXEL_CORRUPT_VISUALIZE_COUNT")
[ "${VOXEL_CORRUPT_VISUALIZE_ONLY:-0}" = "1" ] && ARGS+=(--voxel-corrupt-visualize-only)
[ "${SEMANTIC_AUX:-0}" = "1" ] && ARGS+=(--semantic-aux)
[ -n "${SEMANTIC_LOSS_WEIGHT:-}" ] && ARGS+=(--semantic-loss-weight "$SEMANTIC_LOSS_WEIGHT")
[ "${MASK_AUGMENT:-0}" = "1" ] && ARGS+=(--mask-augment)
[ "${VIEW_DROPOUT:-0}" = "1" ] && ARGS+=(--view-dropout)
[ -n "${MIN_VIEWS:-}" ] && ARGS+=(--min-views "$MIN_VIEWS")
[ -n "${MIN_PROMPT_VIEWS:-}" ] && ARGS+=(--min-prompt-views "$MIN_PROMPT_VIEWS")
[ -n "${VIEW_DROPOUT_START_STEP:-}" ] && ARGS+=(--view-dropout-start-step "$VIEW_DROPOUT_START_STEP")
[ -n "${MASK_TARGET:-}" ] && ARGS+=(--mask-target "$MASK_TARGET")
[ -n "${SUPPORT_MULTIPLIER:-}" ] && ARGS+=(--support-multiplier "$SUPPORT_MULTIPLIER")
[ -n "${SMALL_OVERSAMPLE:-}" ] && ARGS+=(--small-oversample "$SMALL_OVERSAMPLE")
[ -n "${REALAPPLIANCE_OVERSAMPLE:-}" ] && ARGS+=(--realappliance-oversample "$REALAPPLIANCE_OVERSAMPLE")
[ -n "${REALAPPLIANCE_TARGET_SHARE:-}" ] && ARGS+=(--realappliance-target-share "$REALAPPLIANCE_TARGET_SHARE")
[ -n "${REALAPPLIANCE_MAX_OVERSAMPLE:-}" ] && ARGS+=(--realappliance-max-oversample "$REALAPPLIANCE_MAX_OVERSAMPLE")
[ -n "${VERSE_FOCUS_OVERSAMPLE:-}" ] && ARGS+=(--verse-focus-oversample "$VERSE_FOCUS_OVERSAMPLE")
[ -n "${FOCAL_GAMMA:-}" ] && ARGS+=(--focal-gamma "$FOCAL_GAMMA")
[ -n "${BOUNDARY_WEIGHT:-}" ] && ARGS+=(--boundary-weight "$BOUNDARY_WEIGHT")
[ -n "${DECODE_EVAL_STEPS:-}" ] && ARGS+=(--decode-eval-steps "$DECODE_EVAL_STEPS")

if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)
    ARGS+=("${EXTRA_ARR[@]}")
fi

echo "[run_train] repo=$REPO_ROOT"
echo "[run_train] mode=$MODE model_size=$MODEL_SIZE dim=$DIM depth=$DEPTH heads=$HEADS precision=${PRECISION:-legacy-fp16-flag}"
echo "[run_train] num_gpus=$NUM_GPUS gpu_ids='${GPU_IDS:-}' smoke=$SMOKE"
echo "[run_train] packed_dir=$PACKED_DIR"
echo "[run_train] ss_encoder=$PROMPTSEG_SS_ENCODER_CKPT"
echo "[run_train] ss_decoder=$PROMPTSEG_SS_DECODER_CKPT"
echo "[run_train] dataloader num_workers=${NUM_WORKERS:-8} prefetch_factor=${PREFETCH_FACTOR:-4} persistent_workers=${PERSISTENT_WORKERS:-1} pin_memory=${PIN_MEMORY:-1}"
echo "[run_train] seg_discriminative=$SEG_DISCRIMINATIVE group_cost_budget=${GROUP_COST_BUDGET:-0} joint_kmax=${JOINT_KMAX:-default} joint_small_part_threshold=${JOINT_SMALL_PART_THRESHOLD:-32} joint_small_part_weight=${JOINT_SMALL_PART_WEIGHT:-1.5}"
echo "[run_train] joint_smooth weight=${JOINT_SMOOTH_WEIGHT:-0} same=${JOINT_SMOOTH_SAME_LABEL_WEIGHT:-default} all=${JOINT_SMOOTH_ALL_LABEL_WEIGHT:-default} cross=${JOINT_SMOOTH_CROSS_LABEL_WEIGHT:-0} n=${JOINT_SMOOTH_NEIGHBORHOOD:-default}"
echo "[run_train] joint_crf_eval=${JOINT_CRF_EVAL:-0} iters=${JOINT_CRF_ITERS:-default} pairwise=${JOINT_CRF_PAIRWISE:-default} n=${JOINT_CRF_NEIGHBORHOOD:-default}"
echo "[run_train] selection_json=${SELECTION_JSON:-<none>} split_json=${SPLIT_JSON:-<none>} proxy_json=${PROXY_JSON:-<none>}"
echo "[run_train] resume=${RESUME:-<none>} auto_resume=$AUTO_RESUME warm_start=${WARM_START:-<none>}"
echo "[run_train] t1_flags boundary_band_radius=${BOUNDARY_BAND_RADIUS:-<default>} boundary_hard_mining=${BOUNDARY_HARD_MINING:-0} negative_prompt_channel=${NEGATIVE_PROMPT_CHANNEL:-0} voxel_corrupt=${VOXEL_CORRUPT:-0}"
echo "[run_train] old0616_flags semantic_aux=${SEMANTIC_AUX:-0} mask_augment=${MASK_AUGMENT:-0} view_dropout=${VIEW_DROPOUT:-0} mask_target=${MASK_TARGET:-<default>} realappliance_target_share=${REALAPPLIANCE_TARGET_SHARE:-<default>} boundary_weight=${BOUNDARY_WEIGHT:-<default>}"
echo "[run_train] out_dir=$OUT_DIR"

for ckpt in "$PROMPTSEG_SS_ENCODER_CKPT" "$PROMPTSEG_SS_DECODER_CKPT"; do
    cfg="${ckpt%.safetensors}.json"
    if [ ! -f "$ckpt" ] || [ ! -f "$cfg" ]; then
        echo "[run_train] missing SS VAE file: ckpt=$ckpt cfg=$cfg" >&2
        exit 2
    fi
done

cd "$REPO_ROOT"
export WANDB_DISABLED
launch_ddp "$NUM_GPUS" "$GPU_IDS" "$REPO_ROOT/scripts/train/part_promptable_seg/train_part_promptable_seg.py" "${ARGS[@]}"
