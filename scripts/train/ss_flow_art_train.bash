#!/usr/bin/env bash
# ============================================================
# SS Flow Art (paper Stage 2) Training Launcher (Phase 09 D-14/D-15/D-16)
#
# Usage:
#   bash scripts/train/ss_flow_art_train.bash
#   MAX_STEPS=5 bash scripts/train/ss_flow_art_train.bash      # smoke test
#
# MODE switches the variant yaml (resolved against CONFIG_DIR):
#   full     -> ss_flow_art/mv_4view.yaml          (no LoRA, no anchor)
#   lora     -> ss_flow_art/mv_4view_lora.yaml     (LoRA on attn)
#   lora_ext -> ss_flow_art/mv_4view_lora_ext.yaml (LoRA + keep_trainable hybrid)
#   l2sp     -> ss_flow_art/mv_4view_l2sp.yaml     (full + L2-SP anchor)
#   smoke    -> ss_flow_art/smoke_test.yaml        (single-GPU / CPU smoke)
#
# Multi-GPU: torchrun via _ddp_common.sh (single-node).
# Multi-node: see _slurm_common.sh.
#
# Legacy resume (D-19 ckpt resume not auto-continuous after rename):
#   LOAD_DIR=output/stage3_mv_4view RESUME_STEP=50000 bash $0
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/train/_ddp_common.sh
source "$HERE/_ddp_common.sh"

# ------------------------------------------------------------
# Quick config (change these)
# ------------------------------------------------------------
MODE="${MODE:-lora_ext}"                      # full | lora | lora_ext | l2sp | smoke
CONFIG_DIR="TRELLIS-arts/configs/arts/ss_flow_art"

GPU_IDS="${GPU_IDS:-}"                        # e.g. "0,1,2,3"; empty = use NUM_GPUS
NUM_GPUS="${NUM_GPUS:-4}"

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-}"
LR="${LR:-}"
MAX_STEPS="${MAX_STEPS:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_NAME="${WANDB_NAME:-}"

LOAD_DIR="${LOAD_DIR:-}"
RESUME_STEP="${RESUME_STEP:-}"
DUMP_PARAM_STATS="${DUMP_PARAM_STATS:-false}"

# ------------------------------------------------------------
# Resolve config from MODE
# ------------------------------------------------------------
case "$MODE" in
    full)     CONFIG="$CONFIG_DIR/mv_4view.yaml" ;;
    lora)     CONFIG="$CONFIG_DIR/mv_4view_lora.yaml" ;;
    lora_ext) CONFIG="$CONFIG_DIR/mv_4view_lora_ext.yaml" ;;
    l2sp)     CONFIG="$CONFIG_DIR/mv_4view_l2sp.yaml" ;;
    smoke)    CONFIG="$CONFIG_DIR/smoke_test.yaml" ;;
    *) echo "[ERROR] Unknown MODE=$MODE (full|lora|lora_ext|l2sp|smoke)"; exit 1 ;;
esac

SCRIPT="TRELLIS-arts/train_arts.py"

if [ -z "$WANDB_NAME" ]; then
    WANDB_NAME="ss-flow-art-${MODE}-${NUM_GPUS}gpu"
fi

# OmegaConf overrides (positional after --config / --load-dir)
OVERRIDES=(
    "wandb.enabled=$WANDB_ENABLED"
    "wandb.name=$WANDB_NAME"
)
[ -n "$BATCH_SIZE_PER_GPU" ] && OVERRIDES+=("training.batch_size_per_gpu=$BATCH_SIZE_PER_GPU")
[ -n "$LR" ]                 && OVERRIDES+=("training.lr=$LR")
[ -n "$MAX_STEPS" ]          && OVERRIDES+=("training.max_steps=$MAX_STEPS")
[ -n "$CHECKPOINT_EVERY" ]   && OVERRIDES+=("training.checkpoint_every=$CHECKPOINT_EVERY")
[ -n "$OUTPUT_DIR" ]         && OVERRIDES+=("training.output_dir=$OUTPUT_DIR")
[ "$DUMP_PARAM_STATS" = "true" ] && OVERRIDES+=("training.dump_param_stats=true")

EXTRA_ARGS=()
if [ -n "$LOAD_DIR" ] && [ -n "$RESUME_STEP" ]; then
    EXTRA_ARGS+=(--load-dir "$LOAD_DIR" --resume-step "$RESUME_STEP")
fi

echo "============================================================"
echo "SS Flow Art Training"
echo "  MODE / CONFIG:  $MODE / $CONFIG"
echo "  NUM_GPUS:       $NUM_GPUS  GPU_IDS='$GPU_IDS'"
[ -n "$MAX_STEPS" ]  && echo "  MAX_STEPS:      $MAX_STEPS (override)"
[ -n "$LOAD_DIR" ]   && echo "  RESUME from:    $LOAD_DIR step $RESUME_STEP"
[ -n "$OUTPUT_DIR" ] && echo "  OUTPUT_DIR:     $OUTPUT_DIR"
echo "============================================================"

launch_ddp "$NUM_GPUS" "$GPU_IDS" "$SCRIPT" \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}" \
    "${OVERRIDES[@]}"
