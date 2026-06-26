#!/usr/bin/env bash
# ============================================================
# Part Flow Training Launcher (manifest-driven, single canonical yaml).
#
# Quick start:
#   # Default = smoke 20-sample overfit on data/smoke_test/1 (manifest baked in):
#   GPU_IDS="0,1" bash scripts/train/part_flow_train.bash
#
#   # MAX_STEPS forward-only smoke:
#   GPU_IDS="0" MAX_STEPS=5 bash scripts/train/part_flow_train.bash
#
#   # Full training: override data env vars + max_steps. Model arch stays the
#   # same as smoke (same hidden_dim etc.) for consistency:
#   GPU_IDS="0,1,2,3" \
#       DATA_ROOT=data/PhysX-Mobility \
#       RECON_SUBDIR=arts/reconstruction \
#       MASK_SUBDIR=arts/renders \
#       MANIFEST_PATH=arts/manifests/part_completion/full.train.jsonl \
#       MAX_STEPS=100000 \
#       bash scripts/train/part_flow_train.bash
#
# Config:
#   CONFIG (default: TRELLIS-arts/configs/arts/part_flow/part_flow.yaml)
#     Single canonical Part Flow yaml. Override with CONFIG=<path> if you ever
#     need a different yaml file.
#
# GPU env:
#   GPU_IDS               comma list, pins CUDA_VISIBLE_DEVICES + auto NUM_GPUS
#   NUM_GPUS              fallback when GPU_IDS not set (default 4)
#   MASTER_PORT           torchrun rendezvous port (default 29500)
#
# Data env overrides (all OmegaConf-style, applied as CLI overrides):
#   MANIFEST_PATH        -> data.manifest_path
#   DATA_ROOT            -> data.data_root
#   RECON_SUBDIR         -> data.recon_subdir
#   MASK_SUBDIR          -> data.mask_subdir
#   ALLOW_MISSING_MASKS  -> data.allow_missing_masks
#
# Training env overrides:
#   BATCH_SIZE_PER_GPU, LR, MAX_STEPS, CHECKPOINT_EVERY, OUTPUT_DIR
#   FLOW_TYPE, K_MAX, FLOW_T_MAX, SOLVER
#   WANDB_ENABLED (default false), WANDB_NAME
#
# Any extra CLI args after the env vars are passed straight through to
# train_arts.py (after OVERRIDES, so they take precedence). Example:
#   bash $0 data.num_samples=4 training.eval_every=10 model.hidden_dim=256
#
# Multi-GPU: torchrun via _ddp_common.sh (single-node).
# Multi-node: see _slurm_common.sh.
#
# Resume:
#   LOAD_DIR=runs/partflow_xxx RESUME_STEP=20000 bash $0
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/train/_ddp_common.sh
source "$HERE/_ddp_common.sh"

# ------------------------------------------------------------
# Quick config (change these)
# ------------------------------------------------------------
# Single canonical Part Flow yaml. Override via CONFIG=<path> if you want to
# point at a different yaml; otherwise model architecture stays uniform across
# smoke / full runs and only data-side env vars (DATA_ROOT, MANIFEST_PATH, etc.)
# distinguish smoke from full training.
CONFIG="${CONFIG:-TRELLIS-arts/configs/arts/part_flow/part_flow.yaml}"

GPU_IDS="${GPU_IDS:-}"                        # e.g. "0,1,2,3"; empty = use NUM_GPUS
NUM_GPUS="${NUM_GPUS:-4}"

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-}"
LR="${LR:-}"
MAX_STEPS="${MAX_STEPS:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# wandb is OFF by default. To enable, explicitly set WANDB_ENABLED=true.
# part_flow.yaml also sets wandb.enabled: false so the OmegaConf override
# here is doubly safe.
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_NAME="${WANDB_NAME:-}"

LOAD_DIR="${LOAD_DIR:-}"
RESUME_STEP="${RESUME_STEP:-}"

# part_flow specific (flow family / k_max etc.; leave empty to use YAML defaults)
FLOW_TYPE="${FLOW_TYPE:-}"
K_MAX="${K_MAX:-}"
FLOW_T_MAX="${FLOW_T_MAX:-}"
SOLVER="${SOLVER:-}"

# Data overrides (manifest-driven dataset, 2026-05-07). part_flow.yaml ships with
# the smoke 20-sample manifest baked in; override these for full-set training.
MANIFEST_PATH="${MANIFEST_PATH:-}"
DATA_ROOT="${DATA_ROOT:-}"
RECON_SUBDIR="${RECON_SUBDIR:-}"
MASK_SUBDIR="${MASK_SUBDIR:-}"
ALLOW_MISSING_MASKS="${ALLOW_MISSING_MASKS:-}"

# ------------------------------------------------------------
# Reconcile NUM_GPUS with GPU_IDS BEFORE the banner / wandb-name logic.
# launch_ddp also does this internally, but we need NUM_GPUS to reflect
# the real process count when printing diagnostics. Without this the
# banner prints the default NUM_GPUS=4 even when GPU_IDS="0" actually
# launches a single process.
# ------------------------------------------------------------
if [ -n "$GPU_IDS" ]; then
    IFS=',' read -r -a _GPU_ARR <<< "$GPU_IDS"
    NUM_GPUS="${#_GPU_ARR[@]}"
fi

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] CONFIG not found: $CONFIG" >&2
    exit 1
fi

SCRIPT="TRELLIS-arts/train_arts.py"

if [ -z "$WANDB_NAME" ]; then
    WANDB_NAME="part-flow-${NUM_GPUS}gpu"
fi

# OmegaConf overrides (positional after --config / --load-dir)
OVERRIDES=(
    "wandb.enabled=$WANDB_ENABLED"
    "wandb.name=$WANDB_NAME"
)
# Part Flow trainer reads training.batch_size (per-GPU; DDP scales globally).
# Earlier revision wrote training.batch_size_per_gpu which the trainer ignored.
[ -n "$BATCH_SIZE_PER_GPU" ] && OVERRIDES+=("training.batch_size=$BATCH_SIZE_PER_GPU")
[ -n "$LR" ]                 && OVERRIDES+=("training.lr=$LR")
[ -n "$MAX_STEPS" ]          && OVERRIDES+=("training.max_steps=$MAX_STEPS")
[ -n "$CHECKPOINT_EVERY" ]   && OVERRIDES+=("training.checkpoint_every=$CHECKPOINT_EVERY")
[ -n "$OUTPUT_DIR" ]         && OVERRIDES+=("training.output_dir=$OUTPUT_DIR")
[ -n "$FLOW_TYPE" ]          && OVERRIDES+=("flow.type=$FLOW_TYPE")
[ -n "$K_MAX" ]              && OVERRIDES+=("flow.k_max=$K_MAX")
[ -n "$FLOW_T_MAX" ]         && OVERRIDES+=("flow.t_max=$FLOW_T_MAX")
[ -n "$SOLVER" ]             && OVERRIDES+=("flow.solver=$SOLVER")
[ -n "$MANIFEST_PATH" ]      && OVERRIDES+=("data.manifest_path=$MANIFEST_PATH")
[ -n "$DATA_ROOT" ]          && OVERRIDES+=("data.data_root=$DATA_ROOT")
[ -n "$RECON_SUBDIR" ]       && OVERRIDES+=("data.recon_subdir=$RECON_SUBDIR")
[ -n "$MASK_SUBDIR" ]        && OVERRIDES+=("data.mask_subdir=$MASK_SUBDIR")
[ -n "$ALLOW_MISSING_MASKS" ] && OVERRIDES+=("data.allow_missing_masks=$ALLOW_MISSING_MASKS")

EXTRA_ARGS=()
if [ -n "$LOAD_DIR" ] && [ -n "$RESUME_STEP" ]; then
    EXTRA_ARGS+=(--load-dir "$LOAD_DIR" --resume-step "$RESUME_STEP")
fi

echo "============================================================"
echo "Part Flow Training"
echo "  CONFIG:           $CONFIG"
echo "  NUM_GPUS:         $NUM_GPUS  GPU_IDS='$GPU_IDS'"
[ -n "$MAX_STEPS" ]      && echo "  MAX_STEPS:        $MAX_STEPS (override)"
[ -n "$LOAD_DIR" ]       && echo "  RESUME from:      $LOAD_DIR step $RESUME_STEP"
[ -n "$OUTPUT_DIR" ]     && echo "  OUTPUT_DIR:       $OUTPUT_DIR"
[ -n "$FLOW_TYPE" ]      && echo "  FLOW_TYPE:        $FLOW_TYPE"
[ -n "$MANIFEST_PATH" ]  && echo "  MANIFEST_PATH:    $MANIFEST_PATH"
[ -n "$DATA_ROOT" ]      && echo "  DATA_ROOT:        $DATA_ROOT"
[ "$#" -gt 0 ]           && echo "  EXTRA OVERRIDES:  $*"
echo "============================================================"

# Pass-through any extra CLI args ("$@") AFTER OVERRIDES so they take
# precedence (OmegaConf last-wins). This lets users do ad-hoc overrides
# like:  bash $0 data.num_samples=4 training.eval_every=10
launch_ddp "$NUM_GPUS" "$GPU_IDS" "$SCRIPT" \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}" \
    "${OVERRIDES[@]}" \
    "$@"
