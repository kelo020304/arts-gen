#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-TRELLIS-arts/configs/arts/part_ss_latent_flow/part_ss_latent_flow.yaml}"
EXPORT_SCRIPT="TRELLIS-arts/export_part_ss_latent_flow_examples.py"
EVAL_SCRIPT="TRELLIS-arts/eval_part_ss_latent_flow.py"

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
RUN_ID="${RUN_ID:-manual_test_export}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/part_ss_latent_flow/$RUN_ID}"
INSPECTION_ROOT="${INSPECTION_ROOT:-$RUN_DIR/inspections_eval}"
EXPORT_ROOT="${EXPORT_ROOT:-$RUN_DIR/examples}"
EXPORT_LOG="${EXPORT_LOG:-$RUN_DIR/test_export.log}"
DEVICE="${DEVICE:-cuda}"
RUN_EVAL_DECODE="${RUN_EVAL_DECODE:-1}"
if [ -d "pretrained/TRELLIS-image-large/ckpts" ]; then
    DEFAULT_CKPT_DIR="pretrained/TRELLIS-image-large/ckpts"
elif [ -d "pretrained/ckpts" ]; then
    DEFAULT_CKPT_DIR="pretrained/ckpts"
else
    DEFAULT_CKPT_DIR="pretrained/TRELLIS-image-large/ckpts"
fi
CKPT_DIR="${CKPT_DIR:-$DEFAULT_CKPT_DIR}"
EXPORT_SLAT_ASSETS="${EXPORT_SLAT_ASSETS:-1}"
SLAT_FLOW_CKPT="${SLAT_FLOW_CKPT:-$CKPT_DIR/slat_flow_img_dit_L_64l8p2_fp16.safetensors}"
SLAT_GS_DECODER_CKPT="${SLAT_GS_DECODER_CKPT:-$CKPT_DIR/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors}"
SLAT_MESH_DECODER_CKPT="${SLAT_MESH_DECODER_CKPT:-$CKPT_DIR/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors}"
SLAT_NUM_STEPS="${SLAT_NUM_STEPS:-$NUM_STEPS}"
SLAT_SEED="${SLAT_SEED:-}"
SLAT_EMPTY_POLICY="${SLAT_EMPTY_POLICY:-skip}"

mkdir -p "$RUN_DIR" "$INSPECTION_ROOT" "$EXPORT_ROOT"
exec > >(tee -a "$EXPORT_LOG") 2>&1

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

SLAT_ASSET_ARGS=()
if [ "$EXPORT_SLAT_ASSETS" = "1" ] || [ "$EXPORT_SLAT_ASSETS" = "true" ] || [ "$EXPORT_SLAT_ASSETS" = "yes" ]; then
    SLAT_ASSET_ARGS+=(--export-slat-assets)
    SLAT_ASSET_ARGS+=(--slat-flow-ckpt "$SLAT_FLOW_CKPT")
    SLAT_ASSET_ARGS+=(--slat-gs-decoder-ckpt "$SLAT_GS_DECODER_CKPT")
    SLAT_ASSET_ARGS+=(--slat-mesh-decoder-ckpt "$SLAT_MESH_DECODER_CKPT")
    SLAT_ASSET_ARGS+=(--slat-num-steps "$SLAT_NUM_STEPS")
    [ -n "$SLAT_SEED" ] && SLAT_ASSET_ARGS+=(--slat-seed "$SLAT_SEED")
    SLAT_ASSET_ARGS+=(--slat-empty-policy "$SLAT_EMPTY_POLICY")
fi

echo "============================================================"
echo "Part SS Latent Flow Test Example Export Launcher"
echo "  CONFIG:       $CONFIG"
echo "  GPU_ID:       $GPU_ID"
echo "  CHECKPOINT:   ${CHECKPOINT:-<from LOAD_DIR/STEP>}"
echo "  LOAD_DIR:     ${LOAD_DIR:-<none>}"
echo "  STEP:         ${STEP:-<none>}"
echo "  RUN_DIR:      $RUN_DIR"
echo "  INSPECTION_ROOT: $INSPECTION_ROOT"
echo "  EXPORT_ROOT:  $EXPORT_ROOT"
echo "  EXPORT_LOG:   $EXPORT_LOG"
echo "  MAX_SAMPLES:  $MAX_SAMPLES"
echo "  SAMPLE_MODE:  $SAMPLE_MODE"
echo "  OBJECT_IDS:   ${OBJECT_IDS:-<none>}"
echo "  NUM_STEPS:    $NUM_STEPS"
echo "  RUN_EVAL_DECODE: $RUN_EVAL_DECODE"
echo "  EXPORT_SLAT_ASSETS: $EXPORT_SLAT_ASSETS"
echo "  CKPT_DIR: $CKPT_DIR"
echo "  SLAT_FLOW_CKPT: $SLAT_FLOW_CKPT"
echo "  SLAT_GS_DECODER_CKPT: $SLAT_GS_DECODER_CKPT"
echo "  SLAT_MESH_DECODER_CKPT: $SLAT_MESH_DECODER_CKPT"
echo "  SLAT_NUM_STEPS: $SLAT_NUM_STEPS"
echo "  SLAT_SEED: ${SLAT_SEED:-<training seed>}"
echo "  SLAT_EMPTY_POLICY: $SLAT_EMPTY_POLICY"
echo "  EXTRA OVERRIDES: ${OVERRIDES[*]:-<none>}"
echo "============================================================"
if [ ! -d "$CKPT_DIR" ]; then
    echo "[WARN] CKPT_DIR not found: $CKPT_DIR"
fi

if [ "$RUN_EVAL_DECODE" = "1" ] || [ "$RUN_EVAL_DECODE" = "true" ] || [ "$RUN_EVAL_DECODE" = "yes" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVAL_SCRIPT" \
        --config "$CONFIG" \
        "${CKPT_ARGS[@]}" \
        --inspection-root "$INSPECTION_ROOT" \
        --max-samples "$MAX_SAMPLES" \
        --sample-mode "$SAMPLE_MODE" \
        ${OBJECT_IDS:+--object-ids "$OBJECT_IDS"} \
        --num-steps "$NUM_STEPS" \
        --device "$DEVICE" \
        "${OVERRIDES[@]}"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EXPORT_SCRIPT" \
    --config "$CONFIG" \
    "${CKPT_ARGS[@]}" \
    --export-root "$EXPORT_ROOT" \
    --max-samples "$MAX_SAMPLES" \
    --sample-mode "$SAMPLE_MODE" \
    ${OBJECT_IDS:+--object-ids "$OBJECT_IDS"} \
    --num-steps "$NUM_STEPS" \
    --device "$DEVICE" \
    "${SLAT_ASSET_ARGS[@]}" \
    "${OVERRIDES[@]}"
