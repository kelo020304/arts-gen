#!/usr/bin/env bash
set -euo pipefail

SCRIPT="TRELLIS-arts/eval_part_ss_latent_flow_full.py"
EVAL_VIEW_MODE="${EVAL_VIEW_MODE:-${VIEW_MODE:-multiview}}"

case "$EVAL_VIEW_MODE" in
    multiview|multi|4view)
        DEFAULT_CONFIG="TRELLIS-arts/configs/arts/part_ss_latent_flow/part_ss_latent_flow.yaml"
        DEFAULT_RUN_ID="manual_full_eval"
        DEFAULT_OUTPUT_ROOT="/robot/data-lab/arts-gen-data/output"
        DEFAULT_DATA_ROOT=""
        DEFAULT_MANIFEST_PATH=""
        ;;
    single_view|single|1view)
        DEFAULT_CONFIG="TRELLIS-arts/configs/arts/part_ss_latent_flow_single_view/part_ss_latent_flow_single_view.yaml"
        DEFAULT_RUN_ID="manual_single_view_full_eval"
        DEFAULT_OUTPUT_ROOT="/robot/data-lab/jzh/art-gen-output"
        DEFAULT_DATA_ROOT="/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-single-image-0512"
        DEFAULT_MANIFEST_PATH="manifests/part_completion/arts_pc_physx-mobility_train.jsonl"
        ;;
    *)
        echo "[ERROR] EVAL_VIEW_MODE must be multiview|multi|4view or single_view|single|1view, got: $EVAL_VIEW_MODE" >&2
        exit 1
        ;;
esac

CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_IDS="${GPU_IDS:-$GPU_ID}"
IFS=',' read -r -a GPU_ID_ARGS <<< "$GPU_IDS"
if [ "${#GPU_ID_ARGS[@]}" -eq 0 ] || [ -z "${GPU_ID_ARGS[0]}" ]; then
    echo "[ERROR] GPU_IDS must contain at least one GPU id, got: $GPU_IDS" >&2
    exit 1
fi
GPU_ID="${GPU_ID_ARGS[0]}"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_ID_ARGS[@]}}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MERGE_SHARDS="${MERGE_SHARDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHECKPOINT="${CHECKPOINT:-}"
LOAD_DIR="${LOAD_DIR:-}"
STEP="${STEP:-}"
DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
RECON_SUBDIR="${RECON_SUBDIR:-}"
MASK_SUBDIR="${MASK_SUBDIR:-}"
MANIFEST_PATH="${MANIFEST_PATH:-$DEFAULT_MANIFEST_PATH}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
SAMPLE_MODE="${SAMPLE_MODE:-all}"
OBJECT_IDS="${OBJECT_IDS:-}"
NUM_STEPS="${NUM_STEPS:-20}"
WRITE_VOXEL_EXAMPLES="${WRITE_VOXEL_EXAMPLES:-20}"
SIZE_BUCKET_BOUNDARIES="${SIZE_BUCKET_BOUNDARIES:-500 3000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/part_ss_latent_flow/$RUN_ID}"
REPORT_ROOT="${REPORT_ROOT:-$RUN_DIR/full_eval}"
EVAL_LOG="${EVAL_LOG:-$RUN_DIR/full_eval.log}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$RUN_DIR" "$REPORT_ROOT"
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

read -r -a SIZE_BUCKET_ARGS <<< "$SIZE_BUCKET_BOUNDARIES"
if [ "${#SIZE_BUCKET_ARGS[@]}" -ne 2 ]; then
    echo "[ERROR] SIZE_BUCKET_BOUNDARIES must contain exactly two numbers, got: $SIZE_BUCKET_BOUNDARIES" >&2
    exit 1
fi
if [ "$NUM_SHARDS" -lt 1 ]; then
    echo "[ERROR] NUM_SHARDS must be >= 1, got: $NUM_SHARDS" >&2
    exit 1
fi
if [ "$BATCH_SIZE" -lt 1 ]; then
    echo "[ERROR] BATCH_SIZE must be >= 1, got: $BATCH_SIZE" >&2
    exit 1
fi

echo "============================================================"
echo "Part SS Latent Flow Full Eval Launcher"
echo "  EVAL_VIEW_MODE:       $EVAL_VIEW_MODE"
echo "  CONFIG:               $CONFIG"
echo "  GPU_ID:               $GPU_ID"
echo "  GPU_IDS:              $GPU_IDS"
echo "  NUM_SHARDS:           $NUM_SHARDS"
echo "  SHARD_INDEX:          $SHARD_INDEX"
echo "  BATCH_SIZE:           $BATCH_SIZE"
echo "  SKIP_EXISTING:        $SKIP_EXISTING"
echo "  CHECKPOINT:           ${CHECKPOINT:-<from LOAD_DIR/STEP>}"
echo "  LOAD_DIR:             ${LOAD_DIR:-<none>}"
echo "  STEP:                 ${STEP:-<none>}"
echo "  RUN_DIR:              $RUN_DIR"
echo "  REPORT_ROOT:          $REPORT_ROOT"
echo "  EVAL_LOG:             $EVAL_LOG"
echo "  MAX_SAMPLES:          $MAX_SAMPLES"
echo "  SAMPLE_MODE:          $SAMPLE_MODE"
echo "  OBJECT_IDS:           ${OBJECT_IDS:-<none>}"
echo "  NUM_STEPS:            $NUM_STEPS"
echo "  WRITE_VOXEL_EXAMPLES: $WRITE_VOXEL_EXAMPLES"
echo "  SIZE_BUCKETS:         ${SIZE_BUCKET_ARGS[*]}"
echo "  EXTRA OVERRIDES:      ${OVERRIDES[*]:-<none>}"
echo "============================================================"

SKIP_ARGS=()
if [ "$SKIP_EXISTING" != "0" ]; then
    SKIP_ARGS+=(--skip-existing)
fi

run_eval_process() {
    local gpu="$1"
    local shard_index="$2"
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
        --config "$CONFIG" \
        "${CKPT_ARGS[@]}" \
        --report-root "$REPORT_ROOT" \
        --max-samples "$MAX_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --sample-mode "$SAMPLE_MODE" \
        ${OBJECT_IDS:+--object-ids "$OBJECT_IDS"} \
        --num-steps "$NUM_STEPS" \
        --device "$DEVICE" \
        --write-voxel-examples "$WRITE_VOXEL_EXAMPLES" \
        --size-bucket-boundaries "${SIZE_BUCKET_ARGS[0]}" "${SIZE_BUCKET_ARGS[1]}" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard_index" \
        "${SKIP_ARGS[@]}" \
        "${OVERRIDES[@]}"
}

step_root_path() {
    printf "%s/step_%06d" "$REPORT_ROOT" "$STEP"
}

launch_shard() {
    local gpu="$1"
    local shard="$2"
    local shard_log="$RUN_DIR/full_eval_shard_${shard}_of_${NUM_SHARDS}.log"
    echo "[launch] shard $shard/$NUM_SHARDS on GPU $gpu -> $shard_log"
    run_eval_process "$gpu" "$shard" > "$shard_log" 2>&1 &
}

if [ "$NUM_SHARDS" -gt 1 ]; then
    if [ -z "$STEP" ]; then
        echo "[ERROR] parallel shard merge requires STEP=N. Set STEP even when CHECKPOINT is used." >&2
        exit 1
    fi
    pids=()
    next_shard=0
    while [ "$next_shard" -lt "$NUM_SHARDS" ]; do
        gpu="${GPU_ID_ARGS[$((next_shard % ${#GPU_ID_ARGS[@]}))]}"
        launch_shard "$gpu" "$next_shard"
        pids+=("$!")
        next_shard=$((next_shard + 1))
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "[ERROR] one or more full eval shards failed; inspect $RUN_DIR/full_eval_shard_*_of_${NUM_SHARDS}.log" >&2
        exit 1
    fi

    if [ "$MERGE_SHARDS" != "0" ]; then
        echo "[merge] merging $NUM_SHARDS shard outputs"
        python "$SCRIPT" \
            --config "$CONFIG" \
            --report-root "$REPORT_ROOT" \
            --merge-shards-only \
            --merge-step "$STEP" \
            --num-shards "$NUM_SHARDS" \
            --write-voxel-examples "$WRITE_VOXEL_EXAMPLES" \
            --size-bucket-boundaries "${SIZE_BUCKET_ARGS[0]}" "${SIZE_BUCKET_ARGS[1]}" \
            "${OVERRIDES[@]}"
    fi
else
    CUDA_VISIBLE_DEVICES="$GPU_ID" python "$SCRIPT" \
        --config "$CONFIG" \
        "${CKPT_ARGS[@]}" \
        --report-root "$REPORT_ROOT" \
        --max-samples "$MAX_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --sample-mode "$SAMPLE_MODE" \
        ${OBJECT_IDS:+--object-ids "$OBJECT_IDS"} \
        --num-steps "$NUM_STEPS" \
        --device "$DEVICE" \
        --write-voxel-examples "$WRITE_VOXEL_EXAMPLES" \
        --size-bucket-boundaries "${SIZE_BUCKET_ARGS[0]}" "${SIZE_BUCKET_ARGS[1]}" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$SHARD_INDEX" \
        "${SKIP_ARGS[@]}" \
        "${OVERRIDES[@]}"
fi
