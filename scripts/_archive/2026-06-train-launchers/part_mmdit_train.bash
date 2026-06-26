#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_HOME="${TORCH_HOME:-submodules/TRELLIS.1}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export WANDB_IGNORE_GLOBS="${WANDB_IGNORE_GLOBS:-*.pt,*.safetensors,*.ckpt}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

CONFIG="${CONFIG:-TRELLIS-arts/configs/arts/part_mmdit/train_full_uniform.yaml}"
SCRIPT="TRELLIS-arts/train_arts.py"

GPU_IDS="${GPU_IDS:-}"
NUM_GPUS="${NUM_GPUS:-}"
if [ -n "$GPU_IDS" ]; then
    IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS"
    NUM_GPUS="${#GPU_IDS_ARR[@]}"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
NUM_GPUS="${NUM_GPUS:-1}"
if [ "$NUM_GPUS" -le 0 ]; then
    echo "[ERROR] Invalid NUM_GPUS=$NUM_GPUS" >&2
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/robot/data-lab/jzh/art-gen/outputs}"
RUN_ID="${RUN_ID:-part-mmdit-full-uniform-$(date +%m%d%H%M)}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR}"
TRAIN_LOG="${TRAIN_LOG:-$RUN_DIR/train.log}"
MAX_STEPS="${MAX_STEPS:-20000}"
LR="${LR:-}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
SS_DECODER_CKPT="${SS_DECODER_CKPT:-}"
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

OVERRIDES=()
[ -n "$DATA_ROOT" ]        && OVERRIDES+=("data.data_root=$DATA_ROOT" "eval.data.data_root=$DATA_ROOT")
[ -n "$MANIFEST_PATH" ]    && OVERRIDES+=("data.manifest_path=$MANIFEST_PATH" "eval.data.manifest_path=$MANIFEST_PATH")
[ -n "$OUTPUT_DIR" ]       && OVERRIDES+=("training.output_dir=$OUTPUT_DIR")
[ -n "$MAX_STEPS" ]        && OVERRIDES+=("training.max_steps=$MAX_STEPS")
[ -n "$LR" ]               && OVERRIDES+=("training.lr=$LR")
[ -n "$BATCH_SIZE" ]       && OVERRIDES+=("training.batch_size=$BATCH_SIZE")
[ -n "$NUM_WORKERS" ]      && OVERRIDES+=("training.num_workers=$NUM_WORKERS")
[ -n "$EVAL_EVERY" ]       && OVERRIDES+=("eval.fixed_every=$EVAL_EVERY" "eval.eval_every=$EVAL_EVERY")
[ -n "$CHECKPOINT_EVERY" ] && OVERRIDES+=("training.checkpoint_every=$CHECKPOINT_EVERY")
[ -n "$SS_DECODER_CKPT" ]  && OVERRIDES+=("eval.ss_decoder_ckpt=$SS_DECODER_CKPT")

OVERRIDES+=("$@")

CONFIRM_LINE="$(
python - "$CONFIG" "$BATCH_SIZE" "$NUM_GPUS" "${DATA_ROOT:-}" "${MANIFEST_PATH:-}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
batch_size = sys.argv[2]
num_gpus = sys.argv[3]
data_root_override = sys.argv[4]
manifest_override = sys.argv[5]

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
if "_base_" in cfg:
    base_path = config_path.parent / str(cfg["_base_"])
    with base_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    def merge(a, b):
        out = dict(a)
        for key, value in b.items():
            if key == "_base_":
                continue
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge(out[key], value)
            else:
                out[key] = value
        return out
    cfg = merge(base, cfg)
flow = cfg["flow"]
data = cfg["data"]
data_root = data_root_override or data["data_root"]
manifest = manifest_override or data["manifest_path"]
clip_path = data.get("name_emb_cache_path") or "reconstruction/name_emb_cache/clip_vitl14_seq.pt"
print(
    "[CONFIRM] "
    f"t_schedule={flow.get('t_schedule')} "
    f"timestep_shift={flow.get('timestep_shift', 0.0)} "
    f"gpus={num_gpus} batch_size={batch_size} "
    f"data_root={data_root} manifest={manifest} clip_name_cache={clip_path}"
)
PY
)"

echo "============================================================"
echo "PartMMDiT v2 Full Uniform Training"
echo "  CONFIG:          $CONFIG"
echo "  NUM_GPUS:        $NUM_GPUS  GPU_IDS='${GPU_IDS:-<unset>}'"
echo "  NNODES:          ${NNODES:-1}  NODE_RANK=${NODE_RANK:-0}"
echo "  MASTER_ADDR:     ${MASTER_ADDR:-<auto>}  MASTER_PORT=${MASTER_PORT:-29500}"
echo "  RUN_ID:          $RUN_ID"
echo "  RUN_DIR:         $RUN_DIR"
echo "  TRAIN_LOG:       $TRAIN_LOG"
echo "  OUTPUT_DIR:      $OUTPUT_DIR"
echo "  EXTRA OVERRIDES: ${OVERRIDES[*]:-<none>}"
echo "$CONFIRM_LINE"
echo "============================================================"

TORCHRUN_ARGS=(--nproc_per_node="$NUM_GPUS")
MULTINODE=0
if [ "${NNODES:-1}" != "1" ] || [ -n "${NODE_RANK:-}" ] || [ -n "${MASTER_ADDR:-}" ]; then
    MULTINODE=1
fi
if [ "$MULTINODE" -eq 1 ]; then
    TORCHRUN_ARGS+=(--nnodes="${NNODES:-1}")
    TORCHRUN_ARGS+=(--node_rank="${NODE_RANK:-0}")
    TORCHRUN_ARGS+=(--master_addr="${MASTER_ADDR:?MASTER_ADDR must be set for multi-node launch}")
    TORCHRUN_ARGS+=(--master_port="${MASTER_PORT:-29500}")
else
    TORCHRUN_ARGS+=(--master_port="${MASTER_PORT:-29500}")
fi

torchrun "${TORCHRUN_ARGS[@]}" "$SCRIPT" \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}" \
    "${OVERRIDES[@]}"
