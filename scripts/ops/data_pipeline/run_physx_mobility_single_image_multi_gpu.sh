#!/usr/bin/env bash
# ============================================================
# run_physx_mobility_single_image_multi_gpu.sh
#
# Multi-GPU wrapper for the NEW dataset_toolkits pipeline (single-image
# rendering format, commit 8f9b957+). Auto-shards objects across detected
# GPUs, launches one run_physx_mobility_single_image_cloud.sh per GPU,
# each pinned via CUDA_VISIBLE_DEVICES + MESA_VK_DEVICE_SELECT.
#
# CAVEAT: NVIDIA Vulkan does NOT respect CUDA_VISIBLE_DEVICES for device
# selection, and MESA_VK_DEVICE_SELECT (Mesa layer) only filters Mesa-side
# Vulkan drivers. Confirmed empirically: 4 batches end up sharing GPU 0
# for the Blender EEVEE step. CUDA-based steps (DINOv2, SS encoder/decoder)
# DO honor the pin and distribute across GPUs. So this script's main win
# is on steps 5/7/8/10 (CUDA), not step 5 render (Vulkan).
#
# Usage:
#   bash scripts/ops/data_pipeline/run_physx_mobility_single_image_multi_gpu.sh \
#     --data-root /robot/data-lab/.../PhysX-Mobility-single-image \
#     [--workers-per-gpu 1] \
#     [--profile default] \
#     [--source-finaljson /path]   # default: <data-root>/raw/finaljson
#     [--gpus 0,1,2,3]              # default: auto-detect
#     [--object-ids ...]
#     [--log-dir /tmp/physx_single_image_TIMESTAMP]
#     [--no-wait]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

DATA_ROOT=""
WORKERS_PER_GPU=1
PROFILE="default"
SOURCE_FINALJSON=""
GPUS=""
LOG_DIR=""
WAIT_FOR_COMPLETION=1
OBJECT_IDS=""

usage() {
  sed -n '2,30p' "$0"
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root)          DATA_ROOT="$2";          shift 2 ;;
    --workers-per-gpu)    WORKERS_PER_GPU="$2";    shift 2 ;;
    --profile)            PROFILE="$2";            shift 2 ;;
    --source-finaljson)   SOURCE_FINALJSON="$2";   shift 2 ;;
    --gpus)               GPUS="$2";               shift 2 ;;
    --log-dir)            LOG_DIR="$2";            shift 2 ;;
    --no-wait)            WAIT_FOR_COMPLETION=0;   shift   ;;
    --object-ids)         OBJECT_IDS="$2";         shift 2 ;;
    -h|--help)            usage ;;
    *) echo "[error] unknown arg: $1" >&2; usage ;;
  esac
done

[ -n "$DATA_ROOT" ] || { echo "[error] --data-root is required" >&2; exit 2; }
case "$DATA_ROOT" in /*) : ;; *) echo "[error] --data-root must be absolute" >&2; exit 2 ;; esac

# Auto-detect GPUs
if [ -z "$GPUS" ]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "[error] nvidia-smi missing, pass --gpus" >&2; exit 3; }
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
  [ -n "$GPUS" ] || { echo "[error] nvidia-smi listed no GPUs" >&2; exit 3; }
fi
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

# Source finaljson dir
[ -n "$SOURCE_FINALJSON" ] || SOURCE_FINALJSON="$DATA_ROOT/raw/finaljson"
[ -d "$SOURCE_FINALJSON" ] || { echo "[error] finaljson dir not found: $SOURCE_FINALJSON" >&2; exit 4; }

# Log dir
[ -n "$LOG_DIR" ] || LOG_DIR="/tmp/physx_single_image_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Object list (from --object-ids subset or auto-discover)
if [ -n "$OBJECT_IDS" ]; then
  IFS=',' read -ra ALL_OBJS <<< "$OBJECT_IDS"
else
  mapfile -t ALL_OBJS < <(ls "$SOURCE_FINALJSON" | sed -n 's/\.json$//p' | sort)
fi
TOTAL=${#ALL_OBJS[@]}
[ "$TOTAL" -gt 0 ] || { echo "[error] no objects to process" >&2; exit 5; }

PER_GPU=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

# Pre-flight: headless GPU setup
SETUP_SH="$PROJECT_ROOT/scripts/ops/setup/setup_blender_headless_gpu.sh"
if [ -x "$SETUP_SH" ]; then
  if [ ! -f /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.550.144.03 ] \
     || ! pgrep -x Xvfb >/dev/null 2>&1; then
    echo "[setup] running headless GPU setup (one-time per container)"
    bash "$SETUP_SH"
  fi
fi

echo "============================================================"
echo "  Multi-GPU PhysX-Mobility SINGLE-IMAGE pipeline (new toolkit)"
echo "============================================================"
echo "  toolkit dir       : $PROJECT_ROOT/submodules/dataset_toolkits_single_image"
echo "  data_root         : $DATA_ROOT"
echo "  finaljson source  : $SOURCE_FINALJSON"
echo "  total objects     : $TOTAL"
echo "  GPUs              : $GPUS ($NUM_GPUS)"
echo "  per-GPU objects   : $PER_GPU"
echo "  workers/GPU       : $WORKERS_PER_GPU (total parallel: $((NUM_GPUS * WORKERS_PER_GPU)))"
echo "  profile           : $PROFILE"
echo "  log dir           : $LOG_DIR"
echo "============================================================"

PIDS=()
for IDX in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$IDX]}"
  START=$(( IDX * PER_GPU ))
  END=$(( START + PER_GPU ))
  [ $END -gt $TOTAL ] && END=$TOTAL
  if [ $START -ge $TOTAL ]; then
    echo "  GPU $GPU: no objects to process"
    continue
  fi
  COUNT=$(( END - START ))
  GROUP=$( IFS=,; echo "${ALL_OBJS[*]:$START:$COUNT}" )

  LOG_FILE="$LOG_DIR/gpu_${GPU}.log"
  CUDA_VISIBLE_DEVICES=$GPU MESA_VK_DEVICE_SELECT=$GPU \
    nohup bash "$SCRIPT_DIR/run_physx_mobility_single_image_cloud.sh" \
      --data-root "$DATA_ROOT" \
      --profile "$PROFILE" \
      --object-ids "$GROUP" \
      --workers "$WORKERS_PER_GPU" \
      > "$LOG_FILE" 2>&1 &
  PID=$!
  PIDS+=("$PID")
  echo "  GPU $GPU: PID $PID, objs $START..$((END-1)) (count=$COUNT)  log=$LOG_FILE"
done

echo ""
echo "All $((${#PIDS[@]})) batches launched."
echo "Monitor:"
echo "  tail -f $LOG_DIR/gpu_*.log"
echo "  watch -n 3 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'"

if [ "$WAIT_FOR_COMPLETION" -eq 1 ]; then
  echo ""
  echo "Waiting for all batches to finish (Ctrl+C to detach, batches keep running)..."
  FAILED=0
  for PID in "${PIDS[@]}"; do
    if wait "$PID"; then
      echo "  PID $PID: completed"
    else
      RC=$?
      echo "  PID $PID: FAILED with exit code $RC"
      FAILED=$((FAILED + 1))
    fi
  done
  echo ""
  echo "============================================================"
  if [ "$FAILED" -eq 0 ]; then
    echo "  ALL BATCHES COMPLETED SUCCESSFULLY"
  else
    echo "  $FAILED batch(es) had non-zero exit — check $LOG_DIR/gpu_*.log"
  fi

  # Aggregate per-object render failures (if the new pipeline produces a similar log)
  FAILURE_LOG="$DATA_ROOT/manifests/render_failures.jsonl"
  if [ -f "$FAILURE_LOG" ]; then
    NUM_FAILS=$(wc -l < "$FAILURE_LOG")
    echo ""
    echo "  Per-object render failures (across all GPUs): $NUM_FAILS"
    echo "  Failure log: $FAILURE_LOG"
  fi
  echo "============================================================"
  exit $FAILED
else
  echo ""
  echo "[--no-wait] script exits, batches keep running in background"
  echo "PIDs: ${PIDS[*]}"
fi
