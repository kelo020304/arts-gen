#!/usr/bin/env bash
# ============================================================
# run_physx_mobility_multi_gpu.sh
#
# Auto-shard PhysX-Mobility pipeline across all available GPUs.
#
# Detects GPU count from nvidia-smi, splits object IDs into N groups
# (one per GPU), launches N parallel run_physx_mobility_cloud.sh batches
# each pinned via CUDA_VISIBLE_DEVICES.
#
# Each batch runs the FULL pipeline (steps 1-N) on its own slice of objects.
# CUDA_VISIBLE_DEVICES affects both NVIDIA Vulkan (Blender EEVEE) and CUDA
# (PyTorch / DINOv2 / SS encoder) — so the slice stays on one GPU end-to-end.
#
# Usage:
#   bash scripts/ops/data_pipeline/run_physx_mobility_multi_gpu.sh \
#     --data-root /robot/data-lab/.../PhysX-Mobility-full-4view-0511 \
#     [--workers-per-gpu 4] \
#     [--profile complete] \
#     [--source-finaljson /path]   # default: <data-root>/raw/finaljson
#     [--gpus 0,1,2,3]              # default: auto-detect from nvidia-smi
#     [--log-dir /tmp/physx_multi_TIMESTAMP]
#
# All other args (--rgb-engine, --cycles-device, etc.) you'd normally pass to
# run_physx_mobility_cloud.sh are NOT forwarded; this script targets the
# EEVEE_NEXT + Vulkan path (default for run_physx_mobility_cloud.sh). Override
# via env vars if needed (see run_physx_mobility_cloud.sh source).
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

DATA_ROOT=""
WORKERS_PER_GPU=1   # Default 1 for stability — Vulkan multi-context on single
                    # GPU causes occasional Blender SIGSEGV under contention.
                    # Failed objects are skipped + logged, not retried.
PROFILE="complete"
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

# Auto-detect GPUs if not specified
if [ -z "$GPUS" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[error] nvidia-smi not found, can't auto-detect GPUs. Pass --gpus" >&2
    exit 3
  fi
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
  [ -n "$GPUS" ] || { echo "[error] nvidia-smi listed no GPUs" >&2; exit 3; }
fi
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

# Source finaljson dir — where object IDs come from
if [ -z "$SOURCE_FINALJSON" ]; then
  SOURCE_FINALJSON="$DATA_ROOT/raw/finaljson"
fi
[ -d "$SOURCE_FINALJSON" ] || { echo "[error] finaljson dir not found: $SOURCE_FINALJSON" >&2; exit 4; }

# Log dir
if [ -z "$LOG_DIR" ]; then
  LOG_DIR="/tmp/physx_multi_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

# List object IDs (either from --object-ids subset or auto-discover from finaljson dir)
if [ -n "$OBJECT_IDS" ]; then
  IFS=',' read -ra ALL_OBJS <<< "$OBJECT_IDS"
else
  mapfile -t ALL_OBJS < <(ls "$SOURCE_FINALJSON" | sed -n 's/\.json$//p' | sort)
fi
TOTAL=${#ALL_OBJS[@]}
[ "$TOTAL" -gt 0 ] || { echo "[error] no objects to process" >&2; exit 5; }

PER_GPU=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))   # ceiling division

# Pre-flight: setup_blender_headless_gpu present and ready?
SETUP_SH="$PROJECT_ROOT/scripts/ops/setup/setup_blender_headless_gpu.sh"
if [ -x "$SETUP_SH" ]; then
  if [ ! -f /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.550.144.03 ] \
     || ! pgrep -x Xvfb >/dev/null 2>&1; then
    echo "[setup] running headless GPU setup (one-time per container)"
    bash "$SETUP_SH"
  fi
fi

echo "============================================================"
echo "  Multi-GPU PhysX-Mobility pipeline"
echo "============================================================"
echo "  data_root        : $DATA_ROOT"
echo "  finaljson source : $SOURCE_FINALJSON"
echo "  total objects    : $TOTAL"
echo "  GPUs             : $GPUS ($NUM_GPUS)"
echo "  per-GPU objects  : $PER_GPU"
echo "  workers/GPU      : $WORKERS_PER_GPU (total parallel: $((NUM_GPUS * WORKERS_PER_GPU)))"
echo "  profile          : $PROFILE"
echo "  log dir          : $LOG_DIR"
echo "============================================================"

PIDS=()
for IDX in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$IDX]}"
  START=$(( IDX * PER_GPU ))
  END=$(( START + PER_GPU ))
  [ $END -gt $TOTAL ] && END=$TOTAL
  if [ $START -ge $TOTAL ]; then
    echo "  GPU $GPU: no objects to process (TOTAL=$TOTAL, START=$START)"
    continue
  fi
  COUNT=$(( END - START ))
  GROUP=$( IFS=,; echo "${ALL_OBJS[*]:$START:$COUNT}" )

  LOG_FILE="$LOG_DIR/gpu_${GPU}.log"
  # CUDA_VISIBLE_DEVICES filters CUDA (PyTorch, etc.) but NOT NVIDIA Vulkan
  # device enumeration. For Blender EEVEE_NEXT (Vulkan) we additionally need
  # MESA_VK_DEVICE_SELECT (from the VK_LAYER_MESA_device_select layer that
  # ships with vulkan-tools) to pick a specific physical GPU.
  # MESA_VK_DEVICE_SELECT=N picks the Nth Vulkan device by enumeration order,
  # which matches NVIDIA driver's GPU index here.
  CUDA_VISIBLE_DEVICES=$GPU MESA_VK_DEVICE_SELECT=$GPU \
    nohup bash "$SCRIPT_DIR/run_physx_mobility_cloud.sh" \
    --full \
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
echo ""

if [ "$WAIT_FOR_COMPLETION" -eq 1 ]; then
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

  # Aggregate per-object failures across all batches
  FAILURE_LOG="$DATA_ROOT/manifests/render_failures.jsonl"
  if [ -f "$FAILURE_LOG" ]; then
    NUM_FAILS=$(wc -l < "$FAILURE_LOG")
    echo ""
    echo "  Per-object render failures (across all GPUs): $NUM_FAILS"
    echo "  Failure log: $FAILURE_LOG"
    if command -v jq >/dev/null 2>&1; then
      UNIQ_OBJS=$(jq -r '.object_id' "$FAILURE_LOG" | sort -u | wc -l)
      echo "  Unique failed objects: $UNIQ_OBJS"
      echo ""
      echo "  Retry with:"
      echo "    OBJS=\$(jq -r '.object_id' $FAILURE_LOG | sort -u | paste -sd,)"
      echo "    bash scripts/ops/data_pipeline/run_physx_mobility_multi_gpu.sh \\"
      echo "      --data-root $DATA_ROOT --object-ids \"\$OBJS\" --workers-per-gpu 1"
    else
      echo "  (install jq for nicer summary: apt-get install -y jq)"
    fi
  fi
  echo "============================================================"
  exit $FAILED
else
  echo "[--no-wait] script exits, batches keep running in background"
  echo "PIDs: ${PIDS[*]}"
  echo "Wait manually: wait ${PIDS[*]}    (or use 'jobs')"
  echo ""
  echo "After they finish, per-object failures (if any) are at:"
  echo "  $DATA_ROOT/manifests/render_failures.jsonl"
fi
