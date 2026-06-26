#!/usr/bin/env bash
# Run the PhysX-Mobility cloud pipeline on one GPU.
#
# Stable fallback for VolcEngine/Blender EEVEE_NEXT environments where Vulkan
# ignores per-process multi-GPU selectors. This script deliberately uses one
# GPU, records per-render failures, and prints failure summaries at the end.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
TOOLKIT_DIR="$PROJECT_ROOT/submodules/dataset_toolkits"

DATA_ROOT="${DATA_ROOT:-/robot/data-lab/arts-gen-data/data/PhysX-Mobility-full-v1}"
SOURCE_ROOT="${SOURCE_ROOT:-/robot/data-lab/arts-gen-data/data/PhysX-Mobility-4view}"
LEGACY_SOURCE_ROOT="${LEGACY_SOURCE_ROOT:-/robot/data-lab/arts-gen-data/PhysX-Mobility-4view}"
BASE_CONFIG="${BASE_CONFIG:-$TOOLKIT_DIR/configs/PhysX-Mobility.yaml}"
GPU="${GPU:-3}"
VULKAN_GPU="${VULKAN_GPU:-}"
# Multi-GPU fan-out: empty = single-GPU fallback (=$GPU). Opt-in.
RENDER_GPU="${RENDER_GPU:-}"
SHARD_GPUS="${SHARD_GPUS:-}"
SHARD_WORLD_SIZE=""
WORKERS_PER_GPU_LEGACY="${WORKERS_PER_GPU:-}"
GLOBAL_WORKERS_LEGACY="${GLOBAL_WORKERS:-}"
# Per-phase worker defaults (mirror run_physx_mobility_single_image_cloud.sh).
# Legacy env vars are accepted as fallbacks, but new flags/envs are preferred.
WORKERS_CPU="${WORKERS_CPU:-${WORKERS_PER_GPU_LEGACY:-${GLOBAL_WORKERS_LEGACY:-16}}}"
WORKERS_RENDER="${WORKERS_RENDER:-1}"
WORKERS_CUDA="${WORKERS_CUDA:-4}"
VIEWS_PER_QUADRANT="${VIEWS_PER_QUADRANT:-3}"
RGB_ENGINE="${RGB_ENGINE:-BLENDER_EEVEE_NEXT}"
CYCLES_DEVICE="${CYCLES_DEVICE:-CPU}"
LOG_DIR="${LOG_DIR:-/tmp/physx_latest}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
OBJECT_STEPS="${OBJECT_STEPS:-1,2,3,4,5}"
GLOBAL_PRE_STEPS="${GLOBAL_PRE_STEPS:-6,7}"
GLOBAL_MID_STEPS="${GLOBAL_MID_STEPS:-9}"
GLOBAL_PREVIEW_STEPS="${GLOBAL_PREVIEW_STEPS:-11}"
GLOBAL_FINAL_STEPS="${GLOBAL_FINAL_STEPS:-13}"
CHECK_AFTER_STEP="${CHECK_AFTER_STEP:-1}"
AUTO_REPAIR="${AUTO_REPAIR:-1}"
CONFIG_GLOBAL=""
DRY_RUN="0"
FAILED_STEPS=()

usage() {
  cat <<EOF
Run PhysX-Mobility preprocessing on one GPU.

Typical usage:
  bash scripts/ops/data_pipeline/run_physx_mobility_cloud_1gpu.sh \\
    --data-root /robot/data-lab/arts-gen-data/data/PhysX-Mobility-full-4view-0511 \\
    --gpu 3 \\
    --workers-cpu 16 \\
    --workers-render 1 \\
    --workers-cuda 4 \\
    --rgb-engine BLENDER_EEVEE_NEXT

Options:
  --data-root PATH           Full output data_root. Default: $DATA_ROOT
  --source-root PATH         Existing raw dataset root. Default: $SOURCE_ROOT
  --legacy-source-root PATH  Fallback raw dataset root. Default: $LEGACY_SOURCE_ROOT
  --base-config PATH         Base dataset_toolkits YAML to patch for cloud paths. Default: $BASE_CONFIG
  --gpu N                    CUDA GPU id to use. Default: $GPU
  --vulkan-gpu N             Optional Vulkan selector label. Default: same as --gpu.
  --render-gpu N             Render (step 2) GPU id. Single value only. Fallback: --gpu.
  --shard-gpus N,N,...       Fan-out GPU ids for steps 5/8/10/12. Default: --gpu (single, no fan-out).
  --workers-cpu N            Workers for CPU-bound steps (1, 3, 4, 6, 7, 9, 11, 13). Default: $WORKERS_CPU
  --workers-render N         Workers for Blender render step 2. Default: $WORKERS_RENDER (1 = avoid GPU contention)
  --workers-cuda N           Workers for CUDA steps (5, 8, 10, 12). Default: $WORKERS_CUDA
  --workers-per-gpu N        DEPRECATED alias for --workers-cpu (kept for backward compatibility).
  --workers N                DEPRECATED alias for --workers-cpu.
  --global-workers N         DEPRECATED alias for --workers-cpu (was global-phase-only before refactor).
  --views-per-quadrant N     3 means 12 rendered views. Default: $VIEWS_PER_QUADRANT
  --rgb-engine NAME          BLENDER_EEVEE_NEXT or CYCLES. Default: $RGB_ENGINE
  --cycles-device NAME       Cycles RGB device: CPU, CUDA, or OPTIX. Default: $CYCLES_DEVICE
  --log-dir PATH             Detailed logs and reports. Default: $LOG_DIR
  --progress-interval SEC    Print parsed progress every SEC seconds. Default: $PROGRESS_INTERVAL
  --object-steps CSV         Object-local cloud steps. Default: $OBJECT_STEPS
  --check-after-step         Run post-step checks/repairs. Default: enabled.
  --no-check-after-step      Disable post-step checks/repairs.
  --auto-repair              Repair supported gaps after checks. Default: enabled.
  --no-auto-repair           Check only; do not repair supported gaps.
  --config-out PATH          Generated config path. Default: <log-dir>/PhysX-Mobility.cloud.single.yaml
  --dry-run                  Print the schedule, then exit.
  -h, --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --legacy-source-root) LEGACY_SOURCE_ROOT="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --vulkan-gpu) VULKAN_GPU="$2"; shift 2 ;;
    --render-gpu) RENDER_GPU="$2"; shift 2 ;;
    --shard-gpus) SHARD_GPUS="$2"; shift 2 ;;
    --workers|--workers-per-gpu) WORKERS_CPU="$2"; shift 2 ;;
    --global-workers) WORKERS_CPU="$2"; shift 2 ;;
    --workers-cpu) WORKERS_CPU="$2"; shift 2 ;;
    --workers-render) WORKERS_RENDER="$2"; shift 2 ;;
    --workers-cuda) WORKERS_CUDA="$2"; shift 2 ;;
    --views-per-quadrant) VIEWS_PER_QUADRANT="$2"; shift 2 ;;
    --rgb-engine) RGB_ENGINE="$2"; shift 2 ;;
    --cycles-device) CYCLES_DEVICE="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --progress-interval) PROGRESS_INTERVAL="$2"; shift 2 ;;
    --object-steps) OBJECT_STEPS="$2"; shift 2 ;;
    --check-after-step) CHECK_AFTER_STEP="1"; shift ;;
    --no-check-after-step) CHECK_AFTER_STEP="0"; shift ;;
    --auto-repair) AUTO_REPAIR="1"; shift ;;
    --no-auto-repair) AUTO_REPAIR="0"; shift ;;
    --config-out) CONFIG_GLOBAL="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$DATA_ROOT" in
  /*) ;;
  *) echo "[error] --data-root must be absolute: $DATA_ROOT" >&2; exit 2 ;;
esac
case "$RGB_ENGINE" in
  BLENDER_EEVEE_NEXT|CYCLES) ;;
  *) echo "[error] --rgb-engine must be BLENDER_EEVEE_NEXT or CYCLES: $RGB_ENGINE" >&2; exit 2 ;;
esac
case "$CYCLES_DEVICE" in
  CPU|CUDA|OPTIX) ;;
  *) echo "[error] --cycles-device must be CPU, CUDA, or OPTIX: $CYCLES_DEVICE" >&2; exit 2 ;;
esac
if [[ -z "$GPU" || "$GPU" == *","* ]]; then
  echo "[error] --gpu expects one GPU id, got: $GPU" >&2
  exit 2
fi
for value_name in WORKERS_CPU WORKERS_RENDER WORKERS_CUDA PROGRESS_INTERVAL; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
    echo "[error] $value_name must be a positive integer: $value" >&2
    exit 2
  fi
done
for value_name in CHECK_AFTER_STEP AUTO_REPAIR; do
  value="${!value_name}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "[error] $value_name must be 0 or 1: $value" >&2
    exit 2
  fi
done
: "${RENDER_GPU:=$GPU}"
: "${SHARD_GPUS:=$GPU}"
if [[ ! "$RENDER_GPU" =~ ^[0-9]+$ ]]; then
  echo "[error] --render-gpu must be a non-negative integer, got: $RENDER_GPU" >&2
  exit 2
fi
if [[ ! "$SHARD_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[error] --shard-gpus must be comma-separated non-negative integers, got: $SHARD_GPUS" >&2
  exit 2
fi
IFS=',' read -r -a SHARD_GPU_IDS <<< "$SHARD_GPUS"
SHARD_WORLD_SIZE="${#SHARD_GPU_IDS[@]}"
if [[ -z "$VULKAN_GPU" ]]; then
  VULKAN_GPU="$RENDER_GPU"
fi

mkdir -p "$LOG_DIR"
if [[ -z "$CONFIG_GLOBAL" ]]; then
  CONFIG_GLOBAL="$LOG_DIR/PhysX-Mobility.cloud.single.yaml"
fi

PYTHON_BIN="$(command -v python3)"
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python3" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python3"
fi

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

resolve_source_root() {
  if [[ -d "$SOURCE_ROOT/raw/finaljson" && -d "$SOURCE_ROOT/raw/partseg" ]]; then
    printf '%s\n' "$SOURCE_ROOT"
    return 0
  fi
  if [[ -d "$LEGACY_SOURCE_ROOT/raw/finaljson" && -d "$LEGACY_SOURCE_ROOT/raw/partseg" ]]; then
    printf '%s\n' "$LEGACY_SOURCE_ROOT"
    return 0
  fi
  echo "[error] raw data not found under either:" >&2
  echo "        $SOURCE_ROOT/raw" >&2
  echo "        $LEGACY_SOURCE_ROOT/raw" >&2
  exit 3
}

ensure_raw_links() {
  local source_root="$1"

  if [[ "$DATA_ROOT" == "$source_root" ]]; then
    return 0
  fi
  if [[ -L "$DATA_ROOT" ]]; then
    echo "[error] data_root is a symlink; refusing to write full output into an aliased root: $DATA_ROOT" >&2
    exit 3
  fi

  mkdir -p "$DATA_ROOT/raw"
  if [[ ! -e "$DATA_ROOT/raw/finaljson" ]]; then
    ln -s "$source_root/raw/finaljson" "$DATA_ROOT/raw/finaljson"
  fi
  if [[ ! -e "$DATA_ROOT/raw/partseg" ]]; then
    ln -s "$source_root/raw/partseg" "$DATA_ROOT/raw/partseg"
  fi
}

count_objects() {
  find -L "$DATA_ROOT/raw/finaljson" -maxdepth 1 -type f -name '*.json' -printf '%f\n' |
    sed 's/\.json$//' |
    sort |
    wc -l
}

latest_bracket_progress() {
  local log_path="$1"
  local progress

  [[ -f "$log_path" ]] || return 1
  progress="$(
    grep -aEo '\[[0-9]+/[0-9]+\]' "$log_path" 2>/dev/null |
      tail -n 1 |
      tr -d '[]' || true
  )"
  [[ "$progress" =~ ^[0-9]+/[0-9]+$ ]] || return 1
  printf '%s\n' "$progress"
}

latest_ratio_progress() {
  local log_path="$1"
  local progress

  [[ -f "$log_path" ]] || return 1
  progress="$(
    tr '\r' '\n' < "$log_path" |
      grep -aEo '(^|[^0-9])([0-9]+)/([0-9]+)([^0-9]|$)' |
      sed -E 's/[^0-9]*([0-9]+)\/([0-9]+).*/\1\/\2/' |
      tail -n 1 || true
  )"
  [[ "$progress" =~ ^[0-9]+/[0-9]+$ ]] || return 1
  printf '%s\n' "$progress"
}

monitor_single_log_progress() {
  local label="$1"
  local log_path="$2"
  local pid="$3"
  local extractor="$4"
  local last_progress=""
  local progress

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$PROGRESS_INTERVAL"
    if progress="$("$extractor" "$log_path")"; then
      if [[ "$progress" != "$last_progress" ]]; then
        log "[$label] progress=$progress"
        last_progress="$progress"
      fi
    fi
  done
}

export_gpu_env_for() {
  local target_gpu="$1"
  if [[ -z "$target_gpu" ]]; then
    echo "[fatal] export_gpu_env_for called without GPU id" >&2
    return 1
  fi
  export CUDA_VISIBLE_DEVICES="$target_gpu"
}

export_blender_gpu_env_for() {
  local target_gpu="$1"
  export_gpu_env_for "$target_gpu"
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __VK_LAYER_NV_optimus=NVIDIA_only
  export MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1
  export MESA_VK_DEVICE_SELECT="${VULKAN_GPU}"
  export DRI_PRIME="${VULKAN_GPU}!"
  export BLENDER_USER_CONFIG="$LOG_DIR/blender_user_config/config"
  mkdir -p "$BLENDER_USER_CONFIG"
}

export_gpu_env() {
  export_blender_gpu_env_for "$GPU"
}

clear_gpu_env() {
  unset CUDA_VISIBLE_DEVICES
  unset __GLX_VENDOR_LIBRARY_NAME
  unset __NV_PRIME_RENDER_OFFLOAD
  unset __VK_LAYER_NV_optimus
  unset MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE
  unset MESA_VK_DEVICE_SELECT
  unset DRI_PRIME
  unset BLENDER_USER_CONFIG
}

# Intersect a comma-separated phase step list with OBJECT_STEPS.
# Returns the comma-separated intersection (or empty string).
filter_steps() {
  local phase_steps="$1"
  local result=""
  local s

  for s in ${phase_steps//,/ }; do
    case ",$OBJECT_STEPS," in
      *",$s,"*) result="${result:+$result,}$s" ;;
    esac
  done
  printf '%s' "$result"
}

run_cloud_steps() {
  local label="$1"
  local steps_csv="$2"
  local workers="$3"
  local gpu_target="${4:-}"
  local log_path="$LOG_DIR/${label}_steps_${steps_csv//,/}.log"
  local pid monitor_pid

  log "[$label steps $steps_csv] start workers=$workers log=$log_path"
  (
    if [[ -n "$gpu_target" ]]; then
      if [[ "$label" == "render" ]]; then
        export_blender_gpu_env_for "$gpu_target"
      else
        export_gpu_env_for "$gpu_target"
      fi
    else
      clear_gpu_env
    fi
    echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "[env] DRI_PRIME=${DRI_PRIME:-}"
    echo "[env] MESA_VK_DEVICE_SELECT=${MESA_VK_DEVICE_SELECT:-}"
    echo "[env] BLENDER_USER_CONFIG=${BLENDER_USER_CONFIG:-}"
    bash "$SCRIPT_DIR/run_physx_mobility_cloud.sh" \
      --full \
      --data-root "$DATA_ROOT" \
      --legacy-data-root "$SOURCE_ROOT" \
      --base-config "$BASE_CONFIG" \
      --workers "$workers" \
      --views-per-quadrant "$VIEWS_PER_QUADRANT" \
      --rgb-engine "$RGB_ENGINE" \
      --cycles-device "$CYCLES_DEVICE" \
      --steps "$steps_csv" \
      --config-out "$CONFIG_GLOBAL"
  ) >"$log_path" 2>&1 &
  pid="$!"
  monitor_single_log_progress "$label steps $steps_csv" "$log_path" "$pid" latest_bracket_progress &
  monitor_pid="$!"

  if wait "$pid"; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    log "[$label steps $steps_csv] done"
  else
    local rc="$?"
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    log "[$label steps $steps_csv] failed rc=$rc see $log_path"
    return "$rc"
  fi
}

run_sharded_python() {
  local step_label="$1"
  local name="$2"
  local script_name="$3"
  local requires_numba="$4"
  local supports_report="$5"
  shift 5
  local -a extra_args=("$@")
  local -a gpu_ids=("${SHARD_GPU_IDS[@]}")
  local world_size="$SHARD_WORLD_SIZE"

  if (( world_size == 1 )); then
    local log_path="$LOG_DIR/step${step_label}_${name}.log"
    local report_path="$LOG_DIR/step${step_label}_${name}.json"
    local pid monitor_pid

    log "[step $step_label $name] start single-rank gpu=${gpu_ids[0]} log=$log_path"
    (
      cd "$TOOLKIT_DIR"
      export_gpu_env_for "${gpu_ids[0]}"
      if [[ "$requires_numba" == "yes" ]]; then
        export NUMBA_DISABLE_JIT=1
      fi
      cmd=(
        "$PYTHON_BIN" "$script_name" \
        --config "$CONFIG_GLOBAL" \
        --rank 0 \
        --world-size 1 \
        "${extra_args[@]}"
      )
      if [[ "$supports_report" == "yes" ]]; then
        cmd+=(--report-path "$report_path")
        echo "[report] $report_path"
      fi
      "${cmd[@]}"
    ) >"$log_path" 2>&1 &
    pid="$!"
    monitor_single_log_progress "step $step_label $name" "$log_path" "$pid" latest_ratio_progress &
    monitor_pid="$!"

    if wait "$pid"; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
      log "[step $step_label $name] done"
    else
      local rc="$?"
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
      log "[step $step_label $name] failed rc=$rc see $log_path"
      return "$rc"
    fi
    return 0
  fi

  log "[step $step_label $name] start world_size=$world_size shard_gpus=$SHARD_GPUS"
  local -a pids=()
  local -a monitor_pids=()
  local -a ranks=()
  local -a logs=()
  local rank gpu log_path report_path pid monitor_pid
  for rank in "${!gpu_ids[@]}"; do
    gpu="${gpu_ids[$rank]}"
    log_path="$LOG_DIR/step${step_label}_${name}_rank${rank}.log"
    report_path="$LOG_DIR/step${step_label}_${name}_rank${rank}.json"
    logs+=("$log_path")
    ranks+=("$rank")
    (
      cd "$TOOLKIT_DIR"
      export_gpu_env_for "$gpu"
      if [[ "$requires_numba" == "yes" ]]; then
        export NUMBA_DISABLE_JIT=1
      fi
      cmd=(
        "$PYTHON_BIN" "$script_name" \
        --config "$CONFIG_GLOBAL" \
        --rank "$rank" \
        --world-size "$world_size" \
        "${extra_args[@]}"
      )
      if [[ "$supports_report" == "yes" ]]; then
        cmd+=(--report-path "$report_path")
        echo "[report] $report_path"
      fi
      "${cmd[@]}"
    ) >"$log_path" 2>&1 &
    pid="$!"
    pids+=("$pid")
    monitor_single_log_progress "step $step_label $name rank $rank" "$log_path" "$pid" latest_ratio_progress &
    monitor_pid="$!"
    monitor_pids+=("$monitor_pid")
  done

  local rc=0 r
  for r in "${!pids[@]}"; do
    if ! wait "${pids[$r]}"; then
      rc=1
      echo "[fatal] step $step_label $name rank=${ranks[$r]} failed; see ${logs[$r]}" >&2
    fi
    kill "${monitor_pids[$r]}" 2>/dev/null || true
    wait "${monitor_pids[$r]}" 2>/dev/null || true
  done
  log "[step $step_label $name] done rc=$rc"
  return "$rc"
}

post_step_validate() {
  local label="$1"
  local validator_steps="$2"
  local safe_steps="${validator_steps//,/}"
  local report_path="$LOG_DIR/check_${label}_${safe_steps}.json"

  [[ "$CHECK_AFTER_STEP" == "1" ]] || return 0
  log "[check $label] validate steps=$validator_steps report=$report_path"
  (
    cd "$TOOLKIT_DIR"
    "$PYTHON_BIN" "utils/validate_dataset.py" \
      --config "$CONFIG_GLOBAL" \
      --steps "$validator_steps" \
      --report-path "$report_path" \
      --top-n 20
  )
}

post_step_repair_render() {
  local label="$1"
  local expected_views=$((VIEWS_PER_QUADRANT * 4))
  local failure_log="$DATA_ROOT/manifests/render_failures.jsonl"
  local -a cmd

  [[ "$CHECK_AFTER_STEP" == "1" ]] || return 0
  cmd=(
    bash "$SCRIPT_DIR/repair_render_gaps.sh"
    --data-root "$DATA_ROOT"
    --expected-views "$expected_views"
  )
  if [[ -f "$failure_log" ]]; then
    cmd+=(--from-render-failures "$failure_log")
  fi
  if [[ "$AUTO_REPAIR" == "1" ]]; then
    cmd+=(--config "$CONFIG_GLOBAL" --workers "$WORKERS_RENDER" --yes)
    log "[check $label] repair render gaps expected_views=$expected_views"
  else
    cmd+=(--check-only)
    log "[check $label] scan render gaps expected_views=$expected_views"
  fi
  "${cmd[@]}"
}

summarize_failures() {
  local render_failures="$DATA_ROOT/manifests/render_failures.jsonl"

  log "failure summary begin"
  "$PYTHON_BIN" - "$render_failures" "$LOG_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

render_path = Path(sys.argv[1])
log_dir = Path(sys.argv[2])

if render_path.exists():
    rows = []
    for line in render_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    print(f"[failures] render_failures={len(rows)} path={render_path}", flush=True)
    counter = Counter(str(row.get("exit_code", "unknown")) for row in rows)
    print(f"[failures] render exit_code counts={dict(counter)}", flush=True)
    for row in rows[-50:]:
        print(
            "[failures] render "
            f"object={row.get('object_id')} angle={row.get('angle_idx')} "
            f"exit={row.get('exit_code')} reason={row.get('reason', '')}",
            flush=True,
        )
else:
    print(f"[failures] render_failures=0 path={render_path} (missing)", flush=True)

for report_path in sorted(log_dir.glob("step*.json")):
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[failures] report_unreadable path={report_path} reason={exc!r}", flush=True)
        continue
    summary = report.get("summary", {})
    failed = summary.get("failed", 0)
    passed = summary.get("passed", None)
    print(f"[failures] report={report_path.name} passed={passed} failed={failed}", flush=True)
    records = report.get("records", [])
    shown = 0
    for record in records:
        if record.get("status") in {"failed", "fatal"}:
            print(
                "[failures] report_record "
                f"status={record.get('status')} object={record.get('object_id')} "
                f"angle={record.get('angle_idx')} reason={record.get('reason')}",
                flush=True,
            )
            shown += 1
            if shown >= 20:
                break
PY
  log "failure summary end"
}

fail_after_step_failure() {
  local label="$1"

  summarize_failures
  log "pipeline stopped after failed required step: $label"
  log "pipeline finished with failed step(s): ${FAILED_STEPS[*]}"
  log "output=$DATA_ROOT"
  log "logs=$LOG_DIR"
  exit 1
}

if [[ "$DRY_RUN" == "1" ]]; then
  SOURCE_ROOT="<dry-run-not-checked>"
  OBJECT_COUNT="unknown"
else
  SOURCE_ROOT="$(resolve_source_root)"
  ensure_raw_links "$SOURCE_ROOT"
  OBJECT_COUNT="$(count_objects | tr -d '[:space:]')"
fi

log "single-gpu PhysX-Mobility pipeline"
log "data_root=$DATA_ROOT"
log "source_root=$SOURCE_ROOT"
log "base_config=$BASE_CONFIG"
log "objects=$OBJECT_COUNT gpu=$GPU vulkan_gpu=$VULKAN_GPU workers_cpu=$WORKERS_CPU workers_render=$WORKERS_RENDER workers_cuda=$WORKERS_CUDA"
log "render_gpu=$RENDER_GPU"
log "shard_gpus=$SHARD_GPUS world_size=$SHARD_WORLD_SIZE"
log "rgb_engine=$RGB_ENGINE cycles_device=$CYCLES_DEVICE logs=$LOG_DIR"
log "post_step_checks=$CHECK_AFTER_STEP auto_repair=$AUTO_REPAIR"
log "failure policy: per-render failures are skipped and appended to $DATA_ROOT/manifests/render_failures.jsonl"

if [[ "$DRY_RUN" == "1" ]]; then
  CPU_PRE_STEPS=$(filter_steps "1")
  RENDER_STEPS=$(filter_steps "2")
  CPU_MID_STEPS=$(filter_steps "3,4")
  FEATURE_STEPS=$(filter_steps "5")
  log "dry-run only"
  log "render_gpu=$RENDER_GPU"
  log "shard_gpus=$SHARD_GPUS world_size=$SHARD_WORLD_SIZE"
  log "would use log_dir=$LOG_DIR"
  log "would run post-step checks=$CHECK_AFTER_STEP auto_repair=$AUTO_REPAIR"
  [[ -n "$CPU_PRE_STEPS" ]] && log "would run cloud phase cpu-pre steps: $CPU_PRE_STEPS workers=$WORKERS_CPU"
  [[ -n "$RENDER_STEPS" ]] && log "would run cloud phase render steps: $RENDER_STEPS workers=$WORKERS_RENDER gpu=$RENDER_GPU"
  [[ -n "$CPU_MID_STEPS" ]] && log "would run cloud phase cpu-mid steps: $CPU_MID_STEPS workers=$WORKERS_CPU"
  [[ -n "$FEATURE_STEPS" ]] && log "would run python step 05 feature with shard_gpus=$SHARD_GPUS world_size=$SHARD_WORLD_SIZE"
  log "would run cloud global steps: $GLOBAL_PRE_STEPS -> $GLOBAL_MID_STEPS -> $GLOBAL_PREVIEW_STEPS -> $GLOBAL_FINAL_STEPS"
  log "would run python steps: 05, 08, 10, 12 with shard_gpus=$SHARD_GPUS world_size=$SHARD_WORLD_SIZE"
  exit 0
fi

CPU_PRE_STEPS=$(filter_steps "1")
RENDER_STEPS=$(filter_steps "2")
CPU_STEP3=$(filter_steps "3")
CPU_STEP4=$(filter_steps "4")
FEATURE_STEPS=$(filter_steps "5")

if [[ -n "$CPU_PRE_STEPS" ]]; then
  if ! run_cloud_steps "cpu-pre" "$CPU_PRE_STEPS" "$WORKERS_CPU"; then
    FAILED_STEPS+=("cpu-pre:$CPU_PRE_STEPS")
  fi
fi
if [[ -n "$RENDER_STEPS" ]]; then
  if ! run_cloud_steps "render" "$RENDER_STEPS" "$WORKERS_RENDER" "$RENDER_GPU"; then
    FAILED_STEPS+=("render:$RENDER_STEPS")
  fi
  if ! post_step_repair_render "step02_render"; then
    FAILED_STEPS+=("check:step02_render")
    fail_after_step_failure "step02_render-check"
  fi
fi
if [[ -n "$CPU_STEP3" ]]; then
  if ! run_cloud_steps "cpu-step3" "$CPU_STEP3" "$WORKERS_CPU"; then
    FAILED_STEPS+=("cpu-step3:$CPU_STEP3")
  fi
  if ! post_step_validate "step03_bbox" "render"; then
    FAILED_STEPS+=("check:step03_bbox")
    fail_after_step_failure "step03_bbox-check"
  fi
fi
if [[ -n "$CPU_STEP4" ]]; then
  if ! run_cloud_steps "cpu-step4" "$CPU_STEP4" "$WORKERS_CPU"; then
    FAILED_STEPS+=("cpu-step4:$CPU_STEP4")
  fi
  if ! post_step_validate "step04_voxel" "voxel"; then
    FAILED_STEPS+=("check:step04_voxel")
    fail_after_step_failure "step04_voxel-check"
  fi
fi
if [[ -n "$FEATURE_STEPS" ]]; then
  if ! run_sharded_python "05" "feature" "pipeline/05_extract_feature.py" "no" "no"; then
    FAILED_STEPS+=("05:feature")
  fi
  if ! post_step_validate "step05_dinov2" "dinov2"; then
    FAILED_STEPS+=("check:step05_dinov2")
    fail_after_step_failure "step05_dinov2-check"
  fi
fi
if ! run_cloud_steps "global-step6" "6" "$WORKERS_CPU"; then
  FAILED_STEPS+=("global-step6:6")
  fail_after_step_failure "global-step6"
fi
if ! run_cloud_steps "global-step7" "7" "$WORKERS_CPU"; then
  FAILED_STEPS+=("global-step7:7")
  fail_after_step_failure "global-step7"
fi
if ! post_step_validate "step07_vlm" "vlm"; then
  FAILED_STEPS+=("check:step07_vlm")
  fail_after_step_failure "step07_vlm-check"
fi
if ! run_sharded_python "08" "ss_per_part" "pipeline/08_encode_ss_latents_per_part.py" "yes" "yes" --continue-on-error; then
  FAILED_STEPS+=("08:ss_per_part")
fi
if ! run_sharded_python "08" "ss_overall" "utils/encode_ss_latents_expanded.py" "yes" "no"; then
  FAILED_STEPS+=("08:ss_overall")
fi
if ! post_step_validate "step08_ss_latent" "ss_latent"; then
  FAILED_STEPS+=("check:step08_ss_latent")
  fail_after_step_failure "step08_ss_latent-check"
fi
if ! run_cloud_steps "global-mid" "$GLOBAL_MID_STEPS" "$WORKERS_CPU"; then
  FAILED_STEPS+=("global-mid:$GLOBAL_MID_STEPS")
  fail_after_step_failure "global-mid"
fi
if ! run_sharded_python "10" "ss_decode" "pipeline/10_decode_ss_latents.py" "yes" "yes" --continue-on-error; then
  FAILED_STEPS+=("10:ss_decode")
fi
if ! run_cloud_steps "global-preview" "$GLOBAL_PREVIEW_STEPS" "$WORKERS_CPU"; then
  FAILED_STEPS+=("global-preview:$GLOBAL_PREVIEW_STEPS")
  fail_after_step_failure "global-preview"
fi
if ! post_step_validate "step11_preview" "preview"; then
  FAILED_STEPS+=("check:step11_preview")
  fail_after_step_failure "step11_preview-check"
fi
if ! run_sharded_python "12" "part_slat" "pipeline/12_encode_part_synthesis_slat.py" "yes" "yes" --continue-on-error; then
  FAILED_STEPS+=("12:part_slat")
fi
if ! run_cloud_steps "global-final" "$GLOBAL_FINAL_STEPS" "$WORKERS_CPU"; then
  FAILED_STEPS+=("global-final:$GLOBAL_FINAL_STEPS")
  fail_after_step_failure "global-final"
fi

summarize_failures
if (( ${#FAILED_STEPS[@]} > 0 )); then
  log "pipeline finished with failed step(s): ${FAILED_STEPS[*]}"
else
  log "pipeline finished without failed steps"
fi
log "output=$DATA_ROOT"
log "logs=$LOG_DIR"
