#!/usr/bin/env bash
# Run the full PhysX-Mobility pipeline on a 4-GPU cloud dev machine.
#
# The noisy per-step logs go to LOG_DIR. The terminal stays compact: dataset
# size, 13-step progress, and per-rank completion for GPU-sharded stages.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
TOOLKIT_DIR="$PROJECT_ROOT/submodules/dataset_toolkits"
CONFIG_OUT="$TOOLKIT_DIR/configs/PhysX-Mobility.cloud.generated.yaml"

DATA_ROOT="${DATA_ROOT:-/robot/data-lab/arts-gen-data/data/PhysX-Mobility-full-v1}"
SOURCE_ROOT="${SOURCE_ROOT:-/robot/data-lab/arts-gen-data/data/PhysX-Mobility-4view}"
LEGACY_SOURCE_ROOT="${LEGACY_SOURCE_ROOT:-/robot/data-lab/arts-gen-data/PhysX-Mobility-4view}"
WORKERS="${WORKERS:-12}"
GPUS="${GPUS:-0,1,2,3}"
VIEWS_PER_QUADRANT="${VIEWS_PER_QUADRANT:-3}"
RGB_ENGINE="${RGB_ENGINE:-CYCLES}"
CYCLES_DEVICE="${CYCLES_DEVICE:-CPU}"
LOG_DIR="${LOG_DIR:-/tmp/physx_full_$(date +%Y%m%d_%H%M%S)}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"

usage() {
  cat <<EOF
Run full PhysX-Mobility preprocessing with simple terminal progress and
4-GPU sharding for SS/SLat stages.

Typical usage:
  bash scripts/ops/data_pipeline/run_physx_mobility_cloud_4gpu_full.sh \\
    --data-root /robot/data-lab/arts-gen-data/data/PhysX-Mobility-full-v1 \\
    --workers 12 \\
    --gpus 0,1,2,3 \\
    --rgb-engine CYCLES

Options:
  --data-root PATH           Full output data_root. Default: $DATA_ROOT
  --source-root PATH         Existing raw dataset root. Default: $SOURCE_ROOT
  --legacy-source-root PATH  Fallback raw dataset root. Default: $LEGACY_SOURCE_ROOT
  --workers N                Worker count for CPU/render/voxel stages. Default: $WORKERS
  --gpus CSV                 GPU ids for sharded stages. Default: $GPUS
  --views-per-quadrant N     3 means 12 rendered views. Default: $VIEWS_PER_QUADRANT
  --rgb-engine NAME          BLENDER_EEVEE_NEXT or CYCLES. Default: $RGB_ENGINE
  --cycles-device NAME       Cycles RGB device: CPU or CUDA. Default: $CYCLES_DEVICE
  --log-dir PATH             Detailed logs. Default: $LOG_DIR
  --progress-interval SEC    Print parsed progress every SEC seconds. Default: $PROGRESS_INTERVAL
  -h, --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --legacy-source-root)
      LEGACY_SOURCE_ROOT="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --views-per-quadrant)
      VIEWS_PER_QUADRANT="$2"
      shift 2
      ;;
    --rgb-engine)
      RGB_ENGINE="$2"
      shift 2
      ;;
    --cycles-device)
      CYCLES_DEVICE="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --progress-interval)
      PROGRESS_INTERVAL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
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
  CPU|CUDA) ;;
  *) echo "[error] --cycles-device must be CPU or CUDA: $CYCLES_DEVICE" >&2; exit 2 ;;
esac

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if (( ${#GPU_IDS[@]} < 1 )); then
  echo "[error] --gpus cannot be empty" >&2
  exit 2
fi

if [[ ! "$PROGRESS_INTERVAL" =~ ^[0-9]+$ || "$PROGRESS_INTERVAL" -lt 1 ]]; then
  echo "[error] --progress-interval must be a positive integer: $PROGRESS_INTERVAL" >&2
  exit 2
fi

PYTHON_BIN="$(command -v python3)"
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python3" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python3"
fi

mkdir -p "$LOG_DIR"

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
    echo "        remove it first or choose a new --data-root" >&2
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
  find -L "$DATA_ROOT/raw/finaljson" -maxdepth 1 -type f -name '*.json' -printf '.\n' | wc -l
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
  local step_label="$1"
  local name="$2"
  local log_path="$3"
  local pid="$4"
  local extractor="$5"
  local last_progress=""
  local progress

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$PROGRESS_INTERVAL"
    if progress="$("$extractor" "$log_path")"; then
      if [[ "$progress" != "$last_progress" ]]; then
        log "[$step_label/13] $name progress objects=$progress"
        last_progress="$progress"
      fi
    fi
  done
}

run_serial_steps() {
  local step_label="$1"
  local steps_csv="$2"
  local log_path="$LOG_DIR/steps_${steps_csv//,/}_run_pipeline.log"
  local pid monitor_pid

  log "[$step_label/13] start objects=${OBJECT_COUNT}/${OBJECT_COUNT} log=$log_path"
  bash "$SCRIPT_DIR/run_physx_mobility_cloud.sh" \
    --full \
    --data-root "$DATA_ROOT" \
    --legacy-data-root "$SOURCE_ROOT" \
    --workers "$WORKERS" \
    --views-per-quadrant "$VIEWS_PER_QUADRANT" \
    --rgb-engine "$RGB_ENGINE" \
    --cycles-device "$CYCLES_DEVICE" \
    --steps "$steps_csv" \
    --config-out "$CONFIG_OUT" \
    >"$log_path" 2>&1 &
  pid="$!"
  monitor_single_log_progress "$step_label" "step" "$log_path" "$pid" latest_bracket_progress &
  monitor_pid="$!"

  if wait "$pid"; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    log "[$step_label/13] done objects=${OBJECT_COUNT}/${OBJECT_COUNT}"
  else
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    log "[$step_label/13] failed, see $log_path"
    return 1
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
  local world_size="${#GPU_IDS[@]}"
  local -a pids=()
  local -a monitor_pids=()
  local -a ranks=()
  local -a logs=()
  local rank gpu log_path report_path pid

  log "[$step_label/13] $name start ranks=$world_size objects=${OBJECT_COUNT}/${OBJECT_COUNT}"
  for rank in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$rank]}"
    log_path="$LOG_DIR/step${step_label}_${name// /_}_rank${rank}.log"
    logs+=("$log_path")
    ranks+=("$rank")

    (
      cd "$TOOLKIT_DIR"
      export CUDA_VISIBLE_DEVICES="$gpu"
      if [[ "$requires_numba" == "yes" ]]; then
        export NUMBA_DISABLE_JIT=1
      fi

      cmd=(
        "$PYTHON_BIN"
        "$script_name"
        --config "$CONFIG_OUT"
        --rank "$rank"
        --world-size "$world_size"
        "${extra_args[@]}"
      )
      if [[ "$supports_report" == "yes" ]]; then
        report_path="$LOG_DIR/step${step_label}_${name// /_}_rank${rank}.json"
        cmd+=(--report-path "$report_path")
      fi
      "${cmd[@]}"
    ) >"$log_path" 2>&1 &
    pid="$!"
    pids+=("$pid")
    monitor_single_log_progress \
      "$step_label" \
      "$name rank=$((rank + 1))/$world_size" \
      "$log_path" \
      "$pid" \
      latest_ratio_progress &
    monitor_pids+=("$!")
    log "[$step_label/13] $name rank=$((rank + 1))/$world_size gpu=$gpu launched"
  done

  for rank in "${!pids[@]}"; do
    if wait "${pids[$rank]}"; then
      kill "${monitor_pids[$rank]}" 2>/dev/null || true
      wait "${monitor_pids[$rank]}" 2>/dev/null || true
      log "[$step_label/13] $name rank=$((ranks[$rank] + 1))/$world_size done"
    else
      for monitor_pid in "${monitor_pids[@]}"; do
        kill "$monitor_pid" 2>/dev/null || true
      done
      for monitor_pid in "${monitor_pids[@]}"; do
        wait "$monitor_pid" 2>/dev/null || true
      done
      log "[$step_label/13] $name rank=$((ranks[$rank] + 1))/$world_size failed, see ${logs[$rank]}"
      return 1
    fi
  done
  log "[$step_label/13] $name done"
}

SOURCE_ROOT="$(resolve_source_root)"
ensure_raw_links "$SOURCE_ROOT"
OBJECT_COUNT="$(count_objects | tr -d ' ')"

log "dataset objects=${OBJECT_COUNT} data_root=$DATA_ROOT"
log "source_root=$SOURCE_ROOT"
log "gpus=$GPUS workers=$WORKERS rgb_engine=$RGB_ENGINE cycles_device=$CYCLES_DEVICE logs=$LOG_DIR"

run_serial_steps "01" "1"
run_serial_steps "02" "2"
run_serial_steps "03" "3"
run_serial_steps "04" "4"
run_serial_steps "05" "5"
run_serial_steps "06" "6"
run_serial_steps "07" "7"
run_sharded_python "08" "ss_per_part" "pipeline/08_encode_ss_latents_per_part.py" "yes" "yes"
run_sharded_python "08" "ss_overall" "utils/encode_ss_latents_expanded.py" "yes" "no"
run_serial_steps "09" "9"
run_sharded_python "10" "ss_decode" "pipeline/10_decode_ss_latents.py" "yes" "yes"
run_serial_steps "11" "11"
run_sharded_python "12" "part_slat" "pipeline/12_encode_part_synthesis_slat.py" "yes" "yes"
run_serial_steps "13" "13"

log "DONE full pipeline completed: 13/13"
log "output=$DATA_ROOT"
log "logs=$LOG_DIR"
