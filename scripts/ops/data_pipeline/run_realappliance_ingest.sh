#!/usr/bin/env bash
# Run RealAppliance conversion plus the dataset_toolkits multi-view pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
TOOLKIT_ROOT="$PROJECT_ROOT/submodules/dataset_toolkits"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/RealAppliance}"
DEFAULT_SOURCE_TOS="tos://robot-data-lab/arts-reconstruction/data/realappliance_source"
SRC_ROOT="${SRC_ROOT:-$DEFAULT_SOURCE_TOS}"
SOURCE_CACHE_DEFAULTED="0"
if [[ -z "${SOURCE_CACHE+x}" ]]; then
  SOURCE_CACHE="$DATA_ROOT/source"
  SOURCE_CACHE_DEFAULTED="1"
fi
CONFIG="${CONFIG:-$TOOLKIT_ROOT/configs/RealAppliance.yaml}"
CONFIG_OUT="${CONFIG_OUT:-/tmp/realappliance_ingest/RealAppliance.generated.yaml}"
RUNTIME_CONFIG="$CONFIG_OUT"
TOSUTIL="${TOSUTIL:-tosutil}"
DEFAULT_IDS_LOCAL="001,002,003,004,005,006,007,008,009,010"

PROFILE="default"
OBJECT_IDS="$DEFAULT_IDS_LOCAL"
STEPS=""
WORKERS_CPU="${WORKERS_CPU:-16}"
WORKERS_RENDER="${WORKERS_RENDER:-1}"
WORKERS_CUDA="${WORKERS_CUDA:-4}"
GPU="${GPU:-0}"
RENDER_GPU="${RENDER_GPU:-}"
SHARD_GPUS="${SHARD_GPUS:-}"
SHARD_WORLD_SIZE=""
LOG_DIR="${LOG_DIR:-/tmp/realappliance_ingest}"
DRY_RUN="0"
SKIP_CONVERT="0"
SMOKE_ONLY="0"
ALL="0"
OVERWRITE="0"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Default: convert source models 001..010, then run run_pipeline.sh --profile default.

Options:
  --all                Process all 100 source models. Intended for the dev machine.
  --object-ids LIST    Comma-separated source ids, e.g. 001,005,050. ra_001 also works.
  --source-root PATH_OR_TOS
                       RealAppliance source root or TOS prefix.
                       Default: $SRC_ROOT
  --source-cache PATH  Local cache when --source-root is tos://...
                       Default: $SOURCE_CACHE
  --data-root PATH     Output dataset root. Default: $DATA_ROOT
  --config PATH        dataset_toolkits config. Default: $CONFIG
  --config-out PATH    Runtime config generated with current paths.
                       Default: $CONFIG_OUT
  --profile NAME       dataset_toolkits profile. Default: $PROFILE
  --steps CSV          Explicit dataset_toolkits steps, e.g. 1,2,3,4,5.
  --workers-cpu N      Workers for CPU/global steps. Default: $WORKERS_CPU
  --workers-render N   Workers for Blender render step 2. Default: $WORKERS_RENDER
  --workers-cuda N     Per-rank workers/concurrency budget for CUDA shard steps. Default: $WORKERS_CUDA
  --workers N          DEPRECATED alias for --workers-cpu.
  --gpu N              Default GPU id. Default: $GPU
  --render-gpu N       Render GPU id. Default: --gpu.
  --shard-gpus N,N,... GPU ids for CUDA steps 5/8/10/12. Default: --gpu.
  --log-dir PATH       Phase logs. Default: $LOG_DIR
  --skip-convert       Skip converter and run pipeline on existing raw/.
  --smoke-only         Use source model 001 and profile=base.
  --overwrite          Rebuild existing converter raw outputs.
  --dry-run            Print resolved cloud schedule, then exit.
  -h, --help           Show this help.

Paths:
  source: $SRC_ROOT
  source cache: $SOURCE_CACHE
  output: $DATA_ROOT
  base config: $CONFIG
  runtime config: $CONFIG_OUT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ALL="1"; shift ;;
    --object-ids) OBJECT_IDS="$2"; shift 2 ;;
    --source-root) SRC_ROOT="$2"; shift 2 ;;
    --source-cache) SOURCE_CACHE="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --config-out) CONFIG_OUT="$2"; RUNTIME_CONFIG="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --workers|--workers-cpu) WORKERS_CPU="$2"; shift 2 ;;
    --workers-render) WORKERS_RENDER="$2"; shift 2 ;;
    --workers-cuda) WORKERS_CUDA="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --render-gpu) RENDER_GPU="$2"; shift 2 ;;
    --shard-gpus) SHARD_GPUS="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --skip-convert) SKIP_CONVERT="1"; shift ;;
    --smoke-only) SMOKE_ONLY="1"; shift ;;
    --overwrite) OVERWRITE="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$SOURCE_CACHE_DEFAULTED" == "1" ]]; then
  SOURCE_CACHE="$DATA_ROOT/source"
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  OBJECT_IDS="001"
  PROFILE="base"
elif [[ "$ALL" == "1" ]]; then
  OBJECT_IDS="all"
fi

for value_name in WORKERS_CPU WORKERS_RENDER WORKERS_CUDA; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
    echo "[error] $value_name must be a positive integer: $value" >&2
    exit 2
  fi
done
if [[ -z "$GPU" || "$GPU" == *","* || ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "[error] --gpu expects one non-negative GPU id, got: $GPU" >&2
  exit 2
fi
: "${RENDER_GPU:=$GPU}"
: "${SHARD_GPUS:=$GPU}"
if [[ ! "$RENDER_GPU" =~ ^[0-9]+$ ]]; then
  echo "[error] --render-gpu must be one non-negative GPU id, got: $RENDER_GPU" >&2
  exit 2
fi
if [[ ! "$SHARD_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[error] --shard-gpus must be comma-separated non-negative GPU ids, got: $SHARD_GPUS" >&2
  exit 2
fi
IFS=',' read -r -a SHARD_GPU_IDS <<< "$SHARD_GPUS"
SHARD_WORLD_SIZE="${#SHARD_GPU_IDS[@]}"
mkdir -p "$LOG_DIR"

to_pipeline_ids() {
  local ids="$1"
  local out=()
  local item num
  if [[ "$ids" == "all" ]]; then
    printf '%s\n' "all"
    return 0
  fi
  IFS=',' read -r -a raw_items <<< "$ids"
  for item in "${raw_items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "$item" ]] || continue
    if [[ "$item" == ra_* ]]; then
      out+=("$item")
    elif [[ "$item" =~ ^[0-9]+$ ]]; then
      printf -v num 'ra_%03d' "$((10#$item))"
      out+=("$num")
    else
      echo "[error] invalid RealAppliance id: $item (expected 001 or ra_001)" >&2
      exit 2
    fi
  done
  local IFS=,
  printf '%s\n' "${out[*]}"
}

PIPELINE_OBJECT_IDS="$(to_pipeline_ids "$OBJECT_IDS")"

declare -A SELECTED_STEPS=()
if [[ -n "$STEPS" ]]; then
  IFS=',' read -r -a REQUESTED_STEPS <<< "$STEPS"
  for step in "${REQUESTED_STEPS[@]}"; do
    if [[ ! "$step" =~ ^([1-9]|1[0-3])$ ]]; then
      echo "[error] invalid --steps item: $step" >&2
      exit 2
    fi
    SELECTED_STEPS["$step"]=1
  done
else
  case "$PROFILE" in
    default|full|complete|all)
      for step in 1 2 3 4 5 6 7 8 9 10 11 12 13; do
        SELECTED_STEPS["$step"]=1
      done
      ;;
    stable)
      for step in 1 2 3 4 5 6 7 8 9 10 11; do
        SELECTED_STEPS["$step"]=1
      done
      ;;
    base|preview-base)
      for step in 1 2 3 4 5 6 7 11; do
        SELECTED_STEPS["$step"]=1
      done
      ;;
    *)
      echo "[error] unknown --profile: $PROFILE" >&2
      exit 2
      ;;
  esac
fi

filter_steps() {
  local phase_steps="$1"
  local result=""
  local step

  for step in ${phase_steps//,/ }; do
    if [[ -n "${SELECTED_STEPS[$step]:-}" ]]; then
      result="${result:+$result,}$step"
    fi
  done
  printf '%s' "$result"
}

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

export_gpu_env_for() {
  local target_gpu="$1"
  export CUDA_VISIBLE_DEVICES="$target_gpu"
}

start_headless_gpu() {
  # Xvfb + NVIDIA Vulkan is required for Blender EEVEE_NEXT on headless
  # cloud containers. CUDA visibility alone does not provide a graphics
  # surface, and Blender can otherwise crash during EGL/Vulkan startup.
  if ! pgrep -x Xvfb >/dev/null 2>&1; then
    if ! command -v Xvfb >/dev/null 2>&1; then
      echo "[error] Xvfb not found; run scripts/ops/setup/setup_blender_headless_gpu.sh first." >&2
      exit 3
    fi
    Xvfb :99 -screen 0 1024x768x24 &
    sleep 1
  fi
  export DISPLAY=:99
}

export_blender_gpu_env_for() {
  local target_gpu="$1"
  start_headless_gpu
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __VK_LAYER_NV_optimus=NVIDIA_only
  # Keep the render environment aligned with run_physx_mobility_single_image_cloud.sh.
  # That path has already been validated on the cloud machines this launcher targets.
}

clear_gpu_env() {
  unset CUDA_VISIBLE_DEVICES
  unset __GLX_VENDOR_LIBRARY_NAME
  unset __VK_LAYER_NV_optimus
}

prepare_python_env_for_toolkit() {
  # Respect the already-active Python environment. The upstream
  # dataset_toolkits/run_pipeline.sh checks CONDA_* names even when the same
  # dependency stack is provided by a virtualenv, so map VIRTUAL_ENV into the
  # shape it expects without activating or switching environments.
  if [[ -z "${CONDA_PREFIX:-}" && -n "${VIRTUAL_ENV:-}" ]]; then
    export CONDA_PREFIX="$VIRTUAL_ENV"
  fi
  export CONDA_DEFAULT_ENV=dataset_toolkits
}

require_usd_core_for_convert() {
  if ! python3 -c "from pxr import Usd, UsdPhysics" >/dev/null 2>&1; then
    echo "[fatal] pxr (usd-core) is not installed in the active python environment. Run:" >&2
    echo "  python3 -m pip install usd-core" >&2
    exit 3
  fi
}

require_existing_raw_for_skip_convert() {
  local finaljson_dir="$DATA_ROOT/raw/finaljson"
  local count

  if [[ ! -d "$finaljson_dir" ]]; then
    echo "[fatal] --skip-convert requires existing raw/finaljson at $finaljson_dir. Pull raw first:" >&2
    echo "  mkdir -p '$DATA_ROOT'" >&2
    echo "  $TOSUTIL cp -r tos://robot-data-lab/arts-reconstruction/data/RealAppliance-4view-0515/raw '$DATA_ROOT/'" >&2
    exit 4
  fi

  count="$(find "$finaljson_dir" -maxdepth 1 -name '*.json' -type f | wc -l)"
  if [[ "$count" -lt 1 ]]; then
    echo "[fatal] --skip-convert found no finaljson files in $finaljson_dir. Pull raw first:" >&2
    echo "  $TOSUTIL cp -r tos://robot-data-lab/arts-reconstruction/data/RealAppliance-4view-0515/raw '$DATA_ROOT/'" >&2
    exit 4
  fi
  echo "[ingest] existing raw preflight: finaljson=$count dir=$finaljson_dir"
}

run_cloud_steps() {
  local label="$1"
  local steps_csv="$2"
  local workers="$3"
  local gpu_target="${4:-}"
  local log_path="$LOG_DIR/${label}_steps_${steps_csv//,/}.log"

  log "[$label steps $steps_csv] start workers=$workers log=$log_path"
  (
    cd "$TOOLKIT_ROOT"
    prepare_python_env_for_toolkit
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
    cmd=(
      bash run_pipeline.sh
      --config "$RUNTIME_CONFIG"
      --steps "$steps_csv"
    )
    if [[ "$PIPELINE_OBJECT_IDS" != "all" ]]; then
      cmd+=(--object-ids "$PIPELINE_OBJECT_IDS")
    fi
    cmd+=(--workers "$workers")
    "${cmd[@]}"
  ) >"$log_path" 2>&1
  log "[$label steps $steps_csv] done"
}

run_sharded_python() {
  local step_label="$1"
  local name="$2"
  local script_name="$3"
  local requires_numba="$4"
  local supports_report="$5"
  shift 5
  local -a extra_args=("$@")
  local world_size="$SHARD_WORLD_SIZE"
  local -a pids=()
  local -a logs=()
  local rank gpu log_path report_path pid rc

  log "[step $step_label $name] start world_size=$world_size shard_gpus=$SHARD_GPUS workers_cuda=$WORKERS_CUDA"
  for rank in "${!SHARD_GPU_IDS[@]}"; do
    gpu="${SHARD_GPU_IDS[$rank]}"
    log_path="$LOG_DIR/step${step_label}_${name}_rank${rank}.log"
    report_path="$LOG_DIR/step${step_label}_${name}_rank${rank}.json"
    logs+=("$log_path")
    (
      cd "$TOOLKIT_ROOT"
      prepare_python_env_for_toolkit
      export_gpu_env_for "$gpu"
      if [[ "$requires_numba" == "yes" ]]; then
        export NUMBA_DISABLE_JIT=1
      fi
      cmd=(
        python3 "$script_name"
        --config "$RUNTIME_CONFIG"
        --rank "$rank"
        --world-size "$world_size"
      )
      if [[ "$PIPELINE_OBJECT_IDS" != "all" ]]; then
        cmd+=("--object-ids" "$PIPELINE_OBJECT_IDS")
      fi
      if [[ "$supports_report" == "yes" ]]; then
        cmd+=(--report-path "$report_path")
        echo "[report] $report_path"
      fi
      cmd+=("${extra_args[@]}")
      "${cmd[@]}"
    ) >"$log_path" 2>&1 &
    pid="$!"
    pids+=("$pid")
  done

  rc=0
  for rank in "${!pids[@]}"; do
    if ! wait "${pids[$rank]}"; then
      rc=1
      echo "[fatal] step $step_label $name rank=$rank failed; see ${logs[$rank]}" >&2
    fi
  done
  log "[step $step_label $name] done rc=$rc"
  return "$rc"
}

run_cloud_pipeline() {
  local CPU_PRE_STEPS RENDER_STEPS CPU_STEP3 CPU_STEP4 FEATURE_STEPS

  CPU_PRE_STEPS="$(filter_steps "1")"
  RENDER_STEPS="$(filter_steps "2")"
  CPU_STEP3="$(filter_steps "3")"
  CPU_STEP4="$(filter_steps "4")"
  FEATURE_STEPS="$(filter_steps "5")"

  [[ -n "$CPU_PRE_STEPS" ]] && run_cloud_steps "cpu-pre" "$CPU_PRE_STEPS" "$WORKERS_CPU"
  [[ -n "$RENDER_STEPS" ]] && run_cloud_steps "render" "$RENDER_STEPS" "$WORKERS_RENDER" "$RENDER_GPU"
  [[ -n "$CPU_STEP3" ]] && run_cloud_steps "cpu-step3" "$CPU_STEP3" "$WORKERS_CPU"
  [[ -n "$CPU_STEP4" ]] && run_cloud_steps "cpu-step4" "$CPU_STEP4" "$WORKERS_CPU"
  [[ -n "$FEATURE_STEPS" ]] && run_sharded_python "05" "feature" "pipeline/05_extract_feature.py" "no" "no"

  [[ -n "${SELECTED_STEPS[6]:-}" ]] && run_cloud_steps "global-step6" "6" "$WORKERS_CPU"
  [[ -n "${SELECTED_STEPS[7]:-}" ]] && run_cloud_steps "global-step7" "7" "$WORKERS_CPU"
  if [[ -n "${SELECTED_STEPS[8]:-}" ]]; then
    run_sharded_python "08" "ss_per_part" "pipeline/08_encode_ss_latents_per_part.py" "yes" "yes" --continue-on-error
    run_sharded_python "08" "ss_overall" "utils/encode_ss_latents_expanded.py" "yes" "no"
  fi
  [[ -n "${SELECTED_STEPS[9]:-}" ]] && run_cloud_steps "global-mid" "9" "$WORKERS_CPU"
  [[ -n "${SELECTED_STEPS[10]:-}" ]] && run_sharded_python "10" "ss_decode" "pipeline/10_decode_ss_latents.py" "yes" "yes" --continue-on-error
  [[ -n "${SELECTED_STEPS[11]:-}" ]] && run_cloud_steps "global-preview" "11" "$WORKERS_CPU"
  [[ -n "${SELECTED_STEPS[12]:-}" ]] && run_sharded_python "12" "part_slat" "pipeline/12_encode_part_synthesis_slat.py" "yes" "yes" --continue-on-error
  [[ -n "${SELECTED_STEPS[13]:-}" ]] && run_cloud_steps "global-final" "13" "$WORKERS_CPU"
  return 0
}

is_tos_uri() {
  [[ "$1" == tos://* ]]
}

sync_tos_source() {
  local source_uri="$1"

  command -v "$TOSUTIL" >/dev/null 2>&1 || {
    echo "[fatal] '$TOSUTIL' not found in PATH; cannot pull RealAppliance source from TOS." >&2
    exit 127
  }
  mkdir -p "$SOURCE_CACHE"
  if [[ -d "$SOURCE_CACHE/models/001" || -d "$SOURCE_CACHE/model/001" ]]; then
    echo "[source] using existing local TOS cache: $SOURCE_CACHE"
    return 0
  fi

  echo "[source] pulling RealAppliance source from TOS"
  echo "[source] $source_uri/ -> $SOURCE_CACHE/"
  "$TOSUTIL" cp -r "$source_uri/" "$SOURCE_CACHE/"
}

resolve_local_source_root() {
  local root="$1"
  local candidate

  if [[ -d "$root/models/001" || -d "$root/model/001" ]]; then
    printf '%s\n' "$root"
    return 0
  fi

  candidate="$(
    find "$root" -maxdepth 3 -type d \( -path "*/models/001" -o -path "*/model/001" \) 2>/dev/null |
      head -n 1 || true
  )"
  if [[ -n "$candidate" ]]; then
    dirname "$(dirname "$candidate")"
    return 0
  fi

  printf '%s\n' "$root"
}

write_runtime_config() {
  mkdir -p "$(dirname "$RUNTIME_CONFIG")"
  python3 - "$CONFIG" "$RUNTIME_CONFIG" "$DATA_ROOT" "$PROJECT_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, out_path, data_root, project_root = sys.argv[1:5]

with open(base_config, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["data_root"] = str(Path(data_root).resolve())
cfg.setdefault("render", {})["blender"] = f"{project_root}/software/blender-4.4.0-linux-x64/blender"
cfg["render"]["rgb_engine"] = "BLENDER_EEVEE_NEXT"
cfg.setdefault("feature", {})["dinov2_repo"] = f"{project_root}/pretrained/dinov2"
cfg.setdefault("feature", {})["torch_hub_dir"] = f"{project_root}/pretrained/torch_hub"
cfg.setdefault("trellis", {})["root"] = f"{project_root}/TRELLIS-arts"
cfg["trellis"]["ss_encoder"] = f"{project_root}/pretrained/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
cfg["trellis"]["ss_decoder"] = f"{project_root}/pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
cfg["trellis"]["slat_encoder"] = f"{project_root}/pretrained/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
cfg.setdefault("vlm", {})["image_prefix"] = str(Path(project_root).resolve())

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(f"[config] wrote {out_path}")
print(f"[config] data_root={cfg['data_root']}")
print("[config] rgb_engine=BLENDER_EEVEE_NEXT")
PY
}

if [[ "$DRY_RUN" == "1" ]]; then
  log "RealAppliance cloud ingest dry-run"
  log "data_root=$DATA_ROOT"
  log "source_root=$SRC_ROOT"
  log "source_cache=$SOURCE_CACHE"
  log "object_ids=$OBJECT_IDS pipeline_object_ids=$PIPELINE_OBJECT_IDS"
  log "profile=$PROFILE steps=${STEPS:-<profile>}"
  log "workers_cpu=$WORKERS_CPU workers_render=$WORKERS_RENDER workers_cuda=$WORKERS_CUDA"
  log "gpu=$GPU render_gpu=$RENDER_GPU shard_gpus=$SHARD_GPUS world_size=$SHARD_WORLD_SIZE"
  log "rgb_engine=BLENDER_EEVEE_NEXT"
  log "log_dir=$LOG_DIR runtime_config=$RUNTIME_CONFIG"
  [[ -n "$(filter_steps "1")" ]] && log "would run cpu-pre steps $(filter_steps "1") workers=$WORKERS_CPU"
  [[ -n "$(filter_steps "2")" ]] && log "would run render steps $(filter_steps "2") workers=$WORKERS_RENDER gpu=$RENDER_GPU"
  [[ -n "$(filter_steps "3")" ]] && log "would run cpu-step3 steps $(filter_steps "3") workers=$WORKERS_CPU"
  [[ -n "$(filter_steps "4")" ]] && log "would run cpu-step4 steps $(filter_steps "4") workers=$WORKERS_CPU"
  [[ -n "$(filter_steps "5")" ]] && log "would run step05 feature shard_gpus=$SHARD_GPUS"
  [[ -n "${SELECTED_STEPS[6]:-}" ]] && log "would run global-step6 workers=$WORKERS_CPU"
  [[ -n "${SELECTED_STEPS[7]:-}" ]] && log "would run global-step7 workers=$WORKERS_CPU"
  [[ -n "${SELECTED_STEPS[8]:-}" ]] && log "would run step08 ss_per_part + ss_overall shard_gpus=$SHARD_GPUS"
  [[ -n "${SELECTED_STEPS[9]:-}" ]] && log "would run global-mid step9 workers=$WORKERS_CPU"
  [[ -n "${SELECTED_STEPS[10]:-}" ]] && log "would run step10 ss_decode shard_gpus=$SHARD_GPUS"
  [[ -n "${SELECTED_STEPS[11]:-}" ]] && log "would run global-preview step11 workers=$WORKERS_CPU"
  [[ -n "${SELECTED_STEPS[12]:-}" ]] && log "would run step12 part_slat shard_gpus=$SHARD_GPUS"
  [[ -n "${SELECTED_STEPS[13]:-}" ]] && log "would run global-final step13 workers=$WORKERS_CPU"
  exit 0
fi

prepare_python_env_for_toolkit

if is_tos_uri "$SRC_ROOT"; then
  sync_tos_source "$SRC_ROOT"
  SRC_ROOT="$SOURCE_CACHE"
fi
SRC_ROOT="$(resolve_local_source_root "$SRC_ROOT")"
write_runtime_config

if [[ -d "$SRC_ROOT/models" ]]; then
  SOURCE_MODELS_ROOT="$SRC_ROOT/models"
elif [[ -d "$SRC_ROOT/model" ]]; then
  SOURCE_MODELS_ROOT="$SRC_ROOT/model"
else
  SOURCE_MODELS_ROOT="$SRC_ROOT/models"
fi

if [[ "$SKIP_CONVERT" != "1" ]]; then
  if [[ ! -d "$SOURCE_MODELS_ROOT/001" ]]; then
    echo "[fatal] missing RealAppliance source data at $SOURCE_MODELS_ROOT/001. Run:" >&2
    echo "  mkdir -p '$DATA_ROOT'" >&2
    echo "  git clone https://github.com/gaoyz1235/RealAppliance '$SRC_ROOT'" >&2
    exit 4
  fi
  require_usd_core_for_convert
  echo "[ingest] Phase 1: converter source_ids=$OBJECT_IDS source_models=$SOURCE_MODELS_ROOT"
  pushd "$TOOLKIT_ROOT" >/dev/null
  CONVERT_ARGS=(
    --src-root "$SRC_ROOT"
    --dst-root "$DATA_ROOT"
    --object-ids "$OBJECT_IDS"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    CONVERT_ARGS+=(--overwrite)
  fi
  PYTHONPATH="$TOOLKIT_ROOT:${PYTHONPATH:-}" python3 -m converters.convert_realappliance.convert "${CONVERT_ARGS[@]}"
  popd >/dev/null
else
  require_existing_raw_for_skip_convert
  echo "[ingest] Phase 1 skipped (--skip-convert)"
fi

log "[ingest] Phase 2: cloud pipeline profile=$PROFILE object_ids=$PIPELINE_OBJECT_IDS"
log "[ingest] workers_cpu=$WORKERS_CPU workers_render=$WORKERS_RENDER workers_cuda=$WORKERS_CUDA render_gpu=$RENDER_GPU shard_gpus=$SHARD_GPUS rgb_engine=BLENDER_EEVEE_NEXT logs=$LOG_DIR"
run_cloud_pipeline

echo "[ingest] done"
