#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_CONDA_ENV="dataset_toolkits"
PYTHON_BIN=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --config <yaml_path> [--profile <name> | --steps <1,2,3,...>] [--object-ids <id1,id2,...>] [--workers <n>]

Options:
  --config <yaml_path>       Required. Path to pipeline config YAML.
                             Run from the single official conda env: dataset_toolkits.
  --profile <name>           Optional pipeline profile when --steps is not provided.
                             default/full/stable/base/preview-base:
                             1,2,3,4,5,6,7,8,9,10,11.
                             Default profile: default.
                             Step 5 uses pipeline/05_render.py default set:
                             part_complete only. It does not run full-object
                             150-view or valid-parts 150-view renders unless
                             those scripts are invoked directly.
  --steps <1,2,3,...>        Optional. Comma-separated step numbers to run.
                             Mutually exclusive with --profile; selects an explicit custom step set.
                             Steps 12 and 13 are development-only and never
                             run by default profiles.
  --object-ids <id1,id2,...> Optional. Comma-separated object IDs passed to per-object steps.
                             Not passed to step 4 full manifest.
  --workers <n>              Optional. Worker count passed to render/voxelize steps.
  -h, --help                 Show this help message.
EOF
}

CONFIG=""
PROFILE_ARG=""
STEPS_ARG=""
OBJECT_IDS=""
WORKERS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "Error: --config requires a value" >&2; usage; exit 1; }
      CONFIG="$2"
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || { echo "Error: --profile requires a value" >&2; usage; exit 1; }
      PROFILE_ARG="$2"
      shift 2
      ;;
    --steps)
      [[ $# -ge 2 ]] || { echo "Error: --steps requires a value" >&2; usage; exit 1; }
      STEPS_ARG="$2"
      shift 2
      ;;
    --object-ids)
      [[ $# -ge 2 ]] || { echo "Error: --object-ids requires a value" >&2; usage; exit 1; }
      OBJECT_IDS="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "Error: --workers requires a value" >&2; usage; exit 1; }
      WORKERS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required" >&2
  usage
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_CONDA_ENV" ]]; then
  echo "Error: activate the official conda env before running: conda activate ${EXPECTED_CONDA_ENV}" >&2
  echo "Current CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}" >&2
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  echo "Error: PYTHON override is not supported. Use the official ${EXPECTED_CONDA_ENV} env python3." >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: CONDA_PREFIX is unset; activate the official conda env before running." >&2
  exit 1
fi

if ! PYTHON_BIN="$(command -v python3)"; then
  echo "Error: python3 not found in PATH for conda env ${EXPECTED_CONDA_ENV}" >&2
  exit 1
fi

if [[ ! -x "${CONDA_PREFIX}/bin/python3" ]]; then
  echo "Error: expected conda env python3 is missing or not executable: ${CONDA_PREFIX}/bin/python3" >&2
  exit 1
fi

EXPECTED_PYTHON="$(readlink -f "${CONDA_PREFIX}/bin/python3")"
PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
if [[ "$PYTHON_BIN" != "$EXPECTED_PYTHON" ]]; then
  echo "Error: python3 must come from the active ${EXPECTED_CONDA_ENV} conda env." >&2
  echo "Expected: $EXPECTED_PYTHON" >&2
  echo "Actual:   $PYTHON_BIN" >&2
  exit 1
fi

if [[ -n "$PROFILE_ARG" && -n "$STEPS_ARG" ]]; then
  echo "Error: --profile and --steps are mutually exclusive. Use --steps for a custom explicit selection." >&2
  exit 1
fi

declare -A SELECTED_STEPS=()
if [[ -n "$STEPS_ARG" ]]; then
  IFS=',' read -r -a REQUESTED_STEPS <<< "$STEPS_ARG"
  [[ ${#REQUESTED_STEPS[@]} -gt 0 ]] || { echo "Error: --steps cannot be empty" >&2; exit 1; }

  for step in "${REQUESTED_STEPS[@]}"; do
    if [[ ! "$step" =~ ^([1-9]|1[0-3])$ ]]; then
      echo "Error: invalid step '$step'. Valid steps are 1-13." >&2
      exit 1
    fi
    SELECTED_STEPS["$step"]=1
  done
else
  PROFILE_ARG="${PROFILE_ARG:-default}"
  case "$PROFILE_ARG" in
    default|full|stable)
      for step in 1 2 3 4 5 6 7 8 9 10 11; do
        SELECTED_STEPS["$step"]=1
      done
      ;;
    base|preview-base)
      for step in 1 2 3 4 5 6 7 8 9 10 11; do
        SELECTED_STEPS["$step"]=1
      done
      ;;
    *)
      echo "Error: unknown profile '$PROFILE_ARG'. Valid profiles: default, full, stable, base, preview-base." >&2
      exit 1
      ;;
  esac
fi

SELECTED_COUNT=0
for step in 1 2 3 4 5 6 7 8 9 10 11 12 13; do
  if [[ -n "${SELECTED_STEPS[$step]:-}" ]]; then
    ((SELECTED_COUNT += 1))
  fi
done

CURRENT_STEP=""
CURRENT_SCRIPT=""
trap 'if [[ -n "$CURRENT_STEP" ]]; then echo "Error: step $CURRENT_STEP failed: $CURRENT_SCRIPT" >&2; fi' ERR

run_step() {
  local step_num="$1"
  local script_name="$2"
  local uses_workers="$3"
  local uses_object_ids="$4"
  local requires_numba_disable="${5:-no}"
  local -a extra_args=()
  local -a cmd

  if (($# > 5)); then
    shift 5
    extra_args=("$@")
  fi

  CURRENT_STEP="$step_num"
  CURRENT_SCRIPT="$script_name"

  echo "=== Step ${step_num}/13: ${script_name} ==="

  cmd=(
    "$PYTHON_BIN"
    "${SCRIPT_DIR}/${script_name}"
    --config "$CONFIG"
  )

  if [[ "$uses_object_ids" == "yes" && -n "$OBJECT_IDS" ]]; then
    cmd+=(--object-ids "$OBJECT_IDS")
  fi

  if [[ "$uses_workers" == "yes" && -n "$WORKERS" ]]; then
    cmd+=(--workers "$WORKERS")
  fi
  cmd+=("${extra_args[@]}")

  if [[ "$requires_numba_disable" == "yes" ]]; then
    NUMBA_DISABLE_JIT=1 "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
  echo "Step ${step_num} done"
}

for step in 1 2 3 4 5 6 7 8 9 10 11 12 13; do
  [[ -n "${SELECTED_STEPS[$step]:-}" ]] || continue

  case "$step" in
    1) run_step 1 "pipeline/01_joint_transformation.py" "no" "yes" ;;
    2) run_step 2 "pipeline/02_build_canonical_transforms.py" "no" "yes" ;;
    3) run_step 3 "pipeline/03_voxelize.py" "yes" "yes" ;;
    4) run_step 4 "pipeline/04_build_valid_parts_manifest.py" "no" "no" ;;
    5) run_step 5 "pipeline/05_render.py" "yes" "yes" ;;
    6) run_step 6 "pipeline/06_extract_feature.py" "no" "yes" ;;
    7) run_step 7 "pipeline/07_encode_ss_latents_per_part.py" "no" "yes" "yes" ;;
    8) run_step 8 "pipeline/08_decode_ss_latents.py" "no" "yes" "yes" ;;
    9) run_step 9 "pipeline/09_build_vlm_dataset_manifest.py" "no" "yes" ;;
    10) run_step 10 "pipeline/10_build_part_completion_manifest.py" "no" "yes" "no" --overwrite ;;
    11) run_step 11 "pipeline/11_web_preview.py" "no" "yes" ;;
    12) run_step 12 "pipeline/12_encode_part_synthesis_slat.py" "no" "yes" "yes" ;;
    13) run_step 13 "pipeline/13_build_part_synthesis_manifest.py" "no" "yes" ;;
  esac
done

CURRENT_STEP=""
echo "Pipeline completed: ${SELECTED_COUNT} steps"
