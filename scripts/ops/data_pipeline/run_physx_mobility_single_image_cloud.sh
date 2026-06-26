#!/usr/bin/env bash
# ============================================================
# run_physx_mobility_single_image_cloud.sh
#
# Wrapper for the NEW dataset_toolkits pipeline (commit 8f9b957+) which
# produces 16-view "part_complete" single-image render data — separate
# from the old 4-quadrant pipeline (run_physx_mobility_cloud.sh).
#
# Uses submodules/dataset_toolkits_single_image/ (manually cloned, NOT
# a git submodule — coexists with the older dataset_toolkits submodule).
#
# Step 5 (render) default set is `part_complete` which writes 16-view RGB
# + valid-part masks + remaining mask per object. This matches the
# "single image per view" format you want for the new dev machine.
#
# Usage:
#   bash scripts/ops/data_pipeline/run_physx_mobility_single_image_cloud.sh \
#     --data-root /robot/data-lab/.../PhysX-Mobility-single-image \
#     [--profile default] \
#     [--object-ids 100013,100015] \
#     [--workers-cpu 60]
#     [--workers-render 1]
#     [--workers-cuda 4]
#
# Required upfront on a fresh container:
#   bash scripts/ops/setup/setup_blender_headless_gpu.sh
# (installs xvfb + libnvidia-gpucomp + Vulkan ICD; same as old pipeline)
# ============================================================

set -euo pipefail

# ----- Path constants -----
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
TOOLKIT_DIR="$PROJECT_ROOT/submodules/dataset_toolkits_single_image"
BASE_CONFIG="$TOOLKIT_DIR/configs/PhysX-Mobility.yaml"
GENERATED_CONFIG="$TOOLKIT_DIR/configs/PhysX-Mobility.cloud.generated.yaml"

[ -d "$TOOLKIT_DIR" ] || {
  echo "[error] $TOOLKIT_DIR not found." >&2
  echo "Clone first:" >&2
  echo "  cd submodules && git clone git@github.com:mlpchenxl/dataset_toolkits.git dataset_toolkits_single_image" >&2
  exit 2
}
[ -f "$BASE_CONFIG" ] || { echo "[error] base config $BASE_CONFIG missing" >&2; exit 2; }

# ----- Defaults / args -----
DATA_ROOT=""
PROFILE="default"
OBJECT_IDS=""
RENDER_SET=""   # passed to step 5 if non-empty (part_complete / full150 / parts150)
STEPS=""        # runs an explicit step set if provided; otherwise auto-phases
# Per-phase worker defaults. Rationale:
# - CPU phase (steps 1,2,3,4,9,10,11): pure NumPy / JSON / mesh-bbox, IO + CPU bound,
#   benefits from many parallel workers, no GPU pressure.
# - Render phase (step 5): Blender EEVEE on single GPU; multi-worker = GPU OOM risk
#   on 24GB cards with 16-view scenes. Stay conservative.
# - CUDA phase (steps 6,7,8): each worker loads DINOv2/TRELLIS encoder (~1-2 GB GPU
#   each); 4 workers on 24GB GPU fits comfortably.
WORKERS_CPU=16
WORKERS_RENDER=1
WORKERS_CUDA=4

usage() {
  cat <<EOF
Run PhysX-Mobility single-image preprocessing on cloud.

Typical usage:
  bash scripts/ops/data_pipeline/run_physx_mobility_single_image_cloud.sh \\
    --data-root /robot/data-lab/arts-gen-data/data/PhysX-Mobility-single-image-0512 \\
    --workers-cpu 60 \\
    --workers-render 1 \\
    --workers-cuda 4

Options:
  --data-root PATH      Output data root. Required.
  --profile NAME        Pipeline profile when --steps is not set. Default: $PROFILE.
  --object-ids CSV      Optional object subset.
  --workers N          DEPRECATED alias for --workers-cpu; render still uses --workers-render.
  --workers-cpu N      Workers for CPU phases. Default: $WORKERS_CPU.
  --workers-render N   Workers for Blender Step 5. Default: $WORKERS_RENDER.
  --workers-cuda N     Workers for CUDA phases. Default: $WORKERS_CUDA.
  --render-set NAME    Informational only; Step 5 default is part_complete.
  --steps CSV          Explicit step set. Still uses phase-specific workers.
  -h, --help           Show this help.
EOF
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root)        DATA_ROOT="$2";        shift 2 ;;
    --profile)          PROFILE="$2";          shift 2 ;;
    --object-ids)       OBJECT_IDS="$2";       shift 2 ;;
    --workers)          WORKERS_CPU="$2";      shift 2 ;;
    --workers-cpu)      WORKERS_CPU="$2";      shift 2 ;;
    --workers-render)   WORKERS_RENDER="$2";   shift 2 ;;
    --workers-cuda)     WORKERS_CUDA="$2";     shift 2 ;;
    --render-set)       RENDER_SET="$2";       shift 2 ;;
    --steps)            STEPS="$2";            shift 2 ;;
    -h|--help)          usage ;;
    *) echo "[error] unknown arg: $1" >&2; usage ;;
  esac
done

[ -n "$DATA_ROOT" ] || { echo "[error] --data-root is required" >&2; exit 2; }
case "$DATA_ROOT" in /*) : ;; *) echo "[error] --data-root must be absolute" >&2; exit 2 ;; esac

start_headless_gpu() {
  # Xvfb + NVIDIA Vulkan; required for Blender EEVEE_NEXT used by Step 5.
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __VK_LAYER_NV_optimus=NVIDIA_only
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

# ----- Conda env hint for upstream's strict check -----
# Upstream run_pipeline.sh asserts CONDA_DEFAULT_ENV=dataset_toolkits. We use
# arts-gen which has the same deps. Override the var so the check passes.
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "[error] CONDA_PREFIX unset; activate the arts-gen conda env first:" >&2
  echo "        conda activate arts-gen" >&2
  exit 3
fi
export CONDA_DEFAULT_ENV=dataset_toolkits

start_headless_gpu

# ----- Generate config YAML with our data_root -----
# Inline Python instead of yq/sed: handle nested YAML cleanly + preserve
# the long static_objects list from the base config.
python3 - "$BASE_CONFIG" "$GENERATED_CONFIG" "$DATA_ROOT" "$PROJECT_ROOT" <<'PY'
import sys, yaml
from pathlib import Path

base_config, out_path, data_root, project_root = sys.argv[1:5]

with open(base_config, "r") as fh:
    cfg = yaml.safe_load(fh)

cfg["data_root"] = data_root

# Point Blender + DINOv2 + TRELLIS weights at the project's vendored copies
# rather than the absolute paths shipped in the base config (which target a
# different developer's machine).
cfg["render"]["blender"] = f"{project_root}/software/blender-4.4.0-linux-x64/blender"
cfg["feature"]["dinov2_repo"] = f"{project_root}/pretrained/dinov2"
cfg["feature"]["torch_hub_dir"] = f"{project_root}/pretrained/torch_hub"
cfg["trellis"]["root"] = f"{project_root}/TRELLIS-arts"
cfg["trellis"]["ss_encoder"] = f"{project_root}/pretrained/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
cfg["trellis"]["ss_decoder"] = f"{project_root}/pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
cfg["trellis"]["slat_encoder"] = f"{project_root}/pretrained/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
cfg["vlm"]["image_prefix"] = project_root

with open(out_path, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)

print(f"[config] wrote {out_path}")
print(f"[config] data_root={data_root}")
PY

# ----- Override step 5 render-set if requested (informational only) -----
if [ -n "$RENDER_SET" ]; then
  echo "[warn] --render-set is not forwarded to run_pipeline.sh." >&2
  echo "       For non-default sets, run 05_render.py directly after this:" >&2
  echo "         python $TOOLKIT_DIR/pipeline/05_render.py --config $GENERATED_CONFIG --sets $RENDER_SET" >&2
fi

# ----- Execute pipeline -----
# Always keep phase-specific workers. A single global worker count is unsafe for
# Step 5 because it can launch many concurrent Blender processes on one GPU.
run_phase() {
  local phase_name="$1"
  local phase_steps="$2"
  local phase_workers="$3"
  echo ""
  echo "============================================================"
  echo "  Phase '$phase_name': steps $phase_steps, workers $phase_workers"
  echo "============================================================"
  local cmd=( bash "$TOOLKIT_DIR/run_pipeline.sh"
              --config "$GENERATED_CONFIG"
              --workers "$phase_workers"
              --steps "$phase_steps" )
  [ -n "$OBJECT_IDS" ] && cmd+=( --object-ids "$OBJECT_IDS" )
  echo "[run] ${cmd[*]}"
  "${cmd[@]}"
}

if [ -n "$STEPS" ]; then
  echo "[explicit-steps] using WORKERS_CPU=$WORKERS_CPU WORKERS_RENDER=$WORKERS_RENDER WORKERS_CUDA=$WORKERS_CUDA"
  IFS=',' read -r -a REQUESTED_STEPS <<< "$STEPS"
  for step in "${REQUESTED_STEPS[@]}"; do
    case "$step" in
      1|2|3|4|9|10|11) run_phase "explicit-cpu-step-$step" "$step" "$WORKERS_CPU" ;;
      5) run_phase "explicit-render" "5" "$WORKERS_RENDER" ;;
      6|7|8) run_phase "explicit-cuda-step-$step" "$step" "$WORKERS_CUDA" ;;
      *)
        echo "[error] unsupported --steps item for single-image launcher: $step" >&2
        exit 2
        ;;
    esac
  done
else
  # Auto-phase: 3 passes with per-phase optimal workers
  echo "[auto-phase] using WORKERS_CPU=$WORKERS_CPU WORKERS_RENDER=$WORKERS_RENDER WORKERS_CUDA=$WORKERS_CUDA"
  run_phase "cpu-prep"    "1,2,3,4"        "$WORKERS_CPU"
  run_phase "render"      "5"              "$WORKERS_RENDER"
  run_phase "cuda-encode" "6,7,8,9,10,11"  "$WORKERS_CUDA"
fi
