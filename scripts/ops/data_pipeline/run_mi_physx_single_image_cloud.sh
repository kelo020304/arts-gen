#!/usr/bin/env bash
# ============================================================
# run_mi_physx_single_image_cloud.sh
#
# Wrapper for the NEW dataset_toolkits pipeline (commit 8f9b957+) which
# produces 16-view "part_complete" single-image render data — applied to
# the Mi-PhysX dataset (Xiaomi-supplied UUID-keyed articulated objects).
# Same data schema as PhysX-Mobility (raw/{urdf,finaljson,partseg}/...),
# just UUID object_ids instead of integer ones.
#
# Uses submodules/dataset_toolkits_single_image/ (manually cloned, NOT
# a git submodule — coexists with the older dataset_toolkits submodule).
#
# Step 5 (render) default set is `part_complete` which writes 16-view RGB
# + valid-part masks + remaining mask per object.
#
# Usage:
#   bash scripts/ops/data_pipeline/run_mi_physx_single_image_cloud.sh \
#     --data-root /robot/data-lab/.../Mi-PhysX \
#     [--profile default] \
#     [--object-ids 01568948-8fbb-5ce0-8d92-0d658902db64,01c6cde4-a5b8-5fb7-b522-92140fb38921] \
#     [--workers 4]
#
# Required upfront on a fresh container:
#   bash scripts/ops/setup/setup_blender_headless_gpu.sh
# (installs xvfb + libnvidia-gpucomp + Vulkan ICD; same as old pipeline)
# ============================================================

set -euo pipefail

# ----- Headless GPU graphics setup (same as old script) -----
# Xvfb + NVIDIA Vulkan; required for Blender EEVEE_NEXT used by new pipeline.
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
if ! pgrep -x Xvfb >/dev/null 2>&1; then
  Xvfb :99 -screen 0 1024x768x24 &
  sleep 1
fi
export DISPLAY=:99

# ----- Path constants -----
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
TOOLKIT_DIR="$PROJECT_ROOT/submodules/dataset_toolkits_single_image"
BASE_CONFIG="$TOOLKIT_DIR/configs/Mi-PhysX.yaml"
GENERATED_CONFIG="$TOOLKIT_DIR/configs/Mi-PhysX.cloud.generated.yaml"

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
STEPS=""        # overrides --profile + auto-phasing if set
WORKERS=""      # if set, overrides per-phase defaults (single value applied to all)
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
  sed -n '2,30p' "$0"
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root)        DATA_ROOT="$2";        shift 2 ;;
    --profile)          PROFILE="$2";          shift 2 ;;
    --object-ids)       OBJECT_IDS="$2";       shift 2 ;;
    --workers)          WORKERS="$2";          shift 2 ;;
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

# ----- Conda env hint for upstream's strict check -----
# Upstream run_pipeline.sh asserts CONDA_DEFAULT_ENV=dataset_toolkits. We use
# arts-gen which has the same deps. Override the var so the check passes.
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "[error] CONDA_PREFIX unset; activate the arts-gen conda env first:" >&2
  echo "        conda activate arts-gen" >&2
  exit 3
fi
export CONDA_DEFAULT_ENV=dataset_toolkits

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
# If --steps or --workers explicit, run as a single pass (user knows what they
# want). Otherwise auto-phase: 1-4 with WORKERS_CPU, 5 with WORKERS_RENDER,
# 6-11 with WORKERS_CUDA — different steps have very different CPU/GPU profiles
# and one global --workers value is always wrong for at least one phase.
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
  # Explicit --steps: single pass. --workers (if set) overrides per-phase defaults.
  W="${WORKERS:-$WORKERS_CPU}"
  run_phase "explicit" "$STEPS" "$W"
elif [ -n "$WORKERS" ]; then
  # Explicit --workers but no --steps: single pass with profile.
  CMD=( bash "$TOOLKIT_DIR/run_pipeline.sh"
        --config "$GENERATED_CONFIG"
        --workers "$WORKERS"
        --profile "$PROFILE" )
  [ -n "$OBJECT_IDS" ] && CMD+=( --object-ids "$OBJECT_IDS" )
  echo "[run] ${CMD[*]}"
  exec "${CMD[@]}"
else
  # Auto-phase: 3 passes with per-phase optimal workers
  echo "[auto-phase] using WORKERS_CPU=$WORKERS_CPU WORKERS_RENDER=$WORKERS_RENDER WORKERS_CUDA=$WORKERS_CUDA"
  run_phase "cpu-prep"    "1,2,3,4"        "$WORKERS_CPU"
  run_phase "render"      "5"              "$WORKERS_RENDER"
  run_phase "cuda-encode" "6,7,8,9,10,11"  "$WORKERS_CUDA"
fi
