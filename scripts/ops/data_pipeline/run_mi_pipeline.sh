#!/usr/bin/env bash
# ============================================================
# run_mi_pipeline.sh — One-shot launcher for the Xiaomi delivery.
#
# Takes raw Xiaomi articulated-object dumps (under {SRC}/{uuid}/{uuid}.json
# + objs/ + ...), converts them into PhysX-Mobility raw/ layout, and runs
# the full dataset_toolkits pipeline 1→11 with all known workarounds:
#
#   1. converters/convert_mi2physx/convert.py
#        Xiaomi → PhysX (flat raw/, P→B, C-angle deg→pi, .convex.stl rename)
#   2. run_pipeline.sh --steps 1,2,3,4,5
#   3. utils/validate_dataset.py --steps render,voxel
#        Produces the validator report that step 6 needs.
#   4. pipeline/06_build_manifest.py --validator-report <path>
#        (Re)build manifest with summary.validator_status = PASS.
#   5. run_pipeline.sh --steps 7,8
#   6. utils/encode_ss_latents_expanded.py
#        Local gap-fill: writes ss_latents_expanded/{id}/angle_{k}/latent.npz
#        which step 9/10 need but no upstream step produces. See
#        docs/data_comparison.md §upstream pipeline gap.
#   7. run_pipeline.sh --steps 9,10,11
#   8. sed-inject camera.up.set(0,0,1) into the generated HTML so the Three.js
#      voxel viewer renders Z-up correctly. See docs/data_comparison.md
#      §流程陷阱 #4.
#
# Usage:
#   bash scripts/ops/data_pipeline/run_mi_pipeline.sh \
#       --src /media/mi/E2AB72E695F22B61/data_sda/aticulated_data/Mi \
#       --dst /home/mi/jzh/AAAI2027/arts-reconstruction/data/Mi-PhysX \
#       [--object-ids uuid1,uuid2,...] \
#       [--workers N] \
#       [--serve] [--port 8000]
#
# Required:
#   --src PATH         Xiaomi root containing per-uuid subdirectories.
#   --dst PATH         Target data_root (will be wiped if --wipe is given).
#
# Optional:
#   --config PATH      dataset_toolkits YAML (default: submodules/dataset_toolkits/configs/Mi.yaml)
#   --object-ids CSV   Subset of uuids; default: all under --src.
#   --workers N        Worker count for render/voxel steps (default: 1).
#   --wipe             rm -rf --dst before converting (clean slate).
#   --serve            Start python -m http.server on the preview after step 11.
#   --port N           HTTP port (default: 8000). Implies --serve.
#   --skip-convert     Skip the convert step (re-run pipeline on already-converted data).
#   -h | --help
#
# Exit codes:
#   0 success
#   2 bad args
#   3 missing prerequisites (Blender / weights / conda env)
#   4 pipeline failure
# ============================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Locate project root + submodule
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
TOOLKIT_DIR="$PROJECT_ROOT/submodules/dataset_toolkits"
CONVERT_PY="$TOOLKIT_DIR/converters/convert_mi2physx/convert.py"
VALIDATE_PY="$TOOLKIT_DIR/utils/validate_dataset.py"
GAPFILL_PY="$TOOLKIT_DIR/utils/encode_ss_latents_expanded.py"
STEP6_PY="$TOOLKIT_DIR/pipeline/06_build_manifest.py"
RUN_PIPELINE="$TOOLKIT_DIR/run_pipeline.sh"

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
SRC=""
DST=""
CONFIG="$TOOLKIT_DIR/configs/Mi.yaml"
OBJECT_IDS=""
WORKERS=1
WIPE=0
SERVE=0
PORT=8000
SKIP_CONVERT=0

usage() { sed -n '2,55p' "$0"; }

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --src)         SRC="$2";        shift 2 ;;
    --dst)         DST="$2";        shift 2 ;;
    --config)      CONFIG="$2";     shift 2 ;;
    --object-ids)  OBJECT_IDS="$2"; shift 2 ;;
    --workers)     WORKERS="$2";    shift 2 ;;
    --wipe)        WIPE=1;          shift   ;;
    --serve)       SERVE=1;         shift   ;;
    --port)        PORT="$2"; SERVE=1; shift 2 ;;
    --skip-convert) SKIP_CONVERT=1; shift   ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

[ -n "$SRC" ] || { echo "[error] --src is required" >&2; exit 2; }
[ -n "$DST" ] || { echo "[error] --dst is required" >&2; exit 2; }
case "$SRC" in /*) : ;; *) echo "[error] --src must be absolute" >&2; exit 2 ;; esac
case "$DST" in /*) : ;; *) echo "[error] --dst must be absolute" >&2; exit 2 ;; esac
[ -d "$SRC" ] || { echo "[error] --src not found: $SRC" >&2; exit 2; }
[ -f "$CONFIG" ] || { echo "[error] config not found: $CONFIG" >&2; exit 2; }

# -----------------------------------------------------------------------------
# Prerequisite checks
# -----------------------------------------------------------------------------
[ -f "$CONVERT_PY" ]    || { echo "[error] missing $CONVERT_PY" >&2; exit 3; }
[ -f "$VALIDATE_PY" ]   || { echo "[error] missing $VALIDATE_PY" >&2; exit 3; }
[ -f "$GAPFILL_PY" ]    || { echo "[error] missing $GAPFILL_PY (upstream gap-fill)" >&2; exit 3; }
[ -f "$STEP6_PY" ]      || { echo "[error] missing $STEP6_PY" >&2; exit 3; }
[ -f "$RUN_PIPELINE" ]  || { echo "[error] missing $RUN_PIPELINE" >&2; exit 3; }

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "[error] CONDA_PREFIX is unset; activate the arts-gen conda env first:" >&2
  echo "        conda activate arts-gen" >&2
  exit 3
fi

# Tell upstream run_pipeline.sh we're "in" the dataset_toolkits env (we are
# actually in arts-gen but the python interpreter resolved by CONDA_PREFIX
# still points at arts-gen which has all needed deps).
export CONDA_DEFAULT_ENV=dataset_toolkits

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
banner() {
  local msg="$1"
  echo ""
  echo "============================================================"
  echo "  $msg"
  echo "============================================================"
}

# -----------------------------------------------------------------------------
# Optional wipe
# -----------------------------------------------------------------------------
if [ "$WIPE" -eq 1 ] && [ -d "$DST" ]; then
  banner "[wipe] removing $DST"
  rm -rf "$DST"
fi

# -----------------------------------------------------------------------------
# Step 0: convert Xiaomi → PhysX raw/
# -----------------------------------------------------------------------------
if [ "$SKIP_CONVERT" -eq 0 ]; then
  banner "[0/8] convert Xiaomi → PhysX (data_root: $DST)"
  CONV_ARGS=(--src "$SRC" --dst "$DST")
  [ -n "$OBJECT_IDS" ] && CONV_ARGS+=(--object-ids "$OBJECT_IDS")
  python "$CONVERT_PY" "${CONV_ARGS[@]}"
else
  banner "[0/8] convert step skipped (--skip-convert)"
fi

# -----------------------------------------------------------------------------
# Pipeline calls all share these args
# -----------------------------------------------------------------------------
PIPE_ARGS=(--config "$CONFIG" --workers "$WORKERS")
[ -n "$OBJECT_IDS" ] && PIPE_ARGS+=(--object-ids "$OBJECT_IDS")

run_pipeline_steps() {
  local steps="$1"
  ( cd "$TOOLKIT_DIR" && bash run_pipeline.sh "${PIPE_ARGS[@]}" --steps "$steps" )
}

# -----------------------------------------------------------------------------
# Step 1-5: joint_transform / render / bbox / voxel / dinov2
# -----------------------------------------------------------------------------
banner "[1/8] dataset_toolkits steps 1,2,3,4,5"
run_pipeline_steps 1,2,3,4,5

# -----------------------------------------------------------------------------
# Validator (must precede step 6 so manifest.summary.validator_status=PASS)
# -----------------------------------------------------------------------------
VALIDATOR_REPORT="$DST/manifests/validator_report.json"
banner "[2/8] validator (render+voxel) → $VALIDATOR_REPORT"
mkdir -p "$DST/manifests"
( cd "$TOOLKIT_DIR" && python "$VALIDATE_PY" \
    --config "$CONFIG" \
    --steps render,voxel \
    --report-path "$VALIDATOR_REPORT" )

# -----------------------------------------------------------------------------
# Step 6 manifest with explicit --validator-report
# -----------------------------------------------------------------------------
banner "[3/8] step 6 manifest (with validator report → PASS status)"
( cd "$TOOLKIT_DIR" && python "$STEP6_PY" \
    --config "$CONFIG" \
    --validator-report "$VALIDATOR_REPORT" )

# -----------------------------------------------------------------------------
# Step 7-8: VLM JSONL + per-part SS latents
# -----------------------------------------------------------------------------
banner "[4/8] dataset_toolkits steps 7,8"
run_pipeline_steps 7,8

# -----------------------------------------------------------------------------
# Local gap-fill: encode whole-object SS latents
# -----------------------------------------------------------------------------
banner "[5/8] local gap-fill: encode ss_latents_expanded"
( cd "$TOOLKIT_DIR" && python "$GAPFILL_PY" --config "$CONFIG" )

# -----------------------------------------------------------------------------
# Step 9-11: part_completion manifest, decode, preview HTML
# -----------------------------------------------------------------------------
banner "[6/8] dataset_toolkits steps 9,10,11"
# Step 9 fails noisily if its outputs already exist; clear them so re-runs work.
PC_DIR="$DST/manifests/part_completion"
[ -d "$PC_DIR" ] && rm -f "$PC_DIR"/*.{json,jsonl} 2>/dev/null || true
run_pipeline_steps 9,10,11

# -----------------------------------------------------------------------------
# Z-up viewer patch on generated HTML
# -----------------------------------------------------------------------------
INDEX_HTML="$DST/preview/vlm_training/index.html"
if [ ! -f "$INDEX_HTML" ]; then
  echo "[error] step 11 did not produce $INDEX_HTML" >&2
  exit 4
fi

banner "[7/8] patch Three.js viewer to Z-up"
if grep -q 'camera.up.set(0,0,1)' "$INDEX_HTML"; then
  echo "  already patched"
else
  sed -i \
    's#camera=new THREE.PerspectiveCamera(45,1,.1,500);camera.position.set(128,96,128);#camera=new THREE.PerspectiveCamera(45,1,.1,500);camera.up.set(0,0,1);camera.position.set(96,-128,96);#' \
    "$INDEX_HTML"
  if grep -q 'camera.up.set(0,0,1)' "$INDEX_HTML"; then
    echo "  patched"
  else
    echo "  [warn] sed found no match; viewer may stay Y-up. Inspect $INDEX_HTML manually."
  fi
fi

# -----------------------------------------------------------------------------
# Optional: serve preview
# -----------------------------------------------------------------------------
banner "[8/8] done"
echo "  data_root : $DST"
echo "  preview   : $INDEX_HTML"

if [ "$SERVE" -eq 1 ]; then
  echo "  serving at http://localhost:$PORT/index.html (Ctrl+C to stop)"
  echo ""
  ( cd "$DST/preview/vlm_training" && python3 -m http.server "$PORT" )
else
  echo ""
  echo "  to view in browser:"
  echo "      cd $DST/preview/vlm_training && python3 -m http.server 8000"
  echo "      open http://localhost:8000/index.html"
fi
