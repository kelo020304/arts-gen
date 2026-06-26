#!/usr/bin/env bash
# Run dataset_toolkits PhysX-Mobility pipeline on a cloud dev machine.
#
# This wrapper keeps the repo config as the source of truth, but generates a
# cloud-local config with the current checkout paths before running the toolkit.
#
# Typical cloud usage:
#   bash scripts/ops/data_pipeline/run_physx_mobility_cloud.sh --smoke 2 --workers 1 --rgb-engine CYCLES
#   bash scripts/ops/data_pipeline/run_physx_mobility_cloud.sh --full --workers 8
#
# Expected raw data layout:
#   <data_root>/raw/finaljson/*.json
#   <data_root>/raw/partseg/<object_id>/objs/
#
# Default data_root intentionally contains a literal "data" path segment because
# dataset_toolkits step 7 derives preview image paths from that segment.

set -euo pipefail

# Headless GPU graphics context for Blender EEVEE_NEXT (Xvfb + NVIDIA Vulkan).
# On VolcEngine 4090 ML containers /dev/dri is missing and nvidia-drm modeset
# is disabled, so EGL-surfaceless can't get a GPU context. Workaround: a
# virtual X display (Xvfb) lets Blender's startup logic find a "display",
# while actual rendering goes through NVIDIA Vulkan driver bypassing display.
# Requires libnvidia-gpucomp.so.550.144.03 at /usr/lib/x86_64-linux-gnu/
# (download from ml-platform.tos-vpc.cloud.vnet.com/rclone_tmp_dir/ if missing).
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
if ! pgrep -x Xvfb >/dev/null 2>&1; then
  Xvfb :99 -screen 0 1024x768x24 &
  sleep 1
fi
export DISPLAY=:99

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
TOOLKIT_DIR="$PROJECT_ROOT/submodules/dataset_toolkits"
BASE_CONFIG="${BASE_CONFIG:-$TOOLKIT_DIR/configs/PhysX-Mobility.yaml}"

DATA_ROOT="${DATA_ROOT:-/robot/data-lab/arts-gen-data/data/PhysX-Mobility-4view}"
LEGACY_DATA_ROOT="${LEGACY_DATA_ROOT:-/robot/data-lab/arts-gen-data/PhysX-Mobility-4view}"
WORKERS="${WORKERS:-4}"
VIEWS_PER_QUADRANT="${VIEWS_PER_QUADRANT:-3}"
RGB_ENGINE="${RGB_ENGINE:-BLENDER_EEVEE_NEXT}"
CYCLES_DEVICE="${CYCLES_DEVICE:-CPU}"
MODE=""
SMOKE_N="2"
OBJECT_IDS=""
STEPS=""
PROFILE="${PROFILE:-complete}"
PROFILE_ARG_SET="0"
CONFIG_OUT=""

usage() {
  cat <<EOF
Run dataset_toolkits PhysX-Mobility pipeline on a cloud dev machine.

Typical cloud usage:
  bash scripts/ops/data_pipeline/run_physx_mobility_cloud.sh --smoke 2 --workers 1 --rgb-engine CYCLES
  bash scripts/ops/data_pipeline/run_physx_mobility_cloud.sh --full --workers 8

Expected raw data layout:
  <data_root>/raw/finaljson/*.json
  <data_root>/raw/partseg/<object_id>/objs/

Options:
  --smoke [N]              Run on a symlinked N-object smoke root. Default N=2.
  --full                   Run on DATA_ROOT.
  --data-root PATH         Override DATA_ROOT.
  --legacy-data-root PATH  Source path to symlink if DATA_ROOT does not exist.
  --workers N             Worker count for render/voxel steps. Default: $WORKERS.
  --views-per-quadrant N   3 means 12 rendered views. Default: $VIEWS_PER_QUADRANT.
  --rgb-engine NAME        RGB render engine: BLENDER_EEVEE_NEXT or CYCLES. Default: $RGB_ENGINE.
  --cycles-device NAME     Cycles RGB device: CPU or CUDA. Default: $CYCLES_DEVICE.
  --base-config PATH       Base dataset_toolkits YAML to patch for cloud paths. Default: $BASE_CONFIG.
  --object-ids CSV         Restrict per-object steps for full mode.
  --steps CSV              Explicit dataset_toolkits steps, e.g. 1,2,3,4.
  --profile NAME           dataset_toolkits profile when --steps is not set.
                           Default: $PROFILE.
  --config-out PATH        Where to write generated config.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE="smoke"
      if [[ $# -ge 2 && ! "$2" =~ ^-- ]]; then
        SMOKE_N="$2"
        shift 2
      else
        shift
      fi
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --legacy-data-root)
      LEGACY_DATA_ROOT="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
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
    --base-config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --object-ids)
      OBJECT_IDS="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      PROFILE_ARG_SET="1"
      shift 2
      ;;
    --config-out)
      CONFIG_OUT="$2"
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

if [[ -z "$MODE" ]]; then
  echo "[error] choose exactly one mode: --smoke [N] or --full" >&2
  usage >&2
  exit 2
fi

if [[ -n "$STEPS" && "$PROFILE_ARG_SET" == "1" ]]; then
  echo "[error] --steps and --profile are mutually exclusive" >&2
  exit 2
fi

case "$RGB_ENGINE" in
  BLENDER_EEVEE_NEXT|CYCLES) ;;
  *) echo "[error] --rgb-engine must be BLENDER_EEVEE_NEXT or CYCLES: $RGB_ENGINE" >&2; exit 2 ;;
esac
case "$CYCLES_DEVICE" in
  CPU|CUDA|OPTIX) ;;
  *) echo "[error] --cycles-device must be CPU, CUDA, or OPTIX: $CYCLES_DEVICE" >&2; exit 2 ;;
esac

case "$DATA_ROOT" in
  /*) ;;
  *) echo "[error] --data-root must be absolute: $DATA_ROOT" >&2; exit 2 ;;
esac

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[error] missing base config: $BASE_CONFIG" >&2
  exit 3
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[error] activate the cloud Python env first, e.g. source .venv/arts-gen/bin/activate" >&2
  exit 3
fi

ensure_data_root() {
  if [[ -d "$DATA_ROOT/raw" ]]; then
    return 0
  fi

  if [[ -d "$LEGACY_DATA_ROOT/raw" ]]; then
    mkdir -p "$(dirname "$DATA_ROOT")"
    ln -sfn "$LEGACY_DATA_ROOT" "$DATA_ROOT"
    echo "[data] linked $DATA_ROOT -> $LEGACY_DATA_ROOT"
    return 0
  fi

  echo "[error] raw data not found under either:" >&2
  echo "        $DATA_ROOT/raw" >&2
  echo "        $LEGACY_DATA_ROOT/raw" >&2
  echo "        Extract raw.tar so one of those paths contains raw/finaljson and raw/partseg." >&2
  exit 3
}

select_object_ids() {
  local root="$1"
  find "$root/raw/finaljson" -maxdepth 1 -type f -name '*.json' -printf '%f\n' |
    sed 's/\.json$//' |
    sort |
    head -n "$SMOKE_N"
}

prepare_smoke_root() {
  local source_root="$1"
  local smoke_root="${DATA_ROOT}.smoke${SMOKE_N}"
  local ids=() id

  mapfile -t ids < <(select_object_ids "$source_root")
  if (( ${#ids[@]} != SMOKE_N )); then
    echo "[error] requested smoke N=$SMOKE_N but found ${#ids[@]} object json files" >&2
    exit 3
  fi

  rm -rf "$smoke_root"
  mkdir -p "$smoke_root/raw/finaljson" "$smoke_root/raw/partseg"

  for id in "${ids[@]}"; do
    ln -s "$source_root/raw/finaljson/$id.json" "$smoke_root/raw/finaljson/$id.json"
    ln -s "$source_root/raw/partseg/$id" "$smoke_root/raw/partseg/$id"
  done

  echo "[smoke] data_root: $smoke_root" >&2
  echo "[smoke] object_ids: $(IFS=,; echo "${ids[*]}")" >&2
  printf '%s\n' "$smoke_root"
}

make_config() {
  local run_data_root="$1"
  local out_path="$2"

  python3 - "$BASE_CONFIG" "$out_path" "$run_data_root" "$PROJECT_ROOT" "$VIEWS_PER_QUADRANT" "$RGB_ENGINE" "$CYCLES_DEVICE" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

base_config, out_path, data_root, project_root, views_per_quadrant, rgb_engine, cycles_device = sys.argv[1:8]
project = Path(project_root).resolve()
data = Path(data_root).resolve()

with open(base_config, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["data_root"] = str(data)
cfg.setdefault("render", {})["views_per_quadrant"] = int(views_per_quadrant)
cfg["render"]["rgb_engine"] = rgb_engine
cfg["render"]["cycles_device"] = cycles_device
cfg["render"]["blender"] = str(project / "software" / "blender-4.4.0-linux-x64" / "blender")

cfg.setdefault("feature", {})["dinov2_repo"] = str(project / "pretrained" / "dinov2")
cfg["feature"]["torch_hub_dir"] = str(project / "pretrained" / "torch_hub")

cfg.setdefault("trellis", {})["root"] = str(project / "TRELLIS-arts")
cfg["trellis"]["ss_encoder"] = str(
    project / "pretrained" / "TRELLIS-image-large" / "ckpts" / "ss_enc_conv3d_16l8_fp16"
)
cfg["trellis"]["ss_decoder"] = str(
    project / "pretrained" / "TRELLIS-image-large" / "ckpts" / "ss_dec_conv3d_16l8_fp16"
)
cfg["trellis"]["slat_encoder"] = str(
    project / "pretrained" / "TRELLIS-image-large" / "ckpts" / "slat_enc_swin8_B_64l8_fp16"
)

parts = data.parts
if "data" in parts:
    data_index = parts.index("data")
    image_prefix = Path(*parts[:data_index])
    if not str(image_prefix):
        image_prefix = Path("/")
else:
    raise SystemExit(
        f"data_root must contain a literal 'data' path segment for VLM image paths: {data}"
    )
cfg.setdefault("vlm", {})["image_prefix"] = str(image_prefix)

out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=False)
print(f"[config] wrote {out}")
print(f"[config] base_config={Path(base_config).resolve()}")
print(f"[config] data_root={data}")
print(f"[config] dataset_name={cfg.get('dataset_name')}")
print(f"[config] views_per_quadrant={views_per_quadrant} total_views={int(views_per_quadrant) * 4}")
print(f"[config] rgb_engine={rgb_engine}")
print(f"[config] obj_up_axis={cfg.get('render', {}).get('obj_up_axis', 'Y')}")
print(f"[config] cycles_device={cycles_device}")
print(f"[config] image_prefix={image_prefix}")
PY
}

ensure_data_root

RUN_DATA_ROOT="$DATA_ROOT"
if [[ "$MODE" == "smoke" ]]; then
  RUN_DATA_ROOT="$(prepare_smoke_root "$DATA_ROOT" | tail -n 1)"
fi

if [[ -z "$CONFIG_OUT" ]]; then
  CONFIG_OUT="$TOOLKIT_DIR/configs/PhysX-Mobility.cloud.generated.yaml"
fi
make_config "$RUN_DATA_ROOT" "$CONFIG_OUT"

export CONDA_DEFAULT_ENV=dataset_toolkits

CMD=(bash run_pipeline.sh --config "$CONFIG_OUT" --workers "$WORKERS")
if [[ -n "$STEPS" ]]; then
  CMD+=(--steps "$STEPS")
else
  CMD+=(--profile "$PROFILE")
fi
if [[ "$MODE" == "full" && -n "$OBJECT_IDS" ]]; then
  CMD+=(--object-ids "$OBJECT_IDS")
fi

echo "[run] cd $TOOLKIT_DIR"
printf '[run]'
printf ' %q' "${CMD[@]}"
printf '\n'

cd "$TOOLKIT_DIR"
"${CMD[@]}"
