#!/usr/bin/env bash
# Deprecated compatibility wrapper.
#
# The VolcEngine Blender/Vulkan environment used for PhysX-Mobility ignores
# per-process multi-GPU selectors, so this path now intentionally falls back to
# the stable one-GPU runner. New runs should call:
#
#   bash scripts/ops/data_pipeline/run_physx_mobility_cloud_1gpu.sh --gpu 3 ...

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-3}"
VULKAN_GPU="${VULKAN_GPU:-}"
FORWARDED=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --single-gpu|--gpu)
      GPU="$2"
      shift 2
      ;;
    --gpus)
      # Old multi-GPU calls usually passed "0,1,2,3". Pick the last item,
      # because this machine's Blender path has been observed to actually use
      # GPU 3 reliably.
      GPU="${2##*,}"
      shift 2
      ;;
    --vulkan-gpu)
      VULKAN_GPU="$2"
      shift 2
      ;;
    --vulkan-gpus)
      VULKAN_GPU="${2##*,}"
      shift 2
      ;;
    --skip-vulkan-probe|--skip-blender-device-pref|--skip-vulkan-selector-calibration|--prepare-vulkan-configs-only|--vulkan-probe-only)
      shift
      ;;
    *)
      FORWARDED+=("$1")
      shift
      ;;
  esac
done

echo "[deprecated] run_physx_mobility_cloud_4gpu_shards.sh now runs one stable GPU only." >&2
echo "[deprecated] forwarding to run_physx_mobility_cloud_1gpu.sh --gpu $GPU" >&2

cmd=(bash "$SCRIPT_DIR/run_physx_mobility_cloud_1gpu.sh" --gpu "$GPU")
if [[ -n "$VULKAN_GPU" ]]; then
  cmd+=(--vulkan-gpu "$VULKAN_GPU")
fi
cmd+=("${FORWARDED[@]}")
exec "${cmd[@]}"
