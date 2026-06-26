#!/usr/bin/env bash
# Convenience launcher for vanilla TRELLIS Image -> 3D inference.
# Forwards args to scripts/inference/quick_reconstruct.py with sensible defaults.
#
# Usage examples:
#   bash scripts/inference/quick_reconstruct.bash --images img_front.png --output_dir outputs/recon1
#   bash scripts/inference/quick_reconstruct.bash --images a.png b.png c.png --output_dir outputs/recon2
#   FAST=1 bash scripts/inference/quick_reconstruct.bash --images a.png --output_dir out/  # 12 steps each
#   QUALITY=1 bash scripts/inference/quick_reconstruct.bash --images a.png --output_dir out/  # 25 steps each
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

PY="${PY:-python}"

EXTRA_ARGS=()
if [ "${FAST:-0}" = "1" ]; then
  EXTRA_ARGS+=(--ss_steps 12 --slat_steps 12)
fi
if [ "${QUALITY:-0}" = "1" ]; then
  EXTRA_ARGS+=(--ss_steps 25 --slat_steps 25)
fi
if [ "${NO_VIDEO:-0}" = "1" ]; then
  EXTRA_ARGS+=(--no_video)
fi

cd "$ROOT"
"$PY" scripts/inference/quick_reconstruct.py "${EXTRA_ARGS[@]}" "$@"
