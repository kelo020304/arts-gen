#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

export CONFIG="${CONFIG:-TRELLIS-arts/configs/arts/part_ss_latent_flow_single_view/part_ss_latent_flow_single_view.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/robot/data-lab/jzh/art-gen-output}"
export RUN_ID="${RUN_ID:-part-ss-latent-flow-single-view-$(date +%Y%m%d_%H%M%S)_$$}"
export RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$RUN_ID}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR}"
export INSPECTION_ROOT="${INSPECTION_ROOT:-$OUTPUT_DIR/inspections}"
export TRAIN_LOG="${TRAIN_LOG:-$RUN_DIR/train.log}"

exec "$HERE/part_ss_latent_flow_train.bash" "$@"
