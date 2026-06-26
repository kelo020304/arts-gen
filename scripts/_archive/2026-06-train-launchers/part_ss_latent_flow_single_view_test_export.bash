#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-TRELLIS-arts/configs/arts/part_ss_latent_flow_single_view/part_ss_latent_flow_single_view.yaml}"
export RUN_ID="${RUN_ID:-manual_single_view_test_export}"
export DATA_ROOT="${DATA_ROOT:-/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-single-image-0512}"
export MANIFEST_PATH="${MANIFEST_PATH:-manifests/part_completion/arts_pc_physx-mobility_train.jsonl}"

bash scripts/train/part_ss_latent_flow_test_export.bash "$@"
