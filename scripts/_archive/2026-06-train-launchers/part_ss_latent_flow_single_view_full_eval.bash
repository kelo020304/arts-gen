#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

export EVAL_VIEW_MODE="${EVAL_VIEW_MODE:-single_view}"
export GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"

exec bash "$HERE/part_ss_latent_flow_full_eval.bash" "$@"
