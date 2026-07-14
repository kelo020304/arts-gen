#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export MODE="${MODE:-train}"
export MODEL_SIZE="${MODEL_SIZE:-L}"
export NUM_GPUS="${NUM_GPUS:-8}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export OUT_DIR="${OUT_DIR:-/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-joint-edge-graph-v6}"
mkdir -p "$OUT_DIR/logs"

export PACKED_DIR="${PACKED_DIR:-/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v6}"
export DATA_ROOT="$PACKED_DIR"
export SPLIT_JSON="${SPLIT_JSON:-/robot/data-lab/jzh/art-gen/data/part_promptable_seg_manifests/v6/split_official_verse_realappliance_0511dd_v6.json}"
export PROXY_JSON="${PROXY_JSON:-/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/proxy_balanced_three_datasets_v6_eval_stratified.json}"
export WARM_START="${WARM_START:-/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0709-1-joint/ckpts/step_100000.pt}"

export BATCH="${BATCH:-32}"
export STEPS="${STEPS:-100000}"
export EVAL_EVERY="${EVAL_EVERY:-5000}"
export CKPT_EVERY="${CKPT_EVERY:-5000}"
export LOG_EVERY="${LOG_EVERY:-50}"
export PRECISION="${PRECISION:-bf16}"
export FP16="${FP16:-0}"

export USE_PACKED_WHOLE_OCC=1
export ROUTE=voxel
export SEG_DISCRIMINATIVE=1
export JOINT_SEG=1
export VOXEL_MAX_TOKENS=0
export GROUP_COST_BUDGET="${GROUP_COST_BUDGET:-150000}"
export BODY_CLASS_WEIGHT="${BODY_CLASS_WEIGHT:-0.25}"
export JOINT_KMAX="${JOINT_KMAX:-0}"
export JOINT_SMALL_PART_WEIGHT="${JOINT_SMALL_PART_WEIGHT:-1.5}"

export JOINT_LOCAL_MODE=edge_graph
export JOINT_LOCAL_DEPTH="${JOINT_LOCAL_DEPTH:-2}"
export JOINT_BOUNDARY_CE_WEIGHT="${JOINT_BOUNDARY_CE_WEIGHT:-0.5}"
export JOINT_BOUNDARY_CE_NEIGHBORHOOD="${JOINT_BOUNDARY_CE_NEIGHBORHOOD:-6}"
export JOINT_AFFINITY_WEIGHT="${JOINT_AFFINITY_WEIGHT:-0.2}"
export JOINT_AFFINITY_SAME_LABEL_WEIGHT="${JOINT_AFFINITY_SAME_LABEL_WEIGHT:-1.0}"
export JOINT_AFFINITY_CROSS_LABEL_WEIGHT="${JOINT_AFFINITY_CROSS_LABEL_WEIGHT:-1.0}"
export JOINT_AFFINITY_NEIGHBORHOOD="${JOINT_AFFINITY_NEIGHBORHOOD:-6}"

export JOINT_SMOOTH_WEIGHT="${JOINT_SMOOTH_WEIGHT:-0.2}"
export JOINT_SMOOTH_SAME_LABEL_WEIGHT=0
export JOINT_SMOOTH_ALL_LABEL_WEIGHT=0
export JOINT_SMOOTH_CROSS_LABEL_WEIGHT=0
export JOINT_CRF_EVAL=0

export REALAPPLIANCE_OVERSAMPLE="${REALAPPLIANCE_OVERSAMPLE:-8}"
export SMALL_OVERSAMPLE="${SMALL_OVERSAMPLE:-2}"
export VERSE_FOCUS_OVERSAMPLE="${VERSE_FOCUS_OVERSAMPLE:-2}"
export HELDOUT_EVAL_MAX_ROWS="${HELDOUT_EVAL_MAX_ROWS:-128}"
export TRAIN_EVAL_MAX_ROWS="${TRAIN_EVAL_MAX_ROWS:-64}"
export EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-128}"

bash scripts/train/part_promptable_seg/run_train.bash 2>&1 | tee -a "$OUT_DIR/logs/train_tee.log"
