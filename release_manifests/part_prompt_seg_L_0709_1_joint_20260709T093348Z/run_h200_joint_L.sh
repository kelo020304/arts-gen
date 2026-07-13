#!/usr/bin/env bash
set -euo pipefail

cd /root/code/arts-gen
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /opt/venvs/arts-gen

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MODE=train
export MODEL_SIZE=L
export OUT_DIR="${OUT_DIR:-/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0709-1-joint}"
mkdir -p "$OUT_DIR/logs"

export PACKED_DIR=/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v6
export DATA_ROOT="$PACKED_DIR"
export SPLIT_JSON=/robot/data-lab/jzh/art-gen/data/part_promptable_seg_manifests/v6/split_official_verse_realappliance_0511dd_v6.json
export PROXY_JSON=/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/proxy_balanced_three_datasets_v6_eval_stratified.json

export BATCH=32
export STEPS="${STEPS:-100000}"
export EVAL_EVERY=5000
export CKPT_EVERY=5000
export LOG_EVERY=50
export PRECISION=bf16
export FP16=0

export USE_PACKED_WHOLE_OCC=1
export ROUTE=voxel
export SEG_DISCRIMINATIVE=1
export JOINT_SEG=1
export VOXEL_MAX_TOKENS=0
export EXTRA_ARGS="${EXTRA_ARGS:---voxel-max-tokens 0}"

export GROUP_COST_BUDGET=150000
export BODY_CLASS_WEIGHT=0.25
export JOINT_KMAX=0
export JOINT_SMALL_PART_WEIGHT=1.5
export JOINT_SMOOTH_WEIGHT=0.2
export JOINT_SMOOTH_SAME_LABEL_WEIGHT=1.5
export JOINT_SMOOTH_ALL_LABEL_WEIGHT=0.05
export JOINT_SMOOTH_NEIGHBORHOOD=6

export JOINT_CRF_EVAL=1
export JOINT_CRF_ITERS=5
export JOINT_CRF_PAIRWISE=0.30
export JOINT_CRF_NEIGHBORHOOD=6

export REALAPPLIANCE_OVERSAMPLE=8
export SMALL_OVERSAMPLE=2
export VERSE_FOCUS_OVERSAMPLE=2
export HELDOUT_EVAL_MAX_ROWS=128
export TRAIN_EVAL_MAX_ROWS=64
export EVAL_MAX_ROWS=128

export WARM_START="${WARM_START:-/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/ckpts/latest.pt}"

bash scripts/train/part_promptable_seg/run_train.bash 2>&1 | tee -a "$OUT_DIR/logs/train_tee.log"
