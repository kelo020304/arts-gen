#!/usr/bin/env bash
set -euo pipefail

cd /root/code/arts-gen/TRELLIS-arts
export CUDA_VISIBLE_DEVICES=0,1
export ATTN_BACKEND=sdpa
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
exec torchrun --nproc_per_node=2 --master_port="${MASTER_PORT:-29544}" \
    train_arts.py \
    --config configs/arts/part_mmdit/smoke_test.yaml \
    training.max_steps="${MAX_STEPS:-60000}" \
    training.output_dir="$RUN_DIR" \
    eval.fixed_every="${EVAL_EVERY:-5000}" \
    training.checkpoint_every="${CHECKPOINT_EVERY:-5000}"
