#!/bin/bash
# Test training script for 8x Ascend 910B4 (32GB HBM each)

cd /home/ma-user/cfy

# Ascend NPU environment
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
export HCCL_WHITELIST_DISABLE=1

# Model path (local)
llm=/home/ma-user/cfy/pretrain/vlm

# DeepSpeed config (ZeRO-3 without CPU offload)
deepspeed_config=/home/ma-user/cfy/scripts/zero3.json

# Dataset: only physxmobility (we have dummy images for it)
datasets=physxmobility

# Training hyperparameters (same as original paper)
lr=2e-5
batch_size=1
grad_accum_steps=4

# Output
run_name="qwen2vl-ascend-8card-test"
output_dir=./output_ascend_8card_test

# Entry point
entry_file=qwenvl/train/train_qwen.py

# Launch with torchrun on 8 NPUs
torchrun --nproc_per_node=8 \
         --nnodes=1 \
         --master_addr=127.0.0.1 \
         --master_port=29500 \
         ${entry_file} \
    --deepspeed ${deepspeed_config} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 262144 \
    --min_pixels 65536 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name ${run_name} \
    --report_to none \
    --max_steps 10
