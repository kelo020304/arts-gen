#!/usr/bin/env bash
# ============================================================
# SLURM multi-node launcher helper (Phase 09 D-16).
# Used by Xiaomi H200 cluster runs.
#
# Source from <stage>_train.bash inside an SLURM allocation:
#   source "$(dirname "$0")/_slurm_common.sh"
#   launch_slurm <python_script> <python_args...>
# ============================================================
set -euo pipefail

launch_slurm() {
    : "${SLURM_JOB_ID:?must run inside an SLURM allocation}"
    export TORCH_HOME="${TORCH_HOME:-submodules/TRELLIS.1}"
    export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
    export WANDB_IGNORE_GLOBS="${WANDB_IGNORE_GLOBS:-*.pt,*.safetensors,*.ckpt}"

    export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export WORLD_SIZE="${WORLD_SIZE:-$SLURM_NTASKS}"

    echo "[slurm_common] NODES=$SLURM_NNODES  NTASKS=$SLURM_NTASKS  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
    srun --kill-on-bad-exit=1 \
        bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python $*"
}
