#!/usr/bin/env bash
# ============================================================
# Common DDP launcher helper (Phase 09 D-16).
#
# Source from <stage>_train.bash:
#   source "$(dirname "$0")/_ddp_common.sh"
#   launch_ddp <num_gpus> <gpu_ids> <python_script> <python_args...>
#
# Required env vars (set by caller before calling launch_ddp):
#   None — this helper uses positional arguments only.
# Optional env vars (defaults applied if unset):
#   TORCH_HOME, ATTN_BACKEND, WANDB_IGNORE_GLOBS, MASTER_PORT
# ============================================================
set -euo pipefail

_ddp_common_setup_env() {
    # Defaults; caller may override before calling launch_ddp.
    export TORCH_HOME="${TORCH_HOME:-submodules/TRELLIS.1}"
    export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
    export WANDB_IGNORE_GLOBS="${WANDB_IGNORE_GLOBS:-*.pt,*.safetensors,*.ckpt}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    # Stream print() output live. The launcher pipes stdout through `tee`, which
    # makes Python block-buffer stdout (logs appear in delayed bursts); unbuffered
    # output flushes each line immediately to the terminal and the train log.
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
}

_ddp_common_pick_python() {
    if command -v python3 >/dev/null 2>&1; then echo "python3"
    elif command -v python  >/dev/null 2>&1; then echo "python"
    else echo "[ERROR] No python in PATH" >&2; exit 1
    fi
}

launch_ddp() {
    local NUM_GPUS="$1"; shift
    local GPU_IDS="$1"; shift
    local SCRIPT="$1"; shift
    # Remaining args are passed to the python script

    _ddp_common_setup_env
    local PY_BIN
    PY_BIN="$(_ddp_common_pick_python)"

    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        local _GPU_ARR
        IFS=',' read -r -a _GPU_ARR <<< "$GPU_IDS"
        NUM_GPUS="${#_GPU_ARR[@]}"
    fi

    if [ "$NUM_GPUS" -le 0 ]; then
        echo "[ERROR] Invalid NUM_GPUS=$NUM_GPUS" >&2
        exit 1
    fi
    if [ "$NUM_GPUS" -gt 1 ] && ! command -v torchrun >/dev/null 2>&1; then
        echo "[ERROR] NUM_GPUS=$NUM_GPUS but torchrun not in PATH" >&2
        exit 1
    fi

    echo "[ddp_common] NUM_GPUS=$NUM_GPUS  TORCH_HOME=$TORCH_HOME  MASTER_PORT=$MASTER_PORT  GPU_IDS='${GPU_IDS:-<unset>}'"
    if [ "$NUM_GPUS" -eq 1 ]; then
        "$PY_BIN" "$SCRIPT" "$@"
    else
        torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
            "$SCRIPT" "$@"
    fi
}
