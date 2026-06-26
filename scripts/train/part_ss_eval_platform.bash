#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ARTS_GEN_ENV_DIR="${ARTS_GEN_ENV_DIR:-/opt/venvs/arts-gen}"
ARTS_GEN_PYTHON="${ARTS_GEN_PYTHON:-$ARTS_GEN_ENV_DIR/bin/python}"

if [ ! -x "$ARTS_GEN_PYTHON" ]; then
    echo "[ERROR] arts-gen Python not found: $ARTS_GEN_PYTHON" >&2
    echo "        Install the dev-machine env to /opt/venvs/arts-gen, or set ARTS_GEN_ENV_DIR." >&2
    exit 1
fi

export ARTS_GEN_ENV_DIR
export ARTS_GEN_PYTHON
export CONDA_PREFIX="$ARTS_GEN_ENV_DIR"
export CONDA_DEFAULT_ENV="arts-gen"
export PATH="$(dirname "$ARTS_GEN_PYTHON"):$PATH"
export PYTHONPATH="TRELLIS-arts:${PYTHONPATH:-}"
export PART_SS_PLATFORM_HOST="${PART_SS_PLATFORM_HOST:-0.0.0.0}"
export PART_SS_PLATFORM_PORT="${PART_SS_PLATFORM_PORT:-7861}"
export PART_SS_PLATFORM_ROOTS="${PART_SS_PLATFORM_ROOTS:-/robot/data-lab/jzh/art-gen-output,/robot/data-lab/arts-gen-data/output}"
export PART_SS_PLATFORM_OUTPUT_ROOT="${PART_SS_PLATFORM_OUTPUT_ROOT:-/robot/data-lab/jzh/art-gen-output}"

# 先杀掉残留的旧平台实例：否则新进程 bind 同一端口失败、旧 server.py 继续服务，
# 改了服务端代码（如 RGB 越界校验 / artifact 的 npz→bin 转换）刷新页面也看不到效果。
# 用 '\.server' 精确匹配 `python -m part_ss_eval_platform.server`，不会误杀本启动脚本（.bash）。
EXISTING="$(pgrep -f 'part_ss_eval_platform\.server' || true)"
if [ -n "$EXISTING" ]; then
    echo "[restart] 杀掉已在运行的平台进程: $EXISTING"
    # shellcheck disable=SC2086
    kill $EXISTING 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        pgrep -f 'part_ss_eval_platform\.server' >/dev/null 2>&1 || break
        sleep 1
    done
    pkill -9 -f 'part_ss_eval_platform\.server' 2>/dev/null || true
fi

"$ARTS_GEN_PYTHON" -m part_ss_eval_platform.server
