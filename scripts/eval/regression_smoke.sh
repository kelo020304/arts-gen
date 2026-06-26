#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${ARTS_GEN_PYTHON:-/opt/venvs/arts-gen/bin/python}"
exec "$PYTHON_BIN" scripts/eval/regression_smoke.py "$@"
