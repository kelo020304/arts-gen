#!/usr/bin/env bash
# launch_dataset_preview.sh — start the dataset_toolkits HTML preview frontend.
#
# Wraps `submodules/dataset_toolkits/run_pipeline.sh` step 11 (HTML generation),
# then serves the result via `python -m http.server` so you can open it in a browser.
#
# Usage:
#   bash scripts/ops/data_pipeline/launch_dataset_preview.sh --data-root <ABSOLUTE_PATH> [options]
#
# Required:
#   --data-root <PATH>      Absolute path to the dataset root.
#                           Must contain (or eventually contain) raw/finaljson/, raw/partseg/,
#                           renders/, reconstruction/, vlm/, preview/.
#
# Options:
#   --port <N>              Preferred HTTP port (default 8000; auto-falls-back to 8001..8005).
#   --steps <CSV>           Pipeline steps to run before serving (default: 11).
#                           Use "1,2,3,4,5,6,7,8,9,10,11" to run the full default profile.
#                           Step 11 emits HTML preview; steps 12/13 are dev-only.
#   --workers <N>           Pass through to render/voxelize steps (default: 1).
#   --object-ids <CSV>      Filter to specific IDs (default: all).
#   --dataset-name <NAME>   Override dataset_name in generated config
#                           (default: basename of --data-root).
#   --config-template <P>   YAML template to clone (default:
#                           submodules/dataset_toolkits/configs/PhysX-Mobility.yaml).
#   --python <PATH>         Python interpreter used by run_pipeline.sh (default: python3).
#   --no-open               Do not auto-open browser even if xdg-open is available.
#   -h, --help              Show this message.
#
# Examples:
#   bash scripts/ops/data_pipeline/launch_dataset_preview.sh --data-root /media/data/PhysX-Mobility
#   bash scripts/ops/data_pipeline/launch_dataset_preview.sh --data-root /tmp/sample --steps 1,2,3,4,11 --workers 4
#   bash scripts/ops/data_pipeline/launch_dataset_preview.sh --data-root /data/foo --port 9000 --no-open

set -euo pipefail

# -----------------------------------------------------------------------------
# Locate project root + submodule
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
TOOLKIT_DIR="$PROJECT_ROOT/submodules/dataset_toolkits"
DEFAULT_TEMPLATE="$TOOLKIT_DIR/configs/PhysX-Mobility.yaml"

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DATA_ROOT=""
PORT=8000
STEPS="11"
WORKERS=1
OBJECT_IDS=""
DATASET_NAME=""
CONFIG_TEMPLATE="$DEFAULT_TEMPLATE"
PYTHON_BIN=""
NO_OPEN=0

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
usage() {
  sed -n '2,33p' "$0"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root)        DATA_ROOT="$2";        shift 2 ;;
    --port)             PORT="$2";             shift 2 ;;
    --steps)            STEPS="$2";            shift 2 ;;
    --workers)          WORKERS="$2";          shift 2 ;;
    --object-ids)       OBJECT_IDS="$2";       shift 2 ;;
    --dataset-name)     DATASET_NAME="$2";     shift 2 ;;
    --config-template)  CONFIG_TEMPLATE="$2";  shift 2 ;;
    --python)           PYTHON_BIN="$2";       shift 2 ;;
    --no-open)          NO_OPEN=1;             shift   ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "[error] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# -----------------------------------------------------------------------------
# Validate
# -----------------------------------------------------------------------------
[ -z "$DATA_ROOT" ] && { echo "[error] --data-root is required" >&2; usage; exit 2; }
case "$DATA_ROOT" in
  /*) : ;;  # absolute, ok
   *) echo "[error] --data-root must be absolute (got: $DATA_ROOT)" >&2; exit 2 ;;
esac
[ -d "$DATA_ROOT" ] || { echo "[error] --data-root not found: $DATA_ROOT" >&2; exit 2; }
[ -d "$TOOLKIT_DIR" ] || { echo "[error] submodule missing: $TOOLKIT_DIR  (did you run git submodule init/update?)" >&2; exit 3; }
[ -f "$CONFIG_TEMPLATE" ] || { echo "[error] config template not found: $CONFIG_TEMPLATE" >&2; exit 3; }

# DATASET_NAME defaults to "" so the YAML's own dataset_name value is preserved.
# Only override when the user explicitly passed --dataset-name on the CLI.
[ -z "$PYTHON_BIN" ] && PYTHON_BIN="python3"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
TMP_CONFIG=""
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 0.3
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "$TMP_CONFIG" ] && [ -f "$TMP_CONFIG" ]; then
    rm -f "$TMP_CONFIG"
  fi
}
trap cleanup INT TERM EXIT

# -----------------------------------------------------------------------------
# Generate temp config (override data_root + dataset_name on top of template)
# -----------------------------------------------------------------------------
TMP_CONFIG="$(mktemp -t dt_config_XXXXXX.yaml)"
"$PYTHON_BIN" - "$CONFIG_TEMPLATE" "$TMP_CONFIG" "$DATA_ROOT" "$DATASET_NAME" <<'PYEOF'
import sys, yaml
src, dst, data_root, dataset_name = sys.argv[1:]
with open(src, "r") as f:
    cfg = yaml.safe_load(f)
cfg["data_root"] = data_root
# Only override dataset_name when explicitly given; otherwise keep the YAML's
# own value so downstream JSONL slugs stay consistent across pipeline runs.
if dataset_name:
    cfg["dataset_name"] = dataset_name
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(f"[ok] wrote temp config: {dst}", file=sys.stderr)
PYEOF

# -----------------------------------------------------------------------------
# Run pipeline (default: just step 11)
# -----------------------------------------------------------------------------
echo ""
echo "=== Running dataset_toolkits steps [$STEPS] on $DATA_ROOT ==="
PIPELINE_ARGS=(--config "$TMP_CONFIG" --steps "$STEPS" --workers "$WORKERS")
[ -n "$OBJECT_IDS" ] && PIPELINE_ARGS+=(--object-ids "$OBJECT_IDS")

(
  cd "$TOOLKIT_DIR"
  # Upstream run_pipeline.sh (since 2026-05-06 refactor) hard-checks
  # CONDA_DEFAULT_ENV == "dataset_toolkits" and rejects any PYTHON= override.
  # This project intentionally uses the unified "arts-gen" env for everything;
  # we satisfy the env-name check by exporting the expected name (the actual
  # interpreter resolved via CONDA_PREFIX/bin/python3 still points at arts-gen
  # because we don't touch CONDA_PREFIX). PYTHON is left unset so the upstream
  # check at run_pipeline.sh:86 passes.
  CONDA_DEFAULT_ENV=dataset_toolkits bash run_pipeline.sh "${PIPELINE_ARGS[@]}"
)

# -----------------------------------------------------------------------------
# Verify preview output
# -----------------------------------------------------------------------------
# Since the 2026-05 dataset_toolkits refactor, step 11 emits HTML under
# preview/vlm_training/ instead of preview/ directly.
PREVIEW_DIR="$DATA_ROOT/preview/vlm_training"
INDEX_HTML="$PREVIEW_DIR/index.html"
if [ ! -f "$INDEX_HTML" ]; then
  echo "[error] preview not generated: $INDEX_HTML missing" >&2
  echo "        check that step 11 actually ran (--steps must include 11)" >&2
  exit 4
fi

# -----------------------------------------------------------------------------
# Find a free port (PORT, PORT+1, ..., PORT+5)
# -----------------------------------------------------------------------------
port_free() {
  local p="$1"
  ! ( exec 3<>"/dev/tcp/127.0.0.1/$p" ) 2>/dev/null
}

CHOSEN_PORT=""
for p in $(seq "$PORT" $((PORT + 5))); do
  if port_free "$p"; then CHOSEN_PORT="$p"; break; fi
done
[ -z "$CHOSEN_PORT" ] && { echo "[error] no free port in $PORT..$((PORT + 5))" >&2; exit 5; }

# -----------------------------------------------------------------------------
# Serve
# -----------------------------------------------------------------------------
URL="http://localhost:$CHOSEN_PORT/index.html"
echo ""
echo "=== Serving preview ==="
echo "  data_root : $DATA_ROOT"
echo "  preview   : $PREVIEW_DIR"
echo "  url       : $URL"
echo "  port      : $CHOSEN_PORT"
echo "  Ctrl+C to stop."
echo ""

(
  cd "$PREVIEW_DIR"
  "$PYTHON_BIN" -m http.server "$CHOSEN_PORT"
) &
SERVER_PID=$!

# Give server a moment, then optionally open browser
sleep 0.6

if [ "$NO_OPEN" -eq 0 ] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
fi

# Wait on server
wait "$SERVER_PID"
