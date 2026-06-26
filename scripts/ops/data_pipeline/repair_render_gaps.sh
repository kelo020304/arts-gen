#!/usr/bin/env bash
# ============================================================
# repair_render_gaps.sh
#
# Scan <data_root>/renders/ for incomplete (object, angle) renders — i.e.
# angle dirs where rgb/view_*.png or mask/mask_*.npy count is less than
# the expected views. Step 2 (02_render_quadrant_views.py) only checks
# rgb/view_0.png to decide skip-or-render, so partial renders are silently
# treated as "done" and step 3 (extract_bbox_gt) later crashes.
#
# This script:
#   1. Scans every renders/<obj>/<angle_*>/ and reports incomplete ones.
#   2. (Unless --scan-only) Deletes those incomplete angle dirs and re-invokes
#      step 2 on the affected object IDs. Step 2's per-angle skip check will
#      only re-render the deleted angles.
#   3. Re-scans to verify. If any still incomplete, exits 1 (those likely
#      hit a stable rendering bug — bad mesh / SIGSEGV / OOM — and need to
#      be excluded from the dataset).
#
# Scope: OLD 4-quadrant pipeline (submodules/dataset_toolkits, 12 views per
# angle = views_per_quadrant * 4). For the new single_image pipeline use a
# different repair script (mask structure differs).
#
# Usage:
#   bash scripts/ops/data_pipeline/repair_render_gaps.sh \
#     --data-root /robot/data-lab/.../PhysX-Mobility-full-4view-0511 \
#     --config /path/to/PhysX-Mobility.cloud.generated.yaml \
#     [--expected-views 12]   [--from-render-failures PATH]
#     [--scan-only|--check-only]   [--yes]   [--workers 4]
#
# Exit codes:
#   0 = scan was clean, or repair succeeded and re-scan is clean
#   1 = some angles still incomplete after re-render
#   2 = invalid arguments
# ============================================================

set -euo pipefail

DATA_ROOT=""
CONFIG=""
EXPECTED_VIEWS=12
FROM_RENDER_FAILURES=""
SCAN_ONLY=0
ASSUME_YES=0
WORKERS=4

usage() {
  sed -n '2,35p' "$0"
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root)      DATA_ROOT="$2";       shift 2 ;;
    --config)         CONFIG="$2";          shift 2 ;;
    --expected-views) EXPECTED_VIEWS="$2";  shift 2 ;;
    --from-render-failures) FROM_RENDER_FAILURES="$2"; shift 2 ;;
    --scan-only|--check-only) SCAN_ONLY=1;  shift ;;
    --yes|-y)         ASSUME_YES=1;         shift ;;
    --workers)        WORKERS="$2";         shift 2 ;;
    -h|--help)        usage ;;
    *) echo "[error] unknown arg: $1" >&2; usage ;;
  esac
done

[ -n "$DATA_ROOT" ] || { echo "[error] --data-root is required" >&2; exit 2; }
[ -d "$DATA_ROOT/renders" ] || { echo "[error] $DATA_ROOT/renders not found" >&2; exit 2; }
if [ -n "$FROM_RENDER_FAILURES" ]; then
  [ -f "$FROM_RENDER_FAILURES" ] || { echo "[error] render failure log $FROM_RENDER_FAILURES not found" >&2; exit 2; }
fi
if [ "$SCAN_ONLY" -eq 0 ]; then
  [ -n "$CONFIG" ] || { echo "[error] --config is required for repair (use --scan-only to skip)" >&2; exit 2; }
  [ -f "$CONFIG" ] || { echo "[error] config $CONFIG not found" >&2; exit 2; }
fi

TMPLIST=$(mktemp)
REPAIRLIST=$(mktemp)
trap 'rm -f "$TMPLIST" "$REPAIRLIST"' EXIT

# Scan: write tab-separated incomplete list to $TMPLIST, also print to stdout
scan() {
  local label="$1"
  local failure_log_arg="${2:-$FROM_RENDER_FAILURES}"
  echo ""
  echo "=== $label ==="
  python3 - "$DATA_ROOT/renders" "$EXPECTED_VIEWS" "$TMPLIST" "$failure_log_arg" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
outfile = Path(sys.argv[3])
failure_log = Path(sys.argv[4]) if sys.argv[4] else None

incomplete = {}
for obj_dir in sorted(root.iterdir()):
    if not obj_dir.is_dir():
        continue
    for ang_dir in sorted(obj_dir.iterdir()):
        if not ang_dir.is_dir() or not ang_dir.name.startswith("angle_"):
            continue
        mask_dir = ang_dir / "mask"
        rgb_dir = ang_dir / "rgb"
        n_masks = len(list(mask_dir.glob("mask_*.npy"))) if mask_dir.is_dir() else 0
        n_rgb = len(list(rgb_dir.glob("view_*.png"))) if rgb_dir.is_dir() else 0
        if n_masks < expected or n_rgb < expected:
            incomplete[(obj_dir.name, ang_dir.name)] = (n_masks, n_rgb, "incomplete")

failure_seeded = 0
if failure_log and failure_log.is_file():
    for line_no, line in enumerate(failure_log.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL in {failure_log}:{line_no}: {exc}") from exc
        obj = str(record.get("object_id", "")).strip()
        angle_raw = record.get("angle_idx")
        if not obj or angle_raw is None:
            raise SystemExit(f"missing object_id/angle_idx in {failure_log}:{line_no}")
        angle = f"angle_{int(angle_raw)}"
        key = (obj, angle)
        if key not in incomplete:
            ang_dir = root / obj / angle
            mask_dir = ang_dir / "mask"
            rgb_dir = ang_dir / "rgb"
            n_masks = len(list(mask_dir.glob("mask_*.npy"))) if mask_dir.is_dir() else 0
            n_rgb = len(list(rgb_dir.glob("view_*.png"))) if rgb_dir.is_dir() else 0
            incomplete[key] = (n_masks, n_rgb, "render_failure")
            failure_seeded += 1

with open(outfile, "w") as fh:
    for (obj, ang), (nm, nr, _reason) in sorted(incomplete.items()):
        fh.write(f"{obj}\t{ang}\t{nm}\t{nr}\n")

print(f"incomplete = {len(incomplete)}")
if failure_log:
    print(f"from_render_failures = {failure_seeded} ({failure_log})")
for idx, ((obj, ang), (nm, nr, reason)) in enumerate(sorted(incomplete.items())):
    if idx >= 30:
        break
    print(f"  {obj}/{ang}: masks={nm}/{expected} rgb={nr}/{expected} reason={reason}")
if len(incomplete) > 30:
    print(f"  ... ({len(incomplete) - 30} more)")
PY
}

prune_render_failures() {
  local failure_log="$1"
  local repaired_list="$2"
  local backup_path

  [ -n "$failure_log" ] || return 0
  [ -f "$failure_log" ] || return 0
  [ -s "$repaired_list" ] || return 0

  backup_path="${failure_log}.bak.$(date +%Y%m%d_%H%M%S)"
  cp "$failure_log" "$backup_path"

  python3 - "$failure_log" "$repaired_list" <<'PY'
import json
import sys
from pathlib import Path

failure_log = Path(sys.argv[1])
repaired_list = Path(sys.argv[2])

repaired = set()
for line in repaired_list.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj, angle, *_ = line.split("\t")
    repaired.add((obj, int(angle.removeprefix("angle_"))))

kept = []
removed = 0
for line in failure_log.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        kept.append(line)
        continue
    key = (str(record.get("object_id", "")), int(record.get("angle_idx", -1)))
    if key in repaired:
        removed += 1
        continue
    kept.append(line)

failure_log.write_text(
    "".join(f"{line}\n" for line in kept),
    encoding="utf-8",
)
print(f"[repair] render_failures_pruned={removed} path={failure_log}", flush=True)
PY
  echo "[repair] render_failures_backup=$backup_path"
}

scan "Initial scan (expected_views=$EXPECTED_VIEWS)" "$FROM_RENDER_FAILURES"

if [ ! -s "$TMPLIST" ]; then
  echo ""
  echo "[OK] no incomplete angles found"
  exit 0
fi

INCOMPLETE_COUNT=$(wc -l < "$TMPLIST")
cp "$TMPLIST" "$REPAIRLIST"

if [ "$SCAN_ONLY" -eq 1 ]; then
  echo ""
  echo "[scan-only] would repair $INCOMPLETE_COUNT (object, angle) pair(s); run without --scan-only to actually fix"
  exit 0
fi

echo ""
echo "[repair] will delete $INCOMPLETE_COUNT angle directories then re-render via step 2"
if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "Proceed? [y/N] " ans
  case "$ans" in
    [yY]|[yY][eE][sS]) : ;;
    *) echo "[abort] no changes made"; exit 0 ;;
  esac
fi

echo ""
echo "[delete] removing $INCOMPLETE_COUNT angle dir(s)..."
while IFS=$'\t' read -r obj ang nm nr; do
  rm -rf "$DATA_ROOT/renders/$obj/$ang"
  echo "  rm $DATA_ROOT/renders/$obj/$ang  (was masks=$nm rgb=$nr)"
done < "$REPAIRLIST"

OBJECT_IDS=$(cut -f1 "$REPAIRLIST" | sort -u | paste -sd, -)
NUM_OBJECTS=$(cut -f1 "$REPAIRLIST" | sort -u | wc -l)

echo ""
echo "[rerender] re-rendering $NUM_OBJECTS unique object(s) with workers=$WORKERS"
echo "[rerender] object_ids=$OBJECT_IDS"

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ARTS_GEN_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
TOOLKIT_DIR="$PROJECT_ROOT/submodules/dataset_toolkits"

[ -f "$TOOLKIT_DIR/pipeline/02_render_quadrant_views.py" ] || {
  echo "[error] $TOOLKIT_DIR/pipeline/02_render_quadrant_views.py not found" >&2
  exit 2
}

cd "$PROJECT_ROOT"
python3 "$TOOLKIT_DIR/pipeline/02_render_quadrant_views.py" \
  --config "$CONFIG" \
  --object-ids "$OBJECT_IDS" \
  --workers "$WORKERS"

scan "Re-scan after repair" ""

if [ -s "$TMPLIST" ]; then
  STILL=$(wc -l < "$TMPLIST")
  echo ""
  echo "[partial] $STILL angle(s) still incomplete after re-render."
  echo "[partial] These likely have a stable rendering bug (bad mesh / repeatable"
  echo "[partial] SIGSEGV / OOM). Consider excluding them from the dataset entirely:"
  echo ""
  while IFS=$'\t' read -r obj ang nm nr; do
    echo "  rm -rf $DATA_ROOT/renders/$obj/$ang   # masks=$nm rgb=$nr"
  done < "$TMPLIST"
  exit 1
fi

echo ""
echo "[OK] all $INCOMPLETE_COUNT angle(s) successfully re-rendered"
prune_render_failures "$FROM_RENDER_FAILURES" "$REPAIRLIST"
echo ""
echo "Next step: continue pipeline from step 3"
echo "  bash scripts/ops/data_pipeline/run_physx_mobility_cloud_1gpu.sh \\"
echo "    --data-root $DATA_ROOT \\"
echo "    --object-steps 3,4,5"
exit 0
