#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

reject_grep() {
    local pattern="$1"
    shift
    local tmp
    tmp="$(mktemp)"
    if grep -nE "$pattern" "$@" >"$tmp"; then
        cat "$tmp" >&2
        rm -f "$tmp"
        exit 1
    fi
    rm -f "$tmp"
}

for f in solver.py constraints.py backend.py visualize.py; do
    reject_grep "gt_limits|gt_lower|gt_upper|gt\.lower|gt\.upper|Aligned\.usd" "$ROOT/$f"
done

reject_grep "gt_limits|gt_lower|gt_upper|gt\.lower|gt\.upper" "$ROOT/write_predicted_usd.py"
reject_grep "gt_limits|gt_lower|gt_upper|gt\.lower|gt\.upper" "$ROOT/validate.py"

ALL_NON_HELPER=$(find "$ROOT" -name "*.py" \
    ! -name "usd_limit_reader.py" \
    ! -name "usd_limit_writer.py" \
    ! -name "audit_no_cheat.py")
reject_grep "GetLowerLimitAttr|GetUpperLimitAttr|physics:lowerLimit|physics:upperLimit" $ALL_NON_HELPER

NON_READER=$(find "$ROOT" -name "*.py" ! -name "data_prep.py" ! -name "audit_no_cheat.py")
reject_grep "from .*usd_limit_reader|import usd_limit_reader" $NON_READER

NON_WRITER=$(find "$ROOT" -name "*.py" ! -name "write_predicted_usd.py" ! -name "audit_no_cheat.py")
reject_grep "from .*usd_limit_writer|import usd_limit_writer" $NON_WRITER

reject_grep "Aligned\.usd|GetLowerLimitAttr|GetUpperLimitAttr|physics:lowerLimit|physics:upperLimit" \
    "$ROOT/compare.py" "$ROOT/comparison_visualize.py"

reject_grep "from .*compare|from .*comparison_visualize|import compare\b" \
    "$ROOT/solver.py" "$ROOT/constraints.py" "$ROOT/backend.py" \
    "$ROOT/visualize.py" "$ROOT/validate.py" "$ROOT/write_predicted_usd.py"

python3 "$ROOT/tools/audit_no_cheat.py"
echo "[OK] audit_no_cheat passed"
