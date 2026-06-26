"""Bump labels.json from v2 to v3.

v3 adds the optional per-part ``sites[]`` field (semantic AABB regions, see
``scripts/tools/articulator/usd_layout.svg``). v2 files never had any sites, so the
migration is a pure version-field bump — no data is invented or rearranged.

Run::

    python scripts/tools/articulator/migrate_v2_to_v3.py outputs/.../labels.json

By default writes a ``labels.json.v2.bak`` next to the input before mutating.
Pass ``--no-backup`` to skip.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import SchemaError, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="path to labels.json (v2)")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip writing labels.json.v2.bak before mutating")
    args = ap.parse_args()

    path = Path(args.labels).resolve()
    labels = json.loads(path.read_text())
    v = labels.get("version")
    if v == 3:
        print(f"[migrate] {path} already at v3, no-op")
        return
    if v != 2:
        raise SystemExit(f"[migrate] expected version=2, got {v!r}")

    if not args.no_backup:
        bak = path.with_suffix(path.suffix + ".v2.bak")
        shutil.copy(path, bak)
        print(f"[migrate] backup -> {bak}")

    labels["version"] = 3
    # ``sites[]`` stays optional — adding it is left to the editor UI.
    validate(labels)
    path.write_text(json.dumps(labels, indent=2))
    print(f"[migrate] {path} now at v3")


if __name__ == "__main__":
    main()
