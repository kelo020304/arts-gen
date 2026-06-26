#!/usr/bin/env python3
"""Create an object-level subset split for Part-Prompt-Seg boundary A/B runs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_PATH = PROJECT_ROOT / "scripts" / "dev"
if str(DEV_PATH) not in sys.path:
    sys.path.insert(0, str(DEV_PATH))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import format_table  # noqa: E402


DEFAULT_PACKED_DIR = Path("/mnt/robot-data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5")
DEFAULT_OUT = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg_full_S_0618-2/"
    "eval/ab_subset_boundary/subset_split_seed20260622.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--total-objects", type=int, default=1200)
    parser.add_argument("--heldout-objects", type=int, default=200)
    parser.add_argument("--min-represented-per-source", type=int, default=20)
    return parser.parse_args()


def object_key(entry: dict[str, Any]) -> str:
    return f"{entry.get('dataset_id', '')}::{entry.get('obj_id', '')}"


def bucket_name(raw_count: int) -> str:
    if int(raw_count) < 50:
        return "tiny"
    if int(raw_count) < 500:
        return "small"
    if int(raw_count) <= 2000:
        return "medium"
    return "large"


def summarize(keys: list[str], obj_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ds = Counter(obj_meta[key]["dataset_id"] for key in keys)
    bucket = Counter()
    rows = 0
    for key in keys:
        rows += int(obj_meta[key]["rows"])
        bucket.update(obj_meta[key]["buckets"])
    return {
        "objects": len(keys),
        "rows": rows,
        "datasets": dict(sorted(ds.items())),
        "part_buckets": dict(sorted(bucket.items())),
        "medium_large_objects": sum(1 for key in keys if obj_meta[key]["has_medium_large"]),
    }


def main() -> int:
    args = parse_args()
    index_path = Path(args.packed_dir) / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_obj: dict[str, dict[str, Any]] = {}
    for entry in index.get("entries", []):
        key = object_key(entry)
        item = by_obj.setdefault(
            key,
            {
                "object_key": key,
                "dataset_id": str(entry.get("dataset_id", "")),
                "obj_id": str(entry.get("obj_id", "")),
                "rows": 0,
                "max_raw": 0,
                "buckets": Counter(),
                "has_medium_large": False,
            },
        )
        raw_count = int(entry.get("raw_count", 0) or 0)
        item["rows"] += 1
        item["max_raw"] = max(int(item["max_raw"]), raw_count)
        item["buckets"][bucket_name(raw_count)] += 1
        if raw_count >= 500:
            item["has_medium_large"] = True

    rng = random.Random(int(args.seed))
    by_source: dict[str, list[str]] = defaultdict(list)
    for key, item in by_obj.items():
        by_source[str(item["dataset_id"])].append(key)
    for keys in by_source.values():
        keys.sort(key=lambda key: (not by_obj[key]["has_medium_large"], -int(by_obj[key]["rows"]), key))

    total = min(int(args.total_objects), len(by_obj))
    heldout_count = min(max(1, int(args.heldout_objects)), total - 1)
    selected: list[str] = []
    selected_set: set[str] = set()

    def take_source(source: str, count: int) -> None:
        candidates = list(by_source.get(source, []))
        rng.shuffle(candidates)
        candidates.sort(key=lambda key: (not by_obj[key]["has_medium_large"], -int(by_obj[key]["rows"])))
        taken = 0
        for key in candidates:
            if len(selected) >= total or taken >= count:
                break
            if key not in selected_set:
                selected.append(key)
                selected_set.add(key)
                taken += 1

    for source in sorted(by_source):
        take_source(source, min(int(args.min_represented_per_source), len(by_source[source])))

    remaining = [key for key in by_obj if key not in selected_set]
    rng.shuffle(remaining)
    remaining.sort(key=lambda key: (not by_obj[key]["has_medium_large"], -int(by_obj[key]["rows"])))
    for key in remaining:
        if len(selected) >= total:
            break
        selected.append(key)
        selected_set.add(key)

    selected = selected[:total]
    rng.shuffle(selected)

    heldout: list[str] = []
    heldout_set: set[str] = set()
    selected_by_source: dict[str, list[str]] = defaultdict(list)
    for key in selected:
        selected_by_source[by_obj[key]["dataset_id"]].append(key)
    for source, keys in sorted(selected_by_source.items()):
        min_hold = 1 if len(keys) > 1 else 0
        if source == "realappliance":
            min_hold = min(max(10, min_hold), max(0, len(keys) - 1))
        candidates = list(keys)
        rng.shuffle(candidates)
        candidates.sort(key=lambda key: (not by_obj[key]["has_medium_large"], -int(by_obj[key]["rows"])))
        for key in candidates[:min_hold]:
            heldout.append(key)
            heldout_set.add(key)
    remaining_hold = [key for key in selected if key not in heldout_set]
    rng.shuffle(remaining_hold)
    remaining_hold.sort(key=lambda key: (not by_obj[key]["has_medium_large"], -int(by_obj[key]["rows"])))
    for key in remaining_hold:
        if len(heldout) >= heldout_count:
            break
        heldout.append(key)
        heldout_set.add(key)
    train = [key for key in selected if key not in heldout_set]
    if set(train) & set(heldout):
        raise RuntimeError("generated train/heldout overlap")

    payload = {
        "format": "part_promptable_seg_object_split",
        "seed": int(args.seed),
        "packed_dir": str(args.packed_dir),
        "train_keys": train,
        "heldout_keys": heldout,
        "train_ids": train,
        "heldout_ids": heldout,
        "summary": {
            "selected": summarize(selected, by_obj),
            "train": summarize(train, by_obj),
            "heldout": summarize(heldout, by_obj),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = []
    for split_name in ("selected", "train", "heldout"):
        item = payload["summary"][split_name]
        rows.append({
            "split": split_name,
            "objects": item["objects"],
            "rows": item["rows"],
            "medium_large_objects": item["medium_large_objects"],
            "datasets": json.dumps(item["datasets"], sort_keys=True),
            "part_buckets": json.dumps(item["part_buckets"], sort_keys=True),
        })
    print(format_table(rows, ["split", "objects", "rows", "medium_large_objects", "datasets", "part_buckets"]), flush=True)
    print(f"[subset] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
