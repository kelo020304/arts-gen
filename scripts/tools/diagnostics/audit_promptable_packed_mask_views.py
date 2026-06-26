#!/usr/bin/env python3
"""Verify packed prompt masks match raw masks selected by manifest view_indices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_PATH = PROJECT_ROOT / "scripts" / "dev"
if str(DEV_PATH) not in sys.path:
    sys.path.insert(0, str(DEV_PATH))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    OFFICIAL_SPLIT_PATH,
    PACKED_DATA_ROOT,
    PartRow,
    downsample_binary_mask,
    enumerate_part_rows,
    load_official_split,
    make_base_dataset,
    part_row_key,
    rows_for_obj_ids,
)


def _load_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row_spec_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "obj_id": str(record["obj_id"]),
        "angle_idx": int(record["angle_idx"]),
        "part_name": str(record["part_name"]),
    }


def _pick_rows(
    rows_all: list[PartRow],
    *,
    split_json: Path,
    trigger_json: Path | None,
    available_keys: set[str],
    trigger_limit: int,
    clean_limit: int,
) -> list[PartRow]:
    by_key = {part_row_key(row): row for row in rows_all}
    selected: list[PartRow] = []
    seen: set[str] = set()

    trigger_records = _load_json(trigger_json) if trigger_json is not None and trigger_json.is_file() else []
    if isinstance(trigger_records, dict):
        trigger_records = trigger_records.get("records", [])
    for record in list(trigger_records or [])[: int(trigger_limit)]:
        spec = _row_spec_from_record(record)
        key = f"{spec['obj_id']}|{int(spec['angle_idx'])}|{spec['part_name']}"
        if key in by_key and key in available_keys and key not in seen:
            selected.append(by_key[key])
            seen.add(key)

    split = load_official_split(split_json)
    train_rows = rows_for_obj_ids(rows_all, split["train_ids"])
    clean_candidates = [
        row
        for row in train_rows
        if part_row_key(row) not in seen
        and part_row_key(row) in available_keys
        and "button" in row.part_name.lower()
    ]
    clean_candidates.extend(
        row
        for row in train_rows
        if part_row_key(row) not in seen
        and part_row_key(row) in available_keys
    )
    for row in clean_candidates:
        if len(selected) >= int(trigger_limit) + int(clean_limit):
            break
        key = part_row_key(row)
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
    return selected


def _load_packed_sample(packed_dir: Path, index: dict[str, Any], key: str) -> dict[str, Any]:
    entries = {str(entry["key"]): entry for entry in index.get("entries", [])}
    if key not in entries:
        raise KeyError(f"{key} not found in packed index {packed_dir / 'index.json'}")
    entry = entries[key]
    shard_path = packed_dir / str(entry["shard"])
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    return dict(payload[int(entry["index"])])


def _raw_manifest_masks(base_ds, row: PartRow, *, mask_size: int) -> tuple[torch.Tensor, list[int], list[int]]:
    sample = base_ds.samples[int(row.sample_idx)]
    obj_id = str(sample["obj_id"])
    angle_idx = int(sample["angle_idx"])
    view_indices = [int(v) for v in sample["view_indices"]]
    label = int(row.original_label)
    views = []
    raw_visible_by_view = []
    for view_idx in view_indices:
        path = base_ds.mask_root / obj_id / f"angle_{angle_idx}" / "mask" / f"mask_{view_idx}.npy"
        label_map = np.asarray(np.load(path))
        binary = label_map == label
        raw_visible_by_view.append(int(binary.sum()))
        views.append(downsample_binary_mask(binary, int(mask_size)))
    return torch.from_numpy(np.stack(views, axis=0)).float(), view_indices, raw_visible_by_view


def _packed_masks(sample: dict[str, Any]) -> torch.Tensor:
    masks = sample["masks2d"]
    if not isinstance(masks, torch.Tensor):
        masks = torch.as_tensor(masks)
    return masks.float()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-dir", type=Path, default=PACKED_DATA_ROOT)
    parser.add_argument("--split-json", type=Path, default=OFFICIAL_SPLIT_PATH)
    parser.add_argument("--trigger-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("/tmp/promptable_packed_mask_view_audit.json"))
    parser.add_argument("--trigger-limit", type=int, default=10)
    parser.add_argument("--clean-limit", type=int, default=10)
    parser.add_argument("--mask-size", type=int, default=512)
    args = parser.parse_args()

    index_path = args.packed_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"packed index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    available_keys = {str(entry["key"]) for entry in index.get("entries", [])}
    base = make_base_dataset()
    rows_all = enumerate_part_rows(base)
    rows = _pick_rows(
        rows_all,
        split_json=args.split_json,
        trigger_json=args.trigger_json,
        available_keys=available_keys,
        trigger_limit=int(args.trigger_limit),
        clean_limit=int(args.clean_limit),
    )

    records = []
    mismatches = []
    trigger_keys = set()
    trigger_records = _load_json(args.trigger_json) if args.trigger_json is not None and args.trigger_json.is_file() else []
    if isinstance(trigger_records, dict):
        trigger_records = trigger_records.get("records", [])
    for record in list(trigger_records or [])[: int(args.trigger_limit)]:
        spec = _row_spec_from_record(record)
        trigger_keys.add(f"{spec['obj_id']}|{int(spec['angle_idx'])}|{spec['part_name']}")

    for row in rows:
        key = part_row_key(row)
        packed = _load_packed_sample(args.packed_dir, index, key)
        packed_masks = _packed_masks(packed)
        raw_masks, manifest_views, raw_visible_by_view = _raw_manifest_masks(base, row, mask_size=int(args.mask_size))
        equal_by_view = [
            bool(torch.equal(packed_masks[idx].cpu(), raw_masks[idx].cpu()))
            for idx in range(raw_masks.shape[0])
        ]
        abs_diff_by_view = [
            int((packed_masks[idx].cpu() - raw_masks[idx].cpu()).abs().sum().item())
            for idx in range(raw_masks.shape[0])
        ]
        packed_visible_by_view = [int(packed_masks[idx].sum().item()) for idx in range(packed_masks.shape[0])]
        record = {
            "key": key,
            "source": "trigger" if key in trigger_keys else "clean",
            "obj_id": row.obj_id,
            "angle_idx": int(row.angle_idx),
            "sample_id": row.sample_id,
            "part_name": row.part_name,
            "original_label": int(row.original_label),
            "manifest_view_indices": manifest_views,
            "packed_view_indices": [int(v) for v in packed["view_indices"].tolist()],
            "equal_by_view": equal_by_view,
            "abs_diff_by_view": abs_diff_by_view,
            "all_equal": bool(all(equal_by_view)),
            "raw_visible_by_view": raw_visible_by_view,
            "raw_visible_pixels": int(sum(raw_visible_by_view)),
            "packed_visible_by_view": packed_visible_by_view,
            "packed_visible_pixels": int(sum(packed_visible_by_view)),
        }
        records.append(record)
        if not record["all_equal"]:
            mismatches.append(record)

    summary = {
        "packed_dir": str(args.packed_dir),
        "split_json": str(args.split_json),
        "trigger_json": str(args.trigger_json) if args.trigger_json is not None else None,
        "rows_checked": len(records),
        "mismatches": len(mismatches),
        "trigger_rows_checked": sum(1 for rec in records if rec["source"] == "trigger"),
        "clean_rows_checked": sum(1 for rec in records if rec["source"] == "clean"),
        "trigger_rows_visible_in_manifest_views": sum(
            1 for rec in records if rec["source"] == "trigger" and int(rec["raw_visible_pixels"]) > 0
        ),
    }
    payload = {"summary": summary, "records": records, "mismatches": mismatches}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if mismatches:
        print(f"[packed-mask-view-audit] mismatches written to {args.out}", flush=True)
        return 1
    print(f"[packed-mask-view-audit] ok out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
