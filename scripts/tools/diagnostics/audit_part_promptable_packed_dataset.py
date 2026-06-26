#!/usr/bin/env python3
"""Audit packed promptable segmentation samples against source files."""

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

from scripts.train.part_promptable_seg.part_promptable_seg_utils import DATA_ROOT, PACKED_DATA_ROOT, format_table  # noqa: E402


def coords_set(arr: np.ndarray) -> set[tuple[int, int, int]]:
    arr = np.asarray(arr, dtype=np.int64)
    if arr.size == 0:
        return set()
    return set(map(tuple, arr.reshape(-1, 3).tolist()))


def load_manifest_rows(data_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = data_root / "manifests/part_completion/arts_mllm_physx-mobility.train.jsonl"
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows[(str(rec["object_id"]), int(rec["angle_idx"]))] = rec
    return rows


def category(entry: dict[str, Any], part_info: dict[str, Any]) -> str:
    part_name = str(entry["part_name"]).lower()
    part_count = int(part_info.get("num_parts", len(part_info.get("parts", {}))))
    if "button" in part_name:
        return "button"
    if "door" in part_name or "lid" in part_name:
        return "door_lid"
    if part_count <= 1:
        return "single_part"
    return "general"


def pick_entries(index: dict[str, Any], data_root: Path, per_category: int) -> list[dict[str, Any]]:
    picked: dict[str, list[dict[str, Any]]] = {"button": [], "door_lid": [], "single_target": [], "general": []}
    manifest_rows = load_manifest_rows(data_root)
    for entry in index["entries"]:
        obj = str(entry["obj_id"])
        angle = int(entry["angle_idx"])
        info_path = data_root / f"reconstruction/part_info/{obj}/part_info.json"
        part_info = json.loads(info_path.read_text(encoding="utf-8"))
        cat = category(entry, part_info)
        if len(manifest_rows[(obj, angle)].get("target_part_names", [])) == 1:
            cat = "single_target"
        if len(picked[cat]) < int(per_category):
            picked[cat].append(entry)
        if all(len(v) >= int(per_category) for v in picked.values()):
            break
    out = []
    for cat in ("button", "door_lid", "single_target", "general"):
        out.extend(picked[cat])
    if any(len(v) < int(per_category) for v in picked.values()):
        raise RuntimeError(f"not enough audit entries selected: { {k: len(v) for k, v in picked.items()} }")
    return out


def audit_entry(entry: dict[str, Any], sample: dict[str, Any], manifest_rows: dict[tuple[str, int], dict[str, Any]], data_root: Path) -> dict[str, Any]:
    obj = str(entry["obj_id"])
    angle = int(entry["angle_idx"])
    part_name = str(entry["part_name"])
    rec = manifest_rows[(obj, angle)]
    target_part = {str(p.get("name")): p for p in rec.get("target_parts", [])}[part_name]
    original_label = int(target_part["original_label"])
    view_indices = [int(v) for v in rec["view_indices"]]

    z_src = np.load(data_root / f"reconstruction/ss_latents_expanded/{obj}/angle_{angle}/latent.npz")["mean"].astype(np.float32)
    z_ok = np.array_equal(sample["z_global"].numpy().astype(np.float32), z_src)

    raw_src = np.load(data_root / f"reconstruction/voxel_expanded/{obj}/angle_{angle}/64/ind_{part_name}.npy").astype(np.int64)
    raw_ok = np.array_equal(sample["raw_coords"].numpy().astype(np.int64), raw_src)

    masks_ok = []
    for row, view_idx in enumerate(view_indices):
        label_map = np.load(data_root / f"renders/{obj}/angle_{angle}/mask/mask_{view_idx}.npy")
        masks_ok.append(np.array_equal(sample["masks2d"][row].numpy().astype(np.uint8), (label_map == original_label).astype(np.uint8)))

    part_info = json.loads((data_root / f"reconstruction/part_info/{obj}/part_info.json").read_text(encoding="utf-8"))
    part_type = str(part_info["parts"][part_name]["type"])
    type_ok = part_type == str(sample["semantic_type"])

    angle_dir = data_root / f"reconstruction/voxel_expanded/{obj}/angle_{angle}/64"
    union = set()
    for path in sorted(angle_dir.glob("ind_*.npy")):
        union |= coords_set(np.load(path))
    surface = coords_set(np.load(angle_dir / "surface.npy"))
    packed_whole = coords_set(sample["whole_coords"].numpy().astype(np.int64))
    whole_ok = packed_whole == union
    surface_ok = surface == union
    cat = category(entry, part_info)
    if len(rec.get("target_part_names", [])) == 1:
        cat = "single_target"
    return {
        "category": cat,
        "key": entry["key"],
        "z": z_ok,
        "masks": all(masks_ok),
        "raw": raw_ok,
        "type": type_ok,
        "whole": whole_ok,
        "surface": surface_ok,
        "raw_count": int(raw_src.shape[0]),
        "whole_count": len(packed_whole),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-dir", type=Path, default=PACKED_DATA_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/packed_equivalence_audit.json"))
    args = parser.parse_args()
    index_path = args.packed_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = pick_entries(index, args.data_root, int(args.per_category))
    manifest_rows = load_manifest_rows(args.data_root)
    shard_cache: dict[str, list[dict[str, Any]]] = {}
    rows = []
    for entry in entries:
        shard = str(entry["shard"])
        if shard not in shard_cache:
            shard_cache[shard] = torch.load(args.packed_dir / shard, map_location="cpu", weights_only=False)
        sample = shard_cache[shard][int(entry["index"])]
        rows.append(audit_entry(entry, sample, manifest_rows, args.data_root))
    ok = all(row["z"] and row["masks"] and row["raw"] and row["type"] and row["whole"] and row["surface"] for row in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"ok": ok, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_table(rows, ["category", "key", "z", "masks", "raw", "type", "whole", "surface", "raw_count", "whole_count"]))
    print(f"ok={ok} out={args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
