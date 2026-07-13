#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MANIFEST = Path("/robot/data-lab/jzh/art-gen/data/realappliance/manifests/part_completion/arts_mllm_realappliance.train.jsonl")
DEFAULT_DATA_ROOT = Path("/robot/data-lab/jzh/art-gen/data/realappliance")
DEFAULT_SPLIT_JSON = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_verse_realappliance_v3.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _is_prismatic(part: dict[str, Any]) -> bool:
    motion = part.get("motion") if isinstance(part.get("motion"), dict) else {}
    return (
        str(part.get("joint", "")).lower() == "prismatic"
        or str(part.get("joint_type", "")).upper() == "B"
        or str(motion.get("motion_type", "")).lower() == "prismatic"
    )


def _prismatic_parts(rec: dict[str, Any]) -> list[dict[str, Any]]:
    return [part for part in (rec.get("target_parts") or []) if isinstance(part, dict) and _is_prismatic(part)]


def _part_score(part: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(part.get("visible_view_count", 0) or 0),
        int(part.get("raw_count", 0) or 0),
        str(part.get("name") or ""),
    )


def _rooted(data_root: Path, rel_or_abs: str | Path) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else data_root / path


def _part_info_path(rec: dict[str, Any], *, data_root: Path) -> Path:
    paths = dict(rec.get("paths") or {})
    rel = paths.get("part_info", f"reconstruction/part_info/{rec['object_id']}/part_info.json")
    return _rooted(data_root, rel)


def _load_part_info(rec: dict[str, Any], *, data_root: Path) -> dict[str, Any]:
    path = _part_info_path(rec, data_root=data_root)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    parts = payload.get("parts")
    return parts if isinstance(parts, dict) else {}


def _is_fixed_part(info: dict[str, Any]) -> bool:
    return str(info.get("joint", "")).lower() == "fixed" or str(info.get("joint_type", "")).upper() == "E"


def _fixed_child_parts(parts_info: dict[str, Any], part_name: str) -> list[tuple[str, dict[str, Any]]]:
    parent = parts_info.get(part_name)
    if not isinstance(parent, dict):
        return []
    group_id = parent.get("joint_group_id")
    if group_id is None or str(group_id) == "":
        return []
    group_text = str(group_id)
    out: list[tuple[str, dict[str, Any]]] = []
    for child_name, child in sorted(parts_info.items()):
        if child_name == part_name or not isinstance(child, dict) or not _is_fixed_part(child):
            continue
        child_parent = child.get("parent_group")
        child_group = child.get("joint_group_id")
        if (child_parent is not None and str(child_parent) == group_text) or (
            child_group is not None and str(child_group) == group_text
        ):
            out.append((str(child_name), child))
    return out


def _part_voxel_path(data_root: Path, rec: dict[str, Any], part_name: str, part: dict[str, Any] | None = None) -> Path:
    if part is not None:
        part_paths = dict(part.get("paths") or {})
        if part_paths.get("part_voxel"):
            return _rooted(data_root, part_paths["part_voxel"])
    obj_id = str(rec["object_id"])
    angle_idx = int(rec["angle_idx"])
    return data_root / "reconstruction" / "voxel_expanded" / obj_id / f"angle_{angle_idx}" / "64" / f"ind_{part_name}.npy"


def _write_union_part_voxel(
    *,
    data_root: Path,
    merge_voxel_root: Path,
    rec: dict[str, Any],
    picked: dict[str, Any],
    child_names: list[str],
) -> tuple[str | None, int | None, list[str]]:
    part_name = str(picked.get("name") or picked.get("item_name") or "part_00")
    sources = [(part_name, _part_voxel_path(data_root, rec, part_name, picked))]
    sources.extend((child_name, _part_voxel_path(data_root, rec, child_name)) for child_name in child_names)
    arrays: list[np.ndarray] = []
    missing: list[str] = []
    for source_name, path in sources:
        if not path.is_file():
            missing.append(f"{source_name}:{path}")
            continue
        arr = np.asarray(np.load(path, allow_pickle=False))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{path} expected [N,3] coords, got {arr.shape}")
        arrays.append(arr.astype(np.int64, copy=False))
    if len(arrays) <= 1:
        return None, None, missing
    union = np.unique(np.concatenate(arrays, axis=0), axis=0).astype(np.int64, copy=False)
    dst = (
        merge_voxel_root
        / str(rec["object_id"])
        / f"angle_{int(rec['angle_idx'])}"
        / "64"
        / f"ind_{part_name}.npy"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, union)
    return str(dst), int(union.shape[0]), missing


def _filtered_single_part_record(
    rec: dict[str, Any],
    part: dict[str, Any],
    *,
    data_root: Path,
    merge_voxel_root: Path,
) -> dict[str, Any]:
    out = copy.deepcopy(rec)
    picked = copy.deepcopy(part)
    part_name = str(picked.get("name") or picked.get("item_name") or "part_00")
    original_label = int(picked.get("original_label", picked.get("raw_label", 1)) or 1)
    parts_info = _load_part_info(rec, data_root=data_root)
    parent_info = parts_info.get(part_name) if isinstance(parts_info.get(part_name), dict) else {}
    parent_label = int(parent_info.get("label", original_label) or original_label)
    child_parts = _fixed_child_parts(parts_info, part_name)
    child_names = [name for name, _info in child_parts]
    child_labels = sorted({
        int(info["label"])
        for _name, info in child_parts
        if info.get("label") is not None
    })
    prompt_original_labels = [parent_label] + [label for label in child_labels if label != parent_label]
    union_path = None
    union_count = None
    missing_union_sources: list[str] = []
    if child_names:
        union_path, union_count, missing_union_sources = _write_union_part_voxel(
            data_root=data_root,
            merge_voxel_root=merge_voxel_root,
            rec=rec,
            picked=picked,
            child_names=child_names,
        )
    picked["local_label"] = 1
    picked["prompt_original_labels"] = prompt_original_labels
    picked["merged_original_labels"] = prompt_original_labels
    picked["merged_child_parts"] = child_names
    picked["merged_child_labels"] = child_labels
    if union_path is not None:
        picked.setdefault("paths", {})["part_voxel"] = union_path
        picked["raw_count"] = union_count
        picked["merged_part_voxel"] = union_path
        picked["merged_raw_count"] = union_count
    out["target_part_names"] = [part_name]
    out["target_parts"] = [picked]
    out["target_part_count"] = 1
    out["target_original_labels"] = prompt_original_labels
    out["label_remap"] = {str(label): 1 for label in prompt_original_labels}
    out["local_label_to_component"] = {"1": part_name}
    out["selection_override"] = {
        "policy": "one_unique_object_one_angle_one_prismatic_target_part_with_fixed_child_union",
        "source_target_part_count": int(rec.get("target_part_count", len(rec.get("target_parts") or [])) or 0),
        "picked_target_part_name": part_name,
        "picked_original_label": parent_label,
        "prompt_original_labels": prompt_original_labels,
        "merged_child_parts": child_names,
        "merged_child_labels": child_labels,
        "merged_part_voxel": union_path,
        "merged_raw_count": union_count,
        "missing_union_sources": missing_union_sources,
        "picked_joint": str(picked.get("joint") or ""),
        "picked_joint_type": str(picked.get("joint_type") or ""),
    }
    return out


def _eligible_by_object(
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    merge_voxel_root: Path,
) -> list[dict[str, Any]]:
    best_by_object: dict[str, tuple[tuple[int, int, int, str], dict[str, Any], dict[str, Any]]] = {}
    for rec in rows:
        obj_id = str(rec.get("object_id", ""))
        if not obj_id:
            continue
        prismatic = _prismatic_parts(rec)
        if not prismatic:
            continue
        picked_part = max(prismatic, key=_part_score)
        # Prefer views where the picked drawer has stronger prompt visibility, then lower angle.
        score = (
            int(picked_part.get("visible_view_count", 0) or 0),
            int(picked_part.get("raw_count", 0) or 0),
            -int(rec.get("angle_idx", 0) or 0),
            str(picked_part.get("name") or ""),
        )
        if obj_id not in best_by_object or score > best_by_object[obj_id][0]:
            best_by_object[obj_id] = (score, rec, picked_part)
    out = [
        _filtered_single_part_record(rec, part, data_root=data_root, merge_voxel_root=merge_voxel_root)
        for _score, rec, part in best_by_object.values()
    ]
    return sorted(out, key=lambda item: str(item.get("object_id", "")))


def _equal_spaced_indices(n: int, k: int) -> list[int]:
    if k <= 0:
        return []
    if n < k:
        raise ValueError(f"cannot select {k} equal-spaced rows from only {n} eligible rows")
    if k == 1:
        return [0]
    raw = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    indices: list[int] = []
    used = set()
    for idx in raw:
        idx = int(idx)
        if idx not in used:
            indices.append(idx)
            used.add(idx)
            continue
        for delta in range(1, n):
            for cand in (idx - delta, idx + delta):
                if 0 <= cand < n and cand not in used:
                    indices.append(cand)
                    used.add(cand)
                    break
            else:
                continue
            break
    return sorted(indices)


def _sample_payload(rec: dict[str, Any], *, split: str, data_root: Path, manifest_path: Path) -> dict[str, Any]:
    part = (rec.get("target_parts") or [{}])[0]
    raw_count = int(part.get("raw_count", 0) or 0)
    return {
        "split": split,
        "dataset_id": "realappliance",
        "object_key": f"realappliance::{rec['object_id']}",
        "obj_id": str(rec["object_id"]),
        "angle_idx": int(rec["angle_idx"]),
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "bucket": "ra_prismatic_single_target",
        "sample_bucket": "ra_prismatic_single_target",
        "priority_bucket": "ra_prismatic_single_target",
        "part_count": 1,
        "min_raw_voxels": raw_count,
        "max_raw_voxels": raw_count,
        "has_button": False,
        "has_large_keyword": True,
        "selected_reason": "equal_spaced_unique_realappliance_objects_one_angle_one_prismatic_part",
        "original_split": split,
        "target_part_name": str(part.get("name") or ""),
        "target_joint": str(part.get("joint") or ""),
        "target_joint_type": str(part.get("joint_type") or ""),
        "target_original_label": int(part.get("original_label", part.get("raw_label", 1)) or 1),
        "target_original_labels": [int(label) for label in part.get("prompt_original_labels", [])],
        "merged_child_parts": [str(name) for name in part.get("merged_child_parts", [])],
        "merged_child_labels": [int(label) for label in part.get("merged_child_labels", [])],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write ee-eval selection.json for equal-spaced RA prismatic drawer samples.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--train-count", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = _read_jsonl(args.manifest)
    filtered_dir = args.out_dir / "_data_configs" / "realappliance"
    merge_voxel_root = filtered_dir / "merged_voxels"
    rows = _eligible_by_object(source_rows, data_root=args.data_root, merge_voxel_root=merge_voxel_root)
    indices = _equal_spaced_indices(len(rows), int(args.limit))
    selected = [rows[idx] for idx in indices]
    train_count = min(int(args.train_count), len(selected))
    filtered_dir.mkdir(parents=True, exist_ok=True)
    filtered_manifest = filtered_dir / "ra_prismatic_one_part_40objects.jsonl"
    with filtered_manifest.open("w", encoding="utf-8") as handle:
        for rec in selected:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    split_json = filtered_dir / "ra_prismatic_one_part_40objects_split.json"
    split_payload = {
        "name": "0708-ra-40-unique-objects-one-angle-filtered",
        "source_split_json": str(args.split_json),
        "source_manifest": str(args.manifest),
        "selection_policy": (
            "40 equal-spaced unique RealAppliance objects; one angle per object; "
            "each manifest row is filtered to one prismatic/B target part for one drawer joint."
        ),
        "datasets": [
            {
                "dataset_id": "realappliance",
                "data_root": str(args.data_root),
                "manifest_paths": [str(filtered_manifest)],
            }
        ],
        "train_ids": [
            {"dataset_id": "realappliance", "obj_id": str(rec["object_id"]), "object_key": f"realappliance::{rec['object_id']}"}
            for rec in selected[:train_count]
        ],
        "heldout_ids": [
            {"dataset_id": "realappliance", "obj_id": str(rec["object_id"]), "object_key": f"realappliance::{rec['object_id']}"}
            for rec in selected[train_count:]
        ],
    }
    split_json.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    by_split = {"train": [], "held": []}
    for idx, rec in enumerate(selected):
        split = "train" if idx < train_count else "held"
        by_split[split].append(_sample_payload(rec, split=split, data_root=args.data_root, manifest_path=filtered_manifest))
    payload = {
        "name": "0708-ra-40-prismatic-unique-objects-equal-spaced",
        "split_json": str(split_json),
        "filtered_manifest": str(filtered_manifest),
        "selection_policy": (
            "Equal-spaced 40 unique RealAppliance objects with one selected angle per object. "
            "Rows are filtered to one prismatic/B target part, so each exported MJCF should have nq=nv=nu=njnt=1."
        ),
        "sample_selection_unit": "objects",
        "eligible_count": int(len(rows)),
        "eligible_object_count": int(len(rows)),
        "source_row_count": int(len(source_rows)),
        "selected_indices": [int(idx) for idx in indices],
        "selected_objects": [str(row.get("object_id")) for row in selected],
        "selected_angles": [int(row.get("angle_idx", 0)) for row in selected],
        "counts": {
            "train": len(by_split["train"]),
            "held": len(by_split["held"]),
            "objects": len({str(row.get("object_id")) for row in selected}),
            "samples": len(selected),
        },
        "samples": by_split,
    }
    out_path = args.out_dir / "selection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selection": str(out_path),
        "split_json": str(split_json),
        "filtered_manifest": str(filtered_manifest),
        "eligible_objects": len(rows),
        "selected": len(selected),
        "unique_objects": len({str(row.get("object_id")) for row in selected}),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
