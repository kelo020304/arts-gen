#!/usr/bin/env python3
"""Build the v6 promptable-part split with RealAppliance fixed-child unions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_V5_INDEX = Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5/index.json")
DEFAULT_RA_ROOT = Path("/robot/data-lab/jzh/art-gen/data/realappliance")
DEFAULT_OUT_ROOT = Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_manifests/v6")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rooted(data_root: Path, rel_or_abs: str | Path) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else data_root / path


def part_info_path(data_root: Path, rec: dict[str, Any]) -> Path:
    rel = dict(rec.get("paths") or {}).get("part_info", f"reconstruction/part_info/{rec['object_id']}/part_info.json")
    return rooted(data_root, rel)


def load_part_info(data_root: Path, rec: dict[str, Any]) -> dict[str, Any]:
    path = part_info_path(data_root, rec)
    if not path.is_file():
        return {}
    payload = read_json(path)
    parts = payload.get("parts")
    return parts if isinstance(parts, dict) else {}


def is_fixed(meta: dict[str, Any]) -> bool:
    return str(meta.get("joint", "")).lower() == "fixed" or str(meta.get("joint_type", "")).upper() == "E"


def is_movable_target(part: dict[str, Any], info: dict[str, Any] | None) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            part.get("joint"),
            part.get("joint_type"),
            dict(part.get("motion") or {}).get("motion_type"),
            (info or {}).get("joint"),
            (info or {}).get("joint_type"),
        )
    ).lower()
    return "fixed" not in text and ("prismatic" in text or "revolute" in text or "rotate" in text or "b" in text or "c" in text)


def fixed_children(
    parts_info: dict[str, Any],
    *,
    parent_name: str,
    target_names: set[str],
    target_parent_labels: set[int],
) -> list[tuple[str, dict[str, Any]]]:
    parent = parts_info.get(parent_name)
    if not isinstance(parent, dict):
        return []
    group_id = parent.get("joint_group_id")
    if group_id is None or str(group_id) == "":
        return []
    group_text = str(group_id)
    out = []
    for child_name, child in sorted(parts_info.items()):
        if child_name == parent_name or child_name in target_names or not isinstance(child, dict) or not is_fixed(child):
            continue
        child_label = child.get("label")
        if child_label is not None and int(child_label) in target_parent_labels:
            continue
        child_parent = child.get("parent_group")
        child_group = child.get("joint_group_id")
        if (child_parent is not None and str(child_parent) == group_text) or (
            child_group is not None and str(child_group) == group_text
        ):
            out.append((str(child_name), child))
    return out


def part_voxel_path(data_root: Path, rec: dict[str, Any], part_name: str, part: dict[str, Any] | None = None) -> Path:
    if part is not None:
        rel = dict(part.get("paths") or {}).get("part_voxel")
        if rel:
            return rooted(data_root, rel)
    return (
        data_root
        / "reconstruction"
        / "voxel_expanded"
        / str(rec["object_id"])
        / f"angle_{int(rec['angle_idx'])}"
        / "64"
        / f"ind_{part_name}.npy"
    )


def write_union_voxel(
    *,
    data_root: Path,
    out_root: Path,
    rec: dict[str, Any],
    part: dict[str, Any],
    child_names: list[str],
) -> tuple[str | None, int | None, list[str]]:
    part_name = str(part.get("name") or part.get("item_name") or "")
    sources = [(part_name, part_voxel_path(data_root, rec, part_name, part))]
    sources.extend((name, part_voxel_path(data_root, rec, name)) for name in child_names)
    arrays: list[np.ndarray] = []
    missing: list[str] = []
    for name, path in sources:
        if not path.is_file():
            missing.append(f"{name}:{path}")
            continue
        coords = np.asarray(np.load(path, allow_pickle=False))
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"{path} expected [N,3] coords, got {coords.shape}")
        arrays.append(coords.astype(np.int64, copy=False))
    if len(arrays) <= 1:
        return None, None, missing
    union = np.unique(np.concatenate(arrays, axis=0), axis=0).astype(np.int64, copy=False)
    dst = (
        out_root
        / "merged_voxels"
        / str(rec["object_id"])
        / f"angle_{int(rec['angle_idx'])}"
        / "64"
        / f"ind_{part_name}.npy"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, union)
    return str(dst), int(union.shape[0]), missing


def repair_realappliance_manifest(
    *,
    data_root: Path,
    source_manifest: Path,
    out_manifest: Path,
    out_root: Path,
) -> dict[str, Any]:
    rows = read_jsonl(source_manifest)
    repaired: list[dict[str, Any]] = []
    rows_with_merges = 0
    parts_with_merges = 0
    child_labels_added = 0
    union_voxels_written = 0
    missing_union_sources: list[str] = []

    for rec in rows:
        out = copy.deepcopy(rec)
        parts_info = load_part_info(data_root, rec)
        target_parts = [part for part in out.get("target_parts") or [] if isinstance(part, dict)]
        target_names = {str(part.get("name") or "") for part in target_parts}
        target_parent_labels = {
            int(part.get("original_label", part.get("raw_label", 0)) or 0)
            for part in target_parts
        }
        label_remap = {int(k): int(v) for k, v in dict(out.get("label_remap") or {}).items()}
        any_merge = False
        merged_target_labels: list[int] = []
        seen_target_labels: set[int] = set()

        for slot, part in enumerate(target_parts, start=1):
            part_name = str(part.get("name") or part.get("item_name") or "")
            info = parts_info.get(part_name) if isinstance(parts_info.get(part_name), dict) else {}
            parent_label = int(info.get("label", part.get("original_label", part.get("raw_label", slot))) or slot)
            local_label = int(part.get("local_label", label_remap.get(parent_label, slot)) or slot)
            label_remap[parent_label] = local_label
            labels = [parent_label]
            if is_movable_target(part, info):
                children = fixed_children(
                    parts_info,
                    parent_name=part_name,
                    target_names=target_names,
                    target_parent_labels=target_parent_labels,
                )
            else:
                children = []
            child_names = [name for name, _meta in children]
            child_labels = sorted({
                int(meta["label"])
                for _name, meta in children
                if meta.get("label") is not None
            })
            for label in child_labels:
                if label not in labels:
                    labels.append(label)
                    label_remap[label] = local_label
            if child_labels:
                any_merge = True
                parts_with_merges += 1
                child_labels_added += len(child_labels)
                union_path, union_count, missing = write_union_voxel(
                    data_root=data_root,
                    out_root=out_root,
                    rec=rec,
                    part=part,
                    child_names=child_names,
                )
                missing_union_sources.extend(missing)
                part["prompt_original_labels"] = labels
                part["merged_original_labels"] = labels
                part["merged_child_parts"] = child_names
                part["merged_child_labels"] = child_labels
                if union_path is not None:
                    part.setdefault("paths", {})["part_voxel"] = union_path
                    part["merged_part_voxel"] = union_path
                    part["merged_raw_count"] = union_count
                    part["raw_count"] = union_count
                    union_voxels_written += 1
            for label in labels:
                if label not in seen_target_labels:
                    merged_target_labels.append(label)
                    seen_target_labels.add(label)
        out["target_parts"] = target_parts
        out["target_original_labels"] = merged_target_labels
        out["label_remap"] = {str(k): int(v) for k, v in sorted(label_remap.items())}
        if any_merge:
            rows_with_merges += 1
            out["fixed_child_union_v6"] = True
        repaired.append(out)

    write_jsonl(out_manifest, repaired)
    summary = {
        "source_manifest": str(source_manifest),
        "out_manifest": str(out_manifest),
        "rows": len(rows),
        "rows_with_merges": rows_with_merges,
        "parts_with_merges": parts_with_merges,
        "child_labels_added": child_labels_added,
        "union_voxels_written": union_voxels_written,
        "missing_union_sources": missing_union_sources[:200],
        "missing_union_source_count": len(missing_union_sources),
    }
    write_json(out_manifest.with_suffix(".summary.json"), summary)
    return summary


def object_refs_from_entries(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        dataset_id = str(entry["dataset_id"])
        obj_id = str(entry["obj_id"])
        key = f"{dataset_id}::{obj_id}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"dataset_id": dataset_id, "obj_id": obj_id, "object_key": key})
    return out


def build_v6_split(
    *,
    v5_index: Path,
    ra_manifest: Path,
    out_split: Path,
) -> dict[str, Any]:
    index = read_json(v5_index)
    train_rows = int(index["train_rows"])
    entries = list(index["entries"])
    datasets = copy.deepcopy(index["datasets"])
    for dataset in datasets:
        if str(dataset.get("dataset_id")) == "realappliance":
            dataset["manifest_paths"] = [str(ra_manifest)]
    split = {
        "name": "split_official_verse_realappliance_0511dd_v6_fixed_child_union",
        "source": {
            "v5_index": str(v5_index),
            "v5_split_json": str(index.get("split_json", "")),
            "note": "Reconstructed from v5 packed index train/heldout entry order; RealAppliance manifest replaced with fixed-child union v6.",
        },
        "datasets": datasets,
        "train_ids": object_refs_from_entries(entries[:train_rows]),
        "heldout_ids": object_refs_from_entries(entries[train_rows:]),
    }
    write_json(out_split, split)
    return {
        "out_split": str(out_split),
        "train_objects": len(split["train_ids"]),
        "heldout_objects": len(split["heldout_ids"]),
        "datasets": [
            {
                "dataset_id": item.get("dataset_id"),
                "manifest_paths": item.get("manifest_paths"),
            }
            for item in datasets
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-index", type=Path, default=DEFAULT_V5_INDEX)
    parser.add_argument("--realappliance-root", type=Path, default=DEFAULT_RA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--source-ra-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ra_root = Path(args.realappliance_root)
    source_ra_manifest = args.source_ra_manifest or (
        ra_root / "manifests" / "part_completion" / "arts_mllm_realappliance.train.jsonl"
    )
    out_root = Path(args.out_root)
    out_manifest = out_root / "realappliance" / "arts_mllm_realappliance.fixed_child_union_v6.train.jsonl"
    out_split = out_root / "split_official_verse_realappliance_0511dd_v6.json"
    manifest_summary = repair_realappliance_manifest(
        data_root=ra_root,
        source_manifest=Path(source_ra_manifest),
        out_manifest=out_manifest,
        out_root=out_root / "realappliance",
    )
    split_summary = build_v6_split(
        v5_index=Path(args.v5_index),
        ra_manifest=out_manifest,
        out_split=out_split,
    )
    summary = {"manifest": manifest_summary, "split": split_summary}
    write_json(out_root / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
