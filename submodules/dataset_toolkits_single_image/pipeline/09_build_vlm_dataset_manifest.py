#!/usr/bin/env python3
"""Build single-image VLM JSONL from part_complete 16-view assets.

Contract for the new VLM branch:
- reuse ``renders/<object>/angle_i/part_complete/rgb/view_*.png``;
- use the first N fixed part_complete views (default N=8);
- one RGB view is one training sample;
- bbox labels come from part_complete valid-part masks, not historical quadrant bbox_gt.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import load_config  # noqa: E402
from vlm_manifest_helpers import (  # noqa: E402
    build_answer_json,
    build_components_tree,
    build_sample,
    filter_components_tree_for_voxel_kept_parts,
    get_manifest_angle_parts,
    resolve_remote_renders_root,
)


MIN_PART_VOXELS = 5
DEFAULT_VIEW_COUNT = 8
RENDER_SET = "part_complete"


@dataclass(frozen=True)
class VisiblePart:
    key: str
    label: int
    bbox: list[int]
    pixels: int


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"json[{path}] must be an object")
    return payload


def _parse_csv(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",")]
    if not values or any(not item for item in values):
        raise ValueError("comma-separated value must contain non-empty items")
    if len(values) != len(set(values)):
        raise ValueError("comma-separated value contains duplicates")
    return values


def _parse_object_ids(config, raw_value: str | None) -> list[str]:
    available = config.list_object_ids()
    if raw_value is None:
        return available
    requested = _parse_csv(raw_value)
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError("Unknown or filtered-out object IDs: " + ", ".join(missing))
    return requested


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace(" ", "_")


def _default_manifest_path(config) -> Path:
    return Path(config.data_root) / "manifests" / f"{config.dataset_name}.json"


def _default_output_path(config, view_count: int) -> Path:
    slug = _dataset_slug(config.dataset_name)
    return Path(config.vlm_dir) / "training_json" / f"arts_mllm_{slug}_part_complete_{view_count}view_1img.jsonl"


def _remote_renders_root(data_root: str, image_prefix: str) -> str:
    return resolve_remote_renders_root(data_root, image_prefix)


def _manifest_angle_parts(manifest: dict[str, Any], object_id: str, angle_idx: int) -> dict[str, Any]:
    return get_manifest_angle_parts(manifest, object_id, angle_idx)


def _valid_target_keys(manifest_parts: dict[str, Any], components_tree: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    components = components_tree["components"]
    for key, comp in components.items():
        if not isinstance(comp, dict) or comp.get("parent") is None:
            continue
        part_record = manifest_parts.get(key)
        if not isinstance(part_record, dict):
            continue
        if not part_record.get("has_voxel_ind"):
            continue
        if int(part_record.get("num_voxels", 0)) <= MIN_PART_VOXELS:
            continue
        keys.append(key)
    return keys


def _bbox_from_mask(mask: np.ndarray) -> tuple[list[int] | None, int]:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")
    ys, xs = np.nonzero(mask > 0)
    pixels = int(len(xs))
    if pixels == 0:
        return None, 0
    h, w = mask.shape
    x1 = int(np.floor(float(xs.min()) * 1000.0 / float(w)))
    x2 = int(np.ceil(float(xs.max() + 1) * 1000.0 / float(w)))
    y1 = int(np.floor(float(ys.min()) * 1000.0 / float(h)))
    y2 = int(np.ceil(float(ys.max() + 1) * 1000.0 / float(h)))
    bbox = [
        max(0, min(999, x1)),
        max(0, min(999, y1)),
        max(1, min(1000, x2)),
        max(1, min(1000, y2)),
    ]
    if bbox[2] <= bbox[0]:
        bbox[2] = min(1000, bbox[0] + 1)
    if bbox[3] <= bbox[1]:
        bbox[3] = min(1000, bbox[1] + 1)
    return bbox, pixels


def _load_visible_parts(
    *,
    data_root: Path,
    object_id: str,
    angle_idx: int,
    view_idx: int,
    valid_target_keys: list[str],
    part_info: dict[str, Any],
) -> list[VisiblePart]:
    parts_payload = part_info.get("parts")
    if not isinstance(parts_payload, dict):
        raise TypeError(f"part_info parts missing for {object_id}")
    visible: list[VisiblePart] = []
    for key in valid_target_keys:
        part_payload = parts_payload.get(key)
        if not isinstance(part_payload, dict):
            raise KeyError(f"part_info missing target part {key!r}")
        label = int(part_payload.get("label"))
        mask_path = data_root / "renders" / object_id / f"angle_{angle_idx}" / RENDER_SET / "mask" / key / f"mask_{view_idx}.npy"
        if not mask_path.is_file():
            continue
        mask = np.load(mask_path)
        bbox, pixels = _bbox_from_mask(mask)
        if bbox is None or pixels <= 0:
            continue
        visible.append(VisiblePart(key=key, label=label, bbox=bbox, pixels=pixels))
    return visible


def _write_label_mask(
    *,
    data_root: Path,
    object_id: str,
    angle_idx: int,
    view_idx: int,
    visible_parts: list[VisiblePart],
) -> None:
    label_mask: np.ndarray | None = None
    base = data_root / "renders" / object_id / f"angle_{angle_idx}" / RENDER_SET / "mask"
    for part in visible_parts:
        mask_path = base / part.key / f"mask_{view_idx}.npy"
        if not mask_path.is_file():
            continue
        mask = np.load(mask_path)
        if label_mask is None:
            label_mask = np.zeros(mask.shape, dtype=np.int32)
        label_mask[mask > 0] = int(part.label)
    if label_mask is None:
        return
    out_dir = base / "label"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"mask_{view_idx}.npy", label_mask)


def _filter_tree_to_visible_parts(components_tree: dict[str, Any], visible_parts: list[VisiblePart]) -> dict[str, Any] | None:
    visible_keys = {part.key for part in visible_parts}
    if not visible_keys:
        return None
    root_key = str(components_tree["name"])
    components = components_tree["components"]
    root = dict(components[root_key])
    root["children"] = []
    filtered_components: dict[str, Any] = {root_key: root}

    for key in components:
        if key == root_key or key not in visible_keys:
            continue
        comp = dict(components[key])
        parent = comp.get("parent")
        if parent not in visible_keys and parent != root_key:
            parent = root_key
        comp["parent"] = parent
        comp["children"] = [child for child in comp.get("children", []) if child in visible_keys]
        filtered_components[key] = comp

    for key, comp in filtered_components.items():
        if key == root_key:
            continue
        parent = comp.get("parent")
        if parent in filtered_components:
            filtered_components[parent].setdefault("children", []).append(key)
    for comp in filtered_components.values():
        comp["children"] = sorted(set(comp.get("children", [])))

    bbox_map = components_tree.get("_bbox_key_by_component", {})
    filtered_bbox_map = {
        key: value
        for key, value in bbox_map.items()
        if key in visible_keys
    } if isinstance(bbox_map, dict) else {}

    return {
        "name": components_tree["name"],
        "category": components_tree["category"],
        "dimension": components_tree["dimension"],
        "components": filtered_components,
        "_bbox_key_by_component": filtered_bbox_map,
    }


def _bbox_gt_for_visible_parts(visible_parts: list[VisiblePart], view_idx: int) -> dict[str, Any]:
    return {
        "parts": {
            part.key: {
                "views": {
                    str(view_idx): {
                        "visible": True,
                        "bbox": part.bbox,
                        "pixel_count": part.pixels,
                    }
                }
            }
            for part in visible_parts
        }
    }


def build_dataset(config, object_ids: list[str], manifest_path: Path, output_path: Path, view_count: int) -> dict[str, Any]:
    data_root = Path(config.data_root)
    finaljson_dir = Path(config.finaljson_dir)
    part_info_dir = Path(config.part_info_dir)
    manifest = _load_json(manifest_path)
    image_root = _remote_renders_root(config.data_root, config.vlm.image_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    skipped_no_visible = 0
    skipped_missing_rgb = 0
    output_objects: set[str] = set()

    with output_path.open("w", encoding="utf-8") as handle:
        for object_idx, object_id in enumerate(object_ids, start=1):
            finaljson_path = finaljson_dir / f"{object_id}.json"
            part_info_path = part_info_dir / object_id / "part_info.json"
            finaljson = _load_json(finaljson_path)
            part_info = _load_json(part_info_path)
            components_tree = build_components_tree(finaljson, object_id, part_info)
            if components_tree is None:
                continue

            object_rows = 0
            for angle_idx in range(config.get_num_angles(object_id)):
                manifest_parts = _manifest_angle_parts(manifest, object_id, angle_idx)
                voxel_tree, _filtered_count = filter_components_tree_for_voxel_kept_parts(
                    components_tree,
                    manifest_parts,
                    object_id=object_id,
                    angle_idx=angle_idx,
                )
                if voxel_tree is None:
                    continue
                valid_keys = _valid_target_keys(manifest_parts, voxel_tree)
                if not valid_keys:
                    continue

                for view_idx in range(view_count):
                    rgb_path = data_root / "renders" / object_id / f"angle_{angle_idx}" / RENDER_SET / "rgb" / f"view_{view_idx}.png"
                    if not rgb_path.is_file():
                        skipped_missing_rgb += 1
                        continue
                    visible_parts = _load_visible_parts(
                        data_root=data_root,
                        object_id=object_id,
                        angle_idx=angle_idx,
                        view_idx=view_idx,
                        valid_target_keys=valid_keys,
                        part_info=part_info,
                    )
                    if not visible_parts:
                        skipped_no_visible += 1
                        continue
                    visible_tree = _filter_tree_to_visible_parts(voxel_tree, visible_parts)
                    if visible_tree is None:
                        skipped_no_visible += 1
                        continue
                    _write_label_mask(
                        data_root=data_root,
                        object_id=object_id,
                        angle_idx=angle_idx,
                        view_idx=view_idx,
                        visible_parts=visible_parts,
                    )
                    selected_views = [{"view_index": view_idx, "render_set": RENDER_SET}]
                    bbox_gt = _bbox_gt_for_visible_parts(visible_parts, view_idx)
                    answer = build_answer_json(visible_tree, selected_views, bbox_gt)
                    image = f"{image_root}/{object_id}/angle_{angle_idx}/{RENDER_SET}/rgb/view_{view_idx}.png"
                    sample_id = f"{_dataset_slug(config.dataset_name)}_{object_id}_angle_{angle_idx}_view_{view_idx}"
                    record = build_sample(sample_id, [image], answer)
                    record["source_render_set"] = f"{RENDER_SET}_16_first_{view_count}"
                    record["object_id"] = object_id
                    record["angle_index"] = angle_idx
                    record["view_index"] = view_idx
                    record["visible_target_parts"] = [part.key for part in visible_parts]
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows += 1
                    object_rows += 1
            if object_rows:
                output_objects.add(object_id)
            print(f"[{object_idx}/{len(object_ids)}] {object_id} rows={object_rows}", flush=True)

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta = {
        "schema_version": "v1-vlm-part-complete-single-view",
        "dataset": config.dataset_name,
        "render_set": RENDER_SET,
        "view_policy": f"first_{view_count}_part_complete_views",
        "sample_policy": "one RGB view is one training sample; targets are visible valid movable parts only",
        "manifest": str(manifest_path),
        "output_jsonl": str(output_path),
        "object_count": len(object_ids),
        "output_object_count": len(output_objects),
        "sample_count": rows,
        "skipped_no_visible_target": skipped_no_visible,
        "skipped_missing_rgb": skipped_missing_rgb,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    meta["meta_path"] = str(meta_path)
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build part_complete first-8-view single-image VLM JSONL.")
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument("--manifest", help="Delivery manifest path. Default: <data_root>/manifests/<dataset>.json")
    parser.add_argument("--object-ids", help="Optional comma-separated object IDs.")
    parser.add_argument("--view-count", type=int, default=DEFAULT_VIEW_COUNT, help="Use first N part_complete views. Default 8.")
    parser.add_argument("--out", help="Output JSONL path. Default: vlm/training_json/arts_mllm_<dataset>_part_complete_<N>view_1img.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.view_count < 1 or args.view_count > 16:
        raise ValueError("--view-count must be in [1, 16]")
    object_ids = _parse_object_ids(config, args.object_ids)
    manifest_path = Path(args.manifest) if args.manifest else _default_manifest_path(config)
    output_path = Path(args.out) if args.out else _default_output_path(config, args.view_count)
    meta = build_dataset(config, object_ids, manifest_path, output_path, args.view_count)
    print(
        "[vlm-part-complete] written "
        f"{meta['output_jsonl']} samples={meta['sample_count']} "
        f"objects={meta['output_object_count']}/{meta['object_count']} "
        f"skipped_no_visible={meta['skipped_no_visible_target']} "
        f"skipped_missing_rgb={meta['skipped_missing_rgb']}"
    )
    print(f"[vlm-part-complete] meta {meta['meta_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
