#!/usr/bin/env python3
"""Build the PhysX delivery manifest for dataset_toolkits.

The manifest is the downstream single source of truth for object / angle / part
availability across modalities. It deliberately records mask/bbox availability
separately from voxel availability because OmniPart voxel filtering can remove
small parts from `ind_<part>.npy` while their mask/bbox labels remain usable.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import PipelineConfig, load_config  # noqa: E402


# Mirror of pipeline/03_voxelize.py MIN_PART_VOXELS. Parts with <= this count
# are intentionally not persisted as ind_<part>.npy.
MIN_PART_VOXELS = 5
FILTER_REASON_BELOW_MIN_PART_VOXELS = "below_min_part_voxels_5"
SCHEMA_VERSION = "v1-physx"
PART_COMPLETE_NUM_VIEWS = 16
DEFAULT_VALIDATOR_REPORT = Path(
    ".planning/phases/02-validator/02-PHYSX-VALIDATION-REPORT.json"
)


@dataclass(frozen=True)
class PartSpec:
    canonical_name: str
    label: int
    bbox_aliases: tuple[str, ...]


def _load_json(path: Path, default: Any | None = None) -> Any:
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_part_specs(part_info_path: Path) -> tuple[str, int, list[PartSpec]]:
    """Load category, num_parts, and canonical part specs from part_info.json."""
    part_info = _load_json(part_info_path)
    category = str(part_info.get("category", ""))
    num_parts = int(part_info.get("num_parts", 0))
    label_to_key = part_info.get("label_to_key", {})
    parts = part_info.get("parts", {})

    specs: list[PartSpec] = []
    for part_index in range(num_parts):
        canonical_name = str(label_to_key[str(part_index)])
        part_entry = parts[canonical_name]
        label = int(part_entry.get("label", part_index + 1))
        part_type = str(part_entry.get("type", "")).strip()
        aliases = [canonical_name]
        if part_type:
            aliases.append(f"{part_type.replace('_', ' ').title().replace(' ', '_')}_{label}")
        aliases.append(canonical_name.replace("_", " ").title().replace(" ", "_"))
        specs.append(
            PartSpec(
                canonical_name=canonical_name,
                label=label,
                bbox_aliases=tuple(dict.fromkeys(aliases)),
            )
        )
    return category, num_parts, specs


def _relative_to_data_root(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root).as_posix()


def _num_voxels(ind_path: Path) -> int:
    arr = np.load(ind_path, mmap_mode="r")
    if arr.ndim == 0:
        return int(arr.size)
    return int(arr.shape[0])


def build_part_record(
    *,
    data_root: Path,
    name: str,
    label: int,
    ind_path: Path,
    bbox_gt_part_record: dict[str, Any] | None,
    mask_npy_exists: bool,
) -> dict[str, Any]:
    """Build one D-11.5 `parts.<name>` record."""
    has_voxel_ind = ind_path.is_file()
    num_voxels = _num_voxels(ind_path) if has_voxel_ind else 0

    if bbox_gt_part_record is None:
        num_visible_views = 0
        has_bbox = False
    else:
        views = bbox_gt_part_record.get("views", {})
        num_visible_views = sum(
            1 for view in views.values() if bool(view.get("visible", False))
        )
        has_bbox = num_visible_views > 0

    has_mask = bool(mask_npy_exists and num_visible_views > 0)

    return {
        "label": int(label),
        "has_voxel_ind": bool(has_voxel_ind),
        "has_mask": has_mask,
        "has_bbox": bool(has_bbox),
        "num_voxels": int(num_voxels),
        "num_visible_views": int(num_visible_views),
        "filter_reason": None if has_voxel_ind else FILTER_REASON_BELOW_MIN_PART_VOXELS,
        "voxel_ind_path": (
            _relative_to_data_root(ind_path, data_root) if has_voxel_ind else None
        ),
    }


def build_angle_record(
    cfg: PipelineConfig,
    oid: str,
    angle_idx: int,
    part_specs: list[PartSpec],
) -> dict[str, Any]:
    """Build one D-11.5 `angles.<idx>` record."""
    data_root = Path(cfg.data_root)
    renders = Path(cfg.renders_dir)
    recon = Path(cfg.reconstruction_dir)
    res = cfg.voxel.resolution

    angle_render_dir = renders / oid / f"angle_{angle_idx}"
    bbox_gt_abs = angle_render_dir / "bbox_gt.json"
    bbox_gt = _load_json(bbox_gt_abs, default={"parts": {}})
    bbox_parts = bbox_gt.get("parts", {}) if isinstance(bbox_gt, dict) else {}
    mask_npy_exists = (angle_render_dir / "mask" / "mask_0.npy").is_file()

    parts: dict[str, dict[str, Any]] = {}
    for spec in part_specs:
        ind_path = (
            recon
            / "voxel_expanded"
            / oid
            / f"angle_{angle_idx}"
            / str(res)
            / f"ind_{spec.canonical_name}.npy"
        )
        bbox_part_record = None
        for alias in spec.bbox_aliases:
            if alias in bbox_parts:
                bbox_part_record = bbox_parts[alias]
                break
        parts[spec.canonical_name] = build_part_record(
            data_root=data_root,
            name=spec.canonical_name,
            label=spec.label,
            ind_path=ind_path,
            bbox_gt_part_record=bbox_part_record,
            mask_npy_exists=mask_npy_exists,
        )

    voxel_surface_abs = (
        recon / "voxel_expanded" / oid / f"angle_{angle_idx}" / str(res) / "surface.npy"
    )
    joint_abs = Path(cfg.joint_transforms_dir) / f"{oid}.json"
    camera_abs = angle_render_dir / "camera_transforms.json"
    canonical_abs = (
        data_root
        / "canonical_transforms"
        / oid
        / f"angle_{angle_idx}"
        / "canonical_transform.json"
    )

    return {
        "voxel_surface_path": _relative_to_data_root(voxel_surface_abs, data_root),
        "rgb_path_template": _relative_to_data_root(
            angle_render_dir / "part_complete" / "rgb" / "view_{view_index}.png", data_root
        ),
        "mask_path_template": _relative_to_data_root(
            angle_render_dir / "part_complete" / "mask" / "label" / "mask_{view_index}.npy", data_root
        ),
        "bbox_gt_path": _relative_to_data_root(bbox_gt_abs, data_root),
        "joint_transforms_path": _relative_to_data_root(joint_abs, data_root),
        "canonical_transform_path": (
            _relative_to_data_root(canonical_abs, data_root) if canonical_abs.is_file() else None
        ),
        "camera_transforms_path": _relative_to_data_root(camera_abs, data_root),
        "num_views": PART_COMPLETE_NUM_VIEWS,
        "parts": parts,
    }


def build_object_record(cfg: PipelineConfig, oid: str) -> dict[str, Any]:
    """Build one D-11.5 `objects.<oid>` record."""
    part_info_path = Path(cfg.part_info_dir) / oid / "part_info.json"
    category, num_parts_total, part_specs = _load_part_specs(part_info_path)
    num_angles = cfg.get_num_angles(oid)

    angles: dict[str, dict[str, Any]] = {}
    voxel_kept_names: set[str] = set()
    for angle_idx in range(num_angles):
        angle_record = build_angle_record(cfg, oid, angle_idx, part_specs)
        angles[str(angle_idx)] = angle_record
        for part_name, part in angle_record["parts"].items():
            if part["has_voxel_ind"]:
                voxel_kept_names.add(part_name)

    return {
        "category": category,
        "kinematic_type": "articulated" if cfg.is_articulated(oid) else "static",
        "num_angles": int(num_angles),
        "num_parts_total": int(num_parts_total),
        # Union across all angles. This makes angle-specific filtering visible while
        # preserving a compact object-level count.
        "num_parts_voxel_kept": int(len(voxel_kept_names)),
        "angles": angles,
    }


def _validator_status(validator_report_path: Path | None) -> str:
    if validator_report_path is None or not validator_report_path.is_file():
        return "UNKNOWN"
    report = _load_json(validator_report_path)
    passed = report.get("summary", {}).get("passed")
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "PARTIAL"


def build_manifest(
    cfg: PipelineConfig,
    validator_report_path: Path | None = DEFAULT_VALIDATOR_REPORT,
) -> dict[str, Any]:
    """Build the full v1-physx delivery manifest."""
    object_ids = cfg.list_object_ids()
    objects: dict[str, dict[str, Any]] = {}

    total = len(object_ids)
    for idx, oid in enumerate(object_ids, start=1):
        objects[oid] = build_object_record(cfg, oid)
        if idx % 200 == 0 or idx == total:
            print(f"[manifest] processed {idx}/{total} objects", flush=True)

    total_angles = 0
    total_parts = 0
    parts_with_voxel = 0
    parts_filtered = 0
    for obj in objects.values():
        total_angles += int(obj["num_angles"])
        for angle in obj["angles"].values():
            for part in angle["parts"].values():
                total_parts += 1
                if part["has_voxel_ind"]:
                    parts_with_voxel += 1
                else:
                    parts_filtered += 1

    return {
        "dataset": cfg.dataset_name,
        "schema_version": SCHEMA_VERSION,
        "build_date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "voxel_resolution": int(cfg.voxel.resolution),
            "min_part_voxels_threshold": MIN_PART_VOXELS,
            "num_views_per_angle": PART_COMPLETE_NUM_VIEWS,
        },
        "objects": objects,
        "summary": {
            "total_objects": int(len(objects)),
            "total_angles": int(total_angles),
            "total_parts": int(total_parts),
            "parts_with_voxel": int(parts_with_voxel),
            "parts_filtered": int(parts_filtered),
            "validator_status": _validator_status(validator_report_path),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset_toolkits delivery manifest JSON."
    )
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument(
        "--validator-report",
        default=str(DEFAULT_VALIDATOR_REPORT),
        help=(
            "Validator report path used for summary.validator_status "
            "(default: .planning/phases/02-validator/02-PHYSX-VALIDATION-REPORT.json)."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: <data_root>/manifests/<dataset>.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    validator_report_path = Path(args.validator_report) if args.validator_report else None
    out_path = (
        Path(args.out)
        if args.out
        else Path(cfg.data_root) / "manifests" / f"{cfg.dataset_name}.json"
    )

    manifest = build_manifest(cfg, validator_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"[manifest] written {out_path} ({out_path.stat().st_size} bytes)")
    print(
        "[manifest] summary "
        f"objects={manifest['summary']['total_objects']} "
        f"angles={manifest['summary']['total_angles']} "
        f"parts={manifest['summary']['total_parts']} "
        f"parts_with_voxel={manifest['summary']['parts_with_voxel']} "
        f"parts_filtered={manifest['summary']['parts_filtered']} "
        f"validator_status={manifest['summary']['validator_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
