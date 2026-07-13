#!/usr/bin/env python3
"""Build a Part Completion training manifest from part_complete renders.

This Step 10 does not depend on the VLM JSONL.  It uses the Step 04
valid-parts manifest plus the part_complete assets produced by Step 05 render:

    renders/<object_id>/angle_<i>/part_complete/rgb/view_<j>.png
    renders/<object_id>/angle_<i>/part_complete/mask/<part_key>/mask_<j>.npy
    renders/<object_id>/angle_<i>/part_complete/mask/remaining/mask_<j>.npy
    reconstruction/dinov2_tokens/<object_id>/angle_<i>/part_complete/tokens.npz

Every emitted row is one training sample:

    1 RGB image + 1 label mask + the DINO token path/view index + the voxel
    paths for target parts visible in that exact view.

The label mask contract is:

    0 = background
    part_info.label = visible valid movable target component
    remaining_label = merged visible object pixels not belonging to valid targets

Only valid target components are allowed to contribute target voxels:

    movable part (finaljson group_info motion type A/B/C)
    ∩ Step 04 manifest has_voxel_ind=true
    ∩ num_voxels > 5
    ∩ visible in this view's part_complete mask

Output:
    <data_root>/manifests/part_completion/arts_pc_<dataset>_train.jsonl
        row["dinov2"].tokens_path = reconstruction/dinov2_tokens/<object_id>/angle_<i>/part_complete/tokens.npz
        row["dinov2"].view_index = view_idx
    <data_root>/manifests/part_completion/manifest_meta.json
    <data_root>/manifests/part_completion/skip_report.json

Derived label masks are written beside the part_complete masks by default:

    renders/<object_id>/angle_<i>/part_complete/mask/label/mask_<j>.npy
    renders/<object_id>/angle_<i>/part_complete/mask/label/mask_<j>.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional visualization only
    Image = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import load_config  # noqa: E402


MANIFEST_VERSION = 2
TASK_NAME = "part_completion"
ACCEPTED_MOTION_TYPES = {"A", "B", "C"}
MIN_PART_VOXELS = 5
DEFAULT_PART_COMPLETE_SUBDIR = "part_complete"
DEFAULT_LABEL_MASK_SUBDIR = "label"
REMAINING_MASK_DIR = "remaining"
EXPECTED_VIEW_COUNT = 16
VOXEL_RESOLUTION = 64
EXPECTED_FILTER_SKIP_REASONS = frozenset(
    {
        "no_visible_target_part",
        "no_manifest_valid_movable_parts",
    }
)


@dataclass(frozen=True)
class ValidPart:
    name: str
    part_index: int
    label: int
    raw_label: int
    obj_files: tuple[str, ...]
    joint: str
    joint_type: str
    motion_type: str
    group_id: str
    type_name: str | None
    num_voxels: int
    voxel_ind_path: str


@dataclass(frozen=True)
class BuildContext:
    config: Any
    data_root: Path
    dataset_slug: str
    manifest: dict[str, Any]
    part_complete_subdir: str
    label_mask_subdir: str
    min_visible_pixels: int
    write_label_masks: bool
    require_dinov2_tokens: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build single-view Part Completion manifest from part_complete RGB/masks."
    )
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument(
        "--manifest",
        help="Step 04 valid-parts manifest path. Default: <data_root>/manifests/<dataset>.json.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Default: <data_root>/manifests/part_completion",
    )
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset.")
    parser.add_argument("--angle-ids", help="Optional comma-separated angle subset.")
    parser.add_argument("--view-ids", help="Optional comma-separated view subset, default 0..15.")
    parser.add_argument(
        "--part-complete-subdir",
        default=DEFAULT_PART_COMPLETE_SUBDIR,
        help="Subdirectory under renders/<object_id>/angle_<i>/ containing RGB/masks.",
    )
    parser.add_argument(
        "--label-mask-subdir",
        default=DEFAULT_LABEL_MASK_SUBDIR,
        help="Subdirectory under part_complete/mask/ for derived label masks.",
    )
    parser.add_argument(
        "--min-visible-pixels",
        type=int,
        default=1,
        help="Minimum pixels in a part mask for that part to be included in this view sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing train manifest/meta/skip report outputs.",
    )
    parser.add_argument(
        "--strict-zero-skips",
        action="store_true",
        help=(
            "Fail if any candidate view is skipped. By default, expected dataset "
            "eligibility filters such as invisible target parts are allowed, while "
            "missing/corrupt assets remain fatal."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate and validate rows without writing label masks or output files.",
    )
    parser.add_argument(
        "--allow-missing-dinov2",
        action="store_true",
        help=(
            "Do not skip samples when reconstruction/dinov2_tokens/<object>/angle_i/"
            "part_complete/tokens.npz is missing. The manifest still records the "
            "expected path and availability flag. Default is strict: require tokens."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N candidate views; set 0 to disable. Default: 1000.",
    )
    return parser.parse_args(argv)


def _unexpected_skip_reasons(skip_counter: Counter[str]) -> Counter[str]:
    return Counter(
        {
            reason: count
            for reason, count in skip_counter.items()
            if reason not in EXPECTED_FILTER_SKIP_REASONS
        }
    )


def _has_fatal_skips(skip_counter: Counter[str], *, strict_zero_skips: bool) -> bool:
    if not skip_counter:
        return False
    if strict_zero_skips:
        return True
    return bool(_unexpected_skip_reasons(skip_counter))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace(" ", "_")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be null or a non-empty string")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _load_json(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return _require_mapping(json.load(fh), f"json[{path}]")


def _relative_to_data_root(path: Path, data_root: Path) -> str:
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return path.as_posix()


def _validate_simple_name(name: str, field_name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"{field_name} must be a simple path segment, got {name!r}")
    return name


def _parse_csv(value: str, field_name: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise ValueError(f"{field_name} must be a comma-separated list of non-empty values")
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} contains duplicate values")
    return items


def _parse_int_ids(raw_value: str | None, field_name: str) -> list[int] | None:
    if raw_value is None:
        return None
    out: list[int] = []
    for item in _parse_csv(raw_value, field_name):
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{field_name} values must be integers, got {item!r}") from exc
        if value < 0:
            raise ValueError(f"{field_name} values must be >= 0, got {value}")
        out.append(value)
    return out


def _resolve_object_ids(config: Any, object_ids_arg: str | None) -> list[str]:
    available_object_ids = config.list_object_ids()
    if object_ids_arg is None:
        return available_object_ids

    requested_object_ids = _parse_csv(object_ids_arg, "--object-ids")
    available_set = set(available_object_ids)
    unknown_or_filtered = [
        object_id for object_id in requested_object_ids if object_id not in available_set
    ]
    if unknown_or_filtered:
        raise ValueError(
            "Unknown or filtered-out object IDs in --object-ids: "
            + ", ".join(unknown_or_filtered)
        )
    return requested_object_ids


def _manifest_path(config: Any, manifest_arg: str | None) -> Path:
    if manifest_arg:
        return Path(manifest_arg)
    return Path(config.data_root) / "manifests" / f"{config.dataset_name}.json"


def _load_manifest(config: Any, manifest_arg: str | None) -> tuple[Path, dict[str, Any]]:
    path = _manifest_path(config, manifest_arg)
    manifest = _load_json(path, "Step 04 valid-parts manifest")
    _require_mapping(manifest.get("objects"), f"manifest[{path}].objects")
    _require_mapping(manifest.get("summary"), f"manifest[{path}].summary")
    return path, manifest


def _manifest_angle_parts(manifest: dict[str, Any], object_id: str, angle_idx: int) -> dict[str, Any]:
    objects = _require_mapping(manifest.get("objects"), "manifest.objects")
    object_record = _require_mapping(objects.get(object_id), f"manifest.objects[{object_id!r}]")
    angles = _require_mapping(object_record.get("angles"), f"manifest.objects[{object_id!r}].angles")
    angle_record = _require_mapping(
        angles.get(str(angle_idx)),
        f"manifest.objects[{object_id!r}].angles[{angle_idx!r}]",
    )
    return _require_mapping(
        angle_record.get("parts"),
        f"manifest.objects[{object_id!r}].angles[{angle_idx!r}].parts",
    )


def _is_manifest_voxel_kept(part_record: dict[str, Any], part_key: str, context: str) -> bool:
    has_voxel_ind = part_record.get("has_voxel_ind")
    if not isinstance(has_voxel_ind, bool):
        raise TypeError(f"{context}[{part_key!r}].has_voxel_ind must be bool")
    num_voxels = _require_int(part_record.get("num_voxels"), f"{context}[{part_key!r}].num_voxels")
    voxel_ind_path = part_record.get("voxel_ind_path")
    if not has_voxel_ind:
        return False
    if num_voxels <= MIN_PART_VOXELS:
        raise ValueError(
            f"{context}[{part_key!r}] has_voxel_ind=true but num_voxels={num_voxels} "
            f"<= MIN_PART_VOXELS={MIN_PART_VOXELS}"
        )
    if not isinstance(voxel_ind_path, str) or not voxel_ind_path:
        raise ValueError(f"{context}[{part_key!r}] has_voxel_ind=true but voxel_ind_path is empty")
    return True


def _movable_link_ids(finaljson: dict[str, Any]) -> dict[int, tuple[str, str]]:
    group_info = _require_mapping(finaljson.get("group_info"), "finaljson.group_info")
    movable: dict[int, tuple[str, str]] = {}
    for raw_group_id, raw_group in group_info.items():
        group_id = str(raw_group_id)
        if not isinstance(raw_group, list) or len(raw_group) != 4:
            continue
        links, _parent_group, _params, motion_type_raw = raw_group
        motion_type = str(motion_type_raw)
        if motion_type not in ACCEPTED_MOTION_TYPES:
            continue
        if isinstance(links, int):
            link_ids = [links]
        elif isinstance(links, list):
            link_ids = [_require_int(item, f"finaljson.group_info[{group_id!r}][0][]") for item in links]
        else:
            raise TypeError(f"finaljson.group_info[{group_id!r}][0] must be int or list")
        for link_id in link_ids:
            movable[link_id] = (group_id, motion_type)
    return movable


def _load_valid_parts(ctx: BuildContext, object_id: str, angle_idx: int) -> tuple[list[ValidPart], dict[str, Any]]:
    finaljson_path = Path(ctx.config.finaljson_dir) / f"{object_id}.json"
    part_info_path = Path(ctx.config.part_info_dir) / object_id / "part_info.json"
    finaljson = _load_json(finaljson_path, f"finaljson[{object_id}]")
    part_info = _load_json(part_info_path, f"part_info[{object_id}]")
    label_to_key = _require_mapping(part_info.get("label_to_key"), f"part_info[{object_id}].label_to_key")
    parts = _require_mapping(part_info.get("parts"), f"part_info[{object_id}].parts")
    movable_by_raw_label = _movable_link_ids(finaljson)
    manifest_parts = _manifest_angle_parts(ctx.manifest, object_id, angle_idx)

    valid_parts: list[ValidPart] = []
    for raw_label, group_motion in sorted(movable_by_raw_label.items()):
        part_key_raw = label_to_key.get(str(raw_label))
        if not isinstance(part_key_raw, str) or not part_key_raw:
            raise KeyError(f"Object {object_id}: movable raw label {raw_label} has no part_info label_to_key entry")
        part_key = part_key_raw
        part_entry = _require_mapping(parts.get(part_key), f"part_info[{object_id}].parts[{part_key!r}]")
        part_record_raw = manifest_parts.get(part_key)
        if part_record_raw is None:
            raise KeyError(f"manifest object {object_id} angle_{angle_idx} missing movable part {part_key!r}")
        part_record = _require_mapping(part_record_raw, f"manifest.parts[{part_key!r}]")
        if not _is_manifest_voxel_kept(part_record, part_key, "manifest.parts"):
            continue

        label = _require_int(part_entry.get("label"), f"part_info[{object_id}].parts[{part_key!r}].label")
        raw_label_from_part = _require_int(part_entry.get("raw_label"), f"part_info[{object_id}].parts[{part_key!r}].raw_label")
        if label <= 0:
            raise ValueError(f"part_info[{object_id}].parts[{part_key!r}].label must be positive")
        if raw_label_from_part != raw_label or raw_label_from_part != label - 1:
            raise ValueError(
                f"part_info[{object_id}].parts[{part_key!r}] label/raw_label mismatch: "
                f"label={label}, raw_label={raw_label_from_part}, expected_raw={raw_label}"
            )
        label_roundtrip = _require_string(
            label_to_key.get(str(raw_label_from_part)),
            f"part_info[{object_id}].label_to_key[{raw_label_from_part!r}]",
        )
        if label_roundtrip != part_key:
            raise ValueError(
                f"part_info[{object_id}] label_to_key round-trip mismatch for {part_key!r}: "
                f"raw_label {raw_label_from_part} maps to {label_roundtrip!r}"
            )
        joint = _require_string(part_entry.get("joint"), f"part_info[{object_id}].parts[{part_key!r}].joint")
        joint_type = _require_string(part_entry.get("joint_type"), f"part_info[{object_id}].parts[{part_key!r}].joint_type")
        if joint == "fixed" or joint_type == "E":
            raise ValueError(f"Object {object_id}: manifest-valid movable part {part_key!r} is fixed in part_info")
        obj_files = tuple(
            _require_string(item, f"part_info[{object_id}].parts[{part_key!r}].obj_files[]")
            for item in _require_list(part_entry.get("obj_files"), f"part_info[{object_id}].parts[{part_key!r}].obj_files")
        )
        if not obj_files:
            raise ValueError(f"part_info[{object_id}].parts[{part_key!r}].obj_files must not be empty")
        group_id, motion_type = group_motion
        valid_parts.append(
            ValidPart(
                name=part_key,
                part_index=_require_int(part_entry.get("part_index"), f"part_info[{object_id}].parts[{part_key!r}].part_index"),
                label=label,
                raw_label=raw_label_from_part,
                obj_files=obj_files,
                joint=joint,
                joint_type=joint_type,
                motion_type=motion_type,
                group_id=group_id,
                type_name=_require_optional_string(part_entry.get("type"), f"part_info[{object_id}].parts[{part_key!r}].type"),
                num_voxels=_require_int(part_record.get("num_voxels"), f"manifest.parts[{part_key!r}].num_voxels"),
                voxel_ind_path=_require_string(part_record.get("voxel_ind_path"), f"manifest.parts[{part_key!r}].voxel_ind_path"),
            )
        )

    labels = [part.label for part in valid_parts]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Object {object_id} angle_{angle_idx} duplicate valid-part labels: {labels}")
    valid_parts.sort(key=lambda part: (part.part_index, part.name))
    return valid_parts, {"finaljson": finaljson, "part_info": part_info}


def _load_binary_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"mask not found: {path}")
    mask = np.load(path)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D: {path} got shape {mask.shape}")
    unique = set(np.unique(mask).tolist())
    if unique - {0, 1, False, True}:
        raise ValueError(f"mask must be binary 0/1: {path} unique={sorted(unique)}")
    return mask.astype(bool)


def _save_label_mask_png(path: Path, label_mask: np.ndarray) -> None:
    if Image is None:
        return
    max_label = int(label_mask.max()) if label_mask.size else 0
    if max_label <= 0:
        rgb = np.zeros((*label_mask.shape, 3), dtype=np.uint8)
    else:
        palette = np.array(
            [
                (0, 0, 0),
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (128, 0, 0),
                (0, 128, 0),
                (0, 0, 128),
                (255, 128, 0),
                (128, 255, 0),
                (0, 128, 255),
            ],
            dtype=np.uint8,
        )
        rgb = palette[label_mask.astype(np.int64) % len(palette)]
    Image.fromarray(rgb).save(path)


def _load_remaining_label(part_complete_dir: Path, target_labels: list[int]) -> int:
    labels_path = part_complete_dir / "mask_labels.json"
    if labels_path.is_file():
        labels = _load_json(labels_path, f"mask_labels[{part_complete_dir}]")
        remaining = labels.get("remaining")
        if isinstance(remaining, dict):
            remaining_label = remaining.get("blender_pass_index")
            if isinstance(remaining_label, int) and not isinstance(remaining_label, bool):
                if remaining_label <= 0:
                    raise ValueError(f"remaining blender_pass_index must be positive: {labels_path}")
                if remaining_label in target_labels:
                    raise ValueError(
                        f"remaining blender_pass_index {remaining_label} conflicts with target labels {target_labels}: {labels_path}"
                    )
                return int(remaining_label)
    return max(target_labels, default=0) + 1


def _motion_payload(part: ValidPart) -> dict[str, Any]:
    # Keep this minimal and grounded in finaljson's motion family.  Exact axes/ranges
    # remain available from part_info/joint_transforms if a later stage needs them.
    if part.motion_type == "C":
        motion_type = "rotate"
    elif part.motion_type == "B":
        motion_type = "prismatic"
    else:
        motion_type = "articulated"
    return {
        "motion_type": motion_type,
        "finaljson_motion_type": part.motion_type,
        "group_id": part.group_id,
    }


def _build_view_row(
    ctx: BuildContext,
    *,
    object_id: str,
    angle_idx: int,
    view_idx: int,
    valid_parts: list[ValidPart],
    finaljson: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if not valid_parts:
        return None, ["no_manifest_valid_movable_parts"]

    renders_root = Path(ctx.config.renders_dir)
    part_complete_dir = renders_root / object_id / f"angle_{angle_idx}" / ctx.part_complete_subdir
    rgb_path = part_complete_dir / "rgb" / f"view_{view_idx}.png"
    if not rgb_path.is_file():
        return None, ["missing_part_complete_rgb"]

    mask_root = part_complete_dir / "mask"
    label_mask_dir = mask_root / ctx.label_mask_subdir
    if ctx.label_mask_subdir in {REMAINING_MASK_DIR, *[part.name for part in valid_parts]}:
        raise ValueError(f"label mask subdir {ctx.label_mask_subdir!r} conflicts with a reserved mask component")

    visible_parts: list[tuple[ValidPart, int, Path]] = []
    part_masks: list[tuple[ValidPart, np.ndarray, int, Path]] = []
    expected_shape: tuple[int, int] | None = None
    for part in valid_parts:
        _validate_simple_name(part.name, f"part key {part.name!r}")
        part_mask_path = mask_root / part.name / f"mask_{view_idx}.npy"
        try:
            mask = _load_binary_mask(part_mask_path)
        except FileNotFoundError:
            return None, ["missing_part_complete_part_mask"]
        if expected_shape is None:
            expected_shape = tuple(mask.shape)  # type: ignore[assignment]
        elif tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"mask shape mismatch for {object_id} angle_{angle_idx} view_{view_idx}: "
                f"{part_mask_path} got {mask.shape}, expected {expected_shape}"
            )
        pixels = int(mask.sum())
        part_masks.append((part, mask, pixels, part_mask_path))
        if pixels >= ctx.min_visible_pixels:
            visible_parts.append((part, pixels, part_mask_path))

    if not visible_parts:
        return None, ["no_visible_target_part"]
    if expected_shape is None:
        return None, ["no_part_masks"]

    remaining_mask_path = mask_root / REMAINING_MASK_DIR / f"mask_{view_idx}.npy"
    try:
        remaining_mask = _load_binary_mask(remaining_mask_path)
    except FileNotFoundError:
        return None, ["missing_part_complete_remaining_mask"]
    if tuple(remaining_mask.shape) != expected_shape:
        raise ValueError(
            f"remaining mask shape mismatch for {object_id} angle_{angle_idx} view_{view_idx}: "
            f"{remaining_mask_path} got {remaining_mask.shape}, expected {expected_shape}"
        )

    target_labels = [part.label for part in valid_parts]
    remaining_label = _load_remaining_label(part_complete_dir, target_labels)
    label_mask = np.zeros(expected_shape, dtype=np.int32)
    for part, mask, pixels, part_mask_path in part_masks:
        if pixels < ctx.min_visible_pixels:
            continue
        if np.any((label_mask != 0) & mask):
            raise ValueError(
                f"overlapping visible part masks for {object_id} angle_{angle_idx} view_{view_idx}: {part_mask_path}"
            )
        label_mask[mask] = part.label
    if np.any((label_mask != 0) & remaining_mask):
        raise ValueError(f"remaining mask overlaps target mask for {object_id} angle_{angle_idx} view_{view_idx}")
    label_mask[remaining_mask] = remaining_label

    label_mask_path = label_mask_dir / f"mask_{view_idx}.npy"
    label_mask_png_path = label_mask_dir / f"mask_{view_idx}.png"
    if ctx.write_label_masks:
        label_mask_dir.mkdir(parents=True, exist_ok=True)
        np.save(label_mask_path, label_mask)
        _save_label_mask_png(label_mask_png_path, label_mask)

    voxel_root = ctx.data_root / "reconstruction" / "voxel_expanded" / object_id / f"angle_{angle_idx}" / str(VOXEL_RESOLUTION)
    overall_surface_path = voxel_root / "surface.npy"
    if not overall_surface_path.is_file():
        reasons.append("missing_overall_surface")

    dinov2_tokens_path = (
        ctx.data_root
        / "reconstruction"
        / "dinov2_tokens"
        / object_id
        / f"angle_{angle_idx}"
        / "part_complete"
        / "tokens.npz"
    )
    dinov2_tokens_meta_path = dinov2_tokens_path.parent / "tokens_npz_meta.json"
    dinov2_tokens_available = dinov2_tokens_path.is_file()
    if ctx.require_dinov2_tokens and not dinov2_tokens_available:
        reasons.append("missing_part_complete_dinov2_tokens")

    target_parts: list[dict[str, Any]] = []
    target_original_labels: list[int] = []
    label_to_component: dict[str, str] = {str(remaining_label): REMAINING_MASK_DIR}
    visible_pixels_by_part: dict[str, int] = {}
    separated_mask_paths: dict[str, str] = {}
    target_voxel_paths: list[str] = []

    for local_idx, (part, visible_pixels, part_mask_path) in enumerate(visible_parts, start=1):
        part_voxel_path = voxel_root / f"ind_{part.name}.npy"
        if not part_voxel_path.is_file():
            reasons.append("missing_visible_part_voxel")
        target_original_labels.append(part.label)
        label_to_component[str(part.label)] = part.name
        visible_pixels_by_part[part.name] = visible_pixels
        separated_mask_paths[part.name] = _relative_to_data_root(part_mask_path, ctx.data_root)
        target_voxel_paths.append(_relative_to_data_root(part_voxel_path, ctx.data_root))
        target_parts.append(
            {
                "name": part.name,
                "type": part.type_name,
                "original_label": part.label,
                "label": part.label,
                "local_label": local_idx,
                "part_index": part.part_index,
                "raw_label": part.raw_label,
                "joint": part.joint,
                "joint_type": part.joint_type,
                "motion": _motion_payload(part),
                "visible_pixels": visible_pixels,
                "num_voxels": part.num_voxels,
                "paths": {
                    "part_mask": _relative_to_data_root(part_mask_path, ctx.data_root),
                    "part_voxel": _relative_to_data_root(part_voxel_path, ctx.data_root),
                },
            }
        )

    if reasons:
        return None, sorted(set(reasons))

    sample_id = f"{ctx.dataset_slug}_{object_id}_angle_{angle_idx}_view_{view_idx}"
    category = finaljson.get("model_cat") or finaljson.get("category") or finaljson.get("name")
    remaining_pixels = int(remaining_mask.sum())
    separated_mask_paths[REMAINING_MASK_DIR] = _relative_to_data_root(remaining_mask_path, ctx.data_root)
    row = {
        "manifest_version": MANIFEST_VERSION,
        "task": TASK_NAME,
        "sample_id": sample_id,
        "object_id": object_id,
        "angle_idx": angle_idx,
        "view_idx": view_idx,
        "view_indices": [view_idx],
        "sample_unit": "single part_complete RGB view",
        "name": finaljson.get("model_id") or finaljson.get("name"),
        "category": category,
        "image_path": _relative_to_data_root(rgb_path, ctx.data_root),
        "image_paths": [_relative_to_data_root(rgb_path, ctx.data_root)],
        "mask_path": _relative_to_data_root(label_mask_path, ctx.data_root),
        "mask_paths": [_relative_to_data_root(label_mask_path, ctx.data_root)],
        "feature_path": _relative_to_data_root(dinov2_tokens_path, ctx.data_root),
        "feature_paths": [_relative_to_data_root(dinov2_tokens_path, ctx.data_root)],
        "feature_view_index": view_idx,
        "dinov2": {
            "render_set": "part_complete",
            "tokens_path": _relative_to_data_root(dinov2_tokens_path, ctx.data_root),
            "tokens_meta_path": _relative_to_data_root(dinov2_tokens_meta_path, ctx.data_root),
            "tokens_key": "tokens",
            "view_axis": 0,
            "view_index": view_idx,
            "expected_tokens_shape": [EXPECTED_VIEW_COUNT, 1370, 1024],
            "available": dinov2_tokens_available,
            "selection": f"tokens[{view_idx}]",
        },
        "target_part_count": len(target_parts),
        "target_parts": target_parts,
        "target_part_names": [part[0].name for part in visible_parts],
        "target_original_labels": target_original_labels,
        "target_voxel_paths": target_voxel_paths,
        "label_to_component": label_to_component,
        "remaining": {
            "name": REMAINING_MASK_DIR,
            "label": remaining_label,
            "visible_pixels": remaining_pixels,
            "mask_path": _relative_to_data_root(remaining_mask_path, ctx.data_root),
            "description": "merged non-target visible object pixels; no target voxel is attached",
        },
        "mask_rule": (
            "single label mask: 0=background; part_info.label=visible valid target part; "
            f"{remaining_label}=remaining merged non-target object pixels"
        ),
        "visibility": {
            "min_visible_pixels": ctx.min_visible_pixels,
            "visible_pixels_by_part": visible_pixels_by_part,
            "remaining_visible_pixels": remaining_pixels,
            "has_visible_target_part": True,
        },
        "paths": {
            "part_info": _relative_to_data_root(ctx.data_root / "part_info" / object_id / "part_info.json", ctx.data_root),
            "part_complete_dir": _relative_to_data_root(part_complete_dir, ctx.data_root),
            "image": _relative_to_data_root(rgb_path, ctx.data_root),
            "mask": _relative_to_data_root(label_mask_path, ctx.data_root),
            "mask_png": _relative_to_data_root(label_mask_png_path, ctx.data_root),
            "dinov2_tokens": _relative_to_data_root(dinov2_tokens_path, ctx.data_root),
            "dinov2_tokens_meta": _relative_to_data_root(dinov2_tokens_meta_path, ctx.data_root),
            "separated_masks": separated_mask_paths,
            "overall_surface": _relative_to_data_root(overall_surface_path, ctx.data_root),
        },
    }
    return row, []


def _iter_candidate_views(
    config: Any,
    object_ids: list[str],
    angle_ids_arg: list[int] | None,
    view_ids_arg: list[int] | None,
) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    view_ids = view_ids_arg if view_ids_arg is not None else list(range(EXPECTED_VIEW_COUNT))
    invalid_views = [view_idx for view_idx in view_ids if view_idx >= EXPECTED_VIEW_COUNT]
    if invalid_views:
        raise ValueError(f"part_complete views must be in [0,{EXPECTED_VIEW_COUNT - 1}], got {invalid_views}")
    for object_id in object_ids:
        object_num_angles = config.get_num_angles(object_id)
        angle_ids = angle_ids_arg if angle_ids_arg is not None else list(range(object_num_angles))
        invalid_angles = [angle_idx for angle_idx in angle_ids if angle_idx >= object_num_angles]
        if invalid_angles:
            raise ValueError(f"Object {object_id} has {object_num_angles} angle(s); invalid --angle-ids: {invalid_angles}")
        for angle_idx in angle_ids:
            for view_idx in view_ids:
                candidates.append((object_id, angle_idx, view_idx))
    return candidates


def build_manifest(args: argparse.Namespace) -> int:
    if args.min_visible_pixels < 1:
        raise ValueError("--min-visible-pixels must be >= 1")
    part_complete_subdir = _validate_simple_name(args.part_complete_subdir, "--part-complete-subdir")
    label_mask_subdir = _validate_simple_name(args.label_mask_subdir, "--label-mask-subdir")
    if label_mask_subdir == REMAINING_MASK_DIR:
        raise ValueError(f"--label-mask-subdir cannot be reserved name {REMAINING_MASK_DIR!r}")

    config = load_config(args.config)
    data_root = Path(config.data_root)
    dataset_slug = _dataset_slug(config.dataset_name)
    manifest_path, manifest = _load_manifest(config, args.manifest)
    output_dir = Path(args.output_dir) if args.output_dir else (data_root / "manifests" / "part_completion")
    train_path = output_dir / f"arts_pc_{dataset_slug}_train.jsonl"
    meta_path = output_dir / "manifest_meta.json"
    skip_path = output_dir / "skip_report.json"

    existing_outputs = [path for path in (train_path, meta_path, skip_path) if path.exists()]
    if existing_outputs and not args.overwrite and not args.dry_run:
        existing_text = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"output file(s) already exist; pass --overwrite: {existing_text}")

    object_ids = _resolve_object_ids(config, args.object_ids)
    angle_ids = _parse_int_ids(args.angle_ids, "--angle-ids")
    view_ids = _parse_int_ids(args.view_ids, "--view-ids")
    candidates = _iter_candidate_views(config, object_ids, angle_ids, view_ids)
    ctx = BuildContext(
        config=config,
        data_root=data_root,
        dataset_slug=dataset_slug,
        manifest=manifest,
        part_complete_subdir=part_complete_subdir,
        label_mask_subdir=label_mask_subdir,
        min_visible_pixels=args.min_visible_pixels,
        write_label_masks=not args.dry_run,
        require_dinov2_tokens=not args.allow_missing_dinov2,
    )

    t0 = time.time()
    created_at = _utc_now_iso()
    emitted_rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    skip_counter: Counter[str] = Counter()
    target_count_counter: Counter[int] = Counter()
    object_ids_seen: set[str] = set()
    angle_keys_seen: set[tuple[str, int]] = set()
    valid_parts_cache: dict[tuple[str, int], tuple[list[ValidPart], dict[str, Any]]] = {}
    sample_ids_seen: set[str] = set()

    for candidate_idx, (object_id, angle_idx, view_idx) in enumerate(candidates, start=1):
        try:
            cache_key = (object_id, angle_idx)
            if cache_key not in valid_parts_cache:
                valid_parts_cache[cache_key] = _load_valid_parts(ctx, object_id, angle_idx)
            valid_parts, object_payloads = valid_parts_cache[cache_key]
            row, reasons = _build_view_row(
                ctx,
                object_id=object_id,
                angle_idx=angle_idx,
                view_idx=view_idx,
                valid_parts=valid_parts,
                finaljson=_require_mapping(object_payloads.get("finaljson"), "object_payloads.finaljson"),
            )
            if row is not None:
                sample_id = _require_string(row.get("sample_id"), "row.sample_id")
                if sample_id in sample_ids_seen:
                    row = None
                    reasons = ["duplicate_sample_id"]
                else:
                    sample_ids_seen.add(sample_id)
        except (json.JSONDecodeError, TypeError, ValueError, FileNotFoundError, KeyError) as exc:
            row = None
            reasons = [f"invalid_candidate:{type(exc).__name__}: {exc}"]

        if row is None:
            skip_counter.update(reasons)
            skips.append(
                {
                    "object_id": object_id,
                    "angle_idx": angle_idx,
                    "view_idx": view_idx,
                    "reasons": reasons,
                }
            )
        else:
            emitted_rows.append(row)
            object_ids_seen.add(str(row["object_id"]))
            angle_keys_seen.add((str(row["object_id"]), int(row["angle_idx"])))
            target_count_counter.update([int(row["target_part_count"])])

        if args.progress_every and candidate_idx % args.progress_every == 0:
            print(
                f"Processed {candidate_idx}/{len(candidates)} candidate views: "
                f"train={len(emitted_rows)} skipped={len(skips)}",
                flush=True,
            )

    unexpected_skip_counter = _unexpected_skip_reasons(skip_counter)
    has_fatal_skips = _has_fatal_skips(skip_counter, strict_zero_skips=args.strict_zero_skips)

    skip_report = {
        "created_at": created_at,
        "dry_run": bool(args.dry_run),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "train_manifest": str(train_path),
        "total_candidate_views": len(candidates),
        "emitted_rows": len(emitted_rows),
        "skipped_views": len(skips),
        "skip_reasons": dict(skip_counter.most_common()),
        "expected_filter_skip_reasons": sorted(EXPECTED_FILTER_SKIP_REASONS),
        "unexpected_skip_reasons": dict(unexpected_skip_counter.most_common()),
        "strict_zero_skips": bool(args.strict_zero_skips),
        "fatal_skips": bool(has_fatal_skips),
        "skips": skips,
    }

    meta = {
        "manifest_version": MANIFEST_VERSION,
        "task": TASK_NAME,
        "created_at": created_at,
        "dry_run": bool(args.dry_run),
        "created_by": str(Path(__file__).resolve()),
        "config": str(Path(args.config).resolve()),
        "data_root": str(data_root),
        "dataset_name": config.dataset_name,
        "dataset_slug": dataset_slug,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": skip_report["source_manifest_sha256"],
        "train_manifest": str(train_path),
        "skip_report": str(skip_path),
        "split_rule": "no validation split; every valid part_complete RGB view is emitted as one training sample",
        "rules": {
            "source_of_truth": "Step 04 valid-parts manifest plus renders/<object>/angle_i/part_complete assets",
            "no_vlm_dependency": True,
            "sample_unit": "one RGB view + one derived label mask",
            "target_part_rule": "movable A/B/C ∩ has_voxel_ind ∩ visible pixels in that view",
            "voxel_rule": "target voxel path is angle-specific voxel_expanded/<object>/angle_i/64/ind_<part>.npy for each visible target only",
            "dinov2_rule": "one part_complete tokens.npz per object angle; sample view j reads tokens[j] from dinov2_tokens/<object>/angle_i/part_complete/tokens.npz",
            "require_dinov2_tokens": not args.allow_missing_dinov2,
            "mask_rule": "0=background; part_info.label=visible valid target component; remaining_label=merged non-target object pixels",
            "remaining_rule": "remaining is included in the label mask but never attached as a target voxel",
            "skip_policy": (
                "expected candidate filters no_visible_target_part and no_manifest_valid_movable_parts "
                "are allowed by default; missing/corrupt artifacts and duplicate samples are fatal"
            ),
        },
        "paths": {
            "part_complete_subdir": part_complete_subdir,
            "label_mask_subdir": label_mask_subdir,
        },
        "counts": {
            "candidate_views": len(candidates),
            "train_rows": len(emitted_rows),
            "skipped_views": len(skips),
            "unexpected_skipped_reasons": dict(unexpected_skip_counter.most_common()),
            "unique_objects": len(object_ids_seen),
            "unique_object_angles": len(angle_keys_seen),
            "target_part_count_distribution": dict(sorted(target_count_counter.items())),
        },
        "schema": {
            "row_key": "sample_id = <dataset_slug>_<object_id>_angle_<angle_idx>_view_<view_idx>",
            "paths_are_relative_to": str(data_root),
            "view_indices": "single-item list retained for downstream compatibility; use view_idx for the scalar value",
        },
        "runtime_seconds": round(time.time() - t0, 3),
    }

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with train_path.open("w", encoding="utf-8") as out:
            for row in emitted_rows:
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        skip_path.write_text(json.dumps(skip_report, indent=2, ensure_ascii=False), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Source manifest: {manifest_path}")
    print(f"Source manifest sha256: {skip_report['source_manifest_sha256']}")
    print(f"Train manifest: {train_path}")
    print(f"Meta: {meta_path}")
    print(f"Skip report: {skip_path}")
    print(
        f"Rows: candidates={len(candidates)} train={len(emitted_rows)} skipped={len(skips)} "
        f"dry_run={args.dry_run}"
    )
    print(f"Unique objects: {len(object_ids_seen)}")
    print(f"Runtime: {meta['runtime_seconds']}s")
    if skips:
        print(f"Skip reasons: {dict(skip_counter.most_common())}")
        if has_fatal_skips:
            if args.strict_zero_skips:
                print(
                    "ERROR: skipped candidate views found; --strict-zero-skips requires zero skips",
                    file=sys.stderr,
                )
            else:
                print(
                    "ERROR: unexpected skipped candidate views found; expected visibility/eligibility "
                    f"filters are allowed, unexpected reasons={dict(unexpected_skip_counter.most_common())}",
                    file=sys.stderr,
                )
            return 1
        print(
            "Allowed expected candidate filters; continuing because every skip reason is an "
            "eligibility filter, not a missing/corrupt artifact."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return build_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
