#!/usr/bin/env python3
"""
Render 16-view part-completion RGB images and part-separated masks.

Run this after Step 04 has built the valid-parts manifest.  Each object/angle render
contains the full articulated object in RGB.  Masks are saved separately:
one binary mask per valid target component, plus one merged binary mask for the
remaining visible object pixels.

    movable part (finaljson group_info motion type A/B/C)
    ∩ manifest angle part with has_voxel_ind=true and num_voxels > 5

The remaining mask includes fixed parts and movable parts without valid voxel
indices.  Background stays 0 in every binary mask.

Output layout:
    renders/<object_id>/angle_<i>/part_complete/
        rgb/view_0.png ... rgb/view_15.png
        mask/<part_key>/mask_0.npy ... mask/<part_key>/mask_15.npy
        mask/<part_key>/mask_0.png ... mask/<part_key>/mask_15.png
        mask/remaining/mask_0.npy ... mask/remaining/mask_15.npy
        mask/remaining/mask_0.png ... mask/remaining/mask_15.png
        camera_transforms.json
        mask_labels.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_OUTPUT_SUBDIR = "part_complete"
DEFAULT_NUM_VIEWS = 16
DEFAULT_NUM_RANDOM_VIEWS = 8
DEFAULT_LENS_MM = 50.0
DEFAULT_SENSOR_WIDTH_MM = 36.0
DEFAULT_TIMEOUT_SECONDS = 3600
ACCEPTED_MOTION_TYPES = {"A", "B", "C"}
MIN_PART_VOXELS = 5
REMAINING_MASK_DIR = "remaining"


@dataclass(frozen=True)
class TargetPart:
    part_key: str
    part_index: int
    label: int
    obj_files: tuple[str, ...]
    motion_type: str
    group_id: str
    num_voxels: int
    voxel_ind_path: str


@dataclass(frozen=True)
class JobSpec:
    object_id: str
    angle_idx: int
    finaljson_path: Path
    objs_dir: Path
    output_dir: Path
    blender_binary: str
    resolution: int
    num_views: int
    engine: str
    samples: int
    seed: int
    obj_up_axis: str
    angle_payload: dict[str, Any]
    target_parts: tuple[TargetPart, ...]

    @property
    def desc(self) -> str:
        return f"{self.object_id} angle_{self.angle_idx}"


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


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_matrix4x4(value: Any, name: str) -> list[list[float]]:
    rows = _require_list(value, name)
    if len(rows) != 4:
        raise ValueError(f"{name} must contain 4 rows, got {len(rows)}")
    matrix: list[list[float]] = []
    for row_idx, row in enumerate(rows):
        row_values = _require_list(row, f"{name}[{row_idx}]")
        if len(row_values) != 4:
            raise ValueError(f"{name}[{row_idx}] must contain 4 values, got {len(row_values)}")
        matrix_row: list[float] = []
        for col_idx, item in enumerate(row_values):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{name}[{row_idx}][{col_idx}] must be numeric")
            value_float = float(item)
            if not math.isfinite(value_float):
                raise ValueError(f"{name}[{row_idx}][{col_idx}] must be finite")
            matrix_row.append(value_float)
        matrix.append(matrix_row)
    return matrix


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return _require_mapping(json.load(handle), f"{name}[{path}]")


def _parse_csv(value: str, field_name: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise ValueError(f"{field_name} must be a comma-separated list of non-empty values")
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} contains duplicate values")
    return items


def _parse_angle_ids(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    angle_ids: list[int] = []
    for item in _parse_csv(raw_value, "--angle-ids"):
        try:
            angle_idx = int(item)
        except ValueError as exc:
            raise ValueError(f"--angle-ids values must be integers, got {item!r}") from exc
        if angle_idx < 0:
            raise ValueError(f"--angle-ids values must be >= 0, got {angle_idx}")
        angle_ids.append(angle_idx)
    return angle_ids


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


def _load_manifest(config: Any, manifest_arg: str | None) -> dict[str, Any]:
    path = _manifest_path(config, manifest_arg)
    manifest = _load_json(path, "manifest")
    _require_mapping(manifest.get("objects"), f"manifest[{path}]['objects']")
    _require_mapping(manifest.get("summary"), f"manifest[{path}]['summary']")
    return manifest


def _manifest_angle_parts(
    manifest: dict[str, Any],
    object_id: str,
    angle_idx: int,
) -> dict[str, Any]:
    objects = _require_mapping(manifest.get("objects"), "manifest['objects']")
    object_record = _require_mapping(
        objects.get(object_id),
        f"manifest['objects']['{object_id}']",
    )
    angles = _require_mapping(
        object_record.get("angles"),
        f"manifest['objects']['{object_id}']['angles']",
    )
    angle_record = _require_mapping(
        angles.get(str(angle_idx)),
        f"manifest['objects']['{object_id}']['angles']['{angle_idx}']",
    )
    return _require_mapping(
        angle_record.get("parts"),
        f"manifest['objects']['{object_id}']['angles']['{angle_idx}']['parts']",
    )


def _is_manifest_voxel_kept(part_record: dict[str, Any], part_key: str, context: str) -> bool:
    has_voxel_ind = part_record.get("has_voxel_ind")
    if not isinstance(has_voxel_ind, bool):
        raise TypeError(f"{context}['{part_key}']['has_voxel_ind'] must be bool")

    num_voxels = _require_int(part_record.get("num_voxels"), f"{context}['{part_key}']['num_voxels']")
    voxel_ind_path = part_record.get("voxel_ind_path")
    if not has_voxel_ind:
        return False
    if num_voxels <= MIN_PART_VOXELS:
        raise ValueError(
            f"{context}['{part_key}'] has_voxel_ind=true but num_voxels={num_voxels} "
            f"<= MIN_PART_VOXELS={MIN_PART_VOXELS}"
        )
    if not isinstance(voxel_ind_path, str) or not voxel_ind_path:
        raise ValueError(f"{context}['{part_key}'] has_voxel_ind=true but voxel_ind_path is empty")
    return True


def _load_angle_transforms(joint_transforms_path: Path, object_id: str, angle_idx: int) -> dict[str, Any]:
    archive = _load_json(joint_transforms_path, "joint_transforms")
    archive_object_id = str(archive.get("object_id"))
    if archive_object_id != object_id:
        raise ValueError(
            f"object_id mismatch in {joint_transforms_path}: expected {object_id}, got {archive_object_id}"
        )

    num_parts = _require_int(archive.get("num_parts"), "joint_transforms['num_parts']")
    angles = _require_mapping(archive.get("angles"), "joint_transforms['angles']")
    angle_key = str(angle_idx)
    if angle_key not in angles:
        raise KeyError(f"Missing angle '{angle_key}' in {joint_transforms_path}")

    angle_data = _require_mapping(angles[angle_key], f"joint_transforms['angles']['{angle_key}']")
    joint_states = _require_mapping(angle_data.get("joint_states"), f"angles['{angle_key}']['joint_states']")
    part_transforms = _require_mapping(
        angle_data.get("part_transforms"),
        f"angles['{angle_key}']['part_transforms']",
    )

    expected_keys = {str(part_idx) for part_idx in range(num_parts)}
    actual_keys = set(part_transforms)
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys:
        raise KeyError(f"Missing part transforms for object {object_id} angle_{angle_idx}: {missing_keys}")
    if extra_keys:
        raise KeyError(f"Unexpected part transforms for object {object_id} angle_{angle_idx}: {extra_keys}")

    for part_idx in range(num_parts):
        _validate_matrix4x4(part_transforms[str(part_idx)], f"part_transforms['{part_idx}']")

    return {
        "object_id": object_id,
        "num_parts": num_parts,
        "angle_idx": angle_idx,
        "joint_states": joint_states,
        "part_transforms": part_transforms,
    }


def _build_movable_part_candidates(finaljson_path: Path, part_info_path: Path, object_id: str) -> dict[str, dict[str, Any]]:
    finaljson = _load_json(finaljson_path, "finaljson")
    part_info = _load_json(part_info_path, "part_info")
    label_to_key = _require_mapping(part_info.get("label_to_key"), "part_info['label_to_key']")
    parts = _require_mapping(part_info.get("parts"), "part_info['parts']")
    group_info = _require_mapping(finaljson.get("group_info"), "finaljson['group_info']")

    candidates: dict[str, dict[str, Any]] = {}
    for raw_group_id, raw_group in group_info.items():
        group_id = str(raw_group_id)
        if not isinstance(raw_group, list) or len(raw_group) != 4:
            continue
        links, _parent_group, _params, motion_type = raw_group
        motion_type = str(motion_type)
        if motion_type not in ACCEPTED_MOTION_TYPES:
            continue
        if isinstance(links, int):
            link_ids = [links]
        elif isinstance(links, list):
            link_ids = [
                _require_int(link_id, f"finaljson['group_info']['{group_id}'][0][{idx}]")
                for idx, link_id in enumerate(links)
            ]
        else:
            raise TypeError(f"finaljson['group_info']['{group_id}'][0] must be int or list")

        for link_id in link_ids:
            key = label_to_key.get(str(link_id))
            if not isinstance(key, str) or not key:
                raise KeyError(
                    f"Object {object_id}: movable link {link_id} has no part_info label_to_key entry"
                )
            part_entry = _require_mapping(parts.get(key), f"part_info['parts']['{key}']")
            obj_files = tuple(
                _require_string(obj_name, f"part_info['parts']['{key}']['obj_files'][{idx}]")
                for idx, obj_name in enumerate(
                    _require_list(part_entry.get("obj_files"), f"part_info['parts']['{key}']['obj_files']")
                )
            )
            if not obj_files:
                raise ValueError(f"part_info['parts']['{key}']['obj_files'] must not be empty")
            candidates[key] = {
                "part_key": key,
                "part_index": _require_int(part_entry.get("part_index"), f"part_info['parts']['{key}']['part_index']"),
                "label": _require_int(part_entry.get("label"), f"part_info['parts']['{key}']['label']"),
                "obj_files": obj_files,
                "motion_type": motion_type,
                "group_id": group_id,
            }
    return candidates


def select_valid_target_parts(
    finaljson_path: Path,
    part_info_path: Path,
    manifest_parts: dict[str, Any],
    object_id: str,
    requested_part_keys: set[str] | None,
) -> tuple[TargetPart, ...]:
    candidates = _build_movable_part_candidates(finaljson_path, part_info_path, object_id)
    if requested_part_keys is not None:
        candidates = {key: value for key, value in candidates.items() if key in requested_part_keys}

    targets: list[TargetPart] = []
    for part_key, candidate in candidates.items():
        part_record = manifest_parts.get(part_key)
        if part_record is None:
            raise KeyError(f"manifest object {object_id} missing movable part '{part_key}'")
        part_payload = _require_mapping(part_record, f"manifest parts['{part_key}']")
        if not _is_manifest_voxel_kept(part_payload, part_key, "manifest parts"):
            continue
        num_voxels = _require_int(part_payload.get("num_voxels"), f"manifest parts['{part_key}']['num_voxels']")
        voxel_ind_path = _require_string(
            part_payload.get("voxel_ind_path"),
            f"manifest parts['{part_key}']['voxel_ind_path']",
        )
        label = int(candidate["label"])
        if label <= 0:
            raise ValueError(f"part_info['parts']['{part_key}']['label'] must be positive, got {label}")
        targets.append(
            TargetPart(
                part_key=str(candidate["part_key"]),
                part_index=int(candidate["part_index"]),
                label=label,
                obj_files=tuple(candidate["obj_files"]),
                motion_type=str(candidate["motion_type"]),
                group_id=str(candidate["group_id"]),
                num_voxels=num_voxels,
                voxel_ind_path=voxel_ind_path,
            )
        )

    targets.sort(key=lambda part: (part.part_index, part.part_key))
    labels = [part.label for part in targets]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Object {object_id} has duplicate target mask labels: {labels}")
    return tuple(targets)


def _validate_mask_component_name(name: str, context: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"{context} must be a simple directory name, got {name!r}")
    return name


def _expected_mask_component_dirs(target_parts: tuple[TargetPart, ...]) -> list[str]:
    component_dirs = [
        _validate_mask_component_name(part.part_key, f"target part key {part.part_key!r}")
        for part in target_parts
    ]
    if REMAINING_MASK_DIR in component_dirs:
        raise ValueError(f"target part key conflicts with reserved mask directory {REMAINING_MASK_DIR!r}")
    component_dirs.append(REMAINING_MASK_DIR)
    return component_dirs


def _output_is_complete(output_dir: Path, num_views: int, target_parts: tuple[TargetPart, ...]) -> bool:
    if not (output_dir / "camera_transforms.json").is_file():
        return False
    if not (output_dir / "mask_labels.json").is_file():
        return False
    component_dirs = _expected_mask_component_dirs(target_parts)
    return all(
        (output_dir / "rgb" / f"view_{view_idx}.png").is_file()
        and all(
            (output_dir / "mask" / component_dir / f"mask_{view_idx}.npy").is_file()
            and (output_dir / "mask" / component_dir / f"mask_{view_idx}.png").is_file()
            for component_dir in component_dirs
        )
        for view_idx in range(num_views)
    )


def build_omnipart_16_views(object_id: str, angle_idx: int, seed: int, num_views: int) -> list[dict[str, Any]]:
    if num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}")
    fixed_azimuths = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    views: list[dict[str, Any]] = [
        {
            "view_index": view_idx,
            "source": "solid",
            "azimuth_deg": azimuth_deg,
            "elevation_deg": 25.0,
        }
        for view_idx, azimuth_deg in enumerate(fixed_azimuths[: min(len(fixed_azimuths), num_views)])
    ]
    if len(views) >= num_views:
        return views

    digest = hashlib.sha256(f"{seed}:{object_id}:{angle_idx}:part_complete_16".encode("utf-8")).digest()
    rng_seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(rng_seed)
    while len(views) < num_views:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        phi = rng.uniform(0.0, 0.5 * math.pi)
        sign = -1.0 if rng.randint(0, 1) else 1.0
        elevation = math.degrees(math.atan2(math.cos(phi) * sign, math.sin(phi)))
        views.append(
            {
                "view_index": len(views),
                "source": "random",
                "azimuth_deg": math.degrees(theta) % 360.0,
                "elevation_deg": elevation,
            }
        )
    return views


def build_jobs(
    config: Any,
    manifest: dict[str, Any],
    object_ids: list[str],
    *,
    angle_ids: list[int] | None,
    part_keys: set[str] | None,
    num_views: int,
    resolution: int | None,
    engine: str,
    samples: int,
    seed: int,
    output_subdir: str,
    force: bool,
) -> tuple[list[JobSpec], dict[str, int]]:
    sys.path.insert(0, str(SCRIPT_DIR.parent / "utils"))
    from config_loader import resolve_obj_up_axis

    finaljson_root = Path(config.finaljson_dir)
    part_info_root = Path(config.part_info_dir)
    objs_root = Path(config.partseg_dir)
    joint_root = Path(config.joint_transforms_dir)
    renders_root = Path(config.renders_dir)
    render_resolution = resolution if resolution is not None else int(config.render.resolution)

    stats = {
        "movable_parts": 0,
        "target_parts": 0,
        "skipped_existing": 0,
        "angles_without_targets": 0,
    }
    jobs: list[JobSpec] = []

    for object_id in object_ids:
        finaljson_path = finaljson_root / f"{object_id}.json"
        part_info_path = part_info_root / object_id / "part_info.json"
        joint_transforms_path = joint_root / f"{object_id}.json"
        objs_dir = objs_root / object_id / "objs"
        if not finaljson_path.is_file():
            raise FileNotFoundError(f"Missing finaljson: {finaljson_path}")
        if not objs_dir.is_dir():
            raise FileNotFoundError(f"Missing OBJ directory: {objs_dir}")
        if not joint_transforms_path.is_file():
            raise FileNotFoundError(f"Missing joint transforms: {joint_transforms_path}")

        obj_up_axis = resolve_obj_up_axis(config, finaljson_path)
        object_num_angles = config.get_num_angles(object_id)
        selected_angles = angle_ids if angle_ids is not None else list(range(object_num_angles))
        invalid_angles = [angle_idx for angle_idx in selected_angles if angle_idx >= object_num_angles]
        if invalid_angles:
            raise ValueError(
                f"Object {object_id} has {object_num_angles} angle(s); invalid --angle-ids: {invalid_angles}"
            )

        for angle_idx in selected_angles:
            manifest_parts = _manifest_angle_parts(manifest, object_id, angle_idx)
            targets = select_valid_target_parts(
                finaljson_path,
                part_info_path,
                manifest_parts,
                object_id,
                requested_part_keys=part_keys,
            )
            stats["movable_parts"] += len(_build_movable_part_candidates(finaljson_path, part_info_path, object_id))
            stats["target_parts"] += len(targets)
            if not targets:
                stats["angles_without_targets"] += 1
                continue

            output_dir = renders_root / object_id / f"angle_{angle_idx}" / output_subdir
            if not force and _output_is_complete(output_dir, num_views, targets):
                stats["skipped_existing"] += 1
                continue

            jobs.append(
                JobSpec(
                    object_id=object_id,
                    angle_idx=angle_idx,
                    finaljson_path=finaljson_path,
                    objs_dir=objs_dir,
                    output_dir=output_dir,
                    blender_binary=config.render.blender,
                    resolution=render_resolution,
                    num_views=num_views,
                    engine=engine,
                    samples=samples,
                    seed=seed,
                    obj_up_axis=obj_up_axis,
                    angle_payload=_load_angle_transforms(joint_transforms_path, object_id, angle_idx),
                    target_parts=targets,
                )
            )

    return jobs, stats


def _write_temp_json(prefix: str, payload: Any) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=prefix,
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        return Path(handle.name)


def _create_temp_log_path(prefix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=prefix,
        suffix=".log",
        delete=False,
    ) as handle:
        return Path(handle.name)


def _target_parts_payload(parts: tuple[TargetPart, ...]) -> list[dict[str, Any]]:
    return [
        {
            "part_key": part.part_key,
            "part_index": part.part_index,
            "label": part.label,
            "obj_files": list(part.obj_files),
            "motion_type": part.motion_type,
            "group_id": part.group_id,
            "num_voxels": part.num_voxels,
            "voxel_ind_path": part.voxel_ind_path,
        }
        for part in parts
    ]


def run_job(spec: JobSpec, timeout_seconds: int) -> str:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_id": spec.object_id,
        "angle_idx": spec.angle_idx,
        "angle_payload": spec.angle_payload,
        "target_parts": _target_parts_payload(spec.target_parts),
        "mask_rule": (
            "Save one binary mask per movable A/B/C part with manifest has_voxel_ind=true "
            "and num_voxels>5; save all other visible object pixels in mask/remaining. "
            "Background is 0 in every binary mask."
        ),
    }
    views = build_omnipart_16_views(spec.object_id, spec.angle_idx, spec.seed, spec.num_views)
    payload_path = _write_temp_json(
        prefix=f"part_complete_{spec.object_id}_angle_{spec.angle_idx}_payload_",
        payload=payload,
    )
    views_path = _write_temp_json(
        prefix=f"part_complete_{spec.object_id}_angle_{spec.angle_idx}_views_",
        payload=views,
    )
    log_path = _create_temp_log_path(prefix=f"part_complete_{spec.object_id}_angle_{spec.angle_idx}_")
    command = [
        spec.blender_binary,
        "--background",
        "--gpu-backend",
        "vulkan",
        "--python",
        str(SCRIPT_PATH),
        "--",
        "--blender-worker",
        "--payload-json",
        str(payload_path),
        "--views-json",
        str(views_path),
        "--objs-dir",
        str(spec.objs_dir),
        "--finaljson",
        str(spec.finaljson_path),
        "--output-folder",
        str(spec.output_dir),
        "--resolution",
        str(spec.resolution),
        "--engine",
        spec.engine,
        "--samples",
        str(spec.samples),
        "--obj-up-axis",
        spec.obj_up_axis,
    ]
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode != 0:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Blender subprocess failed for {spec.desc} "
                f"(exit code {completed.returncode}) with log:\n{log_text}"
            )
        return spec.desc
    finally:
        for path in (payload_path, views_path, log_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def parse_driver_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 16-view part-completion RGB and target-only masks after Step 04 valid-parts manifest."
    )
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument("--manifest", help="Manifest path; default is <data_root>/manifests/<dataset>.json.")
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset.")
    parser.add_argument("--angle-ids", help="Optional comma-separated angle subset.")
    parser.add_argument("--part-keys", help="Optional comma-separated canonical part key subset.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel Blender subprocess workers.")
    parser.add_argument("--num-views", type=int, default=DEFAULT_NUM_VIEWS, help="Views per object angle.")
    parser.add_argument("--resolution", type=int, default=None, help="Square render resolution; defaults to config.")
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT", help="Blender render engine for RGB pass.")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples when --engine CYCLES is used.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic random OmniPart-style views.")
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdirectory under renders/<object_id>/angle_<i>/ for outputs.",
    )
    parser.add_argument("--force", action="store_true", help="Re-render even when expected files exist.")
    parser.add_argument("--dry-run", action="store_true", help="List jobs without launching Blender.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args(argv)


def main_driver(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(SCRIPT_DIR.parent / "utils"))
    from config_loader import load_config

    args = parse_driver_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.num_views < 1:
        raise ValueError("--num-views must be >= 1")
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")
    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be >= 1")
    if "/" in args.output_subdir or args.output_subdir in {"", ".", ".."}:
        raise ValueError("--output-subdir must be a simple directory name")

    config = load_config(args.config)
    manifest = _load_manifest(config, args.manifest)
    object_ids = _resolve_object_ids(config, args.object_ids)
    angle_ids = _parse_angle_ids(args.angle_ids)
    part_keys = set(_parse_csv(args.part_keys, "--part-keys")) if args.part_keys else None

    jobs, stats = build_jobs(
        config,
        manifest,
        object_ids,
        angle_ids=angle_ids,
        part_keys=part_keys,
        num_views=args.num_views,
        resolution=args.resolution,
        engine=args.engine,
        samples=args.samples,
        seed=args.seed,
        output_subdir=args.output_subdir,
        force=args.force,
    )

    print(
        "Prepared part-complete render jobs: "
        f"objects={len(object_ids)} jobs={len(jobs)} "
        f"movable_parts_seen={stats['movable_parts']} "
        f"target_parts={stats['target_parts']} "
        f"skipped_existing={stats['skipped_existing']} "
        f"angles_without_targets={stats['angles_without_targets']} "
        f"views_per_job={args.num_views}",
        flush=True,
    )
    for job in jobs[:20]:
        target_desc = ", ".join(f"{part.part_key}:label{part.label}" for part in job.target_parts)
        print(f"  {job.desc} -> {job.output_dir} targets=[{target_desc}]", flush=True)
    if len(jobs) > 20:
        print(f"  ... {len(jobs) - 20} more job(s)", flush=True)

    if args.dry_run or not jobs:
        return 0

    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(run_job, job, args.timeout_seconds): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            future.result()
            completed_count += 1
            print(
                f"[{completed_count}/{len(jobs)}] {job.desc} done "
                f"({job.num_views} RGB/mask views)",
                flush=True,
            )

    print(
        f"Summary: objects={len(object_ids)} jobs={len(jobs)} "
        f"views_rendered={len(jobs) * args.num_views}",
        flush=True,
    )
    return 0


def parse_blender_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blender worker for part-complete rendering.")
    parser.add_argument("--blender-worker", action="store_true")
    parser.add_argument("--payload-json", required=True)
    parser.add_argument("--views-json", required=True)
    parser.add_argument("--objs-dir", required=True)
    parser.add_argument("--finaljson", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--obj-up-axis", choices=("Y", "Z"), default="Y")
    if "--" not in argv:
        raise ValueError("Blender script arguments must be passed after '--'")
    args = parser.parse_args(argv[argv.index("--") + 1 :])
    if not args.blender_worker:
        raise ValueError("Missing --blender-worker")
    return args


def main_blender_worker(argv: list[str]) -> int:
    import glob

    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
        (255, 128, 0),
        (128, 255, 0),
        (0, 128, 255),
        (255, 0, 128),
    ]
    y_up_to_z_up = Matrix.Rotation(math.radians(90.0), 4, "X")
    z_up_to_y_up = y_up_to_z_up.inverted()

    def init_render(engine: str, resolution: int, samples: int) -> None:
        scene = bpy.context.scene
        scene.render.engine = engine
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = True
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "None"
        if engine == "CYCLES":
            scene.cycles.samples = samples
            scene.cycles.filter_type = "BOX"
            scene.cycles.filter_width = 1
            scene.cycles.diffuse_bounces = 1
            scene.cycles.glossy_bounces = 1
            scene.cycles.transparent_max_bounces = 3
            scene.cycles.transmission_bounces = 3
            scene.cycles.use_denoising = True

    def init_scene() -> None:
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for material in list(bpy.data.materials):
            bpy.data.materials.remove(material, do_unlink=True)
        for texture in list(bpy.data.textures):
            bpy.data.textures.remove(texture, do_unlink=True)
        for image in list(bpy.data.images):
            bpy.data.images.remove(image, do_unlink=True)

    def init_camera():
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.data.sensor_width = DEFAULT_SENSOR_WIDTH_MM
        cam.data.sensor_height = DEFAULT_SENSOR_WIDTH_MM
        cam.data.lens = DEFAULT_LENS_MM
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        cam_empty = bpy.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, 0)
        cam_empty.empty_display_size = 0
        cam_empty.hide_render = True
        bpy.context.scene.collection.objects.link(cam_empty)
        cam_constraint.target = cam_empty
        return cam

    def get_blender_hdri_path(name: str = "studio.exr") -> str:
        blender_dir = os.path.dirname(bpy.app.binary_path)
        version = f"{bpy.app.version[0]}.{bpy.app.version[1]}"
        hdri_path = os.path.join(blender_dir, version, "datafiles", "studiolights", "world", name)
        if not os.path.isfile(hdri_path):
            raise FileNotFoundError(f"Built-in HDRI not found: {hdri_path}")
        return hdri_path

    def init_lighting() -> None:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        hdri_path = get_blender_hdri_path("studio.exr")
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        for node in nodes:
            nodes.remove(node)
        env_tex = nodes.new("ShaderNodeTexEnvironment")
        env_tex.image = bpy.data.images.load(hdri_path)
        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = 1.0
        links.new(env_tex.outputs["Color"], bg.inputs["Color"])
        output = nodes.new("ShaderNodeOutputWorld")
        links.new(bg.outputs["Background"], output.inputs["Surface"])

    def force_opaque_materials() -> None:
        """Use simple opaque materials for stable multi-view RGB.

        The imported PhysX materials are surface colors, but some OBJ material
        flags select transparent shader variants.  In Cycles that can make
        non-frontal views look stippled; in headless EEVEE it can crash on low
        SSBO limits.  Preserve each material's base color, then rebuild it as a
        single opaque Principled BSDF.
        """
        for material in bpy.data.materials:
            material.use_nodes = True
            base_color = tuple(material.diffuse_color) if len(material.diffuse_color) == 4 else (0.8, 0.0, 0.0, 1.0)
            node_tree = material.node_tree
            if node_tree is not None:
                for node in node_tree.nodes:
                    if node.bl_idname != "ShaderNodeBsdfPrincipled":
                        continue
                    color_input = node.inputs.get("Base Color")
                    if color_input is not None:
                        try:
                            base_color = tuple(color_input.default_value)
                        except Exception:
                            pass
                    break
            base_color = (float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0)
            material.diffuse_color = base_color
            material.blend_method = "OPAQUE"
            if hasattr(material, "show_transparent_back"):
                material.show_transparent_back = False
            if hasattr(material, "use_screen_refraction"):
                material.use_screen_refraction = False
            if node_tree is None:
                continue
            node_tree.nodes.clear()
            bsdf = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            output = node_tree.nodes.new("ShaderNodeOutputMaterial")
            color_input = bsdf.inputs.get("Base Color")
            if color_input is not None:
                color_input.default_value = base_color
            alpha_input = bsdf.inputs.get("Alpha")
            if alpha_input is not None:
                alpha_input.default_value = 1.0
            roughness_input = bsdf.inputs.get("Roughness")
            if roughness_input is not None:
                roughness_input.default_value = 0.55
            metallic_input = bsdf.inputs.get("Metallic")
            if metallic_input is not None:
                metallic_input.default_value = 0.0
            node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    def scene_bbox():
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        for obj in bpy.context.scene.objects.values():
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            found = True
            for coord in obj.bound_box:
                coord = obj.matrix_world @ Vector(coord)
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("No mesh objects in scene")
        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene():
        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if len(scene_root_objects) > 1:
            parent = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(parent)
            parent.empty_display_size = 0
            parent.hide_render = True
            for obj in scene_root_objects:
                obj.parent = parent
            scene = parent
        else:
            scene = scene_root_objects[0]

        bbox_min, bbox_max = scene_bbox()
        scale = 1.0 / max(bbox_max - bbox_min)
        scene.scale = scene.scale * scale
        bpy.context.view_layer.update()
        bbox_min, bbox_max = scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        scene.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")
        return scale, offset

    def get_transform_matrix(obj: Any) -> list[list[float]]:
        pos, rt, _ = obj.matrix_world.decompose()
        rt = rt.to_matrix()
        matrix = []
        for row_idx in range(3):
            row = [rt[row_idx][col_idx] for col_idx in range(3)]
            row.append(pos[row_idx])
            matrix.append(row)
        matrix.append([0.0, 0.0, 0.0, 1.0])
        return matrix

    def load_parts_with_target_pass_indices(
        objs_dir: str,
        finaljson_path: str,
        angle_payload: dict[str, Any],
        target_parts: list[dict[str, Any]],
        remaining_pass_index: int,
        obj_up_axis: str,
    ) -> None:
        finaljson_data = _load_json(Path(finaljson_path), "finaljson")
        parts = _require_list(finaljson_data.get("parts"), "finaljson['parts']")
        part_transforms = _require_mapping(angle_payload.get("part_transforms"), "angle_payload['part_transforms']")
        pass_index_by_part_index = {
            _require_int(part.get("part_index"), "target_parts[].part_index"):
            _require_int(part.get("label"), "target_parts[].label")
            for part in target_parts
        }

        imported_mesh_count = 0
        obj_up_axis = obj_up_axis.upper()
        for part_idx, part_data in enumerate(parts):
            part_mapping = _require_mapping(part_data, f"finaljson['parts'][{part_idx}]")
            obj_names = _require_list(part_mapping.get("obj"), f"parts[{part_idx}]['obj']")
            matrix_key = str(part_idx)
            if matrix_key not in part_transforms:
                raise KeyError(f"Missing transform for part {part_idx}")
            raw_matrix = _validate_matrix4x4(part_transforms[matrix_key], f"part_transforms['{matrix_key}']")
            if obj_up_axis == "Y":
                part_matrix = y_up_to_z_up @ Matrix(raw_matrix) @ z_up_to_y_up
                import_kwargs = {}
            else:
                part_matrix = Matrix(raw_matrix)
                import_kwargs = {"forward_axis": "Y", "up_axis": "Z"}

            pass_index = int(pass_index_by_part_index.get(part_idx, remaining_pass_index))
            for obj_name in obj_names:
                obj_name = _require_string(obj_name, f"parts[{part_idx}]['obj'][]")
                obj_path = os.path.join(objs_dir, f"{obj_name}.obj")
                if not os.path.isfile(obj_path):
                    raise FileNotFoundError(f"Missing OBJ for part {part_idx}: {obj_path}")
                before_objects = set(bpy.data.objects.keys())
                bpy.ops.wm.obj_import(filepath=obj_path, **import_kwargs)
                after_objects = set(bpy.data.objects.keys())
                mesh_objects = []
                for blender_name in after_objects - before_objects:
                    obj = bpy.data.objects[blender_name]
                    if obj.type != "MESH":
                        continue
                    obj.matrix_world = part_matrix @ obj.matrix_world
                    obj.name = f"part_{part_idx:04d}"
                    if obj.data is not None:
                        obj.data.name = f"part_{part_idx:04d}"
                    obj.pass_index = pass_index
                    mesh_objects.append(obj)
                    imported_mesh_count += 1
                if not mesh_objects:
                    raise RuntimeError(f"OBJ import produced no mesh objects: {obj_path}")
        if imported_mesh_count == 0:
            raise RuntimeError("No mesh objects were imported from finaljson")

    def _find_render_layer_output(render_layers: Any, name: str):
        for idx, output in enumerate(render_layers.outputs):
            if output.name == name:
                return render_layers.outputs[idx]
        raise KeyError(f"Render layer output '{name}' not found")

    def configure_object_index_output(output_folder: str) -> None:
        scene = bpy.context.scene
        scene.use_nodes = True
        scene.render.use_compositing = True
        tree = scene.node_tree
        tree.nodes.clear()
        render_layers = tree.nodes.new("CompositorNodeRLayers")
        composite = tree.nodes.new("CompositorNodeComposite")
        file_output = tree.nodes.new("CompositorNodeOutputFile")
        file_output.base_path = output_folder
        file_output.format.file_format = "OPEN_EXR"
        file_output.format.color_depth = "32"
        file_output.file_slots[0].path = "_temp_idx_"
        tree.links.new(_find_render_layer_output(render_layers, "Image"), composite.inputs["Image"])
        tree.links.new(_find_render_layer_output(render_layers, "IndexOB"), file_output.inputs[0])

    def cleanup_temp_exr(output_folder: str) -> None:
        for exr_path in glob.glob(os.path.join(output_folder, "_temp_idx_*.exr")):
            os.remove(exr_path)

    def read_object_index_exr(exr_path: str, width: int, height: int):
        image = bpy.data.images.load(exr_path)
        try:
            pixel_data = np.array(image.pixels[:], dtype=np.float32)
            if image.channels <= 0:
                raise RuntimeError(f"Invalid channel count in EXR: {exr_path}")
            pixel_data = pixel_data.reshape(height, width, image.channels)
            index_map = np.rint(pixel_data[:, :, 0]).astype(np.int32)
            return np.flipud(index_map)
        finally:
            bpy.data.images.remove(image)

    def build_mask_visualization(index_map: Any) -> Any:
        height, width = index_map.shape
        mask_vis = np.zeros((height, width, 3), dtype=np.uint8)
        for part_id in np.unique(index_map):
            if part_id <= 0:
                continue
            mask_vis[index_map == part_id] = colors[(int(part_id) - 1) % len(colors)]
        return mask_vis

    def save_rgb_png(path: str, rgb_array: Any) -> None:
        height, width, channels = rgb_array.shape
        if channels != 3:
            raise ValueError("RGB array must have shape [H, W, 3]")
        image = bpy.data.images.new(name=f"mask_vis_{Path(path).stem}", width=width, height=height, alpha=True)
        try:
            rgba = np.zeros((height, width, 4), dtype=np.float32)
            rgba[:, :, :3] = np.flipud(rgb_array).astype(np.float32) / 255.0
            rgba[:, :, 3] = 1.0
            image.pixels.foreach_set(rgba.reshape(-1))
            image.filepath_raw = path
            image.file_format = "PNG"
            image.save()
        finally:
            bpy.data.images.remove(image)

    def save_binary_mask_png(path: str, binary_mask: Any) -> None:
        height, width = binary_mask.shape
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_array[binary_mask.astype(bool)] = (255, 255, 255)
        save_rgb_png(path, rgb_array)

    def set_camera_pose(camera_obj: Any, azimuth_deg: float, elevation_deg: float, radius: float) -> None:
        azimuth_rad = math.radians(azimuth_deg)
        elevation_rad = math.radians(elevation_deg)
        camera_obj.location = (
            radius * math.cos(elevation_rad) * math.cos(azimuth_rad),
            radius * math.cos(elevation_rad) * math.sin(azimuth_rad),
            radius * math.sin(elevation_rad),
        )
        bpy.context.view_layer.update()

    def render_view(
        camera_obj: Any,
        output_folder: str,
        rgb_dir: str,
        mask_dir: str,
        target_parts: list[dict[str, Any]],
        remaining_pass_index: int,
        view_idx: int,
        view_config: dict[str, Any],
        camera_radius: float,
        rgb_engine: str,
        rgb_samples: int,
    ) -> dict[str, Any]:
        cleanup_temp_exr(output_folder)
        azimuth_deg = float(view_config["azimuth_deg"])
        elevation_deg = float(view_config["elevation_deg"])
        set_camera_pose(camera_obj, azimuth_deg, elevation_deg, camera_radius)
        bpy.context.scene.frame_set(view_idx + 1)

        rgb_path = os.path.join(rgb_dir, f"view_{view_idx}.png")
        bpy.context.scene.render.engine = rgb_engine
        if rgb_engine == "CYCLES":
            bpy.context.scene.cycles.samples = rgb_samples
            bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.render.filepath = rgb_path
        bpy.ops.render.render(write_still=True)

        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.samples = 1
        bpy.context.scene.cycles.use_denoising = False
        try:
            bpy.ops.render.render(write_still=False)
        finally:
            bpy.context.scene.render.engine = rgb_engine

        exr_candidates = glob.glob(os.path.join(output_folder, "_temp_idx_*.exr"))
        if len(exr_candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one Object Index EXR after rendering view {view_idx}, found {len(exr_candidates)}"
            )
        exr_path = exr_candidates[0]
        width = bpy.context.scene.render.resolution_x
        height = bpy.context.scene.render.resolution_y
        index_map = read_object_index_exr(exr_path, width, height)
        mask_files: dict[str, str] = {}
        for raw_part in target_parts:
            part = _require_mapping(raw_part, "target_parts[]")
            part_key = _require_string(part.get("part_key"), "target_parts[].part_key")
            label = _require_int(part.get("label"), f"target_parts[{part_key!r}].label")
            part_mask_dir = os.path.join(mask_dir, part_key)
            os.makedirs(part_mask_dir, exist_ok=True)
            binary_mask = (index_map == label).astype(np.uint8)
            np.save(os.path.join(part_mask_dir, f"mask_{view_idx}.npy"), binary_mask)
            save_binary_mask_png(os.path.join(part_mask_dir, f"mask_{view_idx}.png"), binary_mask)
            mask_files[part_key] = f"mask/{part_key}/mask_{view_idx}.npy"

        remaining_mask_dir = os.path.join(mask_dir, REMAINING_MASK_DIR)
        os.makedirs(remaining_mask_dir, exist_ok=True)
        remaining_mask = (index_map == remaining_pass_index).astype(np.uint8)
        np.save(os.path.join(remaining_mask_dir, f"mask_{view_idx}.npy"), remaining_mask)
        save_binary_mask_png(os.path.join(remaining_mask_dir, f"mask_{view_idx}.png"), remaining_mask)
        mask_files[REMAINING_MASK_DIR] = f"mask/{REMAINING_MASK_DIR}/mask_{view_idx}.npy"
        os.remove(exr_path)

        return {
            "file_path": f"rgb/view_{view_idx}.png",
            "mask_files": mask_files,
            "view_index": view_idx,
            "source": str(view_config.get("source", "unknown")),
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "camera_lens": DEFAULT_LENS_MM,
            "camera_sensor_width": DEFAULT_SENSOR_WIDTH_MM,
            "camera_angle_x": camera_obj.data.angle_x,
            "camera_radius": camera_radius,
            "transform_matrix": get_transform_matrix(camera_obj),
        }

    args = parse_blender_args(argv)
    with open(args.payload_json, "r", encoding="utf-8") as handle:
        payload = _require_mapping(json.load(handle), f"payload[{args.payload_json}]")
    with open(args.views_json, "r", encoding="utf-8") as handle:
        views = _require_list(json.load(handle), f"views[{args.views_json}]")

    output_folder = os.path.abspath(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)
    rgb_dir = os.path.join(output_folder, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    mask_dir = os.path.join(output_folder, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    init_render(args.engine, args.resolution, args.samples)
    init_scene()
    target_parts = _require_list(payload.get("target_parts"), "payload['target_parts']")
    target_labels: list[int] = []
    target_part_keys: list[str] = []
    for raw_part in target_parts:
        part = _require_mapping(raw_part, "payload['target_parts'][]")
        part_key = _validate_mask_component_name(
            _require_string(part.get("part_key"), "payload['target_parts'][].part_key"),
            "payload target part key",
        )
        label = _require_int(part.get("label"), f"payload['target_parts'][{part_key!r}].label")
        if label <= 0:
            raise ValueError(f"payload['target_parts'][{part_key!r}].label must be positive")
        target_part_keys.append(part_key)
        target_labels.append(label)
    if len(target_part_keys) != len(set(target_part_keys)):
        raise ValueError(f"Duplicate target part keys: {target_part_keys}")
    if REMAINING_MASK_DIR in target_part_keys:
        raise ValueError(f"Target part key conflicts with reserved mask directory {REMAINING_MASK_DIR!r}")
    if len(target_labels) != len(set(target_labels)):
        raise ValueError(f"Duplicate target labels: {target_labels}")
    if not target_labels:
        raise ValueError("payload['target_parts'] must contain at least one target")
    remaining_pass_index = max(target_labels) + 1
    load_parts_with_target_pass_indices(
        args.objs_dir,
        args.finaljson,
        _require_mapping(payload.get("angle_payload"), "payload['angle_payload']"),
        target_parts,
        remaining_pass_index,
        args.obj_up_axis,
    )
    force_opaque_materials()
    scale, offset = normalize_scene()
    bbox_min, bbox_max = scene_bbox()
    bbox_size = bbox_max - bbox_min
    camera_radius = DEFAULT_LENS_MM / DEFAULT_SENSOR_WIDTH_MM * math.sqrt(
        bbox_size.x**2 + bbox_size.y**2 + bbox_size.z**2
    )

    camera_obj = init_camera()
    init_lighting()
    for obj in bpy.data.objects:
        if obj.type not in ("MESH", "CAMERA", "LIGHT"):
            obj.hide_render = True
            obj.hide_viewport = True

    bpy.context.view_layer.use_pass_object_index = True
    previous_engine = bpy.context.scene.render.engine
    bpy.context.scene.render.engine = "CYCLES"
    configure_object_index_output(output_folder)
    bpy.context.scene.render.engine = previous_engine

    frames = []
    for view_idx, raw_view in enumerate(views):
        view_config = _require_mapping(raw_view, f"views[{view_idx}]")
        frame_data = render_view(
            camera_obj,
            output_folder,
            rgb_dir,
            mask_dir,
            target_parts,
            remaining_pass_index,
            view_idx,
            view_config,
            camera_radius,
            args.engine,
            args.samples,
        )
        frames.append(frame_data)
        print(
            f"Rendered rgb/view_{view_idx}.png and separated masks "
            f"({frame_data['source']}, azimuth={frame_data['azimuth_deg']:.2f}, "
            f"elevation={frame_data['elevation_deg']:.2f})"
        )

    cleanup_temp_exr(output_folder)
    camera_transforms = {
        "object_id": payload["object_id"],
        "angle_idx": payload["angle_idx"],
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "resolution": args.resolution,
        "camera_lens": DEFAULT_LENS_MM,
        "camera_sensor_width": DEFAULT_SENSOR_WIDTH_MM,
        "camera_radius": camera_radius,
        "total_views": len(views),
        "view_sampling": "OmniPart-style: 8 fixed 25-degree elevation azimuths plus deterministic random sphere views",
        "frames": frames,
    }
    with open(os.path.join(output_folder, "camera_transforms.json"), "w", encoding="utf-8") as handle:
        json.dump(camera_transforms, handle, indent=2)
        handle.write("\n")

    mask_labels = {
        "object_id": payload["object_id"],
        "angle_idx": payload["angle_idx"],
        "mask_format": "binary uint8 masks; 1 means the component is visible in that pixel, 0 means not visible",
        "mask_rule": payload["mask_rule"],
        "movable_parts": target_parts,
        "remaining": {
            "directory": REMAINING_MASK_DIR,
            "blender_pass_index": remaining_pass_index,
            "description": (
                "Merged mask for all visible object pixels that are not in the valid movable target parts."
            ),
        },
    }
    with open(os.path.join(output_folder, "mask_labels.json"), "w", encoding="utf-8") as handle:
        json.dump(mask_labels, handle, indent=2)
        handle.write("\n")

    print(f"Completed {len(views)} part-complete renders in {output_folder}")
    return 0


if __name__ == "__main__":
    if "--blender-worker" in sys.argv:
        raise SystemExit(main_blender_worker(sys.argv))
    raise SystemExit(main_driver())
