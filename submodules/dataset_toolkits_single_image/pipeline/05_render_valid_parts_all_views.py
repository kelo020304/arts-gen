#!/usr/bin/env python3
"""
Render valid target parts from the Step 04 valid-parts manifest with TRELLIS-style 150 views.

Validity for this asset generation is:
    movable part (finaljson group_info motion type A/B/C)
    ∩ manifest angle part with has_voxel_ind=true and num_voxels > 5

The rendered part is placed in the same full-object canonical frame as Step 02
by applying the angle's `camera_transforms.json` scale/offset after the
per-part joint transform.  Therefore each target part keeps its correct
articulated pose and object-relative location.

Output layout:
    renders/<object_id>/angle_<i>/render_part_all_view/<part_key>/
        000.png
        ...
        149.png
        transforms.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_OUTPUT_SUBDIR = "render_part_all_view"
DEFAULT_NUM_VIEWS = 150
DEFAULT_FOV_DEG = 40.0
DEFAULT_RADIUS = 2.0
DEFAULT_TIMEOUT_SECONDS = 7200
ACCEPTED_MOTION_TYPES = {"A", "B", "C"}
MIN_PART_VOXELS = 5


@dataclass(frozen=True)
class TargetPart:
    part_key: str
    part_index: int
    obj_files: tuple[str, ...]
    motion_type: str
    group_id: str


@dataclass(frozen=True)
class JobSpec:
    object_id: str
    angle_idx: int
    part: TargetPart
    objs_dir: Path
    output_dir: Path
    blender_binary: str
    resolution: int
    num_views: int
    fov_deg: float
    radius: float
    engine: str
    samples: int
    seed: int
    obj_up_axis: str
    part_transform: list[list[float]]
    normalization: dict[str, Any]

    @property
    def desc(self) -> str:
        return f"{self.object_id} angle_{self.angle_idx} {self.part.part_key}"


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


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _validate_matrix4x4(value: Any, name: str) -> list[list[float]]:
    rows = _require_list(value, name)
    if len(rows) != 4:
        raise ValueError(f"{name} must contain 4 rows, got {len(rows)}")
    matrix: list[list[float]] = []
    for row_idx, row in enumerate(rows):
        row_values = _require_list(row, f"{name}[{row_idx}]")
        if len(row_values) != 4:
            raise ValueError(f"{name}[{row_idx}] must contain 4 values, got {len(row_values)}")
        matrix.append(
            [
                _require_number(item, f"{name}[{row_idx}][{col_idx}]")
                for col_idx, item in enumerate(row_values)
            ]
        )
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


def radical_inverse(base: int, n: int) -> float:
    value = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        value += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return value


def sphere_hammersley_sequence(n: int, num_samples: int, offset: tuple[float, float]) -> tuple[float, float]:
    """Match TRELLIS dataset_toolkits/utils.py sphere_hammersley_sequence."""
    u = n / num_samples
    v = radical_inverse(2, n)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = math.acos(1 - 2 * u) - math.pi / 2
    phi = v * 2 * math.pi
    return phi, theta


def deterministic_view_offset(object_id: str, angle_idx: int, seed: int) -> tuple[float, float]:
    """Match full-object 150-view sampling for the same object/angle.

    Do not include `part_key` here: render_full_obj_all_view/<idx>.png and
    render_part_all_view/<part>/<idx>.png must share the same camera view.
    """
    digest = hashlib.sha256(f"{seed}:{object_id}:{angle_idx}".encode("utf-8")).digest()
    denom = float(1 << 64)
    return (
        int.from_bytes(digest[:8], "big") / denom,
        int.from_bytes(digest[8:16], "big") / denom,
    )


def build_trellis_views(
    *,
    object_id: str,
    angle_idx: int,
    num_views: int,
    radius: float,
    fov_deg: float,
    seed: int,
) -> list[dict[str, float]]:
    if num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if not 0 < fov_deg < 180:
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")

    offset = deterministic_view_offset(object_id, angle_idx, seed)
    fov = math.radians(fov_deg)
    views = []
    for view_idx in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(view_idx, num_views, offset)
        views.append({"yaw": yaw, "pitch": pitch, "radius": radius, "fov": fov})
    return views


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


def load_movable_parts(finaljson_path: Path, part_info_path: Path, object_id: str) -> dict[str, TargetPart]:
    finaljson = _load_json(finaljson_path, "finaljson")
    part_info = _load_json(part_info_path, "part_info")
    label_to_key = _require_mapping(part_info.get("label_to_key"), "part_info['label_to_key']")
    parts = _require_mapping(part_info.get("parts"), "part_info['parts']")
    group_info = _require_mapping(finaljson.get("group_info"), "finaljson['group_info']")

    targets: dict[str, TargetPart] = {}
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
            part_index = _require_int(part_entry.get("part_index"), f"part_info['parts']['{key}']['part_index']")
            obj_files = tuple(
                _require_string(obj_name, f"part_info['parts']['{key}']['obj_files'][{idx}]")
                for idx, obj_name in enumerate(
                    _require_list(part_entry.get("obj_files"), f"part_info['parts']['{key}']['obj_files']")
                )
            )
            if not obj_files:
                raise ValueError(f"part_info['parts']['{key}']['obj_files'] must not be empty")
            targets[key] = TargetPart(
                part_key=key,
                part_index=part_index,
                obj_files=obj_files,
                motion_type=motion_type,
                group_id=group_id,
            )
    return targets


def load_angle_transform(joint_transforms_path: Path, object_id: str, angle_idx: int, part_index: int) -> list[list[float]]:
    archive = _load_json(joint_transforms_path, "joint_transforms")
    loaded_object_id = str(archive.get("object_id"))
    if loaded_object_id != object_id:
        raise ValueError(
            f"joint_transforms object_id mismatch: expected {object_id}, got {loaded_object_id}"
        )
    angles = _require_mapping(archive.get("angles"), "joint_transforms['angles']")
    angle_data = _require_mapping(
        angles.get(str(angle_idx)),
        f"joint_transforms['angles']['{angle_idx}']",
    )
    part_transforms = _require_mapping(
        angle_data.get("part_transforms"),
        f"joint_transforms['angles']['{angle_idx}']['part_transforms']",
    )
    return _validate_matrix4x4(
        part_transforms.get(str(part_index)),
        f"joint_transforms angle_{angle_idx} part_transforms['{part_index}']",
    )


def load_canonical_normalization(camera_transforms_path: Path) -> dict[str, Any]:
    payload = _load_json(camera_transforms_path, "camera_transforms")
    scale = _require_number(payload.get("scale"), "camera_transforms['scale']")
    if scale <= 0:
        raise ValueError(f"camera_transforms['scale'] must be positive, got {scale}")
    offset = _require_list(payload.get("offset"), "camera_transforms['offset']")
    if len(offset) != 3:
        raise ValueError(f"camera_transforms['offset'] must contain 3 values, got {len(offset)}")
    offset = [
        _require_number(value, f"camera_transforms['offset'][{idx}]")
        for idx, value in enumerate(offset)
    ]
    return {
        "scale": scale,
        "offset": offset,
        "source": camera_transforms_path.as_posix(),
    }


def _output_is_complete(output_dir: Path, num_views: int) -> bool:
    transforms_path = output_dir / "transforms.json"
    if not transforms_path.is_file():
        return False
    return all((output_dir / f"{view_idx:03d}.png").is_file() for view_idx in range(num_views))


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
    fov_deg: float,
    radius: float,
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
        "voxel_filtered_parts": 0,
        "skipped_existing": 0,
        "angles_without_targets": 0,
    }
    jobs: list[JobSpec] = []

    for object_id in object_ids:
        finaljson_path = finaljson_root / f"{object_id}.json"
        part_info_path = part_info_root / object_id / "part_info.json"
        joint_transforms_path = joint_root / f"{object_id}.json"
        objs_dir = objs_root / object_id / "objs"
        if not objs_dir.is_dir():
            raise FileNotFoundError(f"Missing OBJ directory: {objs_dir}")

        movable_parts = load_movable_parts(finaljson_path, part_info_path, object_id)
        if part_keys is not None:
            movable_parts = {
                key: part for key, part in movable_parts.items()
                if key in part_keys
            }
        stats["movable_parts"] += len(movable_parts)
        if not movable_parts:
            continue

        object_num_angles = config.get_num_angles(object_id)
        selected_angles = angle_ids if angle_ids is not None else list(range(object_num_angles))
        invalid_angles = [angle_idx for angle_idx in selected_angles if angle_idx >= object_num_angles]
        if invalid_angles:
            raise ValueError(
                f"Object {object_id} has {object_num_angles} angle(s); invalid --angle-ids: {invalid_angles}"
            )
        obj_up_axis = resolve_obj_up_axis(config, finaljson_path)

        for angle_idx in selected_angles:
            manifest_parts = _manifest_angle_parts(manifest, object_id, angle_idx)
            normalization = load_canonical_normalization(
                renders_root / object_id / f"angle_{angle_idx}" / "camera_transforms.json"
            )
            valid_parts_for_angle = 0
            for part_key, part in movable_parts.items():
                part_record = manifest_parts.get(part_key)
                if part_record is None:
                    raise KeyError(
                        f"manifest object {object_id} angle_{angle_idx} missing movable part '{part_key}'"
                    )
                part_payload = _require_mapping(
                    part_record,
                    f"manifest object {object_id} angle_{angle_idx} parts['{part_key}']",
                )
                context = f"manifest object {object_id} angle_{angle_idx} parts"
                if not _is_manifest_voxel_kept(part_payload, part_key, context):
                    stats["voxel_filtered_parts"] += 1
                    continue
                valid_parts_for_angle += 1

                output_dir = renders_root / object_id / f"angle_{angle_idx}" / output_subdir / part_key
                if not force and _output_is_complete(output_dir, num_views):
                    stats["skipped_existing"] += 1
                    continue

                jobs.append(
                    JobSpec(
                        object_id=object_id,
                        angle_idx=angle_idx,
                        part=part,
                        objs_dir=objs_dir,
                        output_dir=output_dir,
                        blender_binary=config.render.blender,
                        resolution=render_resolution,
                        num_views=num_views,
                        fov_deg=fov_deg,
                        radius=radius,
                        engine=engine,
                        samples=samples,
                        seed=seed,
                        obj_up_axis=obj_up_axis,
                        part_transform=load_angle_transform(
                            joint_transforms_path,
                            object_id,
                            angle_idx,
                            part.part_index,
                        ),
                        normalization=normalization,
                    )
                )
            if valid_parts_for_angle == 0:
                stats["angles_without_targets"] += 1

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


def run_job(spec: JobSpec, timeout_seconds: int) -> str:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_id": spec.object_id,
        "angle_idx": spec.angle_idx,
        "part_key": spec.part.part_key,
        "part_index": spec.part.part_index,
        "obj_files": list(spec.part.obj_files),
        "motion_type": spec.part.motion_type,
        "group_id": spec.part.group_id,
        "part_transform": spec.part_transform,
        "normalization": spec.normalization,
    }
    views = build_trellis_views(
        object_id=spec.object_id,
        angle_idx=spec.angle_idx,
        num_views=spec.num_views,
        radius=spec.radius,
        fov_deg=spec.fov_deg,
        seed=spec.seed,
    )
    payload_path = _write_temp_json(
        prefix=f"render_part_{spec.object_id}_angle_{spec.angle_idx}_{spec.part.part_key}_payload_",
        payload=payload,
    )
    views_path = _write_temp_json(
        prefix=f"render_part_{spec.object_id}_angle_{spec.angle_idx}_{spec.part.part_key}_views_",
        payload=views,
    )
    log_path = _create_temp_log_path(
        prefix=f"render_part_{spec.object_id}_angle_{spec.angle_idx}_{spec.part.part_key}_"
    )
    command = [
        spec.blender_binary,
        "--background",
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
        description="Render manifest-valid movable parts with 150 TRELLIS-style views."
    )
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument("--manifest", help="Manifest path; default is <data_root>/manifests/<dataset>.json.")
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset.")
    parser.add_argument("--angle-ids", help="Optional comma-separated angle subset.")
    parser.add_argument("--part-keys", help="Optional comma-separated canonical part key subset.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel Blender subprocess workers.")
    parser.add_argument("--num-views", type=int, default=DEFAULT_NUM_VIEWS, help="Views per valid part.")
    parser.add_argument("--resolution", type=int, default=None, help="Square render resolution; defaults to config.")
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT", help="Blender render engine.")
    parser.add_argument("--samples", type=int, default=128, help="Cycles samples when --engine CYCLES is used.")
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG, help="Camera FOV in degrees.")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS, help="Camera orbit radius.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic view offsets.")
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdirectory under renders/<object_id>/angle_<i>/ for part-view renders.",
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
        fov_deg=args.fov_deg,
        radius=args.radius,
        seed=args.seed,
        output_subdir=args.output_subdir,
        force=args.force,
    )

    print(
        "Prepared valid part render jobs: "
        f"objects={len(object_ids)} jobs={len(jobs)} "
        f"movable_parts={stats['movable_parts']} "
        f"voxel_filtered_parts={stats['voxel_filtered_parts']} "
        f"skipped_existing={stats['skipped_existing']} "
        f"angles_without_targets={stats['angles_without_targets']} "
        f"views_per_job={args.num_views}",
        flush=True,
    )
    for job in jobs[:20]:
        print(f"  {job.desc} -> {job.output_dir}", flush=True)
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
                f"({job.num_views} views)",
                flush=True,
            )

    print(
        f"Summary: objects={len(object_ids)} jobs={len(jobs)} "
        f"views_rendered={len(jobs) * args.num_views}",
        flush=True,
    )
    return 0


def parse_blender_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blender worker for valid part all-view rendering.")
    parser.add_argument("--blender-worker", action="store_true")
    parser.add_argument("--payload-json", required=True)
    parser.add_argument("--views-json", required=True)
    parser.add_argument("--objs-dir", required=True)
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
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

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
            try:
                scene.cycles.device = "GPU"
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.get_devices()
                prefs.compute_device_type = "CUDA"
            except Exception as exc:
                print(f"[WARN] Could not enable CUDA cycles rendering: {exc}", flush=True)

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
        cam.data.sensor_height = cam.data.sensor_width = 32
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

    def init_lighting() -> None:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        default_light = bpy.data.objects.new(
            "Default_Light",
            bpy.data.lights.new("Default_Light", type="POINT"),
        )
        bpy.context.collection.objects.link(default_light)
        default_light.data.energy = 1000
        default_light.location = (4, 1, 6)

        top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
        bpy.context.collection.objects.link(top_light)
        top_light.data.energy = 10000
        top_light.location = (0, 0, 10)
        top_light.scale = (100, 100, 100)

        bottom_light = bpy.data.objects.new(
            "Bottom_Light",
            bpy.data.lights.new("Bottom_Light", type="AREA"),
        )
        bpy.context.collection.objects.link(bottom_light)
        bottom_light.data.energy = 1000
        bottom_light.location = (0, 0, -10)

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

    def load_target_part(payload: dict[str, Any], objs_dir: str, obj_up_axis: str) -> None:
        obj_files = tuple(
            _require_string(obj_name, f"payload['obj_files'][{idx}]")
            for idx, obj_name in enumerate(_require_list(payload.get("obj_files"), "payload['obj_files']"))
        )
        raw_matrix = _validate_matrix4x4(payload.get("part_transform"), "payload['part_transform']")
        if obj_up_axis == "Y":
            part_matrix = y_up_to_z_up @ Matrix(raw_matrix) @ z_up_to_y_up
            import_kwargs = {}
        else:
            part_matrix = Matrix(raw_matrix)
            import_kwargs = {"forward_axis": "Y", "up_axis": "Z"}

        imported_mesh_count = 0
        for obj_name in obj_files:
            obj_path = os.path.join(objs_dir, f"{obj_name}.obj")
            if not os.path.isfile(obj_path):
                raise FileNotFoundError(f"Missing OBJ for part {payload.get('part_key')}: {obj_path}")
            before_objects = set(bpy.data.objects.keys())
            bpy.ops.wm.obj_import(filepath=obj_path, **import_kwargs)
            after_objects = set(bpy.data.objects.keys())
            for blender_name in after_objects - before_objects:
                obj = bpy.data.objects[blender_name]
                if obj.type != "MESH":
                    continue
                obj.matrix_world = part_matrix @ obj.matrix_world
                obj.name = f"part_{payload['part_key']}"
                if obj.data is not None:
                    obj.data.name = f"part_{payload['part_key']}"
                imported_mesh_count += 1

        if imported_mesh_count == 0:
            raise RuntimeError(f"No mesh objects were imported for part {payload.get('part_key')}")

    def apply_full_object_normalization(normalization: dict[str, Any]) -> None:
        scale = _require_number(normalization.get("scale"), "payload['normalization']['scale']")
        offset = _require_list(normalization.get("offset"), "payload['normalization']['offset']")
        if len(offset) != 3:
            raise ValueError("payload['normalization']['offset'] must contain 3 values")
        offset_vec = Vector(
            (
                _require_number(offset[0], "payload['normalization']['offset'][0]"),
                _require_number(offset[1], "payload['normalization']['offset'][1]"),
                _require_number(offset[2], "payload['normalization']['offset'][2]"),
            )
        )

        mesh_roots = [
            obj for obj in bpy.context.scene.objects.values()
            if obj.type == "MESH" and obj.parent is None
        ]
        if not mesh_roots:
            raise RuntimeError("No root mesh objects available for normalization")
        parent = bpy.data.objects.new("FullObjectNormalizationRoot", None)
        bpy.context.scene.collection.objects.link(parent)
        parent.empty_display_size = 0
        parent.hide_render = True
        for obj in mesh_roots:
            obj.parent = parent
        parent.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        parent.matrix_world.translation += offset_vec
        bpy.context.view_layer.update()

    args = parse_blender_args(argv)
    with open(args.payload_json, "r", encoding="utf-8") as handle:
        payload = _require_mapping(json.load(handle), f"payload[{args.payload_json}]")
    with open(args.views_json, "r", encoding="utf-8") as handle:
        views = _require_list(json.load(handle), f"views[{args.views_json}]")

    output_folder = os.path.abspath(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    init_render(args.engine, args.resolution, args.samples)
    init_scene()
    load_target_part(payload, args.objs_dir, args.obj_up_axis)
    apply_full_object_normalization(
        _require_mapping(payload.get("normalization"), "payload['normalization']")
    )
    cam = init_camera()
    init_lighting()

    to_export: dict[str, Any] = {
        "object_id": payload["object_id"],
        "angle_idx": payload["angle_idx"],
        "part_key": payload["part_key"],
        "part_index": payload["part_index"],
        "motion_type": payload["motion_type"],
        "group_id": payload["group_id"],
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "normalization": payload["normalization"],
        "resolution": args.resolution,
        "engine": args.engine,
        "samples": args.samples,
        "frames": [],
    }

    for view_idx, raw_view in enumerate(views):
        view = _require_mapping(raw_view, f"views[{view_idx}]")
        yaw = float(view["yaw"])
        pitch = float(view["pitch"])
        radius = float(view["radius"])
        fov = float(view["fov"])
        cam.location = (
            radius * np.cos(yaw) * np.cos(pitch),
            radius * np.sin(yaw) * np.cos(pitch),
            radius * np.sin(pitch),
        )
        cam.data.lens = 16 / np.tan(fov / 2)

        bpy.context.scene.render.filepath = os.path.join(output_folder, f"{view_idx:03d}.png")
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        to_export["frames"].append(
            {
                "file_path": f"{view_idx:03d}.png",
                "camera_angle_x": fov,
                "yaw": yaw,
                "pitch": pitch,
                "radius": radius,
                "transform_matrix": get_transform_matrix(cam),
            }
        )

    with open(os.path.join(output_folder, "transforms.json"), "w", encoding="utf-8") as handle:
        json.dump(to_export, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    if "--blender-worker" in sys.argv:
        raise SystemExit(main_blender_worker(sys.argv))
    raise SystemExit(main_driver())
