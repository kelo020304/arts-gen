#!/usr/bin/env python3
"""Build per-object/angle canonical transforms independent of rendering.

The canonical transform is the single geometry normalization record shared by
voxelization and render branches.  It maps the transformed raw mesh coordinates
for one object angle into the pipeline cube [-0.5, 0.5]^3:

    canonical_xyz = raw_xyz * scale + offset

This intentionally mirrors the normalization previously embedded in
``camera_transforms.json`` so voxelization no longer has to depend on a render
artifact being produced first.
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
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import load_config, resolve_obj_up_axis  # noqa: E402


SCHEMA_VERSION = "v1-canonical-transform"
CANONICAL_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])


@dataclass(frozen=True)
class PartSpec:
    canonical_name: str
    part_index: int
    obj_files: tuple[str, ...]


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
    return float(value)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return _require_mapping(json.load(handle), f"{name}[{path}]")


def _parse_object_ids(raw_value: str) -> list[str]:
    object_ids = [item.strip() for item in raw_value.split(",")]
    if not object_ids or any(not item for item in object_ids):
        raise ValueError("--object-ids must be a comma-separated list of non-empty IDs")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("--object-ids contains duplicate IDs")
    return object_ids


def _parse_angle_ids(raw_value: str | None, num_angles: int) -> list[int]:
    if raw_value is None:
        return list(range(num_angles))
    angle_ids: list[int] = []
    for item in raw_value.split(","):
        text = item.strip()
        if not text:
            raise ValueError("--angle-ids must be a comma-separated list of integers")
        angle_idx = int(text)
        if angle_idx < 0 or angle_idx >= num_angles:
            raise ValueError(f"angle index {angle_idx} outside [0, {num_angles})")
        angle_ids.append(angle_idx)
    if len(angle_ids) != len(set(angle_ids)):
        raise ValueError("--angle-ids contains duplicate indices")
    return angle_ids


def _resolve_object_ids(config, object_ids_arg: str | None) -> list[str]:
    available_object_ids = config.list_object_ids()
    if object_ids_arg is None:
        return available_object_ids
    requested = _parse_object_ids(object_ids_arg)
    missing = sorted(set(requested) - set(available_object_ids))
    if missing:
        raise ValueError("Unknown or filtered-out object IDs in --object-ids: " + ", ".join(missing))
    return requested


def _validate_matrix4x4(value: Any, name: str) -> np.ndarray:
    rows = _require_list(value, name)
    if len(rows) != 4:
        raise ValueError(f"{name} must contain 4 rows, got {len(rows)}")
    matrix = np.zeros((4, 4), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        row_values = _require_list(row, f"{name}[{row_idx}]")
        if len(row_values) != 4:
            raise ValueError(f"{name}[{row_idx}] must contain 4 values, got {len(row_values)}")
        for col_idx, item in enumerate(row_values):
            matrix[row_idx, col_idx] = _require_number(item, f"{name}[{row_idx}][{col_idx}]")
    return matrix


def load_part_specs(part_info_path: Path, object_id: str) -> list[PartSpec]:
    part_info = _load_json(part_info_path, "part_info")
    loaded_object_id = _require_string(part_info.get("object_id"), "part_info['object_id']")
    if loaded_object_id != object_id:
        raise ValueError(f"part_info object_id mismatch: expected {object_id}, got {loaded_object_id}")

    num_parts = _require_int(part_info.get("num_parts"), "part_info['num_parts']")
    label_to_key = _require_mapping(part_info.get("label_to_key"), "part_info['label_to_key']")
    parts = _require_mapping(part_info.get("parts"), "part_info['parts']")

    specs: list[PartSpec] = []
    for part_index in range(num_parts):
        canonical_name = _require_string(
            label_to_key.get(str(part_index)), f"part_info['label_to_key']['{part_index}']"
        )
        part_entry = _require_mapping(parts.get(canonical_name), f"part_info['parts']['{canonical_name}']")
        loaded_part_index = _require_int(part_entry.get("part_index"), f"parts['{canonical_name}']['part_index']")
        if loaded_part_index != part_index:
            raise ValueError(
                f"part_info part_index mismatch for {canonical_name}: expected {part_index}, got {loaded_part_index}"
            )
        obj_files = tuple(
            _require_string(obj_name, f"parts['{canonical_name}']['obj_files'][{obj_idx}]")
            for obj_idx, obj_name in enumerate(
                _require_list(part_entry.get("obj_files"), f"parts['{canonical_name}']['obj_files']")
            )
        )
        if not obj_files:
            raise ValueError(f"parts['{canonical_name}']['obj_files'] must not be empty")
        specs.append(PartSpec(canonical_name=canonical_name, part_index=part_index, obj_files=obj_files))
    return specs


def load_joint_archive(path: Path, object_id: str, expected_num_parts: int, expected_num_angles: int) -> dict[str, Any]:
    archive = _load_json(path, "joint_transforms")
    loaded_object_id = _require_string(archive.get("object_id"), "joint_transforms['object_id']")
    if loaded_object_id != object_id:
        raise ValueError(f"joint_transforms object_id mismatch: expected {object_id}, got {loaded_object_id}")
    num_parts = _require_int(archive.get("num_parts"), "joint_transforms['num_parts']")
    if num_parts != expected_num_parts:
        raise ValueError(f"joint_transforms num_parts mismatch: expected {expected_num_parts}, got {num_parts}")
    angles = _require_mapping(archive.get("angles"), "joint_transforms['angles']")
    expected_angle_keys = {str(angle_idx) for angle_idx in range(expected_num_angles)}
    actual_angle_keys = set(angles)
    if expected_angle_keys != actual_angle_keys:
        missing = sorted(expected_angle_keys - actual_angle_keys)
        extra = sorted(actual_angle_keys - expected_angle_keys)
        raise KeyError(f"joint_transforms angle mismatch; missing={missing}, extra={extra}")
    return archive


def extract_part_transforms(archive: dict[str, Any], angle_idx: int, expected_num_parts: int) -> dict[int, np.ndarray]:
    angles = _require_mapping(archive["angles"], "joint_transforms['angles']")
    angle_data = _require_mapping(angles.get(str(angle_idx)), f"joint_transforms['angles']['{angle_idx}']")
    part_transforms = _require_mapping(angle_data.get("part_transforms"), f"angles['{angle_idx}']['part_transforms']")
    return {
        part_idx: _validate_matrix4x4(part_transforms.get(str(part_idx)), f"part_transforms['{part_idx}']")
        for part_idx in range(expected_num_parts)
    }


def convert_part_transform_to_blender_space(raw_matrix: np.ndarray, obj_up_axis: str) -> np.ndarray:
    obj_up_axis = obj_up_axis.upper()
    if obj_up_axis == "Y":
        return Y_UP_TO_Z_UP @ raw_matrix
    if obj_up_axis == "Z":
        return raw_matrix
    raise ValueError(f"obj_up_axis must be 'Y' or 'Z', got {obj_up_axis!r}")


def load_obj_geometry_fast(path: Path) -> trimesh.Trimesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:]])
    vertices_array = np.asarray(vertices, dtype=np.float64)
    if not faces:
        faces_array = np.empty((0, 3), dtype=np.int64)
    else:
        face_sizes = {len(face) for face in faces}
        if len(face_sizes) != 1:
            raise ValueError(f"Mixed face sizes are not supported in {path}")
        face_size = next(iter(face_sizes))
        if face_size not in (3, 4):
            raise ValueError(f"Unsupported face size {face_size} in {path}")
        faces_array = np.asarray(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices_array, faces_array, process=False)


def load_part_meshes(part_specs: list[PartSpec], obj_dir: Path) -> dict[int, trimesh.Trimesh]:
    if not obj_dir.is_dir():
        raise FileNotFoundError(f"Missing OBJ directory: {obj_dir}")
    meshes_by_part: dict[int, trimesh.Trimesh] = {}
    for spec in part_specs:
        meshes: list[trimesh.Trimesh] = []
        for obj_name in spec.obj_files:
            obj_path = obj_dir / f"{obj_name}.obj"
            if not obj_path.is_file():
                raise FileNotFoundError(f"Missing OBJ for part '{spec.canonical_name}': {obj_path}")
            meshes.append(load_obj_geometry_fast(obj_path))
        combined = trimesh.util.concatenate(meshes)
        if len(combined.vertices) == 0:
            raise ValueError(f"Empty combined mesh for part '{spec.canonical_name}'")
        meshes_by_part[spec.part_index] = combined
    return meshes_by_part


def _matrix_payload(scale: float, offset: np.ndarray) -> tuple[list[list[float]], list[list[float]]]:
    raw_to_canonical = np.eye(4, dtype=np.float64)
    raw_to_canonical[0, 0] = scale
    raw_to_canonical[1, 1] = scale
    raw_to_canonical[2, 2] = scale
    raw_to_canonical[:3, 3] = offset
    canonical_to_raw = np.linalg.inv(raw_to_canonical)
    return raw_to_canonical.tolist(), canonical_to_raw.tolist()


def compute_transform_payload(config, object_id: str, angle_idx: int) -> dict[str, Any]:
    data_root = Path(config.data_root)
    num_angles = config.get_num_angles(object_id)
    part_specs = load_part_specs(data_root / "part_info" / object_id / "part_info.json", object_id)
    joint_archive = load_joint_archive(
        Path(config.joint_transforms_dir) / f"{object_id}.json",
        object_id,
        expected_num_parts=len(part_specs),
        expected_num_angles=num_angles,
    )
    part_transforms = extract_part_transforms(joint_archive, angle_idx, len(part_specs))
    part_meshes = load_part_meshes(part_specs, Path(config.raw_dir) / "partseg" / object_id / "objs")
    obj_up_axis = resolve_obj_up_axis(config, object_id)

    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for spec in part_specs:
        mesh = part_meshes[spec.part_index].copy()
        mesh.apply_transform(convert_part_transform_to_blender_space(part_transforms[spec.part_index], obj_up_axis))
        if len(mesh.vertices) == 0:
            raise ValueError(f"Empty transformed mesh for part '{spec.canonical_name}'")
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        mins.append(bounds[0])
        maxs.append(bounds[1])

    raw_min = np.min(np.vstack(mins), axis=0)
    raw_max = np.max(np.vstack(maxs), axis=0)
    extent = raw_max - raw_min
    if not np.isfinite(extent).all() or np.max(extent) <= 0.0:
        raise ValueError(f"Invalid raw AABB extent for {object_id}/angle_{angle_idx}: {extent.tolist()}")

    scale = float(1.0 / np.max(extent))
    offset = -((raw_min * scale) + (raw_max * scale)) / 2.0
    raw_to_canonical, canonical_to_raw = _matrix_payload(scale, offset)

    return {
        "dataset": config.dataset_name,
        "schema_version": SCHEMA_VERSION,
        "object_id": object_id,
        "angle_index": int(angle_idx),
        "source": "computed_from_transformed_mesh_aabb",
        "obj_up_axis": obj_up_axis,
        "raw_aabb": {"min": raw_min.tolist(), "max": raw_max.tolist()},
        "canonical_aabb": {"min": CANONICAL_AABB[0], "max": CANONICAL_AABB[1]},
        # Compatibility fields intentionally match camera_transforms.json.
        "aabb": CANONICAL_AABB,
        "scale": scale,
        "offset": offset.tolist(),
        "raw_to_canonical": raw_to_canonical,
        "canonical_to_raw": canonical_to_raw,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_camera_transform_payload(camera_path: Path, config, object_id: str, angle_idx: int) -> dict[str, Any]:
    camera = _load_json(camera_path, "camera_transforms")
    scale = _require_number(camera.get("scale"), "camera_transforms['scale']")
    offset = np.asarray(
        [_require_number(v, f"camera_transforms['offset'][{idx}]") for idx, v in enumerate(_require_list(camera.get("offset"), "camera_transforms['offset']"))],
        dtype=np.float64,
    )
    if len(offset) != 3:
        raise ValueError("camera_transforms['offset'] must contain 3 values")
    raw_to_canonical, canonical_to_raw = _matrix_payload(scale, offset)
    return {
        "dataset": config.dataset_name,
        "schema_version": SCHEMA_VERSION,
        "object_id": object_id,
        "angle_index": int(angle_idx),
        "source": "copied_from_camera_transforms",
        "camera_transforms_path": str(camera_path),
        "canonical_aabb": {"min": CANONICAL_AABB[0], "max": CANONICAL_AABB[1]},
        "aabb": camera.get("aabb", CANONICAL_AABB),
        "scale": scale,
        "offset": offset.tolist(),
        "raw_to_canonical": raw_to_canonical,
        "canonical_to_raw": canonical_to_raw,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _attach_camera_diff(payload: dict[str, Any], camera_path: Path) -> None:
    if not camera_path.is_file():
        payload["camera_transforms_comparison"] = {"status": "missing", "path": str(camera_path)}
        return
    camera = _load_json(camera_path, "camera_transforms")
    camera_scale = _require_number(camera.get("scale"), "camera_transforms['scale']")
    camera_offset = np.asarray(
        [_require_number(v, f"camera_transforms['offset'][{idx}]") for idx, v in enumerate(_require_list(camera.get("offset"), "camera_transforms['offset']"))],
        dtype=np.float64,
    )
    offset = np.asarray(payload["offset"], dtype=np.float64)
    payload["camera_transforms_comparison"] = {
        "status": "compared",
        "path": str(camera_path),
        "scale_abs_diff": abs(float(payload["scale"]) - camera_scale),
        "offset_abs_diff_max": float(np.max(np.abs(offset - camera_offset))),
    }


def build_one(config, object_id: str, angle_idx: int, source: str, compare_camera: bool) -> dict[str, Any]:
    camera_path = Path(config.renders_dir) / object_id / f"angle_{angle_idx}" / "camera_transforms.json"
    if source == "camera":
        payload = _load_camera_transform_payload(camera_path, config, object_id, angle_idx)
    elif source == "compute":
        payload = compute_transform_payload(config, object_id, angle_idx)
    elif source == "auto":
        payload = (
            _load_camera_transform_payload(camera_path, config, object_id, angle_idx)
            if camera_path.is_file()
            else compute_transform_payload(config, object_id, angle_idx)
        )
    else:
        raise ValueError(f"unsupported source: {source}")
    if compare_camera:
        _attach_camera_diff(payload, camera_path)
    return payload


def canonical_transform_path(config, object_id: str, angle_idx: int, out_root: Path | None = None) -> Path:
    root = Path(config.data_root) if out_root is None else out_root
    return root / "canonical_transforms" / object_id / f"angle_{angle_idx}" / "canonical_transform.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical_transform.json files for dataset_toolkits.")
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset.")
    parser.add_argument("--angle-ids", help="Optional comma-separated angle subset.")
    parser.add_argument(
        "--source",
        choices=("compute", "camera", "auto"),
        default="auto",
        help=(
            "Normalization source. auto copies existing camera_transforms.json when present "
            "for backward-compatible alignment, otherwise computes from transformed mesh AABB."
        ),
    )
    parser.add_argument("--compare-camera", action="store_true", help="Record numeric diff against camera_transforms.json when present.")
    parser.add_argument(
        "--out-root",
        help="Optional output root for smoke tests. Default is config data_root.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing canonical_transform.json files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    object_ids = _resolve_object_ids(config, args.object_ids)
    out_root = Path(args.out_root) if args.out_root else None

    written = 0
    skipped = 0
    planned = 0
    for object_id in object_ids:
        angle_ids = _parse_angle_ids(args.angle_ids, config.get_num_angles(object_id))
        for angle_idx in angle_ids:
            out_path = canonical_transform_path(config, object_id, angle_idx, out_root=out_root)
            planned += 1
            if out_path.is_file() and not args.force:
                skipped += 1
                print(f"[canonical] skip existing {out_path}")
                continue
            if args.dry_run:
                print(f"[canonical] would write {out_path}")
                continue
            payload = build_one(config, object_id, angle_idx, args.source, args.compare_camera)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            written += 1
            comparison = payload.get("camera_transforms_comparison") or {}
            diff = ""
            if comparison.get("status") == "compared":
                diff = (
                    f" camera_diff(scale={comparison['scale_abs_diff']:.3g}, "
                    f"offset={comparison['offset_abs_diff_max']:.3g})"
                )
            print(
                f"[canonical] wrote {out_path} scale={float(payload['scale']):.17g} "
                f"offset={payload['offset']}{diff}",
                flush=True,
            )
    print(f"[canonical] planned={planned} written={written} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
