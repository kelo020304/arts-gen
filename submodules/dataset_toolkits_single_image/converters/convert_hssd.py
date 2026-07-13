#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
ZERO_TOLERANCE = 1e-6
UNIT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ObjectSource:
    object_id: str
    kind: str
    urdf_path: Path | None = None
    glb_path: Path | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert HSSD assets into dataset_toolkits raw/finaljson + raw/partseg format."
        )
    )
    parser.add_argument("--hssd-hab-dir", required=True, help="Path to HSSD _hssd_hab root.")
    parser.add_argument(
        "--hssd-objects-dir",
        required=True,
        help="Path to TRELLIS HSSD raw/objects directory.",
    )
    parser.add_argument(
        "--trellis-metadata",
        required=True,
        help="Path to TRELLIS metadata CSV.",
    )
    parser.add_argument(
        "--semantics-csv",
        required=True,
        help="Path to HSSD semantics/objects.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output HSSD dataset root that will contain raw/finaljson and raw/partseg.",
    )
    parser.add_argument(
        "--object-ids",
        help="Optional comma-separated object ID subset.",
    )
    return parser.parse_args(argv)


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _require_columns(fieldnames: list[str] | None, required: set[str], path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{path} is missing a header row")
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _parse_object_ids_arg(raw_value: str) -> list[str]:
    object_ids = [item.strip() for item in raw_value.split(",")]
    if not object_ids or any(not item for item in object_ids):
        raise ValueError("--object-ids must be a comma-separated list of non-empty IDs")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("--object-ids contains duplicate IDs")
    return object_ids


def _parse_float_list(raw_value: str, expected_len: int, description: str) -> list[float]:
    items = raw_value.replace(",", " ").split()
    if len(items) != expected_len:
        raise ValueError(
            f"{description} must contain {expected_len} numbers, got {len(items)}: {raw_value}"
        )
    return [float(item) for item in items]


def _is_zero_vector(values: list[float]) -> bool:
    return all(abs(value) <= ZERO_TOLERANCE for value in values)


def _is_unit_scale(values: list[float]) -> bool:
    return all(abs(value - 1.0) <= UNIT_TOLERANCE for value in values)


def _trim_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_dimension(bounds: np.ndarray) -> str:
    extents = bounds[1] - bounds[0]
    return "*".join(_trim_float(value) for value in extents.tolist())


def _combine_bounds(bounds_list: list[np.ndarray]) -> np.ndarray:
    if not bounds_list:
        raise ValueError("Cannot combine an empty bounds list")
    mins = np.vstack([bounds[0] for bounds in bounds_list])
    maxs = np.vstack([bounds[1] for bounds in bounds_list])
    return np.vstack([mins.min(axis=0), maxs.max(axis=0)])


def _synset_to_label(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    first = value.split(",")[0].strip()
    if not first:
        return ""
    return first.split(".", 1)[0].replace("_", " ").strip()


def _first_tag(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    return value.split(",")[0].strip()


def _extract_short_name(source_text: str) -> str:
    words = WORD_PATTERN.findall(source_text)
    while words and words[0].lower() in {"a", "an", "the"}:
        words = words[1:]
    if not words:
        raise ValueError(f"Unable to derive object_name from text: {source_text!r}")
    return " ".join(words[:4])


def _resolve_object_name(
    object_id: str,
    metadata_by_id: dict[str, dict[str, Any]],
    semantics_row: dict[str, str],
) -> str:
    metadata_row = metadata_by_id.get(object_id)
    if metadata_row is not None:
        captions_raw = metadata_row["captions_raw"].strip()
        if captions_raw:
            try:
                captions = json.loads(captions_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Metadata captions are invalid JSON for object {object_id}: {captions_raw!r}"
                ) from exc
            if not isinstance(captions, list) or not captions:
                raise ValueError(f"Metadata captions must be a non-empty JSON list for object {object_id}")
            if not all(isinstance(item, str) and item.strip() for item in captions):
                raise ValueError(
                    f"Metadata captions must contain non-empty strings for object {object_id}"
                )
            return _extract_short_name(captions[0])

    semantics_name = semantics_row["name"].strip() if semantics_row else ""
    if semantics_name:
        return _extract_short_name(semantics_name)
    return f"Object_{object_id[:8]}"


def _resolve_category(semantics_row: dict[str, str], kind: str) -> str:
    candidates = [
        semantics_row.get("main_category", "").strip(),
        semantics_row.get("super_category", "").strip(),
        _synset_to_label(semantics_row.get("main_wnsynsetkey", "")),
        _synset_to_label(semantics_row.get("wnsynsetkey", "")),
        _first_tag(semantics_row.get("floorplanner-category-tags", "")),
        semantics_row.get("name", "").strip(),
    ]
    for value in candidates:
        if value:
            return value
    return "Articulated Object" if kind == "articulated" else "Static Object"


def load_metadata_index(metadata_csv: Path) -> dict[str, dict[str, Any]]:
    _require_file(metadata_csv, "TRELLIS metadata CSV")
    index: dict[str, dict[str, Any]] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"file_identifier", "captions"}, metadata_csv)
        for row_idx, row in enumerate(reader, start=2):
            file_identifier = row["file_identifier"].strip()
            object_id = Path(file_identifier).stem
            if not object_id:
                raise ValueError(f"{metadata_csv}:{row_idx} has invalid file_identifier: {file_identifier!r}")
            if object_id in index:
                raise ValueError(f"Duplicate metadata row for object {object_id}")

            index[object_id] = {
                "file_identifier": file_identifier,
                "captions_raw": row["captions"],
            }
    return index


def load_semantics_index(semantics_csv: Path) -> dict[str, dict[str, str]]:
    _require_file(semantics_csv, "semantics CSV")
    required_columns = {
        "id",
        "name",
        "isArticulatable",
        "main_category",
        "super_category",
        "main_wnsynsetkey",
        "wnsynsetkey",
        "floorplanner-category-tags",
    }
    index: dict[str, dict[str, str]] = {}
    with semantics_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, required_columns, semantics_csv)
        for row_idx, row in enumerate(reader, start=2):
            object_id = row["id"].strip()
            if not object_id:
                raise ValueError(f"{semantics_csv}:{row_idx} has an empty id")
            if object_id in index:
                raise ValueError(f"Duplicate semantics row for object {object_id}")
            index[object_id] = row
    return index


def discover_articulated_urdfs(hssd_hab_dir: Path) -> dict[str, Path]:
    urdf_root = _require_directory(hssd_hab_dir / "urdf", "HSSD URDF directory")
    urdf_paths: dict[str, Path] = {}
    for urdf_path in sorted(urdf_root.glob("*/*.urdf")):
        object_id = urdf_path.stem
        if urdf_path.parent.name != object_id:
            raise ValueError(f"Unexpected URDF layout: {urdf_path}")
        if object_id in urdf_paths:
            raise ValueError(f"Duplicate URDF for object {object_id}")
        urdf_paths[object_id] = urdf_path
    return urdf_paths


def discover_static_glbs(objects_dir: Path) -> dict[str, Path]:
    _require_directory(objects_dir, "HSSD objects directory")
    glb_paths: dict[str, Path] = {}
    for bucket_dir in sorted(path for path in objects_dir.iterdir() if path.is_dir()):
        for glb_path in sorted(bucket_dir.glob("*.glb")):
            object_id = glb_path.stem
            if object_id in glb_paths:
                raise ValueError(f"Duplicate static GLB for object {object_id}")
            glb_paths[object_id] = glb_path
    return glb_paths


def resolve_sources(
    requested_ids: list[str] | None,
    articulated_urdfs: dict[str, Path],
    static_glbs: dict[str, Path],
) -> list[ObjectSource]:
    static_only = {object_id: path for object_id, path in static_glbs.items() if object_id not in articulated_urdfs}
    if requested_ids is None:
        object_ids = sorted(set(articulated_urdfs) | set(static_only))
    else:
        object_ids = requested_ids

    sources: list[ObjectSource] = []
    for object_id in object_ids:
        if object_id in articulated_urdfs:
            sources.append(
                ObjectSource(
                    object_id=object_id,
                    kind="articulated",
                    urdf_path=articulated_urdfs[object_id],
                )
            )
            continue
        if object_id in static_only:
            sources.append(
                ObjectSource(
                    object_id=object_id,
                    kind="static",
                    glb_path=static_only[object_id],
                )
            )
            continue
        raise FileNotFoundError(
            f"Object {object_id} was not found as an articulated URDF or static GLB without URDF"
        )
    return sources


def _load_mesh_from_glb(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"GLB contains no geometry: {glb_path}")
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported trimesh payload for {glb_path}: {type(mesh).__name__}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Mesh is empty: {glb_path}")
    return mesh


# Rotation matrix to convert from URDF Z-up to GLB Y-up convention.
# URDF meshes in hssd-hab/urdf/ are Z-up (height along Z).
# Blender's OBJ importer and our pipeline expect Y-up.
# R(-90° around X): (x,y,z) -> (x, z, -y). Z-up height Z becomes Y-up height Y.
Z_UP_TO_Y_UP_R3 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _rotate_mesh_zup_to_yup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    rotated = mesh.copy()
    rotated.vertices = (Z_UP_TO_Y_UP_R3 @ rotated.vertices.T).T
    # Preserve face winding since rotation is right-handed.
    return rotated


def _rotate_vec3_zup_to_yup(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"expected 3-vector, got shape {arr.shape}")
    return (Z_UP_TO_Y_UP_R3 @ arr).tolist()


def _rewrite_obj_mtllib(obj_text: str, target_name: str) -> str:
    lines = obj_text.splitlines()
    rewritten = False
    for index, line in enumerate(lines):
        if line.startswith("mtllib "):
            lines[index] = f"mtllib {target_name}"
            rewritten = True
    if not rewritten:
        lines.insert(0, f"mtllib {target_name}")
    return "\n".join(lines) + "\n"


def export_mesh_bundle(
    mesh: trimesh.Trimesh,
    obj_stem: str,
    objs_dir: Path,
    images_dir: Path,
    texture_prefix: str,
) -> None:
    objs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"convert_hssd_{obj_stem}_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        temp_obj_path = temp_dir / f"{obj_stem}.obj"
        mesh.export(temp_obj_path)

        exported_paths = sorted(temp_dir.iterdir())
        mtl_paths = [path for path in exported_paths if path.suffix.lower() == ".mtl"]
        if len(mtl_paths) > 1:
            raise ValueError(f"Export produced multiple MTL files for {obj_stem}: {mtl_paths}")
        auxiliary_paths = [
            path for path in exported_paths if path != temp_obj_path and path not in mtl_paths
        ]

        obj_text = temp_obj_path.read_text(encoding="utf-8")
        if mtl_paths:
            images_dir.mkdir(parents=True, exist_ok=True)
            source_mtl_path = mtl_paths[0]
            target_mtl_name = f"{obj_stem}.mtl"
            obj_text = _rewrite_obj_mtllib(obj_text, target_mtl_name)

            mtl_text = source_mtl_path.read_text(encoding="utf-8")
            for asset_idx, asset_path in enumerate(auxiliary_paths):
                target_asset_name = f"{texture_prefix}_{asset_idx}{asset_path.suffix.lower()}"
                target_asset_path = images_dir / target_asset_name
                if target_asset_path.exists():
                    raise FileExistsError(f"Texture target already exists: {target_asset_path}")
                shutil.move(str(asset_path), str(target_asset_path))
                mtl_text = mtl_text.replace(asset_path.name, f"../images/{target_asset_name}")

            (objs_dir / target_mtl_name).write_text(mtl_text, encoding="utf-8")
        elif auxiliary_paths:
            raise ValueError(
                f"Export produced auxiliary files without MTL for {obj_stem}: {auxiliary_paths}"
            )

        (objs_dir / f"{obj_stem}.obj").write_text(obj_text, encoding="utf-8")


def _require_attr(element: ET.Element, key: str, description: str) -> str:
    if key not in element.attrib:
        raise ValueError(f"Missing {description} attribute '{key}'")
    return element.attrib[key]


def _expect_single_child(parent: ET.Element, tag: str, description: str) -> ET.Element:
    matches = parent.findall(tag)
    if len(matches) != 1:
        raise ValueError(f"{description} must contain exactly one <{tag}> element, got {len(matches)}")
    return matches[0]


def _parse_link_meshes(urdf_path: Path, robot: ET.Element) -> list[tuple[str, list[Path]]]:
    link_meshes: list[tuple[str, list[Path]]] = []
    for link_element in robot.findall("link"):
        link_name = _require_attr(link_element, "name", "link")
        if link_name == "root":
            if link_element.findall("visual"):
                raise ValueError(f"root link must not contain visuals: {urdf_path}")
            continue

        visual_elements = link_element.findall("visual")
        if not visual_elements:
            raise ValueError(f"link {link_name} must contain at least one <visual> element in {urdf_path}")

        mesh_paths: list[Path] = []
        for vis_idx, visual_element in enumerate(visual_elements):
            origin_element = visual_element.find("origin")
            if origin_element is not None:
                xyz = _parse_float_list(origin_element.attrib.get("xyz", "0 0 0"), 3, f"link {link_name} visual[{vis_idx}] xyz")
                rpy = _parse_float_list(origin_element.attrib.get("rpy", "0 0 0"), 3, f"link {link_name} visual[{vis_idx}] rpy")
                if not _is_zero_vector(xyz) or not _is_zero_vector(rpy):
                    raise ValueError(f"link {link_name} visual[{vis_idx}] origin must be zero in {urdf_path}")

            geometry_element = _expect_single_child(visual_element, "geometry", f"link {link_name} visual[{vis_idx}]")
            mesh_element = _expect_single_child(geometry_element, "mesh", f"link {link_name} visual[{vis_idx}] geometry")
            mesh_filename = _require_attr(mesh_element, "filename", f"link {link_name} mesh[{vis_idx}]")
            mesh_scale = _parse_float_list(
                mesh_element.attrib.get("scale", "1 1 1"),
                3,
                f"link {link_name} mesh[{vis_idx}] scale",
            )
            if not _is_unit_scale(mesh_scale):
                raise ValueError(f"link {link_name} mesh[{vis_idx}] scale must be 1, got {mesh_scale}")

            mesh_path = urdf_path.parent / mesh_filename
            _require_file(mesh_path, f"link mesh for {link_name}")
            mesh_paths.append(mesh_path)

        link_meshes.append((link_name, mesh_paths))
    if not link_meshes:
        raise ValueError(f"No non-root links found in {urdf_path}")
    return link_meshes


def _validate_root_rotation(joint_element: ET.Element, urdf_path: Path) -> None:
    if joint_element.attrib.get("type") != "fixed":
        raise ValueError(f"root_rotation must be a fixed joint in {urdf_path}")
    parent_element = _expect_single_child(joint_element, "parent", "root_rotation")
    child_element = _expect_single_child(joint_element, "child", "root_rotation")
    if _require_attr(parent_element, "link", "root_rotation parent") != "root":
        raise ValueError(f"root_rotation parent must be root in {urdf_path}")
    _require_attr(child_element, "link", "root_rotation child")


def _joint_type_and_params(
    joint_element: ET.Element,
    joint_name: str,
    urdf_path: Path,
) -> tuple[str, list[float]]:
    joint_type = _require_attr(joint_element, "type", f"joint {joint_name}")
    origin_element = _expect_single_child(joint_element, "origin", f"joint {joint_name}")
    origin_xyz = _parse_float_list(
        origin_element.attrib.get("xyz", "0 0 0"),
        3,
        f"joint {joint_name} origin xyz",
    )
    origin_rpy = _parse_float_list(
        origin_element.attrib.get("rpy", "0 0 0"),
        3,
        f"joint {joint_name} origin rpy",
    )
    if not _is_zero_vector(origin_rpy):
        raise ValueError(f"joint {joint_name} has non-zero origin rpy in {urdf_path}")

    if joint_type == "fixed":
        return "E", []

    axis_element = _expect_single_child(joint_element, "axis", f"joint {joint_name}")
    axis = _parse_float_list(
        _require_attr(axis_element, "xyz", f"joint {joint_name} axis"),
        3,
        f"joint {joint_name} axis xyz",
    )

    # URDF axis and origin are in Z-up convention; rotate to Y-up for pipeline.
    axis_yup = _rotate_vec3_zup_to_yup(axis)
    origin_xyz_yup = _rotate_vec3_zup_to_yup(origin_xyz)

    if joint_type == "continuous":
        return "A", axis_yup + origin_xyz_yup + [0.0, 1.0]

    limit_element = _expect_single_child(joint_element, "limit", f"joint {joint_name}")
    lower = float(_require_attr(limit_element, "lower", f"joint {joint_name} limit"))
    upper = float(_require_attr(limit_element, "upper", f"joint {joint_name} limit"))

    if joint_type == "revolute":
        return "C", axis_yup + origin_xyz_yup + [lower / math.pi, upper / math.pi]
    if joint_type == "prismatic":
        return "B", axis_yup + [0.0, 0.0, 0.0] + [lower, upper]

    raise ValueError(f"Unsupported joint type {joint_type!r} in {urdf_path}")


def _simplify_part_name(link_name: str, base_link_name: str | None) -> str:
    if base_link_name is not None:
        if link_name == base_link_name:
            simplified = re.sub(r"\d+$", "", link_name)
            return simplified.lower() or link_name.lower()
        prefix = f"{base_link_name}_"
        if link_name.startswith(prefix):
            return link_name[len(prefix) :].lower()
    return link_name.lower()


def convert_articulated_object(
    source: ObjectSource,
    metadata_by_id: dict[str, dict[str, Any]],
    semantics_by_id: dict[str, dict[str, str]],
    finaljson_dir: Path,
    partseg_root: Path,
) -> int:
    assert source.urdf_path is not None
    semantics_row = semantics_by_id.get(source.object_id, {})

    tree = ET.parse(source.urdf_path)
    robot = tree.getroot()
    link_meshes = _parse_link_meshes(source.urdf_path, robot)
    part_labels = {link_name: index for index, (link_name, _mesh_path) in enumerate(link_meshes)}

    raw_joints: list[dict[str, Any]] = []
    child_links: set[str] = set()
    for joint_element in robot.findall("joint"):
        joint_name = _require_attr(joint_element, "name", "joint")
        if joint_name == "root_rotation":
            _validate_root_rotation(joint_element, source.urdf_path)
            continue

        parent_element = _expect_single_child(joint_element, "parent", f"joint {joint_name}")
        child_element = _expect_single_child(joint_element, "child", f"joint {joint_name}")
        parent_link = _require_attr(parent_element, "link", f"joint {joint_name} parent")
        child_link = _require_attr(child_element, "link", f"joint {joint_name} child")
        if parent_link == "root":
            raise ValueError(f"Unexpected non-root_rotation joint attached to root: {source.urdf_path}")
        if parent_link not in part_labels:
            raise KeyError(f"joint {joint_name} references missing parent link {parent_link}")
        if child_link not in part_labels:
            raise KeyError(f"joint {joint_name} references missing child link {child_link}")
        if child_link in child_links:
            raise ValueError(f"Multiple joints target child link {child_link} in {source.urdf_path}")

        group_type, params = _joint_type_and_params(joint_element, joint_name, source.urdf_path)
        origin_element = _expect_single_child(joint_element, "origin", f"joint {joint_name}")
        origin_xyz = _parse_float_list(
            origin_element.attrib.get("xyz", "0 0 0"),
            3,
            f"joint {joint_name} origin xyz",
        )
        # Convert origin_xyz (Z-up in URDF) to Y-up for rest_offset accumulation.
        raw_joints.append(
            {
                "joint_name": joint_name,
                "parent_link": parent_link,
                "child_link": child_link,
                "group_type": group_type,
                "params": params,
                "origin_xyz": _rotate_vec3_zup_to_yup(origin_xyz),
            }
        )
        child_links.add(child_link)

    base_links = [link_name for link_name, _mesh_path in link_meshes if link_name not in child_links]
    if not base_links:
        raise ValueError(f"No base links found in {source.urdf_path}")

    rest_offsets = {link_name: np.zeros(3, dtype=np.float64) for link_name in base_links}
    pending_offsets = list(raw_joints)
    while pending_offsets:
        progressed = False
        for joint in list(pending_offsets):
            parent_link = joint["parent_link"]
            if parent_link not in rest_offsets:
                continue
            rest_offsets[joint["child_link"]] = rest_offsets[parent_link] + np.asarray(
                joint["origin_xyz"],
                dtype=np.float64,
            )
            pending_offsets.remove(joint)
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(joint["joint_name"] for joint in pending_offsets))
            raise ValueError(f"Unable to resolve rest offsets for {source.object_id}: {unresolved}")

    object_partseg_dir = partseg_root / source.object_id
    if object_partseg_dir.exists():
        shutil.rmtree(object_partseg_dir)
    objs_dir = object_partseg_dir / "objs"
    images_dir = object_partseg_dir / "images"
    objs_dir.mkdir(parents=True, exist_ok=True)

    base_link_name = base_links[0] if len(base_links) == 1 else None
    bounds_list: list[np.ndarray] = []
    parts: list[dict[str, Any]] = []
    for part_index, (link_name, mesh_paths) in enumerate(link_meshes):
        obj_stems: list[str] = []
        for mesh_idx, mesh_path in enumerate(mesh_paths):
            mesh = _load_mesh_from_glb(mesh_path)
            # URDF link meshes are Z-up; rotate vertices to Y-up.
            mesh = _rotate_mesh_zup_to_yup(mesh)
            mesh.apply_translation(rest_offsets[link_name])
            bounds_list.append(mesh.bounds.copy())
            obj_stem = f"part_{part_index}" if len(mesh_paths) == 1 else f"part_{part_index}_{mesh_idx}"
            export_mesh_bundle(
                mesh=mesh,
                obj_stem=obj_stem,
                objs_dir=objs_dir,
                images_dir=images_dir,
                texture_prefix=f"texture_{obj_stem}",
            )
            obj_stems.append(obj_stem)
        parts.append(
            {
                "label": part_index,
                "name": _simplify_part_name(link_name, base_link_name),
                "obj": obj_stems,
            }
        )

    group_info: dict[str, Any] = {
        "0": [part_labels[link_name] for link_name in base_links],
    }
    group_id_by_link: dict[str, str] = {}
    pending_joints = list(raw_joints)
    next_group_id = 1
    while pending_joints:
        progressed = False
        for joint in list(pending_joints):
            parent_link = joint["parent_link"]
            if parent_link in base_links:
                parent_group = "0"
            elif parent_link in group_id_by_link:
                parent_group = group_id_by_link[parent_link]
            else:
                continue

            group_id = str(next_group_id)
            next_group_id += 1
            child_label = part_labels[joint["child_link"]]
            group_info[group_id] = [[child_label], parent_group, joint["params"], joint["group_type"]]
            group_id_by_link[joint["child_link"]] = group_id
            pending_joints.remove(joint)
            progressed = True

        if not progressed:
            unresolved = ", ".join(sorted(joint["joint_name"] for joint in pending_joints))
            raise ValueError(f"Unable to resolve articulated hierarchy for {source.object_id}: {unresolved}")

    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()

    finaljson = {
        "object_name": _resolve_object_name(source.object_id, metadata_by_id, semantics_row),
        "category": _resolve_category(semantics_row, source.kind),
        "dimension": _format_dimension(_combine_bounds(bounds_list)),
        "parts": parts,
        "group_info": group_info,
    }
    finaljson_path = finaljson_dir / f"{source.object_id}.json"
    finaljson_path.write_text(json.dumps(finaljson, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(parts)


def convert_static_object(
    source: ObjectSource,
    metadata_by_id: dict[str, dict[str, Any]],
    semantics_by_id: dict[str, dict[str, str]],
    finaljson_dir: Path,
    partseg_root: Path,
) -> int:
    assert source.glb_path is not None
    semantics_row = semantics_by_id.get(source.object_id, {})

    mesh = _load_mesh_from_glb(source.glb_path)
    object_partseg_dir = partseg_root / source.object_id
    if object_partseg_dir.exists():
        shutil.rmtree(object_partseg_dir)
    objs_dir = object_partseg_dir / "objs"
    images_dir = object_partseg_dir / "images"
    export_mesh_bundle(
        mesh=mesh,
        obj_stem="mesh",
        objs_dir=objs_dir,
        images_dir=images_dir,
        texture_prefix="texture_mesh",
    )
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()

    finaljson = {
        "object_name": _resolve_object_name(source.object_id, metadata_by_id, semantics_row),
        "category": _resolve_category(semantics_row, source.kind),
        "dimension": _format_dimension(mesh.bounds.copy()),
        "parts": [
            {
                "label": 0,
                "name": "body",
                "obj": ["mesh"],
            }
        ],
        "group_info": {"0": [0]},
    }
    finaljson_path = finaljson_dir / f"{source.object_id}.json"
    finaljson_path.write_text(json.dumps(finaljson, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hssd_hab_dir = _require_directory(Path(args.hssd_hab_dir), "HSSD _hssd_hab directory")
    hssd_objects_dir = _require_directory(Path(args.hssd_objects_dir), "HSSD objects directory")
    trellis_metadata = _require_file(Path(args.trellis_metadata), "TRELLIS metadata CSV")
    semantics_csv = _require_file(Path(args.semantics_csv), "semantics CSV")
    output_dir = Path(args.output_dir)

    metadata_by_id = load_metadata_index(trellis_metadata)
    semantics_by_id = load_semantics_index(semantics_csv)
    articulated_urdfs = discover_articulated_urdfs(hssd_hab_dir)
    static_glbs = discover_static_glbs(hssd_objects_dir)
    requested_ids = _parse_object_ids_arg(args.object_ids) if args.object_ids else None
    sources = resolve_sources(requested_ids, articulated_urdfs, static_glbs)

    finaljson_dir = output_dir / "raw" / "finaljson"
    partseg_root = output_dir / "raw" / "partseg"
    finaljson_dir.mkdir(parents=True, exist_ok=True)
    partseg_root.mkdir(parents=True, exist_ok=True)

    total = len(sources)
    for index, source in enumerate(sources, start=1):
        finaljson_path = finaljson_dir / f"{source.object_id}.json"
        if finaljson_path.is_file():
            print(f"[{index}/{total}] {source.object_id} skipped (existing)", flush=True)
            continue

        if source.kind == "articulated":
            try:
                part_count = convert_articulated_object(
                    source=source,
                    metadata_by_id=metadata_by_id,
                    semantics_by_id=semantics_by_id,
                    finaljson_dir=finaljson_dir,
                    partseg_root=partseg_root,
                )
            except FileNotFoundError as exc:
                print(f"  WARNING: {exc}, skipping (no mesh available)", flush=True)
                continue
        elif source.kind == "static":
            part_count = convert_static_object(
                source=source,
                metadata_by_id=metadata_by_id,
                semantics_by_id=semantics_by_id,
                finaljson_dir=finaljson_dir,
                partseg_root=partseg_root,
            )
        else:
            raise ValueError(f"Unsupported source kind: {source.kind}")

        print(
            f"[{index}/{total}] {source.object_id} done ({source.kind}, {part_count} parts)",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
