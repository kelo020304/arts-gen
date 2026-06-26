"""Local KinematicSolver workbench import helpers."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .mjcf_parser import generate_manifest


class WorkbenchError(ValueError):
    """Raised for user-fixable workbench input errors."""


def import_asset_to_workbench(payload: dict[str, Any], workbench_root: Path) -> dict[str, Any]:
    """Import USD/URDF/MJCF into the workbench preview bundle and saved XML/mesh roots."""
    source_path = _required_file(payload.get("source_path"), "source_path")
    object_id = _safe_asset_name(str(payload.get("object_id") or source_path.stem))
    xml_save_root = Path(payload.get("xml_save_root") or workbench_root / "generated_xml").expanduser()
    mesh_save_root = Path(payload.get("mesh_save_root") or workbench_root / "generated_mesh").expanduser()
    orientation_degrees = _orientation_degrees(payload.get("orientation_degrees"))

    preview_assets_root = Path(workbench_root) / "object_assets"
    preview_asset_dir = preview_assets_root / object_id
    preview_mjcf_dir = preview_asset_dir / "mjcf"
    preview_mesh_dir = preview_mjcf_dir / "assets"
    generated_xml_dir = xml_save_root / object_id / "mjcf"
    generated_mesh_dir = mesh_save_root / object_id / "mjcf" / "assets"

    _reset_dir(preview_asset_dir)
    _reset_dir(generated_xml_dir)
    _reset_dir(generated_mesh_dir)
    preview_mesh_dir.mkdir(parents=True, exist_ok=True)
    generated_xml_dir.mkdir(parents=True, exist_ok=True)
    generated_mesh_dir.mkdir(parents=True, exist_ok=True)

    suffix = source_path.suffix.lower()
    if suffix in {".xml", ".mjcf"}:
        preview_root, generated_root = _convert_mjcf(
            source_path,
            preview_mesh_dir=preview_mesh_dir,
            generated_mesh_dir=generated_mesh_dir,
        )
    elif suffix == ".urdf":
        preview_root, generated_root = _convert_urdf(
            source_path,
            object_id=object_id,
            preview_mesh_dir=preview_mesh_dir,
            generated_mesh_dir=generated_mesh_dir,
        )
    elif suffix in {".usd", ".usda", ".usdc"}:
        preview_root, generated_root = _convert_usd(
            source_path,
            object_id=object_id,
            preview_mesh_dir=preview_mesh_dir,
            generated_mesh_dir=generated_mesh_dir,
        )
    else:
        raise WorkbenchError(
            f"Unsupported source type: {source_path.suffix}. Expected .xml, .mjcf, .urdf, .usd, .usda, or .usdc."
        )

    _set_compiler_meshdir(preview_root, ".")
    _set_compiler_meshdir(generated_root, str(generated_mesh_dir.resolve()))
    preview_xml_path = preview_mjcf_dir / f"{object_id}.xml"
    generated_xml_path = generated_xml_dir / f"{object_id}.xml"
    _write_xml(preview_root, preview_xml_path)
    _write_xml(generated_root, generated_xml_path)

    orientation_path = preview_asset_dir / "workbench_orientation.json"
    orientation_path.write_text(
        json.dumps({"orientation_degrees": orientation_degrees}, indent=2),
        encoding="utf-8",
    )

    manifest = generate_manifest(object_id, preview_assets_root)
    if manifest.get("status") != "ok":
        raise WorkbenchError(str(manifest.get("message") or "Imported MJCF failed to compile"))

    return {
        "status": "ok",
        "asset_name": object_id,
        "source_type": suffix.lstrip("."),
        "viewer_url": f"/object-post-process/{object_id}",
        "preview_asset_dir": str(preview_asset_dir),
        "preview_xml_path": str(preview_xml_path),
        "generated_xml_path": str(generated_xml_path),
        "generated_mesh_dir": str(generated_mesh_dir),
        "orientation_path": str(orientation_path),
        "orientation_degrees": orientation_degrees,
        "manifest": manifest,
    }


def save_workbench_orientation(
    *,
    asset_name: str,
    orientation_degrees: dict[str, float],
    workbench_root: Path,
) -> dict[str, Any]:
    """Persist a workbench-level orientation for later preview reloads."""
    safe_asset = _safe_asset_name(asset_name)
    asset_dir = Path(workbench_root) / "object_assets" / safe_asset
    if not asset_dir.is_dir():
        raise WorkbenchError(f"Unknown workbench asset: {safe_asset}")
    orientation = _orientation_degrees(orientation_degrees)
    orientation_path = asset_dir / "workbench_orientation.json"
    orientation_path.write_text(
        json.dumps({"orientation_degrees": orientation}, indent=2),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "asset_name": safe_asset,
        "orientation_degrees": orientation,
        "orientation_path": str(orientation_path),
    }


def save_initial_joints_json(payload: dict[str, Any], workbench_root: Path) -> dict[str, Any]:
    """Validate and persist a manually edited VLM-style initial joints JSON."""
    object_id = _safe_asset_name(str(payload.get("object_id") or "object"))
    raw_json = payload.get("json_text")
    if raw_json is None:
        raw_json = payload.get("initial_joints")
    if isinstance(raw_json, str):
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise WorkbenchError(f"Initial joints input must be valid JSON: {exc.msg}") from exc
    elif isinstance(raw_json, (dict, list)):
        parsed = raw_json
    else:
        raise WorkbenchError("Initial joints input must be a JSON string or object")

    if not isinstance(parsed, dict):
        raise WorkbenchError("Initial joints JSON root must be an object")
    original_object_id = parsed.get("object_id")
    object_id_normalized = original_object_id is not None and str(original_object_id) != object_id
    parsed["object_id"] = object_id

    target_path = _resolve_initial_joints_target_path(
        payload.get("target_path"),
        object_id=object_id,
        workbench_root=workbench_root,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_json_text = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
    target_path.write_text(
        normalized_json_text,
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "object_id": object_id,
        "initial_joints_json": str(target_path),
        "object_id_normalized": object_id_normalized,
        "normalized_json_text": normalized_json_text,
    }


def _resolve_initial_joints_target_path(
    raw_target_path: Any,
    *,
    object_id: str,
    workbench_root: Path,
) -> Path:
    raw_text = str(raw_target_path or "").strip()
    if not raw_text:
        return Path(workbench_root) / "initial_joints" / object_id / "vlm_initial.json"

    target = Path(raw_text).expanduser()
    if target.suffix.lower() == ".json":
        return target
    return target / object_id / "vlm_initial.json"


def _convert_mjcf(
    source_path: Path,
    *,
    preview_mesh_dir: Path,
    generated_mesh_dir: Path,
) -> tuple[ET.Element, ET.Element]:
    preview_root = ET.parse(source_path).getroot()
    generated_root = ET.parse(source_path).getroot()
    preview_meshes = preview_root.findall("./asset/mesh")
    generated_meshes = generated_root.findall("./asset/mesh")
    used_names: set[str] = set()
    for preview_mesh, generated_mesh in zip(preview_meshes, generated_meshes, strict=True):
        mesh_file = preview_mesh.get("file")
        if not mesh_file:
            continue
        source_mesh = _resolve_mjcf_mesh_path(source_path, preview_root, mesh_file)
        target_name = _unique_filename(source_mesh.name, used_names)
        _copy_mesh(source_mesh, preview_mesh_dir / target_name)
        _copy_mesh(source_mesh, generated_mesh_dir / target_name)
        preview_mesh.set("file", f"assets/{target_name}")
        generated_mesh.set("file", target_name)
    return preview_root, generated_root


def _convert_urdf(
    source_path: Path,
    *,
    object_id: str,
    preview_mesh_dir: Path,
    generated_mesh_dir: Path,
) -> tuple[ET.Element, ET.Element]:
    robot = ET.parse(source_path).getroot()
    links = {
        _required_attr(link, "name", "link"): link
        for link in _children(robot, "link")
    }
    joints = list(_children(robot, "joint"))
    used_names: set[str] = set()
    link_meshes: dict[str, str] = {}
    for link_name, link in links.items():
        mesh_element = _first_descendant(link, "mesh")
        if mesh_element is None:
            continue
        raw_filename = mesh_element.get("filename") or mesh_element.get("file")
        if not raw_filename:
            continue
        source_mesh = _resolve_urdf_mesh_path(source_path, raw_filename)
        target_name = _unique_filename(source_mesh.name, used_names)
        _copy_mesh(source_mesh, preview_mesh_dir / target_name)
        _copy_mesh(source_mesh, generated_mesh_dir / target_name)
        link_meshes[link_name] = target_name

    return (
        _build_urdf_mjcf(object_id, links, joints, link_meshes, preview=True),
        _build_urdf_mjcf(object_id, links, joints, link_meshes, preview=False),
    )


def _convert_usd(
    source_path: Path,
    *,
    object_id: str,
    preview_mesh_dir: Path,
    generated_mesh_dir: Path,
) -> tuple[ET.Element, ET.Element]:
    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:  # pragma: no cover - environment-specific import error
        raise WorkbenchError(f"USD import requires pxr: {exc}") from exc

    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        raise WorkbenchError(f"Failed to open USD stage: {source_path}")
    cache = UsdGeom.XformCache()
    mesh_entries: list[tuple[str, str]] = []
    used_filenames: set[str] = set()
    used_body_names: set[str] = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        face_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if not points or not face_counts or not face_indices:
            continue
        body_name = _unique_body_name(_safe_asset_name(prim.GetName()), used_body_names)
        target_name = _unique_filename(body_name + ".obj", used_filenames)
        world_xform = cache.GetLocalToWorldTransform(prim)
        obj_text = _usd_mesh_to_obj(points, face_counts, face_indices, world_xform)
        (preview_mesh_dir / target_name).write_text(obj_text, encoding="utf-8")
        (generated_mesh_dir / target_name).write_text(obj_text, encoding="utf-8")
        mesh_entries.append((body_name, target_name))
    if not mesh_entries:
        raise WorkbenchError(f"No mesh prims found in USD stage: {source_path}")
    return (
        _build_static_mesh_mjcf(object_id, mesh_entries, preview=True),
        _build_static_mesh_mjcf(object_id, mesh_entries, preview=False),
    )


def _build_static_mesh_mjcf(object_id: str, mesh_entries: list[tuple[str, str]], *, preview: bool) -> ET.Element:
    root = ET.Element("mujoco", {"model": object_id})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})
    asset = ET.SubElement(root, "asset")
    for body_name, mesh_name in mesh_entries:
        file_attr = f"assets/{mesh_name}" if preview else mesh_name
        ET.SubElement(asset, "mesh", {"name": f"{body_name}_mesh", "file": file_attr})
    worldbody = ET.SubElement(root, "worldbody")
    used_body_names = {body_name for body_name, _mesh_name in mesh_entries}
    root_body_name = _unique_body_name("root", used_body_names)
    root_body = ET.SubElement(worldbody, "body", {"name": root_body_name, "pos": "0 0 0"})
    for body_name, _mesh_name in mesh_entries:
        body = ET.SubElement(root_body, "body", {"name": body_name, "pos": "0 0 0"})
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{body_name}_visual",
                "type": "mesh",
                "mesh": f"{body_name}_mesh",
                "group": "2",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    return root


def _build_urdf_mjcf(
    object_id: str,
    links: dict[str, ET.Element],
    joints: list[ET.Element],
    link_meshes: dict[str, str],
    *,
    preview: bool,
) -> ET.Element:
    root = ET.Element("mujoco", {"model": object_id})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})
    asset = ET.SubElement(root, "asset")
    for link_name, mesh_name in sorted(link_meshes.items()):
        file_attr = f"assets/{mesh_name}" if preview else mesh_name
        ET.SubElement(asset, "mesh", {"name": f"{_safe_asset_name(link_name)}_mesh", "file": file_attr})
    worldbody = ET.SubElement(root, "worldbody")
    child_to_joint = {
        _required_child_attr(joint, "child", "link", "joint child"): joint
        for joint in joints
    }
    parent_to_children: dict[str, list[str]] = {}
    for child_name, joint in child_to_joint.items():
        parent_name = _required_child_attr(joint, "parent", "link", "joint parent")
        parent_to_children.setdefault(parent_name, []).append(child_name)
    root_links = sorted(set(links) - set(child_to_joint))
    if not root_links and links:
        root_links = [sorted(links)[0]]

    def add_body(parent_xml: ET.Element, link_name: str) -> None:
        joint = child_to_joint.get(link_name)
        body_attrs = {"name": _safe_asset_name(link_name)}
        if joint is not None:
            origin = _first_child(joint, "origin")
            if origin is not None:
                xyz = _float_list(origin.get("xyz"), 3, [0.0, 0.0, 0.0])
                rpy = _float_list(origin.get("rpy"), 3, [0.0, 0.0, 0.0])
                body_attrs["pos"] = _fmt_floats(xyz)
                body_attrs["euler"] = _fmt_floats(rpy)
        body = ET.SubElement(parent_xml, "body", body_attrs)
        if joint is not None:
            joint_type = joint.get("type", "fixed")
            if joint_type in {"revolute", "continuous", "prismatic"}:
                axis_el = _first_child(joint, "axis")
                axis = _float_list(axis_el.get("xyz") if axis_el is not None else None, 3, [1.0, 0.0, 0.0])
                mj_type = "slide" if joint_type == "prismatic" else "hinge"
                joint_attrs = {
                    "name": _safe_asset_name(joint.get("name") or f"{link_name}_joint"),
                    "type": mj_type,
                    "axis": _fmt_floats(axis),
                }
                limit_el = _first_child(joint, "limit")
                if limit_el is not None and limit_el.get("lower") is not None and limit_el.get("upper") is not None:
                    joint_attrs["range"] = f"{float(limit_el.get('lower')):.9g} {float(limit_el.get('upper')):.9g}"
                ET.SubElement(body, "joint", joint_attrs)
        mesh_name = link_meshes.get(link_name)
        if mesh_name is not None:
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{_safe_asset_name(link_name)}_visual",
                    "type": "mesh",
                    "mesh": f"{_safe_asset_name(link_name)}_mesh",
                    "group": "2",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )
        for child in sorted(parent_to_children.get(link_name, [])):
            add_body(body, child)

    for link_name in root_links:
        add_body(worldbody, link_name)
    return root


def _usd_mesh_to_obj(points: Any, face_counts: Any, face_indices: Any, world_xform: Any) -> str:
    lines: list[str] = []
    for point in points:
        transformed = world_xform.Transform(point)
        lines.append(f"v {float(transformed[0]):.9g} {float(transformed[1]):.9g} {float(transformed[2]):.9g}")
    cursor = 0
    for count in face_counts:
        count_int = int(count)
        indices = [int(index) + 1 for index in face_indices[cursor : cursor + count_int]]
        cursor += count_int
        if len(indices) >= 3:
            lines.append("f " + " ".join(str(index) for index in indices))
    return "\n".join(lines) + "\n"


def _resolve_mjcf_mesh_path(source_path: Path, root: ET.Element, mesh_file: str) -> Path:
    raw = Path(mesh_file)
    if raw.is_absolute():
        return _required_file(raw, f"mesh file {mesh_file}")
    compiler = root.find("compiler")
    meshdir_raw = compiler.get("meshdir") if compiler is not None else None
    base = source_path.parent
    if meshdir_raw:
        meshdir = Path(meshdir_raw)
        base = meshdir if meshdir.is_absolute() else source_path.parent / meshdir
    return _required_file(base / mesh_file, f"mesh file {mesh_file}")


def _resolve_urdf_mesh_path(source_path: Path, raw_filename: str) -> Path:
    cleaned = raw_filename
    if cleaned.startswith("file://"):
        cleaned = cleaned[len("file://") :]
    if cleaned.startswith("package://"):
        cleaned = cleaned[len("package://") :]
    candidate = Path(cleaned)
    if candidate.is_absolute():
        return _required_file(candidate, f"mesh file {raw_filename}")
    return _required_file(source_path.parent / candidate, f"mesh file {raw_filename}")


def _set_compiler_meshdir(root: ET.Element, meshdir: str) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", meshdir)


def _write_xml(root: ET.Element, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(root, space="  ")
    except AttributeError:  # pragma: no cover - Python <3.9 fallback
        pass
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _copy_mesh(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _required_file(value: Any, label: str) -> Path:
    if value is None or str(value).strip() == "":
        raise WorkbenchError(f"Missing required path: {label}")
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise WorkbenchError(f"File not found for {label}: {path}")
    return path.resolve()


def _safe_asset_name(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    if not safe:
        raise WorkbenchError("object_id must contain at least one letter, number, or underscore")
    return safe


def _unique_filename(raw_name: str, used_names: set[str]) -> str:
    base = _safe_asset_name(Path(raw_name).stem)
    suffix = Path(raw_name).suffix or ".obj"
    candidate = base + suffix.lower()
    index = 1
    while candidate in used_names:
        candidate = f"{base}_{index}{suffix.lower()}"
        index += 1
    used_names.add(candidate)
    return candidate


def _unique_body_name(raw_name: str, used_names: set[str]) -> str:
    base = _safe_asset_name(raw_name)
    candidate = base
    index = 1
    while candidate in used_names:
        candidate = f"{base}_{index}"
        index += 1
    used_names.add(candidate)
    return candidate


def _orientation_degrees(raw: Any) -> dict[str, float]:
    raw = raw or {}
    return {
        "roll": float(raw.get("roll", 0.0)),
        "pitch": float(raw.get("pitch", 0.0)),
        "yaw": float(raw.get("yaw", 0.0)),
    }


def _children(element: ET.Element, tag_name: str) -> list[ET.Element]:
    return [child for child in list(element) if _strip_ns(child.tag) == tag_name]


def _first_child(element: ET.Element, tag_name: str) -> ET.Element | None:
    for child in list(element):
        if _strip_ns(child.tag) == tag_name:
            return child
    return None


def _first_descendant(element: ET.Element, tag_name: str) -> ET.Element | None:
    for child in element.iter():
        if _strip_ns(child.tag) == tag_name:
            return child
    return None


def _required_attr(element: ET.Element, attr_name: str, label: str) -> str:
    value = element.get(attr_name)
    if value is None or value == "":
        raise WorkbenchError(f"Missing {attr_name} on {label}")
    return value


def _required_child_attr(element: ET.Element, child_name: str, attr_name: str, label: str) -> str:
    child = _first_child(element, child_name)
    if child is None:
        raise WorkbenchError(f"Missing <{child_name}> on {label}")
    return _required_attr(child, attr_name, label)


def _strip_ns(tag_name: str) -> str:
    return tag_name.rsplit("}", 1)[-1]


def _float_list(raw: str | None, expected: int, default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    values = [float(item) for item in raw.split()]
    if len(values) != expected:
        raise WorkbenchError(f"Expected {expected} floats, got {len(values)}: {raw}")
    if any(not math.isfinite(value) for value in values):
        raise WorkbenchError(f"Non-finite float value: {raw}")
    return values


def _fmt_floats(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)
