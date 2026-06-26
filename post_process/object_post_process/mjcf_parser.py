"""MJCF XML parser: compile via MuJoCo and extract manifest for the web editor."""

from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


def generate_manifest(asset_name: str, assets_root: Path) -> dict[str, Any]:
    """Compile MJCF XML for *asset_name* and return a manifest dict."""
    asset_dir = assets_root / asset_name
    try:
        xml_path = _locate_asset_xml(asset_dir)
    except ValueError as exc:
        return _error_result(str(exc))

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:
        return _error_result(str(exc))

    try:
        return _extract_manifest(model, asset_name, xml_path, asset_dir)
    except Exception as exc:
        return _error_result(str(exc))


def generate_manifest_from_xml_bytes(
    xml_bytes: bytes,
    asset_name: str,
    assets_root: Path,
) -> dict[str, Any]:
    """Compile arbitrary XML bytes using *asset_name*'s mesh root."""
    asset_dir = assets_root / asset_name
    mesh_parent = asset_dir / "mjcf"
    if not mesh_parent.is_dir():
        mesh_parent = asset_dir / "xml"
    if not mesh_parent.is_dir():
        return _error_result(f"No mjcf/ or xml/ directory in {asset_dir}")

    tmp_path: Path | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".xml", dir=str(mesh_parent))
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        tmp_path.write_bytes(xml_bytes)

        model = mujoco.MjModel.from_xml_path(str(tmp_path))
        return _extract_manifest(model, asset_name, tmp_path, asset_dir)
    except Exception as exc:
        return _error_result(str(exc))
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _extract_manifest(
    model: mujoco.MjModel,
    asset_name: str,
    xml_path: Path,
    asset_dir: Path,
) -> dict[str, Any]:
    """Build the manifest dict from a compiled MjModel plus the XML source."""
    xml_root = _load_xml_root(xml_path)
    worldbody = xml_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in {xml_path}")

    body_elements = _collect_body_elements(worldbody)
    mesh_name_to_file = _collect_mesh_assets(xml_root)
    pose_config = _read_pose_config(xml_root)
    geom_elements_by_body = {
        body_name: [geom for geom in body_element.findall("geom") if geom.get("mesh")]
        for body_name, body_element in body_elements.items()
    }

    qpos_defaults = model.key_qpos[0].tolist() if model.nkey > 0 else model.qpos0.tolist()

    joints: list[dict[str, Any]] = []
    joint_ids_by_body: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name:
            raise ValueError(f"Joint id {i} is unnamed")

        joint_type = int(model.jnt_type[i])
        if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            joint_type_name = "hinge"
        elif joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            joint_type_name = "slide"
        else:
            raise ValueError(f"Unsupported joint type for Phase 3 preview: joint={name}, type={joint_type}")

        qpos_adr = int(model.jnt_qposadr[i])
        if qpos_adr < 0 or qpos_adr >= len(qpos_defaults):
            raise ValueError(
                f"Joint qpos address out of range for Phase 3 preview: joint={name}, qpos_adr={qpos_adr}, nq={len(qpos_defaults)}"
            )

        body_id = int(model.jnt_bodyid[i])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            raise ValueError(f"Joint {name} is attached to unnamed body id {body_id}")

        joint_ids_by_body[body_name].append((qpos_adr, name))
        joints.append({
            "id": name,
            "name": name,
            "type": joint_type_name,
            "range": model.jnt_range[i].tolist(),
            "default": float(qpos_defaults[qpos_adr]),
            "qpos_adr": qpos_adr,
            "axis": model.jnt_axis[i].tolist(),
            "anchor": model.jnt_pos[i].tolist(),
            "body": body_name,
        })

    joints.sort(key=lambda item: item["qpos_adr"])

    actuators: list[dict[str, Any]] = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            actuators.append({
                "name": name,
                "ctrlrange": model.actuator_ctrlrange[i].tolist(),
            })

    geoms_by_body: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_geom_ids_by_body: dict[str, list[int]] = defaultdict(list)
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        if body_name is None:
            raise ValueError(f"Geom id {geom_id} is attached to unnamed body")
        model_geom_ids_by_body[body_name].append(geom_id)

    for body_name, geom_ids in model_geom_ids_by_body.items():
        xml_geom_elements = geom_elements_by_body.get(body_name, [])
        if len(xml_geom_elements) != len(geom_ids):
            raise ValueError(
                f"Geom count mismatch for body {body_name}: xml={len(xml_geom_elements)} compiled={len(geom_ids)}"
            )
        for geom_id, geom_element in zip(geom_ids, xml_geom_elements, strict=True):
            geom_data = _build_geom_manifest(model, geom_id, geom_element, mesh_name_to_file, pose_config)
            geoms_by_body[body_name].append(geom_data)

    bodies: list[dict[str, Any]] = []
    for i in range(model.nbody):
        if i == 0:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name is None:
            raise ValueError(f"Body id {i} is unnamed")

        parent_id = int(model.body_parentid[i])
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        if parent_name is None:
            raise ValueError(f"Body {body_name} has unnamed parent id {parent_id}")

        body_geoms = geoms_by_body.get(body_name, [])
        visual_geoms = [geom for geom in body_geoms if geom["kind"] == "visual"]
        collision_geoms = [geom for geom in body_geoms if geom["kind"] == "collision"]
        joint_ids = [joint_name for _, joint_name in sorted(joint_ids_by_body.get(body_name, []), key=lambda item: item[0])]

        bodies.append({
            "id": body_name,
            "name": body_name,
            "parent": parent_name,
            "pos": model.body_pos[i].tolist(),
            "quat": model.body_quat[i].tolist(),
            "joint_ids": joint_ids,
            "mesh_file": visual_geoms[0]["mesh_file"] if visual_geoms else None,
            "visual_geoms": [_without_kind(geom) for geom in visual_geoms],
            "collision_geoms": [_without_kind(geom) for geom in collision_geoms],
        })

    preview_image: str | None = None
    for subdir in ("mjcf", "xml"):
        candidate = asset_dir / subdir / "preview.jpg"
        if candidate.is_file():
            preview_image = f"{subdir}/preview.jpg"
            break

    rel_xml = str(xml_path.resolve().relative_to(asset_dir.resolve()))
    asset_base_url = _determine_asset_base_url(asset_name, asset_dir, xml_path)

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "xml_path": rel_xml,
        "asset_base_url": asset_base_url,
        "joints": joints,
        "actuators": actuators,
        "bodies": bodies,
        "preview_image": preview_image,
    }


def _build_geom_manifest(
    model: mujoco.MjModel,
    geom_id: int,
    geom_element: ET.Element,
    mesh_name_to_file: dict[str, str],
    pose_config: dict[str, str],
) -> dict[str, Any]:
    mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom_id]))
    if mesh_name is None:
        raise ValueError(f"Mesh geom id {geom_id} is missing a mesh asset name")
    mesh_file = mesh_name_to_file.get(mesh_name)
    if mesh_file is None:
        raise ValueError(f"Mesh asset {mesh_name} is missing a file attribute in <asset>")

    geom_name = _canonical_geom_name(geom_element, mesh_name_to_file)
    kind = _classify_geom_kind(model, geom_id, geom_element)
    geom_pos, geom_quat = _extract_local_geom_pose(geom_element, pose_config, geom_name)

    geom_record: dict[str, Any] = {
        "kind": kind,
        "name": geom_name,
        "mesh_file": mesh_file,
        "mesh_pos": model.mesh_pos[int(model.geom_dataid[geom_id])].tolist(),
        "mesh_quat": model.mesh_quat[int(model.geom_dataid[geom_id])].tolist(),
        "pos": geom_pos,
        "quat": geom_quat,
        "group": int(model.geom_group[geom_id]),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
    }

    material_name = None
    material_id = int(model.geom_matid[geom_id])
    if material_id >= 0:
        material_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
    if geom_element.get("material") is not None or material_name is not None:
        geom_record["material"] = material_name or geom_element.get("material")

    if geom_element.get("rgba") is not None:
        geom_record["rgba"] = model.geom_rgba[geom_id].tolist()

    return geom_record


def _classify_geom_kind(
    model: mujoco.MjModel,
    geom_id: int,
    geom_element: ET.Element,
) -> str:
    geom_class = geom_element.get("class")
    if geom_class == "visual":
        return "visual"
    if geom_class == "collision":
        return "collision"

    geom_group = int(model.geom_group[geom_id])
    if geom_group == 2:
        return "visual"
    if geom_group == 3:
        return "collision"

    if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
        return "visual"
    return "collision"


def _determine_asset_base_url(asset_name: str, asset_dir: Path, xml_path: Path) -> str:
    rel_xml_dir = str(xml_path.parent.resolve().relative_to(asset_dir.resolve()))
    candidate_assets_dir = xml_path.parent / "assets"
    if candidate_assets_dir.is_dir():
        return f"/assets/{asset_name}/{rel_xml_dir}/assets"

    for subdir in ("mjcf", "xml"):
        assets_dir = asset_dir / subdir / "assets"
        if assets_dir.is_dir():
            return f"/assets/{asset_name}/{subdir}/assets"

    raise ValueError(f"No assets directory found under {asset_dir / 'mjcf'} or {asset_dir / 'xml'}")


def _collect_body_elements(worldbody: ET.Element) -> dict[str, ET.Element]:
    body_elements: dict[str, ET.Element] = {}

    def walk(body_element: ET.Element) -> None:
        body_name = body_element.get("name")
        if not body_name:
            raise ValueError("Encountered unnamed <body> while parsing manifest")
        if body_name in body_elements:
            raise ValueError(f"Duplicate body name in XML: {body_name}")
        body_elements[body_name] = body_element
        for child_body in body_element.findall("body"):
            walk(child_body)

    for body_element in worldbody.findall("body"):
        walk(body_element)
    return body_elements


def _collect_mesh_assets(xml_root: ET.Element) -> dict[str, str]:
    mesh_name_to_file: dict[str, str] = {}
    asset_element = xml_root.find("asset")
    if asset_element is None:
        return mesh_name_to_file

    for mesh_element in asset_element.findall("mesh"):
        mesh_name = mesh_element.get("name")
        mesh_file = mesh_element.get("file")
        if not mesh_name or not mesh_file:
            continue
        mesh_name_to_file[mesh_name] = mesh_file
    return mesh_name_to_file


def _read_pose_config(xml_root: ET.Element) -> dict[str, str]:
    compiler_element = xml_root.find("compiler")
    angle_unit = "degree"
    euler_seq = "xyz"
    if compiler_element is None:
        return {"angle_unit": angle_unit, "euler_seq": euler_seq}

    angle_attr = compiler_element.get("angle")
    if angle_attr is not None:
        if angle_attr not in {"degree", "radian"}:
            raise ValueError(f"Unsupported compiler angle unit: {angle_attr}")
        angle_unit = angle_attr

    euler_seq_attr = compiler_element.get("eulerseq")
    if euler_seq_attr is not None:
        if len(euler_seq_attr) != 3:
            raise ValueError(f"Unsupported compiler eulerseq: {euler_seq_attr}")
        euler_seq = euler_seq_attr

    return {"angle_unit": angle_unit, "euler_seq": euler_seq}


def _extract_local_geom_pose(
    geom_element: ET.Element,
    pose_config: dict[str, str],
    geom_name: str,
) -> tuple[list[float], list[float]]:
    pos = _parse_float_attr(geom_element.get("pos"), 3, [0.0, 0.0, 0.0], f"Geom {geom_name} pos")
    quat = _extract_local_quaternion(geom_element, pose_config, geom_name)
    return pos, quat


def _extract_local_quaternion(
    geom_element: ET.Element,
    pose_config: dict[str, str],
    geom_name: str,
) -> list[float]:
    orientation_attrs = [
        attr_name
        for attr_name in ("quat", "euler", "axisangle", "xyaxes", "zaxis")
        if geom_element.get(attr_name) is not None
    ]
    if len(orientation_attrs) > 1:
        raise ValueError(f"Geom {geom_name} has multiple orientation attributes: {orientation_attrs}")

    angle_unit = pose_config["angle_unit"]
    if not orientation_attrs:
        return [1.0, 0.0, 0.0, 0.0]

    attr_name = orientation_attrs[0]
    if attr_name == "quat":
        quat = np.array(_parse_float_attr(geom_element.get("quat"), 4, None, f"Geom {geom_name} quat"), dtype=float)
        return _normalize_quaternion(quat).tolist()

    if attr_name == "euler":
        euler = np.array(_parse_float_attr(geom_element.get("euler"), 3, None, f"Geom {geom_name} euler"), dtype=float)
        if angle_unit == "degree":
            euler = np.deg2rad(euler)
        quat = np.zeros(4, dtype=float)
        mujoco.mju_euler2Quat(quat, euler, pose_config["euler_seq"])
        return _normalize_quaternion(quat).tolist()

    if attr_name == "axisangle":
        axis_angle = np.array(
            _parse_float_attr(geom_element.get("axisangle"), 4, None, f"Geom {geom_name} axisangle"),
            dtype=float,
        )
        axis = axis_angle[:3]
        angle = axis_angle[3]
        if angle_unit == "degree":
            angle = math.radians(angle)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            if abs(angle) <= 1e-12:
                return [1.0, 0.0, 0.0, 0.0]
            raise ValueError(f"Geom {geom_name} axisangle axis must be non-zero")
        quat = np.zeros(4, dtype=float)
        mujoco.mju_axisAngle2Quat(quat, axis / axis_norm, angle)
        return _normalize_quaternion(quat).tolist()

    if attr_name == "xyaxes":
        xyaxes = np.array(_parse_float_attr(geom_element.get("xyaxes"), 6, None, f"Geom {geom_name} xyaxes"), dtype=float)
        return _quaternion_from_xyaxes(xyaxes[:3], xyaxes[3:], geom_name)

    zaxis = np.array(_parse_float_attr(geom_element.get("zaxis"), 3, None, f"Geom {geom_name} zaxis"), dtype=float)
    return _quaternion_from_zaxis(zaxis, geom_name)


def _parse_float_attr(
    raw_value: str | None,
    expected_length: int,
    default: list[float] | None,
    label: str,
) -> list[float]:
    if raw_value is None:
        if default is None:
            raise ValueError(f"Missing {label}")
        return list(default)

    parts = raw_value.split()
    if len(parts) != expected_length:
        raise ValueError(f"{label} must contain exactly {expected_length} floats")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{label} contains a non-float value") from exc


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(quat))
    if length <= 1e-12:
        raise ValueError("Quaternion length must be > 0")
    return quat / length


def _quaternion_from_xyaxes(x_axis: np.ndarray, y_axis: np.ndarray, geom_name: str) -> list[float]:
    x_axis = _normalize_vector(x_axis, f"Geom {geom_name} xyaxes x-axis")
    y_axis = y_axis - x_axis * float(np.dot(x_axis, y_axis))
    y_axis = _normalize_vector(y_axis, f"Geom {geom_name} xyaxes y-axis")
    z_axis = _normalize_vector(np.cross(x_axis, y_axis), f"Geom {geom_name} xyaxes z-axis")
    y_axis = _normalize_vector(np.cross(z_axis, x_axis), f"Geom {geom_name} xyaxes y-axis")
    return _quaternion_from_rotation_matrix(np.column_stack((x_axis, y_axis, z_axis)))


def _quaternion_from_zaxis(z_axis: np.ndarray, geom_name: str) -> list[float]:
    z_axis = _normalize_vector(z_axis, f"Geom {geom_name} zaxis")
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(z_axis, reference))) > 0.999:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = _normalize_vector(np.cross(reference, z_axis), f"Geom {geom_name} zaxis x-axis")
    y_axis = _normalize_vector(np.cross(z_axis, x_axis), f"Geom {geom_name} zaxis y-axis")
    return _quaternion_from_rotation_matrix(np.column_stack((x_axis, y_axis, z_axis)))


def _normalize_vector(vector: np.ndarray, label: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{label} must be non-zero")
    return vector / length


def _quaternion_from_rotation_matrix(matrix: np.ndarray) -> list[float]:
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ], dtype=float)
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        ], dtype=float)
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = np.array([
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        ], dtype=float)
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = np.array([
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        ], dtype=float)
    return _normalize_quaternion(quat).tolist()


def _canonical_geom_name(geom_element: ET.Element, mesh_name_to_file: dict[str, str]) -> str:
    explicit_name = geom_element.get("name")
    if explicit_name:
        return explicit_name

    mesh_name = geom_element.get("mesh")
    if mesh_name:
        mesh_file = mesh_name_to_file.get(mesh_name)
        if mesh_file:
            return Path(mesh_file).stem
        return mesh_name

    raise ValueError("Encountered <geom> without name or mesh while parsing manifest")


def _load_xml_root(xml_path: Path) -> ET.Element:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(xml_path, parser=parser).getroot()


def _locate_asset_xml(asset_dir: Path) -> Path:
    if not asset_dir.is_dir():
        raise ValueError(f"Asset directory does not exist: {asset_dir}")

    for subdir in ("mjcf", "xml"):
        xml_dir = asset_dir / subdir
        if not xml_dir.is_dir():
            continue
        xml_files = sorted(xml_dir.glob("*.xml"))
        if xml_files:
            return xml_files[0]

    raise ValueError(f"No .xml files found under {asset_dir / 'mjcf'} or {asset_dir / 'xml'}")


def _error_result(message: str) -> dict[str, Any]:
    return {"status": "error", "message": message}


def _without_kind(geom_record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in geom_record.items() if key != "kind"}
