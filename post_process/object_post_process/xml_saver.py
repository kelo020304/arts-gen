"""MJCF save pipeline for the object post-process editor."""

from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

try:
    import mujoco
except Exception as exc:  # pragma: no cover - import failure is surfaced as a structured backend error
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None

from .mjcf_parser import generate_manifest

_BODY_OVERRIDE_ATTRS = {"name", "pos", "quat", "euler", "axisangle", "zaxis", "xyaxes"}
_JOINT_OVERRIDE_ATTRS = {"name", "type", "pos", "axis", "range", "limited"}
_GEOM_OVERRIDE_ATTRS = {
    "name",
    "mesh",
    "pos",
    "quat",
    "euler",
    "axisangle",
    "xyaxes",
    "zaxis",
    "material",
    "rgba",
    "group",
    "contype",
    "conaffinity",
    "condim",
    "friction",
    "density",
}


class SaveError(Exception):
    """Base class for structured save failures."""


class ValidationError(SaveError):
    """Raised when the editor-state payload is invalid."""


class CompileError(SaveError):
    """Raised when MuJoCo validation fails."""


def validate_editor_state(payload: dict, asset_name: str, assets_root: Path) -> dict[str, Any]:
    """Validate and normalize the Phase 3 editor-state payload."""
    if not isinstance(payload, dict):
        raise ValidationError("Save payload must be a JSON object")

    asset_dir = assets_root / asset_name
    if not asset_dir.is_dir():
        raise ValidationError(f"Asset directory does not exist: {asset_dir}")

    xml_path_value = payload.get("xml_path")
    if not isinstance(xml_path_value, str) or not xml_path_value.strip():
        raise ValidationError("Missing xml_path")

    scene_graph = payload.get("scene_graph")
    if not isinstance(scene_graph, dict):
        raise ValidationError("Missing scene_graph")

    source_xml_path, rel_xml_path = _resolve_source_xml_path(asset_dir, xml_path_value)
    xml_root = _load_xml_root(source_xml_path)
    mesh_file_to_name = _collect_mesh_file_to_name(xml_root)

    root_body_name = _require_string(scene_graph.get("root_body"), "scene_graph.root_body")

    bodies_payload = scene_graph.get("bodies")
    if not isinstance(bodies_payload, list) or not bodies_payload:
        raise ValidationError("scene_graph.bodies must be a non-empty list")

    joints_payload = scene_graph.get("joints")
    if not isinstance(joints_payload, list):
        raise ValidationError("scene_graph.joints must be a list")

    geom_names: set[str] = set()
    bodies_by_name: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str, list[str]] = defaultdict(list)

    for index, body_payload in enumerate(bodies_payload):
        if not isinstance(body_payload, dict):
            raise ValidationError(f"scene_graph.bodies[{index}] must be an object")

        body_id = _require_string(body_payload.get("id"), f"scene_graph.bodies[{index}].id")
        body_name = _require_string(body_payload.get("name"), f"scene_graph.bodies[{index}].name")
        if body_id != body_name:
            raise ValidationError(f"Body id/name mismatch: {body_id} != {body_name}")
        if body_name in bodies_by_name:
            raise ValidationError(f"Duplicate body name: {body_name}")

        parent_name = body_payload.get("parent")
        if parent_name is not None:
            parent_name = _require_string(parent_name, f"scene_graph.bodies[{index}].parent")

        pos = _require_numeric_list(body_payload.get("pos"), 3, f"Body {body_name} pos")
        quat = _require_numeric_list(body_payload.get("quat"), 4, f"Body {body_name} quat")

        joint_ids = _require_string_list(body_payload.get("joint_ids"), f"Body {body_name} joint_ids")
        if len(joint_ids) > 1:
            raise ValidationError(f"Body {body_name} has {len(joint_ids)} joints; Phase 3 v1 supports at most one joint per body")

        visual_geoms = _validate_geom_list(
            body_payload.get("visual_geoms"),
            body_name,
            "visual_geoms",
            geom_names,
            asset_dir,
            mesh_file_to_name,
        )
        collision_geoms = _validate_geom_list(
            body_payload.get("collision_geoms"),
            body_name,
            "collision_geoms",
            geom_names,
            asset_dir,
            mesh_file_to_name,
        )

        merged_from = body_payload.get("merged_from", [])
        if not isinstance(merged_from, list):
            raise ValidationError(f"Body {body_name} merged_from must be a list")
        for merged_index, merged_name in enumerate(merged_from):
            _require_string(merged_name, f"Body {body_name} merged_from[{merged_index}]")

        bodies_by_name[body_name] = {
            "id": body_id,
            "name": body_name,
            "parent": parent_name,
            "pos": pos,
            "quat": quat,
            "joint_ids": joint_ids,
            "visual_geoms": visual_geoms,
            "collision_geoms": collision_geoms,
            "merged_from": merged_from,
        }
        if parent_name is not None:
            children_by_parent[parent_name].append(body_name)

    root_bodies = [body_name for body_name, body_data in bodies_by_name.items() if body_data["parent"] is None]
    if len(root_bodies) == 0:
        raise ValidationError("Scene graph must contain exactly one root body; found 0")
    if len(root_bodies) > 1:
        raise ValidationError(f"Scene graph must contain exactly one root body; found {len(root_bodies)}")
    if root_bodies[0] != root_body_name:
        raise ValidationError(
            f"scene_graph.root_body {root_body_name} does not match the only root body {root_bodies[0]}"
        )

    for body_name, body_data in bodies_by_name.items():
        parent_name = body_data["parent"]
        if parent_name is not None and parent_name not in bodies_by_name:
            raise ValidationError(f"Body {body_name} references missing parent {parent_name}")

    visited: dict[str, int] = {}

    def visit(body_name: str) -> None:
        state = visited.get(body_name, 0)
        if state == 1:
            raise ValidationError(f"Cycle detected in body tree at {body_name}")
        if state == 2:
            return
        visited[body_name] = 1
        for child_name in children_by_parent.get(body_name, []):
            visit(child_name)
        visited[body_name] = 2

    visit(root_body_name)
    if len(visited) != len(bodies_by_name):
        unvisited = sorted(set(bodies_by_name) - set(visited))
        raise ValidationError(f"Cycle detected or disconnected body tree: {', '.join(unvisited)}")

    joints_by_name: dict[str, dict[str, Any]] = {}
    orders: list[int] = []
    for index, joint_payload in enumerate(joints_payload):
        if not isinstance(joint_payload, dict):
            raise ValidationError(f"scene_graph.joints[{index}] must be an object")

        joint_id = _require_string(joint_payload.get("id"), f"scene_graph.joints[{index}].id")
        joint_name = _require_string(joint_payload.get("name"), f"scene_graph.joints[{index}].name")
        if joint_id != joint_name:
            raise ValidationError(f"Joint id/name mismatch: {joint_id} != {joint_name}")
        if joint_name in joints_by_name:
            raise ValidationError(f"Duplicate joint name: {joint_name}")

        body_name = _require_string(joint_payload.get("body"), f"Joint {joint_name} body")
        if body_name not in bodies_by_name:
            raise ValidationError(f"Joint {joint_name} references missing body {body_name}")

        joint_type = _require_string(joint_payload.get("type"), f"Joint {joint_name} type")
        if joint_type not in {"hinge", "slide"}:
            raise ValidationError(f"Joint {joint_name} has unsupported type {joint_type}")

        anchor = _require_numeric_list(joint_payload.get("anchor"), 3, f"Joint {joint_name} anchor")
        axis = _require_numeric_list(joint_payload.get("axis"), 3, f"Joint {joint_name} axis")
        axis_norm = math.sqrt(sum(component * component for component in axis))
        if axis_norm <= 1e-9:
            raise ValidationError(f"Joint {joint_name} axis length must be > 1e-9")

        joint_range = _require_numeric_list(joint_payload.get("range"), 2, f"Joint {joint_name} range")
        if joint_range[0] > joint_range[1]:
            raise ValidationError(f"Joint {joint_name} range min must be <= max")

        default_value = _require_number(joint_payload.get("default"), f"Joint {joint_name} default")
        order = _require_int(joint_payload.get("order"), f"Joint {joint_name} order")
        if order < 0:
            raise ValidationError(f"Joint {joint_name} order must be >= 0")

        orders.append(order)
        joints_by_name[joint_name] = {
            "id": joint_id,
            "name": joint_name,
            "body": body_name,
            "type": joint_type,
            "anchor": anchor,
            "axis": axis,
            "range": joint_range,
            "default": default_value,
            "order": order,
        }

    expected_orders = list(range(len(joints_by_name)))
    if sorted(orders) != expected_orders:
        raise ValidationError(
            f"Joint orders must be a contiguous zero-based sequence; got {sorted(orders)}, expected {expected_orders}"
        )

    joints_by_body: dict[str, list[str]] = defaultdict(list)
    for joint_name, joint_data in joints_by_name.items():
        joints_by_body[joint_data["body"]].append(joint_name)

    for body_name, body_data in bodies_by_name.items():
        declared_joint_ids = body_data["joint_ids"]
        actual_joint_ids = joints_by_body.get(body_name, [])
        if sorted(declared_joint_ids) != sorted(actual_joint_ids):
            raise ValidationError(
                f"Body/joint mismatch for {body_name}: joint_ids={declared_joint_ids}, actual={actual_joint_ids}"
            )

    return {
        "asset_dir": asset_dir,
        "source_xml_path": source_xml_path,
        "xml_path": rel_xml_path,
        "root_body": root_body_name,
        "bodies_by_name": bodies_by_name,
        "children_by_parent": dict(children_by_parent),
        "joints_by_name": joints_by_name,
        "joints_in_order": sorted(joints_by_name.values(), key=lambda joint: joint["order"]),
        "mesh_file_to_name": mesh_file_to_name,
    }


def save_editor_state_to_xml(asset_name: str, payload: dict, assets_root: Path) -> dict[str, Any]:
    """Save editor-state JSON back into the asset XML after MuJoCo validation."""
    try:
        _ensure_mjspec_available()
        validated = validate_editor_state(payload, asset_name, assets_root)
        source_xml_path: Path = validated["source_xml_path"]

        xml_tree = _load_xml_tree(source_xml_path)
        xml_root = xml_tree.getroot()
        worldbody = xml_root.find("worldbody")
        if worldbody is None:
            raise ValidationError(f"Missing <worldbody> in {source_xml_path}")

        original_root_body = _find_direct_body_child(worldbody, validated["root_body"])
        if original_root_body is None:
            raise ValidationError(
                f"Root body {validated['root_body']} was not found as a direct child of <worldbody> in {source_xml_path}"
            )

        original_body_data, original_joint_names, original_geom_attrs_by_name, original_joint_attrs_by_name = _collect_original_body_data(
            original_root_body,
            validated["mesh_file_to_name"],
        )
        rebuilt_root_body = _build_body_element(
            validated["root_body"],
            validated,
            original_body_data,
            original_geom_attrs_by_name,
            original_joint_attrs_by_name,
        )
        rebuilt_root_body.tail = original_root_body.tail

        worldbody_children = list(worldbody)
        root_body_index = worldbody_children.index(original_root_body)
        worldbody.remove(original_root_body)
        worldbody.insert(root_body_index, rebuilt_root_body)

        _sync_keyframes(xml_root, validated["joints_in_order"])
        _sync_actuators(xml_root, original_joint_names, validated["joints_in_order"])

        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{source_xml_path.stem}.",
            suffix=".xml",
            dir=str(source_xml_path.parent),
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            ET.indent(xml_tree, space="  ")
            xml_tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
            _compile_xml(tmp_path)
            os.replace(tmp_path, source_xml_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        manifest = generate_manifest(asset_name, assets_root)
        if manifest.get("status") != "ok":
            raise RuntimeError(f"Saved XML but failed to regenerate manifest: {manifest.get('message', 'unknown error')}")

        return {
            "status": "ok",
            "message": "XML saved",
            "xml_path": validated["xml_path"],
            "manifest": manifest,
        }
    except SaveError as exc:
        return _error_result(str(exc))


def _build_body_element(
    body_name: str,
    validated: dict[str, Any],
    original_body_data: dict[str, dict[str, Any]],
    original_geom_attrs_by_name: dict[str, dict[str, str]],
    original_joint_attrs_by_name: dict[str, dict[str, str]],
) -> ET.Element:
    body_state = validated["bodies_by_name"][body_name]
    original_data = original_body_data.get(body_name)

    body_attrs = dict(original_data["attrs"]) if original_data else {}
    _drop_keys(body_attrs, _BODY_OVERRIDE_ATTRS)
    body_attrs["name"] = body_state["name"]
    body_attrs["pos"] = _format_numbers(body_state["pos"])
    body_attrs["quat"] = _format_numbers(body_state["quat"])
    body_element = ET.Element("body", body_attrs)

    joint_payload = None
    if body_state["joint_ids"]:
        joint_payload = validated["joints_by_name"][body_state["joint_ids"][0]]
    geom_payloads = [*body_state["visual_geoms"], *body_state["collision_geoms"]]
    child_body_names = validated["children_by_parent"].get(body_name, [])

    if original_data is None:
        if joint_payload is not None:
            body_element.append(_build_joint_element(joint_payload, original_joint_attrs_by_name.get(joint_payload["name"])))
        for geom_payload in geom_payloads:
            body_element.append(
                _build_geom_element(
                    geom_payload,
                    original_geom_attrs_by_name.get(geom_payload["name"]),
                    validated["mesh_file_to_name"],
                )
            )
        for child_name in child_body_names:
            body_element.append(
                _build_body_element(
                    child_name,
                    validated,
                    original_body_data,
                    original_geom_attrs_by_name,
                    original_joint_attrs_by_name,
                )
            )
        return body_element

    joint_inserted = False
    geom_inserted = False
    child_bodies_inserted = False

    for child in list(original_data["element"]):
        tag = child.tag if isinstance(child.tag, str) else None
        if tag == "joint":
            if not joint_inserted:
                if joint_payload is not None:
                    original_joint_attrs = original_data["joint_attrs"].get(joint_payload["name"]) or original_joint_attrs_by_name.get(
                        joint_payload["name"]
                    )
                    body_element.append(_build_joint_element(joint_payload, original_joint_attrs))
                joint_inserted = True
            continue
        if tag == "geom":
            if not geom_inserted:
                for geom_payload in geom_payloads:
                    original_geom_attrs = original_data["geom_attrs"].get(geom_payload["name"]) or original_geom_attrs_by_name.get(
                        geom_payload["name"]
                    )
                    body_element.append(_build_geom_element(geom_payload, original_geom_attrs, validated["mesh_file_to_name"]))
                geom_inserted = True
            continue
        if tag == "body":
            if not child_bodies_inserted:
                for child_name in child_body_names:
                    body_element.append(
                        _build_body_element(
                            child_name,
                            validated,
                            original_body_data,
                            original_geom_attrs_by_name,
                            original_joint_attrs_by_name,
                        )
                    )
                child_bodies_inserted = True
            continue
        body_element.append(deepcopy(child))

    if not joint_inserted and joint_payload is not None:
        original_joint_attrs = original_data["joint_attrs"].get(joint_payload["name"]) or original_joint_attrs_by_name.get(
            joint_payload["name"]
        )
        body_element.append(_build_joint_element(joint_payload, original_joint_attrs))
    if not geom_inserted:
        for geom_payload in geom_payloads:
            original_geom_attrs = original_data["geom_attrs"].get(geom_payload["name"]) or original_geom_attrs_by_name.get(
                geom_payload["name"]
            )
            body_element.append(_build_geom_element(geom_payload, original_geom_attrs, validated["mesh_file_to_name"]))
    if not child_bodies_inserted:
        for child_name in child_body_names:
            body_element.append(
                _build_body_element(
                    child_name,
                    validated,
                    original_body_data,
                    original_geom_attrs_by_name,
                    original_joint_attrs_by_name,
                )
            )

    return body_element


def _build_joint_element(joint_payload: dict[str, Any], original_attrs: dict[str, str] | None) -> ET.Element:
    joint_attrs = dict(original_attrs) if original_attrs else {}
    _drop_keys(joint_attrs, _JOINT_OVERRIDE_ATTRS)
    joint_attrs["name"] = joint_payload["name"]
    joint_attrs["type"] = joint_payload["type"]
    joint_attrs["pos"] = _format_numbers(joint_payload["anchor"])
    joint_attrs["axis"] = _format_numbers(joint_payload["axis"])
    joint_attrs["range"] = _format_numbers(joint_payload["range"])
    joint_attrs["limited"] = "true"
    return ET.Element("joint", joint_attrs)


def _build_geom_element(
    geom_payload: dict[str, Any],
    original_attrs: dict[str, str] | None,
    mesh_file_to_name: dict[str, str],
) -> ET.Element:
    mesh_name = mesh_file_to_name.get(geom_payload["mesh_file"])
    if mesh_name is None:
        raise ValidationError(f"Mesh file {geom_payload['mesh_file']} is not declared in <asset>")

    geom_attrs = dict(original_attrs) if original_attrs else {}
    _drop_keys(geom_attrs, _GEOM_OVERRIDE_ATTRS)
    geom_attrs.setdefault("type", "mesh")
    geom_attrs["name"] = geom_payload["name"]
    geom_attrs["mesh"] = mesh_name
    geom_attrs["pos"] = _format_numbers(geom_payload["pos"])
    geom_attrs["quat"] = _format_numbers(geom_payload["quat"])
    geom_attrs["group"] = str(geom_payload["group"])
    geom_attrs["contype"] = str(geom_payload["contype"])
    geom_attrs["conaffinity"] = str(geom_payload["conaffinity"])

    if "material" in geom_payload:
        geom_attrs["material"] = geom_payload["material"]
    if "rgba" in geom_payload:
        geom_attrs["rgba"] = _format_numbers(geom_payload["rgba"])
    if "condim" in geom_payload:
        geom_attrs["condim"] = str(geom_payload["condim"])
    if "friction" in geom_payload:
        geom_attrs["friction"] = _format_numbers(geom_payload["friction"])
    if "density" in geom_payload:
        geom_attrs["density"] = _format_number(geom_payload["density"])

    return ET.Element("geom", geom_attrs)


def _sync_keyframes(xml_root: ET.Element, joints_in_order: list[dict[str, Any]]) -> None:
    keyframe_element = xml_root.find("keyframe")
    if not joints_in_order:
        if keyframe_element is None:
            return
        for key_element in keyframe_element.findall("key"):
            key_element.set("qpos", "")
        return

    qpos = [0.0] * len(joints_in_order)
    for joint_payload in joints_in_order:
        qpos[joint_payload["order"]] = joint_payload["default"]
    qpos_text = _format_numbers(qpos)

    if keyframe_element is None:
        keyframe_element = ET.SubElement(xml_root, "keyframe")

    key_elements = keyframe_element.findall("key")
    if not key_elements:
        key_elements = [ET.SubElement(keyframe_element, "key", {"name": "default_pose"})]

    for key_element in key_elements:
        key_element.set("qpos", qpos_text)


def _sync_actuators(
    xml_root: ET.Element,
    original_joint_names: set[str],
    joints_in_order: list[dict[str, Any]],
) -> None:
    new_joint_names = {joint_payload["name"] for joint_payload in joints_in_order}
    managed_joint_names = original_joint_names | new_joint_names

    actuator_element = xml_root.find("actuator")
    if actuator_element is None:
        if not joints_in_order:
            return
        actuator_element = ET.SubElement(xml_root, "actuator")

    reusable_position_actuators: dict[str, ET.Element] = {}
    for actuator_child in list(actuator_element):
        joint_name = actuator_child.get("joint")
        if joint_name not in managed_joint_names:
            continue
        if actuator_child.tag == "position" and joint_name in new_joint_names and joint_name not in reusable_position_actuators:
            reusable_position_actuators[joint_name] = actuator_child
        actuator_element.remove(actuator_child)

    for joint_payload in joints_in_order:
        actuator_child = reusable_position_actuators.get(joint_payload["name"])
        if actuator_child is None:
            actuator_child = ET.Element("position")
        actuator_attrs = dict(actuator_child.attrib)
        actuator_attrs["name"] = actuator_attrs.get("name") or f"{joint_payload['name']}_pos"
        actuator_attrs["joint"] = joint_payload["name"]
        actuator_attrs["ctrlrange"] = _format_numbers(joint_payload["range"])
        actuator_attrs["ctrllimited"] = "true"
        actuator_child.attrib.clear()
        actuator_child.attrib.update(actuator_attrs)
        actuator_element.append(actuator_child)

    if len(actuator_element) == 0:
        xml_root.remove(actuator_element)


def _collect_original_body_data(
    root_body_element: ET.Element,
    mesh_file_to_name: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], set[str], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    body_data: dict[str, dict[str, Any]] = {}
    joint_names: set[str] = set()
    geom_attrs_by_name: dict[str, dict[str, str]] = {}
    joint_attrs_by_name: dict[str, dict[str, str]] = {}

    def walk(body_element: ET.Element) -> None:
        body_name = body_element.get("name")
        if not body_name:
            raise ValidationError("Encountered unnamed <body> while rebuilding XML")
        if body_name in body_data:
            raise ValidationError(f"Duplicate body name in source XML: {body_name}")

        geom_attrs: dict[str, dict[str, str]] = {}
        joint_attrs: dict[str, dict[str, str]] = {}
        for child in list(body_element):
            tag = child.tag if isinstance(child.tag, str) else None
            if tag == "geom":
                geom_name = _canonical_geom_name(child, mesh_file_to_name)
                if geom_name in geom_attrs:
                    raise ValidationError(f"Duplicate geom name in source XML: {geom_name}")
                geom_attrs[geom_name] = dict(child.attrib)
                if geom_name in geom_attrs_by_name:
                    raise ValidationError(f"Duplicate geom name across source XML subtree: {geom_name}")
                geom_attrs_by_name[geom_name] = dict(child.attrib)
            elif tag == "joint":
                joint_name = child.get("name")
                if not joint_name:
                    raise ValidationError(f"Encountered unnamed <joint> under body {body_name}")
                if joint_name in joint_attrs:
                    raise ValidationError(f"Duplicate joint name under body {body_name}: {joint_name}")
                joint_attrs[joint_name] = dict(child.attrib)
                if joint_name in joint_attrs_by_name:
                    raise ValidationError(f"Duplicate joint name across source XML subtree: {joint_name}")
                joint_attrs_by_name[joint_name] = dict(child.attrib)
                joint_names.add(joint_name)

        body_data[body_name] = {
            "element": body_element,
            "attrs": dict(body_element.attrib),
            "geom_attrs": geom_attrs,
            "joint_attrs": joint_attrs,
        }

        for child_body in body_element.findall("body"):
            walk(child_body)

    walk(root_body_element)
    return body_data, joint_names, geom_attrs_by_name, joint_attrs_by_name


def _find_direct_body_child(worldbody: ET.Element, body_name: str) -> ET.Element | None:
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") == body_name:
            return child
    return None


def _collect_mesh_file_to_name(xml_root: ET.Element) -> dict[str, str]:
    asset_element = xml_root.find("asset")
    mesh_file_to_name: dict[str, str] = {}
    if asset_element is None:
        return mesh_file_to_name

    for mesh_element in asset_element.findall("mesh"):
        mesh_name = mesh_element.get("name")
        mesh_file = mesh_element.get("file")
        if not mesh_name or not mesh_file:
            continue
        if mesh_file in mesh_file_to_name and mesh_file_to_name[mesh_file] != mesh_name:
            raise ValidationError(f"Mesh file {mesh_file} is declared multiple times in <asset>")
        mesh_file_to_name[mesh_file] = mesh_name
    return mesh_file_to_name


def _canonical_geom_name(geom_element: ET.Element, mesh_file_to_name: dict[str, str]) -> str:
    explicit_name = geom_element.get("name")
    if explicit_name:
        return explicit_name

    mesh_name = geom_element.get("mesh")
    if mesh_name:
        for mesh_file, candidate_mesh_name in mesh_file_to_name.items():
            if candidate_mesh_name == mesh_name:
                return Path(mesh_file).stem
        return mesh_name

    raise ValidationError("Encountered <geom> without name or mesh")


def _resolve_source_xml_path(asset_dir: Path, xml_path_value: str) -> tuple[Path, str]:
    requested_rel_path = Path(xml_path_value.strip())
    if requested_rel_path.is_absolute():
        raise ValidationError("xml_path must be relative to the asset directory")

    requested_path = (asset_dir / requested_rel_path).resolve()
    try:
        requested_path.relative_to(asset_dir.resolve())
    except ValueError as exc:
        raise ValidationError(f"xml_path escapes the asset directory: {xml_path_value}") from exc

    located_path = _locate_source_xml(asset_dir)
    expected_rel = str(located_path.resolve().relative_to(asset_dir.resolve()))
    if requested_path != located_path.resolve():
        raise ValidationError(f"xml_path must reference the source XML {expected_rel}")

    return located_path, expected_rel


def _locate_source_xml(asset_dir: Path) -> Path:
    for subdir in ("mjcf", "xml"):
        xml_dir = asset_dir / subdir
        if not xml_dir.is_dir():
            continue
        xml_files = sorted(xml_dir.glob("*.xml"))
        if xml_files:
            return xml_files[0]
    raise ValidationError(f"No .xml files found under {asset_dir / 'mjcf'} or {asset_dir / 'xml'}")


def _load_xml_root(xml_path: Path) -> ET.Element:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(xml_path, parser=parser).getroot()


def _load_xml_tree(xml_path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(xml_path, parser=parser)


def _validate_geom_list(
    geom_payloads: Any,
    body_name: str,
    field_name: str,
    geom_names: set[str],
    asset_dir: Path,
    mesh_file_to_name: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(geom_payloads, list):
        raise ValidationError(f"Body {body_name} {field_name} must be a list")

    normalized_geoms: list[dict[str, Any]] = []
    for index, geom_payload in enumerate(geom_payloads):
        if not isinstance(geom_payload, dict):
            raise ValidationError(f"Body {body_name} {field_name}[{index}] must be an object")

        geom_name = _require_string(geom_payload.get("name"), f"Body {body_name} {field_name}[{index}].name")
        if geom_name in geom_names:
            raise ValidationError(f"Duplicate geom name: {geom_name}")
        geom_names.add(geom_name)

        mesh_file = _require_string(geom_payload.get("mesh_file"), f"Geom {geom_name} mesh_file")
        if mesh_file not in mesh_file_to_name:
            raise ValidationError(f"Mesh file {mesh_file} is not declared in <asset>")
        _ensure_mesh_file_exists(asset_dir, mesh_file)

        geom_data: dict[str, Any] = {
            "name": geom_name,
            "mesh_file": mesh_file,
            "pos": _require_numeric_list(geom_payload.get("pos"), 3, f"Geom {geom_name} pos"),
            "quat": _require_numeric_list(geom_payload.get("quat"), 4, f"Geom {geom_name} quat"),
            "group": _require_int(geom_payload.get("group"), f"Geom {geom_name} group"),
            "contype": _require_int(geom_payload.get("contype"), f"Geom {geom_name} contype"),
            "conaffinity": _require_int(geom_payload.get("conaffinity"), f"Geom {geom_name} conaffinity"),
        }

        if "material" in geom_payload:
            geom_data["material"] = _require_string(geom_payload.get("material"), f"Geom {geom_name} material")
        if "rgba" in geom_payload:
            geom_data["rgba"] = _require_numeric_list(geom_payload.get("rgba"), 4, f"Geom {geom_name} rgba")
        if "condim" in geom_payload:
            geom_data["condim"] = _require_int(geom_payload.get("condim"), f"Geom {geom_name} condim")
        if "friction" in geom_payload:
            geom_data["friction"] = _require_numeric_list(geom_payload.get("friction"), 3, f"Geom {geom_name} friction")
        if "density" in geom_payload:
            geom_data["density"] = _require_number(geom_payload.get("density"), f"Geom {geom_name} density")

        normalized_geoms.append(geom_data)

    return normalized_geoms


def _ensure_mesh_file_exists(asset_dir: Path, mesh_file: str) -> None:
    found = False
    for subdir in ("mjcf", "xml"):
        xml_dir = asset_dir / subdir
        if not xml_dir.is_dir():
            continue
        for base_dir in (xml_dir, xml_dir / "assets"):
            candidate = (base_dir / mesh_file).resolve()
            try:
                candidate.relative_to(xml_dir.resolve())
            except ValueError:
                continue
            if candidate.is_file():
                found = True
                break
        if found:
            break
    if not found:
        raise ValidationError(f"Referenced mesh_file is missing from asset directories: {mesh_file}")


def _ensure_mjspec_available() -> None:
    if mujoco is None:
        raise CompileError(f"MuJoCo import failed: {_MUJOCO_IMPORT_ERROR}")
    if not hasattr(mujoco, "MjSpec"):
        raise CompileError("MuJoCo MjSpec is unavailable in the current environment")


def _compile_xml(xml_path: Path) -> None:
    try:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        spec.compile()
    except Exception as exc:
        raise CompileError(f"MjSpec compile failed: {exc}") from exc


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValidationError(f"{field_name} must be a list")
    return [_require_string(item, f"{field_name}[{index}]") for index, item in enumerate(value)]


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field_name} must be a finite number")
    return number


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be an integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValidationError(f"{field_name} must be an integer")
    return int(number)


def _require_numeric_list(value: Any, expected_length: int, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValidationError(f"{field_name} must be a list of length {expected_length}")
    return [_require_number(component, f"{field_name}[{index}]") for index, component in enumerate(value)]


def _drop_keys(mapping: dict[str, str], keys: set[str]) -> None:
    for key in keys:
        mapping.pop(key, None)


def _format_numbers(values: list[float]) -> str:
    return " ".join(_format_number(value) for value in values)


def _format_number(value: float) -> str:
    return f"{float(value):.17g}"


def _error_result(message: str) -> dict[str, Any]:
    return {"status": "error", "message": message, "details": None}
