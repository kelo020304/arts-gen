"""Export decoded-mesh kinematic predictions to MJCF and portable USDA."""

from __future__ import annotations

import math
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from .kin_agent import KinematicCandidate


def export_decoded_mesh_obj(source: Path, destination: Path) -> Path:
    """Convert decoded mesh geometry to OBJ, retaining decoded vertex colors."""
    import trimesh
    from trimesh.exchange.obj import export_obj

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    loaded = trimesh.load(source, force="scene", process=False)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    if getattr(mesh.visual, "kind", None) == "vertex":
        mesh.visual = mesh.visual.to_texture()
        mesh.visual.material.name = destination.stem
    obj_text, auxiliary = export_obj(
        mesh,
        include_color=True,
        include_texture=True,
        return_texture=True,
        mtl_name=f"{destination.stem}.mtl",
    )
    destination.write_text(obj_text, encoding="utf-8")
    for name, payload in auxiliary.items():
        target = destination.parent / Path(name).name
        target.write_text(payload, encoding="utf-8") if isinstance(payload, str) else target.write_bytes(payload)
        if target.suffix.lower() == ".png":
            _ensure_power_of_two_texture(target)
    return destination


def _ensure_power_of_two_texture(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        target_size = 1
        while target_size < max(image.size):
            target_size *= 2
        if image.size == (target_size, target_size):
            return
        image.resize((target_size, target_size), Image.Resampling.NEAREST).save(path)


def write_kinematic_mjcf(
    path: Path,
    *,
    object_name: str,
    body_mesh: Path,
    moving_mesh: Path,
    joint_name: str,
    candidate: KinematicCandidate,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", {"model": object_name})
    ET.SubElement(root, "compiler", {
        "angle": "radian", "balanceinertia": "true", "inertiagrouprange": "3 5",
    })
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "body_mesh", "file": str(Path(body_mesh).resolve())})
    ET.SubElement(asset, "mesh", {"name": "moving_mesh", "file": str(Path(moving_mesh).resolve())})
    world = ET.SubElement(root, "worldbody")
    obj = ET.SubElement(world, "body", {
        "name": "object", "pos": "0 0 0", "quat": "0.707106781 0.707106781 0 0",
    })
    ET.SubElement(obj, "geom", {"name": "body_visual", "type": "mesh", "mesh": "body_mesh", "group": "0"})
    moving = ET.SubElement(obj, "body", {"name": "moving_part", "pos": "0 0 0"})
    joint_type = "slide" if candidate.joint_type == "prismatic" else "hinge"
    axis, lower, upper = _delivery_joint(candidate, joint_type == "slide")
    ET.SubElement(moving, "joint", {
        "name": joint_name,
        "type": joint_type,
        "pos": _values(candidate.origin_world),
        "axis": _values(axis),
        "range": _values((lower, upper)),
        "limited": "true",
    })
    ET.SubElement(moving, "geom", {"name": "moving_visual", "type": "mesh", "mesh": "moving_mesh", "group": "0"})
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return path


def write_kinematic_usda(
    path: Path,
    *,
    object_name: str,
    body_mesh: Path,
    moving_mesh: Path,
    joint_name: str,
    candidate: KinematicCandidate,
) -> Path:
    return write_kinematic_bundle_usda(
        path,
        object_name=object_name,
        body_mesh=body_mesh,
        parts=[{
            "body_name": "moving_part",
            "joint_name": joint_name,
            "source_mesh": moving_mesh,
            "candidate": candidate,
        }],
        force_prismatic_local_z=True,
    )


def write_kinematic_bundle_mjcf(
    path: Path,
    *,
    object_name: str,
    body_mesh: Path,
    parts: list[dict],
    force_prismatic_local_z: bool = False,
    apply_root_correction: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", {"model": object_name})
    ET.SubElement(root, "compiler", {
        "angle": "radian", "meshdir": "assets", "texturedir": "assets",
        "balanceinertia": "true", "inertiagrouprange": "3 5",
    })
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {
        "ambient": "0.35 0.35 0.35", "diffuse": "0.75 0.75 0.75", "specular": "0.1 0.1 0.1",
    })
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "body_mesh", "file": Path(body_mesh).name})
    body_material = _add_mjcf_material(asset, "body", Path(body_mesh))
    for index, part in enumerate(parts):
        ET.SubElement(asset, "mesh", {"name": f"moving_mesh_{index}", "file": Path(part["mesh"]).name})
        part["_mjcf_material"] = _add_mjcf_material(asset, f"moving_{index}", Path(part["mesh"]))
    world = ET.SubElement(root, "worldbody")
    object_attrs = {"name": "object", "pos": "0 0 0"}
    if apply_root_correction:
        object_attrs["quat"] = "0.707106781 0.707106781 0 0"
    obj = ET.SubElement(world, "body", object_attrs)
    body_visual_attrs = {
        "name": "body_visual", "type": "mesh", "mesh": "body_mesh", "group": "0",
        "contype": "0", "conaffinity": "0",
    }
    if body_material:
        body_visual_attrs["material"] = body_material
    ET.SubElement(obj, "geom", body_visual_attrs)
    for index, part in enumerate(parts):
        candidate: KinematicCandidate = part["candidate"]
        moving = ET.SubElement(obj, "body", {"name": str(part["body_name"]), "pos": "0 0 0"})
        joint_type = "slide" if candidate.joint_type == "prismatic" else "hinge"
        axis, lower, upper = _delivery_joint(candidate, force_prismatic_local_z)
        ET.SubElement(moving, "joint", {
            "name": str(part["joint_name"]), "type": joint_type,
            "pos": _values(candidate.origin_world), "axis": _values(axis),
            "range": _values((lower, upper)), "limited": "true",
        })
        visual_attrs = {
            "name": f"moving_visual_{index}", "type": "mesh", "mesh": f"moving_mesh_{index}",
            "group": "0", "contype": "0", "conaffinity": "0",
        }
        if part.get("_mjcf_material"):
            visual_attrs["material"] = str(part["_mjcf_material"])
        ET.SubElement(moving, "geom", visual_attrs)
        center, half_size = _mesh_box(Path(part["mesh"]))
        ET.SubElement(moving, "geom", {
            "name": f"moving_collision_{index}", "type": "box",
            "pos": _values(center), "size": _values(half_size),
            "group": "3", "rgba": "0 0 0 0", "density": "25",
            "contype": "1", "conaffinity": "1",
        })
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return path


def write_kinematic_bundle_usda(
    path: Path,
    *,
    object_name: str,
    body_mesh: Path,
    parts: list[dict],
    force_prismatic_local_z: bool = False,
    apply_root_correction: bool = True,
) -> Path:
    """Write a self-contained, renderable USDA articulation.

    Geometry and decoded vertex appearance are embedded so the result does not
    depend on a renderer understanding GLB/OBJ asset references.  The root
    orientation matches the MJCF delivery coordinate correction.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root_name = _usd_identifier(object_name, "object")
    body_payload = _load_usd_mesh(Path(body_mesh))
    part_payloads = [(part, _load_usd_mesh(Path(part["source_mesh"]))) for part in parts]
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "#usda 1.0\n(\n"
            f'    defaultPrim = "{root_name}"\n'
            "    metersPerUnit = 1\n"
            '    upAxis = "Z"\n'
            ")\n\n"
            f'def Xform "{root_name}" (\n'
            '    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysicsRigidBodyAPI"]\n'
            ")\n{\n"
            "    bool physics:kinematicEnabled = 1\n"
        )
        if apply_root_correction:
            stream.write(
                "    quatf xformOp:orient = (0.707106781, 0.707106781, 0, 0)\n"
                '    uniform token[] xformOpOrder = ["xformOp:orient"]\n'
            )
        stream.write('    def Xform "Visuals"\n    {\n')
        _write_usd_mesh(stream, "body", body_payload, "        ", collision=True)
        stream.write("    }\n")
        for part, payload in part_payloads:
            body_name = _usd_identifier(str(part["body_name"]), "moving_part")
            stream.write(
                f'    def Xform "{body_name}" (\n'
                '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n'
                "    )\n    {\n"
                f"        float physics:mass = {_usd_proxy_mass(payload):.9g}\n"
                '        def Xform "Visuals"\n        {\n'
            )
            _write_usd_mesh(stream, "mesh", payload, "            ")
            stream.write("        }\n")
            _write_usd_collision_box(stream, payload, "        ")
            stream.write("    }\n")
        for part, _payload in part_payloads:
            candidate: KinematicCandidate = part["candidate"]
            body_name = _usd_identifier(str(part["body_name"]), "moving_part")
            joint_name = _usd_identifier(str(part["joint_name"]), "joint")
            schema = "PhysicsPrismaticJoint" if candidate.joint_type == "prismatic" else "PhysicsRevoluteJoint"
            axis, lower, upper = _delivery_joint(candidate, force_prismatic_local_z)
            if candidate.joint_type == "revolute":
                lower, upper = math.degrees(lower), math.degrees(upper)
            quat = _quat_align_x_to(axis)
            stream.write(
                f'    def {schema} "{joint_name}"\n    {{\n'
                f"        rel physics:body0 = </{root_name}>\n"
                f"        rel physics:body1 = </{root_name}/{body_name}>\n"
                f"        point3f physics:localPos0 = ({_values(candidate.origin_world, separator=', ')})\n"
                f"        point3f physics:localPos1 = ({_values(candidate.origin_world, separator=', ')})\n"
                f"        quatf physics:localRot0 = ({_values(quat, separator=', ')})\n"
                f"        quatf physics:localRot1 = ({_values(quat, separator=', ')})\n"
                '        uniform token physics:axis = "X"\n'
                f"        float physics:lowerLimit = {lower:.9g}\n"
                f"        float physics:upperLimit = {upper:.9g}\n"
                "    }\n"
            )
        stream.write("}\n\n")
        _write_usd_vertex_color_material(stream)
    return path


def _load_usd_mesh(path: Path) -> dict:
    import numpy as np
    import trimesh

    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0:
        raise ValueError(f"decoded mesh has no vertices: {path}")
    faces = np.asarray(getattr(mesh, "faces", np.empty((0, 3))), dtype=np.int64).reshape(-1, 3)
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(vertices):
        colors = np.full((len(vertices), 4), 204, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.float32)[:, :3]
    if colors.max(initial=0.0) > 1.0:
        colors /= 255.0
    return {"vertices": vertices, "faces": faces, "colors": np.clip(colors, 0.0, 1.0)}


def _delivery_joint(
    candidate: KinematicCandidate,
    force_prismatic_local_z: bool,
) -> tuple[tuple[float, float, float], float, float]:
    if candidate.joint_type == "prismatic" and force_prismatic_local_z:
        return (0.0, 0.0, 1.0), 0.0, abs(float(candidate.upper) - float(candidate.lower))
    return candidate.axis_world, float(candidate.lower), float(candidate.upper)


def delivery_joint_payload(candidate: KinematicCandidate, force_prismatic_local_z: bool) -> dict:
    axis, lower, upper = _delivery_joint(candidate, force_prismatic_local_z)
    return {
        "joint_type": candidate.joint_type,
        "axis_world": [float(value) for value in axis],
        "origin_world": [float(value) for value in candidate.origin_world],
        "lower": lower,
        "upper": upper,
        "source": "RA local-Z export canonicalization" if force_prismatic_local_z and candidate.joint_type == "prismatic" else "canonical prediction",
    }


def _add_mjcf_material(asset: ET.Element, prefix: str, mesh_path: Path) -> str | None:
    texture_path = mesh_path.with_suffix(".png")
    if not texture_path.is_file():
        return None
    texture_name = f"{prefix}_texture"
    material_name = f"{prefix}_material"
    ET.SubElement(asset, "texture", {
        "name": texture_name, "type": "2d", "file": texture_path.name,
    })
    ET.SubElement(asset, "material", {
        "name": material_name, "texture": texture_name,
        "specular": "0.1", "shininess": "0.1",
    })
    return material_name


def _mesh_box(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    import numpy as np
    import trimesh

    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = (bounds[0] + bounds[1]) * 0.5
    half_size = np.maximum((bounds[1] - bounds[0]) * 0.5, 1e-4)
    return tuple(float(value) for value in center), tuple(float(value) for value in half_size)


def _write_usd_mesh(stream, name: str, payload: dict, indent: str, *, collision: bool = False) -> None:
    import numpy as np

    vertices = payload["vertices"]
    faces = payload["faces"]
    colors = payload["colors"]
    low = np.min(vertices, axis=0)
    high = np.max(vertices, axis=0)
    schemas = ["MaterialBindingAPI"]
    if collision:
        schemas.extend(["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])
    schema_text = ", ".join(f'"{schema}"' for schema in schemas)
    stream.write(
        f'{indent}def Mesh "{_usd_identifier(name, "mesh")}" (\n'
        f"{indent}    prepend apiSchemas = [{schema_text}]\n"
        f"{indent})\n{indent}{{\n"
        f'{indent}    uniform token subdivisionScheme = "none"\n'
        f"{indent}    uniform bool doubleSided = 1\n"
        f"{indent}    float3[] extent = [({_values(low, separator=', ')}), ({_values(high, separator=', ')})]\n"
    )
    _write_usd_array(stream, f"{indent}    point3f[] points", (
        f"({float(x):.8g}, {float(y):.8g}, {float(z):.8g})" for x, y, z in vertices
    ), 1, indent + "        ")
    _write_usd_array(stream, f"{indent}    int[] faceVertexCounts", ("3" for _ in faces), 24, indent + "        ")
    _write_usd_array(stream, f"{indent}    int[] faceVertexIndices", (str(int(value)) for value in faces.reshape(-1)), 18, indent + "        ")
    _write_usd_array(stream, f"{indent}    color3f[] primvars:displayColor", (
        f"({float(r):.6g}, {float(g):.6g}, {float(b):.6g})" for r, g, b in colors
    ), 1, indent + "        ")
    stream.write(f'{indent}    uniform token primvars:displayColor:interpolation = "vertex"\n')
    stream.write(f"{indent}    rel material:binding = </Materials/DecodedVertexColor>\n")
    if collision:
        stream.write(f'{indent}    uniform token physics:approximation = "none"\n')
    stream.write(f"{indent}}}\n")


def _write_usd_collision_box(stream, payload: dict, indent: str) -> None:
    import numpy as np

    vertices = payload["vertices"]
    low = np.min(vertices, axis=0)
    high = np.max(vertices, axis=0)
    center = (low + high) * 0.5
    half_size = np.maximum((high - low) * 0.5, 1e-4)
    stream.write(
        f'{indent}def Cube "Collision" (\n'
        f'{indent}    prepend apiSchemas = ["PhysicsCollisionAPI"]\n'
        f"{indent})\n{indent}{{\n"
        f'{indent}    uniform token purpose = "guide"\n'
        f"{indent}    double size = 2\n"
        f"{indent}    double3 xformOp:translate = ({_values(center, separator=', ')})\n"
        f"{indent}    float3 xformOp:scale = ({_values(half_size, separator=', ')})\n"
        f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n'
        f"{indent}    bool physics:collisionEnabled = 1\n"
        f"{indent}}}\n"
    )


def _usd_proxy_mass(payload: dict) -> float:
    import numpy as np

    extent = np.ptp(payload["vertices"], axis=0)
    return max(0.01, float(np.prod(np.maximum(extent, 1e-4))) * 25.0)


def _write_usd_array(stream, declaration: str, values, per_line: int, indent: str) -> None:
    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration:
        stream.write(f"{declaration} = []\n")
        return
    stream.write(f"{declaration} = [\n{indent}{first}")
    count = 1
    for value in iterator:
        if count:
            stream.write(", ")
        if count and count % per_line == 0:
            stream.write(f"\n{indent}")
        stream.write(value)
        count += 1
    stream.write(f"\n{indent[:-4]}]\n")


def _write_usd_vertex_color_material(stream) -> None:
    stream.write('''def Scope "Materials"
{
    def Material "DecodedVertexColor"
    {
        token outputs:surface.connect = </Materials/DecodedVertexColor/PreviewSurface.outputs:surface>
        def Shader "PrimvarReader"
        {
            uniform token info:id = "UsdPrimvarReader_float3"
            token inputs:varname = "displayColor"
            color3f outputs:result
        }
        def Shader "PreviewSurface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Materials/DecodedVertexColor/PrimvarReader.outputs:result>
            float inputs:roughness = 0.55
            token outputs:surface
        }
    }
}
''')


def _quat_align_x_to(axis) -> tuple[float, float, float, float]:
    import numpy as np

    target = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(target))
    if norm <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    target /= norm
    dot = float(np.clip(target[0], -1.0, 1.0))
    if dot < -0.999999:
        return (0.0, 0.0, 1.0, 0.0)
    w = math.sqrt((1.0 + dot) * 0.5)
    scale = 0.5 / w
    quat = np.asarray((w, 0.0, -target[2] * scale, target[1] * scale), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(value) for value in quat)


def _usd_identifier(value: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(value)).strip("_") or fallback
    return f"_{clean}" if clean[0].isdigit() else clean


def _values(values, separator=" ") -> str:
    return separator.join(f"{float(value):.9g}" for value in values)
