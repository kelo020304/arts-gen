"""Render one GLB mesh preview with Blender vertex colors."""

from __future__ import annotations

import math
import os
from pathlib import Path

import bpy
import mathutils


RUN_DIR = Path(os.environ["RUN_DIR"]).resolve()
GLB_PATH = Path(os.environ.get("GLB_PATH", RUN_DIR / "complete.glb")).resolve()
OUT_DIR = Path(os.environ.get("OUT_DIR", RUN_DIR / "mesh_renders")).resolve()
PREFIX = os.environ.get("PREFIX", GLB_PATH.stem)
RES = int(os.environ.get("RES", "768"))


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def setup_scene() -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 64
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.02, 0.02, 0.02)

    bpy.ops.object.light_add(type="AREA", location=(1.8, -2.2, 2.4))
    key = bpy.context.object
    key.name = "key_light"
    key.data.energy = 450
    key.data.size = 4.0

    bpy.ops.object.light_add(type="AREA", location=(-2.0, 1.5, 1.6))
    fill = bpy.context.object
    fill.name = "fill_light"
    fill.data.energy = 80
    fill.data.size = 5.0


def mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def apply_vertex_color_materials() -> None:
    for obj in mesh_objects():
        mesh = obj.data
        mat = bpy.data.materials.new(f"{obj.name}_vertex_color")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None:
            continue
        bsdf.inputs["Roughness"].default_value = 0.78
        bsdf.inputs["Metallic"].default_value = 0.02
        if mesh.color_attributes:
            attr_name = mesh.color_attributes[0].name
            vc = nodes.new(type="ShaderNodeVertexColor")
            vc.layer_name = attr_name
            mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.72, 1.0)
        mesh.materials.clear()
        mesh.materials.append(mat)


def center_and_scale() -> None:
    objs = mesh_objects()
    if not objs:
        raise RuntimeError("imported GLB contains no mesh objects")
    bpy.context.view_layer.update()
    mins = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    maxs = mathutils.Vector((-float("inf"), -float("inf"), -float("inf")))
    for obj in objs:
        for corner in obj.bound_box:
            world = obj.matrix_world @ mathutils.Vector(corner)
            mins.x = min(mins.x, world.x)
            mins.y = min(mins.y, world.y)
            mins.z = min(mins.z, world.z)
            maxs.x = max(maxs.x, world.x)
            maxs.y = max(maxs.y, world.y)
            maxs.z = max(maxs.z, world.z)
    center = (mins + maxs) * 0.5
    extent = max((maxs - mins).x, (maxs - mins).y, (maxs - mins).z)
    if extent <= 0:
        raise RuntimeError(f"invalid mesh extent: {extent}")
    scale = 1.35 / extent
    root = bpy.data.objects.new("preview_root", None)
    bpy.context.collection.objects.link(root)
    root.location = -center * scale
    root.scale = (scale, scale, scale)
    for obj in objs:
        obj.parent = root
    bpy.context.view_layer.update()


def add_camera(angle_deg: float, elevation_deg: float, dist: float = 3.0):
    az = math.radians(angle_deg)
    el = math.radians(elevation_deg)
    x = dist * math.cos(el) * math.cos(az)
    y = dist * math.cos(el) * math.sin(az)
    z = dist * math.sin(el)
    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.object
    direction = mathutils.Vector((0.0, 0.0, 0.0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 55
    cam.data.sensor_width = 32
    bpy.context.scene.camera = cam


def main() -> None:
    if not GLB_PATH.is_file():
        raise FileNotFoundError(GLB_PATH)
    clean_scene()
    setup_scene()
    bpy.ops.import_scene.gltf(filepath=str(GLB_PATH))
    apply_vertex_color_materials()
    center_and_scale()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, az, el in (("front", 270, 8), ("iso", 315, 24), ("side", 0, 8)):
        add_camera(az, el)
        path = OUT_DIR / f"{PREFIX}_{name}.png"
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        print(f"[render] {path}", flush=True)


if __name__ == "__main__":
    main()
