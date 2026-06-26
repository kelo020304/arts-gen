"""Render closed + open(-110 deg) preview PNGs for the Buds 6 proxy asset.

Reads the meshes built by ``build_xiaomi_buds6.py`` and applies the URDF's
revolute joint at min/max angle to visualize that body, lid, and hinge are
geometrically consistent.

Outputs:
    outputs/xiaomi_buds6_proxy/preview_closed.png
    outputs/xiaomi_buds6_proxy/preview_open.png
"""
import bpy
import os
from math import radians
from pathlib import Path


OUT_DIR = Path(os.environ.get("OUT_DIR", "outputs/xiaomi_buds6_proxy")).resolve()
OBJ_DIR = OUT_DIR / "objs"
HINGE_Y = 0.026285  # = W/2  (must match build script)
HINGE_Z = 0.0
LID_OPEN_DEG = -110.0
RES = 512


def _clean():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _setup_scene():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.film_transparent = True
    # Camera
    bpy.ops.object.camera_add(location=(0.10, -0.10, 0.06),
                              rotation=(radians(70), 0, radians(40)))
    cam = bpy.context.active_object
    cam.data.lens = 50
    scene.camera = cam
    # Light
    bpy.ops.object.light_add(type="SUN", location=(0.1, -0.1, 0.2))
    bpy.context.active_object.data.energy = 5.0


def _import_obj(path: Path, name: str):
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    obj.name = name
    return obj


def _render(out_path: Path, lid_angle_deg: float):
    _clean()
    _setup_scene()

    body = _import_obj(OBJ_DIR / "body.obj", "body")
    lid = _import_obj(OBJ_DIR / "lid.obj", "lid")

    # Lid mesh has its origin already at the hinge. Place it in body frame
    # and rotate around X by the joint angle.
    lid.location = (0, HINGE_Y, HINGE_Z)
    lid.rotation_euler = (radians(lid_angle_deg), 0, 0)

    # Material (matte dark grey to match the Buds 6 case)
    for obj in (body, lid):
        mat = bpy.data.materials.new(name=f"{obj.name}_mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.18, 0.18, 0.20, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.55
        bsdf.inputs["Metallic"].default_value = 0.05
        obj.data.materials.append(mat)

    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    print(f"[render] {out_path}", flush=True)


_render(OUT_DIR / "preview_closed.png", lid_angle_deg=0.0)
_render(OUT_DIR / "preview_open.png", lid_angle_deg=LID_OPEN_DEG)
