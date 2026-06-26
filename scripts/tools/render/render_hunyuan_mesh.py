"""Render 4 turntable views of a single mesh (the Hunyuan3D-2 output).

Run::

    software/blender-4.4.0-linux-x64/blender --background \\
        --python scripts/tools/render_hunyuan_mesh.py
"""
import bpy
import os
from math import radians
from pathlib import Path


MESH = Path(os.environ.get(
    "MESH", "outputs/xiaomi_buds6_hunyuan/shape.obj")).resolve()
OUT_DIR = MESH.parent
RES = 512


def _clean():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _setup():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.film_transparent = True

    bpy.ops.object.light_add(type="SUN", location=(0.5, -0.5, 1.0))
    bpy.context.active_object.data.energy = 5.0
    bpy.ops.object.light_add(type="SUN", location=(-0.5, 0.5, 0.5))
    bpy.context.active_object.data.energy = 2.0


def _import_obj(path: Path):
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    # Material
    mat = bpy.data.materials.new("mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.20, 0.20, 0.22, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.45
    bsdf.inputs["Metallic"].default_value = 0.05
    obj.data.materials.append(mat)
    return obj


def _add_camera(angle_deg: float, dist: float, height: float):
    rad = radians(angle_deg)
    import math
    x = dist * math.cos(rad)
    y = dist * math.sin(rad)
    bpy.ops.object.camera_add(location=(x, y, height))
    cam = bpy.context.active_object
    # Aim at origin
    direction = (0 - x, 0 - y, 0 - height)
    import mathutils
    rot_quat = mathutils.Vector(direction).to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    cam.data.lens = 50
    return cam


_clean()
_setup()
obj = _import_obj(MESH)

# Auto-fit: scale & center mesh so it fits in [-0.5, 0.5]^3 for predictable framing
bpy.context.view_layer.update()
verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
xs, ys, zs = zip(*[(v.x, v.y, v.z) for v in verts])
cx, cy, cz = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2
extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
scale = 1.0 / extent
obj.location = (-cx * scale, -cy * scale, -cz * scale)
obj.scale = (scale, scale, scale)

views = {"front": (270, 1.5, 0.3), "side": (0, 1.5, 0.3),
         "iso": (315, 1.5, 0.6), "top": (270, 0.0, 1.5)}
for name, (ang, dist, h) in views.items():
    cam = _add_camera(ang, dist, h)
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = str(OUT_DIR / f"hunyuan_{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[saved] hunyuan_{name}.png", flush=True)
