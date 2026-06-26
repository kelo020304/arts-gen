"""Render the final USDA from a typical Isaac-Sim-style viewpoint to verify
geometry placement (case + lid + earbuds + ground)."""
import bpy, os, sys
from math import radians
from pathlib import Path
import mathutils

USDA = Path(os.environ["USDA"]).resolve()
OUT = Path(os.environ["OUT"]).resolve()

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Blender's USD importer
bpy.ops.wm.usd_import(filepath=str(USDA))

# Setup scene
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.film_transparent = False
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0.1, 0.1, 0.12, 1.0)

# Lights
bpy.ops.object.light_add(type='SUN', location=(0.1, -0.1, 0.2))
bpy.context.active_object.data.energy = 4.0
bpy.ops.object.light_add(type='SUN', location=(-0.1, 0.1, 0.1))
bpy.context.active_object.data.energy = 2.0

# Camera — case is ~5cm tall in Y-up frame; place camera ~15cm away looking at center
def add_cam(name, pos, target, fov_deg=40):
    bpy.ops.object.camera_add(location=pos)
    cam = bpy.context.active_object
    cam.name = name
    direction = mathutils.Vector(target) - mathutils.Vector(pos)
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens_unit = "FOV"
    cam.data.angle = radians(fov_deg)
    return cam

# Case is in Y-up; centered around (0, 0.025, 0). Earbuds at (~±0.01, 0.025, 0.01)
center = (0, 0.025, 0)
views = [
    ("front", (0.0, 0.025, 0.18), center),
    ("iso",   (0.12, 0.08, 0.12), center),
    ("top",   (0.0, 0.20, 0.0001), center),
    ("side",  (0.18, 0.025, 0.0), center),
    ("back",  (0.0, 0.025, -0.18), center),
    # web editor's exact camera: at (2.5, 2.5, 2.5) looking at origin in
    # GLB-units. Asset is scaled ~0.025x in Isaac Sim, so multiply by that.
    ("web_view", (0.0625, 0.0625 + 0.025, 0.0625), center),
]
for name, pos, tgt in views:
    cam = add_cam(name, pos, tgt)
    scene.camera = cam
    scene.render.filepath = str(OUT / f"final_{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[render] {scene.render.filepath}")
