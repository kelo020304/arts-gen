"""Render the RAW clean.glb body+lid (no part_scale, no lid_offset) from the
web editor's camera angle. This is the baseline geometry — if this matches
the user's web screenshot, then the web editor isn't applying part_scale or
lid_offset (or the geometry is naturally aligned without them).
"""
import bpy, os, sys, json
from math import radians
from pathlib import Path

LABELS = Path(os.environ["LABELS"]).resolve()
CLEAN_GLB = Path(os.environ["CLEAN_GLB"]).resolve()
OUT = Path(os.environ["OUT"]).resolve()

labels_data = json.loads(LABELS.read_text())
cluster_label = labels_data["labels"]

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(CLEAN_GLB))

groups = {"body": [], "lid": []}
for o in list(bpy.context.scene.objects):
    if o.type != "MESH": continue
    lab = cluster_label.get(o.name, "unlabeled")
    if lab in groups:
        groups[lab].append(o)
    else:
        bpy.data.objects.remove(o, do_unlink=True)

LABEL_COLORS = {"body": (0.0, 1.0, 0.0, 1.0), "lid": (0.0, 0.4, 1.0, 1.0)}
for lab, objs in groups.items():
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1: bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = lab
    mat = bpy.data.materials.new(f"{lab}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = LABEL_COLORS[lab]
    bsdf.inputs["Roughness"].default_value = 0.5
    obj.data.materials.append(mat)

import mathutils
cam_pos = mathutils.Vector((2.5, -2.5, 2.5))   # Blender Z-up
target = mathutils.Vector((0, 0, 0))
bpy.ops.object.camera_add(location=cam_pos)
cam = bpy.context.active_object
direction = target - cam_pos
rot_quat = direction.to_track_quat('-Z', 'Y')
cam.rotation_euler = rot_quat.to_euler()
cam.data.lens_unit = "FOV"
cam.data.angle = radians(45)
bpy.context.scene.camera = cam

bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
bpy.context.active_object.data.energy = 4.0
bpy.ops.object.light_add(type="SUN", location=(-2, 2, 2))
bpy.context.active_object.data.energy = 2.0

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = 800
scene.render.resolution_y = 800
scene.render.filepath = str(OUT)
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print(f"[done raw] -> {OUT}")
