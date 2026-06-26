"""Render the assembled body+lid meshes from the same camera angle as the web
editor. If this PNG looks like the user's web screenshot, the geometry math
is right. If not, there's a bug in part_scale/lid_offset/coords.
"""
import bpy
import os
import sys
from math import radians
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Reads:
#   $LABELS  - labels.json
#   $CLEAN_GLB - clean.glb
#   $OUT - output png path
LABELS = Path(os.environ["LABELS"]).resolve()
CLEAN_GLB = Path(os.environ["CLEAN_GLB"]).resolve()
OUT = Path(os.environ["OUT"]).resolve()

import json

labels_data = json.loads(LABELS.read_text())
cluster_label = labels_data["labels"]
part_scales = labels_data.get("part_scales", {})
lid_offset = labels_data.get("lid_offset", [0, 0, 0])

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Import GLB (auto-converts to Blender Z-up internal)
bpy.ops.import_scene.gltf(filepath=str(CLEAN_GLB))

# Group meshes by label, keep only body and lid
groups = {"body": [], "lid": []}
for o in list(bpy.context.scene.objects):
    if o.type != "MESH": continue
    lab = cluster_label.get(o.name, "unlabeled")
    if lab in groups:
        groups[lab].append(o)
    else:
        bpy.data.objects.remove(o, do_unlink=True)

# Join each group + apply scale around bbox center + apply offset for lid
LABEL_COLORS = {"body": (0.0, 1.0, 0.0, 1.0), "lid": (0.0, 0.4, 1.0, 1.0)}

for lab, objs in groups.items():
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1: bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = lab

    # Apply part_scale around bbox center (matches web editor + new build_usd)
    s = part_scales.get(lab, [1, 1, 1])
    if s != [1, 1, 1]:
        bpy.context.view_layer.update()
        coords = [obj.matrix_world @ v.co for v in obj.data.vertices]
        # bbox center IN BLENDER FRAME (Z-up). The web editor used Y-up bbox
        # center so we must convert: glTF Y-up bbox -> Blender Z-up bbox is
        # the same point, just expressed in different axes after import.
        # Blender's gltf importer applied Y-up -> Z-up rotation, so the
        # vertices we see now are already in Z-up. Computing bbox here gives
        # the same physical point as web editor's (just in Z-up coords).
        xs = [c.x for c in coords]; ys = [c.y for c in coords]; zs = [c.z for c in coords]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        cz = (min(zs) + max(zs)) / 2
        # Apply scale in glTF-equivalent axes:
        #   web sx -> Blender sx
        #   web sy -> Blender sz (Y-up Y is Z-up Z)
        #   web sz -> Blender sy (Y-up Z is Z-up -Y; sign doesn't matter for scale)
        for v in obj.data.vertices:
            wv = obj.matrix_world @ v.co
            wv.x = s[0]*(wv.x - cx) + cx
            wv.y = s[2]*(wv.y - cy) + cy   # web sz -> Blender sy
            wv.z = s[1]*(wv.z - cz) + cz   # web sy -> Blender sz
            v.co = obj.matrix_world.inverted() @ wv
        print(f"[scale] {lab}: web s={s} -> Blender (sx,sz,sy)=({s[0]},{s[2]},{s[1]}) around ({cx:.3f},{cy:.3f},{cz:.3f})")

    # Apply lid_open_deg rotation around hinge (the user's slider value at export)
    if lab == "lid":
        hinge = labels_data["hinge"]
        p0 = hinge["p0"]; p1 = hinge["p1"]
        # In Blender Z-up after gltf import: glTF (x,y,z) -> Blender (x,-z,y)
        p0_b = (p0[0], -p0[2], p0[1])
        p1_b = (p1[0], -p1[2], p1[1])
        ax = (p1_b[0]-p0_b[0], p1_b[1]-p0_b[1], p1_b[2]-p0_b[2])
        from math import sqrt, sin, cos, radians
        L = sqrt(ax[0]**2 + ax[1]**2 + ax[2]**2)
        au = (ax[0]/L, ax[1]/L, ax[2]/L)
        ang = radians(float(hinge.get("lid_open_deg", 0)))
        if abs(ang) > 1e-3:
            qw = cos(ang/2); qs = sin(ang/2)
            qx, qy, qz = au[0]*qs, au[1]*qs, au[2]*qs
            for v in obj.data.vertices:
                px = v.co.x - p0_b[0]; py = v.co.y - p0_b[1]; pz = v.co.z - p0_b[2]
                cx1 = qy*pz - qz*py; cy1 = qz*px - qx*pz; cz1 = qx*py - qy*px
                ix = cx1 + qw*px; iy = cy1 + qw*py; iz = cz1 + qw*pz
                cx2 = qy*iz - qz*iy; cy2 = qz*ix - qx*iz; cz2 = qx*iy - qy*ix
                v.co.x = px + 2*cx2 + p0_b[0]
                v.co.y = py + 2*cy2 + p0_b[1]
                v.co.z = pz + 2*cz2 + p0_b[2]
            print(f"[rotate] lid: {hinge.get('lid_open_deg')}° around hinge")

    # Apply lid_offset for lid
    if lab == "lid" and any(abs(c) > 1e-6 for c in lid_offset):
        # Y-up offset (ox, oy, oz) -> Blender (ox, -oz, oy)
        ox = lid_offset[0]; oy = -lid_offset[2]; oz = lid_offset[1]
        for v in obj.data.vertices:
            v.co.x += ox
            v.co.y += oy
            v.co.z += oz
        print(f"[offset] lid: web {lid_offset} -> Blender ({ox:.3f},{oy:.3f},{oz:.3f})")

    # Material color for visibility
    mat = bpy.data.materials.new(f"{lab}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = LABEL_COLORS[lab]
    bsdf.inputs["Roughness"].default_value = 0.5
    obj.data.materials.append(mat)

# Camera matching Three.js: position (2.5, 2.5, 2.5), looking at origin, FOV 45.
# Three.js is Y-up; Blender is Z-up. The Y-up camera (2.5, 2.5, 2.5) in glTF
# maps to Blender (2.5, -2.5, 2.5) (Y-up Z=2.5 -> Z-up -Y=2.5 i.e. Y=-2.5).
import mathutils
cam_pos = mathutils.Vector((2.5, -2.5, 2.5))   # Blender Z-up coords
target = mathutils.Vector((0, 0, 0))
bpy.ops.object.camera_add(location=cam_pos)
cam = bpy.context.active_object
direction = target - cam_pos
rot_quat = direction.to_track_quat('-Z', 'Y')
cam.rotation_euler = rot_quat.to_euler()
cam.data.lens_unit = "FOV"
cam.data.angle = radians(45)
bpy.context.scene.camera = cam

# Lights
bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
bpy.context.active_object.data.energy = 4.0
bpy.ops.object.light_add(type="SUN", location=(-2, 2, 2))
bpy.context.active_object.data.energy = 2.0

# Render settings
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = 800
scene.render.resolution_y = 800
scene.render.filepath = str(OUT)
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
bpy.ops.render.render(write_still=True)
print(f"[done] -> {OUT}")
