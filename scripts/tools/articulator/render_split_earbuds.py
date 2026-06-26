"""Render earbud_L (red) + earbud_R (blue) from 3 angles, and report size
alignment vs the original case clusters (from clean.glb + labels.json).
"""
import bpy
import json
import os
import sys
from math import radians
from pathlib import Path


SEPARATED = Path(os.environ["SEPARATED"]).resolve()
CLEAN_GLB = Path(os.environ["CLEAN_GLB"]).resolve()
LABELS = Path(os.environ["LABELS"]).resolve()
OUT_DIR = Path(os.environ["OUT_DIR"]).resolve(); OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bbox_of_obj(obj):
    bpy.context.view_layer.update()
    coords = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs, ys, zs = zip(*[(c.x, c.y, c.z) for c in coords])
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    ext = (mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])
    return mn, mx, ext


# 1) Compute original earbud bboxes from clean.glb (cluster-based)
labels = json.loads(LABELS.read_text())["labels"]
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(CLEAN_GLB))
orig_bbox = {"earbud_L": [None, None], "earbud_R": [None, None]}
for o in list(bpy.context.scene.objects):
    if o.type != "MESH": continue
    lab = labels.get(o.name, "?")
    if lab not in orig_bbox: continue
    mn, mx, _ = _bbox_of_obj(o)
    cur_min, cur_max = orig_bbox[lab]
    if cur_min is None:
        orig_bbox[lab] = [list(mn), list(mx)]
    else:
        for i in range(3):
            cur_min[i] = min(cur_min[i], mn[i])
            cur_max[i] = max(cur_max[i], mx[i])

orig_ext = {}
for lab, (mn, mx) in orig_bbox.items():
    orig_ext[lab] = (mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])

# Clear scene before loading separated earbuds
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)


# 2) Load split earbuds, color them, compute new bboxes
LABEL_COLORS = {"earbud_L": (0.95, 0.25, 0.25, 1.0), "earbud_R": (0.25, 0.45, 0.95, 1.0)}
new_objs = {}
for tag in ("earbud_L", "earbud_R"):
    path = SEPARATED / f"{tag}.glb"
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in imported: o.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]
    if len(imported) > 1: bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = tag
    # color material override
    mat = bpy.data.materials.new(f"{tag}_overlay")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = LABEL_COLORS[tag]
    bsdf.inputs["Roughness"].default_value = 0.5
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    new_objs[tag] = obj

# Compute new bboxes (in their own GLB frame)
new_ext = {}
for tag, obj in new_objs.items():
    _, _, ext = _bbox_of_obj(obj)
    new_ext[tag] = ext

# 3) Print size comparison report
print("\n=== SIZE ALIGNMENT REPORT (Y-up frame, units = GLB units) ===")
print(f"{'tag':<10} {'original case extent (X,Y,Z)':<35} {'new earbud extent':<35} {'scale factor (orig/new)':<30}")
for tag in ("earbud_L", "earbud_R"):
    o = orig_ext[tag]; n = new_ext[tag]
    sf = (o[0]/n[0], o[1]/n[1], o[2]/n[2])
    print(f"{tag:<10} ({o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f})        "
          f"({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})        "
          f"({sf[0]:.4f}, {sf[1]:.4f}, {sf[2]:.4f})")

# Suggested uniform scale: pick the largest needed (smallest factor inverse)
# i.e., make new earbud fit within original cluster bbox
suggested_scale = {}
for tag in ("earbud_L", "earbud_R"):
    o = orig_ext[tag]; n = new_ext[tag]
    suggested_scale[tag] = min(o[0]/n[0], o[1]/n[1], o[2]/n[2])
    print(f"  {tag} suggested uniform scale (fit-into-bbox): {suggested_scale[tag]:.4f}")

# Save report
report = {
    "original_case_extents": {k: list(v) for k, v in orig_ext.items()},
    "new_earbud_extents": {k: list(v) for k, v in new_ext.items()},
    "suggested_uniform_scale": suggested_scale,
}
(OUT_DIR / "alignment_report.json").write_text(json.dumps(report, indent=2))


# 4) Render side-by-side preview from 3 angles
import mathutils
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = 800
scene.render.resolution_y = 800
scene.render.film_transparent = False

bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
bpy.context.active_object.data.energy = 4.0
bpy.ops.object.light_add(type="SUN", location=(-2, 2, 2))
bpy.context.active_object.data.energy = 2.0

# Auto-frame: target = midpoint of both earbuds
all_pts = []
for obj in new_objs.values():
    bpy.context.view_layer.update()
    all_pts.extend([obj.matrix_world @ v.co for v in obj.data.vertices])
xs = [p.x for p in all_pts]; ys = [p.y for p in all_pts]; zs = [p.z for p in all_pts]
mid = mathutils.Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2))
extent = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
dist = extent * 2.0

views = {
    "top":   (mid + mathutils.Vector((0, 0, dist)),   mid),
    "front": (mid + mathutils.Vector((0, -dist, 0)),  mid),
    "iso":   (mid + mathutils.Vector((dist, -dist, dist*0.6)), mid),
}
for name, (pos, target) in views.items():
    bpy.ops.object.camera_add(location=pos)
    cam = bpy.context.active_object
    direction = target - pos
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    cam.data.lens = 50
    scene.camera = cam
    scene.render.filepath = str(OUT_DIR / f"split_preview_{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"  rendered {name} view -> {scene.render.filepath}")

print(f"[done] -> {OUT_DIR}")
