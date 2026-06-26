"""Generate a parametric proxy for the Xiaomi Buds 6 charging case.

Geometry is a closed-form rounded box (52.34 x 52.57 x 24 mm), bisected at
mid-height into ``body`` (with two ellipsoidal earbud cavities) and ``lid``,
hinged on the back edge along the X axis (matches Mi-PhysX URDF schema where
``axis``/``limit`` are in degrees and lower-bound is negative for the open
state).

Run via Blender 4.4 in background mode::

    software/blender-4.4.0-linux-x64/blender --background \\
        --python scripts/tools/build_xiaomi_buds6.py

Outputs (under ``$OUT_DIR``, default ``outputs/xiaomi_buds6_proxy``):
    objs/body.obj          # closed mesh, body half + carved cavities
    objs/lid.obj           # closed mesh, lid half (origin at hinge)
    objs/body.convex.stl   # convex hull for URDF collision
    objs/lid.convex.stl
    {ASSET_ID}.urdf        # body + revolute joint + lid (axis along X)
    {ASSET_ID}.json        # Mi-PhysX finaljson schema
    part_info.json         # label -> part name map
"""
import bpy
import bmesh
import json
import os
from math import radians
from pathlib import Path


# ---- Config (mm; converted to m for Blender / OBJ / URDF) -----------------
L_MM, W_MM, H_MM = 52.34, 52.57, 24.0    # length(X), width(Y, hinge axis dir), height(Z)
BEVEL_MM = 10.0                           # rounds the vertical corner edges (-> rounded-square footprint)
BEVEL_SEGMENTS = 6
SUBSURF_LEVEL = 3                         # heavy enough to dome the top/bottom; creases keep footprint
CREASE_VERTICAL_EDGES = 0.55              # crease on the 4 vertical bevel-corner edges, keeps sides rect
HINGE_GAP_MM = 0.4                        # body/lid bisect gap so they don't co-plane
CAVITY_RADII_MM = (10.0, 7.0, 6.0)        # earbud well ellipsoid (rx, ry, rz)
CAVITY_OFFSET_MM = 12.0                   # |x| of each cavity center
CAVITY_PROTRUSION_MM = 1.0                # how far cavity sticks above split (gets clipped)

LID_LIMIT_DEG = -110.0                    # open-most angle (negative per Mi-PhysX convention)

OUT_DIR = Path(os.environ.get("OUT_DIR", "outputs/xiaomi_buds6_proxy")).resolve()
ASSET_ID = os.environ.get("ASSET_ID", "xiaomi-buds-6-proxy")
OBJ_DIR = OUT_DIR / "objs"


# ---- Blender helpers ------------------------------------------------------
def _clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _make_pebble(name, size_m, bevel_m, bevel_segments, subsurf_levels):
    """Cube -> heavy bevel -> light Catmull-Clark subsurf -> pebble shape.

    Bevel gives the rounded-square plan view (corners ~bevel_m radius); the
    light subsurf passes turn the still-flat faces into gentle bulges, so the
    object is curved on all sides while keeping a clearly rectangular
    footprint (NOT a sphere/UFO).

    Post-modifier the bounding box shrinks; we measure & rescale to the exact
    target dimensions.
    """
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size_m
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bev = obj.modifiers.new(name="Bevel", type="BEVEL")
    bev.width = bevel_m
    bev.segments = bevel_segments
    bev.profile = 0.7
    bev.limit_method = "ANGLE"
    bev.angle_limit = radians(30)
    _select_only(obj)
    bpy.ops.object.modifier_apply(modifier="Bevel")

    sub = obj.modifiers.new(name="Subsurf", type="SUBSURF")
    sub.subdivision_type = "CATMULL_CLARK"
    sub.levels = subsurf_levels
    sub.render_levels = subsurf_levels
    bpy.ops.object.modifier_apply(modifier="Subsurf")

    bpy.context.view_layer.update()
    coords = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs, ys, zs = zip(*[(c.x, c.y, c.z) for c in coords])
    cur = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    obj.scale = (size_m[0] / cur[0], size_m[1] / cur[1], size_m[2] / cur[2])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def _make_ellipsoid(name, radii_m, location_m):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=48,
                                         ring_count=24, location=location_m)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = radii_m
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def _bisect_keep(obj, plane_co_m, plane_no, keep_above):
    """In-place planar cut, fills the cut, keeps one side."""
    _select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.bisect(
        plane_co=plane_co_m,
        plane_no=plane_no,
        clear_inner=keep_above,        # inner = behind +normal -> below z plane
        clear_outer=not keep_above,    # outer = ahead of +normal -> above z plane
        use_fill=True,
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def _bool_subtract(target, cutter):
    _select_only(target)
    mod = target.modifiers.new(name="bool_sub", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    mod.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier="bool_sub")
    bpy.data.objects.remove(cutter, do_unlink=True)


def _translate(obj, delta_m):
    obj.location = (obj.location[0] + delta_m[0],
                    obj.location[1] + delta_m[1],
                    obj.location[2] + delta_m[2])
    _select_only(obj)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)


def _convex_hull(src_obj, name):
    """Build a convex-hull copy of src_obj for URDF collision."""
    me = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    for v in src_obj.data.vertices:
        bm.verts.new(src_obj.matrix_world @ v.co)
    bm.verts.ensure_lookup_table()
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(me)
    bm.free()
    return obj


def _export_obj(obj, path):
    _select_only(obj)
    bpy.ops.wm.obj_export(
        filepath=str(path),
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        export_materials=False,
    )


def _export_stl(obj, path):
    _select_only(obj)
    bpy.ops.wm.stl_export(filepath=str(path), export_selected_objects=True)


# ---- Build ----------------------------------------------------------------
OBJ_DIR.mkdir(parents=True, exist_ok=True)
_clean_scene()

L = L_MM / 1000.0
W = W_MM / 1000.0
H = H_MM / 1000.0
GAP = HINGE_GAP_MM / 1000.0
CAV_R = tuple(r / 1000.0 for r in CAVITY_RADII_MM)
CAV_OFF = CAVITY_OFFSET_MM / 1000.0
CAV_PROT = CAVITY_PROTRUSION_MM / 1000.0
SPLIT_Z = 0.0  # case is centered at origin; mid-plane = z=0

# Full closed case at origin (pebble shape: bevel + subsurf)
BEVEL = BEVEL_MM / 1000.0
case = _make_pebble("case", (L, W, H), BEVEL, BEVEL_SEGMENTS, SUBSURF_LEVEL)

# Duplicate to derive lid_raw
_select_only(case)
bpy.ops.object.duplicate()
case_copy = bpy.context.active_object
case.name = "body_raw"
case_copy.name = "lid_raw"

# Body = below z = -GAP/2 ; Lid = above z = +GAP/2
_bisect_keep(case, plane_co_m=(0, 0, SPLIT_Z - GAP / 2), plane_no=(0, 0, 1), keep_above=False)
_bisect_keep(case_copy, plane_co_m=(0, 0, SPLIT_Z + GAP / 2), plane_no=(0, 0, 1), keep_above=True)

# Earbud cavities: ellipsoids whose top breaches the body's flat top by CAV_PROT
cav_z = SPLIT_Z - GAP / 2 - CAV_R[2] + CAV_PROT
cav_a = _make_ellipsoid("cav_a", CAV_R, (-CAV_OFF, 0, cav_z))
_bool_subtract(case, cav_a)
cav_b = _make_ellipsoid("cav_b", CAV_R, (+CAV_OFF, 0, cav_z))
_bool_subtract(case, cav_b)

case.name = "body"
case_copy.name = "lid"

# Re-frame lid so its local origin sits at the hinge (back-top edge of body)
HINGE_Y = W / 2.0
HINGE_Z = SPLIT_Z
_translate(case_copy, (0.0, -HINGE_Y, -HINGE_Z))

# Convex hulls (collision)
body_hull = _convex_hull(case, "body_hull")
lid_hull = _convex_hull(case_copy, "lid_hull")

# Export
_export_obj(case, OBJ_DIR / "body.obj")
_export_obj(case_copy, OBJ_DIR / "lid.obj")
_export_stl(body_hull, OBJ_DIR / "body.convex.stl")
_export_stl(lid_hull, OBJ_DIR / "lid.convex.stl")

print(f"[ok] meshes -> {OBJ_DIR}", flush=True)


# ---- URDF (Mi-PhysX schema: limit in degrees, lower<=0=closed) ------------
URDF = """<?xml version="1.0"?>
<robot name="{aid}">
  <link name="body">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="objs/body.obj"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="objs/body.convex.stl"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.025"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <link name="lid">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="objs/lid.obj"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="objs/lid.convex.stl"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.010"/>
      <inertia ixx="0.00005" ixy="0" ixz="0" iyy="0.00005" iyz="0" izz="0.00005"/>
    </inertial>
  </link>

  <joint name="lid_joint" type="revolute">
    <parent link="body"/>
    <child link="lid"/>
    <origin xyz="0 {hy:.9f} {hz:.9f}" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="{lo:.6f}" upper="0.0" effort="1000" velocity="1.0"/>
    <dynamics damping="0.001" friction="0.3"/>
  </joint>
</robot>
"""
(OUT_DIR / f"{ASSET_ID}.urdf").write_text(URDF.format(
    aid=ASSET_ID, hy=HINGE_Y, hz=HINGE_Z, lo=LID_LIMIT_DEG,
))

# ---- finaljson (matches Mi-PhysX raw/finaljson/*.json schema) -------------
finaljson = {
    "object_name": "Xiaomi Buds 6 Charging Case",
    "category": "Earbuds Case",
    "dimension": f"{L_MM/10:.3f}*{W_MM/10:.3f}*{H_MM/10:.3f}",
    "parts": [
        {
            "label": 0, "name": "Body",
            "material": "Plastic", "density": "1.2 g/cm^3",
            "Young's Modulus (GPa)": 2.5, "Poisson's Ratio": 0.35,
            "priority_rank": 1,
            "Basic_description": "This is the body of the earbuds charging case.",
            "Functional_description": "It holds the earbuds and the charging electronics.",
            "Movement_description": "It remains stationary during normal operation.",
            "obj": ["objs/body.obj"],
        },
        {
            "label": 1, "name": "Lid",
            "material": "Plastic", "density": "1.2 g/cm^3",
            "Young's Modulus (GPa)": 2.5, "Poisson's Ratio": 0.35,
            "priority_rank": 2,
            "Basic_description": "This is the flip lid of the earbuds charging case.",
            "Functional_description": "It opens to expose the earbuds and closes to protect them.",
            "Movement_description": "It rotates about the back hinge axis (X).",
            "obj": ["objs/lid.obj"],
        },
    ],
    "group_info": {"0": ["body"], "1": ["lid", "lid_joint", "revolute", "x"]},
}
(OUT_DIR / f"{ASSET_ID}.json").write_text(json.dumps(finaljson, indent=2, ensure_ascii=False))

# ---- part_info.json (label-source-of-truth used by data pipeline) ---------
part_info = {
    "object_id": ASSET_ID,
    "parts": [{"label": 1, "name": "body"}, {"label": 2, "name": "lid"}],
}
(OUT_DIR / "part_info.json").write_text(json.dumps(part_info, indent=2))

print(f"[done] {ASSET_ID} -> {OUT_DIR}", flush=True)
