"""Convert ``clean.glb`` + v2 ``labels.json`` into a physics-ready USDA scene.

Walks the ``parts[]`` / ``joints[]`` schema (see ``schema.py``) and emits one
USDA scene with the appropriate USD physics APIs. Supports any kinematic
tree of revolute / prismatic / fixed joints + free rigid bodies.

Per-rigid-body layout (Visuals / Collisions separation, joint nested under
its child body — matches the convention used by Omniverse-exported assets
like Refrigerator103.usd; see ``scripts/tools/articulator/usd_layout.svg``)::

    World/
      <device>/                            # Xform per articulated component
        <part_a>/                          # Xform + PhysicsRigidBodyAPI
          Visuals/                         #   default purpose, rendered
            mesh                           #     Mesh + MaterialBindingAPI
          Collisions/                      #   purpose="guide", physics-only
            collider_0                     #     Mesh + PhysicsCollisionAPI
        <part_b>/                          # Xform + PhysicsRigidBodyAPI
          Visuals/                         #   (same nested layout)
          Collisions/
          <joint_id>                       #   PhysicsRevoluteJoint nested in child
      <free_part>/                         # standalone body (same V/C layout)

Run (with arts-gen activated)::

    python scripts/tools/articulator/build_usd.py \\
        --clean_glb outputs/xiaomi_buds6_seed3d/clean.glb \\
        --labels    outputs/xiaomi_buds6_seed3d/labels.json \\
        --out       outputs/xiaomi_buds6_usd/xiaomi_buds6.usda
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import joints_by_child, topo_sort, validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BLENDER = PROJECT_ROOT / "software" / "blender-4.4.0-linux-x64" / "blender"


def _schema_point_to_usd_zup(p):
    """Convert a point from the browser/glTF Y-up frame to USD Z-up.

    Three.js labels are stored in the GLB's intrinsic Y-up coordinates.
    Blender imports that GLB into a Z-up world, and ``_export_part_npz`` writes
    Blender world coordinates, so every schema-side point must be mapped before
    it is compared with or emitted alongside mesh vertices.
    """
    return (p[0], -p[2], p[1])


def _schema_vec_to_usd_zup(v):
    """Convert a direction/offset vector from browser/glTF Y-up to USD Z-up."""
    return (v[0], -v[2], v[1])


def _schema_scale_to_usd_zup(s):
    """Convert per-axis schema scale from browser/glTF Y-up to USD Z-up."""
    return (s[0], s[2], s[1])


def _quat_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def _schema_quat_to_usd_zup(q):
    """Convert a schema-frame quaternion to USD Z-up.

    The basis change is a +90 degree rotation around X:
    ``(x, y, z) -> (x, -z, y)``.
    """
    basis = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    return _quat_mul(_quat_mul(basis, q), _quat_conj(basis))


def _schema_aabb_to_usd_zup(amin, amax):
    """Convert a schema-frame AABB to the enclosing USD Z-up AABB."""
    corners = [
        (x, y, z)
        for x in (amin[0], amax[0])
        for y in (amin[1], amax[1])
        for z in (amin[2], amax[2])
    ]
    converted = [_schema_point_to_usd_zup(c) for c in corners]
    return (
        tuple(min(c[i] for c in converted) for i in range(3)),
        tuple(max(c[i] for c in converted) for i in range(3)),
    )


def _apply_stage_transform_point(p, scale, translate):
    s_x, s_y, s_z = scale
    t_x, t_y, t_z = translate
    return (p[0] * s_x + t_x, p[1] * s_y + t_y, p[2] * s_z + t_z)


# ---------------------------------------------------------------------------
# Blender headless: export per-part OBJ from the source GLB.
#
# Receives a JSON config on argv[1] with shape::
#   {
#     "source_glb": "...",
#     "out_dir":    "...",
#     "parts": {part_id: [cluster_ids...], ...},        # from internal cluster GLB
#     "external_meshes": [
#       {"part_id": "...", "glb": "...", "cluster_filter": [...]}
#     ]
#   }
# ---------------------------------------------------------------------------

EXPORT_PARTS_SCRIPT = r'''
import bpy, json, sys, mathutils
from pathlib import Path

cfg = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)


def _split_topo_order(splits):
    """Return split names in dependency order (parent appears before child)."""
    seen, out = set(), []
    def visit(name):
        if name in seen: return
        info = splits.get(name)
        if info: visit(info["parent"])
        seen.add(name)
        if info: out.append(name)
    for n in splits:
        visit(n)
    return out


def _bisect_along_aabb(parent_obj, aabb_min, aabb_max):
    """Slice the mesh along each AABB face that's strictly inside the
    parent's extent. After this, no triangle straddles an AABB boundary —
    centroid-based assignment becomes a clean cut."""
    import bmesh  # noqa: PLC0415
    mesh = parent_obj.data
    px = [v.co.x for v in mesh.vertices]
    py = [v.co.z for v in mesh.vertices]   # Y_up Y == Z_up Z
    pz = [-v.co.y for v in mesh.vertices]  # Y_up Z == -Z_up Y
    pmin = (min(px), min(py), min(pz))
    pmax = (max(px), max(py), max(pz))
    # In Z-up Blender frame, the planes are:
    #   x: plane_no = (1, 0, 0); plane_co = (val_x, 0, 0)
    #   y_yup: plane_no = (0, 0, 1); plane_co = (0, 0, val_y)   ; Y_up Y maps to Z_up Z
    #   z_yup: plane_no = (0, -1, 0); plane_co = (0, -val_z, 0) ; Y_up Z maps to -Z_up Y
    # eps as a fraction of the parent's extent on each axis — guards against
    # numerical-noise cuts at faces that are essentially co-planar with the
    # parent's bounding box (which would just create degenerate slivers).
    extent = (pmax[0] - pmin[0], pmax[1] - pmin[1], pmax[2] - pmin[2])
    eps = [max(1e-4, 0.01 * extent[i]) for i in range(3)]   # 1 % of each axis
    cuts = []
    cut_log = []
    if aabb_min[0] > pmin[0] + eps[0]:
        cuts.append(((aabb_min[0], 0, 0), (1, 0, 0))); cut_log.append(f"x_min={aabb_min[0]:.3f}")
    if aabb_max[0] < pmax[0] - eps[0]:
        cuts.append(((aabb_max[0], 0, 0), (1, 0, 0))); cut_log.append(f"x_max={aabb_max[0]:.3f}")
    if aabb_min[1] > pmin[1] + eps[1]:
        cuts.append(((0, 0, aabb_min[1]), (0, 0, 1))); cut_log.append(f"y_min={aabb_min[1]:.3f}")
    if aabb_max[1] < pmax[1] - eps[1]:
        cuts.append(((0, 0, aabb_max[1]), (0, 0, 1))); cut_log.append(f"y_max={aabb_max[1]:.3f}")
    if aabb_min[2] > pmin[2] + eps[2]:
        cuts.append(((0, -aabb_min[2], 0), (0, -1, 0))); cut_log.append(f"z_min={aabb_min[2]:.3f}")
    if aabb_max[2] < pmax[2] - eps[2]:
        cuts.append(((0, -aabb_max[2], 0), (0, -1, 0))); cut_log.append(f"z_max={aabb_max[2]:.3f}")

    if not cuts:
        return 0
    print(f"[split]   bisect planes: {cut_log}")
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for plane_co, plane_no in cuts:
        bmesh.ops.bisect_plane(bm,
                               geom=bm.faces[:] + bm.edges[:] + bm.verts[:],
                               plane_co=plane_co, plane_no=plane_no,
                               clear_outer=False, clear_inner=False,
                               use_snap_center=False)
    bm.to_mesh(mesh)
    bm.free()
    return len(cuts)


def _apply_split(split_name, info):
    """Re-create a browser box-split on the full-res mesh in Blender. We
    select polygons whose centroid (in world-space coords matching the
    browser's AABB) falls inside the AABB, then mesh.separate(SELECTED).
    Triangles spanning the AABB face are first bisected so the boundary
    cut is clean rather than fuzzy.

    The new object gets renamed to ``split_name``."""
    parent_name = info["parent"]
    parent_obj = bpy.data.objects.get(parent_name)
    if parent_obj is None or parent_obj.type != "MESH":
        print(f"[split] parent '{parent_name}' not found for '{split_name}' — skipping")
        return
    # Skip bmesh bisect — it created sliver triangles at the cut plane that
    # visually appear as "tiny disconnected dots" at the boundary. The pure
    # centroid test produces a fuzzier but cleaner-looking boundary.
    # Original bisect logic is in _bisect_along_aabb if we ever want it back.
    aabb_min = info["aabb_min"]; aabb_max = info["aabb_max"]
    mesh = parent_obj.data
    if len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
        print(f"[split] parent '{parent_name}' is empty (consumed by prior split) — skipping '{split_name}'")
        return
    # Browser AABB is captured from THREE.js geometry.attributes.position
    # (GLB intrinsic Y-up, 1x). Blender's gltf importer keeps vertex.co in
    # 1x and rotates Y-up to Z-up (long axis Y -> Z). To match the browser
    # AABB: use vertex.co (1x), swap Z-up -> Y-up via (x, y, z) -> (x, z, -y).
    inside = []
    for i, poly in enumerate(mesh.polygons):
        c = mathutils.Vector((0, 0, 0))
        for vi in poly.vertices:
            c += mesh.vertices[vi].co
        c = c / len(poly.vertices)
        cx, cy, cz = c.x, c.z, -c.y
        if (aabb_min[0] <= cx <= aabb_max[0]
                and aabb_min[1] <= cy <= aabb_max[1]
                and aabb_min[2] <= cz <= aabb_max[2]):
            inside.append(i)
    # Diagnostic: full XYZ AABB extents + parent mesh bounds so any axis
    # under-coverage is visible (e.g. user's box-rect was too narrow on X).
    print(f"[split] '{split_name}': AABB y_up X[{aabb_min[0]:.3f},{aabb_max[0]:.3f}] "
          f"Y[{aabb_min[1]:.3f},{aabb_max[1]:.3f}] Z[{aabb_min[2]:.3f},{aabb_max[2]:.3f}]")
    parent_xs = [v.co.x for v in mesh.vertices]
    parent_ys = [v.co.z for v in mesh.vertices]   # Z-up Z == Y-up Y
    parent_zs = [-v.co.y for v in mesh.vertices]  # Z-up -Y == Y-up Z
    print(f"[split]   parent '{parent_name}' bounds: X[{min(parent_xs):.3f},{max(parent_xs):.3f}] "
          f"Y[{min(parent_ys):.3f},{max(parent_ys):.3f}] Z[{min(parent_zs):.3f},{max(parent_zs):.3f}]")
    if not inside:
        print(f"[split] '{split_name}': no polys inside AABB — skipping")
        return
    bpy.ops.object.select_all(action="DESELECT")
    parent_obj.select_set(True)
    bpy.context.view_layer.objects.active = parent_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for i in inside:
        mesh.polygons[i].select = True
    before = {o.name for o in bpy.context.scene.objects}
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    new = [o.name for o in bpy.context.scene.objects if o.name not in before and o.type == "MESH"]
    if not new:
        print(f"[split] '{split_name}': separation produced nothing")
        return
    bpy.data.objects[new[0]].name = split_name
    print(f"[split] '{split_name}': {len(inside)} polys split off from '{parent_name}'")


def _export_part_npz(name: str):
    """Pull verts / uvs / faces directly from bpy.context.active_object's
    mesh data and write to <out_dir>/<name>.npz. Skips OBJ entirely —
    OBJ multi-object output had inconsistent global-vs-per-object index
    semantics that produced indices out of range when the merged mesh
    came from a Blender mesh.separate."""
    import numpy as np  # noqa: PLC0415
    obj = bpy.context.active_object
    mesh = obj.data
    mw = obj.matrix_world

    verts = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    for i, v in enumerate(mesh.vertices):
        wv = mw @ v.co
        verts[i] = (wv.x, wv.y, wv.z)

    uv_layer = mesh.uv_layers.active
    has_uv = uv_layer is not None
    if has_uv:
        loop_uvs = np.empty((len(mesh.loops), 2), dtype=np.float32)
        for i, l in enumerate(mesh.loops):
            uv = uv_layer.data[i].uv
            loop_uvs[i] = (uv[0], uv[1])
    else:
        loop_uvs = np.empty((0, 2), dtype=np.float32)

    # Triangulate ngons: for each polygon, fan-triangulate via the loop list
    # so each triangle has its own loop indices (which lookup into loop_uvs).
    tri_v = []          # flat vertex indices, 3 per tri
    tri_loop = []       # flat loop indices (parallel to tri_v); same len
    for poly in mesh.polygons:
        loops = list(poly.loop_indices)
        verts_p = list(poly.vertices)
        for k in range(1, len(loops) - 1):
            tri_v.extend([verts_p[0], verts_p[k], verts_p[k + 1]])
            tri_loop.extend([loops[0], loops[k], loops[k + 1]])
    tri_v = np.asarray(tri_v, dtype=np.int32)
    tri_loop = np.asarray(tri_loop, dtype=np.int32)

    np.savez(out_dir / f"{name}.npz",
             verts=verts, loop_uvs=loop_uvs, tri_v=tri_v, tri_loop=tri_loop)


def _join_cluster_objs(parent_id: str, cluster_ids: list, source_label_for_err: str):
    """Select objects whose name is in cluster_ids, join them, rename to part id, export OBJ."""
    bpy.ops.object.select_all(action="DESELECT")
    found = []
    for o in bpy.context.scene.objects:
        if o.type != "MESH":
            continue
        if o.name in cluster_ids:
            o.select_set(True)
            found.append(o)
    if not found:
        raise SystemExit(f"[error] {source_label_for_err}: no clusters {cluster_ids} found in the imported GLB")
    bpy.context.view_layer.objects.active = found[0]
    if len(found) > 1:
        bpy.ops.object.join()
    merged = bpy.context.active_object
    merged.name = parent_id
    _export_part_npz(parent_id)
    print(f"[export] {parent_id}: joined {len(found)} clusters from {source_label_for_err}")


# Stage A: import source GLB once, export each part that comes from it.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=cfg["source_glb"])

# Replay browser-side box-splits on the full-res mesh BEFORE we look up
# clusters by name. After this runs, the cluster_NN_splitM names exist as
# real objects in the scene, indistinguishable from DBSCAN-produced clusters.
splits = cfg.get("split_clusters", {})
for split_name in _split_topo_order(splits):
    _apply_split(split_name, splits[split_name])

# Determine which parts come from the source GLB vs from external meshes.
ext_attach = {em["part_id"] for em in cfg.get("external_meshes", [])}

for part_id, cluster_ids in cfg["parts"].items():
    if part_id in ext_attach:
        # Geometry overridden by external mesh; skip cluster-based export.
        print(f"[export] {part_id}: SKIPPED (overridden by external mesh)")
        continue
    _join_cluster_objs(part_id, cluster_ids, source_label_for_err="source_glb")

# Stage B: per-external-mesh, import that GLB and export the matching subset.
for em in cfg.get("external_meshes", []):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=em["glb"])
    cluster_ids = em.get("cluster_filter") or [o.name for o in bpy.context.scene.objects if o.type == "MESH"]
    _join_cluster_objs(em["part_id"], cluster_ids, source_label_for_err=f"external:{em['glb']}")
'''


# ---------------------------------------------------------------------------
# OBJ + USDA formatting helpers
# ---------------------------------------------------------------------------


def _read_part_npz(path: Path):
    """Return (verts, faces, uvs, face_uv_idx) from a Blender-produced
    npz. Faces are all triangles (3 verts each). face_uv_idx is the flat
    per-corner UV index list; empty if the npz had no UVs."""
    import numpy as np
    data = np.load(path)
    verts_arr = data["verts"]              # (N, 3)
    loop_uvs = data["loop_uvs"]            # (M, 2) or empty
    tri_v = data["tri_v"]                  # (3T,)
    tri_loop = data["tri_loop"]            # (3T,)

    verts = [(float(v[0]), float(v[1]), float(v[2])) for v in verts_arr]
    n_tri = len(tri_v) // 3
    faces = [[int(tri_v[i * 3]), int(tri_v[i * 3 + 1]), int(tri_v[i * 3 + 2])]
             for i in range(n_tri)]

    if len(loop_uvs):
        uvs = [(float(u[0]), float(u[1])) for u in loop_uvs]
        face_uv_idx = [int(x) for x in tri_loop.tolist()]
    else:
        uvs, face_uv_idx = [], []
    return verts, faces, uvs, face_uv_idx


def _format_points(verts, scale, translate):
    s_x, s_y, s_z = scale
    t_x, t_y, t_z = translate
    return "[" + ", ".join(
        f"({v[0]*s_x + t_x:.6f}, {v[1]*s_y + t_y:.6f}, {v[2]*s_z + t_z:.6f})" for v in verts
    ) + "]"


def _format_face_counts(faces):
    return "[" + ", ".join(str(len(f)) for f in faces) + "]"


def _format_face_indices(faces):
    out = []
    for f in faces:
        out.extend(str(i) for i in f)
    return "[" + ", ".join(out) + "]"


def _format_uvs(uvs):
    return "[" + ", ".join(f"({u:.6f}, {v:.6f})" for u, v in uvs) + "]"


def _format_int_list(idx_list):
    return "[" + ", ".join(str(i) for i in idx_list) + "]"


def _bbox_extent(verts, scale, translate):
    s_x, s_y, s_z = scale
    t_x, t_y, t_z = translate
    xs = [v[0] * s_x + t_x for v in verts]
    ys = [v[1] * s_y + t_y for v in verts]
    zs = [v[2] * s_z + t_z for v in verts]
    return f"[({min(xs):.6f}, {min(ys):.6f}, {min(zs):.6f}), ({max(xs):.6f}, {max(ys):.6f}, {max(zs):.6f})]"


# ---------------------------------------------------------------------------
# Quaternion / rotation math (stdlib only)
# ---------------------------------------------------------------------------


def _quat_align_x_to(axis):
    """Quaternion (w, x, y, z) rotating world +X to ``axis`` (3-tuple, unit)."""
    ax, ay, az = axis
    cx, cy, cz = 0.0, -az, ay
    dot = ax
    angle = math.acos(max(-1.0, min(1.0, dot)))
    if abs(math.sin(angle)) < 1e-9:
        return (1.0, 0.0, 0.0, 0.0) if dot > 0 else (0.0, 0.0, 0.0, 1.0)
    norm = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx, cy, cz = cx / norm, cy / norm, cz / norm
    half = angle / 2
    s = math.sin(half)
    return (math.cos(half), cx * s, cy * s, cz * s)


def _quat_mul(a, b):
    """(w,x,y,z) ⊗ (w,x,y,z) → (w,x,y,z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate_vec(v, q_wxyz):
    qw, qx, qy, qz = q_wxyz
    vx, vy, vz = v
    cx1 = qy * vz - qz * vy
    cy1 = qz * vx - qx * vz
    cz1 = qx * vy - qy * vx
    ix = cx1 + qw * vx
    iy = cy1 + qw * vy
    iz = cz1 + qw * vz
    cx2 = qy * iz - qz * iy
    cy2 = qz * ix - qx * iz
    cz2 = qx * iy - qy * ix
    return (vx + 2 * cx2, vy + 2 * cy2, vz + 2 * cz2)


def _rot_around_axis_pivot(verts, p0, p1, angle_deg):
    """Rotate ``verts`` by ``angle_deg`` around the line through p0 → p1."""
    if abs(angle_deg) < 1e-9:
        return verts
    axis = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    L = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    ax = (axis[0] / L, axis[1] / L, axis[2] / L)
    a = math.radians(angle_deg)
    qw = math.cos(a / 2)
    s = math.sin(a / 2)
    q = (qw, ax[0] * s, ax[1] * s, ax[2] * s)
    out = []
    for v in verts:
        rel = (v[0] - p0[0], v[1] - p0[1], v[2] - p0[2])
        rr = _quat_rotate_vec(rel, q)
        out.append((rr[0] + p0[0], rr[1] + p0[1], rr[2] + p0[2]))
    return out


# ---------------------------------------------------------------------------
# Per-part pipeline: scale -> external transform -> joint bake/offset
# ---------------------------------------------------------------------------


def _apply_part_scale(mesh, scale_xyz):
    """Apply non-uniform scale around the part's bbox center (matches web editor)."""
    if scale_xyz == [1.0, 1.0, 1.0] or scale_xyz == [1, 1, 1]:
        return mesh
    verts, faces, uvs, fuvi = mesh
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    sx, sy, sz = scale_xyz
    new = [(sx * (v[0] - cx) + cx, sy * (v[1] - cy) + cy, sz * (v[2] - cz) + cz) for v in verts]
    return new, faces, uvs, fuvi


def _apply_external_transform(mesh, transform):
    """Bake T*R*S into vertex coords.

    The transform is authored in schema/browser Y-up coordinates, while the
    mesh was exported from Blender in USD Z-up coordinates.
    """
    verts, faces, uvs, fuvi = mesh
    t = _schema_vec_to_usd_zup(transform["t"])
    q = _schema_quat_to_usd_zup(tuple(transform["q_wxyz"]))
    s = _schema_scale_to_usd_zup(transform["s"])
    new = []
    for v in verts:
        vs = (v[0] * s[0], v[1] * s[1], v[2] * s[2])
        vr = _quat_rotate_vec(vs, q)
        new.append((vr[0] + t[0], vr[1] + t[1], vr[2] + t[2]))
    return new, faces, uvs, fuvi


def _apply_joint_bakes_and_offsets(part_meshes, labels):
    """For each joint with a non-zero bake / offset, mutate the child part's verts.

    Walks in topological order so a parent's bake propagates correctly through
    nested joints (B's bake is applied AFTER A's bake for chain root→A→B)."""
    by_child = joints_by_child(labels)
    for pid in topo_sort(labels):
        j = by_child.get(pid)
        if j is None:
            continue  # root or free part — nothing to bake
        verts, faces, uvs, fuvi = part_meshes[pid]
        if j["type"] == "revolute":
            ang = float(j.get("bake_angle", 0))
            if abs(ang) > 1e-6:
                p0 = _schema_point_to_usd_zup(j["axis_p0"])
                p1 = _schema_point_to_usd_zup(j["axis_p1"])
                verts = _rot_around_axis_pivot(verts, p0, p1, ang)
                print(f"  bake-rotate {pid}: {ang:+.2f}° around hinge {j['id']}")
        elif j["type"] == "prismatic":
            d = float(j.get("bake_distance", 0))
            if abs(d) > 1e-6:
                ad = _schema_vec_to_usd_zup(j["axis_dir"])
                L = math.sqrt(sum(c * c for c in ad)) or 1.0
                ax = (ad[0] / L, ad[1] / L, ad[2] / L)
                verts = [(v[0] + d * ax[0], v[1] + d * ax[1], v[2] + d * ax[2]) for v in verts]
                print(f"  bake-translate {pid}: {d:+.4f}m along {j['id']}")
        offset = j.get("offset")
        if offset and any(abs(c) > 1e-6 for c in offset):
            ox, oy, oz = _schema_vec_to_usd_zup(offset)
            verts = [(v[0] + ox, v[1] + oy, v[2] + oz) for v in verts]
            print(f"  bake-offset {pid}: {offset}")
        part_meshes[pid] = (verts, faces, uvs, fuvi)


# ---------------------------------------------------------------------------
# Connected components (parts grouped by joint edges → one Xform per group)
# ---------------------------------------------------------------------------


def _connected_components(labels):
    """Union-find over parts: any two parts joined by a non-free joint share
    the same component. Returns a dict {part_id: component_index}."""
    parent = {p["id"]: p["id"] for p in labels["parts"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for j in labels.get("joints", []):
        if j["type"] == "free":
            continue
        union(j["parent"], j["child"])

    # Map each part id → integer component index in the order roots appear.
    comp_of: dict[str, int] = {}
    next_idx = 0
    for p in labels["parts"]:
        root = find(p["id"])
        if root not in comp_of:
            comp_of[root] = next_idx
            next_idx += 1
    return {pid: comp_of[find(pid)] for pid in parent}, next_idx


def _component_root(labels, comp_id, comp_of):
    """Return the part id that has no incoming joint within its component
    (i.e. the kinematic root). For singleton components this is the only part."""
    by_child = joints_by_child(labels)
    members = [p["id"] for p in labels["parts"] if comp_of[p["id"]] == comp_id]
    for pid in members:
        if pid not in by_child:
            return pid
    # Should never happen if the schema validated (no cycles); fall back to first.
    return members[0]


# ---------------------------------------------------------------------------
# USDA emitters
# ---------------------------------------------------------------------------


def _emit_header(device_name: str, with_ground: bool) -> str:
    out = [f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "{device_name} — generated by scripts/tools/articulator/build_usd.py"
)

def Xform "World"
{{
    def PhysicsScene "physicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}
"""]
    if with_ground:
        # Mesh-based Z-up ground (8 m x 8 m x 0.1 m thick slab). Static collider
        # via PhysicsCollisionAPI + PhysicsMeshCollisionAPI + explicit
        # convexHull approximation. IsaacSim's PhysX consistently picks
        # this up — Cube + bare PhysicsCollisionAPI silently drops the
        # API in some scenes.
        out.append("""
    def Mesh "Ground" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        point3f[] points = [
            (-4, -4, 0), (4, -4, 0), (4, 4, 0), (-4, 4, 0),
            (-4, -4, -0.1), (4, -4, -0.1), (4, 4, -0.1), (-4, 4, -0.1)
        ]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [
            0, 1, 2, 3,
            7, 6, 5, 4,
            0, 4, 5, 1,
            1, 5, 6, 2,
            2, 6, 7, 3,
            3, 7, 4, 0
        ]
        color3f[] primvars:displayColor = [(0.55, 0.55, 0.6)]
        bool physics:collisionEnabled = 1
        uniform token physics:approximation = "convexHull"
    }
""")
    return "".join(out)


def _emit_footer() -> str:
    return "}\n"


def _collision_api_schemas(approx: str) -> list[str]:
    """USD apiSchemas to add for a given collision approximation."""
    schemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    if approx == "sdf":
        schemas.append("PhysxSDFMeshCollisionAPI")
    elif approx == "convexDecomposition":
        schemas.append("PhysxConvexDecompositionCollisionAPI")
    # convexHull / none → no extra physx-specific API needed
    return schemas


def _filtered_pairs_block(prim_paths: list[str], indent: str) -> str:
    if not prim_paths:
        return ""
    if len(prim_paths) == 1:
        return f"{indent}rel physics:filteredPairs = {prim_paths[0]}\n"
    body = ",\n".join(f"{indent}    {p}" for p in prim_paths)
    return f"{indent}rel physics:filteredPairs = [\n{body},\n{indent}]\n"


def _emit_geometry_attrs(mesh, scale, translate, indent: str, *,
                         with_uvs: bool, material_path: str | None) -> str:
    """Inner attrs of a Mesh prim: points, face counts/indices, extent,
    optionally UVs and material binding. Caller wraps with the `def Mesh`
    header + apiSchemas + closing brace."""
    verts, faces, uvs, fuvi = mesh
    out = [
        f"{indent}uniform token subdivisionScheme = \"none\"\n",
        f"{indent}point3f[] points = {_format_points(verts, scale, translate)}\n",
        f"{indent}int[] faceVertexCounts = {_format_face_counts(faces)}\n",
        f"{indent}int[] faceVertexIndices = {_format_face_indices(faces)}\n",
        f"{indent}float3[] extent = {_bbox_extent(verts, scale, translate)}\n",
    ]
    if with_uvs and uvs:
        out.append(f"{indent}texCoord2f[] primvars:st = {_format_uvs(uvs)} (interpolation = \"faceVarying\")\n")
        out.append(f"{indent}int[] primvars:st:indices = {_format_int_list(fuvi)}\n")
    if material_path is not None:
        out.append(f"{indent}rel material:binding = <{material_path}>\n")
    return "".join(out)


def _emit_part_body(part: dict, mesh, scale, translate, prim_path: str,
                    parent_path: str, material_path: str,
                    filtered_pair_paths: list[str],
                    joint_block: str = "",
                    override_materials_out: list[str] | None = None) -> str:
    """Emit ``def Xform "<part_id>"`` containing ``Visuals/`` + ``Collisions/``
    plus an optionally nested joint prim (passed in as a fully-formatted block).

    The Xform is the rigid body — all body-level APIs (RigidBody, Mass,
    FilteredPairs) and attributes (kinematicEnabled, mass, CCD) live here.
    The visual mesh inside ``Visuals/`` carries ``MaterialBindingAPI``;
    the collider inside ``Collisions/`` carries ``PhysicsCollisionAPI`` plus
    the approximation-specific Physx APIs. Renderer skips ``Collisions/`` via
    ``purpose="guide"``."""
    coll_approx = part["collision"]["approx"]
    physics = part["physics"]

    body_apis = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    if filtered_pair_paths:
        body_apis.append("PhysicsFilteredPairsAPI")

    body_indent = "    " * (prim_path.count("/") - 1) if parent_path else "    "
    body_inner = "    " * (prim_path.count("/"))
    nest1 = body_inner + "    "       # inside Visuals/Collisions Xform
    nest2 = nest1 + "    "             # inside the inner Mesh prim

    body_api_str = ", ".join(f'"{s}"' for s in body_apis)
    out = []
    out.append(f"{body_indent}def Xform \"{part['id']}\" (\n")
    out.append(f"{body_indent}    prepend apiSchemas = [{body_api_str}]\n")
    out.append(f"{body_indent})\n")
    out.append(f"{body_indent}{{\n")
    if physics == "kinematic":
        out.append(f"{body_inner}bool physics:kinematicEnabled = 1\n")
    if "mass" in part:
        out.append(f"{body_inner}float physics:mass = {part['mass']}\n")
    if physics == "dynamic":
        out.append(f"{body_inner}bool physxRigidBody:enableCCD = 1\n")
    out.append(_filtered_pairs_block(filtered_pair_paths, body_inner))

    # Visuals/ — render-only mesh with material binding (+ optional screen overlays)
    out.append(f"\n{body_inner}def Xform \"Visuals\"\n")
    out.append(f"{body_inner}{{\n")
    out.append(f"{nest1}def Mesh \"mesh\" (\n")
    out.append(f"{nest1}    prepend apiSchemas = [\"MaterialBindingAPI\"]\n")
    out.append(f"{nest1})\n")
    out.append(f"{nest1}{{\n")
    out.append(f"{nest2}bool doubleSided = 1\n")
    out.append(_emit_geometry_attrs(mesh, scale, translate, nest2,
                                    with_uvs=True, material_path=material_path))
    out.append(f"{nest1}}}\n")
    out.append(_emit_site_overlays(part, scale, translate, nest1, nest2,
                                   override_materials_out))
    out.append(f"{body_inner}}}\n")

    # Collisions/ — purpose="guide" so renderer skips it; PhysX still uses it
    if coll_approx != "none":
        col_apis = _collision_api_schemas(coll_approx)
        col_api_str = ", ".join(f'"{s}"' for s in col_apis)
        out.append(f"\n{body_inner}def Xform \"Collisions\"\n")
        out.append(f"{body_inner}{{\n")
        out.append(f"{nest1}uniform token purpose = \"guide\"\n")
        out.append(f"{nest1}def Mesh \"collider_0\" (\n")
        out.append(f"{nest1}    prepend apiSchemas = [{col_api_str}]\n")
        out.append(f"{nest1})\n")
        out.append(f"{nest1}{{\n")
        out.append(_emit_geometry_attrs(mesh, scale, translate, nest2,
                                        with_uvs=False, material_path=None))
        out.append(f"{nest2}uniform token physics:approximation = \"{coll_approx}\"\n")
        if coll_approx == "sdf":
            out.append(f"{nest2}int physxSDFMeshCollision:sdfResolution = {part['collision']['resolution']}\n")
        out.append(f"{nest1}}}\n")
        out.append(f"{body_inner}}}\n")

    sites_block = _emit_sites_block(part, scale, translate, body_inner, nest1, nest2)
    if sites_block:
        out.append(sites_block)

    if joint_block:
        out.append("\n")
        out.append(joint_block)

    out.append(f"{body_indent}}}\n\n")
    return "".join(out)


def _emit_site_overlays(part: dict, scale, translate,
                        nest1: str, nest2: str,
                        override_materials_out: list[str] | None) -> str:
    """For *any* site with a ``material_override``, emit a visible thin Cube
    inside ``Visuals/`` (slab matching the site AABB, inflated 0.8mm along
    the slab's thin axis to clear z-fighting with the underlying part mesh)
    bound to a freshly-defined override material.

    The site's ``kind`` is purely semantic — the override mechanism is
    generic so the tool stays a domain-agnostic annotator (paint a screen
    black, paint a button red, paint a handle rubber-dark; the overlay
    machinery is identical). See ``scripts/tools/articulator/usd_layout.svg``."""
    out: list[str] = []
    sites = part.get("sites", [])
    for si, site in enumerate(sites):
        mo = site.get("material_override")
        if not mo:
            continue
        amin, amax = _schema_aabb_to_usd_zup(site["aabb_min"], site["aabb_max"])
        wmin = list(_apply_stage_transform_point(amin, scale, translate))
        wmax = list(_apply_stage_transform_point(amax, scale, translate))
        ext = [wmax[i] - wmin[i] for i in range(3)]
        thin = ext.index(min(ext))
        lift = 0.0008
        wmin[thin] -= lift; wmax[thin] += lift
        cx = (wmin[0] + wmax[0]) / 2
        cy = (wmin[1] + wmax[1]) / 2
        cz = (wmin[2] + wmax[2]) / 2
        ex = (wmax[0] - wmin[0]) / 2
        ey = (wmax[1] - wmin[1]) / 2
        ez = (wmax[2] - wmin[2]) / 2

        mat_name = f"{part['id']}_{site['id']}_OverrideMat"
        mat_path = f"/World/Materials/{mat_name}"
        ov_name = f"{site['id']}_overlay"
        col = mo["diffuseColor"]
        rough = mo.get("roughness", 0.15)
        metal = mo.get("metallic", 0.0)
        if override_materials_out is not None:
            override_materials_out.append(_format_override_material(mat_name, mat_path, col, rough, metal))

        out.append(f"{nest1}def Cube \"{ov_name}\" (\n")
        out.append(f"{nest1}    prepend apiSchemas = [\"MaterialBindingAPI\"]\n")
        out.append(f"{nest1})\n")
        out.append(f"{nest1}{{\n")
        out.append(f"{nest2}double size = 2\n")
        out.append(f"{nest2}float3 xformOp:scale = ({ex:.6f}, {ey:.6f}, {ez:.6f})\n")
        out.append(f"{nest2}double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})\n")
        out.append(f"{nest2}uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]\n")
        out.append(f"{nest2}rel material:binding = <{mat_path}>\n")
        out.append(f"{nest1}}}\n")
    return "".join(out)


def _format_override_material(name: str, mat_path: str,
                              diffuse: list[float], rough: float, metal: float) -> str:
    """A standalone UsdPreviewSurface block, spliced into the
    /World/Materials/ Scope by build_usda()."""
    r, g, b = diffuse
    return f"""        def Material "{name}"
        {{
            token outputs:surface.connect = <{mat_path}/PreviewSurface.outputs:surface>
            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({r:.4f}, {g:.4f}, {b:.4f})
                float inputs:roughness = {rough:.4f}
                float inputs:metallic = {metal:.4f}
                token outputs:surface
            }}
        }}
"""


def _emit_sites_block(part: dict, scale, translate,
                      body_inner: str, nest1: str, nest2: str) -> str:
    """``Sites/`` subtree — invisible Cubes for each schema site, scaled to
    its AABB. Site coords are stored in browser/glTF Y-up frame (same as
    split_clusters), so we convert to USD Z-up before applying Stage 5."""
    sites = part.get("sites", [])
    if not sites:
        return ""
    out = [
        f"\n{body_inner}def Xform \"Sites\"\n",
        f"{body_inner}{{\n",
        f"{nest1}uniform token purpose = \"guide\"\n",
    ]
    for site in sites:
        amin, amax = _schema_aabb_to_usd_zup(site["aabb_min"], site["aabb_max"])
        wmin = _apply_stage_transform_point(amin, scale, translate)
        wmax = _apply_stage_transform_point(amax, scale, translate)
        cx = (wmin[0] + wmax[0]) / 2
        cy = (wmin[1] + wmax[1]) / 2
        cz = (wmin[2] + wmax[2]) / 2
        # USD Cube has implicit size=2; scale so the prim spans the AABB.
        ex = (wmax[0] - wmin[0]) / 2
        ey = (wmax[1] - wmin[1]) / 2
        ez = (wmax[2] - wmin[2]) / 2
        out.append(f"{nest1}def Cube \"{site['id']}\"\n")
        out.append(f"{nest1}{{\n")
        out.append(f"{nest2}token visibility = \"invisible\"\n")
        out.append(f"{nest2}double size = 2\n")
        out.append(f"{nest2}custom string userProperties:siteKind = \"{site['kind']}\"\n")
        out.append(f"{nest2}float3 xformOp:scale = ({ex:.6f}, {ey:.6f}, {ez:.6f})\n")
        out.append(f"{nest2}double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})\n")
        out.append(f"{nest2}uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]\n")
        out.append(f"{nest1}}}\n")
    out.append(f"{body_inner}}}\n")
    return "".join(out)


def _emit_revolute_joint(j: dict, body0_path: str, body1_path: str, indent: str,
                         scale=(1, 1, 1), translate=(0, 0, 0)) -> str:
    """Emit a revolute joint in the same USD Z-up frame as the mesh."""
    raw_p0 = _schema_point_to_usd_zup(j["axis_p0"])
    raw_p1 = _schema_point_to_usd_zup(j["axis_p1"])
    p0 = _apply_stage_transform_point(raw_p0, scale, translate)
    p1 = _apply_stage_transform_point(raw_p1, scale, translate)
    axis = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    L = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    axis_unit = (axis[0] / L, axis[1] / L, axis[2] / L)
    quat0 = _quat_align_x_to(axis_unit)

    bake = float(j.get("bake_angle", 0))
    half = math.radians(bake) / 2.0
    rx = (math.cos(half), math.sin(half), 0.0, 0.0)
    quat1 = _quat_mul(quat0, rx)

    drive = j.get("drive") or {}
    target = float(drive.get("target", 0))
    stiffness = float(drive.get("stiffness", 0))
    damping = float(drive.get("damping", 0))
    max_force = float(drive.get("max_force", 0))
    apis = ["PhysicsDriveAPI:angular"]
    if j.get("limit_hard", True):
        apis.append("PhysxLimitAPI:angular")
    api_str = ", ".join(f'"{a}"' for a in apis)

    inner = indent + "    "
    out = [f"{indent}def PhysicsRevoluteJoint \"{j['id']}\" (\n"
           f"{indent}    prepend apiSchemas = [{api_str}]\n"
           f"{indent})\n"
           f"{indent}{{\n"
           f"{inner}rel physics:body0 = <{body0_path}>\n"
           f"{inner}rel physics:body1 = <{body1_path}>\n"
           f"{inner}point3f physics:localPos0 = ({p0[0]:.6f}, {p0[1]:.6f}, {p0[2]:.6f})\n"
           f"{inner}point3f physics:localPos1 = ({p0[0]:.6f}, {p0[1]:.6f}, {p0[2]:.6f})\n"
           f"{inner}quatf physics:localRot0 = ({quat0[0]:.6f}, {quat0[1]:.6f}, {quat0[2]:.6f}, {quat0[3]:.6f})\n"
           f"{inner}quatf physics:localRot1 = ({quat1[0]:.6f}, {quat1[1]:.6f}, {quat1[2]:.6f}, {quat1[3]:.6f})\n"
           f"{inner}uniform token physics:axis = \"X\"\n"
           f"{inner}float physics:lowerLimit = {float(j['lower']):.3f}\n"
           f"{inner}float physics:upperLimit = {float(j['upper']):.3f}\n"]
    if j.get("limit_hard", True):
        out.extend([
            f"{inner}# Hard angular limit (constraint-based, NOT spring): stiffness=0/damping=0\n"
            f"{inner}# tells PhysX to use solver constraints instead of soft springs.\n",
            f"{inner}float physxLimit:angular:stiffness = 0.0\n",
            f"{inner}float physxLimit:angular:damping = 0.0\n",
            f"{inner}float physxLimit:angular:restitution = 0.0\n",
            f"{inner}float physxLimit:angular:contactDistance = 0.1\n",
        ])
    out.extend([
        f"{inner}uniform token drive:angular:physics:type = \"force\"\n",
        f"{inner}float drive:angular:physics:targetPosition = {target:.3f}\n",
        f"{inner}float drive:angular:physics:stiffness = {stiffness}\n",
        f"{inner}float drive:angular:physics:damping = {damping}\n",
        f"{inner}float drive:angular:physics:maxForce = {max_force}\n",
        f"{indent}}}\n\n",
    ])
    return "".join(out)


def _emit_prismatic_joint(j: dict, body0_path: str, body1_path: str, indent: str,
                          scale=(1, 1, 1), translate=(0, 0, 0)) -> str:
    """Prismatic: USD physics axis defaults to X; we orient the joint frame
    via localRot0/1 so its local +X aligns with ``axis_dir`` in world.
    Schema coords are converted from browser/glTF Y-up to USD Z-up before
    applying the Stage-5 transform."""
    s_x = scale[0]
    ad = _schema_vec_to_usd_zup(j["axis_dir"])
    L = math.sqrt(sum(c * c for c in ad)) or 1.0
    axis_unit = (ad[0] / L, ad[1] / L, ad[2] / L)
    quat = _quat_align_x_to(axis_unit)
    raw_o = _schema_point_to_usd_zup(j["axis_origin"])
    o = _apply_stage_transform_point(raw_o, scale, translate)
    # prismatic limits are in meters → also scale by uniform factor (use s_x;
    # uniform scale means s_x == s_y == s_z in our pipeline)
    lower_m = float(j["lower"]) * s_x
    upper_m = float(j["upper"]) * s_x
    drive = j.get("drive") or {}
    target = float(drive.get("target", 0))
    stiffness = float(drive.get("stiffness", 0))
    damping = float(drive.get("damping", 0))
    max_force = float(drive.get("max_force", 0))
    apis = ["PhysicsDriveAPI:linear"]
    if j.get("limit_hard", True):
        apis.append("PhysxLimitAPI:linear")
    api_str = ", ".join(f'"{a}"' for a in apis)

    inner = indent + "    "
    out = [f"{indent}def PhysicsPrismaticJoint \"{j['id']}\" (\n"
           f"{indent}    prepend apiSchemas = [{api_str}]\n"
           f"{indent})\n"
           f"{indent}{{\n"
           f"{inner}rel physics:body0 = <{body0_path}>\n"
           f"{inner}rel physics:body1 = <{body1_path}>\n"
           f"{inner}point3f physics:localPos0 = ({o[0]:.6f}, {o[1]:.6f}, {o[2]:.6f})\n"
           f"{inner}point3f physics:localPos1 = ({o[0]:.6f}, {o[1]:.6f}, {o[2]:.6f})\n"
           f"{inner}quatf physics:localRot0 = ({quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f})\n"
           f"{inner}quatf physics:localRot1 = ({quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f})\n"
           f"{inner}uniform token physics:axis = \"X\"\n"
           f"{inner}float physics:lowerLimit = {lower_m:.6f}\n"
           f"{inner}float physics:upperLimit = {upper_m:.6f}\n"]
    if j.get("limit_hard", True):
        out.extend([
            f"{inner}float physxLimit:linear:stiffness = 0.0\n",
            f"{inner}float physxLimit:linear:damping = 0.0\n",
            f"{inner}float physxLimit:linear:restitution = 0.0\n",
            f"{inner}float physxLimit:linear:contactDistance = 0.001\n",
        ])
    out.extend([
        f"{inner}uniform token drive:linear:physics:type = \"force\"\n",
        f"{inner}float drive:linear:physics:targetPosition = {target}\n",
        f"{inner}float drive:linear:physics:stiffness = {stiffness}\n",
        f"{inner}float drive:linear:physics:damping = {damping}\n",
        f"{inner}float drive:linear:physics:maxForce = {max_force}\n",
        f"{indent}}}\n\n",
    ])
    return "".join(out)


def _emit_fixed_joint(j: dict, body0_path: str, body1_path: str, indent: str,
                      scale=(1, 1, 1), translate=(0, 0, 0)) -> str:
    inner = indent + "    "
    return (
        f"{indent}def PhysicsFixedJoint \"{j['id']}\"\n"
        f"{indent}{{\n"
        f"{inner}rel physics:body0 = <{body0_path}>\n"
        f"{inner}rel physics:body1 = <{body1_path}>\n"
        f"{indent}}}\n\n"
    )


def _emit_free_joint(j, body0_path, body1_path, indent, scale=(1, 1, 1), translate=(0, 0, 0)) -> str:
    return ""  # free bodies have no joint prim


JOINT_EMITTERS = {
    "revolute": _emit_revolute_joint,
    "prismatic": _emit_prismatic_joint,
    "fixed": _emit_fixed_joint,
    "free": _emit_free_joint,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_filtered_pairs(part_id: str, joint: dict | None, part: dict,
                            prim_path_of: dict[str, str], parts_by_id: dict[str, dict]) -> list[str]:
    """Combine filtered_pairs from:
      - part.filtered_pairs (explicit per-part list)
      - joint.filtered_pairs (incoming joint's list)
      - part.collision_group: every other part with the same group is added
        automatically. Lets users say "main + fold are one collision group,
        body shell is another" without listing every pair manually."""
    ids: list[str] = []
    for src in (part.get("filtered_pairs"), joint.get("filtered_pairs") if joint else None):
        if src:
            ids.extend(src)
    group = part.get("collision_group")
    if group:
        for other_id, other in parts_by_id.items():
            if other_id != part_id and other.get("collision_group") == group:
                ids.append(other_id)
    return [f"<{prim_path_of[i]}>" for i in dict.fromkeys(ids) if i in prim_path_of]


def _extract_pbr_textures(glb_path: Path, out_dir: Path) -> dict:
    """Pull baseColor + metallicRoughness textures out of ``glb_path`` and
    write them as PNG files into ``out_dir``. Returns a dict of which
    textures were actually written, e.g. {"baseColor": True, "metallicRoughness": True}.

    Uses raw glTF parsing (we know the structure) instead of trimesh because
    trimesh's PBRMaterial only surfaces the baseColor and silently drops the
    metallicRoughness map. Without metallicRoughness, the asset renders as
    a matte sheet of camo noise — Seed3D outputs put 70%+ of UVs in noise
    atlas regions and rely on the metallic/roughness map to PBR-shade those
    pixels into a uniform mirror, hiding the noise visually."""
    import json, struct  # noqa: PLC0415
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    with open(glb_path, "rb") as f:
        magic, ver, total = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:   # 'glTF'
            print(f"[textures] {glb_path}: not a binary glTF (skipping PBR extract)")
            return written
        j_len, _ = struct.unpack("<II", f.read(8))
        gltf = json.loads(f.read(j_len))
        b_len, _ = struct.unpack("<II", f.read(8))
        bin_data = f.read(b_len)

    materials = gltf.get("materials", [])
    if not materials:
        print(f"[textures] {glb_path}: no materials")
        return written
    pbr = materials[0].get("pbrMetallicRoughness", {})

    def _save_image(img_idx: int, name: str):
        if img_idx is None or img_idx >= len(gltf.get("images", [])):
            return
        img = gltf["images"][img_idx]
        bv_idx = img.get("bufferView")
        if bv_idx is None:
            return
        bv = gltf["bufferViews"][bv_idx]
        offset = bv.get("byteOffset", 0)
        length = bv["byteLength"]
        ext = "png" if img.get("mimeType", "").endswith("png") else "jpg"
        out_path = out_dir / f"{name}.{ext}"
        out_path.write_bytes(bin_data[offset:offset + length])
        print(f"[textures] saved {name} ({length // 1024} KB) -> {out_path}")
        written[name] = True

    bc = pbr.get("baseColorTexture")
    if bc is not None:
        tex = gltf["textures"][bc["index"]]
        _save_image(tex.get("source"), "baseColor")
    mr = pbr.get("metallicRoughnessTexture")
    if mr is not None:
        tex = gltf["textures"][mr["index"]]
        _save_image(tex.get("source"), "metallicRoughness")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_glb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_ground", action="store_true",
                    help="omit the default ground plane")
    ap.add_argument("--texture_glb", default=None,
                    help="GLB to extract baseColor texture from. Defaults to "
                         "--clean_glb. Point at the *original* (full-res) GLB "
                         "if --clean_glb is a downsampled preprocessed copy.")
    args = ap.parse_args()

    labels_path = Path(args.labels).resolve()
    labels = json.loads(labels_path.read_text())
    validate(labels)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="articulator_"))

    # Browser and build_usd both read this exact file — preprocess_glb.py no
    # longer writes a decimated twin, so what you label in the browser is
    # byte-for-byte what we emit to USDA.
    clean_glb_path = Path(args.clean_glb).resolve()

    # --- Stage 0: extract PBR textures (baseColor + metallicRoughness) -------
    texture_src = Path(args.texture_glb).resolve() if args.texture_glb else clean_glb_path
    written_textures = _extract_pbr_textures(texture_src, out_path.parent / "textures")

    # --- Stage 1: export per-part OBJs via Blender headless --------------------
    cfg = {
        "source_glb": str(clean_glb_path),
        "out_dir": str(work),
        "parts": {p["id"]: p["clusters"] for p in labels["parts"]},
        "external_meshes": [],
        "split_clusters": labels.get("split_clusters", {}),
    }
    for em in labels.get("external_meshes", []):
        glb = em["glb"]
        if not Path(glb).is_absolute():
            glb = str((PROJECT_ROOT / glb).resolve())
        cfg["external_meshes"].append({
            "part_id": em["attach_to"],
            "glb": glb,
            "cluster_filter": em.get("cluster_filter"),
        })
    cfg_path = work / "_export.cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    script_path = work / "_export.py"
    script_path.write_text(EXPORT_PARTS_SCRIPT)

    print(f"[stage 1] export per-part OBJ via Blender (scratch: {work})")
    proc = subprocess.run(
        [str(BLENDER), "--background", "--python", str(script_path), "--", str(cfg_path)],
        check=False, capture_output=True, text=True,
    )
    # Surface Blender's output so [split]/[export] diagnostic lines reach the
    # browser log box. Filter the noisy importer chatter so the useful stuff
    # stands out.
    for line in (proc.stdout or "").splitlines():
        if any(line.startswith(t) for t in ("[split]", "[export]")):
            print(line)
    if proc.returncode != 0:
        print("[error] Blender stage 1 returncode", proc.returncode)
        print(proc.stderr)
        raise SystemExit(1)

    # --- Stage 2: load Blender-produced npz files ----------------------------
    part_meshes: dict[str, tuple] = {}
    for p in labels["parts"]:
        path = work / f"{p['id']}.npz"
        if not path.exists():
            raise SystemExit(f"[error] expected {path} from Blender export but it was not produced")
        part_meshes[p["id"]] = _read_part_npz(path)
        v, f, u, _ = part_meshes[p["id"]]
        print(f"  {p['id']}: {len(v)} verts, {len(f)} faces, {len(u)} uvs")

    # --- Stage 3: per-part scale + external transforms -----------------------
    parts_by_id = {p["id"]: p for p in labels["parts"]}
    for pid, part in parts_by_id.items():
        if part.get("scale_xyz") and part["scale_xyz"] != [1, 1, 1]:
            scale_zup = _schema_scale_to_usd_zup(part["scale_xyz"])
            part_meshes[pid] = _apply_part_scale(part_meshes[pid], scale_zup)
            print(f"  part-scale {pid}: schema={part['scale_xyz']} zup={scale_zup}")
    for em in labels.get("external_meshes", []):
        pid = em["attach_to"]
        part_meshes[pid] = _apply_external_transform(part_meshes[pid], em["transform"])
        print(f"  ext-transform {pid}: t={em['transform']['t']} q={em['transform']['q_wxyz']} s={em['transform']['s']}")

    # --- Stage 4: bake joint angles + offsets (topo order) -------------------
    _apply_joint_bakes_and_offsets(part_meshes, labels)

    usda = build_usda(
        labels, part_meshes,
        with_ground=not args.no_ground,
        has_metallic_roughness=bool(written_textures.get("metallicRoughness", False)),
    )
    out_path.write_text(usda)
    size_kb = out_path.stat().st_size / 1024
    print(f"[done] {out_path}  ({size_kb:.0f} KB)")
    shutil.rmtree(work, ignore_errors=True)


def build_usda(labels: dict, part_meshes: dict[str, tuple], *,
               with_ground: bool = True, has_metallic_roughness: bool = False) -> str:
    """Pure in-memory step: take post-stage-4 part meshes (geometry already
    scaled / external-transformed / joint-baked) and emit the USDA string.

    Split out of ``main()`` so structural unit tests can drive it with
    synthetic part meshes — no Blender required."""
    parts_by_id = {p["id"]: p for p in labels["parts"]}

    # --- Stage 5: uniform Z-up scale + centering -----------------------------
    all_xs: list[float] = []; all_ys: list[float] = []; all_zs: list[float] = []
    for v, _, _, _ in part_meshes.values():
        all_xs.extend(p[0] for p in v)
        all_ys.extend(p[1] for p in v)
        all_zs.extend(p[2] for p in v)
    extents = [max(all_xs) - min(all_xs), max(all_ys) - min(all_ys), max(all_zs) - min(all_zs)]
    dims_mm = labels.get("physical_dims_mm", {"x": max(extents) * 1000, "y": max(extents) * 1000, "z": max(extents) * 1000})
    phys_max_m = max(dims_mm["x"], dims_mm["y"], dims_mm["z"]) / 1000.0
    uniform = phys_max_m / max(extents)
    scale = (uniform, uniform, uniform)
    translate = (
        -(min(all_xs) + max(all_xs)) / 2 * uniform,
        -(min(all_ys) + max(all_ys)) / 2 * uniform,
        -min(all_zs) * uniform,
    )
    print(f"[uniform-scale] x{uniform:.4f}  extents={[round(e,3) for e in extents]}  "
          f"-> phys_max={phys_max_m:.4f}m  t={tuple(round(t,4) for t in translate)}")

    # --- Stage 6: build USD prim path map ------------------------------------
    device_name = labels.get("device", "device")
    comp_of, n_comp = _connected_components(labels)
    component_xform_name: dict[int, str] = {}
    # The "main" articulated component (multi-part) gets the device name as
    # its Xform. Additional multi-part components get device_name_<root>.
    main_assigned = False
    for ci in range(n_comp):
        members = [p["id"] for p in labels["parts"] if comp_of[p["id"]] == ci]
        if len(members) <= 1:
            continue  # singleton — emit at /World/<part_id> directly
        if not main_assigned:
            component_xform_name[ci] = device_name
            main_assigned = True
        else:
            root = _component_root(labels, ci, comp_of)
            component_xform_name[ci] = f"{device_name}_{root}"

    prim_path_of: dict[str, str] = {}
    for pid in [p["id"] for p in labels["parts"]]:
        ci = comp_of[pid]
        if ci in component_xform_name:
            prim_path_of[pid] = f"/World/{component_xform_name[ci]}/{pid}"
        else:
            prim_path_of[pid] = f"/World/{pid}"

    # --- Stage 7: emit USDA --------------------------------------------------
    chunks: list[str] = [_emit_header(device_name, with_ground=with_ground)]

    material_name = f"{device_name}Mat"
    material_path = f"/World/Materials/{material_name}"
    override_material_defs: list[str] = []

    by_child = joints_by_child(labels)
    for ci in range(n_comp):
        members = [p["id"] for p in labels["parts"] if comp_of[p["id"]] == ci]
        if len(members) > 1:
            xform_name = component_xform_name[ci]
            chunks.append(f"    def Xform \"{xform_name}\" (\n        kind = \"component\"\n    )\n    {{\n")
            body_indent = "        "
        else:
            body_indent = "    "
        joint_indent = body_indent + "    "  # joint nests inside its child body's Xform

        for pid in members:
            part = parts_by_id[pid]
            j = by_child.get(pid)
            filtered = _resolve_filtered_pairs(pid, j, part, prim_path_of, parts_by_id)
            joint_block = ""
            if j is not None:
                emit = JOINT_EMITTERS[j["type"]]
                joint_block = emit(
                    j, prim_path_of[j["parent"]], prim_path_of[j["child"]], joint_indent,
                    scale=scale, translate=translate,
                )
            chunks.append(_emit_part_body(
                part, part_meshes[pid], scale, translate,
                prim_path=prim_path_of[pid],
                parent_path=prim_path_of[pid].rsplit("/", 1)[0],
                material_path=material_path,
                filtered_pair_paths=filtered,
                joint_block=joint_block,
                override_materials_out=override_material_defs,
            ))

        if len(members) > 1:
            chunks.append("    }\n\n")

    chunks.append(_emit_materials_block(material_path, has_metallic_roughness,
                                        extra_material_defs=override_material_defs))
    chunks.append(_emit_footer())
    return "".join(chunks)


def _emit_materials_block(material_path: str, has_metallic_roughness: bool,
                          extra_material_defs: list[str] | None = None) -> str:
    """UsdPreviewSurface bound to baseColor + (optional) metallicRoughness.
    Without the metallicRoughness map, Seed3D-style assets render their
    "noise atlas" regions as raw camo speckle. With it, those regions read
    high-metallic/low-roughness and PBR-shade into a uniform mirror, which
    is what Blender's render shows."""
    name = material_path.rsplit("/", 1)[-1]
    if has_metallic_roughness:
        surface_extra = (
            f'                float inputs:roughness.connect = <{material_path}/metallicRoughnessTex.outputs:g>\n'
            f'                float inputs:metallic.connect = <{material_path}/metallicRoughnessTex.outputs:b>\n'
        )
        mr_shader = f"""
            def Shader "metallicRoughnessTex"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./textures/metallicRoughness.png@
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float2 inputs:st.connect = <{material_path}/uvReader.outputs:result>
                float outputs:g
                float outputs:b
            }}
"""
    else:
        surface_extra = (
            "                float inputs:roughness = 0.5\n"
            "                float inputs:metallic = 0.0\n"
        )
        mr_shader = ""
    extras = "".join(extra_material_defs or [])
    return f"""    def Scope "Materials"
    {{
        def Material "{name}"
        {{
            token outputs:surface.connect = <{material_path}/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{material_path}/baseColorTex.outputs:rgb>
{surface_extra}                token outputs:surface
            }}

            def Shader "baseColorTex"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./textures/baseColor.png@
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float2 inputs:st.connect = <{material_path}/uvReader.outputs:result>
                float3 outputs:rgb
            }}
{mr_shader}
            def Shader "uvReader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
        }}
{extras}    }}
"""


if __name__ == "__main__":
    main()
