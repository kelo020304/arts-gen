"""Inspect a Seed3D GLB: split by loose parts, list each part's stats.

Run:
    software/blender-4.4.0-linux-x64/blender --background \\
        --python scripts/tools/inspect_seed3d_glb.py \\
        -- --glb /home/mi/jzh/earphone/pbr/mesh_textured_pbr.glb \\
           --out outputs/xiaomi_buds6_seed3d
"""
import bpy
import os
import sys
import json
from pathlib import Path


def _argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _parse(argv):
    args = {"glb": None, "out": None}
    i = 0
    while i < len(argv):
        if argv[i] == "--glb":
            args["glb"] = argv[i + 1]; i += 2
        elif argv[i] == "--out":
            args["out"] = argv[i + 1]; i += 2
        else:
            i += 1
    return args


def main():
    a = _parse(_argv())
    glb = Path(a["glb"]).resolve()
    out = Path(a["out"]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.import_scene.gltf(filepath=str(glb))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    print(f"[gltf] imported {len(meshes)} mesh objects")

    # Join all meshes (if any) into one before separating by loose parts.
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    print(f"[joined] verts={len(obj.data.vertices)}  polys={len(obj.data.polygons)}")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")

    parts = sorted(
        (o for o in bpy.context.scene.objects if o.type == "MESH"),
        key=lambda o: -len(o.data.vertices),
    )
    print(f"[split] {len(parts)} connected components")

    summary = []
    for i, p in enumerate(parts):
        bpy.context.view_layer.update()
        coords = [p.matrix_world @ v.co for v in p.data.vertices]
        xs, ys, zs = zip(*[(c.x, c.y, c.z) for c in coords])
        ext = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        ctr = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2)
        info = {
            "idx": i, "name": p.name,
            "verts": len(p.data.vertices),
            "polys": len(p.data.polygons),
            "extents": [round(v, 4) for v in ext],
            "center": [round(v, 4) for v in ctr],
        }
        summary.append(info)
        print(f"  cc{i}: name={p.name!r} verts={info['verts']:>6} ext={info['extents']} ctr={info['center']}")

        # Export each part as OBJ for downstream pipeline
        bpy.ops.object.select_all(action="DESELECT")
        p.select_set(True)
        bpy.context.view_layer.objects.active = p
        bpy.ops.wm.obj_export(
            filepath=str(out / f"part_{i:02d}.obj"),
            export_selected_objects=True,
            forward_axis="Y", up_axis="Z",
            export_materials=False,
        )

    (out / "parts_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] -> {out}")


main()
