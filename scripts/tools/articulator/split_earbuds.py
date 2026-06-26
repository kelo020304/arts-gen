"""Split a Seed3D-style "two earbuds together on table" GLB into two separate
GLBs (earbud_L.glb + earbud_R.glb) by connected components + auto-detected
separation axis.

Auto-detects which axis (X, Y, or Z) the two earbuds are spread along by
finding the axis with the largest centroid-difference magnitude. The earbud
on the *smaller* side of that axis becomes earbud_L by default; pass
``--swap`` if your physical L/R got flipped.

Run via Blender headless (no env activation needed)::

    ./software/blender-4.4.0-linux-x64/blender --background \\
        --python scripts/tools/articulator/split_earbuds.py -- \\
        --in /path/to/both_earbuds.glb \\
        --out_dir outputs/earbuds_split

Outputs:
    outputs/earbuds_split/earbud_L.glb
    outputs/earbuds_split/earbud_R.glb
    outputs/earbuds_split/info.json   # vert count, bbox, centroid, split axis
"""
import bpy
import json
import sys
from pathlib import Path


def _argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _parse(argv):
    a = {"in": None, "out_dir": None, "swap": False, "axis": None}
    i = 0
    while i < len(argv):
        if argv[i] in ("--in", "-i"):
            a["in"] = argv[i + 1]; i += 2
        elif argv[i] in ("--out_dir", "-o"):
            a["out_dir"] = argv[i + 1]; i += 2
        elif argv[i] == "--swap":
            a["swap"] = True; i += 1
        elif argv[i] == "--axis":
            a["axis"] = argv[i + 1]; i += 2  # force splitting axis: x|y|z
        else:
            i += 1
    if not a["in"] or not a["out_dir"]:
        print("usage: -- --in <glb> --out_dir <dir> [--swap] [--axis x|y|z]", file=sys.stderr); sys.exit(2)
    return a


def main():
    args = _parse(_argv())
    out_dir = Path(args["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=args["in"])

    # Join everything → one big mesh
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("[error] no meshes in input GLB")
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()

    # Split by loose parts
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")

    parts = sorted([o for o in bpy.context.scene.objects if o.type == "MESH"],
                   key=lambda o: -len(o.data.vertices))
    if len(parts) < 2:
        raise SystemExit(f"[error] expected ≥2 connected components, got {len(parts)}. "
                         "The two earbuds may be touching geometrically — separate them in the source photo.")

    # Two largest = the actual earbuds. Discard noise (UV island fragments etc).
    big = parts[:2]
    noise = parts[2:]
    print(f"[split] {len(big)} earbud parts kept, {len(noise)} noise parts dropped "
          f"(noise total verts: {sum(len(p.data.vertices) for p in noise)})")
    for p in noise:
        bpy.data.objects.remove(p, do_unlink=True)

    # Compute centroids
    centers = []
    for p in big:
        bpy.context.view_layer.update()
        coords = [p.matrix_world @ v.co for v in p.data.vertices]
        cx = sum(c.x for c in coords) / len(coords)
        cy = sum(c.y for c in coords) / len(coords)
        cz = sum(c.z for c in coords) / len(coords)
        centers.append((cx, cy, cz, p))

    # Auto-detect split axis: pick the axis with the largest centroid spread.
    # User can override with --axis x|y|z.
    dx = abs(centers[0][0] - centers[1][0])
    dy = abs(centers[0][1] - centers[1][1])
    dz = abs(centers[0][2] - centers[1][2])
    spread = {"x": dx, "y": dy, "z": dz}
    if args["axis"] in ("x", "y", "z"):
        axis_name = args["axis"]
        print(f"[split-axis] forced --axis={axis_name}  spreads: {spread}")
    else:
        axis_name = max(spread, key=spread.get)
        print(f"[split-axis] auto-detected '{axis_name}' (spreads: x={dx:.4f} y={dy:.4f} z={dz:.4f})")
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name]

    # Smaller value on that axis = earbud_L (default); --swap inverts.
    centers.sort(key=lambda t: t[axis_idx])
    if args["swap"]:
        centers = list(reversed(centers))
        print("[split-axis] --swap requested; L/R inverted")
    L_data = centers[0]; R_data = centers[1]
    L_obj = L_data[3]; R_obj = R_data[3]
    L_obj.name = "earbud_L"
    R_obj.name = "earbud_R"

    summary = {}
    for tag, obj, ctr in [("earbud_L", L_obj, L_data[:3]), ("earbud_R", R_obj, R_data[:3])]:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        coords = [obj.matrix_world @ v.co for v in obj.data.vertices]
        xs, ys, zs = zip(*[(c.x, c.y, c.z) for c in coords])
        bbox_min = [min(xs), min(ys), min(zs)]
        bbox_max = [max(xs), max(ys), max(zs)]
        out_glb = out_dir / f"{tag}.glb"
        bpy.ops.export_scene.gltf(
            filepath=str(out_glb),
            export_format="GLB",
            use_selection=True,
            export_apply=True,
            export_materials="EXPORT",
        )
        summary[tag] = {
            "glb": f"{tag}.glb",
            "verts": len(obj.data.vertices),
            "centroid": list(ctr),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        }
        print(f"[ok] {tag}: {len(obj.data.vertices)} verts, "
              f"centroid=({ctr[0]:.3f}, {ctr[1]:.3f}, {ctr[2]:.3f})")

    summary["_split_axis"] = axis_name
    summary["_centroid_spreads"] = spread
    (out_dir / "info.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] -> {out_dir}")
    print("  if L/R look swapped after rendering, re-run with --swap.")
    print("  if the wrong axis was picked (rare), re-run with --axis x|y|z.")


main()
