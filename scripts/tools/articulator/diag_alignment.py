"""Diagnose lid/body alignment in the same Y-up frame the web editor uses.

Reads the merged-by-Blender per-label OBJs (re-exporting them once for the
diagnostic), applies user's part_scales + lid_offset, prints body vs lid
bboxes, and computes the gap. Lets you confirm whether the web editor
actually shows the lid closed before debugging the USD export further.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BLENDER = PROJECT_ROOT / "software" / "blender-4.4.0-linux-x64" / "blender"


SPLIT_SCRIPT = r'''
import bpy, json, sys
from pathlib import Path
argv = sys.argv[sys.argv.index("--") + 1:]
glb_path, labels_path, out_dir = argv[0], argv[1], Path(argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=glb_path)

cluster_label = json.loads(Path(labels_path).read_text())["labels"]
groups = {}
for o in bpy.context.scene.objects:
    if o.type != "MESH": continue
    label = cluster_label.get(o.name, "unlabeled")
    if label in ("unlabeled", "ignore"): continue
    groups.setdefault(label, []).append(o)

for label, objs in groups.items():
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1: bpy.ops.object.join()
    bpy.ops.wm.obj_export(filepath=str(out_dir / f"{label}.obj"),
                          export_selected_objects=True,
                          forward_axis="NEGATIVE_Z", up_axis="Y",
                          export_materials=False)
'''


def read_obj(p):
    verts = []
    for line in p.read_text().splitlines():
        if line.startswith("v "):
            x, y, z = line.split()[1:4]
            verts.append((float(x), float(y), float(z)))
    return verts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_glb", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="diag_"))
    script = work / "split.py"
    script.write_text(SPLIT_SCRIPT)
    subprocess.run([str(BLENDER), "--background", "--python", str(script),
                    "--", args.clean_glb, args.labels, str(work)],
                   check=True, capture_output=True)

    labels_data = json.loads(Path(args.labels).read_text())
    part_scales = labels_data.get("part_scales") or {}
    lid_offset = labels_data.get("lid_offset", [0, 0, 0])
    hinge = labels_data["hinge"]

    parts = {}
    for name in ["body", "lid"]:
        verts = read_obj(work / f"{name}.obj")
        s = part_scales.get(name, [1, 1, 1])
        if s != [1, 1, 1]:
            xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
            cx = (min(xs)+max(xs))/2  # bbox center, matches web editor
            cy = (min(ys)+max(ys))/2
            cz = (min(zs)+max(zs))/2
            verts = [(s[0]*(v[0]-cx)+cx, s[1]*(v[1]-cy)+cy, s[2]*(v[2]-cz)+cz) for v in verts]
            print(f"  scaled {name} by {s} around bbox-center ({cx:.3f},{cy:.3f},{cz:.3f})")
        if name == "lid" and any(abs(c) > 1e-6 for c in lid_offset):
            verts = [(v[0]+lid_offset[0], v[1]+lid_offset[1], v[2]+lid_offset[2]) for v in verts]
            print(f"  applied lid_offset {lid_offset}")
        parts[name] = verts

    def bbox(verts):
        xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    print("\n=== bbox in Y-up frame (web editor's coord system) ===")
    for name in ["body", "lid"]:
        mn, mx = bbox(parts[name])
        print(f"  {name}: X[{mn[0]:+.3f},{mx[0]:+.3f}]  Y[{mn[1]:+.3f},{mx[1]:+.3f}]  Z[{mn[2]:+.3f},{mx[2]:+.3f}]")

    bb = bbox(parts["body"]); lb = bbox(parts["lid"])
    body_center = [(bb[0][i]+bb[1][i])/2 for i in range(3)]
    lid_center  = [(lb[0][i]+lb[1][i])/2 for i in range(3)]
    print(f"\n  body center: ({body_center[0]:+.3f}, {body_center[1]:+.3f}, {body_center[2]:+.3f})")
    print(f"  lid  center: ({lid_center[0]:+.3f}, {lid_center[1]:+.3f}, {lid_center[2]:+.3f})")
    print(f"  delta (lid - body, in Y-up): "
          f"X={lid_center[0]-body_center[0]:+.3f}  "
          f"Y={lid_center[1]-body_center[1]:+.3f}  "
          f"Z={lid_center[2]-body_center[2]:+.3f}")
    print("  -> for a CLOSED clamshell, only Y delta should be large positive (lid above body).")
    print("     X and Z deltas should be near zero. Big X/Z delta = lid not aligned with body.")

    print(f"\n  hinge p0 (Y-up): {hinge['p0']}")
    print(f"  hinge p1 (Y-up): {hinge['p1']}")
    import shutil; shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
