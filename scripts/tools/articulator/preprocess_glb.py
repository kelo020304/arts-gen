"""Preprocess a messy Seed3D GLB into a clean labelable GLB for the web editor.

Seed3D-style outputs split a single physical object into thousands of UV-island
connected components. This script:

  1) Splits the input GLB by loose parts (Blender headless).
  2) Reads back per-component centroids, drops low-vertex artifacts.
  3) Clusters the remaining components by spatial centroid via DBSCAN.
  4) Re-merges per cluster + writes a new GLB where each cluster is a named
     submesh (cluster_00, cluster_01, ...). This is what the web editor loads.

Usage (with arts-gen env activated)::

    python scripts/tools/articulator/preprocess_glb.py \\
        --in  /home/mi/jzh/earphone/pbr/mesh_textured_pbr.glb \\
        --out outputs/xiaomi_buds6_seed3d/clean.glb \\
        --eps 0.20 --min_verts 200
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BLENDER = PROJECT_ROOT / "software" / "blender-4.4.0-linux-x64" / "blender"


SPLIT_SCRIPT = r'''
import bpy, json, sys
from pathlib import Path

# Stage 1: Import GLB, split by loose, write per-part centroids to
# summary.json. NO OBJ export — that round-trip dropped embedded textures
# from bpy.data.images, leaving the merged GLB with empty materials.
# Stage 2 re-imports the same original GLB (textures intact) and re-runs
# the same deterministic loose-parts split to get the same parts back.

argv = sys.argv[sys.argv.index("--") + 1:]
glb_path = argv[0]
work_dir = Path(argv[1]); work_dir.mkdir(parents=True, exist_ok=True)
min_verts = int(argv[2])

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=glb_path)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for m in meshes: m.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.separate(type="LOOSE")
bpy.ops.object.mode_set(mode="OBJECT")

parts = sorted([o for o in bpy.context.scene.objects if o.type == "MESH"],
               key=lambda o: -len(o.data.vertices))

summary = []
for i, p in enumerate(parts):
    if len(p.data.vertices) < min_verts:
        continue
    bpy.context.view_layer.update()
    coords = [p.matrix_world @ v.co for v in p.data.vertices]
    xs, ys, zs = zip(*[(c.x, c.y, c.z) for c in coords])
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    cz = (max(zs) + min(zs)) / 2
    summary.append({
        "idx": i, "name": p.name,
        "verts": len(p.data.vertices),
        "center": [cx, cy, cz],
    })

(work_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"[split] kept {len(summary)} parts >= {min_verts} verts (no OBJ exports — textures preserved)")
'''


MERGE_SCRIPT = r'''
import bpy, json, sys
from pathlib import Path

# Stage 2: Re-import the original GLB (textures intact), re-run the same
# deterministic loose-parts split as Stage 1, then group parts by cluster
# id (provided in clusters.json as ``list[list[part_idx]]``).

argv = sys.argv[sys.argv.index("--") + 1:]
in_glb = argv[0]                  # original GLB (with textures)
out_glb = argv[1]                 # output (no decimation — same mesh used by build_usd.py)
clusters_json = Path(argv[2])     # list[list[idx]]
min_verts = int(argv[3])

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=in_glb)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for m in meshes: m.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.separate(type="LOOSE")
bpy.ops.object.mode_set(mode="OBJECT")

parts_sorted = sorted([o for o in bpy.context.scene.objects if o.type == "MESH"],
                      key=lambda o: -len(o.data.vertices))
# Same min_verts filter as SPLIT phase — but keep idx of the *original* sort
# so cluster references stay valid.
parts_by_idx = {i: p for i, p in enumerate(parts_sorted) if len(p.data.vertices) >= min_verts}
# Drop the rest
for i, p in enumerate(parts_sorted):
    if i not in parts_by_idx:
        bpy.data.objects.remove(p, do_unlink=True)

clusters = json.loads(clusters_json.read_text())  # list[list[part_idx]]

cluster_objs = []
for ci, member_idxs in enumerate(clusters):
    members = [parts_by_idx[i] for i in member_idxs if i in parts_by_idx]
    if not members:
        continue
    bpy.ops.object.select_all(action="DESELECT")
    for o in members:
        o.select_set(True)
    bpy.context.view_layer.objects.active = members[0]
    if len(members) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = f"cluster_{ci:02d}"
    obj.data.name = obj.name
    cluster_objs.append(obj)


# Single full-res output. Browser and build_usd both use this exact same mesh
# so what you label / box-split in the browser is byte-for-byte what build_usd
# emits to USDA. No decimation, no twin file — only DBSCAN clustering.
bpy.ops.object.select_all(action="DESELECT")
for o in cluster_objs:
    o.select_set(True)
bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format="GLB",
    use_selection=True,
    export_apply=True,
    export_materials="EXPORT",
    # AUTO keeps original image formats; quality 100 avoids JPEG re-encoding
    # loss. Without these the embedded baseColor gets downscaled to a tiny
    # 256×256 PNG, which then renders as a blurry asset in IsaacSim.
    export_image_format="AUTO",
    export_image_quality=100,
)
total = sum(len(o.data.vertices) for o in cluster_objs)
print(f"[merge] wrote {out_glb} with {len(cluster_objs)} clusters, {total} verts (no decimation)")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=0.20,
                    help="DBSCAN radius in normalized mesh coords")
    ap.add_argument("--min_verts", type=int, default=1,
                    help="drop loose-component parts with fewer vertices than "
                         "this. Default 1 = keep everything (lets DBSCAN deal "
                         "with tiny stray fragments). Bump up only if you have "
                         "obvious noise blobs you want gone.")
    ap.add_argument("--workdir", default=None,
                    help="scratch dir (default: <out_parent>/_split)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(args.workdir).resolve() if args.workdir else out.parent / "_split"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    print(f"[stage 1] split {args.inp} into per-cc OBJs (min_verts={args.min_verts})")
    split_py = work / "split.py"
    split_py.write_text(SPLIT_SCRIPT)
    subprocess.run([str(BLENDER), "--background", "--python", str(split_py),
                    "--", args.inp, str(work), str(args.min_verts)],
                   check=True, capture_output=True)

    summary = json.loads((work / "summary.json").read_text())
    print(f"[stage 1] kept {len(summary)} parts >= {args.min_verts} verts")

    # DBSCAN on centroids
    import numpy as np
    from sklearn.cluster import DBSCAN

    centers = np.array([s["center"] for s in summary], dtype=float)
    labels = DBSCAN(eps=args.eps, min_samples=1).fit_predict(centers)
    n_clusters = int(labels.max()) + 1
    print(f"[stage 2] DBSCAN(eps={args.eps}) -> {n_clusters} clusters")

    clusters: list[list[int]] = [[] for _ in range(n_clusters)]
    for s, lab in zip(summary, labels):
        clusters[lab].append(int(s["idx"]))

    # Sort clusters by size (largest first) for stable IDs
    cluster_sizes = []
    for ci, members in enumerate(clusters):
        verts = sum(s["verts"] for s, lab in zip(summary, labels) if lab == ci)
        cluster_sizes.append((ci, verts, len(members)))
    cluster_sizes.sort(key=lambda x: -x[1])

    sorted_clusters = [clusters[ci] for ci, _, _ in cluster_sizes]
    print("[stage 2] clusters by size:")
    for new_idx, (ci, verts, count) in enumerate(cluster_sizes):
        cluster_centers = np.array(
            [s["center"] for s, lab in zip(summary, labels) if lab == ci]
        )
        ctr = cluster_centers.mean(axis=0).round(3).tolist()
        print(f"  cluster_{new_idx:02d}: {count} parts, {verts} verts, center={ctr}")

    clusters_json = work / "clusters.json"
    clusters_json.write_text(json.dumps(sorted_clusters))

    # Single output. The browser editor and build_usd.py both read this exact
    # file, so what you see / box-split / label in the browser is byte-for-
    # byte what build_usd emits to USDA. No decimated twin.
    print(f"[stage 3] merge clusters -> {out}")
    merge_py = work / "merge.py"
    merge_py.write_text(MERGE_SCRIPT)
    subprocess.run([str(BLENDER), "--background", "--python", str(merge_py),
                    "--", str(Path(args.inp).resolve()), str(out), str(clusters_json),
                    str(args.min_verts)],
                   check=True, capture_output=True)
    print(f"[done] -> {out}  ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
