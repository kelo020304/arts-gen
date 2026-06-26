"""PCA-based alignment of split earbud GLBs to original case cluster bboxes.

Computes initial translate / rotate / scale for each external earbud so its
principal axes align with the original case cluster's principal axes. Writes
the transforms into labels.json under ``external_earbuds`` so the web editor
loads them already aligned (user fine-tunes from there).

Run (with arts-gen activated)::

    python scripts/tools/articulator/align_earbuds.py \\
        --clean_glb outputs/xiaomi_buds6_seed3d/clean.glb \\
        --labels    outputs/xiaomi_buds6_seed3d/labels.json \\
        --separated outputs/xiaomi_buds6_seed3d/ear_split/separated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _pca(verts: np.ndarray):
    """Return (centroid, eigvals_desc, eigvecs_desc as columns).
    Eigenvectors are orthonormal and ordered by descending eigenvalue.
    """
    c = verts.mean(axis=0)
    centered = verts - c
    cov = (centered.T @ centered) / len(verts)
    eigvals, eigvecs = np.linalg.eigh(cov)        # ascending
    idx = np.argsort(eigvals)[::-1]
    return c, eigvals[idx], eigvecs[:, idx]


def _orient_consistent(eigvecs: np.ndarray, verts_centered: np.ndarray) -> np.ndarray:
    """Eigenvectors have sign ambiguity. Force the principal axis to point
    toward the side with more vertex mass, for a deterministic orientation.
    """
    out = eigvecs.copy()
    for i in range(3):
        proj = verts_centered @ out[:, i]
        # Pick orientation where the cube root of the sum of CUBE projections is positive
        # (skewness sign, robust to symmetric distributions where mean is ~0).
        skew = float(np.mean(proj ** 3))
        if skew < 0:
            out[:, i] *= -1
    # Make right-handed if det < 0 (flip the SHORTEST axis since its sign is least informative)
    if np.linalg.det(out) < 0:
        out[:, 2] *= -1
    return out


def _mat_to_quat_wxyz(R: np.ndarray):
    """3x3 rotation matrix -> (w, x, y, z) quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return [float(w), float(x), float(y), float(z)]


def _gather_cluster_verts(clean_glb: Path, labels: dict, label: str) -> np.ndarray:
    """Concatenate vertex coordinates of every cluster labelled `label`."""
    scene = trimesh.load(clean_glb, force="scene")
    out = []
    for name, mesh in scene.geometry.items():
        if labels.get(name) != label:
            continue
        out.append(np.asarray(mesh.vertices, dtype=np.float64))
    if not out:
        raise SystemExit(f"[error] no clusters labelled '{label}' in clean.glb")
    return np.vstack(out)


def _load_external_verts(glb_path: Path) -> np.ndarray:
    scene = trimesh.load(glb_path, force="scene")
    out = []
    for mesh in scene.geometry.values():
        out.append(np.asarray(mesh.vertices, dtype=np.float64))
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_glb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--separated", required=True,
                    help="dir containing earbud_L.glb + earbud_R.glb")
    ap.add_argument("--scale_mode", default="longest",
                    choices=["longest", "mean", "max", "per_axis"],
                    help="how to derive scale. 'longest' (default) uses just the "
                         "ratio along the longest principal axis — best when the "
                         "original cluster is contaminated by surrounding case "
                         "geometry but its main length is still trustworthy.")
    args = ap.parse_args()

    labels_path = Path(args.labels).resolve()
    labels_data = json.loads(labels_path.read_text())
    labels = labels_data["labels"]

    sep_dir = Path(args.separated).resolve()

    external_earbuds = {}
    for tag in ("earbud_L", "earbud_R"):
        # Original case cluster verts
        orig = _gather_cluster_verts(Path(args.clean_glb), labels, tag)
        c_orig, ev_orig, R_orig = _pca(orig)
        R_orig = _orient_consistent(R_orig, orig - c_orig)

        # New earbud verts
        glb = sep_dir / f"{tag}.glb"
        if not glb.exists():
            raise SystemExit(f"[error] missing {glb}")
        new = _load_external_verts(glb)
        c_new, ev_new, R_new = _pca(new)
        R_new = _orient_consistent(R_new, new - c_new)

        # Rotation: map new principal frame -> original principal frame
        R = R_orig @ R_new.T
        # Final det check (should be +1 by construction)
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1

        # Use ORIENTED-BBOX EXTENTS along principal axes for scale (NOT std-dev,
        # because std-dev gets inflated when the cluster is multiple disjoint
        # parts spread out — we want span, not spread).
        def _obb_extent(verts, c, R):
            centered = verts - c
            return np.array([
                (centered @ R[:, i]).max() - (centered @ R[:, i]).min()
                for i in range(3)
            ])
        ext_orig = _obb_extent(orig, c_orig, R_orig)
        ext_new = _obb_extent(new, c_new, R_new)
        per_axis = ext_orig / np.maximum(ext_new, 1e-9)
        if args.scale_mode == "longest":
            # principal axes are sorted by descending eigenvalue, so [0] is longest
            s = float(per_axis[0])
            scale = [s, s, s]
        elif args.scale_mode == "mean":
            s = float(per_axis.mean())
            scale = [s, s, s]
        elif args.scale_mode == "max":
            s = float(per_axis.max())
            scale = [s, s, s]
        else:  # per_axis
            scale = per_axis.tolist()

        quat = _mat_to_quat_wxyz(R)

        # Translation: place new earbud's centroid at original centroid.
        # (Translation is applied AFTER rotation+scale around new centroid in
        #  the web editor's transform; equivalent here since we send the
        #  desired world centroid for the transformed earbud.)
        external_earbuds[tag] = {
            "glb": str(glb.relative_to(Path(__file__).resolve().parent.parent.parent))
                   if glb.is_relative_to(Path(__file__).resolve().parent.parent.parent)
                   else str(glb),
            "translate": c_orig.tolist(),
            "rotate_quat_wxyz": quat,
            "scale": scale,
            "_pivot_in_new_glb_frame": c_new.tolist(),
            "_obb_extent_orig": ext_orig.tolist(),
            "_obb_extent_new": ext_new.tolist(),
        }
        print(f"[{tag}]")
        print(f"  obb extent (orig): {ext_orig.round(4).tolist()}")
        print(f"  obb extent (new):  {ext_new.round(4).tolist()}")
        print(f"  per-axis ratio:    {per_axis.round(4).tolist()}")
        print(f"  scale ({args.scale_mode}): {[round(x, 4) for x in scale]}")
        print(f"  quat (wxyz):       {[round(q, 4) for q in quat]}")
        print(f"  target centroid:   {c_orig.round(4).tolist()}")

    labels_data["external_earbuds"] = external_earbuds
    labels_path.write_text(json.dumps(labels_data, indent=2))
    print(f"\n[written] {labels_path}")


if __name__ == "__main__":
    main()
