"""Post-hoc alignment of sam3d_objects' splat.ply with mesh.glb.

sam3d's mesh export rotates from Z-up to Y-up (in postprocessing_utils.to_glb,
line 665-666: `vertices = vertices @ [[1,0,0],[0,0,-1],[0,1,0]]`), but its
gaussian PLY export does NOT. That asymmetry leaves splat.ply 90° rotated
relative to mesh.glb. Mesh viewers (Three.js GLTFLoader) and splat viewers
(SuperSplat, 3DGS viewers) both default to Y-up, so the .glb looks correct
out of the box and the .ply does not.

This module applies the same rotation to splat.ply so both end up in Y-up.
Touches positions, normals (if present), AND each gaussian's rotation
quaternion — the last of which the reference allign_assets.py script
in generate_obj_assets/ skipped (which is fine for plain point clouds but
wrong for anisotropic gaussians).
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


# Row-vector convention: v_new = v @ R_ROW   (matches sam3d's to_glb style)
# Column-vector form is R_ROW.T, which is rotation around +X by -90°.
_R_ROW = np.array(
    [[1, 0, 0],
     [0, 0, -1],
     [0, 1, 0]],
    dtype=np.float64,
)

# Same rotation as quaternion (w, x, y, z): around +X by -90°.
_H = math.sqrt(0.5)
_QR_W, _QR_X, _QR_Y, _QR_Z = _H, -_H, 0.0, 0.0


def _rotate_xyz(arr_x, arr_y, arr_z):
    """Apply v @ R_ROW to three same-length numpy arrays. Returns three arrays."""
    stacked = np.stack([arr_x, arr_y, arr_z], axis=1).astype(np.float64)
    rotated = stacked @ _R_ROW
    return rotated[:, 0], rotated[:, 1], rotated[:, 2]


def _qmul_columns(w1, x1, y1, z1, w2, x2, y2, z2):
    """Hamilton product (w,x,y,z) of two batches of quaternions, returns 4 arrays."""
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def align_ply_to_yup(path: Path) -> None:
    """Rotate a sam3d-produced gaussian splat .ply from Z-up into Y-up, in place.

    Rotates: positions (x,y,z), normals (nx,ny,nz if present), and each
    gaussian's orientation quaternion (rot_0..rot_3, treated as w,x,y,z per
    INRIA 3DGS convention).
    """
    path = Path(path)
    plydata = PlyData.read(str(path))
    if "vertex" not in plydata:
        raise ValueError(f"PLY missing vertex element: {path}")

    v = plydata["vertex"].data
    names = v.dtype.names

    if all(n in names for n in ("x", "y", "z")):
        nx, ny, nz = _rotate_xyz(v["x"], v["y"], v["z"])
        v["x"] = nx.astype(v["x"].dtype, copy=False)
        v["y"] = ny.astype(v["y"].dtype, copy=False)
        v["z"] = nz.astype(v["z"].dtype, copy=False)

    if all(n in names for n in ("nx", "ny", "nz")):
        a, b, c = _rotate_xyz(v["nx"], v["ny"], v["nz"])
        v["nx"] = a.astype(v["nx"].dtype, copy=False)
        v["ny"] = b.astype(v["ny"].dtype, copy=False)
        v["nz"] = c.astype(v["nz"].dtype, copy=False)

    if all(n in names for n in ("rot_0", "rot_1", "rot_2", "rot_3")):
        # sam3d stores quaternions as (w, x, y, z) → rot_0..rot_3
        qw = v["rot_0"].astype(np.float64)
        qx = v["rot_1"].astype(np.float64)
        qy = v["rot_2"].astype(np.float64)
        qz = v["rot_3"].astype(np.float64)
        nw, nx2, ny2, nz2 = _qmul_columns(
            _QR_W, _QR_X, _QR_Y, _QR_Z,
            qw, qx, qy, qz,
        )
        v["rot_0"] = nw.astype(v["rot_0"].dtype, copy=False)
        v["rot_1"] = nx2.astype(v["rot_1"].dtype, copy=False)
        v["rot_2"] = ny2.astype(v["rot_2"].dtype, copy=False)
        v["rot_3"] = nz2.astype(v["rot_3"].dtype, copy=False)

    elements = [PlyElement.describe(v, "vertex")]
    for el in plydata.elements:
        if el.name != "vertex":
            elements.append(el)

    with tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=path.parent, suffix=path.suffix,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        PlyData(elements, text=plydata.text).write(str(tmp_path))
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def align_sam3d_outputs(out_dir: Path) -> dict:
    """Align Stage B outputs in `out_dir` to a shared Y-up frame.

    Currently only rotates splat.ply (mesh.glb is already Y-up via sam3d's to_glb).
    Idempotent if called twice — but should NOT be: each call rotates again.
    Caller is responsible for tracking whether a directory has already been aligned.

    Returns a small dict for logging.
    """
    out_dir = Path(out_dir)
    info = {"splat_ply_aligned": False, "mesh_glb_aligned": False}

    ply = out_dir / "splat.ply"
    if ply.exists():
        align_ply_to_yup(ply)
        info["splat_ply_aligned"] = True
    # mesh.glb already Y-up, no-op

    return info
