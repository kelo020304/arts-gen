from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AppearanceOutput:
    gs: Any | None
    mesh: Any | None
    ply_path: Path | None = None
    glb_path: Path | None = None
    num_gaussians: int | None = None

    def save(self, output_dir: Path, *, save_mesh: bool = True) -> None:
        """Persist the raw splat.ply / mesh.glb to disk.

        NOTE: writes sam3d's native frames as-is. The splat PLY ends up Z-up
        while the mesh GLB is Y-up (sam3d's `to_glb` rotates internally).
        Callers that need a common frame should call `align_sam3d_outputs()`
        on the output dir afterwards.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.gs is not None:
            ply_path = output_dir / "splat.ply"
            self.gs.save_ply(str(ply_path))
            self.ply_path = ply_path

        if save_mesh and self.mesh is not None:
            glb_path = output_dir / "mesh.glb"
            _export_mesh_glb(self.mesh, glb_path)
            self.glb_path = glb_path


# Same row-vector rotation sam3d's postprocessing_utils.to_glb applies (line 666):
# v @ [[1,0,0],[0,0,-1],[0,1,0]]  →  (x, y, z) → (x, z, -y), i.e. Z-up → Y-up.
# Our export goes through trimesh directly rather than sam3d's to_glb, so we
# apply the same rotation here. align.py's docstring assumes the GLB is Y-up.
_SAM3D_GLB_ROT_ROW = np.array(
    [[1, 0, 0],
     [0, 0, -1],
     [0, 1, 0]],
    dtype=np.float64,
)


def _export_mesh_glb(mesh: Any, glb_path: Path) -> None:
    """Export a slat mesh decoder result to GLB via trimesh, rotated to Y-up."""
    import trimesh

    verts = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    verts = verts @ _SAM3D_GLB_ROT_ROW
    faces = mesh.faces.detach().cpu().numpy()

    vertex_colors = None
    vattrs = getattr(mesh, "vertex_attrs", None)
    if vattrs is not None:
        attrs = vattrs.detach().cpu().numpy()
        if attrs.ndim == 2 and attrs.shape[1] >= 3:
            rgb = np.clip(attrs[:, :3], 0.0, 1.0)
            vertex_colors = (rgb * 255).astype(np.uint8)

    tm = trimesh.Trimesh(
        vertices=verts.astype(np.float32),
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    tm.export(str(glb_path))
