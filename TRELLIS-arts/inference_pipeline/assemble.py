from __future__ import annotations
from pathlib import Path
import trimesh


def assemble_complete(run_dir, *, part_mesh_names, part_gaussian_names, parts_subdir: str = "parts") -> dict:
    """逐 part 资产拼装成 complete。mesh -> complete.glb（每 part 一个 named geometry，保留标签）；
    gaussian ply -> complete.ply（顶点拼接，property 布局须一致，否则 raise）。坐标不变（canonical grid）。"""
    run_dir = Path(run_dir)
    parts_dir = run_dir / parts_subdir
    out: dict = {}
    if part_mesh_names:
        scene = trimesh.Scene()
        for name in part_mesh_names:
            geom = trimesh.load(str(parts_dir / name), force="mesh", process=False)
            scene.add_geometry(geom, geom_name=Path(name).stem)
        scene.export(str(run_dir / "complete.glb"))
        out["complete_mesh"] = "complete.glb"
    if part_gaussian_names:
        _concat_gaussian_ply([parts_dir / n for n in part_gaussian_names], run_dir / "complete.ply")
        out["complete_gaussian"] = "complete.ply"
    return out


def _concat_gaussian_ply(srcs, dst) -> None:
    from plyfile import PlyData, PlyElement
    import numpy as np
    verts = [PlyData.read(str(p))["vertex"].data for p in srcs]
    if not verts:
        return
    ref_dtype = verts[0].dtype
    for v in verts[1:]:
        if v.dtype != ref_dtype:
            raise ValueError("逐 part gaussian ply 的属性布局不一致，无法拼接")
    merged = np.concatenate(verts)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(str(dst))
