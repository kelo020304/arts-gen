"""共用 SLat 资产写出：从 export_part_ss_latent_flow_examples 抽出，供 export 与 full-pipeline 复用。"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
import trimesh


_SAM3D_Z_UP_TO_Y_UP = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    dtype=np.float32,
)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _mesh_vertices_y_up(vertices: Any) -> np.ndarray:
    """SAM3D SLat mesh decoder returns Z-up vertices; GLB/web viewer uses Y-up."""
    vertices = np.asarray(_to_numpy(vertices), dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"decoded mesh vertices must have shape (N, 3); got {vertices.shape}")
    return vertices @ _SAM3D_Z_UP_TO_Y_UP


def _mesh_vertex_colors(mesh: Any, vertex_count: int) -> np.ndarray | None:
    """Return RGBA uint8 vertex colors from SAM3D mesh.vertex_attrs when present."""
    attrs = getattr(mesh, "vertex_attrs", None)
    if attrs is None:
        return None
    colors = _to_numpy(attrs)
    if colors.ndim != 2 or colors.shape[0] != vertex_count or colors.shape[1] < 3:
        raise ValueError(
            "decoded mesh vertex_attrs must have shape (num_vertices, >=3); "
            f"got {colors.shape}, vertices={vertex_count}"
        )
    colors = np.asarray(colors[:, :3], dtype=np.float32)
    if colors.size == 0:
        return None
    if colors.max(initial=0.0) <= 1.0:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([colors, alpha], axis=1)


def save_decoded_slat_assets(
    decoded: dict[str, Any],
    asset_dir: Path,
    *,
    mesh_name: str = "mesh.glb",
    gaussian_name: str = "gaussians.ply",
) -> dict[str, str]:
    """把 decode_slat 的输出 {gaussian, mesh} 写盘。返回 {gaussian, mesh} -> 写出的文件名(相对 asset_dir)。
    失败暴露：gaussian 不暴露 save_ply / mesh 缺 vertices|faces / mesh.success=False 直接抛错。"""
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, str] = {}

    gaussian = decoded.get("gaussian")
    if gaussian is not None:
        save_ply = getattr(gaussian, "save_ply", None)
        if not callable(save_ply):
            raise TypeError(f"gaussian 不暴露 save_ply(): {type(gaussian).__name__}")
        save_ply(str(asset_dir / gaussian_name))
        record["gaussian"] = gaussian_name

    mesh = decoded.get("mesh")
    if mesh is not None:
        if not getattr(mesh, "success", True):
            raise ValueError(f"decoded mesh success=False: {asset_dir}")
        vertices = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        if vertices is None or faces is None:
            raise TypeError(f"decoded mesh 缺 vertices/faces: {type(mesh).__name__}")
        vertices = _mesh_vertices_y_up(vertices)
        faces = _to_numpy(faces)
        tri_mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        vertex_colors = _mesh_vertex_colors(mesh, len(tri_mesh.vertices))
        if vertex_colors is not None:
            tri_mesh.visual.vertex_colors = vertex_colors
        tri_mesh.export(str(asset_dir / mesh_name))
        record["mesh"] = mesh_name

    return record
