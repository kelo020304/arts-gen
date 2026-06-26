"""Matplotlib rendering for collision backends that expose hull poses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RenderHull:
    part_name: str
    vertices: np.ndarray
    faces: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray


def render_backend_frame(backend, out_path: Path) -> None:
    iter_hulls = getattr(backend, "iter_render_hulls", None)
    if iter_hulls is None:
        raise TypeError("backend must expose iter_render_hulls() for rendering")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []
    palette = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756",
        "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D",
    ]

    part_to_color: dict[str, str] = {}
    for hull in iter_hulls():
        part_to_color.setdefault(
            hull.part_name,
            palette[len(part_to_color) % len(palette)],
        )
        vertices = np.asarray(hull.vertices, dtype=np.float64)
        faces = np.asarray(hull.faces, dtype=np.int32)
        rotation = np.asarray(hull.rotation, dtype=np.float64)
        translation = np.asarray(hull.translation, dtype=np.float64)
        verts_world = (vertices @ rotation.T) + translation
        all_points.append(verts_world)
        triangles = verts_world[faces]
        ax.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolors=part_to_color[hull.part_name],
                edgecolors="black",
                linewidths=0.1,
                alpha=0.55,
            )
        )

    if all_points:
        pts = np.concatenate(all_points, axis=0)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = (lo + hi) * 0.5
        radius = max(float((hi - lo).max()) * 0.6, 1e-3)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    else:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
