#!/usr/bin/env python3
"""Render shaded GLB mesh previews with Open3D offscreen rendering."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


VIEWS = {
    "front": (270.0, 8.0),
    "iso": (315.0, 24.0),
    "side": (0.0, 8.0),
}


def _camera_vectors(azimuth_deg: float, elevation_deg: float, distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = np.array(
        [
            distance * math.cos(el) * math.cos(az),
            distance * math.cos(el) * math.sin(az),
            distance * math.sin(el),
        ],
        dtype=np.float64,
    )
    center = np.zeros(3, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return eye, center, up


def _load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"{path}: no triangle mesh geometry")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        colors = np.full((len(mesh.vertices), 3), 0.72, dtype=np.float64)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def _normalizing_transform(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, float]:
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = float(np.max(bbox.get_extent()))
    if extent <= 0:
        raise ValueError("invalid zero-size mesh bounds")
    return center, extent


def render(
    glb: Path,
    *,
    out_dir: Path,
    views: dict[str, tuple[float, float]],
    resolution: int,
    use_vertex_colors: bool = False,
) -> list[Path]:
    mesh = _load_mesh(glb)
    center, extent = _normalizing_transform(mesh)
    mesh = mesh.translate(-center, relative=True)

    renderer = o3d.visualization.rendering.OffscreenRenderer(resolution, resolution)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.MED_SHADOWS,
        (0.35, -0.45, -0.82),
    )

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    if use_vertex_colors and mesh.has_vertex_colors():
        material.base_color = [1.0, 1.0, 1.0, 1.0]
    else:
        material.base_color = [0.72, 0.72, 0.72, 1.0]
    material.base_roughness = 0.78
    material.base_metallic = 0.0
    scene.add_geometry("mesh", mesh, material)

    distance = extent * 2.2
    vertical_fov = 35.0
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, (az, el) in views.items():
        eye, target, up = _camera_vectors(az, el, distance)
        scene.camera.look_at(target, eye, up)
        scene.camera.set_projection(
            vertical_fov,
            1.0,
            max(0.001, distance - extent * 1.5),
            distance + extent * 1.5,
            o3d.visualization.rendering.Camera.FovType.Vertical,
        )
        image = renderer.render_to_image()
        arr = np.asarray(image)
        out = out_dir / f"{glb.stem}_{name}.png"
        Image.fromarray(arr).save(out)
        print(out, flush=True)
        written.append(out)

    return written


def main() -> None:
    os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")
    parser = argparse.ArgumentParser()
    parser.add_argument("glbs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--orbit-count", type=int, default=0)
    parser.add_argument("--orbit-elevation", type=float, default=16.0)
    parser.add_argument(
        "--use-vertex-colors",
        action="store_true",
        help="Render mesh vertex colors instead of overriding with neutral gray.",
    )
    args = parser.parse_args()

    views = dict(VIEWS)
    if args.orbit_count > 0:
        for idx in range(args.orbit_count):
            views[f"orbit_{idx:02d}"] = (360.0 * idx / args.orbit_count, args.orbit_elevation)

    for glb in args.glbs:
        render(
            glb,
            out_dir=args.out_dir,
            views=views,
            resolution=args.resolution,
            use_vertex_colors=bool(args.use_vertex_colors),
        )


if __name__ == "__main__":
    main()
