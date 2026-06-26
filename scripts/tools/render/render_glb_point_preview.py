#!/usr/bin/env python3
"""Render lightweight PNG previews for GLB meshes without Blender/OpenGL."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh


VIEWS = {
    "front": (270.0, 8.0),
    "iso": (315.0, 24.0),
    "side": (0.0, 8.0),
}


def _as_mesh(path: Path) -> trimesh.Trimesh:
    obj = trimesh.load(str(path), force="scene", process=False)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    meshes = []
    for geom in obj.geometry.values():
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices):
            meshes.append(geom)
    if not meshes:
        raise ValueError(f"{path}: no mesh geometry")
    return trimesh.util.concatenate(meshes)


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(mesh.vertices):
        arr = np.asarray(colors[:, :3], dtype=np.float32)
        if arr.max(initial=0.0) <= 1.0:
            arr *= 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    light = np.array([0.35, -0.45, 0.82], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0.0, 1.0)
    shade = 0.35 + 0.65 * shade
    base = np.array([186, 192, 184], dtype=np.float32)
    return np.clip(base[None, :] * shade[:, None], 0, 255).astype(np.uint8)


def _camera_basis(azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float32,
    )
    eye /= np.linalg.norm(eye)
    forward = -eye
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, eye


def render(mesh: trimesh.Trimesh, *, azimuth: float, elevation: float, resolution: int, radius: int) -> Image.Image:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    colors = _vertex_colors(mesh)
    center = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
    verts = verts - center[None, :]

    right, up, view_to_camera = _camera_basis(azimuth, elevation)
    x = verts @ right
    y = verts @ up
    depth = verts @ view_to_camera

    span = max(float(np.ptp(x)), float(np.ptp(y)), 1e-6)
    scale = (resolution * 0.82) / span
    px = np.round(x * scale + resolution * 0.5).astype(np.int32)
    py = np.round(-y * scale + resolution * 0.5).astype(np.int32)

    offsets = [(0, 0)]
    for r in range(1, max(1, radius) + 1):
        offsets.extend((dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r)
    offsets = np.asarray(offsets, dtype=np.int32)

    pxs = px[:, None] + offsets[None, :, 0]
    pys = py[:, None] + offsets[None, :, 1]
    valid = (pxs >= 0) & (pxs < resolution) & (pys >= 0) & (pys < resolution)
    flat = (pys[valid] * resolution + pxs[valid]).astype(np.int64)
    src = np.repeat(np.arange(len(px), dtype=np.int64), len(offsets))[valid.reshape(-1)]
    dep = depth[src]

    order = np.lexsort((dep, flat))
    flat_s = flat[order]
    src_s = src[order]
    keep = np.r_[flat_s[1:] != flat_s[:-1], True]
    chosen_flat = flat_s[keep]
    chosen_src = src_s[keep]

    img = np.full((resolution * resolution, 3), 245, dtype=np.uint8)
    img[chosen_flat] = colors[chosen_src]
    img = img.reshape(resolution, resolution, 3)
    return Image.fromarray(img, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("glbs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--orbit-count", type=int, default=0,
                        help="Also render this many evenly spaced azimuth views.")
    parser.add_argument("--orbit-elevation", type=float, default=16.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    views = dict(VIEWS)
    if args.orbit_count > 0:
        for idx in range(args.orbit_count):
            az = 360.0 * idx / args.orbit_count
            views[f"orbit_{idx:02d}"] = (az, args.orbit_elevation)

    for glb in args.glbs:
        mesh = _as_mesh(glb)
        for name, (az, el) in views.items():
            image = render(mesh, azimuth=az, elevation=el, resolution=args.resolution, radius=args.radius)
            out = args.out_dir / f"{glb.stem}_{name}.png"
            image.save(out)
            print(out)


if __name__ == "__main__":
    main()
