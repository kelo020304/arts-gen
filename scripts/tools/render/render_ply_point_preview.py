#!/usr/bin/env python3
"""Render lightweight PNG previews for PLY point clouds without OpenGL."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


VIEWS = {
    "front": (270.0, 8.0),
    "iso": (315.0, 24.0),
    "side": (0.0, 8.0),
}


def _load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = PlyData.read(str(path))["vertex"].data
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    names = set(data.dtype.names or ())
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh0 = 0.28209479177387814
        rgb = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1).astype(np.float32)
        rgb = np.clip((rgb * sh0 + 0.5) * 255.0, 0, 255)
    else:
        rgb = np.full((xyz.shape[0], 3), 145, dtype=np.float32)
    return xyz, np.clip(rgb, 0, 255).astype(np.uint8)


def _camera_basis(azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float32)
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


def render(points: np.ndarray, colors: np.ndarray, *, azimuth: float, elevation: float, resolution: int, radius: int) -> Image.Image:
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    pts = points - center[None, :]
    right, up, view_to_camera = _camera_basis(azimuth, elevation)
    x = pts @ right
    y = pts @ up
    depth = pts @ view_to_camera
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

    img = np.full((resolution * resolution, 3), 245, dtype=np.uint8)
    img[flat_s[keep]] = colors[src_s[keep]]
    return Image.fromarray(img.reshape(resolution, resolution, 3), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plys", nargs="+", type=Path)
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

    for ply in args.plys:
        points, colors = _load_ply(ply)
        for name, (az, el) in views.items():
            image = render(points, colors, azimuth=az, elevation=el, resolution=args.resolution, radius=args.radius)
            out = args.out_dir / f"{ply.stem}_{name}.png"
            image.save(out)
            print(out)


if __name__ == "__main__":
    main()
