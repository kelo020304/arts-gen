#!/usr/bin/env python3
"""Render before/after scene GLBs with vertex colors when available."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from scripts.eval.post.holopart_smooth import (
    _fallback_vertex_colors,
    _transfer_vertex_colors,
    render_component,
)


def _scene_parts(path: Path) -> list[tuple[str, trimesh.Trimesh]]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return [(path.stem, loaded)]
    out: list[tuple[str, trimesh.Trimesh]] = []
    for name, geom in loaded.geometry.items():
        mesh = geom if isinstance(geom, trimesh.Trimesh) else trimesh.util.concatenate(tuple(geom.dump()))
        mesh.remove_unreferenced_vertices()
        if len(mesh.vertices) and len(mesh.faces):
            out.append((str(mesh.metadata.get("name") or name), mesh))
    return out


def _concat(parts: list[tuple[str, trimesh.Trimesh]]) -> trimesh.Trimesh:
    return trimesh.util.concatenate([mesh for _name, mesh in parts])


def _concat_colors(parts: list[tuple[str, trimesh.Trimesh]]) -> np.ndarray:
    return np.concatenate([_fallback_vertex_colors(mesh) for _name, mesh in parts], axis=0)


def _camera_for_bounds(mesh: trimesh.Trimesh) -> tuple[Any, Any]:
    import torch

    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    center = bounds.mean(axis=0)
    radius = float(np.linalg.norm(bounds[1] - bounds[0]) * 0.75)
    radius = max(radius, 1.0)
    eye = center + np.asarray([radius * 1.35, radius * 1.15, radius * 1.35], dtype=np.float32)
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up_hint, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    rot = np.stack([right, up, forward], axis=0)
    trans = -rot @ eye
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rot
    extrinsic[:3, 3] = trans
    fov = math.radians(36.0)
    focal = 0.5 / math.tan(fov / 2.0)
    intrinsic = np.asarray([[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return torch.from_numpy(extrinsic), torch.from_numpy(intrinsic)


def _tile(image: Image.Image, label: str, width: int, height: int) -> Image.Image:
    body_h = max(1, int(height) - 30)
    image = image.convert("RGB")
    image.thumbnail((int(width), body_h), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, int(width), 30), fill=(0, 0, 0))
    draw.text((8, 9), label[:96], fill=(255, 255, 255))
    tile.paste(image, ((int(width) - image.width) // 2, 30 + (body_h - image.height) // 2))
    return tile


def render_pair(before_path: Path, after_path: Path, out_path: Path, *, resolution: int) -> dict[str, Any]:
    before_parts = _scene_parts(before_path)
    after_parts = _scene_parts(after_path)
    before = _concat(before_parts)
    after = _concat(after_parts)
    before_colors = _concat_colors(before_parts)
    if len(before_parts) == len(after_parts):
        after_colors = np.concatenate([
            _transfer_vertex_colors(before_mesh, after_mesh)
            for (_bn, before_mesh), (_an, after_mesh) in zip(before_parts, after_parts, strict=True)
        ], axis=0)
    else:
        after_colors = _transfer_vertex_colors(before, after)
    extrinsic, intrinsic = _camera_for_bounds(before)
    before_img = render_component(
        before,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=0,
        vertex_colors=before_colors,
        color_mode="color",
    )
    after_img = render_component(
        after,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=0,
        vertex_colors=after_colors,
        color_mode="color",
    )
    canvas = Image.new("RGB", (int(resolution) * 2, int(resolution) + 30), (255, 255, 255))
    canvas.paste(_tile(before_img, f"before {before_path.name}", int(resolution), int(resolution) + 30), (0, 0))
    canvas.paste(_tile(after_img, f"after {after_path.name}", int(resolution), int(resolution) + 30), (int(resolution), 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "before": str(before_path),
        "after": str(after_path),
        "out": str(out_path),
        "before_parts": len(before_parts),
        "after_parts": len(after_parts),
        "before_faces": int(len(before.faces)),
        "after_faces": int(len(after.faces)),
        "after_watertight_parts": sum(bool(mesh.is_watertight) for _name, mesh in after_parts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=768)
    args = parser.parse_args()
    report = render_pair(args.before, args.after, args.out, resolution=int(args.resolution))
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[scene-render] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
