#!/usr/bin/env python3
"""Render three-way colored comparison panels for post-smooth reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.post.holopart_smooth import (  # noqa: E402
    _component_list,
    _load_json,
    _load_mesh,
    _load_render_camera,
    _mesh_vertex_colors_float,
    _safe_name,
    _tile,
    _transfer_vertex_colors,
    render_component,
)


def _after_meshes(report_path: Path) -> dict[str, trimesh.Trimesh]:
    report = _load_json(Path(report_path))
    out: dict[str, trimesh.Trimesh] = {}
    for row in report.get("metrics") or []:
        label = str(row.get("component") or "")
        path = Path(str(row.get("after_mesh") or ""))
        if label and path.is_file():
            out[label] = _load_mesh(path)
    return out


def _component_rows(components: list[Any], xpart: dict[str, trimesh.Trimesh], classic: dict[str, trimesh.Trimesh]) -> list[Any]:
    return [comp for comp in components if comp.label in xpart and comp.label in classic]


def _render_tile(
    mesh: trimesh.Trimesh,
    *,
    label: str,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
    max_faces: int,
    vertex_colors: np.ndarray | None = None,
) -> Image.Image:
    image = render_component(
        mesh,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=int(max_faces),
        vertex_colors=vertex_colors,
        color_mode="color",
    )
    return _tile(image, label, int(resolution), int(resolution) + 30)


def _write_threeway_panels(
    components: list[Any],
    xpart: dict[str, trimesh.Trimesh],
    classic: dict[str, trimesh.Trimesh],
    out_dir: Path,
    *,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
    max_faces: int,
    filter_regex: str | None = None,
    overview_name: str,
) -> None:
    rows = _component_rows(components, xpart, classic)
    if filter_regex:
        pattern = re.compile(filter_regex)
        rows = [comp for comp in rows if pattern.search(comp.label)]
    if not rows:
        return
    panel_dir = out_dir / "threeway_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    overview = Image.new("RGB", (int(resolution) * 3, len(rows) * (int(resolution) + 30)), (255, 255, 255))
    for row_idx, comp in enumerate(rows):
        before = comp.before_mesh
        xmesh = xpart[comp.label]
        cmesh = classic[comp.label]
        xcolors = _transfer_vertex_colors(before, xmesh)
        ccolors = _transfer_vertex_colors(before, cmesh)
        tiles = [
            _render_tile(
                before,
                label=f"before {comp.label}",
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                resolution=int(resolution),
                max_faces=int(max_faces),
            ),
            _render_tile(
                xmesh,
                label="X-Part corrected",
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                resolution=int(resolution),
                max_faces=int(max_faces),
                vertex_colors=xcolors,
            ),
            _render_tile(
                cmesh,
                label="classic trimesh_fill",
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                resolution=int(resolution),
                max_faces=int(max_faces),
                vertex_colors=ccolors,
            ),
        ]
        canvas = Image.new("RGB", (int(resolution) * 3, int(resolution) + 30), (255, 255, 255))
        for col, tile in enumerate(tiles):
            canvas.paste(tile, (col * int(resolution), 0))
            overview.paste(tile, (col * int(resolution), row_idx * (int(resolution) + 30)))
        canvas.save(panel_dir / f"{_safe_name(comp.label)}__before_xpart_classic.png")
    overview.save(out_dir / overview_name)


def _exploded_mesh(
    components: list[Any],
    meshes_by_label: dict[str, trimesh.Trimesh],
    *,
    source: str,
    scale: float,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    selected = [comp for comp in components if comp.label in meshes_by_label]
    if not selected:
        raise ValueError("no meshes to explode")
    centers = np.stack([np.asarray(meshes_by_label[comp.label].bounds, dtype=np.float64).mean(axis=0) for comp in selected])
    global_center = centers.mean(axis=0)
    extent = float(np.max(np.ptp(centers, axis=0))) if len(selected) > 1 else 1.0
    if not np.isfinite(extent) or extent <= 1.0e-6:
        extent = 1.0
    exploded: list[trimesh.Trimesh] = []
    colors: list[np.ndarray] = []
    for comp, center in zip(selected, centers, strict=True):
        mesh = meshes_by_label[comp.label].copy()
        if source == "before":
            color = _mesh_vertex_colors_float(comp.before_mesh)
            if color is None:
                color = _transfer_vertex_colors(comp.before_mesh, comp.before_mesh)
        else:
            color = _transfer_vertex_colors(comp.before_mesh, mesh)
        direction = center - global_center
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-6:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            norm = 1.0
        mesh.apply_translation((direction / norm) * float(scale) * extent)
        exploded.append(mesh)
        colors.append(np.asarray(color, dtype=np.float32))
    merged = trimesh.util.concatenate(exploded)
    merged_colors = np.concatenate(colors, axis=0)
    return merged, merged_colors


def _write_exploded_threeway(
    components: list[Any],
    xpart: dict[str, trimesh.Trimesh],
    classic: dict[str, trimesh.Trimesh],
    out_path: Path,
    *,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
    max_faces: int,
) -> None:
    before_map = {comp.label: comp.before_mesh for comp in components}
    specs = [
        ("before exploded", before_map, "before"),
        ("X-Part corrected exploded", xpart, "after"),
        ("classic trimesh_fill exploded", classic, "after"),
    ]
    tiles: list[Image.Image] = []
    for label, meshes, source in specs:
        mesh, colors = _exploded_mesh(components, meshes, source=source, scale=0.35)
        tiles.append(
            _render_tile(
                mesh,
                label=label,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                resolution=int(resolution),
                max_faces=int(max_faces),
                vertex_colors=colors,
            )
        )
    canvas = Image.new("RGB", (int(resolution) * 3, int(resolution) + 30), (255, 255, 255))
    for col, tile in enumerate(tiles):
        canvas.paste(tile, (col * int(resolution), 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_dir, _whole_voxel, components, meta = _component_list(Path(args.summary).resolve())
    summary = meta["summary"]
    xpart = _after_meshes(Path(args.xpart_report).resolve())
    classic = _after_meshes(Path(args.classic_report).resolve())
    rows = _component_rows(components, xpart, classic)
    extrinsic, intrinsic = _load_render_camera(summary, render_view=int(args.render_view))
    _write_threeway_panels(
        components,
        xpart,
        classic,
        out_dir,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(args.resolution),
        max_faces=int(args.max_faces),
        overview_name="before_xpart_classic_overview.png",
    )
    _write_threeway_panels(
        components,
        xpart,
        classic,
        out_dir,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(args.zoom_resolution),
        max_faces=int(args.max_faces),
        filter_regex=str(args.zoom_regex),
        overview_name="before_xpart_classic_zoom.png",
    )
    _write_exploded_threeway(
        rows,
        xpart,
        classic,
        out_dir / "before_xpart_classic_exploded.png",
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(args.resolution),
        max_faces=int(args.max_faces),
    )
    manifest = {
        "summary": str(Path(args.summary).resolve()),
        "xpart_report": str(Path(args.xpart_report).resolve()),
        "classic_report": str(Path(args.classic_report).resolve()),
        "out_dir": str(out_dir),
        "component_count": len(rows),
        "overview": str(out_dir / "before_xpart_classic_overview.png"),
        "zoom": str(out_dir / "before_xpart_classic_zoom.png"),
        "exploded": str(out_dir / "before_xpart_classic_exploded.png"),
        "panels": str(out_dir / "threeway_panels"),
    }
    (out_dir / "compare_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--xpart-report", type=Path, required=True)
    parser.add_argument("--classic-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--zoom-resolution", type=int, default=768)
    parser.add_argument("--render-view", type=int, default=0)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--zoom-regex", default="handle")
    return parser.parse_args()


def main() -> int:
    manifest = render(parse_args())
    print(f"[compare-post-tools] wrote {manifest['out_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
