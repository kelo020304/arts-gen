#!/usr/bin/env python3
"""Rerender saved voxel coords as semi-transparent cube voxel panels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_RESULT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/ss_flow_official_single_multiflow_16obj_0610")


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_coords(path: Path) -> np.ndarray:
    coords = np.load(require_file(path, "coords"))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: expected coords [N,3], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: coords are empty")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords must be integer, got {coords.dtype}")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise ValueError(f"{path}: coords out of [0,64), min={coords.min()} max={coords.max()}")
    return np.ascontiguousarray(coords.astype(np.int64, copy=False))


def downsample_coords(coords: np.ndarray, max_voxels: int) -> np.ndarray:
    if max_voxels <= 0 or coords.shape[0] <= max_voxels:
        return coords
    idx = np.linspace(0, coords.shape[0] - 1, int(max_voxels), dtype=np.int64)
    return np.ascontiguousarray(coords[idx])


def render_voxel_grid(
    coords: np.ndarray,
    *,
    color: tuple[float, float, float],
    alpha: float,
    resolution: int,
    max_voxels: int,
) -> Image.Image:
    import open3d as o3d

    draw_coords = downsample_coords(coords, int(max_voxels))
    points = draw_coords.astype(np.float64) / 63.0 - 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (points.shape[0], 1)))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0 / 64.0)

    renderer = o3d.visualization.rendering.OffscreenRenderer(int(resolution), int(resolution))
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [float(color[0]), float(color[1]), float(color[2]), float(alpha)]
    material.base_roughness = 0.65
    scene.add_geometry("voxels", voxel_grid, material)
    scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
        (0.35, -0.45, -0.82),
    )
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = float(max(bounds.get_extent()))
    if extent <= 0:
        extent = 1.0
    distance = extent * 2.8
    az = math.radians(315.0)
    el = math.radians(24.0)
    eye = center + np.array([
        distance * math.cos(el) * math.cos(az),
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
    ])
    scene.camera.look_at(center, eye, np.array([0.0, 0.0, 1.0]))
    scene.camera.set_projection(
        35.0,
        1.0,
        max(0.001, distance - extent * 2.0),
        distance + extent * 2.0,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )
    image = renderer.render_to_image()
    return Image.fromarray(np.asarray(image)).convert("RGB")


def merge_coords_with_colors(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_keys = {tuple(map(int, row)) for row in gt}
    pred_keys = {tuple(map(int, row)) for row in pred}
    both = sorted(gt_keys & pred_keys)
    gt_only = sorted(gt_keys - pred_keys)
    pred_only = sorted(pred_keys - gt_keys)
    rows: list[tuple[int, int, int]] = []
    colors: list[tuple[float, float, float]] = []
    for row in both:
        rows.append(row)
        colors.append((0.16, 0.62, 0.35))
    for row in gt_only:
        rows.append(row)
        colors.append((0.10, 0.36, 0.82))
    for row in pred_only:
        rows.append(row)
        colors.append((0.88, 0.20, 0.14))
    if not rows:
        raise ValueError("overlay has no voxels")
    return np.asarray(rows, dtype=np.int64), np.asarray(colors, dtype=np.float64)


def render_overlay(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    alpha: float,
    resolution: int,
    max_voxels: int,
) -> Image.Image:
    import open3d as o3d

    coords, colors = merge_coords_with_colors(gt, pred)
    if max_voxels > 0 and coords.shape[0] > max_voxels:
        idx = np.linspace(0, coords.shape[0] - 1, int(max_voxels), dtype=np.int64)
        coords = coords[idx]
        colors = colors[idx]
    points = coords.astype(np.float64) / 63.0 - 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0 / 64.0)

    renderer = o3d.visualization.rendering.OffscreenRenderer(int(resolution), int(resolution))
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [1.0, 1.0, 1.0, float(alpha)]
    material.base_roughness = 0.65
    scene.add_geometry("overlay", voxel_grid, material)
    scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
        (0.35, -0.45, -0.82),
    )
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = float(max(bounds.get_extent()))
    if extent <= 0:
        extent = 1.0
    distance = extent * 2.8
    az = math.radians(315.0)
    el = math.radians(24.0)
    eye = center + np.array([
        distance * math.cos(el) * math.cos(az),
        distance * math.sin(az) * math.cos(el),
        distance * math.sin(el),
    ])
    scene.camera.look_at(center, eye, np.array([0.0, 0.0, 1.0]))
    scene.camera.set_projection(
        35.0,
        1.0,
        max(0.001, distance - extent * 2.0),
        distance + extent * 2.0,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )
    image = renderer.render_to_image()
    return Image.fromarray(np.asarray(image)).convert("RGB")


def make_panel(
    columns: list[tuple[str, Image.Image, dict[str, Any]]],
    out_path: Path,
    *,
    title: str,
) -> None:
    width, height = columns[0][1].size
    title_h = 34
    label_h = 70
    panel = Image.new("RGB", (width * len(columns), height + title_h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), title, fill=(0, 0, 0))
    for idx, (label, image, stats) in enumerate(columns):
        x = idx * width
        text = label
        if "voxels" in stats:
            text += f" vox={int(stats['voxels'])}"
        if "iou" in stats:
            text += f" IoU={float(stats['iou']):.3f}"
        if "precision" in stats and "recall" in stats:
            text += f" P={float(stats['precision']):.3f} R={float(stats['recall']):.3f}"
        draw.text((x + 8, title_h + 8), text, fill=(0, 0, 0))
        panel.paste(image, (x, title_h + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def coord_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_keys = {tuple(map(int, row)) for row in pred}
    gt_keys = {tuple(map(int, row)) for row in gt}
    inter = len(pred_keys & gt_keys)
    union = len(pred_keys | gt_keys)
    return {
        "iou": float(inter / union) if union else 1.0,
        "precision": float(inter / len(pred_keys)) if pred_keys else 0.0,
        "recall": float(inter / len(gt_keys)) if gt_keys else 0.0,
    }


def rerender_sample(sample: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    sample_dir = Path(sample["render_path"]).parent if "render_path" in sample else args.result_dir / f"{sample['object_id']}_angle_{int(sample['angle_idx']):02d}"
    gt_path = sample_dir / "gt_surface_coords.npy"
    pred_path = sample_dir / "pred_multiflow_coords.npy"
    gt = load_coords(gt_path)
    pred = load_coords(pred_path)
    metrics = coord_metrics(pred, gt)
    gt_img = render_voxel_grid(
        gt,
        color=(0.10, 0.36, 0.82),
        alpha=float(args.alpha),
        resolution=int(args.resolution),
        max_voxels=int(args.max_voxels),
    )
    pred_img = render_voxel_grid(
        pred,
        color=(0.88, 0.20, 0.14),
        alpha=float(args.alpha),
        resolution=int(args.resolution),
        max_voxels=int(args.max_voxels),
    )
    overlay_img = render_overlay(
        gt,
        pred,
        alpha=float(args.overlay_alpha),
        resolution=int(args.resolution),
        max_voxels=int(args.max_overlay_voxels),
    )
    out_path = sample_dir / "pred_vs_gt_cube_alpha.png"
    make_panel(
        [
            ("GT cube alpha", gt_img, {"voxels": gt.shape[0]}),
            ("Pred cube alpha", pred_img, {"voxels": pred.shape[0], **metrics}),
            ("Overlay green=hit blue=GT red=Pred", overlay_img, metrics),
        ],
        out_path,
        title=f"{sample['object_id']} angle_{int(sample['angle_idx'])} views={sample.get('view_indices')} cube alpha={args.alpha}",
    )
    print(
        f"[write] {sample['object_id']} angle_{int(sample['angle_idx'])} "
        f"IoU={metrics['iou']:.4f} panel={out_path}",
        flush=True,
    )
    return {
        "object_id": sample["object_id"],
        "angle_idx": int(sample["angle_idx"]),
        "view_indices": sample.get("view_indices"),
        "iou": metrics["iou"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "gt_voxels": int(gt.shape[0]),
        "pred_voxels": int(pred.shape[0]),
        "panel_path": str(out_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--alpha", type=float, default=0.36)
    parser.add_argument("--overlay-alpha", type=float, default=0.42)
    parser.add_argument("--max-voxels", type=int, default=35000)
    parser.add_argument("--max-overlay-voxels", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_dir = args.result_dir.expanduser().resolve()
    summary_path = args.summary or args.result_dir / "summary.json"
    summary = json.loads(require_file(summary_path, "summary").read_text(encoding="utf-8"))
    samples = summary.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{summary_path}: expected non-empty samples list")
    rendered = [rerender_sample(sample, args=args) for sample in samples]
    out_summary = {
        "source_summary": str(summary_path.resolve()),
        "renderer": "Open3D VoxelGrid defaultLitTransparency",
        "resolution": int(args.resolution),
        "alpha": float(args.alpha),
        "overlay_alpha": float(args.overlay_alpha),
        "max_voxels": int(args.max_voxels),
        "max_overlay_voxels": int(args.max_overlay_voxels),
        "samples": rendered,
    }
    out_path = args.result_dir / "cube_alpha_render_summary.json"
    out_path.write_text(json.dumps(out_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] summary={out_path}", flush=True)


if __name__ == "__main__":
    main()
