#!/usr/bin/env python3
"""Render SS-flow eval coordinates as true voxel-cube overlays in a flat folder."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ENC_Y = 64
ENC_X = 64 * 64

COLORS = {
    "gt_only": (58, 132, 224),
    "pred_only": (220, 72, 72),
    "overlap": (46, 184, 102),
}

FACE_SHADE = {
    "x+": 0.92,
    "y-": 0.78,
    "z+": 1.12,
}


def encode_coords(coords: np.ndarray) -> set[int]:
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must be [N,3], got {arr.shape}")
    return set((arr[:, 0] * ENC_X + arr[:, 1] * ENC_Y + arr[:, 2]).astype(np.int64).tolist())


def decode_key(key: int) -> tuple[int, int, int]:
    x = key // ENC_X
    rem = key - x * ENC_X
    y = rem // ENC_Y
    z = rem - y * ENC_Y
    return int(x), int(y), int(z)


def adjust(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(c * factor)))) for c in rgb)


def project(point: tuple[float, float, float], scale: float, ox: float, oy: float) -> tuple[float, float]:
    x, y, z = point
    px = (x - y) * scale + ox
    py = (x + y) * 0.50 * scale - z * 0.92 * scale + oy
    return px, py


def face_corners(x: int, y: int, z: int, face: str) -> list[tuple[float, float, float]]:
    if face == "x+":
        return [(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)]
    if face == "y-":
        return [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)]
    if face == "z+":
        return [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]
    raise ValueError(face)


def build_visible_faces(all_keys: set[int], cat_by_key: dict[int, str]) -> list[tuple[float, str, str, tuple[int, int, int]]]:
    faces: list[tuple[float, str, str, tuple[int, int, int]]] = []
    for key in all_keys:
        x, y, z = decode_key(key)
        cat = cat_by_key[key]
        if x < 63 and key + ENC_X not in all_keys:
            faces.append((x - y + z + 0.35, "x+", cat, (x, y, z)))
        if y > 0 and key - ENC_Y not in all_keys:
            faces.append((x - y + z + 0.20, "y-", cat, (x, y, z)))
        if z < 63 and key + 1 not in all_keys:
            faces.append((x - y + z + 0.50, "z+", cat, (x, y, z)))
    faces.sort(key=lambda item: item[0])
    return faces


def bbox_project_extent(all_keys: set[int]) -> tuple[float, float, float, float]:
    xs: list[int] = []
    ys: list[int] = []
    zs: list[int] = []
    for key in all_keys:
        x, y, z = decode_key(key)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    mnx, mxx = min(xs), max(xs) + 1
    mny, mxy = min(ys), max(ys) + 1
    mnz, mxz = min(zs), max(zs) + 1
    corners = [
        (x, y, z)
        for x in (mnx, mxx)
        for y in (mny, mxy)
        for z in (mnz, mxz)
    ]
    projected = [((x - y), (x + y) * 0.50 - z * 0.92) for x, y, z in corners]
    px = [p[0] for p in projected]
    py = [p[1] for p in projected]
    return min(px), max(px), min(py), max(py)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_overlay(
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    *,
    title: str,
    metrics: dict[str, Any],
    out_path: Path,
    width: int,
    height: int,
) -> None:
    gt_keys = encode_coords(gt_coords)
    pred_keys = encode_coords(pred_coords)
    overlap = gt_keys & pred_keys
    gt_only = gt_keys - pred_keys
    pred_only = pred_keys - gt_keys
    all_keys = gt_keys | pred_keys
    if not all_keys:
        raise ValueError("no voxels to render")

    cat_by_key: dict[int, str] = {}
    for key in gt_only:
        cat_by_key[key] = "gt_only"
    for key in pred_only:
        cat_by_key[key] = "pred_only"
    for key in overlap:
        cat_by_key[key] = "overlap"

    header_h = 92
    margin = 28
    min_px, max_px, min_py, max_py = bbox_project_extent(all_keys)
    proj_w = max_px - min_px
    proj_h = max_py - min_py
    scale = min((width - margin * 2) / max(proj_w, 1.0), (height - header_h - margin * 2) / max(proj_h, 1.0))
    scale = max(2.0, min(scale, 12.0))
    ox = margin - min_px * scale + (width - margin * 2 - proj_w * scale) * 0.5
    oy = header_h + margin - min_py * scale + (height - header_h - margin * 2 - proj_h * scale) * 0.5

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = load_font(16)
    small = load_font(13)
    draw.text((14, 10), title, fill=(0, 0, 0), font=font)
    draw.text(
        (14, 34),
        (
            f"IoU={float(metrics.get('iou', 0.0)):.4f}  "
            f"P={float(metrics.get('precision', 0.0)):.4f}  "
            f"R={float(metrics.get('recall', 0.0)):.4f}  "
            f"GT={len(gt_keys)} Pred={len(pred_keys)} Overlap={len(overlap)}"
        ),
        fill=(0, 0, 0),
        font=small,
    )
    legend = [("GT only", COLORS["gt_only"]), ("Pred only", COLORS["pred_only"]), ("Overlap", COLORS["overlap"])]
    lx = 14
    for label, color in legend:
        draw.rectangle((lx, 62, lx + 18, 80), fill=color, outline=(40, 40, 40))
        draw.text((lx + 24, 62), label, fill=(0, 0, 0), font=small)
        lx += 115

    faces = build_visible_faces(all_keys, cat_by_key)
    for _, face, cat, (x, y, z) in faces:
        corners = face_corners(x, y, z, face)
        pts = [project(p, scale, ox, oy) for p in corners]
        fill = adjust(COLORS[cat], FACE_SHADE[face])
        outline = adjust(fill, 0.62)
        draw.polygon(pts, fill=fill, outline=outline)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def convert_one(summary_path: Path, out_dir: Path, width: int, height: int) -> tuple[str, bool, str]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    sample_dir = summary_path.parent
    object_id = str(data["object_id"])
    angle_idx = int(data["angle_idx"])
    stem = f"{object_id}_angle{angle_idx:02d}"
    gt_path = sample_dir / "gt_surface_coords.npy"
    pred_path = sample_dir / "pred_multiflow_coords.npy"
    gt = np.load(gt_path)
    pred = np.load(pred_path)
    metrics = data.get("metrics_vs_gt_surface", {})
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"
    title = f"{object_id} angle_{angle_idx} true voxel overlay"
    render_overlay(gt, pred, title=title, metrics=metrics, out_path=png_path, width=width, height=height)
    payload = {
        "object_id": object_id,
        "angle_idx": angle_idx,
        "category": data.get("category"),
        "name": data.get("name"),
        "target_part_count": data.get("target_part_count"),
        "view_indices": data.get("view_indices"),
        "render_type": "true_voxel_cube_overlay",
        "color_legend": {
            "blue": "GT only",
            "red": "Pred only",
            "green": "GT and Pred overlap",
        },
        "metrics_vs_gt_surface": metrics,
        "pred_stats": data.get("pred_stats"),
        "gt_latent_decoded_stats": data.get("gt_latent_decoded_stats"),
        "source_summary": str(summary_path.resolve()),
        "source_gt_surface_coords": str(gt_path.resolve()),
        "source_pred_multiflow_coords": str(pred_path.resolve()),
        "png_path": str(png_path.resolve()),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stem, True, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--width", type=int, default=1180)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = sorted(p for p in args.src_root.glob("shard*/*/summary.json"))
    if args.limit > 0:
        summaries = summaries[: args.limit]
    if not summaries:
        raise RuntimeError(f"no sample summaries found under {args.src_root}/shard*/*/summary.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = len(summaries)
    done = 0
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(convert_one, path, args.out_dir, int(args.width), int(args.height)): path
            for path in summaries
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                stem, _, _ = future.result()
                done += 1
                if done == 1 or done % 50 == 0 or done == total:
                    print(f"[render] {done}/{total} {stem}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append((str(path), repr(exc)))
                print(f"[render][ERROR] {path}: {exc!r}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} render failures: {failures[:5]}")
    print(f"[done] wrote {total} png/json pairs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
