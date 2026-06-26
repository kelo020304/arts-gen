#!/usr/bin/env python3
"""Render SS-flow eval coordinates as flat GT / Pred / Overlay voxel-block panels."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ENC_Y = 64
ENC_X = 64 * 64

COLORS = {
    "gt": (58, 132, 224),
    "pred": (220, 72, 72),
    "gt_only": (58, 132, 224),
    "pred_only": (220, 72, 72),
    "overlap": (46, 184, 102),
}

FACE_SHADE = {
    "x+": 0.90,
    "x-": 0.68,
    "y+": 0.82,
    "y-": 0.72,
    "z+": 1.12,
    "z-": 0.58,
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
    return (x - y) * scale + ox, (x + y) * 0.50 * scale - z * 0.92 * scale + oy


def face_corners(x: int, y: int, z: int, face: str) -> list[tuple[float, float, float]]:
    if face == "x+":
        return [(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)]
    if face == "x-":
        return [(x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z)]
    if face == "y+":
        return [(x, y + 1, z), (x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y + 1, z)]
    if face == "y-":
        return [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)]
    if face == "z+":
        return [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]
    if face == "z-":
        return [(x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z)]
    raise ValueError(face)


def build_visible_faces(keys: set[int], cat_by_key: dict[int, str]) -> list[tuple[float, str, str, tuple[int, int, int]]]:
    faces: list[tuple[float, str, str, tuple[int, int, int]]] = []
    for key in keys:
        x, y, z = decode_key(key)
        cat = cat_by_key[key]
        for face, neighbor_missing in (
            ("x-", x == 0 or key - ENC_X not in keys),
            ("x+", x == 63 or key + ENC_X not in keys),
            ("y-", y == 0 or key - ENC_Y not in keys),
            ("y+", y == 63 or key + ENC_Y not in keys),
            ("z-", z == 0 or key - 1 not in keys),
            ("z+", z == 63 or key + 1 not in keys),
        ):
            if not neighbor_missing:
                continue
            corners = face_corners(x, y, z, face)
            cx = sum(p[0] for p in corners) / 4.0
            cy = sum(p[1] for p in corners) / 4.0
            cz = sum(p[2] for p in corners) / 4.0
            depth = 0.92 * cx + 0.92 * cy + cz
            faces.append((depth, face, cat, (x, y, z)))
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
    projected = [
        ((x - y), (x + y) * 0.50 - z * 0.92)
        for x in (mnx, mxx)
        for y in (mny, mxy)
        for z in (mnz, mxz)
    ]
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


def draw_voxel_panel(
    draw: ImageDraw.ImageDraw,
    *,
    keys: set[int],
    cat_by_key: dict[int, str],
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
    scale: float,
    bbox: tuple[float, float, float, float],
    title: str,
    subtitle: str,
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    draw.rectangle((panel_x, panel_y, panel_x + panel_w - 1, panel_y + panel_h - 1), fill=(255, 255, 255), outline=(220, 220, 220))
    draw.text((panel_x + 12, panel_y + 10), title, fill=(0, 0, 0), font=font)
    draw.text((panel_x + 12, panel_y + 34), subtitle, fill=(20, 20, 20), font=small)
    if not keys:
        draw.text((panel_x + 12, panel_y + 64), "empty", fill=(180, 0, 0), font=small)
        return

    min_px, max_px, min_py, max_py = bbox
    proj_w = max_px - min_px
    proj_h = max_py - min_py
    plot_y = panel_y + 68
    plot_h = panel_h - 82
    ox = panel_x + 18 - min_px * scale + (panel_w - 36 - proj_w * scale) * 0.5
    oy = plot_y + 8 - min_py * scale + (plot_h - 16 - proj_h * scale) * 0.5

    for _, face, cat, (x, y, z) in build_visible_faces(keys, cat_by_key):
        pts = [project(p, scale, ox, oy) for p in face_corners(x, y, z, face)]
        fill = adjust(COLORS[cat], FACE_SHADE[face])
        outline = adjust(fill, 0.55)
        draw.polygon(pts, fill=fill, outline=outline)


def render_tripanel(
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
    union = gt_keys | pred_keys
    if not union:
        raise ValueError("no voxels to render")

    gt_cats = {key: "gt" for key in gt_keys}
    pred_cats = {key: "pred" for key in pred_keys}
    overlay_cats: dict[int, str] = {}
    overlay_cats.update({key: "gt_only" for key in gt_only})
    overlay_cats.update({key: "pred_only" for key in pred_only})
    overlay_cats.update({key: "overlap" for key in overlap})

    font = load_font(17)
    small = load_font(13)
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.text((16, 10), title, fill=(0, 0, 0), font=font)
    draw.text(
        (16, 34),
        (
            f"IoU={float(metrics.get('iou', 0.0)):.4f}  "
            f"P={float(metrics.get('precision', 0.0)):.4f}  "
            f"R={float(metrics.get('recall', 0.0)):.4f}  "
            f"GT={len(gt_keys)} Pred={len(pred_keys)} Overlap={len(overlap)}"
        ),
        fill=(0, 0, 0),
        font=small,
    )
    legend_x = 16
    for label, color in (("GT only", COLORS["gt_only"]), ("Pred only", COLORS["pred_only"]), ("Overlap", COLORS["overlap"])):
        draw.rectangle((legend_x, 58, legend_x + 18, 76), fill=color, outline=(40, 40, 40))
        draw.text((legend_x + 24, 58), label, fill=(0, 0, 0), font=small)
        legend_x += 124

    panel_gap = 14
    outer_margin = 16
    panel_y = 88
    panel_h = height - panel_y - 16
    panel_w = int((width - outer_margin * 2 - panel_gap * 2) / 3)
    bbox = bbox_project_extent(union)
    min_px, max_px, min_py, max_py = bbox
    proj_w = max_px - min_px
    proj_h = max_py - min_py
    scale = min((panel_w - 36) / max(proj_w, 1.0), (panel_h - 98) / max(proj_h, 1.0))
    scale = max(1.2, min(scale, 10.0))

    panels = [
        (outer_margin, gt_keys, gt_cats, "GT", f"voxels={len(gt_keys)}"),
        (outer_margin + panel_w + panel_gap, pred_keys, pred_cats, "Pred", f"voxels={len(pred_keys)}"),
        (
            outer_margin + (panel_w + panel_gap) * 2,
            union,
            overlay_cats,
            "Overlay",
            f"green={len(overlap)} blue={len(gt_only)} red={len(pred_only)}",
        ),
    ]
    for panel_x, keys, cats, panel_title, subtitle in panels:
        draw_voxel_panel(
            draw,
            keys=keys,
            cat_by_key=cats,
            panel_x=panel_x,
            panel_y=panel_y,
            panel_w=panel_w,
            panel_h=panel_h,
            scale=scale,
            bbox=bbox,
            title=panel_title,
            subtitle=subtitle,
            font=font,
            small=small,
        )

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
    render_tripanel(
        gt,
        pred,
        title=f"{object_id} angle_{angle_idx} voxel blocks",
        metrics=metrics,
        out_path=png_path,
        width=width,
        height=height,
    )
    payload = {
        "object_id": object_id,
        "angle_idx": angle_idx,
        "category": data.get("category"),
        "name": data.get("name"),
        "target_part_count": data.get("target_part_count"),
        "view_indices": data.get("view_indices"),
        "render_type": "tripanel_true_voxel_blocks",
        "panels": ["GT", "Pred", "Overlay"],
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
    parser.add_argument("--width", type=int, default=2100)
    parser.add_argument("--height", type=int, default=860)
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
