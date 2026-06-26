#!/usr/bin/env python3
"""Compare SAM3D stage4 PLY color statistics against TRELLIS baseline PLY/render."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData


SH_C0 = 0.2820948


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ply_rgb(path: Path) -> tuple[np.ndarray, str]:
    vertex = PlyData.read(str(path.resolve()))["vertex"].data
    names = vertex.dtype.names or ()
    if all(name in names for name in ("red", "green", "blue")):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0
        return np.clip(rgb, 0, 1), "rgb"
    if all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        fdc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        rgb = np.clip(0.5 + SH_C0 * fdc, 0, 1)
        return rgb, "f_dc_to_rgb"
    raise RuntimeError(f"{path} missing RGB/f_dc color properties; props={names}")


def stats_rgb(rgb: np.ndarray) -> dict:
    if rgb.ndim != 2 or rgb.shape[1] != 3 or rgb.shape[0] == 0:
        raise ValueError(f"rgb must be [N,3], got {rgb.shape}")
    return {
        "count": int(rgb.shape[0]),
        "mean_rgb": [float(x) for x in rgb.mean(axis=0).tolist()],
        "min_rgb": [float(x) for x in rgb.min(axis=0).tolist()],
        "max_rgb": [float(x) for x in rgb.max(axis=0).tolist()],
    }


def input_stats(image_path: Path, mask_path: Path | None) -> dict:
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    else:
        rgba = Image.open(image_path).convert("RGBA")
        mask = np.asarray(rgba.getchannel("A")) > 0
    vals = img[mask] if mask.any() else img.reshape(-1, 3)
    out = stats_rgb(vals.reshape(-1, 3))
    out["mask_pixels"] = int(vals.shape[0])
    return out


def swatch_image(rgb: np.ndarray, label: str, *, size: tuple[int, int] = (256, 128)) -> Image.Image:
    mean = np.clip(rgb.mean(axis=0), 0, 1)
    color = tuple(int(round(float(v) * 255)) for v in mean)
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, size[0], 26), fill=(0, 0, 0))
    draw.text((6, 7), label, fill=(255, 255, 255))
    draw.text((6, size[1] - 22), f"mean RGB {color}", fill=(255, 255, 255))
    return img


def resize_keep(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def make_panel(path: Path, cells: list[tuple[str, Image.Image]], *, cell_size: tuple[int, int] = (512, 512)) -> None:
    label_h = 30
    w, h = cell_size
    canvas = Image.new("RGB", (len(cells) * w, h + label_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(cells):
        x = idx * w
        draw.text((x + 8, 9), label, fill=(255, 255, 255))
        canvas.paste(resize_keep(img, cell_size), (x, label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--input-mask", type=Path)
    parser.add_argument("--sam3d-parts-dir", type=Path, required=True)
    parser.add_argument("--trellis-ply", type=Path, required=True)
    parser.add_argument("--trellis-render", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    input_image = require_file(args.input_image, "input image")
    input_mask = require_file(args.input_mask, "input mask") if args.input_mask else None
    sam3d_parts = require_dir(args.sam3d_parts_dir, "SAM3D parts dir")
    trellis_ply = require_file(args.trellis_ply, "TRELLIS baseline PLY")
    trellis_render = require_file(args.trellis_render, "TRELLIS baseline render")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = {
        "input": input_stats(input_image, input_mask),
        "sam3d": {},
        "trellis_baseline": {},
        "paths": {
            "input_image": str(input_image),
            "input_mask": str(input_mask) if input_mask else None,
            "sam3d_parts_dir": str(sam3d_parts),
            "trellis_ply": str(trellis_ply),
            "trellis_render": str(trellis_render),
        },
    }
    sam3d_rgbs = []
    for ply in sorted(sam3d_parts.glob("*.ply")):
        rgb, source = ply_rgb(ply)
        rec = stats_rgb(rgb)
        rec["color_source"] = source
        records["sam3d"][ply.name] = rec
        sam3d_rgbs.append(rgb)
    if not sam3d_rgbs:
        raise RuntimeError(f"no SAM3D .ply files found in {sam3d_parts}")
    sam3d_all = np.concatenate(sam3d_rgbs, axis=0)
    records["sam3d"]["ALL"] = stats_rgb(sam3d_all)

    trellis_rgb, trellis_source = ply_rgb(trellis_ply)
    trellis_rec = stats_rgb(trellis_rgb)
    trellis_rec["color_source"] = trellis_source
    records["trellis_baseline"]["assembled_shared_grid.ply"] = trellis_rec

    panel_path = out_dir / "100058_sam3d_vs_trellis_color_panel.png"
    make_panel(
        panel_path,
        [
            ("input RGB", Image.open(input_image).convert("RGB")),
            ("SAM3D PLY mean color", swatch_image(sam3d_all, "SAM3D all PLY")),
            ("TRELLIS PLY mean color", swatch_image(trellis_rgb, "TRELLIS PLY")),
            ("TRELLIS rendered baseline", Image.open(trellis_render).convert("RGB")),
        ],
    )
    records["outputs"] = {"panel": str(panel_path), "report": str(out_dir / "report.json")}
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[input] mean_rgb={records['input']['mean_rgb']} mask_pixels={records['input']['mask_pixels']}")
    print(f"[sam3d] ALL mean_rgb={records['sam3d']['ALL']['mean_rgb']} count={records['sam3d']['ALL']['count']}")
    for name, rec in records["sam3d"].items():
        if name == "ALL":
            continue
        print(f"[sam3d] {name} mean_rgb={rec['mean_rgb']} count={rec['count']}")
    print(f"[trellis] assembled_shared_grid.ply mean_rgb={trellis_rec['mean_rgb']} count={trellis_rec['count']}")
    print(f"[done] panel={panel_path}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()
