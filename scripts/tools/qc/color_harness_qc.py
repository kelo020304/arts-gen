#!/usr/bin/env python3
"""Color-path diagnostics for TRELLIS Gaussian render/save harness."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import diff_gaussian_rasterization  # noqa: E402,F401
import inference  # noqa: E402

sys.modules.pop("trellis.renderers", None)
gaussian_render_mod = importlib.import_module("trellis.renderers.gaussian_render")
GaussianRenderer = gaussian_render_mod.GaussianRenderer


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


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def foreground_mask_from_rgba(image: Image.Image) -> np.ndarray:
    if image.mode == "RGBA":
        return np.asarray(image.getchannel("A")) > 0
    rgb = np.asarray(image.convert("RGB"))
    return rgb.max(axis=2) > 8


def channel_stats_from_image(image: Image.Image, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if mask is not None:
        if mask.shape != arr.shape[:2]:
            raise RuntimeError(f"mask shape {mask.shape} does not match image {arr.shape[:2]}")
        vals = arr[mask]
    else:
        vals = arr.reshape(-1, 3)
    if vals.shape[0] == 0:
        raise RuntimeError("cannot compute channel stats on empty mask")
    return {
        "pixels": int(vals.shape[0]),
        "mean_rgb": [float(x) for x in vals.mean(axis=0).tolist()],
        "min_rgb": [float(x) for x in vals.min(axis=0).tolist()],
        "max_rgb": [float(x) for x in vals.max(axis=0).tolist()],
    }


def load_camera_matrices(camera_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    import utils3d

    payload = json.loads(require_file(camera_path, "camera_transforms").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"camera_transforms.frames must be a non-empty list: {camera_path}")
    extrinsics = []
    intrinsics = []
    for idx, frame in enumerate(frames):
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device="cuda")
        if tuple(c2w.shape) != (4, 4):
            raise RuntimeError(f"camera frame {idx} transform_matrix must be 4x4: {camera_path}")
        c2w = c2w.clone()
        c2w[:3, 1:3] *= -1
        extrinsics.append(torch.inverse(c2w))
        fov = torch.tensor(float(frame["camera_angle_x"]), dtype=torch.float32, device="cuda")
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    return (
        torch.stack(extrinsics),
        torch.stack(intrinsics),
        {
            "resolution": int(payload.get("resolution", 512)),
            "total_views": len(frames),
        },
    )


def make_renderer(resolution: int, kernel_size: float = 0.1) -> Any:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = float(kernel_size)
    return renderer


def load_ply_gaussian(ply_path: Path, decoder_ckpt: Path):
    decoder = inference._load_slat_vae_decoder(str(decoder_ckpt.resolve()))
    from trellis.representations.gaussian import Gaussian

    rep_config = getattr(decoder, "rep_config", None)
    if not isinstance(rep_config, dict):
        raise RuntimeError("SLat Gaussian decoder missing rep_config")
    gaussian = Gaussian(
        sh_degree=0,
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        mininum_kernel_size=float(rep_config["3d_filter_kernel_size"]),
        scaling_bias=float(rep_config["scaling_bias"]),
        opacity_bias=float(rep_config["opacity_bias"]),
        scaling_activation=str(rep_config["scaling_activation"]),
    )
    gaussian.load_ply(str(ply_path.resolve()), transform=None)
    return gaussian, decoder


@torch.no_grad()
def render_one(gaussian: Any, renderer: Any, extrinsic: torch.Tensor, intrinsic: torch.Tensor, colors: torch.Tensor | None = None) -> torch.Tensor:
    out = renderer.render(gaussian, extrinsic, intrinsic, colors_overwrite=colors)
    if "color" not in out:
        raise RuntimeError("GaussianRenderer output missing color")
    color = out["color"].detach().float().cpu().clamp(0, 1)
    if color.ndim != 3 or color.shape[0] != 3:
        raise RuntimeError(f"rendered color must be [3,H,W], got {tuple(color.shape)}")
    return color


def save_labeled_grid(path: Path, cells: list[tuple[str, Image.Image]]) -> None:
    if not cells:
        raise ValueError("save_labeled_grid requires at least one cell")
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = cells[0][1].size
    label_h = 28
    canvas = Image.new("RGB", (w * len(cells), h + label_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(cells):
        x = idx * w
        draw.rectangle((x, 0, x + w, label_h), fill=(0, 0, 0))
        draw.text((x + 6, 7), label, fill=(255, 255, 255))
        canvas.paste(image.convert("RGB"), (x, label_h))
    canvas.save(path)


def find_pinkest_view(data_root: Path, object_id: str, angle_idx: int) -> dict[str, Any]:
    rgb_root = require_dir(data_root / "renders" / object_id / f"angle_{angle_idx}" / "rgb", "RGB render root")
    best: dict[str, Any] | None = None
    for path in sorted(rgb_root.glob("view_*.png")):
        view_idx = int(path.stem.split("_")[-1])
        image = Image.open(path).convert("RGBA")
        mask = foreground_mask_from_rgba(image)
        stats = channel_stats_from_image(image, mask)
        mean = np.asarray(stats["mean_rgb"], dtype=np.float32)
        pink_score = float(mean[0] - mean[1] + 0.5 * (mean[2] - mean[1]))
        rec = {
            "view": view_idx,
            "path": str(path.resolve()),
            "pink_score": pink_score,
            "foreground_stats": stats,
        }
        if best is None or pink_score > float(best["pink_score"]):
            best = rec
    if best is None:
        raise RuntimeError(f"no view_*.png under {rgb_root}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--object-id", default="100058")
    parser.add_argument("--angle-idx", type=int, default=0)
    parser.add_argument("--view-idx", type=int, default=0)
    parser.add_argument("--decoder-ckpt", type=Path, required=True)
    parser.add_argument("--assembled-ply", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--skip-overwrites", action="store_true")
    parser.add_argument("--overwrite-color", choices=("red", "green", "blue", "white"), default=None)
    args = parser.parse_args()

    data_root = require_dir(args.data_root, "DATA_ROOT")
    decoder_ckpt = require_file(args.decoder_ckpt, "SLat Gaussian decoder ckpt")
    assembled_ply = require_file(args.assembled_ply, "assembled gaussian ply")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = require_file(
        data_root / "renders" / args.object_id / f"angle_{args.angle_idx}" / "rgb" / f"view_{args.view_idx}.png",
        "GT RGB view",
    )
    gt_rgba = Image.open(rgb_path).convert("RGBA")
    gt_rgb = gt_rgba.convert("RGB")
    gt_mask = foreground_mask_from_rgba(gt_rgba)
    gt_resaved = tensor_to_image(pil_rgb_to_tensor(gt_rgb))
    gt_resaved_path = out_dir / "gt_resave_same_function.png"
    gt_resaved.save(gt_resaved_path)
    gt_panel_path = out_dir / "gt_resave_panel.png"
    save_labeled_grid(gt_panel_path, [("GT original", gt_rgb), ("same save fn", gt_resaved)])

    diff = np.asarray(gt_rgb, dtype=np.int16) - np.asarray(gt_resaved, dtype=np.int16)
    save_absdiff = np.abs(diff).astype(np.uint8)
    Image.fromarray(save_absdiff, mode="RGB").save(out_dir / "gt_resave_absdiff.png")

    extrinsics, intrinsics, camera_meta = load_camera_matrices(
        data_root / "renders" / args.object_id / f"angle_{args.angle_idx}" / "camera_transforms.json"
    )
    gaussian, decoder = load_ply_gaussian(assembled_ply, decoder_ckpt)
    rep_config = getattr(decoder, "rep_config", {})
    renderer = make_renderer(camera_meta["resolution"], kernel_size=float(rep_config.get("2d_filter_kernel_size", 0.1)))
    view = int(args.view_idx)
    rendered = render_one(gaussian, renderer, extrinsics[view], intrinsics[view])
    render_img = tensor_to_image(rendered)
    render_path = out_dir / "assembled_original_same_function.png"
    render_img.save(render_path)

    n = int(gaussian.get_xyz.shape[0])
    red = torch.zeros((n, 3), dtype=torch.float32, device="cuda")
    red[:, 0] = 1.0
    green = torch.zeros((n, 3), dtype=torch.float32, device="cuda")
    green[:, 1] = 1.0
    blue = torch.zeros((n, 3), dtype=torch.float32, device="cuda")
    blue[:, 2] = 1.0
    white = torch.ones((n, 3), dtype=torch.float32, device="cuda")
    color_by_name = {
        "red": red,
        "green": green,
        "blue": blue,
        "white": white,
    }
    override_cells: list[tuple[str, Image.Image]] = [("original SH/DC", render_img)]
    override_stats = {}
    if args.skip_overwrites and args.overwrite_color is not None:
        raise ValueError("--skip-overwrites and --overwrite-color are mutually exclusive")
    if args.overwrite_color is not None:
        selected = [(f"{args.overwrite_color} overwrite", color_by_name[args.overwrite_color])]
    elif args.skip_overwrites:
        selected = []
    else:
        selected = [
            ("red overwrite", red),
            ("green overwrite", green),
            ("blue overwrite", blue),
            ("white overwrite", white),
        ]
    for label, colors in selected:
        frame = render_one(gaussian, renderer, extrinsics[view], intrinsics[view], colors=colors.contiguous())
        img = tensor_to_image(frame)
        img.save(out_dir / f"{label.replace(' ', '_')}.png")
        override_cells.append((label, img))
        mask = np.asarray(img.convert("RGB")).max(axis=2) > 8
        override_stats[label] = channel_stats_from_image(img, mask)
    override_panel_path = out_dir / "renderer_channel_overwrite_panel.png"
    if len(override_cells) > 1:
        save_labeled_grid(override_panel_path, override_cells)
    else:
        override_panel_path = None

    features = gaussian._features_dc.detach().float().cpu()
    get_features = gaussian.get_features.detach().float().cpu()
    render_mask = np.asarray(render_img.convert("RGB")).max(axis=2) > 8
    pinkest = find_pinkest_view(data_root, args.object_id, args.angle_idx)
    report = {
        "python": sys.executable,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "official_renderer": {
            "gaussian_renderer_module": gaussian_render_mod.__file__,
            "diff_gaussian_module": diff_gaussian_rasterization.__file__,
            "used_gsplat": False,
        },
        "inputs": {
            "data_root": str(data_root),
            "object_id": args.object_id,
            "angle_idx": int(args.angle_idx),
            "view_idx": view,
            "gt_rgb": str(rgb_path.resolve()),
            "assembled_ply": str(assembled_ply.resolve()),
            "decoder_ckpt": str(decoder_ckpt.resolve()),
        },
        "gt_resave_same_function": {
            "original_foreground_stats": channel_stats_from_image(gt_rgb, gt_mask),
            "resaved_foreground_stats": channel_stats_from_image(gt_resaved, gt_mask),
            "max_abs_diff": int(np.abs(diff).max()),
            "mean_abs_diff": float(np.abs(diff).mean()),
            "panel": str(gt_panel_path.resolve()),
            "resaved_png": str(gt_resaved_path.resolve()),
        },
        "pinkest_view_for_this_object_angle": pinkest,
        "renderer_channel_overwrite": {
            "panel": str(override_panel_path.resolve()) if override_panel_path is not None else None,
            "stats": override_stats,
        },
        "assembled_render_same_function": {
            "path": str(render_path.resolve()),
            "foreground_stats": channel_stats_from_image(render_img, render_mask),
        },
        "gaussian_feature_stats": {
            "_features_dc_shape": list(features.shape),
            "get_features_shape": list(get_features.shape),
            "_features_dc_mean": [float(x) for x in features.reshape(-1, 3).mean(dim=0).tolist()],
            "_features_dc_min": [float(x) for x in features.reshape(-1, 3).min(dim=0).values.tolist()],
            "_features_dc_max": [float(x) for x in features.reshape(-1, 3).max(dim=0).values.tolist()],
            "note": "For sh_degree=0, diff_gaussian consumes get_features as SH DC coefficients; renderer converts SH to RGB internally.",
        },
        "outputs": {
            "gt_resave_panel": str(gt_panel_path.resolve()),
            "gt_resave_absdiff": str((out_dir / "gt_resave_absdiff.png").resolve()),
            "renderer_channel_overwrite_panel": str(override_panel_path.resolve()) if override_panel_path is not None else None,
            "assembled_original_same_function": str(render_path.resolve()),
            "report_json": str((out_dir / "report.json").resolve()),
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[python] {sys.executable}", flush=True)
    print(f"[torch] {torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"[renderer] {gaussian_render_mod.__file__}", flush=True)
    print(f"[diff_gaussian] {diff_gaussian_rasterization.__file__}", flush=True)
    print(f"[gt-resave] max_abs_diff={report['gt_resave_same_function']['max_abs_diff']} mean_abs_diff={report['gt_resave_same_function']['mean_abs_diff']:.6f}", flush=True)
    print(f"[gt-stats] original_fg_mean_rgb={report['gt_resave_same_function']['original_foreground_stats']['mean_rgb']}", flush=True)
    print(f"[gt-stats] resaved_fg_mean_rgb={report['gt_resave_same_function']['resaved_foreground_stats']['mean_rgb']}", flush=True)
    print(f"[pinkest-view] view={pinkest['view']} score={pinkest['pink_score']:.6f} mean_rgb={pinkest['foreground_stats']['mean_rgb']}", flush=True)
    print(f"[render-stats] assembled_fg_mean_rgb={report['assembled_render_same_function']['foreground_stats']['mean_rgb']}", flush=True)
    for label, stats in override_stats.items():
        print(f"[override] {label} mean_rgb={stats['mean_rgb']}", flush=True)
    print(f"[features-dc] shape={list(features.shape)} mean={report['gaussian_feature_stats']['_features_dc_mean']}", flush=True)
    print(f"[done] report={report_path}", flush=True)
    print(f"[done] panels={gt_panel_path} {override_panel_path}", flush=True)


if __name__ == "__main__":
    main()
