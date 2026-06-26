#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import inference  # noqa: E402
from scripts.tools.roundtrip.trellis_full_voxel_mesh_roundtrip import load_camera_matrices  # noqa: E402
from trellis.modules.sparse import SparseTensor  # noqa: E402


def _restore_trellis_renderer_package() -> None:
    for name in list(sys.modules):
        if name == "trellis.renderers" or name.startswith("trellis.renderers."):
            sys.modules.pop(name, None)
    import trellis.renderers  # noqa: F401


_restore_trellis_renderer_package()
from trellis.renderers.gaussian_render import GaussianRenderer  # noqa: E402


DEFAULT_DATA_ROOT = Path("/mnt/robot-data-lab/jzh/art-gen/data/realappliance")
DEFAULT_OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-realappliance-gaussian-sharpness")
DEFAULT_DECODER = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"


@dataclass(frozen=True)
class Variant:
    name: str
    scale_mult: float = 1.0
    opacity_power: float = 1.0
    opacity_mult: float = 1.0
    kernel_size: float = 0.1
    scale_modifier: float = 1.0
    ssaa: int = 1


def _load_latent(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        coords = np.asarray(data["coords"])
        feats = np.asarray(data["feats"])
    return (
        np.ascontiguousarray(coords.astype(np.int32, copy=False)),
        np.ascontiguousarray(feats.astype(np.float32, copy=False)),
    )


def _make_sparse(coords_np: np.ndarray, feats_np: np.ndarray) -> SparseTensor:
    coords = torch.from_numpy(coords_np).to(device="cuda", dtype=torch.int32)
    feats = torch.from_numpy(feats_np).to(device="cuda", dtype=torch.float32)
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    return SparseTensor(coords=torch.cat([batch, coords], dim=1), feats=feats)


def _clone_and_adjust_gaussian(gaussian: Any, variant: Variant) -> Any:
    adjusted = type(gaussian)(**gaussian.init_params)
    adjusted.active_sh_degree = gaussian.active_sh_degree
    for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        setattr(adjusted, name, getattr(gaussian, name).detach().clone())
    adjusted._features_rest = None if gaussian._features_rest is None else gaussian._features_rest.detach().clone()
    if variant.scale_mult != 1.0:
        scaling = torch.clamp(adjusted.get_scaling * float(variant.scale_mult), min=adjusted.mininum_kernel_size + 1e-7)
        adjusted.from_scaling(scaling)
    if variant.opacity_power != 1.0 or variant.opacity_mult != 1.0:
        opacity = adjusted.get_opacity
        opacity = torch.clamp((opacity ** float(variant.opacity_power)) * float(variant.opacity_mult), 1e-5, 0.995)
        adjusted.from_opacity(opacity)
    return adjusted


def _make_renderer(resolution: int, bg_color: tuple[int, int, int], variant: Variant) -> GaussianRenderer:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = bg_color
    renderer.rendering_options.ssaa = int(variant.ssaa)
    renderer.pipe.kernel_size = float(variant.kernel_size)
    renderer.pipe.scale_modifier = float(variant.scale_modifier)
    return renderer


@torch.no_grad()
def _render_variant(
    gaussian: Any,
    variant: Variant,
    *,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    resolution: int,
    bg_color: tuple[int, int, int],
    out_png: Path,
) -> dict[str, Any]:
    adjusted = _clone_and_adjust_gaussian(gaussian, variant)
    renderer = _make_renderer(resolution, bg_color, variant)
    color = renderer.render(adjusted, extrinsic, intrinsic)["color"].detach().float().cpu().clamp(0, 1)
    arr = (color.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_png)
    op = adjusted.get_opacity.detach()
    sc = adjusted.get_scaling.detach()
    return {
        "name": variant.name,
        "png": str(out_png.resolve()),
        "scale_mult": variant.scale_mult,
        "opacity_power": variant.opacity_power,
        "opacity_mult": variant.opacity_mult,
        "kernel_size": variant.kernel_size,
        "scale_modifier": variant.scale_modifier,
        "ssaa": variant.ssaa,
        "opacity_mean": float(op.mean().cpu().item()),
        "opacity_p50": float(op.median().cpu().item()),
        "opacity_p90": float(torch.quantile(op.flatten(), 0.9).cpu().item()),
        "scale_mean": float(sc.mean().cpu().item()),
        "scale_p50": float(sc.median().cpu().item()),
        "scale_p90": float(torch.quantile(sc.flatten(), 0.9).cpu().item()),
        "gaussians": int(adjusted.get_xyz.shape[0]),
    }


def _tile(path: Path, label: str, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 44), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size, 44), fill=(0, 0, 0))
    draw.text((8, 8), label[:80], fill=(255, 255, 255))
    tile.paste(image, ((size - image.width) // 2, 44 + (size - image.height) // 2))
    return tile


def _panel(records: list[dict[str, Any]], out_png: Path, *, tile_size: int, cols: int) -> None:
    cols = max(1, min(int(cols), len(records)))
    rows = int(np.ceil(len(records) / cols))
    canvas = Image.new("RGB", (cols * tile_size, rows * (tile_size + 44)), (255, 255, 255))
    for idx, rec in enumerate(records):
        label = (
            f"{rec['name']} s={rec['scale_mult']} op_pow={rec['opacity_power']} "
            f"op_mul={rec['opacity_mult']} k={rec['kernel_size']} smod={rec['scale_modifier']}"
        )
        canvas.paste(_tile(Path(rec["png"]), label, tile_size), ((idx % cols) * tile_size, (idx // cols) * (tile_size + 44)))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def _variants() -> list[Variant]:
    return [
        Variant("base"),
        Variant("kernel_0.05", kernel_size=0.05),
        Variant("kernel_0.02", kernel_size=0.02),
        Variant("scale_0.75", scale_mult=0.75),
        Variant("scale_0.60", scale_mult=0.60),
        Variant("opacity_x1.5", opacity_mult=1.5),
        Variant("opacity_x2.0", opacity_mult=2.0),
        Variant("opacity_pow0.70", opacity_power=0.70),
        Variant("scale0.75_opx1.5_k0.05", scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
        Variant("scale0.60_opx1.5_k0.05", scale_mult=0.60, opacity_mult=1.5, kernel_size=0.05),
        Variant("scale0.60_oppow0.7_k0.05", scale_mult=0.60, opacity_power=0.70, kernel_size=0.05),
        Variant("scale_modifier0.75_k0.05", scale_modifier=0.75, kernel_size=0.05),
        Variant("scale_modifier0.60_opx1.5_k0.05", scale_modifier=0.60, opacity_mult=1.5, kernel_size=0.05),
        Variant("hires_ssaa2_base", ssaa=2),
        Variant("hires_ssaa2_scale0.75_opx1.5_k0.05", scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05, ssaa=2),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune RealAppliance precomputed raw SLat Gaussian sharpness without retraining.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--object-id", default="039")
    parser.add_argument("--angle", type=int, default=1)
    parser.add_argument("--render-view", type=int, default=1)
    parser.add_argument("--decoder-ckpt", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--tile-size", type=int, default=360)
    parser.add_argument("--panel-cols", type=int, default=3)
    parser.add_argument("--bg", choices=("white", "black"), default="white")
    parser.add_argument("--export-best-ply", action="store_true")
    parser.add_argument("--best-variant", default="scale0.75_opx1.5_k0.05")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inst = f"{args.object_id}_angle_{int(args.angle)}"
    latent_path = args.data_root / "part_synthesis_slat" / args.object_id[:2] / inst / "overall" / "latent.npz"
    coords, feats = _load_latent(latent_path)
    slat = _make_sparse(coords, feats)
    decoded = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(args.decoder_ckpt.resolve()),
        slat_is_normalized=False,
    )
    gaussian = decoded["gaussian"]
    extrinsics, intrinsics = load_camera_matrices(
        args.data_root / "renders" / args.object_id / f"angle_{int(args.angle)}" / "camera_transforms.json",
        [int(args.render_view)],
    )
    bg_color = (1, 1, 1) if args.bg == "white" else (0, 0, 0)
    case_dir = args.out_dir / f"{args.object_id}-{int(args.angle)}" / f"view_{int(args.render_view)}_{args.bg}"
    records: list[dict[str, Any]] = []
    for variant in _variants():
        out_png = case_dir / f"{variant.name}.png"
        records.append(
            _render_variant(
                gaussian,
                variant,
                extrinsic=extrinsics[0],
                intrinsic=intrinsics[0],
                resolution=int(args.resolution),
                bg_color=bg_color,
                out_png=out_png,
            )
        )
        if args.export_best_ply and variant.name == str(args.best_variant):
            best_gaussian = _clone_and_adjust_gaussian(gaussian, variant)
            best_gaussian.save_ply(case_dir / f"{variant.name}.ply")
    _panel(records, case_dir / "sharpness_panel.png", tile_size=int(args.tile_size), cols=int(args.panel_cols))
    report = {
        "object_id": args.object_id,
        "angle": int(args.angle),
        "render_view": int(args.render_view),
        "latent_path": str(latent_path.resolve()),
        "resolution": int(args.resolution),
        "bg": args.bg,
        "note": "num_gaussians is fixed by the decoder checkpoint representation_config/output layout; this sweep only adjusts decoded Gaussian opacity/scale and renderer kernel/ssaa.",
        "best_variant": str(args.best_variant),
        "best_ply": str((case_dir / f"{args.best_variant}.ply").resolve()) if args.export_best_ply else None,
        "records": records,
        "panel": str((case_dir / "sharpness_panel.png").resolve()),
    }
    (case_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
