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
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


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
from trellis.utils import postprocessing_utils  # noqa: E402
from trellis.renderers.gaussian_render import GaussianRenderer  # noqa: E402


DEFAULT_DATA_ROOT = Path("/mnt/robot-data-lab/jzh/art-gen/data/realappliance")
DEFAULT_OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-realappliance-gaussian-filtered")
DEFAULT_GAUSSIAN_DECODER = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
DEFAULT_MESH_DECODER = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
DEFAULT_BAKED_REF = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/"
    "0617-realappliance-textured-bake-fixed-direction/039-1/overall/textured_mesh_nvdiffrast_render.png"
)


@dataclass(frozen=True)
class Variant:
    name: str
    max_mesh_dist: float | None = None
    max_scale_quantile: float | None = None
    max_scale_abs: float | None = None
    min_opacity: float | None = None
    scale_mult: float = 1.0
    opacity_mult: float = 1.0
    kernel_size: float = 0.1


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


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = mesh.vertices.detach().float().cpu().numpy() if torch.is_tensor(mesh.vertices) else np.asarray(mesh.vertices, dtype=np.float32)
    faces = mesh.faces.detach().long().cpu().numpy() if torch.is_tensor(mesh.faces) else np.asarray(mesh.faces, dtype=np.int64)
    return vertices.reshape(-1, 3), faces.reshape(-1, 3)


def _surface_points(
    mesh: Any,
    *,
    simplify: float,
    fill_holes: bool,
    fill_holes_num_views: int,
    samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices, faces = _mesh_arrays(mesh)
    stats: dict[str, Any] = {
        "mesh_vertices_before": int(vertices.shape[0]),
        "mesh_faces_before": int(faces.shape[0]),
    }
    vertices, faces = postprocessing_utils.postprocess_mesh(
        vertices,
        faces,
        simplify=float(simplify) > 0,
        simplify_ratio=float(simplify),
        fill_holes=bool(fill_holes),
        fill_holes_num_views=int(fill_holes_num_views),
        verbose=False,
    )
    stats.update(
        {
            "mesh_vertices_post": int(vertices.shape[0]),
            "mesh_faces_post": int(faces.shape[0]),
        }
    )
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rng = np.random.default_rng(20260617)
    sample_count = min(int(samples), max(int(faces.shape[0]) * 2, 1_000))
    sampled, _ = trimesh.sample.sample_surface(tri_mesh, sample_count, seed=rng)
    centers = vertices[faces].mean(axis=1)
    points = np.concatenate([vertices, centers, sampled.astype(np.float32, copy=False)], axis=0)
    stats["surface_points"] = int(points.shape[0])
    stats["surface_sample_points"] = int(sampled.shape[0])
    stats["mesh_bounds_min"] = [float(x) for x in vertices.min(axis=0).tolist()]
    stats["mesh_bounds_max"] = [float(x) for x in vertices.max(axis=0).tolist()]
    return np.ascontiguousarray(points.astype(np.float32, copy=False)), stats


def _new_like_gaussian(gaussian: Any) -> Any:
    out = type(gaussian)(**gaussian.init_params)
    out.active_sh_degree = gaussian.active_sh_degree
    return out


def _copy_gaussian(gaussian: Any) -> Any:
    out = _new_like_gaussian(gaussian)
    for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        setattr(out, name, getattr(gaussian, name).detach().clone())
    out._features_rest = None if gaussian._features_rest is None else gaussian._features_rest.detach().clone()
    return out


def _subset_gaussian(gaussian: Any, keep: torch.Tensor) -> Any:
    out = _new_like_gaussian(gaussian)
    keep = keep.to(device=gaussian.get_xyz.device, dtype=torch.bool)
    for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        setattr(out, name, getattr(gaussian, name).detach()[keep].clone())
    out._features_rest = None if gaussian._features_rest is None else gaussian._features_rest.detach()[keep].clone()
    return out


def _adjust_gaussian(gaussian: Any, *, scale_mult: float, opacity_mult: float) -> Any:
    out = _copy_gaussian(gaussian)
    if float(scale_mult) != 1.0:
        scaling = torch.clamp(out.get_scaling * float(scale_mult), min=out.mininum_kernel_size + 1e-7)
        out.from_scaling(scaling)
    if float(opacity_mult) != 1.0:
        opacity = torch.clamp(out.get_opacity * float(opacity_mult), 1e-5, 0.995)
        out.from_opacity(opacity)
    return out


def _apply_variant(
    gaussian: Any,
    variant: Variant,
    *,
    mesh_dist: torch.Tensor,
    scale_max: torch.Tensor,
    opacity: torch.Tensor,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    keep = torch.ones_like(opacity, dtype=torch.bool)
    scale_limit = None
    if variant.max_mesh_dist is not None:
        keep &= mesh_dist <= float(variant.max_mesh_dist)
    if variant.max_scale_quantile is not None:
        scale_limit = float(torch.quantile(scale_max, float(variant.max_scale_quantile)).detach().cpu().item())
        keep &= scale_max <= scale_limit
    if variant.max_scale_abs is not None:
        scale_limit = float(variant.max_scale_abs) if scale_limit is None else min(scale_limit, float(variant.max_scale_abs))
        keep &= scale_max <= float(variant.max_scale_abs)
    if variant.min_opacity is not None:
        keep &= opacity >= float(variant.min_opacity)
    filtered = _subset_gaussian(gaussian, keep)
    adjusted = _adjust_gaussian(filtered, scale_mult=float(variant.scale_mult), opacity_mult=float(variant.opacity_mult))
    stats = {
        "name": variant.name,
        "max_mesh_dist": variant.max_mesh_dist,
        "max_scale_quantile": variant.max_scale_quantile,
        "max_scale_abs": variant.max_scale_abs,
        "scale_limit_used": scale_limit,
        "min_opacity": variant.min_opacity,
        "scale_mult": float(variant.scale_mult),
        "opacity_mult": float(variant.opacity_mult),
        "kernel_size": float(variant.kernel_size),
        "kept": int(keep.sum().detach().cpu().item()),
        "removed": int((~keep).sum().detach().cpu().item()),
        "kept_fraction": float(keep.float().mean().detach().cpu().item()),
    }
    return adjusted, keep, stats


def _make_renderer(resolution: int, bg_color: tuple[int, int, int], kernel_size: float) -> GaussianRenderer:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = bg_color
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = float(kernel_size)
    renderer.pipe.scale_modifier = 1.0
    return renderer


@torch.no_grad()
def _render_png(
    gaussian: Any,
    *,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    resolution: int,
    kernel_size: float,
    out_png: Path,
) -> np.ndarray:
    renderer = _make_renderer(int(resolution), (1, 1, 1), float(kernel_size))
    color = renderer.render(gaussian, extrinsic, intrinsic)["color"].detach().float().cpu().clamp(0, 1)
    arr = (color.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_png)
    return arr


def _mask_from_white_bg(path: Path, *, resolution: int, dilation: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    image = Image.open(path).convert("RGB").resize((int(resolution), int(resolution)), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.uint8)
    mask = np.any(arr < 245, axis=-1)
    if int(dilation) > 0:
        mask = binary_dilation(mask, iterations=int(dilation))
    return np.asarray(mask, dtype=bool)


def _artifact_stats(arr: np.ndarray, ref_mask: np.ndarray | None) -> dict[str, float] | None:
    if ref_mask is None:
        return None
    outside = ~ref_mask
    if not np.any(outside):
        return None
    dark = 255.0 - arr.astype(np.float32).mean(axis=-1)
    outside_dark = dark[outside]
    return {
        "outside_dark_mean": float(outside_dark.mean()),
        "outside_dark_p95": float(np.quantile(outside_dark, 0.95)),
        "outside_dark_pixels_gt10": float((outside_dark > 10.0).mean()),
        "outside_dark_pixels_gt25": float((outside_dark > 25.0).mean()),
    }


def _tile(path: Path, label: str, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 44), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size, 44), fill=(0, 0, 0))
    draw.text((8, 8), label[:92], fill=(255, 255, 255))
    tile.paste(image, ((size - image.width) // 2, 44 + (size - image.height) // 2))
    return tile


def _panel(records: list[dict[str, Any]], baked_ref: Path | None, out_png: Path, *, tile_size: int, cols: int) -> None:
    items = records[:]
    if baked_ref is not None and baked_ref.is_file():
        items.append({"name": "baked_mesh_ref", "png": str(baked_ref)})
    cols = max(1, min(int(cols), len(items)))
    rows = int(np.ceil(len(items) / cols))
    canvas = Image.new("RGB", (cols * tile_size, rows * (tile_size + 44)), (255, 255, 255))
    for idx, rec in enumerate(items):
        label = rec["name"]
        if "kept_fraction" in rec:
            label += f" keep={rec['kept_fraction']:.3f}"
        canvas.paste(_tile(Path(rec["png"]), label, tile_size), ((idx % cols) * tile_size, (idx // cols) * (tile_size + 44)))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def _variants() -> list[Variant]:
    return [
        Variant("base_k0.10"),
        Variant("old_scale0.75_opx1.5_k0.05", scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
        Variant("scaleq997_scale0.75_opx1.5", max_scale_quantile=0.997, scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
        Variant("scaleq995_scale0.75_opx1.5", max_scale_quantile=0.995, scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
        Variant("scaleq992_scale0.75_opx1.5", max_scale_quantile=0.992, scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
        Variant("scaleq990_scale0.75_opx1.5", max_scale_quantile=0.990, scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
        Variant("scaleq985_scale0.75_opx1.5", max_scale_quantile=0.985, scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
        Variant("scaleq980_scale0.75_opx1.5", max_scale_quantile=0.980, scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
        Variant("meshdist0.015_scaleq990", max_mesh_dist=0.015, max_scale_quantile=0.990, scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
        Variant("opmin0.80_scaleq990", max_scale_quantile=0.990, min_opacity=0.80, scale_mult=0.75, opacity_mult=1.5, kernel_size=0.05),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter decoded RealAppliance Gaussians using same-SLat mesh surface proximity.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--object-id", default="039")
    parser.add_argument("--angle", type=int, default=1)
    parser.add_argument("--render-view", type=int, default=1)
    parser.add_argument("--gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER)
    parser.add_argument("--mesh-decoder-ckpt", type=Path, default=DEFAULT_MESH_DECODER)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--mesh-simplify", type=float, default=0.75)
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument("--fill-holes-num-views", type=int, default=1000)
    parser.add_argument("--surface-samples", type=int, default=250000)
    parser.add_argument("--baked-ref-png", type=Path, default=DEFAULT_BAKED_REF)
    parser.add_argument("--ref-mask-dilation", type=int, default=7)
    parser.add_argument("--tile-size", type=int, default=360)
    parser.add_argument("--panel-cols", type=int, default=3)
    parser.add_argument("--best-variant", default="scaleq990_scale0.75_opx1.5")
    parser.add_argument("--export-best-ply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inst = f"{args.object_id}_angle_{int(args.angle)}"
    latent_path = args.data_root / "part_synthesis_slat" / args.object_id[:2] / inst / "overall" / "latent.npz"
    coords_np, feats_np = _load_latent(latent_path)
    slat = _make_sparse(coords_np, feats_np)
    decoded = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(args.gaussian_decoder_ckpt.resolve()),
        mesh_decoder_ckpt=str(args.mesh_decoder_ckpt.resolve()),
        slat_is_normalized=False,
    )
    gaussian = decoded["gaussian"]
    mesh = decoded["mesh"]
    if mesh is None or not getattr(mesh, "success", True):
        raise RuntimeError("mesh decoder failed; cannot use mesh surface proximity filter")

    surface_points, mesh_stats = _surface_points(
        mesh,
        simplify=float(args.mesh_simplify),
        fill_holes=not bool(args.no_fill_holes),
        fill_holes_num_views=int(args.fill_holes_num_views),
        samples=int(args.surface_samples),
    )
    xyz_np = gaussian.get_xyz.detach().float().cpu().numpy()
    tree = cKDTree(surface_points)
    mesh_dist_np, _ = tree.query(xyz_np, k=1, workers=-1)
    mesh_dist = torch.from_numpy(mesh_dist_np.astype(np.float32, copy=False)).to(device="cuda")
    scale = gaussian.get_scaling.detach()
    scale_max = scale.max(dim=1).values
    opacity = gaussian.get_opacity.detach().flatten()

    extrinsics, intrinsics = load_camera_matrices(
        args.data_root / "renders" / args.object_id / f"angle_{int(args.angle)}" / "camera_transforms.json",
        [int(args.render_view)],
    )
    case_dir = args.out_dir / f"{args.object_id}-{int(args.angle)}" / f"view_{int(args.render_view)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    ref_mask = _mask_from_white_bg(args.baked_ref_png, resolution=int(args.resolution), dilation=int(args.ref_mask_dilation))

    records: list[dict[str, Any]] = []
    best_gaussian = None
    for variant in _variants():
        adjusted, keep, rec = _apply_variant(
            gaussian,
            variant,
            mesh_dist=mesh_dist,
            scale_max=scale_max,
            opacity=opacity,
        )
        out_png = case_dir / f"{variant.name}.png"
        arr = _render_png(
            adjusted,
            extrinsic=extrinsics[0],
            intrinsic=intrinsics[0],
            resolution=int(args.resolution),
            kernel_size=float(variant.kernel_size),
            out_png=out_png,
        )
        rec["png"] = str(out_png.resolve())
        rec["artifact_vs_baked_mask"] = _artifact_stats(arr, ref_mask)
        rec["mesh_dist_kept_p95"] = float(torch.quantile(mesh_dist[keep], 0.95).detach().cpu().item()) if bool(keep.any()) else None
        rec["opacity_kept_mean"] = float(adjusted.get_opacity.detach().mean().cpu().item()) if adjusted.get_xyz.shape[0] else None
        rec["scale_kept_p95"] = float(torch.quantile(adjusted.get_scaling.detach().max(dim=1).values, 0.95).cpu().item()) if adjusted.get_xyz.shape[0] else None
        records.append(rec)
        if variant.name == str(args.best_variant):
            best_gaussian = adjusted

    if bool(args.export_best_ply):
        if best_gaussian is None:
            raise ValueError(f"best variant not found: {args.best_variant}")
        best_gaussian.save_ply(case_dir / f"{args.best_variant}.ply")

    _panel(records, args.baked_ref_png, case_dir / "floater_filter_panel.png", tile_size=int(args.tile_size), cols=int(args.panel_cols))
    dist_quantiles = {
        f"p{int(q * 1000):03d}": float(torch.quantile(mesh_dist, q).detach().cpu().item())
        for q in (0.5, 0.75, 0.9, 0.95, 0.975, 0.99)
    }
    threshold_keep = {
        f"{thr:.3f}": float((mesh_dist <= thr).float().mean().detach().cpu().item())
        for thr in (0.015, 0.020, 0.025, 0.030, 0.040, 0.050)
    }
    report = {
        "object_id": args.object_id,
        "angle": int(args.angle),
        "render_view": int(args.render_view),
        "latent_path": str(latent_path.resolve()),
        "gaussians_before": int(gaussian.get_xyz.shape[0]),
        "coords": int(coords_np.shape[0]),
        "mesh_stats": mesh_stats,
        "mesh_distance_quantiles": dist_quantiles,
        "mesh_distance_threshold_keep_fraction": threshold_keep,
        "scale_max_quantiles": {
            f"p{int(q * 1000):03d}": float(torch.quantile(scale_max, q).detach().cpu().item())
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "opacity_quantiles": {
            f"p{int(q * 1000):03d}": float(torch.quantile(opacity, q).detach().cpu().item())
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "baked_ref_png": str(args.baked_ref_png.resolve()) if args.baked_ref_png.is_file() else None,
        "panel": str((case_dir / "floater_filter_panel.png").resolve()),
        "best_variant": str(args.best_variant),
        "best_ply": str((case_dir / f"{args.best_variant}.ply").resolve()) if bool(args.export_best_ply) else None,
        "records": records,
    }
    (case_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
