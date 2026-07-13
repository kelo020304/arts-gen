#!/usr/bin/env python3
"""Track2 embed-only gate evaluation.

Compares pretrained SLat flow against the view-embedding-only checkpoint on
the Phase2 overfit cache.  Both paths use the same sparse coords, conditioning
tokens, initial noise, CFG sampler, and pretrained SLat decoders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT), str(Path(__file__).resolve().parent)):
    if item not in sys.path:
        sys.path.insert(0, item)

import inference  # noqa: E402
import utils3d.torch  # noqa: E402
from train_track2_overfit import (  # noqa: E402
    DEFAULT_CACHE_MANIFEST,
    DEFAULT_CKPT_DIR,
    DEFAULT_FLOW_CKPT,
    DEFAULT_MEAN,
    DEFAULT_STD,
    Phase2SLatFlowDataset,
    build_model,
    sparse_from_batch,
)
from official_trellis_cond import dump_preprocessed_dino_inputs  # noqa: E402
from trellis.pipelines.samplers import FlowEulerCfgSampler  # noqa: E402
from trellis.renderers.gaussian_render import GaussianRenderer  # noqa: E402


DEFAULT_OUT_DIR = DEFAULT_CKPT_DIR / "embed_gate_eval"
DEFAULT_GAUSSIAN_DECODER = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
DEFAULT_MESH_DECODER = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
GS_PRESET = {
    "max_scale_quantile": 0.997,
    "max_scale_abs": 0.020,
    "scale_mult": 0.75,
    "opacity_mult": 1.5,
    "kernel_size": 0.05,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest_objects(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = load_json(path)
    out = {}
    for obj in payload.get("objects", []):
        key = (str(obj["dataset_id"]), str(obj["obj_id"]), int(obj["angle_idx"]))
        out[key] = obj
    return out


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_embed_model(base_ckpt: Path, latest_ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(base_ckpt, device)
    payload = torch.load(latest_ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Track2 latest load mismatch missing={missing} unexpected={unexpected}")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def freeze_eval(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def sample_slat(
    model: torch.nn.Module,
    x0,
    cond_tokens: torch.Tensor,
    noise_feats: torch.Tensor,
    *,
    steps: int,
    cfg_strength: float,
) -> Any:
    noise = x0.replace(noise_feats)
    sampler = FlowEulerCfgSampler(sigma_min=1.0e-5)
    neg_cond = torch.zeros_like(cond_tokens)
    result = sampler.sample(
        model,
        noise=noise,
        cond=cond_tokens,
        neg_cond=neg_cond,
        steps=int(steps),
        cfg_strength=float(cfg_strength),
        verbose=False,
    )
    return result.samples


def decode_assets(slat_norm, *, gaussian_decoder: Path, mesh_decoder: Path) -> dict[str, Any]:
    return inference.decode_slat_assets(
        slat_norm,
        gaussian_decoder_ckpt=str(gaussian_decoder),
        mesh_decoder_ckpt=str(mesh_decoder),
        slat_is_normalized=True,
    )


def mesh_to_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if mesh is None or not bool(getattr(mesh, "success", False)):
        return None
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if vertices is None or faces is None:
        return None
    v = vertices.detach().float().cpu().numpy()
    f = faces.detach().long().cpu().numpy()
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or len(v) == 0 or len(f) == 0:
        return None
    return v, f


def mesh_stats(mesh: Any) -> dict[str, Any]:
    arrays = mesh_to_arrays(mesh)
    if arrays is None:
        return {"mesh_success": False, "vertices": 0, "faces": 0, "watertight": False}
    import trimesh

    v, f = arrays
    tri = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return {
        "mesh_success": True,
        "vertices": int(len(v)),
        "faces": int(len(f)),
        "watertight": bool(tri.is_watertight),
    }


def sample_mesh_points(mesh: Any, *, count: int, seed: int) -> np.ndarray | None:
    arrays = mesh_to_arrays(mesh)
    if arrays is None:
        return None
    import trimesh

    v, f = arrays
    tri = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if len(tri.faces) == 0:
        return None
    pts, _face_idx = trimesh.sample.sample_surface(tri, int(count), seed=int(seed))
    return np.asarray(pts, dtype=np.float32)


def nearest_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    return cKDTree(b).query(a, k=1)[0]


def chamfer_to_gt(pred_mesh: Any, gt_mesh: Any, *, samples: int, seed: int) -> dict[str, float]:
    pred = sample_mesh_points(pred_mesh, count=samples, seed=seed)
    gt = sample_mesh_points(gt_mesh, count=samples, seed=seed + 17)
    if pred is None or gt is None:
        return {"chamfer_mean": float("nan"), "chamfer_p95": float("nan")}
    p2g = nearest_dist(pred, gt)
    g2p = nearest_dist(gt, pred)
    both = np.concatenate([p2g, g2p])
    return {
        "chamfer_mean": float(np.mean(both)),
        "chamfer_p95": float(np.percentile(both, 95)),
    }


def new_like_gaussian(gaussian: Any) -> Any:
    out = type(gaussian)(**gaussian.init_params)
    out.active_sh_degree = gaussian.active_sh_degree
    return out


def subset_gaussian(gaussian: Any, keep: torch.Tensor) -> Any:
    out = new_like_gaussian(gaussian)
    keep = keep.to(device=gaussian.get_xyz.device, dtype=torch.bool)
    for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        setattr(out, name, getattr(gaussian, name).detach()[keep].clone())
    out._features_rest = None if gaussian._features_rest is None else gaussian._features_rest.detach()[keep].clone()
    return out


def adjust_gaussian(gaussian: Any) -> Any:
    out = new_like_gaussian(gaussian)
    for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        setattr(out, name, getattr(gaussian, name).detach().clone())
    out._features_rest = None if gaussian._features_rest is None else gaussian._features_rest.detach().clone()
    scaling = torch.clamp(out.get_scaling * float(GS_PRESET["scale_mult"]), min=out.mininum_kernel_size + 1e-7)
    out.from_scaling(scaling)
    opacity = torch.clamp(out.get_opacity * float(GS_PRESET["opacity_mult"]), 1e-5, 0.995)
    out.from_opacity(opacity)
    return out


def apply_gs_preset(gaussian: Any) -> Any:
    scale_max = gaussian.get_scaling.detach().max(dim=1).values
    quantile_limit = torch.quantile(scale_max, float(GS_PRESET["max_scale_quantile"]))
    abs_limit = scale_max.new_tensor(float(GS_PRESET["max_scale_abs"]))
    keep = scale_max <= torch.minimum(quantile_limit, abs_limit)
    return adjust_gaussian(subset_gaussian(gaussian, keep))


def make_renderer(resolution: int) -> GaussianRenderer:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (1, 1, 1)
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = float(GS_PRESET["kernel_size"])
    renderer.pipe.scale_modifier = 1.0
    return renderer


def load_camera_matrices(camera_path: Path, view_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    payload = load_json(camera_path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) < max(view_indices) + 1:
        raise ValueError(f"{camera_path}: invalid camera frames for view_indices={view_indices}")
    extrinsics = []
    intrinsics = []
    for idx in view_indices:
        frame = frames[int(idx)]
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device="cuda")
        c2w = c2w.clone()
        c2w[:3, 1:3] *= -1
        extrinsics.append(torch.inverse(c2w))
        fov = torch.tensor(float(frame["camera_angle_x"]), dtype=torch.float32, device="cuda")
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    return torch.stack(extrinsics), torch.stack(intrinsics)


@torch.no_grad()
def render_gaussian_views(gaussian: Any, extrinsics: torch.Tensor, intrinsics: torch.Tensor, *, resolution: int) -> torch.Tensor:
    renderer = make_renderer(int(resolution))
    gaussian = apply_gs_preset(gaussian)
    views = []
    for i in range(extrinsics.shape[0]):
        color = renderer.render(gaussian, extrinsics[i], intrinsics[i])["color"]
        views.append(color.detach().float().clamp(0, 1))
    return torch.stack(views, dim=0)


def load_reference_rgbs(paths: list[str], *, resolution: int) -> torch.Tensor:
    views = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize((int(resolution), int(resolution)), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        views.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(views, dim=0).cuda()


def load_reference_masks(paths: list[str], *, resolution: int) -> torch.Tensor:
    views = []
    for path in paths:
        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        mask = (arr > 0).astype(np.uint8) * 255
        img = Image.fromarray(mask, mode="L").resize((int(resolution), int(resolution)), Image.Resampling.NEAREST)
        views.append(torch.from_numpy((np.asarray(img) > 0).astype(np.float32)))
    return torch.stack(views, dim=0).cuda()


def color_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    fg = mask[:, None].bool()
    if int(fg.sum().item()) == 0:
        fg = torch.ones_like(mask[:, None], dtype=torch.bool)
    diff = torch.abs(pred - gt)
    l1 = diff[fg.expand_as(diff)].mean()
    pred_luma = (pred[:, 0] * 0.299 + pred[:, 1] * 0.587 + pred[:, 2] * 0.114)
    gt_luma = (gt[:, 0] * 0.299 + gt[:, 1] * 0.587 + gt[:, 2] * 0.114)
    fg2 = mask.bool()
    bright_fg = fg2 & (gt_luma > 0.15)
    if int(bright_fg.sum().item()) == 0:
        bright_fg = fg2
    deficit = torch.clamp(gt_luma - pred_luma, min=0.0)
    black_frac = ((pred_luma < 0.08) & (gt_luma > 0.15) & fg2).float()
    denom = max(int(fg2.sum().item()), 1)
    return {
        "rgb_l1_fg": float(l1.detach().cpu().item()),
        "dark_deficit_fg": float(deficit[bright_fg].mean().detach().cpu().item()) if int(bright_fg.sum().item()) else float("nan"),
        "black_frac_fg": float(black_frac.sum().detach().cpu().item() / denom),
    }


def save_viz(path: Path, *, gt: torch.Tensor, base: torch.Tensor, embed: torch.Tensor | None, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(4, gt.shape[0])
    rows = [("GT", gt), ("Base", base)]
    if embed is not None:
        rows.append(("Embed", embed))
    fig, axes = plt.subplots(len(rows), count, figsize=(4 * count, 3 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, (name, tensor) in enumerate(rows):
        for c in range(count):
            arr = tensor[c].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            axes[r, c].imshow(arr)
            axes[r, c].axis("off")
            axes[r, c].set_title(f"{name} v{c}")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    seed_all(int(args.seed))
    device = torch.device(f"cuda:{int(args.gpu)}")
    torch.cuda.set_device(device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Phase2SLatFlowDataset(
        args.manifest,
        view_dropout=False,
        cond_layer_norm=bool(args.cond_layer_norm),
        cond_source=str(args.cond_source),
    )
    manifest_objects = load_manifest_objects(args.manifest)
    base_model = freeze_eval(build_model(args.flow_ckpt, device))
    embed_model = None if bool(args.base_only) else load_embed_model(args.flow_ckpt, args.track2_ckpt, device)
    print(
        json.dumps(
            {
                "event": "track2_eval_start",
                "cond_source": dataset.cond_source,
                "base_only": bool(args.base_only),
                "slat_normalization": {
                    "mean": DEFAULT_MEAN.flatten().tolist(),
                    "std": DEFAULT_STD.flatten().tolist(),
                    "decode_slat_assets_slat_is_normalized": True,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    limit = len(dataset) if int(args.limit) <= 0 else min(len(dataset), int(args.limit))
    for idx in range(limit):
        sample = dataset[idx]
        batch = dataset.collate_fn([sample])
        meta = batch["meta"][0]
        key = (str(meta["dataset_id"]), str(meta["obj_id"]), int(meta["angle_idx"]))
        manifest = manifest_objects[key]
        x0 = sparse_from_batch(batch, device)
        cond = batch["cond_tokens"].to(device=device, dtype=torch.float32)
        if bool(args.dump_dino_inputs) and dataset.cond_source == "live_official_trellis_rgba" and idx < int(args.dino_dump_limit):
            dump_preprocessed_dino_inputs(
                meta["reference_rgb"],
                meta["reference_masks"],
                out_dir=out_dir / "dino_inputs",
                prefix=f"{idx:02d}_{meta['tag']}_{meta['obj_id']}",
                view_indices=meta["view_indices"],
            )
        gen = torch.Generator(device=device).manual_seed(int(args.seed) + idx * 1009)
        noise_feats = torch.randn(x0.feats.shape, generator=gen, device=device, dtype=x0.feats.dtype)

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        base_slat = sample_slat(base_model, x0, cond, noise_feats, steps=args.steps, cfg_strength=args.cfg_strength)
        t1.record(); torch.cuda.synchronize(device)
        base_ms = float(t0.elapsed_time(t1))
        embed_slat = None
        embed_ms = float("nan")
        if embed_model is not None:
            t0.record()
            embed_slat = sample_slat(embed_model, x0, cond, noise_feats, steps=args.steps, cfg_strength=args.cfg_strength)
            t1.record(); torch.cuda.synchronize(device)
            embed_ms = float(t0.elapsed_time(t1))

        gt_assets = decode_assets(x0, gaussian_decoder=args.gaussian_decoder, mesh_decoder=args.mesh_decoder)
        base_assets = decode_assets(base_slat, gaussian_decoder=args.gaussian_decoder, mesh_decoder=args.mesh_decoder)
        embed_assets = None if embed_slat is None else decode_assets(embed_slat, gaussian_decoder=args.gaussian_decoder, mesh_decoder=args.mesh_decoder)

        camera_path = Path(str(manifest["reference_rgb"][0])).parents[1] / "camera_transforms.json"
        view_indices = [int(x) for x in manifest["view_indices"]]
        extrinsics, intrinsics = load_camera_matrices(camera_path, view_indices)
        gt_rgb = load_reference_rgbs([str(x) for x in manifest["reference_rgb"]], resolution=int(args.resolution))
        gt_mask = load_reference_masks([str(x) for x in manifest["reference_masks"]], resolution=int(args.resolution))
        base_rgb = render_gaussian_views(base_assets["gaussian"], extrinsics, intrinsics, resolution=int(args.resolution))
        embed_rgb = None if embed_assets is None else render_gaussian_views(embed_assets["gaussian"], extrinsics, intrinsics, resolution=int(args.resolution))
        base_color = color_metrics(base_rgb, gt_rgb, gt_mask)
        embed_color = {"rgb_l1_fg": float("nan"), "dark_deficit_fg": float("nan"), "black_frac_fg": float("nan")}
        if embed_rgb is not None:
            embed_color = color_metrics(embed_rgb, gt_rgb, gt_mask)

        base_geom = mesh_stats(base_assets["mesh"])
        embed_geom = {"mesh_success": False, "vertices": 0, "faces": 0, "watertight": False}
        if embed_assets is not None:
            embed_geom = mesh_stats(embed_assets["mesh"])
        gt_geom = mesh_stats(gt_assets["mesh"])
        base_ch = chamfer_to_gt(base_assets["mesh"], gt_assets["mesh"], samples=int(args.chamfer_samples), seed=int(args.seed) + idx)
        embed_ch = {"chamfer_mean": float("nan"), "chamfer_p95": float("nan")}
        if embed_assets is not None:
            embed_ch = chamfer_to_gt(embed_assets["mesh"], gt_assets["mesh"], samples=int(args.chamfer_samples), seed=int(args.seed) + idx + 991)

        row = {
            "tag": meta["tag"],
            "dataset_id": meta["dataset_id"],
            "obj_id": meta["obj_id"],
            "angle_idx": int(meta["angle_idx"]),
            "coords": int(x0.feats.shape[0]),
            "steps": int(args.steps),
            "cfg_strength": float(args.cfg_strength),
            "cond_source": meta["token_source"],
            "cond_token_shape": "x".join(str(x) for x in tuple(cond.shape)),
            "base_seconds": base_ms / 1000.0,
            "embed_seconds": embed_ms / 1000.0,
            "gt_mesh_faces": gt_geom["faces"],
            "base_rgb_l1_fg": base_color["rgb_l1_fg"],
            "embed_rgb_l1_fg": embed_color["rgb_l1_fg"],
            "delta_rgb_l1_fg": base_color["rgb_l1_fg"] - embed_color["rgb_l1_fg"],
            "base_dark_deficit_fg": base_color["dark_deficit_fg"],
            "embed_dark_deficit_fg": embed_color["dark_deficit_fg"],
            "delta_dark_deficit_fg": base_color["dark_deficit_fg"] - embed_color["dark_deficit_fg"],
            "base_black_frac_fg": base_color["black_frac_fg"],
            "embed_black_frac_fg": embed_color["black_frac_fg"],
            "delta_black_frac_fg": base_color["black_frac_fg"] - embed_color["black_frac_fg"],
            "base_mesh_success": base_geom["mesh_success"],
            "embed_mesh_success": embed_geom["mesh_success"],
            "base_faces": base_geom["faces"],
            "embed_faces": embed_geom["faces"],
            "base_watertight": base_geom["watertight"],
            "embed_watertight": embed_geom["watertight"],
            "base_chamfer_mean": base_ch["chamfer_mean"],
            "embed_chamfer_mean": embed_ch["chamfer_mean"],
            "delta_chamfer_mean": base_ch["chamfer_mean"] - embed_ch["chamfer_mean"],
            "base_chamfer_p95": base_ch["chamfer_p95"],
            "embed_chamfer_p95": embed_ch["chamfer_p95"],
            "delta_chamfer_p95": base_ch["chamfer_p95"] - embed_ch["chamfer_p95"],
        }
        rows.append(row)
        if embed_model is None:
            print(
                "[track2-base] {tag} {obj_id}: rgb_l1={b:.4f} dark={bd:.4f} "
                "black_frac={bf:.4f} chamfer_p95={bc:.4f}".format(
                    tag=row["tag"],
                    obj_id=row["obj_id"],
                    b=row["base_rgb_l1_fg"],
                    bd=row["base_dark_deficit_fg"],
                    bf=row["base_black_frac_fg"],
                    bc=row["base_chamfer_p95"],
                ),
                flush=True,
            )
        else:
            print(
                "[track2-eval] {tag} {obj_id}: rgb_l1 base={b:.4f} embed={e:.4f} "
                "dark base={bd:.4f} embed={ed:.4f} chamfer_p95 base={bc:.4f} embed={ec:.4f}".format(
                    tag=row["tag"],
                    obj_id=row["obj_id"],
                    b=row["base_rgb_l1_fg"],
                    e=row["embed_rgb_l1_fg"],
                    bd=row["base_dark_deficit_fg"],
                    ed=row["embed_dark_deficit_fg"],
                    bc=row["base_chamfer_p95"],
                    ec=row["embed_chamfer_p95"],
                ),
                flush=True,
            )
        if idx < int(args.viz_limit):
            save_viz(out_dir / f"viz_{idx:02d}_{meta['tag']}_{meta['obj_id']}.png", gt=gt_rgb, base=base_rgb, embed=embed_rgb, title=f"{meta['tag']} {meta['obj_id']}")

    csv_path = out_dir / "track2_embed_gate_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    aggregate = {
        "n": len(rows),
        "track2_ckpt": str(args.track2_ckpt),
        "flow_ckpt": str(args.flow_ckpt),
        "cond_source": dataset.cond_source,
        "base_only": bool(args.base_only),
        "steps": int(args.steps),
        "cfg_strength": float(args.cfg_strength),
        "slat_normalization_mean": DEFAULT_MEAN.flatten().tolist(),
        "slat_normalization_std": DEFAULT_STD.flatten().tolist(),
    }
    for key in (
        "base_rgb_l1_fg",
        "base_dark_deficit_fg",
        "base_black_frac_fg",
        "base_chamfer_mean",
        "base_chamfer_p95",
    ):
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        aggregate[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
        aggregate[f"median_{key}"] = float(np.median(vals)) if vals else float("nan")
    for key in (
        "delta_rgb_l1_fg",
        "delta_dark_deficit_fg",
        "delta_black_frac_fg",
        "delta_chamfer_mean",
        "delta_chamfer_p95",
    ):
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        aggregate[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
        aggregate[f"median_{key}"] = float(np.median(vals)) if vals else float("nan")
    aggregate["embed_improves_rgb_l1_count"] = int(sum(float(r["delta_rgb_l1_fg"]) > 0 for r in rows))
    aggregate["embed_improves_dark_count"] = int(sum(float(r["delta_dark_deficit_fg"]) > 0 for r in rows))
    aggregate["embed_improves_chamfer_p95_count"] = int(sum(float(r["delta_chamfer_p95"]) > 0 for r in rows))
    report = {"aggregate": aggregate, "rows": rows}
    json_path = out_dir / "track2_embed_gate_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "track2_embed_gate_report.md"
    lines = [
        "# Track2 Embed-Only Gate",
        "",
        f"- Samples: {len(rows)}",
        f"- Cond source: `{dataset.cond_source}`",
        f"- Base only: {bool(args.base_only)}",
        f"- Steps: {int(args.steps)}",
        f"- CFG: {float(args.cfg_strength)}",
        f"- SLat normalization mean: `{DEFAULT_MEAN.flatten().tolist()}`",
        f"- SLat normalization std: `{DEFAULT_STD.flatten().tolist()}`",
        f"- Mean base rgb_l1_fg: {aggregate['mean_base_rgb_l1_fg']:.6f}",
        f"- Mean base dark_deficit_fg: {aggregate['mean_base_dark_deficit_fg']:.6f}",
        f"- Mean base black_frac_fg: {aggregate['mean_base_black_frac_fg']:.6f}",
        f"- Mean delta rgb_l1_fg (base - embed): {aggregate['mean_delta_rgb_l1_fg']:.6f}",
        f"- Mean delta dark_deficit_fg (base - embed): {aggregate['mean_delta_dark_deficit_fg']:.6f}",
        f"- Mean delta chamfer_p95 (base - embed): {aggregate['mean_delta_chamfer_p95']:.6f}",
        f"- RGB improved: {aggregate['embed_improves_rgb_l1_count']}/{len(rows)}",
        f"- Dark deficit improved: {aggregate['embed_improves_dark_count']}/{len(rows)}",
        f"- Chamfer p95 improved: {aggregate['embed_improves_chamfer_p95_count']}/{len(rows)}",
        "",
        "| tag | obj | rgb_l1 base/embed | dark base/embed | black_frac base/embed | chamfer_p95 base/embed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {tag} | {obj_id} | {base_rgb_l1_fg:.4f}/{embed_rgb_l1_fg:.4f} | {base_dark_deficit_fg:.4f}/{embed_dark_deficit_fg:.4f} | {base_black_frac_fg:.4f}/{embed_black_frac_fg:.4f} | {base_chamfer_p95:.4f}/{embed_chamfer_p95:.4f} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "md": str(md_path), "aggregate": aggregate}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--flow-ckpt", type=Path, default=DEFAULT_FLOW_CKPT)
    parser.add_argument("--track2-ckpt", type=Path, default=DEFAULT_CKPT_DIR / "latest.pt")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gaussian-decoder", type=Path, default=DEFAULT_GAUSSIAN_DECODER)
    parser.add_argument("--mesh-decoder", type=Path, default=DEFAULT_MESH_DECODER)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg-strength", type=float, default=3.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--chamfer-samples", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--viz-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--cond-layer-norm", action="store_true")
    parser.add_argument("--cond-source", choices=["live_official_trellis_rgba", "cache"], default="live_official_trellis_rgba")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--dump-dino-inputs", action="store_true", default=True)
    parser.add_argument("--no-dump-dino-inputs", dest="dump_dino_inputs", action="store_false")
    parser.add_argument("--dino-dump-limit", type=int, default=12)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    main()
