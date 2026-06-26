#!/usr/bin/env python3
"""Separate SLat VAE decode ceiling from SLat flow error on true per-part GT."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")

import inference  # noqa: E402

# inference.py installs a light shell for trellis.renderers at import time.
# Rendering needs the real lazy __init__ so GaussianRenderer is exposed.
sys.modules.pop("trellis.renderers", None)
importlib.import_module("trellis.renderers")


SLAT_MEAN = inference._DEFAULT_SLAT_MEAN.detach().cpu().float()
SLAT_STD = inference._DEFAULT_SLAT_STD.detach().cpu().float()


def require_file(path: Path, what: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def require_dir(path: Path, what: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def load_gt_latent(path: Path) -> dict[str, np.ndarray]:
    path = require_file(path, "GT per-part latent")
    z = np.load(path)
    keys = set(z.files)
    if keys != {"coords", "feats"}:
        raise RuntimeError(f"{path} keys must be exactly {{'coords','feats'}}, got {sorted(keys)}")
    coords = z["coords"]
    feats = z["feats"]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"{path} coords must be [N,3], got {coords.shape}")
    if feats.ndim != 2 or feats.shape[1] != 8:
        raise RuntimeError(f"{path} feats must be [N,8], got {feats.shape}")
    if coords.shape[0] != feats.shape[0]:
        raise RuntimeError(f"{path} coords/feats row mismatch: {coords.shape[0]} vs {feats.shape[0]}")
    if coords.shape[0] == 0:
        raise RuntimeError(f"{path} contains zero voxels")
    return {"coords": coords.astype(np.int64, copy=False), "feats": feats.astype(np.float32, copy=False)}


def make_sparse_tensor(coords_np: np.ndarray, feats_np: np.ndarray):
    from trellis.modules.sparse import SparseTensor

    coords = torch.from_numpy(coords_np.astype(np.int32, copy=False)).cuda()
    feats = torch.from_numpy(feats_np.astype(np.float32, copy=False)).cuda()
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    sp_coords = torch.cat([batch, coords], dim=1)
    return SparseTensor(coords=sp_coords, feats=feats)


def require_trellis_safetensors(path: Path, what: str) -> Path:
    path = require_file(path, what)
    if path.suffix != ".safetensors":
        raise ValueError(f"{what} must be a .safetensors file, got {path}")
    cfg = path.with_suffix(".json")
    if not cfg.is_file():
        raise FileNotFoundError(f"{what} config json not found next to checkpoint: {cfg}")
    return path.resolve()


def probe_official_ckpts(flow_ckpt: Path, decoder_ckpt: Path) -> None:
    _ = inference._load_slat_flow(str(flow_ckpt))
    _ = inference._load_slat_vae_decoder(str(decoder_ckpt))


def parse_instance(instance: str) -> tuple[str, int]:
    m = re.fullmatch(r"(.+)_angle_(\d+)", instance)
    if not m:
        raise ValueError(f"instance must look like '<object>_angle_<idx>', got {instance!r}")
    return m.group(1), int(m.group(2))


def tokens_path(data_root: Path, object_id: str, angle_idx: int) -> Path:
    return require_file(
        data_root / "reconstruction" / "dinov2_tokens" / object_id / f"angle_{angle_idx}" / "tokens.npz",
        "DINOv2 tokens",
    )


def part_latent_path(data_root: Path, object_id: str, angle_idx: int, part: str) -> Path:
    instance = f"{object_id}_angle_{angle_idx}"
    return require_file(data_root / "part_synthesis_slat" / object_id[:2] / instance / part / "latent.npz", "part latent")


def render_dir(data_root: Path, object_id: str, angle_idx: int) -> Path:
    return require_dir(data_root / "renders" / object_id / f"angle_{angle_idx}", "render dir")


def part_info_path(data_root: Path, object_id: str) -> Path:
    return require_file(data_root / "reconstruction" / "part_info" / object_id / "part_info.json", "part_info")


def default_manifest_path(data_root: Path) -> Path:
    return data_root / "manifests" / "part_synthesis" / "arts_mllm_physx-mobility.jsonl"


def resolve_manifest_rel_path(data_root: Path, rel: Any, *, field: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise RuntimeError(f"manifest field {field} must be a non-empty relative path string, got {rel!r}")
    path = Path(rel)
    if path.is_absolute():
        raise RuntimeError(f"manifest field {field} must be relative to DATA_ROOT, got absolute path {rel!r}")
    return data_root / path


def load_manifest_rows(
    manifest_path: Path,
    data_root: Path,
    target_keys: set[tuple[str, int]] | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    manifest_path = require_file(manifest_path, "part synthesis manifest")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "part_synthesis":
                raise RuntimeError(f"{manifest_path}:{line_no}: task must be 'part_synthesis', got {row.get('task')!r}")
            object_id = row.get("object_id")
            angle_idx = row.get("angle_idx")
            if not isinstance(object_id, str) or not object_id:
                raise RuntimeError(f"{manifest_path}:{line_no}: object_id must be a non-empty string")
            if isinstance(angle_idx, bool) or not isinstance(angle_idx, int) or angle_idx < 0:
                raise RuntimeError(f"{manifest_path}:{line_no}: angle_idx must be a non-negative int")
            key = (object_id, angle_idx)
            if target_keys is not None and key not in target_keys:
                continue
            view_indices = row.get("view_indices")
            bad_views = (
                not isinstance(view_indices, list)
                or len(view_indices) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > 11 for v in view_indices)
                or len(set(view_indices)) != 4
            )
            if bad_views:
                raise RuntimeError(f"{manifest_path}:{line_no}: view_indices must be four unique ints in [0,11]")

            paths = row.get("paths")
            if not isinstance(paths, dict):
                raise RuntimeError(f"{manifest_path}:{line_no}: paths must be an object")
            rgb_paths = paths.get("rgb")
            mask_paths = paths.get("masks")
            if not isinstance(rgb_paths, list) or len(rgb_paths) != 4:
                raise RuntimeError(f"{manifest_path}:{line_no}: paths.rgb must contain four entries")
            if not isinstance(mask_paths, list) or len(mask_paths) != 4:
                raise RuntimeError(f"{manifest_path}:{line_no}: paths.masks must contain four entries")
            for idx, (view_idx, rgb_rel, mask_rel) in enumerate(zip(view_indices, rgb_paths, mask_paths)):
                rgb_path = resolve_manifest_rel_path(data_root, rgb_rel, field=f"paths.rgb[{idx}]")
                mask_path = resolve_manifest_rel_path(data_root, mask_rel, field=f"paths.masks[{idx}]")
                expected_rgb = data_root / "renders" / object_id / f"angle_{angle_idx}" / "rgb" / f"view_{view_idx}.png"
                expected_mask = data_root / "renders" / object_id / f"angle_{angle_idx}" / "mask" / f"mask_{view_idx}.npy"
                if rgb_path != expected_rgb:
                    raise RuntimeError(f"{manifest_path}:{line_no}: RGB path/view mismatch: {rgb_path} != {expected_rgb}")
                if mask_path != expected_mask:
                    raise RuntimeError(f"{manifest_path}:{line_no}: mask path/view mismatch: {mask_path} != {expected_mask}")
                require_file(rgb_path, f"manifest RGB view {view_idx}")
                require_file(mask_path, f"manifest mask view {view_idx}")

            if key in rows:
                raise RuntimeError(f"{manifest_path}:{line_no}: duplicate manifest row for object={object_id} angle={angle_idx}")
            rows[key] = row
    if target_keys is not None:
        missing = sorted(target_keys.difference(rows))
        if missing:
            raise RuntimeError(f"manifest missing target rows: {missing}")
    elif not rows:
        raise RuntimeError(f"manifest contains no usable rows: {manifest_path}")
    return rows


def manifest_case_row(
    data_root: Path,
    manifest_rows: dict[tuple[str, int], dict[str, Any]],
    object_id: str,
    angle_idx: int,
    part: str,
) -> dict[str, Any]:
    key = (object_id, angle_idx)
    if key not in manifest_rows:
        raise RuntimeError(f"manifest has no row for object={object_id} angle={angle_idx}")
    row = manifest_rows[key]
    part_ids = row.get("part_ids")
    if not isinstance(part_ids, list) or any(not isinstance(p, str) or not p for p in part_ids):
        raise RuntimeError(f"manifest row {object_id}/angle_{angle_idx} part_ids must be a list of strings")
    part_latents = row.get("paths", {}).get("part_latents")
    if not isinstance(part_latents, dict):
        raise RuntimeError(f"manifest row {object_id}/angle_{angle_idx} paths.part_latents must be an object")
    if part not in part_ids or part not in part_latents:
        raise RuntimeError(
            f"part {part!r} is not a manifest target for {object_id}/angle_{angle_idx}; "
            f"available parts={part_ids}"
        )
    expected = part_latent_path(data_root, object_id, angle_idx, part).resolve()
    manifest_latent = require_file(
        resolve_manifest_rel_path(data_root, part_latents[part], field=f"paths.part_latents.{part}"),
        f"manifest latent {part}",
    ).resolve()
    if manifest_latent != expected:
        raise RuntimeError(f"manifest latent path mismatch for {part}: {manifest_latent} != {expected}")
    return row


def load_tokens(path: Path) -> torch.Tensor:
    npz = np.load(path)
    if set(npz.files) != {"tokens"}:
        raise RuntimeError(f"{path} keys must be exactly ['tokens'], got {npz.files}")
    arr = npz["tokens"]
    if arr.shape != (12, 1370, 1024):
        raise RuntimeError(f"{path} tokens must be [12,1370,1024], got {arr.shape}")
    return torch.from_numpy(arr.astype(np.float32, copy=False))


def load_part_label(data_root: Path, object_id: str, part: str) -> int:
    info = json.loads(part_info_path(data_root, object_id).read_text(encoding="utf-8"))
    parts = info.get("parts", {})
    if part not in parts:
        raise RuntimeError(f"{part!r} missing in {part_info_path(data_root, object_id)}")
    label = int(parts[part]["label"])
    if label <= 0:
        raise RuntimeError(f"{part!r} has invalid mask label {label}")
    return label


def image_to_tensor(path: Path) -> torch.Tensor:
    img = Image.open(require_file(path, "RGB render")).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    pred = pred.float()
    gt = gt.float()
    if mask is not None:
        mask = mask.bool()
        if mask.sum().item() == 0:
            return float("nan")
        err = (pred - gt).pow(2)
        mse = err[:, mask].mean()
    else:
        mse = (pred - gt).pow(2).mean()
    if mse.item() <= 1e-12:
        return 99.0
    return float((-10.0 * torch.log10(mse)).item())


def ssim_global(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Simple global SSIM, masked by tight bbox when a mask is provided."""
    pred = pred.float()
    gt = gt.float()
    if mask is not None and mask.any():
        ys, xs = torch.where(mask.bool())
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        pred = pred[:, y0:y1, x0:x1]
        gt = gt[:, y0:y1, x0:x1]
        m = mask[y0:y1, x0:x1].bool()
        pred = pred * m.unsqueeze(0)
        gt = gt * m.unsqueeze(0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred.mean()
    mu_y = gt.mean()
    var_x = pred.var(unbiased=False)
    var_y = gt.var(unbiased=False)
    cov = ((pred - mu_x) * (gt - mu_y)).mean()
    val = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2))
    return float(val.clamp(-1, 1).item())


def canonical_cameras(num_views: int, *, width: int = 512, height: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    import utils3d

    extrinsics = []
    intrinsics = []
    for i in range(num_views):
        yaw = torch.tensor(float(i * 2 * math.pi / num_views), device="cuda")
        pitch = torch.tensor(0.0, device="cuda")
        fov = torch.deg2rad(torch.tensor(40.0, device="cuda"))
        origin = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ], device="cuda") * 2.0
        extrinsics.append(
            utils3d.torch.extrinsics_look_at(
                origin,
                torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
                torch.tensor([0, 0, 1], dtype=torch.float32, device="cuda"),
            )
        )
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    return torch.stack(extrinsics), torch.stack(intrinsics)


def dataset_cameras(data_root: Path, object_id: str, angle_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    import utils3d

    camera_path = require_file(
        render_dir(data_root, object_id, angle_idx) / "camera_transforms.json",
        "camera_transforms",
    )
    payload = json.loads(camera_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"camera_transforms.frames must be a non-empty list: {camera_path}")
    extrinsics = []
    intrinsics = []
    for idx, frame in enumerate(frames):
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device="cuda")
        if tuple(c2w.shape) != (4, 4):
            raise RuntimeError(f"camera frame {idx} transform_matrix must be 4x4: {camera_path}")
        # Same Blender -> TRELLIS convention as 12_encode_part_synthesis_slat.py.
        c2w = c2w.clone()
        c2w[:3, 1:3] *= -1
        extrinsics.append(torch.inverse(c2w))
        fov = torch.tensor(float(frame["camera_angle_x"]), dtype=torch.float32, device="cuda")
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    return torch.stack(extrinsics), torch.stack(intrinsics)


def _intrinsics_to_pixel_k(intrinsics: torch.Tensor, width: int, height: int) -> torch.Tensor:
    k = intrinsics.clone()
    k[..., 0, 0] *= float(width)
    k[..., 1, 1] *= float(height)
    k[..., 0, 2] *= float(width)
    k[..., 1, 2] *= float(height)
    return k


def render_gaussian(gaussian: Any, renderer: Any, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    backend = renderer["backend"] if isinstance(renderer, dict) else "diff-gaussian"
    if backend == "diff-gaussian":
        from trellis.utils.arts.slat_render_utils import render_sample_to_views
        return render_sample_to_views(gaussian, extrinsics.cuda(), intrinsics.cuda(), renderer["renderer"]).detach().cpu()

    if backend != "gsplat":
        raise ValueError(f"unknown renderer backend: {backend}")

    from gsplat.rendering import rasterization

    width = int(renderer["width"])
    height = int(renderer["height"])
    means = gaussian.get_xyz.detach().float().cuda().contiguous()
    quats = gaussian.get_rotation.detach().float().cuda().contiguous()
    scales = gaussian.get_scaling.detach().float().cuda().contiguous()
    opacities = gaussian.get_opacity.detach().float().cuda().reshape(-1).contiguous()
    colors = gaussian.get_features.detach().float().cuda().contiguous()
    sh_degree = int(getattr(gaussian, "active_sh_degree", 0))
    viewmats = extrinsics.detach().float().cuda().contiguous()
    ks = _intrinsics_to_pixel_k(intrinsics.detach().float().cuda(), width, height).contiguous()
    with torch.no_grad():
        render, _alpha, _meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            ks,
            width,
            height,
            near_plane=float(renderer["near"]),
            far_plane=float(renderer["far"]),
            sh_degree=sh_degree,
            backgrounds=None,
            rasterize_mode="classic",
        )
    return render.clamp(0, 1).permute(0, 3, 1, 2).detach().cpu()


def decode_render(
    slat: Any,
    decoder_ckpt: Path,
    normalized: bool,
    renderer: Any,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    assets = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(decoder_ckpt),
        slat_is_normalized=normalized,
    )
    return render_gaussian(assets["gaussian"], renderer=renderer, extrinsics=extrinsics, intrinsics=intrinsics)


def decode_gaussian_stats(slat: Any, decoder_ckpt: Path, normalized: bool) -> dict[str, Any]:
    assets = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(decoder_ckpt),
        slat_is_normalized=normalized,
    )
    gaussian = assets["gaussian"]

    def tensor_range(t: torch.Tensor) -> list[float]:
        t = t.detach().float()
        return [float(t.min().item()), float(t.max().item())]

    return {
        "num_gaussians": int(gaussian.get_xyz.shape[0]),
        "xyz_range": tensor_range(gaussian.get_xyz),
        "opacity_range": tensor_range(gaussian.get_opacity),
        "scaling_range": tensor_range(gaussian.get_scaling),
        "features_range": tensor_range(gaussian.get_features),
    }


def norm_feats(raw: np.ndarray) -> np.ndarray:
    return ((torch.from_numpy(raw).float() - SLAT_MEAN.view(1, -1)) / SLAT_STD.view(1, -1)).numpy()


def feat_metrics(pred_norm: torch.Tensor, gt_norm_np: np.ndarray) -> tuple[float, float]:
    gt = torch.from_numpy(gt_norm_np).to(pred_norm.device).float()
    pred = pred_norm.float()
    if pred.shape != gt.shape:
        raise RuntimeError(f"pred/gt feature shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    l2 = torch.linalg.norm(pred - gt, dim=1).mean()
    cos = torch.nn.functional.cosine_similarity(pred, gt, dim=1).mean()
    return float(l2.item()), float(cos.item())


def save_strip(path: Path, views: torch.Tensor, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = []
    for v in range(views.shape[0]):
        arr = (views[v].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        imgs.append(Image.fromarray(arr))
    w, h = imgs[0].size
    title_h = 28 if title else 0
    canvas = Image.new("RGB", (w * len(imgs), h + title_h), (0, 0, 0))
    if title:
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), title, fill=(255, 255, 255))
    for i, img in enumerate(imgs):
        canvas.paste(img, (i * w, title_h))
    canvas.save(path)


def save_triptych(path: Path, gt: torch.Tensor, pred4: torch.Tensor, pred1: torch.Tensor, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # first canonical view for compact report; full strips are saved separately.
    panels = [("GT-latent", gt[0]), ("pred-4v", pred4[0]), ("pred-1v", pred1[0])]
    imgs = []
    for label, t in panels:
        arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 150, 24), fill=(0, 0, 0))
        draw.text((6, 5), label, fill=(255, 255, 255))
        imgs.append(img)
    w, h = imgs[0].size
    title_h = 32
    canvas = Image.new("RGB", (3 * w, h + title_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(255, 255, 255))
    for i, img in enumerate(imgs):
        canvas.paste(img, (i * w, title_h))
    canvas.save(path)


def assess_quality(psnr_vs_blender: float | None) -> str:
    if psnr_vs_blender is None or math.isnan(psnr_vs_blender):
        return "see-render"
    if psnr_vs_blender >= 20:
        return "好"
    if psnr_vs_blender >= 14:
        return "中"
    return "差"


def run_case(
    data_root: Path,
    manifest_row: dict[str, Any],
    object_id: str,
    angle_idx: int,
    part: str,
    flow_ckpt: Path,
    decoder_ckpt: Path,
    renderer: Any,
    out_dir: Path,
    dataset_index: int,
    num_steps: int,
    skip_render: bool,
    render_blocker: str | None,
) -> dict[str, Any]:
    print(f"[case] object={object_id} angle={angle_idx} part={part}", flush=True)
    view_indices = list(manifest_row["view_indices"])
    manifest_paths = manifest_row["paths"]
    print(f"[case] manifest_view_indices={view_indices}", flush=True)
    latent = load_gt_latent(part_latent_path(data_root, object_id, angle_idx, part))
    token_file = tokens_path(data_root, object_id, angle_idx)
    tokens = load_tokens(token_file)
    tokens4 = tokens[view_indices]
    tokens1 = tokens[[view_indices[0]]]

    gt_slat = make_sparse_tensor(latent["coords"], latent["feats"])
    gt_norm_np = norm_feats(latent["feats"])

    gt_render = None
    a_psnr = float("nan")
    a_ssim = float("nan")
    a_quality = f"NA: {render_blocker}" if skip_render else "NA"
    b_psnr_4 = float("nan")
    b_psnr_1 = float("nan")
    triptych = None
    gt_gaussian_stats = decode_gaussian_stats(gt_slat, decoder_ckpt, normalized=False)
    a_visible_views: list[int] = []

    if not skip_render:
        data_ext, data_intr = dataset_cameras(data_root, object_id, angle_idx)
        gt_data_render = decode_render(
            gt_slat,
            decoder_ckpt,
            normalized=False,
            renderer=renderer,
            extrinsics=data_ext,
            intrinsics=data_intr,
        )

    label = load_part_label(data_root, object_id, part)
    if not skip_render:
        view_psnrs = []
        view_ssims = []
        for local_idx, view_idx in enumerate(view_indices):
            rgb_path = resolve_manifest_rel_path(data_root, manifest_paths["rgb"][local_idx], field=f"paths.rgb[{local_idx}]")
            mask_path = resolve_manifest_rel_path(data_root, manifest_paths["masks"][local_idx], field=f"paths.masks[{local_idx}]")
            rgb = image_to_tensor(rgb_path)
            mask_np = np.load(require_file(mask_path, f"manifest mask view {view_idx}"))
            if mask_np.shape != tuple(rgb.shape[-2:]):
                raise RuntimeError(
                    f"mask/RGB shape mismatch for {object_id} angle {angle_idx} view {view_idx}: "
                    f"{mask_np.shape} vs {tuple(rgb.shape[-2:])}"
                )
            mask = torch.from_numpy(mask_np == label)
            if mask.sum().item() == 0:
                continue
            a_visible_views.append(view_idx)
            view_psnrs.append(psnr(gt_data_render[view_idx], rgb, mask))
            view_ssims.append(ssim_global(gt_data_render[view_idx], rgb, mask))
        if view_psnrs:
            a_psnr = float(np.nanmean(np.array(view_psnrs, dtype=np.float32)))
            a_ssim = float(np.nanmean(np.array(view_ssims, dtype=np.float32)))
        gt_ext, gt_intr = canonical_cameras(4)
        gt_render = decode_render(
            gt_slat,
            decoder_ckpt,
            normalized=False,
            renderer=renderer,
            extrinsics=gt_ext,
            intrinsics=gt_intr,
        )
        a_quality = assess_quality(a_psnr)

    part_coords = {part: torch.from_numpy(latent["coords"]).long()}
    pred4 = inference.run_slat_flow_per_part(
        tokens4,
        part_coords,
        str(flow_ckpt),
        num_steps=num_steps,
        base_seed=42,
        dataset_index=dataset_index,
    )[part]
    pred1 = inference.run_slat_flow_per_part(
        tokens1,
        part_coords,
        str(flow_ckpt),
        num_steps=num_steps,
        base_seed=42,
        dataset_index=dataset_index,
    )[part]

    pred4_feats = pred4.feats.detach().float().cpu()
    pred1_feats = pred1.feats.detach().float().cpu()
    l2_4, cos_4 = feat_metrics(pred4_feats, gt_norm_np)
    l2_1, cos_1 = feat_metrics(pred1_feats, gt_norm_np)
    pred4_gaussian_stats = decode_gaussian_stats(pred4, decoder_ckpt, normalized=True)
    pred1_gaussian_stats = decode_gaussian_stats(pred1, decoder_ckpt, normalized=True)

    base = f"{object_id}_angle_{angle_idx}_{part}"
    if not skip_render:
        pred4_render = decode_render(
            pred4,
            decoder_ckpt,
            normalized=True,
            renderer=renderer,
            extrinsics=gt_ext,
            intrinsics=gt_intr,
        )
        pred1_render = decode_render(
            pred1,
            decoder_ckpt,
            normalized=True,
            renderer=renderer,
            extrinsics=gt_ext,
            intrinsics=gt_intr,
        )
        b_psnr_4 = psnr(pred4_render, gt_render)
        b_psnr_1 = psnr(pred1_render, gt_render)

        triptych = out_dir / "triptychs" / f"{base}.png"
        save_triptych(triptych, gt_render, pred4_render, pred1_render, f"{object_id}/angle_{angle_idx} {part}")
        save_strip(out_dir / "strips" / f"{base}_gt_latent.png", gt_render, f"{base} GT-latent canonical views")
        save_strip(out_dir / "strips" / f"{base}_pred4.png", pred4_render, f"{base} pred-4v canonical views")
        save_strip(out_dir / "strips" / f"{base}_pred1.png", pred1_render, f"{base} pred-1v canonical views")

    raw = latent["feats"]
    row = {
        "object_id": object_id,
        "angle_idx": angle_idx,
        "part": part,
        "voxel_count": int(latent["coords"].shape[0]),
        "tokens_path": str(token_file.resolve()),
        "manifest_view_indices": view_indices,
        "manifest_rgb_paths": [
            str(resolve_manifest_rel_path(data_root, value, field=f"paths.rgb[{idx}]").resolve())
            for idx, value in enumerate(manifest_paths["rgb"])
        ],
        "manifest_mask_paths": [
            str(resolve_manifest_rel_path(data_root, value, field=f"paths.masks[{idx}]").resolve())
            for idx, value in enumerate(manifest_paths["masks"])
        ],
        "cond_tokens4_shape": list(tokens4.shape),
        "cond_tokens1_shape": list(tokens1.shape),
        "latent_path": str(part_latent_path(data_root, object_id, angle_idx, part).resolve()),
        "mask_label": label,
        "gt_raw_range": [float(raw.min()), float(raw.max())],
        "gt_norm_range": [float(gt_norm_np.min()), float(gt_norm_np.max())],
        "pred4_norm_range": [float(pred4_feats.min()), float(pred4_feats.max())],
        "pred1_norm_range": [float(pred1_feats.min()), float(pred1_feats.max())],
        "test_a_quality": a_quality,
        "test_a_blender_mask_psnr_view0": a_psnr,
        "test_a_blender_mask_ssim_view0": a_ssim,
        "test_a_visible_views": a_visible_views,
        "gt_gaussian_stats": gt_gaussian_stats,
        "feat_l2_4v": l2_4,
        "feat_l2_1v": l2_1,
        "cosine_4v": cos_4,
        "cosine_1v": cos_1,
        "pred4_gaussian_stats": pred4_gaussian_stats,
        "pred1_gaussian_stats": pred1_gaussian_stats,
        "b_render_psnr_4v_vs_gt_latent": b_psnr_4,
        "b_render_psnr_1v_vs_gt_latent": b_psnr_1,
        "triptych": None if triptych is None else str(triptych.resolve()),
        "render_blocker": render_blocker,
    }
    print(
        "[case-done] "
        f"{object_id}/angle_{angle_idx} {part} vox={row['voxel_count']} "
        f"A={a_quality} Apsnr={a_psnr:.3f} "
        f"L2(4/1)={l2_4:.3f}/{l2_1:.3f} cos(4/1)={cos_4:.3f}/{cos_1:.3f} "
        f"Bpsnr(4/1)={b_psnr_4:.3f}/{b_psnr_1:.3f}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--flow-ckpt", type=Path, required=True)
    parser.add_argument("--decoder-ckpt", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "slat_gt_ceiling_gap")
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--renderer", choices=("diff-gaussian", "gsplat"), default="diff-gaussian")
    parser.add_argument(
        "--case",
        action="append",
        help="object_id:angle_idx:part. May be repeated.",
    )
    args = parser.parse_args()

    data_root = require_dir(args.data_root, "DATA_ROOT")
    if args.case:
        cases = []
        for item in args.case:
            bits = item.split(":", 2)
            if len(bits) != 3:
                raise ValueError(f"--case must be object:angle:part, got {item!r}")
            cases.append((bits[0], int(bits[1]), bits[2]))
    else:
        cases = [
            ("100058", 0, "button_(top_handle)_0"),
            ("100058", 0, "lid_0"),
            ("100033", 0, "lid_0"),
        ]

    manifest_path = require_file(args.manifest_path or default_manifest_path(data_root), "part synthesis manifest")
    manifest_rows = load_manifest_rows(manifest_path, data_root, {(object_id, angle_idx) for object_id, angle_idx, _part in cases})
    flow_ckpt = require_trellis_safetensors(args.flow_ckpt, "SLat Flow DiT checkpoint")
    decoder_ckpt = require_trellis_safetensors(args.decoder_ckpt, "SLat gaussian VAE decoder checkpoint")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[env] project_root={PROJECT_ROOT}", flush=True)
    print(f"[env] data_root={data_root}", flush=True)
    print(f"[env] manifest_path={manifest_path}", flush=True)
    print(f"[env] output_dir={out_dir}", flush=True)
    print(f"[norm] mean={SLAT_MEAN.tolist()}", flush=True)
    print(f"[norm] std={SLAT_STD.tolist()}", flush=True)

    probe_official_ckpts(flow_ckpt, decoder_ckpt)
    print(f"[ckpt] slat_flow={flow_ckpt}", flush=True)
    print(f"[ckpt] gaussian_decoder={decoder_ckpt}", flush=True)

    renderer = None
    render_blocker = None
    if args.skip_render:
        render_blocker = "skipped by --skip-render"
    elif args.renderer == "diff-gaussian":
        try:
            import diff_gaussian_rasterization  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Gaussian rendering requires diff_gaussian_rasterization; "
                "install the compiled CUDA extension or rerun with --skip-render "
                "to compute latent-only metrics."
            ) from exc
        from trellis.utils.arts.slat_render_utils import load_gaussian_renderer
        renderer = {"backend": "diff-gaussian", "renderer": load_gaussian_renderer()}
    else:
        try:
            import gsplat.rendering  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "gsplat renderer requested, but gsplat is not importable. "
                "Set PYTHONPATH to the local gsplat source root."
            ) from exc
        renderer = {"backend": "gsplat", "width": 512, "height": 512, "near": 0.8, "far": 10.0}

    rows = []
    for i, (object_id, angle_idx, part) in enumerate(cases):
        row = manifest_case_row(data_root, manifest_rows, object_id, angle_idx, part)
        rows.append(
            run_case(
                data_root=data_root,
                manifest_row=row,
                object_id=object_id,
                angle_idx=angle_idx,
                part=part,
                flow_ckpt=flow_ckpt,
                decoder_ckpt=decoder_ckpt,
                renderer=renderer,
                out_dir=out_dir,
                dataset_index=i,
                num_steps=args.num_steps,
                skip_render=args.skip_render,
                render_blocker=render_blocker,
            )
        )

    report = {
        "project_root": str(PROJECT_ROOT),
        "data_root": str(data_root),
        "manifest_path": str(manifest_path.resolve()),
        "flow_ckpt": str(flow_ckpt),
        "gaussian_decoder_ckpt": str(decoder_ckpt),
        "normalization": {"mean": SLAT_MEAN.tolist(), "std": SLAT_STD.tolist()},
        "num_steps": args.num_steps,
        "renderer": args.renderer,
        "rows": rows,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
