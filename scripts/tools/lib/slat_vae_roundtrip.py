#!/usr/bin/env python3
"""SLat Gaussian VAE appearance round-trip QC on dataset GT latents."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
SAM3D_ROOT = PROJECT_ROOT / "submodules" / "sam3d-stage" / "submodules" / "sam-3d-objects"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import diff_gaussian_rasterization  # noqa: F401,E402
import inference  # noqa: E402

# inference.py registers a lightweight trellis.renderers shell. Replace it with
# the real lazy package so importing trellis.renderers.gaussian_render exercises
# the official TRELLIS renderer that imports diff_gaussian_rasterization.
sys.modules.pop("trellis.renderers", None)
gaussian_render_mod = importlib.import_module("trellis.renderers.gaussian_render")
GaussianRenderer = gaussian_render_mod.GaussianRenderer


DEFAULT_CASES = [
    "100058:0",
    "100033:0",
]

EXPECTED_TOKEN_SHAPE = (12, 1370, 1024)
EXPECTED_PATCH_GRID = 37
EXPECTED_FEATURE_DIM = 1024


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


def parse_case(value: str) -> tuple[str, int]:
    bits = value.split(":")
    if len(bits) != 2 or not bits[0]:
        raise ValueError(f"--case must be object_id:angle_idx, got {value!r}")
    return bits[0], int(bits[1])


def instance_id(object_id: str, angle_idx: int) -> str:
    return f"{object_id}_angle_{angle_idx}"


def slat_instance_root(data_root: Path, object_id: str, angle_idx: int) -> Path:
    inst = instance_id(object_id, angle_idx)
    return require_dir(data_root / "part_synthesis_slat" / object_id[:2] / inst, "SLat instance root")


def render_root(data_root: Path, object_id: str, angle_idx: int) -> Path:
    return require_dir(data_root / "renders" / object_id / f"angle_{angle_idx}", "render root")


def part_info_path(data_root: Path, object_id: str) -> Path:
    return require_file(data_root / "reconstruction" / "part_info" / object_id / "part_info.json", "part_info")


def token_path(data_root: Path, object_id: str, angle_idx: int) -> Path:
    return require_file(
        data_root / "reconstruction" / "dinov2_tokens" / object_id / f"angle_{angle_idx}" / "tokens.npz",
        "DINOv2 tokens",
    )


def surface_path(data_root: Path, object_id: str, angle_idx: int) -> Path:
    return require_file(
        data_root / "reconstruction" / "voxel_expanded" / object_id / f"angle_{angle_idx}" / "64" / "surface.npy",
        "overall surface voxel",
    )


def part_voxel_path(data_root: Path, object_id: str, angle_idx: int, part: str) -> Path:
    return require_file(
        data_root / "reconstruction" / "voxel_expanded" / object_id / f"angle_{angle_idx}" / "64" / f"ind_{part}.npy",
        f"part voxel {part}",
    )


def load_latent(path: Path) -> dict[str, np.ndarray]:
    path = require_file(path, "SLat latent")
    with np.load(path) as data:
        keys = set(data.files)
        if keys != {"coords", "feats"}:
            raise RuntimeError(f"{path} keys must be exactly {{'coords','feats'}}, got {sorted(keys)}")
        coords = data["coords"]
        feats = data["feats"]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"{path} coords must be [N,3], got {coords.shape}")
    if not np.issubdtype(coords.dtype, np.integer):
        raise RuntimeError(f"{path} coords dtype must be integer, got {coords.dtype}")
    if coords.size and (int(coords.min()) < 0 or int(coords.max()) >= 64):
        raise RuntimeError(f"{path} coords out of [0,64): min={int(coords.min())} max={int(coords.max())}")
    if feats.ndim != 2 or feats.shape[1] != 8:
        raise RuntimeError(f"{path} feats must be [N,8], got {feats.shape}")
    if feats.dtype != np.float32:
        raise RuntimeError(f"{path} feats dtype must be float32, got {feats.dtype}")
    if coords.shape[0] != feats.shape[0]:
        raise RuntimeError(f"{path} coords/feats row mismatch: {coords.shape[0]} vs {feats.shape[0]}")
    if coords.shape[0] == 0:
        raise RuntimeError(f"{path} contains zero voxels")
    if not np.isfinite(feats).all():
        raise RuntimeError(f"{path} feats contain NaN or Inf")
    return {"coords": coords.astype(np.int64, copy=False), "feats": feats.astype(np.float32, copy=False)}


def load_voxel_coords(path: Path) -> np.ndarray:
    path = require_file(path, "voxel coords")
    coords = np.load(path)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"{path} voxel coords must be [N,3], got {coords.shape}")
    if not np.issubdtype(coords.dtype, np.integer):
        raise RuntimeError(f"{path} voxel coords dtype must be integer, got {coords.dtype}")
    if coords.shape[0] == 0:
        raise RuntimeError(f"{path} contains zero voxels")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise RuntimeError(f"{path} coords out of [0,64): min={int(coords.min())} max={int(coords.max())}")
    return np.ascontiguousarray(coords.astype(np.int64, copy=False))


def make_sparse_tensor(latent: dict[str, np.ndarray]):
    from trellis.modules.sparse import SparseTensor

    coords = torch.from_numpy(latent["coords"].astype(np.int32, copy=False)).cuda()
    feats = torch.from_numpy(latent["feats"].astype(np.float32, copy=False)).cuda()
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    return SparseTensor(coords=torch.cat([batch, coords], dim=1), feats=feats)


def make_sparse_tensor_from_coords_feats(coords_np: np.ndarray, feats: torch.Tensor):
    from trellis.modules.sparse import SparseTensor

    coords = torch.as_tensor(coords_np.astype(np.int32, copy=False), dtype=torch.int32, device=feats.device)
    if coords.shape[0] != feats.shape[0]:
        raise RuntimeError(f"coords/features row mismatch: {coords.shape[0]} vs {feats.shape[0]}")
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=feats.device)
    return SparseTensor(coords=torch.cat([batch, coords], dim=1), feats=feats)


def trellis_slat_to_sam3d_sparse(slat: Any):
    if not SAM3D_ROOT.is_dir():
        raise FileNotFoundError(f"SAM3D root not found: {SAM3D_ROOT}")
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    if str(SAM3D_ROOT) not in sys.path:
        sys.path.insert(0, str(SAM3D_ROOT))
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sam3d_sp

    if not hasattr(slat, "coords") or not hasattr(slat, "feats"):
        raise RuntimeError(f"cannot convert unsupported SLat type {type(slat).__name__}")
    coords = slat.coords.detach().to(device="cuda", dtype=torch.int32).contiguous()
    feats = slat.feats.detach().to(device="cuda", dtype=torch.float32).contiguous()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise RuntimeError(f"SLat coords must be [N,4] with batch column, got {tuple(coords.shape)}")
    if feats.ndim != 2 or feats.shape[1] != 8:
        raise RuntimeError(f"SLat feats must be [N,8], got {tuple(feats.shape)}")
    if coords.shape[0] != feats.shape[0]:
        raise RuntimeError(f"SLat coords/features row mismatch: {coords.shape[0]} vs {feats.shape[0]}")
    return sam3d_sp.SparseTensor(feats=feats, coords=coords)


def load_slat_encoder(encoder_ckpt: Path):
    import json
    from trellis.models.structured_latent_vae.encoder import SLatEncoder

    encoder_ckpt = require_file(encoder_ckpt, "SLat encoder checkpoint")
    if encoder_ckpt.suffix != ".safetensors":
        raise ValueError(f"SLat encoder checkpoint must be .safetensors, got {encoder_ckpt}")
    cfg_path = require_file(encoder_ckpt.with_suffix(".json"), "SLat encoder config")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("name") != "SLatEncoder":
        raise RuntimeError(f"{cfg_path} expected name='SLatEncoder', got {cfg.get('name')!r}")
    encoder = SLatEncoder(**cfg["args"]).cuda().eval()
    state = load_file(str(encoder_ckpt.resolve()), device="cuda")
    missing, unexpected = encoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"SLat encoder strict load mismatch: missing={missing}, unexpected={unexpected}")
    for param in encoder.parameters():
        param.requires_grad_(False)
    print(f"[ckpt] encoder={encoder_ckpt.resolve()}", flush=True)
    return encoder


def load_trellis_decoder(decoder_ckpt: Path) -> tuple[Any, dict[str, Any]]:
    decoder_ckpt = require_file(decoder_ckpt, "TRELLIS SLat Gaussian decoder checkpoint")
    if decoder_ckpt.suffix != ".safetensors":
        raise ValueError(f"TRELLIS decoder checkpoint must be .safetensors, got {decoder_ckpt}")
    cfg_path = require_file(decoder_ckpt.with_suffix(".json"), "TRELLIS SLat Gaussian decoder config")
    decoder = inference._load_slat_vae_decoder(str(decoder_ckpt.resolve()))
    for param in decoder.parameters():
        param.requires_grad_(False)
    info = {
        "family": "trellis",
        "checkpoint": str(decoder_ckpt.resolve()),
        "config": str(cfg_path.resolve()),
        "class": type(decoder).__name__,
        "module": type(decoder).__module__,
    }
    return decoder, info


def load_sam3d_decoder(decoder_ckpt: Path, decoder_config: Path | None) -> tuple[Any, dict[str, Any]]:
    decoder_ckpt = require_file(decoder_ckpt, "SAM3D SLat Gaussian decoder checkpoint")
    if decoder_ckpt.suffix != ".ckpt":
        raise ValueError(f"SAM3D decoder checkpoint must be .ckpt, got {decoder_ckpt}")
    config_path = decoder_config if decoder_config is not None else decoder_ckpt.with_suffix(".yaml")
    config_path = require_file(config_path, "SAM3D SLat Gaussian decoder config")
    if not SAM3D_ROOT.is_dir():
        raise FileNotFoundError(f"SAM3D root not found: {SAM3D_ROOT}")
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    if str(SAM3D_ROOT) not in sys.path:
        sys.path.insert(0, str(SAM3D_ROOT))
    import yaml
    from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs import (
        SLatGaussianDecoderTdfyWrapper,
    )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError(f"SAM3D decoder config must be a mapping: {config_path}")
    target = cfg.pop("_target_", None)
    expected_target = (
        "sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs."
        "SLatGaussianDecoderTdfyWrapper"
    )
    if target != expected_target:
        raise RuntimeError(f"{config_path} _target_ must be {expected_target!r}, got {target!r}")
    decoder = SLatGaussianDecoderTdfyWrapper(**cfg, pretrained_ckpt_path=str(decoder_ckpt.resolve())).cuda().eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    rep_config = getattr(decoder, "rep_config", {})
    info = {
        "family": "sam3d",
        "checkpoint": str(decoder_ckpt.resolve()),
        "config": str(config_path.resolve()),
        "target": target,
        "class": type(decoder).__name__,
        "module": type(decoder).__module__,
        "num_gaussians": int(rep_config["num_gaussians"]) if isinstance(rep_config, dict) and "num_gaussians" in rep_config else None,
        "parameter_count": int(sum(param.numel() for param in decoder.parameters())),
    }
    return decoder, info


def load_patchtokens(path: Path) -> torch.Tensor:
    with np.load(require_file(path, "DINOv2 tokens")) as data:
        if set(data.files) != {"tokens"}:
            raise RuntimeError(f"{path} keys must be exactly ['tokens'], got {data.files}")
        tokens = data["tokens"]
    if tuple(tokens.shape) != EXPECTED_TOKEN_SHAPE:
        raise RuntimeError(f"{path} expected tokens shape {EXPECTED_TOKEN_SHAPE}, got {tokens.shape}")
    patch = torch.from_numpy(tokens[:, 1:, :].astype(np.float32, copy=False)).cuda()
    return patch.permute(0, 2, 1).reshape(12, EXPECTED_FEATURE_DIM, EXPECTED_PATCH_GRID, EXPECTED_PATCH_GRID)


def coords_to_positions(coords_np: np.ndarray, resolution: int = 64) -> torch.Tensor:
    coords = torch.as_tensor(coords_np, dtype=torch.float32, device="cuda")
    return (coords + 0.5) / float(resolution) - 0.5


def project_features(coords_np: np.ndarray, patchtokens: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    import utils3d

    positions = coords_to_positions(coords_np, resolution=64)
    uv = utils3d.torch.project_cv(positions, extrinsics, intrinsics)[0] * 2 - 1
    sampled = F.grid_sample(
        patchtokens,
        uv.unsqueeze(1),
        mode="bilinear",
        align_corners=False,
    ).squeeze(2).permute(0, 2, 1)
    return sampled.mean(dim=0).float()


def encode_coords_to_slat(encoder: Any, coords_np: np.ndarray, patchtokens: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor):
    feats = project_features(coords_np, patchtokens, extrinsics, intrinsics)
    sparse = make_sparse_tensor_from_coords_feats(coords_np, feats)
    with torch.no_grad():
        latent = encoder(sparse, sample_posterior=False)
    if not hasattr(latent, "coords") or not hasattr(latent, "feats"):
        raise RuntimeError(f"SLat encoder returned unsupported type {type(latent).__name__}")
    if not torch.isfinite(latent.feats).all():
        raise RuntimeError("SLat encoder returned NaN or Inf feats")
    if latent.feats.ndim != 2 or latent.feats.shape[1] != 8:
        raise RuntimeError(f"SLat encoder feats must be [N,8], got {tuple(latent.feats.shape)}")
    return latent


def sparse_to_np(slat: Any) -> dict[str, np.ndarray]:
    coords = slat.coords[:, 1:].detach().cpu().numpy().astype(np.int64, copy=False)
    feats = slat.feats.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return {"coords": coords, "feats": feats}


def latent_feature_delta(encoded: dict[str, np.ndarray], cached: dict[str, np.ndarray]) -> dict[str, Any]:
    if encoded["coords"].shape != cached["coords"].shape:
        return {"coords_match": False, "reason": f"shape {encoded['coords'].shape} != {cached['coords'].shape}"}
    coords_match = bool(np.array_equal(encoded["coords"], cached["coords"]))
    if not coords_match:
        return {"coords_match": False, "reason": "coords arrays differ"}
    diff = encoded["feats"] - cached["feats"]
    l2 = np.linalg.norm(diff, axis=1)
    cos = np.sum(encoded["feats"] * cached["feats"], axis=1) / (
        np.linalg.norm(encoded["feats"], axis=1) * np.linalg.norm(cached["feats"], axis=1) + 1e-12
    )
    return {
        "coords_match": True,
        "feat_l2_mean": float(l2.mean()),
        "feat_l2_max": float(l2.max()),
        "feat_abs_max": float(np.abs(diff).max()),
        "cosine_mean": float(cos.mean()),
    }


def load_camera_matrices(camera_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    import utils3d

    payload = json.loads(require_file(camera_path, "camera_transforms").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != 12:
        raise RuntimeError(f"camera_transforms.frames must have length 12: {camera_path}")
    extrinsics = []
    intrinsics = []
    centers = []
    for idx, frame in enumerate(frames):
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device="cuda")
        if tuple(c2w.shape) != (4, 4):
            raise RuntimeError(f"camera frame {idx} transform_matrix must be 4x4: {camera_path}")
        c2w = c2w.clone()
        c2w[:3, 1:3] *= -1
        centers.append(c2w[:3, 3].detach().cpu())
        extrinsics.append(torch.inverse(c2w))
        fov = torch.tensor(float(frame["camera_angle_x"]), dtype=torch.float32, device="cuda")
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    center_norm = torch.stack(centers).norm(dim=1)
    meta = {
        "aabb": payload.get("aabb"),
        "scale": payload.get("scale"),
        "offset": payload.get("offset"),
        "resolution": payload.get("resolution"),
        "fov_deg": payload.get("fov_deg"),
        "total_views": payload.get("total_views"),
        "camera_center_norm_range": [float(center_norm.min()), float(center_norm.max())],
    }
    return torch.stack(extrinsics), torch.stack(intrinsics), meta


def load_rgb(path: Path) -> torch.Tensor:
    img = Image.open(require_file(path, "RGB image")).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_mask(path: Path) -> torch.Tensor:
    arr = np.load(require_file(path, "mask npy"))
    if arr.ndim != 2:
        raise RuntimeError(f"{path} mask must be [H,W], got {arr.shape}")
    return torch.from_numpy(arr.astype(np.int64, copy=False))


def load_tokens_shape(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        if set(data.files) != {"tokens"}:
            raise RuntimeError(f"{path} keys must be exactly ['tokens'], got {data.files}")
        tokens = data["tokens"]
        if tokens.shape != (12, 1370, 1024):
            raise RuntimeError(f"{path} tokens must be [12,1370,1024], got {tokens.shape}")
        return {
            "path": str(path.resolve()),
            "shape": list(tokens.shape),
            "dtype": str(tokens.dtype),
            "range": [float(tokens.min()), float(tokens.max())],
        }


def load_part_labels(data_root: Path, object_id: str) -> dict[str, int]:
    payload = json.loads(part_info_path(data_root, object_id).read_text(encoding="utf-8"))
    parts = payload.get("parts")
    if not isinstance(parts, dict):
        raise RuntimeError(f"part_info.parts must be object: {part_info_path(data_root, object_id)}")
    out: dict[str, int] = {}
    for name, item in parts.items():
        if not isinstance(item, dict):
            raise RuntimeError(f"part_info.parts[{name!r}] must be object")
        label = item.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
            raise RuntimeError(f"part_info.parts[{name!r}].label must be positive int, got {label!r}")
        out[name] = label
    return out


def make_renderer(resolution: int, near: float, far: float, decoder: Any) -> Any:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = float(near)
    renderer.rendering_options.far = float(far)
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 1
    rep_config = getattr(decoder, "rep_config", None)
    if isinstance(rep_config, dict) and "2d_filter_kernel_size" in rep_config:
        renderer.pipe.kernel_size = float(rep_config["2d_filter_kernel_size"])
    else:
        renderer.pipe.kernel_size = 0.1
    return renderer


def render_views(gaussian: Any, renderer: Any, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    frames = []
    with torch.no_grad():
        for idx in range(extrinsics.shape[0]):
            out = renderer.render(gaussian, extrinsics[idx], intrinsics[idx], colors_overwrite=None)
            if "color" not in out:
                raise RuntimeError(f"GaussianRenderer output for view {idx} missing 'color'")
            color = out["color"].detach().float().cpu().clamp(0, 1)
            if color.ndim != 3 or color.shape[0] != 3:
                raise RuntimeError(f"GaussianRenderer color for view {idx} must be [3,H,W], got {tuple(color.shape)}")
            frames.append(color)
    return torch.stack(frames, dim=0)


def _first_sample(out: Any) -> Any:
    return out[0] if hasattr(out, "__len__") else out


def decode_gaussian(slat: Any, decoder: Any, decoder_family: str) -> Any:
    if decoder_family == "trellis":
        slat_in = slat
    elif decoder_family == "sam3d":
        slat_in = trellis_slat_to_sam3d_sparse(slat)
    else:
        raise ValueError(f"unknown decoder_family {decoder_family!r}")
    with torch.no_grad():
        gaussian = _first_sample(decoder(slat_in))
    for attr in ("get_xyz", "get_features", "get_opacity", "get_scaling", "get_rotation"):
        if not hasattr(gaussian, attr):
            raise RuntimeError(f"decoded gaussian from {decoder_family} missing {attr}")
    return gaussian


def psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    if pred.shape != gt.shape:
        raise RuntimeError(f"PSNR shape mismatch: pred {tuple(pred.shape)} vs gt {tuple(gt.shape)}")
    mask = mask.bool()
    if mask.shape != pred.shape[-2:]:
        raise RuntimeError(f"PSNR mask shape mismatch: mask {tuple(mask.shape)} vs image {tuple(pred.shape[-2:])}")
    if mask.sum().item() == 0:
        return float("nan")
    mse = (pred[:, mask] - gt[:, mask]).pow(2).mean()
    if mse.item() <= 1e-12:
        return 99.0
    return float((-10.0 * torch.log10(mse)).item())


def ssim_global(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    if pred.shape != gt.shape:
        raise RuntimeError(f"SSIM shape mismatch: pred {tuple(pred.shape)} vs gt {tuple(gt.shape)}")
    mask = mask.bool()
    if mask.shape != pred.shape[-2:]:
        raise RuntimeError(f"SSIM mask shape mismatch: mask {tuple(mask.shape)} vs image {tuple(pred.shape[-2:])}")
    if not mask.any():
        return float("nan")
    ys, xs = torch.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pred_crop = pred[:, y0:y1, x0:x1]
    gt_crop = gt[:, y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1].float().unsqueeze(0)
    pred_crop = pred_crop * mask_crop
    gt_crop = gt_crop * mask_crop
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred_crop.mean()
    mu_y = gt_crop.mean()
    var_x = pred_crop.var(unbiased=False)
    var_y = gt_crop.var(unbiased=False)
    cov = ((pred_crop - mu_x) * (gt_crop - mu_y)).mean()
    val = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    )
    return float(val.clamp(-1, 1).item())


def quality(psnr_value: float, ssim_value: float) -> str:
    if math.isnan(psnr_value) or math.isnan(ssim_value):
        return "差"
    if psnr_value >= 22 and ssim_value >= 0.90:
        return "好"
    if psnr_value >= 16 and ssim_value >= 0.75:
        return "中"
    return "差"


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def make_pair_grid(
    out_path: Path,
    gt_views: list[torch.Tensor],
    recon_views: list[torch.Tensor],
    view_indices: list[int],
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(gt_views) != len(recon_views) or len(gt_views) != len(view_indices):
        raise RuntimeError("pair grid input length mismatch")
    cells: list[Image.Image] = []
    labels: list[str] = []
    for view_idx, gt, recon in zip(view_indices, gt_views, recon_views):
        cells.append(tensor_to_image(gt))
        labels.append(f"Blender GT v{view_idx}")
        cells.append(tensor_to_image(recon))
        labels.append(f"decode(GT) v{view_idx}")
    w, h = cells[0].size
    title_h = 30
    label_h = 22
    cols = 2
    rows = len(view_indices)
    canvas = Image.new("RGB", (cols * w, title_h + rows * (label_h + h)), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(255, 255, 255))
    for row_idx in range(rows):
        for col_idx in range(cols):
            idx = row_idx * cols + col_idx
            x = col_idx * w
            y = title_h + row_idx * (label_h + h)
            draw.rectangle((x, y, x + w, y + label_h), fill=(0, 0, 0))
            draw.text((x + 6, y + 5), labels[idx], fill=(255, 255, 255))
            canvas.paste(cells[idx], (x, y + label_h))
    canvas.save(out_path)


def make_part_grid(
    out_path: Path,
    renders: list[tuple[str, int, torch.Tensor]],
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cells = [(name, view_idx, tensor_to_image(image)) for name, view_idx, image in renders]
    if not cells:
        raise RuntimeError("part grid needs at least one render")
    w, h = cells[0][2].size
    cols = min(3, len(cells))
    rows = int(math.ceil(len(cells) / cols))
    title_h = 30
    label_h = 22
    canvas = Image.new("RGB", (cols * w, title_h + rows * (label_h + h)), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(255, 255, 255))
    for idx, (name, view_idx, image) in enumerate(cells):
        row = idx // cols
        col = idx % cols
        x = col * w
        y = title_h + row * (label_h + h)
        draw.rectangle((x, y, x + w, y + label_h), fill=(0, 0, 0))
        draw.text((x + 6, y + 5), f"{name} v{view_idx}", fill=(255, 255, 255))
        canvas.paste(image, (x, y + label_h))
    canvas.save(out_path)


def metric_across_views(
    recon: torch.Tensor,
    data_root: Path,
    object_id: str,
    angle_idx: int,
    view_indices: list[int],
    mask_mode: str,
    part_label: int | None = None,
) -> tuple[float, float, list[dict[str, float]]]:
    rroot = render_root(data_root, object_id, angle_idx)
    per_view = []
    psnrs = []
    ssims = []
    for view_idx in view_indices:
        rgb = load_rgb(rroot / "rgb" / f"view_{view_idx}.png")
        mask_int = load_mask(rroot / "mask" / f"mask_{view_idx}.npy")
        if mask_int.shape != tuple(rgb.shape[-2:]):
            raise RuntimeError(f"mask/RGB shape mismatch for {object_id}/angle_{angle_idx} view {view_idx}")
        if tuple(recon[view_idx].shape[-2:]) != tuple(rgb.shape[-2:]):
            raise RuntimeError(
                f"render/RGB shape mismatch for {object_id}/angle_{angle_idx} view {view_idx}: "
                f"{tuple(recon[view_idx].shape[-2:])} vs {tuple(rgb.shape[-2:])}"
            )
        if mask_mode == "foreground":
            mask = mask_int > 0
        elif mask_mode == "part":
            if part_label is None:
                raise RuntimeError("part_label is required when mask_mode='part'")
            mask = mask_int == int(part_label)
        else:
            raise ValueError(f"unknown mask_mode {mask_mode!r}")
        p = psnr(recon[view_idx], rgb, mask)
        s = ssim_global(recon[view_idx], rgb, mask)
        per_view.append({"view": int(view_idx), "mask_pixels": int(mask.sum().item()), "psnr": p, "ssim": s})
        if not math.isnan(p):
            psnrs.append(p)
        if not math.isnan(s):
            ssims.append(s)
    mean_psnr = float(np.mean(np.array(psnrs, dtype=np.float32))) if psnrs else float("nan")
    mean_ssim = float(np.mean(np.array(ssims, dtype=np.float32))) if ssims else float("nan")
    return mean_psnr, mean_ssim, per_view


def choose_views(data_root: Path, object_id: str, angle_idx: int, max_views: int) -> list[int]:
    rroot = render_root(data_root, object_id, angle_idx)
    visible = []
    for view_idx in range(12):
        mask = load_mask(rroot / "mask" / f"mask_{view_idx}.npy")
        if (mask > 0).any():
            visible.append(view_idx)
    if not visible:
        raise RuntimeError(f"no foreground mask pixels for {object_id}/angle_{angle_idx}")
    return visible[:max_views]


def inspect_sample_paths(data_root: Path, object_id: str, angle_idx: int) -> dict[str, Any]:
    inst = instance_id(object_id, angle_idx)
    sroot = slat_instance_root(data_root, object_id, angle_idx)
    rroot = render_root(data_root, object_id, angle_idx)
    overall = require_file(sroot / "overall" / "latent.npz", "overall SLat latent")
    part_latents = sorted(str(path.resolve()) for path in sroot.glob("*/latent.npz") if path.parent.name != "overall")
    return {
        "instance": inst,
        "mapping": {"instance": inst, "object_id": object_id, "angle_idx": int(angle_idx)},
        "overall_surface": str(surface_path(data_root, object_id, angle_idx).resolve()),
        "overall_latent": str(overall.resolve()),
        "part_latents": part_latents,
        "camera_transforms": str((rroot / "camera_transforms.json").resolve()),
        "rgb_dir": str((rroot / "rgb").resolve()),
        "mask_dir": str((rroot / "mask").resolve()),
        "tokens": str(token_path(data_root, object_id, angle_idx).resolve()),
    }


def run_case(
    data_root: Path,
    decoder: Any,
    decoder_family: str,
    encoder: Any,
    renderer: Any,
    object_id: str,
    angle_idx: int,
    out_dir: Path,
    max_views: int,
    tiny_vox: int,
) -> dict[str, Any]:
    print(f"[case] object={object_id} angle={angle_idx}", flush=True)
    paths = inspect_sample_paths(data_root, object_id, angle_idx)
    print(f"[paths] {json.dumps(paths, ensure_ascii=False)}", flush=True)
    rroot = render_root(data_root, object_id, angle_idx)
    extrinsics, intrinsics, camera_meta = load_camera_matrices(rroot / "camera_transforms.json")
    token_info = load_tokens_shape(token_path(data_root, object_id, angle_idx))
    view_indices = choose_views(data_root, object_id, angle_idx, max_views=max_views)
    print(f"[camera] {object_id}/angle_{angle_idx} meta={camera_meta} metric_views={view_indices}", flush=True)

    patchtokens = load_patchtokens(token_path(data_root, object_id, angle_idx))
    overall_latent_path = slat_instance_root(data_root, object_id, angle_idx) / "overall" / "latent.npz"
    cached_overall_latent = load_latent(overall_latent_path)
    overall_surface = load_voxel_coords(surface_path(data_root, object_id, angle_idx))
    overall_slat = encode_coords_to_slat(encoder, overall_surface, patchtokens, extrinsics, intrinsics)
    overall_latent = sparse_to_np(overall_slat)
    overall_cache_delta = latent_feature_delta(overall_latent, cached_overall_latent)
    overall_gaussian = decode_gaussian(overall_slat, decoder, decoder_family)
    overall_recon = render_views(overall_gaussian, renderer, extrinsics, intrinsics)
    overall_psnr, overall_ssim, overall_per_view = metric_across_views(
        overall_recon,
        data_root,
        object_id,
        angle_idx,
        view_indices,
        mask_mode="foreground",
    )
    overall_quality = quality(overall_psnr, overall_ssim)
    pair_views_gt = [load_rgb(rroot / "rgb" / f"view_{idx}.png") for idx in view_indices]
    pair_views_recon = [overall_recon[idx] for idx in view_indices]
    pair_path = out_dir / "images" / f"{object_id}_angle_{angle_idx}_overall_blender_vs_decode.png"
    make_pair_grid(pair_path, pair_views_gt, pair_views_recon, view_indices, f"{object_id}/angle_{angle_idx} overall")

    labels = load_part_labels(data_root, object_id)
    part_rows = []
    part_grid_renders = []
    for part_path in sorted(slat_instance_root(data_root, object_id, angle_idx).glob("*/latent.npz")):
        part_name = part_path.parent.name
        if part_name == "overall":
            continue
        if part_name not in labels:
            raise RuntimeError(f"{part_name!r} missing from part_info labels for {object_id}")
        cached_latent = load_latent(part_path)
        voxel_path = part_voxel_path(data_root, object_id, angle_idx, part_name)
        voxel_coords = load_voxel_coords(voxel_path)
        slat = encode_coords_to_slat(encoder, voxel_coords, patchtokens, extrinsics, intrinsics)
        latent = sparse_to_np(slat)
        cache_delta = latent_feature_delta(latent, cached_latent)
        gaussian = decode_gaussian(slat, decoder, decoder_family)
        recon = render_views(gaussian, renderer, extrinsics, intrinsics)
        visible_views = []
        for idx in range(12):
            mask_int = load_mask(rroot / "mask" / f"mask_{idx}.npy")
            if (mask_int == labels[part_name]).any():
                visible_views.append(idx)
        metric_views = visible_views[:max_views]
        if metric_views:
            part_psnr, part_ssim, per_view = metric_across_views(
                recon,
                data_root,
                object_id,
                angle_idx,
                metric_views,
                mask_mode="part",
                part_label=labels[part_name],
            )
        else:
            part_psnr, part_ssim, per_view = float("nan"), float("nan"), []
        part_quality = quality(part_psnr, part_ssim)
        part_rows.append(
            {
                "name": part_name,
                "voxel_count": int(latent["coords"].shape[0]),
                "tiny": bool(latent["coords"].shape[0] < tiny_vox),
                "voxel_path": str(voxel_path.resolve()),
                "latent_path": str(part_path.resolve()),
                "cache_delta": cache_delta,
                "mask_label": int(labels[part_name]),
                "visible_views": visible_views,
                "metric_views": metric_views,
                "psnr": part_psnr,
                "ssim": part_ssim,
                "quality": part_quality,
                "per_view": per_view,
                "raw_feat_range": [float(latent["feats"].min()), float(latent["feats"].max())],
                "gaussian_count": int(gaussian.get_xyz.shape[0]),
            }
        )
        if metric_views:
            part_grid_renders.append((part_name, metric_views[0], recon[metric_views[0]]))
        print(
            f"[part] {object_id}/angle_{angle_idx} {part_name} vox={latent['coords'].shape[0]} "
            f"tiny={latent['coords'].shape[0] < tiny_vox} PSNR={part_psnr:.3f} SSIM={part_ssim:.3f} Q={part_quality}",
            flush=True,
        )
    part_grid_path = None
    if part_grid_renders:
        part_grid_path = out_dir / "images" / f"{object_id}_angle_{angle_idx}_parts_decode.png"
        make_part_grid(part_grid_path, part_grid_renders, f"{object_id}/angle_{angle_idx} part decode(GT)")

    row = {
        "object_id": object_id,
        "angle_idx": int(angle_idx),
        "instance": instance_id(object_id, angle_idx),
        "paths": paths,
        "token_info": token_info,
        "camera_meta": camera_meta,
        "metric_views": view_indices,
        "overall": {
            "voxel_count": int(overall_latent["coords"].shape[0]),
            "voxel_path": str(surface_path(data_root, object_id, angle_idx).resolve()),
            "latent_path": str(overall_latent_path.resolve()),
            "cache_delta": overall_cache_delta,
            "raw_feat_range": [float(overall_latent["feats"].min()), float(overall_latent["feats"].max())],
            "gaussian_count": int(overall_gaussian.get_xyz.shape[0]),
            "psnr": overall_psnr,
            "ssim": overall_ssim,
            "quality": overall_quality,
            "per_view": overall_per_view,
            "pair_image": str(pair_path.resolve()),
        },
        "parts": part_rows,
        "part_grid_image": None if part_grid_path is None else str(part_grid_path.resolve()),
    }
    print(
        f"[case-done] {object_id}/angle_{angle_idx} overall vox={row['overall']['voxel_count']} "
        f"PSNR={overall_psnr:.3f} SSIM={overall_ssim:.3f} Q={overall_quality}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--encoder-ckpt", type=Path, required=True)
    parser.add_argument("--decoder-ckpt", type=Path, required=True)
    parser.add_argument("--decoder-config", type=Path, help="Required/used for --decoder-family sam3d unless decoder .yaml is adjacent.")
    parser.add_argument("--decoder-family", choices=("trellis", "sam3d"), default="trellis")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "slat_vae_roundtrip")
    parser.add_argument("--case", action="append", help="object_id:angle_idx. May be repeated.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--near", type=float, default=0.8)
    parser.add_argument("--far", type=float, default=1.6)
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--tiny-vox", type=int, default=50)
    args = parser.parse_args()

    if args.max_views <= 0 or args.max_views > 12:
        raise ValueError("--max-views must be in [1,12]")
    data_root = require_dir(args.data_root, "DATA_ROOT")
    encoder_ckpt = require_file(args.encoder_ckpt, "SLat encoder checkpoint")
    if encoder_ckpt.suffix != ".safetensors":
        raise ValueError(f"--encoder-ckpt must be .safetensors, got {encoder_ckpt}")
    require_file(encoder_ckpt.with_suffix(".json"), "SLat encoder config")
    decoder_ckpt = require_file(args.decoder_ckpt, "SLat Gaussian decoder checkpoint")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [parse_case(item) for item in (args.case or DEFAULT_CASES)]

    encoder = load_slat_encoder(encoder_ckpt)
    if args.decoder_family == "trellis":
        decoder, decoder_info = load_trellis_decoder(decoder_ckpt)
    elif args.decoder_family == "sam3d":
        decoder, decoder_info = load_sam3d_decoder(decoder_ckpt, args.decoder_config)
    else:
        raise ValueError(f"unknown --decoder-family {args.decoder_family!r}")
    renderer = make_renderer(args.resolution, args.near, args.far, decoder)
    print(f"[env] python={sys.executable}", flush=True)
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"[env] data_root={data_root}", flush=True)
    print(f"[ckpt] encoder={encoder_ckpt.resolve()}", flush=True)
    print(f"[ckpt] decoder={decoder_ckpt.resolve()}", flush=True)
    print(f"[decoder] {json.dumps(decoder_info, ensure_ascii=False)}", flush=True)
    print("[renderer] trellis.renderers.gaussian_render.GaussianRenderer", flush=True)
    print(f"[renderer] module={gaussian_render_mod.__file__}", flush=True)
    print(f"[renderer] diff_gaussian_module={diff_gaussian_rasterization.__file__}", flush=True)
    print(
        f"[renderer] resolution={renderer.rendering_options.resolution} near={renderer.rendering_options.near} "
        f"far={renderer.rendering_options.far} bg={renderer.rendering_options.bg_color} "
        f"kernel_size={renderer.pipe.kernel_size}",
        flush=True,
    )

    rows = []
    for object_id, angle_idx in cases:
        rows.append(
            run_case(
                data_root=data_root,
                decoder=decoder,
                decoder_family=args.decoder_family,
                encoder=encoder,
                renderer=renderer,
                object_id=object_id,
                angle_idx=angle_idx,
                out_dir=out_dir,
                max_views=args.max_views,
                tiny_vox=args.tiny_vox,
            )
        )

    report = {
        "project_root": str(PROJECT_ROOT),
        "data_root": str(data_root),
        "encoder_ckpt": str(encoder_ckpt.resolve()),
        "decoder_ckpt": str(decoder_ckpt.resolve()),
        "decoder_family": args.decoder_family,
        "decoder_info": decoder_info,
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "renderer": {
            "class": "trellis.renderers.gaussian_render.GaussianRenderer",
            "module": str(Path(gaussian_render_mod.__file__).resolve()),
            "diff_gaussian_module": str(Path(diff_gaussian_rasterization.__file__).resolve()),
            "resolution": int(renderer.rendering_options.resolution),
            "near": float(renderer.rendering_options.near),
            "far": float(renderer.rendering_options.far),
            "bg_color": list(renderer.rendering_options.bg_color),
            "kernel_size": float(renderer.pipe.kernel_size),
        },
        "tiny_vox_threshold": int(args.tiny_vox),
        "rows": rows,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
