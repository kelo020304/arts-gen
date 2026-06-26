#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
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
from trellis.pipelines.samplers import FlowEulerCfgSampler  # noqa: E402
from trellis.pipelines.samplers.flow_euler import FlowEulerSampler  # noqa: E402


def _restore_trellis_renderer_package() -> None:
    for name in list(sys.modules):
        if name == "trellis.renderers" or name.startswith("trellis.renderers."):
            sys.modules.pop(name, None)
    import trellis.renderers  # noqa: F401


_restore_trellis_renderer_package()
from trellis.renderers.gaussian_render import GaussianRenderer  # noqa: E402


DEFAULT_DATA_ROOT = Path("/mnt/robot-data-lab/jzh/art-gen/data/realappliance")
DEFAULT_OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-realappliance-gaussian-qc")
DEFAULT_DECODER = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
DEFAULT_SLAT_FLOW = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors"
DEFAULT_FLOW_RENDER = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-realappliance-gaussian-4view/train/realappliance/039-1/overall.png"
)


def _load_latent(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"coords", "feats"}:
            raise ValueError(f"{path}: expected keys coords/feats, got {sorted(data.files)}")
        coords = np.asarray(data["coords"])
        feats = np.asarray(data["feats"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: coords must be [N,3], got {coords.shape}")
    if feats.ndim != 2 or feats.shape[1] != 8:
        raise ValueError(f"{path}: feats must be [N,8], got {feats.shape}")
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(f"{path}: coords/feats row mismatch: {coords.shape[0]} vs {feats.shape[0]}")
    return (
        np.ascontiguousarray(coords.astype(np.int32, copy=False)),
        np.ascontiguousarray(feats.astype(np.float32, copy=False)),
    )


def _make_sparse(coords_np: np.ndarray, feats_np: np.ndarray) -> SparseTensor:
    coords = torch.from_numpy(coords_np).to(device="cuda", dtype=torch.int32)
    feats = torch.from_numpy(feats_np).to(device="cuda", dtype=torch.float32)
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    return SparseTensor(coords=torch.cat([batch, coords], dim=1), feats=feats)


def _make_slat_noise(coords_np: np.ndarray, *, seed: int | None) -> SparseTensor:
    coords = torch.from_numpy(np.ascontiguousarray(coords_np.astype(np.int32, copy=False))).to(
        device="cuda",
        dtype=torch.int32,
    )
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    sp_coords = torch.cat([batch, coords], dim=1)
    if seed is None:
        feats = torch.randn(coords.shape[0], 8, device=coords.device, dtype=torch.float32)
    else:
        gen = torch.Generator(device=coords.device)
        gen.manual_seed(int(seed))
        feats = torch.randn(coords.shape[0], 8, device=coords.device, dtype=torch.float32, generator=gen)
    return SparseTensor(coords=sp_coords, feats=feats)


def _sample_slat_native_multi_image(
    cond_views: torch.Tensor,
    coords_np: np.ndarray,
    ckpt_path: Path,
    *,
    steps: int,
    seed: int,
    mode: str,
    cfg_strength: float = 3.0,
) -> SparseTensor:
    if cond_views.dim() != 3:
        raise ValueError(f"cond_views must be [V,T,D], got {tuple(cond_views.shape)}")
    if cond_views.shape[0] < 1:
        raise ValueError("cond_views must contain at least one view")
    cond = cond_views.float().cuda()
    if mode == "single":
        cond = cond[:1]
    elif mode not in {"stochastic", "multidiffusion"}:
        raise ValueError(f"unknown native SLat flow mode={mode!r}")
    neg_cond = torch.zeros_like(cond[:1])
    noise = _make_slat_noise(coords_np, seed=seed)
    model = inference._load_slat_flow(str(ckpt_path.resolve()))
    sampler = FlowEulerCfgSampler(sigma_min=1e-5)

    if mode == "single":
        with torch.no_grad():
            return sampler.sample(
                model,
                noise=noise,
                cond=cond,
                neg_cond=neg_cond,
                steps=int(steps),
                cfg_strength=float(cfg_strength),
                verbose=False,
            ).samples

    old_inference = sampler._inference_model
    if mode == "stochastic":
        cond_indices = (np.arange(int(steps)) % int(cond.shape[0])).tolist()

        def _new_inference_model(self, model, x_t, t, cond, **kwargs):
            cond_idx = cond_indices.pop(0)
            return old_inference(model, x_t, t, cond=cond[cond_idx : cond_idx + 1], **kwargs)

    else:

        def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
            pred = None
            for view_idx in range(int(cond.shape[0])):
                view_pred = FlowEulerSampler._inference_model(
                    self,
                    model,
                    x_t,
                    t,
                    cond[view_idx : view_idx + 1],
                    **kwargs,
                )
                pred = view_pred if pred is None else pred + view_pred
            pred = pred / int(cond.shape[0])
            neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
            return (1.0 + float(cfg_strength)) * pred - float(cfg_strength) * neg_pred

    sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))
    try:
        with torch.no_grad():
            return sampler.sample(
                model,
                noise=noise,
                cond=cond,
                neg_cond=neg_cond,
                steps=int(steps),
                cfg_strength=float(cfg_strength),
                verbose=False,
            ).samples
    finally:
        sampler._inference_model = old_inference


def _load_cond_tokens(data_root: Path, object_id: str, angle: int, view_indices: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    token_path = data_root / "reconstruction" / "dinov2_tokens" / object_id / f"angle_{int(angle)}" / "tokens.npz"
    if not token_path.is_file():
        raise FileNotFoundError(token_path)
    with np.load(token_path, allow_pickle=False) as data:
        if "tokens" not in data.files:
            raise KeyError(f"{token_path}: expected key tokens, got {data.files}")
        tokens = np.asarray(data["tokens"], dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[-1] != 1024:
        raise ValueError(f"{token_path}: expected [V,T,1024], got {tokens.shape}")
    if not view_indices:
        raise ValueError("view_indices must be non-empty")
    if min(view_indices) < 0 or max(view_indices) >= tokens.shape[0]:
        raise ValueError(f"{token_path}: cannot select views {view_indices} from shape {tokens.shape}")
    picked = torch.from_numpy(np.ascontiguousarray(tokens[np.asarray(view_indices, dtype=np.int64)])).float()
    picked = torch.nn.functional.layer_norm(picked, picked.shape[-1:])
    return picked, {
        "token_path": str(token_path.resolve()),
        "available_token_shape": list(tokens.shape),
        "view_indices": [int(v) for v in view_indices],
        "picked_token_shape": list(picked.shape),
        "flow_input_shape": [int(np.prod(picked.shape[:2])), int(picked.shape[-1])],
        "normalization": "torch.nn.functional.layer_norm over last dim",
    }


def _make_renderer(resolution: int, bg_color: tuple[int, int, int]) -> GaussianRenderer:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = bg_color
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = 0.1
    return renderer


@torch.no_grad()
def _decode_render(
    slat: SparseTensor,
    *,
    decoder_ckpt: Path,
    slat_is_normalized: bool,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    out_png: Path,
    resolution: int,
    bg_color: tuple[int, int, int],
) -> dict[str, Any]:
    decoded = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(decoder_ckpt.resolve()),
        slat_is_normalized=slat_is_normalized,
    )
    gaussian = decoded.get("gaussian")
    if gaussian is None:
        raise RuntimeError("gaussian decoder returned None")
    renderer = _make_renderer(resolution, bg_color)
    color = renderer.render(gaussian, extrinsic, intrinsic)["color"].detach().float().cpu().clamp(0, 1)
    arr = (color.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_png)
    return {
        "png": str(out_png.resolve()),
        "slat_is_normalized": bool(slat_is_normalized),
        "gaussians": int(gaussian.get_xyz.shape[0]),
        "gaussian_xyz_min": [float(v) for v in gaussian.get_xyz.detach().min(dim=0).values.cpu().tolist()],
        "gaussian_xyz_max": [float(v) for v in gaussian.get_xyz.detach().max(dim=0).values.cpu().tolist()],
        "gaussian_opacity_mean": float(gaussian.get_opacity.detach().mean().cpu().item()),
        "gaussian_feature_dc_mean": [
            float(v) for v in gaussian._features_dc.detach().mean(dim=0).reshape(-1).cpu().tolist()
        ],
    }


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    if path.suffix == ".npy":
        mask = np.load(path)
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
        image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    else:
        image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def _image_stats(path: Path, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    fg_auto = arr.max(axis=-1) > 5.0
    out: dict[str, Any] = {
        "path": str(path.resolve()),
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean_rgb": [float(v) for v in arr.reshape(-1, 3).mean(axis=0)],
        "auto_nonblack_fraction": float(fg_auto.mean()),
    }
    if fg_auto.any():
        out["auto_fg_mean_rgb"] = [float(v) for v in arr[fg_auto].mean(axis=0)]
    if mask is not None and mask.shape == arr.shape[:2] and mask.any():
        out["gt_mask_fg_mean_rgb"] = [float(v) for v in arr[mask].mean(axis=0)]
        out["gt_mask_fraction"] = float(mask.mean())
    return out


def _tile(path: Path, label: str, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 30), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size, 30), fill=(0, 0, 0))
    draw.text((8, 9), label[:58], fill=(255, 255, 255))
    tile.paste(img, ((size - img.width) // 2, 30 + (size - img.height) // 2))
    return tile


def _make_panel(items: list[tuple[str, Path]], out_path: Path, tile_size: int) -> None:
    cols = len(items)
    canvas = Image.new("RGB", (cols * tile_size, tile_size + 30), (255, 255, 255))
    for idx, (label, path) in enumerate(items):
        canvas.paste(_tile(path, label, tile_size), (idx * tile_size, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC RealAppliance SLat latent vs flow Gaussian color.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--object-id", default="039")
    parser.add_argument("--angle", type=int, default=1)
    parser.add_argument("--view", type=int, default=1)
    parser.add_argument("--decoder-ckpt", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--slat-flow-ckpt", type=Path, default=DEFAULT_SLAT_FLOW)
    parser.add_argument("--flow-render", type=Path, default=DEFAULT_FLOW_RENDER)
    parser.add_argument("--run-flow-on-gt-coords", action="store_true")
    parser.add_argument("--run-native-flow-ab", action="store_true")
    parser.add_argument("--flow-view-indices", type=int, nargs="+", default=[1, 3, 8, 11])
    parser.add_argument("--flow-steps", type=int, default=25)
    parser.add_argument("--flow-seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=320)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    for path, label in ((args.data_root, "data root"), (args.decoder_ckpt, "Gaussian decoder")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    case_dir = args.out_dir / f"{args.object_id}_angle_{int(args.angle)}_view_{int(args.view)}"
    case_dir.mkdir(parents=True, exist_ok=True)

    inst = f"{args.object_id}_angle_{int(args.angle)}"
    latent_path = (
        args.data_root / "part_synthesis_slat" / args.object_id[:2] / inst / "overall" / "latent.npz"
    )
    gt_png = args.data_root / "renders" / args.object_id / f"angle_{int(args.angle)}" / "rgb" / f"view_{int(args.view)}.png"
    mask_path = args.data_root / "renders" / args.object_id / f"angle_{int(args.angle)}" / "mask" / f"mask_{int(args.view)}.npy"
    camera_path = args.data_root / "renders" / args.object_id / f"angle_{int(args.angle)}" / "camera_transforms.json"

    coords_np, feats_np = _load_latent(latent_path)
    slat = _make_sparse(coords_np, feats_np)
    extrinsics, intrinsics = load_camera_matrices(camera_path, [int(args.view)])
    extrinsic = extrinsics[0]
    intrinsic = intrinsics[0]

    raw_black = case_dir / "precomputed_slat_raw_blackbg.png"
    raw_white = case_dir / "precomputed_slat_raw_whitebg.png"
    norm_black = case_dir / "precomputed_slat_as_normalized_blackbg.png"
    raw_rec = _decode_render(
        slat,
        decoder_ckpt=args.decoder_ckpt,
        slat_is_normalized=False,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        out_png=raw_black,
        resolution=int(args.resolution),
        bg_color=(0, 0, 0),
    )
    raw_white_rec = _decode_render(
        slat,
        decoder_ckpt=args.decoder_ckpt,
        slat_is_normalized=False,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        out_png=raw_white,
        resolution=int(args.resolution),
        bg_color=(1, 1, 1),
    )
    norm_rec = _decode_render(
        slat,
        decoder_ckpt=args.decoder_ckpt,
        slat_is_normalized=True,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        out_png=norm_black,
        resolution=int(args.resolution),
        bg_color=(0, 0, 0),
    )

    mask = _load_mask(mask_path, (int(args.resolution), int(args.resolution)))
    stats = {
        "gt": _image_stats(gt_png, mask),
        "precomputed_slat_raw_blackbg": _image_stats(raw_black, mask),
        "precomputed_slat_raw_whitebg": _image_stats(raw_white, mask),
        "precomputed_slat_as_normalized_blackbg": _image_stats(norm_black, mask),
    }
    panel_items = [("GT rgb", gt_png)]
    if args.flow_render.is_file():
        stats["flow_existing_4view_blackbg"] = _image_stats(args.flow_render, mask)
        panel_items.append(("flow existing", args.flow_render))
    panel_items.extend(
        [
            ("GT SLat raw black bg", raw_black),
            ("GT SLat raw white bg", raw_white),
            ("GT SLat wrong normalized", norm_black),
        ]
    )
    flow_gt_report = None
    native_flow_report = None
    if args.run_flow_on_gt_coords:
        cond_tokens, cond_meta = _load_cond_tokens(
            args.data_root,
            str(args.object_id),
            int(args.angle),
            [int(v) for v in args.flow_view_indices],
        )
        coords_t = torch.from_numpy(np.ascontiguousarray(coords_np.astype(np.int64, copy=False))).long()
        print(
            f"[qc] running SLat flow on GT coords N={coords_np.shape[0]} "
            f"views={cond_meta['view_indices']} steps={int(args.flow_steps)} seed={int(args.flow_seed)}",
            flush=True,
        )
        flow_slat = inference.run_slat_flow_from_tokens(
            cond_tokens,
            coords_t,
            str(args.slat_flow_ckpt.resolve()),
            num_steps=int(args.flow_steps),
            seed=int(args.flow_seed),
        )
        flow_norm = case_dir / "flow_on_gt_coords_decode_normalized.png"
        flow_raw = case_dir / "flow_on_gt_coords_decode_raw.png"
        flow_norm_rec = _decode_render(
            flow_slat,
            decoder_ckpt=args.decoder_ckpt,
            slat_is_normalized=True,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            out_png=flow_norm,
            resolution=int(args.resolution),
            bg_color=(0, 0, 0),
        )
        flow_raw_rec = _decode_render(
            flow_slat,
            decoder_ckpt=args.decoder_ckpt,
            slat_is_normalized=False,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            out_png=flow_raw,
            resolution=int(args.resolution),
            bg_color=(0, 0, 0),
        )
        stats["flow_on_gt_coords_decode_normalized"] = _image_stats(flow_norm, mask)
        stats["flow_on_gt_coords_decode_raw"] = _image_stats(flow_raw, mask)
        flow_gt_report = {
            "condition": cond_meta,
            "slat_flow_ckpt": str(args.slat_flow_ckpt.resolve()),
            "steps": int(args.flow_steps),
            "seed": int(args.flow_seed),
            "coords_source": "precomputed part_synthesis_slat overall coords",
            "decode_normalized": flow_norm_rec,
            "decode_raw": flow_raw_rec,
        }
        panel_items.extend(
            [
                ("flow GT coords norm", flow_norm),
                ("flow GT coords raw", flow_raw),
            ]
        )

    if args.run_native_flow_ab:
        cond_tokens, cond_meta = _load_cond_tokens(
            args.data_root,
            str(args.object_id),
            int(args.angle),
            [int(v) for v in args.flow_view_indices],
        )
        native_flow_report = {
            "condition": cond_meta,
            "slat_flow_ckpt": str(args.slat_flow_ckpt.resolve()),
            "steps": int(args.flow_steps),
            "seed": int(args.flow_seed),
            "coords_source": "precomputed part_synthesis_slat overall coords",
            "contract": (
                "Native TRELLIS multi-image sampler keeps cond as [V,T,D] and injects one view "
                "per denoise step (stochastic) or averages per-view predictions (multidiffusion)."
            ),
            "variants": {},
        }
        for mode in ("single", "stochastic", "multidiffusion"):
            print(
                f"[qc] native SLat flow mode={mode} GT coords N={coords_np.shape[0]} "
                f"views={cond_meta['view_indices']} steps={int(args.flow_steps)} seed={int(args.flow_seed)}",
                flush=True,
            )
            flow_slat = _sample_slat_native_multi_image(
                cond_tokens,
                coords_np,
                args.slat_flow_ckpt,
                steps=int(args.flow_steps),
                seed=int(args.flow_seed),
                mode=mode,
                cfg_strength=3.0,
            )
            out_png = case_dir / f"native_{mode}_flow_gt_coords_decode_normalized.png"
            rec = _decode_render(
                flow_slat,
                decoder_ckpt=args.decoder_ckpt,
                slat_is_normalized=True,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                out_png=out_png,
                resolution=int(args.resolution),
                bg_color=(0, 0, 0),
            )
            stat_key = f"native_{mode}_flow_gt_coords_decode_normalized"
            stats[stat_key] = _image_stats(out_png, mask)
            native_flow_report["variants"][mode] = rec
            panel_items.append((f"native {mode}", out_png))

    panel = case_dir / "qc_panel.png"
    _make_panel(panel_items, panel, int(args.tile_size))

    report = {
        "object_id": args.object_id,
        "angle": int(args.angle),
        "view": int(args.view),
        "case_dir": str(case_dir.resolve()),
        "panel": str(panel.resolve()),
        "latent_path": str(latent_path.resolve()),
        "latent_coords_shape": list(coords_np.shape),
        "latent_feats_shape": list(feats_np.shape),
        "latent_feats_min": float(feats_np.min()),
        "latent_feats_max": float(feats_np.max()),
        "latent_feats_mean": [float(v) for v in feats_np.mean(axis=0)],
        "latent_feats_std": [float(v) for v in feats_np.std(axis=0)],
        "raw_decode": raw_rec,
        "raw_white_decode": raw_white_rec,
        "wrong_normalized_decode": norm_rec,
        "flow_on_gt_coords": flow_gt_report,
        "native_flow_ab": native_flow_report,
        "stats": stats,
        "interpretation": (
            "precomputed part_synthesis_slat is raw SLat, so raw decode uses "
            "slat_is_normalized=False; wrong_normalized is intentionally incorrect."
        ),
    }
    (case_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
