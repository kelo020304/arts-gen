#!/usr/bin/env python3
"""A/B check TRELLIS SLat flow conditioning tokens on fixed PhysX voxels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import inference  # noqa: E402
from scripts.tools.render.render_glb_open3d_preview import render as render_open3d  # noqa: E402
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_RESOLUTION = 518
EXPECTED_DIM = 1024


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


def load_manifest_row(manifest: Path, object_id: str, angle_idx: int) -> dict[str, Any]:
    manifest = require_file(manifest, "manifest")
    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("object_id")) == str(object_id) and int(row.get("angle_idx", -1)) == int(angle_idx):
                row["_manifest_line"] = line_no
                return row
    raise ValueError(f"{manifest}: no row for object_id={object_id} angle_idx={angle_idx}")


def resolve_manifest_paths(data_root: Path, row: dict[str, Any]) -> tuple[list[Path], list[Path], list[int]]:
    paths = row.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("manifest row missing paths dict")
    rgb_rel = paths.get("rgb")
    mask_rel = paths.get("masks")
    if not isinstance(rgb_rel, list) or not isinstance(mask_rel, list):
        raise ValueError(f"manifest paths.rgb/masks must be lists, got rgb={type(rgb_rel)} masks={type(mask_rel)}")
    if len(rgb_rel) != len(mask_rel):
        raise ValueError(f"manifest rgb/mask length mismatch: {len(rgb_rel)} vs {len(mask_rel)}")
    rgb_paths = [require_file(data_root / rel, f"manifest rgb[{idx}]") for idx, rel in enumerate(rgb_rel)]
    mask_paths = [require_file(data_root / rel, f"manifest mask[{idx}]") for idx, rel in enumerate(mask_rel)]
    view_indices = []
    for rel in rgb_rel:
        stem = Path(rel).stem
        if not stem.startswith("view_"):
            raise ValueError(f"cannot parse view index from rgb path: {rel}")
        view_indices.append(int(stem[len("view_") :]))
    return rgb_paths, mask_paths, view_indices


def load_rgba_views(rgb_paths: list[Path], mask_paths: list[Path]) -> list[Image.Image]:
    out = []
    for rgb_path, mask_path in zip(rgb_paths, mask_paths):
        rgb = Image.open(rgb_path).convert("RGB")
        if mask_path.suffix == ".npy":
            mask_arr = np.load(mask_path)
            if mask_arr.ndim != 2:
                raise ValueError(f"{mask_path}: npy mask must be [H,W], got {mask_arr.shape}")
            if mask_arr.shape != (rgb.height, rgb.width):
                raise ValueError(f"{mask_path}: npy mask shape {mask_arr.shape} does not match rgb {rgb.size}")
            mask = Image.fromarray((mask_arr > 0).astype(np.uint8) * 255, mode="L")
        else:
            mask_image = Image.open(mask_path)
            if mask_image.mode == "RGBA":
                mask_arr = np.asarray(mask_image.convert("RGBA"), dtype=np.uint8)
                mask = Image.fromarray(mask_arr[:, :, :3].max(axis=2), mode="L")
            else:
                mask = mask_image.convert("L")
        if rgb.size != mask.size:
            raise ValueError(f"rgb/mask size mismatch: {rgb_path} {rgb.size} vs {mask_path} {mask.size}")
        rgba = Image.new("RGBA", rgb.size)
        rgba.paste(rgb)
        rgba.putalpha(mask)
        out.append(rgba)
    return out


def preprocess_official(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        raise ValueError(f"official preprocessing expects RGBA image with alpha, got {image.mode}")
    arr = np.asarray(image.convert("RGBA"))
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > int(0.8 * 255))
    if xs.size == 0 or ys.size == 0:
        raise ValueError("official preprocessing found empty alpha foreground")
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    size = int(max(right - left, bottom - top) * 1.2)
    if size <= 0:
        raise ValueError(f"invalid official crop size from bbox={(left, top, right, bottom)}")
    bbox = (
        center_x - size // 2,
        center_y - size // 2,
        center_x + size // 2,
        center_y + size // 2,
    )
    cropped = image.crop(bbox).resize((DINO_RESOLUTION, DINO_RESOLUTION), Image.Resampling.LANCZOS)
    out = np.asarray(cropped.convert("RGBA")).astype(np.float32) / 255.0
    premultiplied_black = out[:, :, :3] * out[:, :, 3:4]
    return Image.fromarray((premultiplied_black * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")


def preprocess_current_white(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert("RGB")


def images_to_tensor(images: list[Image.Image]) -> torch.Tensor:
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(
                (DINO_RESOLUTION, DINO_RESOLUTION),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
    return torch.stack([transform(image.convert("RGB")) for image in images], dim=0).cuda()


@torch.no_grad()
def encode_tokens(images: list[Image.Image], mode: str) -> tuple[torch.Tensor, list[Image.Image]]:
    if mode == "prenorm_official":
        processed = [preprocess_official(image) for image in images]
        batch = images_to_tensor(processed)
        feats = inference._load_dinov2()(batch, is_training=True)
        if "x_prenorm" not in feats:
            raise KeyError(f"DINOv2 output missing x_prenorm; keys={sorted(feats.keys())}")
        tokens = F.layer_norm(feats["x_prenorm"], feats["x_prenorm"].shape[-1:]).float()
        expected = (len(images), 1374, EXPECTED_DIM)
    elif mode == "xnorm_current":
        processed = [preprocess_current_white(image) for image in images]
        batch = images_to_tensor(processed)
        feats = inference._load_dinov2().forward_features(batch)
        for key in ("x_norm_clstoken", "x_norm_patchtokens"):
            if key not in feats:
                raise KeyError(f"DINOv2 output missing {key}; keys={sorted(feats.keys())}")
        tokens = torch.cat([feats["x_norm_clstoken"].unsqueeze(1), feats["x_norm_patchtokens"]], dim=1).float()
        expected = (len(images), 1370, EXPECTED_DIM)
    else:
        raise ValueError(f"unknown token mode: {mode}")
    if tuple(tokens.shape) != expected:
        raise ValueError(f"{mode}: token shape {tuple(tokens.shape)} != expected {expected}")
    if not torch.isfinite(tokens).all():
        raise RuntimeError(f"{mode}: DINO tokens contain NaN/Inf")
    return tokens, processed


def load_surface(data_root: Path, object_id: str, angle_idx: int) -> np.ndarray:
    path = require_file(
        data_root / "reconstruction" / "voxel_expanded" / object_id / f"angle_{angle_idx}" / "64" / "surface.npy",
        "surface voxel",
    )
    coords = np.load(path)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: expected coords [N,3], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: empty coords")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords dtype must be integer, got {coords.dtype}")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise ValueError(f"{path}: coords out of [0,64): min={coords.min()} max={coords.max()}")
    return np.ascontiguousarray(coords.astype(np.int64, copy=False))


def save_processed_views(out_dir: Path, label: str, images: list[Image.Image]) -> list[str]:
    view_dir = out_dir / label / "processed_views"
    view_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, image in enumerate(images):
        path = view_dir / f"view_{idx:02d}.png"
        image.save(path)
        written.append(str(path.resolve()))
    return written


def render_case(
    *,
    label: str,
    tokens: torch.Tensor,
    coords_np: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_dir = args.out_dir.resolve() / label
    case_dir.mkdir(parents=True, exist_ok=True)
    coords = torch.from_numpy(coords_np).long()
    slat = inference.run_slat_flow_from_tokens(
        tokens,
        coords,
        str(args.slat_flow_ckpt.resolve()),
        num_steps=int(args.steps),
        seed=int(args.seed),
    )
    decoded = inference.decode_slat_assets(
        slat,
        mesh_decoder_ckpt=str(args.mesh_decoder_ckpt.resolve()),
        slat_is_normalized=True,
    )
    mesh = decoded.get("mesh")
    if mesh is None:
        raise RuntimeError(f"{label}: mesh decoder returned None")
    if not getattr(mesh, "success", True):
        raise RuntimeError(f"{label}: mesh decoder returned success=False")
    assets = save_decoded_slat_assets(decoded, case_dir, mesh_name=f"{label}.glb")
    glb = case_dir / assets["mesh"]
    render_dir = case_dir / "open3d"
    render_paths = render_open3d(
        glb,
        out_dir=render_dir,
        views={"gt_like_iso": (315.0, 24.0), "front": (270.0, 8.0), "side": (0.0, 8.0)},
        resolution=int(args.render_resolution),
        use_vertex_colors=True,
    )
    feats = slat.feats.detach().float().cpu()
    return {
        "label": label,
        "tokens_shape": list(tokens.shape),
        "glb": str(glb.resolve()),
        "open3d_renders": [str(path.resolve()) for path in render_paths],
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "slat_rows": int(feats.shape[0]),
        "slat_feat_min": float(feats.min().item()),
        "slat_feat_max": float(feats.max().item()),
    }


def make_panel(out_path: Path, cases: list[dict[str, Any]]) -> None:
    cells = []
    for case in cases:
        image_path = next(path for path in case["open3d_renders"] if path.endswith("_gt_like_iso.png"))
        cells.append((case["label"], Image.open(image_path).convert("RGB")))
    width, height = cells[0][1].size
    label_h = 32
    panel = Image.new("RGB", (width * len(cells), height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    for idx, (label, image) in enumerate(cells):
        x = idx * width
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
        panel.paste(image, (x, label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def run(args: argparse.Namespace) -> None:
    data_root = require_dir(args.data_root, "data root")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path, label in (
        (args.slat_flow_ckpt, "SLat flow ckpt"),
        (args.mesh_decoder_ckpt, "SLat mesh decoder ckpt"),
    ):
        require_file(path, label)
        if path.suffix == ".safetensors":
            require_file(path.with_suffix(".json"), f"{label} json")

    row = load_manifest_row(args.manifest, args.object_id, args.angle_idx)
    rgb_paths, mask_paths, view_indices = resolve_manifest_paths(data_root, row)
    rgba_views = load_rgba_views(rgb_paths, mask_paths)
    coords = load_surface(data_root, args.object_id, args.angle_idx)
    print(f"[input] object={args.object_id} angle={args.angle_idx} views={view_indices}", flush=True)
    print(f"[input] coords={coords.shape[0]} seed={args.seed}", flush=True)

    tokens_a, processed_a = encode_tokens(rgba_views, "prenorm_official")
    tokens_b, processed_b = encode_tokens(rgba_views, "xnorm_current")
    processed_a_paths = save_processed_views(args.out_dir, "prenorm_official", processed_a)
    processed_b_paths = save_processed_views(args.out_dir, "xnorm_current", processed_b)
    print(f"[tokens] prenorm_official={tuple(tokens_a.shape)} xnorm_current={tuple(tokens_b.shape)}", flush=True)

    cases = [
        render_case(label="prenorm_official_1374_black_crop", tokens=tokens_a, coords_np=coords, args=args),
        render_case(label="xnorm_current_1370_white_resize", tokens=tokens_b, coords_np=coords, args=args),
    ]
    panel_path = args.out_dir / "ab_open3d_mesh_panel.png"
    make_panel(panel_path, cases)
    report = {
        "object_id": args.object_id,
        "angle_idx": int(args.angle_idx),
        "manifest": str(args.manifest.resolve()),
        "manifest_line": int(row["_manifest_line"]),
        "view_indices": view_indices,
        "rgb_paths": [str(path.resolve()) for path in rgb_paths],
        "mask_paths": [str(path.resolve()) for path in mask_paths],
        "surface_path": str(
            (data_root / "reconstruction" / "voxel_expanded" / args.object_id / f"angle_{args.angle_idx}" / "64" / "surface.npy").resolve()
        ),
        "surface_voxels": int(coords.shape[0]),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "slat_flow_ckpt": str(args.slat_flow_ckpt.resolve()),
        "mesh_decoder_ckpt": str(args.mesh_decoder_ckpt.resolve()),
        "processed_views": {
            "prenorm_official": processed_a_paths,
            "xnorm_current": processed_b_paths,
        },
        "cases": cases,
        "panel": str(panel_path.resolve()),
        "notes": [
            "A uses DINOv2 forward(..., is_training=True)['x_prenorm'] + non-affine F.layer_norm and keeps CLS+4 register+1369 patch tokens.",
            "B mirrors current TRELLIS-arts inference: x_norm_clstoken + x_norm_patchtokens on white-background resize.",
            "Both cases use identical surface coords, flow ckpt, seed, steps, and mesh decoder.",
        ],
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] panel={panel_path}", flush=True)
    print(f"[done] report={report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object-id", default="100058")
    parser.add_argument("--angle-idx", type=int, default=0)
    parser.add_argument("--slat-flow-ckpt", type=Path, required=True)
    parser.add_argument("--mesh-decoder-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-resolution", type=int, default=768)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
