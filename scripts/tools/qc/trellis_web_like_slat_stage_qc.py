#!/usr/bin/env python3
"""Run a web-stage-like TRELLIS SLat appearance baseline on existing voxels.

The component contract mirrors the SAM3D web stage:
  input_rgb + input_mask are shared by body and every part
  body coords = whole voxel - part coords
  each component is sampled independently from the same condition image

No SAM3D weights are used here.  The appearance model is TRELLIS
slat_flow_img_dit and the outputs are TRELLIS SLat decoders.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
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
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


PART_RE = re.compile(r"^part_(\d+)_voxel\.npz$")


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


def load_coords_np(path: Path) -> np.ndarray:
    with np.load(require_file(path, "voxel npz")) as data:
        if "coords" not in data.files:
            raise ValueError(f"{path}: missing coords key; keys={data.files}")
        coords = data["coords"]
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: coords must be [N,3], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: coords is empty")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords dtype must be integer, got {coords.dtype}")
    lo = int(coords.min())
    hi = int(coords.max())
    if lo < 0 or hi >= 64:
        raise ValueError(f"{path}: coords out of [0,64), min={lo} max={hi}")
    return np.ascontiguousarray(coords.astype(np.int64, copy=False))


def coord_keys(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    return coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]


def discover_part_voxels(parts_dir: Path) -> list[tuple[int, Path]]:
    parts_dir = require_dir(parts_dir, "parts dir")
    found: list[tuple[int, Path]] = []
    for child in sorted(parts_dir.iterdir()):
        match = PART_RE.match(child.name)
        if match:
            found.append((int(match.group(1)), child.resolve()))
    if not found:
        raise FileNotFoundError(f"no part_NN_voxel.npz files found in {parts_dir}")
    return sorted(found, key=lambda x: x[0])


def build_body_coords(run_dir: Path, part_paths: list[tuple[int, Path]]) -> tuple[Path, np.ndarray]:
    whole_path = require_file(run_dir / "voxel.npz", "whole voxel")
    whole = load_coords_np(whole_path)
    part_chunks = [load_coords_np(path) for _, path in part_paths]
    part_keys = np.unique(coord_keys(np.concatenate(part_chunks, axis=0)))
    keep = ~np.isin(coord_keys(whole), part_keys)
    body = np.ascontiguousarray(whole[keep].astype(np.int64, copy=False))
    if body.shape[0] == 0:
        raise ValueError(f"{whole_path}: body became empty after subtracting parts")
    return whole_path, body


def load_rgba(image_path: Path, mask_path: Path) -> Image.Image:
    rgb = Image.open(require_file(image_path, "input RGB")).convert("RGB")
    mask = Image.open(require_file(mask_path, "input mask")).convert("L")
    if rgb.size != mask.size:
        raise ValueError(f"input image/mask size mismatch: rgb={rgb.size} mask={mask.size}")
    rgba = Image.new("RGBA", rgb.size)
    rgba.paste(rgb)
    rgba.putalpha(mask)
    return rgba


def make_renderer(resolution: int, decoder: Any) -> Any:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 1
    rep_config = getattr(decoder, "rep_config", None)
    if isinstance(rep_config, dict) and "2d_filter_kernel_size" in rep_config:
        renderer.pipe.kernel_size = float(rep_config["2d_filter_kernel_size"])
    else:
        renderer.pipe.kernel_size = 0.1
    return renderer


@torch.no_grad()
def render_gaussian(gaussian: Any, renderer: Any, view_count: int) -> list[Image.Image]:
    from trellis.utils.arts.slat_render_utils import get_canonical_cameras

    extrinsics, intrinsics = get_canonical_cameras(num_views=int(view_count))
    extrinsics = extrinsics.cuda()
    intrinsics = intrinsics.cuda()
    images = []
    for view_idx in range(int(view_count)):
        out = renderer.render(gaussian, extrinsics[view_idx], intrinsics[view_idx], colors_overwrite=None)
        if "color" not in out:
            raise RuntimeError(f"GaussianRenderer output for view {view_idx} missing color")
        color = out["color"].detach().float().cpu().clamp(0, 1)
        if color.ndim != 3 or color.shape[0] != 3:
            raise RuntimeError(f"rendered color must be [3,H,W], got {tuple(color.shape)}")
        arr = (color.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def merge_gaussians(gaussians: list[Any]) -> Any:
    if not gaussians:
        raise RuntimeError("cannot merge zero gaussians")
    from trellis.representations.gaussian import Gaussian

    ref = gaussians[0]
    merged = Gaussian(**ref.init_params)
    for attr in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        vals = [getattr(g, attr) for g in gaussians]
        if any(v is None for v in vals):
            raise RuntimeError(f"cannot merge gaussian attr {attr}: contains None")
        setattr(merged, attr, torch.cat(vals, dim=0))
    rest_vals = [getattr(g, "_features_rest") for g in gaussians]
    if all(v is None for v in rest_vals):
        merged._features_rest = None
    elif any(v is None for v in rest_vals):
        raise RuntimeError("cannot merge gaussian attr _features_rest: mixed None/non-None")
    else:
        merged._features_rest = torch.cat(rest_vals, dim=0)
    return merged


def make_panel(path: Path, input_img: Image.Image, component_images: dict[str, Image.Image], assembled: Image.Image) -> None:
    labels = ["input_rgb", "assembled", *component_images.keys()]
    images = [input_img.convert("RGB"), assembled.convert("RGB"), *[img.convert("RGB") for img in component_images.values()]]
    thumb = 288
    label_h = 30
    pad = 8
    cols = 3
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * thumb + (cols + 1) * pad, rows * (thumb + label_h) + (rows + 1) * pad), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(zip(labels, images)):
        tile = Image.new("RGB", (thumb, thumb), (30, 30, 30))
        image = image.copy()
        image.thumbnail((thumb, thumb), Image.LANCZOS)
        tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        x = pad + (idx % cols) * (thumb + pad)
        y = pad + (idx // cols) * (thumb + label_h + pad)
        draw.text((x + 6, y + 6), label, fill=(20, 20, 20))
        canvas.paste(tile, (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run(args: argparse.Namespace) -> None:
    run_dir = require_dir(args.run_dir, "web-like source run dir")
    parts_dir = require_dir(args.parts_dir or run_dir / "parts", "parts dir")
    image_path = require_file(args.image or run_dir / "input_rgb.png", "input RGB")
    mask_path = require_file(args.mask or run_dir / "input_mask.png", "input mask")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, label in (
        (args.slat_flow_ckpt, "TRELLIS SLat flow ckpt"),
        (args.gaussian_decoder_ckpt, "TRELLIS SLat gaussian decoder ckpt"),
        (args.mesh_decoder_ckpt, "TRELLIS SLat mesh decoder ckpt"),
    ):
        require_file(path, label)

    part_paths = discover_part_voxels(parts_dir)
    body_source, body_coords = build_body_coords(run_dir, part_paths)
    components: list[tuple[str, int, np.ndarray, str]] = [
        ("body", int(args.seed), body_coords, str(body_source)),
    ]
    for part_index, part_path in part_paths:
        stem = part_path.name[: -len("_voxel.npz")]
        components.append((stem, int(args.seed) + part_index + 1, load_coords_np(part_path), str(part_path)))

    rgba = load_rgba(image_path, mask_path)
    input_rgb = Image.open(image_path).convert("RGB")
    print(f"[input] image={image_path} mask={mask_path} size={rgba.size}", flush=True)
    print(f"[input] shared RGBA condition for {len(components)} component(s)", flush=True)
    print(f"[renderer] diff_gaussian={diff_gaussian_rasterization.__file__}", flush=True)
    print(f"[renderer] GaussianRenderer={gaussian_render_mod.__file__}", flush=True)

    tokens = inference._images_to_tokens([rgba])
    print(f"[tokens] {tuple(tokens.shape)} from single web input image", flush=True)

    gs_decoder = inference._load_slat_vae_decoder(str(args.gaussian_decoder_ckpt.resolve()))
    renderer = make_renderer(int(args.render_resolution), gs_decoder)
    gaussians = []
    component_images: dict[str, Image.Image] = {}
    rows = []

    for comp_idx, (name, seed, coords_np, source) in enumerate(components):
        coords = torch.from_numpy(coords_np).long()
        print(f"[component] {name} voxels={coords_np.shape[0]} seed={seed} source={source}", flush=True)
        slat = inference.run_slat_flow_from_tokens(
            tokens,
            coords,
            str(args.slat_flow_ckpt.resolve()),
            num_steps=int(args.steps),
            seed=seed,
        )
        decoded = inference.decode_slat_assets(
            slat,
            gaussian_decoder_ckpt=str(args.gaussian_decoder_ckpt.resolve()),
            mesh_decoder_ckpt=str(args.mesh_decoder_ckpt.resolve()),
            slat_is_normalized=True,
        )
        asset_dir = out_dir / "components"
        record = save_decoded_slat_assets(
            decoded,
            asset_dir,
            mesh_name=f"{name}.glb",
            gaussian_name=f"{name}.ply",
        )
        gaussian = decoded.get("gaussian")
        if gaussian is None:
            raise RuntimeError(f"{name}: gaussian decoder returned None")
        gaussians.append(copy.deepcopy(gaussian))
        rendered = render_gaussian(gaussian, renderer, 1)[0]
        rendered.save(out_dir / f"{name}_render.png")
        component_images[name] = rendered
        rows.append(
            {
                "name": name,
                "component_index": comp_idx,
                "seed": int(seed),
                "voxel_count": int(coords_np.shape[0]),
                "coord_min": [int(x) for x in coords_np.min(axis=0).tolist()],
                "coord_max": [int(x) for x in coords_np.max(axis=0).tolist()],
                "source": source,
                "assets": record,
                "render": str((out_dir / f"{name}_render.png").resolve()),
            }
        )

    assembled = merge_gaussians(gaussians)
    assembled_ply = out_dir / "assembled_trellis_web_like.ply"
    assembled.save_ply(str(assembled_ply), transform=None)
    assembled_views = render_gaussian(assembled, renderer, int(args.view_count))
    for view_idx, image in enumerate(assembled_views):
        image.save(out_dir / f"assembled_view{view_idx}.png")
    panel = out_dir / "trellis_web_like_panel.png"
    make_panel(panel, input_rgb, component_images, assembled_views[0])

    report = {
        "source_run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "input_image": str(image_path),
        "input_mask": str(mask_path),
        "ckpts": {
            "slat_flow": str(args.slat_flow_ckpt.resolve()),
            "gaussian_decoder": str(args.gaussian_decoder_ckpt.resolve()),
            "mesh_decoder": str(args.mesh_decoder_ckpt.resolve()),
        },
        "official_renderer": {
            "diff_gaussian": diff_gaussian_rasterization.__file__,
            "gaussian_renderer": gaussian_render_mod.__file__,
            "used_gsplat": False,
        },
        "sampling": {
            "steps": int(args.steps),
            "seed": int(args.seed),
            "condition": "single shared web input RGBA image for body and all parts",
            "tokens_shape": list(tokens.shape),
            "flow_output_slat_is_normalized": True,
            "decode_slat_assets_slat_is_normalized": True,
        },
        "components": rows,
        "outputs": {
            "panel": str(panel.resolve()),
            "assembled_ply": str(assembled_ply.resolve()),
            "assembled_views": [str((out_dir / f"assembled_view{i}.png").resolve()) for i in range(len(assembled_views))],
            "component_dir": str((out_dir / "components").resolve()),
        },
        "notes": [
            "This is a TRELLIS baseline following the SAM3D/web component scheduling contract.",
            "No SAM3D slat_generator or SAM3D decoder weights are used.",
            "The input condition is only one object image, so part-specific colors are not explicitly conditioned.",
        ],
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] panel={panel}", flush=True)
    print(f"[done] report={report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--slat-flow-ckpt", type=Path, required=True)
    parser.add_argument("--gaussian-decoder-ckpt", type=Path, required=True)
    parser.add_argument("--mesh-decoder-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-count", type=int, default=4)
    parser.add_argument("--render-resolution", type=int, default=512)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
