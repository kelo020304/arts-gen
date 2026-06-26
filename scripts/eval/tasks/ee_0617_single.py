#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SS_FLOW_FUSION_MODE", "concat")

import inference  # noqa: E402
from part_ss_eval_platform.eval_0617_1 import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_SLAT_FLOW_CKPT,
    DEFAULT_SLAT_MESH_DECODER_CKPT,
    DEFAULT_SS_DECODER_CKPT,
    _command_for_sample,
    _find_dataset_sample,
    _load_datasets,
    _run_dir_for_sample,
    _sample_data_config_path,
)
from part_ss_eval_platform.eval_real_0615 import _execute, _load_coords, render_preview_voxel  # noqa: E402


def _restore_trellis_renderer_package() -> None:
    for name in list(sys.modules):
        if name == "trellis.renderers" or name.startswith("trellis.renderers."):
            sys.modules.pop(name, None)
    import trellis.renderers  # noqa: F401


_restore_trellis_renderer_package()
from trellis.modules.sparse import SparseTensor  # noqa: E402
from trellis.renderers.gaussian_render import GaussianRenderer  # noqa: E402
from trellis.renderers.mesh_renderer import MeshRenderer  # noqa: E402
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


DEFAULT_OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-128ee")
DEFAULT_SPLIT_JSON = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_verse_realappliance_v3.json"
)
DEFAULT_OBJECT_ID = "05a035c3347645b8a7ceb6d65f825ac3"
DEFAULT_DATASET_ID = "phyx-verse"
DEFAULT_ANGLE = 0
DEFAULT_PART_SEG_CKPT = Path(
    "/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0616-1/ckpts/step_50000.pt"
)
DEFAULT_SS_FLOW_CKPT = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/tre-ss-concat-0616-1/ckpts/denoiser_ema0.999_step0012500.pt"
)
DEFAULT_GAUSSIAN_DECODER = (
    REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
)

GS_PRESET = {
    "name": "scaleq997_abs020_scale0.75_opx1.5",
    "max_scale_quantile": 0.997,
    "max_scale_abs": 0.020,
    "scale_mult": 0.75,
    "opacity_mult": 1.5,
    "kernel_size": 0.05,
}


def _safe_name(value: str, max_len: int = 80) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value)).strip("_")
    return (out or "part")[:max_len]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        child = None
        for child in elem:
            _indent_xml(child, level + 1)
        if child is not None and (not child.tail or not child.tail.strip()):
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def _xml_float_list(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def _write_static_mujoco_xml(
    *,
    out_xml: Path,
    model_name: str,
    part_meshes: list[dict[str, Any]],
) -> Path:
    root = ET.Element("mujoco", {"model": _safe_name(model_name, 120)})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        {
            "type": "mesh",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
            "rgba": "0.72 0.76 0.80 1",
        },
    )
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    object_body = ET.SubElement(worldbody, "body", {"name": "object", "pos": "0 0 0"})

    colors = [
        (0.80, 0.16, 0.18, 1.0),
        (0.13, 0.47, 0.70, 1.0),
        (0.17, 0.63, 0.17, 1.0),
        (0.58, 0.40, 0.74, 1.0),
        (1.00, 0.50, 0.05, 1.0),
        (0.55, 0.34, 0.29, 1.0),
        (0.89, 0.47, 0.76, 1.0),
        (0.50, 0.50, 0.50, 1.0),
    ]
    for idx, item in enumerate(part_meshes):
        label = _safe_name(str(item["label"]), 80)
        mesh_name = f"{label}_mesh"
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(item["mesh_file"])})
        part_body = ET.SubElement(object_body, "body", {"name": label, "pos": "0 0 0"})
        ET.SubElement(
            part_body,
            "geom",
            {
                "name": f"{label}_visual",
                "mesh": mesh_name,
                "rgba": _xml_float_list(colors[idx % len(colors)]),
            },
        )

    _indent_xml(root)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return out_xml


def _find_sample(ds: Any, object_id: str, angle: int, dataset_id: str) -> SimpleNamespace:
    for row in ds.samples:
        if str(row["obj_id"]) == object_id and int(row["angle_idx"]) == int(angle):
            return SimpleNamespace(
                split=str(row.get("split", "held")),
                dataset_id=dataset_id,
                obj_id=object_id,
                angle_idx=int(angle),
                data_root=str(row.get("_eval_data_root") or ds.data_root),
                manifest_path=str(row.get("_eval_manifest_path") or ds.manifest_path),
            )
    raise KeyError(f"{dataset_id}::{object_id} angle={angle} not found")


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_camera_matrices(camera_path: Path, view_indices: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    import utils3d

    payload = json.loads(_require_file(camera_path, "camera_transforms").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != 12:
        raise ValueError(f"{camera_path}: frames must have length 12")
    if view_indices is None:
        selected = list(range(len(frames)))
    else:
        selected = list(view_indices)
        if not selected:
            raise ValueError("view_indices must be non-empty")
        bad = [idx for idx in selected if idx < 0 or idx >= len(frames)]
        if bad:
            raise ValueError(f"{camera_path}: view_indices out of range [0,{len(frames)}): {bad}")
    extrinsics = []
    intrinsics = []
    for idx in selected:
        frame = frames[idx]
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device="cuda")
        if tuple(c2w.shape) != (4, 4):
            raise ValueError(f"{camera_path}: frame {idx} transform_matrix must be 4x4")
        c2w = c2w.clone()
        c2w[:3, 1:3] *= -1
        extrinsics.append(torch.inverse(c2w))
        fov = torch.tensor(float(frame["camera_angle_x"]), dtype=torch.float32, device="cuda")
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    return torch.stack(extrinsics), torch.stack(intrinsics)


def _ensure_ss_and_part(args: argparse.Namespace, ds: Any, sample: SimpleNamespace, ds_sample: dict[str, Any]) -> Path:
    run_dir = _run_dir_for_sample(args.out_dir, sample)
    progress_path = args.out_dir / "progress_single.jsonl"
    expected_parts = len(ds_sample["parts"])
    ss_done = (run_dir / "ss_latent.npy").is_file() and (run_dir / "voxel.npz").is_file()
    part_done = len(list((run_dir / "parts").glob("part_*_voxel.npz"))) >= expected_parts
    local_args = argparse.Namespace(**vars(args))
    local_args.data_config = str(_sample_data_config_path(args.out_dir, sample, ds))

    for stage, done in (("ss", ss_done), ("part", part_done)):
        if done and not args.force_stage:
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"stage": stage, "status": "skipped", "run_dir": str(run_dir)}) + "\n")
            continue
        spec = _command_for_sample(args.out_dir, sample, local_args, stage, ds)
        rec = _execute(
            spec,
            gpu=str(args.gpu),
            progress_path=progress_path,
            label=f"0617-128ee/{stage}/{sample.dataset_id}/{sample.obj_id}/{int(sample.angle_idx)}",
        )
        rec["status"] = "done"
    if not (run_dir / "voxel.npz").is_file():
        raise FileNotFoundError(f"missing whole voxel after ss stage: {run_dir / 'voxel.npz'}")
    if len(list((run_dir / "parts").glob("part_*_voxel.npz"))) < expected_parts:
        raise FileNotFoundError(f"missing part voxels after part stage: {run_dir / 'parts'}")
    return run_dir


def _rgba_view_image(data_root: Path, object_id: str, angle: int, view_idx: int) -> Image.Image:
    rgb_path = data_root / "renders" / object_id / f"angle_{int(angle)}" / "rgb" / f"view_{int(view_idx)}.png"
    if not rgb_path.is_file():
        raise FileNotFoundError(f"SLat input RGB view not found: {rgb_path}")
    image = Image.open(rgb_path)
    if image.mode == "RGBA" or "A" in image.getbands():
        return image.convert("RGBA")

    mask_candidates = [
        data_root / "renders" / object_id / f"angle_{int(angle)}" / "mask" / f"mask_{int(view_idx)}.npy",
        data_root / "renders" / object_id / f"angle_{int(angle)}" / "mask" / f"mask_{int(view_idx)}.png",
    ]
    mask_path = next((path for path in mask_candidates if path.is_file()), None)
    if mask_path is None:
        raise FileNotFoundError(f"SLat input view has no alpha and mask is missing for view {view_idx}")
    if mask_path.suffix == ".npy":
        mask = np.asarray(np.load(mask_path))
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
        alpha = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    else:
        alpha = Image.open(mask_path).convert("L")
    if alpha.size != image.size:
        alpha = alpha.resize(image.size, Image.Resampling.NEAREST)
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _load_slat_cond_tokens_for_views(
    ds: Any,
    sample: dict[str, Any],
    view_indices: list[int],
    *,
    token_source: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    data_root = Path(ds.data_root)
    object_id = str(sample["obj_id"])
    angle = int(sample["angle_idx"])
    picked_views = [int(v) for v in view_indices]
    if not picked_views:
        raise ValueError("slat view list must be non-empty")

    if token_source == "live":
        images = [_rgba_view_image(data_root, object_id, angle, view_idx) for view_idx in picked_views]
        picked = inference._images_to_tokens(images).detach().float().cpu()
        return picked, {
            "token_source": "live_official_trellis_rgba",
            "preprocess": "TRELLIS RGBA alpha crop + black premultiply + 518 resize + DINO x_prenorm layer_norm",
            "view_indices": picked_views,
            "picked_token_shape": list(picked.shape),
            "flow_input_shape": [int(np.prod(picked.shape[:2])), int(picked.shape[-1])],
            "image_paths": [
                str((data_root / "renders" / object_id / f"angle_{angle}" / "rgb" / f"view_{view_idx}.png").resolve())
                for view_idx in picked_views
            ],
        }
    if token_source != "cache":
        raise ValueError(f"unsupported slat token source: {token_source!r}")

    token_candidates = [
        data_root / ds.recon_subdir / "dinov2_tokens" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
        data_root / ds.recon_subdir / "dinov2_tokens_prenorm" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
        data_root / ds.recon_subdir / "dinov2_tokens_official_prenorm1374" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
    ]
    token_path = next((path for path in token_candidates if path.is_file()), token_candidates[0])
    if not token_path.is_file():
        raise FileNotFoundError(f"TRELLIS SLat DINO tokens not found: {token_path}")
    with np.load(token_path, allow_pickle=False) as data:
        tokens = np.asarray(data["tokens"], dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[-1] != 1024:
        raise ValueError(f"{token_path} expected [V,T,1024], got {tokens.shape}")
    if max(picked_views) >= tokens.shape[0] or min(picked_views) < 0:
        raise ValueError(f"{token_path} has {tokens.shape[0]} views, cannot select {picked_views}")
    picked = torch.from_numpy(np.ascontiguousarray(tokens[picked_views])).float()
    picked = torch.nn.functional.layer_norm(picked, picked.shape[-1:])
    return picked, {
        "token_source": "cache",
        "token_path": str(token_path.resolve()),
        "available_token_shape": list(tokens.shape),
        "view_indices": picked_views,
        "picked_token_shape": list(picked.shape),
        "flow_input_shape": [int(np.prod(picked.shape[:2])), int(picked.shape[-1])],
    }


def _sparse_subset_from_coords(slat: SparseTensor, coords: np.ndarray, label: str) -> tuple[SparseTensor, int]:
    coords_np = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords_np.size == 0:
        raise ValueError(f"{label}: empty part coords")
    spatial = slat.coords[:, 1:].detach().long().cpu().numpy()
    resolution = int(max(int(spatial.max(initial=0)) + 1, int(coords_np.max(initial=0)) + 1, 64))
    slat_keys = spatial[:, 0] * resolution * resolution + spatial[:, 1] * resolution + spatial[:, 2]
    part_keys = coords_np[:, 0] * resolution * resolution + coords_np[:, 1] * resolution + coords_np[:, 2]
    part_set = set(int(x) for x in part_keys.tolist())
    keep_np = np.fromiter((int(k) in part_set for k in slat_keys.tolist()), dtype=bool, count=len(slat_keys))
    matched = int(keep_np.sum())
    if matched == 0:
        raise ValueError(f"{label}: no part coords matched whole SLat coords ({len(coords_np)} requested)")
    keep = torch.from_numpy(keep_np).to(device=slat.feats.device)
    return SparseTensor(feats=slat.feats[keep].contiguous(), coords=slat.coords[keep].contiguous()), matched


def _new_like_gaussian(gaussian: Any) -> Any:
    out = type(gaussian)(**gaussian.init_params)
    out.active_sh_degree = gaussian.active_sh_degree
    return out


def _subset_gaussian(gaussian: Any, keep: torch.Tensor) -> Any:
    out = _new_like_gaussian(gaussian)
    keep = keep.to(device=gaussian.get_xyz.device, dtype=torch.bool)
    for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        setattr(out, name, getattr(gaussian, name).detach()[keep].clone())
    out._features_rest = None if gaussian._features_rest is None else gaussian._features_rest.detach()[keep].clone()
    return out


def _adjust_gaussian(gaussian: Any) -> Any:
    out = _new_like_gaussian(gaussian)
    for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        setattr(out, name, getattr(gaussian, name).detach().clone())
    out._features_rest = None if gaussian._features_rest is None else gaussian._features_rest.detach().clone()
    scale_mult = float(GS_PRESET["scale_mult"])
    opacity_mult = float(GS_PRESET["opacity_mult"])
    if scale_mult != 1.0:
        scaling = torch.clamp(out.get_scaling * scale_mult, min=out.mininum_kernel_size + 1e-7)
        out.from_scaling(scaling)
    if opacity_mult != 1.0:
        opacity = torch.clamp(out.get_opacity * opacity_mult, 1e-5, 0.995)
        out.from_opacity(opacity)
    return out


def _apply_gs_preset(gaussian: Any) -> tuple[Any, dict[str, Any]]:
    scale_max = gaussian.get_scaling.detach().max(dim=1).values
    quantile_limit = torch.quantile(scale_max, float(GS_PRESET["max_scale_quantile"]))
    abs_limit = scale_max.new_tensor(float(GS_PRESET["max_scale_abs"]))
    limit = torch.minimum(quantile_limit, abs_limit)
    keep = scale_max <= limit
    adjusted = _adjust_gaussian(_subset_gaussian(gaussian, keep))
    return adjusted, {
        **GS_PRESET,
        "scale_quantile_limit": float(quantile_limit.detach().cpu().item()),
        "scale_abs_limit": float(abs_limit.detach().cpu().item()),
        "scale_limit_used": float(limit.detach().cpu().item()),
        "gaussians_before": int(gaussian.get_xyz.shape[0]),
        "gaussians_after": int(adjusted.get_xyz.shape[0]),
        "removed": int((~keep).sum().detach().cpu().item()),
    }


def _make_gaussian_renderer(resolution: int) -> GaussianRenderer:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (1, 1, 1)
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = float(GS_PRESET["kernel_size"])
    renderer.pipe.scale_modifier = 1.0
    return renderer


def _make_mesh_renderer(resolution: int) -> MeshRenderer:
    return MeshRenderer({"resolution": int(resolution), "near": 0.1, "far": 10.0, "ssaa": 1})


@torch.no_grad()
def _render_gaussian(gaussian: Any, extrinsic: torch.Tensor, intrinsic: torch.Tensor, resolution: int) -> Image.Image:
    color = _make_gaussian_renderer(resolution).render(gaussian, extrinsic, intrinsic)["color"]
    arr = (color.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
def _render_mesh(mesh: Any, extrinsic: torch.Tensor, intrinsic: torch.Tensor, resolution: int) -> Image.Image:
    renderer = _make_mesh_renderer(resolution)
    if getattr(mesh, "vertex_attrs", None) is None:
        ret = renderer.render(mesh, extrinsic, intrinsic, return_types=["normal", "mask"])
        color = ret["normal"].detach().float().cpu().clamp(0, 1)
    else:
        ret = renderer.render(mesh, extrinsic, intrinsic, return_types=["color", "mask"])
        color = ret["color"].detach().float().cpu().clamp(0, 1)
    mask = ret["mask"].detach().float().cpu().clamp(0, 1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    color = color * mask + torch.full_like(color, 0.94) * (1.0 - mask)
    arr = (color.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _tile(image: Image.Image, label: str, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 32), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size, 32), fill=(0, 0, 0))
    draw.text((7, 9), label[:64], fill=(255, 255, 255))
    tile.paste(image, ((size - image.width) // 2, 32 + (size - image.height) // 2))
    return tile


def _error_image(message: str, resolution: int) -> Image.Image:
    size = max(128, int(resolution))
    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(200, 60, 60), width=3)
    text = str(message)[:160]
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 28 and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    y = max(12, size // 2 - 10 * len(lines))
    for line in lines[:8]:
        draw.text((12, y), line, fill=(120, 0, 0))
        y += 22
    return image


def _panel(tiles: list[tuple[str, Image.Image]], out_png: Path, *, tile_size: int, cols: int) -> None:
    cols = max(1, min(int(cols), len(tiles)))
    rows = int(np.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * tile_size, rows * (tile_size + 32)), (255, 255, 255))
    for idx, (label, image) in enumerate(tiles):
        canvas.paste(_tile(image, label, tile_size), ((idx % cols) * tile_size, (idx // cols) * (tile_size + 32)))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def _labeled_fit(image: Image.Image, label: str, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    body_h = max(1, int(height) - 30)
    image.thumbnail((int(width), body_h), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, int(width), 30), fill=(0, 0, 0))
    draw.text((8, 9), label[:96], fill=(255, 255, 255))
    tile.paste(image, ((int(width) - image.width) // 2, 30 + (body_h - image.height) // 2))
    return tile


def _input_views_panel(ds: Any, ds_sample: dict[str, Any]) -> Image.Image:
    view_indices = [int(v) for v in ds_sample.get("view_indices", [])]
    image_paths = list(ds_sample.get("image_paths", []))
    tiles: list[Image.Image] = []
    for idx, rel_path in enumerate(image_paths[:4]):
        path = Path(ds.data_root) / str(rel_path)
        if not path.is_file():
            continue
        view_id = view_indices[idx] if idx < len(view_indices) else idx
        tiles.append(_labeled_fit(Image.open(path), f"input view {view_id}", 360, 220))
    if not tiles:
        return _labeled_fit(Image.new("RGB", (720, 440), (245, 245, 245)), "input views missing", 720, 440)
    while len(tiles) < 4:
        tiles.append(_labeled_fit(Image.new("RGB", (360, 220), (245, 245, 245)), "missing", 360, 220))
    canvas = Image.new("RGB", (720, 440), (255, 255, 255))
    for idx, tile in enumerate(tiles[:4]):
        canvas.paste(tile, ((idx % 2) * 360, (idx // 2) * 220))
    return canvas


def _write_diagnostic_panel(
    *,
    ds: Any,
    ds_sample: dict[str, Any],
    whole_coords: np.ndarray,
    part_items: list[tuple[str, np.ndarray]],
    gaussian_png: Path,
    mesh_png: Path,
    out_png: Path,
    object_id: str,
    angle: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="0617_128ee_diag_") as tmp:
        tmp_dir = Path(tmp)
        ss_voxel_png = tmp_dir / "ss_decode_voxel.png"
        partseg_voxel_png = tmp_dir / "partseg_voxel.png"
        render_preview_voxel(whole_coords, [], ss_voxel_png, object_id, int(angle))
        render_preview_voxel(whole_coords, part_items, partseg_voxel_png, object_id, int(angle))
        input_panel = _input_views_panel(ds, ds_sample)
        ss_img = Image.open(ss_voxel_png).copy()
        partseg_img = Image.open(partseg_voxel_png).copy()
    gaussian_img = Image.open(gaussian_png).copy()
    mesh_img = Image.open(mesh_png).copy()

    canvas = Image.new("RGB", (1600, 940), (255, 255, 255))
    top = [
        _labeled_fit(input_panel, "input 4 views", 760, 440),
        _labeled_fit(ss_img, "SS decode voxel", 410, 440),
        _labeled_fit(partseg_img, "PartSeg voxel", 410, 440),
    ]
    x = 0
    for tile in top:
        canvas.paste(tile, (x, 0))
        x += tile.width + 10
    canvas.paste(_labeled_fit(gaussian_img, "Gaussian overall + parts", 790, 500), (0, 440))
    canvas.paste(_labeled_fit(mesh_img, "Mesh overall + parts", 790, 500), (810, 440))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def _prefix(dataset_id: str, object_id: str, angle: int) -> str:
    return f"{dataset_id}__{_safe_name(object_id)}__angle_{int(angle):02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="0617-128ee single-object EE smoke from input tokens, SS concat, partseg, SLat.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--object-id", default=DEFAULT_OBJECT_ID)
    parser.add_argument("--angle", type=int, default=DEFAULT_ANGLE)
    parser.add_argument("--part-seg-ckpt", type=Path, default=DEFAULT_PART_SEG_CKPT)
    parser.add_argument("--ss-flow-ckpt", type=Path, default=DEFAULT_SS_FLOW_CKPT)
    parser.add_argument("--ss-decoder-ckpt", type=Path, default=DEFAULT_SS_DECODER_CKPT)
    parser.add_argument("--slat-flow-ckpt", type=Path, default=DEFAULT_SLAT_FLOW_CKPT)
    parser.add_argument("--slat-mesh-decoder-ckpt", type=Path, default=DEFAULT_SLAT_MESH_DECODER_CKPT)
    parser.add_argument("--slat-gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--slat-steps", type=int, default=25)
    parser.add_argument("--slat-seed", type=int, default=42)
    parser.add_argument(
        "--slat-token-source",
        choices=("live", "cache"),
        default="live",
        help="SLat flow condition source. live uses accepted TRELLIS RGBA preprocessing from input renders; cache is diagnostic.",
    )
    parser.add_argument(
        "--slat-view-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override SLat appearance condition views. Default uses the sample's manifest view_indices.",
    )
    parser.add_argument("--render-view", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=240)
    parser.add_argument("--panel-cols", type=int, default=4)
    parser.add_argument(
        "--export-mujoco",
        action="store_true",
        help="Export per-part OBJ meshes and a static no-joint MJCF XML for MuJoCo visualization.",
    )
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["SS_FLOW_FUSION_MODE"] = "concat"
    args.out_dir = Path(args.out_dir).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for attr, label in (
        ("data_config", "data config"),
        ("split_json", "split json"),
        ("part_seg_ckpt", "part_promptable_seg_full_S_0616-1 step_50000 ckpt"),
        ("ss_flow_ckpt", "tre-ss-concat-0616-1 SS-flow ckpt"),
        ("ss_decoder_ckpt", "SS decoder ckpt"),
        ("slat_flow_ckpt", "SLat flow ckpt"),
        ("slat_mesh_decoder_ckpt", "SLat mesh decoder ckpt"),
        ("slat_gaussian_decoder_ckpt", "SLat gaussian decoder ckpt"),
    ):
        setattr(args, attr, _require_file(Path(getattr(args, attr)), label))

    datasets, dataset_meta = _load_datasets(args)
    if args.dataset_id not in datasets:
        raise KeyError(f"dataset_id={args.dataset_id!r} not found; available={sorted(datasets)}")
    ds = datasets[args.dataset_id]
    sample = _find_sample(ds, args.object_id, int(args.angle), args.dataset_id)
    ds_sample = _find_dataset_sample(ds, sample)
    started = time.time()
    run_dir = _ensure_ss_and_part(args, ds, sample, ds_sample)

    slat_view_indices = (
        [int(v) for v in args.slat_view_indices]
        if args.slat_view_indices is not None
        else [int(v) for v in ds_sample.get("view_indices", [])]
    )
    if not slat_view_indices:
        raise ValueError(f"{args.dataset_id}::{args.object_id} angle={int(args.angle)} has no manifest view_indices")
    cond_tokens, slat_cond_meta = _load_slat_cond_tokens_for_views(
        ds,
        ds_sample,
        slat_view_indices,
        token_source=str(args.slat_token_source),
    )
    whole_coords = _load_coords(run_dir / "voxel.npz")
    whole_coords_t = torch.from_numpy(np.ascontiguousarray(whole_coords.astype(np.int64, copy=False))).long()
    print(
        f"[0617-128ee] SLat flow ONCE {args.dataset_id}::{args.object_id} "
        f"angle={int(args.angle)} coords={whole_coords.shape[0]} views={slat_cond_meta['view_indices']}",
        flush=True,
    )
    overall_slat = inference.run_slat_flow_from_tokens(
        cond_tokens,
        whole_coords_t,
        str(args.slat_flow_ckpt.resolve()),
        num_steps=int(args.slat_steps),
        seed=int(args.slat_seed),
    )

    extrinsics, intrinsics = load_camera_matrices(
        Path(ds.data_root) / "renders" / sample.obj_id / f"angle_{int(sample.angle_idx)}" / "camera_transforms.json",
        [int(args.render_view)],
    )
    extrinsic, intrinsic = extrinsics[0], intrinsics[0]

    components: list[tuple[str, np.ndarray, SparseTensor | None, str, int, str | None]] = [
        ("overall", whole_coords, overall_slat, "whole_slat_flow_once", int(whole_coords.shape[0]), None),
    ]
    part_items: list[tuple[str, np.ndarray]] = []
    for part_idx, part in enumerate(ds_sample["parts"]):
        coords = _load_coords(run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz")
        part_items.append((str(part["part_name"]), coords))
        label = f"part_{part_idx:02d}_{_safe_name(str(part['part_name']))}"
        try:
            part_slat, matched = _sparse_subset_from_coords(overall_slat, coords, label)
            components.append((label, coords, part_slat, "subset_from_whole_slat_by_coords", matched, None))
        except ValueError as exc:
            components.append((label, coords, None, "subset_from_whole_slat_by_coords_failed", 0, str(exc)))

    gauss_tiles: list[tuple[str, Image.Image]] = []
    mesh_tiles: list[tuple[str, Image.Image]] = []
    records: list[dict[str, Any]] = []
    prefix = _prefix(args.dataset_id, args.object_id, int(args.angle))
    mujoco_dir = args.out_dir / f"{prefix}__mujoco"
    mujoco_assets_dir = mujoco_dir / "assets"
    mujoco_part_meshes: list[dict[str, Any]] = []
    for label, coords, slat, slat_source, matched, subset_error in components:
        t0 = time.time()
        print(f"[0617-128ee] decode+render {label} coords={len(coords)} matched={matched}", flush=True)
        if slat is None:
            error_text = subset_error or "missing part SLat"
            gauss_tiles.append((label, _error_image(error_text, int(args.resolution))))
            mesh_tiles.append((label, _error_image(error_text, int(args.resolution))))
            records.append(
                {
                    "label": label,
                    "coords": int(len(coords)),
                    "matched_coords": int(matched),
                    "slat_source": slat_source,
                    "slat_subset_error": error_text,
                    "mesh_vertices": 0,
                    "mesh_faces": 0,
                    "mesh_has_vertex_attrs": False,
                    "mesh_error": error_text,
                    "gaussian_error": error_text,
                    "gs_preset": None,
                    "seconds": round(time.time() - t0, 3),
                }
            )
            continue
        decoded = inference.decode_slat_assets(
            slat,
            gaussian_decoder_ckpt=str(args.slat_gaussian_decoder_ckpt.resolve()),
            mesh_decoder_ckpt=str(args.slat_mesh_decoder_ckpt.resolve()),
            slat_is_normalized=True,
        )
        gaussian = decoded.get("gaussian")
        mesh = decoded.get("mesh")
        if gaussian is None:
            raise RuntimeError(f"{label}: gaussian decoder returned None")
        gaussian, gs_stats = _apply_gs_preset(gaussian)
        gauss_tiles.append((label, _render_gaussian(gaussian, extrinsic, intrinsic, int(args.resolution))))
        mesh_error = None
        mesh_vertices = 0
        mesh_faces = 0
        mesh_has_vertex_attrs = False
        if mesh is None or not getattr(mesh, "success", True):
            mesh_error = "mesh decoder failed"
            mesh_tiles.append((label, _error_image(mesh_error, int(args.resolution))))
        else:
            mesh_vertices = int(mesh.vertices.shape[0])
            mesh_faces = int(mesh.faces.shape[0])
            mesh_has_vertex_attrs = bool(getattr(mesh, "vertex_attrs", None) is not None)
            mesh_tiles.append((label, _render_mesh(mesh, extrinsic, intrinsic, int(args.resolution))))
            if args.export_mujoco and label != "overall":
                obj_name = f"{_safe_name(label, 96)}.obj"
                assets = save_decoded_slat_assets({"mesh": mesh}, mujoco_assets_dir, mesh_name=obj_name)
                mujoco_part_meshes.append(
                    {
                        "label": label,
                        "mesh_file": f"assets/{assets['mesh']}",
                        "mesh_path": str((mujoco_assets_dir / assets["mesh"]).resolve()),
                        "coords": int(len(coords)),
                        "matched_coords": int(matched),
                        "vertices": int(mesh_vertices),
                        "faces": int(mesh_faces),
                    }
                )
        records.append(
            {
                "label": label,
                "coords": int(len(coords)),
                "matched_coords": int(matched),
                "slat_source": slat_source,
                "slat_subset_error": subset_error,
                "mesh_vertices": mesh_vertices,
                "mesh_faces": mesh_faces,
                "mesh_has_vertex_attrs": mesh_has_vertex_attrs,
                "mesh_error": mesh_error,
                "gaussian_error": None,
                "gs_preset": gs_stats,
                "seconds": round(time.time() - t0, 3),
            }
        )
        del decoded, gaussian, mesh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gaussian_png = args.out_dir / f"{prefix}__gaussian.png"
    mesh_png = args.out_dir / f"{prefix}__mesh.png"
    diagnostic_png = args.out_dir / f"{prefix}__diagnostic.png"
    summary_path = args.out_dir / f"{prefix}__summary.json"
    mujoco_xml = None
    if args.export_mujoco and mujoco_part_meshes:
        mujoco_xml = _write_static_mujoco_xml(
            out_xml=mujoco_dir / f"{prefix}.xml",
            model_name=prefix,
            part_meshes=mujoco_part_meshes,
        )
    if not gaussian_png.is_file() or args.force_export:
        _panel(gauss_tiles, gaussian_png, tile_size=int(args.tile_size), cols=int(args.panel_cols))
    if not mesh_png.is_file() or args.force_export:
        _panel(mesh_tiles, mesh_png, tile_size=int(args.tile_size), cols=int(args.panel_cols))
    if not diagnostic_png.is_file() or args.force_export:
        _write_diagnostic_panel(
            ds=ds,
            ds_sample=ds_sample,
            whole_coords=whole_coords,
            part_items=part_items,
            gaussian_png=gaussian_png,
            mesh_png=mesh_png,
            out_png=diagnostic_png,
            object_id=args.object_id,
            angle=int(args.angle),
        )

    summary = {
        "status": "done",
        "dataset_id": args.dataset_id,
        "object_id": args.object_id,
        "angle": int(args.angle),
        "out_dir": str(args.out_dir),
        "run_dir": str(run_dir.resolve()),
        "gaussian_png": str(gaussian_png.resolve()),
        "mesh_png": str(mesh_png.resolve()),
        "diagnostic_png": str(diagnostic_png.resolve()),
        "mujoco_xml": None if mujoco_xml is None else str(mujoco_xml.resolve()),
        "mujoco_assets_dir": None if mujoco_xml is None else str(mujoco_assets_dir.resolve()),
        "mujoco_part_meshes": mujoco_part_meshes,
        "component_count": len(records),
        "components": records,
        "ss_stage": {
            "source": "input 4-view DINO tokens used by SS stage, not part_synthesis_slat",
            "fusion_mode": "concat",
            "ckpt": str(args.ss_flow_ckpt.resolve()),
        },
        "part_stage": {
            "backend": "promptable_seg",
            "ckpt": str(args.part_seg_ckpt.resolve()),
        },
        "slat_stage": {
            "flow_calls": 1,
            "part_rule": "whole SLat flow once, then slice by part voxel coords",
            "ckpt": str(args.slat_flow_ckpt.resolve()),
            "condition": slat_cond_meta,
        },
        "gs_preset": GS_PRESET,
        "datasets": dataset_meta,
        "seconds": round(time.time() - started, 3),
    }
    _write_json(summary_path, summary)
    print(f"[0617-128ee] gaussian -> {gaussian_png}", flush=True)
    print(f"[0617-128ee] mesh -> {mesh_png}", flush=True)
    if mujoco_xml is not None:
        print(f"[0617-128ee] mujoco -> {mujoco_xml}", flush=True)
    print(f"[0617-128ee] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
