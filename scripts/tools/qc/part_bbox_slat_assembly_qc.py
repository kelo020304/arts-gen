#!/usr/bin/env python3
"""End-to-end per-part bbox normalization and SLat assembly QC."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
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

from inference_pipeline.object_inputs import load_object_inputs  # noqa: E402


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


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(require_file(path, "yaml config").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"yaml config must be a mapping: {path}")
    return payload


def data_config_from_yaml(path: Path, data_root: Path, *, include_obj_id: str) -> dict[str, Any]:
    payload = load_yaml(path)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} missing mapping key 'data'")
    cfg = dict(data)
    cfg["data_root"] = str(data_root.resolve())
    cfg["include_obj_ids"] = [str(include_obj_id)]
    cfg["filter_zero_mask_coverage"] = False
    return cfg


def load_images_for_item(data_root: Path, object_id: str, angle_idx: int, view_indices: list[int]) -> list[Image.Image]:
    rgb_root = require_dir(data_root / "renders" / object_id / f"angle_{angle_idx}" / "rgb", "RGB render root")
    images = []
    for view_idx in view_indices:
        images.append(Image.open(require_file(rgb_root / f"view_{int(view_idx)}.png", f"RGB view {view_idx}")).convert("RGBA"))
    return images


def save_part_voxels(
    run_dir: Path,
    part_coords: dict[str, np.ndarray],
    part_names: list[str],
    part_sources: dict[str, str],
) -> list[dict[str, Any]]:
    parts_dir = run_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for part_index, name in enumerate(part_names):
        coords = np.asarray(part_coords[name], dtype=np.int64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise RuntimeError(f"{name}: predicted coords must be [N,3], got {coords.shape}")
        if coords.shape[0] == 0:
            raise RuntimeError(f"{name}: predicted coords are empty")
        if int(coords.min()) < 0 or int(coords.max()) >= 64:
            raise RuntimeError(f"{name}: predicted coords out of [0,64): min={coords.min()} max={coords.max()}")
        path = parts_dir / f"part_{part_index:02d}_voxel.npz"
        np.savez_compressed(
            path,
            coords=coords.astype(np.int32),
            resolution=np.int32(64),
            coord_frame="canonical_grid",
            source=str(part_sources[name]),
            part_index=np.int32(part_index),
            target_part_name=str(name),
        )
        records.append(
            {
                "part": name,
                "path": str(path.resolve()),
                "source": str(part_sources[name]),
                "voxel_count": int(coords.shape[0]),
            }
        )
    return records


def bbox_normalize_coords(coords_np: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    coords = np.asarray(coords_np, dtype=np.float32)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    extent = bbox_max - bbox_min
    scale = float(extent.max())
    if scale <= 0:
        raise RuntimeError(f"bbox scale must be > 0, got {scale} for bbox {bbox_min.tolist()}..{bbox_max.tolist()}")
    center = (bbox_min + bbox_max) * 0.5
    local = (coords - center[None, :]) / scale
    grid_f = (local + 0.5) * 64.0 - 0.5
    norm = np.rint(grid_f).astype(np.int64)
    norm = np.clip(norm, 0, 63)
    unique = np.unique(norm, axis=0)
    if unique.shape[0] == 0:
        raise RuntimeError("normalized coords became empty")
    meta = {
        "bbox_min": [int(x) for x in bbox_min.tolist()],
        "bbox_max": [int(x) for x in bbox_max.tolist()],
        "center_grid": [float(x) for x in center.tolist()],
        "scale_grid": scale,
        "voxel_count_shared": int(coords_np.shape[0]),
        "voxel_count_normalized": int(unique.shape[0]),
        "normalized_min": [int(x) for x in unique.min(axis=0).tolist()],
        "normalized_max": [int(x) for x in unique.max(axis=0).tolist()],
    }
    return unique.astype(np.int64), meta


def grid_center_to_world(center_grid: list[float], scale_grid: float) -> tuple[torch.Tensor, float]:
    center = torch.tensor(center_grid, dtype=torch.float32, device="cuda")
    center_world = (center + 0.5) / 64.0 - 0.5
    scale_world = float(scale_grid) / 64.0
    return center_world, scale_world


def transform_gaussian_from_part_cube(gaussian: Any, center_grid: list[float], scale_grid: float) -> Any:
    out = copy.deepcopy(gaussian)
    center_world, scale_world = grid_center_to_world(center_grid, scale_grid)
    xyz_local = gaussian.get_xyz.detach().float()
    xyz_world = xyz_local * float(scale_world) + center_world[None, :]
    out.from_xyz(xyz_world)
    scaled = gaussian.get_scaling.detach().float() * float(scale_world)
    out.from_scaling(scaled.clamp_min(float(out.mininum_kernel_size) + 1e-7))
    return out


def clone_gaussian(gaussian: Any) -> Any:
    return copy.deepcopy(gaussian)


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


def load_camera_matrices(camera_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    import utils3d

    payload = json.loads(require_file(camera_path, "camera_transforms").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"camera_transforms.frames must be a non-empty list: {camera_path}")
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
        "resolution": int(payload.get("resolution", 512)),
        "total_views": len(frames),
        "camera_center_norm_range": [float(center_norm.min()), float(center_norm.max())],
        "fov_deg": payload.get("fov_deg"),
    }
    return torch.stack(extrinsics), torch.stack(intrinsics), meta


@torch.no_grad()
def render_views(gaussian: Any, renderer: Any, extrinsics: torch.Tensor, intrinsics: torch.Tensor, views: list[int]) -> list[torch.Tensor]:
    frames = []
    for view_idx in views:
        out = renderer.render(gaussian, extrinsics[int(view_idx)], intrinsics[int(view_idx)], colors_overwrite=None)
        if "color" not in out:
            raise RuntimeError(f"GaussianRenderer output for view {view_idx} missing color")
        color = out["color"].detach().float().cpu().clamp(0, 1)
        if color.ndim != 3 or color.shape[0] != 3:
            raise RuntimeError(f"rendered color must be [3,H,W], got {tuple(color.shape)}")
        frames.append(color)
    return frames


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def make_render_grid(path: Path, norm_images: list[Image.Image], shared_images: list[Image.Image], view_indices: list[int]) -> None:
    if len(norm_images) != len(shared_images) or len(norm_images) != len(view_indices):
        raise RuntimeError("render grid inputs length mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = norm_images[0].size
    title_h = 30
    label_h = 24
    rows = len(view_indices)
    cols = 2
    canvas = Image.new("RGB", (cols * w, title_h + rows * (label_h + h)), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "assembled render: bbox-normalized parts | shared-grid parts", fill=(255, 255, 255))
    for row, view_idx in enumerate(view_indices):
        y = title_h + row * (label_h + h)
        labels = [f"norm+bbox putback view {view_idx}", f"shared-grid baseline view {view_idx}"]
        for col, (img, label) in enumerate(zip((norm_images[row], shared_images[row]), labels)):
            x = col * w
            draw.rectangle((x, y, x + w, y + label_h), fill=(0, 0, 0))
            draw.text((x + 6, y + 6), label, fill=(255, 255, 255))
            canvas.paste(img.convert("RGB"), (x, y + label_h))
    canvas.save(path)


def load_fixed_body_parts(data_root: Path, object_id: str, angle_idx: int, existing_parts: set[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    part_info_path = require_file(
        data_root / "reconstruction" / "part_info" / object_id / "part_info.json",
        "part_info",
    )
    payload = json.loads(part_info_path.read_text(encoding="utf-8"))
    parts = payload.get("parts")
    if not isinstance(parts, dict):
        raise RuntimeError(f"part_info.parts must be a mapping: {part_info_path}")
    voxel_root = require_dir(
        data_root / "reconstruction" / "voxel_expanded" / object_id / f"angle_{angle_idx}" / "64",
        "voxel_expanded part root",
    )
    names = []
    coords_by_name: dict[str, np.ndarray] = {}
    for name, meta in parts.items():
        if name in existing_parts:
            continue
        if not isinstance(meta, dict):
            raise RuntimeError(f"part_info part {name!r} must be a mapping")
        joint = str(meta.get("joint", "")).lower()
        ptype = str(meta.get("type", "")).lower()
        if joint != "fixed" and "body" not in ptype:
            continue
        voxel_path = require_file(voxel_root / f"ind_{name}.npy", f"fixed/body voxel {name}")
        coords = np.asarray(np.load(voxel_path), dtype=np.int64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise RuntimeError(f"{voxel_path} expected [N,3], got {coords.shape}")
        if coords.shape[0] == 0:
            raise RuntimeError(f"{voxel_path} contains zero voxels")
        if int(coords.min()) < 0 or int(coords.max()) >= 64:
            raise RuntimeError(f"{voxel_path} coords out of [0,64): min={coords.min()} max={coords.max()}")
        names.append(str(name))
        coords_by_name[str(name)] = coords
    return names, coords_by_name


def load_part_voxel_from_data(data_root: Path, object_id: str, angle_idx: int, part_name: str) -> np.ndarray:
    voxel_path = require_file(
        data_root
        / "reconstruction"
        / "voxel_expanded"
        / object_id
        / f"angle_{angle_idx}"
        / "64"
        / f"ind_{part_name}.npy",
        f"cached part voxel {part_name}",
    )
    coords = np.asarray(np.load(voxel_path), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"{voxel_path} expected [N,3], got {coords.shape}")
    if coords.shape[0] == 0:
        raise RuntimeError(f"{voxel_path} contains zero voxels")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise RuntimeError(f"{voxel_path} coords out of [0,64): min={coords.min()} max={coords.max()}")
    return coords


def load_ss_flow_model_any(ckpt_path: Path):
    from safetensors.torch import load_file
    from trellis.models.sparse_structure_flow import SparseStructureFlowModel

    model = SparseStructureFlowModel(**inference._SS_FLOW_DEFAULT_ARGS).cuda().eval()
    ckpt_path = require_file(ckpt_path, "SS flow ckpt")
    if ckpt_path.suffix == ".safetensors":
        state = load_file(str(ckpt_path.resolve()))
    elif ckpt_path.suffix == ".pt":
        state = torch.load(str(ckpt_path.resolve()), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not hasattr(state, "keys"):
            raise RuntimeError(f"unsupported SS flow .pt checkpoint payload: {ckpt_path}")
    else:
        raise ValueError(f"unsupported SS flow checkpoint suffix {ckpt_path.suffix}: {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"SS flow checkpoint did not load cleanly: missing={missing[:10]} unexpected={unexpected[:10]} "
            f"counts=({len(missing)},{len(unexpected)}) path={ckpt_path}"
        )
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[ss-flow] loaded {ckpt_path.resolve()} missing=0 unexpected=0", flush=True)
    return model


def run_ss_flow_any(images: list[Image.Image], ckpt_path: Path, *, num_steps: int, cfg_strength: float) -> torch.Tensor:
    from trellis.pipelines.samplers.flow_euler import FlowEulerCfgSampler

    if not images:
        raise ValueError("run_ss_flow_any requires at least one image")
    tokens = inference._images_to_tokens(images)
    cond = tokens.reshape(1, tokens.shape[0] * tokens.shape[1], tokens.shape[2])
    model = load_ss_flow_model_any(ckpt_path)
    sampler = FlowEulerCfgSampler(sigma_min=1e-5)
    noise = torch.randn(1, 8, 16, 16, 16, device=cond.device, dtype=torch.float32)
    with torch.no_grad():
        result = sampler.sample(
            model=model,
            noise=noise,
            cond=cond,
            neg_cond=torch.zeros_like(cond),
            steps=int(num_steps),
            cfg_strength=float(cfg_strength),
            verbose=False,
        )
    return result.samples[0].detach().float().cpu()


def decode_ss_with_threshold(z_s: torch.Tensor, decoder_ckpt_path: Path, threshold: float) -> torch.Tensor:
    if z_s.dim() != 4:
        raise ValueError(f"z_s must have 4 dims [C,H,W,D], got {tuple(z_s.shape)}")
    decoder = inference._load_ss_decoder(str(decoder_ckpt_path.resolve()))
    z = z_s.unsqueeze(0).cuda()
    if next(decoder.parameters()).dtype == torch.float16:
        z = z.half()
    with torch.no_grad():
        logits = decoder(z)
    occ = logits[0, 0].float() > float(threshold)
    return torch.nonzero(occ, as_tuple=False).long().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-config-yaml", type=Path, required=True)
    parser.add_argument("--object-id", default="100058")
    parser.add_argument("--angle-idx", type=int, default=0)
    parser.add_argument("--part-flow-ckpt", type=Path, required=True)
    parser.add_argument("--ss-flow-ckpt", type=Path, required=True)
    parser.add_argument("--ss-decoder-ckpt", type=Path, required=True)
    parser.add_argument("--slat-flow-ckpt", type=Path, required=True)
    parser.add_argument("--slat-decoder-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-fixed-body", action="store_true")
    parser.add_argument("--force-cached-parts-if-empty", action="store_true")
    parser.add_argument("--ss-source", choices=("predicted", "cached_gt"), default="predicted")
    parser.add_argument("--ss-steps", type=int, default=20)
    parser.add_argument("--ss-cfg-strength", type=float, default=3.0)
    parser.add_argument("--ss-decode-threshold", type=float, default=0.0)
    parser.add_argument("--part-decode-threshold", type=float, default=0.0)
    parser.add_argument("--part-steps", type=int, default=None)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-views", nargs="+", type=int, default=[0, 4, 7, 11])
    args = parser.parse_args()

    data_root = require_dir(args.data_root, "DATA_ROOT")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"{args.object_id}_angle_{args.angle_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for path, label in (
        (args.part_flow_ckpt, "part flow ckpt"),
        (args.ss_flow_ckpt, "SS flow ckpt"),
        (args.ss_decoder_ckpt, "SS decoder ckpt"),
        (args.slat_flow_ckpt, "SLat flow ckpt"),
        (args.slat_decoder_ckpt, "SLat gaussian decoder ckpt"),
    ):
        require_file(path, label)

    data_cfg = data_config_from_yaml(args.data_config_yaml, data_root, include_obj_id=args.object_id)
    item = load_object_inputs(data_cfg, object_id=args.object_id, angle_idx=args.angle_idx, view_mode="multi")
    target_part_names = list(item["target_part_names"])
    view_indices = [int(v) for v in item["view_indices"]]
    print(f"[object] {args.object_id}/angle_{args.angle_idx} dataset_index={item['dataset_index']}", flush=True)
    print(f"[object] target_part_names={target_part_names}", flush=True)
    print(f"[object] manifest_view_indices={view_indices}", flush=True)
    print(f"[object] cond_shape={tuple(item['cond'].shape)}", flush=True)
    if "part_token_weights" in item:
        print(f"[object] part_token_weights_shape={tuple(item['part_token_weights'].shape)}", flush=True)

    model, cfg = inference._load_part_ss_latent_flow(str(args.part_flow_ckpt.resolve()))
    ckpt = torch.load(str(args.part_flow_ckpt.resolve()), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    lora_count = sum(1 for key in state.keys() if "lora" in str(key).lower())
    ckpt_type = "LoRA" if lora_count else "full"
    print(f"[ckpt] part_flow_type={ckpt_type} lora_key_count={lora_count}", flush=True)
    print(f"[ckpt] part_flow_param_count={sum(p.numel() for p in model.parameters())}", flush=True)

    images = load_images_for_item(data_root, args.object_id, args.angle_idx, view_indices)
    if args.ss_source == "predicted":
        z_global = run_ss_flow_any(
            images,
            args.ss_flow_ckpt.resolve(),
            num_steps=args.ss_steps,
            cfg_strength=args.ss_cfg_strength,
        )
    else:
        z_path = require_file(
            data_root / "reconstruction" / "ss_latents_expanded" / args.object_id / f"angle_{args.angle_idx}" / "latent.npz",
            "cached GT overall SS latent",
        )
        with np.load(z_path) as data:
            if "mean" not in data.files:
                raise RuntimeError(f"{z_path} expected key 'mean', got {data.files}")
            z_global = torch.from_numpy(np.asarray(data["mean"])).float()
        print(f"[ss] loaded cached GT z_global={z_path}", flush=True)
    np.save(run_dir / "ss_latent.npy", z_global.numpy().astype(np.float32))
    ss_coords = decode_ss_with_threshold(z_global, args.ss_decoder_ckpt.resolve(), threshold=args.ss_decode_threshold)
    print(
        f"[ss] source={args.ss_source} z_global_shape={tuple(z_global.shape)} "
        f"decode_threshold={args.ss_decode_threshold} decoded_voxels={int(ss_coords.shape[0])}",
        flush=True,
    )

    result = inference.run_part_ss_latent_flow(
        z_global,
        item["cond"],
        str(args.part_flow_ckpt.resolve()),
        target_slots=item["target_slots"].tolist(),
        mask_token_labels=item["mask_token_labels"],
        target_part_names=target_part_names,
        part_token_weights=item.get("part_token_weights"),
        ss_decoder_ckpt=str(args.ss_decoder_ckpt.resolve()),
        num_steps=args.part_steps,
        decode_threshold=float(args.part_decode_threshold),
        decode=True,
    )
    all_part_names = list(target_part_names)
    part_coord_arrays: dict[str, np.ndarray] = {}
    part_sources: dict[str, str] = {}
    empty_pred_parts = []
    for name in target_part_names:
        coords = np.asarray(result["part_coords"][name], dtype=np.int64)
        if coords.shape[0] == 0:
            empty_pred_parts.append(name)
            if not args.force_cached_parts_if_empty:
                part_coord_arrays[name] = coords
                part_sources[name] = "pred_part_ss_latent_flow_600k_empty"
                continue
            coords = load_part_voxel_from_data(data_root, args.object_id, args.angle_idx, name)
            part_coord_arrays[name] = coords
            part_sources[name] = "cached_part_voxel_due_to_empty_pred"
            print(f"[part-empty] {name}: predicted empty; using cached ind voxel", flush=True)
        else:
            part_coord_arrays[name] = coords
            part_sources[name] = "pred_part_ss_latent_flow_600k"
    fixed_body_names: list[str] = []
    if args.include_fixed_body:
        fixed_body_names, fixed_body_coords = load_fixed_body_parts(
            data_root,
            args.object_id,
            args.angle_idx,
            set(all_part_names),
        )
        for name in fixed_body_names:
            all_part_names.append(name)
            part_coord_arrays[name] = fixed_body_coords[name]
            part_sources[name] = "cached_fixed_body_voxel"
        print(f"[body] included_fixed_body_parts={fixed_body_names}", flush=True)
    part_voxel_records = save_part_voxels(run_dir, part_coord_arrays, all_part_names, part_sources)

    part_coords_shared: dict[str, torch.Tensor] = {}
    part_coords_norm: dict[str, torch.Tensor] = {}
    part_rows = []
    bbox_meta_by_part: dict[str, dict[str, Any]] = {}
    for rec in part_voxel_records:
        part = rec["part"]
        coords = np.asarray(part_coord_arrays[part], dtype=np.int64)
        norm_coords, bbox_meta = bbox_normalize_coords(coords)
        bbox_meta_by_part[part] = bbox_meta
        part_coords_shared[part] = torch.from_numpy(coords).long()
        part_coords_norm[part] = torch.from_numpy(norm_coords).long()
        row = {
            "part": part,
            "source": rec["source"],
            "voxel_path": rec["path"],
            **bbox_meta,
        }
        part_rows.append(row)
        print(
            f"[bbox] {part} vox_shared={bbox_meta['voxel_count_shared']} "
            f"vox_norm={bbox_meta['voxel_count_normalized']} "
            f"bbox_min={bbox_meta['bbox_min']} bbox_max={bbox_meta['bbox_max']} "
            f"center={bbox_meta['center_grid']} scale={bbox_meta['scale_grid']}",
            flush=True,
        )

    cond4 = item["cond"].reshape(len(view_indices), -1, item["cond"].shape[-1]).contiguous()
    norm_gaussians = []
    shared_gaussians = []
    decoder = inference._load_slat_vae_decoder(str(args.slat_decoder_ckpt.resolve()))
    for part_index, part in enumerate(all_part_names):
        seed = int(args.seed) + int(item["dataset_index"]) * 1_000_003 + part_index * 9_176
        slat_norm = inference.run_slat_flow_from_tokens(
            cond4,
            part_coords_norm[part],
            str(args.slat_flow_ckpt.resolve()),
            num_steps=args.slat_steps,
            seed=seed,
        )
        decoded_norm = inference.decode_slat_assets(
            slat_norm,
            gaussian_decoder_ckpt=str(args.slat_decoder_ckpt.resolve()),
            slat_is_normalized=True,
        )["gaussian"]
        putback = transform_gaussian_from_part_cube(
            decoded_norm,
            bbox_meta_by_part[part]["center_grid"],
            bbox_meta_by_part[part]["scale_grid"],
        )
        norm_gaussians.append(putback)

        slat_shared = inference.run_slat_flow_from_tokens(
            cond4,
            part_coords_shared[part],
            str(args.slat_flow_ckpt.resolve()),
            num_steps=args.slat_steps,
            seed=seed,
        )
        decoded_shared = inference.decode_slat_assets(
            slat_shared,
            gaussian_decoder_ckpt=str(args.slat_decoder_ckpt.resolve()),
            slat_is_normalized=True,
        )["gaussian"]
        shared_gaussians.append(clone_gaussian(decoded_shared))
        print(
            f"[slat] {part} norm_vox={part_coords_norm[part].shape[0]} "
            f"shared_vox={part_coords_shared[part].shape[0]} seed={seed}",
            flush=True,
        )

    assembled_norm = merge_gaussians(norm_gaussians)
    assembled_shared = merge_gaussians(shared_gaussians)
    camera_path = data_root / "renders" / args.object_id / f"angle_{args.angle_idx}" / "camera_transforms.json"
    extr, intr, camera_meta = load_camera_matrices(camera_path)
    renderer = make_renderer(camera_meta["resolution"], decoder)
    render_views_list = [idx for idx in args.render_views if 0 <= int(idx) < int(camera_meta["total_views"])]
    if not render_views_list:
        raise RuntimeError(f"no valid render views from {args.render_views}")
    norm_frames = render_views(assembled_norm, renderer, extr, intr, render_views_list)
    shared_frames = render_views(assembled_shared, renderer, extr, intr, render_views_list)
    norm_images = [tensor_to_image(frame) for frame in norm_frames]
    shared_images = [tensor_to_image(frame) for frame in shared_frames]
    grid_path = run_dir / "assembled_norm_vs_shared.png"
    make_render_grid(grid_path, norm_images, shared_images, render_views_list)

    norm_ply = run_dir / "assembled_norm_putback.ply"
    shared_ply = run_dir / "assembled_shared_grid.ply"
    assembled_norm.save_ply(str(norm_ply), transform=None)
    assembled_shared.save_ply(str(shared_ply), transform=None)

    report = {
        "object_id": args.object_id,
        "angle_idx": int(args.angle_idx),
        "data_root": str(data_root),
        "official_renderer": {
            "gaussian_renderer_module": gaussian_render_mod.__file__,
            "diff_gaussian_module": diff_gaussian_rasterization.__file__,
            "used_gsplat": False,
        },
        "ckpts": {
            "part_flow": str(args.part_flow_ckpt.resolve()),
            "part_flow_type": ckpt_type,
            "part_flow_lora_key_count": int(lora_count),
            "part_flow_strict_load_clean": True,
            "part_flow_param_count": int(sum(p.numel() for p in model.parameters())),
            "ss_flow": str(args.ss_flow_ckpt.resolve()),
            "ss_decoder": str(args.ss_decoder_ckpt.resolve()),
            "slat_flow": str(args.slat_flow_ckpt.resolve()),
            "slat_gaussian_decoder": str(args.slat_decoder_ckpt.resolve()),
        },
        "sampling": {
            "ss_steps": int(args.ss_steps),
            "ss_source": args.ss_source,
            "ss_cfg_strength": float(args.ss_cfg_strength),
            "ss_decode_threshold": float(args.ss_decode_threshold),
            "part_steps": args.part_steps,
            "part_decode_threshold": float(args.part_decode_threshold),
            "slat_steps": int(args.slat_steps),
            "seed": int(args.seed),
            "manifest_view_indices": view_indices,
            "render_views": render_views_list,
            "slat_flow_output_is_normalized": True,
            "decode_slat_assets_slat_is_normalized_for_flow": True,
        },
        "dataset_index": int(item["dataset_index"]),
        "target_part_names": target_part_names,
        "fixed_body_part_names": fixed_body_names,
        "assembled_part_names": all_part_names,
        "ss_decoded_voxel_count": int(ss_coords.shape[0]),
        "empty_predicted_parts": empty_pred_parts,
        "parts": part_rows,
        "outputs": {
            "run_dir": str(run_dir),
            "parts_dir": str((run_dir / "parts").resolve()),
            "assembled_render_grid": str(grid_path.resolve()),
            "assembled_norm_ply": str(norm_ply.resolve()),
            "assembled_shared_ply": str(shared_ply.resolve()),
        },
        "camera_meta": camera_meta,
        "notes": [
            "SLat flow is TRELLIS pretrained baseline, not part-finetuned.",
            "bbox is derived only from predicted part voxel min/max; no bbox generation model is used.",
            "normalized part coords are quantized back to 64^3 before SLat flow.",
        ],
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] report={report_path}", flush=True)
    print(f"[done] render_grid={grid_path}", flush=True)


if __name__ == "__main__":
    main()
