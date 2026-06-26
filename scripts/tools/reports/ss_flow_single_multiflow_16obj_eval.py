#!/usr/bin/env python3
"""Evaluate single-view SS flow checkpoints with 4-view velocity averaging."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPCONV_ALGO", "native")


def _register_trellis_package() -> None:
    if "trellis" not in sys.modules:
        pkg = types.ModuleType("trellis")
        pkg.__path__ = [str(TRELLIS_ROOT / "trellis")]
        pkg.__package__ = "trellis"
        sys.modules["trellis"] = pkg
    for subpkg in ("models", "modules", "pipelines", "utils", "datasets", "trainers"):
        name = f"trellis.{subpkg}"
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(TRELLIS_ROOT / "trellis" / subpkg)]
            mod.__package__ = name
            sys.modules[name] = mod


_register_trellis_package()

from safetensors.torch import load_file  # noqa: E402
from trellis.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402
from trellis.utils.arts.config_utils import config_to_dict, load_config  # noqa: E402


DEFAULT_CONFIG = TRELLIS_ROOT / "configs/arts/ss_flow_global_z/official_single_multiflow_16obj.yaml"
DEFAULT_MODEL_CONFIG = Path("/mnt/robot-data-lab/jzh/art-gen/weights/ss_flow_img_dit_L_16l8_fp16.json")
DEFAULT_OFFICIAL_CKPT = Path("/mnt/robot-data-lab/jzh/art-gen/weights/ss_flow_img_dit_L_16l8_fp16.safetensors")
DEFAULT_DECODER_CKPT = PROJECT_ROOT / "pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
DEFAULT_OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/ss_flow_official_single_multiflow_16obj_0610")


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def rooted(data_root: Path, rel_or_abs: str | Path) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else data_root / path


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object, got {type(payload)}")
    return payload


def latest_ema_ckpt(output_dir: Path, ema_rate: str) -> Path:
    ckpt_dir = output_dir / "ckpts"
    require_dir(ckpt_dir, "checkpoint directory")
    pattern = f"denoiser_ema{ema_rate}_step*.pt"
    candidates = sorted(ckpt_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no EMA checkpoint matching {pattern} under {ckpt_dir}")
    return candidates[-1].resolve()


def iter_manifest(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    require_file(path, "manifest")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{line_no}: expected object record, got {type(rec)}")
            obj_id = str(rec.get("object_id", rec.get("obj_id", "")))
            if not obj_id:
                raise KeyError(f"{path}:{line_no}: missing object_id/obj_id")
            if "angle_idx" not in rec and "angle" not in rec:
                raise KeyError(f"{path}:{line_no}: missing angle_idx/angle")
            angle_idx = int(rec.get("angle_idx", rec.get("angle")))
            rec["_manifest_line"] = line_no
            rows.setdefault((obj_id, angle_idx), rec)
    if not rows:
        raise ValueError(f"manifest has no samples: {path}")
    return rows


def parse_sample_keys(value: Any) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for item in value:
        if isinstance(item, str):
            if ":angle_" in item:
                obj_id, angle_text = item.split(":angle_", 1)
            elif ":" in item:
                obj_id, angle_text = item.split(":", 1)
            else:
                raise ValueError(f"sample string must be '<obj_id>:angle_<idx>', got {item!r}")
            keys.append((str(obj_id), int(angle_text)))
            continue
        if not isinstance(item, dict):
            raise TypeError(f"sample entries must be dicts or strings, got {type(item)}")
        obj_id = str(item.get("obj_id", item.get("object_id", "")))
        if not obj_id:
            raise KeyError(f"sample entry missing obj_id/object_id: {item}")
        if "angle_idx" not in item and "angle" not in item:
            raise KeyError(f"sample entry missing angle_idx/angle: {item}")
        keys.append((obj_id, int(item.get("angle_idx", item.get("angle")))))
    return keys


def load_samples_from_config(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = config_to_dict(load_config(str(config_path)))
    data_cfg = dict(cfg["data"])
    data_root = Path(data_cfg["data_root"]).resolve()
    manifest_path = Path(str(data_cfg["manifest_path"]))
    manifest = manifest_path if manifest_path.is_absolute() else data_root / manifest_path
    manifest_rows = iter_manifest(manifest)
    sample_keys = parse_sample_keys(data_cfg.get("test_samples", []))
    if not sample_keys:
        raise ValueError(f"{config_path}: data.test_samples is required for this eval")
    tokens_subdir = str(data_cfg.get("tokens_subdir", "dinov2_tokens_official_prenorm1374"))
    latent_subdir = str(data_cfg.get("latent_subdir", "ss_latents_expanded"))
    renders_subdir = str(data_cfg.get("renders_subdir", "renders"))
    recon_subdir = str(data_cfg.get("recon_subdir", "reconstruction"))
    samples: list[dict[str, Any]] = []
    missing = [key for key in sample_keys if key not in manifest_rows]
    if missing:
        missing_str = ", ".join(f"{obj_id}:angle_{angle_idx}" for obj_id, angle_idx in missing)
        raise RuntimeError(f"requested samples not found in manifest: {missing_str}")
    for obj_id, angle_idx in sample_keys:
        rec = manifest_rows[(obj_id, angle_idx)]
        paths = rec.get("paths")
        if not isinstance(paths, dict):
            raise ValueError(f"{obj_id} angle_{angle_idx}: missing paths dict")
        view_indices = [int(v) for v in rec.get("view_indices", [])]
        if len(view_indices) != 4:
            raise ValueError(f"{obj_id} angle_{angle_idx}: expected 4 view_indices, got {view_indices}")
        if len(set(view_indices)) != 4:
            raise ValueError(f"{obj_id} angle_{angle_idx}: duplicate view_indices={view_indices}")
        token_path = data_root / recon_subdir / tokens_subdir / obj_id / f"angle_{angle_idx}" / "tokens.npz"
        surface_path = rooted(data_root, paths.get(
            "overall_surface",
            f"{recon_subdir}/voxel_expanded/{obj_id}/angle_{angle_idx}/64/surface.npy",
        ))
        latent_path = rooted(data_root, paths.get(
            "overall_latent",
            f"{recon_subdir}/{latent_subdir}/{obj_id}/angle_{angle_idx}/latent.npz",
        ))
        camera_path = data_root / renders_subdir / obj_id / f"angle_{angle_idx}" / "camera_transforms.json"
        for path, label in (
            (token_path, "official DINO token cache"),
            (surface_path, "GT overall surface"),
            (latent_path, "GT overall latent"),
            (camera_path, "camera transforms"),
        ):
            require_file(path, f"{obj_id} angle_{angle_idx} {label}")
        samples.append({
            "object_id": obj_id,
            "angle_idx": angle_idx,
            "manifest_line": int(rec["_manifest_line"]),
            "category": rec.get("category"),
            "name": rec.get("name"),
            "target_part_count": int(rec.get("target_part_count", 0)),
            "view_indices": view_indices,
            "token_path": token_path,
            "surface_path": surface_path,
            "latent_path": latent_path,
            "camera_path": camera_path,
        })
    return cfg, samples


def load_tokens(path: Path, view_indices: list[int], device: torch.device) -> torch.Tensor:
    require_file(path, "official DINO token cache")
    with np.load(path) as data:
        if "tokens" not in data.files:
            raise KeyError(f"{path}: expected key 'tokens', found {data.files}")
        arr = np.asarray(data["tokens"])
    expected_all = (12, 1374, 1024)
    if tuple(arr.shape) != expected_all:
        raise ValueError(f"{path}: expected full token shape {expected_all}, got {tuple(arr.shape)}")
    if min(view_indices) < 0 or max(view_indices) >= arr.shape[0]:
        raise ValueError(f"{path}: cannot select view_indices={view_indices} from V={arr.shape[0]}")
    selected = torch.from_numpy(np.ascontiguousarray(arr[view_indices])).float().to(device)
    expected = (4, 1374, 1024)
    if tuple(selected.shape) != expected:
        raise ValueError(f"{path}: selected token shape {tuple(selected.shape)} != {expected}")
    if not torch.isfinite(selected).all():
        raise RuntimeError(f"{path}: selected tokens contain NaN/Inf")
    return selected


def instantiate_ss_flow(ckpt_path: Path, model_config: Path, device: torch.device) -> torch.nn.Module:
    ckpt_path = require_file(ckpt_path, "SS flow checkpoint")
    config = load_json(model_config)
    args = dict(config.get("args", config.get("models", {}).get("denoiser", {}).get("args", {})))
    if not args:
        raise KeyError(f"{model_config}: missing model args")
    args["use_camera_pose"] = False
    model = SparseStructureFlowModel(**args).to(device).eval()
    if ckpt_path.suffix == ".safetensors":
        state = load_file(str(ckpt_path))
    elif ckpt_path.suffix == ".pt":
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"unsupported checkpoint suffix: {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"SS flow checkpoint load failed: missing={missing[:20]} unexpected={unexpected[:20]}")
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[load] SS flow ckpt={ckpt_path} model_config={model_config}", flush=True)
    return model


def instantiate_ss_decoder(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt_path = require_file(ckpt_path, "SS decoder checkpoint")
    config_path = ckpt_path.with_suffix(".json")
    config = load_json(config_path)
    if config.get("name") != "SparseStructureDecoder":
        raise ValueError(f"{config_path}: expected SparseStructureDecoder, got {config.get('name')}")
    decoder = SparseStructureDecoder(**dict(config["args"])).to(device).eval()
    decoder.load_state_dict(load_file(str(ckpt_path)), strict=True)
    for param in decoder.parameters():
        param.requires_grad_(False)
    print(f"[load] SS decoder ckpt={ckpt_path} config={config_path}", flush=True)
    return decoder


@torch.no_grad()
def sample_multiflow(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    seed: int,
    steps: int,
    cfg_strength: float,
    sigma_min: float,
) -> torch.Tensor:
    if tuple(tokens.shape) != (4, 1374, 1024):
        raise ValueError(f"multiflow tokens expected [4,1374,1024], got {tuple(tokens.shape)}")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    sample = torch.randn(
        1,
        model.in_channels,
        model.resolution,
        model.resolution,
        model.resolution,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    neg_cond = torch.zeros(1, 1374, 1024, device=device, dtype=tokens.dtype)
    t_seq = np.linspace(1, 0, int(steps) + 1)
    for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
        t_model = torch.tensor([1000.0 * float(t)], device=device, dtype=torch.float32)
        preds = [
            model(sample, t_model, tokens[i:i + 1])
            for i in range(tokens.shape[0])
        ]
        pred = torch.stack(preds, dim=0).mean(dim=0)
        neg_pred = model(sample, t_model, neg_cond)
        pred_v = (1.0 + float(cfg_strength)) * pred - float(cfg_strength) * neg_pred
        sample = sample - (float(t) - float(t_prev)) * pred_v
    latent = sample[0].detach().float().cpu()
    if tuple(latent.shape) != (8, 16, 16, 16):
        raise ValueError(f"sampled latent shape {tuple(latent.shape)} != (8,16,16,16)")
    if not torch.isfinite(latent).all():
        raise RuntimeError("sampled latent contains NaN/Inf")
    return latent


@torch.no_grad()
def decode_latent_to_coords(
    decoder: torch.nn.Module,
    latent: torch.Tensor,
    *,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if tuple(latent.shape) != (8, 16, 16, 16):
        raise ValueError(f"latent expected shape (8,16,16,16), got {tuple(latent.shape)}")
    device = next(decoder.parameters()).device
    logits = decoder(latent.unsqueeze(0).to(device=device))[0, 0].detach().float().cpu()
    coords = torch.nonzero(logits > float(threshold), as_tuple=False).long().numpy()
    flat = logits.reshape(-1)
    stats = {
        "voxels": int(coords.shape[0]),
        "logit_min": float(flat.min().item()),
        "logit_max": float(flat.max().item()),
        "logit_mean": float(flat.mean().item()),
        "threshold": float(threshold),
    }
    return np.ascontiguousarray(coords.astype(np.int64, copy=False)), stats


def load_gt_surface(path: Path) -> np.ndarray:
    coords = np.load(require_file(path, "GT surface"))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: expected GT coords [N,3], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: GT coords are empty")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: GT coords must be integer, got {coords.dtype}")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise ValueError(f"{path}: GT coords out of [0,64), min={coords.min()} max={coords.max()}")
    return np.ascontiguousarray(coords.astype(np.int64, copy=False))


def load_gt_latent(path: Path) -> torch.Tensor:
    with np.load(require_file(path, "GT latent")) as data:
        if "mean" not in data.files:
            raise KeyError(f"{path}: expected key 'mean', found {data.files}")
        latent = torch.from_numpy(np.asarray(data["mean"])).float()
    if tuple(latent.shape) != (8, 16, 16, 16):
        raise ValueError(f"{path}: expected latent shape (8,16,16,16), got {tuple(latent.shape)}")
    return latent


def coord_keys(coords: np.ndarray) -> set[int]:
    return {
        int(x) * 4096 + int(y) * 64 + int(z)
        for x, y, z in coords.astype(np.int64, copy=False)
    }


def coords_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_keys = coord_keys(pred)
    gt_keys = coord_keys(gt)
    inter = len(pred_keys & gt_keys)
    union = len(pred_keys | gt_keys)
    return {
        "iou": float(inter / union) if union > 0 else 1.0,
        "precision": float(inter / len(pred_keys)) if pred_keys else 0.0,
        "recall": float(inter / len(gt_keys)) if gt_keys else 0.0,
        "intersection": float(inter),
        "pred_voxels": float(len(pred_keys)),
        "gt_voxels": float(len(gt_keys)),
    }


def render_voxel_open3d(
    coords: np.ndarray,
    *,
    color: tuple[float, float, float],
    resolution: int,
) -> Image.Image:
    import open3d as o3d

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"render coords expected [N,3], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError("cannot render empty voxel coords")
    points = coords.astype(np.float64) / 63.0 - 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (points.shape[0], 1)))
    renderer = o3d.visualization.rendering.OffscreenRenderer(int(resolution), int(resolution))
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = max(2.0, float(resolution) / 180.0)
    scene.add_geometry("voxels", pcd, material)
    scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        (0.35, -0.45, -0.82),
    )
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = float(max(bounds.get_extent()))
    if extent <= 0:
        extent = 1.0
    distance = extent * 2.4
    az = math.radians(315.0)
    el = math.radians(24.0)
    eye = center + np.array([
        distance * math.cos(el) * math.cos(az),
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
    ])
    scene.camera.look_at(center, eye, np.array([0.0, 0.0, 1.0]))
    scene.camera.set_projection(
        35.0,
        1.0,
        max(0.001, distance - extent * 1.8),
        distance + extent * 1.8,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )
    image = renderer.render_to_image()
    return Image.fromarray(np.asarray(image)).convert("RGB")


def make_panel(
    columns: list[tuple[str, Image.Image, dict[str, Any]]],
    out_path: Path,
    *,
    title: str,
) -> None:
    if not columns:
        raise ValueError("no columns for panel")
    width, height = columns[0][1].size
    label_h = 58
    title_h = 32
    panel = Image.new("RGB", (width * len(columns), height + label_h + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), title, fill=(0, 0, 0))
    for idx, (label, image, stats) in enumerate(columns):
        x = idx * width
        text = label
        if "voxels" in stats:
            text += f" vox={int(stats['voxels'])}"
        if "iou" in stats:
            text += f" IoU={float(stats['iou']):.3f}"
        if "precision" in stats and "recall" in stats:
            text += f" P={float(stats['precision']):.3f} R={float(stats['recall']):.3f}"
        draw.text((x + 8, title_h + 8), text, fill=(0, 0, 0))
        panel.paste(image, (x, title_h + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def run_sample(
    sample: dict[str, Any],
    *,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    obj_id = str(sample["object_id"])
    angle_idx = int(sample["angle_idx"])
    sample_dir = args.out_dir / f"{obj_id}_angle_{angle_idx:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    tokens = load_tokens(sample["token_path"], sample["view_indices"], device)
    pred_latent = sample_multiflow(
        model,
        tokens,
        seed=int(args.seed),
        steps=int(args.steps),
        cfg_strength=float(args.cfg_strength),
        sigma_min=float(args.sigma_min),
    )
    pred_coords, pred_stats = decode_latent_to_coords(decoder, pred_latent, threshold=float(args.decode_threshold))
    if pred_coords.shape[0] == 0:
        raise RuntimeError(f"{obj_id} angle_{angle_idx}: predicted coords are empty")
    gt_surface = load_gt_surface(sample["surface_path"])
    gt_latent = load_gt_latent(sample["latent_path"])
    gt_decoded_coords, gt_decoded_stats = decode_latent_to_coords(
        decoder,
        gt_latent,
        threshold=float(args.decode_threshold),
    )
    if gt_decoded_coords.shape[0] == 0:
        raise RuntimeError(f"{obj_id} angle_{angle_idx}: decoded GT latent coords are empty")
    surface_metrics = coords_metrics(pred_coords, gt_surface)
    decoded_metrics = coords_metrics(pred_coords, gt_decoded_coords)
    ceiling_metrics = coords_metrics(gt_decoded_coords, gt_surface)
    np.save(sample_dir / "gt_surface_coords.npy", gt_surface)
    np.save(sample_dir / "gt_latent_decoded_coords.npy", gt_decoded_coords)
    np.save(sample_dir / "pred_multiflow_coords.npy", pred_coords)
    torch.save(pred_latent, sample_dir / "pred_multiflow_latent.pt")
    panel_path = sample_dir / "pred_vs_gt_open3d.png"
    if not bool(args.skip_open3d_render):
        gt_img = render_voxel_open3d(gt_surface, color=(0.10, 0.36, 0.72), resolution=int(args.render_resolution))
        pred_img = render_voxel_open3d(pred_coords, color=(0.82, 0.22, 0.18), resolution=int(args.render_resolution))
        make_panel(
            [
                ("GT surface", gt_img, {"voxels": int(gt_surface.shape[0])}),
                ("pred multiflow", pred_img, {
                    "voxels": int(pred_coords.shape[0]),
                    "iou": surface_metrics["iou"],
                    "precision": surface_metrics["precision"],
                    "recall": surface_metrics["recall"],
                }),
            ],
            panel_path,
            title=f"{obj_id} angle_{angle_idx} views={sample['view_indices']} seed={args.seed}",
        )
    summary = {
        "object_id": obj_id,
        "angle_idx": angle_idx,
        "manifest_line": int(sample["manifest_line"]),
        "category": sample.get("category"),
        "name": sample.get("name"),
        "target_part_count": int(sample.get("target_part_count", 0)),
        "view_indices": [int(v) for v in sample["view_indices"]],
        "token_path": str(sample["token_path"].resolve()),
        "surface_path": str(sample["surface_path"].resolve()),
        "latent_path": str(sample["latent_path"].resolve()),
        "camera_path": str(sample["camera_path"].resolve()),
        "tokens_shape": list(tokens.shape),
        "pred_stats": pred_stats,
        "gt_latent_decoded_stats": gt_decoded_stats,
        "metrics_vs_gt_surface": surface_metrics,
        "metrics_vs_gt_latent_decoded": decoded_metrics,
        "gt_latent_decoded_vs_gt_surface": ceiling_metrics,
        "pred_coords_path": str((sample_dir / "pred_multiflow_coords.npy").resolve()),
        "pred_latent_path": str((sample_dir / "pred_multiflow_latent.pt").resolve()),
        "render_path": str(panel_path.resolve()) if panel_path.is_file() else None,
    }
    (sample_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[sample] {obj_id} angle_{angle_idx} views={sample['view_indices']} "
        f"tokens={tuple(tokens.shape)} IoU={surface_metrics['iou']:.4f} "
        f"P={surface_metrics['precision']:.4f} R={surface_metrics['recall']:.4f} "
        f"pred_vox={int(surface_metrics['pred_voxels'])} "
        f"panel={panel_path if panel_path.is_file() else 'skipped'}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ss-flow-ckpt", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--ss-decoder-ckpt", type=Path, default=DEFAULT_DECODER_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg-strength", type=float, default=7.5)
    parser.add_argument("--sigma-min", type=float, default=1.0e-5)
    parser.add_argument("--decode-threshold", type=float, default=0.0)
    parser.add_argument("--render-resolution", type=int, default=560)
    parser.add_argument("--skip-open3d-render", action="store_true")
    parser.add_argument("--ema-rate", default="0.9999")
    parser.add_argument(
        "--official-baseline",
        action="store_true",
        help="Use the official safetensors checkpoint instead of discovering latest EMA under training.output_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = require_file(args.config, "eval config")
    args.model_config = require_file(args.model_config, "SS flow model config")
    args.ss_decoder_ckpt = require_file(args.ss_decoder_ckpt, "SS decoder checkpoint")
    cfg, samples = load_samples_from_config(args.config)
    output_dir = Path(cfg["training"]["output_dir"]).resolve()
    if args.official_baseline:
        args.ss_flow_ckpt = require_file(DEFAULT_OFFICIAL_CKPT, "official SS flow checkpoint")
    elif args.ss_flow_ckpt is None:
        args.ss_flow_ckpt = latest_ema_ckpt(output_dir, str(args.ema_rate))
    else:
        args.ss_flow_ckpt = require_file(args.ss_flow_ckpt, "SS flow checkpoint")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError(f"CUDA requested ({args.device}) but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(int(args.seed))
    torch.manual_seed(int(args.seed))
    print(f"[select] {len(samples)} samples from {args.config}", flush=True)
    for sample in samples:
        print(
            f"[select] {sample['object_id']} angle_{sample['angle_idx']} "
            f"parts={sample['target_part_count']} views={sample['view_indices']} "
            f"line={sample['manifest_line']}",
            flush=True,
        )
    model = instantiate_ss_flow(args.ss_flow_ckpt, args.model_config, device)
    decoder = instantiate_ss_decoder(args.ss_decoder_ckpt, device)
    summaries = [
        run_sample(sample, model=model, decoder=decoder, device=device, args=args)
        for sample in samples
    ]
    ious = [s["metrics_vs_gt_surface"]["iou"] for s in summaries]
    ps = [s["metrics_vs_gt_surface"]["precision"] for s in summaries]
    rs = [s["metrics_vs_gt_surface"]["recall"] for s in summaries]
    pred_voxels = [s["metrics_vs_gt_surface"]["pred_voxels"] for s in summaries]
    gt_voxels = [s["metrics_vs_gt_surface"]["gt_voxels"] for s in summaries]
    summary = {
        "task": "official_single_view_finetune_multiflow_eval",
        "config": str(args.config.resolve()),
        "ss_flow_ckpt": str(args.ss_flow_ckpt.resolve()),
        "model_config": str(args.model_config.resolve()),
        "ss_decoder_ckpt": str(args.ss_decoder_ckpt.resolve()),
        "sampler": "per-step average of 4 single-view velocities, then CFG",
        "condition": "cached official DINO x_prenorm + F.layer_norm, selected by manifest view_indices",
        "expected_tokens_shape": [4, 1374, 1024],
        "seed": int(args.seed),
        "steps": int(args.steps),
        "cfg_strength": float(args.cfg_strength),
        "decode_threshold": float(args.decode_threshold),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "mean_precision": float(np.mean(ps)),
        "mean_recall": float(np.mean(rs)),
        "mean_pred_voxels": float(np.mean(pred_voxels)),
        "mean_gt_voxels": float(np.mean(gt_voxels)),
        "samples": summaries,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[done] summary={summary_path} mean_iou={summary['mean_iou']:.4f} "
        f"median_iou={summary['median_iou']:.4f} mean_P={summary['mean_precision']:.4f} "
        f"mean_R={summary['mean_recall']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
