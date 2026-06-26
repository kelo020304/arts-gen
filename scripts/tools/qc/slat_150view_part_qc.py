#!/usr/bin/env python3
"""Compare 12-view vs 150-view projected DINO features on fixed part voxels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
TOOLKIT_PIPELINE = PROJECT_ROOT / "submodules" / "dataset_toolkits" / "pipeline"
TOOLKIT_UTILS = PROJECT_ROOT / "submodules" / "dataset_toolkits" / "utils"
for item in (str(TRELLIS_ROOT), str(TOOLKIT_PIPELINE), str(TOOLKIT_UTILS)):
    if item not in sys.path:
        sys.path.insert(0, item)

import scripts.tools.lib.slat_vae_roundtrip as rt
from importlib import util as importlib_util


STEP05_PATH = TOOLKIT_PIPELINE / "05_extract_feature.py"
spec = importlib_util.spec_from_file_location("dataset_toolkits_step05_feature", STEP05_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load Step05 feature module spec: {STEP05_PATH}")
step05 = importlib_util.module_from_spec(spec)
spec.loader.exec_module(step05)


DEFAULT_PARTS = ("lid_0", "button_(top_handle)_0")


class FeatureOnlyConfig:
    def __init__(self, model: str, dinov2_repo: str, torch_hub_dir: str) -> None:
        self.feature = type(
            "FeatureConfig",
            (),
            {
                "model": model,
                "dinov2_repo": dinov2_repo,
                "torch_hub_dir": torch_hub_dir,
            },
        )()


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


def load_feature_config(config_path: Path) -> FeatureOnlyConfig:
    payload = yaml.safe_load(require_file(config_path, "dataset config").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"config must be a mapping: {config_path}")
    feature = payload.get("feature")
    if not isinstance(feature, dict):
        raise RuntimeError(f"config missing mapping key 'feature': {config_path}")
    required = {}
    for key in ("model", "dinov2_repo", "torch_hub_dir"):
        value = feature.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"config feature.{key} must be a non-empty string: {config_path}")
        required[key] = value
    require_dir(Path(required["dinov2_repo"]), "DINOv2 repo")
    require_dir(Path(required["torch_hub_dir"]), "torch hub dir")
    print(
        f"[dino-config] model={required['model']} repo={required['dinov2_repo']} "
        f"torch_hub={required['torch_hub_dir']}",
        flush=True,
    )
    return FeatureOnlyConfig(**required)


def load_camera_matrices(camera_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    return rt.load_camera_matrices(camera_path)


def load_camera_matrices_any(camera_path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    import utils3d

    payload = json.loads(require_file(camera_path, "camera_transforms").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"camera_transforms.frames must be a non-empty list: {camera_path}")
    expected_total = payload.get("total_views")
    if expected_total is not None and int(expected_total) != len(frames):
        raise RuntimeError(f"{camera_path} total_views={expected_total} but frames={len(frames)}")

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
        "total_views": len(frames),
        "view_sampler": payload.get("view_sampler"),
        "render_engine": payload.get("render_engine"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "camera_center_norm_range": [float(center_norm.min()), float(center_norm.max())],
    }
    return torch.stack(extrinsics), torch.stack(intrinsics), meta


def load_patchtokens_from_npz(tokens_path: Path) -> torch.Tensor:
    with np.load(require_file(tokens_path, "12-view DINO tokens")) as data:
        if set(data.files) != {"tokens"}:
            raise RuntimeError(f"{tokens_path} keys must be exactly ['tokens'], got {data.files}")
        tokens = data["tokens"]
    if tokens.ndim != 3 or tokens.shape[1:] != (1370, 1024):
        raise RuntimeError(f"{tokens_path} tokens must be [V,1370,1024], got {tokens.shape}")
    patch = torch.from_numpy(tokens[:, 1:, :].astype(np.float32, copy=False)).cuda()
    return patch.permute(0, 2, 1).reshape(tokens.shape[0], 1024, 37, 37)


def list_view_paths(rgb_dir: Path, num_views: int) -> list[Path]:
    rgb_dir = require_dir(rgb_dir, "RGB directory")
    paths = []
    for view_idx in range(num_views):
        paths.append(require_file(rgb_dir / f"view_{view_idx}.png", f"RGB view {view_idx}"))
    return paths


def extract_dino_tokens(view_paths: list[Path], cfg: FeatureOnlyConfig, device: torch.device, batch_size: int) -> torch.Tensor:
    transform = step05._build_transform()
    model = step05._load_model(cfg.feature.model, cfg.feature.dinov2_repo, cfg.feature.torch_hub_dir, device)
    patches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(view_paths), batch_size):
            batch_paths = view_paths[start : start + batch_size]
            image_tensors = [step05._load_rgba_as_white_rgb(path, transform) for path in batch_paths]
            batch = torch.stack(image_tensors, dim=0).to(device)
            features = model.forward_features(batch)
            patch_tokens = features["x_norm_patchtokens"]
            if patch_tokens.shape[1:] != (1369, 1024):
                raise RuntimeError(f"unexpected DINO patch token shape {tuple(patch_tokens.shape)}")
            patches.append(patch_tokens.detach().float())
            print(f"[dino] encoded {min(start + batch_size, len(view_paths))}/{len(view_paths)} views", flush=True)
    all_patch = torch.cat(patches, dim=0)
    if all_patch.shape[0] != len(view_paths):
        raise RuntimeError(f"DINO view count mismatch: {all_patch.shape[0]} vs {len(view_paths)}")
    return all_patch.permute(0, 2, 1).reshape(len(view_paths), 1024, 37, 37)


def make_part_slat(encoder: Any, voxel_path: Path, patchtokens: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> Any:
    coords = rt.load_voxel_coords(voxel_path)
    return rt.encode_coords_to_slat(encoder, coords, patchtokens, extrinsics, intrinsics)


def rgb_tensor(path: Path) -> torch.Tensor:
    return rt.load_rgb(path)


def masked_gt(rgb: torch.Tensor, mask: torch.Tensor, label: int) -> torch.Tensor:
    keep = (mask == int(label)).float().unsqueeze(0)
    return rgb * keep


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    return rt.tensor_to_image(tensor.detach().cpu().float().clamp(0, 1))


def make_triptych(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = tensor_to_image(rows[0]["gt"]).size
    title_h = 30
    label_h = 24
    cols = 3
    canvas = Image.new("RGB", (cols * cell_w, title_h + len(rows) * (label_h + cell_h)), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "fixed part voxels: GT render | 150-view decode | 12-view decode", fill=(255, 255, 255))
    for row_idx, row in enumerate(rows):
        images = [tensor_to_image(row["gt"]), tensor_to_image(row["decode150"]), tensor_to_image(row["decode12"])]
        labels = [
            f"{row['part']} GT mask v{row['view']}",
            f"150v decode vox={row['voxels']}",
            f"12v decode vox={row['voxels']}",
        ]
        y = title_h + row_idx * (label_h + cell_h)
        for col_idx, (image, label) in enumerate(zip(images, labels)):
            x = col_idx * cell_w
            draw.rectangle((x, y, x + cell_w, y + label_h), fill=(0, 0, 0))
            draw.text((x + 6, y + 6), label, fill=(255, 255, 255))
            canvas.paste(image, (x, y + label_h))
    canvas.save(out_path)


def psnr_vs_gt(pred: torch.Tensor, gt_rgb: torch.Tensor, mask: torch.Tensor) -> float:
    return rt.psnr(pred, gt_rgb, mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--object-id", default="100058")
    parser.add_argument("--angle-idx", type=int, default=0)
    parser.add_argument("--parts", nargs="+", default=list(DEFAULT_PARTS))
    parser.add_argument("--render150-dir", type=Path, required=True)
    parser.add_argument("--encoder-ckpt", type=Path, required=True)
    parser.add_argument("--decoder-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compare-view", type=int, default=3)
    args = parser.parse_args()

    data_root = require_dir(args.data_root, "DATA_ROOT")
    cfg = load_feature_config(args.config.resolve())
    render150_dir = require_dir(args.render150_dir, "150-view render directory")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    object_id = args.object_id
    angle_idx = args.angle_idx
    rroot12 = rt.render_root(data_root, object_id, angle_idx)
    voxel_root = require_dir(
        data_root / "reconstruction" / "voxel_expanded" / object_id / f"angle_{angle_idx}" / "64",
        "voxel root",
    )
    token12_path = rt.token_path(data_root, object_id, angle_idx)
    camera12_path = require_file(rroot12 / "camera_transforms.json", "12-view camera_transforms")
    camera150_path = require_file(render150_dir / "camera_transforms.json", "150-view camera_transforms")

    print(f"[paths] 12_tokens={token12_path}", flush=True)
    print(f"[paths] render150={render150_dir}", flush=True)
    print(f"[paths] voxel_root={voxel_root}", flush=True)

    extr12, intr12, camera12_meta = load_camera_matrices(camera12_path)
    extr150, intr150, camera150_meta = load_camera_matrices_any(camera150_path)
    patch12 = load_patchtokens_from_npz(token12_path)
    if patch12.shape[0] != 12:
        raise RuntimeError(f"expected 12-view tokens, got {patch12.shape[0]}")
    view150_paths = list_view_paths(render150_dir / "rgb", int(camera150_meta["total_views"]))
    if len(view150_paths) != 150:
        raise RuntimeError(f"expected 150 rendered views, got {len(view150_paths)}")
    patch150 = extract_dino_tokens(view150_paths, cfg, torch.device("cuda"), args.batch_size)
    if patch150.shape[0] != 150:
        raise RuntimeError(f"expected 150 DINO patch views, got {patch150.shape[0]}")

    encoder = rt.load_slat_encoder(args.encoder_ckpt)
    decoder, decoder_info = rt.load_trellis_decoder(args.decoder_ckpt)
    renderer = rt.make_renderer(camera12_meta["resolution"], 0.8, 1.6, decoder)

    labels = rt.load_part_labels(data_root, object_id)
    rows = []
    report_rows = []
    for part in args.parts:
        if part not in labels:
            raise RuntimeError(f"part {part!r} missing from part_info labels")
        voxel_path = require_file(voxel_root / f"ind_{part}.npy", f"part voxel {part}")
        coords = rt.load_voxel_coords(voxel_path)
        slat12 = make_part_slat(encoder, voxel_path, patch12, extr12, intr12)
        slat150 = make_part_slat(encoder, voxel_path, patch150, extr150, intr150)
        coords12 = slat12.coords[:, 1:].detach().cpu().numpy().astype(np.int64)
        coords150 = slat150.coords[:, 1:].detach().cpu().numpy().astype(np.int64)
        if not np.array_equal(coords12, coords150):
            raise RuntimeError(f"{part}: 12v and 150v encoded coords differ")
        if not np.array_equal(coords, coords12):
            raise RuntimeError(f"{part}: encoded coords differ from fixed voxel file")

        gaussian12 = rt.decode_gaussian(slat12, decoder, "trellis")
        gaussian150 = rt.decode_gaussian(slat150, decoder, "trellis")
        recon12 = rt.render_views(gaussian12, renderer, extr12, intr12)
        recon150 = rt.render_views(gaussian150, renderer, extr12, intr12)

        label = labels[part]
        view_idx = int(args.compare_view)
        gt_rgb = rgb_tensor(rroot12 / "rgb" / f"view_{view_idx}.png")
        mask_int = rt.load_mask(rroot12 / "mask" / f"mask_{view_idx}.npy")
        part_mask = mask_int == int(label)
        if not part_mask.any():
            visible = []
            for idx in range(12):
                m = rt.load_mask(rroot12 / "mask" / f"mask_{idx}.npy")
                if (m == int(label)).any():
                    visible.append(idx)
            if not visible:
                raise RuntimeError(f"{part}: no visible 12-view GT mask pixels")
            view_idx = visible[0]
            gt_rgb = rgb_tensor(rroot12 / "rgb" / f"view_{view_idx}.png")
            mask_int = rt.load_mask(rroot12 / "mask" / f"mask_{view_idx}.npy")
            part_mask = mask_int == int(label)
        gt_vis = masked_gt(gt_rgb, mask_int, label)
        p12 = psnr_vs_gt(recon12[view_idx], gt_rgb, part_mask)
        p150 = psnr_vs_gt(recon150[view_idx], gt_rgb, part_mask)
        rows.append(
            {
                "part": part,
                "view": view_idx,
                "voxels": int(coords.shape[0]),
                "gt": gt_vis,
                "decode150": recon150[view_idx],
                "decode12": recon12[view_idx],
            }
        )
        report_rows.append(
            {
                "object_id": object_id,
                "angle_idx": int(angle_idx),
                "part": part,
                "voxel_path": str(voxel_path.resolve()),
                "voxel_count": int(coords.shape[0]),
                "coords_identical_12v_150v": True,
                "compare_view": int(view_idx),
                "mask_pixels": int(part_mask.sum().item()),
                "psnr_150_vs_gt_partmask": float(p150),
                "psnr_12_vs_gt_partmask": float(p12),
                "feat_range_150": [float(slat150.feats.min().item()), float(slat150.feats.max().item())],
                "feat_range_12": [float(slat12.feats.min().item()), float(slat12.feats.max().item())],
            }
        )
        print(
            f"[part] {part} vox={coords.shape[0]} view={view_idx} "
            f"PSNR150={p150:.3f} PSNR12={p12:.3f}",
            flush=True,
        )

    triptych = out_dir / f"{object_id}_angle_{angle_idx}_fixed_vox_150v_vs_12v_triptych.png"
    make_triptych(triptych, rows)
    report = {
        "object_id": object_id,
        "angle_idx": int(angle_idx),
        "data_root": str(data_root),
        "render150_dir": str(render150_dir),
        "camera12_meta": camera12_meta,
        "camera150_meta": camera150_meta,
        "encoder_ckpt": str(args.encoder_ckpt.resolve()),
        "decoder_ckpt": str(args.decoder_ckpt.resolve()),
        "decoder_info": decoder_info,
        "dino": {"views_12": int(patch12.shape[0]), "views_150": int(patch150.shape[0])},
        "triptych": str(triptych.resolve()),
        "rows": report_rows,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] report={report_path}", flush=True)
    print(f"[done] triptych={triptych}", flush=True)


if __name__ == "__main__":
    main()
