#!/usr/bin/env python3
"""Cache official TRELLIS DINOv2 tokens for PhysX renders.

Writes a new token tree without touching the legacy
``reconstruction/dinov2_tokens`` cache.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    "/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/"
    "PhysX-Mobility-full-4view-0511"
)
MANIFEST = DATA_ROOT / "manifests/part_completion/arts_mllm_physx-mobility.train.jsonl"
DINO_CKPT = Path("/robot/data-lab/jzh/art-gen/weights/dinov2_vitl14_reg4_pretrain.pth")
OUT_SUBDIR = "dinov2_tokens_official_prenorm1374"
DINO_RESOLUTION = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_TOKENS = 1374
EXPECTED_CHANNELS = 1024


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    if image.mode != "RGBA":
        raise ValueError(f"alpha_bbox expects RGBA image, got {image.mode}")
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.nonzero(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("RGBA image has empty alpha foreground")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def preprocess_official(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        x0, y0, x1, y1 = alpha_bbox(image)
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        size = int(max(x1 - x0, y1 - y0) * 1.2)
        if size <= 0:
            raise ValueError(f"invalid foreground crop from bbox={(x0, y0, x1, y1)}")
        bbox = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )
        cropped = image.crop(bbox).resize(
            (DINO_RESOLUTION, DINO_RESOLUTION),
            Image.Resampling.LANCZOS,
        )
        rgba = np.asarray(cropped.convert("RGBA")).astype(np.float32) / 255.0
        rgb = rgba[:, :, :3] * rgba[:, :, 3:4]
        return Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")
    if image.mode != "RGB":
        if "A" in image.getbands():
            return preprocess_official(image.convert("RGBA"))
        image = image.convert("RGB")
    return image.resize((DINO_RESOLUTION, DINO_RESOLUTION), Image.Resampling.LANCZOS)


def tensorize(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])
    return torch.stack([transform(image.convert("RGB")) for image in images], dim=0).to(device)


def load_dinov2(device: torch.device, ckpt_path: Path) -> torch.nn.Module:
    require_file(ckpt_path, "DINOv2 ckpt")
    torch_home = os.environ.get("TORCH_HOME", str(PROJECT_ROOT / "pretrained/torch_hub"))
    repo_candidates = [
        PROJECT_ROOT / "pretrained/torch_hub/hub/facebookresearch_dinov2_main",
        PROJECT_ROOT / "pretrained/dinov2",
        Path(torch_home) / "hub/facebookresearch_dinov2_main",
        Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main",
    ]
    repo = next((path for path in repo_candidates if (path / "hubconf.py").is_file()), None)
    if repo is None:
        raise FileNotFoundError(
            "local DINOv2 repo not found; expected one of: "
            + ", ".join(str(path) for path in repo_candidates)
        )
    model = torch.hub.load(str(repo), "dinov2_vitl14_reg", source="local", pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[load] DINOv2 repo={repo} ckpt={ckpt_path}", flush=True)
    return model


def iter_manifest(manifest: Path) -> list[dict[str, Any]]:
    require_file(manifest, "manifest")
    samples: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            obj_id = str(rec.get("object_id", rec.get("obj_id", "")))
            if not obj_id:
                raise KeyError(f"{manifest}:{line_no}: missing object_id/obj_id")
            if "angle_idx" not in rec and "angle" not in rec:
                raise KeyError(f"{manifest}:{line_no}: missing angle_idx/angle")
            angle_idx = int(rec.get("angle_idx", rec.get("angle")))
            key = (obj_id, angle_idx)
            if key in seen:
                continue
            seen.add(key)
            samples.append({
                "object_id": obj_id,
                "angle_idx": angle_idx,
                "line": line_no,
                "view_indices": [int(v) for v in rec.get("view_indices", [])],
            })
    if not samples:
        raise ValueError(f"manifest has no samples: {manifest}")
    return samples


def load_view_images(data_root: Path, obj_id: str, angle_idx: int, num_views: int) -> list[Image.Image]:
    rgb_dir = data_root / "renders" / obj_id / f"angle_{angle_idx}" / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB render directory not found: {rgb_dir}")
    images: list[Image.Image] = []
    for view_idx in range(num_views):
        path = rgb_dir / f"view_{view_idx}.png"
        require_file(path, f"view_{view_idx} image")
        image = Image.open(path)
        if image.mode == "RGBA" or "A" in image.getbands():
            images.append(image.convert("RGBA"))
        else:
            images.append(image.convert("RGB"))
    return images


@torch.no_grad()
def encode_views(
    model: torch.nn.Module,
    images: list[Image.Image],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    encoded: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        chunk = [preprocess_official(image) for image in images[start:start + batch_size]]
        batch = tensorize(chunk, device)
        feats = model(batch, is_training=True)
        if "x_prenorm" not in feats:
            raise KeyError(f"DINOv2 output missing x_prenorm; keys={sorted(feats.keys())}")
        tokens = F.layer_norm(feats["x_prenorm"], feats["x_prenorm"].shape[-1:]).float()
        if tuple(tokens.shape[1:]) != (EXPECTED_TOKENS, EXPECTED_CHANNELS):
            raise ValueError(
                f"official DINO tokens chunk shape {tuple(tokens.shape)} has wrong tail; "
                f"expected (*,{EXPECTED_TOKENS},{EXPECTED_CHANNELS})"
            )
        if not torch.isfinite(tokens).all():
            raise RuntimeError("official DINO tokens contain NaN/Inf")
        encoded.append(tokens.detach().cpu())
    out = torch.cat(encoded, dim=0)
    if tuple(out.shape) != (len(images), EXPECTED_TOKENS, EXPECTED_CHANNELS):
        raise ValueError(
            f"official DINO tokens shape {tuple(out.shape)} != "
            f"expected {(len(images), EXPECTED_TOKENS, EXPECTED_CHANNELS)}"
        )
    return out


def atomic_save_npz(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, tokens=tokens)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dinov2-ckpt", type=Path, default=DINO_CKPT)
    parser.add_argument("--out-subdir", default=OUT_SUBDIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--object-ids",
        nargs="*",
        default=None,
        help="Optional explicit object ids to cache; uses --angle-idx for each id.",
    )
    parser.add_argument(
        "--angle-idx",
        type=int,
        default=0,
        help="Angle index used with --object-ids.",
    )
    parser.add_argument("--num-views", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--start-after-existing",
        action="store_true",
        help="Skip existing valid caches instead of requiring --overwrite.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError(f"--count must be positive, got {args.count}")
    if args.num_views <= 0:
        raise ValueError(f"--num-views must be positive, got {args.num_views}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if str(args.out_subdir).strip() in {"dinov2_tokens", "reconstruction/dinov2_tokens"}:
        raise ValueError("refusing to write into legacy dinov2_tokens directory")
    data_root = args.data_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else data_root / args.manifest
    require_file(manifest, "manifest")
    out_root = data_root / "reconstruction" / str(args.out_subdir)
    samples = iter_manifest(manifest)
    if args.object_ids:
        by_key = {
            (str(sample["object_id"]), int(sample["angle_idx"])): sample
            for sample in samples
        }
        requested = [(str(obj_id), int(args.angle_idx)) for obj_id in args.object_ids]
        missing = [key for key in requested if key not in by_key]
        if missing:
            missing_str = ", ".join(f"{obj_id}:angle_{angle_idx}" for obj_id, angle_idx in missing)
            raise RuntimeError(f"requested samples not found in manifest: {missing_str}")
        selected = [by_key[key] for key in requested]
    else:
        selected = samples[int(args.offset):int(args.offset) + int(args.count)]
        if len(selected) != args.count:
            raise RuntimeError(f"selected {len(selected)} samples, expected {args.count}")
    print(f"[select] total_manifest_samples={len(samples)} selected={len(selected)} out_root={out_root}", flush=True)
    for sample in selected:
        print(
            f"[select] {sample['object_id']} angle_{sample['angle_idx']} "
            f"manifest_views={sample['view_indices']} line={sample['line']}",
            flush=True,
        )
    if args.dry_run:
        return
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError(f"CUDA requested ({args.device}) but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = load_dinov2(device, args.dinov2_ckpt)

    total_encode_s = 0.0
    total_write_s = 0.0
    written = 0
    started = time.perf_counter()
    for idx, sample in enumerate(selected, start=1):
        obj_id = sample["object_id"]
        angle_idx = int(sample["angle_idx"])
        out_path = out_root / obj_id / f"angle_{angle_idx}" / "tokens.npz"
        if out_path.exists() and not args.overwrite:
            with np.load(out_path) as data:
                shape = tuple(data["tokens"].shape)
            if shape != (args.num_views, EXPECTED_TOKENS, EXPECTED_CHANNELS):
                raise ValueError(f"existing cache has wrong shape {shape}: {out_path}")
            if args.start_after_existing:
                print(f"[skip] {idx}/{len(selected)} existing {out_path} shape={shape}", flush=True)
                continue
            raise FileExistsError(
                f"cache already exists: {out_path}; pass --overwrite or --start-after-existing"
            )
        images = load_view_images(data_root, obj_id, angle_idx, int(args.num_views))
        t0 = time.perf_counter()
        tokens = encode_views(model, images, device=device, batch_size=int(args.batch_size))
        if tuple(tokens.shape) != (args.num_views, EXPECTED_TOKENS, EXPECTED_CHANNELS):
            raise ValueError(f"{obj_id} angle_{angle_idx}: bad token shape {tuple(tokens.shape)}")
        encode_s = time.perf_counter() - t0
        arr = np.ascontiguousarray(tokens.numpy().astype(np.float32, copy=False))
        t1 = time.perf_counter()
        atomic_save_npz(out_path, arr)
        write_s = time.perf_counter() - t1
        size_mb = out_path.stat().st_size / (1024 * 1024)
        total_encode_s += encode_s
        total_write_s += write_s
        written += 1
        print(
            f"[write] {idx}/{len(selected)} {obj_id} angle_{angle_idx} "
            f"shape={tuple(arr.shape)} size={size_mb:.1f}MiB "
            f"encode={encode_s:.2f}s write={write_s:.2f}s path={out_path}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    per_written = elapsed / max(1, written)
    estimate_all_hours = per_written * len(samples) / 3600.0
    print(
        f"[done] written={written} selected={len(selected)} elapsed={elapsed:.2f}s "
        f"avg_wall_per_written={per_written:.2f}s avg_encode={total_encode_s / max(1, written):.2f}s "
        f"avg_write={total_write_s / max(1, written):.2f}s "
        f"estimated_all_{len(samples)}={estimate_all_hours:.2f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
