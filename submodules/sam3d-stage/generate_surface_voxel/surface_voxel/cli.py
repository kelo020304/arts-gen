from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image at {path}, got shape {arr.shape}")
    return arr


def _load_mask(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        # take last channel (alpha if RGBA, else collapse via max)
        arr = arr[..., -1] if arr.shape[-1] in (2, 4) else arr.max(axis=-1)
    if arr.dtype == bool:
        return arr
    return arr > 127


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="surface-voxel",
        description="Run SAM 3D Objects Stage A: image+mask -> voxel surface coords.",
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to RGB image (PNG/JPG).")
    parser.add_argument("--mask", type=Path, required=True, help="Path to binary mask PNG.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="Output directory (created)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./checkpoints/hf/pipeline.yaml"),
        help="Path to pipeline.yaml (default: ./checkpoints/hf/pipeline.yaml).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-layout-aux",
        action="store_true",
        help="Skip writing pointmap_unnorm.npy (smaller output).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"--image not found: {args.image}")
    if not args.mask.exists():
        raise FileNotFoundError(f"--mask not found: {args.mask}")
    if not args.config.exists():
        raise FileNotFoundError(f"--config not found: {args.config}")

    image = _load_rgb(args.image)
    mask = _load_mask(args.mask)
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image HxW {image.shape[:2]}"
        )

    # Import here so --help is fast and doesn't pull in torch/sam3d_objects.
    from surface_voxel.pipeline import SurfaceVoxelPipeline

    pipeline = SurfaceVoxelPipeline(config_path=args.config, device=args.device)
    try:
        out = pipeline(
            image,
            mask,
            seed=args.seed,
            keep_layout_aux=not args.no_layout_aux,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out.save(args.output_dir)
        print(f"wrote {out.coords.shape[0]} surface voxels to {args.output_dir}")
    finally:
        pipeline.unload()


if __name__ == "__main__":
    main()
