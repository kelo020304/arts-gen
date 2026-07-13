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
        arr = arr[..., -1] if arr.shape[-1] in (2, 4) else arr.max(axis=-1)
    if arr.dtype == bool:
        return arr
    return arr > 127


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="texture",
        description="Run SAM 3D Objects Stage B: voxel coords + image+mask -> Gaussian splat + mesh.",
    )
    parser.add_argument(
        "--voxel-dir",
        type=Path,
        required=True,
        help="Directory containing surface.npy + pose.json (Stage A output).",
    )
    parser.add_argument("--image", type=Path, required=True, help="Same RGB image used in Stage A.")
    parser.add_argument("--mask", type=Path, required=True, help="Same binary mask used in Stage A.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="Output directory (created)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./checkpoints/hf/pipeline.yaml"),
        help="Path to pipeline.yaml (default: ./checkpoints/hf/pipeline.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed (default: read from pose.json, +1 applied internally).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["gaussian", "mesh", "gaussian_4"],
        default=["gaussian", "mesh"],
        help="Decode formats (default: gaussian mesh).",
    )
    parser.add_argument(
        "--with-layout-postprocess",
        action="store_true",
        help="Run layout post-optimization (requires pointmap; usually run from Stage A).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.voxel_dir.is_dir():
        raise FileNotFoundError(f"--voxel-dir not a directory: {args.voxel_dir}")
    if not (args.voxel_dir / "surface.npy").exists():
        raise FileNotFoundError(f"surface.npy missing in {args.voxel_dir}")
    if not (args.voxel_dir / "pose.json").exists():
        raise FileNotFoundError(f"pose.json missing in {args.voxel_dir}")
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
    from texture.pipeline import TexturePipeline

    formats = tuple(args.formats)
    pipeline = TexturePipeline(
        config_path=args.config,
        device=args.device,
        load_mesh_decoder="mesh" in formats,
        load_gs4_decoder="gaussian_4" in formats,
    )
    try:
        out = pipeline(
            args.voxel_dir,
            image,
            mask,
            seed=args.seed,
            formats=formats,
            with_layout_postprocess=args.with_layout_postprocess,
        )
        out.save(args.output_dir, save_mesh="mesh" in formats)

        msgs = []
        if out.ply_path is not None:
            msgs.append(f"splat ({out.num_gaussians} gaussians) -> {out.ply_path}")
        if out.glb_path is not None:
            msgs.append(f"mesh -> {out.glb_path}")
        print("wrote: " + ", ".join(msgs) if msgs else "no outputs produced")
    finally:
        pipeline.unload()


if __name__ == "__main__":
    main()
