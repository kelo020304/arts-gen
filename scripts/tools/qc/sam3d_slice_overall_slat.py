#!/usr/bin/env python3
"""Decode SAM3D whole-object SLat slices on GT part voxels.

This diagnostic differs from the current web SLat path. The web path samples
one SLat per component. This script samples one whole-object SLat on GT
surface coords, then slices the resulting SparseTensor by GT part coords and
decodes each slice with the same SAM3D decoders.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
GLUE = REPO / "submodules" / "sam3d-stage" / "infer_glue"
sys.path.insert(0, str(GLUE))

import slat_stage  # noqa: E402


def _load_coords(path: Path) -> np.ndarray:
    coords = np.load(path)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: expected coords shape (N,3), got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: empty coords")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise ValueError(f"{path}: coords out of [0,63], min={coords.min()} max={coords.max()}")
    return np.ascontiguousarray(coords.astype(np.int32, copy=False))


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., -1] if arr.shape[-1] in (2, 4) else arr.max(axis=-1)
    return arr > 127


def _coord_keys_np(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    return coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]


def _sparse_slice_by_coords(slat, part_coords_np: np.ndarray):
    import torch
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    slat_coords = slat.coords
    slat_feats = slat.feats
    keys = (
        slat_coords[:, 1].to(torch.int64) * 4096
        + slat_coords[:, 2].to(torch.int64) * 64
        + slat_coords[:, 3].to(torch.int64)
    )
    wanted_np = _coord_keys_np(part_coords_np)
    wanted = torch.as_tensor(wanted_np, dtype=keys.dtype, device=keys.device)
    keep = torch.isin(keys, wanted)
    kept = int(keep.sum().item())
    if kept == 0:
        raise ValueError("part coords have zero overlap with whole SLat coords")
    return sp.SparseTensor(coords=slat_coords[keep].contiguous(), feats=slat_feats[keep].contiguous())


def _save_sparse_npz(path: Path, sparse_tensor) -> None:
    coords = sparse_tensor.coords.detach().cpu().numpy().astype(np.int32, copy=False)
    feats = sparse_tensor.feats.detach().float().cpu().numpy().astype(np.float32, copy=False)
    np.savez_compressed(path, coords=coords, feats=feats)


def _load_sparse_npz(path: Path, device):
    import torch
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    with np.load(path) as data:
        coords = data["coords"].astype(np.int32, copy=False)
        feats = data["feats"].astype(np.float32, copy=False)
    return sp.SparseTensor(
        coords=torch.from_numpy(coords).to(device),
        feats=torch.from_numpy(feats).to(device),
    )


def _part_stem(path: Path) -> str:
    stem = path.stem
    if stem.startswith("ind_"):
        stem = stem[len("ind_") :]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel-dir", type=Path, required=True,
                        help="GT voxel dir containing surface.npy and ind_*.npy")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=["gaussian", "mesh"],
                        default=["gaussian", "mesh"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-overall", action="store_true",
                        help="Also decode the unsliced whole SLat as overall.glb/.ply")
    parser.add_argument("--save-latents", action="store_true",
                        help="Write whole_slat.npz and slice_<part>.npz with coords/feats.")
    parser.add_argument("--reuse-whole-slat", type=Path, default=None,
                        help="Load whole_slat.npz instead of sampling a new whole-object SLat.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault("SPCONV_ALGO", "native")

    surface_path = args.voxel_dir / "surface.npy"
    if not surface_path.is_file():
        raise FileNotFoundError(f"surface.npy not found: {surface_path}")
    part_paths = sorted(args.voxel_dir.glob("ind_*.npy"))
    if not part_paths:
        raise FileNotFoundError(f"no ind_*.npy found in {args.voxel_dir}")
    for path in (args.image, args.mask, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    surface = _load_coords(surface_path)
    image = _load_rgb(args.image)
    mask = _load_mask(args.mask)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image {image.shape[:2]}")
    rgba = slat_stage._merge_mask_to_rgba(image, mask)
    save_fn = slat_stage._resolve_save_fn()

    want_mesh = "mesh" in args.formats
    pipe = slat_stage._build_pipeline(args.config, args.device, load_mesh_decoder=want_mesh)
    device = torch.device(args.device)
    summary: dict[str, object] = {
        "voxel_dir": str(args.voxel_dir),
        "image": str(args.image),
        "mask": str(args.mask),
        "surface_voxels": int(surface.shape[0]),
        "parts": [],
    }
    try:
        with device:
            slat_input = pipe.preprocess_image(rgba, pipe.slat_preprocessor)
            if args.reuse_whole_slat is not None:
                whole_slat = _load_sparse_npz(args.reuse_whole_slat, device)
            else:
                whole_coords = slat_stage._coords_np_to_torch(surface).to(device)
                torch.manual_seed(int(args.seed))
                whole_slat = pipe.sample_slat(
                    slat_input,
                    whole_coords,
                    inference_steps=None,
                    use_distillation=False,
                )
                if args.save_latents:
                    _save_sparse_npz(args.out / "whole_slat.npz", whole_slat)

            if args.include_overall:
                outputs = pipe.decode_slat(whole_slat, list(args.formats))
                save_fn(
                    {
                        "gaussian": outputs["gaussian"][0] if "gaussian" in outputs else None,
                        "mesh": outputs["mesh"][0] if "mesh" in outputs else None,
                    },
                    args.out,
                    mesh_name="overall.glb",
                    gaussian_name="overall.ply",
                )

            for part_path in part_paths:
                stem = _part_stem(part_path)
                part = _load_coords(part_path)
                sliced = _sparse_slice_by_coords(whole_slat, part).to(device)
                if args.save_latents:
                    _save_sparse_npz(args.out / f"slice_{stem}.npz", sliced)
                outputs = pipe.decode_slat(sliced, list(args.formats))
                save_fn(
                    {
                        "gaussian": outputs["gaussian"][0] if "gaussian" in outputs else None,
                        "mesh": outputs["mesh"][0] if "mesh" in outputs else None,
                    },
                    args.out,
                    mesh_name=f"{stem}.glb",
                    gaussian_name=f"{stem}.ply",
                )
                summary["parts"].append(
                    {
                        "stem": stem,
                        "source": str(part_path),
                        "part_voxels": int(part.shape[0]),
                        "sliced_voxels": int(sliced.coords.shape[0]),
                    }
                )
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
