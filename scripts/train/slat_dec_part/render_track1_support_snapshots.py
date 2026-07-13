#!/usr/bin/env python3
"""CPU support-projection snapshots for Track1 decoder gates.

The nvdiffrast preview path can crash native code on very large FlexiCubes
outputs.  This diagnostic renders occupancy support instead: GT component mask
voxels versus predicted mesh surface voxels, projected along XY/XZ/YZ.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT), str(Path(__file__).resolve().parent)):
    if item not in sys.path:
        sys.path.insert(0, item)

from render_track1_snapshots import load_decoder_from_snapshot, subset_collate  # noqa: E402
from track1_online_render import DEFAULT_CACHE_MANIFEST, DEFAULT_DECODER_CKPT, PartMaskedOnlineRenderDataset  # noqa: E402
from trellis.modules.sparse import SparseTensor  # noqa: E402


def _surface_coords64(rep: Any, *, resolution: int = 64) -> np.ndarray:
    if not bool(getattr(rep, "success", False)):
        return np.empty((0, 3), dtype=np.int32)
    vertices = rep.vertices.detach().float().cpu()
    faces = rep.faces.detach().long().cpu()
    if vertices.numel() == 0 or faces.numel() == 0:
        return np.empty((0, 3), dtype=np.int32)
    centroids = vertices[faces].mean(dim=1)
    pts = torch.cat([vertices, centroids], dim=0)
    q = torch.floor((pts + 0.5) * float(resolution)).long().clamp(0, int(resolution) - 1)
    return torch.unique(q, dim=0).numpy().astype(np.int32, copy=False)


def _mask_coords_from_batch(batch: dict[str, Any]) -> np.ndarray:
    coords = batch["coords"].detach().cpu().numpy()[:, 1:4].astype(np.int32, copy=False)
    mask = batch["feats"].detach().cpu().numpy()[:, -1] > 0.5
    return coords[mask]


def _projection(coords: np.ndarray, axes: tuple[int, int], *, resolution: int = 64) -> Image.Image:
    canvas = np.zeros((resolution, resolution), dtype=np.uint8)
    if coords.size:
        xy = np.asarray(coords[:, axes], dtype=np.int64)
        xy = np.clip(xy, 0, resolution - 1)
        canvas[resolution - 1 - xy[:, 1], xy[:, 0]] = 255
    return Image.fromarray(canvas, mode="L").convert("RGB").resize((160, 160), Image.Resampling.NEAREST)


def _tile(img: Image.Image, title: str) -> Image.Image:
    out = Image.new("RGB", (160, 188), (255, 255, 255))
    out.paste(img, (0, 28))
    draw = ImageDraw.Draw(out)
    draw.text((6, 6), title[:28], fill=(0, 0, 0))
    return out


def _panel(gt_coords: np.ndarray, pred_coords: np.ndarray, title: str, out_path: Path) -> None:
    axes = [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]
    tiles: list[Image.Image] = []
    for name, axis_pair in axes:
        tiles.append(_tile(_projection(gt_coords, axis_pair), f"GT {name}"))
        tiles.append(_tile(_projection(pred_coords, axis_pair), f"Pred {name}"))
    canvas = Image.new("RGB", (160 * 6, 188 + 30), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * 160, 30))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[2, 3, 14, 16, 20, 6])
    parser.add_argument("--mask-profile", choices=["gt", "front_only"], default="gt")
    parser.add_argument("--latent-input-mode", choices=["whole", "expanded_subset"], default="whole")
    parser.add_argument("--subset-dilation", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    device = torch.device(f"cuda:{int(args.gpu)}")
    torch.cuda.set_device(device)
    degrade_prob = 1.0 if args.mask_profile == "front_only" else 0.0
    dataset = PartMaskedOnlineRenderDataset(
        DEFAULT_CACHE_MANIFEST,
        resolution=128,
        include_body=True,
        normalize_gt_mesh=True,
        mask_degrade_prob=degrade_prob,
        front_only_prob=1.0,
        latent_input_mode=str(args.latent_input_mode),
        subset_dilation=int(args.subset_dilation),
    )
    snapshot = args.ckpt_dir / f"step_{int(args.step):07d}.pt"
    decoder = load_decoder_from_snapshot(DEFAULT_DECODER_CKPT, snapshot, device=device)
    rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for sample_index in [int(x) for x in args.sample_indices]:
        batch = subset_collate(dataset, [sample_index])
        meta = batch["sample_meta"][0]
        latents = SparseTensor(
            coords=batch["coords"].to(device=device, dtype=torch.int32),
            feats=batch["feats"].to(device=device, dtype=torch.float32),
        )
        with torch.no_grad():
            rep = decoder(latents)[0]
        gt_coords = _mask_coords_from_batch(batch)
        pred_coords = _surface_coords64(rep)
        panel = args.out_dir / str(args.mask_profile) / f"{int(args.step):07d}_{sample_index:03d}_{meta['tag']}_{meta['component_name']}.png"
        _panel(gt_coords, pred_coords, f"{meta['tag']} {meta['component_name']} {meta['mask_mode']}", panel)
        panel_paths.append(panel)
        rows.append(
            {
                "step": int(args.step),
                "sample_index": sample_index,
                "profile": str(args.mask_profile),
                "tag": meta["tag"],
                "component": meta["component_name"],
                "mask_mode": meta["mask_mode"],
                "mask_voxels": int(gt_coords.shape[0]),
                "slat_voxels": int(meta["slat_voxels"]),
                "subset_matched_slat_voxels": int(meta.get("subset_matched_slat_voxels", meta["slat_voxels"])),
                "pred_surface_voxels64": int(pred_coords.shape[0]),
                "pred_surface_over_mask": float(pred_coords.shape[0] / max(1, gt_coords.shape[0])),
                "pred_surface_over_subset": float(pred_coords.shape[0] / max(1, int(meta.get("subset_matched_slat_voxels", meta["slat_voxels"])))),
                "panel": str(panel.resolve()),
            }
        )
    csv_path = args.out_dir / str(args.mask_profile) / "support_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / str(args.mask_profile) / "support_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "panels": [str(p) for p in panel_paths]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
