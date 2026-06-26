#!/usr/bin/env python3
"""Round-trip a sparse 64^3 surface through TRELLIS SS-VAE encode -> decode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))

# Avoid executing TRELLIS-arts/trellis/__init__.py; it imports pipelines and
# optional deps such as rembg that are irrelevant for SS-VAE roundtrip.
import types  # noqa: E402

_trellis_pkg = types.ModuleType("trellis")
_trellis_pkg.__path__ = [str(TRELLIS_PATH / "trellis")]
_trellis_pkg.__package__ = "trellis"
sys.modules.setdefault("trellis", _trellis_pkg)
for _subpackage in ("models", "modules"):
    _module = types.ModuleType(f"trellis.{_subpackage}")
    _module.__path__ = [str(TRELLIS_PATH / "trellis" / _subpackage)]
    _module.__package__ = f"trellis.{_subpackage}"
    sys.modules.setdefault(f"trellis.{_subpackage}", _module)

from trellis.models.sparse_structure_vae import (  # noqa: E402
    SparseStructureDecoder,
    SparseStructureEncoder,
)


def load_coords(path: Path, resolution: int) -> np.ndarray:
    coords = np.asarray(np.load(path), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} expected [N,3], got {coords.shape}")
    if len(coords) and np.any((coords < 0) | (coords >= resolution)):
        bad = coords[np.any((coords < 0) | (coords >= resolution), axis=1)][0]
        raise ValueError(f"{path} contains out-of-range coord {bad.tolist()}")
    return np.unique(coords.reshape(-1, 3), axis=0)


def ckpt_paths(path: Path) -> tuple[Path, Path]:
    if path.suffix == ".safetensors":
        weights = path
        config = path.with_suffix(".json")
    else:
        weights = path.with_suffix(".safetensors")
        config = path.with_suffix(".json")
    if not config.is_file():
        raise FileNotFoundError(f"config not found: {config}")
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    return config, weights


def load_encoder(path: Path, device: torch.device) -> SparseStructureEncoder:
    config_path, weights_path = ckpt_paths(path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("name") != "SparseStructureEncoder":
        raise ValueError(f"{config_path} expected SparseStructureEncoder, got {cfg.get('name')!r}")
    model = SparseStructureEncoder(**cfg["args"]).to(device).eval()
    model.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_decoder(path: Path, device: torch.device) -> SparseStructureDecoder:
    config_path, weights_path = ckpt_paths(path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("name") != "SparseStructureDecoder":
        raise ValueError(f"{config_path} expected SparseStructureDecoder, got {cfg.get('name')!r}")
    model = SparseStructureDecoder(**cfg["args"]).to(device).eval()
    model.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def dense_from_coords(coords: np.ndarray, resolution: int, device: torch.device) -> torch.Tensor:
    grid = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.float32, device=device)
    if len(coords):
        idx = torch.as_tensor(coords, dtype=torch.long, device=device)
        grid[:, :, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return grid


def coord_key(coords: np.ndarray, resolution: int) -> np.ndarray:
    if len(coords) == 0:
        return np.empty((0,), dtype=np.int64)
    coords = np.asarray(coords, dtype=np.int64)
    return coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]


def key_set(coords: np.ndarray, resolution: int) -> set[int]:
    return {int(v) for v in coord_key(coords, resolution)}


def direct_outer_6dir(coords: np.ndarray) -> np.ndarray:
    coords = np.unique(np.asarray(coords, dtype=np.int64).reshape(-1, 3), axis=0)
    keep: set[tuple[int, int, int]] = set()
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        groups: dict[tuple[int, int], list[int]] = {}
        for row in coords:
            key = (int(row[other[0]]), int(row[other[1]]))
            value = int(row[axis])
            if key not in groups:
                groups[key] = [value, value]
            else:
                groups[key][0] = min(groups[key][0], value)
                groups[key][1] = max(groups[key][1], value)
        for key, (lo, hi) in groups.items():
            for value in {lo, hi}:
                row = [0, 0, 0]
                row[axis] = value
                row[other[0]] = key[0]
                row[other[1]] = key[1]
                keep.add(tuple(row))
    if not keep:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(sorted(keep), dtype=np.int64)


def set_to_coords(values: set[int], resolution: int) -> np.ndarray:
    if not values:
        return np.empty((0, 3), dtype=np.int64)
    keys = np.asarray(sorted(values), dtype=np.int64)
    x = keys // (resolution * resolution)
    rem = keys % (resolution * resolution)
    y = rem // resolution
    z = rem % resolution
    return np.stack([x, y, z], axis=1).astype(np.int64)


def metrics(gt: np.ndarray, pred: np.ndarray, resolution: int) -> dict[str, object]:
    gt_set = key_set(gt, resolution)
    pred_set = key_set(pred, resolution)
    inter = gt_set & pred_set
    union = gt_set | pred_set
    outer = direct_outer_6dir(gt)
    interior_set = gt_set - key_set(outer, resolution)
    interior_hit = interior_set & pred_set
    false_neg = gt_set - pred_set
    false_pos = pred_set - gt_set
    return {
        "gt_voxels": int(len(gt_set)),
        "pred_voxels": int(len(pred_set)),
        "intersection": int(len(inter)),
        "union": int(len(union)),
        "iou": float(len(inter) / len(union)) if union else 1.0,
        "recall": float(len(inter) / len(gt_set)) if gt_set else 1.0,
        "precision": float(len(inter) / len(pred_set)) if pred_set else 1.0,
        "false_negative": int(len(false_neg)),
        "false_positive": int(len(false_pos)),
        "outer_6dir_voxels": int(len(outer)),
        "interior_voxels": int(len(interior_set)),
        "interior_recovered": int(len(interior_hit)),
        "interior_recall": float(len(interior_hit) / len(interior_set)) if interior_set else 1.0,
    }


def scatter_group(ax, coords: np.ndarray, *, color: str, size: float, alpha: float) -> None:
    if len(coords) == 0:
        return
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=color,
        s=size,
        alpha=alpha,
        marker="s",
        linewidths=0,
        depthshade=False,
    )


def setup_ax(ax, title: str, elev: float, azim: float, resolution: int) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, resolution - 1)
    ax.set_ylim(0, resolution - 1)
    ax.set_zlim(0, resolution - 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6, pad=0)


def make_fig(gt: np.ndarray, pred: np.ndarray, fig_path: Path, title: str, resolution: int) -> None:
    views = [
        ("front", 20, -65),
        ("back", 20, 115),
        ("top", 90, -90),
        ("iso", 30, 45),
    ]
    fig = plt.figure(figsize=(16, 8), dpi=180)
    for i, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        scatter_group(ax, gt, color="#111111", size=2.5, alpha=0.38)
        setup_ax(ax, f"GT {name} ({len(gt)})", elev, azim, resolution)
        ax = fig.add_subplot(2, 4, i + 5, projection="3d")
        scatter_group(ax, pred, color="#2ca02c", size=2.5, alpha=0.42)
        setup_ax(ax, f"decoded {name} ({len(pred)})", elev, azim, resolution)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_path)
    plt.close(fig)


def parse_object_angle(surface: Path) -> tuple[str | None, int | None]:
    parts = surface.parts
    try:
        idx = parts.index("voxel_expanded")
        obj = parts[idx + 1]
        angle_name = parts[idx + 2]
        if angle_name.startswith("angle_"):
            return obj, int(angle_name[len("angle_") :])
    except Exception:
        pass
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--enc", required=True, type=Path)
    parser.add_argument("--dec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fig", type=Path)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gt = load_coords(args.surface, args.resolution)
    encoder = load_encoder(args.enc, device)
    decoder = load_decoder(args.dec, device)
    grid = dense_from_coords(gt, args.resolution, device)
    if next(encoder.parameters()).dtype == torch.float16:
        grid = grid.half()
    with torch.no_grad():
        latent = encoder(grid, sample_posterior=False)
        dec_in = latent
        if next(decoder.parameters()).dtype == torch.float16:
            dec_in = dec_in.half()
        logits = decoder(dec_in)
    pred = torch.nonzero(logits[0, 0].float() > float(args.threshold), as_tuple=False).cpu().numpy().astype(np.int64)

    m = metrics(gt, pred, args.resolution)
    obj, angle = parse_object_angle(args.surface)
    meta = {
        "surface": str(args.surface),
        "encoder": str(args.enc),
        "decoder": str(args.dec),
        "object_id": obj,
        "angle_idx": angle,
        "resolution": args.resolution,
        "threshold": args.threshold,
        "device": str(device),
        **m,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        gt_coords=gt.astype(np.int16),
        pred_coords=pred.astype(np.int16),
        latent=latent[0].detach().float().cpu().numpy().astype(np.float32),
        metrics=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    (args.out.with_suffix(".json")).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.fig:
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        title_obj = obj or args.surface.parent.parent.parent.name
        title_ang = f"angle_{angle}" if angle is not None else args.surface.parent.parent.name
        make_fig(gt, pred, args.fig, f"{title_obj} {title_ang} SS-VAE roundtrip IoU={m['iou']:.4f}", args.resolution)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
