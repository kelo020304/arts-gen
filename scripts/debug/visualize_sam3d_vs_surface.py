#!/usr/bin/env python3
"""Visualize SAM3D StageA voxels against dataset surface voxels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_coords(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if "coords" not in data.files:
            raise KeyError(f"{path} expected key 'coords', found {data.files}")
        arr = data["coords"]
    else:
        arr = np.load(path)
    arr = np.asarray(arr, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path} expected [N,3] coords, got {arr.shape}")
    if np.any(arr < 0) or np.any(arr >= 64):
        bad = arr[np.any((arr < 0) | (arr >= 64), axis=1)][0]
        raise ValueError(f"{path} contains out-of-range coord {bad.tolist()}")
    return np.unique(arr, axis=0)


def coords_to_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in coords}


def set_to_coords(values: set[tuple[int, int, int]]) -> np.ndarray:
    if not values:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(sorted(values), dtype=np.int64)


def plot_group(ax, coords: np.ndarray, *, color: str, label: str, size: float, alpha: float) -> None:
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
        label=label,
        depthshade=False,
    )


def setup_ax(ax, elev: float, azim: float, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 63)
    ax.set_ylim(0, 63)
    ax.set_zlim(0, 63)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.tick_params(labelsize=7, pad=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True, type=Path, help="GT surface.npy")
    parser.add_argument("--pred", required=True, type=Path, help="SAM3D voxel.npz")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", default="SAM3D StageA vs GT surface")
    parser.add_argument("--surface-label", default="GT")
    parser.add_argument("--pred-label", default="SAM3D")
    args = parser.parse_args()

    surface = load_coords(args.surface)
    pred = load_coords(args.pred)
    surface_set = coords_to_set(surface)
    pred_set = coords_to_set(pred)

    overlap_set = surface_set & pred_set
    gt_only_set = surface_set - pred_set
    pred_only_set = pred_set - surface_set

    overlap = set_to_coords(overlap_set)
    gt_only = set_to_coords(gt_only_set)
    pred_only = set_to_coords(pred_only_set)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "surface_path": str(args.surface),
        "pred_path": str(args.pred),
        "surface_count": int(len(surface_set)),
        "pred_count": int(len(pred_set)),
        "overlap_count": int(len(overlap_set)),
        "gt_only_count": int(len(gt_only_set)),
        "pred_only_count": int(len(pred_only_set)),
        "pred_recall_vs_surface": float(len(overlap_set) / len(surface_set)) if surface_set else 0.0,
        "pred_precision_vs_surface": float(len(overlap_set) / len(pred_set)) if pred_set else 0.0,
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    views = [
        ("front", 20, -65),
        ("back", 20, 115),
        ("top", 90, -90),
        ("iso", 30, 45),
    ]
    colors = {
        "gt_only": "#d62728",
        "pred_only": "#1f77b4",
        "overlap": "#2ca02c",
    }

    for name, elev, azim in views:
        fig = plt.figure(figsize=(9, 8), dpi=180)
        ax = fig.add_subplot(111, projection="3d")
        plot_group(ax, gt_only, color=colors["gt_only"], label=f"{args.surface_label} only ({len(gt_only)})", size=5, alpha=0.22)
        plot_group(ax, pred_only, color=colors["pred_only"], label=f"{args.pred_label} only ({len(pred_only)})", size=9, alpha=0.65)
        plot_group(ax, overlap, color=colors["overlap"], label=f"overlap ({len(overlap)})", size=8, alpha=0.7)
        setup_ax(ax, elev, azim, f"{args.title} - {name}")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"sam3d_vs_surface_{name}.png")
        plt.close(fig)

    fig = plt.figure(figsize=(14, 11), dpi=180)
    for idx, (name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        plot_group(ax, gt_only, color=colors["gt_only"], label=f"{args.surface_label} only", size=3, alpha=0.18)
        plot_group(ax, pred_only, color=colors["pred_only"], label=f"{args.pred_label} only", size=6, alpha=0.6)
        plot_group(ax, overlap, color=colors["overlap"], label="overlap", size=5, alpha=0.68)
        setup_ax(ax, elev, azim, name)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11)
    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out_dir / "sam3d_vs_surface_grid.png")
    plt.close(fig)

    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
