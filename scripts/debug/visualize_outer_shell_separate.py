#!/usr/bin/env python3
"""Separate visualizations for GT full surface, outer shell, and removed voxels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_coords(path: Path, *, resolution: int) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path} expected [N,3], got {arr.shape}")
    if len(arr) == 0:
        return arr.reshape(0, 3)
    bad = np.any((arr < 0) | (arr >= resolution), axis=1)
    if np.any(bad):
        raise ValueError(f"{path} contains out-of-range coord {arr[bad][0].tolist()}")
    return np.unique(arr, axis=0)


def bbox_payload(coords: np.ndarray) -> dict[str, list[int] | None]:
    if len(coords) == 0:
        return {"min": None, "max": None, "span": None}
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return {
        "min": lo.astype(int).tolist(),
        "max": hi.astype(int).tolist(),
        "span": (hi - lo + 1).astype(int).tolist(),
    }


def coords_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in coords}


def set_to_coords(values: set[tuple[int, int, int]]) -> np.ndarray:
    if not values:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(sorted(values), dtype=np.int64)


def set_3d_axes(ax, *, resolution: int, title: str, elev: float, azim: float) -> None:
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(0, resolution - 1)
    ax.set_ylim(0, resolution - 1)
    ax.set_zlim(0, resolution - 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6, pad=0)


def scatter3d(
    ax,
    coords: np.ndarray,
    *,
    color: str,
    size: float,
    alpha: float,
    marker: str = "s",
) -> None:
    if len(coords) == 0:
        return
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=color,
        s=size,
        alpha=alpha,
        marker=marker,
        linewidths=0,
        depthshade=False,
    )


def make_3d_grid(
    *,
    full: np.ndarray,
    outer: np.ndarray,
    removed: np.ndarray,
    out_path: Path,
    title: str,
    resolution: int,
) -> None:
    views = [
        ("front", 20, -65),
        ("back", 20, 115),
        ("top", 90, -90),
        ("iso", 30, 45),
    ]
    columns = [
        ("full surface", "full"),
        ("outer shell", "outer"),
        ("removed/internal only", "removed"),
        ("outer shell + removed highlight", "overlay"),
    ]

    fig = plt.figure(figsize=(22, 18), dpi=160)
    for row, (view_name, elev, azim) in enumerate(views):
        for col, (col_title, mode) in enumerate(columns):
            ax = fig.add_subplot(len(views), len(columns), row * len(columns) + col + 1, projection="3d")
            if mode == "full":
                scatter3d(ax, full, color="#111111", size=2.2, alpha=0.22)
            elif mode == "outer":
                scatter3d(ax, outer, color="#2ca02c", size=3.0, alpha=0.48)
            elif mode == "removed":
                scatter3d(ax, removed, color="#d62728", size=16.0, alpha=0.92, marker="o")
            elif mode == "overlay":
                scatter3d(ax, outer, color="#9e9e9e", size=2.0, alpha=0.12)
                scatter3d(ax, removed, color="#d62728", size=18.0, alpha=0.95, marker="o")
            set_3d_axes(
                ax,
                resolution=resolution,
                title=f"{view_name} | {col_title}",
                elev=elev,
                azim=azim,
            )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path)
    plt.close(fig)


def set_2d_axes(ax, *, resolution: int, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1, resolution)
    ax.set_ylim(-1, resolution)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.tick_params(labelsize=7)


def scatter2d(
    ax,
    coords: np.ndarray,
    axes: tuple[int, int],
    *,
    color: str,
    size: float,
    alpha: float,
    marker: str = "s",
) -> None:
    if len(coords) == 0:
        return
    ax.scatter(
        coords[:, axes[0]],
        coords[:, axes[1]],
        c=color,
        s=size,
        alpha=alpha,
        marker=marker,
        linewidths=0,
    )


def make_projection_grid(
    *,
    full: np.ndarray,
    outer: np.ndarray,
    removed: np.ndarray,
    out_path: Path,
    title: str,
    resolution: int,
) -> None:
    projections = [
        ("xy", (0, 1), "x", "y"),
        ("xz", (0, 2), "x", "z"),
        ("yz", (1, 2), "y", "z"),
    ]
    columns = [
        ("full surface", "full"),
        ("outer shell", "outer"),
        ("removed/internal only", "removed"),
        ("outer + removed highlight", "overlay"),
    ]
    fig, axes = plt.subplots(
        len(projections),
        len(columns),
        figsize=(18, 13),
        dpi=180,
        squeeze=False,
    )
    for row, (proj_name, axes_idx, xlabel, ylabel) in enumerate(projections):
        for col, (col_title, mode) in enumerate(columns):
            ax = axes[row][col]
            if mode == "full":
                scatter2d(ax, full, axes_idx, color="#111111", size=3.0, alpha=0.22)
            elif mode == "outer":
                scatter2d(ax, outer, axes_idx, color="#2ca02c", size=4.0, alpha=0.46)
            elif mode == "removed":
                scatter2d(ax, removed, axes_idx, color="#d62728", size=16.0, alpha=0.86, marker="o")
            elif mode == "overlay":
                scatter2d(ax, outer, axes_idx, color="#bdbdbd", size=3.0, alpha=0.18)
                scatter2d(ax, removed, axes_idx, color="#d62728", size=20.0, alpha=0.92, marker="o")
            set_2d_axes(
                ax,
                resolution=resolution,
                title=f"{proj_name} | {col_title}",
                xlabel=xlabel,
                ylabel=ylabel,
            )
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True, type=Path, help="Original full surface.npy")
    parser.add_argument("--outer", required=True, type=Path, help="Extracted outer_surface.npy")
    parser.add_argument(
        "--removed",
        type=Path,
        help="Optional inner_removed.npy; if omitted, computed as full - outer.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", default="GT full surface / outer shell / removed")
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()

    full = load_coords(args.full, resolution=args.resolution)
    outer = load_coords(args.outer, resolution=args.resolution)
    if args.removed is not None:
        removed = load_coords(args.removed, resolution=args.resolution)
    else:
        removed = set_to_coords(coords_set(full) - coords_set(outer))

    outer_set = coords_set(outer)
    full_set = coords_set(full)
    removed_set = coords_set(removed)
    overlap_outer_full = outer_set & full_set
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "full_path": str(args.full),
        "outer_path": str(args.outer),
        "removed_path": str(args.removed) if args.removed else None,
        "full_count": int(len(full_set)),
        "outer_count": int(len(outer_set)),
        "removed_count": int(len(removed_set)),
        "outer_subset_of_full": bool(outer_set <= full_set),
        "removed_equals_full_minus_outer": bool(removed_set == (full_set - outer_set)),
        "outer_recall_vs_full": float(len(overlap_outer_full) / len(full_set)) if full_set else 0.0,
        "full_bbox": bbox_payload(full),
        "outer_bbox": bbox_payload(outer),
        "removed_bbox": bbox_payload(removed),
    }
    (args.out_dir / "separate_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_3d_grid(
        full=full,
        outer=outer,
        removed=removed,
        out_path=args.out_dir / "separate_3d_grid.png",
        title=args.title,
        resolution=args.resolution,
    )
    make_projection_grid(
        full=full,
        outer=outer,
        removed=removed,
        out_path=args.out_dir / "separate_projection_grid.png",
        title=args.title,
        resolution=args.resolution,
    )

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
