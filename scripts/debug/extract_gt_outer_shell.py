#!/usr/bin/env python3
"""Extract the outside-air reachable shell from sparse GT surface voxels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np


NEIGHBORS = np.asarray(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int64,
)


def load_coords(path: Path, resolution: int) -> np.ndarray:
    coords = np.asarray(np.load(path), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} expected [N,3], got {coords.shape}")
    if len(coords) == 0:
        return coords.reshape(0, 3)
    bad_mask = np.any((coords < 0) | (coords >= resolution), axis=1)
    if np.any(bad_mask):
        bad = coords[bad_mask][0].tolist()
        raise ValueError(f"{path} contains out-of-range coord {bad} for res={resolution}")
    return np.unique(coords, axis=0)


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


def coords_key(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64)
    return coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]


def enqueue_boundary_empty(occ: np.ndarray) -> deque[tuple[int, int, int]]:
    outside = np.zeros_like(occ, dtype=bool)
    boundary = np.zeros_like(occ, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    seed_mask = boundary & ~occ
    outside[seed_mask] = True
    q: deque[tuple[int, int, int]] = deque(
        tuple(map(int, item)) for item in np.argwhere(seed_mask)
    )
    return q, outside


def flood_outside_empty(occ: np.ndarray) -> np.ndarray:
    q, outside = enqueue_boundary_empty(occ)
    n = occ.shape[0]
    while q:
        x, y, z = q.popleft()
        for dx, dy, dz in NEIGHBORS:
            nx = x + int(dx)
            ny = y + int(dy)
            nz = z + int(dz)
            if nx < 0 or ny < 0 or nz < 0 or nx >= n or ny >= n or nz >= n:
                continue
            if occ[nx, ny, nz] or outside[nx, ny, nz]:
                continue
            outside[nx, ny, nz] = True
            q.append((nx, ny, nz))
    return outside


def extract_outer_shell(
    coords: np.ndarray,
    *,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(coords) == 0:
        empty_mask = np.zeros((0,), dtype=bool)
        return coords, coords, empty_mask

    padded_size = resolution + 2
    occ = np.zeros((padded_size, padded_size, padded_size), dtype=bool)
    shifted = coords + 1
    occ[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = True
    outside = flood_outside_empty(occ)

    outer_mask = np.zeros(len(coords), dtype=bool)
    for dx, dy, dz in NEIGHBORS:
        nb = shifted + np.asarray([dx, dy, dz], dtype=np.int64)
        outer_mask |= outside[nb[:, 0], nb[:, 1], nb[:, 2]]

    outer = coords[outer_mask]
    inner_removed = coords[~outer_mask]
    return outer, inner_removed, outer_mask


def filter_coords_by_allowed(
    coords: np.ndarray,
    allowed_keys: np.ndarray,
    *,
    resolution: int,
) -> np.ndarray:
    coords = np.unique(np.asarray(coords, dtype=np.int64).reshape(-1, 3), axis=0)
    if len(coords) == 0:
        return coords
    mask = np.isin(coords_key(coords, resolution), allowed_keys, assume_unique=False)
    return coords[mask]


def export_filtered_voxel_dir(
    *,
    source_voxel_dir: Path,
    out_dir: Path,
    outer: np.ndarray,
    resolution: int,
) -> dict[str, object]:
    voxel_out = out_dir / "voxel_outer" / str(resolution)
    voxel_out.mkdir(parents=True, exist_ok=True)
    np.save(voxel_out / "surface.npy", outer.astype(np.int64))

    allowed = coords_key(outer, resolution)
    part_rows: list[dict[str, object]] = []
    for ind_path in sorted(source_voxel_dir.glob("ind_*.npy")):
        coords = load_coords(ind_path, resolution)
        filtered = filter_coords_by_allowed(coords, allowed, resolution=resolution)
        np.save(voxel_out / ind_path.name, filtered.astype(np.int64))
        part_rows.append(
            {
                "part_file": ind_path.name,
                "source_count": int(len(coords)),
                "outer_count": int(len(filtered)),
                "removed_count": int(len(coords) - len(filtered)),
                "outer_ratio": float(len(filtered) / len(coords)) if len(coords) else 0.0,
            }
        )

    for optional_name in ("viz",):
        src = source_voxel_dir / optional_name
        dst = voxel_out / optional_name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)

    payload: dict[str, object] = {
        "source_voxel_dir": str(source_voxel_dir),
        "voxel_outer_dir": str(voxel_out),
        "part_files": part_rows,
    }
    (out_dir / "filtered_part_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def write_one(
    surface_path: Path,
    out_dir: Path,
    *,
    resolution: int,
    source_voxel_dir: Path | None = None,
) -> dict[str, object]:
    coords = load_coords(surface_path, resolution)
    outer, inner_removed, _outer_mask = extract_outer_shell(coords, resolution=resolution)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "outer_surface.npy", outer.astype(np.int64))
    np.save(out_dir / "inner_removed.npy", inner_removed.astype(np.int64))

    stats: dict[str, object] = {
        "surface_path": str(surface_path),
        "out_dir": str(out_dir),
        "resolution": resolution,
        "surface_count": int(len(coords)),
        "outer_count": int(len(outer)),
        "inner_removed_count": int(len(inner_removed)),
        "outer_ratio": float(len(outer) / len(coords)) if len(coords) else 0.0,
        "surface_bbox": bbox_payload(coords),
        "outer_bbox": bbox_payload(outer),
        "inner_removed_bbox": bbox_payload(inner_removed),
    }
    if source_voxel_dir is not None:
        stats["filtered_voxel_export"] = export_filtered_voxel_dir(
            source_voxel_dir=source_voxel_dir,
            out_dir=out_dir,
            outer=outer,
            resolution=resolution,
        )
    (out_dir / "outer_shell_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--source-voxel-dir",
        type=Path,
        action="append",
        help="Optional source dir containing ind_*.npy files to filter by the same outer shell.",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--summary-csv", type=Path)
    args = parser.parse_args()

    if len(args.surface) != len(args.out_dir):
        raise ValueError("--surface and --out-dir must be passed the same number of times")
    source_voxel_dirs = args.source_voxel_dir or [None] * len(args.surface)
    if len(source_voxel_dirs) != len(args.surface):
        raise ValueError(
            "--source-voxel-dir must be omitted or passed the same number of times as --surface"
        )

    rows = [
        write_one(
            surface_path,
            out_dir,
            resolution=args.resolution,
            source_voxel_dir=source_voxel_dir,
        )
        for surface_path, out_dir, source_voxel_dir in zip(
            args.surface, args.out_dir, source_voxel_dirs, strict=True
        )
    ]

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "surface_path",
            "out_dir",
            "resolution",
            "surface_count",
            "outer_count",
            "inner_removed_count",
            "outer_ratio",
            "surface_bbox",
            "outer_bbox",
            "inner_removed_bbox",
        ]
        with args.summary_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(row[key], ensure_ascii=False)
                        if isinstance(row[key], dict)
                        else row[key]
                        for key in fieldnames
                    }
                )

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
