#!/usr/bin/env python3
"""Directly keep the outermost occupied voxels along the six axis directions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_coords(path: Path, resolution: int) -> np.ndarray:
    coords = np.asarray(np.load(path), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} expected [N,3], got {coords.shape}")
    if len(coords) and np.any((coords < 0) | (coords >= resolution)):
        bad = coords[np.any((coords < 0) | (coords >= resolution), axis=1)][0]
        raise ValueError(f"{path} has out-of-range coord {bad.tolist()}")
    return np.unique(coords.reshape(-1, 3), axis=0)


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


def coord_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    if len(coords) == 0:
        return np.empty((0,), dtype=np.int64)
    return coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]


def direct_outer_6dir(coords: np.ndarray) -> np.ndarray:
    """For every axis-parallel occupied line, keep min and max along that axis."""
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


def filter_coords(coords: np.ndarray, allowed_keys: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.unique(np.asarray(coords, dtype=np.int64).reshape(-1, 3), axis=0)
    if len(coords) == 0:
        return coords
    mask = np.isin(coord_keys(coords, resolution), allowed_keys)
    return coords[mask]


def process_one(
    *,
    surface_path: Path,
    out_dir: Path,
    resolution: int,
    source_voxel_dir: Path | None,
) -> dict[str, object]:
    full = load_coords(surface_path, resolution)
    outer = direct_outer_6dir(full)
    outer_keys = coord_keys(outer, resolution)
    removed = filter_coords(full, np.setdiff1d(coord_keys(full, resolution), outer_keys), resolution)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "outer_surface.npy", outer.astype(np.int64))
    np.save(out_dir / "inner_removed.npy", removed.astype(np.int64))

    stats: dict[str, object] = {
        "surface_path": str(surface_path),
        "out_dir": str(out_dir),
        "method": "direct_outer_6dir_minmax_per_axis_line",
        "resolution": resolution,
        "surface_count": int(len(full)),
        "outer_count": int(len(outer)),
        "inner_removed_count": int(len(removed)),
        "outer_ratio": float(len(outer) / len(full)) if len(full) else 0.0,
        "surface_bbox": bbox_payload(full),
        "outer_bbox": bbox_payload(outer),
        "inner_removed_bbox": bbox_payload(removed),
    }

    if source_voxel_dir is not None:
        voxel_out = out_dir / "voxel_direct_outer" / str(resolution)
        voxel_out.mkdir(parents=True, exist_ok=True)
        np.save(voxel_out / "surface.npy", outer.astype(np.int64))
        part_rows = []
        for ind_path in sorted(source_voxel_dir.glob("ind_*.npy")):
            coords = load_coords(ind_path, resolution)
            filtered = filter_coords(coords, outer_keys, resolution)
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
        stats["filtered_voxel_export"] = {
            "source_voxel_dir": str(source_voxel_dir),
            "voxel_direct_outer_dir": str(voxel_out),
            "part_files": part_rows,
        }

    (out_dir / "direct_outer_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, action="append", required=True)
    parser.add_argument("--source-voxel-dir", type=Path, action="append")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--summary-csv", type=Path)
    args = parser.parse_args()

    if len(args.surface) != len(args.out_dir):
        raise ValueError("--surface and --out-dir counts must match")
    source_dirs = args.source_voxel_dir or [None] * len(args.surface)
    if len(source_dirs) != len(args.surface):
        raise ValueError("--source-voxel-dir must be omitted or passed for every surface")

    rows = [
        process_one(
            surface_path=surface,
            out_dir=out_dir,
            resolution=args.resolution,
            source_voxel_dir=source_dir,
        )
        for surface, out_dir, source_dir in zip(args.surface, args.out_dir, source_dirs, strict=True)
    ]

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "surface_path",
            "out_dir",
            "method",
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
