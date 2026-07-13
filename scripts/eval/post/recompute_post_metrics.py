#!/usr/bin/env python3
"""Recompute post-smoothing metrics on existing before/after mesh reports."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from scripts.eval.post.holopart_smooth import (
    _chamfer_to_overall,
    _load_mesh,
    _sample_points,
    _smoothness,
    _surface_distance_metrics,
    _voxel_overlap_volume,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scene_to_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(tuple(meshes))
    mesh.remove_unreferenced_vertices()
    return mesh


def _overall_mesh(report: dict[str, Any], report_path: Path):
    source = str(report.get("overall_mesh_source") or "")
    if source and source != "merged_body_parts":
        path = Path(source)
        if path.is_file():
            return _load_mesh(path)
    merged = report_path.parent / "inputs" / "overall_merged_before.obj"
    if merged.is_file():
        return _load_mesh(merged)
    glb = report_path.parent / "inputs" / "component_scene_input.glb"
    if glb.is_file():
        return _scene_to_mesh(glb)
    raise FileNotFoundError(f"cannot find overall mesh for {report_path}")


def _to_overall_quantiles(mesh, overall_mesh, *, samples: int, seed: int) -> dict[str, float]:
    pts = _sample_points(mesh, int(samples), int(seed))
    overall_pts = _sample_points(overall_mesh, int(samples), int(seed) + 17)
    if len(pts) == 0 or len(overall_pts) == 0:
        return {
            "after_to_overall_p99": float("nan"),
            "after_to_overall_max": float("nan"),
        }
    tree = cKDTree(overall_pts)
    dist, _idx = tree.query(pts, k=1, workers=-1)
    return {
        "after_to_overall_p99": float(np.quantile(dist, 0.99)),
        "after_to_overall_max": float(np.max(dist)),
    }


def recompute_report(report_path: Path, *, metric_samples: int, completeness_samples: int, overlap_pitch: float, overlap_max_faces: int) -> dict[str, Any]:
    report = _load_json(report_path)
    overall = _overall_mesh(report, report_path)
    overall_points = _sample_points(overall, int(metric_samples), seed=123)
    rows: list[dict[str, Any]] = []
    after_meshes = []
    for idx, old in enumerate(report.get("metrics") or []):
        before_path = Path(str(old.get("before_mesh")))
        after_path = Path(str(old.get("after_mesh")))
        before = _load_mesh(before_path)
        after = _load_mesh(after_path)
        after_meshes.append((str(old.get("component")), after))
        before_s = _smoothness(before)
        after_s = _smoothness(after)
        after_to_overall = _chamfer_to_overall(after, overall_points, samples=int(metric_samples), seed=456 + idx)
        after_to_overall.update(_to_overall_quantiles(after, overall, samples=int(completeness_samples), seed=1701 + idx))
        before_to_after = _surface_distance_metrics(
            before,
            after,
            samples=int(completeness_samples),
            seed=789 + idx,
            prefix="before_to_after",
        )
        bidir_mean = max(float(after_to_overall["after_to_overall_mean"]), float(before_to_after["before_to_after_mean"]))
        bidir_p95 = max(float(after_to_overall["after_to_overall_p95"]), float(before_to_after["before_to_after_p95"]))
        row = {
            **old,
            "seconds": old.get("seconds"),
            "before_mean_dihedral_rad": before_s["mean_dihedral_rad"],
            "after_mean_dihedral_rad": after_s["mean_dihedral_rad"],
            "before_normal_variance": before_s["normal_variance"],
            "after_normal_variance": after_s["normal_variance"],
            "before_is_watertight": before_s["is_watertight"],
            "after_is_watertight": after_s["is_watertight"],
            "before_vertices": before_s["vertex_count"],
            "after_vertices": after_s["vertex_count"],
            "before_faces": before_s["face_count"],
            "after_faces": after_s["face_count"],
            **after_to_overall,
            **before_to_after,
            "bidirectional_chamfer_mean_max": float(bidir_mean),
            "bidirectional_chamfer_p95_max": float(bidir_p95),
        }
        rows.append(row)
    report["metrics"] = rows
    report["after_intersection"] = _voxel_overlap_volume(
        after_meshes,
        pitch=float(overlap_pitch),
        max_faces=int(overlap_max_faces),
        mode="voxel",
    )
    _write_json(report_path, report)
    _write_csv(report_path.parent / "metrics.csv", rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--report-glob", default="")
    parser.add_argument("--metric-samples", type=int, default=12000)
    parser.add_argument("--completeness-samples", type=int, default=50000)
    parser.add_argument("--overlap-pitch", type=float, default=0.02)
    parser.add_argument("--overlap-max-faces", type=int, default=100000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [Path(path) for path in args.report]
    if args.report_glob:
        reports.extend(Path(path) for path in sorted(glob.glob(args.report_glob)))
    if not reports:
        raise SystemExit("provide --report or --report-glob")
    for report in reports:
        recompute_report(
            report.resolve(),
            metric_samples=int(args.metric_samples),
            completeness_samples=int(args.completeness_samples),
            overlap_pitch=float(args.overlap_pitch),
            overlap_max_faces=int(args.overlap_max_faces),
        )
        print(f"[recompute] {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
