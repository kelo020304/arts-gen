#!/usr/bin/env python3
"""Classic geometry-only repair baselines for ee-eval component meshes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.post.holopart_smooth import (  # noqa: E402
    _build_overall_mesh,
    _chamfer_to_overall,
    _component_list,
    _load_mesh,
    _load_render_camera,
    _safe_name,
    _sample_points,
    _smoothness,
    _surface_distance_metrics,
    _voxel_overlap_volume,
    _write_component_panels,
    _write_csv,
    _write_exploded_panel,
    _write_json,
    _write_overview_panel,
)


def _run_pymeshlab_subprocess(
    before_path: Path,
    out_path: Path,
    *,
    python: Path,
    max_hole_size: int,
    remesh_iterations: int,
    target_len_fraction: float,
    smooth_steps: int,
    timeout: int,
) -> dict[str, Any]:
    inline = r"""
import json
import sys
from pathlib import Path

import pymeshlab

inp = Path(sys.argv[1])
out = Path(sys.argv[2])
max_hole_size = int(sys.argv[3])
remesh_iterations = int(sys.argv[4])
target_len_fraction = float(sys.argv[5])
smooth_steps = int(sys.argv[6])

ms = pymeshlab.MeshSet()
ms.load_new_mesh(str(inp))
before_faces = ms.current_mesh().face_number()
try:
    ms.meshing_close_holes(maxholesize=max_hole_size, refinehole=True, selected=False)
except Exception as exc:
    print(f"[classic:pymeshlab] close_holes failed: {exc!r}", flush=True)
if remesh_iterations > 0:
    try:
        bbox = ms.current_mesh().bounding_box()
        diag = float(bbox.diagonal())
        target = pymeshlab.PercentageValue(float(target_len_fraction) * 100.0)
        ms.meshing_isotropic_explicit_remeshing(
            iterations=remesh_iterations,
            targetlen=target,
            checksurfdist=True,
        )
    except Exception as exc:
        print(f"[classic:pymeshlab] isotropic remesh failed: {exc!r}", flush=True)
if smooth_steps > 0:
    try:
        ms.apply_coord_taubin_smoothing(stepsmoothnum=smooth_steps, lambda_=0.5, mu=-0.53)
    except Exception as exc:
        print(f"[classic:pymeshlab] taubin failed: {exc!r}", flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
ms.save_current_mesh(str(out), save_vertex_color=True, save_face_color=True)
payload = {
    "before_faces": int(before_faces),
    "after_faces": int(ms.current_mesh().face_number()),
}
print(json.dumps(payload), flush=True)
"""
    started = time.time()
    proc = subprocess.run(
        [
            str(python),
            "-c",
            inline,
            str(before_path),
            str(out_path),
            str(int(max_hole_size)),
            str(int(remesh_iterations)),
            str(float(target_len_fraction)),
            str(int(smooth_steps)),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(timeout),
        check=False,
    )
    return {
        "returncode": int(proc.returncode),
        "seconds": float(time.time() - started),
        "log": proc.stdout[-4000:],
    }


def _o3d_mesh_to_trimesh(mesh: Any) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )


def _run_open3d_repair(
    before: trimesh.Trimesh,
    out_path: Path,
    *,
    method: str,
    samples: int,
    alpha_fraction: float,
    poisson_depth: int,
) -> None:
    import open3d as o3d

    pts = _sample_points(before, int(samples), seed=314)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(np.max(before.extents)) * 0.05,
            max_nn=30,
        )
    )
    if str(method) == "o3d_alpha":
        alpha = max(float(np.max(before.extents)) * float(alpha_fraction), 1.0e-4)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)
    elif str(method) == "o3d_poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=int(poisson_depth))
        densities_np = np.asarray(densities)
        if len(densities_np):
            keep = densities_np >= np.quantile(densities_np, 0.02)
            mesh.remove_vertices_by_mask(~keep)
    else:
        raise ValueError(f"unsupported open3d method: {method}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    out = _o3d_mesh_to_trimesh(mesh)
    if len(out.vertices) == 0 or len(out.faces) == 0:
        raise ValueError(f"{method} produced empty mesh")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.export(out_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = Path(args.summary).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_dir, _whole_voxel, components, meta = _component_list(
        summary_path,
        assets_dir=Path(args.assets_dir).resolve() if args.assets_dir else None,
    )
    summary = meta["summary"]
    assets_dir = Path(meta["assets_dir"])
    overall_path = None
    for name in ("overall.obj", "overall.glb", "whole.obj", "whole.glb"):
        candidate = assets_dir / name
        if candidate.is_file():
            overall_path = candidate
            break
    if overall_path is not None:
        overall_mesh = _load_mesh(overall_path)
        overall_source_text = str(overall_path)
    else:
        overall_mesh = _build_overall_mesh(components, out_dir / "inputs" / "overall_merged_before.obj")
        overall_source_text = "merged_body_parts"

    after_paths: dict[str, Path] = {}
    times: dict[str, float] = {}
    method_logs: dict[str, Any] = {}
    after_raw = out_dir / "after_components"
    after_raw.mkdir(parents=True, exist_ok=True)
    for comp in components:
        out_path = after_raw / f"{_safe_name(comp.label)}.obj"
        started = time.time()
        if str(args.method) == "pymeshlab_close_remesh":
            status = _run_pymeshlab_subprocess(
                comp.before_mesh_path,
                out_path,
                python=Path(args.pymeshlab_python),
                max_hole_size=int(args.max_hole_size),
                remesh_iterations=int(args.remesh_iterations),
                target_len_fraction=float(args.target_len_fraction),
                smooth_steps=int(args.smooth_steps),
                timeout=int(args.timeout),
            )
            method_logs[comp.label] = status
            if int(status["returncode"]) != 0 or not out_path.is_file():
                raise RuntimeError(f"pymeshlab repair failed for {comp.label}: {status['log']}")
            times[comp.label] = float(status["seconds"])
        elif str(args.method) in {"o3d_alpha", "o3d_poisson"}:
            _run_open3d_repair(
                comp.before_mesh,
                out_path,
                method=str(args.method),
                samples=int(args.open3d_samples),
                alpha_fraction=float(args.alpha_fraction),
                poisson_depth=int(args.poisson_depth),
            )
            times[comp.label] = float(time.time() - started)
        elif str(args.method) == "trimesh_fill":
            mesh = comp.before_mesh.copy()
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fill_holes(mesh)
            if int(args.smooth_steps) > 0:
                trimesh.smoothing.filter_taubin(mesh, iterations=int(args.smooth_steps))
            mesh.export(out_path)
            times[comp.label] = float(time.time() - started)
        else:
            raise ValueError(f"unsupported method: {args.method}")
        after_paths[comp.label] = out_path

    overall_points = _sample_points(overall_mesh, int(args.metric_samples), seed=123)
    after_meshes: dict[str, trimesh.Trimesh] = {}
    rows: list[dict[str, Any]] = []
    for idx, comp in enumerate(components):
        after = _load_mesh(after_paths[comp.label])
        after_meshes[comp.label] = after
        before_s = _smoothness(comp.before_mesh)
        after_s = _smoothness(after)
        a2o = _chamfer_to_overall(after, overall_points, samples=int(args.metric_samples), seed=456 + idx)
        b2a = _surface_distance_metrics(
            comp.before_mesh,
            after,
            samples=int(args.completeness_samples),
            seed=789 + idx,
            prefix="before_to_after",
        )
        rows.append(
            {
                "component": comp.label,
                "role": comp.role,
                "before_mesh": str(comp.before_mesh_path),
                "after_mesh": str(after_paths[comp.label].resolve()),
                "voxel_path": None if comp.voxel_path is None else str(comp.voxel_path),
                "voxel_count": None if comp.coords is None else int(comp.coords.shape[0]),
                "seconds": float(times.get(comp.label, float("nan"))),
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
                **a2o,
                **b2a,
                "bidirectional_chamfer_mean_max": float(max(a2o["after_to_overall_mean"], b2a["before_to_after_mean"])),
                "bidirectional_chamfer_p95_max": float(max(a2o["after_to_overall_p95"], b2a["before_to_after_p95"])),
            }
        )

    intersection = _voxel_overlap_volume(
        list(after_meshes.items()),
        pitch=float(args.intersection_pitch),
        max_faces=int(args.intersection_max_faces),
        mode=str(args.intersection_mode),
    )
    if bool(args.render):
        extrinsic, intrinsic = _load_render_camera(summary, render_view=int(args.render_view))
        _write_component_panels(
            components,
            after_meshes,
            out_dir,
            method=str(args.method),
            max_faces=int(args.render_max_faces),
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(args.render_resolution),
        )
        _write_overview_panel(
            components,
            after_meshes,
            out_dir / "before_after_overview_color.png",
            method=str(args.method),
            max_faces=int(args.render_max_faces),
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(args.render_resolution),
        )
        _write_exploded_panel(
            components,
            after_meshes,
            out_dir / "after_exploded_overview_color.png",
            max_faces=int(args.render_max_faces),
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(args.render_resolution),
        )
        panel_dir = out_dir / "panels_color"
        if panel_dir.exists():
            import shutil

            shutil.rmtree(panel_dir)
        if (out_dir / "panels").is_dir():
            import shutil

            shutil.copytree(out_dir / "panels", panel_dir)

    _write_csv(out_dir / "metrics.csv", rows)
    report = {
        "method": str(args.method),
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),
        "overall_mesh_source": overall_source_text,
        "component_count": int(len(components)),
        "components": [
            {
                "label": comp.label,
                "role": comp.role,
                "before_mesh": str(comp.before_mesh_path),
                "before_mesh_stats": comp.mesh_stats,
                "voxel_path": None if comp.voxel_path is None else str(comp.voxel_path),
            }
            for comp in components
        ],
        "method_status": {
            str(args.method): {
                "returncode": 0,
                "mode": str(args.method),
                "logs": method_logs,
                "max_hole_size": int(args.max_hole_size),
                "remesh_iterations": int(args.remesh_iterations),
                "target_len_fraction": float(args.target_len_fraction),
                "smooth_steps": int(args.smooth_steps),
                "alpha_fraction": float(args.alpha_fraction),
                "poisson_depth": int(args.poisson_depth),
            },
            f"{args.method}_match": {
                "matched_count": int(len(rows)),
                "expected_count": int(len(components)),
            },
        },
        "metrics": rows,
        "after_intersection": intersection,
    }
    _write_json(out_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument("--method", choices=("pymeshlab_close_remesh", "o3d_alpha", "o3d_poisson", "trimesh_fill"), default="pymeshlab_close_remesh")
    parser.add_argument("--pymeshlab-python", type=Path, default=Path("/opt/venvs/holopart/bin/python"))
    parser.add_argument("--max-hole-size", type=int, default=100000)
    parser.add_argument("--remesh-iterations", type=int, default=2)
    parser.add_argument("--target-len-fraction", type=float, default=0.006)
    parser.add_argument("--smooth-steps", type=int, default=3)
    parser.add_argument("--open3d-samples", type=int, default=80000)
    parser.add_argument("--alpha-fraction", type=float, default=0.025)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--metric-samples", type=int, default=12000)
    parser.add_argument("--completeness-samples", type=int, default=50000)
    parser.add_argument("--intersection-mode", choices=("bbox", "voxel", "skip"), default="voxel")
    parser.add_argument("--intersection-pitch", type=float, default=0.02)
    parser.add_argument("--intersection-max-faces", type=int, default=100000)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-max-faces", type=int, default=0)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--render-view", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(f"[classic] report -> {report['out_dir']}/report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
