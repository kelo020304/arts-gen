#!/usr/bin/env python3
"""Render existing part-route outputs from a fixed source-frame direction."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "TRELLIS-arts"))

from scripts.eval.post.holopart_smooth import (
    SAM3D_Z_UP_TO_Y_UP,
    _component_list,
    _load_mesh,
    _safe_name,
    _tile,
    render_component,
)
from scripts.eval.post.part_mesh_routes import (
    OBJECTS,
    _render_route_panel,
    _summary_path,
    _write_exploded,
    _write_json,
)


ROUTES = ("R-P", "R-P-fallback", "R-V", "R-D", "B-dec", "B-dec+clip", "R-X")


def _render_space_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Match render_component(default): exported OBJ Y-up -> SAM3D/source Z-up."""

    return np.asarray(mesh.vertices, dtype=np.float64) @ SAM3D_Z_UP_TO_Y_UP.T


def _source_neg_y_camera(mesh: trimesh.Trimesh, *, fov_degrees: float = 32.0) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Camera on source -Y side, looking toward +Y, with +Z up."""

    vertices = _render_space_vertices(mesh)
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)], axis=0)
    center = bounds.mean(axis=0)
    camera_side = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    forward = -camera_side
    up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-8)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1.0e-8)
    # MeshRenderer uses OpenCV-style camera coordinates: +Y points down in the
    # image.  Use -up as the camera Y row so source +Z appears upward on screen.
    camera_down = -up

    rel = vertices - center[None, :]
    half_w = float(np.max(np.abs(rel @ right)))
    half_h = float(np.max(np.abs(rel @ up)))
    half_d = float(np.max(np.abs(rel @ forward)))
    fov = math.radians(float(fov_degrees))
    distance = max(max(half_w, half_h) / max(math.tan(fov / 2.0), 1.0e-6) + half_d + 0.15, 1.25)
    eye = center + camera_side * distance

    rot = np.stack([right, camera_down, forward], axis=0).astype(np.float32)
    trans = (-rot @ eye.astype(np.float32)).astype(np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rot
    extrinsic[:3, 3] = trans
    focal = 0.5 / math.tan(fov / 2.0)
    intrinsic = np.asarray([[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    report = {
        "camera": "source_neg_y_outward",
        "definition": "exported route meshes are rendered after OBJ Y-up -> source/SAM3D Z-up conversion; camera is on source -Y side, looks toward +Y, screen up is +Z",
        "camera_side": [0.0, -1.0, 0.0],
        "look_direction": [0.0, 1.0, 0.0],
        "screen_up": [0.0, 0.0, 1.0],
        "camera_y_down": camera_down.tolist(),
        "screen_right": right.tolist(),
        "fov_degrees": float(fov_degrees),
        "distance": float(distance),
        "source_space_bounds": bounds.tolist(),
    }
    return torch.from_numpy(extrinsic), torch.from_numpy(intrinsic), report


def _route_paths(source_object_dir: Path, components: list[Any]) -> dict[str, dict[str, Path]]:
    routes: dict[str, dict[str, Path]] = {route: {} for route in ROUTES}
    for route in ROUTES:
        route_dir = source_object_dir / "routes" / route
        if not route_dir.is_dir():
            continue
        for comp in components:
            path = route_dir / f"{_safe_name(comp.label)}.obj"
            if path.is_file():
                routes[route][comp.label] = path
    return routes


def _overall_path(summary: dict[str, Any], assets_dir: Path) -> Path:
    path = Path((summary.get("mujoco_overall_mesh") or {}).get("mesh_path") or assets_dir / "overall.obj")
    if not path.is_file():
        raise FileNotFoundError(f"overall mesh missing: {path}")
    return path


def _write_gate(
    *,
    overall_mesh: trimesh.Trimesh,
    out_path: Path,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    resolution: int,
    max_faces: int,
) -> None:
    image = render_component(
        overall_mesh,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=int(max_faces),
        color_mode="color",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _tile(image, "overall source -Y outward", int(resolution), int(resolution) + 30).save(out_path)


def render_object(
    *,
    tag: str,
    dataset_id: str,
    object_id: str,
    angle: int,
    eval_dir: Path,
    source_root: Path,
    out_root: Path,
    resolution: int,
    max_faces: int,
) -> dict[str, Any]:
    summary_path = _summary_path(eval_dir, dataset_id, object_id, angle)
    _run_dir, _whole_voxel, components, meta = _component_list(summary_path)
    summary = meta["summary"]
    assets_dir = Path(meta["assets_dir"])
    overall_mesh = _load_mesh(_overall_path(summary, assets_dir))
    extrinsic, intrinsic, camera_report = _source_neg_y_camera(overall_mesh)

    object_dir = out_root / f"{tag}_{_safe_name(object_id)}"
    source_object_dir = source_root / f"{tag}_{_safe_name(object_id)}"
    routes = _route_paths(source_object_dir, components)
    if not any(routes.values()):
        raise FileNotFoundError(f"no route meshes found under {source_object_dir / 'routes'}")

    _write_gate(
        overall_mesh=overall_mesh,
        out_path=object_dir / "render_gate" / "overall_source_neg_y.png",
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=int(max_faces),
    )
    _render_route_panel(
        components=components,
        routes=routes,
        out_path=object_dir / "six_column_components.png",
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=int(max_faces),
    )
    _write_exploded(
        components=components,
        routes={"before": {}, **routes},
        out_dir=object_dir / "exploded",
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=int(max_faces),
    )
    report = {
        "object_tag": tag,
        "object_key": f"{dataset_id}::{object_id}::angle_{int(angle):02d}",
        "summary": str(summary_path),
        "source_routes": str(source_object_dir / "routes"),
        "camera": camera_report,
        "six_column_panel": str(object_dir / "six_column_components.png"),
        "exploded_dir": str(object_dir / "exploded"),
        "render_gate": str(object_dir / "render_gate" / "overall_source_neg_y.png"),
    }
    _write_json(object_dir / "visual_report.json", report)
    return report


def run(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_dir).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for tag, dataset_id, object_id, angle in OBJECTS:
        print(f"[route-render] {tag} {dataset_id}::{object_id} angle={angle}", flush=True)
        reports.append(
            render_object(
                tag=tag,
                dataset_id=dataset_id,
                object_id=object_id,
                angle=angle,
                eval_dir=eval_dir,
                source_root=source_root,
                out_root=out_root,
                resolution=int(args.render_resolution),
                max_faces=int(args.render_max_faces),
            )
        )

    for name in ("all_metrics.csv", "all_route_summary.csv", "aggregate_by_route.csv"):
        src = source_root / name
        if src.is_file():
            shutil.copy2(src, out_root / name)
    lines = [
        "# Source -Y Route Visuals",
        "",
        f"source_root: `{source_root}`",
        "",
        "Camera: source/SAM3D Z-up, camera on `-Y`, looking toward `+Y`, screen up `+Z`.",
        "Route geometry and metrics are reused; no R-P/R-V/R-D/R-X generation was rerun.",
        "",
        "| object | six-column panel | exploded dir | gate |",
        "|---|---|---|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report['object_tag']} `{report['object_key']}` | `{report['six_column_panel']}` | "
            f"`{report['exploded_dir']}` | `{report['render_gate']}` |"
        )
    (out_root / "visual_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(out_root / "visual_report.json", {"reports": reports, "source_root": str(source_root)})
    print(f"[route-render] report -> {out_root / 'visual_report.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/robot/data-lab/jzh/art-gen/ee-eval/part_mesh_routes_0702/routes_compare_true_rp_0703_full"),
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("/robot/data-lab/jzh/art-gen/ee-eval/part_mesh_routes_0702/ee_eval_seed42"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/robot/data-lab/jzh/art-gen/ee-eval/part_mesh_routes_0702/routes_compare_true_rp_0703_full_front_source_neg_y"),
    )
    parser.add_argument("--render-resolution", type=int, default=256)
    parser.add_argument("--render-max-faces", type=int, default=400000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
