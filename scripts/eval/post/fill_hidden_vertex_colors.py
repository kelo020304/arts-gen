#!/usr/bin/env python3
"""Fill hidden black vertex colors on decoded ee-eval component meshes.

This is a model-free Track2 fallback.  It keeps decoded component geometry
unchanged, detects vertices not visible in the four conditioning views, and
fills their colors from nearest visible vertices on the same component.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "TRELLIS-arts"))

from scripts.eval.post.holopart_smooth import (  # noqa: E402
    SAM3D_Z_UP_TO_Y_UP,
    _load_mesh,
    _load_render_camera,
    _mesh_vertex_colors_float,
    _safe_name,
    _tile,
    render_component,
)
from scripts.eval.tasks.ee_0617_single import load_camera_matrices  # noqa: E402


DEFAULT_SUMMARIES = [
    Path("/robot/data-lab/jzh/art-gen/ee-eval/part_mesh_routes_0702/ee_eval_seed42/phyx-verse__74c7791c8ac64c55a08704202b8cbf38__angle_01__summary.json"),
    Path("/robot/data-lab/jzh/art-gen/ee-eval/part_mesh_routes_0702/ee_eval_seed42/physx-0511-drawer-door__22367__angle_00__summary.json"),
]
DEFAULT_OUT_DIR = Path("/robot/data-lab/jzh/art-gen/ee-eval/track2_hidden_color_fill_0703")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mesh_items(summary: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    body = summary.get("mujoco_body_mesh")
    if body:
        out.append({**body, "label": "body_without_parts"})
    for item in summary.get("mujoco_part_meshes") or []:
        out.append(dict(item))
    if not out:
        raise ValueError(f"{summary.get('summary_path') or summary.get('mesh_png')}: no mujoco component meshes")
    return out


def _condition_view_indices(summary: dict[str, Any]) -> list[int]:
    cond = ((summary.get("slat_stage") or {}).get("condition") or {})
    view_indices = cond.get("view_indices")
    if not isinstance(view_indices, list) or len(view_indices) != 4:
        raise ValueError(f"{summary.get('mesh_png')}: expected 4 slat_stage.condition.view_indices, got {view_indices}")
    return [int(v) for v in view_indices]


def _data_root(summary: dict[str, Any]) -> Path:
    dataset_id = str(summary.get("dataset_id"))
    for item in summary.get("datasets") or []:
        if str(item.get("dataset_id")) == dataset_id and item.get("data_root"):
            return Path(str(item["data_root"]))
    image_paths = (((summary.get("slat_stage") or {}).get("condition") or {}).get("image_paths") or [])
    if image_paths:
        object_id = str(summary.get("object_id"))
        angle = int(summary.get("angle", 0))
        marker = f"/renders/{object_id}/angle_{angle}/"
        image_path = str(image_paths[0])
        if marker in image_path:
            return Path(image_path.split(marker, 1)[0])
    raise RuntimeError(f"cannot infer data root from {summary.get('mesh_png')}")


def _mask_path(data_root: Path, object_id: str, angle: int, view: int) -> Path:
    base = data_root / "renders" / object_id / f"angle_{int(angle)}"
    candidates = [
        base / "mask" / f"mask_{int(view)}.npy",
        base / "masks" / f"mask_{int(view)}.npy",
        base / "mask" / f"view_{int(view)}.npy",
        base / "masks" / f"view_{int(view)}.npy",
        base / "alpha" / f"view_{int(view)}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing mask for view {view}: tried {[str(p) for p in candidates]}")


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.asarray(np.load(path))
        if arr.ndim == 3:
            arr = arr.max(axis=-1)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected 2-D mask, got {arr.shape}")
        return arr > 0
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 0


def _project_points(vertices_y_up: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray, mask_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices_y_up, dtype=np.float64) @ SAM3D_Z_UP_TO_Y_UP.T
    homo = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
    cam = (np.asarray(extrinsic, dtype=np.float64) @ homo.T).T[:, :3]
    z = cam[:, 2]
    x = np.asarray(intrinsic)[0, 0] * (cam[:, 0] / np.maximum(z, 1.0e-8)) + np.asarray(intrinsic)[0, 2]
    y = np.asarray(intrinsic)[1, 1] * (cam[:, 1] / np.maximum(z, 1.0e-8)) + np.asarray(intrinsic)[1, 2]
    h, w = mask_shape
    px = np.round(x * (w - 1)).astype(np.int64)
    py = np.round(y * (h - 1)).astype(np.int64)
    inside = (z > 1.0e-5) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    return px, py, z, inside


def _project_visible(
    vertices_y_up: np.ndarray,
    *,
    zbuffer_vertices_y_up: np.ndarray,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    masks: list[np.ndarray],
    z_epsilon: float,
) -> np.ndarray:
    visible = np.zeros((len(vertices_y_up),), dtype=bool)
    extr_np = extrinsics.detach().float().cpu().numpy()
    intr_np = intrinsics.detach().float().cpu().numpy()
    for view_idx, mask in enumerate(masks):
        h, w = mask.shape[:2]
        px_z, py_z, z_z, inside_z = _project_points(zbuffer_vertices_y_up, extr_np[view_idx], intr_np[view_idx], (h, w))
        zbuf = np.full((h, w), np.inf, dtype=np.float64)
        zidx = np.flatnonzero(inside_z)
        if len(zidx):
            np.minimum.at(zbuf, (py_z[zidx], px_z[zidx]), z_z[zidx])
        px, py, z, inside = _project_points(vertices_y_up, extr_np[view_idx], intr_np[view_idx], (h, w))
        hit = np.zeros_like(inside)
        idx = np.flatnonzero(inside)
        if len(idx):
            near_surface = z[idx] <= (zbuf[py[idx], px[idx]] + float(z_epsilon))
            hit[idx] = mask[py[idx], px[idx]] & near_surface
        visible |= hit
    return visible


def _fill_colors(mesh: trimesh.Trimesh, visible: np.ndarray, *, dark_threshold: float) -> tuple[np.ndarray, dict[str, Any]]:
    colors = _mesh_vertex_colors_float(mesh)
    if colors is None:
        raise ValueError("mesh has no vertex colors; this fallback is color-only")
    colors = np.asarray(colors[:, :3], dtype=np.float32).copy()
    visible = np.asarray(visible, dtype=bool)
    if visible.shape != (len(mesh.vertices),):
        raise ValueError(f"visible shape mismatch: {visible.shape} vs vertices={len(mesh.vertices)}")
    luma = colors @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    hidden = ~visible
    fill_mask = hidden & (luma <= float(dark_threshold))
    source_mask = visible & (luma > float(dark_threshold) * 0.5)
    if int(source_mask.sum()) == 0:
        source_mask = visible
    if int(source_mask.sum()) == 0:
        source_mask = ~fill_mask
    if int(source_mask.sum()) == 0:
        source_mask = np.ones((len(mesh.vertices),), dtype=bool)
    if int(fill_mask.sum()) > 0:
        source_vertices = np.asarray(mesh.vertices, dtype=np.float64)[source_mask]
        source_colors = colors[source_mask]
        tree = cKDTree(source_vertices)
        _dist, idx = tree.query(np.asarray(mesh.vertices, dtype=np.float64)[fill_mask], k=1, workers=-1)
        colors[fill_mask] = source_colors[np.asarray(idx, dtype=np.int64)]
    before_hidden_dark_mean = float(luma[hidden].mean()) if int(hidden.sum()) else float("nan")
    after_luma = colors @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    stats = {
        "vertices": int(len(mesh.vertices)),
        "visible_vertices": int(visible.sum()),
        "hidden_vertices": int(hidden.sum()),
        "filled_vertices": int(fill_mask.sum()),
        "hidden_fraction": float(hidden.mean()) if len(hidden) else 0.0,
        "filled_fraction": float(fill_mask.mean()) if len(fill_mask) else 0.0,
        "luma_mean_before": float(luma.mean()),
        "luma_mean_after": float(after_luma.mean()),
        "hidden_luma_mean_before": before_hidden_dark_mean,
        "hidden_luma_mean_after": float(after_luma[hidden].mean()) if int(hidden.sum()) else float("nan"),
        "dark_threshold": float(dark_threshold),
    }
    return np.clip(colors, 0.0, 1.0), stats


def _export_colored_obj(mesh: trimesh.Trimesh, colors: np.ndarray, path: Path) -> None:
    out = mesh.copy()
    rgba = np.pad((np.clip(colors, 0.0, 1.0) * 255.0).round().astype(np.uint8), ((0, 0), (0, 1)), constant_values=255)
    out.visual.vertex_colors = rgba
    path.parent.mkdir(parents=True, exist_ok=True)
    out.export(path)


def _render_panel(before: trimesh.Trimesh, after_colors: np.ndarray, out_path: Path, *, label: str, extrinsic: Any, intrinsic: Any, resolution: int, max_faces: int) -> None:
    before_img = render_component(before, extrinsic=extrinsic, intrinsic=intrinsic, resolution=int(resolution), max_faces=int(max_faces), color_mode="color")
    after_img = render_component(
        before,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=int(max_faces),
        color_mode="color",
        vertex_colors=after_colors,
    )
    tile_w = int(resolution)
    tile_h = int(resolution) + 30
    canvas = Image.new("RGB", (tile_w * 2, tile_h), (255, 255, 255))
    canvas.paste(_tile(before_img, f"{label} before", tile_w, tile_h), (0, 0))
    canvas.paste(_tile(after_img, f"{label} color-fill", tile_w, tile_h), (tile_w, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def process_summary(summary_path: Path, out_root: Path, *, resolution: int, render_view: int, max_faces: int, dark_threshold: float) -> dict[str, Any]:
    summary = _load_json(summary_path)
    summary["summary_path"] = str(summary_path.resolve())
    prefix = f"{summary['dataset_id']}__{summary['object_id']}__angle_{int(summary['angle']):02d}"
    out_dir = out_root / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = _data_root(summary)
    view_indices = _condition_view_indices(summary)
    camera_path = data_root / "renders" / str(summary["object_id"]) / f"angle_{int(summary['angle'])}" / "camera_transforms.json"
    extrinsics, intrinsics = load_camera_matrices(camera_path, view_indices)
    masks = [_load_mask(_mask_path(data_root, str(summary["object_id"]), int(summary["angle"]), view)) for view in view_indices]
    render_extrinsic, render_intrinsic = _load_render_camera(summary, render_view=int(render_view))
    overall_item = summary.get("mujoco_overall_mesh") or {}
    overall_path = Path(str(overall_item.get("mesh_path") or ""))
    if not overall_path.is_file():
        raise FileNotFoundError(f"{summary_path}: missing mujoco overall mesh for visibility z-buffer")
    overall_mesh = _load_mesh(overall_path)
    zbuffer_vertices = np.asarray(overall_mesh.vertices, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    panels: list[Path] = []
    for item in _mesh_items(summary):
        label = str(item["label"])
        mesh_path = Path(str(item["mesh_path"]))
        mesh = _load_mesh(mesh_path)
        visible = _project_visible(
            np.asarray(mesh.vertices, dtype=np.float64),
            zbuffer_vertices_y_up=zbuffer_vertices,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            masks=masks,
            z_epsilon=0.015,
        )
        colors, stats = _fill_colors(mesh, visible, dark_threshold=float(dark_threshold))
        safe = _safe_name(label)
        after_path = out_dir / "components_color_filled" / f"{safe}.obj"
        panel_path = out_dir / "panels" / f"{safe}__before_after.png"
        _export_colored_obj(mesh, colors, after_path)
        _render_panel(mesh, colors, panel_path, label=label, extrinsic=render_extrinsic, intrinsic=render_intrinsic, resolution=resolution, max_faces=max_faces)
        panels.append(panel_path)
        rows.append(
            {
                "object": prefix,
                "component": label,
                "before_mesh": str(mesh_path.resolve()),
                "after_mesh": str(after_path.resolve()),
                "panel": str(panel_path.resolve()),
                **stats,
            }
        )
    # Overview uses all component panels at reduced size to keep browsing cheap.
    thumb_w = int(resolution)
    thumb_h = int(resolution) + 30
    cols = 2
    rows_n = int(np.ceil(len(panels) / cols))
    overview = Image.new("RGB", (thumb_w * cols * 2, thumb_h * rows_n), (255, 255, 255))
    for i, panel in enumerate(panels):
        img = Image.open(panel).convert("RGB")
        img.thumbnail((thumb_w * 2, thumb_h), Image.Resampling.LANCZOS)
        overview.paste(img, ((i % cols) * thumb_w * 2, (i // cols) * thumb_h))
    overview_path = out_dir / "before_after_overview_color_fill.png"
    overview.save(overview_path)
    _write_csv(out_dir / "metrics.csv", rows)
    report = {
        "method": "hidden_vertex_color_fill",
        "summary_path": str(summary_path.resolve()),
        "object": prefix,
        "condition_view_indices": view_indices,
        "data_root": str(data_root),
        "render_view": int(render_view),
        "resolution": int(resolution),
        "max_faces": int(max_faces),
        "dark_threshold": float(dark_threshold),
        "overview": str(overview_path.resolve()),
        "metrics": rows,
    }
    _write_json(out_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--render-view", type=int, default=0)
    parser.add_argument("--max-faces", type=int, default=120000)
    parser.add_argument("--dark-threshold", type=float, default=0.18)
    args = parser.parse_args()
    summaries = args.summary or DEFAULT_SUMMARIES
    reports = [
        process_summary(Path(path), Path(args.out_dir), resolution=int(args.resolution), render_view=int(args.render_view), max_faces=int(args.max_faces), dark_threshold=float(args.dark_threshold))
        for path in summaries
    ]
    all_rows = [row for report in reports for row in report["metrics"]]
    _write_csv(Path(args.out_dir) / "aggregate_metrics.csv", all_rows)
    md = [
        "# Track2 hidden vertex color fill",
        "",
        "Model-free fallback: geometry is unchanged; vertices invisible in all four live conditioning views and dark are recolored from nearest visible same-component vertices.",
        "",
        "| object | components | filled vertices | hidden fraction mean | overview |",
        "|---|---:|---:|---:|---|",
    ]
    for report in reports:
        rows = report["metrics"]
        filled = sum(int(row["filled_vertices"]) for row in rows)
        hidden_mean = float(np.mean([float(row["hidden_fraction"]) for row in rows])) if rows else 0.0
        md.append(f"| `{report['object']}` | {len(rows)} | {filled} | {hidden_mean:.4f} | `{report['overview']}` |")
    (Path(args.out_dir) / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_json(Path(args.out_dir) / "aggregate_report.json", {"reports": reports, "metrics": all_rows})
    print(json.dumps({"out_dir": str(Path(args.out_dir).resolve()), "objects": len(reports), "components": len(all_rows)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
