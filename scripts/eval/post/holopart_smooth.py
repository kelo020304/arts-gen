#!/usr/bin/env python3
"""Post-process EE eval component meshes with HoloPart/X-Part style smoothing.

This script is intentionally outside the ee-eval core.  It consumes
``ee_0617_single.py --export-mujoco`` outputs plus the platform voxel npz files,
builds a face-level segmentation for the whole mesh, runs a post-processor when
available, and reports before/after metrics for body_without_parts and parts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY_ROOT = Path("/robot/data-lab/jzh/art-gen/third-party-weights/post_smooth_eval")
DEFAULT_HOLOPART_ROOT = THIRD_PARTY_ROOT / "holopart"
DEFAULT_XPART_ROOT = THIRD_PARTY_ROOT / "hunyuan3d-part" / "XPart"
DEFAULT_HOLOPART_PYTHON = Path("/opt/venvs/holopart/bin/python")
DEFAULT_XPART_PYTHON = Path("/opt/venvs/xpart/bin/python")
DEFAULT_HOLOPART_WEIGHTS = DEFAULT_HOLOPART_ROOT / "pretrained_weights" / "holopart"
DEFAULT_XPART_WEIGHTS = THIRD_PARTY_ROOT / "hunyuan3d-part" / "pretrained_weights" / "hunyuan3d-part"
ARTS_GEN_SITE_PACKAGES = Path("/opt/venvs/arts-gen/lib/python3.10/site-packages")

SAM3D_Z_UP_TO_Y_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
ROTX_POS_90 = SAM3D_Z_UP_TO_Y_UP
ROTX_NEG_90 = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def _safe_name(value: str, max_len: int = 120) -> str:
    chars: list[str] = []
    for ch in str(value):
        if ch.isascii() and (ch.isalnum() or ch in ("-", "_", ".")):
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
        else:
            chars.append(f"_u{ord(ch):04x}_")
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return (out or "component")[:max_len]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {path}")
    mesh.remove_unreferenced_vertices()
    return mesh


def _mesh_stats(mesh: trimesh.Trimesh, path: Path | None = None) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return {
        "path": None if path is None else str(Path(path).resolve()),
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "bbox_min": bounds[0].tolist() if bounds.shape == (2, 3) else None,
        "bbox_max": bounds[1].tolist() if bounds.shape == (2, 3) else None,
        "is_watertight": bool(mesh.is_watertight),
    }


def _load_component_mesh(path: Path, label: str) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    if path.suffix.lower() in {".npz", ".npy"}:
        raise ValueError(f"{label}: refusing to use voxel/point array as mesh: {path}")
    mesh = _load_mesh(path)
    stats = _mesh_stats(mesh, path)
    if int(stats["n_faces"]) <= 0:
        raise ValueError(f"{label}: decoded component OBJ has no faces; refusing point/voxel input: {path}")
    print(
        "[post-smooth] before mesh "
        f"{label}: vertices={stats['n_vertices']} faces={stats['n_faces']} "
        f"bbox={stats['bbox_min']}..{stats['bbox_max']} path={path}",
        flush=True,
    )
    return mesh, stats


def _load_coords(path: Path) -> np.ndarray:
    payload = np.load(path)
    if "coords" not in payload:
        raise KeyError(f"{path}: missing coords")
    coords = np.asarray(payload["coords"], dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: coords must be Nx3, got {coords.shape}")
    return coords


def _coords_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    return set(map(tuple, np.asarray(coords, dtype=np.int64).tolist()))


def _coords_to_mask(coords: np.ndarray, *, resolution: int = 64) -> np.ndarray:
    mask = np.zeros((int(resolution), int(resolution), int(resolution)), dtype=bool)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    valid = np.all((coords >= 0) & (coords < int(resolution)), axis=1)
    if bool(valid.any()):
        idx = coords[valid]
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def _voxel_centers(coords: np.ndarray, *, resolution: int = 64) -> np.ndarray:
    return (np.asarray(coords, dtype=np.float64).reshape(-1, 3) + 0.5) / float(resolution) - 0.5


def _mesh_to_voxel_indices(vertices: np.ndarray, *, scale: float, rotation: np.ndarray, resolution: int = 64) -> np.ndarray:
    zup_norm = (np.asarray(vertices, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T) / float(scale)
    return np.floor((zup_norm + 0.5) * float(resolution)).astype(np.int64)


def _voxel_to_mesh_points(coords: np.ndarray, *, scale: float, rotation: np.ndarray, resolution: int = 64) -> np.ndarray:
    return (_voxel_centers(coords, resolution=int(resolution)) * float(scale)) @ np.asarray(rotation, dtype=np.float64)


def _candidate_scales(vertices: np.ndarray, whole_coords: np.ndarray, *, resolution: int = 64) -> list[float]:
    mesh_extent = np.ptp(np.asarray(vertices, dtype=np.float64), axis=0)
    voxel_extent = np.ptp(_voxel_centers(whole_coords, resolution=int(resolution)), axis=0)
    scales: list[float] = []
    for mesh_axis, voxel_axis in zip(mesh_extent, voxel_extent):
        if mesh_axis > 1.0e-8 and voxel_axis > 1.0e-8:
            scales.append(float(mesh_axis / voxel_axis))
    max_abs = float(np.max(np.abs(vertices))) if len(vertices) else 0.0
    if max_abs > 1.0e-8:
        scales.append(max_abs * 2.0)
    scales.extend([1.0, 2.0])
    out: list[float] = []
    for value in scales:
        if not np.isfinite(value) or value <= 0:
            continue
        for factor in (0.70, 0.85, 1.0, 1.15, 1.30):
            candidate = float(value * factor)
            if candidate > 0 and all(abs(candidate - old) > 1.0e-5 for old in out):
                out.append(candidate)
    return out


def _surface_points_for_self_check(
    mesh: trimesh.Trimesh,
    *,
    max_vertices: int,
    max_face_centers: int,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) > int(max_vertices):
        rng = np.random.default_rng(0)
        vertices = vertices[rng.choice(len(vertices), size=int(max_vertices), replace=False)]
    if len(mesh.faces):
        centers = np.asarray(mesh.triangles_center, dtype=np.float64)
        if len(centers) > int(max_face_centers):
            face_idx = np.linspace(0, len(centers) - 1, int(max_face_centers), dtype=np.int64)
            centers = centers[face_idx]
        vertices = np.concatenate([vertices, centers], axis=0)
    return vertices


def _occupancy_iou(
    surface_points: np.ndarray,
    whole_mask: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    resolution: int = 64,
) -> dict[str, Any]:
    idx = _mesh_to_voxel_indices(surface_points, scale=float(scale), rotation=rotation, resolution=int(resolution))
    valid = np.all((idx >= 0) & (idx < int(resolution)), axis=1)
    pred = np.zeros_like(whole_mask, dtype=bool)
    if bool(valid.any()):
        v = idx[valid]
        pred[v[:, 0], v[:, 1], v[:, 2]] = True
    inter = int(np.count_nonzero(pred & whole_mask))
    union = int(np.count_nonzero(pred | whole_mask))
    return {
        "iou": float(inter / union) if union else 0.0,
        "intersection": inter,
        "pred_voxels": int(np.count_nonzero(pred)),
        "gt_voxels": int(np.count_nonzero(whole_mask)),
        "valid_surface_points": int(np.count_nonzero(valid)),
        "surface_points": int(len(surface_points)),
    }


def _axis_self_check(
    mesh: trimesh.Trimesh,
    whole_coords: np.ndarray,
    *,
    resolution: int = 64,
    max_vertices: int = 200_000,
    max_face_centers: int = 200_000,
) -> dict[str, Any]:
    """Calibrate mesh-frame <-> 64^3 voxel-frame transform.

    EE meshes are exported after SAM3D's Z-up decoded vertices are converted into
    a Y-up viewer frame.  This checks the inverse mapping by projecting mesh
    surface points back into the occupied whole-object voxel grid.  The reported
    IoU is a surface occupancy proxy, not a volumetric fill metric.
    """

    surface_points = _surface_points_for_self_check(
        mesh,
        max_vertices=int(max_vertices),
        max_face_centers=int(max_face_centers),
    )
    whole_mask = _coords_to_mask(whole_coords, resolution=int(resolution))
    rotations = [
        ("sam3d_zup_to_yup", SAM3D_Z_UP_TO_Y_UP),
        ("x_pos_90", ROTX_POS_90),
        ("x_neg_90", ROTX_NEG_90),
        ("identity", np.eye(3, dtype=np.float64)),
    ]
    best: dict[str, Any] | None = None
    for scale in _candidate_scales(surface_points, whole_coords, resolution=int(resolution)):
        for name, rotation in rotations:
            stats = _occupancy_iou(
                surface_points,
                whole_mask,
                scale=float(scale),
                rotation=rotation,
                resolution=int(resolution),
            )
            item = {
                **stats,
                "name": name,
                "scale": float(scale),
                "rotation": np.asarray(rotation, dtype=np.float64).tolist(),
                "bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
                "resolution": int(resolution),
            }
            if best is None or float(item["iou"]) > float(best["iou"]):
                best = item
    assert best is not None
    return best


@dataclass
class Component:
    label: str
    role: str
    before_mesh_path: Path
    voxel_path: Path | None
    before_mesh: trimesh.Trimesh
    coords: np.ndarray | None
    mesh_stats: dict[str, Any]


def _component_index_from_label(label: str) -> int | None:
    parts = str(label).split("_")
    if len(parts) >= 2 and parts[0] == "part":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _component_list(summary_path: Path, *, assets_dir: Path | None = None) -> tuple[Path, Path, list[Component], dict[str, Any]]:
    summary = _load_json(summary_path)
    run_dir = Path(summary["run_dir"])
    if assets_dir is None:
        assets_dir = Path(summary.get("mujoco_assets_dir") or "")
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"mujoco assets dir missing; run ee_0617_single with --export-mujoco: {assets_dir}")
    whole_voxel = run_dir / "voxel.npz"
    if not whole_voxel.is_file():
        raise FileNotFoundError(f"whole voxel missing: {whole_voxel}")

    body_item = summary.get("mujoco_body_mesh") or {}
    body_path = Path(body_item.get("mesh_path") or assets_dir / "body.obj")
    if not body_path.is_file():
        raise FileNotFoundError(
            f"body_without_parts mesh missing: {body_path}; run ee_0617_single with --export-mujoco on a current "
            "summary. Refusing to use overall.obj as body because that would invalidate body metrics."
        )

    part_items = list(summary.get("mujoco_part_meshes") or [])
    components: list[Component] = []
    whole_coords = _load_coords(whole_voxel)
    part_coord_sets: list[set[tuple[int, int, int]]] = []
    for item in part_items:
        label = str(item["label"])
        part_idx = _component_index_from_label(label)
        voxel_path = None if part_idx is None else run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz"
        coords = _load_coords(voxel_path) if voxel_path is not None and voxel_path.is_file() else None
        if coords is not None:
            part_coord_sets.append(_coords_set(coords))
        mesh_path = Path(item["mesh_path"])
        if mesh_path.is_file():
            before_mesh, mesh_stats = _load_component_mesh(mesh_path, label)
            components.append(
                Component(
                    label=label,
                    role="part",
                    before_mesh_path=mesh_path,
                    voxel_path=voxel_path,
                    before_mesh=before_mesh,
                    coords=coords,
                    mesh_stats=mesh_stats,
                )
            )
    part_union = set().union(*part_coord_sets) if part_coord_sets else set()
    body_residual = np.asarray(sorted(_coords_set(whole_coords) - part_union), dtype=np.int64)
    body_recorded_coords = int((summary.get("body_without_parts") or {}).get("coords") or body_item.get("coords") or -1)
    body_is_residual = body_recorded_coords == int(body_residual.shape[0])
    body_mesh, body_mesh_stats = _load_component_mesh(body_path, "body_without_parts")
    components.insert(
        0,
        Component(
            label="body_without_parts",
            role="body",
            before_mesh_path=body_path,
            voxel_path=None,
            before_mesh=body_mesh,
            coords=body_residual,
            mesh_stats=body_mesh_stats,
        ),
    )
    meta = {
        "summary_path": str(summary_path),
        "run_dir": str(run_dir),
        "assets_dir": str(assets_dir),
        "whole_voxel": str(whole_voxel),
        "whole_coords": int(whole_coords.shape[0]),
        "body_residual_coords": int(body_residual.shape[0]),
        "body_recorded_coords": int(body_recorded_coords),
        "body_mesh_is_declared_residual": bool(body_is_residual),
        "part_count": int(len(components) - 1),
        "summary": summary,
    }
    return run_dir, whole_voxel, components, meta


def _first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _default_holopart_weights(holopart_root: Path) -> Path | None:
    return _first_existing_path(
        [
            holopart_root / "pretrained_weights" / "HoloPart",
            holopart_root / "pretrained_weights" / "holopart",
        ]
    )


def _overall_mesh_path(summary: dict[str, Any], assets_dir: Path) -> Path | None:
    for name in ("overall.obj", "overall.glb", "whole.obj", "whole.glb"):
        path = assets_dir / name
        if path.is_file():
            return path
    return None


def _build_overall_mesh(components: list[Component], out_path: Path) -> trimesh.Trimesh:
    meshes = [comp.before_mesh.copy() for comp in components if comp.before_mesh is not None]
    merged = trimesh.util.concatenate(meshes)
    merged.remove_unreferenced_vertices()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.export(out_path)
    return merged


def _nearest_face_seg(
    overall_mesh: trimesh.Trimesh,
    components: list[Component],
    *,
    transform: dict[str, Any],
    resolution: int = 64,
    min_chunk: int = 200_000,
) -> np.ndarray:
    centroids = np.asarray(overall_mesh.triangles_center, dtype=np.float64)
    labels = np.full(len(centroids), -1, dtype=np.int32)
    comp_points = []
    comp_ids = []
    for comp_idx, comp in enumerate(components):
        if comp.coords is None or len(comp.coords) == 0:
            continue
        points = _voxel_to_mesh_points(
            comp.coords,
            scale=float(transform["scale"]),
            rotation=np.asarray(transform["rotation"], dtype=np.float64),
            resolution=int(resolution),
        )
        comp_points.append(points)
        comp_ids.append(np.full(len(points), comp_idx, dtype=np.int32))
    if not comp_points:
        raise ValueError("no component voxel centers for face segmentation")
    points = np.concatenate(comp_points, axis=0)
    point_comp = np.concatenate(comp_ids, axis=0)
    tree = cKDTree(points)
    for start in range(0, len(centroids), int(min_chunk)):
        end = min(start + int(min_chunk), len(centroids))
        _dist, idx = tree.query(centroids[start:end], k=1, workers=-1)
        labels[start:end] = point_comp[idx]
    if np.any(labels < 0):
        raise RuntimeError("internal error: unassigned face labels")
    return labels


def _export_segmented_glb(overall_mesh: trimesh.Trimesh, face_seg: np.ndarray, components: list[Component], out_path: Path) -> Path:
    scene = trimesh.Scene()
    for comp_idx, comp in enumerate(components):
        face_mask = np.asarray(face_seg) == int(comp_idx)
        if not np.any(face_mask):
            continue
        sub = overall_mesh.submesh([face_mask], append=True, repair=False)
        sub.metadata["name"] = _safe_name(comp.label)
        scene.add_geometry(sub, geom_name=_safe_name(comp.label))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)
    return out_path


def _export_component_scene_glb(components: list[Component], out_path: Path) -> Path:
    """Export one GLB scene with one named submesh per decoded ee-eval component.

    This is HoloPart's official input shape: a single whole-object scene split
    into submeshes.  The meshes remain in the original ee-eval world frame; no
    voxel points or per-component point clouds are used.
    """

    scene = trimesh.Scene()
    for comp in components:
        mesh = comp.before_mesh.copy()
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"{comp.label}: cannot export empty component mesh to HoloPart GLB")
        mesh.metadata["name"] = _safe_name(comp.label)
        scene.add_geometry(mesh, geom_name=_safe_name(comp.label))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)
    return out_path


def _laplacian_smooth(mesh: trimesh.Trimesh, *, iterations: int, lamb: float) -> trimesh.Trimesh:
    out = mesh.copy()
    vertices = np.asarray(out.vertices, dtype=np.float64).copy()
    faces = np.asarray(out.faces, dtype=np.int64)
    adjacency: list[set[int]] = [set() for _ in range(len(vertices))]
    for tri in faces:
        a, b, c = map(int, tri)
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    for _ in range(int(iterations)):
        new_vertices = vertices.copy()
        for idx, nbrs in enumerate(adjacency):
            if not nbrs:
                continue
            nbr_mean = vertices[list(nbrs)].mean(axis=0)
            new_vertices[idx] = vertices[idx] + float(lamb) * (nbr_mean - vertices[idx])
        vertices = new_vertices
    out.vertices = vertices
    out.remove_unreferenced_vertices()
    return out


def _run_fallback(components: list[Component], out_dir: Path, *, iterations: int, lamb: float) -> tuple[dict[str, Path], dict[str, float]]:
    after_paths: dict[str, Path] = {}
    times: dict[str, float] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for comp in components:
        started = time.time()
        after = _laplacian_smooth(comp.before_mesh, iterations=int(iterations), lamb=float(lamb))
        out_path = out_dir / f"{_safe_name(comp.label)}.obj"
        after.export(out_path)
        after_paths[comp.label] = out_path
        times[comp.label] = float(time.time() - started)
    return after_paths, times


def _run_holopart(
    segmented_glb: Path,
    out_dir: Path,
    *,
    holopart_root: Path,
    holopart_python: Path,
    weights_dir: Path | None,
    num_inference_steps: int,
    guidance_scale: float,
    batch_size: int,
    timeout: int,
    use_flash_decoder: bool = True,
    normalize_input: bool = False,
) -> tuple[dict[str, Path], dict[str, float], dict[str, Any]]:
    status: dict[str, Any] = {
        "enabled": True,
        "available": False,
        "reason": "",
        "command": None,
    }
    if not holopart_root.is_dir():
        status["reason"] = f"HoloPart root missing: {holopart_root}"
        return {}, {}, status
    if not holopart_python.is_file():
        status["reason"] = f"HoloPart venv python missing: {holopart_python}"
        return {}, {}, status
    if weights_dir is None:
        weights_dir = _default_holopart_weights(holopart_root)
    if weights_dir is not None and not Path(weights_dir).exists():
        status["reason"] = f"HoloPart weights missing: {weights_dir}"
        return {}, {}, status
    if weights_dir is None:
        status["reason"] = f"HoloPart weights missing under {holopart_root / 'pretrained_weights'}"
        return {}, {}, status
    script = holopart_root / "scripts" / "inference_holopart.py"
    if not script.is_file():
        status["reason"] = f"HoloPart inference utilities missing: {script}"
        return {}, {}, status
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pythonpath_parts = [str(holopart_root), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = ":".join(part for part in pythonpath_parts if part)
    env["HOLOPART_MESH_INPUT"] = str(segmented_glb)
    env["HOLOPART_OUTPUT_DIR"] = str(out_dir)
    env["HOLOPART_WEIGHTS_DIR"] = str(weights_dir)
    env["HOLOPART_STEPS"] = str(int(num_inference_steps))
    env["HOLOPART_GUIDANCE_SCALE"] = str(float(guidance_scale))
    env["HOLOPART_BATCH_SIZE"] = str(int(batch_size))
    env["HOLOPART_USE_FLASH_DECODER"] = "1" if bool(use_flash_decoder) else "0"
    env["HOLOPART_NORMALIZE_INPUT"] = "1" if bool(normalize_input) else "0"
    inline = r"""
import os
import sys
import types

import numpy as np
from scipy.spatial import cKDTree

import torch

if "torch_cluster" not in sys.modules:
    torch_cluster = types.ModuleType("torch_cluster")

    def nearest(x, y):
        x_np = x.detach().float().cpu().numpy()
        y_np = y.detach().float().cpu().numpy()
        _dist, idx = cKDTree(y_np).query(x_np, k=1, workers=-1)
        return torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device=x.device)

    def fps(x, batch=None, ratio=0.5, random_start=True, batch_size=None):
        points = x.detach().float()
        if batch is None:
            batch = torch.zeros((points.shape[0],), dtype=torch.long, device=points.device)
        out = []
        for batch_id in torch.unique(batch).detach().cpu().tolist():
            local = torch.nonzero(batch == int(batch_id), as_tuple=False).flatten()
            if local.numel() == 0:
                continue
            count = max(1, int(np.ceil(float(local.numel()) * float(ratio))))
            count = min(count, int(local.numel()))
            local_points = points[local]
            selected = [0 if not random_start else int((int(batch_id) * 9973) % int(local.numel()))]
            min_dist = torch.full((local_points.shape[0],), float("inf"), device=points.device)
            for _ in range(1, count):
                last = local_points[selected[-1]].unsqueeze(0)
                dist = torch.sum((local_points - last) ** 2, dim=1)
                min_dist = torch.minimum(min_dist, dist)
                selected.append(int(torch.argmax(min_dist).detach().cpu().item()))
            out.append(local[torch.as_tensor(selected, dtype=torch.long, device=local.device)])
        if not out:
            return torch.empty((0,), dtype=torch.long, device=points.device)
        return torch.cat(out, dim=0)

    torch_cluster.nearest = nearest
    torch_cluster.fps = fps
    sys.modules["torch_cluster"] = torch_cluster

from holopart.pipelines.pipeline_holopart import HoloPartPipeline
from scripts.inference_holopart import prepare_data, run_holopart

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
weights_dir = os.environ["HOLOPART_WEIGHTS_DIR"]
mesh_input = os.environ["HOLOPART_MESH_INPUT"]
output_dir = os.environ["HOLOPART_OUTPUT_DIR"]
steps = int(os.environ.get("HOLOPART_STEPS", "50"))
guidance = float(os.environ.get("HOLOPART_GUIDANCE_SCALE", "3.5"))
batch_size = int(os.environ.get("HOLOPART_BATCH_SIZE", "8"))
use_flash_decoder = os.environ.get("HOLOPART_USE_FLASH_DECODER", "1") == "1"
normalize_input = os.environ.get("HOLOPART_NORMALIZE_INPUT", "0") == "1"
os.makedirs(output_dir, exist_ok=True)
pipe = HoloPartPipeline.from_pretrained(weights_dir).to(device, dtype)
if normalize_input:
    import trimesh
    scene_in = trimesh.load(mesh_input, force="scene", process=False)
    bounds = scene_in.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = max(float((bounds[1] - bounds[0]).max()) / 1.9, 1.0e-6)
    scene_out = trimesh.Scene()
    for name, geom in scene_in.geometry.items():
        mesh = geom.copy()
        mesh.apply_translation(-center)
        mesh.apply_scale(1.0 / scale)
        scene_out.add_geometry(mesh, geom_name=name)
    normalized_path = os.path.join(output_dir, "normalized_input.glb")
    scene_out.export(normalized_path)
    mesh_input = normalized_path
parts_data = prepare_data(mesh_input, device=device)
scene = run_holopart(
    pipe,
    batch=parts_data,
    batch_size=batch_size,
    seed=42,
    num_inference_steps=steps,
    guidance_scale=guidance,
    use_flash_decoder=use_flash_decoder,
    device=device,
)
if normalize_input:
    for geom in scene.geometry.values():
        geom.apply_scale(scale)
        geom.apply_translation(center)
scene.export(os.path.join(output_dir, "output.glb"))
"""
    command = [
        str(holopart_python),
        "-c",
        inline,
    ]
    status["command"] = command
    status["weights_dir"] = str(weights_dir)
    status["num_inference_steps"] = int(num_inference_steps)
    status["guidance_scale"] = float(guidance_scale)
    status["batch_size"] = int(batch_size)
    status["use_flash_decoder"] = bool(use_flash_decoder)
    status["normalize_input"] = bool(normalize_input)
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=str(holopart_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(timeout),
        check=False,
    )
    status["available"] = proc.returncode == 0
    status["returncode"] = int(proc.returncode)
    status["seconds"] = float(time.time() - started)
    status["log"] = proc.stdout[-8000:]
    if proc.returncode != 0:
        status["reason"] = "HoloPart command failed"
        return {}, {}, status
    output_glb = out_dir / "output.glb"
    if not output_glb.is_file():
        status["available"] = False
        status["reason"] = f"HoloPart output missing: {output_glb}"
        return {}, {}, status
    scene = trimesh.load(output_glb, force="scene", process=False)
    after_paths: dict[str, Path] = {}
    times: dict[str, float] = {}
    denom = max(1, len(scene.geometry))
    for idx, (name, geom) in enumerate(scene.geometry.items()):
        mesh = geom if isinstance(geom, trimesh.Trimesh) else trimesh.util.concatenate(tuple(geom.dump()))
        label = str(mesh.metadata.get("name") or name or f"component_{idx:02d}")
        out_path = out_dir / f"{_safe_name(label)}.obj"
        mesh.export(out_path)
        after_paths[label] = out_path
        times[label] = float(status["seconds"]) / denom
    return after_paths, times, status


def _run_xpart(
    overall_mesh_path: Path,
    face_seg_labels: Path,
    components: list[Component],
    out_dir: Path,
    *,
    xpart_root: Path,
    xpart_python: Path,
    weights_dir: Path,
    timeout: int,
    steps: int,
    octree_resolution: int,
    num_chunks: int,
    p3sam_point_num: int,
    p3sam_prompt_num: int,
    p3sam_batch_size: int,
    progress: bool,
    seg_mode: str,
) -> tuple[dict[str, Path], dict[str, float], dict[str, Any]]:
    status: dict[str, Any] = {
        "enabled": True,
        "available": False,
        "reason": "",
        "command": None,
    }
    runner = REPO_ROOT / "scripts" / "eval" / "post" / "xpart_run.py"
    if not runner.is_file():
        status["reason"] = f"X-Part runner missing: {runner}"
        return {}, {}, status
    if not xpart_root.is_dir():
        status["reason"] = f"X-Part root missing: {xpart_root}"
        return {}, {}, status
    if not xpart_python.is_file():
        status["reason"] = f"X-Part venv python missing: {xpart_python}"
        return {}, {}, status
    if not weights_dir.exists():
        status["reason"] = f"X-Part weights missing: {weights_dir}"
        return {}, {}, status
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(xpart_python),
        str(runner),
        "--mesh-input",
        str(overall_mesh_path),
        "--out-dir",
        str(out_dir),
        "--xpart-root",
        str(xpart_root),
        "--weights",
        str(weights_dir),
        "--steps",
        str(int(steps)),
        "--octree-resolution",
        str(int(octree_resolution)),
        "--num-chunks",
        str(int(num_chunks)),
        "--p3sam-point-num",
        str(int(p3sam_point_num)),
        "--p3sam-prompt-num",
        str(int(p3sam_prompt_num)),
        "--p3sam-batch-size",
        str(int(p3sam_batch_size)),
    ]
    if str(seg_mode) in {"seg_surface", "legacy_bbox"}:
        command.extend(["--component-labels", str(face_seg_labels)])
        for comp in components:
            command.extend(["--component-mesh", str(comp.before_mesh_path)])
        if str(seg_mode) == "legacy_bbox":
            command.extend(["--conditioning-mode", "legacy_bbox"])
    elif str(seg_mode) != "p3sam":
        raise ValueError(f"unsupported X-Part segmentation mode: {seg_mode}")
    command.append("--disable-dataparallel")
    command.append("--progress" if progress else "--no-progress")
    status["command"] = command
    status["weights_dir"] = str(weights_dir)
    status["steps"] = int(steps)
    status["octree_resolution"] = int(octree_resolution)
    status["num_chunks"] = int(num_chunks)
    status["p3sam_point_num"] = int(p3sam_point_num)
    status["p3sam_prompt_num"] = int(p3sam_prompt_num)
    status["p3sam_batch_size"] = int(p3sam_batch_size)
    status["seg_mode"] = str(seg_mode)
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part)
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(timeout),
        check=False,
    )
    status["available"] = proc.returncode == 0
    status["returncode"] = int(proc.returncode)
    status["seconds"] = float(time.time() - started)
    status["log"] = proc.stdout[-12000:]
    if proc.returncode != 0:
        status["reason"] = "X-Part command failed"
        return {}, {}, status
    report_path = out_dir / "report.json"
    if not report_path.is_file():
        status["available"] = False
        status["reason"] = f"X-Part report missing: {report_path}"
        return {}, {}, status
    report = _load_json(report_path)
    component_paths = report.get("component_paths") or {}
    after_paths = {str(label): Path(path) for label, path in component_paths.items() if Path(path).is_file()}
    denom = max(1, len(after_paths))
    per_component = float(report.get("inference_seconds") or status["seconds"]) / denom
    times = {label: per_component for label in after_paths}
    status["report"] = report
    status["mode"] = report.get("mode")
    status["output_glb"] = report.get("output_glb")
    return after_paths, times, status


def _sample_points(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.asarray(mesh.vertices, dtype=np.float64)
    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        pts = mesh.sample(int(n))
    finally:
        np.random.set_state(rng_state)
    return np.asarray(pts, dtype=np.float64)


def _chamfer_to_overall(mesh: trimesh.Trimesh, overall_points: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    pts = _sample_points(mesh, int(samples), int(seed))
    if len(pts) == 0 or len(overall_points) == 0:
        return {
            "after_to_overall_mean": float("nan"),
            "after_to_overall_p95": float("nan"),
            "after_to_overall_p99": float("nan"),
            "after_to_overall_max": float("nan"),
        }
    tree = cKDTree(overall_points)
    dist, _idx = tree.query(pts, k=1, workers=-1)
    return {
        "after_to_overall_mean": float(np.mean(dist)),
        "after_to_overall_p95": float(np.quantile(dist, 0.95)),
        "after_to_overall_p99": float(np.quantile(dist, 0.99)),
        "after_to_overall_max": float(np.max(dist)),
    }


def _surface_distance_metrics(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    samples: int,
    seed: int,
    prefix: str,
    epsilons: tuple[float, ...] = (0.01, 0.02),
) -> dict[str, float]:
    """Approximate one-way surface distance by dense surface sampling.

    This is intentionally asymmetric.  ``before_to_after`` is the completeness
    check: if an after mesh deletes a panel and leaves only a frame, samples on
    the missing before panel become far from the after surface and coverage
    drops even when watertight and after->overall distances look good.
    """

    source_pts = _sample_points(source, int(samples), int(seed))
    target_pts = _sample_points(target, int(samples), int(seed) + 1)
    base = {
        f"{prefix}_mean": float("nan"),
        f"{prefix}_p95": float("nan"),
        f"{prefix}_p99": float("nan"),
        f"{prefix}_max": float("nan"),
    }
    for eps in epsilons:
        base[f"{prefix}_coverage_{str(eps).replace('.', 'p')}"] = float("nan")
    if len(source_pts) == 0 or len(target_pts) == 0:
        return base
    tree = cKDTree(target_pts)
    dist, _idx = tree.query(source_pts, k=1, workers=-1)
    out = {
        f"{prefix}_mean": float(np.mean(dist)),
        f"{prefix}_p95": float(np.quantile(dist, 0.95)),
        f"{prefix}_p99": float(np.quantile(dist, 0.99)),
        f"{prefix}_max": float(np.max(dist)),
    }
    for eps in epsilons:
        out[f"{prefix}_coverage_{str(eps).replace('.', 'p')}"] = float(np.mean(dist <= float(eps)))
    return out


def _smoothness(mesh: trimesh.Trimesh) -> dict[str, float]:
    out = {
        "mean_dihedral_rad": float("nan"),
        "normal_variance": float("nan"),
        "face_count": int(len(mesh.faces)),
        "vertex_count": int(len(mesh.vertices)),
        "is_watertight": bool(mesh.is_watertight),
    }
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    if len(normals):
        mean = normals.mean(axis=0, keepdims=True)
        out["normal_variance"] = float(np.mean(np.sum((normals - mean) ** 2, axis=1)))
    adjacency = getattr(mesh, "face_adjacency", np.empty((0, 2), dtype=np.int64))
    if len(adjacency):
        dots = np.einsum("ij,ij->i", normals[adjacency[:, 0]], normals[adjacency[:, 1]])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        out["mean_dihedral_rad"] = float(np.mean(angles))
    return out


def _bbox_overlap_volume(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    amin, amax = np.asarray(a.bounds[0]), np.asarray(a.bounds[1])
    bmin, bmax = np.asarray(b.bounds[0]), np.asarray(b.bounds[1])
    overlap = np.maximum(0.0, np.minimum(amax, bmax) - np.maximum(amin, bmin))
    return float(np.prod(overlap))


def _voxel_overlap_volume(
    meshes: list[tuple[str, trimesh.Trimesh]],
    *,
    pitch: float,
    max_faces: int,
    mode: str,
) -> dict[str, Any]:
    if str(mode) == "skip":
        return {
            "mode": "skip",
            "pitch": float(pitch),
            "total_overlap_voxels": None,
            "max_pair_overlap_voxels": None,
            "pairs": [],
        }
    if str(mode) == "bbox":
        pairs = []
        total_volume = 0.0
        max_volume = 0.0
        for i in range(len(meshes)):
            for j in range(i + 1, len(meshes)):
                volume = _bbox_overlap_volume(meshes[i][1], meshes[j][1])
                total_volume += volume
                max_volume = max(max_volume, volume)
                if volume > 0:
                    pairs.append({"a": meshes[i][0], "b": meshes[j][0], "bbox_overlap_volume": float(volume)})
        return {
            "mode": "bbox",
            "pitch": float(pitch),
            "total_bbox_overlap_volume": float(total_volume),
            "max_pair_bbox_overlap_volume": float(max_volume),
            "total_overlap_voxels": None,
            "max_pair_overlap_voxels": None,
            "pairs": pairs,
        }

    voxels: list[tuple[str, set[tuple[int, int, int]]]] = []
    for label, mesh in meshes:
        if len(mesh.faces) > int(max_faces):
            mesh = mesh.copy()
            face_idx = np.linspace(0, len(mesh.faces) - 1, int(max_faces), dtype=np.int64)
            mesh.update_faces(face_idx)
            mesh.remove_unreferenced_vertices()
        try:
            vg = mesh.voxelized(pitch=float(pitch))
            pts = np.asarray(vg.points, dtype=np.float64)
            keys = set(map(tuple, np.rint(pts / float(pitch)).astype(np.int64).tolist()))
        except Exception:
            keys = set()
        voxels.append((label, keys))
    pairs = []
    total = 0
    max_pair = 0
    for i in range(len(voxels)):
        for j in range(i + 1, len(voxels)):
            count = len(voxels[i][1] & voxels[j][1])
            volume = float(count * (float(pitch) ** 3))
            total += count
            max_pair = max(max_pair, count)
            if count:
                pairs.append({
                    "a": voxels[i][0],
                    "b": voxels[j][0],
                    "overlap_voxels": int(count),
                    "overlap_volume": volume,
                })
    return {
        "mode": "voxel",
        "pitch": float(pitch),
        "total_overlap_voxels": int(total),
        "max_pair_overlap_voxels": int(max_pair),
        "pairs": pairs,
    }


def _display_mesh(
    mesh: trimesh.Trimesh,
    max_faces: int,
    vertex_colors: np.ndarray | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray | None]:
    if int(max_faces) <= 0 or len(mesh.faces) <= int(max_faces):
        return mesh, vertex_colors
    out = mesh.copy()
    if vertex_colors is not None:
        colors = np.asarray(vertex_colors, dtype=np.float32)
        if colors.ndim == 2 and colors.shape[0] == len(out.vertices) and colors.shape[1] >= 3:
            rgba = np.pad(
                (np.clip(colors[:, :3], 0.0, 1.0) * 255.0).round().astype(np.uint8),
                ((0, 0), (0, 1)),
                constant_values=255,
            )
            out.visual.vertex_colors = rgba
    face_idx = np.linspace(0, len(out.faces) - 1, int(max_faces), dtype=np.int64)
    out.update_faces(face_idx)
    out.remove_unreferenced_vertices()
    return out, _mesh_vertex_colors_float(out)


def _load_render_camera(summary: dict[str, Any], *, render_view: int) -> tuple[Any, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "TRELLIS-arts"))
    from scripts.eval.tasks.ee_0617_single import load_camera_matrices

    dataset_id = str(summary.get("dataset_id"))
    object_id = str(summary.get("object_id"))
    angle = int(summary.get("angle", 0))
    data_root = None
    for item in summary.get("datasets") or []:
        if str(item.get("dataset_id")) == dataset_id:
            data_root = item.get("data_root")
            break
    if not data_root:
        # Most current summaries include absolute image paths in the SLat condition.
        image_paths = (((summary.get("slat_stage") or {}).get("condition") or {}).get("image_paths") or [])
        if image_paths:
            marker = f"/renders/{object_id}/angle_{angle}/"
            image_path = str(image_paths[0])
            if marker in image_path:
                data_root = image_path.split(marker, 1)[0]
    if not data_root:
        raise RuntimeError(f"cannot infer data_root for mesh render camera from {summary.get('summary_path')}")
    camera_path = Path(data_root) / "renders" / object_id / f"angle_{angle}" / "camera_transforms.json"
    extrinsics, intrinsics = load_camera_matrices(camera_path, [int(render_view)])
    return extrinsics[0], intrinsics[0]


def _mesh_vertex_colors_float(mesh: trimesh.Trimesh) -> np.ndarray | None:
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None:
        return None
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != len(mesh.vertices) or colors.shape[1] < 3:
        return None
    colors = colors[:, :3].astype(np.float32)
    if colors.size and colors.max(initial=0.0) > 1.0:
        colors /= 255.0
    return np.clip(colors, 0.0, 1.0)


def _mesh_face_colors_float(mesh: trimesh.Trimesh) -> np.ndarray | None:
    colors = getattr(mesh.visual, "face_colors", None)
    if colors is None:
        return None
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != len(mesh.faces) or colors.shape[1] < 3:
        return None
    colors = colors[:, :3].astype(np.float32)
    if colors.size and colors.max(initial=0.0) > 1.0:
        colors /= 255.0
    return np.clip(colors, 0.0, 1.0)


def _fallback_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    vertex_colors = _mesh_vertex_colors_float(mesh)
    if vertex_colors is not None:
        return vertex_colors
    face_colors = _mesh_face_colors_float(mesh)
    if face_colors is not None:
        out = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
        counts = np.zeros((len(mesh.vertices), 1), dtype=np.float32)
        for corner in range(3):
            idx = np.asarray(mesh.faces[:, corner], dtype=np.int64)
            np.add.at(out, idx, face_colors)
            np.add.at(counts, idx, 1.0)
        valid = counts[:, 0] > 0
        out[valid] /= counts[valid]
        out[~valid] = np.asarray([0.7, 0.7, 0.7], dtype=np.float32)
        return np.clip(out, 0.0, 1.0)
    return np.tile(np.asarray([[0.7, 0.7, 0.7]], dtype=np.float32), (len(mesh.vertices), 1))


def _transfer_vertex_colors(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    k_faces: int = 16,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Transfer source vertex colors to target vertices by nearest source surface.

    ``trimesh.proximity.closest_point`` requires rtree in this environment.  This
    uses a face-centroid KDTree to choose local triangle candidates, then computes
    closest points and barycentric colors on those candidate triangles.
    """

    if len(target.vertices) == 0:
        return np.empty((0, 3), dtype=np.float32)
    source_colors = _fallback_vertex_colors(source)
    if len(source.faces) == 0 or len(source.vertices) == 0:
        return np.tile(np.asarray([[0.7, 0.7, 0.7]], dtype=np.float32), (len(target.vertices), 1))
    triangles = np.asarray(source.triangles, dtype=np.float64)
    face_centers = np.asarray(source.triangles_center, dtype=np.float64)
    tree = cKDTree(face_centers)
    k = min(max(1, int(k_faces)), len(triangles))
    out = np.empty((len(target.vertices), 3), dtype=np.float32)
    target_vertices = np.asarray(target.vertices, dtype=np.float64)
    import trimesh.triangles as tri

    for start in range(0, len(target_vertices), int(chunk_size)):
        end = min(start + int(chunk_size), len(target_vertices))
        points = target_vertices[start:end]
        _dist, candidate_idx = tree.query(points, k=k, workers=-1)
        candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
        if candidate_idx.ndim == 1:
            candidate_idx = candidate_idx[:, None]
        best_dist2 = np.full((len(points),), np.inf, dtype=np.float64)
        best_color = np.zeros((len(points), 3), dtype=np.float32)
        for candidate_col in range(candidate_idx.shape[1]):
            face_idx = candidate_idx[:, candidate_col]
            cand_triangles = triangles[face_idx]
            closest = tri.closest_point(cand_triangles, points)
            dist2 = np.sum((closest - points) ** 2, axis=1)
            improve = dist2 < best_dist2
            if not bool(np.any(improve)):
                continue
            bary = tri.points_to_barycentric(cand_triangles[improve], closest[improve], method="cramer")
            bary = np.nan_to_num(bary, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0)
            bary = np.clip(bary, 0.0, 1.0)
            denom = bary.sum(axis=1, keepdims=True)
            bary = np.divide(bary, np.maximum(denom, 1.0e-8))
            face_vertices = np.asarray(source.faces[face_idx[improve]], dtype=np.int64)
            colors = np.einsum("ij,ijk->ik", bary.astype(np.float32), source_colors[face_vertices])
            best_color[improve] = colors.astype(np.float32)
            best_dist2[improve] = dist2[improve]
        out[start:end] = np.clip(best_color, 0.0, 1.0)
    return out


def _trimesh_to_mesh_extract(
    mesh: trimesh.Trimesh,
    *,
    device: str,
    vertex_colors: np.ndarray | None = None,
    convert_y_up_to_z_up: bool = True,
) -> Any:
    sys.path.insert(0, str(REPO_ROOT / "TRELLIS-arts"))
    from trellis.representations.mesh import MeshExtractResult
    import torch

    vertices_np = np.asarray(mesh.vertices, dtype=np.float32)
    if convert_y_up_to_z_up:
        # ee-eval renders decoder meshes in SAM3D Z-up, while exported OBJ/GLB
        # assets are written through slat_asset_writer as Y-up.  Post panels
        # consume those exported assets, so rotate them back before using the
        # exact ee-eval MeshRenderer camera.
        vertices_np = vertices_np @ SAM3D_Z_UP_TO_Y_UP.T
    vertices_np = np.asarray(vertices_np, dtype=np.float32)
    vertices = torch.as_tensor(vertices_np, device=device)
    faces = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int64), device=device)
    attrs = None
    if vertex_colors is None:
        vertex_colors = _mesh_vertex_colors_float(mesh)
    if vertex_colors is not None:
        colors = np.asarray(vertex_colors, dtype=np.float32)
        if colors.ndim != 2 or colors.shape[0] != len(mesh.vertices) or colors.shape[1] < 3:
            raise ValueError(f"vertex colors must be Nx3 for render, got {colors.shape}")
        attrs = torch.as_tensor(np.clip(colors[:, :3], 0.0, 1.0), device=device)
    return MeshExtractResult(vertices, faces, vertex_attrs=attrs, res=64)


def render_component(
    mesh: trimesh.Trimesh,
    *,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
    max_faces: int,
    vertex_colors: np.ndarray | None = None,
    color_mode: str = "color",
    convert_y_up_to_z_up: bool = True,
) -> Image.Image:
    """Render a component with the same MeshRenderer path as ee-eval."""

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("render_component received an empty mesh")
    import torch
    from trellis.renderers.mesh_renderer import MeshRenderer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("TRELLIS MeshRenderer requires CUDA for this eval render; refusing scatter fallback")
    display, display_colors = _display_mesh(mesh, int(max_faces), vertex_colors=vertex_colors)
    render_mesh = _trimesh_to_mesh_extract(
        display,
        device=device,
        vertex_colors=display_colors,
        convert_y_up_to_z_up=bool(convert_y_up_to_z_up),
    )
    renderer = MeshRenderer({"resolution": int(resolution), "near": 0.1, "far": 10.0, "ssaa": 1})
    with torch.no_grad():
        if str(color_mode) == "normal" or getattr(render_mesh, "vertex_attrs", None) is None:
            ret = renderer.render(
                render_mesh,
                extrinsic.to(device).float(),
                intrinsic.to(device).float(),
                return_types=["normal", "mask"],
            )
            color = ret["normal"].detach().float().cpu().clamp(0, 1)
        else:
            ret = renderer.render(
                render_mesh,
                extrinsic.to(device).float(),
                intrinsic.to(device).float(),
                return_types=["color", "mask"],
            )
            color = ret["color"].detach().float().cpu().clamp(0, 1)
    mask = ret["mask"].detach().float().cpu().clamp(0, 1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    color = color * mask + torch.full_like(color, 0.94) * (1.0 - mask)
    arr = (color.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _tile(image: Image.Image, label: str, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    body_h = max(1, int(height) - 30)
    image.thumbnail((int(width), body_h), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, int(width), 30), fill=(0, 0, 0))
    draw.text((8, 9), label[:96], fill=(255, 255, 255))
    tile.paste(image, ((int(width) - image.width) // 2, 30 + (body_h - image.height) // 2))
    return tile


def _write_component_panels(
    components: list[Component],
    after_meshes: dict[str, trimesh.Trimesh],
    out_dir: Path,
    *,
    method: str,
    max_faces: int,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
) -> None:
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for comp in components:
        after = after_meshes.get(comp.label)
        if after is None:
            continue
        after_colors = _transfer_vertex_colors(comp.before_mesh, after)
        before_img = render_component(
            comp.before_mesh,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(resolution),
            max_faces=int(max_faces),
            color_mode="color",
        )
        after_img = render_component(
            after,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(resolution),
            max_faces=int(max_faces),
            vertex_colors=after_colors,
            color_mode="color",
        )
        canvas = Image.new("RGB", (int(resolution) * 2, int(resolution) + 30), (255, 255, 255))
        canvas.paste(_tile(before_img, f"before {comp.label}", int(resolution), int(resolution) + 30), (0, 0))
        canvas.paste(_tile(after_img, f"after {method}", int(resolution), int(resolution) + 30), (int(resolution), 0))
        canvas.save(panel_dir / f"{_safe_name(comp.label)}__before_after.png")


def _write_overview_panel(
    components: list[Component],
    after_meshes: dict[str, trimesh.Trimesh],
    out_path: Path,
    *,
    method: str,
    max_faces: int,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
    max_components: int = 8,
) -> None:
    shown = [comp for comp in components if comp.label in after_meshes][: int(max_components)]
    if not shown:
        return
    rows = len(shown)
    canvas = Image.new("RGB", (int(resolution) * 2, rows * (int(resolution) + 30)), (255, 255, 255))
    for row_idx, comp in enumerate(shown):
        after = after_meshes[comp.label]
        after_colors = _transfer_vertex_colors(comp.before_mesh, after)
        before_img = render_component(
            comp.before_mesh,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(resolution),
            max_faces=int(max_faces),
            color_mode="color",
        )
        after_img = render_component(
            after,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(resolution),
            max_faces=int(max_faces),
            vertex_colors=after_colors,
            color_mode="color",
        )
        y = row_idx * (int(resolution) + 30)
        canvas.paste(_tile(before_img, f"before {comp.label}", int(resolution), int(resolution) + 30), (0, y))
        canvas.paste(_tile(after_img, f"after {method}", int(resolution), int(resolution) + 30), (int(resolution), y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _exploded_meshes(
    components: list[Component],
    after_meshes: dict[str, trimesh.Trimesh],
    *,
    scale: float = 0.35,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    meshes: list[trimesh.Trimesh] = []
    colors: list[np.ndarray] = []
    selected = [comp for comp in components if comp.label in after_meshes]
    if not selected:
        raise ValueError("no after meshes for exploded view")
    centers = []
    for comp in selected:
        mesh = after_meshes[comp.label]
        centers.append(np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0))
    centers_arr = np.stack(centers, axis=0)
    global_center = centers_arr.mean(axis=0)
    extent = float(np.max(np.ptp(centers_arr, axis=0))) if len(centers_arr) > 1 else 1.0
    if not np.isfinite(extent) or extent <= 1.0e-6:
        extent = 1.0
    for comp, center in zip(selected, centers_arr, strict=True):
        mesh = after_meshes[comp.label].copy()
        direction = center - global_center
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-6:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            norm = 1.0
        mesh.apply_translation((direction / norm) * float(scale) * extent)
        color = _transfer_vertex_colors(comp.before_mesh, mesh)
        colors.append(color)
        meshes.append(mesh)
    merged = trimesh.util.concatenate(meshes)
    merged_colors = np.concatenate(colors, axis=0) if colors else None
    if merged_colors is None or len(merged_colors) != len(merged.vertices):
        merged_colors = _fallback_vertex_colors(merged)
    return merged, merged_colors


def _write_exploded_panel(
    components: list[Component],
    after_meshes: dict[str, trimesh.Trimesh],
    out_path: Path,
    *,
    max_faces: int,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
) -> None:
    if not after_meshes:
        return
    exploded, colors = _exploded_meshes(components, after_meshes)
    image = render_component(
        exploded,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=int(max_faces),
        vertex_colors=colors,
        color_mode="color",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _tile(image, "after exploded components", int(resolution), int(resolution) + 30).save(out_path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _markdown_report(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {report['method']} post-process report",
        "",
        f"summary: `{report['summary_path']}`",
        f"out_dir: `{report['out_dir']}`",
        f"holopart_input_glb: `{report.get('holopart_input_glb')}`",
        f"body_mesh_is_declared_residual: `{report['body_mesh_is_declared_residual']}`",
        f"axis_iou: `{report['axis_self_check']['iou']:.4f}`",
        f"intersections total voxels: `{report['after_intersection']['total_overlap_voxels']}`",
        "",
        "## Components",
        "",
        "| component | role | before_dihedral | after_dihedral | after_overall_mean | before_to_after_p95 | coverage@0.01 | coverage@0.02 | bidir_p95 | before_water | after_water | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {component} | {role} | {before_mean_dihedral_rad:.4f} | {after_mean_dihedral_rad:.4f} | "
            "{after_to_overall_mean:.5f} | {before_to_after_p95:.5f} | "
            "{before_to_after_coverage_0p01:.5f} | {before_to_after_coverage_0p02:.5f} | "
            "{bidirectional_chamfer_p95_max:.5f} | {before_is_watertight} | {after_is_watertight} | "
            "{seconds:.3f} |".format(**row)
        )
    lines.extend([
        "",
        "## Method Status",
        "",
        "```json",
        json.dumps(report.get("method_status", {}), indent=2, ensure_ascii=False),
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _match_after_paths_by_label(after_paths: dict[str, Path], components: list[Component]) -> dict[str, Path]:
    if not after_paths:
        return {}
    by_safe = {_safe_name(label).lower(): path for label, path in after_paths.items()}
    by_raw = {label.lower(): path for label, path in after_paths.items()}
    out = {}
    for comp in components:
        candidates = [
            comp.label.lower(),
            _safe_name(comp.label).lower(),
            comp.role.lower(),
            "body" if comp.role == "body" else "",
        ]
        for key in candidates:
            if key and key in by_raw:
                out[comp.label] = by_raw[key]
                break
            if key and key in by_safe:
                out[comp.label] = by_safe[key]
                break
    return out


def _match_after_paths(after_paths: dict[str, Path], components: list[Component]) -> tuple[dict[str, Path], dict[str, Any]]:
    matched = _match_after_paths_by_label(after_paths, components)
    strategy = "label"
    if len(matched) < len(components) and len(after_paths) == len(components):
        used = {path.resolve() for path in matched.values()}
        unmatched_components = [comp for comp in components if comp.label not in matched]
        unmatched_paths = [path for path in after_paths.values() if path.resolve() not in used]
        for comp, path in zip(unmatched_components, unmatched_paths, strict=True):
            matched[comp.label] = path
        strategy = "label_plus_order_fallback"
    return matched, {
        "strategy": strategy,
        "matched_count": int(len(matched)),
        "expected_count": int(len(components)),
        "raw_output_labels": list(after_paths.keys()),
        "matched_labels": list(matched.keys()),
    }


def _remap_times_by_matched_paths(after_paths: dict[str, Path], times: dict[str, float], matched_paths: dict[str, Path]) -> dict[str, float]:
    raw_by_path = {path.resolve(): label for label, path in after_paths.items()}
    out: dict[str, float] = {}
    for label, path in matched_paths.items():
        raw_label = raw_by_path.get(path.resolve())
        if raw_label is not None and raw_label in times:
            out[label] = float(times[raw_label])
        elif label in times:
            out[label] = float(times[label])
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = Path(args.summary).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else summary_path.parent / "holopart"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_dir, whole_voxel, components, meta = _component_list(summary_path, assets_dir=Path(args.assets_dir).resolve() if args.assets_dir else None)
    assets_dir = Path(meta["assets_dir"])
    summary = meta["summary"]
    overall_source = _overall_mesh_path(summary, assets_dir)
    if overall_source is None:
        overall_mesh = _build_overall_mesh(components, out_dir / "inputs" / "overall_merged_before.obj")
        overall_source_text = "merged_body_parts"
    else:
        overall_mesh = _load_mesh(overall_source)
        overall_source_text = str(overall_source)
        shutil.copy2(overall_source, out_dir / "inputs" / overall_source.name)

    whole_coords = _load_coords(whole_voxel)
    axis_check = _axis_self_check(overall_mesh, whole_coords, resolution=int(args.voxel_resolution))
    if float(axis_check["iou"]) < float(args.min_axis_iou):
        raise RuntimeError(
            f"voxel<->mesh axis self-check failed: IoU={axis_check['iou']:.4f} < {args.min_axis_iou}; "
            f"best={axis_check}"
        )

    face_seg = _nearest_face_seg(
        overall_mesh,
        components,
        transform=axis_check,
        resolution=int(args.voxel_resolution),
    )
    np.save(out_dir / "inputs" / "face_seg.npy", face_seg)
    _write_json(
        out_dir / "inputs" / "face_seg_labels.json",
        [{"id": idx, "label": comp.label, "role": comp.role, "faces": int(np.sum(face_seg == idx))} for idx, comp in enumerate(components)],
    )
    component_scene_glb = _export_component_scene_glb(components, out_dir / "inputs" / "component_scene_input.glb")
    voxel_segmented_glb = _export_segmented_glb(overall_mesh, face_seg, components, out_dir / "inputs" / "voxel_segmented_input.glb")
    holopart_input_glb = component_scene_glb
    if str(args.holopart_input_mode) == "voxel_seg":
        holopart_input_glb = voxel_segmented_glb
    if args.holopart_input_glb:
        holopart_input_glb = Path(args.holopart_input_glb).resolve()
        if not holopart_input_glb.is_file():
            raise FileNotFoundError(f"--holopart-input-glb missing: {holopart_input_glb}")

    method_status: dict[str, Any] = {}
    if args.method == "fallback":
        after_paths, times = _run_fallback(components, out_dir / "fallback", iterations=int(args.smooth_iterations), lamb=float(args.smooth_lambda))
        method_status["fallback"] = {"enabled": True, "iterations": int(args.smooth_iterations), "lambda": float(args.smooth_lambda)}
    elif args.method == "holopart":
        after_paths, times, status = _run_holopart(
            holopart_input_glb,
            out_dir / "holopart_raw",
            holopart_root=Path(args.holopart_root).resolve(),
            holopart_python=Path(args.holopart_python),
            weights_dir=Path(args.holopart_weights).resolve() if args.holopart_weights else None,
            timeout=int(args.timeout),
            num_inference_steps=int(args.holopart_steps),
            guidance_scale=float(args.holopart_guidance_scale),
            batch_size=int(args.holopart_batch_size),
            use_flash_decoder=bool(args.holopart_flash_decoder),
            normalize_input=bool(args.holopart_normalize_input),
        )
        method_status["holopart"] = status
        raw_after_paths = dict(after_paths)
        raw_times = dict(times)
        after_paths, match_status = _match_after_paths(after_paths, components)
        times = _remap_times_by_matched_paths(raw_after_paths, raw_times, after_paths)
        method_status["holopart_match"] = match_status
        if not after_paths and args.fallback_on_unavailable:
            fallback_paths, fallback_times = _run_fallback(
                components,
                out_dir / "fallback",
                iterations=int(args.smooth_iterations),
                lamb=float(args.smooth_lambda),
            )
            method_status["fallback"] = {
                "enabled": True,
                "reason": "HoloPart unavailable or unmatched output; used deterministic smoothing baseline",
                "iterations": int(args.smooth_iterations),
                "lambda": float(args.smooth_lambda),
            }
            after_paths, times = fallback_paths, fallback_times
    elif args.method == "xpart":
        after_paths, times, status = _run_xpart(
            out_dir / "inputs" / "overall_merged_before.obj" if overall_source is None else out_dir / "inputs" / Path(overall_source_text).name,
            out_dir / "inputs" / "face_seg_labels.json",
            components,
            out_dir / "xpart_raw",
            xpart_root=Path(args.xpart_root).resolve(),
            xpart_python=Path(args.xpart_python),
            weights_dir=Path(args.xpart_weights).resolve(),
            timeout=int(args.timeout),
            steps=int(args.xpart_steps),
            octree_resolution=int(args.xpart_octree_resolution),
            num_chunks=int(args.xpart_num_chunks),
            p3sam_point_num=int(args.xpart_p3sam_point_num),
            p3sam_prompt_num=int(args.xpart_p3sam_prompt_num),
            p3sam_batch_size=int(args.xpart_p3sam_batch_size),
            progress=bool(args.xpart_progress),
            seg_mode=str(args.xpart_seg_mode),
        )
        method_status["xpart"] = status
        raw_after_paths = dict(after_paths)
        raw_times = dict(times)
        after_paths, match_status = _match_after_paths(after_paths, components)
        times = _remap_times_by_matched_paths(raw_after_paths, raw_times, after_paths)
        method_status["xpart_match"] = match_status
        if not after_paths and args.fallback_on_unavailable:
            fallback_paths, fallback_times = _run_fallback(
                components,
                out_dir / "fallback_xpart_placeholder",
                iterations=int(args.smooth_iterations),
                lamb=float(args.smooth_lambda),
            )
            method_status["fallback"] = {
                "enabled": True,
                "reason": "X-Part unavailable or unmatched output; used deterministic smoothing baseline",
                "iterations": int(args.smooth_iterations),
                "lambda": float(args.smooth_lambda),
            }
            after_paths, times = fallback_paths, fallback_times
    else:
        raise ValueError(f"unsupported method: {args.method}")

    after_meshes: dict[str, trimesh.Trimesh] = {}
    after_dir = out_dir / "after_components"
    after_dir.mkdir(parents=True, exist_ok=True)
    for comp in components:
        path = after_paths.get(comp.label)
        if path is None:
            continue
        mesh = _load_mesh(path)
        canonical = after_dir / f"{_safe_name(comp.label)}.obj"
        if Path(path).resolve() != canonical.resolve():
            mesh.export(canonical)
        after_meshes[comp.label] = mesh

    overall_points = _sample_points(overall_mesh, int(args.metric_samples), seed=123)
    metric_rows: list[dict[str, Any]] = []
    for comp in components:
        after = after_meshes.get(comp.label)
        if after is None:
            continue
        before_s = _smoothness(comp.before_mesh)
        after_s = _smoothness(after)
        chamfer = _chamfer_to_overall(after, overall_points, samples=int(args.metric_samples), seed=456)
        completeness = _surface_distance_metrics(
            comp.before_mesh,
            after,
            samples=int(args.completeness_samples),
            seed=789,
            prefix="before_to_after",
        )
        bidirectional = {
            "bidirectional_chamfer_mean_max": float(
                max(
                    float(chamfer.get("after_to_overall_mean", float("nan"))),
                    float(completeness.get("before_to_after_mean", float("nan"))),
                )
            ),
            "bidirectional_chamfer_p95_max": float(
                max(
                    float(chamfer.get("after_to_overall_p95", float("nan"))),
                    float(completeness.get("before_to_after_p95", float("nan"))),
                )
            ),
        }
        row = {
            "component": comp.label,
            "role": comp.role,
            "before_mesh": str(comp.before_mesh_path),
            "after_mesh": str((after_dir / f"{_safe_name(comp.label)}.obj").resolve()),
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
            **chamfer,
            **completeness,
            **bidirectional,
        }
        metric_rows.append(row)

    after_intersection = _voxel_overlap_volume(
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
            out_dir / "before_after_overview.png",
            method=str(args.method),
            max_faces=int(args.render_max_faces),
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(args.render_resolution),
        )
        _write_exploded_panel(
            components,
            after_meshes,
            out_dir / "after_exploded_overview.png",
            max_faces=int(args.render_max_faces),
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(args.render_resolution),
        )
    _write_csv(out_dir / "metrics.csv", metric_rows)

    report = {
        "method": str(args.method),
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),
        "overall_mesh_source": overall_source_text,
        "holopart_input_glb": str(holopart_input_glb),
        "component_scene_input_glb": str(component_scene_glb),
        "voxel_segmented_input_glb": str(voxel_segmented_glb),
        "face_seg_npy": str((out_dir / "inputs" / "face_seg.npy").resolve()),
        "axis_self_check": axis_check,
        "min_axis_iou": float(args.min_axis_iou),
        "body_mesh_is_declared_residual": bool(meta["body_mesh_is_declared_residual"]),
        "body_residual_coords": int(meta["body_residual_coords"]),
        "body_recorded_coords": int(meta["body_recorded_coords"]),
        "component_count": int(len(components)),
        "components": [
            {
                "label": comp.label,
                "role": comp.role,
                "before_mesh": str(comp.before_mesh_path),
                "before_mesh_stats": comp.mesh_stats,
                "voxel_path": None if comp.voxel_path is None else str(comp.voxel_path),
                "voxel_count": None if comp.coords is None else int(comp.coords.shape[0]),
            }
            for comp in components
        ],
        "method_status": method_status,
        "metrics": metric_rows,
        "after_intersection": after_intersection,
    }
    _write_json(out_dir / "report.json", report)
    _markdown_report(out_dir / "report.md", report, metric_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Path to ee_0617_single *__summary.json")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--assets-dir", type=Path, default=None, help="Override *__mujoco/assets directory")
    parser.add_argument("--method", choices=("holopart", "xpart", "fallback"), default="holopart")
    parser.add_argument("--fallback-on-unavailable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--holopart-root", type=Path, default=DEFAULT_HOLOPART_ROOT)
    parser.add_argument("--holopart-python", type=Path, default=DEFAULT_HOLOPART_PYTHON)
    parser.add_argument("--holopart-weights", type=Path, default=DEFAULT_HOLOPART_WEIGHTS)
    parser.add_argument("--holopart-steps", type=int, default=25)
    parser.add_argument("--holopart-guidance-scale", type=float, default=3.5)
    parser.add_argument("--holopart-batch-size", type=int, default=4)
    parser.add_argument("--holopart-flash-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--holopart-normalize-input", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--holopart-input-glb", type=Path, default=None, help="Override HoloPart input GLB for diagnostics.")
    parser.add_argument(
        "--holopart-input-mode",
        choices=("component_scene", "voxel_seg"),
        default="component_scene",
        help="component_scene is the official single-GLB scene made from decoded component OBJ meshes; voxel_seg is diagnostic fallback.",
    )
    parser.add_argument("--xpart-root", type=Path, default=DEFAULT_XPART_ROOT)
    parser.add_argument("--xpart-python", type=Path, default=DEFAULT_XPART_PYTHON)
    parser.add_argument("--xpart-weights", type=Path, default=DEFAULT_XPART_WEIGHTS)
    parser.add_argument("--xpart-steps", type=int, default=50)
    parser.add_argument("--xpart-octree-resolution", type=int, default=512)
    parser.add_argument("--xpart-num-chunks", type=int, default=400000)
    parser.add_argument("--xpart-p3sam-point-num", type=int, default=30000)
    parser.add_argument("--xpart-p3sam-prompt-num", type=int, default=64)
    parser.add_argument("--xpart-p3sam-batch-size", type=int, default=64)
    parser.add_argument("--xpart-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--xpart-seg-mode",
        choices=("seg_surface", "legacy_bbox", "p3sam"),
        default="seg_surface",
        help=(
            "seg_surface samples true ee-eval component mesh surfaces for X-Part conditioning; "
            "legacy_bbox reproduces the old bbox-only path; p3sam uses X-Part/P3-SAM auto segmentation."
        ),
    )
    parser.add_argument("--voxel-resolution", type=int, default=64)
    parser.add_argument("--min-axis-iou", type=float, default=0.5)
    parser.add_argument("--metric-samples", type=int, default=12000)
    parser.add_argument(
        "--completeness-samples",
        type=int,
        default=50000,
        help="Before-surface samples for before->after completeness and coverage metrics.",
    )
    parser.add_argument("--intersection-mode", choices=("bbox", "voxel", "skip"), default="bbox")
    parser.add_argument("--intersection-pitch", type=float, default=0.02)
    parser.add_argument("--intersection-max-faces", type=int, default=50000)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--render-max-faces",
        type=int,
        default=0,
        help="0 renders full meshes, matching ee-eval. Positive values are diagnostic-only face subsampling.",
    )
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--render-view", type=int, default=0)
    parser.add_argument("--smooth-iterations", type=int, default=4)
    parser.add_argument("--smooth-lambda", type=float, default=0.18)
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    print(f"[post-smooth] report -> {report['out_dir']}/report.md", flush=True)
    if not bool(report.get("body_mesh_is_declared_residual", False)):
        print("[post-smooth] warning: body mesh in summary is not declared as residual body_without_parts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
