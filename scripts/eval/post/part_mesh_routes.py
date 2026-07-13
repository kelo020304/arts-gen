#!/usr/bin/env python3
"""Compare part mesh production routes from a good ee-eval overall mesh.

This is a post-processing script only.  It consumes current
``ee_0617_single.py --export-mujoco`` outputs and never changes model or
ee-eval core behavior.
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
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.post.holopart_smooth import (  # noqa: E402
    SAM3D_Z_UP_TO_Y_UP,
    _axis_self_check,
    _chamfer_to_overall,
    _component_list,
    _load_coords,
    _load_mesh,
    _load_render_camera,
    _mesh_to_voxel_indices,
    _mesh_stats,
    _safe_name,
    _sample_points,
    _smoothness,
    _surface_distance_metrics,
    _tile,
    _transfer_vertex_colors,
    _coords_set,
    _voxel_to_mesh_points,
    _voxel_overlap_volume,
    render_component,
)


OBJECTS = [
    ("A", "phyx-verse", "74c7791c8ac64c55a08704202b8cbf38", 1),
    ("B", "physx-0511-drawer-door", "22367", 0),
    ("C", "phyx-verse", "0786542d0f7549208f889113fc384a7f", 0),
    ("D", "phyx-verse", "0a46621504c24197b5653608f474f73b", 0),
]

DEFAULT_BDEC_CKPT = Path(
    "/robot/data-lab/jzh/art-gen/ckpts/slat-dec-part/routeB_expanded_subset_fix_0706/step_0003000.pt"
)
DEFAULT_BDEC_BASE_DECODER_CKPT = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
DEFAULT_BDEC_CACHE_MANIFEST = Path(
    "/robot/data-lab/jzh/art-gen/data/slat_dec_part_cache/phase2_shared/phase2_shared_cache_manifest.json"
)


_BDEC_DECODER_CACHE: dict[tuple[str, str, int], Any] = {}


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
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bdec_data_roots(dataset_id: str) -> list[Path]:
    roots: dict[str, list[Path]] = {
        "phyx-verse": [
            Path("/robot/data-lab/jzh/art-gen/data/phyx-verse"),
        ],
        "physx-0511-drawer-door": [
            Path("/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511"),
            Path("/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/PhysX-Mobility-full-4view-0511"),
            Path("/robot/data-lab/arts-gen-data/data/PhysX-Mobility-full-4view-0511"),
        ],
    }
    return roots.get(str(dataset_id), [])


def _bdec_manifest_slat_candidates(
    *,
    manifest_path: Path,
    object_tag: str,
    dataset_id: str,
    object_id: str,
    angle: int,
) -> list[Path]:
    if not manifest_path.is_file():
        return []
    payload = _load_json(manifest_path)
    cache_root = manifest_path.parent
    candidates: list[Path] = []
    for obj in payload.get("objects") or []:
        tag_match = str(obj.get("tag") or "") == str(object_tag)
        exact_match = (
            str(obj.get("dataset_id") or "") == str(dataset_id)
            and str(obj.get("obj_id") or "") == str(object_id)
            and int(obj.get("angle_idx", -9999)) == int(angle)
        )
        # The historical route_A cache is angle_0 while the route harness uses
        # angle_1.  Prefer an exact angle match from data roots, but keep the
        # tag fallback as a last resort and record the source in route_status.
        if not (exact_match or tag_match):
            continue
        rel = obj.get("overall_slat_rel")
        if rel:
            candidates.append(cache_root / str(rel))
        src = obj.get("overall_slat_source")
        if src:
            candidates.append(Path(str(src)))
    return candidates


def _find_bdec_overall_slat(
    *,
    object_tag: str,
    dataset_id: str,
    object_id: str,
    angle: int,
    manifest_path: Path,
) -> Path:
    candidates: list[Path] = []
    for root in _bdec_data_roots(dataset_id):
        candidates.extend(
            [
                root / "part_synthesis_slat" / object_id[:2] / f"{object_id}_angle_{int(angle)}" / "overall" / "latent.npz",
                root
                / "reconstruction"
                / "part_synthesis_slat"
                / object_id[:2]
                / f"{object_id}_angle_{int(angle)}"
                / "overall"
                / "latent.npz",
                root / "reconstruction" / "slat_latents_expanded" / object_id / f"angle_{int(angle)}" / "latent.npz",
            ]
        )
    candidates.extend(
        _bdec_manifest_slat_candidates(
            manifest_path=manifest_path,
            object_tag=object_tag,
            dataset_id=dataset_id,
            object_id=object_id,
            angle=int(angle),
        )
    )
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    for path in unique:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"B-dec overall SLat latent.npz not found for {dataset_id}::{object_id} angle={angle}; "
        f"tried {unique}"
    )


def _load_bdec_decoder(
    *,
    ckpt: Path,
    base_decoder_ckpt: Path,
    gpu: int,
) -> Any:
    key = (str(ckpt.resolve()), str(base_decoder_ckpt.resolve()), int(gpu))
    if key in _BDEC_DECODER_CACHE:
        return _BDEC_DECODER_CACHE[key]
    if not ckpt.is_file():
        raise FileNotFoundError(f"B-dec checkpoint missing: {ckpt}")
    if not base_decoder_ckpt.is_file():
        raise FileNotFoundError(f"B-dec base decoder checkpoint missing: {base_decoder_ckpt}")
    train_dir = REPO_ROOT / "scripts" / "train" / "slat_dec_part"
    trellis_root = REPO_ROOT / "TRELLIS-arts"
    for path in (trellis_root, train_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch  # noqa: PLC0415
    from render_track1_snapshots import load_decoder_from_snapshot  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("B-dec route requires CUDA for the TRELLIS sparse decoder")
    device = torch.device(f"cuda:{int(gpu)}")
    torch.cuda.set_device(device)
    decoder = load_decoder_from_snapshot(base_decoder_ckpt, ckpt, device=device)
    decoder.eval()
    _BDEC_DECODER_CACHE[key] = (decoder, device)
    return _BDEC_DECODER_CACHE[key]


def _finite_floats(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _min_float(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_floats(rows, key)
    return float(min(values)) if values else float("nan")


def _max_float(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_floats(rows, key)
    return float(max(values)) if values else float("nan")


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_floats(rows, key)
    return float(np.mean(values)) if values else float("nan")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"", "0", "false", "none", "nan", "no"}:
        return False
    if text in {"1", "true", "yes"}:
        return True
    return bool(value)


def _summary_path(eval_dir: Path, dataset_id: str, object_id: str, angle: int) -> Path:
    return eval_dir / f"{dataset_id}__{object_id}__angle_{int(angle):02d}__summary.json"


def _coords_mask(coords: np.ndarray, *, resolution: int = 64) -> np.ndarray:
    mask = np.zeros((int(resolution), int(resolution), int(resolution)), dtype=bool)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    valid = np.all((coords >= 0) & (coords < int(resolution)), axis=1)
    if bool(valid.any()):
        c = coords[valid]
        mask[c[:, 0], c[:, 1], c[:, 2]] = True
    return mask


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool)).astype(np.int64, copy=False)


def _ensure_manifold3d_available() -> dict[str, Any]:
    try:
        import manifold3d  # noqa: F401

        return {
            "available": True,
            "module": "manifold3d",
            "version": str(getattr(manifold3d, "__version__", "unknown")),
        }
    except Exception as exc:
        raise RuntimeError(
            "manifold3d import failed; true R-P is required and silent fallback is forbidden. "
            f"Install the TOS wheel into /opt/venvs/arts-gen first. error={exc!r}"
        ) from exc


def _grid_vertices_to_mesh_points(vertices: np.ndarray, *, transform: dict[str, Any], resolution: int = 64) -> np.ndarray:
    norm = np.asarray(vertices, dtype=np.float64) / float(resolution) - 0.5
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    return (norm * float(transform["scale"])) @ rotation


def _voxel_mesh_from_coords(
    coords: np.ndarray,
    *,
    transform: dict[str, Any],
    out_path: Path,
    resolution: int = 64,
    smooth_sigma: float = 0.6,
    taubin_iterations: int = 8,
) -> trimesh.Trimesh:
    mask = _coords_mask(coords, resolution=int(resolution))
    if not bool(mask.any()):
        raise ValueError("empty voxel coords")
    filled = binary_fill_holes(mask)
    grid = np.pad(filled.astype(np.float32), 1, mode="constant")
    if float(smooth_sigma) > 0:
        grid = gaussian_filter(grid, sigma=float(smooth_sigma))
        level = 0.35
    else:
        level = 0.5
    verts, faces, _normals, _values = measure.marching_cubes(grid, level=float(level))
    # Remove pad and map 64^3 index coordinates into the decoded overall mesh bbox.
    verts = verts - 1.0
    vertices = _grid_vertices_to_mesh_points(verts, transform=transform, resolution=int(resolution))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int64), process=True)
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces"):
        keep = mesh.nondegenerate_faces()
        mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    if int(taubin_iterations) > 0 and len(mesh.faces):
        trimesh.smoothing.filter_taubin(mesh, iterations=int(taubin_iterations))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    return mesh


def _mesh_perforation(mesh: trimesh.Trimesh) -> dict[str, Any]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    out: dict[str, Any] = {
        "open_edge_count": 0,
        "total_unique_edges": 0,
        "open_edge_length": 0.0,
        "total_edge_length": 0.0,
        "open_edge_ratio": 0.0,
        "connected_components": 0,
        "hole_count": 0,
    }
    if len(vertices) == 0 or len(faces) == 0:
        return out
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if len(unique) == 0:
        return out
    lengths = np.linalg.norm(vertices[unique[:, 0]] - vertices[unique[:, 1]], axis=1)
    boundary_mask = counts == 1
    out["open_edge_count"] = int(np.count_nonzero(boundary_mask))
    out["total_unique_edges"] = int(len(unique))
    out["open_edge_length"] = float(np.sum(lengths[boundary_mask]))
    out["total_edge_length"] = float(np.sum(lengths))
    out["open_edge_ratio"] = float(out["open_edge_length"] / max(out["total_edge_length"], 1.0e-12))
    try:
        comps = trimesh.graph.connected_components(
            mesh.face_adjacency,
            nodes=np.arange(len(faces)),
            min_len=1,
        )
        out["connected_components"] = int(len(comps))
    except Exception:
        out["connected_components"] = int(len(mesh.split(only_watertight=False)))
    boundary_edges = unique[boundary_mask]
    if len(boundary_edges):
        try:
            loops = trimesh.graph.connected_components(boundary_edges, min_len=1)
            out["hole_count"] = int(len(loops))
        except Exception:
            out["hole_count"] = int(out["open_edge_count"])
    return out


def _repair_boolean_input(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    repaired = mesh.copy()
    before = _mesh_perforation(repaired)
    status: dict[str, Any] = {
        "input_watertight_before_repair": bool(repaired.is_watertight),
        "input_open_edge_ratio_before_repair": before["open_edge_ratio"],
        "input_hole_count_before_repair": before["hole_count"],
        "repair_attempted": False,
        "input_watertight_after_repair": bool(repaired.is_watertight),
    }
    if not bool(repaired.is_watertight):
        status["repair_attempted"] = True
        try:
            trimesh.repair.fix_normals(repaired)
            trimesh.repair.fill_holes(repaired)
            repaired.remove_unreferenced_vertices()
        except Exception as exc:
            status["repair_error"] = repr(exc)
    after = _mesh_perforation(repaired)
    status["input_watertight_after_repair"] = bool(repaired.is_watertight)
    status["input_open_edge_ratio_after_repair"] = after["open_edge_ratio"]
    status["input_hole_count_after_repair"] = after["hole_count"]
    return repaired, status


def _try_boolean_intersection(
    overall_mesh: trimesh.Trimesh,
    coords: np.ndarray,
    *,
    transform: dict[str, Any],
    out_path: Path,
    resolution: int,
) -> tuple[trimesh.Trimesh | None, dict[str, Any]]:
    """Try R-P boolean intersection; return None with reason when unavailable.

    The intended engine is manifold3d.  The development image may not have it
    installed; in that case this function records the exact failure and lets the
    caller use R-V fallback.
    """

    status: dict[str, Any] = {
        "engine": "trimesh.boolean.intersection",
        "fallback_used": False,
        "error": None,
    }
    try:
        region = _voxel_mesh_from_coords(
            coords,
            transform=transform,
            out_path=out_path.with_name(out_path.stem + "__region.obj"),
            resolution=int(resolution),
            smooth_sigma=0.0,
            taubin_iterations=0,
        )
        started = time.time()
        result = trimesh.boolean.intersection([overall_mesh, region], engine="manifold")
        status["seconds_boolean"] = float(time.time() - started)
        if result is None or len(result.vertices) == 0 or len(result.faces) == 0:
            raise ValueError("boolean returned empty mesh")
        result.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(result)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.export(out_path)
        return result, status
    except Exception as exc:
        status["error"] = repr(exc)
        return None, status


def _run_bdec_clip_route(
    *,
    bdec_paths: dict[str, Path],
    components: list[Any],
    transform: dict[str, Any],
    out_dir: Path,
    resolution: int = 64,
    reuse_existing: bool = False,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Clip B-dec meshes by single-owner component voxel regions.

    This is a pure geometry post-process: B-dec mesh ∩ component voxel region.
    It does not dilate the region and does not delete faces by ownership after
    the boolean result.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    route_paths: dict[str, Path] = {}
    route_status: dict[str, dict[str, Any]] = {}
    for comp in components:
        started = time.time()
        out_path = out_dir / f"{_safe_name(comp.label)}.obj"
        status: dict[str, Any] = {
            "source": "B-dec mesh boolean-intersected with single-owner voxel region",
            "region": "component coords, no dilation",
            "engine": "trimesh.boolean.intersection(engine='manifold')",
            "available": False,
        }
        try:
            if bool(reuse_existing) and out_path.is_file():
                route_paths[comp.label] = out_path
                status.update({"available": True, "reused_existing_route_mesh": True, "seconds": 0.0, "mesh": str(out_path)})
                route_status[comp.label] = status
                continue
            source = bdec_paths.get(comp.label)
            if source is None or not source.is_file():
                raise FileNotFoundError(f"B-dec source missing for {comp.label}")
            if comp.coords is None or len(comp.coords) == 0:
                raise ValueError("missing component coords")
            bdec_mesh = _load_mesh(source)
            before_repair = _mesh_perforation(bdec_mesh)
            repaired = bdec_mesh.copy()
            repair_attempted = False
            repair_error = None
            if not bool(repaired.is_watertight):
                repair_attempted = True
                try:
                    trimesh.repair.fix_normals(repaired)
                    trimesh.repair.fill_holes(repaired)
                    repaired.remove_unreferenced_vertices()
                except Exception as exc:
                    repair_error = repr(exc)
            after_repair = _mesh_perforation(repaired)
            status.update(
                {
                    "source_mesh": str(source),
                    "source_watertight": bool(bdec_mesh.is_watertight),
                    "repair_attempted": bool(repair_attempted),
                    "repair_error": repair_error,
                    "source_open_edge_ratio": before_repair["open_edge_ratio"],
                    "source_hole_count": before_repair["hole_count"],
                    "after_fill_holes_watertight": bool(repaired.is_watertight),
                    "after_fill_holes_open_edge_ratio": after_repair["open_edge_ratio"],
                    "after_fill_holes_hole_count": after_repair["hole_count"],
                    "repair_unresolved": bool(repair_attempted and not repaired.is_watertight),
                }
            )
            region_path = out_path.with_name(out_path.stem + "__region.obj")
            region = _voxel_mesh_from_coords(
                comp.coords,
                transform=transform,
                out_path=region_path,
                resolution=int(resolution),
                smooth_sigma=0.0,
                taubin_iterations=0,
            )
            result = trimesh.boolean.intersection([repaired, region], engine="manifold")
            if result is None or len(result.vertices) == 0 or len(result.faces) == 0:
                raise ValueError("boolean returned empty mesh")
            result.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(result)
            result.export(out_path)
            route_paths[comp.label] = out_path
            result_perf = _mesh_perforation(result)
            status.update(
                {
                    "available": True,
                    "seconds": float(time.time() - started),
                    "mesh": str(out_path),
                    "region_mesh": str(region_path),
                    "result_watertight": bool(result.is_watertight),
                    "result_open_edge_ratio": result_perf["open_edge_ratio"],
                    "result_hole_count": result_perf["hole_count"],
                    "source_faces": int(len(bdec_mesh.faces)),
                    "region_faces": int(len(region.faces)),
                    "result_faces": int(len(result.faces)),
                }
            )
        except Exception as exc:
            status.update({"available": False, "seconds": float(time.time() - started), "error": repr(exc)})
        route_status[comp.label] = status
    return route_paths, route_status


def _face_owner_by_voxel(
    mesh: trimesh.Trimesh,
    components: list[Any],
    *,
    transform: dict[str, Any],
    resolution: int = 64,
) -> np.ndarray:
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    idx = _mesh_to_voxel_indices(
        centers,
        scale=float(transform["scale"]),
        rotation=np.asarray(transform["rotation"], dtype=np.float64),
        resolution=int(resolution),
    )
    owner = np.full((int(resolution), int(resolution), int(resolution)), -1, dtype=np.int32)
    for comp_idx, comp in enumerate(components):
        if comp.coords is None or len(comp.coords) == 0:
            continue
        coords = np.asarray(comp.coords, dtype=np.int64).reshape(-1, 3)
        valid = np.all((coords >= 0) & (coords < int(resolution)), axis=1)
        coords = coords[valid]
        owner[coords[:, 0], coords[:, 1], coords[:, 2]] = int(comp_idx)
    valid = np.all((idx >= 0) & (idx < int(resolution)), axis=1)
    labels = np.full((len(centers),), -1, dtype=np.int32)
    if bool(valid.any()):
        v = idx[valid]
        labels[valid] = owner[v[:, 0], v[:, 1], v[:, 2]]
    missing = labels < 0
    if bool(np.any(missing)):
        comp_points: list[np.ndarray] = []
        comp_ids: list[np.ndarray] = []
        for comp_idx, comp in enumerate(components):
            if comp.coords is None or len(comp.coords) == 0:
                continue
            pts = _voxel_to_mesh_points(
                comp.coords,
                scale=float(transform["scale"]),
                rotation=np.asarray(transform["rotation"], dtype=np.float64),
                resolution=int(resolution),
            )
            comp_points.append(pts)
            comp_ids.append(np.full((len(pts),), int(comp_idx), dtype=np.int32))
        if comp_points:
            tree = cKDTree(np.concatenate(comp_points, axis=0))
            all_ids = np.concatenate(comp_ids, axis=0)
            _dist, nearest = tree.query(centers[missing], k=1, workers=-1)
            labels[missing] = all_ids[nearest]
    return labels


def _majority_smooth_face_labels(labels: np.ndarray, adjacency: np.ndarray, *, iterations: int = 2) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32).copy()
    adjacency = np.asarray(adjacency, dtype=np.int64)
    if len(labels) == 0 or len(adjacency) == 0:
        return labels
    neighbors: list[list[int]] = [[] for _ in range(len(labels))]
    for a, b in adjacency:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
    for _ in range(int(iterations)):
        new_labels = labels.copy()
        for idx, nbrs in enumerate(neighbors):
            if not nbrs:
                continue
            votes = labels[nbrs + [idx]]
            votes = votes[votes >= 0]
            if len(votes) == 0:
                continue
            values, counts = np.unique(votes, return_counts=True)
            new_labels[idx] = int(values[int(np.argmax(counts))])
        labels = new_labels
    return labels


def _trim_dilated_meshes(
    *,
    dilated_paths: dict[str, Path],
    components: list[Any],
    transform: dict[str, Any],
    out_dir: Path,
    resolution: int = 64,
    reuse_existing: bool = False,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_by_label = {comp.label: idx for idx, comp in enumerate(components)}
    out: dict[str, Path] = {}
    status: dict[str, dict[str, Any]] = {}
    for comp in components:
        out_path = out_dir / f"{_safe_name(comp.label)}.obj"
        if bool(reuse_existing) and out_path.is_file():
            out[comp.label] = out_path
            status[comp.label] = {
                "available": True,
                "source": str(out_path),
                "reused_existing_route_mesh": True,
                "trim_rule": "voxel-owner face labels with 2-step face-adjacency majority vote",
            }
            continue
        source = dilated_paths.get(comp.label)
        if source is None or not source.is_file():
            status[comp.label] = {"available": False, "error": "missing dilated source"}
            continue
        started = time.time()
        try:
            mesh = _load_mesh(source)
            raw_perf = _mesh_perforation(mesh)
            face_owner = _face_owner_by_voxel(mesh, components, transform=transform, resolution=int(resolution))
            smoothed = _majority_smooth_face_labels(face_owner, mesh.face_adjacency, iterations=2)
            keep = smoothed == int(comp_by_label[comp.label])
            if not bool(np.any(keep)):
                raise ValueError("no faces kept after majority-vote trim")
            trimmed = mesh.submesh([keep], append=True, repair=False)
            trimmed.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(trimmed)
            trimmed.export(out_path)
            out[comp.label] = out_path
            trim_perf = _mesh_perforation(trimmed)
            status[comp.label] = {
                "available": True,
                "source": str(source),
                "seconds": float(time.time() - started),
                "raw_faces": int(len(mesh.faces)),
                "kept_faces": int(len(trimmed.faces)),
                "kept_face_fraction": float(len(trimmed.faces) / max(len(mesh.faces), 1)),
                "raw_open_edge_ratio": raw_perf["open_edge_ratio"],
                "trim_open_edge_ratio": trim_perf["open_edge_ratio"],
                "raw_hole_count": raw_perf["hole_count"],
                "trim_hole_count": trim_perf["hole_count"],
                "trim_rule": "voxel-owner face labels with 2-step face-adjacency majority vote",
            }
        except Exception as exc:
            status[comp.label] = {
                "available": False,
                "source": str(source),
                "seconds": float(time.time() - started),
                "error": repr(exc),
            }
    return out, status


def _run_bdec_route(
    *,
    object_tag: str,
    dataset_id: str,
    object_id: str,
    angle: int,
    components: list[Any],
    overall_mesh: trimesh.Trimesh,
    transform: dict[str, Any],
    out_dir: Path,
    ckpt: Path,
    base_decoder_ckpt: Path,
    cache_manifest: Path,
    subset_dilation: int,
    gpu: int,
    reuse_existing: bool = False,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Decode RouteB-3K expanded-subset meshes for the current harness components.

    This route deliberately does not trim/delete faces after decoding.  The only
    post steps are coordinate-frame conversion into the ee-eval mesh frame,
    normal repair, and vertex color transfer for visualization.
    """

    started_all = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(reuse_existing):
        existing = {comp.label: out_dir / f"{_safe_name(comp.label)}.obj" for comp in components}
        if all(path.is_file() for path in existing.values()):
            status = {
                comp.label: {
                    "source": "RouteB-3K PartMaskedSLatMeshDecoder",
                    "checkpoint": str(ckpt),
                    "base_decoder_ckpt": str(base_decoder_ckpt),
                    "subset_dilation": int(subset_dilation),
                    "no_posthoc_face_delete": True,
                    "available": True,
                    "reused_existing_route_mesh": True,
                    "mesh": str(existing[comp.label]),
                    "seconds": 0.0,
                }
                for comp in components
            }
            status["_route"] = {
                "seconds_total": 0.0,
                "components_written": int(len(existing)),
                "expected_components": int(len(components)),
                "checkpoint": str(ckpt),
                "reused_existing_route_meshes": True,
            }
            return existing, status
    slat_path = _find_bdec_overall_slat(
        object_tag=object_tag,
        dataset_id=dataset_id,
        object_id=object_id,
        angle=int(angle),
        manifest_path=cache_manifest,
    )
    with np.load(slat_path, allow_pickle=False) as data:
        coords = np.asarray(data["coords"], dtype=np.int32)
        feats = np.asarray(data["feats"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{slat_path}: B-dec coords expected [N,3], got {coords.shape}")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0] or feats.shape[1] != 8:
        raise ValueError(f"{slat_path}: B-dec feats expected [N,8] matching coords, got {feats.shape}")

    decoder, device = _load_bdec_decoder(ckpt=ckpt, base_decoder_ckpt=base_decoder_ckpt, gpu=int(gpu))
    import torch  # noqa: PLC0415
    from trellis.modules.sparse import SparseTensor  # noqa: PLC0415
    from track1_online_render import (  # noqa: PLC0415
        component_mask_for_slat_coords,
        dilate_coords64,
        subset_slat_by_coords64,
    )

    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    scale = float(transform["scale"])
    route_paths: dict[str, Path] = {}
    route_status: dict[str, dict[str, Any]] = {}
    for comp in components:
        started = time.time()
        out_path = out_dir / f"{_safe_name(comp.label)}.obj"
        status: dict[str, Any] = {
            "source": "RouteB-3K PartMaskedSLatMeshDecoder",
            "checkpoint": str(ckpt),
            "base_decoder_ckpt": str(base_decoder_ckpt),
            "overall_slat": str(slat_path),
            "subset_dilation": int(subset_dilation),
            "no_posthoc_face_delete": True,
            "coordinate_transform": "rep.vertices * axis_self_check.scale @ axis_self_check.rotation",
            "available": False,
        }
        try:
            if bool(reuse_existing) and out_path.is_file():
                route_paths[comp.label] = out_path
                status.update(
                    {
                        "available": True,
                        "seconds": 0.0,
                        "reused_existing_route_mesh": True,
                        "mesh": str(out_path),
                    }
                )
                route_status[comp.label] = status
                continue
            if comp.coords is None or len(comp.coords) == 0:
                raise ValueError("missing component coords")
            component_coords = np.asarray(comp.coords, dtype=np.int16).reshape(-1, 3)
            expanded = dilate_coords64(component_coords, int(subset_dilation))
            sub_coords, sub_feats, matched = subset_slat_by_coords64(
                coords,
                feats,
                expanded,
                label=f"{object_tag}:{comp.label}",
            )
            mask = component_mask_for_slat_coords(sub_coords, component_coords).reshape(-1, 1)
            if float(mask.sum()) <= 0.0:
                raise ValueError("mask has no active SLat voxels after subset matching")
            batch_col = np.zeros((sub_coords.shape[0], 1), dtype=np.int32)
            sparse_coords = np.concatenate([batch_col, sub_coords.astype(np.int32, copy=False)], axis=1)
            sparse_feats = np.concatenate([sub_feats.astype(np.float32, copy=False), mask.astype(np.float32, copy=False)], axis=1)
            latents = SparseTensor(
                coords=torch.from_numpy(sparse_coords).to(device=device, dtype=torch.int32),
                feats=torch.from_numpy(sparse_feats).to(device=device, dtype=torch.float32),
            )
            with torch.no_grad():
                rep = decoder(latents)[0]
            success = bool(getattr(rep, "success", False))
            if not success:
                raise ValueError("decoder rep.success is false")
            vertices_t = getattr(rep, "vertices", None)
            faces_t = getattr(rep, "faces", None)
            if vertices_t is None or faces_t is None or vertices_t.numel() == 0 or faces_t.numel() == 0:
                raise ValueError("decoder returned empty vertices/faces")
            raw_vertices = vertices_t.detach().float().cpu().numpy()
            faces = faces_t.detach().long().cpu().numpy()
            vertices = (raw_vertices.astype(np.float64, copy=False) * scale) @ rotation
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int64, copy=False), process=False)
            mesh.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(mesh)
            colors = _transfer_vertex_colors(overall_mesh, mesh)
            if len(colors) == len(mesh.vertices):
                rgba = np.pad((np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8), ((0, 0), (0, 1)), constant_values=255)
                mesh.visual.vertex_colors = rgba
            mesh.export(out_path)
            route_paths[comp.label] = out_path
            status.update(
                {
                    "available": True,
                    "seconds": float(time.time() - started),
                    "component_coords64": int(component_coords.shape[0]),
                    "expanded_coords64": int(expanded.shape[0]),
                    "matched_slat_voxels": int(matched),
                    "subset_slat_voxels": int(sub_coords.shape[0]),
                    "mask_voxels_on_slat": int(float(mask.sum())),
                    "success": success,
                    "raw_vertices": int(raw_vertices.shape[0]),
                    "raw_faces": int(faces.shape[0]),
                    "raw_bbox_min": raw_vertices.min(axis=0).astype(float).tolist(),
                    "raw_bbox_max": raw_vertices.max(axis=0).astype(float).tolist(),
                    "mapped_bbox_min": vertices.min(axis=0).astype(float).tolist(),
                    "mapped_bbox_max": vertices.max(axis=0).astype(float).tolist(),
                    "mesh": str(out_path),
                }
            )
        except Exception as exc:
            status.update(
                {
                    "available": False,
                    "seconds": float(time.time() - started),
                    "error": repr(exc),
                }
            )
        route_status[comp.label] = status
    route_status["_route"] = {
        "seconds_total": float(time.time() - started_all),
        "components_written": int(len(route_paths)),
        "expected_components": int(len(components)),
        "overall_slat": str(slat_path),
        "checkpoint": str(ckpt),
    }
    return route_paths, route_status


def _load_xpart_report(object_tag: str, dataset_id: str, object_id: str, angle: int, root: Path) -> Path | None:
    candidates = [
        root / "xpart_sweep" / "s8_o384" / f"{object_tag}_{object_id}" / "report.json",
        root / "xpart_sweep" / "s8_o384" / f"{object_tag}_{object_id}" / "xpart_raw" / "report.json",
        Path("/robot/data-lab/jzh/art-gen/ee-eval/post_smooth_corrected_0702")
        / "xpart"
        / f"{dataset_id}__{object_id}__angle_{int(angle):02d}"
        / "report.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _after_meshes_from_report(report_path: Path | None) -> dict[str, Path]:
    if report_path is None or not report_path.is_file():
        return {}
    data = _load_json(report_path)
    out: dict[str, Path] = {}
    for row in data.get("metrics") or []:
        label = str(row.get("component") or "")
        path = Path(str(row.get("after_mesh") or ""))
        if label and path.is_file():
            out[label] = path
    if not out:
        for path in (report_path.parent / "after_components").glob("*.obj"):
            out[path.stem] = path
    return out


def _route_mesh_map_from_dir(after_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(after_dir.glob("*.obj"))}


def _copy_route_meshes(source: dict[str, Path], dest_dir: Path, labels: list[str]) -> dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for label in labels:
        path = source.get(label)
        if path is None or not path.is_file():
            continue
        dest = dest_dir / f"{_safe_name(label)}.obj"
        shutil.copy2(path, dest)
        out[label] = dest
    return out


def _dilated_meshes_from_summary(summary_path: Path | None, dest_dir: Path, labels: list[str]) -> dict[str, Path]:
    if summary_path is None or not summary_path.is_file():
        return {}
    data = _load_json(summary_path)
    items = []
    if data.get("mujoco_body_mesh"):
        item = dict(data["mujoco_body_mesh"])
        item["label"] = "body_without_parts"
        items.append(item)
    items.extend(data.get("mujoco_part_meshes") or [])
    source = {str(item["label"]): Path(item["mesh_path"]) for item in items if Path(item.get("mesh_path", "")).is_file()}
    return _copy_route_meshes(source, dest_dir, labels)


def _metric_row(
    *,
    object_tag: str,
    object_key: str,
    route: str,
    comp: Any,
    after_path: Path,
    after_mesh: trimesh.Trimesh,
    overall_points: np.ndarray,
    route_status: dict[str, Any] | None,
    metric_samples: int,
    completeness_samples: int,
    idx: int,
) -> dict[str, Any]:
    before = comp.before_mesh
    before_s = _smoothness(before)
    after_s = _smoothness(after_mesh)
    before_p = _mesh_perforation(before)
    after_p = _mesh_perforation(after_mesh)
    a2o = _chamfer_to_overall(after_mesh, overall_points, samples=int(metric_samples), seed=456 + idx)
    b2a = _surface_distance_metrics(
        before,
        after_mesh,
        samples=int(completeness_samples),
        seed=789 + idx,
        prefix="before_to_after",
    )
    after_to_before = _surface_distance_metrics(
        after_mesh,
        before,
        samples=int(completeness_samples),
        seed=987 + idx,
        prefix="after_to_before",
    )
    row = {
        "object_tag": object_tag,
        "object_key": object_key,
        "route": route,
        "component": comp.label,
        "role": comp.role,
        "before_mesh": str(comp.before_mesh_path),
        "after_mesh": str(after_path),
        "voxel_count": None if comp.coords is None else int(comp.coords.shape[0]),
        "before_vertices": before_s["vertex_count"],
        "before_faces": before_s["face_count"],
        "after_vertices": after_s["vertex_count"],
        "after_faces": after_s["face_count"],
        "before_is_watertight": before_s["is_watertight"],
        "after_is_watertight": after_s["is_watertight"],
        "before_open_edge_ratio": before_p["open_edge_ratio"],
        "after_open_edge_ratio": after_p["open_edge_ratio"],
        "before_open_edge_count": before_p["open_edge_count"],
        "after_open_edge_count": after_p["open_edge_count"],
        "before_connected_components": before_p["connected_components"],
        "after_connected_components": after_p["connected_components"],
        "before_hole_count": before_p["hole_count"],
        "after_hole_count": after_p["hole_count"],
        "before_mean_dihedral_rad": before_s["mean_dihedral_rad"],
        "after_mean_dihedral_rad": after_s["mean_dihedral_rad"],
        "before_normal_variance": before_s["normal_variance"],
        "after_normal_variance": after_s["normal_variance"],
        **a2o,
        **b2a,
        **after_to_before,
        "bidirectional_chamfer_mean_max": float(max(a2o["after_to_overall_mean"], b2a["before_to_after_mean"])),
        "bidirectional_chamfer_p95_max": float(max(a2o["after_to_overall_p95"], b2a["before_to_after_p95"])),
    }
    if route_status:
        for key, value in route_status.items():
            row[f"route_status_{key}"] = value
    return row


def _render_route_panel(
    *,
    components: list[Any],
    routes: dict[str, dict[str, Path]],
    out_path: Path,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
    max_faces: int,
) -> None:
    route_names = ["before", "R-D", "R-V", "B-dec", "B-dec+clip", "remark"]
    rows: list[Image.Image] = []
    for comp in components:
        tiles: list[Image.Image] = []
        for route in route_names:
            if route == "remark":
                image = Image.new("RGB", (int(resolution), int(resolution)), (245, 245, 245))
                draw = ImageDraw.Draw(image)
                draw.text((10, 12), comp.label[:48], fill=(0, 0, 0))
                draw.text((10, 38), f"role={comp.role}", fill=(0, 0, 0))
                draw.text((10, 64), f"vox={0 if comp.coords is None else len(comp.coords)}", fill=(0, 0, 0))
                tiles.append(_tile(image, "remark", int(resolution), int(resolution) + 30))
                continue
            if route == "before":
                mesh = comp.before_mesh
                colors = None
            else:
                path = routes.get(route, {}).get(comp.label)
                if path is None and route == "R-P":
                    path = routes.get("R-P-fallback", {}).get(comp.label)
                if path is None or not path.is_file():
                    image = Image.new("RGB", (int(resolution), int(resolution)), (250, 240, 240))
                    ImageDraw.Draw(image).text((10, 12), "missing", fill=(120, 0, 0))
                    tiles.append(_tile(image, route, int(resolution), int(resolution) + 30))
                    continue
                mesh = _load_mesh(path)
                colors = _transfer_vertex_colors(comp.before_mesh, mesh)
            image = render_component(
                mesh,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                resolution=int(resolution),
                max_faces=int(max_faces),
                vertex_colors=colors,
                color_mode="color",
            )
            tiles.append(_tile(image, route, int(resolution), int(resolution) + 30))
        row = Image.new("RGB", (len(route_names) * int(resolution), int(resolution) + 30), (255, 255, 255))
        for col, tile in enumerate(tiles):
            row.paste(tile, (col * int(resolution), 0))
        rows.append(row)
    canvas = Image.new("RGB", (len(route_names) * int(resolution), len(rows) * (int(resolution) + 30)), (255, 255, 255))
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * (int(resolution) + 30)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _exploded_mesh(
    components: list[Any],
    route_paths: dict[str, Path],
) -> tuple[trimesh.Trimesh, np.ndarray]:
    selected = [comp for comp in components if comp.label in route_paths]
    meshes: list[trimesh.Trimesh] = []
    colors: list[np.ndarray] = []
    if not selected:
        raise ValueError("no meshes for exploded view")
    centers = np.stack([_load_mesh(route_paths[comp.label]).bounds.mean(axis=0) for comp in selected])
    center0 = centers.mean(axis=0)
    extent = float(np.max(np.ptp(centers, axis=0))) if len(selected) > 1 else 1.0
    if not np.isfinite(extent) or extent <= 1.0e-6:
        extent = 1.0
    for comp, center in zip(selected, centers, strict=True):
        mesh = _load_mesh(route_paths[comp.label]).copy()
        direction = center - center0
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            direction = np.asarray([1.0, 0.0, 0.0])
            norm = 1.0
        mesh.apply_translation((direction / norm) * 0.35 * extent)
        meshes.append(mesh)
        colors.append(_transfer_vertex_colors(comp.before_mesh, mesh))
    merged = trimesh.util.concatenate(meshes)
    merged_colors = np.concatenate(colors, axis=0)
    return merged, merged_colors


def _write_exploded(
    *,
    components: list[Any],
    routes: dict[str, dict[str, Path]],
    out_dir: Path,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
    max_faces: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    render_routes = dict(routes)
    if "R-P" in render_routes and "R-P-fallback" in render_routes:
        merged_rp = dict(render_routes["R-P-fallback"])
        merged_rp.update(render_routes["R-P"])
        render_routes["R-P"] = merged_rp
        render_routes.pop("R-P-fallback", None)
    for route, paths in render_routes.items():
        if route == "before":
            paths = {comp.label: comp.before_mesh_path for comp in components}
        if not paths:
            continue
        try:
            mesh, colors = _exploded_mesh(components, paths)
            image = render_component(
                mesh,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                resolution=int(resolution),
                max_faces=int(max_faces),
                vertex_colors=colors,
                color_mode="color",
            )
            _tile(image, f"{route} exploded", int(resolution), int(resolution) + 30).save(
                out_dir / f"{_safe_name(route)}__exploded.png"
            )
        except Exception as exc:
            _write_json(out_dir / f"{_safe_name(route)}__exploded_error.json", {"error": repr(exc)})


def _write_render_gate(
    *,
    summary: dict[str, Any],
    overall_mesh: trimesh.Trimesh,
    object_dir: Path,
    extrinsic: Any,
    intrinsic: Any,
    resolution: int,
    max_faces: int,
) -> dict[str, Any]:
    ee_panel_path = Path(str(summary.get("mesh_png") or ""))
    post = render_component(
        overall_mesh,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
        max_faces=int(max_faces),
        color_mode="color",
    )
    out_path = object_dir / "render_gate" / "overall_gate.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    post_tile = _tile(post, "post overall.obj rendered with ee-eval camera", int(resolution), int(resolution) + 30)
    if ee_panel_path.is_file():
        ee = Image.open(ee_panel_path).convert("RGB")
        max_h = int(resolution) + 30
        ee.thumbnail((max(1, int(resolution) * 3), max_h), Image.Resampling.LANCZOS)
        ee_tile = Image.new("RGB", (ee.width, max_h), (255, 255, 255))
        draw = ImageDraw.Draw(ee_tile)
        draw.rectangle((0, 0, ee_tile.width, 30), fill=(0, 0, 0))
        draw.text((8, 9), "ee-eval original *__mesh.png", fill=(255, 255, 255))
        ee_tile.paste(ee, ((ee_tile.width - ee.width) // 2, 30 + (max_h - 30 - ee.height) // 2))
    else:
        ee_tile = Image.new("RGB", (int(resolution) * 2, int(resolution) + 30), (255, 245, 245))
        ImageDraw.Draw(ee_tile).text((8, 9), f"missing mesh_png: {ee_panel_path}", fill=(120, 0, 0))
    canvas = Image.new("RGB", (ee_tile.width + post_tile.width, int(resolution) + 30), (255, 255, 255))
    canvas.paste(ee_tile, (0, 0))
    canvas.paste(post_tile, (ee_tile.width, 0))
    canvas.save(out_path)
    return {
        "gate_image": str(out_path),
        "ee_mesh_png": str(ee_panel_path),
        "post_render_resolution": int(resolution),
        "post_render_max_faces": int(max_faces),
        "required_human_check": "post overall must match the run's own mesh.png orientation before batch panels are trusted",
    }


def _write_seg_diagnostic_22367(
    *,
    summary_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    run_dir = Path(summary["run_dir"])
    whole_path = run_dir / "voxel.npz"
    whole = _load_coords(whole_path)
    part_items = []
    rows: list[dict[str, Any]] = []
    part_sets: list[set[tuple[int, int, int]]] = []
    for item in summary.get("mujoco_part_meshes") or []:
        label = str(item.get("label") or "")
        part_idx = None
        bits = label.split("_")
        if len(bits) >= 2 and bits[0] == "part":
            try:
                part_idx = int(bits[1])
            except ValueError:
                part_idx = None
        if part_idx is None:
            continue
        path = run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz"
        coords = _load_coords(path) if path.is_file() else np.empty((0, 3), dtype=np.int64)
        part_items.append((label, coords))
        part_sets.append(_coords_set(coords))
        if len(coords):
            bbox_min = coords.min(axis=0).tolist()
            bbox_max = coords.max(axis=0).tolist()
            extent = (coords.max(axis=0) - coords.min(axis=0) + 1).tolist()
            thin_axes = [axis for axis, value in zip(("x", "y", "z"), extent) if int(value) <= 3]
        else:
            bbox_min = bbox_max = extent = []
            thin_axes = []
        rows.append(
            {
                "label": label,
                "voxel_count": int(len(coords)),
                "whole_fraction": float(len(coords) / max(len(whole), 1)),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bbox_extent_voxels": extent,
                "thin_axes_extent_le_3": ",".join(thin_axes),
            }
        )
    part_union = set().union(*part_sets) if part_sets else set()
    body_residual = np.asarray(sorted(_coords_set(whole) - part_union), dtype=np.int64)
    rows.append(
        {
            "label": "body_without_parts_residual",
            "voxel_count": int(len(body_residual)),
            "whole_fraction": float(len(body_residual) / max(len(whole), 1)),
            "bbox_min": body_residual.min(axis=0).tolist() if len(body_residual) else [],
            "bbox_max": body_residual.max(axis=0).tolist() if len(body_residual) else [],
            "bbox_extent_voxels": (body_residual.max(axis=0) - body_residual.min(axis=0) + 1).tolist() if len(body_residual) else [],
            "thin_axes_extent_le_3": "",
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "22367_partseg_voxel_counts.csv"
    _write_csv(csv_path, rows)
    png_path = out_dir / "22367_partseg_voxel.png"
    try:
        sys.path.insert(0, str(REPO_ROOT / "TRELLIS-arts"))
        from part_ss_eval_platform.eval_real_0615 import render_preview_voxel

        render_preview_voxel(whole, [("body_without_parts", body_residual), *part_items], png_path, "22367", 0)
    except Exception as exc:
        _write_json(out_dir / "22367_partseg_voxel_render_error.json", {"error": repr(exc)})
    ckpt = ((summary.get("part_stage") or {}).get("ckpt") or (summary.get("part_seg") or {}).get("ckpt") or "")
    ss_ckpt = ((summary.get("ss_stage") or {}).get("ckpt") or (summary.get("ss_flow") or {}).get("ckpt") or "")
    expected = "part_promptable_seg_full_S_0618-1/ckpts/step_100000.pt"
    diagnosis = "weight_ok_model_or_seed_sample_weak"
    if expected not in ckpt:
        diagnosis = "weight_mismatch"
    report = {
        "summary": str(summary_path),
        "run_dir": str(run_dir),
        "part_seg_ckpt": ckpt,
        "ss_flow_ckpt": ss_ckpt,
        "expected_part_seg_suffix": expected,
        "diagnosis": diagnosis,
        "voxel_counts_csv": str(csv_path),
        "voxel_preview_png": str(png_path) if png_path.is_file() else None,
        "rows": rows,
    }
    _write_json(out_dir / "22367_partseg_report.json", report)
    return report


def _write_mujoco_xml(out_xml: Path, route_paths: dict[str, Path]) -> Path:
    root = ET.Element("mujoco", {"model": _safe_name(out_xml.stem, 120)})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81"})
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", {"name": "object", "pos": "0 0 0"})
    for idx, (label, path) in enumerate(route_paths.items()):
        mesh_name = _safe_name(label, 80)
        rel = os.path.relpath(path, out_xml.parent)
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": rel})
        rgba = "0.72 0.76 0.80 1" if label == "body_without_parts" else f"{0.25 + (idx % 5) * 0.13:.3f} 0.45 0.70 1"
        ET.SubElement(body, "geom", {"name": mesh_name, "type": "mesh", "mesh": mesh_name, "rgba": rgba})
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return out_xml


def _mujoco_smoke(out_xml: Path, *, deps_dir: Path | None) -> dict[str, Any]:
    env = os.environ.copy()
    if deps_dir is not None and deps_dir.is_dir():
        env["PYTHONPATH"] = f"{deps_dir}:{env.get('PYTHONPATH', '')}"
    code = "import mujoco, sys; m=mujoco.MjModel.from_xml_path(sys.argv[1]); d=mujoco.MjData(m); mujoco.mj_step(m,d); print(m.ngeom)"
    proc = subprocess.run(
        [sys.executable, "-c", code, str(out_xml)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
        timeout=60,
    )
    return {"returncode": int(proc.returncode), "log": proc.stdout[-2000:]}


def _volume_sum(meshes: list[trimesh.Trimesh]) -> float:
    total = 0.0
    for mesh in meshes:
        try:
            total += abs(float(mesh.volume))
        except Exception:
            pass
    return total


def process_object(
    *,
    object_tag: str,
    dataset_id: str,
    object_id: str,
    angle: int,
    summary_path: Path,
    dilated_summary_path: Path | None,
    xpart_root: Path,
    out_dir: Path,
    deps_dir: Path | None,
    render_resolution: int,
    render_max_faces: int,
    metric_samples: int,
    completeness_samples: int,
    intersection_pitch: float,
    intersection_max_faces: int,
    overlap_mode: str,
    bdec_ckpt: Path,
    bdec_base_decoder_ckpt: Path,
    bdec_cache_manifest: Path,
    bdec_subset_dilation: int,
    bdec_gpu: int,
    reuse_existing_routes: bool,
    skip_exploded: bool,
    skip_mujoco: bool,
) -> dict[str, Any]:
    _ensure_manifold3d_available()
    run_dir, _whole_voxel, components, meta = _component_list(summary_path)
    summary = meta["summary"]
    object_key = f"{dataset_id}::{object_id}::angle_{int(angle):02d}"
    object_dir = out_dir / f"{object_tag}_{_safe_name(object_id)}"
    routes_dir = object_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    labels = [comp.label for comp in components]
    assets_dir = Path(meta["assets_dir"])
    overall_path = Path((summary.get("mujoco_overall_mesh") or {}).get("mesh_path") or assets_dir / "overall.obj")
    if not overall_path.is_file():
        raise FileNotFoundError(f"overall mesh missing: {overall_path}")
    overall_mesh = _load_mesh(overall_path)
    overall_boolean_mesh, overall_repair_status = _repair_boolean_input(overall_mesh)
    overall_points = _sample_points(overall_mesh, int(metric_samples), seed=123)
    whole_coords = _load_coords(Path(meta["whole_voxel"]))
    transform = _axis_self_check(overall_mesh, whole_coords, resolution=64)

    route_paths: dict[str, dict[str, Path]] = {
        "R-P": {},
        "R-P-fallback": {},
        "R-V": {},
        "R-D-raw": {},
        "R-D": {},
        "B-dec": {},
        "B-dec+clip": {},
        "R-X": {},
    }
    route_status: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in route_paths}

    extrinsic, intrinsic = _load_render_camera(summary, render_view=0)
    render_gate = _write_render_gate(
        summary=summary,
        overall_mesh=overall_mesh,
        object_dir=object_dir,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(render_resolution),
        max_faces=int(render_max_faces),
    )
    if bool(getattr(process_object, "_gate_only", False)):
        report = {
            "object_tag": object_tag,
            "object_key": object_key,
            "summary": str(summary_path),
            "overall_mesh": str(overall_path),
            "render_gate": render_gate,
            "voxel_mesh_transform": transform,
            "overall_boolean_repair": overall_repair_status,
            "components": labels,
        }
        _write_json(object_dir / "report.json", report)
        return {"report": report, "metrics": [], "route_summaries": []}

    # R-V: direct voxel marching cubes.
    for comp in components:
        out_path = routes_dir / "R-V" / f"{_safe_name(comp.label)}.obj"
        if bool(reuse_existing_routes) and out_path.is_file():
            route_paths["R-V"][comp.label] = out_path
            route_status["R-V"][comp.label] = {
                "seconds": 0.0,
                "source": "voxel_marching_cubes",
                "reused_existing_route_mesh": True,
            }
            continue
        if comp.coords is None or len(comp.coords) == 0:
            route_status["R-V"][comp.label] = {"error": "missing coords"}
            continue
        started = time.time()
        mesh = _voxel_mesh_from_coords(comp.coords, transform=transform, out_path=out_path)
        colors = _transfer_vertex_colors(overall_mesh, mesh)
        mesh.visual.vertex_colors = np.pad((np.clip(colors, 0, 1) * 255).astype(np.uint8), ((0, 0), (0, 1)), constant_values=255)
        mesh.export(out_path)
        route_paths["R-V"][comp.label] = out_path
        route_status["R-V"][comp.label] = {"seconds": float(time.time() - started), "source": "voxel_marching_cubes"}

    # R-P: boolean cut, with R-V fallback if the boolean engine is unavailable or fails.
    for comp in components:
        out_path = routes_dir / "R-P" / f"{_safe_name(comp.label)}.obj"
        fb_path = routes_dir / "R-P-fallback" / f"{_safe_name(comp.label)}.obj"
        if bool(reuse_existing_routes) and (out_path.is_file() or fb_path.is_file()):
            status = {
                "engine": "trimesh.boolean.intersection",
                "reused_existing_route_mesh": True,
                "seconds": 0.0,
            }
            if out_path.is_file():
                route_paths["R-P"][comp.label] = out_path
                status["true_boolean"] = True
                status["fallback_used"] = False
            else:
                route_paths["R-P-fallback"][comp.label] = fb_path
                route_status["R-P-fallback"][comp.label] = {
                    **status,
                    "fallback_used": True,
                    "fallback_route": "R-V",
                }
                status["fallback_used"] = True
                status["fallback_route"] = "R-V"
            route_status["R-P"][comp.label] = status
            continue
        if comp.coords is None or len(comp.coords) == 0:
            route_status["R-P"][comp.label] = {"error": "missing coords", "fallback_used": True}
            continue
        started = time.time()
        mesh, status = _try_boolean_intersection(
            overall_boolean_mesh,
            comp.coords,
            transform=transform,
            out_path=out_path,
            resolution=64,
        )
        if mesh is None:
            rv_path = route_paths["R-V"].get(comp.label)
            if rv_path is not None and rv_path.is_file():
                fb_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rv_path, fb_path)
                mesh = _load_mesh(fb_path)
                status["fallback_used"] = True
                status["fallback_route"] = "R-V"
                route_paths["R-P-fallback"][comp.label] = fb_path
            else:
                status["fallback_used"] = True
                status["fallback_route"] = None
        if mesh is not None:
            if bool(status.get("fallback_used")):
                route_status["R-P-fallback"][comp.label] = dict(status)
            else:
                route_paths["R-P"][comp.label] = out_path
                status["true_boolean"] = True
        status["seconds"] = float(time.time() - started)
        route_status["R-P"][comp.label] = status

    # R-D: dilated sparse subset decode from a separately-run ee_0617_single.
    route_paths["R-D-raw"] = _dilated_meshes_from_summary(dilated_summary_path, routes_dir / "R-D-raw", labels)
    for comp in components:
        route_status["R-D-raw"][comp.label] = {
            "source": "dilated_sparse_subset_decode",
            "available": comp.label in route_paths["R-D-raw"],
            "summary": None if dilated_summary_path is None else str(dilated_summary_path),
        }
    route_paths["R-D"], route_status["R-D"] = _trim_dilated_meshes(
        dilated_paths=route_paths["R-D-raw"],
        components=components,
        transform=transform,
        out_dir=routes_dir / "R-D",
        reuse_existing=bool(reuse_existing_routes),
    )

    # B-dec: RouteB-3K expanded-subset decoder.  This is a pure evaluation
    # slot: no training, no post-hoc deletion, and current ee-eval component
    # voxels define the support for every component.
    route_paths["B-dec"], route_status["B-dec"] = _run_bdec_route(
        object_tag=object_tag,
        dataset_id=dataset_id,
        object_id=object_id,
        angle=int(angle),
        components=components,
        overall_mesh=overall_mesh,
        transform=transform,
        out_dir=routes_dir / "B-dec",
        ckpt=bdec_ckpt,
        base_decoder_ckpt=bdec_base_decoder_ckpt,
        cache_manifest=bdec_cache_manifest,
        subset_dilation=int(bdec_subset_dilation),
        gpu=int(bdec_gpu),
        reuse_existing=bool(reuse_existing_routes),
    )

    route_paths["B-dec+clip"], route_status["B-dec+clip"] = _run_bdec_clip_route(
        bdec_paths=route_paths["B-dec"],
        components=components,
        transform=transform,
        out_dir=routes_dir / "B-dec+clip",
        resolution=64,
        reuse_existing=bool(reuse_existing_routes),
    )

    # R-X: reuse corrected X-Part s8_o384/seg_surface when present.
    xpart_report = _load_xpart_report(object_tag, dataset_id, object_id, angle, xpart_root)
    route_paths["R-X"] = _copy_route_meshes(
        _after_meshes_from_report(xpart_report),
        routes_dir / "R-X",
        labels,
    )
    for comp in components:
        route_status["R-X"][comp.label] = {
            "source": "xpart_corrected_s8_o384_or_existing_report",
            "available": comp.label in route_paths["R-X"],
            "report": None if xpart_report is None else str(xpart_report),
        }

    rows: list[dict[str, Any]] = []
    route_summaries: list[dict[str, Any]] = []
    for route, paths in route_paths.items():
        after_meshes: list[tuple[str, trimesh.Trimesh]] = []
        for idx, comp in enumerate(components):
            path = paths.get(comp.label)
            if path is None or not path.is_file():
                continue
            mesh = _load_mesh(path)
            after_meshes.append((comp.label, mesh))
            rows.append(
                _metric_row(
                    object_tag=object_tag,
                    object_key=object_key,
                    route=route,
                    comp=comp,
                    after_path=path,
                    after_mesh=mesh,
                    overall_points=overall_points,
                    route_status=route_status.get(route, {}).get(comp.label),
                    metric_samples=int(metric_samples),
                    completeness_samples=int(completeness_samples),
                    idx=idx,
                )
            )
        overlap = _voxel_overlap_volume(
            after_meshes,
            pitch=float(intersection_pitch),
            max_faces=int(intersection_max_faces),
            mode=str(overlap_mode),
        )
        meshes_only = [mesh for _label, mesh in after_meshes]
        route_summaries.append(
            {
                "object_tag": object_tag,
                "object_key": object_key,
                "route": route,
                "components": len(after_meshes),
                "expected_components": len(components),
                "watertight": sum(1 for _label, mesh in after_meshes if bool(mesh.is_watertight)),
                "total_overlap_voxels": overlap.get("total_overlap_voxels"),
                "max_pair_overlap_voxels": overlap.get("max_pair_overlap_voxels"),
                "bbox_overlap_volume": overlap.get("total_bbox_overlap_volume"),
                "max_pair_bbox_overlap_volume": overlap.get("max_pair_bbox_overlap_volume"),
                "overlap_mode": overlap.get("mode"),
                "sum_abs_volume": _volume_sum(meshes_only),
                "overall_abs_volume": abs(float(overall_mesh.volume)) if np.isfinite(overall_mesh.volume) else float("nan"),
                "sum_volume_over_overall": (
                    _volume_sum(meshes_only) / max(abs(float(overall_mesh.volume)), 1.0e-12)
                    if np.isfinite(overall_mesh.volume)
                    else float("nan")
                ),
                "boolean_failures": sum(
                    1 for status in route_status.get(route, {}).values() if status.get("fallback_used")
                )
                if route == "R-P"
                else None,
            }
        )

    _render_route_panel(
        components=components,
        routes=route_paths,
        out_path=object_dir / "six_column_components.png",
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(render_resolution),
        max_faces=int(render_max_faces),
    )
    if not bool(skip_exploded):
        _write_exploded(
            components=components,
            routes={"before": {}, **route_paths},
            out_dir=object_dir / "exploded",
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=int(render_resolution),
            max_faces=int(render_max_faces),
        )

    mujoco_reports: dict[str, Any] = {}
    if not bool(skip_mujoco):
        for route, paths in route_paths.items():
            if not paths:
                continue
            xml = _write_mujoco_xml(object_dir / "mujoco" / route / f"{_safe_name(object_tag)}_{route}.xml", paths)
            mujoco_reports[route] = _mujoco_smoke(xml, deps_dir=deps_dir)

    _write_csv(object_dir / "metrics.csv", rows)
    _write_csv(object_dir / "route_summary.csv", route_summaries)
    report = {
        "object_tag": object_tag,
        "object_key": object_key,
        "summary": str(summary_path),
        "dilated_summary": None if dilated_summary_path is None else str(dilated_summary_path),
        "overall_mesh": str(overall_path),
        "overall_boolean_repair": overall_repair_status,
        "render_gate": render_gate,
        "voxel_mesh_transform": transform,
        "components": labels,
        "route_status": route_status,
        "route_summaries": route_summaries,
        "metrics_csv": str(object_dir / "metrics.csv"),
        "route_summary_csv": str(object_dir / "route_summary.csv"),
        "six_column_panel": str(object_dir / "six_column_components.png"),
        "exploded_dir": None if bool(skip_exploded) else str(object_dir / "exploded"),
        "skip_exploded": bool(skip_exploded),
        "mujoco_smoke": mujoco_reports,
        "skip_mujoco": bool(skip_mujoco),
    }
    _write_json(object_dir / "report.json", report)
    return {"report": report, "metrics": rows, "route_summaries": route_summaries}


def _aggregate(out_dir: Path, all_metrics: list[dict[str, Any]], all_route_summaries: list[dict[str, Any]], reports: list[dict[str, Any]]) -> None:
    _write_csv(out_dir / "all_metrics.csv", all_metrics)
    _write_csv(out_dir / "all_route_summary.csv", all_route_summaries)
    agg: list[dict[str, Any]] = []
    for route in ["R-P", "R-P-fallback", "R-V", "R-D-raw", "R-D", "B-dec", "B-dec+clip", "R-X"]:
        rows = [row for row in all_metrics if row.get("route") == route]
        if not rows:
            continue
        agg.append(
            {
                "route": route,
                "objects": len({row["object_key"] for row in rows}),
                "components": len(rows),
                "watertight": sum(1 for row in rows if _truthy(row.get("after_is_watertight"))),
                "min_coverage_0p01": _min_float(rows, "before_to_after_coverage_0p01"),
                "min_coverage_0p02": _min_float(rows, "before_to_after_coverage_0p02"),
                "max_bidir_p95": _max_float(rows, "bidirectional_chamfer_p95_max"),
                "max_open_edge_ratio": _max_float(rows, "after_open_edge_ratio"),
                "max_hole_count": _max_float(rows, "after_hole_count"),
                "max_connected_components": _max_float(rows, "after_connected_components"),
                "mean_after_dihedral": _mean_float(rows, "after_mean_dihedral_rad"),
                "mean_before_dihedral": _mean_float(rows, "before_mean_dihedral_rad"),
            }
        )
    _write_csv(out_dir / "aggregate_by_route.csv", agg)

    def _route_agg(route: str) -> dict[str, Any] | None:
        for row in agg:
            if row.get("route") == route:
                return row
        return None

    def _route_summary_values(route: str, key: str) -> list[float]:
        values = []
        for row in all_route_summaries:
            if row.get("route") != route:
                continue
            try:
                value = float(row.get(key, float("nan")))
            except Exception:
                continue
            if np.isfinite(value):
                values.append(value)
        return values

    rd = _route_agg("R-D")
    bdec = _route_agg("B-dec")
    rd_rows = [row for row in all_metrics if row.get("route") == "R-D"]
    bdec_rows = [row for row in all_metrics if row.get("route") == "B-dec"]
    rd_open_mean = _mean_float(rd_rows, "after_open_edge_ratio")
    bdec_open_mean = _mean_float(bdec_rows, "after_open_edge_ratio")
    rd_overlap_values = _route_summary_values("R-D", "total_overlap_voxels")
    bdec_overlap_values = _route_summary_values("B-dec", "total_overlap_voxels")
    rd_pair_values = _route_summary_values("R-D", "max_pair_overlap_voxels")
    bdec_pair_values = _route_summary_values("B-dec", "max_pair_overlap_voxels")
    overlap_metric = "voxel_overlap"
    if not rd_overlap_values or not bdec_overlap_values:
        rd_overlap_values = _route_summary_values("R-D", "bbox_overlap_volume")
        bdec_overlap_values = _route_summary_values("B-dec", "bbox_overlap_volume")
        rd_pair_values = rd_overlap_values
        bdec_pair_values = bdec_overlap_values
        overlap_metric = "bbox_overlap_volume_fallback"
    rd_overlap_sum = float(np.sum(rd_overlap_values)) if rd_overlap_values else float("nan")
    bdec_overlap_sum = float(np.sum(bdec_overlap_values)) if bdec_overlap_values else float("nan")
    rd_max_pair_overlap = float(np.max(rd_pair_values)) if rd_pair_values else float("nan")
    bdec_max_pair_overlap = float(np.max(bdec_pair_values)) if bdec_pair_values else float("nan")
    open_better = bool(
        rd is not None
        and bdec is not None
        and np.isfinite(rd_open_mean)
        and np.isfinite(bdec_open_mean)
        and np.isfinite(float(rd["max_open_edge_ratio"]))
        and np.isfinite(float(bdec["max_open_edge_ratio"]))
        and bdec_open_mean <= rd_open_mean * 0.5
        and float(bdec["max_open_edge_ratio"]) < float(rd["max_open_edge_ratio"])
    )
    overlap_better = bool(
        rd is not None
        and bdec is not None
        and np.isfinite(rd_overlap_sum)
        and np.isfinite(bdec_overlap_sum)
        and np.isfinite(rd_max_pair_overlap)
        and np.isfinite(bdec_max_pair_overlap)
        and bdec_overlap_sum <= rd_overlap_sum * 0.5
        and bdec_max_pair_overlap < rd_max_pair_overlap
    )
    decision = {
        "criterion": (
            "continue RouteB only if B-dec is significantly better than R-D on both open edges and neighbor overlap; "
            "significant is fixed here as mean open_edge_ratio <= 50% of R-D plus lower max open_edge_ratio, "
            "and total/max-pair overlap <= 50%/lower than R-D"
        ),
        "rd_mean_open_edge_ratio": rd_open_mean,
        "bdec_mean_open_edge_ratio": bdec_open_mean,
        "rd_max_open_edge_ratio": None if rd is None else rd.get("max_open_edge_ratio"),
        "bdec_max_open_edge_ratio": None if bdec is None else bdec.get("max_open_edge_ratio"),
        "rd_total_overlap_voxels_sum": rd_overlap_sum,
        "bdec_total_overlap_voxels_sum": bdec_overlap_sum,
        "rd_max_pair_overlap_voxels_max": rd_max_pair_overlap,
        "bdec_max_pair_overlap_voxels_max": bdec_max_pair_overlap,
        "overlap_metric_used": overlap_metric,
        "open_edge_significantly_better": open_better,
        "overlap_significantly_better": overlap_better,
        "verdict": (
            "RouteB learned boundary closure; continue training with seam-focused camera sampling, boundary-band render loss weighting, and LR sweep."
            if open_better and overlap_better
            else "Track1 RouteB has no decisive incremental value over R-D under the fixed rule; park Track1 and keep before visual mesh plus R-V collision mesh."
        ),
    }
    _write_json(out_dir / "bdec_decision.json", decision)

    clip = _route_agg("B-dec+clip")
    clip_rows = [row for row in all_metrics if row.get("route") == "B-dec+clip"]
    clip_overlap_values = _route_summary_values("B-dec+clip", "total_overlap_voxels")
    clip_pair_values = _route_summary_values("B-dec+clip", "max_pair_overlap_voxels")
    clip_overlap_metric = "voxel_overlap"
    if not clip_overlap_values:
        clip_overlap_values = _route_summary_values("B-dec+clip", "bbox_overlap_volume")
        clip_pair_values = _route_summary_values("B-dec+clip", "max_pair_bbox_overlap_volume")
        clip_overlap_metric = "bbox_overlap_volume_fallback"
    clip_overlap_sum = float(np.sum(clip_overlap_values)) if clip_overlap_values else float("nan")
    clip_max_pair_overlap = float(np.max(clip_pair_values)) if clip_pair_values else float("nan")
    clip_max_open = float(clip["max_open_edge_ratio"]) if clip is not None and np.isfinite(float(clip["max_open_edge_ratio"])) else float("nan")
    bdec_by_comp = {(row.get("object_key"), row.get("component")): row for row in bdec_rows}
    coverage_drops: list[dict[str, Any]] = []
    for row in clip_rows:
        key = (row.get("object_key"), row.get("component"))
        base = bdec_by_comp.get(key)
        if base is None:
            continue
        item: dict[str, Any] = {
            "object_tag": row.get("object_tag"),
            "component": row.get("component"),
        }
        for metric in ("before_to_after_coverage_0p01", "before_to_after_coverage_0p02"):
            try:
                before_value = float(base.get(metric, float("nan")))
                after_value = float(row.get(metric, float("nan")))
                drop = before_value - after_value
            except Exception:
                before_value = after_value = drop = float("nan")
            item[f"bdec_{metric}"] = before_value
            item[f"clip_{metric}"] = after_value
            item[f"drop_{metric}"] = drop
        coverage_drops.append(item)
    max_cov_drop_01 = _max_float(coverage_drops, "drop_before_to_after_coverage_0p01")
    max_cov_drop_02 = _max_float(coverage_drops, "drop_before_to_after_coverage_0p02")
    worst_drop = None
    if coverage_drops:
        worst_drop = max(
            coverage_drops,
            key=lambda row: max(
                float(row.get("drop_before_to_after_coverage_0p01", float("-inf"))),
                float(row.get("drop_before_to_after_coverage_0p02", float("-inf"))),
            ),
        )
    bdec_min_cov_row = None
    if bdec_rows:
        bdec_min_cov_row = min(
            bdec_rows,
            key=lambda row: float(row.get("before_to_after_coverage_0p01", float("inf"))),
        )
    nonwatertight_sources: list[dict[str, Any]] = []
    unresolved_repairs: list[dict[str, Any]] = []
    for report in reports:
        route_status = ((report.get("route_status") or {}).get("B-dec+clip") or {})
        for label, status in route_status.items():
            if str(label).startswith("_") or not isinstance(status, dict):
                continue
            if status.get("source_watertight") is False:
                item = {
                    "object_tag": report.get("object_tag"),
                    "object_key": report.get("object_key"),
                    "component": label,
                    "source_open_edge_ratio": status.get("source_open_edge_ratio"),
                    "source_hole_count": status.get("source_hole_count"),
                    "after_fill_holes_watertight": status.get("after_fill_holes_watertight"),
                    "after_fill_holes_open_edge_ratio": status.get("after_fill_holes_open_edge_ratio"),
                    "after_fill_holes_hole_count": status.get("after_fill_holes_hole_count"),
                    "repair_error": status.get("repair_error"),
                    "clip_available": status.get("available"),
                    "clip_error": status.get("error"),
                }
                nonwatertight_sources.append(item)
                if status.get("repair_unresolved") or status.get("repair_error") or not status.get("available"):
                    unresolved_repairs.append(item)
    overlap_ok = bool(
        clip_overlap_metric == "voxel_overlap"
        and np.isfinite(clip_overlap_sum)
        and np.isfinite(clip_max_pair_overlap)
        and clip_overlap_sum <= 20.0
        and clip_max_pair_overlap <= 10.0
    )
    open_ok = bool(np.isfinite(clip_max_open) and clip_max_open <= 0.002)
    coverage_ok = bool(
        np.isfinite(max_cov_drop_01)
        and np.isfinite(max_cov_drop_02)
        and max_cov_drop_01 <= 0.02
        and max_cov_drop_02 <= 0.02
    )
    clip_decision = {
        "criterion": (
            "flip only if B-dec+clip has voxel overlap total<=20 and max_pair<=10, "
            "max open_edge_ratio<=0.002, and per-component coverage@0.01/@0.02 drops by <=0.02 versus B-dec"
        ),
        "clip_overlap_metric_used": clip_overlap_metric,
        "clip_overlap_sum": clip_overlap_sum,
        "clip_max_pair_overlap": clip_max_pair_overlap,
        "clip_max_open_edge_ratio": clip_max_open,
        "max_coverage_drop_0p01_vs_bdec": max_cov_drop_01,
        "max_coverage_drop_0p02_vs_bdec": max_cov_drop_02,
        "worst_coverage_drop": worst_drop,
        "bdec_min_coverage_0p01_component": None
        if bdec_min_cov_row is None
        else {
            "object_tag": bdec_min_cov_row.get("object_tag"),
            "object_key": bdec_min_cov_row.get("object_key"),
            "component": bdec_min_cov_row.get("component"),
            "coverage_0p01": bdec_min_cov_row.get("before_to_after_coverage_0p01"),
            "coverage_0p02": bdec_min_cov_row.get("before_to_after_coverage_0p02"),
            "after_mesh": bdec_min_cov_row.get("after_mesh"),
        },
        "bdec_source_nonwatertight_components": nonwatertight_sources,
        "bdec_source_repair_unresolved_components": unresolved_repairs,
        "overlap_ok": overlap_ok,
        "open_edge_ok": open_ok,
        "coverage_ok": coverage_ok,
        "verdict": (
            "FLIP: B-dec+clip passes; use B-dec+clip as visual mesh candidate and propose scale-up training only."
            if overlap_ok and open_ok and coverage_ok
            else "NO FLIP: B-dec+clip does not satisfy the fixed rule; keep Track1 parked."
        ),
    }
    _write_json(out_dir / "bdec_clip_decision.json", clip_decision)

    lines = [
        "# Part Mesh Route Comparison",
        "",
        f"out_dir: `{out_dir}`",
        "",
        "## Gate 0",
        "",
        "- Four ee-eval runs passed runbook checks: `flow_calls=1`, `live_official_trellis_rgba`, `concat`, `body_without_parts`, `overall.obj`, and component OBJ faces > 0.",
        "- Seed is recorded in summaries; SS flow seed is passed into `infer_stage.py`, and SLat seed is passed into `run_slat_flow_from_tokens`.",
        "",
        "## Route Summary",
        "",
        "| route | objects | components | watertight | min cov@0.01 | min cov@0.02 | max bidir p95 | max open-edge | max holes | mean dihedral before->after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            "| {route} | {objects} | {components} | {watertight}/{components} | {cov1:.4f} | {cov2:.4f} | {p95:.5f} | {openr:.5f} | {holes:.0f} | {bd:.4f}->{ad:.4f} |".format(
                route=row["route"],
                objects=row["objects"],
                components=row["components"],
                watertight=row["watertight"],
                cov1=float(row["min_coverage_0p01"]),
                cov2=float(row["min_coverage_0p02"]),
                p95=float(row["max_bidir_p95"]),
                openr=float(row["max_open_edge_ratio"]),
                holes=float(row["max_hole_count"]),
                bd=float(row["mean_before_dihedral"]),
                ad=float(row["mean_after_dihedral"]),
            )
        )
    lines.extend(
        [
            "",
            "## Panels",
            "",
            "| object | six-column panel | exploded dir | render gate |",
            "|---|---|---|---|",
        ]
    )
    for report in reports:
        panel = report.get("six_column_panel", "")
        exploded = report.get("exploded_dir", "")
        gate = (report.get("render_gate") or {}).get("gate_image", "")
        lines.append(f"| {report['object_tag']} `{report['object_key']}` | `{panel}` | `{exploded}` | gate `{gate}` |")
    rp_rows = [row for row in all_route_summaries if row.get("route") == "R-P"]
    rp_fail = sum(int(row.get("boolean_failures") or 0) for row in rp_rows)
    rp_total = sum(int(row.get("expected_components") or 0) for row in rp_rows)
    rp_true = sum(int(row.get("components") or 0) for row in rp_rows)
    lines.extend(
        [
            "",
            "## R-P Boolean Status",
            "",
            f"- `manifold3d` import gate passed; true boolean components: `{rp_true}`, fallback components: `{rp_fail}/{rp_total}`.",
            "- `R-P` aggregate excludes fallback components. Fallback meshes are written and measured under `R-P-fallback`.",
            "- Render gate images are listed in the panel table; post panels use exported OBJ Y-up -> renderer Z-up conversion and the ee-eval MeshRenderer camera.",
            "",
            "## R-D Speckle Diagnosis",
            "",
            "- `R-D-raw` is the untrimmed dilation=1 sparse subset decode baseline.",
            "- `R-D` is the repaired trim path: face ownership from original component voxels, smoothed by face-adjacency majority vote before clipping.",
            "- `B-dec` is RouteB-3K expanded-subset decoder output, using current ee-eval component voxels as support and no post-hoc face deletion.",
            "- Verdict weighting treats open-edge ratio / holes / connected components as first-class failures because coverage can stay high on pinholed meshes.",
            "",
            "## B-dec Decision",
            "",
            f"- Fixed criterion: {decision['criterion']}.",
            f"- R-D mean/max open_edge_ratio: `{rd_open_mean:.6f}` / `{float('nan') if rd is None else float(rd['max_open_edge_ratio']):.6f}`.",
            f"- B-dec mean/max open_edge_ratio: `{bdec_open_mean:.6f}` / `{float('nan') if bdec is None else float(bdec['max_open_edge_ratio']):.6f}`.",
            f"- Overlap metric used: `{overlap_metric}`.",
            f"- R-D overlap sum/max: `{rd_overlap_sum:.6f}` / `{rd_max_pair_overlap:.6f}`.",
            f"- B-dec overlap sum/max: `{bdec_overlap_sum:.6f}` / `{bdec_max_pair_overlap:.6f}`.",
            f"- Verdict: **{decision['verdict']}**",
            "",
            "## B-dec+clip Supplement Decision",
            "",
            f"- Fixed criterion: {clip_decision['criterion']}.",
            f"- Clip overlap metric: `{clip_overlap_metric}`; sum/max-pair: `{clip_overlap_sum:.6f}` / `{clip_max_pair_overlap:.6f}`.",
            f"- Clip max open_edge_ratio: `{clip_max_open:.6f}`.",
            f"- Worst coverage drop vs B-dec: cov@0.01 `{max_cov_drop_01:.6f}`, cov@0.02 `{max_cov_drop_02:.6f}`.",
            f"- B-dec min cov@0.01 component: `{(clip_decision['bdec_min_coverage_0p01_component'] or {}).get('object_tag')}` / `{(clip_decision['bdec_min_coverage_0p01_component'] or {}).get('component')}`.",
            f"- B-dec source non-watertight before clip: `{len(nonwatertight_sources)}`; unresolved after fill_holes or failed clip: `{len(unresolved_repairs)}`.",
            f"- Supplement verdict: **{clip_decision['verdict']}**",
            "",
            "## Files",
            "",
            f"- Metrics: `{out_dir / 'all_metrics.csv'}`",
            f"- Route summaries: `{out_dir / 'all_route_summary.csv'}`",
            f"- Aggregate by route: `{out_dir / 'aggregate_by_route.csv'}`",
            f"- B-dec decision JSON: `{out_dir / 'bdec_decision.json'}`",
            f"- B-dec+clip decision JSON: `{out_dir / 'bdec_clip_decision.json'}`",
        ]
    )
    (out_dir / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        out_dir / "aggregate_report.json",
        {"routes": agg, "reports": reports, "bdec_decision": decision, "bdec_clip_decision": clip_decision},
    )


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(args.eval_dir).resolve()
    dilated_dir = Path(args.dilated_eval_dir).resolve() if args.dilated_eval_dir else None
    _ensure_manifold3d_available()
    process_object._gate_only = bool(args.gate_only)  # type: ignore[attr-defined]
    seg_summary = _summary_path(eval_dir, "physx-0511-drawer-door", "22367", 0)
    if seg_summary.is_file():
        _write_seg_diagnostic_22367(summary_path=seg_summary, out_dir=out_dir / "seg_diagnostics")
    all_metrics: list[dict[str, Any]] = []
    all_route_summaries: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for tag, dataset_id, object_id, angle in OBJECTS:
        summary = _summary_path(eval_dir, dataset_id, object_id, angle)
        dilated = _summary_path(dilated_dir, dataset_id, object_id, angle) if dilated_dir is not None else None
        if dilated is not None and not dilated.is_file():
            dilated = None
        print(f"[part-routes] process {tag} {dataset_id}::{object_id} angle={angle}", flush=True)
        result = process_object(
            object_tag=tag,
            dataset_id=dataset_id,
            object_id=object_id,
            angle=angle,
            summary_path=summary,
            dilated_summary_path=dilated,
            xpart_root=Path(args.xpart_root).resolve(),
            out_dir=out_dir,
            deps_dir=Path(args.deps_dir).resolve() if args.deps_dir else None,
            render_resolution=int(args.render_resolution),
            render_max_faces=int(args.render_max_faces),
            metric_samples=int(args.metric_samples),
            completeness_samples=int(args.completeness_samples),
            intersection_pitch=float(args.intersection_pitch),
            intersection_max_faces=int(args.intersection_max_faces),
            overlap_mode=str(args.overlap_mode),
            bdec_ckpt=Path(args.bdec_ckpt).resolve(),
            bdec_base_decoder_ckpt=Path(args.bdec_base_decoder_ckpt).resolve(),
            bdec_cache_manifest=Path(args.bdec_cache_manifest).resolve(),
            bdec_subset_dilation=int(args.bdec_subset_dilation),
            bdec_gpu=int(args.bdec_gpu),
            reuse_existing_routes=bool(args.reuse_existing_routes),
            skip_exploded=bool(args.skip_exploded),
            skip_mujoco=bool(args.skip_mujoco),
        )
        all_metrics.extend(result["metrics"])
        all_route_summaries.extend(result["route_summaries"])
        reports.append(result["report"])
    _aggregate(out_dir, all_metrics, all_route_summaries, reports)
    print(f"[part-routes] aggregate -> {out_dir / 'aggregate_report.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--dilated-eval-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--xpart-root", type=Path, default=Path("/robot/data-lab/jzh/art-gen/ee-eval/xpart_corrected_0702"))
    parser.add_argument("--deps-dir", type=Path, default=Path("/robot/data-lab/jzh/art-gen/ee-eval/part_mesh_routes_0702/_deps"))
    parser.add_argument("--render-resolution", type=int, default=256)
    parser.add_argument("--render-max-faces", type=int, default=400000)
    parser.add_argument("--metric-samples", type=int, default=30000)
    parser.add_argument("--completeness-samples", type=int, default=50000)
    parser.add_argument("--intersection-pitch", type=float, default=0.02)
    parser.add_argument("--intersection-max-faces", type=int, default=100000)
    parser.add_argument("--overlap-mode", choices=["bbox", "voxel", "skip"], default="bbox")
    parser.add_argument("--bdec-ckpt", type=Path, default=DEFAULT_BDEC_CKPT)
    parser.add_argument("--bdec-base-decoder-ckpt", type=Path, default=DEFAULT_BDEC_BASE_DECODER_CKPT)
    parser.add_argument("--bdec-cache-manifest", type=Path, default=DEFAULT_BDEC_CACHE_MANIFEST)
    parser.add_argument("--bdec-subset-dilation", type=int, default=1)
    parser.add_argument("--bdec-gpu", type=int, default=0)
    parser.add_argument("--reuse-existing-routes", action="store_true")
    parser.add_argument("--skip-exploded", action="store_true")
    parser.add_argument("--skip-mujoco", action="store_true")
    parser.add_argument("--gate-only", action="store_true", help="Only render overall orientation gate and diagnostics; skip route generation.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
