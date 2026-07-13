#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


SAM3D_Z_UP_TO_Y_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
ROTX_POS_90 = SAM3D_Z_UP_TO_Y_UP
ROTX_NEG_90 = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
COLORS = [
    (214, 57, 65),
    (47, 124, 188),
    (63, 158, 82),
    (145, 91, 181),
    (229, 142, 40),
    (111, 91, 74),
    (210, 93, 154),
    (88, 150, 153),
    (172, 120, 42),
    (82, 82, 82),
]


def _safe_name(value: str, max_len: int = 96) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value)).strip("_")
    return (out or "part")[:max_len]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_npz_coords(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        for key in ("coords", "points", "indices", "arr_0"):
            if key in data:
                arr = np.asarray(data[key])
                break
        else:
            if len(data.files) != 1:
                raise ValueError(f"{path}: cannot infer coords key from {data.files}")
            arr = np.asarray(data[data.files[0]])
    if arr.ndim in (3, 4):
        arr = np.argwhere(arr > 0)
    elif arr.ndim == 2 and arr.shape[1] >= 4:
        arr = arr[:, -3:]
    elif arr.ndim == 2 and arr.shape[1] == 3:
        pass
    else:
        raise ValueError(f"{path}: unsupported voxel array shape {arr.shape}")
    coords = np.asarray(arr[:, :3], dtype=np.int64)
    if coords.size == 0:
        raise ValueError(f"{path}: empty voxel coords")
    if bool(((coords < 0) | (coords >= 64)).any()):
        raise ValueError(f"{path}: voxel coords outside [0,64)")
    return np.ascontiguousarray(coords)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    mask = np.zeros((64, 64, 64), dtype=bool)
    coords = np.asarray(coords, dtype=np.int64)
    mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    return np.ascontiguousarray(coords.astype(np.int64, copy=False))


def _voxel_centers(coords: np.ndarray) -> np.ndarray:
    return (np.asarray(coords, dtype=np.float64) + 0.5) / 64.0 - 0.5


def _mesh_to_voxel_indices(vertices: np.ndarray, scale: float, rot: np.ndarray) -> np.ndarray:
    zup_norm = (np.asarray(vertices, dtype=np.float64) @ rot.T) / float(scale)
    return np.floor((zup_norm + 0.5) * 64.0).astype(np.int64)


def _voxel_to_mesh_points(coords: np.ndarray, scale: float, rot: np.ndarray) -> np.ndarray:
    return (_voxel_centers(coords) * float(scale)) @ rot


def _occupancy_iou(vertices: np.ndarray, whole_mask: np.ndarray, scale: float, rot: np.ndarray) -> float:
    idx = _mesh_to_voxel_indices(vertices, scale, rot)
    valid = np.all((idx >= 0) & (idx < 64), axis=1)
    pred = np.zeros((64, 64, 64), dtype=bool)
    if bool(valid.any()):
        v = idx[valid]
        pred[v[:, 0], v[:, 1], v[:, 2]] = True
    inter = np.count_nonzero(pred & whole_mask)
    union = np.count_nonzero(pred | whole_mask)
    return float(inter / union) if union else 0.0


def _candidate_scales(vertices: np.ndarray, whole_coords: np.ndarray) -> list[float]:
    mesh_extent = np.ptp(np.asarray(vertices, dtype=np.float64), axis=0)
    voxel_extent = np.ptp(_voxel_centers(whole_coords), axis=0)
    scales: list[float] = []
    for m, v in zip(mesh_extent, voxel_extent):
        if v > 1e-8 and m > 1e-8:
            scales.append(float(m / v))
    max_abs = float(np.max(np.abs(vertices)))
    if max_abs > 1e-8:
        scales.append(max_abs * 2.0)
    scales.extend([1.0, 2.0])
    out: list[float] = []
    for value in scales:
        if not np.isfinite(value) or value <= 0:
            continue
        for factor in (0.70, 0.85, 1.0, 1.15, 1.30):
            candidate = float(value * factor)
            if candidate > 0 and all(abs(candidate - x) > 1e-5 for x in out):
                out.append(candidate)
    return out


def _calibrate_transform(mesh: trimesh.Trimesh, whole_coords: np.ndarray) -> dict[str, Any]:
    whole_mask = _coords_to_mask(whole_coords)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rotations = [
        ("sam3d_zup_to_yup", SAM3D_Z_UP_TO_Y_UP),
        ("x_pos_90", ROTX_POS_90),
        ("x_neg_90", ROTX_NEG_90),
    ]
    best: dict[str, Any] | None = None
    for scale in _candidate_scales(vertices, whole_coords):
        for name, rot in rotations:
            iou = _occupancy_iou(vertices, whole_mask, scale, rot)
            record = {"name": name, "scale": float(scale), "rotation": rot.tolist(), "iou": float(iou)}
            if best is None or iou > best["iou"]:
                best = record
    if best is None:
        raise RuntimeError("failed to build coordinate transform candidates")
    if float(best["iou"]) < 0.5:
        raise RuntimeError(f"coordinate self-check failed: IoU={best['iou']:.6f} < 0.5 transform={best}")
    return best


def _part_index_from_path(path: Path) -> int:
    match = re.search(r"part_(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer part index from {path}")
    return int(match.group(1))


def _load_part_voxels(pattern: str) -> list[dict[str, Any]]:
    paths = sorted((Path(p) for p in glob.glob(pattern)), key=_part_index_from_path)
    if not paths:
        raise FileNotFoundError(f"part voxel glob matched nothing: {pattern}")
    return [{"index": _part_index_from_path(path), "path": str(path.resolve()), "coords": _load_npz_coords(path)} for path in paths]


def _load_before_meshes(pattern: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for raw in glob.glob(pattern):
        path = Path(raw)
        if path.name in {"body.obj", "overall.obj"}:
            continue
        match = re.search(r"part_(\d+)", path.name)
        if match:
            out.setdefault(int(match.group(1)), path)
    return out


def _part_label(path: Path | None, index: int) -> str:
    if path is None:
        return f"part_{index:02d}"
    stem = path.stem
    return stem if stem.startswith(f"part_{index:02d}") else f"part_{index:02d}_{stem}"


def _make_class_items(whole_coords: np.ndarray, part_items: list[dict[str, Any]], before_paths: dict[int, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    whole_mask = _coords_to_mask(whole_coords)
    movable_union = np.zeros_like(whole_mask)
    for item in part_items:
        movable_union |= _coords_to_mask(item["coords"])
    body_mask = whole_mask & ~movable_union
    body_coords = _mask_to_coords(body_mask)
    class_items: list[dict[str, Any]] = []
    if len(body_coords) > 0:
        class_items.append(
            {
                "index": -1,
                "label": "part_body",
                "path": None,
                "coords": body_coords,
                "is_body": True,
                "before_path": None,
            }
        )
    for item in part_items:
        index = int(item["index"])
        before_path = before_paths.get(index)
        class_items.append(
            {
                "index": index,
                "label": _part_label(before_path, index),
                "path": item["path"],
                "coords": item["coords"],
                "is_body": False,
                "before_path": before_path,
            }
        )
    body_info = {
        "has_body": bool(len(body_coords) > 0),
        "body_voxel_count": int(len(body_coords)),
        "whole_voxel_count": int(np.count_nonzero(whole_mask)),
        "movable_union_voxel_count": int(np.count_nonzero(movable_union & whole_mask)),
        "movable_outside_whole_voxel_count": int(np.count_nonzero(movable_union & ~whole_mask)),
    }
    return class_items, body_info


def _assign_faces(mesh: trimesh.Trimesh, class_items: list[dict[str, Any]], transform: dict[str, Any]) -> np.ndarray:
    rot = np.asarray(transform["rotation"], dtype=np.float64)
    scale = float(transform["scale"])
    points = []
    labels = []
    for label, item in enumerate(class_items):
        pts = _voxel_to_mesh_points(item["coords"], scale, rot)
        points.append(pts)
        labels.append(np.full(len(pts), label, dtype=np.int32))
    all_points = np.concatenate(points, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    tree = cKDTree(all_points)
    _, nearest = tree.query(np.asarray(mesh.triangles_center, dtype=np.float64), k=1, workers=-1)
    return all_labels[np.asarray(nearest, dtype=np.int64)].astype(np.int32)


def _face_adjacency_dihedral(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return adjacency, np.zeros(0, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    dots = np.einsum("ij,ij->i", normals[adjacency[:, 0]], normals[adjacency[:, 1]])
    return adjacency, np.arccos(np.clip(dots, -1.0, 1.0))


def _weighted_boundary_smooth(mesh: trimesh.Trimesh, labels: np.ndarray, iterations: int, sigma: float) -> np.ndarray:
    adjacency, dihedral = _face_adjacency_dihedral(mesh)
    if len(adjacency) == 0:
        return labels.copy()
    weights = np.exp(-dihedral / max(float(sigma), 1e-6))
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(len(mesh.faces))]
    for (a, b), w in zip(adjacency, weights):
        neighbors[int(a)].append((int(b), float(w)))
        neighbors[int(b)].append((int(a), float(w)))
    current = labels.astype(np.int32, copy=True)
    for _ in range(max(0, int(iterations))):
        next_labels = current.copy()
        changed = 0
        for face_idx, nbs in enumerate(neighbors):
            own = int(current[face_idx])
            if not nbs or all(int(current[n]) == own for n, _ in nbs):
                continue
            scores: dict[int, float] = {own: 0.05}
            for nb, weight in nbs:
                label = int(current[nb])
                scores[label] = scores.get(label, 0.0) + float(weight)
            best_label, best_score = max(scores.items(), key=lambda item: (item[1], -item[0]))
            if best_label != own and best_score > scores.get(own, 0.0) + 1e-9:
                next_labels[face_idx] = best_label
                changed += 1
        current = next_labels
        if changed == 0:
            break
    return current


def _boundary_stats(mesh: trimesh.Trimesh, labels: np.ndarray) -> dict[str, float]:
    adjacency, dihedral = _face_adjacency_dihedral(mesh)
    if len(adjacency) == 0:
        return {"edge_count": 0, "roughness": 0.0, "mean_dihedral_rad": 0.0, "cut_length": 0.0}
    boundary_mask = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
    if not bool(boundary_mask.any()):
        return {"edge_count": 0, "roughness": 0.0, "mean_dihedral_rad": 0.0, "cut_length": 0.0}
    boundary_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)[boundary_mask]
    edge_vec = mesh.vertices[boundary_edges[:, 0]] - mesh.vertices[boundary_edges[:, 1]]
    lengths = np.linalg.norm(edge_vec, axis=1)
    roughness = float(np.sum(lengths * (1.0 - np.clip(dihedral[boundary_mask] / math.pi, 0.0, 1.0))))
    return {
        "edge_count": int(boundary_mask.sum()),
        "roughness": roughness,
        "mean_dihedral_rad": float(np.mean(dihedral[boundary_mask])),
        "cut_length": float(np.sum(lengths)),
    }


def _extract_submesh(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> trimesh.Trimesh:
    if len(face_indices) == 0:
        raise ValueError("empty face selection")
    sub = mesh.submesh([np.asarray(face_indices, dtype=np.int64)], append=True, repair=False)
    if isinstance(sub, list):
        if not sub:
            raise ValueError("submesh extraction returned no meshes")
        sub = sub[0]
    sub = trimesh.Trimesh(vertices=np.asarray(sub.vertices), faces=np.asarray(sub.faces), process=False)
    sub.remove_unreferenced_vertices()
    sub.merge_vertices()
    trimesh.repair.fix_normals(sub)
    trimesh.repair.fix_winding(sub)
    trimesh.repair.fill_holes(sub)
    trimesh.repair.fix_normals(sub)
    return sub


def _mesh_summary(mesh: trimesh.Trimesh | None) -> dict[str, Any]:
    if mesh is None:
        return {
            "vertices": 0,
            "faces": 0,
            "is_watertight": False,
            "volume": 0.0,
            "area": 0.0,
            "bounds": None,
            "extents": None,
        }
    bounds = np.asarray(mesh.bounds, dtype=np.float64) if len(mesh.vertices) else np.zeros((2, 3), dtype=np.float64)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "volume": float(mesh.volume) if np.isfinite(mesh.volume) else 0.0,
        "area": float(mesh.area) if np.isfinite(mesh.area) else 0.0,
        "bounds": bounds.tolist(),
        "extents": (bounds[1] - bounds[0]).tolist(),
    }


def _coacd_decompose(mesh: trimesh.Trimesh, out_dir: Path, stem: str, threshold: float) -> dict[str, Any]:
    import coacd

    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError(f"{stem}: mesh too small for CoACD")
    coacd_mesh = coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64))
    pieces = coacd.run_coacd(coacd_mesh, threshold=float(threshold))
    out_paths: list[str] = []
    for idx, (vertices, faces) in enumerate(pieces):
        piece = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        if len(piece.vertices) < 4 or len(piece.faces) < 4:
            continue
        path = out_dir / f"{stem}_coacd_{idx:02d}.obj"
        piece.export(path)
        out_paths.append(str(path.resolve()))
    if not out_paths:
        raise RuntimeError(f"{stem}: CoACD returned zero usable pieces")
    return {"status": "ok", "pieces": len(out_paths), "paths": out_paths, "error": None}


def _obb_box_decompose(mesh: trimesh.Trimesh, out_dir: Path, stem: str, reason: str) -> dict[str, Any]:
    if len(mesh.vertices) == 0:
        raise ValueError(f"{stem}: empty mesh cannot build fallback box")
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if (not np.isfinite(extents).all()) or float(np.max(extents)) <= 1e-8:
        raise ValueError(f"{stem}: invalid extents for fallback box")
    box = mesh.bounding_box_oriented.to_mesh()
    trimesh.repair.fix_normals(box)
    path = out_dir / f"{stem}_coacd_degenerate_box_00.obj"
    box.export(path)
    return {
        "status": "degenerate_box",
        "pieces": 1,
        "paths": [str(path.resolve())],
        "error": reason,
    }


def _camera_for_meshes(meshes: list[trimesh.Trimesh | None]) -> tuple[np.ndarray, float]:
    bounds = [np.asarray(mesh.bounds, dtype=np.float64) for mesh in meshes if mesh is not None and len(mesh.vertices)]
    if not bounds:
        return np.zeros(3, dtype=np.float64), 1.0
    stacked = np.stack(bounds, axis=0)
    mn = stacked[:, 0, :].min(axis=0)
    mx = stacked[:, 1, :].max(axis=0)
    return (mn + mx) * 0.5, max(float(np.linalg.norm(mx - mn)) * 0.5, 1e-6)


def _project_points(vertices: np.ndarray, center: np.ndarray, radius: float, size: int) -> np.ndarray:
    rel = vertices - center[None, :]
    xy = np.column_stack([rel[:, 0] + 0.30 * rel[:, 2], -rel[:, 1] + 0.16 * rel[:, 2]])
    return xy * ((size * 0.36) / max(radius, 1e-6)) + np.array([size * 0.5, size * 0.54])


def _draw_single_mesh(mesh: trimesh.Trimesh | None, label: str, color: tuple[int, int, int], size: int, title: str) -> Image.Image:
    image = Image.new("RGB", (size, size), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size, 24), fill=(30, 30, 30))
    draw.text((7, 7), title[:34], fill=(255, 255, 255))
    draw.rectangle((7, 31, 18, 42), fill=color)
    draw.text((24, 30), label[:40], fill=(15, 15, 15))
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        draw.text((size * 0.34, size * 0.52), "missing", fill=(80, 80, 80))
        return image
    center, radius = _camera_for_meshes([mesh])
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) > 6000:
        faces = faces[np.linspace(0, len(faces) - 1, 6000).astype(np.int64)]
    points = _project_points(vertices, center, radius, size)
    face_order = np.mean(vertices[faces][:, :, 2], axis=1)
    outline = tuple(max(0, int(c * 0.45)) for c in color)
    reverse_fill = tuple(min(255, int(c * 0.82 + 35)) for c in color)
    for face_idx in np.argsort(face_order):
        poly = points[faces[face_idx]]
        if np.isfinite(poly).all():
            coords = [tuple(x) for x in poly]
            draw.polygon(coords, fill=reverse_fill, outline=outline)
            draw.line(coords + [coords[0]], fill=outline, width=1)
    for face_idx in np.argsort(face_order):
        poly = points[faces[face_idx]]
        if np.isfinite(poly).all():
            coords = [tuple(x) for x in poly]
            draw.polygon(coords, fill=color, outline=outline)
    return image


def _write_compare_png(compare_items: list[dict[str, Any]], out_png: Path) -> None:
    cell_w = 320
    cell_h = 320
    label_h = 38
    header_h = 34
    rows = max(1, len(compare_items))
    canvas = Image.new("RGB", (cell_w * 2, header_h + rows * (cell_h + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, cell_w * 2, header_h), fill=(20, 20, 20))
    draw.text((10, 10), "before: independent decode", fill=(255, 255, 255))
    draw.text((cell_w + 10, 10), "after: cut from overall + body", fill=(255, 255, 255))
    for row, item in enumerate(compare_items):
        y0 = header_h + row * (cell_h + label_h)
        color = item["color"]
        draw.rectangle((0, y0, cell_w * 2, y0 + label_h), fill=(238, 238, 238))
        draw.rectangle((8, y0 + 10, 24, y0 + 26), fill=color)
        draw.text((32, y0 + 10), item["label"][:82], fill=(20, 20, 20))
        before = _draw_single_mesh(item.get("before_mesh"), item["label"], color, cell_w, "before")
        after = _draw_single_mesh(item.get("after_mesh"), item["label"], color, cell_w, "after")
        canvas.paste(before, (0, y0 + label_h))
        canvas.paste(after, (cell_w, y0 + label_h))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        child = None
        for child in elem:
            _indent_xml(child, level + 1)
        if child is not None and (not child.tail or not child.tail.strip()):
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def _write_mjcf(out_xml: Path, visual_items: list[dict[str, Any]], density: float, freejoint: bool) -> None:
    root = ET.Element("mujoco", {"model": "route1_partition"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81"})
    asset = ET.SubElement(root, "asset")
    for item in visual_items:
        ET.SubElement(asset, "mesh", {"name": item["mesh_name"], "file": item["mesh_file"]})
        for piece in item.get("coacd", {}).get("paths", []):
            path = Path(piece)
            ET.SubElement(asset, "mesh", {"name": path.stem, "file": f"collision/{path.name}"})
    worldbody = ET.SubElement(root, "worldbody")
    base = ET.SubElement(worldbody, "body", {"name": "object", "pos": "0 0 0"})
    if freejoint:
        ET.SubElement(base, "freejoint")
    colors = ["0.82 0.24 0.26 1", "0.20 0.52 0.77 1", "0.25 0.62 0.32 1", "0.58 0.39 0.70 1", "0.90 0.55 0.17 1", "0.45 0.36 0.28 1"]
    for idx, item in enumerate(visual_items):
        body = ET.SubElement(base, "body", {"name": item["label"], "pos": "0 0 0"})
        mass = max(abs(float(item.get("volume", 0.0))) * float(density), 1e-5)
        ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": f"{mass:.6g}", "diaginertia": "1e-4 1e-4 1e-4"})
        ET.SubElement(body, "geom", {"name": f"{item['label']}_visual", "type": "mesh", "mesh": item["mesh_name"], "group": "2", "contype": "0", "conaffinity": "0", "rgba": colors[idx % len(colors)]})
        for piece in item.get("coacd", {}).get("paths", []):
            path = Path(piece)
            ET.SubElement(body, "geom", {"name": path.stem, "type": "mesh", "mesh": path.stem, "group": "3", "contype": "1", "conaffinity": "1", "rgba": "0.2 0.2 0.2 0.35"})
    _indent_xml(root)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def _mesh_vertex_mask(mesh: trimesh.Trimesh, transform: dict[str, Any]) -> np.ndarray:
    mask = np.zeros((64, 64, 64), dtype=bool)
    if len(mesh.vertices) == 0:
        return mask
    rot = np.asarray(transform["rotation"], dtype=np.float64)
    scale = float(transform["scale"])
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(mesh.faces):
        triangles = vertices[np.asarray(mesh.faces, dtype=np.int64)]
        centers = triangles.mean(axis=1)
        vertices = np.concatenate([vertices, centers], axis=0)
    idx = _mesh_to_voxel_indices(vertices, scale, rot)
    valid = np.all((idx >= 0) & (idx < 64), axis=1)
    if bool(valid.any()):
        v = idx[valid]
        mask[v[:, 0], v[:, 1], v[:, 2]] = True
    return mask


def _label_boundary_masks(class_items: list[dict[str, Any]]) -> list[np.ndarray]:
    source_masks = [_coords_to_mask(item["coords"]) for item in class_items]
    boundary_masks: list[np.ndarray] = []
    for idx, mask in enumerate(source_masks):
        others = np.zeros_like(mask)
        for other_idx, other in enumerate(source_masks):
            if other_idx != idx:
                others |= other
        boundary_masks.append(binary_dilation(mask, iterations=1) & binary_dilation(others, iterations=1))
    return boundary_masks


def _pairwise_overlap_voxels(meshes: list[trimesh.Trimesh], labels: list[str], transform: dict[str, Any], class_items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(meshes) < 2:
        return {"max": 0, "raw_max": 0, "pairs": [], "raw_pairs": [], "boundary_excluded": True}
    masks = [_mesh_vertex_mask(mesh, transform) for mesh in meshes]
    boundary_masks = _label_boundary_masks(class_items)
    pairs: list[dict[str, Any]] = []
    raw_pairs: list[dict[str, Any]] = []
    max_overlap = 0
    raw_max = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            raw_mask = masks[i] & masks[j]
            raw_overlap = int(np.count_nonzero(raw_mask))
            boundary = boundary_masks[i] | boundary_masks[j]
            overlap = int(np.count_nonzero(raw_mask & ~boundary))
            if raw_overlap:
                raw_pairs.append({"a": labels[i], "b": labels[j], "overlap_voxels": raw_overlap})
            if overlap:
                pairs.append({"a": labels[i], "b": labels[j], "overlap_voxels": overlap})
            raw_max = max(raw_max, raw_overlap)
            max_overlap = max(max_overlap, overlap)
    pairs.sort(key=lambda item: item["overlap_voxels"], reverse=True)
    raw_pairs.sort(key=lambda item: item["overlap_voxels"], reverse=True)
    return {"max": int(max_overlap), "raw_max": int(raw_max), "pairs": pairs, "raw_pairs": raw_pairs, "boundary_excluded": True}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_dir = Path(args.out_dir).resolve()
    visual_dir = out_dir / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "status": "running",
        "overall_mesh": str(Path(args.overall_mesh).resolve()),
        "whole_voxel": str(Path(args.whole_voxel).resolve()),
        "part_voxels_glob": args.part_voxels_glob,
        "part_meshes_glob": args.part_meshes_glob,
        "out_dir": str(out_dir),
        "coacd_import": "SKIPPED",
        "coacd_skipped": True,
        "mjcf_skipped": True,
        "mujoco_py": "UNKNOWN",
        "anomalies": [],
    }

    try:
        import mujoco_py  # noqa: F401

        report["mujoco_py"] = "有"
    except Exception:
        report["mujoco_py"] = "无"

    overall_mesh = trimesh.load_mesh(args.overall_mesh, process=False)
    if isinstance(overall_mesh, trimesh.Scene):
        overall_mesh = trimesh.util.concatenate([geom for geom in overall_mesh.geometry.values()])
    if len(overall_mesh.vertices) == 0 or len(overall_mesh.faces) == 0:
        raise ValueError(f"empty overall mesh: {args.overall_mesh}")
    whole_coords = _load_npz_coords(Path(args.whole_voxel))
    part_items = _load_part_voxels(args.part_voxels_glob)
    before_paths = _load_before_meshes(args.part_meshes_glob)
    class_items, body_info = _make_class_items(whole_coords, part_items, before_paths)

    transform = _calibrate_transform(overall_mesh, whole_coords)
    report["coordinate_check"] = transform
    print(f"[partition] coord transform={transform['name']} scale={transform['scale']:.6f} IoU={transform['iou']:.6f}", flush=True)
    print(
        f"[partition] body has_body={body_info['has_body']} body_voxels={body_info['body_voxel_count']} "
        f"whole_voxels={body_info['whole_voxel_count']}",
        flush=True,
    )

    labels_before = _assign_faces(overall_mesh, class_items, transform)
    boundary_before = _boundary_stats(overall_mesh, labels_before)
    labels_after = _weighted_boundary_smooth(overall_mesh, labels_before, int(args.smooth_iterations), float(args.smooth_sigma))
    boundary_after = _boundary_stats(overall_mesh, labels_after)

    after_meshes: list[trimesh.Trimesh] = []
    after_labels: list[str] = []
    compare_items: list[dict[str, Any]] = [
        {
            "label": "overall",
            "before_mesh": overall_mesh,
            "after_mesh": overall_mesh,
            "color": (92, 92, 92),
        }
    ]
    parts: list[dict[str, Any]] = []

    for label_idx, item in enumerate(class_items):
        index = int(item["index"])
        before_path = item.get("before_path")
        label = str(item["label"])
        face_indices = np.flatnonzero(labels_after == label_idx)
        if len(face_indices) == 0:
            raise RuntimeError(f"{label}: no faces assigned after smoothing")
        submesh = _extract_submesh(overall_mesh, face_indices)
        visual_path = visual_dir / f"{_safe_name(label)}.obj"
        submesh.export(visual_path)
        before_mesh = None
        if before_path is not None and Path(before_path).is_file():
            before_loaded = trimesh.load_mesh(before_path, process=False)
            if isinstance(before_loaded, trimesh.Scene):
                before_loaded = trimesh.util.concatenate([geom for geom in before_loaded.geometry.values()])
            before_mesh = before_loaded
        after_meshes.append(submesh)
        after_labels.append(label)
        compare_items.append(
            {
                "label": label,
                "before_mesh": before_mesh,
                "after_mesh": submesh,
                "color": COLORS[label_idx % len(COLORS)],
            }
        )

        part_report = {
            "index": index,
            "label": label,
            "is_body": bool(item.get("is_body", False)),
            "part_voxel_path": item["path"],
            "before_mesh_path": None if before_path is None else str(before_path.resolve()),
            "after_mesh_path": str(visual_path.resolve()),
            "voxel_count": int(len(item["coords"])),
            "assigned_faces": int(len(face_indices)),
            "watertight_before": bool(before_mesh.is_watertight) if before_mesh is not None else None,
            "watertight_after": bool(submesh.is_watertight),
            "mesh_before": _mesh_summary(before_mesh),
            "mesh_after": _mesh_summary(submesh),
        }
        parts.append(part_report)

    compare_png = out_dir / "compare.png"
    _write_compare_png(compare_items, compare_png)
    overlap = _pairwise_overlap_voxels(after_meshes, after_labels, transform, class_items)
    report.update(
        {
            "status": "done",
            "compare_png": str(compare_png.resolve()),
            "n_parts": int(len(parts)),
            "has_body": bool(body_info["has_body"]),
            "body": body_info,
            "unassigned_faces": 0,
            "face_count": int(len(overall_mesh.faces)),
            "face_label_changes_after_smoothing": int(np.count_nonzero(labels_after != labels_before)),
            "wt_before_percent": float(100.0 * sum(1 for p in parts if p["watertight_before"]) / len(parts)) if parts else 0.0,
            "wt_after_percent": float(100.0 * sum(1 for p in parts if p["watertight_after"]) / len(parts)) if parts else 0.0,
            "overlap_voxels": overlap,
            "compare_viz": {"colored_by_part": True, "explode_layout": True, "double_sided": True},
            "boundary": {"before": boundary_before, "after": boundary_after},
            "parts": parts,
            "seconds": round(time.time() - started, 3),
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut an overall decoded mesh into parts using 64^3 part seg voxels.")
    parser.add_argument("--overall-mesh", required=True, type=Path)
    parser.add_argument("--whole-voxel", required=True, type=Path)
    parser.add_argument("--part-voxels-glob", required=True)
    parser.add_argument("--part-meshes-glob", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--coacd-threshold", type=float, default=0.05)
    parser.add_argument("--density", type=float, default=300.0)
    parser.add_argument("--smooth-iterations", type=int, default=8)
    parser.add_argument("--smooth-sigma", type=float, default=0.55)
    parser.add_argument("--freejoint", action="store_true")
    parser.add_argument("--allow-missing-coacd", action="store_true")
    parser.add_argument("--allow-coacd-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    try:
        report = _run(args)
        _write_json(report_path, report)
        print(f"[partition] report -> {report_path}", flush=True)
        return 0
    except Exception as exc:
        failure_report = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "overall_mesh": str(Path(args.overall_mesh).resolve()),
            "whole_voxel": str(Path(args.whole_voxel).resolve()),
            "part_voxels_glob": args.part_voxels_glob,
            "part_meshes_glob": args.part_meshes_glob,
            "out_dir": str(out_dir),
        }
        _write_json(report_path, failure_report)
        print(f"[partition] FAILED: {failure_report['error']}", file=sys.stderr, flush=True)
        print(f"[partition] report -> {report_path}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
