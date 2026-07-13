#!/usr/bin/env python3
"""Quantitative and visual diagnostics for a saved joint voxel partition.

The evaluator intentionally has no Torch or TRELLIS dependency.  It can run in
the lightweight ee-eval post-processing environment with only NumPy and PIL.
Classification metrics are evaluated on the complete 64^3 grid.  Voxels that
are claimed by more than one GT part are ignored instead of being assigned by
part order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


GRID_SIZE = 64
GT_IGNORE = -1
OUTSIDE = -2

BODY_COLOR = (142, 148, 154)
PART_COLORS = (
    (34, 113, 179),
    (225, 119, 38),
    (42, 157, 79),
    (184, 61, 104),
    (118, 82, 167),
    (23, 153, 164),
    (196, 157, 41),
    (104, 92, 82),
)
ERROR_COLOR = (218, 44, 44)
OVERLAP_COLOR = (35, 35, 35)
INTERFACE_COLOR = (0, 145, 190)
LOW_MARGIN_COLOR = (247, 190, 25)
IMPROVED_COLOR = (15, 156, 80)
REGRESSED_COLOR = (207, 44, 131)
NEUTRAL_COLOR = (235, 116, 30)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _read_refinement_json(value: Any) -> dict[str, Any]:
    value = _json_scalar(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None or str(value).strip() == "":
        return {}
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("refinement_json must encode an object")
    return parsed


def _rooted(data_root: str | Path, rel_or_abs: str | Path) -> Path:
    path = Path(str(rel_or_abs))
    return path if path.is_absolute() else Path(data_root) / path


def _coords_from_array(value: np.ndarray, *, source: str, unique: bool = True) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (GRID_SIZE, GRID_SIZE, GRID_SIZE):
        coords = np.argwhere(array.astype(bool, copy=False))
    else:
        try:
            coords = np.asarray(array, dtype=np.int64).reshape(-1, 3)
        except ValueError as exc:
            raise ValueError(f"{source}: expected [N,3] coords or a 64^3 mask, got {array.shape}") from exc
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    valid = np.all((coords >= 0) & (coords < GRID_SIZE), axis=1)
    if not bool(valid.all()):
        bad = coords[~valid][:5].tolist()
        raise ValueError(f"{source}: coordinates outside [0,{GRID_SIZE}): {bad}")
    if unique:
        coords = np.unique(coords, axis=0)
    return coords.astype(np.int32, copy=False)


def _load_coords(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            for key in ("coords", "indices", "arr_0"):
                if key in loaded.files:
                    return _coords_from_array(loaded[key], source=f"{path}:{key}")
            raise KeyError(f"{path}: no coords/indices/arr_0 key; available={loaded.files}")
        finally:
            loaded.close()
    return _coords_from_array(loaded, source=str(path))


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    mask = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords.size:
        mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return mask


def _sample_parts(ds_sample: dict[str, Any]) -> list[dict[str, Any]]:
    parts = ds_sample.get("parts")
    if isinstance(parts, list) and parts:
        return [dict(part) for part in parts]

    target_names = [str(name) for name in ds_sample.get("target_part_names", [])]
    target_parts = [dict(part) for part in (ds_sample.get("target_parts") or [])]
    by_name = {str(part.get("name", part.get("part_name", ""))): part for part in target_parts}
    return [
        {
            "part_name": name,
            "target_part": by_name.get(name, {}),
        }
        for name in target_names
    ]


def _part_name(part: dict[str, Any]) -> str:
    return str(part.get("part_name") or part.get("name") or (part.get("target_part") or {}).get("name") or "")


def _part_voxel_path(data_root: Path, ds_sample: dict[str, Any], part: dict[str, Any]) -> Path:
    direct = part.get("raw_ind_rel")
    target_part = part.get("target_part") or {}
    paths = part.get("paths") or target_part.get("paths") or {}
    candidate = direct or paths.get("part_voxel")
    if candidate:
        return _rooted(data_root, candidate)
    name = _part_name(part)
    return (
        data_root
        / "reconstruction"
        / "voxel_expanded"
        / str(ds_sample.get("obj_id", ds_sample.get("object_id")))
        / f"angle_{int(ds_sample.get('angle_idx', ds_sample.get('angle', 0)))}"
        / "64"
        / f"ind_{name}.npy"
    )


def _surface_path(data_root: Path, ds_sample: dict[str, Any]) -> Path:
    paths = ds_sample.get("manifest_paths") or ds_sample.get("paths") or {}
    candidate = ds_sample.get("surface_rel") or paths.get("overall_surface")
    if candidate:
        return _rooted(data_root, candidate)
    return (
        data_root
        / "reconstruction"
        / "voxel_expanded"
        / str(ds_sample.get("obj_id", ds_sample.get("object_id")))
        / f"angle_{int(ds_sample.get('angle_idx', ds_sample.get('angle', 0)))}"
        / "64"
        / "surface.npy"
    )


def _load_partition(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"coords", "logits", "labels_raw", "class_names"}
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path}: missing keys {missing}")
        coords_raw = np.asarray(data["coords"], dtype=np.int64).reshape(-1, 3)
        if coords_raw.shape[0] != np.unique(coords_raw, axis=0).shape[0]:
            raise ValueError(f"{path}: duplicate candidate coordinates")
        coords = _coords_from_array(coords_raw, source=f"{path}:coords", unique=False)
        logits = np.asarray(data["logits"], dtype=np.float32)
        labels_raw = np.asarray(data["labels_raw"], dtype=np.int64).reshape(-1)
        labels_refined = np.asarray(data.get("labels_refined", labels_raw), dtype=np.int64).reshape(-1)
        class_names = [str(value) for value in np.asarray(data["class_names"]).reshape(-1).tolist()]
        refinement = _read_refinement_json(data.get("refinement_json"))

    if logits.ndim != 2 or logits.shape[0] != coords.shape[0]:
        raise ValueError(f"{path}: logits {logits.shape} do not match coords {coords.shape}")
    if logits.shape[1] != len(class_names):
        raise ValueError(f"{path}: logits has {logits.shape[1]} classes, class_names has {len(class_names)}")
    if labels_raw.shape[0] != coords.shape[0] or labels_refined.shape[0] != coords.shape[0]:
        raise ValueError(f"{path}: label lengths do not match coords")
    for name, labels in (("labels_raw", labels_raw), ("labels_refined", labels_refined)):
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= len(class_names)):
            raise ValueError(f"{path}: {name} contains an invalid class index")
    if not np.isfinite(logits).all():
        raise ValueError(f"{path}: logits contain NaN or infinity")
    return {
        "coords": coords,
        "logits": logits,
        "labels_raw": labels_raw,
        "labels_refined": labels_refined,
        "class_names": class_names,
        "refinement": refinement,
    }


def _load_ground_truth(
    data_root: Path,
    ds_sample: dict[str, Any],
    class_names: Sequence[str],
) -> dict[str, Any]:
    if len(class_names) < 1:
        raise ValueError("joint partition must contain the body class")
    sample_parts = _sample_parts(ds_sample)
    by_name = {_part_name(part): part for part in sample_parts}
    ordered_parts: list[dict[str, Any]] = []
    for index, class_name in enumerate(class_names[1:]):
        part = by_name.get(str(class_name))
        if part is None and len(sample_parts) == len(class_names) - 1:
            part = sample_parts[index]
        if part is None:
            raise KeyError(f"GT part for joint class {class_name!r} was not found in ds_sample")
        ordered_parts.append(part)

    surface_path = _surface_path(data_root, ds_sample)
    whole_coords = _load_coords(surface_path)
    whole = _coords_to_mask(whole_coords)
    claims = np.zeros((len(ordered_parts), GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
    part_paths: list[str] = []
    outside_counts: list[int] = []
    for index, part in enumerate(ordered_parts):
        part_path = _part_voxel_path(data_root, ds_sample, part)
        part_paths.append(str(part_path))
        part_mask = _coords_to_mask(_load_coords(part_path))
        outside_counts.append(int((part_mask & ~whole).sum()))
        claims[index] = part_mask & whole

    claim_count = claims.sum(axis=0, dtype=np.uint16) if len(ordered_parts) else np.zeros_like(whole, dtype=np.uint16)
    overlap = whole & (claim_count > 1)
    labels = np.full(whole.shape, OUTSIDE, dtype=np.int16)
    labels[whole] = 0
    for part_index in range(len(ordered_parts)):
        labels[whole & (claim_count == 1) & claims[part_index]] = int(part_index + 1)
    labels[overlap] = GT_IGNORE
    return {
        "labels": labels,
        "whole": whole,
        "claims": claims,
        "overlap": overlap,
        "claim_count": claim_count,
        "surface_path": str(surface_path),
        "part_paths": part_paths,
        "part_voxels_outside_whole": outside_counts,
    }


def _prediction_grid(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), OUTSIDE, dtype=np.int16)
    if coords.size:
        grid[coords[:, 0], coords[:, 1], coords[:, 2]] = np.asarray(labels, dtype=np.int16)
    return grid


def _iou_counts(pred: np.ndarray, gt: np.ndarray, class_index: int, ignore: np.ndarray) -> dict[str, Any]:
    gt_mask = gt == int(class_index)
    pred_mask = (pred == int(class_index)) & ~ignore
    intersection = int((gt_mask & pred_mask).sum())
    union = int((gt_mask | pred_mask).sum())
    gt_count = int(gt_mask.sum())
    pred_count = int(pred_mask.sum())
    return {
        "iou": None if union == 0 else float(intersection / union),
        "intersection": intersection,
        "union": union,
        "gt_voxels": gt_count,
        "pred_voxels": pred_count,
        "recall": None if gt_count == 0 else float(intersection / gt_count),
        "precision": None if pred_count == 0 else float(intersection / pred_count),
    }


def _cross_label_pairs(gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_ids = np.arange(gt.size, dtype=np.int64).reshape(gt.shape)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    boundary = np.zeros(gt.shape, dtype=bool)
    for axis in range(3):
        a_slice = [slice(None)] * 3
        b_slice = [slice(None)] * 3
        a_slice[axis] = slice(0, -1)
        b_slice[axis] = slice(1, None)
        a_slice_t = tuple(a_slice)
        b_slice_t = tuple(b_slice)
        a_gt = gt[a_slice_t]
        b_gt = gt[b_slice_t]
        cross = (a_gt >= 0) & (b_gt >= 0) & (a_gt != b_gt)
        if not bool(cross.any()):
            continue
        left.append(flat_ids[a_slice_t][cross])
        right.append(flat_ids[b_slice_t][cross])
        a_boundary = boundary[a_slice_t]
        b_boundary = boundary[b_slice_t]
        a_boundary[cross] = True
        b_boundary[cross] = True
    if not left:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, boundary
    return np.concatenate(left), np.concatenate(right), boundary


def _connected_component_stats(mask: np.ndarray) -> dict[str, Any]:
    coords = np.argwhere(mask)
    total = int(coords.shape[0])
    if total == 0:
        return {
            "count": 0,
            "largest_voxels": 0,
            "largest_fraction": 0.0,
            "singletons": 0,
            "tiny_components_le_8": 0,
            "tiny_voxels_le_8": 0,
        }
    remaining = {tuple(int(value) for value in row) for row in coords.tolist()}
    sizes: list[int] = []
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 0
        while stack:
            x, y, z = stack.pop()
            size += 1
            for dx, dy, dz in offsets:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    sizes.sort(reverse=True)
    return {
        "count": int(len(sizes)),
        "largest_voxels": int(sizes[0]),
        "largest_fraction": float(sizes[0] / total),
        "singletons": int(sum(size == 1 for size in sizes)),
        "tiny_components_le_8": int(sum(size <= 8 for size in sizes)),
        "tiny_voxels_le_8": int(sum(size for size in sizes if size <= 8)),
    }


def _method_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    candidate: np.ndarray,
    class_names: Sequence[str],
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    boundary: np.ndarray,
) -> dict[str, Any]:
    ignore = gt == GT_IGNORE
    per_class = {
        str(name): _iou_counts(pred, gt, index, ignore)
        for index, name in enumerate(class_names)
    }
    ious = [row["iou"] for row in per_class.values() if row["iou"] is not None]
    part_ious = [per_class[str(name)]["iou"] for name in class_names[1:] if per_class[str(name)]["iou"] is not None]

    boundary_count = int(boundary.sum())
    boundary_correct = int((pred[boundary] == gt[boundary]).sum()) if boundary_count else 0
    boundary_errors = boundary_count - boundary_correct
    boundary_covered = boundary & candidate
    boundary_covered_count = int(boundary_covered.sum())
    boundary_covered_correct = (
        int((pred[boundary_covered] == gt[boundary_covered]).sum()) if boundary_covered_count else 0
    )
    pred_flat = pred.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    covered = candidate_flat[pair_a] & candidate_flat[pair_b] if pair_a.size else np.empty((0,), dtype=bool)
    covered_count = int(covered.sum())
    same_covered = int((pred_flat[pair_a[covered]] == pred_flat[pair_b[covered]]).sum()) if covered_count else 0
    pair_correct = (
        int(
            (
                (pred_flat[pair_a[covered]] == gt.reshape(-1)[pair_a[covered]])
                & (pred_flat[pair_b[covered]] == gt.reshape(-1)[pair_b[covered]])
            ).sum()
        )
        if covered_count
        else 0
    )
    component_rows = {
        str(name): _connected_component_stats(pred == index)
        for index, name in enumerate(class_names)
    }
    component_counts = [int(row["count"]) for row in component_rows.values()]
    part_component_counts = [int(component_rows[str(name)]["count"]) for name in class_names[1:]]
    predicted_pair_a, _predicted_pair_b, _predicted_boundary = _cross_label_pairs(pred)
    predicted_interface_pairs = int(predicted_pair_a.shape[0])
    gt_interface_pairs = int(pair_a.shape[0])
    return {
        "mean_iou": float(np.mean(ious)) if ious else None,
        "part_mean_iou": float(np.mean(part_ious)) if part_ious else None,
        "per_class": per_class,
        "boundary_voxels": boundary_count,
        "boundary_correct": boundary_correct,
        "boundary_error": None if boundary_count == 0 else float(boundary_errors / boundary_count),
        "boundary_candidate_voxels": boundary_covered_count,
        "boundary_candidate_coverage": (
            None if boundary_count == 0 else float(boundary_covered_count / boundary_count)
        ),
        "boundary_candidate_correct": boundary_covered_correct,
        "boundary_error_covered": (
            None
            if boundary_covered_count == 0
            else float((boundary_covered_count - boundary_covered_correct) / boundary_covered_count)
        ),
        "interface_pairs": int(pair_a.shape[0]),
        "predicted_interface_pairs": predicted_interface_pairs,
        "predicted_to_gt_interface_ratio": (
            None if gt_interface_pairs == 0 else float(predicted_interface_pairs / gt_interface_pairs)
        ),
        "interface_pairs_covered": covered_count,
        "interface_pair_coverage": None if pair_a.size == 0 else float(covered_count / pair_a.size),
        "cross_label_same_pred": same_covered,
        "cross_label_same_pred_rate": None if covered_count == 0 else float(same_covered / covered_count),
        "cross_label_pair_correct": pair_correct,
        "cross_label_pair_correct_rate": None if covered_count == 0 else float(pair_correct / covered_count),
        "components": {
            "per_class": component_rows,
            "total": int(sum(component_counts)),
            "mean_per_class": float(np.mean(component_counts)) if component_counts else 0.0,
            "part_total": int(sum(part_component_counts)),
            "part_mean": float(np.mean(part_component_counts)) if part_component_counts else 0.0,
            "tiny_components_le_8": int(
                sum(int(row["tiny_components_le_8"]) for row in component_rows.values())
            ),
            "tiny_voxels_le_8": int(sum(int(row["tiny_voxels_le_8"]) for row in component_rows.values())),
            "part_tiny_components_le_8": int(
                sum(int(component_rows[str(name)]["tiny_components_le_8"]) for name in class_names[1:])
            ),
            "part_tiny_voxels_le_8": int(
                sum(int(component_rows[str(name)]["tiny_voxels_le_8"]) for name in class_names[1:])
            ),
        },
    }


def _binary_iou(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    intersection = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return {
        "iou": 1.0 if union == 0 else float(intersection / union),
        "intersection": intersection,
        "union": union,
        "pred_voxels": int(pred.sum()),
        "gt_voxels": int(gt.sum()),
    }


def _raw_logit_gap(logits: np.ndarray) -> np.ndarray:
    if logits.shape[1] <= 1:
        return np.full((logits.shape[0],), np.inf, dtype=np.float32)
    top2 = np.partition(logits, kth=logits.shape[1] - 2, axis=1)[:, -2:]
    return (top2.max(axis=1) - top2.min(axis=1)).astype(np.float32, copy=False)


def _probability_margin(logits: np.ndarray) -> np.ndarray:
    if logits.shape[1] <= 1:
        return np.ones((logits.shape[0],), dtype=np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    top2 = np.partition(probability, kth=probability.shape[1] - 2, axis=1)[:, -2:]
    return (top2.max(axis=1) - top2.min(axis=1)).astype(np.float32, copy=False)


def _low_margin_mask(logits: np.ndarray, refinement: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    raw_gap = _raw_logit_gap(logits)
    probability_margin = _probability_margin(logits)
    quantile = float(refinement.get("margin_quantile", 0.0) or 0.0)
    raw_threshold_value = refinement.get("raw_margin_quantile_threshold")
    if raw_threshold_value is None and quantile > 0.0 and raw_gap.size:
        raw_threshold_value = float(np.quantile(raw_gap.astype(np.float64), quantile))
        raw_threshold_source = "derived_from_raw_logits"
    elif raw_threshold_value is None:
        raw_threshold_source = "disabled"
    else:
        raw_threshold_value = float(raw_threshold_value)
        raw_threshold_source = "refinement_json"
    probability_threshold = float(refinement.get("margin_threshold", 0.0) or 0.0)
    low = np.zeros((logits.shape[0],), dtype=bool)
    if raw_threshold_value is not None:
        low |= raw_gap <= float(raw_threshold_value)
    if probability_threshold > 0.0:
        low |= probability_margin < probability_threshold
    finite_gap = raw_gap[np.isfinite(raw_gap)]
    return low, {
        "raw_logit_gap_threshold": raw_threshold_value,
        "raw_logit_gap_threshold_source": raw_threshold_source,
        "probability_margin_threshold": probability_threshold,
        "margin_quantile": quantile,
        "raw_logit_gap_min": None if finite_gap.size == 0 else float(finite_gap.min()),
        "raw_logit_gap_median": None if finite_gap.size == 0 else float(np.median(finite_gap)),
        "raw_logit_gap_max": None if finite_gap.size == 0 else float(finite_gap.max()),
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    interior = mask.copy()
    interior &= padded[:-2, 1:-1, 1:-1]
    interior &= padded[2:, 1:-1, 1:-1]
    interior &= padded[1:-1, :-2, 1:-1]
    interior &= padded[1:-1, 2:, 1:-1]
    interior &= padded[1:-1, 1:-1, :-2]
    interior &= padded[1:-1, 1:-1, 2:]
    return mask & ~interior


def _project(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return (x - y) * 0.86, (x + y) * 0.43 - z * 0.92, x + y + z


def _projection_frame(coords: np.ndarray, width: int, height: int) -> dict[str, float]:
    if coords.size == 0:
        return {"u_mid": 0.0, "v_mid": 0.0, "scale": 1.0, "cx": width / 2, "cy": height / 2 + 20}
    u, v, _ = _project(coords)
    u_span = max(float(u.max() - u.min()), 1.0)
    v_span = max(float(v.max() - v.min()), 1.0)
    scale = min((width - 36) / u_span, (height - 118) / v_span)
    return {
        "u_mid": float((u.max() + u.min()) * 0.5),
        "v_mid": float((v.max() + v.min()) * 0.5),
        "scale": float(scale),
        "cx": width / 2,
        "cy": 58 + (height - 98) / 2,
    }


def _project_pixels(coords: np.ndarray, frame: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v, depth = _project(coords)
    px = (u - frame["u_mid"]) * frame["scale"] + frame["cx"]
    py = (v - frame["v_mid"]) * frame["scale"] + frame["cy"]
    return px, py, depth


def _draw_layers(
    draw: ImageDraw.ImageDraw,
    layers: Iterable[tuple[np.ndarray, tuple[int, int, int], int]],
    frame: dict[str, float],
    *,
    ellipse: bool = False,
) -> None:
    points: list[tuple[float, float, float, tuple[int, int, int], int]] = []
    for coords, color, size in layers:
        coords = np.asarray(coords, dtype=np.int32).reshape(-1, 3)
        if coords.size == 0:
            continue
        px, py, depth = _project_pixels(coords, frame)
        points.extend(
            (float(d), float(x), float(y), color, int(size))
            for x, y, d in zip(px.tolist(), py.tolist(), depth.tolist())
        )
    points.sort(key=lambda row: row[0])
    for _depth, x, y, color, size in points:
        half = max(1, int(size) // 2)
        box = (x - half, y - half, x + half, y + half)
        if ellipse:
            draw.ellipse(box, fill=color)
        else:
            draw.rectangle(box, fill=color)


def _render_projection_panel(
    *,
    title: str,
    subtitle: Sequence[str],
    base_layers: Iterable[tuple[np.ndarray, tuple[int, int, int], int]],
    marker_layers: Iterable[tuple[np.ndarray, tuple[int, int, int], int]],
    frame_coords: np.ndarray,
    width: int = 430,
    height: int = 470,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 54), fill=(30, 33, 36))
    draw.text((10, 8), title, fill=(255, 255, 255))
    for line_index, line in enumerate(subtitle[:2]):
        draw.text((10, 25 + line_index * 13), str(line)[:68], fill=(215, 220, 224))
    draw.rectangle((0, height - 22, width, height), fill=(238, 238, 235))
    frame = _projection_frame(np.asarray(frame_coords), width, height)
    _draw_layers(draw, base_layers, frame, ellipse=False)
    # Marker layers are deliberately last.  They include internal interface,
    # low-margin, and changed voxels that the exterior shell would hide.
    _draw_layers(draw, marker_layers, frame, ellipse=True)
    return image


def _class_color(class_index: int) -> tuple[int, int, int]:
    return BODY_COLOR if class_index == 0 else PART_COLORS[(class_index - 1) % len(PART_COLORS)]


def _label_layers(labels: np.ndarray, class_count: int, *, voxel_size: int = 5) -> list[tuple[np.ndarray, tuple[int, int, int], int]]:
    layers = []
    for class_index in range(class_count):
        class_surface = _surface(labels == class_index)
        layers.append((np.argwhere(class_surface), _class_color(class_index), voxel_size))
    return layers


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _render_diagnostic(
    output_png: Path,
    *,
    title: str,
    gt: dict[str, Any],
    raw_grid: np.ndarray,
    refined_grid: np.ndarray,
    candidate: np.ndarray,
    pred_whole: np.ndarray,
    raw_metrics: dict[str, Any],
    refined_metrics: dict[str, Any],
    boundary: np.ndarray,
    low_margin_grid: np.ndarray,
    improved_grid: np.ndarray,
    regressed_grid: np.ndarray,
    neutral_grid: np.ndarray,
    class_names: Sequence[str],
    whole_iou: float,
    interface_coverage: Any,
) -> None:
    frame_mask = gt["whole"] | candidate | pred_whole
    frame_coords = np.argwhere(frame_mask)
    gt_error_overlap = np.argwhere(gt["overlap"])
    # Keep the class panels focused on the joint interface. Whole-shape misses
    # are already quantified by candidate recall and whole IoU.
    raw_error = np.argwhere(boundary & (raw_grid != gt["labels"]))
    refined_error = np.argwhere(boundary & (refined_grid != gt["labels"]))

    gt_panel = _render_projection_panel(
        title="GT (multi-claim ignored)",
        subtitle=(f"whole={int(gt['whole'].sum())} overlap={int(gt['overlap'].sum())}", title),
        base_layers=_label_layers(gt["labels"], len(class_names)),
        marker_layers=[(gt_error_overlap, OVERLAP_COLOR, 5)],
        frame_coords=frame_coords,
    )
    raw_panel = _render_projection_panel(
        title="Raw joint argmax (red=GT interface error)",
        subtitle=(
            f"mIoU={_fmt(raw_metrics['mean_iou'])} part={_fmt(raw_metrics['part_mean_iou'])}",
            f"berr={_fmt(raw_metrics['boundary_error'])}/{_fmt(raw_metrics['boundary_error_covered'])} iface={_fmt(raw_metrics['predicted_to_gt_interface_ratio'])} tiny={raw_metrics['components']['part_tiny_components_le_8']}",
        ),
        base_layers=_label_layers(raw_grid, len(class_names)),
        marker_layers=[(raw_error, ERROR_COLOR, 4)],
        frame_coords=frame_coords,
    )
    refined_panel = _render_projection_panel(
        title="Refined joint labels (red=GT interface error)",
        subtitle=(
            f"mIoU={_fmt(refined_metrics['mean_iou'])} part={_fmt(refined_metrics['part_mean_iou'])}",
            f"berr={_fmt(refined_metrics['boundary_error'])}/{_fmt(refined_metrics['boundary_error_covered'])} iface={_fmt(refined_metrics['predicted_to_gt_interface_ratio'])} tiny={refined_metrics['components']['part_tiny_components_le_8']}",
        ),
        base_layers=_label_layers(refined_grid, len(class_names)),
        marker_layers=[(refined_error, ERROR_COLOR, 4)],
        frame_coords=frame_coords,
    )
    xray_base = np.argwhere(_surface(gt["whole"] | pred_whole))
    xray_panel = _render_projection_panel(
        title="Boundary x-ray (markers on top)",
        subtitle=(
            f"whole IoU={whole_iou:.4f} interface coverage={_fmt(interface_coverage)}",
            "cyan=interface yellow=low green=better pink=worse orange=neutral",
        ),
        base_layers=[(xray_base, (205, 210, 213), 3)],
        marker_layers=[
            (np.argwhere(boundary), INTERFACE_COLOR, 3),
            (np.argwhere(low_margin_grid), LOW_MARGIN_COLOR, 4),
            (np.argwhere(neutral_grid), NEUTRAL_COLOR, 5),
            (np.argwhere(regressed_grid), REGRESSED_COLOR, 6),
            (np.argwhere(improved_grid), IMPROVED_COLOR, 6),
        ],
        frame_coords=frame_coords,
    )

    panels = [gt_panel, raw_panel, refined_panel, xray_panel]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), 540), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title[:180], fill=(20, 20, 20))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 34))
        x += panel.width
    legend_y = 512
    legend_x = 12
    for class_index, name in enumerate(class_names[:10]):
        color = _class_color(class_index)
        draw.rectangle((legend_x, legend_y, legend_x + 10, legend_y + 10), fill=color)
        draw.text((legend_x + 14, legend_y - 1), str(name)[:22], fill=(25, 25, 25))
        legend_x += min(180, 25 + len(str(name)[:22]) * 7)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)


def run_joint_boundary_diagnostics(
    joint_partition_path: str | Path,
    *,
    data_root: str | Path,
    ds_sample: dict[str, Any],
    output_png: str | Path,
    output_json: str | Path | None = None,
    whole_pred_path: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Evaluate and render one ``joint_partition.npz`` artifact."""
    joint_partition_path = Path(joint_partition_path)
    output_png = Path(output_png)
    output_json = Path(output_json) if output_json is not None else output_png.with_suffix(".json")
    data_root = Path(data_root)
    partition = _load_partition(joint_partition_path)
    class_names = partition["class_names"]
    gt = _load_ground_truth(data_root, ds_sample, class_names)

    coords = partition["coords"]
    candidate = _coords_to_mask(coords)
    raw_grid = _prediction_grid(coords, partition["labels_raw"])
    refined_grid = _prediction_grid(coords, partition["labels_refined"])
    pair_a, pair_b, boundary = _cross_label_pairs(gt["labels"])
    raw_metrics = _method_metrics(raw_grid, gt["labels"], candidate, class_names, pair_a, pair_b, boundary)
    refined_metrics = _method_metrics(refined_grid, gt["labels"], candidate, class_names, pair_a, pair_b, boundary)

    inferred_whole = joint_partition_path.parent.parent / "voxel.npz"
    selected_whole_path = Path(whole_pred_path) if whole_pred_path is not None else inferred_whole
    if selected_whole_path.is_file():
        pred_whole = _coords_to_mask(_load_coords(selected_whole_path))
        whole_source = str(selected_whole_path)
        whole_fallback = False
    else:
        pred_whole = candidate.copy()
        whole_source = "candidate_coords_fallback"
        whole_fallback = True
    whole_metrics = _binary_iou(pred_whole, gt["whole"])

    gt_valid = gt["labels"] >= 0
    gt_parts_unique = gt["labels"] > 0
    gt_part_claim_union = gt["claim_count"] > 0
    candidate_metrics = {
        "voxels": int(candidate.sum()),
        "recall": float((candidate & gt_valid).sum() / max(1, int(gt_valid.sum()))),
        "gt_unique_voxels": int(gt_valid.sum()),
        "gt_whole_recall": float((candidate & gt["whole"]).sum() / max(1, int(gt["whole"].sum()))),
        "gt_part_unique_recall": float((candidate & gt_parts_unique).sum() / max(1, int(gt_parts_unique.sum()))),
        "gt_part_claim_recall": float((candidate & gt_part_claim_union).sum() / max(1, int(gt_part_claim_union.sum()))),
        "pred_whole_coverage": float((candidate & pred_whole).sum() / max(1, int(pred_whole.sum()))),
        "outside_pred_whole": int((candidate & ~pred_whole).sum()),
    }

    changed = partition["labels_raw"] != partition["labels_refined"]
    gt_at_candidate = gt["labels"][coords[:, 0], coords[:, 1], coords[:, 2]]
    raw_correct = (gt_at_candidate >= 0) & (partition["labels_raw"] == gt_at_candidate)
    refined_correct = (gt_at_candidate >= 0) & (partition["labels_refined"] == gt_at_candidate)
    improved = changed & ~raw_correct & refined_correct
    regressed = changed & raw_correct & ~refined_correct
    neutral = changed & ~improved & ~regressed
    improved_grid = _coords_to_mask(coords[improved])
    regressed_grid = _coords_to_mask(coords[regressed])
    neutral_grid = _coords_to_mask(coords[neutral])

    low_margin, low_margin_meta = _low_margin_mask(partition["logits"], partition["refinement"])
    low_margin_grid = _coords_to_mask(coords[low_margin])
    boundary_at_candidate = boundary[coords[:, 0], coords[:, 1], coords[:, 2]]
    logits_argmax = partition["logits"].argmax(axis=1)
    low_margin_metrics = {
        **low_margin_meta,
        "voxels": int(low_margin.sum()),
        "low_margin_voxels": int(low_margin.sum()),
        "fraction": float(low_margin.mean()) if low_margin.size else 0.0,
        "changed_voxels": int((low_margin & changed).sum()),
        "interface_voxels": int((low_margin & boundary_at_candidate).sum()),
    }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "joint_partition_path": str(joint_partition_path.resolve()),
        "data_root": str(data_root.resolve()),
        "candidate_mode": str(partition["refinement"].get("candidate_mode", "unknown")),
        "dataset_id": str(ds_sample.get("_eval_dataset_id") or ds_sample.get("dataset_id") or ""),
        "sample": {
            "dataset_id": str(ds_sample.get("_eval_dataset_id") or ds_sample.get("dataset_id") or ""),
            "object_id": str(ds_sample.get("obj_id", ds_sample.get("object_id", ""))),
            "angle": int(ds_sample.get("angle_idx", ds_sample.get("angle", 0))),
        },
        "class_names": class_names,
        "gt": {
            "whole_voxels": int(gt["whole"].sum()),
            "unique_label_voxels": int(gt_valid.sum()),
            "multi_claim_ignore_voxels": int(gt["overlap"].sum()),
            "part_voxels_outside_whole": gt["part_voxels_outside_whole"],
            "surface_path": gt["surface_path"],
            "part_paths": gt["part_paths"],
        },
        "candidate": candidate_metrics,
        "whole": {
            **whole_metrics,
            "source": whole_source,
            "candidate_fallback": whole_fallback,
        },
        "interface": {
            "pairs": int(pair_a.shape[0]),
            "boundary_voxels": int(boundary.sum()),
            "pairs_covered": int(raw_metrics["interface_pairs_covered"]),
            "pair_coverage": raw_metrics["interface_pair_coverage"],
        },
        "raw": raw_metrics,
        "refined": refined_metrics,
        "changed": {
            "voxels": int(changed.sum()),
            "improved": int(improved.sum()),
            "regressed": int(regressed.sum()),
            "neutral": int(neutral.sum()),
            "net_improved": int(improved.sum() - regressed.sum()),
            "ignored_gt": int((changed & (gt_at_candidate < 0)).sum()),
        },
        "low_margin": low_margin_metrics,
        "artifact_checks": {
            "raw_label_logit_argmax_mismatch": int((partition["labels_raw"] != logits_argmax).sum()),
            "refinement_json": partition["refinement"],
        },
        "artifacts": {
            "png": str(output_png.resolve()),
            "json": str(output_json.resolve()),
        },
    }

    object_id = payload["sample"]["object_id"]
    angle = payload["sample"]["angle"]
    render_title = title or f"joint boundary diagnostics: {object_id} angle={angle}"
    render_title = f"{render_title} | candidate={payload['candidate_mode']}"
    _render_diagnostic(
        output_png,
        title=render_title,
        gt=gt,
        raw_grid=raw_grid,
        refined_grid=refined_grid,
        candidate=candidate,
        pred_whole=pred_whole,
        raw_metrics=raw_metrics,
        refined_metrics=refined_metrics,
        boundary=boundary,
        low_margin_grid=low_margin_grid,
        improved_grid=improved_grid,
        regressed_grid=regressed_grid,
        neutral_grid=neutral_grid,
        class_names=class_names,
        whole_iou=float(whole_metrics["iou"]),
        interface_coverage=raw_metrics["interface_pair_coverage"],
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize a saved joint voxel partition.")
    parser.add_argument("--joint-partition", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, required=True, help="Normalized dataset sample JSON")
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--whole-pred", type=Path, default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ds_sample = json.loads(args.sample_json.read_text(encoding="utf-8"))
    if not isinstance(ds_sample, dict):
        raise ValueError(f"{args.sample_json}: expected a JSON object")
    payload = run_joint_boundary_diagnostics(
        args.joint_partition,
        data_root=args.data_root,
        ds_sample=ds_sample,
        output_png=args.output_png,
        output_json=args.output_json,
        whole_pred_path=args.whole_pred,
        title=args.title,
    )
    print(json.dumps({"raw": payload["raw"], "refined": payload["refined"], "artifacts": payload["artifacts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
