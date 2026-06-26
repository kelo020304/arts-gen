#!/usr/bin/env python3
"""Pure-GT motion-grounded boundary owner oracle diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


OUT_DEFAULT = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg_full_S_0618-2/eval/diagnostics_motion_oracle"
)
SPLIT_DEFAULT = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_verse_realappliance_0511dd_v4.json"
)

Y_UP_TO_Z_UP_3 = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
Y_UP_TO_Z_UP_4 = np.eye(4, dtype=np.float64)
Y_UP_TO_Z_UP_4[:3, :3] = Y_UP_TO_Z_UP_3

NEIGHBORS6 = np.asarray(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int16,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_roots(split_path: Path) -> dict[str, Path]:
    split = read_json(split_path)
    return {str(item["dataset_id"]): Path(item["data_root"]) for item in split.get("datasets", [])}


def held_keys(split_path: Path) -> list[str]:
    split = read_json(split_path)
    keys: list[str] = []
    for name in ("heldout_keys", "realappliance_heldout_keys", "physx0511_drawer_door_heldout_keys"):
        for key in split.get(name, []) or []:
            if key not in keys:
                keys.append(str(key))
    return keys


def split_key(key: str) -> tuple[str, str]:
    if "::" in key:
        ds, obj = key.split("::", 1)
        return ds, obj
    return "", key


def needs_y_up_fix(dataset_id: str, data_root: Path) -> bool:
    text = f"{dataset_id} {data_root}".lower()
    return (
        "phyx-verse" in text
        or "phyx_verse" in text
        or "physx-0511" in text
        or "physx-mobility" in text
    )


def part_transform_to_voxel_frame(matrix: Any, *, dataset_id: str, data_root: Path) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"expected 4x4 part transform, got {mat.shape}")
    if needs_y_up_fix(dataset_id, data_root):
        return Y_UP_TO_Z_UP_4 @ mat @ np.linalg.inv(Y_UP_TO_Z_UP_4)
    return mat


def load_camera_norm(data_root: Path, obj_id: str, angle: int) -> tuple[float, np.ndarray] | None:
    path = data_root / "renders" / obj_id / f"angle_{int(angle)}" / "camera_transforms.json"
    if not path.is_file():
        return None
    payload = read_json(path)
    if "scale" not in payload or "offset" not in payload:
        return None
    return float(payload["scale"]), np.asarray(payload["offset"], dtype=np.float64)


def coords_path(data_root: Path, obj_id: str, angle: int, part_name: str) -> Path:
    return data_root / "reconstruction" / "voxel_expanded" / obj_id / f"angle_{int(angle)}" / "64" / f"ind_{part_name}.npy"


def load_coords(data_root: Path, obj_id: str, angle: int, part_name: str) -> np.ndarray | None:
    path = coords_path(data_root, obj_id, angle, part_name)
    if not path.is_file():
        return None
    arr = np.asarray(np.load(path), dtype=np.int16)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def dense_labels(part_coords: dict[str, np.ndarray]) -> np.ndarray:
    labels = np.full((64, 64, 64), -1, dtype=np.int16)
    for idx, coords in enumerate(part_coords.values()):
        if coords.size == 0:
            continue
        c = coords.astype(np.int64, copy=False)
        labels[c[:, 0], c[:, 1], c[:, 2]] = idx
    return labels


def coord_keys(coords: np.ndarray) -> set[int]:
    if coords.size == 0:
        return set()
    c = coords.astype(np.int64, copy=False)
    return set((c[:, 0] * 4096 + c[:, 1] * 64 + c[:, 2]).tolist())


def bucket_name(count: int) -> str:
    if count < 50:
        return "tiny"
    if count < 500:
        return "small"
    if count < 2000:
        return "medium"
    return "large"


def grid_to_world(coords: np.ndarray, scale: float, offset: np.ndarray) -> np.ndarray:
    unit = (coords.astype(np.float64) + 0.5) / 64.0 - 0.5
    return (unit - offset.reshape(1, 3)) / max(float(scale), 1.0e-12)


def world_to_grid(world: np.ndarray, scale: float, offset: np.ndarray) -> np.ndarray:
    unit = world * float(scale) + offset.reshape(1, 3)
    return (unit + 0.5) * 64.0 - 0.5


def transform_grid(
    coords: np.ndarray,
    transform_ab: np.ndarray,
    norm_a: tuple[float, np.ndarray],
    norm_b: tuple[float, np.ndarray],
) -> np.ndarray:
    scale_a, offset_a = norm_a
    scale_b, offset_b = norm_b
    world_a = grid_to_world(coords, scale_a, offset_a)
    ones = np.ones((world_a.shape[0], 1), dtype=np.float64)
    world_b = (np.concatenate([world_a, ones], axis=1) @ transform_ab.T)[:, :3]
    return world_to_grid(world_b, scale_b, offset_b)


def nearest_iou(coords_a: np.ndarray, coords_b: np.ndarray, grid_b: np.ndarray) -> tuple[float, float, float]:
    if coords_a.size == 0 or coords_b.size == 0 or grid_b.size == 0:
        return float("nan"), 0.0, 0.0
    rounded = np.rint(grid_b).astype(np.int16)
    valid = np.all((rounded >= 0) & (rounded < 64), axis=1)
    pred = rounded[valid]
    pred_keys = coord_keys(pred)
    gt_keys = coord_keys(coords_b)
    inter = len(pred_keys & gt_keys)
    union = len(pred_keys | gt_keys)
    iou = inter / max(1, union)
    precision = inter / max(1, len(pred_keys))
    recall = inter / max(1, len(gt_keys))
    return iou, precision, recall


def nearest_residual_metrics(coords_b: np.ndarray, grid_b: np.ndarray) -> dict[str, float]:
    """Bidirectional nearest-neighbor residuals in 64^3 voxel units."""

    if coords_b.size == 0 or grid_b.size == 0:
        return {
            "self_p2g_mean": float("nan"),
            "self_p2g_median": float("nan"),
            "self_p2g_p90": float("nan"),
            "self_g2p_mean": float("nan"),
            "self_g2p_median": float("nan"),
            "self_g2p_p90": float("nan"),
            "self_frac_p2g_le_0p5": float("nan"),
            "self_frac_p2g_le_1": float("nan"),
            "self_frac_p2g_le_2": float("nan"),
            "self_frac_g2p_le_0p5": float("nan"),
            "self_frac_g2p_le_1": float("nan"),
            "self_frac_g2p_le_2": float("nan"),
            "self_tol_f1_0p5": float("nan"),
            "self_tol_f1_1": float("nan"),
            "self_tol_f1_2": float("nan"),
        }

    target = coords_b.astype(np.float64, copy=False)
    pred = grid_b.astype(np.float64, copy=False)
    p2g, _ = cKDTree(target).query(pred, k=1, workers=1)
    g2p, _ = cKDTree(pred).query(target, k=1, workers=1)

    def stats(prefix: str, vals: np.ndarray) -> dict[str, float]:
        return {
            f"{prefix}_mean": float(np.mean(vals)),
            f"{prefix}_median": float(np.median(vals)),
            f"{prefix}_p90": float(np.percentile(vals, 90)),
        }

    out: dict[str, float] = {}
    out.update(stats("self_p2g", p2g))
    out.update(stats("self_g2p", g2p))
    for radius, name in [(0.5, "0p5"), (1.0, "1"), (2.0, "2")]:
        precision = float(np.mean(p2g <= radius))
        recall = float(np.mean(g2p <= radius))
        out[f"self_frac_p2g_le_{name}"] = precision
        out[f"self_frac_g2p_le_{name}"] = recall
        out[f"self_tol_f1_{name}"] = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    return out


def boundary_mask_for_part(labels: np.ndarray, part_idx: int, coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0,), dtype=bool)
    out = np.zeros((coords.shape[0],), dtype=bool)
    c32 = coords.astype(np.int16, copy=False)
    for nb in NEIGHBORS6:
        n = c32 + nb.reshape(1, 3)
        valid = np.all((n >= 0) & (n < 64), axis=1)
        if not np.any(valid):
            continue
        vals = np.full((coords.shape[0],), part_idx, dtype=np.int16)
        nv = n[valid].astype(np.int64, copy=False)
        vals[valid] = labels[nv[:, 0], nv[:, 1], nv[:, 2]]
        out |= valid & (vals >= 0) & (vals != part_idx)
    return out


def boundary_neighbor_indices(labels: np.ndarray, part_idx: int, coords: np.ndarray) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(coords.shape[0])]
    if coords.size == 0:
        return neighbors
    c32 = coords.astype(np.int16, copy=False)
    for nb in NEIGHBORS6:
        n = c32 + nb.reshape(1, 3)
        valid = np.all((n >= 0) & (n < 64), axis=1)
        if not np.any(valid):
            continue
        nv = n[valid].astype(np.int64, copy=False)
        vals = labels[nv[:, 0], nv[:, 1], nv[:, 2]]
        valid_indices = np.flatnonzero(valid)
        for row_idx, val in zip(valid_indices.tolist(), vals.tolist()):
            if int(val) >= 0 and int(val) != int(part_idx):
                neighbors[row_idx].add(int(val))
    return neighbors


def relative_motion_norm(transform_true: np.ndarray, transform_other: np.ndarray) -> float:
    delta = np.linalg.inv(transform_other) @ transform_true
    rot = delta[:3, :3]
    trans = delta[:3, 3]
    rot_delta = float(np.linalg.norm(rot - np.eye(3), ord="fro"))
    trans_delta = float(np.linalg.norm(trans))
    return rot_delta + trans_delta


def summarize(rows: list[dict[str, Any]], key: str = "correct") -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "mean": float("nan")}
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return {"n": n, "mean": float(vals.mean())}


def summarize_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = np.asarray([float(r[key]) for r in rows if math.isfinite(float(r[key]))], dtype=np.float64)
    if vals.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_case_png(path: Path, case: dict[str, Any], coords: np.ndarray, true_owner: np.ndarray, pred_owner: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if coords.size == 0:
        return
    # Use the slice with most boundary voxels for a compact diagnostic image.
    z_vals, counts = np.unique(coords[:, 2], return_counts=True)
    z = int(z_vals[int(np.argmax(counts))])
    sel = coords[:, 2] == z
    coords2 = coords[sel]
    true2 = true_owner[sel]
    pred2 = pred_owner[sel]
    max_owner = max(int(true_owner.max(initial=0)), int(pred_owner.max(initial=0)), 1)
    true_img = np.full((64, 64), np.nan)
    pred_img = np.full((64, 64), np.nan)
    err_img = np.zeros((64, 64), dtype=np.float32)
    for c, t, p in zip(coords2, true2, pred2):
        true_img[int(c[1]), int(c[0])] = int(t)
        pred_img[int(c[1]), int(c[0])] = int(p)
        err_img[int(c[1]), int(c[0])] = 1.0 if int(t) != int(p) else 0.0
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=160)
    for ax, img, title in [
        (axes[0], true_img, "true owner"),
        (axes[1], pred_img, "motion owner"),
        (axes[2], err_img, "error"),
    ]:
        cmap = "tab20" if title != "error" else "Reds"
        ax.imshow(img, origin="lower", interpolation="nearest", cmap=cmap, vmin=0, vmax=max_owner)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{case['dataset_id']}::{case['obj_id']} a{case['angle_a']}->{case['angle_b']} z={z} acc={case['boundary_acc']:.3f}",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_margin_histogram(path: Path, owner_rows: list[dict[str, Any]]) -> None:
    vals = np.asarray(
        [
            float(r["runnerup_minus_true"])
            for r in owner_rows
            if r.get("scope") == "boundary" and math.isfinite(float(r["runnerup_minus_true"]))
        ],
        dtype=np.float64,
    )
    if vals.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=160)
    clipped = np.clip(vals, -5.0, 5.0)
    ax.hist(clipped, bins=80, color="#356a9a", alpha=0.9)
    ax.axvline(0.0, color="#b33a3a", linewidth=1.0)
    ax.axvline(float(np.median(vals)), color="#222222", linewidth=1.0, linestyle="--")
    ax.set_title(f"Boundary residual margin, median={np.median(vals):.3f}")
    ax.set_xlabel("runnerup residual - true residual (voxels, clipped to [-5,5])")
    ax.set_ylabel("voxels")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def object_candidates(split_path: Path, max_objects: int, seed: int, objects_per_dataset: int = 0) -> list[tuple[str, str]]:
    roots = dataset_roots(split_path)
    keys = held_keys(split_path)
    rng = random.Random(seed)
    rng.shuffle(keys)

    if objects_per_dataset > 0:
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key in keys:
            ds, obj = split_key(key)
            root = roots.get(ds)
            if root is None:
                continue
            if not (root / "joint_transforms" / f"{obj}.json").is_file():
                continue
            if not (root / "reconstruction" / "part_info" / obj / "part_info.json").is_file():
                continue
            if not (root / "reconstruction" / "voxel_expanded" / obj).is_dir():
                continue
            grouped[ds].append((ds, obj))
        out_balanced: list[tuple[str, str]] = []
        for ds in sorted(grouped):
            out_balanced.extend(grouped[ds][:objects_per_dataset])
        return out_balanced[:max_objects] if max_objects > 0 else out_balanced

    out: list[tuple[str, str]] = []
    for key in keys:
        ds, obj = split_key(key)
        root = roots.get(ds)
        if root is None:
            continue
        if not (root / "joint_transforms" / f"{obj}.json").is_file():
            continue
        if not (root / "reconstruction" / "part_info" / obj / "part_info.json").is_file():
            continue
        if not (root / "reconstruction" / "voxel_expanded" / obj).is_dir():
            continue
        out.append((ds, obj))
        if max_objects > 0 and len(out) >= max_objects:
            break
    return out


def analyze_object(
    dataset_id: str,
    obj_id: str,
    data_root: Path,
    *,
    max_pairs_per_object: int,
    max_boundary_voxels_per_part: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]]]:
    part_info_path = data_root / "reconstruction" / "part_info" / obj_id / "part_info.json"
    jt_path = data_root / "joint_transforms" / f"{obj_id}.json"
    part_info = read_json(part_info_path).get("parts", {})
    jt = read_json(jt_path)
    angles_meta = jt.get("angles", {})
    angles = sorted(int(a) for a in angles_meta.keys() if str(a).isdigit())
    if len(angles) < 2:
        return [], [], [], []

    parts: list[dict[str, Any]] = []
    for name, meta in part_info.items():
        pidx = meta.get("part_index", meta.get("label"))
        if pidx is None:
            continue
        joint_type = str(meta.get("joint_type") or meta.get("joint") or "unknown")
        parts.append({"name": str(name), "part_index": int(pidx), "joint_type": joint_type, "meta": meta})
    if len(parts) < 2:
        return [], [], [], []
    movable = [p for p in parts if p["joint_type"] in {"B", "C", "prismatic", "revolute"}]
    if not movable:
        return [], [], [], []

    norms = {a: load_camera_norm(data_root, obj_id, a) for a in angles}
    angles = [a for a in angles if norms.get(a) is not None]
    if len(angles) < 2:
        return [], [], [], []

    # Prefer wider angle gaps, but keep a few pairs per object.
    pairs = [(a, b) for a in angles for b in angles if a != b]
    pairs.sort(key=lambda x: (-abs(x[1] - x[0]), x[0], x[1]))
    pairs = pairs[: max_pairs_per_object]

    sanity_rows: list[dict[str, Any]] = []
    owner_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    visuals: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []

    coords_cache: dict[tuple[int, str], np.ndarray | None] = {}
    label_cache: dict[int, tuple[list[str], dict[str, np.ndarray], np.ndarray]] = {}
    transform_cache: dict[tuple[int, str], np.ndarray] = {}

    def coords_for(angle: int, part_name: str) -> np.ndarray | None:
        key = (angle, part_name)
        if key not in coords_cache:
            coords_cache[key] = load_coords(data_root, obj_id, angle, part_name)
        return coords_cache[key]

    def transform_for(angle: int, part: dict[str, Any]) -> np.ndarray | None:
        key = (angle, part["name"])
        if key in transform_cache:
            return transform_cache[key]
        meta = angles_meta.get(str(angle), {})
        raw = (meta.get("part_transforms") or {}).get(str(int(part["part_index"])))
        if raw is None:
            return None
        transform_cache[key] = part_transform_to_voxel_frame(raw, dataset_id=dataset_id, data_root=data_root)
        return transform_cache[key]

    def labels_for(angle: int) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
        if angle not in label_cache:
            present: dict[str, np.ndarray] = {}
            names: list[str] = []
            for p in parts:
                c = coords_for(angle, p["name"])
                if c is None or c.size == 0:
                    continue
                present[p["name"]] = c
                names.append(p["name"])
            label_cache[angle] = (names, present, dense_labels(present))
        return label_cache[angle]

    for angle_a, angle_b in pairs:
        norm_a = norms[angle_a]
        norm_b = norms[angle_b]
        if norm_a is None or norm_b is None:
            continue
        names_a, coords_a_by_name, labels_a = labels_for(angle_a)
        _names_b, coords_b_by_name, _labels_b = labels_for(angle_b)
        if len(names_a) < 2:
            continue

        candidate_parts = [p for p in parts if p["name"] in coords_a_by_name and p["name"] in coords_b_by_name]
        if len(candidate_parts) < 2:
            continue

        trees: dict[str, cKDTree] = {}
        for p in candidate_parts:
            cb = coords_b_by_name[p["name"]]
            if cb.size > 0:
                trees[p["name"]] = cKDTree(cb.astype(np.float64))
        if len(trees) < 2:
            continue

        transforms_ab: dict[str, np.ndarray] = {}
        for p in candidate_parts:
            ta = transform_for(angle_a, p)
            tb = transform_for(angle_b, p)
            if ta is None or tb is None:
                continue
            transforms_ab[p["name"]] = tb @ np.linalg.inv(ta)

        for p in candidate_parts:
            if p["name"] not in transforms_ab:
                continue
            ca = coords_a_by_name[p["name"]]
            cb = coords_b_by_name[p["name"]]
            grid_b = transform_grid(ca, transforms_ab[p["name"]], norm_a, norm_b)
            iou, precision, recall = nearest_iou(ca, cb, grid_b)
            residual_metrics = nearest_residual_metrics(cb, grid_b)
            sanity_rows.append(
                {
                    "dataset_id": dataset_id,
                    "obj_id": obj_id,
                    "angle_a": angle_a,
                    "angle_b": angle_b,
                    "angle_gap": abs(angle_b - angle_a),
                    "part_name": p["name"],
                    "joint_type": p["joint_type"],
                    "bucket": bucket_name(int(ca.shape[0])),
                    "raw_count": int(ca.shape[0]),
                    "self_iou": iou,
                    "self_precision": precision,
                    "self_recall": recall,
                    **residual_metrics,
                }
            )

        part_index_local = {name: idx for idx, name in enumerate(coords_a_by_name.keys())}
        local_name_by_idx = list(coords_a_by_name.keys())
        part_meta_by_name = {p["name"]: p for p in candidate_parts}
        pair_boundary_total = 0
        pair_motion_total = 0
        pair_motion_correct = 0
        pair_all_total = 0
        pair_all_correct = 0
        vis_coords: list[np.ndarray] = []
        vis_true: list[np.ndarray] = []
        vis_pred: list[np.ndarray] = []

        for p in candidate_parts:
            pname = p["name"]
            if pname not in transforms_ab:
                continue
            ca = coords_a_by_name[pname]
            local_idx = part_index_local[pname]
            bmask = boundary_mask_for_part(labels_a, local_idx, ca)
            neighbor_sets = boundary_neighbor_indices(labels_a, local_idx, ca)
            for scope, selected in [("boundary", bmask), ("all", np.ones((ca.shape[0],), dtype=bool))]:
                idxs = np.flatnonzero(selected)
                if idxs.size == 0:
                    continue
                if scope == "boundary" and idxs.size > max_boundary_voxels_per_part:
                    idxs = np.asarray(rng.sample(idxs.tolist(), max_boundary_voxels_per_part), dtype=np.int64)
                coords_sel = ca[idxs]
                transformed_by_candidate: dict[str, np.ndarray] = {}
                residuals: list[np.ndarray] = []
                candidate_names: list[str] = []
                for cand in candidate_parts:
                    cname = cand["name"]
                    if cname not in transforms_ab or cname not in trees:
                        continue
                    grid = transform_grid(coords_sel, transforms_ab[cname], norm_a, norm_b)
                    dist, _ = trees[cname].query(grid, k=1, workers=1)
                    residuals.append(dist.astype(np.float32))
                    candidate_names.append(cname)
                    transformed_by_candidate[cname] = grid
                if len(residuals) < 2:
                    continue
                res = np.stack(residuals, axis=1)
                pred_idx = np.argmin(res, axis=1)
                true_cand_idx = candidate_names.index(pname) if pname in candidate_names else -1
                if true_cand_idx < 0:
                    continue
                true_res = res[:, true_cand_idx]
                sorted_res = np.sort(res, axis=1)
                runnerup = sorted_res[:, 1] if res.shape[1] > 1 else sorted_res[:, 0]
                margin = runnerup - true_res
                pred_names = [candidate_names[int(i)] for i in pred_idx]
                correct = np.asarray([name == pname for name in pred_names], dtype=bool)
                rel_motion_by_candidate = {
                    cname: relative_motion_norm(transforms_ab[pname], transforms_ab[cname])
                    for cname in candidate_names
                }
                rel_motion_nontrivial_any = bool(max(rel_motion_by_candidate.values(), default=0.0) > 1.0e-4)
                if scope == "boundary":
                    rel_motion_flags: list[bool] = []
                    for original_idx in idxs.tolist():
                        neighbor_names = [
                            local_name_by_idx[nidx]
                            for nidx in neighbor_sets[original_idx]
                            if nidx < len(local_name_by_idx) and local_name_by_idx[nidx] in rel_motion_by_candidate
                        ]
                        rel_motion_flags.append(
                            bool(max((rel_motion_by_candidate[nname] for nname in neighbor_names), default=0.0) > 1.0e-4)
                        )
                    rel_motion_flags_arr = np.asarray(rel_motion_flags, dtype=bool)
                else:
                    rel_motion_flags_arr = np.full((coords_sel.shape[0],), rel_motion_nontrivial_any, dtype=bool)
                for i in range(coords_sel.shape[0]):
                    row = {
                        "dataset_id": dataset_id,
                        "obj_id": obj_id,
                        "angle_a": angle_a,
                        "angle_b": angle_b,
                        "angle_gap": abs(angle_b - angle_a),
                        "scope": scope,
                        "part_name": pname,
                        "joint_type": p["joint_type"],
                        "bucket": bucket_name(int(ca.shape[0])),
                        "raw_count": int(ca.shape[0]),
                        "true_residual": float(true_res[i]),
                        "runnerup_minus_true": float(margin[i]),
                        "pred_owner": pred_names[i],
                        "true_owner": pname,
                        "correct": int(correct[i]),
                        "relative_motion_nontrivial": int(rel_motion_flags_arr[i]),
                    }
                    owner_rows.append(row)
                if scope == "boundary":
                    pair_boundary_total += int(coords_sel.shape[0])
                    pair_motion_total += int(rel_motion_flags_arr.sum())
                    pair_motion_correct += int(correct.sum())
                    if len(visuals) < 10:
                        vis_coords.append(coords_sel)
                        vis_true.append(np.full((coords_sel.shape[0],), true_cand_idx, dtype=np.int16))
                        vis_pred.append(pred_idx.astype(np.int16))
                else:
                    pair_all_total += int(coords_sel.shape[0])
                    pair_all_correct += int(correct.sum())

        if pair_boundary_total > 0:
            cov = pair_motion_total / max(1, pair_boundary_total)
            acc = pair_motion_correct / max(1, pair_boundary_total)
            case = {
                "dataset_id": dataset_id,
                "obj_id": obj_id,
                "angle_a": angle_a,
                "angle_b": angle_b,
                "angle_gap": abs(angle_b - angle_a),
                "boundary_voxels": pair_boundary_total,
                "motion_coverable_voxels": pair_motion_total,
                "motion_coverable_ratio": cov,
                "boundary_acc": acc,
                "all_voxels": pair_all_total,
                "all_acc": pair_all_correct / max(1, pair_all_total),
            }
            coverage_rows.append(case)
            if vis_coords:
                coords_vis = np.concatenate(vis_coords, axis=0)
                true_vis = np.concatenate(vis_true, axis=0)
                pred_vis = np.concatenate(vis_pred, axis=0)
                visuals.append((case, coords_vis, true_vis, pred_vis))

    return sanity_rows, owner_rows, coverage_rows, visuals


def aggregate_tables(owner_rows: list[dict[str, Any]], sanity_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}

    sanity_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sanity_rows:
        sanity_groups["all"].append(r)
        sanity_groups[f"dataset:{r['dataset_id']}"].append(r)
        sanity_groups[f"joint:{r['joint_type']}"].append(r)
        sanity_groups[f"bucket:{r['bucket']}"].append(r)
        sanity_groups[f"gap:{r['angle_gap']}"].append(r)
    tables["sanity"] = [
        {
            "group": k,
            **summarize_numeric(v, "self_iou"),
            "p2g_median": summarize_numeric(v, "self_p2g_median")["median"],
            "p2g_p90": summarize_numeric(v, "self_p2g_p90")["median"],
            "tol_f1_0p5": summarize_numeric(v, "self_tol_f1_0p5")["mean"],
            "tol_f1_1": summarize_numeric(v, "self_tol_f1_1")["mean"],
            "tol_f1_2": summarize_numeric(v, "self_tol_f1_2")["mean"],
        }
        for k, v in sorted(sanity_groups.items())
    ]

    for scope in ("boundary", "all"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        selected = [r for r in owner_rows if r["scope"] == scope]
        for r in selected:
            groups["all"].append(r)
            groups[f"dataset:{r['dataset_id']}"].append(r)
            groups[f"joint:{r['joint_type']}"].append(r)
            groups[f"bucket:{r['bucket']}"].append(r)
            groups[f"gap:{r['angle_gap']}"].append(r)
            groups[f"coverable:{int(r['relative_motion_nontrivial'])}"].append(r)
        tables[f"{scope}_accuracy"] = [
            {"group": k, "n": len(v), "accuracy": summarize(v, "correct")["mean"], **summarize_numeric(v, "runnerup_minus_true")}
            for k, v in sorted(groups.items())
        ]

    cov_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in coverage_rows:
        cov_groups["all"].append(r)
        cov_groups[f"dataset:{r['dataset_id']}"].append(r)
        cov_groups[f"gap:{r['angle_gap']}"].append(r)
    tables["coverage"] = []
    for k, vals in sorted(cov_groups.items()):
        boundary = sum(int(v["boundary_voxels"]) for v in vals)
        cover = sum(int(v["motion_coverable_voxels"]) for v in vals)
        correct = sum(float(v["boundary_acc"]) * int(v["boundary_voxels"]) for v in vals)
        tables["coverage"].append(
            {
                "group": k,
                "cases": len(vals),
                "boundary_voxels": boundary,
                "motion_coverable_voxels": cover,
                "motion_coverable_ratio": cover / max(1, boundary),
                "boundary_accuracy": correct / max(1, boundary),
            }
        )
    return tables


def print_table(title: str, rows: list[dict[str, Any]], max_rows: int = 20) -> None:
    print(f"\n[{title}]")
    if not rows:
        print("(empty)")
        return
    fields = list(rows[0].keys())
    widths = {f: max(len(f), *(len(f"{r.get(f, ''):.4f}") if isinstance(r.get(f), float) else len(str(r.get(f, ""))) for r in rows[:max_rows])) for f in fields}
    print(" | ".join(f.ljust(widths[f]) for f in fields))
    print("-+-".join("-" * widths[f] for f in fields))
    for r in rows[:max_rows]:
        vals = []
        for f in fields:
            v = r.get(f, "")
            vals.append((f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(widths[f]))
        print(" | ".join(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, default=SPLIT_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--max-objects", type=int, default=30)
    parser.add_argument(
        "--objects-per-dataset",
        type=int,
        default=0,
        help="If >0, sample up to this many held objects per dataset before max-objects is applied.",
    )
    parser.add_argument("--max-pairs-per-object", type=int, default=3)
    parser.add_argument("--max-boundary-voxels-per-part", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--visuals", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = dataset_roots(args.split_json)
    objects = object_candidates(
        args.split_json,
        int(args.max_objects),
        int(args.seed),
        objects_per_dataset=int(args.objects_per_dataset),
    )
    rng = random.Random(int(args.seed))

    all_sanity: list[dict[str, Any]] = []
    all_owner: list[dict[str, Any]] = []
    all_coverage: list[dict[str, Any]] = []
    all_visuals: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []

    for idx, (dataset_id, obj_id) in enumerate(objects):
        root = roots[dataset_id]
        try:
            sanity, owner, coverage, visuals = analyze_object(
                dataset_id,
                obj_id,
                root,
                max_pairs_per_object=int(args.max_pairs_per_object),
                max_boundary_voxels_per_part=int(args.max_boundary_voxels_per_part),
                rng=rng,
            )
        except Exception as exc:
            print(f"[warn] skip {dataset_id}::{obj_id}: {exc}", flush=True)
            continue
        all_sanity.extend(sanity)
        all_owner.extend(owner)
        all_coverage.extend(coverage)
        all_visuals.extend(visuals)
        print(
            f"[object {idx + 1}/{len(objects)}] {dataset_id}::{obj_id} "
            f"sanity={len(sanity)} owner_voxels={len(owner)} cases={len(coverage)}",
            flush=True,
        )

    write_csv(args.out_dir / "self搬_sanity_rows.csv", all_sanity)
    write_csv(args.out_dir / "owner_oracle_voxels.csv", all_owner)
    write_csv(args.out_dir / "coverage_cases.csv", all_coverage)
    tables = aggregate_tables(all_owner, all_sanity, all_coverage)
    for name, rows in tables.items():
        write_csv(args.out_dir / f"{name}.csv", rows)

    worst = sorted(all_coverage, key=lambda r: float(r["boundary_acc"]))[: int(args.visuals)]
    selected = []
    for case in worst:
        for item in all_visuals:
            if item[0] is case or (
                item[0]["dataset_id"] == case["dataset_id"]
                and item[0]["obj_id"] == case["obj_id"]
                and item[0]["angle_a"] == case["angle_a"]
                and item[0]["angle_b"] == case["angle_b"]
            ):
                selected.append(item)
                break
    manifest = []
    for idx, (case, coords, true_owner, pred_owner) in enumerate(selected[: int(args.visuals)]):
        png = args.out_dir / "visuals" / f"{idx:02d}_{case['dataset_id']}__{case['obj_id']}_a{case['angle_a']}_to_a{case['angle_b']}.png"
        save_case_png(png, case, coords, true_owner, pred_owner)
        manifest.append({**case, "png": str(png)})
    write_csv(args.out_dir / "visuals_manifest.csv", manifest)
    save_margin_histogram(args.out_dir / "boundary_margin_hist.png", all_owner)

    worst_cases = sorted(all_coverage, key=lambda r: float(r["boundary_acc"]))[:20]
    write_csv(args.out_dir / "worst_cases.csv", worst_cases)

    summary = {
        "objects_requested": len(objects),
        "sanity_rows": len(all_sanity),
        "owner_rows": len(all_owner),
        "coverage_cases": len(all_coverage),
        "tables": tables,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print_table("STEP0 self搬 IoU", tables.get("sanity", []))
    print_table("STEP1 boundary owner accuracy / margin", tables.get("boundary_accuracy", []))
    print_table("STEP1 all-voxel owner accuracy / margin", tables.get("all_accuracy", []))
    print_table("STEP3 coverage", tables.get("coverage", []))
    print_table("Lowest-accuracy cases", worst_cases[:10])
    print(f"\n[out] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
