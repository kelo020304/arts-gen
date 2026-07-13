#!/usr/bin/env python3
"""Phase 0/1 audit for part decoder / multi-view SLat flow overfit data.

This script is intentionally read-only except for writing reports under the
requested cache directory. It checks the v5 packed dataset at object granularity
and records blockers before any training or cache materialization is attempted.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_PACKED_DIR = Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5")
DEFAULT_OUT_DIR = Path("/robot/data-lab/jzh/art-gen/data/slat_dec_part_cache")
TRELLIS_ROOT = Path("/root/code/arts-gen/TRELLIS-arts")

FIXED_OBJECTS = [
    ("route_A", "phyx-verse", "74c7791c8ac64c55a08704202b8cbf38"),
    ("route_B", "physx-0511-drawer-door", "22367"),
    ("route_C", "phyx-verse", "0786542d0f7549208f889113fc384a7f"),
    ("route_D", "phyx-verse", "0a46621504c24197b5653608f474f73b"),
]

CANONICAL_ROUTE_PRIORITY = {
    "route_A": 0,
    "route_B": 1,
    "route_C": 2,
    "route_D": 3,
}

ROTX_POS_90 = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
ROTX_NEG_90 = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
IDENTITY = np.eye(3, dtype=np.float64)
AXIS_PERMUTATIONS: list[tuple[str, np.ndarray]] = [
    ("identity", IDENTITY),
    ("x_pos_90", ROTX_POS_90),
    ("x_neg_90", ROTX_NEG_90),
]


@dataclass(frozen=True)
class ObjectKey:
    dataset_id: str
    obj_id: str

    @property
    def label(self) -> str:
        return f"{self.dataset_id}::{self.obj_id}"


@dataclass(frozen=True)
class RawMeshHit:
    path: Path
    raw_root: Path
    layout: str


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def coords_to_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    return {tuple(map(int, row)) for row in arr}


def coords_iou(a: np.ndarray, b: np.ndarray) -> float:
    sa = coords_to_set(a)
    sb = coords_to_set(b)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return float(len(sa & sb) / union)


def coords_bbox(coords: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        return {"min": None, "max": None, "extent": None}
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    return {"min": lo.tolist(), "max": hi.tolist(), "extent": (hi - lo + 1).tolist()}


def infer_interior_heuristic(coords: np.ndarray) -> dict[str, Any]:
    """Cheap interior-signal heuristic on 64-grid component coords.

    v5 rows are sparse component voxels. A true solid component commonly has at
    least some voxels with occupied neighbors on both sides of every axis. Thin
    handles and doors can fail this even when valid, so this is diagnostic only.
    """
    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        return {"interior_like": 0, "interior_like_ratio": 0.0, "bbox_fill_ratio": 0.0}
    occ = coords_to_set(arr)
    interior = 0
    for x, y, z in occ:
        if (
            (x - 1, y, z) in occ
            and (x + 1, y, z) in occ
            and (x, y - 1, z) in occ
            and (x, y + 1, z) in occ
            and (x, y, z - 1) in occ
            and (x, y, z + 1) in occ
        ):
            interior += 1
    bbox = coords_bbox(arr)
    extent = bbox["extent"] or [0, 0, 0]
    bbox_volume = int(extent[0] * extent[1] * extent[2]) if all(v is not None for v in extent) else 0
    return {
        "interior_like": int(interior),
        "interior_like_ratio": float(interior / max(1, len(arr))),
        "bbox_fill_ratio": float(len(arr) / max(1, bbox_volume)),
    }


def build_object_index(entries: list[dict[str, Any]]) -> dict[ObjectKey, list[dict[str, Any]]]:
    out: dict[ObjectKey, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        out[ObjectKey(str(entry["dataset_id"]), str(entry["obj_id"]))].append(entry)
    for rows in out.values():
        rows.sort(key=lambda r: (int(r["angle_idx"]), str(r["part_name"])))
    return dict(out)


def dataset_roots(index: dict[str, Any]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for item in index.get("datasets", []):
        dataset_id = str(item["dataset_id"])
        roots[dataset_id] = Path(str(item["data_root"]))
    return roots


def _existing_unique(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        if path.exists():
            out.append(path)
            seen.add(key)
    return out


def dataset_raw_roots(index: dict[str, Any]) -> dict[str, list[Path]]:
    """Return dataset-specific raw OBJ roots.

    v5 `data_root` is the packed/reconstruction root. Most datasets colocate raw
    mesh sources there, but PhysX 0511 keeps the complete `raw/partseg` tree in
    the original arts-gen-data root while the jzh mirror is nested/incomplete.
    Keep this explicit so Phase 0 does not silently assume a single layout.
    """
    data_roots = dataset_roots(index)
    raw_roots: dict[str, list[Path]] = {}
    for dataset_id, data_root in data_roots.items():
        candidates: list[Path] = [data_root]
        if dataset_id == "physx-0511-drawer-door":
            parent = data_root.parent
            candidates = [
                Path("/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511"),
                data_root,
                Path("/robot/data-lab/arts-gen-data/data/PhysX-Mobility-full-4view-0511"),
                parent,
            ]
            manifest_meta = data_root / "manifests" / "part_completion" / "manifest_meta.json"
            if manifest_meta.is_file():
                try:
                    manifest_root = Path(str(_json_load(manifest_meta).get("data_root", "")))
                    if str(manifest_root):
                        candidates.append(manifest_root)
                except Exception:
                    pass
        raw_roots[dataset_id] = _existing_unique(candidates)
    return raw_roots


def load_rows_for_entries(packed_dir: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_shard[str(entry["shard"])].append(entry)
    rows: list[dict[str, Any]] = []
    for shard, shard_entries in sorted(by_shard.items()):
        shard_path = packed_dir / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"packed shard missing: {shard_path}")
        payload = torch.load(shard_path, map_location="cpu")
        for entry in shard_entries:
            row = dict(payload[int(entry["index"])])
            row["_packed_shard"] = shard
            row["_packed_index"] = int(entry["index"])
            rows.append(row)
    rows.sort(key=lambda r: (int(r["angle_idx"]), str(r["part_name"])))
    return rows


def object_summary_from_index(key: ObjectKey, entries: list[dict[str, Any]]) -> dict[str, Any]:
    angles = sorted({_safe_int(e.get("angle_idx")) for e in entries})
    by_angle: dict[int, list[str]] = defaultdict(list)
    for e in entries:
        by_angle[_safe_int(e["angle_idx"])].append(str(e["part_name"]))
    part_names = sorted({str(e["part_name"]) for e in entries})
    return {
        "dataset_id": key.dataset_id,
        "obj_id": key.obj_id,
        "rows": len(entries),
        "angles": angles,
        "num_angles": len(angles),
        "part_names": part_names,
        "num_target_parts": len(part_names),
        "parts_per_angle": {str(k): len(v) for k, v in sorted(by_angle.items())},
        "raw_count_min": min((_safe_int(e.get("raw_count")) for e in entries), default=0),
        "raw_count_max": max((_safe_int(e.get("raw_count")) for e in entries), default=0),
    }


def select_candidates(
    object_index: dict[ObjectKey, list[dict[str, Any]]],
    *,
    max_extra: int,
) -> list[tuple[str, ObjectKey]]:
    selected: list[tuple[str, ObjectKey]] = []
    selected_keys: set[ObjectKey] = set()
    for tag, dataset_id, obj_id in FIXED_OBJECTS:
        key = ObjectKey(dataset_id, obj_id)
        selected.append((tag, key))
        selected_keys.add(key)

    # Varied part-count candidates, favor phyx/realappliance because raw GT mesh
    # roots are complete there. Keep deterministic ordering by part count bucket.
    pool: list[tuple[int, int, int, ObjectKey]] = []
    for key, rows in object_index.items():
        if key in selected_keys:
            continue
        if key.dataset_id not in {"phyx-verse", "realappliance"}:
            continue
        summary = object_summary_from_index(key, rows)
        if summary["num_angles"] < 1:
            continue
        k_parts = int(summary["num_target_parts"])
        if k_parts < 1:
            continue
        rows_count = int(summary["rows"])
        # Buckets make the chosen list cover small/medium/larger K without a scan
        # over raw meshes before the actual audit.
        bucket = min(6, k_parts)
        pool.append((bucket, k_parts, rows_count, key))
    pool.sort(key=lambda x: (x[0], x[1], -x[2], x[3].dataset_id, x[3].obj_id))

    bucket_counts: Counter[int] = Counter()
    for bucket, _k_parts, _rows, key in pool:
        if len(selected) >= len(FIXED_OBJECTS) + max_extra:
            break
        if bucket_counts[bucket] >= 3:
            continue
        selected.append((f"extra_{len(selected) - len(FIXED_OBJECTS) + 1:02d}", key))
        selected_keys.add(key)
        bucket_counts[bucket] += 1
    return selected


def load_part_info(root: Path, obj_id: str) -> dict[str, Any] | None:
    path = root / "reconstruction" / "part_info" / obj_id / "part_info.json"
    if not path.is_file():
        return None
    data = _json_load(path)
    if not isinstance(data, dict) or not isinstance(data.get("parts"), dict):
        raise ValueError(f"malformed part_info: {path}")
    return data


def raw_mesh_layout_dirs(raw_root: Path, obj_id: str) -> list[tuple[str, Path]]:
    layouts = [
        ("raw/partseg/<obj>/objs", raw_root / "raw" / "partseg" / obj_id / "objs"),
        ("partseg/<obj>/objs", raw_root / "partseg" / obj_id / "objs"),
        ("<obj>/objs", raw_root / obj_id / "objs"),
    ]
    nested = raw_root / raw_root.name
    if nested.is_dir():
        layouts.extend(
            [
                (
                    "<rootname>/raw/partseg/<obj>/objs",
                    nested / "raw" / "partseg" / obj_id / "objs",
                ),
                ("<rootname>/partseg/<obj>/objs", nested / "partseg" / obj_id / "objs"),
                ("<rootname>/<obj>/objs", nested / obj_id / "objs"),
            ]
        )
    return layouts


def raw_mesh_stems_for_part(part_name: str, part_info: dict[str, Any] | None) -> list[str]:
    stems: list[str] = []
    if part_info and part_name in part_info.get("parts", {}):
        meta = part_info["parts"][part_name]
        stems.extend(str(x) for x in meta.get("obj_files", []) if str(x))
        raw_label = meta.get("raw_label")
        if raw_label is not None:
            stems.append(str(raw_label))
    if not stems:
        # Best-effort fallback for legacy data whose v5 part names end in _N.
        m = re.search(r"_(\d+)$", part_name)
        if m:
            stems.append(m.group(1))
    out: list[str] = []
    seen: set[str] = set()
    for stem in stems:
        stem = str(stem)
        if stem and stem not in seen:
            out.append(stem)
            seen.add(stem)
    return out


def raw_mesh_paths_for_part(
    raw_roots: list[Path],
    obj_id: str,
    part_name: str,
    part_info: dict[str, Any] | None,
) -> list[RawMeshHit]:
    stems = raw_mesh_stems_for_part(part_name, part_info)
    hits: list[RawMeshHit] = []
    seen: set[str] = set()
    for raw_root in raw_roots:
        for layout, obj_dir in raw_mesh_layout_dirs(raw_root, obj_id):
            if not obj_dir.is_dir():
                continue
            for stem in stems:
                path = obj_dir / f"{stem}.obj"
                key = str(path)
                if path.is_file() and key not in seen:
                    hits.append(RawMeshHit(path=path, raw_root=raw_root, layout=layout))
                    seen.add(key)
            if hits:
                # A part can reference several stems; once a layout resolves,
                # avoid mixing with duplicate mirrors later in the candidate list.
                return hits
    return hits


def count_obj_faces(path: Path) -> tuple[int, int]:
    vertices = 0
    faces = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    return vertices, faces


def load_obj_vertices(path: Path, *, max_vertices: int = 50000) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.asarray(vertices, dtype=np.float64)
    if len(arr) > max_vertices:
        idx = np.linspace(0, len(arr) - 1, max_vertices).round().astype(np.int64)
        arr = arr[idx]
    return arr


def best_vertex_voxel_iou(vertices: np.ndarray, target_coords: np.ndarray) -> dict[str, Any]:
    """Approximate raw OBJ/v5 alignment by voxelizing raw vertices.

    This is not a replacement for full mesh voxelization, but catches wrong
    label/axis/scale cases cheaply. The report names it as an approximation.
    """
    if vertices.size == 0 or len(target_coords) == 0:
        return {"iou": 0.0, "transform": None, "pred_count": 0}
    target = coords_to_set(target_coords)
    target_arr = np.asarray(target_coords, dtype=np.float64)
    target_center = (target_arr.mean(axis=0) + 0.5) / 64.0 - 0.5
    target_extent = np.maximum(np.ptp(target_arr, axis=0) / 64.0, 1.0 / 64.0)

    best: dict[str, Any] = {"iou": 0.0, "transform": None, "pred_count": 0}
    for rot_name, rot in AXIS_PERMUTATIONS:
        v = vertices @ rot.T
        v_center = v.mean(axis=0)
        v_extent = np.maximum(np.ptp(v, axis=0), 1.0e-8)
        scale_candidates = []
        for m_extent, t_extent in zip(v_extent, target_extent):
            if m_extent > 1.0e-8 and t_extent > 1.0e-8:
                scale_candidates.append(float(t_extent / m_extent))
        if not scale_candidates:
            scale_candidates = [1.0]
        for base_scale in scale_candidates:
            for factor in (0.85, 1.0, 1.15):
                scale = float(base_scale * factor)
                mapped = (v - v_center) * scale + target_center
                idx = np.floor((mapped + 0.5) * 64.0).astype(np.int64)
                valid = np.all((idx >= 0) & (idx < 64), axis=1)
                pred = {tuple(map(int, row)) for row in idx[valid]}
                union = len(pred | target)
                iou = float(len(pred & target) / union) if union else 0.0
                if iou > best["iou"]:
                    best = {
                        "iou": iou,
                        "transform": {"rotation": rot_name, "scale": scale},
                        "pred_count": len(pred),
                    }
    return best


def check_tokens(path: Path) -> dict[str, Any]:
    out = {"exists": path.is_file(), "shape": None, "dtype": None, "ok": False, "error": None}
    if not path.is_file():
        out["error"] = "missing"
        return out
    try:
        with np.load(path, allow_pickle=False) as data:
            if "tokens" not in data:
                raise ValueError(f"missing tokens key; keys={data.files}")
            tokens = data["tokens"]
            out["shape"] = list(tokens.shape)
            out["dtype"] = str(tokens.dtype)
            out["ok"] = bool(tokens.ndim == 3 and tokens.shape[0] >= 4 and tokens.shape[-1] == 1024 and tokens.dtype == np.float32)
            if not out["ok"]:
                out["error"] = "schema_mismatch"
    except Exception as exc:
        out["error"] = str(exc)
    return out


def check_render_paths(root: Path, obj_id: str, angle_idx: int, view_indices: list[int]) -> dict[str, Any]:
    base = root / "renders" / obj_id / f"angle_{angle_idx}"
    rgb = []
    masks = []
    for view in view_indices:
        rgb_path = base / "rgb" / f"view_{int(view)}.png"
        mask_path = base / "mask" / f"mask_{int(view)}.npy"
        rgb.append(rgb_path.is_file())
        masks.append(mask_path.is_file())
    return {
        "base_exists": base.is_dir(),
        "camera_exists": (base / "camera_transforms.json").is_file(),
        "rgb_present": int(sum(rgb)),
        "mask_present": int(sum(masks)),
        "requested_views": [int(v) for v in view_indices],
        "ok": bool(base.is_dir() and all(rgb) and all(masks)),
    }


def check_closed_angle(root: Path, obj_id: str) -> dict[str, Any]:
    path = root / "joint_transforms" / f"{obj_id}.json"
    if not path.is_file():
        return {"status": "unknown", "path": str(path), "reason": "missing_joint_transforms"}
    data = _json_load(path)
    angle0 = (data.get("angles") or {}).get("0")
    if not isinstance(angle0, dict):
        return {"status": "fail", "path": str(path), "reason": "missing_angle_0"}
    states = angle0.get("joint_states")
    if not isinstance(states, dict):
        return {"status": "fail", "path": str(path), "reason": "missing_joint_states"}
    values = [abs(float(v or 0.0)) for v in states.values()]
    max_abs = max(values, default=0.0)
    return {
        "status": "pass" if max_abs <= 1.0e-6 else "fail",
        "path": str(path),
        "angle_0_max_abs_joint_state": max_abs,
        "num_joint_states": len(values),
    }


def surface_completeness(root: Path, obj_id: str, angle_idx: int, part_names: list[str], whole_coords: np.ndarray) -> dict[str, Any]:
    voxel_dir = root / "reconstruction" / "voxel_expanded" / obj_id / f"angle_{angle_idx}" / "64"
    surface_path = voxel_dir / "surface.npy"
    out = {
        "voxel_dir": str(voxel_dir),
        "surface_exists": surface_path.is_file(),
        "surface_count": 0,
        "target_ind_present": 0,
        "target_ind_count_sum": 0,
        "union_surface_iou": 0.0,
        "union_vs_whole_iou": 0.0,
        "ok": False,
    }
    if not surface_path.is_file():
        return out
    surface = np.asarray(np.load(surface_path), dtype=np.int64).reshape(-1, 3)
    out["surface_count"] = int(len(surface))
    union_sets: set[tuple[int, int, int]] = set()
    for part_name in part_names:
        ind_path = voxel_dir / f"ind_{part_name}.npy"
        if not ind_path.is_file():
            continue
        coords = np.asarray(np.load(ind_path), dtype=np.int64).reshape(-1, 3)
        out["target_ind_present"] += 1
        out["target_ind_count_sum"] += int(len(coords))
        union_sets |= coords_to_set(coords)
    surface_set = coords_to_set(surface)
    whole_set = coords_to_set(whole_coords)
    out["union_surface_iou"] = float(len(union_sets & surface_set) / max(1, len(union_sets | surface_set)))
    out["union_vs_whole_iou"] = float(len(union_sets & whole_set) / max(1, len(union_sets | whole_set)))
    out["ok"] = bool(surface_path.is_file() and out["target_ind_present"] == len(part_names))
    return out


def load_component_ind_coords(root: Path, obj_id: str, angle_idx: int, part_name: str) -> tuple[Path, np.ndarray | None]:
    path = root / "reconstruction" / "voxel_expanded" / obj_id / f"angle_{angle_idx}" / "64" / f"ind_{part_name}.npy"
    if not path.is_file():
        return path, None
    arr = np.asarray(np.load(path), dtype=np.int64).reshape(-1, 3)
    return path, arr


def part_info_consistency(part_info: dict[str, Any] | None, part_names: list[str]) -> dict[str, Any]:
    if part_info is None:
        return {"exists": False, "ok": False, "num_parts": 0, "missing_target_parts": part_names}
    parts = part_info.get("parts") or {}
    labels = sorted(int(meta.get("label")) for meta in parts.values() if isinstance(meta, dict) and "label" in meta)
    expected = list(range(1, int(part_info.get("num_parts", len(parts))) + 1))
    missing = [name for name in part_names if name not in parts]
    return {
        "exists": True,
        "ok": bool(not missing and labels == expected),
        "num_parts": int(part_info.get("num_parts", len(parts))),
        "labels_contiguous": labels == expected,
        "missing_target_parts": missing,
    }


def audit_object(
    *,
    tag: str,
    key: ObjectKey,
    entries: list[dict[str, Any]] | None,
    packed_dir: Path,
    roots: dict[str, Path],
    raw_roots_by_dataset: dict[str, list[Path]],
    max_parts_for_mesh_iou: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reasons: list[str] = []
    if not entries:
        return (
            {
                "tag": tag,
                "dataset_id": key.dataset_id,
                "obj_id": key.obj_id,
                "pass": False,
                "fail_reasons": ["missing_from_v5_index"],
            },
            [],
        )
    root = roots.get(key.dataset_id)
    if root is None:
        return (
            {
                "tag": tag,
                "dataset_id": key.dataset_id,
                "obj_id": key.obj_id,
                "pass": False,
                "fail_reasons": ["dataset_root_missing_from_index"],
            },
            [],
        )
    raw_roots = raw_roots_by_dataset.get(key.dataset_id, [])
    if not raw_roots:
        reasons.append("dataset_raw_root_missing")
    index_summary = object_summary_from_index(key, entries)
    rows = load_rows_for_entries(packed_dir, entries)
    part_names = index_summary["part_names"]
    angle0_rows = [r for r in rows if int(r["angle_idx"]) == 0]
    representative_rows = angle0_rows if angle0_rows else rows[: len(part_names)]
    if not representative_rows:
        reasons.append("no_loadable_rows")

    part_info = load_part_info(root, key.obj_id)
    part_info_check = part_info_consistency(part_info, part_names)
    if not part_info_check["ok"]:
        reasons.append("part_info_missing_or_inconsistent")

    closed = check_closed_angle(root, key.obj_id)
    if closed["status"] != "pass":
        reasons.append(f"closed_angle_{closed['status']}")

    sample_rows_by_part: dict[str, dict[str, Any]] = {}
    for row in representative_rows:
        sample_rows_by_part.setdefault(str(row["part_name"]), row)
    if len(sample_rows_by_part) < len(part_names):
        # Some objects have no angle 0 in v5; this still reports but cannot pass
        # the closed-state overfit gate.
        reasons.append("angle0_missing_some_parts")

    whole_coords = None
    if representative_rows:
        whole_coords = _as_numpy(representative_rows[0]["whole_coords"]).astype(np.int64).reshape(-1, 3)
    whole_count = int(len(whole_coords)) if whole_coords is not None else 0
    if whole_count <= 0:
        reasons.append("empty_whole_coords")

    union_parts: set[tuple[int, int, int]] = set()
    part_rows_out: list[dict[str, Any]] = []
    tiny_parts = 0
    all_zero_prompt_parts = 0
    selected_zero_view_parts = 0
    mesh_iou_failures = 0
    mesh_missing_parts = 0
    mesh_face_failures = 0
    component_voxel_iou_failures = 0
    component_voxel_missing_parts = 0
    component_voxel_iou_values: list[float] = []

    for part_name in part_names:
        row = sample_rows_by_part.get(part_name)
        if row is None:
            part_rows_out.append(
                {
                    "tag": tag,
                    "dataset_id": key.dataset_id,
                    "obj_id": key.obj_id,
                    "part_name": part_name,
                    "angle_idx": None,
                    "pass": False,
                    "fail_reason": "missing_representative_row",
                }
            )
            continue
        raw_coords = _as_numpy(row["raw_coords"]).astype(np.int64).reshape(-1, 3)
        union_parts |= coords_to_set(raw_coords)
        m_gt = _as_numpy(row["m_gt"])
        masks2d = _as_numpy(row["masks2d"])
        view_indices = [int(x) for x in _as_numpy(row["view_indices"]).reshape(-1).tolist()]
        mask_px = [int(masks2d[i].sum()) for i in range(masks2d.shape[0])]
        if int(m_gt.sum()) < 10:
            tiny_parts += 1
        if sum(mask_px) == 0:
            all_zero_prompt_parts += 1
        if any(v == 0 for v in mask_px):
            selected_zero_view_parts += 1

        ind_path, ind_coords = load_component_ind_coords(root, key.obj_id, int(row["angle_idx"]), part_name)
        if ind_coords is None:
            component_voxel_missing_parts += 1
            component_voxel_iou = 0.0
            component_voxel_status = "missing"
        else:
            component_voxel_iou = coords_iou(raw_coords, ind_coords)
            component_voxel_iou_values.append(component_voxel_iou)
            if component_voxel_iou < 0.7:
                component_voxel_iou_failures += 1
                component_voxel_status = "low_iou"
            else:
                component_voxel_status = "ok"

        mesh_hits = raw_mesh_paths_for_part(raw_roots, key.obj_id, part_name, part_info)
        mesh_paths = [hit.path for hit in mesh_hits]
        mesh_raw_roots = sorted({str(hit.raw_root) for hit in mesh_hits})
        mesh_layouts = sorted({str(hit.layout) for hit in mesh_hits})
        mesh_vertices = 0
        mesh_faces = 0
        mesh_iou = None
        mesh_status = "ok"
        if not mesh_paths:
            mesh_status = "missing"
            mesh_missing_parts += 1
        else:
            for path in mesh_paths:
                nv, nf = count_obj_faces(path)
                mesh_vertices += nv
                mesh_faces += nf
            if mesh_faces <= 0:
                mesh_status = "zero_faces"
                mesh_face_failures += 1
            elif mesh_iou_failures < max_parts_for_mesh_iou:
                vertices = np.concatenate([load_obj_vertices(path) for path in mesh_paths], axis=0)
                best = best_vertex_voxel_iou(vertices, raw_coords)
                mesh_iou = float(best["iou"])
                if mesh_iou < 0.05:
                    mesh_iou_failures += 1
                    mesh_status = "low_approx_vertex_iou"

        if mesh_status in {"missing", "zero_faces"}:
            pass
        elif mesh_status == "low_approx_vertex_iou":
            # Low approximate vertex IoU is a warning unless all checked parts fail.
            pass

        part_rows_out.append(
            {
                "tag": tag,
                "dataset_id": key.dataset_id,
                "obj_id": key.obj_id,
                "part_name": part_name,
                "angle_idx": int(row["angle_idx"]),
                "semantic_type": str(row.get("semantic_type", "")),
                "original_label": int(row.get("original_label", -1)),
                "raw_count": int(row.get("raw_count", len(raw_coords))),
                "raw_coords_count": int(len(raw_coords)),
                "m_gt_voxels_16": int(m_gt.sum()),
                "tiny_lt10_mgt": bool(int(m_gt.sum()) < 10),
                "view_indices": view_indices,
                "selected_mask_px": mask_px,
                "selected_mask_px_sum": int(sum(mask_px)),
                "all_selected_masks_zero": bool(sum(mask_px) == 0),
                "selected_has_zero_view": bool(any(v == 0 for v in mask_px)),
                "bbox64": coords_bbox(raw_coords),
                **{f"interior_{k}": v for k, v in infer_interior_heuristic(raw_coords).items()},
                "raw_mesh_paths": [str(p) for p in mesh_paths],
                "raw_mesh_roots": mesh_raw_roots,
                "raw_mesh_layouts": mesh_layouts,
                "raw_mesh_vertices": int(mesh_vertices),
                "raw_mesh_faces": int(mesh_faces),
                "raw_mesh_status": mesh_status,
                "approx_vertex_voxel_iou": mesh_iou,
                "component_ind_path": str(ind_path),
                "component_ind_exists": bool(ind_coords is not None),
                "component_ind_count": int(len(ind_coords)) if ind_coords is not None else 0,
                "component_ind_v5_raw_iou": float(component_voxel_iou),
                "component_ind_status": component_voxel_status,
            }
        )

    if component_voxel_missing_parts > 0:
        reasons.append("component_ind_missing")
    if component_voxel_iou_failures > 0:
        reasons.append("component_ind_v5_iou_lt_0_7")
    if mesh_missing_parts > 0:
        reasons.append("missing_raw_gt_mesh")
    if mesh_face_failures > 0:
        reasons.append("raw_gt_mesh_zero_faces")
    if representative_rows and all_zero_prompt_parts == len(part_names):
        reasons.append("all_parts_prompt_masks_zero")

    token_path = root / "reconstruction" / "dinov2_tokens" / key.obj_id / "angle_0" / "tokens.npz"
    token_check = check_tokens(token_path)
    if not token_check["ok"]:
        reasons.append("dinov2_tokens_missing_or_bad")
    render_check = {"ok": False, "reason": "no_representative_row"}
    if representative_rows:
        view_indices = [int(x) for x in _as_numpy(representative_rows[0]["view_indices"]).reshape(-1).tolist()]
        render_check = check_render_paths(root, key.obj_id, int(representative_rows[0]["angle_idx"]), view_indices)
        if not render_check["ok"]:
            reasons.append("renders_or_masks_missing")

    surface_check = {"ok": False, "reason": "no_whole_coords"}
    if whole_coords is not None:
        surface_check = surface_completeness(root, key.obj_id, 0, part_names, whole_coords)
        if not surface_check["ok"]:
            reasons.append("voxel_expanded_surface_or_part_ind_missing")

    whole_set = coords_to_set(whole_coords) if whole_coords is not None else set()
    union_iou_whole = float(len(union_parts & whole_set) / max(1, len(union_parts | whole_set)))
    body_residual = int(max(0, len(whole_set - union_parts)))
    part_outside_whole = int(max(0, len(union_parts - whole_set)))
    if union_iou_whole <= 0:
        reasons.append("component_union_no_overlap_whole")

    # The hard pass condition is strict for training supervision. Approximate
    # vertex IoU is diagnostic and not included because v5 stores voxelized GT
    # directly; full mesh voxelization will be part of Phase 2 cache build.
    pass_gate = not any(
        reason
        for reason in reasons
        if reason
        in {
            "missing_from_v5_index",
            "dataset_root_missing_from_index",
            "no_loadable_rows",
            "part_info_missing_or_inconsistent",
            "closed_angle_fail",
            "closed_angle_unknown",
            "angle0_missing_some_parts",
            "empty_whole_coords",
            "dataset_raw_root_missing",
            "missing_raw_gt_mesh",
            "raw_gt_mesh_zero_faces",
            "component_ind_missing",
            "component_ind_v5_iou_lt_0_7",
            "all_parts_prompt_masks_zero",
            "dinov2_tokens_missing_or_bad",
            "renders_or_masks_missing",
            "voxel_expanded_surface_or_part_ind_missing",
            "component_union_no_overlap_whole",
        }
    )
    # Normalize unknown closed-angle reason key.
    reasons = ["closed_angle_unknown" if r == "closed_angle_unknown" else r for r in reasons]
    if closed["status"] == "unknown":
        pass_gate = False

    object_out = {
        "tag": tag,
        "dataset_id": key.dataset_id,
        "obj_id": key.obj_id,
        "data_root": str(root),
        "raw_root_candidates": [str(path) for path in raw_roots],
        "raw_roots_used": sorted({root for row in part_rows_out for root in row.get("raw_mesh_roots", [])}),
        "raw_layouts_used": sorted({layout for row in part_rows_out for layout in row.get("raw_mesh_layouts", [])}),
        "pass": bool(pass_gate),
        "fail_reasons": sorted(set(reasons)),
        **index_summary,
        "whole_coords_count": whole_count,
        "component_union_vs_whole_iou": union_iou_whole,
        "body_residual_voxels": body_residual,
        "part_voxels_outside_whole": part_outside_whole,
        "part_info": part_info_check,
        "closed_angle": closed,
        "tokens": token_check,
        "renders": render_check,
        "voxel_expanded": surface_check,
        "voxel_surface_count": int(surface_check.get("surface_count", 0)) if isinstance(surface_check, dict) else 0,
        "tiny_parts_mgt_lt10": int(tiny_parts),
        "all_selected_masks_zero_parts": int(all_zero_prompt_parts),
        "selected_zero_view_parts": int(selected_zero_view_parts),
        "mesh_missing_parts": int(mesh_missing_parts),
        "mesh_zero_face_parts": int(mesh_face_failures),
        "mesh_low_approx_iou_parts": int(mesh_iou_failures),
        "component_ind_missing_parts": int(component_voxel_missing_parts),
        "component_ind_v5_iou_fail_parts": int(component_voxel_iou_failures),
        "component_ind_v5_iou_min": float(min(component_voxel_iou_values)) if component_voxel_iou_values else 0.0,
        "component_ind_v5_iou_mean": float(sum(component_voxel_iou_values) / len(component_voxel_iou_values)) if component_voxel_iou_values else 0.0,
    }
    return object_out, part_rows_out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(title for title, _key in columns) + " |")
    lines.append("| " + " | ".join("---" for _title, _key in columns) + " |")
    for row in rows:
        vals = []
        for _title, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            elif isinstance(value, (list, dict)):
                vals.append(str(value).replace("\n", " ")[:160])
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_phase0_report(
    path: Path,
    *,
    object_rows: list[dict[str, Any]],
    selected_pass: list[dict[str, Any]],
    part_rows: list[dict[str, Any]],
    packed_dir: Path,
    elapsed: float,
) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    pass_count = sum(1 for r in object_rows if r.get("pass"))
    fixed_rows = [r for r in object_rows if str(r.get("tag", "")).startswith("route_")]
    fail_rows = [r for r in object_rows if not r.get("pass")]
    fail_counter: Counter[str] = Counter()
    for row in fail_rows:
        fail_counter.update(row.get("fail_reasons", []))

    lines = [
        "# Phase 0 v5 Data Check",
        "",
        f"- generated: {now}",
        f"- packed_dir: `{packed_dir}`",
        f"- audited_objects: {len(object_rows)}",
        f"- pass_objects: {pass_count}",
        f"- selected_training_pass_objects: {len(selected_pass)}",
        f"- elapsed_seconds: {elapsed:.2f}",
        "",
        "## Hard Gate",
        "",
    ]
    if len(selected_pass) >= 8:
        lines.append("PASS: at least 8 objects satisfy the strict supervision gate. Training still must not start until this report is reviewed.")
    else:
        lines.append("FAIL: fewer than 8 objects satisfy the strict supervision gate. Do not prepare caches or start training.")
    lines.extend(
        [
            "",
            "Strict gate requires v5 rows, angle_0 closed state, part_info consistency, non-empty whole/component voxels, raw per-part OBJ faces, render masks/RGB, DINOv2 tokens, voxel_expanded surface/ind files, and per-component `ind_<part>.npy` versus v5 `raw_coords` IoU >= 0.7.",
            "",
            "Approximate raw OBJ vertex-to-v5 voxel IoU is reported as a diagnostic only. The hard consistency check uses already materialized component voxel supervision (`ind_<part>.npy`) against v5 `raw_coords`.",
            "",
            "## Fixed Route Objects",
            "",
            markdown_table(
                fixed_rows,
                [
                    ("tag", "tag"),
                    ("dataset", "dataset_id"),
                    ("obj", "obj_id"),
                    ("pass", "pass"),
                    ("parts", "num_target_parts"),
                    ("angles", "num_angles"),
                    ("whole", "whole_coords_count"),
                    ("union_iou", "component_union_vs_whole_iou"),
                    ("ind_iou_min", "component_ind_v5_iou_min"),
                    ("raw_roots", "raw_roots_used"),
                    ("tiny", "tiny_parts_mgt_lt10"),
                    ("zero_masks", "all_selected_masks_zero_parts"),
                    ("missing_mesh", "mesh_missing_parts"),
                    ("fail_reasons", "fail_reasons"),
                ],
            ),
            "",
            "## Selected Pass Objects",
            "",
        ]
    )
    if selected_pass:
        lines.append(
            markdown_table(
                selected_pass,
                [
                    ("tag", "tag"),
                    ("dataset", "dataset_id"),
                    ("obj", "obj_id"),
                    ("parts", "num_target_parts"),
                    ("angles", "num_angles"),
                    ("whole", "whole_coords_count"),
                    ("body_residual", "body_residual_voxels"),
                    ("ind_iou_min", "component_ind_v5_iou_min"),
                    ("raw_roots", "raw_roots_used"),
                    ("tiny", "tiny_parts_mgt_lt10"),
                    ("surface", "voxel_surface_count"),
                ],
            )
        )
    else:
        lines.append("(none)")
    lines.extend(["", "## Fail Reason Counts", ""])
    if fail_counter:
        for reason, count in fail_counter.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## All Audited Objects",
            "",
            markdown_table(
                object_rows,
                [
                    ("tag", "tag"),
                    ("dataset", "dataset_id"),
                    ("obj", "obj_id"),
                    ("pass", "pass"),
                    ("parts", "num_target_parts"),
                    ("angles", "num_angles"),
                    ("whole", "whole_coords_count"),
                    ("union_iou", "component_union_vs_whole_iou"),
                    ("ind_iou_min", "component_ind_v5_iou_min"),
                    ("raw_roots", "raw_roots_used"),
                    ("body", "body_residual_voxels"),
                    ("tiny", "tiny_parts_mgt_lt10"),
                    ("fail_reasons", "fail_reasons"),
                ],
            ),
            "",
            "## Known Pit Scan",
            "",
            f"- mask_px all selected views zero parts: {sum(int(r.get('all_selected_masks_zero_parts', 0)) for r in object_rows)}",
            f"- selected-view has at least one zero mask parts: {sum(int(r.get('selected_zero_view_parts', 0)) for r in object_rows)}",
            f"- tiny m_gt<10 components: {sum(int(r.get('tiny_parts_mgt_lt10', 0)) for r in object_rows)}",
            "",
            "## Output Files",
            "",
            f"- object table: `{path.with_name('phase0_object_table.csv')}`",
            f"- part table: `{path.with_name('phase0_part_table.csv')}`",
            f"- machine JSON: `{path.with_name('phase0_data_check.json')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_ref(path: str, needle: str) -> str:
    file_path = Path(path)
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return f"{path}:missing"
    for idx, line in enumerate(lines, start=1):
        if needle in line:
            return f"{path}:{idx}"
    return f"{path}:not-found"


def dependency_status() -> dict[str, Any]:
    modules = ["torch", "trimesh", "numpy", "nvdiffrast", "kaolin"]
    out = {}
    for name in modules:
        try:
            spec = importlib.util.find_spec(name)
            out[name] = {"available": spec is not None, "origin": spec.origin if spec else None}
        except Exception as exc:
            out[name] = {"available": False, "error": str(exc)}
    return out


def write_phase1_report(path: Path, *, selected_pass: list[dict[str, Any]], object_rows: list[dict[str, Any]]) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    deps = dependency_status()
    refs = {
        "mesh_decoder_model": source_ref(
            "TRELLIS-arts/trellis/models/structured_latent_vae/decoder_mesh.py",
            "class SLatMeshDecoder",
        ),
        "mesh_decoder_trainer": source_ref(
            "TRELLIS-arts/trellis/trainers/vae/structured_latent_vae_mesh_dec.py",
            "class SLatVaeMeshDecoderTrainer",
        ),
        "mesh_decoder_training_losses": source_ref(
            "TRELLIS-arts/trellis/trainers/vae/structured_latent_vae_mesh_dec.py",
            "def training_losses",
        ),
        "slat_flow_dataset": source_ref(
            "TRELLIS-arts/trellis/datasets/arts/slat_flow_art.py",
            "class MvImageConditionedSparseLatentDataset",
        ),
        "slat_flow_select_views": source_ref(
            "TRELLIS-arts/trellis/datasets/arts/slat_flow_art.py",
            "def _select_views",
        ),
        "slat_flow_trainer": source_ref(
            "TRELLIS-arts/trellis/trainers/arts/slat_flow_art.py",
            "class Stage4Trainer",
        ),
        "slat_flow_get_cond": source_ref(
            "TRELLIS-arts/trellis/trainers/arts/slat_flow_art.py",
            "def get_cond",
        ),
        "slat_flow_snapshot_cfg": source_ref(
            "TRELLIS-arts/trellis/trainers/arts/slat_flow_art.py",
            "cfg_strength=3.0",
        ),
    }
    pass_ids = [f"{r['dataset_id']}::{r['obj_id']}" for r in selected_pass]
    physx_22367 = next((r for r in object_rows if r.get("obj_id") == "22367"), None)
    lines = [
        "# Phase 1 Inventory",
        "",
        f"- generated: {now}",
        f"- strict pass objects available from Phase 0: {len(selected_pass)}",
        f"- pass object ids: {', '.join(pass_ids) if pass_ids else '(none)'}",
        "",
        "## Track 1: Mask-Conditioned Mesh Decoder",
        "",
        "- Base model location: `TRELLIS-arts/trellis/models/structured_latent_vae/decoder_mesh.py`.",
        f"- `SLatMeshDecoder` reference: `{refs['mesh_decoder_model']}`.",
        "- Safe change shape: add a new variant class under `TRELLIS-arts/trellis/models/` that accepts whole SLat plus one zero-initialized mask channel; do not edit the original class.",
        f"- Native trainer: `{refs['mesh_decoder_trainer']}`.",
        f"- Native loss entry: `{refs['mesh_decoder_training_losses']}`; it renders GT mesh and predicted mesh and uses mask/depth/normal/TSDF/color terms.",
        "- GT supervision state: v5 contains dense `z_global`, `latent_gt`, 16^3 masks, 64-grid raw component coords, and whole coords. Raw per-part OBJ roots are resolved by dataset-id and audited in Phase 0.",
        "- Rendering supervision gap: object-level renders and DINO tokens exist; per-part mesh decoder render packs for component GT are not precomputed. Phase 2 must offline render component GT views, including drawer interior views, before Track1 training.",
        "- Dependency state: nvdiffrast is required by TRELLIS `MeshRenderer`; current import check is recorded below.",
        "",
        "## Track 2: 4-View SLat Flow + Per-View Embedding",
        "",
        f"- Existing dataset reference: `{refs['slat_flow_dataset']}`.",
        f"- Current view dropout selection/padding reference: `{refs['slat_flow_select_views']}`. It pads dropped views with zero token blocks after random selection; Track2 should preserve view slot identity with an explicit keep mask or zeroed original slots before flattening.",
        f"- Existing trainer reference: `{refs['slat_flow_trainer']}`.",
        f"- Training CFG conditioning reference: `{refs['slat_flow_get_cond']}`; it supplies zero negative condition.",
        f"- Snapshot CFG reference: `{refs['slat_flow_snapshot_cfg']}`; inference path already calls sampler with CFG strength in snapshots, but the new entry should expose/record train condition dropout and eval CFG explicitly.",
        "- Target SLat source: prefer `part_synthesis_slat/<prefix>/<obj>_angle_<n>/overall/latent.npz` where present; v5 packed rows also contain dense `latent_gt`/`z_global` per row and can seed cache validation.",
        "- 4-view tokens: v5 rows store selected `view_indices`; DINOv2 tokens are under `reconstruction/dinov2_tokens/<obj>/angle_<n>/tokens.npz` and are [V,T,1024] float32.",
        "",
        "## GT Encode / Cache Plan",
        "",
        "- Shared Phase 2 cache should write one object-angle record per strict-pass object: whole SLat, four view tokens, component masks including body residual, raw GT mesh paths, and render supervision paths.",
        "- Do not cache failed objects. The fixed route object `22367` can remain an evaluation-only diagnostic until a raw GT mesh source is found.",
        "- Mask degradation pairs should be generated from component coords: GT mask, eroded mask, and front-only thin-layer masks for drawer/door semantics.",
        "",
        "## Cost Estimate",
        "",
        "- Track1 single-card smoke: 1 GPU, 1-2 objects, batch 1-2 components, 200-500 steps, expected memory dominated by FlexiCubes render loss.",
        "- Track1 4-card overfit: 8-12 objects, batch about 1 component/GPU initially, 2k-5k steps; increase only after renderer memory is measured.",
        "- Track2 single-card smoke: 1 GPU, 1-2 objects, batch 1 sparse SLat, 200-500 steps with pretrained decoder eval snapshots disabled or sparse.",
        "- Track2 4-card overfit: 8-12 objects x closed angles, batch 1-2/GPU, 2k-5k steps; NUM_WORKERS <= 8 per user constraint.",
        "- Full Track2 training should wait for queue window; prepare config only after overfit gate passes.",
        "",
        "## Dependency Check",
        "",
    ]
    for name, status in deps.items():
        lines.append(f"- {name}: {status}")
    lines.extend(["", "## 22367 Status", ""])
    if physx_22367:
        lines.append(f"- pass: {physx_22367.get('pass')}")
        lines.append(f"- fail_reasons: {physx_22367.get('fail_reasons')}")
        lines.append(f"- data_root: `{physx_22367.get('data_root')}`")
        lines.append(f"- raw_root_candidates: `{physx_22367.get('raw_root_candidates')}`")
        lines.append(f"- raw_roots_used: `{physx_22367.get('raw_roots_used')}`")
    else:
        lines.append("- not audited")
    lines.extend(
        [
            "",
            "## Pre-Training Gate",
            "",
            "No training has been launched by this script. Phase 2 cache preparation and both training tracks require report review first.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(path.with_suffix(".json"), {"generated": now, "refs": refs, "dependencies": deps, "pass_ids": pass_ids})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-extra", type=int, default=16)
    parser.add_argument("--max-parts-for-mesh-iou", type=int, default=8)
    args = parser.parse_args()

    start = time.perf_counter()
    packed_dir = args.packed_dir.resolve()
    index_path = packed_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"v5 index not found: {index_path}")
    index = _json_load(index_path)
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{index_path}: expected top-level entries list")
    roots = dataset_roots(index)
    raw_roots_by_dataset = dataset_raw_roots(index)
    object_index = build_object_index(entries)
    candidates = select_candidates(object_index, max_extra=int(args.max_extra))

    object_rows: list[dict[str, Any]] = []
    part_rows: list[dict[str, Any]] = []
    for tag, key in candidates:
        obj_out, obj_part_rows = audit_object(
            tag=tag,
            key=key,
            entries=object_index.get(key),
            packed_dir=packed_dir,
            roots=roots,
            raw_roots_by_dataset=raw_roots_by_dataset,
            max_parts_for_mesh_iou=int(args.max_parts_for_mesh_iou),
        )
        object_rows.append(obj_out)
        part_rows.extend(obj_part_rows)

    # Keep only strict-pass objects for Phase 2. Prefer route objects if pass,
    # then deterministic extras. Cap at 12 to match requested overfit set size.
    pass_rows = [row for row in object_rows if row.get("pass")]
    pass_rows.sort(
        key=lambda r: (
            CANONICAL_ROUTE_PRIORITY.get(str(r.get("tag")), 100),
            int(r.get("num_target_parts", 0)),
            str(r.get("dataset_id")),
            str(r.get("obj_id")),
        )
    )
    selected_pass = pass_rows[:12]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - start
    payload = {
        "packed_dir": str(packed_dir),
        "elapsed_seconds": elapsed,
        "audited_objects": object_rows,
        "parts": part_rows,
        "selected_pass_objects": selected_pass,
        "phase0_gate_pass": len(selected_pass) >= 8,
    }
    _write_json(out_dir / "phase0_data_check.json", payload)
    object_csv_rows: list[dict[str, Any]] = []
    for row in object_rows:
        flat = dict(row)
        flat["fail_reasons"] = ",".join(row.get("fail_reasons", []))
        flat["raw_root_candidates"] = "|".join(row.get("raw_root_candidates", []))
        flat["raw_roots_used"] = "|".join(row.get("raw_roots_used", []))
        flat["raw_layouts_used"] = "|".join(row.get("raw_layouts_used", []))
        flat["voxel_surface_count"] = (row.get("voxel_expanded") or {}).get("surface_count", 0)
        object_csv_rows.append(flat)
    write_csv(
        out_dir / "phase0_object_table.csv",
        object_csv_rows,
        [
            "tag",
            "dataset_id",
            "obj_id",
            "pass",
            "fail_reasons",
            "data_root",
            "raw_root_candidates",
            "raw_roots_used",
            "raw_layouts_used",
            "rows",
            "num_angles",
            "num_target_parts",
            "whole_coords_count",
            "component_union_vs_whole_iou",
            "component_ind_v5_iou_min",
            "component_ind_v5_iou_mean",
            "component_ind_v5_iou_fail_parts",
            "component_ind_missing_parts",
            "body_residual_voxels",
            "part_voxels_outside_whole",
            "tiny_parts_mgt_lt10",
            "all_selected_masks_zero_parts",
            "selected_zero_view_parts",
            "mesh_missing_parts",
            "mesh_zero_face_parts",
            "mesh_low_approx_iou_parts",
            "voxel_surface_count",
        ],
    )
    write_csv(
        out_dir / "phase0_part_table.csv",
        part_rows,
        [
            "tag",
            "dataset_id",
            "obj_id",
            "part_name",
            "angle_idx",
            "semantic_type",
            "original_label",
            "raw_count",
            "raw_coords_count",
            "m_gt_voxels_16",
            "tiny_lt10_mgt",
            "view_indices",
            "selected_mask_px",
            "selected_mask_px_sum",
            "all_selected_masks_zero",
            "selected_has_zero_view",
            "raw_mesh_vertices",
            "raw_mesh_faces",
            "raw_mesh_status",
            "raw_mesh_roots",
            "raw_mesh_layouts",
            "approx_vertex_voxel_iou",
            "component_ind_exists",
            "component_ind_count",
            "component_ind_v5_raw_iou",
            "component_ind_status",
            "component_ind_path",
        ],
    )
    write_phase0_report(
        out_dir / "data_check_report.md",
        object_rows=object_rows,
        selected_pass=selected_pass,
        part_rows=part_rows,
        packed_dir=packed_dir,
        elapsed=elapsed,
    )
    write_phase1_report(
        out_dir / "phase1_inventory_report.md",
        selected_pass=selected_pass,
        object_rows=object_rows,
    )
    print(f"[phase0] audited={len(object_rows)} pass={len(pass_rows)} selected={len(selected_pass)} gate={len(selected_pass) >= 8}")
    print(f"[phase0] report={out_dir / 'data_check_report.md'}")
    print(f"[phase1] report={out_dir / 'phase1_inventory_report.md'}")


if __name__ == "__main__":
    main()
