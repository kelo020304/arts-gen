#!/usr/bin/env python3
"""Read-only v5 packed-data joint-axis gate.

This script audits whether joint axes parsed by scripts/tools/joint_head.py align
with the packed v5 part geometry. It does not start training or modify data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.lib.joint_head import parse_joint_part  # noqa: E402


DEFAULT_PACKED = Path("/mnt/robot-data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5")
DEFAULT_OUT = Path("/mnt/robot-data-lab/jzh/art-gen/debug/v5_axis_gate")


@dataclass(frozen=True)
class GateRow:
    entry: dict[str, Any]
    data_root: Path
    joint_type: str
    axis: tuple[float, float, float]
    pivot: tuple[float, float, float]
    limits: tuple[float, float]
    parent_group: str | None
    joint_group_id: str
    ignore_reason: str


def fix_path(text: str) -> Path:
    if text.startswith("/robot/data-lab/"):
        text = "/mnt/robot-data-lab/" + text[len("/robot/data-lab/") :]
    return Path(text)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} expected JSON object")
    return value


def load_part_info(root: Path, obj_id: str) -> dict[str, Any]:
    path = root / "reconstruction" / "part_info" / str(obj_id) / "part_info.json"
    return load_json(path)


def load_camera_frame(root: Path, obj_id: str, angle_idx: int) -> tuple[float, np.ndarray]:
    path = root / "renders" / str(obj_id) / f"angle_{int(angle_idx)}" / "camera_transforms.json"
    data = load_json(path)
    return float(data["scale"]), np.asarray(data["offset"], dtype=np.float64)


def part_key(dataset_id: str, obj_id: str, angle_idx: int, part_name: str) -> str:
    return f"{dataset_id}::{obj_id}|{int(angle_idx)}|{part_name}"


def source_label(dataset_id: str) -> str:
    if dataset_id == "realappliance":
        return "RA"
    if dataset_id == "physx-0511-drawer-door":
        return "0511"
    if dataset_id == "phyx-verse":
        return "verse"
    return dataset_id


def raw_to_grid(point: np.ndarray, scale: float, offset: np.ndarray) -> np.ndarray:
    return (point * float(scale) + offset + 0.5) * 63.0


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-8:
        return v * 0.0
    return v / n


def load_item(packed_dir: Path, cache: dict[str, list[dict[str, Any]]], entry: dict[str, Any]) -> dict[str, Any]:
    shard = str(entry["shard"])
    if shard not in cache:
        payload = torch.load(packed_dir / shard, map_location="cpu", weights_only=False)
        if not isinstance(payload, list):
            raise TypeError(f"{packed_dir / shard} expected list")
        cache[shard] = payload
    return cache[shard][int(entry["index"])]


def coords_np(item: dict[str, Any], key: str) -> np.ndarray:
    value = item.get(key)
    if value is None:
        return np.empty((0, 3), dtype=np.float32)
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    arr = arr.reshape(-1, 3).astype(np.float32, copy=False)
    return arr


def sample_points(points: np.ndarray, max_points: int, rng: random.Random) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = rng.sample(range(len(points)), int(max_points))
    return points[np.asarray(idx, dtype=np.int64)]


def plot_projection(
    ax: Any,
    *,
    dims: tuple[int, int],
    title: str,
    whole: np.ndarray,
    child: np.ndarray,
    parent: np.ndarray,
    pivot_grid: np.ndarray,
    axis_grid: np.ndarray,
    limit: float,
) -> None:
    a, b = dims
    if len(whole):
        ax.scatter(whole[:, a], whole[:, b], s=1, c="#b8b8b8", alpha=0.16, linewidths=0)
    if len(parent):
        ax.scatter(parent[:, a], parent[:, b], s=5, c="#f28e2b", alpha=0.45, linewidths=0)
    if len(child):
        ax.scatter(child[:, a], child[:, b], s=6, c="#4e79a7", alpha=0.72, linewidths=0)
    p = pivot_grid
    v = normalize(axis_grid)
    start = p - v * float(limit)
    end = p + v * float(limit)
    ax.plot([start[a], end[a]], [start[b], end[b]], color="#d62728", linewidth=2.0)
    ax.scatter([p[a]], [p[b]], s=42, marker="x", c="#d62728", linewidths=2.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-2, 65)
    ax.set_ylim(-2, 65)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.tick_params(labelsize=7)


def render_gate_image(
    *,
    packed_dir: Path,
    out_path: Path,
    row: GateRow,
    entries_by_key: dict[str, dict[str, Any]],
    part_infos: dict[str, Any],
    cache: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> dict[str, Any]:
    entry = row.entry
    item = load_item(packed_dir, cache, entry)
    child = coords_np(item, "raw_coords")
    whole = coords_np(item, "whole_coords")
    scale, offset = load_camera_frame(row.data_root, str(entry["obj_id"]), int(entry["angle_idx"]))

    parent_coords: list[np.ndarray] = []
    if row.parent_group is not None:
        for name, info in sorted(part_infos.get("parts", {}).items()):
            if str(info.get("joint_group_id", "")) != str(row.parent_group):
                continue
            key = part_key(str(entry["dataset_id"]), str(entry["obj_id"]), int(entry["angle_idx"]), str(name))
            parent_entry = entries_by_key.get(key)
            if parent_entry is not None:
                parent_coords.append(coords_np(load_item(packed_dir, cache, parent_entry), "raw_coords"))
    parent = np.concatenate(parent_coords, axis=0) if parent_coords else np.empty((0, 3), dtype=np.float32)

    pivot_raw = np.asarray(row.pivot, dtype=np.float64)
    axis = normalize(np.asarray(row.axis, dtype=np.float64))
    pivot_grid = raw_to_grid(pivot_raw, scale, offset)
    axis_grid = axis
    child_center = child.mean(axis=0) if len(child) else np.asarray([math.nan, math.nan, math.nan])
    parent_center = parent.mean(axis=0) if len(parent) else np.asarray([math.nan, math.nan, math.nan])
    child_extent = child.max(axis=0) - child.min(axis=0) if len(child) else np.zeros(3)
    axis_len = max(10.0, min(30.0, float(np.linalg.norm(child_extent)) * 0.75))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    title = (
        f"{source_label(str(entry['dataset_id']))} {entry['obj_id']} angle={entry['angle_idx']} "
        f"{entry['part_name']} type={row.joint_type}\\n"
        f"axis=({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}) "
        f"pivot_grid=({pivot_grid[0]:.1f},{pivot_grid[1]:.1f},{pivot_grid[2]:.1f})"
    )
    fig.suptitle(title, fontsize=10)
    whole_s = sample_points(whole, 6000, rng)
    child_s = sample_points(child, 2500, rng)
    parent_s = sample_points(parent, 2500, rng)
    plot_projection(axes[0], dims=(0, 1), title="XY", whole=whole_s, child=child_s, parent=parent_s, pivot_grid=pivot_grid, axis_grid=axis_grid, limit=axis_len)
    plot_projection(axes[1], dims=(0, 2), title="XZ", whole=whole_s, child=child_s, parent=parent_s, pivot_grid=pivot_grid, axis_grid=axis_grid, limit=axis_len)
    plot_projection(axes[2], dims=(1, 2), title="YZ", whole=whole_s, child=child_s, parent=parent_s, pivot_grid=pivot_grid, axis_grid=axis_grid, limit=axis_len)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    dist_child = float(np.linalg.norm(pivot_grid - child_center)) if len(child) else math.nan
    dist_parent = float(np.linalg.norm(pivot_grid - parent_center)) if len(parent) else math.nan
    in_grid = bool(np.all(pivot_grid >= -1.0) and np.all(pivot_grid <= 64.0))
    return {
        "image": str(out_path),
        "dataset_id": entry["dataset_id"],
        "source": source_label(str(entry["dataset_id"])),
        "obj_id": entry["obj_id"],
        "angle_idx": int(entry["angle_idx"]),
        "part_name": entry["part_name"],
        "joint_type": row.joint_type,
        "axis_x": float(axis[0]),
        "axis_y": float(axis[1]),
        "axis_z": float(axis[2]),
        "pivot_grid_x": float(pivot_grid[0]),
        "pivot_grid_y": float(pivot_grid[1]),
        "pivot_grid_z": float(pivot_grid[2]),
        "pivot_in_64_grid": int(in_grid),
        "pivot_to_child_center_grid": dist_child,
        "pivot_to_parent_center_grid": dist_parent,
        "child_voxels": int(len(child)),
        "parent_voxels": int(len(parent)),
        "whole_voxels": int(len(whole)),
        "limits_lo": float(row.limits[0]),
        "limits_hi": float(row.limits[1]),
        "ignore_reason": row.ignore_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples-per-source", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()

    packed_dir = Path(args.packed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))

    index = load_json(packed_dir / "index.json")
    roots_by_dataset = {str(item["dataset_id"]): fix_path(str(item["data_root"])) for item in index.get("datasets", [])}
    entries = [dict(e) for e in index.get("entries", [])]
    entries_by_key = {str(e["key"]): e for e in entries}
    entries_by_obj_angle_part = {
        (str(e["dataset_id"]), str(e["obj_id"]), int(e["angle_idx"]), str(e["part_name"])): e for e in entries
    }

    object_angles = {(str(e["dataset_id"]), str(e["obj_id"]), int(e["angle_idx"])) for e in entries}
    objects = {(str(e["dataset_id"]), str(e["obj_id"])) for e in entries}
    entry_counts = Counter(str(e["dataset_id"]) for e in entries)

    part_info_cache: dict[tuple[str, str], dict[str, Any]] = {}
    valid_rows: list[GateRow] = []
    ignored = Counter()
    missing_part_info = Counter()
    missing_part_name = Counter()
    type_counts = Counter()
    source_valid_counts = Counter()
    source_object_with_valid: dict[str, set[str]] = defaultdict(set)
    source_angle_with_valid: dict[str, set[tuple[str, int]]] = defaultdict(set)

    for entry in entries:
        dataset_id = str(entry["dataset_id"])
        root = roots_by_dataset[dataset_id]
        obj_id = str(entry["obj_id"])
        cache_key = (dataset_id, obj_id)
        try:
            if cache_key not in part_info_cache:
                part_info_cache[cache_key] = load_part_info(root, obj_id)
            part_infos = part_info_cache[cache_key]
        except FileNotFoundError:
            missing_part_info[dataset_id] += 1
            continue
        info = (part_infos.get("parts") or {}).get(str(entry["part_name"]))
        if not isinstance(info, dict):
            missing_part_name[dataset_id] += 1
            continue
        meta = parse_joint_part(
            str(entry["part_name"]),
            info,
            object_id=obj_id,
            dataset_id=dataset_id,
            data_root=root,
        )
        if meta.joint_type not in ("B", "C"):
            continue
        if meta.ignored:
            ignored[(dataset_id, meta.ignore_reason)] += 1
            continue
        valid_rows.append(
            GateRow(
                entry=entry,
                data_root=root,
                joint_type=meta.joint_type,
                axis=meta.axis,
                pivot=meta.pivot,
                limits=meta.limits,
                parent_group=meta.parent_group,
                joint_group_id=meta.joint_group_id,
                ignore_reason=meta.ignore_reason,
            )
        )
        type_counts[(dataset_id, meta.joint_type)] += 1
        source_valid_counts[dataset_id] += 1
        source_object_with_valid[dataset_id].add(obj_id)
        source_angle_with_valid[dataset_id].add((obj_id, int(entry["angle_idx"])))

    stats = {
        "packed_dir": str(packed_dir),
        "entries_total": len(entries),
        "unique_objects_total": len(objects),
        "object_angle_pairs_total": len(object_angles),
        "entries_by_source": dict(sorted(entry_counts.items())),
        "valid_bc_parts_total": len(valid_rows),
        "valid_bc_parts_by_source": dict(sorted(source_valid_counts.items())),
        "valid_bc_parts_by_source_type": {f"{k[0]}:{k[1]}": int(v) for k, v in sorted(type_counts.items())},
        "unique_objects_with_valid_bc_by_source": {k: len(v) for k, v in sorted(source_object_with_valid.items())},
        "object_angle_pairs_with_valid_bc_by_source": {k: len(v) for k, v in sorted(source_angle_with_valid.items())},
        "missing_part_info_rows_by_source": dict(sorted(missing_part_info.items())),
        "missing_part_name_rows_by_source": dict(sorted(missing_part_name.items())),
        "ignored_bc_rows_by_source_reason": {f"{k[0]}:{k[1]}": int(v) for k, v in sorted(ignored.items())},
        "datasets": {k: str(v) for k, v in sorted(roots_by_dataset.items())},
    }
    (out_dir / "v5_axis_gate_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    with (out_dir / "v5_axis_gate_valid_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset_id",
            "source",
            "obj_id",
            "angle_idx",
            "part_name",
            "joint_type",
            "axis_x",
            "axis_y",
            "axis_z",
            "pivot_x",
            "pivot_y",
            "pivot_z",
            "limits_lo",
            "limits_hi",
            "raw_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in valid_rows:
            e = row.entry
            writer.writerow(
                {
                    "dataset_id": e["dataset_id"],
                    "source": source_label(str(e["dataset_id"])),
                    "obj_id": e["obj_id"],
                    "angle_idx": int(e["angle_idx"]),
                    "part_name": e["part_name"],
                    "joint_type": row.joint_type,
                    "axis_x": row.axis[0],
                    "axis_y": row.axis[1],
                    "axis_z": row.axis[2],
                    "pivot_x": row.pivot[0],
                    "pivot_y": row.pivot[1],
                    "pivot_z": row.pivot[2],
                    "limits_lo": row.limits[0],
                    "limits_hi": row.limits[1],
                    "raw_count": int(e.get("raw_count", 0)),
                }
            )

    by_source_obj: dict[str, dict[str, list[GateRow]]] = defaultdict(lambda: defaultdict(list))
    for row in valid_rows:
        e = row.entry
        by_source_obj[str(e["dataset_id"])][str(e["obj_id"])].append(row)

    selected: list[GateRow] = []
    for dataset_id in sorted(by_source_obj):
        obj_ids = sorted(by_source_obj[dataset_id])
        rng.shuffle(obj_ids)
        for obj_id in obj_ids[: int(args.samples_per_source)]:
            candidates = by_source_obj[dataset_id][obj_id]
            candidates.sort(key=lambda r: (-int(r.entry.get("raw_count", 0)), str(r.entry["part_name"])))
            selected.append(candidates[0])

    cache: dict[str, list[dict[str, Any]]] = {}
    rendered: list[dict[str, Any]] = []
    for row in selected:
        e = row.entry
        part_infos = part_info_cache[(str(e["dataset_id"]), str(e["obj_id"]))]
        out_path = out_dir / "images" / source_label(str(e["dataset_id"])) / (
            f"{e['obj_id']}_a{int(e['angle_idx']):03d}_{str(e['part_name']).replace('/', '_')}.png"
        )
        rendered.append(
            render_gate_image(
                packed_dir=packed_dir,
                out_path=out_path,
                row=row,
                entries_by_key=entries_by_key,
                part_infos=part_infos,
                cache=cache,
                rng=rng,
            )
        )

    with (out_dir / "v5_axis_gate_rendered.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({k for row in rendered for k in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rendered)
    (out_dir / "v5_axis_gate_rendered.json").write_text(json.dumps(rendered, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    print(f"[v5-axis-gate] rendered={len(rendered)} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
