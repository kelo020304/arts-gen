#!/usr/bin/env python3
"""Materialize the shared Phase 2 cache for Track1/Track2 overfit work.

This is a foreground, read-mostly preparation step. It does not launch training
or encode/render with GPUs. The cache keeps the real TRELLIS SLat
`coords/feats` tensors separate from the 16^3 dense part-seg tensors so later
train scripts cannot silently consume the wrong representation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from phase0_v5_data_check import (  # noqa: E402
    ObjectKey,
    _as_numpy,
    _json_load,
    build_object_index,
    coords_to_set,
    count_obj_faces,
    dataset_raw_roots,
    dataset_roots,
    load_part_info,
    load_rows_for_entries,
    raw_mesh_paths_for_part,
)


DEFAULT_PACKED_DIR = Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5")
DEFAULT_PHASE0_JSON = Path("/robot/data-lab/jzh/art-gen/data/slat_dec_part_cache/phase0_data_check.json")
DEFAULT_OUT_DIR = Path("/robot/data-lab/jzh/art-gen/data/slat_dec_part_cache/phase2_shared")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_name(*parts: str) -> str:
    return "__".join("".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(part)) for part in parts)


def coords64_to_mask16(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    mask = np.zeros((16, 16, 16), dtype=np.bool_)
    if arr.size:
        idx = np.clip(arr // 4, 0, 15)
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def coords64_erode6(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        return arr.astype(np.int16)
    occ = coords_to_set(arr)
    kept = []
    for x, y, z in occ:
        if (
            (x - 1, y, z) in occ
            and (x + 1, y, z) in occ
            and (x, y - 1, z) in occ
            and (x, y + 1, z) in occ
            and (x, y, z - 1) in occ
            and (x, y, z + 1) in occ
        ):
            kept.append((x, y, z))
    if not kept:
        return np.zeros((0, 3), dtype=np.int16)
    return np.asarray(sorted(kept), dtype=np.int16)


def front_only_coords64(coords: np.ndarray, whole_coords: np.ndarray, semantic_type: str) -> tuple[np.ndarray, dict[str, Any]]:
    semantic = str(semantic_type).lower()
    applicable = ("drawer" in semantic or "door" in semantic) and "handle" not in semantic
    if not applicable:
        return np.zeros((0, 3), dtype=np.int16), {"applicable": False}

    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    whole = np.asarray(whole_coords, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0 or whole.size == 0:
        return np.zeros((0, 3), dtype=np.int16), {"applicable": True, "status": "empty_input"}

    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    extent = np.maximum(hi - lo + 1, 1)
    comp_center = arr.mean(axis=0)
    whole_center = whole.mean(axis=0)
    axis = int(np.argmax(np.abs(comp_center - whole_center)))
    side = "max" if comp_center[axis] >= whole_center[axis] else "min"
    thickness = max(1, int(np.ceil(float(extent[axis]) * 0.20)))
    if side == "max":
        keep = arr[:, axis] >= (hi[axis] - thickness + 1)
    else:
        keep = arr[:, axis] <= (lo[axis] + thickness - 1)
    face = arr[keep]
    return face.astype(np.int16), {
        "applicable": True,
        "status": "ok" if len(face) else "empty_face",
        "axis": axis,
        "side": side,
        "thickness": thickness,
        "bbox_min": lo.astype(int).tolist(),
        "bbox_max": hi.astype(int).tolist(),
    }


def find_overall_slat_path(data_root: Path, obj_id: str, angle_idx: int) -> Path:
    candidates = [
        data_root / "part_synthesis_slat" / obj_id[:2] / f"{obj_id}_angle_{angle_idx}" / "overall" / "latent.npz",
        data_root
        / "reconstruction"
        / "part_synthesis_slat"
        / obj_id[:2]
        / f"{obj_id}_angle_{angle_idx}"
        / "overall"
        / "latent.npz",
        data_root / "reconstruction" / "slat_latents_expanded" / obj_id / f"angle_{angle_idx}" / "latent.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"overall SLat latent.npz not found for {obj_id} angle {angle_idx}; tried {candidates}")


def validate_slat_npz(path: Path, expected_coords_count: int) -> dict[str, Any]:
    with np.load(path) as data:
        if set(data.files) != {"coords", "feats"}:
            raise ValueError(f"{path}: expected keys coords/feats, got {sorted(data.files)}")
        coords = np.asarray(data["coords"])
        feats = np.asarray(data["feats"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: coords expected [N,3], got {coords.shape}")
    if feats.ndim != 2 or feats.shape[1] != 8:
        raise ValueError(f"{path}: feats expected [N,8], got {feats.shape}")
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(f"{path}: coords/feats length mismatch {coords.shape[0]} vs {feats.shape[0]}")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords dtype must be integer, got {coords.dtype}")
    if feats.dtype != np.float32:
        raise ValueError(f"{path}: feats dtype must be float32, got {feats.dtype}")
    if coords.size and (int(coords.min()) < 0 or int(coords.max()) >= 64):
        raise ValueError(f"{path}: coords out of [0,64), min={int(coords.min())} max={int(coords.max())}")
    return {
        "coords_shape": list(coords.shape),
        "coords_dtype": str(coords.dtype),
        "feats_shape": list(feats.shape),
        "feats_dtype": str(feats.dtype),
        "coords_count_matches_whole": bool(coords.shape[0] == expected_coords_count),
    }


def copy_mesh_hits(hits: list[Any], dst_dir: Path) -> tuple[list[dict[str, Any]], int, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total_vertices = 0
    total_faces = 0
    for idx, hit in enumerate(hits):
        src = Path(hit.path)
        if not src.is_file():
            raise FileNotFoundError(f"raw mesh source missing: {src}")
        vertices, faces = count_obj_faces(src)
        if faces <= 0:
            raise ValueError(f"raw mesh has no faces: {src}")
        dst_name = f"{idx:03d}_{src.name}"
        dst = dst_dir / dst_name
        shutil.copy2(src, dst)
        records.append(
            {
                "source_path": str(src),
                "cache_path": str(dst),
                "cache_rel": str(dst.relative_to(DEFAULT_OUT_DIR)) if dst.is_relative_to(DEFAULT_OUT_DIR) else str(dst),
                "raw_root": str(hit.raw_root),
                "layout": str(hit.layout),
                "vertices": int(vertices),
                "faces": int(faces),
            }
        )
        total_vertices += int(vertices)
        total_faces += int(faces)
    return records, total_vertices, total_faces


def selected_angle0_rows(
    *,
    packed_dir: Path,
    object_index: dict[ObjectKey, list[dict[str, Any]]],
    dataset_id: str,
    obj_id: str,
    part_names: list[str],
    angle_idx: int,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in object_index.get(ObjectKey(dataset_id, obj_id), [])
        if int(entry["angle_idx"]) == int(angle_idx) and str(entry["part_name"]) in set(part_names)
    ]
    if len(entries) != len(part_names):
        got = sorted(str(e["part_name"]) for e in entries)
        raise ValueError(f"{dataset_id}::{obj_id} angle {angle_idx}: expected parts {part_names}, got {got}")
    rows = load_rows_for_entries(packed_dir, entries)
    rows.sort(key=lambda row: str(row["part_name"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_object(
    *,
    packed_dir: Path,
    object_index: dict[ObjectKey, list[dict[str, Any]]],
    roots: dict[str, Path],
    raw_roots_by_dataset: dict[str, list[Path]],
    selected: dict[str, Any],
    out_dir: Path,
    angle_idx: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_id = str(selected["dataset_id"])
    obj_id = str(selected["obj_id"])
    tag = str(selected["tag"])
    part_names = [str(name) for name in selected["part_names"]]
    data_root = roots.get(dataset_id)
    if data_root is None:
        raise KeyError(f"dataset root missing for {dataset_id}")
    raw_roots = raw_roots_by_dataset.get(dataset_id, [])
    if not raw_roots:
        raise FileNotFoundError(f"raw roots missing for {dataset_id}")

    rows = selected_angle0_rows(
        packed_dir=packed_dir,
        object_index=object_index,
        dataset_id=dataset_id,
        obj_id=obj_id,
        part_names=part_names,
        angle_idx=angle_idx,
    )
    rows_by_part = {str(row["part_name"]): row for row in rows}
    first = rows[0]
    whole_coords = _as_numpy(first["whole_coords"]).astype(np.int16).reshape(-1, 3)
    z_global_dense = _as_numpy(first["z_global"]).astype(np.float32)
    for row in rows[1:]:
        z_other = _as_numpy(row["z_global"]).astype(np.float32)
        if not np.array_equal(z_global_dense, z_other):
            raise ValueError(f"{dataset_id}::{obj_id}: z_global differs across part rows")

    sample_dir = out_dir / "objects" / safe_name(tag, dataset_id, obj_id, f"angle_{angle_idx}")
    sample_dir.mkdir(parents=True, exist_ok=True)

    slat_src = find_overall_slat_path(data_root, obj_id, angle_idx)
    slat_check = validate_slat_npz(slat_src, expected_coords_count=int(len(whole_coords)))
    slat_dst = sample_dir / "overall_slat.npz"
    shutil.copy2(slat_src, slat_dst)

    token_path = data_root / "reconstruction" / "dinov2_tokens" / obj_id / f"angle_{angle_idx}" / "tokens.npz"
    if not token_path.is_file():
        raise FileNotFoundError(f"DINO tokens missing: {token_path}")
    view_indices = _as_numpy(first["view_indices"]).astype(np.int64).reshape(-1)
    with np.load(token_path, allow_pickle=False) as token_data:
        tokens_all = np.asarray(token_data["tokens"], dtype=np.float32)
    if tokens_all.ndim != 3 or tokens_all.shape[-1] != 1024:
        raise ValueError(f"{token_path}: expected tokens [V,T,1024], got {tokens_all.shape}")
    if int(view_indices.max()) >= tokens_all.shape[0]:
        raise ValueError(f"{token_path}: view index {int(view_indices.max())} outside token count {tokens_all.shape[0]}")
    cond_tokens = tokens_all[view_indices]
    cond_dst = sample_dir / "cond_4view_tokens.npz"
    np.savez_compressed(cond_dst, tokens=cond_tokens, view_indices=view_indices.astype(np.int16), source_path=str(token_path))

    dense_dst = sample_dir / "dense_partseg_latents.npz"
    np.savez_compressed(
        dense_dst,
        z_global=z_global_dense,
        whole_coords64=whole_coords,
        whole_mask16=coords64_to_mask16(whole_coords),
    )

    render_base = data_root / "renders" / obj_id / f"angle_{angle_idx}"
    image_paths = [render_base / "rgb" / f"view_{int(view)}.png" for view in view_indices]
    mask_paths = [render_base / "mask" / f"mask_{int(view)}.npy" for view in view_indices]
    for path in image_paths + mask_paths:
        if not path.is_file():
            raise FileNotFoundError(f"4-view render supervision file missing: {path}")

    part_info = load_part_info(data_root, obj_id)
    if part_info is None:
        raise FileNotFoundError(f"part_info missing for {dataset_id}::{obj_id}")

    component_records: list[dict[str, Any]] = []
    render_jobs: list[dict[str, Any]] = []
    union_part_coords: set[tuple[int, int, int]] = set()
    components = []

    for part_name in part_names:
        row = rows_by_part[part_name]
        semantic_type = str(row.get("semantic_type", ""))
        raw_coords = _as_numpy(row["raw_coords"]).astype(np.int16).reshape(-1, 3)
        union_part_coords |= coords_to_set(raw_coords)
        eroded = coords64_erode6(raw_coords)
        front, front_meta = front_only_coords64(raw_coords, whole_coords, semantic_type)
        mesh_hits = raw_mesh_paths_for_part(raw_roots, obj_id, part_name, part_info)
        if not mesh_hits:
            raise FileNotFoundError(f"{dataset_id}::{obj_id} {part_name}: raw GT mesh missing after Phase 0 pass")
        comp_dir = sample_dir / "components" / safe_name(part_name)
        mesh_records, mesh_vertices, mesh_faces = copy_mesh_hits(mesh_hits, comp_dir / "gt_mesh_objs")
        comp_npz = comp_dir / "component_cache.npz"
        np.savez_compressed(
            comp_npz,
            coords64=raw_coords,
            mask16_gt=_as_numpy(row["m_gt"]).astype(np.bool_),
            mask16_from_coords64=coords64_to_mask16(raw_coords),
            coords64_erode=eroded,
            mask16_erode=coords64_to_mask16(eroded),
            coords64_front_only=front,
            mask16_front_only=coords64_to_mask16(front),
            latent_gt16=_as_numpy(row["latent_gt"]).astype(np.float32),
            masks2d=_as_numpy(row["masks2d"]).astype(np.uint8),
            view_indices=view_indices.astype(np.int16),
        )
        record = {
            "tag": tag,
            "dataset_id": dataset_id,
            "obj_id": obj_id,
            "angle_idx": int(angle_idx),
            "component_name": part_name,
            "component_role": "part",
            "semantic_type": semantic_type,
            "component_cache": str(comp_npz),
            "component_cache_rel": str(comp_npz.relative_to(out_dir)),
            "coords64_count": int(len(raw_coords)),
            "mask16_voxels": int(_as_numpy(row["m_gt"]).sum()),
            "erode64_count": int(len(eroded)),
            "front_only64_count": int(len(front)),
            "front_only_meta": front_meta,
            "gt_mesh_obj_count": int(len(mesh_records)),
            "gt_mesh_faces": int(mesh_faces),
            "gt_mesh_vertices": int(mesh_vertices),
            "gt_mesh_records": mesh_records,
            "component_render_status": "missing_offline_component_renders",
        }
        component_records.append(record)
        components.append(record)
        render_jobs.append(
            {
                "dataset_id": dataset_id,
                "obj_id": obj_id,
                "angle_idx": int(angle_idx),
                "component_name": part_name,
                "component_role": "part",
                "gt_mesh_objs": [item["cache_path"] for item in mesh_records],
                "reference_rgb": [str(path) for path in image_paths],
                "reference_masks": [str(path) for path in mask_paths],
                "status": "planned",
                "reason": "component GT render supervision not precomputed in v5; render from staged raw OBJ with object angle cameras",
            }
        )

    whole_set = coords_to_set(whole_coords)
    body_coords = np.asarray(sorted(whole_set - union_part_coords), dtype=np.int16).reshape(-1, 3)
    body_eroded = coords64_erode6(body_coords)
    part_info_parts = part_info.get("parts", {})
    body_mesh_hits = []
    for info_part_name in sorted(part_info_parts):
        if info_part_name in part_names:
            continue
        body_mesh_hits.extend(raw_mesh_paths_for_part(raw_roots, obj_id, str(info_part_name), part_info))
    if not body_mesh_hits:
        raise FileNotFoundError(f"{dataset_id}::{obj_id}: body_without_parts GT mesh sources missing")
    body_dir = sample_dir / "components" / "body_without_parts"
    body_mesh_records, body_vertices, body_faces = copy_mesh_hits(body_mesh_hits, body_dir / "gt_mesh_objs")
    body_npz = body_dir / "component_cache.npz"
    np.savez_compressed(
        body_npz,
        coords64=body_coords,
        mask16_gt=coords64_to_mask16(body_coords),
        coords64_erode=body_eroded,
        mask16_erode=coords64_to_mask16(body_eroded),
        coords64_front_only=np.zeros((0, 3), dtype=np.int16),
        mask16_front_only=np.zeros((16, 16, 16), dtype=np.bool_),
        view_indices=view_indices.astype(np.int16),
    )
    body_record = {
        "tag": tag,
        "dataset_id": dataset_id,
        "obj_id": obj_id,
        "angle_idx": int(angle_idx),
        "component_name": "body_without_parts",
        "component_role": "body",
        "semantic_type": "body_without_parts",
        "component_cache": str(body_npz),
        "component_cache_rel": str(body_npz.relative_to(out_dir)),
        "coords64_count": int(len(body_coords)),
        "mask16_voxels": int(coords64_to_mask16(body_coords).sum()),
        "erode64_count": int(len(body_eroded)),
        "front_only64_count": 0,
        "front_only_meta": {"applicable": False},
        "gt_mesh_obj_count": int(len(body_mesh_records)),
        "gt_mesh_faces": int(body_faces),
        "gt_mesh_vertices": int(body_vertices),
        "gt_mesh_records": body_mesh_records,
        "component_render_status": "missing_offline_component_renders",
    }
    component_records.append(body_record)
    components.append(body_record)
    render_jobs.append(
        {
            "dataset_id": dataset_id,
            "obj_id": obj_id,
            "angle_idx": int(angle_idx),
            "component_name": "body_without_parts",
            "component_role": "body",
            "gt_mesh_objs": [item["cache_path"] for item in body_mesh_records],
            "reference_rgb": [str(path) for path in image_paths],
            "reference_masks": [str(path) for path in mask_paths],
            "status": "planned",
            "reason": "body GT render supervision must be rendered from non-target fixed/body OBJ sources",
        }
    )

    object_manifest = {
        "tag": tag,
        "dataset_id": dataset_id,
        "obj_id": obj_id,
        "angle_idx": int(angle_idx),
        "data_root": str(data_root),
        "raw_root_candidates": [str(path) for path in raw_roots],
        "sample_dir": str(sample_dir),
        "overall_slat_source": str(slat_src),
        "overall_slat_cache": str(slat_dst),
        "overall_slat_rel": str(slat_dst.relative_to(out_dir)),
        "overall_slat_schema": slat_check,
        "dense_partseg_cache": str(dense_dst),
        "dense_partseg_rel": str(dense_dst.relative_to(out_dir)),
        "cond_4view_tokens_source": str(token_path),
        "cond_4view_tokens_cache": str(cond_dst),
        "cond_4view_tokens_rel": str(cond_dst.relative_to(out_dir)),
        "cond_4view_tokens_shape": list(cond_tokens.shape),
        "view_indices": [int(v) for v in view_indices.tolist()],
        "reference_rgb": [str(path) for path in image_paths],
        "reference_masks": [str(path) for path in mask_paths],
        "whole_coords64_count": int(len(whole_coords)),
        "body_without_parts_coords64_count": int(len(body_coords)),
        "part_count": int(len(part_names)),
        "component_count_with_body": int(len(components)),
        "components": components,
        "render_supervision_ready": False,
        "render_plan_status": "planned_not_rendered",
    }
    _write_json(sample_dir / "object_manifest.json", object_manifest)
    return object_manifest, component_records, render_jobs


def write_report(
    path: Path,
    *,
    objects: list[dict[str, Any]],
    components: list[dict[str, Any]],
    render_jobs: list[dict[str, Any]],
    elapsed: float,
) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    by_dataset = defaultdict(int)
    for obj in objects:
        by_dataset[str(obj["dataset_id"])] += 1
    missing_render = sum(1 for row in components if row.get("component_render_status") != "ready")
    lines = [
        "# Phase 2 Shared Cache Prep",
        "",
        f"- generated: {now}",
        f"- objects: {len(objects)}",
        f"- components_including_body: {len(components)}",
        f"- render_jobs_planned: {len(render_jobs)}",
        f"- elapsed_seconds: {elapsed:.2f}",
        "",
        "## Gate State",
        "",
        "PASS: shared tensor/mesh cache materialized for the selected Phase 0 objects.",
        f"BLOCKER BEFORE TRAINING: component render supervision is still offline-planned, not rendered yet (`missing_offline_component_renders` components: {missing_render}). Do not start Track1 mesh-decoder training until those renders exist or the trainer is configured to render staged GT meshes on the fly.",
        "",
        "## Dataset Mix",
        "",
    ]
    for dataset_id, count in sorted(by_dataset.items()):
        lines.append(f"- {dataset_id}: {count}")
    lines.extend(["", "## Objects", ""])
    lines.append("| tag | dataset | obj | parts | components | slat_N | token_shape | body_vox64 | render_ready |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for obj in objects:
        token_shape = obj.get("cond_4view_tokens_shape")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(obj["tag"]),
                    str(obj["dataset_id"]),
                    str(obj["obj_id"]),
                    str(obj["part_count"]),
                    str(obj["component_count_with_body"]),
                    str((obj.get("overall_slat_schema") or {}).get("coords_shape", ["?"])[0]),
                    str(token_shape),
                    str(obj["body_without_parts_coords64_count"]),
                    str(obj["render_supervision_ready"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- manifest: `{path.with_name('phase2_shared_cache_manifest.json')}`",
            f"- component table: `{path.with_name('phase2_component_table.csv')}`",
            f"- render plan: `{path.with_name('phase2_render_plan.json')}`",
            "",
            "No training was launched by this prep step.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED_DIR)
    parser.add_argument("--phase0-json", type=Path, default=DEFAULT_PHASE0_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--angle-idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    start = time.perf_counter()
    packed_dir = args.packed_dir.resolve()
    phase0_json = args.phase0_json.resolve()
    out_dir = args.out_dir.resolve()
    index_path = packed_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"v5 index not found: {index_path}")
    if not phase0_json.is_file():
        raise FileNotFoundError(f"Phase 0 JSON not found: {phase0_json}")

    index = _json_load(index_path)
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{index_path}: expected entries list")
    roots = dataset_roots(index)
    raw_roots_by_dataset = dataset_raw_roots(index)
    object_index = build_object_index(entries)
    phase0 = _json_load(phase0_json)
    if not phase0.get("phase0_gate_pass"):
        raise RuntimeError(f"Phase 0 gate did not pass in {phase0_json}")
    selected = list(phase0.get("selected_pass_objects", []))[: int(args.limit)]
    if len(selected) < 8:
        raise RuntimeError(f"need at least 8 selected pass objects, got {len(selected)}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    object_manifests: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    render_jobs: list[dict[str, Any]] = []
    for item in selected:
        obj_manifest, obj_components, obj_render_jobs = prepare_object(
            packed_dir=packed_dir,
            object_index=object_index,
            roots=roots,
            raw_roots_by_dataset=raw_roots_by_dataset,
            selected=item,
            out_dir=out_dir,
            angle_idx=int(args.angle_idx),
        )
        object_manifests.append(obj_manifest)
        component_rows.extend(obj_components)
        render_jobs.extend(obj_render_jobs)

    elapsed = time.perf_counter() - start
    manifest = {
        "packed_dir": str(packed_dir),
        "phase0_json": str(phase0_json),
        "out_dir": str(out_dir),
        "elapsed_seconds": elapsed,
        "object_count": len(object_manifests),
        "component_count": len(component_rows),
        "render_job_count": len(render_jobs),
        "render_supervision_ready": False,
        "objects": object_manifests,
    }
    _write_json(out_dir / "phase2_shared_cache_manifest.json", manifest)
    _write_json(out_dir / "phase2_render_plan.json", {"jobs": render_jobs})
    csv_rows = []
    for row in component_rows:
        flat = dict(row)
        flat["front_only_meta"] = json.dumps(row.get("front_only_meta", {}), sort_keys=True)
        flat["gt_mesh_records"] = json.dumps(row.get("gt_mesh_records", []), sort_keys=True)
        csv_rows.append(flat)
    write_csv(
        out_dir / "phase2_component_table.csv",
        csv_rows,
        [
            "tag",
            "dataset_id",
            "obj_id",
            "angle_idx",
            "component_name",
            "component_role",
            "semantic_type",
            "component_cache_rel",
            "coords64_count",
            "mask16_voxels",
            "erode64_count",
            "front_only64_count",
            "front_only_meta",
            "gt_mesh_obj_count",
            "gt_mesh_faces",
            "gt_mesh_vertices",
            "component_render_status",
            "gt_mesh_records",
        ],
    )
    write_report(
        out_dir / "phase2_cache_report.md",
        objects=object_manifests,
        components=component_rows,
        render_jobs=render_jobs,
        elapsed=elapsed,
    )
    print(f"[phase2] objects={len(object_manifests)} components={len(component_rows)} render_jobs={len(render_jobs)}")
    print(f"[phase2] manifest={out_dir / 'phase2_shared_cache_manifest.json'}")
    print(f"[phase2] report={out_dir / 'phase2_cache_report.md'}")
    print("[phase2] training_started=False render_supervision_ready=False")


if __name__ == "__main__":
    main()
