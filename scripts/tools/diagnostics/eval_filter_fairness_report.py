#!/usr/bin/env python3
"""Diagnose promptability filter misses and recompute detectable-only eval tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
for path in (PROJECT_ROOT, TRELLIS_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    bucket_name,
    format_table,
)


VALUE_KEYS = (
    "cell_iou",
    "head2_gtm_decode_iou",
    "e2e_decode_iou",
    "partition_e2e_decode_iou",
    "part_overlap_voxels",
    "object_overlap_voxels",
    "partition_object_overlap_voxels",
)
BUCKETS = ("tiny", "small", "medium", "large", "all")
TARGET_NAME_TERMS = ("rotational_shaft", "gift_box", "feed_horn")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, Any]) -> str:
    dataset_id = str(row.get("dataset_id", ""))
    prefix = f"{dataset_id}::" if dataset_id else ""
    return f"{prefix}{row.get('obj_id')}|{int(row.get('angle_idx', 0))}|{row.get('part_name')}"


def object_angle_key(row: dict[str, Any]) -> str:
    dataset_id = str(row.get("dataset_id", ""))
    prefix = f"{dataset_id}::" if dataset_id else ""
    return f"{prefix}{row.get('obj_id')}|{int(row.get('angle_idx', 0))}"


def is_promptable_eval_row(row: dict[str, Any]) -> bool:
    if str(row.get("joint_class_kind", "")) == "body":
        return False
    if str(row.get("part_name", "")) == "body":
        return False
    return True


def old_filter_class(selected_px: int, all_px: int) -> str:
    if selected_px > 0:
        return "visible_selected_views"
    if all_px > 0:
        return "undetectable_selected_views"
    return "label_absent_all_views"


def new_filter_class(selected_px: int, all_px: int) -> str:
    if selected_px > 0:
        return "visible_selected_views"
    if all_px > 0:
        return "undetectable_selected_views"
    return "undetectable_all_views"


def is_undetectable_new(cls: str) -> bool:
    return cls in {"undetectable_selected_views", "undetectable_all_views"}


def index_entries(packed_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    index_path = packed_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"packed index not found: {index_path}")
    index = load_json(index_path)
    by_key = {str(entry["key"]): dict(entry) for entry in index.get("entries", [])}
    datasets = {
        str(item.get("dataset_id", "")): dict(item)
        for item in index.get("datasets", [])
        if isinstance(item, dict)
    }
    return by_key, datasets, index


def load_packed_sample(packed_dir: Path, entry: dict[str, Any], cache: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    shard = str(entry["shard"])
    if shard not in cache:
        cache[shard] = torch.load(packed_dir / shard, map_location="cpu", weights_only=False)
    return dict(cache[shard][int(entry["index"])])


def mask_dir_for_entry(entry: dict[str, Any], datasets: dict[str, dict[str, Any]]) -> Path:
    dataset_id = str(entry.get("dataset_id", ""))
    data_root = Path(str(datasets.get(dataset_id, {}).get("data_root", "")))
    if not data_root:
        raise KeyError(f"no data_root in packed index for dataset_id={dataset_id!r}")
    return data_root / "renders" / str(entry["obj_id"]) / f"angle_{int(entry['angle_idx'])}" / "mask"


def raw_voxel_path_for_entry(entry: dict[str, Any], datasets: dict[str, dict[str, Any]]) -> Path:
    dataset_id = str(entry.get("dataset_id", ""))
    data_root = Path(str(datasets.get(dataset_id, {}).get("data_root", "")))
    return (
        data_root
        / "reconstruction"
        / "voxel_expanded"
        / str(entry["obj_id"])
        / f"angle_{int(entry['angle_idx'])}"
        / "64"
        / f"ind_{entry['part_name']}.npy"
    )


def all_mask_counts(
    entry: dict[str, Any],
    datasets: dict[str, dict[str, Any]],
    original_label: int,
    *,
    expected_views: int,
    cache: dict[tuple[str, str, int], tuple[dict[int, dict[int, int]], list[int]]],
) -> tuple[dict[int, int], list[int]]:
    cache_key = (str(entry.get("dataset_id", "")), str(entry["obj_id"]), int(entry["angle_idx"]))
    if cache_key not in cache:
        mask_dir = mask_dir_for_entry(entry, datasets)
        found: dict[int, Path] = {}
        if mask_dir.is_dir():
            for path in mask_dir.glob("mask_*.npy"):
                stem = path.stem
                try:
                    idx = int(stem.split("_")[-1])
                except ValueError:
                    continue
                found[idx] = path
        view_ids = set(found)
        if int(expected_views) > 0:
            view_ids.update(range(int(expected_views)))
        counts_by_label: dict[int, dict[int, int]] = {}
        missing: list[int] = []
        for view_idx in sorted(view_ids):
            path = found.get(view_idx, mask_dir / f"mask_{view_idx}.npy")
            if not path.is_file():
                missing.append(int(view_idx))
                counts_by_label[int(view_idx)] = {}
                continue
            labels, counts = np.unique(np.asarray(np.load(path)), return_counts=True)
            counts_by_label[int(view_idx)] = {int(label): int(count) for label, count in zip(labels.tolist(), counts.tolist())}
        cache[cache_key] = (counts_by_label, missing)
    counts_by_label, missing = cache[cache_key]
    return (
        {view_idx: counts.get(int(original_label), 0) for view_idx, counts in counts_by_label.items()},
        missing,
    )


def diagnose_row(
    row: dict[str, Any],
    *,
    packed_dir: Path,
    entries: dict[str, dict[str, Any]],
    datasets: dict[str, dict[str, Any]],
    shard_cache: dict[str, list[dict[str, Any]]],
    mask_cache: dict[tuple[str, str, int], tuple[dict[int, dict[int, int]], list[int]]],
    expected_views: int,
) -> dict[str, Any]:
    key = row_key(row)
    entry = entries.get(key)
    if entry is None:
        return {
            "key": key,
            "matched_packed": False,
            "new_filter_class": "unmatched_eval_row",
            "old_filter_class": "unmatched_eval_row",
            "new_filter_drop": False,
            "gt_voxel_nonempty": int(row.get("raw_count", 0)) > 0,
            "reason": "no packed promptable row for eval row",
        }
    sample = load_packed_sample(packed_dir, entry, shard_cache)
    original_label = int(sample.get("original_label", 0))
    view_indices = [int(v) for v in torch.as_tensor(sample.get("view_indices", [])).detach().cpu().tolist()]
    selected_from_packed = [int(v) for v in torch.as_tensor(sample.get("masks2d")).sum(dim=(1, 2)).detach().cpu().tolist()]
    all_by_view, missing = all_mask_counts(
        entry,
        datasets,
        original_label,
        expected_views=int(expected_views),
        cache=mask_cache,
    )
    selected_by_view = {
        int(view_idx): int(all_by_view.get(int(view_idx), 0))
        for view_idx in view_indices
    }
    selected_px = int(sum(selected_by_view.values()))
    all_px = int(sum(all_by_view.values()))
    old_cls = old_filter_class(selected_px, all_px)
    new_cls = new_filter_class(selected_px, all_px)
    raw_count = int(row.get("raw_count", entry.get("raw_count", 0)) or 0)
    raw_path = raw_voxel_path_for_entry(entry, datasets)
    raw_file_count = None
    if raw_path.is_file():
        raw_file_count = int(np.asarray(np.load(raw_path)).reshape(-1, 3).shape[0])
    return {
        "key": key,
        "matched_packed": True,
        "dataset_id": row.get("dataset_id", entry.get("dataset_id", "")),
        "obj_id": row.get("obj_id"),
        "angle_idx": int(row.get("angle_idx", 0)),
        "part_name": row.get("part_name"),
        "bucket": row.get("bucket", bucket_name(raw_count)),
        "raw_count": raw_count,
        "raw_file_count": raw_file_count,
        "gt_voxel_nonempty": raw_count > 0,
        "original_label": original_label,
        "selected_view_indices": view_indices,
        "selected_mask_px": selected_px,
        "selected_mask_px_by_view": selected_by_view,
        "selected_packed_mask_px_by_view": selected_from_packed,
        "all_mask_px": all_px,
        "all_mask_px_by_view": all_by_view,
        "missing_mask_views": missing,
        "old_filter_class": old_cls,
        "new_filter_class": new_cls,
        "old_filter_drop": old_cls == "undetectable_selected_views",
        "new_filter_drop": is_undetectable_new(new_cls),
        "cell_iou": float(row.get("cell_iou", float("nan"))),
        "head2_gtm_decode_iou": float(row.get("head2_gtm_decode_iou", float("nan"))),
        "e2e_decode_iou": float(row.get("e2e_decode_iou", float("nan"))),
        "part_overlap_voxels": float(row.get("part_overlap_voxels", 0.0)),
        "partition_object_overlap_voxels": float(row.get("partition_object_overlap_voxels", 0.0)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for key in VALUE_KEYS:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def bucket_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for bucket in ("tiny", "small", "medium", "large"):
        out[bucket] = summarize_rows([row for row in rows if str(row.get("bucket", bucket_name(int(row.get("raw_count", 0))))) == bucket])
    out["all"] = summarize_rows(rows)
    return out


def compact_table(summary: dict[str, dict[str, Any]], *, prefix: str = "") -> list[dict[str, Any]]:
    rows = []
    for bucket in BUCKETS:
        item = summary.get(bucket, {})
        rows.append({
            f"{prefix}bucket": bucket,
            f"{prefix}n": int(item.get("n", 0)),
            f"{prefix}cell": f"{float(item.get('cell_iou', float('nan'))):.4f}",
            f"{prefix}GTcand": f"{float(item.get('head2_gtm_decode_iou', float('nan'))):.4f}",
            f"{prefix}Predcand": f"{float(item.get('e2e_decode_iou', float('nan'))):.4f}",
            f"{prefix}part_ov": f"{float(item.get('partition_object_overlap_voxels', float('nan'))):.1f}",
        })
    return rows


def compare_table(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> str:
    rows = []
    for bucket in BUCKETS:
        b = before.get(bucket, {})
        a = after.get(bucket, {})
        rows.append({
            "bucket": bucket,
            "before_n": int(b.get("n", 0)),
            "before_cell": f"{float(b.get('cell_iou', float('nan'))):.4f}",
            "before_GTcand": f"{float(b.get('head2_gtm_decode_iou', float('nan'))):.4f}",
            "before_Predcand": f"{float(b.get('e2e_decode_iou', float('nan'))):.4f}",
            "after_n": int(a.get("n", 0)),
            "after_cell": f"{float(a.get('cell_iou', float('nan'))):.4f}",
            "after_GTcand": f"{float(a.get('head2_gtm_decode_iou', float('nan'))):.4f}",
            "after_Predcand": f"{float(a.get('e2e_decode_iou', float('nan'))):.4f}",
            "after_part_ov": f"{float(a.get('partition_object_overlap_voxels', float('nan'))):.1f}",
        })
    return format_table(rows, ["bucket", "before_n", "before_cell", "before_GTcand", "before_Predcand", "after_n", "after_cell", "after_GTcand", "after_Predcand", "after_part_ov"])


def selected_zero_rows(rows: list[dict[str, Any]], *, max_rows: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if is_promptable_eval_row(row)
        and float(row.get("cell_iou", 0.0)) <= 0.0
        and (
            str(row.get("bucket", "")) in {"tiny", "small"}
            or any(term in str(row.get("part_name", "")).lower() for term in TARGET_NAME_TERMS)
        )
    ]
    candidates.sort(key=lambda r: (str(r.get("bucket", "")), str(r.get("obj_id", "")), int(r.get("angle_idx", 0)), str(r.get("part_name", ""))))
    return candidates[: int(max_rows)]


def split_report(
    *,
    split: str,
    rows_path: Path,
    packed_dir: Path,
    entries: dict[str, dict[str, Any]],
    datasets: dict[str, dict[str, Any]],
    expected_views: int,
    out_dir: Path,
    max_diag_rows: int,
) -> dict[str, Any]:
    rows = load_json(rows_path)
    prompt_rows = [row for row in rows if is_promptable_eval_row(row)]
    shard_cache: dict[str, list[dict[str, Any]]] = {}
    mask_cache: dict[tuple[str, str, int], tuple[dict[int, dict[int, int]], list[int]]] = {}
    diag_targets = selected_zero_rows(prompt_rows, max_rows=max_diag_rows)
    diag_rows = [
        diagnose_row(
            row,
            packed_dir=packed_dir,
            entries=entries,
            datasets=datasets,
            shard_cache=shard_cache,
            mask_cache=mask_cache,
            expected_views=int(expected_views),
        )
        for row in diag_targets
    ]

    row_filter: dict[str, dict[str, Any]] = {}
    for row in prompt_rows:
        key = row_key(row)
        entry = entries.get(key)
        if entry is None:
            row_filter[key] = {
                "new_filter_class": "unmatched_eval_row",
                "old_filter_class": "unmatched_eval_row",
                "new_filter_drop": False,
                "matched_packed": False,
            }
            continue
        sample = load_packed_sample(packed_dir, entry, shard_cache)
        original_label = int(sample.get("original_label", 0))
        view_indices = [int(v) for v in torch.as_tensor(sample.get("view_indices", [])).detach().cpu().tolist()]
        selected_px = int(torch.as_tensor(sample.get("masks2d")).sum().detach().cpu().item())
        if selected_px > 0:
            all_px = selected_px
            old_cls = "visible_selected_views"
            new_cls = "visible_selected_views"
        else:
            all_by_view, _missing = all_mask_counts(
                entry,
                datasets,
                original_label,
                expected_views=int(expected_views),
                cache=mask_cache,
            )
            all_px = int(sum(all_by_view.values()))
            old_cls = old_filter_class(selected_px, all_px)
            new_cls = new_filter_class(selected_px, all_px)
        row_filter[key] = {
            "new_filter_class": new_cls,
            "old_filter_class": old_cls,
            "new_filter_drop": is_undetectable_new(new_cls),
            "old_filter_drop": old_cls == "undetectable_selected_views",
            "matched_packed": True,
            "selected_mask_px": selected_px,
            "all_mask_px": all_px,
        }

    before_summary = bucket_summary(rows)
    detectable_rows = [
        row
        for row in rows
        if not (
            is_promptable_eval_row(row)
            and bool(row_filter.get(row_key(row), {}).get("new_filter_drop", False))
        )
    ]
    after_summary = bucket_summary(detectable_rows)
    prompt_total = len(prompt_rows)
    dropped_prompt_keys = {
        key for key, info in row_filter.items() if bool(info.get("new_filter_drop", False))
    }
    old_drop_keys = {
        key for key, info in row_filter.items() if bool(info.get("old_filter_drop", False))
    }
    class_counts = Counter(str(info.get("new_filter_class", "")) for info in row_filter.values())
    drop_rows = []
    for row in prompt_rows:
        key = row_key(row)
        info = row_filter.get(key, {})
        if not bool(info.get("new_filter_drop", False)):
            continue
        drop_rows.append({
            "key": key,
            "split": split,
            "obj_id": row.get("obj_id"),
            "angle_idx": int(row.get("angle_idx", 0)),
            "part_name": row.get("part_name"),
            "bucket": row.get("bucket"),
            "raw_count": int(row.get("raw_count", 0)),
            "selected_mask_px": int(info.get("selected_mask_px", 0)),
            "all_mask_px": int(info.get("all_mask_px", 0)),
            "new_filter_class": info.get("new_filter_class", ""),
            "cell_iou": float(row.get("cell_iou", float("nan"))),
            "GTcand": float(row.get("head2_gtm_decode_iou", float("nan"))),
            "Predcand": float(row.get("e2e_decode_iou", float("nan"))),
        })
    drop_counts_by_class = Counter(row["new_filter_class"] for row in drop_rows)
    drop_counts_by_bucket = Counter(str(row["bucket"]) for row in drop_rows)
    no_visible_drops = all(int(row["selected_mask_px"]) == 0 for row in drop_rows)
    no_promptable_false_drop = all(
        int(row["selected_mask_px"]) <= 0
        for row in drop_rows
    )
    split_dir = out_dir / split
    dump_json(split_dir / "diagnostic_zero_rows.json", diag_rows)
    write_csv(split_dir / "diagnostic_zero_rows.csv", diag_rows)
    dump_json(split_dir / "dropped_rows.json", drop_rows)
    write_csv(split_dir / "dropped_rows.csv", drop_rows)
    dump_json(split_dir / "summary_before.json", before_summary)
    dump_json(split_dir / "summary_detectable_only.json", after_summary)
    return {
        "split": split,
        "rows_path": str(rows_path),
        "total_rows": len(rows),
        "promptable_rows": prompt_total,
        "unmatched_promptable_rows": sum(1 for info in row_filter.values() if not bool(info.get("matched_packed", False))),
        "old_filtered_promptable_rows": len(old_drop_keys),
        "new_filtered_promptable_rows": len(dropped_prompt_keys),
        "newly_filtered_promptable_rows": len(dropped_prompt_keys - old_drop_keys),
        "new_filtered_promptable_ratio": float(len(dropped_prompt_keys) / max(1, prompt_total)),
        "new_class_counts": dict(sorted(class_counts.items())),
        "drop_counts_by_class": dict(sorted(drop_counts_by_class.items())),
        "drop_counts_by_bucket": dict(sorted(drop_counts_by_bucket.items())),
        "drop_rows_all_selected_mask_px_zero": bool(no_visible_drops),
        "no_promptable_false_drop": bool(no_promptable_false_drop),
        "diagnostic_zero_rows": diag_rows,
        "before_summary": before_summary,
        "detectable_only_summary": after_summary,
        "compare_table": compare_table(before_summary, after_summary),
        "part_ov_max_after": max([float(row.get("partition_object_overlap_voxels", 0.0)) for row in detectable_rows] or [0.0]),
    }


def audit_row_list(
    *,
    name: str,
    rows_path: Path,
    packed_dir: Path,
    entries: dict[str, dict[str, Any]],
    datasets: dict[str, dict[str, Any]],
    expected_views: int,
    out_dir: Path,
    shard_cache: dict[str, list[dict[str, Any]]] | None = None,
    mask_cache: dict[tuple[str, str, int], tuple[dict[int, dict[int, int]], list[int]]] | None = None,
) -> dict[str, Any]:
    rows = load_json(rows_path)
    if mask_cache is None:
        mask_cache = {}
    by_shard: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    unmatched = 0
    for row in rows:
        key = row_key(row)
        entry = entries.get(key)
        if entry is None:
            unmatched += 1
            continue
        by_shard[str(entry["shard"])].append((row, entry))
    records: list[dict[str, Any]] = []
    for shard, items in sorted(by_shard.items()):
        samples = torch.load(packed_dir / shard, map_location="cpu", weights_only=False)
        for row, entry in items:
            sample = samples[int(entry["index"])]
            selected_px = int(torch.as_tensor(sample.get("masks2d")).sum().detach().cpu().item())
            if selected_px > 0:
                all_px = selected_px
                old_cls = "visible_selected_views"
                new_cls = "visible_selected_views"
            else:
                original_label = int(sample.get("original_label", 0))
                all_by_view, _missing = all_mask_counts(
                    entry,
                    datasets,
                    original_label,
                    expected_views=int(expected_views),
                    cache=mask_cache,
                )
                all_px = int(sum(all_by_view.values()))
                old_cls = old_filter_class(selected_px, all_px)
                new_cls = new_filter_class(selected_px, all_px)
            raw_count = int(row.get("raw_count", entry.get("raw_count", 0)) or 0)
            records.append({
                "key": row_key(row),
                "dataset_id": row.get("dataset_id", entry.get("dataset_id", "")),
                "obj_id": row.get("obj_id"),
                "angle_idx": int(row.get("angle_idx", 0)),
                "part_name": row.get("part_name"),
                "bucket": row.get("bucket", bucket_name(raw_count)),
                "raw_count": raw_count,
                "selected_mask_px": selected_px,
                "all_mask_px": all_px,
                "old_filter_class": old_cls,
                "new_filter_class": new_cls,
                "old_filter_drop": old_cls == "undetectable_selected_views",
                "new_filter_drop": is_undetectable_new(new_cls),
            })
        del samples
    old_drop = [rec for rec in records if bool(rec["old_filter_drop"])]
    new_drop = [rec for rec in records if bool(rec["new_filter_drop"])]
    newly_drop = [rec for rec in new_drop if not bool(rec["old_filter_drop"])]
    false_drop = [rec for rec in new_drop if int(rec["selected_mask_px"]) > 0]
    out = {
        "name": name,
        "rows_path": str(rows_path),
        "total_rows": len(rows),
        "matched_rows": len(records),
        "unmatched_rows": int(unmatched),
        "old_drop_rows": len(old_drop),
        "new_drop_rows": len(new_drop),
        "newly_drop_rows": len(newly_drop),
        "new_drop_ratio": float(len(new_drop) / max(1, len(records))),
        "newly_drop_ratio": float(len(newly_drop) / max(1, len(records))),
        "false_drop_selected_px_gt0": len(false_drop),
        "class_counts": dict(sorted(Counter(str(rec["new_filter_class"]) for rec in records).items())),
        "drop_counts_by_class": dict(sorted(Counter(str(rec["new_filter_class"]) for rec in new_drop).items())),
        "drop_counts_by_bucket": dict(sorted(Counter(str(rec["bucket"]) for rec in new_drop).items())),
        "newly_drop_counts_by_class": dict(sorted(Counter(str(rec["new_filter_class"]) for rec in newly_drop).items())),
        "newly_drop_counts_by_bucket": dict(sorted(Counter(str(rec["bucket"]) for rec in newly_drop).items())),
        "all_new_drops_selected_px_zero": all(int(rec["selected_mask_px"]) == 0 for rec in new_drop),
        "all_newly_drops_all_px_zero": all(int(rec["all_mask_px"]) == 0 for rec in newly_drop),
        "sample_new_drops": new_drop[:50],
        "sample_newly_drops": newly_drop[:50],
        "sample_false_drops": false_drop[:50],
    }
    safe_name = name.replace("/", "_")
    dump_json(out_dir / "row_list_audit" / f"{safe_name}.json", out)
    write_csv(out_dir / "row_list_audit" / f"{safe_name}_new_drops.csv", new_drop)
    write_csv(out_dir / "row_list_audit" / f"{safe_name}_newly_drops.csv", newly_drop)
    return out


def report_for_run(args: argparse.Namespace, run_dir: Path, label: str) -> dict[str, Any]:
    eval_dir = run_dir / "eval" / str(args.step_dir)
    packed_dir = Path(args.packed_dir)
    entries, datasets, packed_index = index_entries(packed_dir)
    out_dir = Path(args.out_dir) / label
    split_reports = {}
    for split in ("train", "heldout"):
        rows_path = eval_dir / split / "rows.json"
        if not rows_path.is_file():
            raise FileNotFoundError(f"eval rows not found: {rows_path}")
        split_reports[split] = split_report(
            split=split,
            rows_path=rows_path,
            packed_dir=packed_dir,
            entries=entries,
            datasets=datasets,
            expected_views=int(args.expected_views),
            out_dir=out_dir,
            max_diag_rows=int(args.max_diag_rows),
        )
    row_list_audits = {}
    if bool(args.audit_row_lists):
        row_list_shard_cache: dict[str, list[dict[str, Any]]] = {}
        row_list_mask_cache: dict[tuple[str, str, int], tuple[dict[int, dict[int, int]], list[int]]] = {}
        for name in ("train_rows", "proxy_train_rows", "proxy_eval_rows", "full_eval_rows"):
            path = run_dir / f"{name}.json"
            if path.is_file():
                row_list_audits[name] = audit_row_list(
                    name=name,
                    rows_path=path,
                    packed_dir=packed_dir,
                    entries=entries,
                    datasets=datasets,
                    expected_views=int(args.expected_views),
                    out_dir=out_dir,
                    shard_cache=row_list_shard_cache,
                    mask_cache=row_list_mask_cache,
                )
    combined = {
        "label": label,
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "packed_dir": str(packed_dir),
        "packed_split_json": packed_index.get("split_json"),
        "reports": split_reports,
        "row_list_audits": row_list_audits,
    }
    lines = [
        f"# {label} {args.step_dir}",
        "",
        f"run_dir: {run_dir}",
        f"packed_dir: {packed_dir}",
        "",
    ]
    for split in ("train", "heldout"):
        rep = split_reports[split]
        lines.extend([
            f"## {split}",
            "",
            f"promptable filtered: {rep['new_filtered_promptable_rows']}/{rep['promptable_rows']} ({rep['new_filtered_promptable_ratio']:.2%}); newly filtered vs old: {rep['newly_filtered_promptable_rows']}",
            f"class counts: {json.dumps(rep['new_class_counts'], ensure_ascii=False, sort_keys=True)}",
            f"drop classes: {json.dumps(rep['drop_counts_by_class'], ensure_ascii=False, sort_keys=True)}",
            f"drop buckets: {json.dumps(rep['drop_counts_by_bucket'], ensure_ascii=False, sort_keys=True)}",
            f"drop selected_mask_px all zero: {rep['drop_rows_all_selected_mask_px_zero']}",
            f"part_ov max after: {rep['part_ov_max_after']:.1f}",
            "",
            "```",
            rep["compare_table"],
            "```",
            "",
        ])
        if rep["diagnostic_zero_rows"]:
            preview = []
            for item in rep["diagnostic_zero_rows"][:12]:
                preview.append({
                    "part": item.get("part_name"),
                    "bucket": item.get("bucket"),
                    "selected_px": item.get("selected_mask_px"),
                    "all_px": item.get("all_mask_px"),
                    "old": item.get("old_filter_class"),
                    "new": item.get("new_filter_class"),
                    "gt": item.get("gt_voxel_nonempty"),
                })
            lines.extend(["zero-row diagnostic preview:", "```json", json.dumps(preview, indent=2, ensure_ascii=False), "```", ""])
    if row_list_audits:
        lines.extend(["## row-list filter audit", ""])
        audit_rows = []
        for name, audit in row_list_audits.items():
            audit_rows.append({
                "list": name,
                "rows": int(audit["matched_rows"]),
                "old_drop": int(audit["old_drop_rows"]),
                "new_drop": int(audit["new_drop_rows"]),
                "newly_drop": int(audit["newly_drop_rows"]),
                "new_drop_ratio": f"{float(audit['new_drop_ratio']):.2%}",
                "false_drop": int(audit["false_drop_selected_px_gt0"]),
                "drop_classes": json.dumps(audit["drop_counts_by_class"], ensure_ascii=False, sort_keys=True),
                "drop_buckets": json.dumps(audit["drop_counts_by_bucket"], ensure_ascii=False, sort_keys=True),
            })
        lines.extend([
            "```",
            format_table(audit_rows, ["list", "rows", "old_drop", "new_drop", "newly_drop", "new_drop_ratio", "false_drop", "drop_classes", "drop_buckets"]),
            "```",
            "",
        ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    dump_json(out_dir / "report.json", combined)
    print("\n".join(lines), flush=True)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "RUN_DIR"), required=True)
    parser.add_argument("--packed-dir", type=Path, default=Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5"))
    parser.add_argument("--step-dir", default="step_001000")
    parser.add_argument("--expected-views", type=int, default=12)
    parser.add_argument("--max-diag-rows", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=Path("/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/eval_filter_fairness_step001000"))
    parser.add_argument("--audit-row-lists", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for label, run_text in args.run:
        reports.append(report_for_run(args, Path(run_text), label))
    dump_json(args.out_dir / "combined_report.json", reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
