#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from part_ss_eval_platform.eval_0617_1 import _load_datasets  # noqa: E402


def _safe_name(value: str, max_len: int = 96) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value)).strip("_")
    return (out or "item")[:max_len]


def _prefix(dataset_id: str, object_id: str, angle: int) -> str:
    return f"{dataset_id}__{_safe_name(object_id)}__angle_{int(angle):02d}"


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_coords(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        coords = np.asarray(data["coords"], dtype=np.int64).reshape(-1, 3)
    valid = np.all((coords >= 0) & (coords < 64), axis=1)
    return np.unique(coords[valid], axis=0).astype(np.int32, copy=False)


def _coords_to_occ(coords: np.ndarray) -> np.ndarray:
    occ = np.zeros((64, 64, 64), dtype=bool)
    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if arr.size:
        valid = np.all((arr >= 0) & (arr < 64), axis=1)
        arr = arr[valid]
        occ[arr[:, 0], arr[:, 1], arr[:, 2]] = True
    return occ


def _occ_to_coords(occ: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(occ, dtype=bool)).astype(np.int32, copy=False)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return float((a & b).sum() / union)


def _shift_label_and_mask(labels: np.ndarray, mask: np.ndarray, dx: int, dy: int, dz: int) -> tuple[np.ndarray, np.ndarray]:
    shifted_labels = np.zeros_like(labels, dtype=labels.dtype)
    shifted_mask = np.zeros_like(mask, dtype=bool)
    src = [
        slice(max(0, -dx), labels.shape[0] - max(0, dx)),
        slice(max(0, -dy), labels.shape[1] - max(0, dy)),
        slice(max(0, -dz), labels.shape[2] - max(0, dz)),
    ]
    dst = [
        slice(max(0, dx), labels.shape[0] - max(0, -dx)),
        slice(max(0, dy), labels.shape[1] - max(0, -dy)),
        slice(max(0, dz), labels.shape[2] - max(0, -dz)),
    ]
    shifted_labels[tuple(dst)] = labels[tuple(src)]
    shifted_mask[tuple(dst)] = mask[tuple(src)]
    return shifted_labels, shifted_mask


def _dilate26(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(radius))):
        grown = out.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    shifted, _ = _shift_label_and_mask(out.astype(np.int8), np.ones_like(out, dtype=bool), dx, dy, dz)
                    grown |= shifted.astype(bool)
        out = grown
    return out


def _interface_band(labels: np.ndarray, support: np.ndarray, radius: int = 2) -> np.ndarray:
    boundary = np.zeros_like(support, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                shifted_labels, shifted_support = _shift_label_and_mask(labels, support, dx, dy, dz)
                boundary |= support & shifted_support & (shifted_labels != labels)
    return _dilate26(boundary, int(radius)) & support


def _label_map(whole_occ: np.ndarray, part_occs: list[np.ndarray]) -> np.ndarray:
    labels = np.zeros((64, 64, 64), dtype=np.int16)
    labels[~whole_occ] = -1
    for idx, occ in enumerate(part_occs, start=1):
        labels[np.asarray(occ, dtype=bool) & whole_occ] = int(idx)
    return labels


def _unique_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.unique(arr, axis=0).astype(np.int32, copy=False)


def _connected_components(coords: np.ndarray) -> list[np.ndarray]:
    arr = _unique_coords(coords).astype(np.int64, copy=False)
    if arr.size == 0:
        return []
    coord_set = {tuple(map(int, row)) for row in arr.tolist()}
    seen: set[tuple[int, int, int]] = set()
    comps: list[np.ndarray] = []
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for start in coord_set:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int, int]] = []
        while stack:
            x, y, z = stack.pop()
            comp.append((x, y, z))
            for dx, dy, dz in neighbors:
                nxt = (x + dx, y + dy, z + dz)
                if nxt in coord_set and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(np.asarray(comp, dtype=np.int32))
    comps.sort(key=lambda item: int(item.shape[0]), reverse=True)
    return comps


def _part_overlap(part_occs: list[np.ndarray]) -> int:
    counts = np.zeros((64, 64, 64), dtype=np.uint16)
    for occ in part_occs:
        counts += np.asarray(occ, dtype=bool)
    return int((counts > 1).sum())


def _find_sample(ds: Any, object_id: str, angle: int) -> dict[str, Any]:
    for row in ds.samples:
        if str(row["obj_id"]) == str(object_id) and int(row["angle_idx"]) == int(angle):
            return row
    raise KeyError(f"{object_id} angle={angle} not found")


def _selection_rows(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path, {}) or {}
    rows: list[dict[str, Any]] = []
    for split in ("train", "held"):
        for item in data.get("samples", {}).get(split, []):
            rows.append(
                {
                    "split": split,
                    "dataset_id": str(item.get("dataset_id")),
                    "object_id": str(item.get("object_id") or item.get("obj_id")),
                    "angle": int(item.get("angle", item.get("angle_idx", 0))),
                    "part_count": int(item.get("part_count", 0)),
                    "reason": str(item.get("selected_reason") or item.get("priority_bucket") or ""),
                }
            )
    if rows:
        return rows
    raise RuntimeError(f"{path}: no samples found")


def _load_method(method_dir: Path, row: dict[str, Any], part_count: int) -> dict[str, Any]:
    prefix = _prefix(row["dataset_id"], row["object_id"], int(row["angle"]))
    summary_path = method_dir / f"{prefix}__summary.json"
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        raise FileNotFoundError(f"missing summary: {summary_path}")
    run_dir = Path(str(summary["run_dir"]))
    whole_occ = _coords_to_occ(_load_coords(run_dir / "voxel.npz"))
    part_occs = []
    component_counts = []
    largest_fractions = []
    for idx in range(int(part_count)):
        coords = _load_coords(run_dir / "parts" / f"part_{idx:02d}_voxel.npz")
        comps = _connected_components(coords)
        component_counts.append(int(len(comps)))
        largest = int(comps[0].shape[0]) if comps else 0
        largest_fractions.append(float(largest / max(1, int(coords.shape[0]))))
        part_occs.append(_coords_to_occ(coords) & whole_occ)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "run_dir": run_dir,
        "whole_occ": whole_occ,
        "part_occs": part_occs,
        "part_overlap": _part_overlap(part_occs),
        "component_counts": component_counts,
        "largest_component_fractions": largest_fractions,
        "diagnostic_png": Path(str(summary.get("diagnostic_png") or method_dir / f"{prefix}__diagnostic.png")),
    }


def _metrics_for(
    *,
    pred: dict[str, Any],
    gt_part_occs: list[np.ndarray],
) -> dict[str, Any]:
    whole_occ = np.asarray(pred["whole_occ"], dtype=bool)
    pred_part_occs = [np.asarray(occ, dtype=bool) & whole_occ for occ in pred["part_occs"]]
    gt_part_occs = [np.asarray(occ, dtype=bool) & whole_occ for occ in gt_part_occs]
    pred_union = np.logical_or.reduce(pred_part_occs) if pred_part_occs else np.zeros((64, 64, 64), dtype=bool)
    gt_union = np.logical_or.reduce(gt_part_occs) if gt_part_occs else np.zeros((64, 64, 64), dtype=bool)
    pred_body = whole_occ & ~pred_union
    gt_body = whole_occ & ~gt_union
    pred_labels = _label_map(whole_occ, pred_part_occs)
    gt_labels = _label_map(whole_occ, gt_part_occs)
    band = _interface_band(gt_labels, whole_occ, radius=2)
    boundary_total = int(band.sum())
    boundary_acc = float((pred_labels[band] == gt_labels[band]).sum() / boundary_total) if boundary_total else 1.0
    part_ious = [_iou(pred_part_occs[idx], gt_part_occs[idx]) for idx in range(len(gt_part_occs))]
    return {
        "cell_iou": _iou(pred_union, gt_union),
        "body_iou": _iou(pred_body, gt_body),
        "boundary_band_acc": boundary_acc,
        "boundary_band_voxels": boundary_total,
        "part_ov": int(pred["part_overlap"]),
        "part_iou_mean": float(np.mean(part_ious)) if part_ious else 1.0,
        "component_count_mean": float(np.mean(pred["component_counts"])) if pred["component_counts"] else 0.0,
        "largest_component_fraction_mean": (
            float(np.mean(pred["largest_component_fractions"])) if pred["largest_component_fractions"] else 1.0
        ),
        "pred_part_union_voxels": int(pred_union.sum()),
        "gt_part_union_voxels": int(gt_union.sum()),
        "body_voxels": int(pred_body.sum()),
    }


def _side_by_side(old_png: Path, t0_png: Path, out_png: Path, title: str) -> None:
    images = []
    for label, path in (("old", old_png), ("old+T0", t0_png)):
        if path.is_file():
            image = Image.open(path).convert("RGB")
        else:
            image = Image.new("RGB", (800, 470), (245, 245, 245))
            ImageDraw.Draw(image).text((20, 20), f"missing: {path}", fill=(120, 0, 0))
        image.thumbnail((800, 470), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (800, 500), (255, 255, 255))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 800, 30), fill=(0, 0, 0))
        draw.text((8, 9), label, fill=(255, 255, 255))
        tile.paste(image, ((800 - image.width) // 2, 30 + (470 - image.height) // 2))
        images.append(tile)
    canvas = Image.new("RGB", (1600, 530), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:180], fill=(0, 0, 0))
    canvas.paste(images[0], (0, 30))
    canvas.paste(images[1], (800, 30))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _compare_label(delta: float, eps: float = 1.0e-4) -> str:
    if delta > eps:
        return "better"
    if delta < -eps:
        return "worse"
    return "same"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare EE part-seg old vs old+T0 outputs on the same selection.")
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument("--boundary-radius", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.old_dir = args.old_dir.resolve()
    args.t0_dir = args.t0_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    old_config = _read_json(args.old_dir / "run_config.json", {}) or {}
    data_config = args.data_config or Path(str(old_config.get("data_config")))
    split_json = args.split_json or Path(str(old_config.get("split_json")))
    selection_json = args.selection_json or (args.old_dir / "selection.json")
    rows = _selection_rows(selection_json)
    datasets, _dataset_meta = _load_datasets(SimpleNamespace(data_config=data_config, split_json=split_json))

    detail_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        ds = datasets[row["dataset_id"]]
        ds_sample = _find_sample(ds, row["object_id"], int(row["angle"]))
        gt_part_occs = [
            _coords_to_occ(ds._load_raw_ind_coords(ds_sample, part).numpy().astype(np.int64))
            for part in ds_sample["parts"]
        ]
        row["part_count"] = int(len(ds_sample["parts"]))
        old = _load_method(args.old_dir, row, int(row["part_count"]))
        t0 = _load_method(args.t0_dir, row, int(row["part_count"]))
        old_metrics = _metrics_for(pred=old, gt_part_occs=gt_part_occs)
        t0_metrics = _metrics_for(pred=t0, gt_part_occs=gt_part_occs)
        prefix = _prefix(row["dataset_id"], row["object_id"], int(row["angle"]))
        side_by_side = args.out_dir / "diagnostics" / f"{idx:02d}__{prefix}__old_vs_t0.png"
        _side_by_side(
            Path(old["diagnostic_png"]),
            Path(t0["diagnostic_png"]),
            side_by_side,
            f"{idx:02d} {row['dataset_id']}::{row['object_id']} angle={row['angle']}",
        )
        detail = {
            **row,
            "idx": int(idx),
            "old": old_metrics,
            "old_t0": t0_metrics,
            "delta_cell_iou": float(t0_metrics["cell_iou"] - old_metrics["cell_iou"]),
            "delta_body_iou": float(t0_metrics["body_iou"] - old_metrics["body_iou"]),
            "delta_boundary_band_acc": float(t0_metrics["boundary_band_acc"] - old_metrics["boundary_band_acc"]),
            "cell_iou_verdict": _compare_label(float(t0_metrics["cell_iou"] - old_metrics["cell_iou"])),
            "diagnostic_png": str(side_by_side),
            "old_summary": str(old["summary_path"]),
            "t0_summary": str(t0["summary_path"]),
        }
        detail_rows.append(detail)
        diag_rows.append(
            {
                "idx": str(idx),
                "dataset_id": row["dataset_id"],
                "object_id": row["object_id"],
                "angle": str(row["angle"]),
                "diagnostic_png": str(side_by_side),
            }
        )

    old_means = {
        key: float(np.mean([float(row["old"][key]) for row in detail_rows]))
        for key in ("cell_iou", "body_iou", "boundary_band_acc", "part_iou_mean", "part_ov")
    }
    t0_means = {
        key: float(np.mean([float(row["old_t0"][key]) for row in detail_rows]))
        for key in ("cell_iou", "body_iou", "boundary_band_acc", "part_iou_mean", "part_ov")
    }
    verdict_counts = {
        name: int(sum(1 for row in detail_rows if row["cell_iou_verdict"] == name))
        for name in ("better", "same", "worse")
    }
    boundary_better = int(sum(1 for row in detail_rows if float(row["delta_boundary_band_acc"]) > 1.0e-4))
    boundary_worse = int(sum(1 for row in detail_rows if float(row["delta_boundary_band_acc"]) < -1.0e-4))
    recommend = verdict_counts["worse"] == 0 and t0_means["boundary_band_acc"] >= old_means["boundary_band_acc"]
    report = {
        "old_dir": str(args.old_dir),
        "old_t0_dir": str(args.t0_dir),
        "selection_json": str(selection_json),
        "data_config": str(data_config),
        "split_json": str(split_json),
        "old_ckpt": ((old_config.get("part_stage") or {}).get("ckpt") or ""),
        "old_t0_ckpt": (((_read_json(args.t0_dir / "run_config.json", {}) or {}).get("part_stage") or {}).get("ckpt") or ""),
        "count": int(len(detail_rows)),
        "old_means": old_means,
        "old_t0_means": t0_means,
        "cell_iou_verdict_counts": verdict_counts,
        "boundary_better": boundary_better,
        "boundary_worse": boundary_worse,
        "recommend_t0": bool(recommend),
        "details": detail_rows,
        "diagnostics": diag_rows,
    }
    _write_json(args.out_dir / "comparison.json", report)

    lines = [
        "# EE part-seg old vs old+T0",
        "",
        f"- old dir: `{args.old_dir}`",
        f"- old+T0 dir: `{args.t0_dir}`",
        f"- selection: `{selection_json}`",
        f"- old ckpt: `{report['old_ckpt']}`",
        f"- old+T0 ckpt: `{report['old_t0_ckpt']}`",
        f"- verdict by cell_iou: better={verdict_counts['better']} same={verdict_counts['same']} worse={verdict_counts['worse']}",
        f"- boundary band: better={boundary_better} worse={boundary_worse}",
        f"- T0 recommendation: `{'recommend' if recommend else 'do_not_default'}`",
        "",
        "## Aggregate",
        "",
        "| metric | old | old+T0 | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ("cell_iou", "body_iou", "boundary_band_acc", "part_iou_mean", "part_ov"):
        lines.append(f"| {key} | {_fmt(old_means[key])} | {_fmt(t0_means[key])} | {_fmt(t0_means[key] - old_means[key])} |")
    lines += [
        "",
        "## Per Object",
        "",
        "| # | dataset | object | cell old | cell T0 | d_cell | body old | body T0 | boundary old | boundary T0 | part_ov old | part_ov T0 | verdict | diagnostic |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in detail_rows:
        lines.append(
            "| {idx} | {dataset} | `{obj}` | {old_cell} | {t0_cell} | {d_cell} | {old_body} | {t0_body} | "
            "{old_boundary} | {t0_boundary} | {old_ov} | {t0_ov} | {verdict} | `{diag}` |".format(
                idx=row["idx"],
                dataset=row["dataset_id"],
                obj=row["object_id"],
                old_cell=_fmt(row["old"]["cell_iou"]),
                t0_cell=_fmt(row["old_t0"]["cell_iou"]),
                d_cell=_fmt(row["delta_cell_iou"]),
                old_body=_fmt(row["old"]["body_iou"]),
                t0_body=_fmt(row["old_t0"]["body_iou"]),
                old_boundary=_fmt(row["old"]["boundary_band_acc"]),
                t0_boundary=_fmt(row["old_t0"]["boundary_band_acc"]),
                old_ov=int(row["old"]["part_ov"]),
                t0_ov=int(row["old_t0"]["part_ov"]),
                verdict=row["cell_iou_verdict"],
                diag=row["diagnostic_png"],
            )
        )
    lines += [
        "",
        "## 22367",
        "",
    ]
    match_22367 = [row for row in detail_rows if row["object_id"] == "22367"]
    if match_22367:
        item = match_22367[0]
        lines.append(f"- old vs old+T0 diagnostic: `{item['diagnostic_png']}`")
        lines.append(
            f"- cell_iou {_fmt(item['old']['cell_iou'])} -> {_fmt(item['old_t0']['cell_iou'])}, "
            f"boundary {_fmt(item['old']['boundary_band_acc'])} -> {_fmt(item['old_t0']['boundary_band_acc'])}"
        )
    else:
        lines.append("- 22367 not present in selection.")
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare_ee_partseg_t0] report={args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
