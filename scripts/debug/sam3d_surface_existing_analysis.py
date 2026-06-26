#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/PhysX-Mobility-full-4view-0511")
FULL_STAGE = Path("/robot/data-lab/jzh/art-gen-output/full-stage")
OUT_ROOT = Path("/robot/data-lab/jzh/arts-gen/surface_debug")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for meta_path in sorted(FULL_STAGE.glob("*/*/run_*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("mode") != "B":
            continue
        run_dir = meta_path.parent
        pred = run_dir / "voxel.npz"
        if not pred.is_file():
            continue
        object_id = str(meta.get("object_id") or run_dir.parent.name)
        angle_idx = int(meta.get("angle_idx", 0))
        sample_dir = OUT_ROOT / f"{object_id}_angle_{angle_idx}" / f"existing_{run_dir.name}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        surface = DATA_ROOT / "reconstruction/voxel_expanded" / object_id / f"angle_{angle_idx}" / "64/surface.npy"
        image = run_dir / "input_rgb/view_0.png"
        if image.is_file():
            shutil.copyfile(image, sample_dir / "input_image.png")
        shutil.copyfile(pred, sample_dir / "voxel.npz")
        compare_dir = sample_dir / "compare"
        _run_visualize(surface, pred, compare_dir, f"SAM3D existing {object_id} angle {angle_idx}")
        row = _metrics(surface, pred)
        row.update({"object_id": object_id, "angle_idx": angle_idx, "run_id": run_dir.name, "source": "existing_mode_B"})
        rows.append(row)
    _write_outputs(rows)
    return 0


def _run_visualize(surface: Path, pred: Path, out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/debug/visualize_sam3d_vs_surface.py",
            "--surface",
            str(surface),
            "--pred",
            str(pred),
            "--out-dir",
            str(out_dir),
            "--title",
            title,
        ],
        cwd=str(REPO),
        check=True,
    )


def _coords(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as data:
            arr = data["coords"]
    else:
        arr = np.load(path)
    return np.unique(np.asarray(arr, dtype=np.int64).reshape(-1, 3), axis=0)


def _bbox(coords: np.ndarray) -> dict[str, object]:
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    ext = mx - mn + 1
    center = (mn + mx) / 2.0
    return {
        "min": mn.tolist(),
        "max": mx.tolist(),
        "extent": ext.tolist(),
        "center": center.tolist(),
        "count": int(coords.shape[0]),
    }


def _metrics(surface: Path, pred: Path) -> dict[str, object]:
    gt = _coords(surface)
    pr = _coords(pred)
    gt_set = {tuple(x) for x in gt.tolist()}
    pr_set = {tuple(x) for x in pr.tolist()}
    overlap = len(gt_set & pr_set)
    gt_bbox = _bbox(gt)
    pred_bbox = _bbox(pr)
    gt_ext = np.asarray(gt_bbox["extent"], dtype=np.float64)
    pred_ext = np.asarray(pred_bbox["extent"], dtype=np.float64)
    ratio = pred_ext / np.maximum(gt_ext, 1.0)
    center_delta = np.asarray(pred_bbox["center"], dtype=np.float64) - np.asarray(gt_bbox["center"], dtype=np.float64)
    return {
        "surface_count": len(gt_set),
        "pred_count": len(pr_set),
        "overlap_count": overlap,
        "recall": overlap / len(gt_set) if gt_set else 0.0,
        "precision": overlap / len(pr_set) if pr_set else 0.0,
        "gt_extent": " ".join(map(str, gt_bbox["extent"])),
        "pred_extent": " ".join(map(str, pred_bbox["extent"])),
        "extent_ratio_xyz": " ".join(f"{v:.3f}" for v in ratio),
        "mean_extent_ratio": float(ratio.mean()),
        "center_delta_xyz": " ".join(f"{v:.3f}" for v in center_delta),
        "gt_min": " ".join(map(str, gt_bbox["min"])),
        "gt_max": " ".join(map(str, gt_bbox["max"])),
        "pred_min": " ".join(map(str, pred_bbox["min"])),
        "pred_max": " ".join(map(str, pred_bbox["max"])),
    }


def _write_outputs(rows: list[dict[str, object]]) -> None:
    fields = [
        "object_id",
        "angle_idx",
        "run_id",
        "source",
        "surface_count",
        "pred_count",
        "overlap_count",
        "recall",
        "precision",
        "gt_extent",
        "pred_extent",
        "extent_ratio_xyz",
        "mean_extent_ratio",
        "center_delta_xyz",
        "gt_min",
        "gt_max",
        "pred_min",
        "pred_max",
    ]
    with (OUT_ROOT / "existing_sam3d_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Existing SAM3D Surface Analysis", ""]
    lines.append("| object | run | pred | overlap | recall | precision | gt extent | pred extent | ratio | center delta |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['object_id']} | {r['run_id']} | {r['pred_count']} | {r['overlap_count']} | "
            f"{float(r['recall']):.4f} | {float(r['precision']):.4f} | {r['gt_extent']} | "
            f"{r['pred_extent']} | {r['extent_ratio_xyz']} | {r['center_delta_xyz']} |"
        )
    (OUT_ROOT / "existing_sam3d_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
