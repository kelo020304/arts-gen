#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/PhysX-Mobility-full-4view-0511")
OUT_ROOT = Path("/robot/data-lab/jzh/arts-gen/surface_debug")
WEIGHTS = Path("/robot/data-lab/jzh/art-gen/weights")
SAM3D_PY = Path(sys.executable)
SAM3D_SS = REPO / "submodules/sam3d-stage/infer_glue/ss_stage.py"
SAM3D_PIPELINE = WEIGHTS / "pipeline.yaml"
SAM3D_MOGE_MODEL = WEIGHTS / "hub/models--Ruicheng--moge-vitl/snapshots/979e84da9415762c30e6c0cf8dc0962896c793df/model.pt"
SAM3D_EXTRA_PYTHONPATHS = [
    REPO / "sam3d_cu118_deps/utils3d",
    REPO / "sam3d_cu118_src_deps/utils3d",
    REPO / "sam3d_cu118_deps/MoGe",
    REPO / "sam3d_cu118_src_deps/MoGe",
    REPO / "submodules/sam3d-stage/submodules/sam-3d-objects",
    REPO / "submodules/sam3d-stage/generate_surface_voxel",
    REPO / "submodules/sam3d-stage/generate_mask",
    REPO / "submodules/sam3d-stage/generate_texture",
]

SAMPLES = [
    ("102252", 0),
    ("100599", 0),
    ("103234", 0),
    ("102985", 0),
    ("100113", 0),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--mode", choices=["all", "metrics-only"], default="all")
    parser.add_argument("--limit", type=int, default=len(SAMPLES))
    parser.add_argument("--sam3d-python", type=Path, default=SAM3D_PY)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for object_id, angle_idx in SAMPLES[: args.limit]:
        sample_dir = args.out_root / f"{object_id}_angle_{angle_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        paths = _paths(object_id, angle_idx)
        _copy_inputs(sample_dir, paths)
        if not paths["surface"].is_file():
            failures.append({"sample": sample_dir.name, "stage": "input", "error": f"missing {paths['surface']}"})
            continue

        for method in ("sam3d",):
            pred = sample_dir / method / "voxel.npz"
            if args.mode == "all" and not pred.is_file():
                try:
                    _run_sam3d(sample_dir / method, paths["image"], paths["mask"], args.sam3d_python)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"sample": sample_dir.name, "stage": method, "error": str(exc)})
                    continue
            if pred.is_file():
                try:
                    row = _write_method_outputs(sample_dir, method, paths["surface"], pred)
                    row.update({"object_id": object_id, "angle_idx": angle_idx, "method": method})
                    all_rows.append(row)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"sample": sample_dir.name, "stage": f"{method}_metrics", "error": str(exc)})

    _write_tables(args.out_root, all_rows, failures)
    return 0 if not failures else 1


def _paths(object_id: str, angle_idx: int) -> dict[str, Path]:
    angle = f"angle_{angle_idx}"
    return {
        "image": DATA_ROOT / "renders" / object_id / angle / "rgb/view_0.png",
        "mask": DATA_ROOT / "renders" / object_id / angle / "mask/mask_0.png",
        "surface": DATA_ROOT / "reconstruction/voxel_expanded" / object_id / angle / "64/surface.npy",
        "camera": DATA_ROOT / "renders" / object_id / angle / "camera_transforms.json",
    }


def _copy_inputs(sample_dir: Path, paths: dict[str, Path]) -> None:
    for key in ("image", "mask", "surface", "camera"):
        src = paths[key]
        if src.is_file():
            dst = sample_dir / f"input_{key}{src.suffix}"
            if not dst.exists():
                shutil.copyfile(src, dst)


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.update(env)
    with log_path.open("ab") as log:
        log.write(("\n$ " + " ".join(cmd) + "\n").encode())
        log.flush()
        result = subprocess.run(cmd, cwd=str(cwd), env=merged_env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}; see {log_path}")


def _run_sam3d(out_dir: Path, image: Path, mask: Path, sam3d_python: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not sam3d_python.is_file():
        raise FileNotFoundError(f"missing SAM3D python: {sam3d_python}")
    pythonpath = [str(p) for p in SAM3D_EXTRA_PYTHONPATHS if p.exists()]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    env = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "LIDRA_SKIP_INIT": "true",
        "ATTN_BACKEND": "sdpa",
        "SPARSE_ATTN_BACKEND": "sdpa",
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "SAM3D_MOGE_MODEL_PATH": str(SAM3D_MOGE_MODEL),
    }
    _run(
        [
            str(sam3d_python),
            str(SAM3D_SS),
            "--image",
            str(image),
            "--mask",
            str(mask),
            "--config",
            str(SAM3D_PIPELINE),
            "--out",
            str(out_dir),
            "--device",
            "cuda:0",
        ],
        env=env,
        cwd=REPO,
        log_path=out_dir / "sam3d_ss.log",
    )


def _write_method_outputs(sample_dir: Path, method: str, surface: Path, pred: Path) -> dict[str, object]:
    compare_dir = sample_dir / method / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "scripts/debug/visualize_sam3d_vs_surface.py",
            "--surface",
            str(surface),
            "--pred",
            str(pred),
            "--out-dir",
            str(compare_dir),
            "--title",
            f"{method.upper()} vs GT surface",
        ],
        env={},
        cwd=REPO,
        log_path=compare_dir / "visualize.log",
    )
    surface_coords = _load_coords(surface)
    pred_coords = _load_coords(pred)
    stats = json.loads((compare_dir / "stats.json").read_text())
    bbox = {
        "gt": _bbox(surface_coords),
        "pred": _bbox(pred_coords),
    }
    aligned_coords, align_info = _bbox_align_coords(pred_coords, surface_coords)
    aligned_npz = sample_dir / method / "voxel_bbox_aligned.npz"
    np.savez_compressed(
        aligned_npz,
        coords=aligned_coords.astype(np.int32),
        resolution=np.int32(64),
        coord_frame="gt_bbox_aligned_grid",
        source=f"{method}_bbox_aligned_debug",
    )
    aligned_compare_dir = sample_dir / method / "compare_bbox_aligned"
    _run(
        [
            sys.executable,
            "scripts/debug/visualize_sam3d_vs_surface.py",
            "--surface",
            str(surface),
            "--pred",
            str(aligned_npz),
            "--out-dir",
            str(aligned_compare_dir),
            "--title",
            f"{method.upper()} bbox-aligned vs GT surface",
        ],
        env={},
        cwd=REPO,
        log_path=aligned_compare_dir / "visualize.log",
    )
    aligned_stats = json.loads((aligned_compare_dir / "stats.json").read_text())
    row = _flatten_stats(stats, bbox, aligned_stats, align_info)
    (sample_dir / method / "bbox.json").write_text(json.dumps(bbox, indent=2), encoding="utf-8")
    (sample_dir / method / "bbox_align_debug.json").write_text(
        json.dumps(align_info, indent=2), encoding="utf-8"
    )
    return row


def _load_coords(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as data:
            arr = data["coords"]
    else:
        arr = np.load(path)
    arr = np.asarray(arr, dtype=np.int64).reshape(-1, 3)
    return np.unique(arr, axis=0)


def _bbox(coords: np.ndarray) -> dict[str, object]:
    if coords.size == 0:
        return {"count": 0, "min": None, "max": None, "extent": None, "center": None}
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    ext = mx - mn + 1
    center = (mn + mx) / 2.0
    return {
        "count": int(coords.shape[0]),
        "min": mn.astype(int).tolist(),
        "max": mx.astype(int).tolist(),
        "extent": ext.astype(int).tolist(),
        "center": center.astype(float).round(3).tolist(),
    }


def _bbox_align_coords(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    pred_bbox = _bbox(pred)
    gt_bbox = _bbox(gt)
    if pred.size == 0 or gt.size == 0:
        return np.empty((0, 3), dtype=np.int64), {"scale": None, "offset": None}

    pred_min = np.asarray(pred_bbox["min"], dtype=np.float64)
    pred_max = np.asarray(pred_bbox["max"], dtype=np.float64)
    gt_min = np.asarray(gt_bbox["min"], dtype=np.float64)
    gt_max = np.asarray(gt_bbox["max"], dtype=np.float64)
    pred_span = np.maximum(pred_max - pred_min, 1.0)
    gt_span = np.maximum(gt_max - gt_min, 1.0)
    scale = gt_span / pred_span
    offset = gt_min - pred_min * scale
    aligned = np.rint(pred.astype(np.float64) * scale + offset).astype(np.int64)
    aligned = np.clip(aligned, 0, 63)
    aligned = np.unique(aligned, axis=0)
    return aligned, {
        "pred_min": pred_min.tolist(),
        "pred_max": pred_max.tolist(),
        "gt_min": gt_min.tolist(),
        "gt_max": gt_max.tolist(),
        "scale_xyz": scale.round(6).tolist(),
        "offset_xyz": offset.round(6).tolist(),
        "aligned_count": int(aligned.shape[0]),
    }


def _flatten_stats(
    stats: dict[str, object],
    bbox: dict[str, dict[str, object]],
    aligned_stats: dict[str, object],
    align_info: dict[str, object],
) -> dict[str, object]:
    gt = bbox["gt"]
    pred = bbox["pred"]
    gt_ext = np.asarray(gt["extent"], dtype=np.float64)
    pred_ext = np.asarray(pred["extent"], dtype=np.float64)
    ext_ratio = pred_ext / np.maximum(gt_ext, 1.0)
    center_delta = np.asarray(pred["center"], dtype=np.float64) - np.asarray(gt["center"], dtype=np.float64)
    return {
        "surface_count": int(stats["surface_count"]),
        "pred_count": int(stats["pred_count"]),
        "overlap_count": int(stats["overlap_count"]),
        "recall": float(stats["pred_recall_vs_surface"]),
        "precision": float(stats["pred_precision_vs_surface"]),
        "gt_extent": " ".join(map(str, gt["extent"])),
        "pred_extent": " ".join(map(str, pred["extent"])),
        "extent_ratio_xyz": " ".join(f"{v:.3f}" for v in ext_ratio.tolist()),
        "mean_extent_ratio": float(ext_ratio.mean()),
        "center_delta_xyz": " ".join(f"{v:.3f}" for v in center_delta.tolist()),
        "gt_min": " ".join(map(str, gt["min"])),
        "gt_max": " ".join(map(str, gt["max"])),
        "pred_min": " ".join(map(str, pred["min"])),
        "pred_max": " ".join(map(str, pred["max"])),
        "bbox_aligned_count": int(aligned_stats["pred_count"]),
        "bbox_aligned_overlap_count": int(aligned_stats["overlap_count"]),
        "bbox_aligned_recall": float(aligned_stats["pred_recall_vs_surface"]),
        "bbox_aligned_precision": float(aligned_stats["pred_precision_vs_surface"]),
        "bbox_align_scale_xyz": " ".join(f"{v:.3f}" for v in align_info["scale_xyz"]),
    }


def _write_tables(out_root: Path, rows: list[dict[str, object]], failures: list[dict[str, str]]) -> None:
    fieldnames = [
        "object_id",
        "angle_idx",
        "method",
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
        "bbox_aligned_count",
        "bbox_aligned_overlap_count",
        "bbox_aligned_recall",
        "bbox_aligned_precision",
        "bbox_align_scale_xyz",
    ]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Surface Scale Debug",
        "",
        "| object | angle | method | pred | overlap | recall | precision | gt extent | pred extent | extent ratio | center delta | bbox-align recall | bbox-align scale |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['object_id']} | {row['angle_idx']} | {row['method']} | {row['pred_count']} | "
            f"{row['overlap_count']} | {float(row['recall']):.4f} | {float(row['precision']):.4f} | "
            f"{row['gt_extent']} | {row['pred_extent']} | {row['extent_ratio_xyz']} | {row['center_delta_xyz']} |"
            f" {float(row['bbox_aligned_recall']):.4f} | {row['bbox_align_scale_xyz']} |"
        )
    if rows:
        mean_ratio = sum(float(row["mean_extent_ratio"]) for row in rows) / len(rows)
        mean_recall = sum(float(row["recall"]) for row in rows) / len(rows)
        mean_aligned_recall = sum(float(row["bbox_aligned_recall"]) for row in rows) / len(rows)
        lines.extend(
            [
                "",
                "## Scale Notes",
                "",
                f"- mean raw extent ratio pred/GT: `{mean_ratio:.3f}`",
                f"- mean raw recall: `{mean_recall:.4f}`",
                f"- mean bbox-aligned recall: `{mean_aligned_recall:.4f}`",
                "- GT `surface.npy` is in the dataset 64-grid frame; in these samples its bbox center is close to `[31.5, 31.5, 31.5]` and often one axis spans the full grid.",
                "- SAM3D Stage-A uses image-mask cropping plus ObjectCentricSSI pointmap normalization, then writes coords as a canonical 64-grid without applying the dataset reconstruction transform.",
                "- Therefore raw SAM3D coords and GT part/surface coords should not be mixed unless they are registered into the same grid frame.",
            ]
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['sample']}` `{failure['stage']}`: {failure['error']}")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
