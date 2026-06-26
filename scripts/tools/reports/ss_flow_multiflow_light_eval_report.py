#!/usr/bin/env python3
"""Lightweight 1024-sample multiflow eval and final report.

This is a disk-frugal companion for ss_flow_single_multiflow_16obj_eval.py.
It can reuse already-written per-sample summaries/coords, runs missing samples
without saving per-sample latents, and keeps only enough coords to render the
global best/worst examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.render.render_voxel_eval_tripanel_flat import render_tripanel  # noqa: E402
from scripts.tools.reports.ss_flow_single_multiflow_16obj_eval import (  # noqa: E402
    DEFAULT_DECODER_CKPT,
    DEFAULT_MODEL_CONFIG,
    coords_metrics,
    decode_latent_to_coords,
    instantiate_ss_decoder,
    instantiate_ss_flow,
    load_gt_surface,
    load_samples_from_config,
    load_tokens,
    require_file,
    sample_multiflow,
)
from scripts.tools.reports.summarize_ss_flow_eval_best_worst_bins import (  # noqa: E402
    build_iou_bins,
    draw_hist_png,
    draw_table_png,
)


def sample_stem(sample: dict[str, Any]) -> str:
    return f"{sample['object_id']}_angle_{int(sample['angle_idx']):02d}"


def row_sort_key(row: dict[str, Any]) -> tuple[float, int, str, int]:
    return (
        float(row["iou"]),
        -int(row.get("target_part_count", 0) or 0),
        str(row["object_id"]),
        int(row["angle_idx"]),
    )


def row_id(row: dict[str, Any]) -> str:
    return f"{row['object_id']}:angle_{int(row['angle_idx'])}"


def make_row(data: dict[str, Any], summary_path: Path | None = None) -> dict[str, Any]:
    metrics = data.get("metrics_vs_gt_surface", data)
    row = {
        "object_id": str(data["object_id"]),
        "angle_idx": int(data["angle_idx"]),
        "category": data.get("category"),
        "name": data.get("name"),
        "target_part_count": int(data.get("target_part_count", 0) or 0),
        "view_indices": data.get("view_indices"),
        "iou": float(metrics.get("iou", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "intersection": int(metrics.get("intersection", 0.0)),
        "pred_voxels": int(metrics.get("pred_voxels", 0.0)),
        "gt_voxels": int(metrics.get("gt_voxels", 0.0)),
    }
    if summary_path is not None:
        row["summary_path"] = str(summary_path.resolve())
    return row


class CandidateKeeper:
    def __init__(self, keep_n: int) -> None:
        self.keep_n = int(keep_n)
        self.worst: list[dict[str, Any]] = []
        self.best: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any], gt: np.ndarray, pred: np.ndarray) -> None:
        item = {
            "row": dict(row),
            "gt": np.ascontiguousarray(gt.astype(np.int64, copy=False)),
            "pred": np.ascontiguousarray(pred.astype(np.int64, copy=False)),
        }
        self.worst.append(item)
        self.worst.sort(key=lambda x: row_sort_key(x["row"]))
        self.worst = self.worst[: self.keep_n]
        self.best.append(item)
        self.best.sort(key=lambda x: row_sort_key(x["row"]))
        self.best = self.best[-self.keep_n :]

    def write(self, out_dir: Path) -> dict[str, Any]:
        cand_dir = out_dir / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"best": [], "worst": []}
        groups = {
            "worst": self.worst,
            "best": list(reversed(self.best)),
        }
        for group, items in groups.items():
            for rank, item in enumerate(items, start=1):
                row = item["row"]
                stem = f"{group}{rank:02d}_{row['object_id']}_angle{int(row['angle_idx']):02d}"
                coords_path = cand_dir / f"{stem}.npz"
                np.savez_compressed(coords_path, gt=item["gt"], pred=item["pred"])
                rec = {
                    "rank": rank,
                    "row": row,
                    "coords_path": str(coords_path.resolve()),
                }
                payload[group].append(rec)
        (out_dir / "candidates.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload


def load_existing_sample(sample: dict[str, Any], existing_root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray] | None:
    sample_dir = existing_root / sample_stem(sample)
    summary_path = sample_dir / "summary.json"
    gt_path = sample_dir / "gt_surface_coords.npy"
    pred_path = sample_dir / "pred_multiflow_coords.npy"
    if not (summary_path.is_file() and gt_path.is_file() and pred_path.is_file()):
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    row = make_row(data, summary_path)
    gt = np.load(gt_path)
    pred = np.load(pred_path)
    return row, gt, pred


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_shard(args: argparse.Namespace) -> None:
    args.config = require_file(args.config, "eval config")
    args.model_config = require_file(args.model_config, "SS flow model config")
    args.ss_decoder_ckpt = require_file(args.ss_decoder_ckpt, "SS decoder checkpoint")
    args.ss_flow_ckpt = require_file(args.ss_flow_ckpt, "SS flow checkpoint")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, samples = load_samples_from_config(args.config)

    rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, int]] = set()
    keeper = CandidateKeeper(args.keep_candidates)

    if args.existing_root:
        existing_root = args.existing_root.resolve()
        for sample in samples:
            loaded = load_existing_sample(sample, existing_root)
            if loaded is None:
                continue
            row, gt, pred = loaded
            rows.append(row)
            done_keys.add((row["object_id"], int(row["angle_idx"])))
            keeper.add(row, gt, pred)
        print(f"[reuse] {len(done_keys)} completed samples from {existing_root}", flush=True)

    todo = [
        sample for sample in samples
        if (str(sample["object_id"]), int(sample["angle_idx"])) not in done_keys
    ]
    print(f"[select] total={len(samples)} reuse={len(done_keys)} todo={len(todo)} config={args.config}", flush=True)

    if todo:
        if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
            raise RuntimeError(f"CUDA requested ({args.device}) but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(int(args.seed))
        torch.manual_seed(int(args.seed))
        model = instantiate_ss_flow(args.ss_flow_ckpt, args.model_config, device)
        decoder = instantiate_ss_decoder(args.ss_decoder_ckpt, device)
        for idx, sample in enumerate(todo, start=1):
            tokens = load_tokens(sample["token_path"], sample["view_indices"], device)
            pred_latent = sample_multiflow(
                model,
                tokens,
                seed=int(args.seed),
                steps=int(args.steps),
                cfg_strength=float(args.cfg_strength),
                sigma_min=float(args.sigma_min),
            )
            pred, pred_stats = decode_latent_to_coords(
                decoder,
                pred_latent,
                threshold=float(args.decode_threshold),
            )
            if pred.shape[0] == 0:
                raise RuntimeError(f"{sample['object_id']} angle_{sample['angle_idx']}: predicted coords are empty")
            gt = load_gt_surface(sample["surface_path"])
            metrics = coords_metrics(pred, gt)
            row = {
                "object_id": str(sample["object_id"]),
                "angle_idx": int(sample["angle_idx"]),
                "manifest_line": int(sample["manifest_line"]),
                "category": sample.get("category"),
                "name": sample.get("name"),
                "target_part_count": int(sample.get("target_part_count", 0) or 0),
                "view_indices": [int(v) for v in sample["view_indices"]],
                "iou": float(metrics["iou"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "intersection": int(metrics["intersection"]),
                "pred_voxels": int(metrics["pred_voxels"]),
                "gt_voxels": int(metrics["gt_voxels"]),
                "pred_stats": pred_stats,
            }
            rows.append(row)
            keeper.add(row, gt, pred)
            if idx == 1 or idx % 25 == 0 or idx == len(todo):
                print(
                    f"[sample] {idx}/{len(todo)} {row['object_id']} angle_{row['angle_idx']} "
                    f"IoU={row['iou']:.4f} P={row['precision']:.4f} R={row['recall']:.4f}",
                    flush=True,
                )

    rows.sort(key=lambda row: (str(row["object_id"]), int(row["angle_idx"])))
    write_jsonl(args.out_dir / "metrics.jsonl", rows)
    candidates = keeper.write(args.out_dir)
    ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    summary = {
        "config": str(args.config.resolve()),
        "ss_flow_ckpt": str(args.ss_flow_ckpt.resolve()),
        "sample_count": len(rows),
        "reused_count": len(done_keys),
        "new_count": len(todo),
        "mean_iou": float(ious.mean()) if len(ious) else None,
        "median_iou": float(np.median(ious)) if len(ious) else None,
        "metrics_jsonl": str((args.out_dir / "metrics.jsonl").resolve()),
        "candidates_json": str((args.out_dir / "candidates.json").resolve()),
        "candidate_counts": {key: len(value) for key, value in candidates.items()},
    }
    (args.out_dir / "shard_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_candidate_map(shard_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for shard_dir in shard_dirs:
        path = shard_dir / "candidates.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for group in ("best", "worst"):
            for item in data.get(group, []):
                row = item["row"]
                out[row_id(row)] = item
    return out


def render_selected(
    rows: list[dict[str, Any]],
    candidate_map: dict[str, dict[str, Any]],
    out_dir: Path,
    best_n: int,
    worst_n: int,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=row_sort_key)
    selected: list[tuple[str, int, dict[str, Any]]] = []
    for rank, row in enumerate(sorted_rows[:worst_n], start=1):
        selected.append(("worst", rank, row))
    for rank, row in enumerate(reversed(sorted_rows[-best_n:]), start=1):
        selected.append(("best", rank, row))

    rendered: list[dict[str, Any]] = []
    for group, rank, row in selected:
        item = candidate_map.get(row_id(row))
        if item is None:
            raise RuntimeError(f"missing candidate coords for selected row {row_id(row)}")
        with np.load(item["coords_path"]) as data:
            gt = np.asarray(data["gt"])
            pred = np.asarray(data["pred"])
        stem = f"{row['object_id']}_angle{int(row['angle_idx']):02d}"
        dst_stem = f"{group}{rank:02d}_{stem}_iou{row['iou']:.4f}_parts{row['target_part_count']}"
        png_path = out_dir / f"{dst_stem}.png"
        json_path = out_dir / f"{dst_stem}.json"
        metrics = {
            "iou": row["iou"],
            "precision": row["precision"],
            "recall": row["recall"],
            "intersection": row["intersection"],
            "pred_voxels": row["pred_voxels"],
            "gt_voxels": row["gt_voxels"],
        }
        render_tripanel(
            gt,
            pred,
            title=f"{row['object_id']} angle_{int(row['angle_idx'])} voxel blocks",
            metrics=metrics,
            out_path=png_path,
            width=width,
            height=height,
        )
        payload = {
            **row,
            "selection_group": group,
            "selection_rank": rank,
            "render_type": "tripanel_true_voxel_blocks",
            "panels": ["GT", "Pred", "Overlay"],
            "color_legend": {
                "blue": "GT only",
                "red": "Pred only",
                "green": "GT and Pred overlap",
            },
            "metrics_vs_gt_surface": metrics,
            "png_path": str(png_path.resolve()),
            "json_path": str(json_path.resolve()),
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rendered.append(payload)
    return rendered


def finalize(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [path.resolve() for path in args.shard_dir]
    rows: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        rows.extend(load_jsonl(shard_dir / "metrics.jsonl"))
    if args.expected_count and len(rows) != int(args.expected_count):
        raise RuntimeError(f"expected {args.expected_count} rows, got {len(rows)}")
    rows.sort(key=row_sort_key)
    candidate_map = load_candidate_map(shard_dirs)
    selection_report = None
    if args.selection_report and args.selection_report.is_file():
        selection_report = json.loads(args.selection_report.read_text(encoding="utf-8"))

    write_jsonl(args.out_dir / "metrics_all.jsonl", rows)
    bins = build_iou_bins(rows)
    table_png = args.out_dir / "iou_bins_table.png"
    hist_png = args.out_dir / "iou_histogram.png"
    draw_table_png(table_png, rows=rows, bins=bins, selection_report=selection_report, ckpt=str(args.ckpt))
    draw_hist_png(hist_png, rows)
    rendered = render_selected(
        rows,
        candidate_map,
        args.out_dir,
        best_n=int(args.best_n),
        worst_n=int(args.worst_n),
        width=int(args.width),
        height=int(args.height),
    )
    ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    ps = np.asarray([row["precision"] for row in rows], dtype=np.float64)
    rs = np.asarray([row["recall"] for row in rows], dtype=np.float64)
    summary = {
        "eval_root": str(args.out_dir.resolve()),
        "sample_count": len(rows),
        "ckpt": str(args.ckpt),
        "metrics_vs_gt_surface": {
            "mean_iou": float(ious.mean()),
            "median_iou": float(np.median(ious)),
            "mean_precision": float(ps.mean()),
            "mean_recall": float(rs.mean()),
            "min_iou": float(ious.min()),
            "max_iou": float(ious.max()),
        },
        "iou_bins": bins,
        "rendered_best_worst": rendered,
        "table_png": str(table_png.resolve()),
        "histogram_png": str(hist_png.resolve()),
        "metrics_jsonl": str((args.out_dir / "metrics_all.jsonl").resolve()),
        "selection_report": selection_report,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "iou_bins_table.json").write_text(
        json.dumps({"iou_bins": bins}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sample_count": len(rows),
        "mean_iou": summary["metrics_vs_gt_surface"]["mean_iou"],
        "median_iou": summary["metrics_vs_gt_surface"]["median_iou"],
        "out_dir": str(args.out_dir),
        "renders": len(rendered),
    }, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    eval_p = sub.add_parser("eval-shard")
    eval_p.add_argument("--config", type=Path, required=True)
    eval_p.add_argument("--ss-flow-ckpt", type=Path, required=True)
    eval_p.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    eval_p.add_argument("--ss-decoder-ckpt", type=Path, default=DEFAULT_DECODER_CKPT)
    eval_p.add_argument("--existing-root", type=Path, default=None)
    eval_p.add_argument("--out-dir", type=Path, required=True)
    eval_p.add_argument("--device", default="cuda:0")
    eval_p.add_argument("--seed", type=int, default=20260610)
    eval_p.add_argument("--steps", type=int, default=20)
    eval_p.add_argument("--cfg-strength", type=float, default=7.5)
    eval_p.add_argument("--sigma-min", type=float, default=1.0e-5)
    eval_p.add_argument("--decode-threshold", type=float, default=0.0)
    eval_p.add_argument("--keep-candidates", type=int, default=20)

    fin_p = sub.add_parser("finalize")
    fin_p.add_argument("--shard-dir", type=Path, action="append", required=True)
    fin_p.add_argument("--out-dir", type=Path, required=True)
    fin_p.add_argument("--ckpt", type=Path, required=True)
    fin_p.add_argument("--selection-report", type=Path, default=None)
    fin_p.add_argument("--expected-count", type=int, default=1024)
    fin_p.add_argument("--best-n", type=int, default=10)
    fin_p.add_argument("--worst-n", type=int, default=10)
    fin_p.add_argument("--width", type=int, default=2100)
    fin_p.add_argument("--height", type=int, default=860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "eval-shard":
        eval_shard(args)
    elif args.cmd == "finalize":
        finalize(args)
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
