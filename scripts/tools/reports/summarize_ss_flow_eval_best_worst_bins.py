#!/usr/bin/env python3
"""Write flat best/worst voxel renders and IoU-bin table for SS-flow eval."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.render.render_voxel_eval_tripanel_flat import convert_one  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--best-n", type=int, default=10)
    parser.add_argument("--worst-n", type=int, default=10)
    parser.add_argument("--width", type=int, default=2100)
    parser.add_argument("--height", type=int, default=860)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_summaries(eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_root.glob("shard*/*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics_vs_gt_surface", {})
        row = {
            "summary_path": str(path.resolve()),
            "sample_dir": str(path.parent.resolve()),
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
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no summaries found under {eval_root}/shard*/*/summary.json")
    return rows


def bin_label(index: int) -> str:
    if index == 9:
        return "90-100"
    return f"{index * 10}-{(index + 1) * 10}"


def build_iou_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for idx in range(10):
        lo = idx / 10.0
        hi = 1.0 if idx == 9 else (idx + 1) / 10.0
        if idx == 9:
            members = [row for row in rows if lo <= row["iou"] <= hi]
        else:
            members = [row for row in rows if lo <= row["iou"] < hi]
        members = sorted(members, key=lambda row: (row["iou"], -row["target_part_count"], row["object_id"], row["angle_idx"]))
        worst = members[0] if members else None
        best = members[-1] if members else None
        bins.append({
            "bin": bin_label(idx),
            "range": [lo, hi],
            "count": len(members),
            "worst": worst,
            "best": best,
        })
    return bins


def draw_table_png(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    selection_report: dict[str, Any] | None,
    ckpt: str | None,
) -> None:
    width = 1800
    row_h = 54
    top_h = 188
    header_h = 42
    height = top_h + header_h + row_h * len(bins) + 38
    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30)
    font = load_font(20)
    small = load_font(16)
    tiny = load_font(14)
    ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    ps = np.asarray([row["precision"] for row in rows], dtype=np.float64)
    rs = np.asarray([row["recall"] for row in rows], dtype=np.float64)
    parts = Counter(int(row["target_part_count"]) for row in rows)
    draw.text((28, 22), "SS Flow Eval IoU Distribution", fill=(0, 0, 0), font=title_font)
    draw.text(
        (28, 66),
        (
            f"N={len(rows)}  mean IoU={ious.mean():.4f}  median={np.median(ious):.4f}  "
            f"mean P={ps.mean():.4f}  mean R={rs.mean():.4f}"
        ),
        fill=(0, 0, 0),
        font=font,
    )
    draw.text((28, 98), f"Checkpoint: {ckpt or 'unknown'}", fill=(35, 35, 35), font=small)
    part_text = ", ".join(f"{part}:{count}" for part, count in sorted(parts.items(), reverse=True))
    draw.text((28, 126), f"Selected target_part_count counts: {part_text}", fill=(35, 35, 35), font=small)
    if selection_report:
        strategy = (
            f"Selection: high-part first; full groups={selection_report.get('full_part_count_groups')} "
            f"boundary={selection_report.get('boundary_fill_part_count')} "
            f"{selection_report.get('boundary_fill_selected')}/{selection_report.get('boundary_fill_available')}"
        )
        draw.text((28, 152), strategy, fill=(35, 35, 35), font=small)

    columns = [
        ("IoU %", 28, 140),
        ("Count", 168, 92),
        ("Worst in bin", 270, 610),
        ("Worst IoU/P/R", 900, 210),
        ("Worst parts", 1130, 130),
        ("Best in bin", 1280, 410),
    ]
    y = top_h
    draw.rectangle((20, y, width - 20, y + header_h), fill=(36, 42, 52))
    for label, x, _ in columns:
        draw.text((x, y + 10), label, fill=(255, 255, 255), font=font)
    y += header_h
    max_count = max((item["count"] for item in bins), default=1)
    for idx, item in enumerate(bins):
        fill = (255, 255, 255) if idx % 2 == 0 else (240, 243, 247)
        draw.rectangle((20, y, width - 20, y + row_h), fill=fill)
        bar_w = int(220 * item["count"] / max(max_count, 1))
        draw.rectangle((168, y + 14, 168 + bar_w, y + 38), fill=(86, 141, 214))
        draw.text((28, y + 15), item["bin"], fill=(0, 0, 0), font=font)
        draw.text((178, y + 15), str(item["count"]), fill=(0, 0, 0), font=font)
        worst = item["worst"]
        best = item["best"]
        if worst:
            worst_name = f"{worst['object_id']} angle_{worst['angle_idx']:02d}  {worst.get('category') or ''}"
            draw.text((270, y + 8), worst_name[:70], fill=(0, 0, 0), font=small)
            draw.text((270, y + 29), str(worst.get("name") or "")[:76], fill=(55, 55, 55), font=tiny)
            draw.text(
                (900, y + 15),
                f"{worst['iou']:.4f} / {worst['precision']:.4f} / {worst['recall']:.4f}",
                fill=(0, 0, 0),
                font=small,
            )
            draw.text((1130, y + 15), str(worst["target_part_count"]), fill=(0, 0, 0), font=small)
        else:
            draw.text((270, y + 15), "-", fill=(90, 90, 90), font=small)
            draw.text((900, y + 15), "-", fill=(90, 90, 90), font=small)
            draw.text((1130, y + 15), "-", fill=(90, 90, 90), font=small)
        if best:
            best_name = f"{best['object_id']} angle_{best['angle_idx']:02d}  IoU={best['iou']:.4f}  parts={best['target_part_count']}"
            draw.text((1280, y + 15), best_name[:54], fill=(0, 0, 0), font=small)
        else:
            draw.text((1280, y + 15), "-", fill=(90, 90, 90), font=small)
        y += row_h
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_hist_png(path: Path, rows: list[dict[str, Any]]) -> None:
    ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    ax.hist(ious * 100.0, bins=np.arange(0, 110, 10), color="#568dd6", edgecolor="#222222")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.set_xlabel("IoU (%)")
    ax.set_ylabel("Object count")
    ax.set_title("IoU Histogram")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_selected_renders(rows: list[dict[str, Any]], out_dir: Path, best_n: int, worst_n: int, width: int, height: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: (row["iou"], -row["target_part_count"], row["object_id"], row["angle_idx"]))
    selected: list[tuple[str, int, dict[str, Any]]] = []
    for rank, row in enumerate(sorted_rows[:worst_n], start=1):
        selected.append(("worst", rank, row))
    for rank, row in enumerate(reversed(sorted_rows[-best_n:]), start=1):
        selected.append(("best", rank, row))

    render_rows: list[dict[str, Any]] = []
    for group, rank, row in selected:
        tmp_dir = out_dir / "_tmp_render"
        summary_path = Path(row["summary_path"])
        stem = f"{row['object_id']}_angle{row['angle_idx']:02d}"
        convert_one(summary_path, tmp_dir, width, height)
        src_png = tmp_dir / f"{stem}.png"
        src_json = tmp_dir / f"{stem}.json"
        dst_stem = f"{group}{rank:02d}_{stem}_iou{row['iou']:.4f}_parts{row['target_part_count']}"
        dst_png = out_dir / f"{dst_stem}.png"
        dst_json = out_dir / f"{dst_stem}.json"
        src_png.replace(dst_png)
        payload = json.loads(src_json.read_text(encoding="utf-8"))
        payload.update({
            "selection_group": group,
            "selection_rank": rank,
            "flat_png_path": str(dst_png.resolve()),
            "flat_json_path": str(dst_json.resolve()),
        })
        dst_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        src_json.unlink(missing_ok=True)
        render_rows.append({**row, "selection_group": group, "selection_rank": rank, "png_path": str(dst_png.resolve()), "json_path": str(dst_json.resolve())})
    tmp_dir = out_dir / "_tmp_render"
    if tmp_dir.exists():
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
    return render_rows


def main() -> None:
    args = parse_args()
    args.eval_root = args.eval_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summaries(args.eval_root)
    selection_report = None
    if args.selection_report and args.selection_report.is_file():
        selection_report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    bins = build_iou_bins(rows)
    selected_renders = write_selected_renders(rows, args.out_dir, int(args.best_n), int(args.worst_n), int(args.width), int(args.height))
    table_png = args.out_dir / "iou_bins_table.png"
    hist_png = args.out_dir / "iou_histogram.png"
    ckpt = str(args.ckpt.resolve()) if args.ckpt else None
    draw_table_png(table_png, rows=rows, bins=bins, selection_report=selection_report, ckpt=ckpt)
    draw_hist_png(hist_png, rows)

    ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    ps = np.asarray([row["precision"] for row in rows], dtype=np.float64)
    rs = np.asarray([row["recall"] for row in rows], dtype=np.float64)
    summary = {
        "eval_root": str(args.eval_root),
        "out_dir": str(args.out_dir),
        "sample_count": len(rows),
        "ckpt": ckpt,
        "metrics_vs_gt_surface": {
            "mean_iou": float(ious.mean()),
            "median_iou": float(np.median(ious)),
            "mean_precision": float(ps.mean()),
            "mean_recall": float(rs.mean()),
            "min_iou": float(ious.min()),
            "max_iou": float(ious.max()),
        },
        "selected_by_target_part_count": dict(sorted(Counter(int(row["target_part_count"]) for row in rows).items(), reverse=True)),
        "iou_bins": bins,
        "rendered_best_worst": selected_renders,
        "table_png": str(table_png.resolve()),
        "histogram_png": str(hist_png.resolve()),
        "selection_report": selection_report,
    }
    summary_path = args.out_dir / "summary.json"
    bins_path = args.out_dir / "iou_bins_table.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    bins_path.write_text(json.dumps({"iou_bins": bins}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_count": len(rows),
        "mean_iou": summary["metrics_vs_gt_surface"]["mean_iou"],
        "median_iou": summary["metrics_vs_gt_surface"]["median_iou"],
        "out_dir": str(args.out_dir),
        "summary": str(summary_path),
        "table_png": str(table_png),
        "histogram_png": str(hist_png),
        "renders": len(selected_renders),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
