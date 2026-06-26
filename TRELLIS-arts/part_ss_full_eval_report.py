"""Reporting helpers for Part SS Latent Flow full evaluation.

This module is intentionally CPU-only and model-free.  The full eval runner
produces per-part rows and object panels; these helpers enrich, aggregate, and
write those rows into Markdown/JSON-friendly summaries.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from trellis.trainers.arts.part_ss_latent_flow_losses import k_bucket_name


SIZE_BUCKETS = ("small", "medium", "large")
K_BUCKETS = ("k_1_2", "k_3_5", "k_6_10", "k_11_15", "k_16_plus")


def size_bucket_name(count: float, size_boundaries: tuple[float, float] = (500.0, 3000.0)) -> str:
    small_hi, medium_hi = float(size_boundaries[0]), float(size_boundaries[1])
    if not small_hi < medium_hi:
        raise ValueError(f"size_boundaries must be increasing, got {size_boundaries}")
    value = float(count)
    if value < small_hi:
        return "small"
    if value < medium_hi:
        return "medium"
    return "large"


def _part_count(row: dict[str, Any]) -> int:
    return int(row.get("part_raw_voxel_count", row.get("raw_ind_count", 0)))


def _part_iou(row: dict[str, Any]) -> float:
    return float(row.get("part_iou", row.get("decode_iou_pred_vs_raw_ind", 0.0)))


def _part_recall(row: dict[str, Any]) -> float:
    return float(row.get("part_recall", row.get("decode_recall_pred_vs_raw_ind", 0.0)))


def _part_precision(row: dict[str, Any]) -> float:
    return float(row.get("part_precision", row.get("decode_precision_pred_vs_raw_ind", 0.0)))


def _part_cos(row: dict[str, Any]) -> float:
    return float(row.get("part_cos", row.get("latent_cos", float("nan"))))


def _object_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("obj_id", "")),
        str(row.get("sample_id", "")),
        int(row.get("dataset_index", -1)),
    )


def enrich_part_rows(
    rows: Iterable[dict[str, Any]],
    *,
    size_boundaries: tuple[float, float] = (500.0, 3000.0),
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        count = _part_count(out)
        k = int(out.get("object_part_count", 0))
        out["part_raw_voxel_count"] = count
        out["size_bucket"] = size_bucket_name(count, size_boundaries)
        out["k_bucket"] = k_bucket_name(k)
        out["part_iou"] = _part_iou(out)
        out["part_recall"] = _part_recall(out)
        out["part_precision"] = _part_precision(out)
        out["part_cos"] = _part_cos(out)
        enriched.append(out)
    return enriched


def _panel_key(panel: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(panel.get("obj_id", "")),
        str(panel.get("sample_id", "")),
        int(panel.get("dataset_index", -1)),
    )


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_min(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.min(vals)) if vals else float("nan")


def build_object_rows(
    part_rows: list[dict[str, Any]],
    object_panels: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    panel_by_key = {_panel_key(panel): panel for panel in (object_panels or [])}
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in part_rows:
        grouped[_object_key(row)].append(row)

    object_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        obj_id, sample_id, dataset_index = key
        counts = [max(0, _part_count(row)) for row in rows]
        positive_counts = [count for count in counts if count > 0]
        size_counts = {name: sum(1 for row in rows if row.get("size_bucket") == name) for name in SIZE_BUCKETS}
        k = int(rows[0].get("object_part_count", len(rows)))
        panel = panel_by_key.get(key, {})
        size_mix_ratio = (
            float(max(positive_counts) / min(positive_counts))
            if positive_counts
            else float("nan")
        )
        object_rows.append({
            "obj_id": obj_id,
            "sample_id": sample_id,
            "dataset_index": dataset_index,
            "object_part_count": k,
            "k_bucket": k_bucket_name(k),
            "parts": len(rows),
            "small_count": size_counts["small"],
            "medium_count": size_counts["medium"],
            "large_count": size_counts["large"],
            "size_mix_ratio": size_mix_ratio,
            "has_small_large_mix": bool(size_counts["small"] > 0 and size_counts["large"] > 0),
            "target_parts_iou_pred_vs_gt": float(
                panel.get("target_parts_iou_pred_vs_gt", _safe_mean(_part_iou(row) for row in rows))
            ),
            "target_parts_precision_pred_vs_gt": float(
                panel.get("target_parts_precision_pred_vs_gt", _safe_mean(_part_precision(row) for row in rows))
            ),
            "target_parts_recall_pred_vs_gt": float(
                panel.get("target_parts_recall_pred_vs_gt", _safe_mean(_part_recall(row) for row in rows))
            ),
            "part_iou_mean": _safe_mean(_part_iou(row) for row in rows),
            "part_iou_min": _safe_min(_part_iou(row) for row in rows),
        })
    return sorted(object_rows, key=lambda row: (int(row["dataset_index"]), str(row["obj_id"])))


def _aggregate_part_group(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "parts": len(rows),
        "iou_mean": _safe_mean(_part_iou(row) for row in rows),
        "recall_mean": _safe_mean(_part_recall(row) for row in rows),
        "precision_mean": _safe_mean(_part_precision(row) for row in rows),
        "cos_mean": _safe_mean(_part_cos(row) for row in rows),
        "latent_mse_mean": _safe_mean(float(row.get("latent_mse", float("nan"))) for row in rows),
        "mse_vs_zero_mean": _safe_mean(float(row.get("mse_vs_zero", float("nan"))) for row in rows),
        "assignment_diag_iou_mean": _safe_mean(float(row.get("assignment_diag_iou", float("nan"))) for row in rows),
        "assignment_offdiag_max_mean": _safe_mean(float(row.get("assignment_offdiag_max", float("nan"))) for row in rows),
    }


def _aggregate_object_group(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "objects": len(rows),
        "target_iou_mean": _safe_mean(float(row.get("target_parts_iou_pred_vs_gt", float("nan"))) for row in rows),
        "target_recall_mean": _safe_mean(float(row.get("target_parts_recall_pred_vs_gt", float("nan"))) for row in rows),
        "target_precision_mean": _safe_mean(float(row.get("target_parts_precision_pred_vs_gt", float("nan"))) for row in rows),
        "part_iou_min_mean": _safe_mean(float(row.get("part_iou_min", float("nan"))) for row in rows),
        "size_mix_ratio_mean": _safe_mean(float(row.get("size_mix_ratio", float("nan"))) for row in rows),
    }


def summarize_tables(part_rows: list[dict[str, Any]], object_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_size = {
        name: _aggregate_part_group([row for row in part_rows if row.get("size_bucket") == name])
        for name in SIZE_BUCKETS
    }
    by_k = {
        name: _aggregate_part_group([row for row in part_rows if row.get("k_bucket") == name])
        for name in K_BUCKETS
    }
    by_size_x_k = {
        size: {
            k: _aggregate_part_group([
                row for row in part_rows
                if row.get("size_bucket") == size and row.get("k_bucket") == k
            ])
            for k in K_BUCKETS
        }
        for size in SIZE_BUCKETS
    }
    object_by_k = {
        name: _aggregate_object_group([row for row in object_rows if row.get("k_bucket") == name])
        for name in K_BUCKETS
    }
    mixed_objects = [row for row in object_rows if bool(row.get("has_small_large_mix", False))]
    return {
        "overall": {
            "parts": len(part_rows),
            "objects": len(object_rows),
            "part_iou_mean": _safe_mean(_part_iou(row) for row in part_rows),
            "part_recall_mean": _safe_mean(_part_recall(row) for row in part_rows),
            "part_cos_mean": _safe_mean(_part_cos(row) for row in part_rows),
            "target_parts_iou_mean": _safe_mean(
                float(row.get("target_parts_iou_pred_vs_gt", float("nan"))) for row in object_rows
            ),
            "mixed_objects": len(mixed_objects),
            "mixed_object_target_iou_mean": _safe_mean(
                float(row.get("target_parts_iou_pred_vs_gt", float("nan"))) for row in mixed_objects
            ),
        },
        "by_size": by_size,
        "by_k": by_k,
        "by_size_x_k": by_size_x_k,
        "object_by_k": object_by_k,
    }


def _wrap_float(value: Any) -> float | None:
    """Coerce to float; return None for missing values or NaN."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _part_count_error(row: dict[str, Any]) -> float | None:
    pred_count = row.get("pred_count")
    if pred_count is None:
        return None
    raw_count = _part_count(row)
    return abs(float(pred_count) - float(raw_count)) / max(float(raw_count), 1.0)


def _part_example_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    return {
        "recall": _wrap_float(_part_recall(row)),
        "precision": _wrap_float(_part_precision(row)),
        "count_error": _wrap_float(_part_count_error(row)),
        "confusion": _wrap_float(row.get("assignment_offdiag_max")),
        "worst_part": _wrap_float(_part_iou(row)),
    }


def _object_example_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    return {
        "recall": _wrap_float(row.get("target_parts_recall_pred_vs_gt")),
        "precision": _wrap_float(row.get("target_parts_precision_pred_vs_gt")),
        "count_error": _wrap_float(row.get("object_count_error")),
        "confusion": _wrap_float(row.get("object_assignment_offdiag_max")),
        "worst_part": _wrap_float(row.get("part_iou_min")),
    }


def select_visualization_examples(
    part_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    *,
    limit_per_group: int = 20,
) -> list[dict[str, Any]]:
    limit = max(0, int(limit_per_group))
    selected: list[dict[str, Any]] = []
    if limit == 0:
        return selected

    def add_part(group: str, rows: list[dict[str, Any]], kind: str) -> None:
        for row in rows[:limit]:
            selected.append({
                "group": group,
                "kind": kind,
                "obj_id": row.get("obj_id"),
                "sample_id": row.get("sample_id"),
                "dataset_index": int(row.get("dataset_index", -1)),
                "part_index": int(row.get("part_index", 0)),
                "label": row.get("target_part_name", "part"),
                "metric": _part_iou(row),
                "metrics": _part_example_metrics(row),
            })

    def add_object(group: str, rows: list[dict[str, Any]], kind: str) -> None:
        for row in rows[:limit]:
            selected.append({
                "group": group,
                "kind": kind,
                "obj_id": row.get("obj_id"),
                "sample_id": row.get("sample_id"),
                "dataset_index": int(row.get("dataset_index", -1)),
                "label": f"K={int(row.get('object_part_count', 0))}",
                "metric": float(row.get("target_parts_iou_pred_vs_gt", float("nan"))),
                "metrics": _object_example_metrics(row),
            })

    small_parts = [row for row in part_rows if row.get("size_bucket") == "small"]
    worst_small_parts = sorted(
        small_parts,
        key=lambda row: (_part_iou(row), _part_recall(row)),
    )
    best_small_parts = sorted(
        small_parts,
        key=lambda row: (_part_iou(row), _part_recall(row)),
        reverse=True,
    )

    high_k_objects = [row for row in object_rows if int(row.get("object_part_count", 0)) >= 6]
    worst_high_k_objects = sorted(
        high_k_objects,
        key=lambda row: float(row.get("target_parts_iou_pred_vs_gt", float("inf"))),
    )
    best_high_k_objects = sorted(
        high_k_objects,
        key=lambda row: float(row.get("target_parts_iou_pred_vs_gt", float("-inf"))),
        reverse=True,
    )

    mixed_objects = [row for row in object_rows if bool(row.get("has_small_large_mix", False))]
    worst_mixed_size_objects = sorted(
        mixed_objects,
        key=lambda row: float(row.get("target_parts_iou_pred_vs_gt", float("inf"))),
    )
    best_mixed_size_objects = sorted(
        mixed_objects,
        key=lambda row: float(row.get("target_parts_iou_pred_vs_gt", float("-inf"))),
        reverse=True,
    )

    add_part("best_small_parts", best_small_parts, "good")
    add_object("best_high_k_objects", best_high_k_objects, "good")
    add_object("best_mixed_size_objects", best_mixed_size_objects, "good")

    add_part("worst_small_parts", worst_small_parts, "bad")
    add_object("worst_high_k_objects", worst_high_k_objects, "bad")
    add_object("worst_mixed_size_objects", worst_mixed_size_objects, "bad")

    # General fallback so EVERY eval with >=1 object yields example images, even
    # when no object falls into the special buckets above. Sorted by target IoU.
    all_objects = sorted(
        object_rows,
        key=lambda row: float(row.get("target_parts_iou_pred_vs_gt", float("inf"))),
    )
    add_object("best_objects", list(reversed(all_objects)), "good")
    add_object("worst_objects", all_objects, "bad")

    return selected


def _fmt(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if math.isnan(number) else f"{number:.6f}"


def _rel_link(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    return f"[{Path(value).name}]({value})"


def _named_link(label: str, path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    return f"[{label}]({value})"


def write_markdown_report(
    path: str | Path,
    *,
    summary: dict[str, Any],
    part_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    selected_examples: list[dict[str, Any]],
    plots: list[str | Path],
    step: int,
    stage: str,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# Part SS Latent Flow Full Eval - step {int(step)}",
        "",
        f"- stage: `{stage}`",
        f"- parts: `{int(summary['overall']['parts'])}`",
        f"- objects: `{int(summary['overall']['objects'])}`",
        f"- part_iou_mean: `{_fmt(summary['overall']['part_iou_mean'])}`",
        f"- part_cos_mean: `{_fmt(summary['overall']['part_cos_mean'])}`",
        f"- target_parts_iou_mean: `{_fmt(summary['overall']['target_parts_iou_mean'])}`",
        "",
    ]

    if plots:
        lines.extend(["## Plots", ""])
        for plot in plots:
            lines.append(f"- {_rel_link(plot)}")
        lines.append("")

    lines.extend([
        "## Size Buckets",
        "",
        "| size | parts | iou_mean | recall_mean | cos_mean | precision_mean | mse_vs_zero_mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for size in SIZE_BUCKETS:
        row = summary["by_size"][size]
        lines.append(
            f"| {size} | {int(row['parts'])} | {_fmt(row['iou_mean'])} | {_fmt(row['recall_mean'])} | "
            f"{_fmt(row['cos_mean'])} | {_fmt(row['precision_mean'])} | {_fmt(row['mse_vs_zero_mean'])} |"
        )

    lines.extend([
        "",
        "## K Buckets",
        "",
        "| k_bucket | parts | iou_mean | recall_mean | cos_mean | precision_mean | mse_vs_zero_mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for bucket in K_BUCKETS:
        row = summary["by_k"][bucket]
        lines.append(
            f"| {bucket} | {int(row['parts'])} | {_fmt(row['iou_mean'])} | {_fmt(row['recall_mean'])} | "
            f"{_fmt(row['cos_mean'])} | {_fmt(row['precision_mean'])} | {_fmt(row['mse_vs_zero_mean'])} |"
        )

    lines.extend([
        "",
        "## Size x K IoU",
        "",
        "| size | k_1_2 | k_3_5 | k_6_10 | k_11_15 | k_16_plus |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for size in SIZE_BUCKETS:
        cells = [_fmt(summary["by_size_x_k"][size][bucket]["iou_mean"]) for bucket in K_BUCKETS]
        lines.append(f"| {size} | {' | '.join(cells)} |")

    lines.extend([
        "",
        "## Object K Buckets",
        "",
        "| k_bucket | objects | target_iou_mean | target_recall_mean | part_iou_min_mean | size_mix_ratio_mean |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for bucket in K_BUCKETS:
        row = summary["object_by_k"][bucket]
        lines.append(
            f"| {bucket} | {int(row['objects'])} | {_fmt(row['target_iou_mean'])} | "
            f"{_fmt(row['target_recall_mean'])} | {_fmt(row['part_iou_min_mean'])} | "
            f"{_fmt(row['size_mix_ratio_mean'])} |"
        )

    if selected_examples:
        lines.extend([
            "",
            "## Voxel Examples",
            "",
            "| group | obj_id | dataset_index | label | metric | png |",
            "|---|---|---:|---|---:|---|",
        ])
        for item in selected_examples:
            png = item.get("png")
            png_link = _named_link("png", png) if png else ""
            lines.append(
                f"| {item.get('group')} | {item.get('obj_id')} | {int(item.get('dataset_index', -1))} | "
                f"{item.get('label')} | {_fmt(item.get('metric'))} | {png_link} |"
            )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
