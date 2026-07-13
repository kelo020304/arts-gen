#!/usr/bin/env python3
"""Aggregate and visualize joint-boundary candidate/refinement A/B reports."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


REPORT_SUFFIX = "__joint_boundary.json"
PNG_SUFFIX = "__joint_boundary.png"
QUALITY_METRICS = ("mIoU", "part_mIoU", "boundary_error", "boundary_error_covered", "cross_same")
SHARED_METRICS = ("candidate_recall", "whole_iou", "interface_ratio")
DIAGNOSTIC_METRICS = ("changed", "improved", "regressed", "neutral")
LOWER_IS_BETTER = frozenset(("boundary_error", "boundary_error_covered", "cross_same"))

ALIASES: dict[str, tuple[str, ...]] = {
    "mIoU": ("mIoU", "miou", "mean_iou", "joint_miou"),
    "part_mIoU": ("part_mIoU", "part_miou", "part_mean_iou", "parts_miou"),
    "boundary_error": ("boundary_error", "joint_boundary_error", "boundary_band_error"),
    "boundary_error_covered": (
        "boundary_error_covered",
        "covered_boundary_error",
        "candidate_boundary_error",
    ),
    "cross_same": (
        "cross_same",
        "cross_same_rate",
        "cross_label_same_pred_rate",
        "joint_cross_label_same_pred_rate",
    ),
    "candidate_recall": ("candidate_recall", "candidate_coverage", "gt_candidate_recall", "recall"),
    "whole_iou": ("whole_iou", "whole_occ_iou", "whole_occupancy_iou", "iou"),
    "interface_ratio": (
        "interface_ratio",
        "interface_voxel_ratio",
        "predicted_to_gt_interface_ratio",
        "boundary_ratio",
    ),
    "changed": ("changed", "changed_voxels", "voxels"),
    "improved": ("improved", "improved_voxels", "changed_improved"),
    "regressed": ("regressed", "regressed_voxels", "changed_regressed"),
    "neutral": ("neutral", "neutral_voxels", "changed_neutral"),
    "low_margin_voxels": ("low_margin_voxels", "ambiguous_voxels", "voxels"),
    "gt_overlap_voxels": ("gt_overlap_voxels", "overlap_voxels", "multi_claim_ignore_voxels"),
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalized_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _as_float(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("value", "mean", "count", "total"):
            if key in value:
                return _as_float(value[key])
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _lookup_number(mapping: Any, canonical: str) -> float | None:
    if not isinstance(mapping, Mapping):
        return None
    normalized = {_normalized_key(str(key)): value for key, value in mapping.items()}
    for alias in ALIASES.get(canonical, (canonical,)):
        key = _normalized_key(alias)
        if key in normalized:
            return _as_float(normalized[key])
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _prefixed_metric(metrics: Mapping[str, Any], prefix: str, canonical: str) -> float | None:
    aliases = ALIASES.get(canonical, (canonical,))
    normalized = {_normalized_key(str(key)): value for key, value in metrics.items()}
    for alias in aliases:
        key = _normalized_key(f"{prefix}_{alias}")
        if key in normalized:
            return _as_float(normalized[key])
    return None


def _metric(group: Mapping[str, Any], metrics: Mapping[str, Any], prefix: str, canonical: str) -> float | None:
    value = _lookup_number(group, canonical)
    return value if value is not None else _prefixed_metric(metrics, prefix, canonical)


def _safe_name(value: str, max_len: int = 180) -> str:
    name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value)).strip("_")
    return (name or "object")[:max_len]


def _resolve_png(report_path: Path, payload: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> Path:
    artifacts = _mapping(payload.get("artifacts"))
    values = (
        payload.get("png_path"),
        payload.get("boundary_png"),
        diagnostics.get("png_path"),
        diagnostics.get("boundary_png"),
        artifacts.get("png"),
    )
    candidates: list[Path] = []
    for value in values:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = report_path.parent / path
        candidates.append(path.resolve())
    candidates.append(report_path.with_name(report_path.name.replace(REPORT_SUFFIX, PNG_SUFFIX)).resolve())
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _object_key(report_path: Path, payload: Mapping[str, Any]) -> str:
    explicit = payload.get("object_key") or payload.get("prefix")
    if explicit:
        return str(explicit)
    sample = _mapping(payload.get("sample"))
    dataset_id = payload.get("dataset_id") or sample.get("dataset_id")
    object_id = payload.get("object_id") or sample.get("object_id")
    angle = payload.get("angle") if payload.get("angle") is not None else sample.get("angle")
    if dataset_id and object_id and angle is not None:
        try:
            angle_text = f"{int(angle):02d}"
        except (TypeError, ValueError):
            angle_text = _safe_name(str(angle))
        return f"{_safe_name(str(dataset_id))}__{_safe_name(str(object_id))}__angle_{angle_text}"
    return report_path.name[: -len(REPORT_SUFFIX)]


def _outcome(raw_value: float | None, refined_value: float | None, *, lower_is_better: bool, eps: float = 1e-9) -> str:
    if raw_value is None or refined_value is None:
        return "missing"
    gain = raw_value - refined_value if lower_is_better else refined_value - raw_value
    if gain > eps:
        return "improved"
    if gain < -eps:
        return "regressed"
    return "neutral"


def report_to_row(variant: str, report_path: Path) -> dict[str, Any]:
    """Normalize one producer report into a stable, flat per-object row."""

    payload = _read_json(report_path)
    metrics = _mapping(payload.get("metrics"))
    shared = _mapping(metrics.get("shared"))
    raw = _mapping(metrics.get("raw") or payload.get("raw"))
    refined = _mapping(metrics.get("refined") or payload.get("refined"))
    producer_delta = _mapping(metrics.get("delta"))
    diagnostics = _mapping(payload.get("diagnostics"))
    direct_changed = _mapping(payload.get("changed"))
    direct_candidate = _mapping(payload.get("candidate"))
    direct_whole = _mapping(payload.get("whole"))
    direct_interface = _mapping(payload.get("interface"))
    direct_low_margin = _mapping(payload.get("low_margin"))
    direct_gt = _mapping(payload.get("gt"))
    sample = _mapping(payload.get("sample"))
    png_path = _resolve_png(report_path, payload, diagnostics)
    object_key = _object_key(report_path, payload)

    row: dict[str, Any] = {
        "object_key": object_key,
        "variant": str(variant),
        "dataset_id": payload.get("dataset_id") or sample.get("dataset_id"),
        "object_id": payload.get("object_id") or sample.get("object_id"),
        "angle": payload.get("angle") if payload.get("angle") is not None else sample.get("angle"),
        "candidate_mode": payload.get("candidate_mode") or variant,
        "stage": payload.get("stage"),
        "report_path": str(report_path.resolve()),
        "artifact_path": payload.get("artifact_path") or payload.get("joint_partition_path"),
        "png_path": str(png_path),
        "png_exists": png_path.is_file(),
    }

    for name in QUALITY_METRICS:
        raw_value = _metric(raw, metrics, "raw", name)
        refined_value = _metric(refined, metrics, "refined", name)
        if raw_value is not None and refined_value is not None:
            delta_value = refined_value - raw_value
        else:
            delta_value = _metric(producer_delta, metrics, "delta", name)
        row[f"raw_{name}"] = raw_value
        row[f"refined_{name}"] = refined_value
        row[f"delta_{name}"] = delta_value
        row[name] = refined_value if refined_value is not None else raw_value
        row[f"{name}_outcome"] = _outcome(
            raw_value,
            refined_value,
            lower_is_better=name in LOWER_IS_BETTER,
        )

    for name in SHARED_METRICS:
        value = _lookup_number(shared, name)
        if value is None:
            value = _lookup_number(metrics, name)
        if value is None and name == "candidate_recall":
            value = _lookup_number(direct_candidate, "candidate_recall")
        if value is None and name == "whole_iou":
            value = _lookup_number(direct_whole, "whole_iou")
        if value is None and name == "interface_ratio":
            value = _lookup_number(refined, "interface_ratio")
        if value is None and name == "interface_ratio":
            value = _lookup_number(raw, "interface_ratio")
        if value is None and name == "interface_ratio":
            value = _lookup_number(direct_interface, "interface_ratio")
        row[name] = value

    for name in DIAGNOSTIC_METRICS + ("low_margin_voxels", "gt_overlap_voxels"):
        row[name] = _lookup_number(diagnostics, name)
    for name in DIAGNOSTIC_METRICS:
        if row[name] is None:
            row[name] = _lookup_number(direct_changed, name)
    if row["changed"] is None:
        row["changed"] = _lookup_number(refined, "changed")
    if row["low_margin_voxels"] is None:
        row["low_margin_voxels"] = _lookup_number(direct_low_margin, "low_margin_voxels")
    if row["gt_overlap_voxels"] is None:
        row["gt_overlap_voxels"] = _lookup_number(direct_gt, "gt_overlap_voxels")

    class_names = diagnostics.get("class_names") or payload.get("class_names") or []
    row["class_names"] = ",".join(str(item) for item in class_names) if isinstance(class_names, list) else str(class_names)
    diagnostic_payload = diagnostics or {
        "changed": direct_changed,
        "low_margin": direct_low_margin,
        "gt_overlap_voxels": row["gt_overlap_voxels"],
        "class_names": class_names,
    }
    row["diagnostics_json"] = json.dumps(_json_safe(diagnostic_payload), ensure_ascii=False, sort_keys=True)
    for phase, group in (("raw", raw), ("refined", refined)):
        components = _mapping(group.get("components"))
        row[f"{phase}_part_components"] = _lookup_number(components, "part_total")
        row[f"{phase}_part_tiny_components_le_8"] = _lookup_number(
            components,
            "part_tiny_components_le_8",
        )
        row[f"{phase}_part_tiny_voxels_le_8"] = _lookup_number(
            components,
            "part_tiny_voxels_le_8",
        )
        row[f"{phase}_interface_ratio"] = _lookup_number(group, "interface_ratio")
        row[f"{phase}_boundary_candidate_coverage"] = _lookup_number(
            group,
            "boundary_candidate_coverage",
        )
    return row


def collect_rows(variants: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, root in variants.items():
        paths = sorted(Path(root).glob(f"*{REPORT_SUFFIX}"))
        for report_path in paths:
            rows.append(report_to_row(str(variant), report_path))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = [
        "object_key",
        "variant",
        "dataset_id",
        "object_id",
        "angle",
        "candidate_mode",
        "stage",
        *QUALITY_METRICS,
        *SHARED_METRICS,
        "changed",
        "improved",
        "regressed",
        "neutral",
    ]
    fields = list(preferred)
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = {}
            for key in fields:
                value = source.get(key)
                row[key] = "" if value is None else value
            writer.writerow(row)


def _stats(values: Iterable[Any], *, include_sum: bool = False) -> dict[str, Any]:
    finite = [number for value in values if (number := _as_float(value)) is not None]
    if not finite:
        result: dict[str, Any] = {"count": 0, "mean": None, "median": None, "min": None, "max": None}
        if include_sum:
            result["sum"] = 0.0
        return result
    arr = np.asarray(finite, dtype=np.float64)
    result = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    if include_sum:
        result["sum"] = float(arr.sum())
    return result


def _outcome_counts(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, int]:
    values = [str(row.get(f"{metric}_outcome") or "missing") for row in rows]
    return {name: values.count(name) for name in ("improved", "regressed", "neutral", "missing")}


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "report_count": int(len(rows)),
        "object_count": int(len({str(row.get("object_key")) for row in rows})),
        "candidate_modes": sorted({str(row.get("candidate_mode")) for row in rows if row.get("candidate_mode")}),
        "raw": {name: _stats(row.get(f"raw_{name}") for row in rows) for name in QUALITY_METRICS},
        "refined": {name: _stats(row.get(f"refined_{name}") for row in rows) for name in QUALITY_METRICS},
        "delta_refined_minus_raw": {
            name: _stats(row.get(f"delta_{name}") for row in rows) for name in QUALITY_METRICS
        },
        "shared": {name: _stats(row.get(name) for row in rows) for name in SHARED_METRICS},
        "diagnostics": {
            name: _stats((row.get(name) for row in rows), include_sum=True) for name in DIAGNOSTIC_METRICS
        },
        "refinement_outcomes": {name: _outcome_counts(rows, name) for name in QUALITY_METRICS},
        "missing_png_count": int(sum(not bool(row.get("png_exists")) for row in rows)),
    }


def _pairwise_summaries(rows: Sequence[Mapping[str, Any]], variant_order: Sequence[str]) -> list[dict[str, Any]]:
    by_variant: dict[str, dict[str, Mapping[str, Any]]] = {}
    for variant in variant_order:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        by_variant[variant] = {str(row.get("object_key")): row for row in variant_rows}

    output: list[dict[str, Any]] = []
    for left, right in itertools.combinations(variant_order, 2):
        common = sorted(set(by_variant[left]) & set(by_variant[right]))
        deltas: dict[str, dict[str, Any]] = {}
        right_better: dict[str, int] = {}
        for metric in QUALITY_METRICS + SHARED_METRICS:
            values: list[float] = []
            better = 0
            for key in common:
                column = f"refined_{metric}" if metric in QUALITY_METRICS else metric
                left_value = _as_float(by_variant[left][key].get(column))
                right_value = _as_float(by_variant[right][key].get(column))
                if left_value is None or right_value is None:
                    continue
                delta = right_value - left_value
                values.append(delta)
                if (delta < 0.0) if metric in LOWER_IS_BETTER else (delta > 0.0):
                    better += 1
            deltas[metric] = _stats(values)
            right_better[metric] = int(better)
        output.append(
            {
                "left": left,
                "right": right,
                "common_object_count": int(len(common)),
                "delta_right_minus_left": deltas,
                "right_better_object_count": right_better,
            }
        )
    return output


def _format_metric(value: Any) -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number:.4f}"


def _load_or_placeholder(path_value: Any, width: int, height: int) -> Image.Image:
    path = Path(str(path_value)) if path_value else Path("")
    if path_value and path.is_file():
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    image = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(180, 40, 40), width=3)
    draw.text((18, 20), "missing boundary PNG", fill=(130, 0, 0))
    if path_value:
        draw.text((18, 44), str(path_value)[:100], fill=(90, 90, 90))
    return image


def _candidate_tile(row: Mapping[str, Any] | None, variant: str, width: int, height: int) -> Image.Image:
    body_height = max(1, height - 64)
    source = _load_or_placeholder(None if row is None else row.get("png_path"), width, body_height)
    source.thumbnail((width, body_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, width, 64), fill=(18, 18, 18))
    if row is None:
        line1 = f"{variant}: report missing"
        line2 = ""
    else:
        mode = str(row.get("candidate_mode") or variant)
        line1 = f"{variant} | candidate={mode}"
        line2 = (
            f"refined mIoU={_format_metric(row.get('refined_mIoU'))}  "
            f"part={_format_metric(row.get('refined_part_mIoU'))}  "
            f"boundary={_format_metric(row.get('refined_boundary_error'))}  "
            f"changed={_format_metric(row.get('changed'))}"
        )
    draw.text((9, 9), line1[:120], fill=(255, 255, 255))
    draw.text((9, 34), line2[:140], fill=(225, 225, 225))
    tile.paste(source, ((width - source.width) // 2, 64 + (body_height - source.height) // 2))
    return tile


def write_object_panels(
    rows: Sequence[Mapping[str, Any]],
    variant_order: Sequence[str],
    out_dir: Path,
    *,
    tile_width: int = 760,
    tile_height: int = 620,
) -> list[dict[str, Any]]:
    by_object: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_object.setdefault(str(row.get("object_key")), {})[str(row.get("variant"))] = row

    panel_dir = out_dir / "object_boundary_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for object_key in sorted(by_object):
        canvas = Image.new(
            "RGB",
            (tile_width * len(variant_order), tile_height + 38),
            (255, 255, 255),
        )
        ImageDraw.Draw(canvas).text((9, 12), object_key[:180], fill=(0, 0, 0))
        for idx, variant in enumerate(variant_order):
            tile = _candidate_tile(by_object[object_key].get(variant), variant, tile_width, tile_height)
            canvas.paste(tile, (idx * tile_width, 38))
        out_png = panel_dir / f"{_safe_name(object_key)}__candidate_boundary_ab.png"
        canvas.save(out_png)
        records.append(
            {
                "object_key": object_key,
                "path": str(out_png.resolve()),
                "variants": list(variant_order),
            }
        )
    return records


def _pair_text(raw_value: Any, refined_value: Any, *, digits: int = 3) -> str:
    raw_number = _as_float(raw_value)
    refined_number = _as_float(refined_value)
    if raw_number is None or refined_number is None:
        return "n/a"
    return f"{raw_number:.{digits}f} -> {refined_number:.{digits}f}"


def _pair_color(raw_value: Any, refined_value: Any, *, lower_is_better: bool) -> tuple[float, float, float, float]:
    raw_number = _as_float(raw_value)
    refined_number = _as_float(refined_value)
    neutral = (0.95, 0.95, 0.95, 1.0)
    if raw_number is None or refined_number is None:
        return neutral
    gain = raw_number - refined_number if lower_is_better else refined_number - raw_number
    if gain > 1.0e-9:
        return (0.84, 0.94, 0.86, 1.0)
    if gain < -1.0e-9:
        return (0.98, 0.86, 0.86, 1.0)
    return neutral


def write_variant_metric_tables(
    rows: Sequence[Mapping[str, Any]],
    variant_order: Sequence[str],
    out_dir: Path,
) -> dict[str, str]:
    """Write one compact per-object quantitative table for each candidate variant."""

    outputs: dict[str, str] = {}
    columns = (
        "candidate\nrecall",
        "whole\nIoU",
        "mIoU\nraw -> ref",
        "covered BErr\nraw -> ref",
        "cross-same\nraw -> ref",
        "interface ratio\nraw -> ref",
        "part CC\nraw -> ref",
        "tiny <=8\nraw -> ref",
        "changed\n(+good/-bad)",
    )
    for variant in variant_order:
        variant_rows = sorted(
            (row for row in rows if str(row.get("variant")) == str(variant)),
            key=lambda row: (str(row.get("object_id")), int(_as_float(row.get("angle")) or 0)),
        )
        if not variant_rows:
            continue
        cell_text: list[list[str]] = []
        cell_colors: list[list[tuple[float, float, float, float]]] = []
        row_labels: list[str] = []
        for row in variant_rows:
            object_id = str(row.get("object_id") or row.get("object_key"))
            angle = int(_as_float(row.get("angle")) or 0)
            row_labels.append(f"{object_id}  a{angle}")
            candidate_recall = _as_float(row.get("candidate_recall"))
            whole_iou = _as_float(row.get("whole_iou"))
            improved = int(_as_float(row.get("improved")) or 0)
            regressed = int(_as_float(row.get("regressed")) or 0)
            changed = int(_as_float(row.get("changed")) or 0)
            cell_text.append(
                [
                    "n/a" if candidate_recall is None else f"{candidate_recall:.3f}",
                    "n/a" if whole_iou is None else f"{whole_iou:.3f}",
                    _pair_text(row.get("raw_mIoU"), row.get("refined_mIoU")),
                    _pair_text(
                        row.get("raw_boundary_error_covered"),
                        row.get("refined_boundary_error_covered"),
                    ),
                    _pair_text(row.get("raw_cross_same"), row.get("refined_cross_same")),
                    _pair_text(row.get("raw_interface_ratio"), row.get("refined_interface_ratio")),
                    _pair_text(
                        row.get("raw_part_components"),
                        row.get("refined_part_components"),
                        digits=0,
                    ),
                    _pair_text(
                        row.get("raw_part_tiny_components_le_8"),
                        row.get("refined_part_tiny_components_le_8"),
                        digits=0,
                    ),
                    f"{changed} (+{improved}/-{regressed})",
                ]
            )
            candidate_color = (
                (0.84, 0.94, 0.86, 1.0)
                if candidate_recall is not None and candidate_recall >= 0.9
                else (0.98, 0.86, 0.86, 1.0)
                if candidate_recall is not None and candidate_recall < 0.7
                else (0.95, 0.95, 0.95, 1.0)
            )
            whole_color = (
                (0.84, 0.94, 0.86, 1.0)
                if whole_iou is not None and whole_iou >= 0.8
                else (0.98, 0.86, 0.86, 1.0)
                if whole_iou is not None and whole_iou < 0.5
                else (0.95, 0.95, 0.95, 1.0)
            )
            cell_colors.append(
                [
                    candidate_color,
                    whole_color,
                    _pair_color(row.get("raw_mIoU"), row.get("refined_mIoU"), lower_is_better=False),
                    _pair_color(
                        row.get("raw_boundary_error_covered"),
                        row.get("refined_boundary_error_covered"),
                        lower_is_better=True,
                    ),
                    _pair_color(
                        row.get("raw_cross_same"),
                        row.get("refined_cross_same"),
                        lower_is_better=True,
                    ),
                    (0.95, 0.95, 0.95, 1.0),
                    _pair_color(
                        row.get("raw_part_components"),
                        row.get("refined_part_components"),
                        lower_is_better=True,
                    ),
                    _pair_color(
                        row.get("raw_part_tiny_components_le_8"),
                        row.get("refined_part_tiny_components_le_8"),
                        lower_is_better=True,
                    ),
                    (0.84, 0.94, 0.86, 1.0)
                    if improved > regressed
                    else (0.98, 0.86, 0.86, 1.0)
                    if regressed > improved
                    else (0.95, 0.95, 0.95, 1.0),
                ]
            )

        fig_height = max(3.2, 0.62 * len(variant_rows) + 1.9)
        fig, ax = plt.subplots(figsize=(18, fig_height), dpi=150)
        ax.axis("off")
        table = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            rowLabels=row_labels,
            colLabels=columns,
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.65)
        for (row_index, _col_index), cell in table.get_celld().items():
            cell.set_edgecolor((0.75, 0.75, 0.75, 1.0))
            if row_index == 0:
                cell.set_facecolor((0.16, 0.18, 0.20, 1.0))
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
        ax.set_title(
            f"Joint boundary per-object diagnostics: {variant} | green=improved, red=regressed/hard",
            fontsize=14,
            pad=16,
        )
        fig.tight_layout()
        out_png = out_dir / f"per_object_metrics__{_safe_name(str(variant))}.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        outputs[str(variant)] = str(out_png.resolve())
    return outputs


def _summary_mean(summary: Mapping[str, Any], section: str, metric: str) -> float:
    value = (((summary.get(section) or {}).get(metric) or {}).get("mean"))
    number = _as_float(value)
    return float("nan") if number is None else number


def _plot_grouped(
    ax: Any,
    summaries: Mapping[str, Mapping[str, Any]],
    variant_order: Sequence[str],
    metrics: Sequence[str],
    sections: Sequence[str],
    *,
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(metrics), dtype=np.float64)
    series = [(variant, section) for variant in variant_order for section in sections]
    width = 0.8 / max(1, len(series))
    finite_seen = False
    for index, (variant, section) in enumerate(series):
        values = [_summary_mean(summaries[variant], section, metric) for metric in metrics]
        finite_seen |= any(math.isfinite(value) for value in values)
        offset = (index - (len(series) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=f"{variant} {section}")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, metrics)
    ax.grid(axis="y", alpha=0.25)
    if finite_seen:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no finite metrics", ha="center", va="center", transform=ax.transAxes)


def write_aggregate_plot(
    summaries: Mapping[str, Mapping[str, Any]],
    variant_order: Sequence[str],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=140)
    _plot_grouped(
        axes[0, 0],
        summaries,
        variant_order,
        ("mIoU", "part_mIoU"),
        ("raw", "refined"),
        title="Segmentation quality (macro mean)",
        ylabel="IoU",
    )
    _plot_grouped(
        axes[0, 1],
        summaries,
        variant_order,
        ("boundary_error", "boundary_error_covered", "cross_same"),
        ("raw", "refined"),
        title="Boundary errors (lower is better)",
        ylabel="rate",
    )
    _plot_grouped(
        axes[1, 0],
        summaries,
        variant_order,
        SHARED_METRICS,
        ("shared",),
        title="Candidate and whole-shape diagnostics",
        ylabel="rate",
    )

    ax = axes[1, 1]
    x = np.arange(len(DIAGNOSTIC_METRICS), dtype=np.float64)
    width = 0.8 / max(1, len(variant_order))
    for idx, variant in enumerate(variant_order):
        values = [
            _as_float((((summaries[variant].get("diagnostics") or {}).get(name) or {}).get("sum"))) or 0.0
            for name in DIAGNOSTIC_METRICS
        ]
        offset = (idx - (len(variant_order) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=variant)
    ax.set_title("Refinement voxel changes (sum)")
    ax.set_ylabel("voxels")
    ax.set_xticks(x, DIAGNOSTIC_METRICS)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Joint boundary candidate/refinement A/B", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def aggregate_joint_boundary_ab(
    variants: Mapping[str, Path],
    out_dir: Path,
    *,
    panel_width: int = 760,
    panel_height: int = 620,
) -> dict[str, Any]:
    """Collect reports and write tabular, aggregate, and per-object visual outputs."""

    if not variants:
        raise ValueError("at least one variant is required")
    variant_order = list(variants)
    rows = collect_rows(variants)
    if not rows:
        roots = ", ".join(f"{name}={path}" for name, path in variants.items())
        raise FileNotFoundError(f"no *{REPORT_SUFFIX} reports found under: {roots}")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "metrics_per_object.csv", rows)
    _write_json(out_dir / "metrics_per_object.json", rows)

    panels = write_object_panels(
        rows,
        variant_order,
        out_dir,
        tile_width=int(panel_width),
        tile_height=int(panel_height),
    )
    metric_tables = write_variant_metric_tables(rows, variant_order, out_dir)
    variant_summaries = {
        variant: summarize_rows([row for row in rows if row.get("variant") == variant])
        for variant in variant_order
    }
    aggregate = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "delta_definition": "refined - raw; negative is better for boundary_error/cross_same",
        "variant_order": variant_order,
        "variant_dirs": {name: str(Path(path).resolve()) for name, path in variants.items()},
        "report_count": int(len(rows)),
        "unique_object_count": int(len({str(row.get("object_key")) for row in rows})),
        "variants": variant_summaries,
        "pairwise": _pairwise_summaries(rows, variant_order),
        "object_panels": panels,
        "per_variant_metric_tables": metric_tables,
        "outputs": {
            "metrics_csv": str((out_dir / "metrics_per_object.csv").resolve()),
            "metrics_json": str((out_dir / "metrics_per_object.json").resolve()),
            "aggregate_json": str((out_dir / "aggregate.json").resolve()),
            "aggregate_png": str((out_dir / "aggregate.png").resolve()),
            "per_variant_metric_tables": metric_tables,
        },
    }
    write_aggregate_plot(variant_summaries, variant_order, out_dir / "aggregate.png")
    _write_json(out_dir / "aggregate.json", aggregate)
    return aggregate


def _parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must be NAME=DIR")
    name, path_value = value.split("=", 1)
    name = name.strip()
    path_value = path_value.strip()
    if not name or not path_value:
        raise argparse.ArgumentTypeError("variant must have a non-empty NAME and DIR")
    path = Path(path_value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"variant directory does not exist: {path}")
    return name, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate joint-boundary JSON/PNG reports across candidate/refinement variants."
    )
    parser.add_argument(
        "--variant",
        action="append",
        type=_parse_variant,
        required=True,
        metavar="NAME=DIR",
        help=f"Variant name and directory containing *{REPORT_SUFFIX}; repeat for A/B.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=760)
    parser.add_argument("--panel-height", type=int, default=620)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    variants: dict[str, Path] = {}
    for name, path in args.variant:
        if name in variants:
            raise ValueError(f"duplicate variant name: {name}")
        variants[name] = path
    aggregate = aggregate_joint_boundary_ab(
        variants,
        args.out_dir,
        panel_width=max(160, int(args.panel_width)),
        panel_height=max(160, int(args.panel_height)),
    )
    print(json.dumps(_json_safe(aggregate), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
