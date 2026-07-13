#!/usr/bin/env python3
"""Generate a static per-sample preview for Part Completion manifests.

The voxel view intentionally follows the older web preview: it uses vendored
Three.js + OrbitControls, lazy-loads one sample's voxel payload, and renders
interactive draggable/zoomable instanced voxel cubes.

Each Step 10 row is one sample:
    1 RGB image + 1 derived label mask + visible target part voxels

For every sample the page shows:
- RGB image
- label mask image
- RGB/mask overlay
- separated masks for visible target parts and remaining
- an interactive voxel viewer with GT target parts vs SS-decoded target parts
- remaining/overall comparison as GT surface vs SS-decoded overall
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import load_config  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
THREE_VERSION = "0.160.0"
VENDOR_ROOT = REPO_ROOT / "vendor" / "three" / THREE_VERSION
CLASSIC_VENDOR_ASSETS = {
    "three.min.js": VENDOR_ROOT / "classic" / "three.min.js",
    "OrbitControls.js": VENDOR_ROOT / "classic" / "OrbitControls.js",
}

REMAINING_NAME = "remaining"
BACKGROUND_LABEL = 0
DEFAULT_ALPHA = 0.45
VOXEL_GRID_SIZE = 64

PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (238, 75, 43),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (231, 76, 60),
    (52, 73, 94),
    (127, 140, 141),
]
REMAINING_COLOR = (180, 180, 180)


@dataclass(frozen=True)
class PreviewPaths:
    output_dir: Path
    asset_dir: Path
    overlay_dir: Path
    label_dir: Path
    voxel_dir: Path
    vendor_dir: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static Part Completion sample preview HTML.")
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument(
        "--manifest",
        help="Part Completion manifest JSONL. Default: <data_root>/manifests/part_completion/arts_pc_<dataset_slug>_train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        help="Preview output directory. Default: <data_root>/preview/part_completion",
    )
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset.")
    parser.add_argument("--angle-ids", help="Optional comma-separated angle subset.")
    parser.add_argument("--view-ids", help="Optional comma-separated view subset.")
    parser.add_argument("--max-samples", type=int, help="Optional cap after filtering, for quick previews.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Overlay alpha in [0,1]. Default: 0.45.",
    )
    parser.add_argument(
        "--skip-voxel-previews",
        action="store_true",
        help="Do not emit interactive voxel payload JS files.",
    )
    return parser.parse_args(argv)


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace(" ", "_")


def _parse_csv(raw: str | None, field_name: str) -> set[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",")]
    if not values or any(not item for item in values):
        raise ValueError(f"{field_name} must be comma-separated non-empty values")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} contains duplicate values")
    return set(values)


def _parse_int_csv(raw: str | None, field_name: str) -> set[int] | None:
    values = _parse_csv(raw, field_name)
    if values is None:
        return None
    out: set[int] = set()
    for value in values:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain integers, got {value!r}") from exc
        if parsed < 0:
            raise ValueError(f"{field_name} must contain non-negative integers, got {parsed}")
        out.add(parsed)
    return out


def _relative_path(path: Path, base_dir: Path) -> str:
    resolved = path.resolve()
    base = base_dir.resolve()
    return resolved.relative_to(base).as_posix() if _is_relative_to(resolved, base) else resolved.as_posix()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def _row_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"row[{key!r}] must be int, got {type(value).__name__}")
    return int(value)


def _row_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"row[{key!r}] must be non-empty string")
    return value


def _hex_color(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _load_rows(
    manifest_path: Path,
    *,
    object_filter: set[str] | None,
    angle_filter: set[int] | None,
    view_filter: set[int] | None,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{manifest_path}:{line_no}: row must be object")
            if row.get("task") != "part_completion":
                raise ValueError(f"{manifest_path}:{line_no}: task must be part_completion")
            object_id = _row_string(row, "object_id")
            angle_idx = _row_int(row, "angle_idx")
            view_idx = _row_int(row, "view_idx")
            if object_filter is not None and object_id not in object_filter:
                continue
            if angle_filter is not None and angle_idx not in angle_filter:
                continue
            if view_filter is not None and view_idx not in view_filter:
                continue
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _color_for_label(label: int, label_to_component: dict[str, str]) -> tuple[int, int, int]:
    if label == BACKGROUND_LABEL:
        return (0, 0, 0)
    if label_to_component.get(str(label)) == REMAINING_NAME:
        return REMAINING_COLOR
    return PALETTE[label % len(PALETTE)]


def _colorize_label_mask(label_mask: np.ndarray, label_to_component: dict[str, str]) -> Image.Image:
    if label_mask.ndim != 2:
        raise ValueError(f"label mask must be 2D, got {label_mask.shape}")
    rgb = np.zeros((*label_mask.shape, 3), dtype=np.uint8)
    for label in np.unique(label_mask):
        label_int = int(label)
        rgb[label_mask == label_int] = _color_for_label(label_int, label_to_component)
    return Image.fromarray(rgb, mode="RGB")


def _make_overlay(rgb_path: Path, mask_path: Path, label_to_component: dict[str, str], alpha: float) -> tuple[Image.Image, Image.Image, list[int]]:
    if not rgb_path.is_file():
        raise FileNotFoundError(f"missing RGB image: {rgb_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"missing label mask: {mask_path}")
    rgb = Image.open(rgb_path).convert("RGB")
    label_mask = np.load(mask_path)
    if label_mask.ndim != 2:
        raise ValueError(f"label mask must be 2D: {mask_path} got {label_mask.shape}")
    if rgb.size != (label_mask.shape[1], label_mask.shape[0]):
        raise ValueError(f"RGB/mask size mismatch: {rgb_path} size={rgb.size}, {mask_path} shape={label_mask.shape}")
    colorized = _colorize_label_mask(label_mask.astype(np.int32), label_to_component)
    overlay = rgb.convert("RGBA")
    color_rgba = colorized.convert("RGBA")
    alpha_channel = np.zeros(label_mask.shape, dtype=np.uint8)
    alpha_channel[label_mask != 0] = int(max(0.0, min(1.0, alpha)) * 255)
    color_rgba.putalpha(Image.fromarray(alpha_channel, mode="L"))
    overlay.alpha_composite(color_rgba)
    return overlay.convert("RGB"), colorized, [int(item) for item in np.unique(label_mask).tolist()]


def _binary_mask_png_path(mask_npy_path: Path) -> Path | None:
    candidate = mask_npy_path.with_suffix(".png")
    return candidate if candidate.is_file() else None


def _load_voxel_coords(voxel_path: Path) -> list[list[int]]:
    if not voxel_path.is_file():
        raise FileNotFoundError(f"missing voxel path: {voxel_path}")
    data = np.load(voxel_path)
    if data.ndim == 2 and data.shape[1] == 3:
        coords = data
    elif data.ndim == 3:
        coords = np.argwhere(data > 0)
    else:
        raise ValueError(f"voxel data must be coords [N,3] or occupancy [D,H,W]: {voxel_path} got {data.shape}")
    if coords.size == 0:
        return []
    coords = np.asarray(coords, dtype=np.int64)
    coords = np.clip(coords, 0, VOXEL_GRID_SIZE - 1)
    coords = np.unique(coords, axis=0)
    return coords.astype(int).tolist()


def _coords_to_set(coords: list[list[int]]) -> set[tuple[int, int, int]]:
    return {tuple(int(v) for v in item) for item in coords}


def _set_to_coords(values: set[tuple[int, int, int]]) -> list[list[int]]:
    if not values:
        return []
    return [[int(x), int(y), int(z)] for x, y, z in sorted(values)]


def _metrics_from_sets(gt: set[tuple[int, int, int]], decoded: set[tuple[int, int, int]]) -> dict[str, Any]:
    intersection = gt & decoded
    union = gt | decoded
    return {
        "gt_count": len(gt),
        "decoded_count": len(decoded),
        "intersection": len(intersection),
        "union": len(union),
        "iou": (len(intersection) / len(union)) if union else 1.0,
        "precision": (len(intersection) / len(decoded)) if decoded else (1.0 if not gt else 0.0),
        "recall": (len(intersection) / len(gt)) if gt else 1.0,
        "false_positive": len(decoded - gt),
        "false_negative": len(gt - decoded),
    }


def _metric_float(metric: dict[str, Any], key: str) -> float | None:
    value = metric.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _metric_int(metric: dict[str, Any], key: str) -> int:
    value = metric.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0
    return int(value)


def _summarize_iou(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [metric for metric in metrics if _metric_float(metric, "iou") is not None]
    if not valid:
        return {
            "count": 0,
            "micro_iou": None,
            "mean_iou": None,
            "min_iou": None,
            "gt_count": 0,
            "decoded_count": 0,
            "intersection": 0,
            "union": 0,
        }

    intersections = sum(_metric_int(metric, "intersection") for metric in valid)
    unions = sum(_metric_int(metric, "union") for metric in valid)
    ious = [_metric_float(metric, "iou") for metric in valid]
    iou_values = [value for value in ious if value is not None]
    return {
        "count": len(valid),
        "micro_iou": (intersections / unions) if unions else 1.0,
        "mean_iou": sum(iou_values) / len(iou_values),
        "min_iou": min(iou_values),
        "gt_count": sum(_metric_int(metric, "gt_count") for metric in valid),
        "decoded_count": sum(_metric_int(metric, "decoded_count") for metric in valid),
        "intersection": intersections,
        "union": unions,
    }


def _build_iou_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    part_metrics: list[dict[str, Any]] = []
    surface_metrics: list[dict[str, Any]] = []
    for item in items:
        for target in item.get("targets", []):
            if isinstance(target, dict) and isinstance(target.get("voxel_metrics"), dict):
                part_metrics.append(target["voxel_metrics"])
        remaining = item.get("remaining")
        if isinstance(remaining, dict) and isinstance(remaining.get("voxel_metrics"), dict):
            surface_metrics.append(remaining["voxel_metrics"])
    return {
        "part": _summarize_iou(part_metrics),
        "surface": _summarize_iou(surface_metrics),
    }


def _format_summary_metric(summary: dict[str, Any]) -> str:
    count = int(summary.get("count", 0))
    if count <= 0:
        return "n/a"

    def fmt(value: Any) -> str:
        return f"{float(value):.4f}" if isinstance(value, (int, float)) and math.isfinite(float(value)) else "n/a"

    return (
        f"micro {fmt(summary.get('micro_iou'))} / "
        f"mean {fmt(summary.get('mean_iou'))} / "
        f"min {fmt(summary.get('min_iou'))} "
        f"(n={count}, ∩={summary.get('intersection', 0)}, ∪={summary.get('union', 0)})"
    )


def _format_iou_summary_header(iou_summary: dict[str, Any]) -> str:
    return (
        "IoU summary: "
        f"part {_format_summary_metric(iou_summary.get('part', {}))} · "
        f"surface {_format_summary_metric(iou_summary.get('surface', {}))}"
    )


def _voxel_expanded_dir(data_root: Path, object_id: str, angle_idx: int) -> Path:
    return data_root / "reconstruction" / "voxel_expanded" / object_id / f"angle_{angle_idx}" / str(VOXEL_GRID_SIZE)


def _ss_decoded_dir(data_root: Path, object_id: str, angle_idx: int) -> Path:
    return data_root / "reconstruction" / "ss_latent_decoded" / object_id / f"angle_{angle_idx}" / str(VOXEL_GRID_SIZE)


def _read_decoded_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - optional QC sidecar should not block preview
        return {}
    return data if isinstance(data, dict) else {}


def _js_safe_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_js_assignment(path: Path, prefix: str, payload: Any) -> None:
    _write_text_atomic(path, f"{prefix}{_js_safe_json(payload)};\n")


def _copy_vendor_assets(output_dir: Path) -> None:
    vendor_dir = output_dir / "vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for asset_name, src in CLASSIC_VENDOR_ASSETS.items():
        if not src.is_file():
            missing.append(f"{asset_name}: expected vendored file at {src}")
            continue
        shutil.copy2(src, vendor_dir / asset_name)
    if missing:
        raise FileNotFoundError("Missing Three.js vendor assets: " + " | ".join(missing))


def _reset_generated_dir(directory: Path, suffixes: set[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            path.unlink()


def _prepare_paths(output_dir: Path) -> PreviewPaths:
    asset_dir = output_dir / "assets"
    paths = PreviewPaths(
        output_dir=output_dir,
        asset_dir=asset_dir,
        overlay_dir=asset_dir / "overlays",
        label_dir=asset_dir / "labels",
        voxel_dir=output_dir / "voxels",
        vendor_dir=output_dir / "vendor",
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.asset_dir.mkdir(parents=True, exist_ok=True)
    _reset_generated_dir(paths.overlay_dir, {".png", ".jpg", ".jpeg"})
    _reset_generated_dir(paths.label_dir, {".png", ".jpg", ".jpeg"})
    _reset_generated_dir(paths.voxel_dir, {".js"})
    # Legacy location from the previous static-PNG implementation.
    _reset_generated_dir(paths.asset_dir / "voxels", {".png", ".jpg", ".jpeg"})
    _copy_vendor_assets(output_dir)
    return paths


def _target_label(part: dict[str, Any]) -> int:
    value = part.get("label", part.get("original_label"))
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"target part label must be int: {part}")
    return int(value)


def _voxel_payload_path(paths: PreviewPaths, sample_index: int) -> Path:
    return paths.voxel_dir / f"sample_{sample_index:06d}.js"


def _build_and_write_voxel_payload(
    *,
    row: dict[str, Any],
    data_root: Path,
    paths: PreviewPaths,
    sample_index: int,
    label_to_component: dict[str, str],
) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    sample_id = _row_string(row, "sample_id")
    object_id = _row_string(row, "object_id")
    angle_idx = _row_int(row, "angle_idx")
    voxel_dir = _voxel_expanded_dir(data_root, object_id, angle_idx)
    decoded_dir = _ss_decoded_dir(data_root, object_id, angle_idx)
    metrics_path = decoded_dir / "metrics.json"
    metrics_sidecar = _read_decoded_metrics(metrics_path)

    surface_path = voxel_dir / "surface.npy"
    decoded_overall_path = decoded_dir / "overall.npy"
    fatal_errors: list[str] = []

    def load_or_error(path: Path) -> list[list[int]]:
        try:
            return _load_voxel_coords(path)
        except Exception as exc:  # noqa: BLE001 - capture per-sample data issue in viewer
            fatal_errors.append(f"{path}: {exc}")
            return []

    surface = load_or_error(surface_path)
    decoded_overall = load_or_error(decoded_overall_path)
    surface_set = _coords_to_set(surface)
    decoded_overall_set = _coords_to_set(decoded_overall)
    surface_metrics = _metrics_from_sets(surface_set, decoded_overall_set)
    if isinstance(metrics_sidecar.get("overall"), dict):
        surface_metrics.update(metrics_sidecar["overall"])

    target_voxels: dict[str, list[list[int]]] = {}
    decoded_target_voxels: dict[str, list[list[int]]] = {}
    target_overlaps: dict[str, list[list[int]]] = {}
    part_metrics: dict[str, dict[str, Any]] = {}
    path_payload: dict[str, Any] = {
        "surface_gt": _relative_path(surface_path, paths.output_dir),
        "surface_decoded": _relative_path(decoded_overall_path, paths.output_dir),
        "metrics": _relative_path(metrics_path, paths.output_dir),
    }

    target_parts = row.get("target_parts")
    if not isinstance(target_parts, list):
        raise TypeError(f"{sample_id}: target_parts must be list")
    sidecar_parts = metrics_sidecar.get("parts") if isinstance(metrics_sidecar.get("parts"), dict) else {}
    for part in target_parts:
        if not isinstance(part, dict):
            raise TypeError(f"{sample_id}: target part must be object")
        part_name = _row_string(part, "name")
        part_paths = part.get("paths")
        if not isinstance(part_paths, dict):
            raise TypeError(f"{sample_id}: target part paths must be object")
        gt_path = _data_path(data_root, str(part_paths.get("part_voxel")))
        decoded_path = decoded_dir / "parts" / f"{part_name}.npy"
        gt_coords = load_or_error(gt_path)
        decoded_coords = load_or_error(decoded_path)
        gt_set = _coords_to_set(gt_coords)
        decoded_set = _coords_to_set(decoded_coords)
        target_voxels[part_name] = gt_coords
        decoded_target_voxels[part_name] = decoded_coords
        target_overlaps[part_name] = _set_to_coords(gt_set & decoded_set)
        computed = _metrics_from_sets(gt_set, decoded_set)
        if isinstance(sidecar_parts, dict) and isinstance(sidecar_parts.get(part_name), dict):
            computed.update(sidecar_parts[part_name])
        part_metrics[part_name] = computed
        path_payload.setdefault("target_gt", {})[part_name] = _relative_path(gt_path, paths.output_dir)
        path_payload.setdefault("target_decoded", {})[part_name] = _relative_path(decoded_path, paths.output_dir)

    payload = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "object_id": object_id,
        "angle_idx": angle_idx,
        "voxel_resolution": VOXEL_GRID_SIZE,
        "fatal_errors": fatal_errors,
        "paths": path_payload,
        "surface": surface,
        "decoded_overall": decoded_overall,
        "surface_overlap": _set_to_coords(surface_set & decoded_overall_set),
        "target_voxels": target_voxels,
        "decoded_target_voxels": decoded_target_voxels,
        "target_overlaps": target_overlaps,
        "decoded_metrics": {
            "overall": surface_metrics,
            "parts": part_metrics,
        },
        "label_to_component": label_to_component,
    }
    js_path = _voxel_payload_path(paths, sample_index)
    _write_js_assignment(
        js_path,
        f"window.__PC_VOXEL_PAYLOADS=window.__PC_VOXEL_PAYLOADS||{{}};window.__PC_VOXEL_PAYLOADS[{sample_index}]=",
        payload,
    )
    return _relative_path(js_path, paths.output_dir), surface_metrics, part_metrics


def _build_preview_items(
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    paths: PreviewPaths,
    alpha: float,
    skip_voxel_previews: bool,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = _row_string(row, "sample_id")
        object_id = _row_string(row, "object_id")
        angle_idx = _row_int(row, "angle_idx")
        view_idx = _row_int(row, "view_idx")
        image_path = _data_path(data_root, _row_string(row, "image_path"))
        mask_path = _data_path(data_root, _row_string(row, "mask_path"))
        label_to_component_raw = row.get("label_to_component")
        if not isinstance(label_to_component_raw, dict):
            raise TypeError(f"{sample_id}: label_to_component must be object")
        label_to_component = {str(key): str(value) for key, value in label_to_component_raw.items()}
        overlay, label_img, mask_unique = _make_overlay(image_path, mask_path, label_to_component, alpha)
        overlay_path = paths.overlay_dir / f"sample_{idx:06d}.jpg"
        label_png_path = paths.label_dir / f"sample_{idx:06d}.png"
        overlay.save(overlay_path, quality=92)
        label_img.save(label_png_path)

        paths_payload = row.get("paths")
        if not isinstance(paths_payload, dict):
            raise TypeError(f"{sample_id}: paths must be object")
        separated_raw = paths_payload.get("separated_masks")
        separated_masks = separated_raw if isinstance(separated_raw, dict) else {}

        voxel_payload_src: str | None = None
        surface_metrics: dict[str, Any] | None = None
        part_metrics: dict[str, dict[str, Any]] = {}
        if not skip_voxel_previews:
            voxel_payload_src, surface_metrics, part_metrics = _build_and_write_voxel_payload(
                row=row,
                data_root=data_root,
                paths=paths,
                sample_index=idx,
                label_to_component=label_to_component,
            )

        target_parts = row.get("target_parts")
        if not isinstance(target_parts, list):
            raise TypeError(f"{sample_id}: target_parts must be list")
        target_payloads: list[dict[str, Any]] = []
        for part in target_parts:
            if not isinstance(part, dict):
                raise TypeError(f"{sample_id}: target part must be object")
            part_name = _row_string(part, "name")
            label = _target_label(part)
            part_paths = part.get("paths")
            if not isinstance(part_paths, dict):
                raise TypeError(f"{sample_id}: target part paths must be object")
            part_mask_path = _data_path(data_root, str(part_paths.get("part_mask") or separated_masks.get(part_name)))
            part_mask_png = _binary_mask_png_path(part_mask_path)
            part_voxel_path = _data_path(data_root, str(part_paths.get("part_voxel")))
            decoded_part_path = _ss_decoded_dir(data_root, object_id, angle_idx) / "parts" / f"{part_name}.npy"
            target_payloads.append(
                {
                    "name": part_name,
                    "label": label,
                    "color": _hex_color(_color_for_label(label, label_to_component)),
                    "visible_pixels": int(part.get("visible_pixels", 0)),
                    "num_voxels": int(part.get("num_voxels", -1)),
                    "mask": _relative_path(part_mask_png or part_mask_path, paths.output_dir),
                    "voxel_gt_path": _relative_path(part_voxel_path, paths.output_dir),
                    "voxel_decoded_path": _relative_path(decoded_part_path, paths.output_dir),
                    "voxel_metrics": part_metrics.get(part_name),
                }
            )

        remaining = row.get("remaining")
        remaining_payload: dict[str, Any] = {}
        if isinstance(remaining, dict):
            remaining_mask_path = _data_path(data_root, str(remaining.get("mask_path")))
            remaining_mask_png = _binary_mask_png_path(remaining_mask_path)
            surface_gt_path = _voxel_expanded_dir(data_root, object_id, angle_idx) / "surface.npy"
            surface_decoded_path = _ss_decoded_dir(data_root, object_id, angle_idx) / "overall.npy"
            remaining_payload = {
                "label": int(remaining.get("label", -1)),
                "visible_pixels": int(remaining.get("visible_pixels", 0)),
                "mask": _relative_path(remaining_mask_png or remaining_mask_path, paths.output_dir),
                "surface_gt_path": _relative_path(surface_gt_path, paths.output_dir),
                "surface_decoded_path": _relative_path(surface_decoded_path, paths.output_dir),
                "voxel_metrics": surface_metrics,
            }

        items.append(
            {
                "index": idx,
                "sample_id": sample_id,
                "object_id": object_id,
                "angle_idx": angle_idx,
                "view_idx": view_idx,
                "target_part_count": int(row.get("target_part_count", len(target_payloads))),
                "target_part_names": [part["name"] for part in target_payloads],
                "mask_unique": mask_unique,
                "rgb": _relative_path(image_path, paths.output_dir),
                "label_mask": _relative_path(label_png_path, paths.output_dir),
                "overlay": _relative_path(overlay_path, paths.output_dir),
                "label_mask_npy": _relative_path(mask_path, paths.output_dir),
                "targets": target_payloads,
                "remaining": remaining_payload,
                "voxel_payload_src": voxel_payload_src,
            }
        )
    return items


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Part Completion Preview</title>
<style>
:root{color-scheme:dark;--bg:#0b0f17;--panel:#111827;--panel2:#0b1220;--muted:#9ca3af;--text:#f9fafb;--accent:#60a5fa;--border:#263244;--good:#22c55e;--bad:#ef4444;--warn:#f59e0b}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}button,input{font:inherit}header{height:96px;padding:14px 20px;border-bottom:1px solid var(--border);background:#0f172a}h1{margin:0 0 5px;font-size:20px}.meta{color:var(--muted);font-size:13px;line-height:1.45}.meta.stats{color:#dbeafe}.app{display:grid;grid-template-columns:370px minmax(0,1fr);height:calc(100vh - 96px)}.sidebar{border-right:1px solid var(--border);overflow:hidden;background:var(--panel2);display:flex;flex-direction:column;min-height:0}#list{flex:1;min-height:0;overflow:auto;position:relative}.filters{padding:12px;position:sticky;top:0;background:var(--panel2);border-bottom:1px solid var(--border);z-index:2}input{width:100%;background:#111827;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 10px}.pager{position:sticky;top:0;z-index:2;padding:10px 12px;border-bottom:1px solid var(--border);background:rgba(11,18,32,.96);backdrop-filter:blur(10px)}.pager-row{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}.pager-row:last-child{margin-bottom:0}.pager button{height:30px;border:1px solid #3e5066;background:#172033;color:var(--text);border-radius:6px;padding:0 10px;cursor:pointer}.pager button:not([disabled]):hover{border-color:var(--accent);color:#dbeafe}.pager button[disabled]{opacity:.42;cursor:not-allowed}.pager .range{color:#dbeafe;font-weight:750}.pager small{color:var(--muted);font-size:11px}.pager input{width:74px;height:30px;border-radius:6px;padding:0 7px}.card{width:100%;text-align:left;border:0;border-bottom:1px solid var(--border);background:transparent;color:var(--text);padding:10px 12px;cursor:pointer;display:grid;grid-template-columns:72px 1fr;gap:10px}.card:hover,.card.active{background:#172033}.card img{width:72px;height:72px;object-fit:cover;border-radius:6px;border:1px solid var(--border)}.card b{display:block;font-size:13px;line-height:1.25}.card small{display:block;color:var(--muted);margin-top:4px}.detail{min-width:0;overflow:auto;padding:16px}.sample-title{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:14px}h2{margin:0;font-size:20px}.badge{display:inline-block;padding:3px 8px;border:1px solid var(--border);border-radius:999px;color:#dbeafe;background:#1e3a8a55;font-size:12px;margin-right:6px}.grid3{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:14px;margin-bottom:14px}.panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:10px}.panel h3{margin:0 0 10px;font-size:14px;color:#d1d5db;display:flex;justify-content:space-between;gap:8px}.panel img{width:100%;background-color:#fff;background-image:linear-gradient(45deg,#e5e7eb 25%,transparent 25%),linear-gradient(-45deg,#e5e7eb 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#e5e7eb 75%),linear-gradient(-45deg,transparent 75%,#e5e7eb 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0;border-radius:8px;border:1px solid var(--border)}.workspace{display:block}.targets{display:grid;grid-template-columns:1fr;gap:12px}.target h4{margin:0 0 8px}.target-meta{color:var(--muted);font-size:12px;line-height:1.35;margin-bottom:6px}.target-pair{display:grid;grid-template-columns:minmax(120px,.24fr) minmax(300px,1fr);gap:10px;align-items:start}.target-pair img{max-height:120px;object-fit:contain}.remaining{border-color:#4b5563}.voxel-panel{min-height:300px;display:flex;flex-direction:column}.voxel-wrap{position:relative;height:260px;min-height:260px;flex:0 0 260px;background:#070c12;border-radius:10px;overflow:hidden;border:1px solid var(--border)}.voxel-wrap canvas{width:100%!important;height:100%!important;display:block}.voxel-controls{position:absolute;right:8px;top:8px;z-index:3;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:4px;max-width:calc(100% - 16px);pointer-events:auto}.voxel-layer-btn{min-height:24px;border:1px solid #3e5066;background:rgba(22,33,48,.9);color:var(--muted);border-radius:6px;padding:0 6px;font-size:10.5px;font-weight:750;cursor:pointer}.voxel-layer-btn.active{border-color:var(--accent);background:rgba(96,165,250,.95);color:#061014}.voxel-layer-btn:not(.active){opacity:.72;text-decoration:line-through}.voxel-info{position:absolute;left:8px;bottom:8px;right:8px;padding:5px 8px;border-radius:7px;background:rgba(0,0,0,.68);color:#d1d5db;font-size:11px;line-height:1.3;max-height:58px;overflow:auto}.legend{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 0;color:var(--muted);font-size:11px}.legend span{display:inline-flex;align-items:center;gap:5px}.swatch{width:10px;height:10px;border-radius:2px;display:inline-block}.metric{display:inline-block;margin-right:8px;color:#cbd5e1}.metric strong{color:#fff}.metric .good{color:var(--good)}.metric .bad{color:var(--bad)}.links a{margin-right:12px}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.empty{padding:24px;color:var(--muted);text-align:center}.error{border:1px solid rgba(239,68,68,.6);background:rgba(239,68,68,.12);color:#fecaca;padding:9px;border-radius:10px}
@media(max-width:1250px){body{overflow:auto}.app{display:block;height:auto}.sidebar{max-height:360px;border-right:0;border-bottom:1px solid var(--border)}.detail{overflow:visible}.grid3{grid-template-columns:1fr}.workspace{grid-template-columns:1fr}.voxel-panel{position:relative}.voxel-wrap{height:520px}.target-pair{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1>Part Completion Preview</h1><div class="meta">manifest: __MANIFEST__ · data_root: __DATA_ROOT__ · samples: <span id="total">__SAMPLE_COUNT__</span> · voxel: Three.js draggable GT / SS decoder comparison</div><div class="meta stats">__IOU_SUMMARY__</div></header>
<div class="app"><aside class="sidebar"><div class="filters"><input id="filter" placeholder="filter: object / sample / target / angle / view"></div><div id="list"></div></aside><main class="detail" id="detail"><div class="empty">选择左侧 sample</div></main></div>
<script src="vendor/three.min.js"></script>
<script src="vendor/OrbitControls.js"></script>
<script>
window.__PC_VOXEL_PAYLOADS=window.__PC_VOXEL_PAYLOADS||{};
let samples=[];let filtered=[];let current=null;let activeScript=new Set();let scriptPromises=new Map();let scriptElements=new Map();let loadedVoxelIndex=null;let loadedVoxelSrc=null;let currentVoxelPayload=null;let voxelViewers=[];let page=0;
const PAGE_SIZE=60;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function cardText(s){return `${s.sample_id} ${s.object_id} angle_${s.angle_idx} view_${s.view_idx} ${s.target_part_names.join(' ')}`;}
function scriptOnce(src){if(!src)return Promise.reject(new Error('missing script src'));if(activeScript.has(src))return Promise.resolve();if(scriptPromises.has(src))return scriptPromises.get(src);const p=new Promise((res,rej)=>{const el=document.createElement('script');el.src=src;el.onload=()=>{activeScript.add(src);scriptElements.set(src,el);res();};el.onerror=()=>{el.remove();rej(new Error('加载失败 '+src));};document.body.appendChild(el);}).finally(()=>scriptPromises.delete(src));scriptPromises.set(src,p);return p;}
function forgetScript(src){if(!src)return;activeScript.delete(src);scriptPromises.delete(src);const oldEl=scriptElements.get(src);if(oldEl)oldEl.remove();scriptElements.delete(src);}
function metricText(m){if(!m)return 'metrics n/a';const f=x=>Number.isFinite(Number(x))?Number(x).toFixed(4):'n/a';const cls=Number(m.iou)>=.98?'good':'bad';return `<span class="metric">IoU <strong class="${cls}">${f(m.iou)}</strong></span><span class="metric">P <strong>${f(m.precision)}</strong></span><span class="metric">R <strong>${f(m.recall)}</strong></span><span class="metric">GT <strong>${m.gt_count??'?'}</strong></span><span class="metric">SS <strong>${m.decoded_count??'?'}</strong></span><span class="metric">∩ <strong>${m.intersection??'?'}</strong></span><span class="metric">FP <strong>${m.false_positive??'?'}</strong></span><span class="metric">FN <strong>${m.false_negative??'?'}</strong></span>`;}
function metricPlain(m){if(!m)return 'metrics n/a';const f=x=>Number.isFinite(Number(x))?Number(x).toFixed(4):'n/a';return `IoU ${f(m.iou)} P ${f(m.precision)} R ${f(m.recall)} GT ${m.gt_count??'?'} SS ${m.decoded_count??'?'} ∩ ${m.intersection??'?'} FP ${m.false_positive??'?'} FN ${m.false_negative??'?'}`;}
function imgPanel(title,src,link){return `<div class="panel"><h3><span>${esc(title)}</span><a href="${esc(link||src)}" target="_blank">open</a></h3><img src="${esc(src)}"></div>`;}
function listEl(){return document.getElementById('list');}
function pageCount(){return Math.max(1,Math.ceil(filtered.length/PAGE_SIZE));}
function gotoPage(next){page=Math.min(Math.max(0,next),pageCount()-1);renderList();listEl().scrollTop=0;}
function renderList(){const l=listEl();const total=pageCount();page=Math.min(Math.max(0,page),total-1);const start=page*PAGE_SIZE;const end=Math.min(start+PAGE_SIZE,filtered.length);let rows='';for(let i=start;i<end;i++){const s=filtered[i];const active=current&&current.index===s.index?'active':'';rows+=`<button class="card ${active}" data-index="${s.index}"><img src="${esc(s.overlay)}"><span><b>#${s.index} ${esc(s.object_id)} angle_${s.angle_idx} view_${s.view_idx}</b><small>${esc(s.sample_id)}</small><small>targets(${s.target_part_count}): ${esc(s.target_part_names.join(', '))}</small><small>labels: ${esc(s.mask_unique.join(', '))}</small></span></button>`;}l.innerHTML=`<div class="pager"><div class="pager-row"><button data-page="prev" ${page===0?'disabled':''}>上一页</button><span class="range">${filtered.length?start+1:0}-${end} / ${filtered.length}</span><button data-page="next" ${page>=total-1?'disabled':''}>下一页</button></div><div class="pager-row"><small>第 ${page+1} / ${total} 页，每页 ${PAGE_SIZE} 条</small><span><input id="page-input" type="number" min="1" max="${total}" value="${page+1}"><button data-page="jump">跳页</button></span></div></div>${rows||'<div class="empty">没有匹配样本</div>'}`;l.querySelector('[data-page="prev"]')?.addEventListener('click',()=>gotoPage(page-1));l.querySelector('[data-page="next"]')?.addEventListener('click',()=>gotoPage(page+1));l.querySelector('[data-page="jump"]')?.addEventListener('click',()=>gotoPage(Number(document.getElementById('page-input').value||1)-1));l.querySelector('#page-input')?.addEventListener('keydown',e=>{if(e.key==='Enter')gotoPage(Number(e.currentTarget.value||1)-1);});l.querySelectorAll('[data-index]').forEach(btn=>btn.onclick=()=>openSample(Number(btn.dataset.index)));}
function layerBtn(layer,label){return `<button class="voxel-layer-btn active" data-layer="${layer}" type="button">${label}</button>`;}
function viewerMarkup(kind,partName,title,metricsHtml,linksHtml){const safeKind=esc(kind),safePart=esc(partName||'');return `<div class="panel voxel-panel"><h3><span>${esc(title)}</span><span>drag / wheel / toggle</span></h3><div class="target-meta">${metricsHtml}<br>${linksHtml||''}</div><div class="voxel-wrap" data-viewer-kind="${safeKind}" data-part="${safePart}"><div class="voxel-controls">${layerBtn('gt','GT')} ${layerBtn('decoded','SS decoder')} ${layerBtn('overlap','overlap')}</div><div class="empty voxel-empty">加载 voxel payload...</div><div class="voxel-info"></div></div><div class="legend"><span><i class="swatch" style="background:#888"></i>GT</span><span><i class="swatch" style="background:#ff4de3"></i>SS decoder</span><span><i class="swatch" style="background:#22c55e"></i>overlap</span><span>坐标轴 X红 Y绿 Z蓝</span></div></div>`;}
function targetCard(t){const links=`<span class="links"><a href="${esc(t.voxel_gt_path)}" target="_blank">GT npy</a><a href="${esc(t.voxel_decoded_path)}" target="_blank">SS decoder npy</a></span>`;return `<div class="panel target"><h4><span class="swatch" style="background:${esc(t.color)}"></span>${esc(t.name)} <span class="badge">label ${t.label}</span></h4><div class="target-meta">visible px: ${t.visible_pixels} · manifest voxels: ${t.num_voxels}</div><div class="target-pair">${imgPanel('separated mask',t.mask,t.mask)}${viewerMarkup('target',t.name,`${t.name}: GT vs SS decoder`,metricText(t.voxel_metrics),links)}</div></div>`;}
function remainingCard(r){if(!r||!r.mask)return '';const links=`<span class="links"><a href="${esc(r.surface_gt_path)}" target="_blank">surface GT npy</a><a href="${esc(r.surface_decoded_path)}" target="_blank">overall SS decoder npy</a></span>`;return `<div class="panel target remaining"><h4>remaining <span class="badge">label ${r.label}</span></h4><div class="target-meta">visible px: ${r.visible_pixels} · voxel viewer 使用整体 surface GT vs overall SS decoder</div><div class="target-pair">${imgPanel('remaining mask',r.mask,r.mask)}${viewerMarkup('surface','',`surface: GT vs overall SS decoder`,metricText(r.voxel_metrics),links)}</div></div>`;}
function renderDetail(){const d=current;if(!d){document.getElementById('detail').innerHTML='<div class="empty">没有样本</div>';return;}if(!Array.isArray(d.targets)){document.getElementById('detail').innerHTML='<div class="empty">选择左侧 sample</div>';return;}disposeVoxelViewers();document.getElementById('detail').innerHTML=`<div class="sample-title"><div><h2>#${d.index} ${esc(d.sample_id)}</h2><div class="meta">object=${esc(d.object_id)} · angle_${d.angle_idx} · view_${d.view_idx} · target_count=${d.target_part_count} · interactive viewers=${d.targets.length+1} · mask labels=[${d.mask_unique.join(', ')}]</div></div><div>${d.target_part_names.map(x=>`<span class="badge">${esc(x)}</span>`).join('')}</div></div><div class="grid3">${imgPanel('RGB',d.rgb,d.rgb)}${imgPanel('label mask',d.label_mask,d.label_mask_npy)}${imgPanel('overlay',d.overlay,d.overlay)}</div><div class="workspace"><div class="targets">${d.targets.map(targetCard).join('')}${remainingCard(d.remaining)}</div></div>`;initVoxelViewers();loadVoxel(d.index).catch(e=>setAllViewerErrors(e.message));}
async function openSample(index){const thin=samples.find(x=>x.index===index)||samples[index];if(!thin)return;current=thin;currentVoxelPayload=null;releaseVoxelPayload(thin.index);disposeVoxelViewers();renderList();const det=document.getElementById('detail');det.innerHTML='<div class="empty">加载 sample 详情...</div>';const src='details/sample_'+String(thin.index).padStart(6,'0')+'.js';try{await scriptOnce(src);if(!current||current.index!==thin.index)return;const d=(window.__PC_DETAILS||{})[thin.index];if(!d)throw new Error('detail payload missing');current=Object.assign({},thin,d);}catch(e){if(current&&current.index===thin.index)det.innerHTML='<div class="empty error">详情加载失败：'+esc(e.message)+'</div>';return;}renderDetail();}
function releaseVoxelPayload(nextIndex){if(loadedVoxelIndex!==null&&loadedVoxelIndex!==nextIndex){delete window.__PC_VOXEL_PAYLOADS[loadedVoxelIndex];forgetScript(loadedVoxelSrc);loadedVoxelSrc=null;}loadedVoxelIndex=nextIndex;}
function makeAxisLabel(text,color){const canvas=document.createElement('canvas');canvas.width=128;canvas.height=64;const ctx=canvas.getContext('2d');ctx.font='700 34px ui-monospace,monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.lineWidth=8;ctx.strokeStyle='rgba(0,0,0,.85)';ctx.fillStyle=color;ctx.strokeText(text,64,32);ctx.fillText(text,64,32);const tex=new THREE.CanvasTexture(canvas);tex.minFilter=THREE.LinearFilter;const mat=new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false,depthWrite:false});const sprite=new THREE.Sprite(mat);sprite.scale.set(10,5,1);return sprite;}
function addVoxelAxes(scene){const axisLen=72;const origin=new THREE.Vector3(0,0,0);const group=new THREE.Group();const axes=[{name:'X',dir:new THREE.Vector3(1,0,0),color:0xff4d4d,css:'#ff4d4d'},{name:'Y',dir:new THREE.Vector3(0,1,0),color:0x42e66f,css:'#42e66f'},{name:'Z',dir:new THREE.Vector3(0,0,1),color:0x4da3ff,css:'#4da3ff'}];axes.forEach(a=>{const arrow=new THREE.ArrowHelper(a.dir,origin,axisLen,a.color,5,2.6);arrow.line.material.depthTest=false;arrow.cone.material.depthTest=false;group.add(arrow);const label=makeAxisLabel(a.name,a.css);label.position.copy(a.dir.clone().multiplyScalar(axisLen+7));group.add(label);});const boxGeo=new THREE.EdgesGeometry(new THREE.BoxGeometry(63,63,63));const boxMat=new THREE.LineBasicMaterial({color:0x33485f,transparent:true,opacity:.68});const box=new THREE.LineSegments(boxGeo,boxMat);box.position.set(31.5,31.5,31.5);group.add(box);scene.add(group);}
function initVoxelViewers(){voxelViewers=[];document.querySelectorAll('.voxel-wrap[data-viewer-kind]').forEach((root,idx)=>{const scene=new THREE.Scene();const camera=new THREE.PerspectiveCamera(45,1,.1,500);camera.position.set(128,96,128);const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setClearColor(0x070c12);root.appendChild(renderer.domElement);const controls=new THREE.OrbitControls(camera,renderer.domElement);controls.target.set(32,32,32);controls.enableDamping=true;scene.add(new THREE.AmbientLight(0xffffff,.62));const dl=new THREE.DirectionalLight(0xffffff,.85);dl.position.set(60,90,60);scene.add(dl);addVoxelAxes(scene);const viewer={id:idx,root,kind:root.dataset.viewerKind,part:root.dataset.part||'',scene,camera,renderer,controls,meshes:[],layers:{gt:true,decoded:true,overlap:true}};root.querySelectorAll('[data-layer]').forEach(btn=>{btn.onclick=()=>{const layer=btn.dataset.layer;viewer.layers[layer]=!viewer.layers[layer];btn.classList.toggle('active',viewer.layers[layer]);btn.setAttribute('aria-pressed',String(viewer.layers[layer]));if(currentVoxelPayload)renderSingleViewer(viewer,currentVoxelPayload);};});const resize=()=>{const w=root.clientWidth,h=root.clientHeight;if(w&&h){renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}};new ResizeObserver(resize).observe(root);resize();voxelViewers.push(viewer);});ensureVoxelAnimation();}
function ensureVoxelAnimation(){if(window.__pcVoxelAnim)return;window.__pcVoxelAnim=true;(function anim(){requestAnimationFrame(anim);voxelViewers.forEach(v=>{v.controls.update();v.renderer.render(v.scene,v.camera);});})();}
function disposeVoxelViewers(){voxelViewers.forEach(v=>{clearViewer(v);v.renderer.dispose();v.root.querySelector('canvas')?.remove();});voxelViewers=[];}
function mesh(coords,color,opacity,size=.9){const geo=new THREE.BoxGeometry(size,size,size);const mat=new THREE.MeshLambertMaterial({color:new THREE.Color(color),transparent:opacity<1,opacity,depthWrite:opacity>=1});const inst=new THREE.InstancedMesh(geo,mat,coords.length);const d=new THREE.Object3D();coords.forEach((c,i)=>{d.position.set(c[0],c[1],c[2]);d.updateMatrix();inst.setMatrixAt(i,d.matrix);});inst.instanceMatrix.needsUpdate=true;return inst;}
function clearViewer(v){v.meshes.forEach(m=>{v.scene.remove(m);m.geometry.dispose();m.material.dispose();});v.meshes=[];}
function addMeshToViewer(v,coords,color,opacity,size){if(!coords||!coords.length)return 0;const m=mesh(coords,color,opacity,size);v.scene.add(m);v.meshes.push(m);return coords.length;}
function renderSingleViewer(viewer,payload){clearViewer(viewer);const empty=viewer.root.querySelector('.voxel-empty');const info=viewer.root.querySelector('.voxel-info');if(payload.fatal_errors&&payload.fatal_errors.length){empty.style.display='block';empty.innerHTML='<div class="error"><b>Voxel fatal error</b><br>'+payload.fatal_errors.map(esc).join('<br>')+'</div>';info.textContent='voxel not rendered; no fallback used';return;}empty.style.display='none';let gt=[],decoded=[],overlap=[],gtColor='#888888',decodedColor='#ff4de3',title='';let metrics=null;if(viewer.kind==='surface'){gt=payload.surface||[];decoded=payload.decoded_overall||[];overlap=payload.surface_overlap||[];gtColor='#888888';decodedColor='#00e5ff';metrics=payload.decoded_metrics?.overall;title='surface';}else{const t=current.targets.find(x=>x.name===viewer.part);gt=payload.target_voxels?.[viewer.part]||[];decoded=payload.decoded_target_voxels?.[viewer.part]||[];overlap=payload.target_overlaps?.[viewer.part]||[];gtColor=t?.color||'#f97316';decodedColor='#ff4de3';metrics=payload.decoded_metrics?.parts?.[viewer.part];title=viewer.part;}const shownGt=viewer.layers.gt?addMeshToViewer(viewer,gt,gtColor,.58,.9):'off';const shownDecoded=viewer.layers.decoded?addMeshToViewer(viewer,decoded,decodedColor,.44,1.06):'off';const shownOverlap=viewer.layers.overlap?addMeshToViewer(viewer,overlap,'#22c55e',.96,1.14):'off';info.textContent=`${title} | GT ${shownGt==='off'?'off':gt.length} | SS ${shownDecoded==='off'?'off':decoded.length} | overlap ${shownOverlap==='off'?'off':overlap.length} | ${metricPlain(metrics)} | 拖动旋转 / 滚轮缩放`;}
function renderAllVoxelViewers(payload){voxelViewers.forEach(v=>renderSingleViewer(v,payload));}
function setAllViewerErrors(message){document.querySelectorAll('.voxel-info').forEach(info=>{info.textContent=message;});}
async function loadVoxel(index){setAllViewerErrors('loading voxel payload...');document.querySelectorAll('.voxel-empty').forEach(e=>{e.style.display='block';e.textContent='loading voxel payload...';});if(!current?.voxel_payload_src){setAllViewerErrors('voxel payload skipped');return;}const src=current.voxel_payload_src;await scriptOnce(src);if(!current||current.index!==index){delete window.__PC_VOXEL_PAYLOADS[index];forgetScript(src);return;}loadedVoxelIndex=index;loadedVoxelSrc=src;const v=window.__PC_VOXEL_PAYLOADS[index];if(!v)throw new Error('voxel payload missing');currentVoxelPayload=v;renderAllVoxelViewers(v);}
(async function init(){const l=listEl();l.innerHTML='<div class="empty">加载 samples.js...</div>';try{await scriptOnce('samples.js');if(!Array.isArray(window.__PC_SAMPLES))throw new Error('window.__PC_SAMPLES not an array');samples=window.__PC_SAMPLES;}catch(e){l.innerHTML='<div class="empty error">samples.js 加载失败：'+esc(e.message)+'</div>';return;}filtered=samples.slice();current=null;document.getElementById('filter').addEventListener('input',e=>{const q=e.target.value.trim().toLowerCase();filtered=q?samples.filter(s=>cardText(s).toLowerCase().includes(q)):samples.slice();page=0;const keep=current&&filtered.some(s=>s.index===current.index);if(!keep){current=null;currentVoxelPayload=null;releaseVoxelPayload(null);disposeVoxelViewers();document.getElementById('detail').innerHTML='<div class="empty">选择左侧 sample</div>';}l.scrollTop=0;renderList();});renderList();document.getElementById('detail').innerHTML='<div class="empty">选择左侧 sample</div>';})();
</script>
</body>
</html>
'''


# Fields kept in the small samples.js index (used for the sidebar list,
# filter, and thumbnail). Everything else lives in per-sample
# details/sample_<N>.js files that are only fetched when the user opens
# that sample. Pages opened over file:// load scripts via <script src=...>
# tags (fetch() is blocked), so we materialise both index and details as
# JS assignments.
_PC_INDEX_FIELDS = (
    "index",
    "sample_id",
    "object_id",
    "angle_idx",
    "view_idx",
    "target_part_count",
    "target_part_names",
    "mask_unique",
    "overlay",
)


def _write_html(
    path: Path,
    items: list[dict[str, Any]],
    manifest_path: Path,
    data_root: Path,
    iou_summary: dict[str, Any],
) -> None:
    output_dir = path.parent
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    thin_items: list[dict[str, Any]] = []
    for sample in items:
        thin = {key: sample[key] for key in _PC_INDEX_FIELDS if key in sample}
        thin_items.append(thin)
        detail = {key: value for key, value in sample.items() if key not in _PC_INDEX_FIELDS}
        idx = sample["index"]
        detail_path = details_dir / f"sample_{int(idx):06d}.js"
        _write_js_assignment(
            detail_path,
            f"window.__PC_DETAILS=window.__PC_DETAILS||{{}};window.__PC_DETAILS[{int(idx)}]=",
            detail,
        )

    samples_path = output_dir / "samples.js"
    _write_js_assignment(samples_path, "window.__PC_SAMPLES=", thin_items)

    html_text = (
        HTML_TEMPLATE
        .replace("__MANIFEST__", html.escape(str(manifest_path)))
        .replace("__DATA_ROOT__", html.escape(str(data_root)))
        .replace("__SAMPLE_COUNT__", str(len(items)))
        .replace("__IOU_SUMMARY__", html.escape(_format_iou_summary_header(iou_summary)))
    )
    path.write_text(html_text, encoding="utf-8")


def build_preview(args: argparse.Namespace) -> int:
    if args.alpha < 0.0 or args.alpha > 1.0 or not math.isfinite(args.alpha):
        raise ValueError("--alpha must be finite and in [0,1]")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1")
    cfg = load_config(args.config)
    data_root = Path(cfg.data_root)
    dataset_slug = _dataset_slug(cfg.dataset_name)
    manifest_path = Path(args.manifest) if args.manifest else (
        data_root / "manifests" / "part_completion" / f"arts_pc_{dataset_slug}_train.jsonl"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Part Completion manifest not found: {manifest_path}")
    output_dir = Path(args.output_dir) if args.output_dir else (data_root / "preview" / "part_completion")
    paths = _prepare_paths(output_dir)

    rows = _load_rows(
        manifest_path,
        object_filter=_parse_csv(args.object_ids, "--object-ids"),
        angle_filter=_parse_int_csv(args.angle_ids, "--angle-ids"),
        view_filter=_parse_int_csv(args.view_ids, "--view-ids"),
        max_samples=args.max_samples,
    )
    if not rows:
        raise ValueError("No Part Completion samples selected")

    items = _build_preview_items(
        rows,
        data_root=data_root,
        paths=paths,
        alpha=args.alpha,
        skip_voxel_previews=args.skip_voxel_previews,
    )
    iou_summary = _build_iou_summary(items)
    index_path = output_dir / "index.html"
    _write_html(index_path, items, manifest_path, data_root, iou_summary)
    voxel_payload_count = len(list(paths.voxel_dir.glob("sample_*.js")))
    manifest_copy = output_dir / "preview_manifest.json"
    manifest_copy.write_text(
        json.dumps(
            {
                "source_manifest": str(manifest_path),
                "data_root": str(data_root),
                "sample_count": len(items),
                "index": str(index_path),
                "voxel_preview_mode": "threejs_interactive_gt_vs_ss_decoder",
                "voxel_payload_count": voxel_payload_count,
                "iou_summary": iou_summary,
                "filters": {
                    "object_ids": args.object_ids,
                    "angle_ids": args.angle_ids,
                    "view_ids": args.view_ids,
                    "max_samples": args.max_samples,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Part Completion preview: {index_path}")
    print(f"Samples: {len(items)}")
    print(f"Voxel payloads: {voxel_payload_count}")
    print(_format_iou_summary_header(iou_summary))
    print(f"Assets: {paths.asset_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return build_preview(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
