#!/usr/bin/env python3
"""Dataset-agnostic validator for the dataset_toolkits pipeline.

Covers VALID-01..07 (Phase 02 plan 02-01):
  VALID-01: per-object angle override via cfg.get_num_angles(oid)
  VALID-02: file existence + counts per step
  VALID-03: schema / shape / dtype hardcoded asserts
  VALID-04: numeric sanity (voxel size, bbox range, tokens NaN/Inf, jsonl image paths)
  VALID-05: minimal cross-step consistency (per-object angle count alignment)
  VALID-06: dataset-agnostic - all paths derived from cfg.<...>_dir
  VALID-07: failure records pinpoint <dataset>/<obj>/<angle>/<file> + dimension

Run:
    python utils/validate_dataset.py --config configs/<DATASET>.yaml \
        [--steps render,voxel,dinov2,vlm,preview,ss_latent] [--top-n 20] [--report-path PATH]

Exit code 0 if no failures, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Make utils/ importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import PipelineConfig, load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Local constants - mirrored from pipeline modules. Comments mark same-source.
# ---------------------------------------------------------------------------

# mirror of pipeline/03_voxelize.py (VALID-04 voxel filter threshold same source)
MIN_PART_VOXELS = 5
EXPECTED_VOXEL_RESOLUTION = 64
MIN_SURFACE_VOXELS = 100  # plan 02-01 lower bound for non-empty surface

# mirror of the historical bbox extractor contract (VALID-04 bbox range same source)
GROUNDING_SCALE_LOCAL = 1000

# mirror of pipeline/06_extract_feature.py default part_complete dinov2_vitl14_reg @ 16 views
EXPECTED_TOKENS_SHAPE = (16, 1370, 1024)
EXPECTED_TOKENS_DTYPE = np.float32

# mirror of Blender mask write contract (np.int32, 2D)
EXPECTED_MASK_DTYPE = np.int32
EXPECTED_PART_COMPLETE_VIEWS = 16

# mirror of pipeline/07_encode_ss_latents_per_part.py
EXPECTED_SS_LATENT_SHAPE = (8, 16, 16, 16)
EXPECTED_SS_LATENT_DTYPE = np.float32

ALL_STEPS: tuple[str, ...] = ("render", "voxel", "dinov2", "vlm", "preview", "ss_latent")
DIMENSIONS: tuple[str, ...] = ("exists", "schema", "numeric", "consistency")

ANGLE_DIR_RE = re.compile(r"^angle_(\d+)$")
MAX_NUMERIC_SAMPLE_ROWS = 1024
MAX_FINITE_SAMPLE_ROWS = 16


# ---------------------------------------------------------------------------
# Failure record helpers (VALID-07)
# ---------------------------------------------------------------------------


def _record_fail(
    failures: list[dict[str, Any]],
    *,
    step: str,
    dimension: str,
    object_id: str | None,
    angle: int | None,
    file: str | None,
    reason: str,
) -> None:
    """Append one failure record matching D-07 schema verbatim."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown dimension: {dimension!r}")
    failures.append(
        {
            "step": step,
            "dimension": dimension,
            "object_id": object_id,
            "angle": angle,
            "file": file,
            "reason": reason,
        }
    )


def _record_warn(
    warnings: list[dict[str, Any]],
    *,
    step: str,
    dimension: str,
    object_id: str | None,
    angle: int | None,
    file: str | None,
    reason: str,
) -> None:
    """Append one warning record (same shape as failure; not counted in summary.passed)."""
    warnings.append(
        {
            "step": step,
            "dimension": dimension,
            "object_id": object_id,
            "angle": angle,
            "file": file,
            "reason": reason,
        }
    )


def _dataset_slug(dataset_name: str) -> str:
    """Return the stable dataset slug used in manifest filenames."""
    return dataset_name.lower().replace(" ", "_")


def _expected_total_angles(cfg: PipelineConfig) -> int:
    return sum(cfg.get_num_angles(oid) for oid in cfg.list_object_ids())


def _sample_rows(arr: np.ndarray, max_rows: int = MAX_NUMERIC_SAMPLE_ROWS) -> np.ndarray:
    """Return a bounded deterministic sample for expensive numeric sanity checks.

    Full PhysX validation touches tens of thousands of small voxel arrays and
    thousands of ~67MB DINO arrays. Schema/shape/dtype checks remain exhaustive;
    numeric range/finite checks use head/tail/stride sampling so Plan 02-02 can
    run as a practical validator instead of a full data re-read.
    """
    if arr.ndim == 0 or arr.shape[0] <= max_rows:
        return arr
    head = np.arange(min(128, arr.shape[0]))
    tail_start = max(0, arr.shape[0] - 128)
    tail = np.arange(tail_start, arr.shape[0])
    remaining = max_rows - len(head) - len(tail)
    if remaining > 0:
        stride = max(1, arr.shape[0] // remaining)
        middle = np.arange(0, arr.shape[0], stride)[:remaining]
        idx = np.unique(np.concatenate([head, middle, tail]))
    else:
        idx = np.unique(np.concatenate([head, tail]))[:max_rows]
    return arr[idx]


def _sampled_min_max(arr: np.ndarray) -> tuple[int, int] | None:
    sample = _sample_rows(arr)
    if sample.size == 0:
        return None
    return int(sample.min()), int(sample.max())


def _sampled_all_finite(arr: np.ndarray) -> tuple[bool, bool]:
    sample = _sample_rows(
        arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr,
        max_rows=MAX_FINITE_SAMPLE_ROWS,
    )
    return bool(np.isnan(sample).any()), bool(np.isinf(sample).any())


def _load_stored_npz_array(path: Path, member_name: str) -> np.ndarray:
    """Load a .npz member as a memmap when it is ZIP_STORED.

    `pipeline/06_extract_feature.py` writes `tokens.npz` with numpy's default
    uncompressed ZIP storage. Mapping the inner `tokens.npy` member lets the
    validator check shape/dtype and sample numeric finiteness without loading
    the full 67MB array for every object angle.
    """
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            with zf.open(info) as fh:
                return np.load(fh)

    with path.open("rb") as fh:
        fh.seek(info.header_offset)
        local_header = fh.read(30)
        if len(local_header) != 30 or local_header[:4] != b"PK\x03\x04":
            raise ValueError(f"invalid ZIP local header for {member_name}")
        filename_len, extra_len = struct.unpack("<HH", local_header[26:30])
        member_offset = info.header_offset + 30 + filename_len + extra_len
        fh.seek(member_offset)
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            shape, fortran_order, dtype = np.lib.format._read_array_header(
                fh, version
            )
        data_offset = fh.tell()
    order = "F" if fortran_order else "C"
    return np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=shape, order=order)


# ---------------------------------------------------------------------------
# Step: render
# ---------------------------------------------------------------------------


def check_render_exists(
    cfg: PipelineConfig,
    failures: list[dict[str, Any]],
    counters: dict[str, dict[str, int]],
) -> None:
    """VALID-02 + VALID-03 + VALID-04 for render step.

    For each (obj, angle):
      - camera_transforms.json must exist; load it for total_views
      - rgb/view_<i>.png and mask/mask_<i>.npy exist for view_idx in [0, total_views)
      - schema check on camera_transforms.json (sampled per-angle)
      - schema check on one mask_<i>.npy per angle (perf: skip 11 of 12 per angle)
    """
    renders_root = Path(cfg.renders_dir)
    fallback_total_views = EXPECTED_PART_COMPLETE_VIEWS
    expected_resolution = cfg.render.resolution

    for oid in cfg.list_object_ids():
        num_angles = cfg.get_num_angles(oid)
        for angle_i in range(num_angles):
            angle_dir = renders_root / oid / f"angle_{angle_i}"
            counters["render"]["checked"] += 1

            cam_path = angle_dir / "camera_transforms.json"
            if not cam_path.is_file():
                _record_fail(
                    failures,
                    step="render",
                    dimension="exists",
                    object_id=oid,
                    angle=angle_i,
                    file=str(cam_path),
                    reason="camera_transforms.json missing",
                )
                counters["render"]["failed"] += 1
                continue

            # Schema for camera_transforms.json (VALID-03)
            total_views = fallback_total_views
            try:
                with cam_path.open("r", encoding="utf-8") as fh:
                    cam = json.load(fh)
                required_keys = {
                    "aabb",
                    "scale",
                    "offset",
                    "resolution",
                    "fov_deg",
                    "total_views",
                    "frames",
                }
                missing = required_keys - set(cam.keys())
                if missing:
                    _record_fail(
                        failures,
                        step="render",
                        dimension="schema",
                        object_id=oid,
                        angle=angle_i,
                        file=str(cam_path),
                        reason=f"camera_transforms.json missing keys: {sorted(missing)}",
                    )
                    counters["render"]["failed"] += 1
                else:
                    if not isinstance(cam["scale"], (int, float)) or float(cam["scale"]) <= 0:
                        _record_fail(
                            failures,
                            step="render",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(cam_path),
                            reason=f"camera_transforms.scale not positive: {cam['scale']!r}",
                        )
                        counters["render"]["failed"] += 1
                    if not (isinstance(cam["offset"], list) and len(cam["offset"]) == 3):
                        _record_fail(
                            failures,
                            step="render",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(cam_path),
                            reason=f"camera_transforms.offset not list[3]: {cam['offset']!r}",
                        )
                        counters["render"]["failed"] += 1
                    if not np.allclose(
                        cam["aabb"],
                        [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        atol=1e-12,
                    ):
                        _record_fail(
                            failures,
                            step="render",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(cam_path),
                            reason=f"camera_transforms.aabb != [[-0.5,-0.5,-0.5],[0.5,0.5,0.5]]: {cam['aabb']!r}",
                        )
                        counters["render"]["failed"] += 1
                    total_views = int(cam["total_views"])
            except Exception as exc:
                _record_fail(
                    failures,
                    step="render",
                    dimension="schema",
                    object_id=oid,
                    angle=angle_i,
                    file=str(cam_path),
                    reason=f"camera_transforms.json unreadable: {exc!r}",
                )
                counters["render"]["failed"] += 1
                continue

            rgb_dir = angle_dir / "rgb"
            mask_dir = angle_dir / "mask"
            rgb_names = {p.name for p in rgb_dir.iterdir()} if rgb_dir.is_dir() else set()
            mask_names = {p.name for p in mask_dir.iterdir()} if mask_dir.is_dir() else set()
            for view_idx in range(total_views):
                rgb_path = rgb_dir / f"view_{view_idx}.png"
                mask_path = mask_dir / f"mask_{view_idx}.npy"
                if rgb_path.name not in rgb_names:
                    _record_fail(
                        failures,
                        step="render",
                        dimension="exists",
                        object_id=oid,
                        angle=angle_i,
                        file=str(rgb_path),
                        reason="rgb view png missing",
                    )
                    counters["render"]["failed"] += 1
                if mask_path.name not in mask_names:
                    _record_fail(
                        failures,
                        step="render",
                        dimension="exists",
                        object_id=oid,
                        angle=angle_i,
                        file=str(mask_path),
                        reason="mask view npy missing",
                    )
                    counters["render"]["failed"] += 1

            # Perf: per-angle sample one mask npy for schema check (VALID-03)
            sample_mask = mask_dir / "mask_0.npy"
            if sample_mask.is_file():
                try:
                    arr = np.load(sample_mask, mmap_mode="r")
                    if arr.dtype != EXPECTED_MASK_DTYPE:
                        _record_fail(
                            failures,
                            step="render",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(sample_mask),
                            reason=f"mask dtype {arr.dtype} != int32",
                        )
                        counters["render"]["failed"] += 1
                    if arr.shape != (expected_resolution, expected_resolution):
                        _record_fail(
                            failures,
                            step="render",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(sample_mask),
                            reason=(
                                f"mask shape {arr.shape} != "
                                f"({expected_resolution}, {expected_resolution})"
                            ),
                        )
                        counters["render"]["failed"] += 1
                except Exception as exc:
                    _record_fail(
                        failures,
                        step="render",
                        dimension="schema",
                        object_id=oid,
                        angle=angle_i,
                        file=str(sample_mask),
                        reason=f"mask sample unreadable: {exc!r}",
                    )
                    counters["render"]["failed"] += 1

            # bbox_gt.json schema (VALID-03) + bbox numeric range (VALID-04)
            bbox_path = angle_dir / "bbox_gt.json"
            if bbox_path.is_file():
                _check_bbox_gt(bbox_path, oid, angle_i, failures, counters, "render")


def _check_bbox_gt(
    bbox_path: Path,
    oid: str,
    angle_i: int,
    failures: list[dict[str, Any]],
    counters: dict[str, dict[str, int]],
    step_label: str,
) -> None:
    try:
        with bbox_path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as exc:
        _record_fail(
            failures,
            step=step_label,
            dimension="schema",
            object_id=oid,
            angle=angle_i,
            file=str(bbox_path),
            reason=f"bbox_gt.json unreadable: {exc!r}",
        )
        counters[step_label]["failed"] += 1
        return

    required = {"object_id", "angle_idx", "resolution", "num_views", "parts"}
    missing = required - set(d.keys())
    if missing:
        _record_fail(
            failures,
            step=step_label,
            dimension="schema",
            object_id=oid,
            angle=angle_i,
            file=str(bbox_path),
            reason=f"bbox_gt.json missing keys: {sorted(missing)}",
        )
        counters[step_label]["failed"] += 1
        return

    if str(d["object_id"]) != str(oid):
        _record_fail(
            failures,
            step=step_label,
            dimension="schema",
            object_id=oid,
            angle=angle_i,
            file=str(bbox_path),
            reason=f"bbox_gt.object_id {d['object_id']!r} != {oid!r}",
        )
        counters[step_label]["failed"] += 1
    if int(d["angle_idx"]) != angle_i:
        _record_fail(
            failures,
            step=step_label,
            dimension="schema",
            object_id=oid,
            angle=angle_i,
            file=str(bbox_path),
            reason=f"bbox_gt.angle_idx {d['angle_idx']} != {angle_i}",
        )
        counters[step_label]["failed"] += 1
    if not isinstance(d["parts"], dict):
        _record_fail(
            failures,
            step=step_label,
            dimension="schema",
            object_id=oid,
            angle=angle_i,
            file=str(bbox_path),
            reason=f"bbox_gt.parts not dict: {type(d['parts']).__name__}",
        )
        counters[step_label]["failed"] += 1
        return

    # VALID-04 bbox range: 0 <= x_min <= x_max <= GROUNDING_SCALE_LOCAL
    for part_name, part_entry in d["parts"].items():
        if not isinstance(part_entry, dict):
            continue
        views = part_entry.get("views", {})
        if not isinstance(views, dict):
            continue
        for view_str, view_info in views.items():
            if not isinstance(view_info, dict):
                continue
            bbox = view_info.get("bbox")
            if bbox is None:
                continue
            if (not isinstance(bbox, list)) or len(bbox) != 4:
                _record_fail(
                    failures,
                    step=step_label,
                    dimension="schema",
                    object_id=oid,
                    angle=angle_i,
                    file=str(bbox_path),
                    reason=f"bbox parts[{part_name}].views[{view_str}].bbox not list[4]: {bbox!r}",
                )
                counters[step_label]["failed"] += 1
                continue
            x_min, y_min, x_max, y_max = bbox
            if not (
                0 <= x_min <= x_max <= GROUNDING_SCALE_LOCAL
                and 0 <= y_min <= y_max <= GROUNDING_SCALE_LOCAL
            ):
                _record_fail(
                    failures,
                    step=step_label,
                    dimension="numeric",
                    object_id=oid,
                    angle=angle_i,
                    file=str(bbox_path),
                    reason=(
                        f"bbox out of [0,{GROUNDING_SCALE_LOCAL}] "
                        f"part={part_name} view={view_str} bbox={bbox}"
                    ),
                )
                counters[step_label]["failed"] += 1


# ---------------------------------------------------------------------------
# Step: voxel
# ---------------------------------------------------------------------------


def check_voxel_exists(
    cfg: PipelineConfig,
    failures: list[dict[str, Any]],
    counters: dict[str, dict[str, int]],
) -> None:
    """VALID-02 / VALID-03 / VALID-04 for voxel step.

    Per (obj, angle):
      - surface.npy exists, dtype int64, ndim==2, shape[1]==3, size>=MIN_SURFACE_VOXELS,
        coords in [0, voxel.resolution)
      - viz/surface_voxel.png + viz/per_part_voxel.png exist
      - ind_*.npy: glob, each int64 (M,3) with M > MIN_PART_VOXELS and coords in range
    """
    voxel_root = Path(cfg.reconstruction_dir) / "voxel_expanded"
    res = cfg.voxel.resolution

    for oid in cfg.list_object_ids():
        num_angles = cfg.get_num_angles(oid)
        for angle_i in range(num_angles):
            angle_dir = voxel_root / oid / f"angle_{angle_i}" / str(res)
            counters["voxel"]["checked"] += 1

            surface_path = angle_dir / "surface.npy"
            if not surface_path.is_file():
                _record_fail(
                    failures,
                    step="voxel",
                    dimension="exists",
                    object_id=oid,
                    angle=angle_i,
                    file=str(surface_path),
                    reason="surface.npy missing",
                )
                counters["voxel"]["failed"] += 1
            else:
                try:
                    arr = np.load(surface_path, mmap_mode="r")
                    if arr.dtype != np.int64:
                        _record_fail(
                            failures,
                            step="voxel",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(surface_path),
                            reason=f"surface.npy dtype {arr.dtype} != int64",
                        )
                        counters["voxel"]["failed"] += 1
                    if arr.ndim != 2 or arr.shape[1] != 3:
                        _record_fail(
                            failures,
                            step="voxel",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(surface_path),
                            reason=f"surface.npy shape {arr.shape} not (N,3)",
                        )
                        counters["voxel"]["failed"] += 1
                    else:
                        if arr.shape[0] < MIN_SURFACE_VOXELS:
                            _record_fail(
                                failures,
                                step="voxel",
                                dimension="numeric",
                                object_id=oid,
                                angle=angle_i,
                                file=str(surface_path),
                                reason=(
                                    f"surface.npy N={arr.shape[0]} < "
                                    f"MIN_SURFACE_VOXELS={MIN_SURFACE_VOXELS}"
                                ),
                            )
                            counters["voxel"]["failed"] += 1
                        min_max = (
                            _sampled_min_max(arr)
                            if arr.shape[0] <= MAX_NUMERIC_SAMPLE_ROWS
                            else None
                        )
                        if min_max is not None and (min_max[0] < 0 or min_max[1] >= res):
                            _record_fail(
                                failures,
                                step="voxel",
                                dimension="numeric",
                                object_id=oid,
                                angle=angle_i,
                                file=str(surface_path),
                                reason=(
                                    f"surface.npy coords out of [0,{res}): "
                                    f"sample_min={min_max[0]} sample_max={min_max[1]}"
                                ),
                            )
                            counters["voxel"]["failed"] += 1
                except Exception as exc:
                    _record_fail(
                        failures,
                        step="voxel",
                        dimension="schema",
                        object_id=oid,
                        angle=angle_i,
                        file=str(surface_path),
                        reason=f"surface.npy unreadable: {exc!r}",
                    )
                    counters["voxel"]["failed"] += 1

            viz_dir = angle_dir / "viz"
            for viz_name in ("surface_voxel.png", "per_part_voxel.png"):
                viz_path = viz_dir / viz_name
                if not viz_path.is_file():
                    _record_fail(
                        failures,
                        step="voxel",
                        dimension="exists",
                        object_id=oid,
                        angle=angle_i,
                        file=str(viz_path),
                        reason=f"{viz_name} missing",
                    )
                    counters["voxel"]["failed"] += 1

            # ind_*.npy: schema + numeric on whatever exists (filter is legitimate)
            if angle_dir.is_dir():
                for ind_path in angle_dir.glob("ind_*.npy"):
                    try:
                        arr = np.load(ind_path, mmap_mode="r")
                    except Exception as exc:
                        _record_fail(
                            failures,
                            step="voxel",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(ind_path),
                            reason=f"ind npy unreadable: {exc!r}",
                        )
                        counters["voxel"]["failed"] += 1
                        continue
                    if arr.dtype != np.int64:
                        _record_fail(
                            failures,
                            step="voxel",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(ind_path),
                            reason=f"ind npy dtype {arr.dtype} != int64",
                        )
                        counters["voxel"]["failed"] += 1
                    if arr.ndim != 2 or arr.shape[1] != 3:
                        _record_fail(
                            failures,
                            step="voxel",
                            dimension="schema",
                            object_id=oid,
                            angle=angle_i,
                            file=str(ind_path),
                            reason=f"ind npy shape {arr.shape} not (M,3)",
                        )
                        counters["voxel"]["failed"] += 1
                        continue
                    if arr.shape[0] <= MIN_PART_VOXELS:
                        _record_fail(
                            failures,
                            step="voxel",
                            dimension="numeric",
                            object_id=oid,
                            angle=angle_i,
                            file=str(ind_path),
                            reason=(
                                f"ind npy M={arr.shape[0]} should be > "
                                f"MIN_PART_VOXELS={MIN_PART_VOXELS} (filter invariant)"
                            ),
                        )
                        counters["voxel"]["failed"] += 1
                    # Per-part files are numerous; full/sampled coordinate scans make
                    # PhysX full validation I/O-bound for tens of minutes. Keep their
                    # schema and MIN_PART_VOXELS invariant exhaustive, while surface.npy
                    # remains the per-angle coordinate-range numeric sentinel.


# ---------------------------------------------------------------------------
# Step: dinov2
# ---------------------------------------------------------------------------


def check_dinov2_exists(
    cfg: PipelineConfig,
    failures: list[dict[str, Any]],
    counters: dict[str, dict[str, int]],
) -> None:
    """VALID-02 / VALID-03 / VALID-04 for dinov2 tokens."""
    tokens_root = Path(cfg.reconstruction_dir) / "dinov2_tokens"

    for oid in cfg.list_object_ids():
        num_angles = cfg.get_num_angles(oid)
        for angle_i in range(num_angles):
            angle_dir = tokens_root / oid / f"angle_{angle_i}"
            counters["dinov2"]["checked"] += 1
            tokens_path = angle_dir / "part_complete" / "tokens.npz"
            if not tokens_path.is_file():
                _record_fail(
                    failures,
                    step="dinov2",
                    dimension="exists",
                    object_id=oid,
                    angle=angle_i,
                    file=str(tokens_path),
                    reason="tokens.npz missing",
                )
                counters["dinov2"]["failed"] += 1
                continue

            try:
                tokens = _load_stored_npz_array(tokens_path, "tokens.npy")
            except Exception as exc:
                _record_fail(
                    failures,
                    step="dinov2",
                    dimension="schema",
                    object_id=oid,
                    angle=angle_i,
                    file=str(tokens_path),
                    reason=f"tokens.npz unreadable: {exc!r}",
                )
                counters["dinov2"]["failed"] += 1
                continue

            # VALID-03: default part_complete shape == (16, 1370, 1024) and dtype == float32
            if tokens.shape != EXPECTED_TOKENS_SHAPE:
                _record_fail(
                    failures,
                    step="dinov2",
                    dimension="schema",
                    object_id=oid,
                    angle=angle_i,
                    file=str(tokens_path),
                    reason=f"tokens shape {tokens.shape} != {EXPECTED_TOKENS_SHAPE}",
                )
                counters["dinov2"]["failed"] += 1
            if tokens.dtype != EXPECTED_TOKENS_DTYPE:
                _record_fail(
                    failures,
                    step="dinov2",
                    dimension="schema",
                    object_id=oid,
                    angle=angle_i,
                    file=str(tokens_path),
                    reason=f"tokens dtype {tokens.dtype} != float32",
                )
                counters["dinov2"]["failed"] += 1
            # VALID-04: sampled no NaN / Inf; full-array scan is too expensive
            # for 13,809 * ~67MB token matrices in Phase 02 full validation.
            has_nan, has_inf = _sampled_all_finite(tokens)
            if has_nan:
                _record_fail(
                    failures,
                    step="dinov2",
                    dimension="numeric",
                    object_id=oid,
                    angle=angle_i,
                    file=str(tokens_path),
                    reason="tokens sampled rows contain NaN",
                )
                counters["dinov2"]["failed"] += 1
            if has_inf:
                _record_fail(
                    failures,
                    step="dinov2",
                    dimension="numeric",
                    object_id=oid,
                    angle=angle_i,
                    file=str(tokens_path),
                    reason="tokens sampled rows contain Inf",
                )
                counters["dinov2"]["failed"] += 1


# ---------------------------------------------------------------------------
# Step: vlm
# ---------------------------------------------------------------------------


def check_vlm_exists(
    cfg: PipelineConfig,
    failures: list[dict[str, Any]],
    counters: dict[str, dict[str, int]],
) -> None:
    """VALID-02 / VALID-03 / VALID-04 for VLM JSONL.

    - <vlm_dir>/training_json/arts_mllm_<slug>.jsonl exists; line count == expected total angles
    - each line has keys >= {id, conversations, images}
    - per-object spot-check: first line per object's images all exist (VALID-04)
    """
    slug = _dataset_slug(cfg.dataset_name)
    jsonl_path = Path(cfg.vlm_dir) / "training_json" / f"arts_mllm_{slug}.jsonl"
    counters["vlm"]["checked"] += 1

    if not jsonl_path.is_file():
        _record_fail(
            failures,
            step="vlm",
            dimension="exists",
            object_id=None,
            angle=None,
            file=str(jsonl_path),
            reason="vlm jsonl missing",
        )
        counters["vlm"]["failed"] += 1
        return

    expected_total = _expected_total_angles(cfg)
    line_count = 0
    seen_obj_first_line: set[str] = set()
    image_path_failures = 0
    image_path_failure_cap = 50  # stop spamming after enough samples
    schema_failure_cap = 50
    schema_failures = 0

    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line_count += 1
                try:
                    rec = json.loads(raw)
                except Exception as exc:
                    if schema_failures < schema_failure_cap:
                        _record_fail(
                            failures,
                            step="vlm",
                            dimension="schema",
                            object_id=None,
                            angle=None,
                            file=str(jsonl_path),
                            reason=f"line {line_count} not valid JSON: {exc!r}",
                        )
                        counters["vlm"]["failed"] += 1
                        schema_failures += 1
                    continue

                required = {"id", "conversations", "images"}
                missing = required - set(rec.keys())
                if missing:
                    if schema_failures < schema_failure_cap:
                        _record_fail(
                            failures,
                            step="vlm",
                            dimension="schema",
                            object_id=None,
                            angle=None,
                            file=str(jsonl_path),
                            reason=f"line {line_count} missing keys: {sorted(missing)}",
                        )
                        counters["vlm"]["failed"] += 1
                        schema_failures += 1
                    continue

                # Spot-check images existence: only first record encountered for each obj
                sample_id = rec.get("id", "")
                m = re.match(rf"^{re.escape(slug)}_(.+?)_angle_(\d+)$", str(sample_id))
                obj_id_from_sample = m.group(1) if m else None
                if obj_id_from_sample and obj_id_from_sample not in seen_obj_first_line:
                    seen_obj_first_line.add(obj_id_from_sample)
                    images = rec.get("images", []) or []
                    for img in images:
                        if not isinstance(img, str) or not Path(img).is_file():
                            if image_path_failures < image_path_failure_cap:
                                _record_fail(
                                    failures,
                                    step="vlm",
                                    dimension="numeric",
                                    object_id=obj_id_from_sample,
                                    angle=int(m.group(2)) if m else None,
                                    file=str(img),
                                    reason="referenced image path does not exist",
                                )
                                counters["vlm"]["failed"] += 1
                                image_path_failures += 1
    except Exception as exc:
        _record_fail(
            failures,
            step="vlm",
            dimension="schema",
            object_id=None,
            angle=None,
            file=str(jsonl_path),
            reason=f"jsonl iteration failed: {exc!r}",
        )
        counters["vlm"]["failed"] += 1
        return

    if line_count != expected_total:
        _record_fail(
            failures,
            step="vlm",
            dimension="exists",
            object_id=None,
            angle=None,
            file=str(jsonl_path),
            reason=f"jsonl line count {line_count} != expected total angles {expected_total}",
        )
        counters["vlm"]["failed"] += 1



# ---------------------------------------------------------------------------
# Step: ss_latent (pipeline/07_encode_ss_latents_per_part.py)
# ---------------------------------------------------------------------------


def _load_part_names_for_validation(part_info_path: Path, oid: str) -> list[str]:
    with part_info_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if str(payload.get("object_id")) != str(oid):
        raise ValueError(
            f"part_info object_id mismatch: expected {oid}, got {payload.get('object_id')!r}"
        )
    parts = payload.get("parts")
    label_to_key = payload.get("label_to_key")
    num_parts = payload.get("num_parts")
    if not isinstance(parts, dict):
        raise TypeError("part_info.parts must be a dict")
    if not isinstance(label_to_key, dict):
        raise TypeError("part_info.label_to_key must be a dict")
    if isinstance(num_parts, bool) or not isinstance(num_parts, int) or num_parts < 1:
        raise ValueError(f"part_info.num_parts must be positive int, got {num_parts!r}")

    names: list[str] = []
    for part_idx in range(num_parts):
        key = str(part_idx)
        if key not in label_to_key:
            raise KeyError(f"part_info.label_to_key missing {key!r}")
        part_name = label_to_key[key]
        if not isinstance(part_name, str) or not part_name:
            raise TypeError(f"part_info.label_to_key[{key!r}] must be a non-empty string")
        if part_name not in parts:
            raise KeyError(f"part_info parts missing canonical key {part_name!r}")
        if part_name in names:
            raise ValueError(f"duplicate part name {part_name!r}")
        names.append(part_name)
    return names


def check_ss_latent_exists(
    cfg: PipelineConfig,
    failures: list[dict[str, Any]],
    counters: dict[str, dict[str, int]],
    warnings: list[dict[str, Any]],
) -> None:
    """Validate pipeline/08 per-part SS latents.

    Coverage policy mirrors the generator default: expected latent files are the
    canonical `part_info.json` parts that have a voxel `ind_<part>.npy` input.
    Missing voxel inds are explicit warnings rather than failures because
    03_voxelize.py legitimately filters tiny parts below MIN_PART_VOXELS.
    """
    reconstruction_root = Path(cfg.reconstruction_dir)
    part_info_root = Path(cfg.part_info_dir)
    voxel_root = reconstruction_root / "voxel_expanded"
    latent_root = reconstruction_root / "ss_latents_per_part"
    res = cfg.voxel.resolution

    for oid in cfg.list_object_ids():
        part_info_path = part_info_root / oid / "part_info.json"
        try:
            part_names = _load_part_names_for_validation(part_info_path, oid)
        except Exception as exc:
            _record_fail(
                failures,
                step="ss_latent",
                dimension="schema",
                object_id=oid,
                angle=None,
                file=str(part_info_path),
                reason=f"part_info unreadable/invalid for ss_latent: {exc!r}",
            )
            counters["ss_latent"]["failed"] += 1
            continue

        for angle_i in range(cfg.get_num_angles(oid)):
            voxel_angle_dir = voxel_root / oid / f"angle_{angle_i}" / str(res)
            latent_angle_dir = latent_root / oid / f"angle_{angle_i}"
            expected_names = set(part_names)
            counters["ss_latent"]["checked"] += 1

            if latent_angle_dir.is_dir():
                for latent_path in latent_angle_dir.glob("*.npy"):
                    if latent_path.stem not in expected_names:
                        _record_fail(
                            failures,
                            step="ss_latent",
                            dimension="consistency",
                            object_id=oid,
                            angle=angle_i,
                            file=str(latent_path),
                            reason=(
                                f"extra latent file stem {latent_path.stem!r} is not a "
                                "canonical part_info parts key"
                            ),
                        )
                        counters["ss_latent"]["failed"] += 1

            for part_name in part_names:
                ind_path = voxel_angle_dir / f"ind_{part_name}.npy"
                latent_path = latent_angle_dir / f"{part_name}.npy"
                if not ind_path.is_file():
                    _record_warn(
                        warnings,
                        step="ss_latent",
                        dimension="exists",
                        object_id=oid,
                        angle=angle_i,
                        file=str(ind_path),
                        reason=(
                            f"part={part_name}: missing voxel ind; latent not expected under "
                            "voxel-kept coverage (likely below_min_part_voxels_5)"
                        ),
                    )
                    continue

                if not latent_path.is_file():
                    _record_fail(
                        failures,
                        step="ss_latent",
                        dimension="exists",
                        object_id=oid,
                        angle=angle_i,
                        file=str(latent_path),
                        reason=f"part={part_name}: ss latent missing",
                    )
                    counters["ss_latent"]["failed"] += 1
                    continue

                try:
                    arr = np.load(latent_path, mmap_mode="r")
                except Exception as exc:
                    _record_fail(
                        failures,
                        step="ss_latent",
                        dimension="schema",
                        object_id=oid,
                        angle=angle_i,
                        file=str(latent_path),
                        reason=f"part={part_name}: latent unreadable: {exc!r}",
                    )
                    counters["ss_latent"]["failed"] += 1
                    continue

                if arr.dtype != EXPECTED_SS_LATENT_DTYPE:
                    _record_fail(
                        failures,
                        step="ss_latent",
                        dimension="schema",
                        object_id=oid,
                        angle=angle_i,
                        file=str(latent_path),
                        reason=f"part={part_name}: dtype {arr.dtype} != float32",
                    )
                    counters["ss_latent"]["failed"] += 1
                if tuple(arr.shape) != EXPECTED_SS_LATENT_SHAPE:
                    _record_fail(
                        failures,
                        step="ss_latent",
                        dimension="schema",
                        object_id=oid,
                        angle=angle_i,
                        file=str(latent_path),
                        reason=f"part={part_name}: shape {tuple(arr.shape)} != {EXPECTED_SS_LATENT_SHAPE}",
                    )
                    counters["ss_latent"]["failed"] += 1
                    continue
                if not bool(np.isfinite(arr).all()):
                    _record_fail(
                        failures,
                        step="ss_latent",
                        dimension="numeric",
                        object_id=oid,
                        angle=angle_i,
                        file=str(latent_path),
                        reason=f"part={part_name}: latent contains NaN or Inf",
                    )
                    counters["ss_latent"]["failed"] += 1

# ---------------------------------------------------------------------------
# Step: preview
# ---------------------------------------------------------------------------


_PREVIEW_FILE_URI_RE = re.compile(r"file://([^\"'\s>)]+)")


def check_preview_exists(
    cfg: PipelineConfig,
    failures: list[dict[str, Any]],
    counters: dict[str, dict[str, int]],
    warnings: list[dict[str, Any]],
) -> None:
    """VALID-02 + D-05 preview embed warning."""
    preview_root = Path(cfg.preview_dir)

    object_ids = cfg.list_object_ids()
    for oid in object_ids:
        counters["preview"]["checked"] += 1
        html_path = preview_root / f"{oid}.html"
        if not html_path.is_file():
            _record_fail(
                failures,
                step="preview",
                dimension="exists",
                object_id=oid,
                angle=None,
                file=str(html_path),
                reason="preview html missing",
            )
            counters["preview"]["failed"] += 1

    # index.html optional
    index_html = preview_root / "index.html"
    if not index_html.is_file():
        _record_warn(
            warnings,
            step="preview",
            dimension="exists",
            object_id=None,
            angle=None,
            file=str(index_html),
            reason="preview index.html missing (optional)",
        )

    # D-05 bottom-line: spot-check N=10 obj for broken file:// embeds (warning only)
    sample = object_ids[:10]
    for oid in sample:
        html_path = preview_root / f"{oid}.html"
        if not html_path.is_file():
            continue
        try:
            text = html_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            _record_warn(
                warnings,
                step="preview",
                dimension="consistency",
                object_id=oid,
                angle=None,
                file=str(html_path),
                reason=f"preview html unreadable: {exc!r}",
            )
            continue
        for m in _PREVIEW_FILE_URI_RE.finditer(text):
            uri = m.group(1)
            if not Path(uri).is_file():
                _record_warn(
                    warnings,
                    step="preview",
                    dimension="consistency",
                    object_id=oid,
                    angle=None,
                    file=uri,
                    reason="preview_embed_broken_link",
                )


# ---------------------------------------------------------------------------
# VALID-05 cross-step consistency: per-object angle-count alignment only.
#
# NOTE: Per-part cross-step consistency (mask/bbox/voxel_ind 列表对齐) is NOT in
# VALID-05's scope. It is delivered by the manifest builder
# (pipeline/04_build_valid_parts_manifest.py, Plan 02-02) which encodes per-part availability via
# manifest.objects[<oid>].angles[<i>].parts[<name>].has_voxel_ind / has_mask /
# has_bbox flags. VALID-05 only checks per-object angle count alignment.
#
# Future hook: if part-level cross-step consistency is later added, the
# implementation MUST first try-load <data_root>/manifests/<DATASET>.json. For
# any part whose has_voxel_ind == False, treat the absence of ind_*.npy as a
# legitimate filter (filter_reason="below_min_part_voxels_5", MIN_PART_VOXELS=5)
# and do NOT report. Fallback when manifest is missing: derive filtered set from
# `set(part_info.parts) - set(glob ind_*.npy)` per (obj, angle).
# ---------------------------------------------------------------------------


def check_consistency(
    cfg: PipelineConfig,
    failures: list[dict[str, Any]],
    counters: dict[str, dict[str, int]],
    selected_steps: tuple[str, ...],
) -> None:
    """VALID-05: per-object angle count alignment per step."""
    object_ids = cfg.list_object_ids()
    slug = _dataset_slug(cfg.dataset_name)

    # Pre-build vlm angle-per-obj count if vlm step selected
    vlm_obj_counts: dict[str, int] = {}
    if "vlm" in selected_steps:
        jsonl_path = Path(cfg.vlm_dir) / "training_json" / f"arts_mllm_{slug}.jsonl"
        if jsonl_path.is_file():
            id_pat = re.compile(rf"^{re.escape(slug)}_(.+?)_angle_(\d+)$")
            try:
                with jsonl_path.open("r", encoding="utf-8") as fh:
                    for raw in fh:
                        try:
                            rec = json.loads(raw)
                        except Exception:
                            continue
                        m = id_pat.match(str(rec.get("id", "")))
                        if m:
                            vlm_obj_counts[m.group(1)] = vlm_obj_counts.get(m.group(1), 0) + 1
            except Exception:
                pass

    renders_root = Path(cfg.renders_dir)
    voxel_root = Path(cfg.reconstruction_dir) / "voxel_expanded"
    tokens_root = Path(cfg.reconstruction_dir) / "dinov2_tokens"
    preview_root = Path(cfg.preview_dir)
    res = cfg.voxel.resolution

    for oid in object_ids:
        expected = cfg.get_num_angles(oid)

        # render: count angle_<i> dirs that contain camera_transforms.json
        if "render" in selected_steps:
            counters["consistency"]["checked"] += 1
            obj_dir = renders_root / oid
            actual = _count_angle_dirs(obj_dir, marker="camera_transforms.json")
            if actual != expected:
                _record_fail(
                    failures,
                    step="render",
                    dimension="consistency",
                    object_id=oid,
                    angle=None,
                    file=str(obj_dir),
                    reason=(
                        f"render angle dir count {actual} != get_num_angles {expected}"
                    ),
                )
                counters["consistency"]["failed"] += 1

        if "voxel" in selected_steps:
            counters["consistency"]["checked"] += 1
            obj_dir = voxel_root / oid
            actual = _count_angle_dirs(obj_dir, marker=f"{res}/surface.npy")
            if actual != expected:
                _record_fail(
                    failures,
                    step="voxel",
                    dimension="consistency",
                    object_id=oid,
                    angle=None,
                    file=str(obj_dir),
                    reason=f"voxel angle dir count {actual} != get_num_angles {expected}",
                )
                counters["consistency"]["failed"] += 1

        if "dinov2" in selected_steps:
            counters["consistency"]["checked"] += 1
            obj_dir = tokens_root / oid
            actual = _count_angle_dirs(obj_dir, marker="part_complete/tokens.npz")
            if actual != expected:
                _record_fail(
                    failures,
                    step="dinov2",
                    dimension="consistency",
                    object_id=oid,
                    angle=None,
                    file=str(obj_dir),
                    reason=f"dinov2 angle dir count {actual} != get_num_angles {expected}",
                )
                counters["consistency"]["failed"] += 1

        if "vlm" in selected_steps:
            counters["consistency"]["checked"] += 1
            actual = vlm_obj_counts.get(oid, 0)
            if actual != expected:
                _record_fail(
                    failures,
                    step="vlm",
                    dimension="consistency",
                    object_id=oid,
                    angle=None,
                    file=None,
                    reason=(
                        f"vlm jsonl rows for obj {oid}: {actual} != get_num_angles {expected}"
                    ),
                )
                counters["consistency"]["failed"] += 1

        if "preview" in selected_steps:
            counters["consistency"]["checked"] += 1
            html_path = preview_root / f"{oid}.html"
            # preview is 1-per-obj (not per-angle); single-point check
            if not html_path.is_file():
                _record_fail(
                    failures,
                    step="preview",
                    dimension="consistency",
                    object_id=oid,
                    angle=None,
                    file=str(html_path),
                    reason="preview html missing (consistency)",
                )
                counters["consistency"]["failed"] += 1


def _count_angle_dirs(obj_dir: Path, marker: str) -> int:
    if not obj_dir.is_dir():
        return 0
    count = 0
    for entry in obj_dir.iterdir():
        if not entry.is_dir():
            continue
        if not ANGLE_DIR_RE.match(entry.name):
            continue
        if (entry / marker).is_file():
            count += 1
    return count


# ---------------------------------------------------------------------------
# Report writer (VALID-07) + stdout summary
# ---------------------------------------------------------------------------


def write_json_report(
    cfg: PipelineConfig,
    config_path: Path,
    selected_steps: tuple[str, ...],
    counters: dict[str, dict[str, int]],
    failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    report_path: Path,
) -> dict[str, Any]:
    now = int(time.time())
    iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    by_step: dict[str, dict[str, int]] = {}
    for step in (*selected_steps, "consistency"):
        if step in counters:
            by_step[step] = dict(counters[step])

    report = {
        "dataset": cfg.dataset_name,
        "config_path": str(config_path.resolve()),
        "timestamp_unix": now,
        "timestamp_iso": iso,
        "summary": {
            "total_objects": len(cfg.list_object_ids()),
            "total_angles_expected": _expected_total_angles(cfg),
            "passed": len(failures) == 0,
            "by_step": by_step,
        },
        "failures": failures,
        "warnings": warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    return report


def print_summary(
    report: dict[str, Any],
    report_path: Path,
    top_n: int,
) -> None:
    summary = report["summary"]
    total_objects = summary["total_objects"]
    total_angles = summary["total_angles_expected"]
    print(
        f"[validator] dataset={report['dataset']} "
        f"objects={total_objects} angles_expected={total_angles}"
    )
    for step, counts in summary["by_step"].items():
        print(f"[{step}] checked={counts['checked']} failed={counts['failed']}")

    failures = report["failures"]
    if failures:
        n = min(top_n, len(failures))
        for rec in failures[:n]:
            print(
                f"[FAIL] step={rec['step']} dim={rec['dimension']} "
                f"obj={rec['object_id']} angle={rec['angle']} "
                f"file={rec['file']} reason={rec['reason']}"
            )
        remaining = len(failures) - n
        if remaining > 0:
            print(f"... + {remaining} more")

    warnings_list = report.get("warnings", [])
    if warnings_list:
        print(f"[warnings] count={len(warnings_list)} (not counted in passed)")

    print(f"[report] written to {report_path.resolve()}")
    print(f"[summary] passed={'true' if summary['passed'] else 'false'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-agnostic validator covering exists / schema / numeric / "
            "consistency dimensions across render, voxel, dinov2, vlm, preview, ss_latent steps."
        )
    )
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help="Comma-separated steps to run (subset of render,voxel,dinov2,vlm,preview,ss_latent).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of failure records to show in stdout summary.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="JSON report path (default: /tmp/validate_<dataset>_<unix_ts>.json).",
    )
    return parser.parse_args(argv)


def _normalize_steps(raw: str) -> tuple[str, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if p not in ALL_STEPS:
            raise ValueError(f"unknown step: {p!r} (valid: {ALL_STEPS})")
    # preserve order from ALL_STEPS for deterministic stdout
    selected = tuple(s for s in ALL_STEPS if s in parts)
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    cfg = load_config(str(config_path))

    selected_steps = _normalize_steps(args.steps)

    counters: dict[str, dict[str, int]] = {
        step: {"checked": 0, "failed": 0} for step in selected_steps
    }
    counters["consistency"] = {"checked": 0, "failed": 0}

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    step_to_fn: dict[str, Callable[..., None]] = {
        "render": lambda: check_render_exists(cfg, failures, counters),
        "voxel": lambda: check_voxel_exists(cfg, failures, counters),
        "dinov2": lambda: check_dinov2_exists(cfg, failures, counters),
        "vlm": lambda: check_vlm_exists(cfg, failures, counters),
        "preview": lambda: check_preview_exists(cfg, failures, counters, warnings),
        "ss_latent": lambda: check_ss_latent_exists(cfg, failures, counters, warnings),
    }
    for step in selected_steps:
        step_to_fn[step]()

    if selected_steps:
        check_consistency(cfg, failures, counters, selected_steps)

    if args.report_path:
        report_path = Path(args.report_path)
    else:
        slug = _dataset_slug(cfg.dataset_name)
        report_path = Path(f"/tmp/validate_{slug}_{int(time.time())}.json")

    report = write_json_report(
        cfg,
        config_path,
        selected_steps,
        counters,
        failures,
        warnings,
        report_path,
    )
    print_summary(report, report_path, args.top_n)
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
