#!/usr/bin/env python3
"""Encode OmniPart/TRELLIS part-synthesis SLat caches from pipeline outputs.

This stage is distinct from Step 8 sparse-structure (SS) latents.  It builds
structured-latent (SLat) sparse tensors used by OmniPart part-synthesis
training:

    <output_root>/<instance_id[:2]>/<instance_id>/overall/latent.npz
    <output_root>/<instance_id[:2]>/<instance_id>/<part_name>/latent.npz
    <output_root>/<instance_id[:2]>/<instance_id>/overall/part_id.txt
    <output_root>/<instance_id[:2]>/<instance_id>/all_latent.npz

where ``instance_id`` is ``<object_id>_angle_<angle_idx>``.  Each per-component
``latent.npz`` stores TRELLIS SLat sparse ``coords`` and ``feats``.  The merged
``all_latent.npz`` mirrors OmniPart's ``dataset_toolkits/merge_slat.py`` output
with ``coords``, ``feats`` and ``offsets``.

The Part Completion manifest is the source of truth for sample and target-part
selection.  This stage must not enumerate ``part_info.parts`` as a fallback: it
only encodes the exact ``object_id``/``angle_idx`` rows and ``target_part_names``
already accepted by the VLM/Part Completion pipeline.

Inputs come only from dataset_toolkits outputs: voxel indices, rendered camera
transforms, cached DINOv2 tokens, and the strict Part Completion manifest.  The
script loads the TRELLIS SLat encoder but does not modify OmniPart model code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import PipelineConfig, load_config, resolve_repo_path  # noqa: E402


EXPECTED_RESOLUTION = 64
EXPECTED_TOKEN_SHAPE = (16, 1370, 1024)
EXPECTED_PATCH_GRID = 37
EXPECTED_PATCH_COUNT = EXPECTED_PATCH_GRID * EXPECTED_PATCH_GRID
EXPECTED_FEATURE_DIM = 1024
EXPECTED_LATENT_CHANNELS = 8


@dataclass(frozen=True)
class Target:
    name: str
    kind: str  # "overall" or "part"
    voxel_path: Path
    latent_path: Path


@dataclass(frozen=True)
class Instance:
    object_id: str
    angle_idx: int
    instance_id: str
    root: Path
    camera_path: Path
    token_path: Path
    part_info_path: Path
    targets: tuple[Target, ...]
    source_sample_id: str
    view_indices: tuple[int, ...]


@dataclass
class Counters:
    objects_seen: int = 0
    angles_seen: int = 0
    instances_seen: int = 0
    targets_seen: int = 0
    queued: int = 0
    generated: int = 0
    existing_valid: int = 0
    skipped_missing_voxel: int = 0
    merged: int = 0
    failed: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode TRELLIS/OmniPart part-synthesis SLat sparse caches."
    )
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset.")
    parser.add_argument("--object-list", help="Optional newline-delimited object ID file.")
    parser.add_argument(
        "--output-root",
        help="Default: <data_root>/part_synthesis_slat. This is the OmniPart SLat data_root.",
    )
    parser.add_argument(
        "--completion-manifest",
        help=(
            "Part Completion train JSONL source of truth. Default: "
            "<data_root>/manifests/part_completion/arts_pc_<dataset_slug>_train.jsonl"
        ),
    )
    parser.add_argument(
        "--enc-pretrained",
        help="Absolute or repo-relative TRELLIS SLat encoder prefix/path. Default: trellis.slat_encoder from config.",
    )
    parser.add_argument(
        "--trellis-root",
        help="Absolute or repo-relative path containing the trellis Python package. Default: trellis.root from config.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, default cuda.")
    parser.add_argument("--rank", type=int, default=0, help="Shard rank for manual splitting.")
    parser.add_argument("--world-size", type=int, default=1, help="Shard count for manual splitting.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing valid latents.")
    parser.add_argument("--dry-run", action="store_true", help="Enumerate work without importing TRELLIS or writing outputs.")
    parser.add_argument("--report-path", help="JSON report path (default: /tmp/part_synthesis_slat_<dataset>_<ts>.json).")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after per-target failures.")
    parser.add_argument("--progress-every", type=int, default=500, help="Print preflight progress every N manifest rows; 0 disables.")
    return parser.parse_args(argv)


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace(" ", "_")


def _utc_now() -> tuple[int, str]:
    now = int(time.time())
    return now, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_object_ids(raw_value: str) -> list[str]:
    object_ids = [item.strip() for item in raw_value.split(",")]
    if not object_ids or any(not item for item in object_ids):
        raise ValueError("--object-ids must be comma-separated non-empty IDs")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("--object-ids contains duplicate IDs")
    return object_ids


def _requested_object_ids(args: argparse.Namespace) -> set[str] | None:
    if args.object_ids and args.object_list:
        raise ValueError("--object-ids and --object-list are mutually exclusive")
    if args.object_ids:
        return set(_parse_object_ids(args.object_ids))
    elif args.object_list:
        object_list = Path(args.object_list)
        requested = [line.strip() for line in object_list.read_text().splitlines() if line.strip()]
        if len(requested) != len(set(requested)):
            raise ValueError(f"--object-list contains duplicate IDs: {object_list}")
        if any(not item for item in requested):
            raise ValueError(f"--object-list contains empty object IDs: {object_list}")
        return set(requested)
    else:
        return None


def _load_json_mapping(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object: {path}")
    return payload


def load_part_info_parts(part_info_path: Path, object_id: str) -> dict[str, Any]:
    payload = _load_json_mapping(part_info_path, "part_info")
    if str(payload.get("object_id")) != str(object_id):
        raise ValueError(
            f"part_info object_id mismatch for {part_info_path}: expected {object_id}, got {payload.get('object_id')!r}"
        )
    parts = payload.get("parts")
    if not isinstance(parts, dict):
        raise TypeError(f"part_info.parts must be a dict: {part_info_path}")
    return parts


def validate_manifest_target_parts(row: dict[str, Any], source: str, line_no: int) -> tuple[list[str], tuple[int, ...]]:
    if row.get("manifest_version") != 1:
        raise ValueError(f"{source}:{line_no}: manifest_version must be 1")
    if row.get("task") != "part_completion":
        raise ValueError(f"{source}:{line_no}: task must be 'part_completion'")
    target_part_count = row.get("target_part_count")
    if isinstance(target_part_count, bool) or not isinstance(target_part_count, int) or target_part_count < 1:
        raise ValueError(f"{source}:{line_no}: target_part_count must be positive int")
    target_part_names = row.get("target_part_names")
    if not isinstance(target_part_names, list) or len(target_part_names) != target_part_count:
        raise ValueError(f"{source}:{line_no}: target_part_names length must equal target_part_count")
    if any(not isinstance(name, str) or not name for name in target_part_names):
        raise ValueError(f"{source}:{line_no}: target_part_names must be non-empty strings")
    if len(set(target_part_names)) != len(target_part_names):
        raise ValueError(f"{source}:{line_no}: target_part_names contains duplicates")
    target_parts = row.get("target_parts")
    if not isinstance(target_parts, list) or len(target_parts) != target_part_count:
        raise ValueError(f"{source}:{line_no}: target_parts length must equal target_part_count")
    for idx, (expected_name, part) in enumerate(zip(target_part_names, target_parts), start=1):
        if not isinstance(part, dict):
            raise ValueError(f"{source}:{line_no}: target_parts[{idx}] must be object")
        if part.get("name") != expected_name:
            raise ValueError(f"{source}:{line_no}: target_parts[{idx}].name does not match target_part_names")
        local_label = part.get("local_label")
        if local_label != idx:
            raise ValueError(f"{source}:{line_no}: target_parts[{idx}].local_label must be {idx}")
        motion = part.get("motion")
        if not isinstance(motion, dict) or motion.get("motion_type") not in {"rotate", "prismatic"}:
            raise ValueError(f"{source}:{line_no}: target_parts[{idx}].motion.motion_type must be rotate/prismatic")
    view_indices = row.get("view_indices")
    if not isinstance(view_indices, list) or len(view_indices) != 4:
        raise ValueError(f"{source}:{line_no}: view_indices must contain 4 entries")
    if any(isinstance(view, bool) or not isinstance(view, int) or view < 0 or view > 11 for view in view_indices):
        raise ValueError(f"{source}:{line_no}: view_indices must be ints in [0,11]")
    if len(set(view_indices)) != 4:
        raise ValueError(f"{source}:{line_no}: view_indices contains duplicates")
    quadrants = row.get("quadrants")
    if quadrants != [0, 1, 2, 3] or [view // 3 for view in view_indices] != [0, 1, 2, 3]:
        raise ValueError(f"{source}:{line_no}: quadrants/view_indices must match Part Completion contract")
    return list(target_part_names), tuple(int(view) for view in view_indices)


def load_completion_samples(
    cfg: PipelineConfig,
    manifest_path: Path,
    requested_object_ids: set[str] | None,
) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Part Completion manifest not found: {manifest_path}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    manifest_objects: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}:{line_no}: row must be JSON object")
            object_id = row.get("object_id")
            angle_idx = row.get("angle_idx")
            sample_id = row.get("sample_id")
            if not isinstance(object_id, str) or not object_id:
                raise ValueError(f"{manifest_path}:{line_no}: object_id must be non-empty string")
            manifest_objects.add(object_id)
            if requested_object_ids is not None and object_id not in requested_object_ids:
                continue
            if isinstance(angle_idx, bool) or not isinstance(angle_idx, int) or angle_idx < 0:
                raise ValueError(f"{manifest_path}:{line_no}: angle_idx must be non-negative int")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{manifest_path}:{line_no}: sample_id must be non-empty string")
            target_part_names, view_indices = validate_manifest_target_parts(row, str(manifest_path), line_no)
            key = (object_id, angle_idx)
            if key in seen:
                raise ValueError(f"{manifest_path}:{line_no}: duplicate object_id/angle_idx sample {key}")
            seen.add(key)
            rows.append(
                {
                    "object_id": object_id,
                    "angle_idx": angle_idx,
                    "sample_id": sample_id,
                    "target_part_names": target_part_names,
                    "view_indices": view_indices,
                }
            )
    if requested_object_ids is not None:
        missing = sorted(requested_object_ids - manifest_objects)
        if missing:
            raise ValueError("Requested object IDs are absent from Part Completion manifest: " + ", ".join(missing))
    if not rows:
        raise ValueError(f"No Part Completion manifest rows selected from {manifest_path}")
    return rows


def validate_target_against_part_info(part_info_parts: dict[str, Any], object_id: str, part_name: str) -> None:
    entry = part_info_parts.get(part_name)
    if not isinstance(entry, dict):
        raise ValueError(f"target part {part_name!r} not found in part_info[{object_id}].parts")
    label = entry.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
        raise ValueError(f"part_info[{object_id}].parts[{part_name!r}].label must be positive int")
    joint = entry.get("joint")
    joint_type = entry.get("joint_type")
    if not isinstance(joint, str) or not joint:
        raise ValueError(f"part_info[{object_id}].parts[{part_name!r}].joint must be non-empty string")
    if not isinstance(joint_type, str) or not joint_type:
        raise ValueError(f"part_info[{object_id}].parts[{part_name!r}].joint_type must be non-empty string")
    if joint == "fixed" or joint_type == "E":
        raise ValueError(f"Part Completion target {part_name!r} is not movable in part_info[{object_id}]")


def validate_voxel_file(path: Path, resolution: int) -> tuple[np.ndarray | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable: {exc!r}"
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None, f"shape {arr.shape} not (N,3)"
    if arr.shape[0] < 1:
        return None, "empty voxel array"
    if int(arr.min()) < 0 or int(arr.max()) >= resolution:
        return None, f"coords out of [0,{resolution})"
    return arr.astype(np.int64, copy=False), None


def validate_slat_file(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        with np.load(path) as data:
            if set(data.files) != {"coords", "feats"}:
                return f"keys {sorted(data.files)} != ['coords', 'feats']"
            coords = data["coords"]
            feats = data["feats"]
    except Exception as exc:  # noqa: BLE001
        return f"unreadable: {exc!r}"
    if coords.ndim != 2 or coords.shape[1] != 3:
        return f"coords shape {coords.shape} not (N,3)"
    if feats.ndim != 2 or feats.shape[1] != EXPECTED_LATENT_CHANNELS:
        return f"feats shape {feats.shape} not (N,{EXPECTED_LATENT_CHANNELS})"
    if coords.shape[0] != feats.shape[0]:
        return f"coords rows {coords.shape[0]} != feats rows {feats.shape[0]}"
    if coords.shape[0] < 1:
        return "empty latent"
    if not np.issubdtype(coords.dtype, np.integer):
        return f"coords dtype {coords.dtype} is not integer"
    if int(coords.min()) < 0 or int(coords.max()) >= EXPECTED_RESOLUTION:
        return f"coords out of [0,{EXPECTED_RESOLUTION})"
    if feats.dtype != np.float32:
        return f"feats dtype {feats.dtype} != float32"
    if not bool(np.isfinite(feats).all()):
        return "feats contain NaN or Inf"
    return None


def validate_all_latent_file(path: Path, expected_part_count: int | None = None) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        with np.load(path) as data:
            if set(data.files) != {"coords", "feats", "offsets"}:
                return f"keys {sorted(data.files)} != ['coords', 'feats', 'offsets']"
            coords = data["coords"]
            feats = data["feats"]
            offsets = data["offsets"]
    except Exception as exc:  # noqa: BLE001
        return f"unreadable: {exc!r}"
    if coords.ndim != 2 or coords.shape[1] != 3:
        return f"coords shape {coords.shape} not (N,3)"
    if feats.ndim != 2 or feats.shape[1] != EXPECTED_LATENT_CHANNELS:
        return f"feats shape {feats.shape} not (N,{EXPECTED_LATENT_CHANNELS})"
    if coords.shape[0] != feats.shape[0]:
        return "coords/feats row mismatch"
    if offsets.ndim != 1 or offsets.shape[0] < 2:
        return f"offsets shape {offsets.shape} invalid"
    if expected_part_count is not None:
        expected_offsets = expected_part_count + 2  # 0 + overall segment + each target part
        if offsets.shape[0] != expected_offsets:
            return f"offsets length {offsets.shape[0]} != expected {expected_offsets}"
    if int(offsets[0]) != 0 or int(offsets[-1]) != coords.shape[0]:
        return "offset endpoints invalid"
    if np.any(offsets[1:] < offsets[:-1]):
        return "offsets are not monotonic"
    if feats.dtype != np.float32:
        return f"feats dtype {feats.dtype} != float32"
    if not bool(np.isfinite(feats).all()):
        return "feats contain NaN or Inf"
    return None


def read_part_ids(path: Path) -> tuple[list[str] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    part_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not part_ids:
        return None, "empty"
    if len(part_ids) != len(set(part_ids)):
        return None, "duplicate part ids"
    return part_ids, None


def latent_row_count(path: Path) -> tuple[int | None, str | None]:
    reason = validate_slat_file(path)
    if reason is not None:
        return None, reason
    with np.load(path) as data:
        return int(data["coords"].shape[0]), None


def validate_all_latent_segments(path: Path, component_paths: list[Path]) -> str | None:
    all_reason = validate_all_latent_file(path, len(component_paths) - 1)
    if all_reason is not None:
        return all_reason
    with np.load(path) as data:
        offsets = data["offsets"].astype(np.int64)
    segment_lengths = np.diff(offsets)
    for idx, component_path in enumerate(component_paths):
        rows, reason = latent_row_count(component_path)
        if reason is not None or rows is None:
            return f"component[{idx}] invalid:{reason}"
        if int(segment_lengths[idx]) != rows:
            return f"segment[{idx}] length {int(segment_lengths[idx])} != component rows {rows}: {component_path}"
    return None


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def iter_instances(
    cfg: PipelineConfig,
    completion_rows: list[dict[str, Any]],
    output_root: Path,
    counters: Counters,
    records: list[dict[str, Any]],
) -> list[Instance]:
    reconstruction = Path(cfg.reconstruction_dir)
    renders = Path(cfg.renders_dir)
    resolution = cfg.voxel.resolution
    instances: list[Instance] = []
    seen_objects: set[str] = set()
    part_info_cache: dict[str, dict[str, Any]] = {}
    total_rows = len(completion_rows)
    for row_index, sample in enumerate(completion_rows, start=1):
        progress_every = getattr(cfg, "_part_synthesis_progress_every", 0)
        if progress_every and (row_index == 1 or row_index % progress_every == 0):
            print(
                f"[part-synthesis-slat][preflight] row={row_index}/{total_rows} "
                f"instances={len(instances)} failed={counters.failed}",
                flush=True,
            )
        oid = str(sample["object_id"])
        angle_idx = int(sample["angle_idx"])
        target_part_names = list(sample["target_part_names"])
        sample_id = str(sample["sample_id"])
        view_indices = tuple(sample["view_indices"])
        if oid not in seen_objects:
            counters.objects_seen += 1
            seen_objects.add(oid)
        part_info_path = Path(cfg.part_info_dir) / oid / "part_info.json"
        try:
            part_info_parts = part_info_cache.get(oid)
            if part_info_parts is None:
                part_info_parts = load_part_info_parts(part_info_path, oid)
                part_info_cache[oid] = part_info_parts
            for part_name in target_part_names:
                validate_target_against_part_info(part_info_parts, oid, part_name)
        except Exception as exc:  # noqa: BLE001
            counters.failed += 1
            records.append({"status": "failed", "object_id": oid, "angle_idx": angle_idx, "sample_id": sample_id, "reason": repr(exc)})
            continue
        counters.angles_seen += 1
        counters.instances_seen += 1
        instance_id = f"{oid}_angle_{angle_idx}"
        instance_root = output_root / instance_id[:2] / instance_id
        voxel_dir = reconstruction / "voxel_expanded" / oid / f"angle_{angle_idx}" / str(resolution)
        camera_path = renders / oid / f"angle_{angle_idx}" / "part_complete" / "camera_transforms.json"
        token_path = reconstruction / "dinov2_tokens" / oid / f"angle_{angle_idx}" / "part_complete" / "tokens.npz"
        surface_path = voxel_dir / "surface.npy"
        missing_required = [
            name for name, path in (
                ("surface", surface_path),
                ("camera_transforms", camera_path),
                ("dinov2_tokens", token_path),
            ) if not path.is_file()
        ]
        if missing_required:
            counters.failed += 1
            records.append(
                {
                    "status": "failed",
                    "object_id": oid,
                    "angle_idx": angle_idx,
                    "sample_id": sample_id,
                    "instance_id": instance_id,
                    "reason": "missing required input(s): " + ",".join(missing_required),
                }
            )
            continue
        targets: list[Target] = [
            Target("overall", "overall", surface_path, instance_root / "overall" / "latent.npz")
        ]
        missing_part_voxels: list[str] = []
        for part_name in target_part_names:
            ind_path = voxel_dir / f"ind_{part_name}.npy"
            if not ind_path.is_file():
                missing_part_voxels.append(str(ind_path))
            else:
                targets.append(Target(part_name, "part", ind_path, instance_root / part_name / "latent.npz"))
        if missing_part_voxels:
            counters.failed += 1
            counters.skipped_missing_voxel += len(missing_part_voxels)
            records.append(
                {
                    "status": "failed",
                    "object_id": oid,
                    "angle_idx": angle_idx,
                    "sample_id": sample_id,
                    "instance_id": instance_id,
                    "reason": "missing Part Completion target voxel(s); instance not encoded",
                    "missing": missing_part_voxels,
                }
            )
            continue
        counters.targets_seen += len(targets)
        instances.append(
            Instance(
                oid,
                angle_idx,
                instance_id,
                instance_root,
                camera_path,
                token_path,
                part_info_path,
                tuple(targets),
                sample_id,
                view_indices,
            )
        )
    return instances


def shard_instances(instances, rank: int, world_size: int):
    if world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("--rank must satisfy 0 <= rank < --world-size")
    # Round-robin sharding balances variable target-part counts better than
    # contiguous chunks while still giving each rank a disjoint deterministic set.
    return instances[rank::world_size]


def import_trellis(args: argparse.Namespace):
    trellis_root = Path(resolve_repo_path(args.trellis_root))
    if not trellis_root.is_absolute():
        raise ValueError(f"--trellis-root must be an absolute or repo-relative path: {trellis_root}")
    if not trellis_root.is_dir():
        raise FileNotFoundError(f"--trellis-root does not exist or is not a directory: {trellis_root}")
    sys.path.insert(0, str(trellis_root))
    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import torch in the official dataset_toolkits environment: {exc!r}") from exc
    try:
        trellis_pkg_dir = trellis_root / "trellis"
        if trellis_pkg_dir.is_dir() and "trellis" not in sys.modules:
            pkg = types.ModuleType("trellis")
            pkg.__path__ = [str(trellis_pkg_dir)]  # type: ignore[attr-defined]
            pkg.__file__ = str(trellis_pkg_dir / "__init__.py")
            sys.modules["trellis"] = pkg
        import trellis.models as models  # noqa: PLC0415
        import trellis.modules.sparse as sp  # noqa: PLC0415
        import utils3d.torch as utils3d_torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import trellis.models / trellis.modules.sparse / utils3d.torch. "
            "Pass --trellis-root to a TRELLIS checkout and run in the official dataset_toolkits environment. "
            f"Original error: {exc!r}"
        ) from exc
    torch.set_grad_enabled(False)
    return torch, F, models, sp, utils3d_torch


def require_trellis_checkpoint_prefix(raw_path: str, label: str) -> str:
    path = Path(resolve_repo_path(raw_path))
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute or repo-relative local path, got: {raw_path}")
    if path.exists():
        return str(path)
    json_path = path.with_suffix(".json")
    weights_path = path.with_suffix(".safetensors")
    if json_path.is_file() and weights_path.is_file():
        return str(path)
    raise FileNotFoundError(
        f"{label} must point to an existing TRELLIS checkpoint prefix/path. "
        f"Missing {json_path} or {weights_path}"
    )


def load_encoder(args: argparse.Namespace):
    torch, F, models, sp, utils3d_torch = import_trellis(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    enc_pretrained = require_trellis_checkpoint_prefix(args.enc_pretrained, "--enc-pretrained")
    encoder = models.from_pretrained(enc_pretrained).eval().to(device)
    return torch, F, sp, utils3d_torch, encoder, device


def load_patchtokens(torch, token_path: Path, device):
    with np.load(token_path) as data:
        if "tokens" not in data.files:
            raise KeyError(f"tokens.npz missing 'tokens': {token_path}")
        tokens = data["tokens"]
    if tuple(tokens.shape) != EXPECTED_TOKEN_SHAPE:
        raise ValueError(f"{token_path}: expected tokens shape {EXPECTED_TOKEN_SHAPE}, got {tokens.shape}")
    patch = torch.from_numpy(tokens[:, 1:, :].astype(np.float32, copy=False)).to(device)
    patch = patch.permute(0, 2, 1).reshape(EXPECTED_TOKEN_SHAPE[0], EXPECTED_FEATURE_DIM, EXPECTED_PATCH_GRID, EXPECTED_PATCH_GRID)
    return patch


def load_camera_matrices(torch, utils3d_torch, camera_path: Path, device):
    payload = _load_json_mapping(camera_path, "camera_transforms")
    frames = payload.get("frames")
    expected_views = EXPECTED_TOKEN_SHAPE[0]
    if not isinstance(frames, list) or len(frames) != expected_views:
        raise ValueError(f"camera_transforms.frames must have length {expected_views}: {camera_path}")
    extrinsics = []
    intrinsics = []
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise TypeError(f"camera frame {idx} must be object: {camera_path}")
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
        if tuple(c2w.shape) != (4, 4):
            raise ValueError(f"camera frame {idx} transform_matrix must be 4x4")
        # Match TRELLIS dataset_toolkits/extract_feature.py convention.
        c2w = c2w.clone()
        c2w[:3, 1:3] *= -1
        extrinsics.append(torch.inverse(c2w))
        fov = torch.tensor(float(frame["camera_angle_x"]), dtype=torch.float32, device=device)
        intrinsics.append(utils3d_torch.intrinsics_from_fov_xy(fov, fov))
    return torch.stack(extrinsics, dim=0), torch.stack(intrinsics, dim=0)


def coords_to_positions(torch, coords: np.ndarray, resolution: int, device):
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=device)
    return (coords_t + 0.5) / float(resolution) - 0.5


def project_features(torch, F, utils3d_torch, coords: np.ndarray, patchtokens, extrinsics, intrinsics, resolution: int, device):
    positions = coords_to_positions(torch, coords, resolution, device)
    uv = utils3d_torch.project_cv(positions, extrinsics, intrinsics)[0] * 2 - 1
    sampled = F.grid_sample(
        patchtokens,
        uv.unsqueeze(1),
        mode="bilinear",
        align_corners=False,
    ).squeeze(2).permute(0, 2, 1)
    return sampled.mean(dim=0).float()


def encode_target(torch, F, sp, utils3d_torch, encoder, device, target: Target, resolution: int, patchtokens, extrinsics, intrinsics):
    coords, reason = validate_voxel_file(target.voxel_path, resolution)
    if reason is not None or coords is None:
        raise ValueError(f"invalid voxel file {target.voxel_path}: {reason}")
    feats = project_features(torch, F, utils3d_torch, coords, patchtokens, extrinsics, intrinsics, resolution, device)
    sparse = sp.SparseTensor(
        feats=feats,
        coords=torch.cat(
            [
                torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=device),
                torch.as_tensor(coords, dtype=torch.int32, device=device),
            ],
            dim=1,
        ),
    )
    latent = encoder(sparse, sample_posterior=False)
    if not hasattr(latent, "feats") or not hasattr(latent, "coords"):
        raise TypeError(f"encoder returned unsupported type: {type(latent).__name__}")
    if not torch.isfinite(latent.feats).all():
        raise ValueError("encoder returned NaN or Inf feats")
    out_coords = latent.coords[:, 1:].detach().cpu().numpy().astype(np.uint8, copy=False)
    out_feats = latent.feats.detach().cpu().numpy().astype(np.float32, copy=False)
    if out_feats.ndim != 2 or out_feats.shape[1] != EXPECTED_LATENT_CHANNELS:
        raise ValueError(f"encoder feats shape {out_feats.shape} != (N,{EXPECTED_LATENT_CHANNELS})")
    return np.ascontiguousarray(out_coords), np.ascontiguousarray(out_feats)


def merge_instance(instance: Instance) -> None:
    part_ids = [target.name for target in instance.targets if target.kind == "part"]
    coords_list: list[np.ndarray] = []
    feats_list: list[np.ndarray] = []
    offsets = [0]
    for target in instance.targets:
        reason = validate_slat_file(target.latent_path)
        if reason is not None:
            raise ValueError(f"cannot merge invalid SLat {target.latent_path}: {reason}")
        with np.load(target.latent_path) as data:
            coords = data["coords"]
            feats = data["feats"]
        coords_list.append(coords)
        feats_list.append(feats)
        offsets.append(offsets[-1] + coords.shape[0])
    all_coords = np.concatenate(coords_list, axis=0)
    all_feats = np.concatenate(feats_list, axis=0).astype(np.float32, copy=False)
    all_offsets = np.asarray(offsets, dtype=np.int64)
    atomic_save_npz(instance.root / "all_latent.npz", coords=all_coords, feats=all_feats, offsets=all_offsets)
    part_id_path = instance.root / "overall" / "part_id.txt"
    part_id_path.parent.mkdir(parents=True, exist_ok=True)
    part_id_path.write_text("".join(f"{name}\n" for name in part_ids), encoding="utf-8")
    meta = {
        "object_id": instance.object_id,
        "angle_idx": instance.angle_idx,
        "instance_id": instance.instance_id,
        "source_sample_id": instance.source_sample_id,
        "view_indices": list(instance.view_indices),
        "part_ids": part_ids,
        "camera_transforms": str(instance.camera_path),
        "dinov2_tokens": str(instance.token_path),
        "part_info": str(instance.part_info_path),
    }
    (instance.root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    reason = validate_all_latent_file(instance.root / "all_latent.npz", len(part_ids))
    if reason is not None:
        raise ValueError(f"merged all_latent failed validation: {reason}")


def validate_instance_cache(instance: Instance) -> str | None:
    part_ids = [target.name for target in instance.targets if target.kind == "part"]
    component_paths = [target.latent_path for target in instance.targets]
    all_reason = validate_all_latent_segments(instance.root / "all_latent.npz", component_paths)
    if all_reason is not None:
        return "invalid_all_latent:" + all_reason
    cached_part_ids, part_reason = read_part_ids(instance.root / "overall" / "part_id.txt")
    if part_reason is not None or cached_part_ids is None:
        return "invalid_part_id_txt:" + str(part_reason)
    if cached_part_ids != part_ids:
        return f"part_id_txt_mismatch:expected={','.join(part_ids)} actual={','.join(cached_part_ids)}"
    for target in instance.targets:
        latent_reason = validate_slat_file(target.latent_path)
        if latent_reason is not None:
            return f"invalid_component_latent:{target.name}:{latent_reason}"
    return None


def process_instances(instances: list[Instance], cfg: PipelineConfig, args: argparse.Namespace, counters: Counters, records: list[dict[str, Any]]) -> None:
    if args.dry_run:
        for instance in instances:
            cache_reason = validate_instance_cache(instance)
            status = "existing_valid" if cache_reason is None and not args.overwrite else "would_generate"
            if status == "existing_valid":
                counters.existing_valid += len(instance.targets)
            else:
                counters.queued += len(instance.targets)
            records.append(
                {
                    "status": status,
                    "object_id": instance.object_id,
                    "angle_idx": instance.angle_idx,
                    "instance_id": instance.instance_id,
                    "source_sample_id": instance.source_sample_id,
                    "part_ids": [target.name for target in instance.targets if target.kind == "part"],
                    "target_count": len(instance.targets),
                    "output": str(instance.root / "all_latent.npz"),
                    "reason": cache_reason,
                }
            )
        return

    torch, F, sp, utils3d_torch, encoder, device = load_encoder(args)
    resolution = cfg.voxel.resolution
    for instance in tqdm(instances, desc="Encoding part-synthesis SLat instances"):
        instance_failed = False
        instance_context = None
        for target in instance.targets:
            try:
                existing_reason = validate_slat_file(target.latent_path)
                if existing_reason is None and not args.overwrite:
                    counters.existing_valid += 1
                    records.append(
                        {
                            "status": "existing_valid",
                            "object_id": instance.object_id,
                            "angle_idx": instance.angle_idx,
                            "instance_id": instance.instance_id,
                            "target": target.name,
                            "output": str(target.latent_path),
                        }
                    )
                    continue
                counters.queued += 1
                if instance_context is None:
                    patchtokens = load_patchtokens(torch, instance.token_path, device)
                    extrinsics, intrinsics = load_camera_matrices(torch, utils3d_torch, instance.camera_path, device)
                    instance_context = (patchtokens, extrinsics, intrinsics)
                else:
                    patchtokens, extrinsics, intrinsics = instance_context
                coords, feats = encode_target(torch, F, sp, utils3d_torch, encoder, device, target, resolution, patchtokens, extrinsics, intrinsics)
                atomic_save_npz(target.latent_path, coords=coords, feats=feats)
                post_reason = validate_slat_file(target.latent_path)
                if post_reason is not None:
                    raise ValueError(f"saved latent failed validation: {post_reason}")
                counters.generated += 1
                records.append(
                    {
                        "status": "generated",
                        "object_id": instance.object_id,
                        "angle_idx": instance.angle_idx,
                        "instance_id": instance.instance_id,
                        "target": target.name,
                        "kind": target.kind,
                        "input": str(target.voxel_path),
                        "output": str(target.latent_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                counters.failed += 1
                instance_failed = True
                records.append(
                    {
                        "status": "failed",
                        "object_id": instance.object_id,
                        "angle_idx": instance.angle_idx,
                        "instance_id": instance.instance_id,
                        "target": target.name,
                        "input": str(target.voxel_path),
                        "output": str(target.latent_path),
                        "reason": repr(exc),
                    }
                )
                if not args.continue_on_error:
                    raise
        if instance_failed:
            continue
        try:
            merge_instance(instance)
            counters.merged += 1
            records.append(
                {
                    "status": "merged",
                    "object_id": instance.object_id,
                    "angle_idx": instance.angle_idx,
                    "instance_id": instance.instance_id,
                    "output": str(instance.root / "all_latent.npz"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            counters.failed += 1
            records.append(
                {
                    "status": "failed",
                    "object_id": instance.object_id,
                    "angle_idx": instance.angle_idx,
                    "instance_id": instance.instance_id,
                    "reason": f"merge failed: {exc!r}",
                }
            )
            if not args.continue_on_error:
                raise


def write_report(cfg: PipelineConfig, config_path: Path, args: argparse.Namespace, counters: Counters, records: list[dict[str, Any]], report_path: Path, output_root: Path) -> dict[str, Any]:
    ts, iso = _utc_now()
    report = {
        "dataset": cfg.dataset_name,
        "config_path": str(config_path.resolve()),
        "timestamp_unix": ts,
        "timestamp_iso": iso,
        "step": "12_encode_part_synthesis_slat",
        "output_root": str(output_root),
        "completion_manifest": str(Path(args.completion_manifest).resolve()) if args.completion_manifest else str((Path(cfg.data_root) / "manifests" / "part_completion" / f"arts_pc_{_dataset_slug(cfg.dataset_name)}_train.jsonl").resolve()),
        "source_of_truth": "Part Completion manifest target_part_names; no part_info fallback enumeration",
        "latent_contract": {
            "component_template": "{output_root}/{instance_id[:2]}/{instance_id}/{overall|part_name}/latent.npz",
            "merged_template": "{output_root}/{instance_id[:2]}/{instance_id}/all_latent.npz",
            "coords": "uint8/integer (N,3) in [0,64)",
            "feats": f"float32 (N,{EXPECTED_LATENT_CHANNELS}) finite",
        },
        "options": {
            "object_ids": args.object_ids,
            "object_list": args.object_list,
            "completion_manifest": args.completion_manifest,
            "rank": args.rank,
            "world_size": args.world_size,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "enc_pretrained": args.enc_pretrained,
            "trellis_root": args.trellis_root,
        },
        "summary": {
            "passed": counters.failed == 0,
            "objects_seen": counters.objects_seen,
            "angles_seen": counters.angles_seen,
            "instances_seen": counters.instances_seen,
            "targets_seen": counters.targets_seen,
            "queued": counters.queued,
            "generated": counters.generated,
            "existing_valid": counters.existing_valid,
            "skipped_missing_voxel": counters.skipped_missing_voxel,
            "merged": counters.merged,
            "failed": counters.failed,
        },
        "records": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def print_summary(report: dict[str, Any], report_path: Path) -> None:
    s = report["summary"]
    print(
        "[part-synthesis-slat] "
        f"dataset={report['dataset']} instances={s['instances_seen']} targets={s['targets_seen']} "
        f"generated={s['generated']} existing_valid={s['existing_valid']} merged={s['merged']} "
        f"skipped_missing_voxel={s['skipped_missing_voxel']} failed={s['failed']}"
    )
    print(f"[report] written to {report_path.resolve()}")
    print(f"[summary] passed={'true' if s['passed'] else 'false'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    cfg = load_config(str(config_path))
    args.enc_pretrained = args.enc_pretrained or cfg.trellis.slat_encoder
    args.trellis_root = args.trellis_root or cfg.trellis.root
    if cfg.voxel.resolution != EXPECTED_RESOLUTION:
        raise ValueError(f"Part-synthesis SLat encoder expects voxel resolution {EXPECTED_RESOLUTION}, got {cfg.voxel.resolution}")
    output_root = Path(args.output_root) if args.output_root else Path(cfg.data_root) / "part_synthesis_slat"
    completion_manifest = Path(args.completion_manifest) if args.completion_manifest else (
        Path(cfg.data_root) / "manifests" / "part_completion" / f"arts_pc_{_dataset_slug(cfg.dataset_name)}_train.jsonl"
    )
    requested = _requested_object_ids(args)
    counters = Counters()
    records: list[dict[str, Any]] = []
    completion_rows = load_completion_samples(cfg, completion_manifest, requested)
    sharded_completion_rows = shard_instances(completion_rows, args.rank, args.world_size)
    setattr(cfg, "_part_synthesis_progress_every", args.progress_every)
    instances = iter_instances(cfg, sharded_completion_rows, output_root, counters, records)
    if counters.failed:
        records.append(
            {
                "status": "fatal",
                "reason": "preflight failed; refusing to encode partial Part Synthesis SLat cache",
            }
        )
    else:
        try:
            process_instances(instances, cfg, args, counters, records)
        except Exception as exc:  # noqa: BLE001
            counters.failed += 1
            records.append({"status": "fatal", "reason": repr(exc)})
    report_path = Path(args.report_path) if args.report_path else Path(
        f"/tmp/part_synthesis_slat_{_dataset_slug(cfg.dataset_name)}_{int(time.time())}.json"
    )
    report = write_report(cfg, config_path, args, counters, records, report_path, output_root)
    print_summary(report, report_path)
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
