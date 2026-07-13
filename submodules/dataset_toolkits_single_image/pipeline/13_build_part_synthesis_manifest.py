#!/usr/bin/env python3
"""Build a Part Synthesis manifest and OmniPart mesh lists from SLat caches.

The Part Completion manifest is the source of truth for sample, view, and
target-part selection.  This stage validates Step 12 ``part_synthesis_slat``
caches against those exact rows and emits both a rich JSONL manifest and the
lightweight ``train_mesh_list`` / ``val_mesh_list`` text files expected by
OmniPart's current ``ImageConditionedSLat`` config.

No fallback enumeration is allowed: ``part_info.parts`` validates target parts,
and ``overall/part_id.txt`` validates cache consistency, but neither can add
samples or parts beyond the Part Completion manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import load_config  # noqa: E402


MANIFEST_VERSION = 1
TASK_NAME = "part_synthesis"
COMPLETION_TASK_NAME = "part_completion"
EXPECTED_LATENT_CHANNELS = 8
EXPECTED_VIEW_COUNT = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build part_synthesis manifest from Step 12 SLat cache.")
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset.")
    parser.add_argument("--object-list", help="Optional newline-delimited object ID file.")
    parser.add_argument(
        "--completion-manifest",
        help=(
            "Part Completion train JSONL source of truth. Default: "
            "<data_root>/manifests/part_completion/arts_pc_<dataset_slug>_train.jsonl"
        ),
    )
    parser.add_argument("--slat-root", help="Default: <data_root>/part_synthesis_slat.")
    parser.add_argument("--output-dir", help="Default: <data_root>/manifests/part_synthesis.")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Deterministic validation ratio in [0,1). Default train-only.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest outputs.")
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="Deprecated/no effect for this strict stage; skipped source rows still exit non-zero.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing outputs.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print progress every N rows; 0 disables.")
    return parser.parse_args(argv)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace(" ", "_")


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
    if args.object_list:
        object_list = Path(args.object_list)
        requested = [line.strip() for line in object_list.read_text().splitlines() if line.strip()]
        if len(requested) != len(set(requested)):
            raise ValueError(f"--object-list contains duplicate IDs: {object_list}")
        if any(not item for item in requested):
            raise ValueError(f"--object-list contains empty object IDs: {object_list}")
        return set(requested)
    return None


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _resolve_manifest_path(data_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def _load_json(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be a JSON object: {path}")
    return payload


def validate_component_latent(path: Path) -> str | None:
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
    if coords.shape[0] != feats.shape[0] or coords.shape[0] < 1:
        return "coords/feats rows invalid"
    if feats.dtype != np.float32:
        return f"feats dtype {feats.dtype} != float32"
    if not bool(np.isfinite(feats).all()):
        return "feats contain NaN or Inf"
    return None


def latent_row_count(path: Path) -> tuple[int | None, str | None]:
    reason = validate_component_latent(path)
    if reason is not None:
        return None, reason
    with np.load(path) as data:
        return int(data["coords"].shape[0]), None


def validate_all_latent(path: Path, expected_part_count: int) -> tuple[int | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        with np.load(path) as data:
            if set(data.files) != {"coords", "feats", "offsets"}:
                return None, f"keys {sorted(data.files)} != ['coords', 'feats', 'offsets']"
            coords = data["coords"]
            feats = data["feats"]
            offsets = data["offsets"]
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable: {exc!r}"
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None, f"coords shape {coords.shape} not (N,3)"
    if feats.ndim != 2 or feats.shape[1] != EXPECTED_LATENT_CHANNELS:
        return None, f"feats shape {feats.shape} not (N,{EXPECTED_LATENT_CHANNELS})"
    if coords.shape[0] != feats.shape[0]:
        return None, "coords/feats row mismatch"
    expected_offsets = expected_part_count + 2  # 0 + overall + each target part
    if offsets.ndim != 1 or offsets.shape[0] != expected_offsets:
        return None, f"offsets length {offsets.shape[0]} != expected {expected_offsets}"
    if int(offsets[0]) != 0 or int(offsets[-1]) != coords.shape[0]:
        return None, "offset endpoints invalid"
    if np.any(offsets[1:] < offsets[:-1]):
        return None, "offsets not monotonic"
    if feats.dtype != np.float32:
        return None, f"feats dtype {feats.dtype} != float32"
    if not bool(np.isfinite(feats).all()):
        return None, "feats contain NaN or Inf"
    return int(coords.shape[0]), None


def validate_all_latent_segments(all_latent_path: Path, component_paths: list[Path]) -> tuple[int | None, str | None]:
    voxel_num, all_reason = validate_all_latent(all_latent_path, len(component_paths) - 1)
    if all_reason is not None:
        return None, all_reason
    with np.load(all_latent_path) as data:
        offsets = data["offsets"].astype(np.int64)
    segment_lengths = np.diff(offsets)
    for idx, component_path in enumerate(component_paths):
        rows, reason = latent_row_count(component_path)
        if reason is not None or rows is None:
            return None, f"component[{idx}] invalid:{reason}"
        if int(segment_lengths[idx]) != rows:
            return None, f"segment[{idx}] length {int(segment_lengths[idx])} != component rows {rows}: {component_path}"
    return voxel_num, None


def read_part_ids(path: Path) -> tuple[list[str] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    part_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not part_ids:
        return None, "empty"
    if len(part_ids) != len(set(part_ids)):
        return None, "duplicate part ids"
    return part_ids, None


def deterministic_split(instance_id: str, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    h = hashlib.sha1(instance_id.encode("utf-8")).hexdigest()
    bucket = int(h[:12], 16) / float(16**12)
    return "val" if bucket < val_ratio else "train"


def validate_completion_row(row: dict[str, Any], data_root: Path, source: str, line_no: int) -> dict[str, Any]:
    if row.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"{source}:{line_no}: manifest_version must be {MANIFEST_VERSION}")
    if row.get("task") != COMPLETION_TASK_NAME:
        raise ValueError(f"{source}:{line_no}: task must be {COMPLETION_TASK_NAME!r}")
    sample_id = row.get("sample_id")
    object_id = row.get("object_id")
    angle_idx = row.get("angle_idx")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{source}:{line_no}: sample_id must be non-empty string")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError(f"{source}:{line_no}: object_id must be non-empty string")
    if isinstance(angle_idx, bool) or not isinstance(angle_idx, int) or angle_idx < 0:
        raise ValueError(f"{source}:{line_no}: angle_idx must be non-negative int")

    view_indices = row.get("view_indices")
    if not isinstance(view_indices, list) or len(view_indices) != EXPECTED_VIEW_COUNT:
        raise ValueError(f"{source}:{line_no}: view_indices must contain {EXPECTED_VIEW_COUNT} entries")
    if any(isinstance(view, bool) or not isinstance(view, int) or view < 0 or view > 11 for view in view_indices):
        raise ValueError(f"{source}:{line_no}: view_indices must be ints in [0,11]")
    if len(set(view_indices)) != EXPECTED_VIEW_COUNT:
        raise ValueError(f"{source}:{line_no}: view_indices contains duplicates")
    if row.get("quadrants") != [0, 1, 2, 3] or [view // 3 for view in view_indices] != [0, 1, 2, 3]:
        raise ValueError(f"{source}:{line_no}: quadrants/view_indices violate Part Completion contract")

    target_count = row.get("target_part_count")
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
        raise ValueError(f"{source}:{line_no}: target_part_count must be positive int")
    target_names = row.get("target_part_names")
    if not isinstance(target_names, list) or len(target_names) != target_count:
        raise ValueError(f"{source}:{line_no}: target_part_names length must equal target_part_count")
    if any(not isinstance(name, str) or not name for name in target_names):
        raise ValueError(f"{source}:{line_no}: target_part_names must be non-empty strings")
    if len(set(target_names)) != len(target_names):
        raise ValueError(f"{source}:{line_no}: target_part_names contains duplicates")
    target_parts = row.get("target_parts")
    if not isinstance(target_parts, list) or len(target_parts) != target_count:
        raise ValueError(f"{source}:{line_no}: target_parts length must equal target_part_count")
    for idx, (expected_name, part) in enumerate(zip(target_names, target_parts), start=1):
        if not isinstance(part, dict):
            raise ValueError(f"{source}:{line_no}: target_parts[{idx}] must be object")
        if part.get("name") != expected_name:
            raise ValueError(f"{source}:{line_no}: target_parts[{idx}].name does not match target_part_names")
        if part.get("local_label") != idx:
            raise ValueError(f"{source}:{line_no}: target_parts[{idx}].local_label must be {idx}")
        motion = part.get("motion")
        if not isinstance(motion, dict) or motion.get("motion_type") not in {"rotate", "prismatic"}:
            raise ValueError(f"{source}:{line_no}: target_parts[{idx}].motion.motion_type must be rotate/prismatic")

    paths = row.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"{source}:{line_no}: paths must be object")
    image_paths_raw = row.get("image_paths")
    masks_raw = paths.get("masks")
    if not isinstance(image_paths_raw, list) or len(image_paths_raw) != EXPECTED_VIEW_COUNT:
        raise ValueError(f"{source}:{line_no}: image_paths must contain {EXPECTED_VIEW_COUNT} entries")
    if not isinstance(masks_raw, list) or len(masks_raw) != EXPECTED_VIEW_COUNT:
        raise ValueError(f"{source}:{line_no}: paths.masks must contain {EXPECTED_VIEW_COUNT} entries")
    image_paths = [
        _resolve_manifest_path(data_root, value, f"{source}:{line_no}.image_paths[{idx}]")
        for idx, value in enumerate(image_paths_raw)
    ]
    mask_paths = [
        _resolve_manifest_path(data_root, value, f"{source}:{line_no}.paths.masks[{idx}]")
        for idx, value in enumerate(masks_raw)
    ]
    for view, image_path, mask_path in zip(view_indices, image_paths, mask_paths):
        expected_image = data_root / "renders" / object_id / f"angle_{angle_idx}" / "rgb" / f"view_{view}.png"
        expected_mask = data_root / "renders" / object_id / f"angle_{angle_idx}" / "mask" / f"mask_{view}.npy"
        if image_path != expected_image:
            raise ValueError(f"{source}:{line_no}: image path mismatch for view {view}: {image_path}")
        if mask_path != expected_mask:
            raise ValueError(f"{source}:{line_no}: mask path mismatch for view {view}: {mask_path}")

    part_info_path = _resolve_manifest_path(data_root, paths.get("part_info"), f"{source}:{line_no}.paths.part_info")
    expected_part_info = data_root / "part_info" / object_id / "part_info.json"
    if part_info_path != expected_part_info:
        raise ValueError(f"{source}:{line_no}: part_info path mismatch: {part_info_path}")
    return {
        "sample_id": sample_id,
        "object_id": object_id,
        "angle_idx": angle_idx,
        "view_indices": tuple(int(view) for view in view_indices),
        "image_paths": image_paths,
        "mask_paths": mask_paths,
        "target_part_names": list(target_names),
        "target_parts": target_parts,
        "part_info_path": part_info_path,
    }


def load_completion_rows(config, args: argparse.Namespace, data_root: Path) -> tuple[list[dict[str, Any]], Path]:
    dataset_slug = _dataset_slug(config.dataset_name)
    manifest_path = Path(args.completion_manifest) if args.completion_manifest else (
        data_root / "manifests" / "part_completion" / f"arts_pc_{dataset_slug}_train.jsonl"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Part Completion manifest not found: {manifest_path}")

    requested = _requested_object_ids(args)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    manifest_objects: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{manifest_path}:{line_no}: row must be JSON object")
            object_id = raw.get("object_id")
            if isinstance(object_id, str):
                manifest_objects.add(object_id)
            if requested is not None and object_id not in requested:
                continue
            row = validate_completion_row(raw, data_root, str(manifest_path), line_no)
            key = (row["object_id"], row["angle_idx"])
            if key in seen:
                raise ValueError(f"{manifest_path}:{line_no}: duplicate object_id/angle_idx sample {key}")
            seen.add(key)
            rows.append(row)

    if requested is not None:
        missing = sorted(requested - manifest_objects)
        if missing:
            raise ValueError("Requested object IDs are absent from Part Completion manifest: " + ", ".join(missing))
    if not rows:
        raise ValueError(f"No Part Completion manifest rows selected from {manifest_path}")
    return rows, manifest_path


def validate_target_against_part_info(part_info_parts: dict[str, Any], object_id: str, part_name: str) -> str | None:
    entry = part_info_parts.get(part_name)
    if not isinstance(entry, dict):
        return f"target_part_not_in_part_info:{part_name}"
    label = entry.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
        return f"invalid_part_info_label:{part_name}"
    joint = entry.get("joint")
    joint_type = entry.get("joint_type")
    if not isinstance(joint, str) or not joint:
        return f"invalid_part_info_joint:{part_name}"
    if not isinstance(joint_type, str) or not joint_type:
        return f"invalid_part_info_joint_type:{part_name}"
    if joint == "fixed" or joint_type == "E":
        return f"target_part_not_movable_in_part_info:{part_name}"
    return None


def build_rows(config, args: argparse.Namespace):
    data_root = Path(config.data_root)
    reconstruction = Path(config.reconstruction_dir)
    renders = Path(config.renders_dir)
    slat_root = Path(args.slat_root) if args.slat_root else data_root / "part_synthesis_slat"
    completion_rows, completion_manifest = load_completion_rows(config, args, data_root)
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    skip_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    voxel_num_by_split: dict[str, list[tuple[str, int]]] = {"train": [], "val": []}
    part_info_cache: dict[str, dict[str, Any]] = {}

    for row_index, source_row in enumerate(completion_rows, start=1):
        object_id = str(source_row["object_id"])
        angle_idx = int(source_row["angle_idx"])
        target_part_names = list(source_row["target_part_names"])
        selected_rgb_paths = list(source_row["image_paths"])
        selected_mask_paths = list(source_row["mask_paths"])
        instance_id = f"{object_id}_angle_{angle_idx}"
        sample_id = str(source_row["sample_id"])
        instance_root = slat_root / instance_id[:2] / instance_id
        all_latent_path = instance_root / "all_latent.npz"
        part_id_path = instance_root / "overall" / "part_id.txt"
        component_paths = [instance_root / "overall" / "latent.npz"] + [
            instance_root / part_id / "latent.npz" for part_id in target_part_names
        ]
        voxel_num, all_reason = validate_all_latent_segments(all_latent_path, component_paths)
        part_ids, part_reason = read_part_ids(part_id_path)

        reasons: list[str] = []
        if all_reason is not None:
            reasons.append("invalid_all_latent:" + all_reason)
        if part_reason is not None or part_ids is None:
            reasons.append("invalid_part_id_txt:" + str(part_reason))
        elif part_ids != target_part_names:
            reasons.append(
                "part_id_txt_mismatch:"
                f"expected={','.join(target_part_names)} actual={','.join(part_ids)}"
            )

        part_info_path = Path(source_row["part_info_path"])
        part_info_parts: dict[str, Any] = {}
        cached_part_info = part_info_cache.get(object_id)
        if cached_part_info is not None:
            part_info_parts = cached_part_info
        elif not part_info_path.is_file():
            reasons.append("missing_part_info")
        else:
            try:
                part_info = _load_json(part_info_path, f"part_info[{object_id}]")
                raw_parts = part_info.get("parts")
                if not isinstance(raw_parts, dict):
                    reasons.append("invalid_part_info_parts")
                else:
                    part_info_parts = raw_parts
                    part_info_cache[object_id] = raw_parts
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"invalid_part_info:{type(exc).__name__}:{exc}")

        camera_path = renders / object_id / f"angle_{angle_idx}" / "camera_transforms.json"
        if not camera_path.is_file():
            reasons.append("missing_camera_transforms")
        if any(not path.is_file() for path in selected_rgb_paths):
            reasons.append("missing_rgb")
        if any(not path.is_file() for path in selected_mask_paths):
            reasons.append("missing_mask")

        target_part_paths: dict[str, str] = {}
        for part_id in target_part_names:
            if part_info_parts:
                part_info_reason = validate_target_against_part_info(part_info_parts, object_id, part_id)
                if part_info_reason is not None:
                    reasons.append(part_info_reason)
            latent_path = instance_root / part_id / "latent.npz"
            latent_reason = validate_component_latent(latent_path)
            if latent_reason is not None:
                reasons.append(f"invalid_part_latent:{part_id}:{latent_reason}")
            target_part_paths[part_id] = _relative_to(latent_path, data_root)

        overall_reason = validate_component_latent(instance_root / "overall" / "latent.npz")
        if overall_reason is not None:
            reasons.append("invalid_overall_latent:" + overall_reason)

        if reasons:
            unique_reasons = sorted(set(reasons))
            skip_counter.update(unique_reasons)
            skips.append({"sample_id": sample_id, "object_id": object_id, "angle_idx": angle_idx, "reasons": unique_reasons})
            if args.progress_every and (len(rows) + len(skips)) % args.progress_every == 0:
                print(f"Processed {len(rows) + len(skips)} instances: rows={len(rows)} skipped={len(skips)}", flush=True)
            continue

        assert voxel_num is not None
        split = deterministic_split(instance_id, args.val_ratio)
        split_counter.update([split])
        voxel_num_by_split[split].append((instance_id, voxel_num))
        rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "task": TASK_NAME,
                "sample_id": sample_id,
                "source_completion_sample_id": sample_id,
                "instance_id": instance_id,
                "object_id": object_id,
                "angle_idx": angle_idx,
                "split": split,
                "view_indices": list(source_row["view_indices"]),
                "voxel_num": voxel_num,
                "part_count": len(target_part_names),
                "part_ids": target_part_names,
                "target_parts": source_row["target_parts"],
                "paths": {
                    "slat_root": _relative_to(instance_root, data_root),
                    "all_latent": _relative_to(all_latent_path, data_root),
                    "overall_latent": _relative_to(instance_root / "overall" / "latent.npz", data_root),
                    "part_latents": target_part_paths,
                    "part_id_txt": _relative_to(part_id_path, data_root),
                    "part_info": _relative_to(part_info_path, data_root),
                    "camera_transforms": _relative_to(camera_path, data_root),
                    "rgb": [_relative_to(path, data_root) for path in selected_rgb_paths],
                    "masks": [_relative_to(path, data_root) for path in selected_mask_paths],
                },
            }
        )
        if args.progress_every and (len(rows) + len(skips)) % args.progress_every == 0:
            print(f"Processed {len(rows) + len(skips)} instances: rows={len(rows)} skipped={len(skips)}", flush=True)
        if args.progress_every and row_index % max(args.progress_every, 1) == 0:
            print(f"Completion rows processed: {row_index}/{len(completion_rows)}", flush=True)

    return rows, skips, skip_counter, split_counter, voxel_num_by_split, slat_root, completion_manifest, len(completion_rows)


def write_outputs(config, args: argparse.Namespace) -> int:
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio must be in [0,1)")
    data_root = Path(config.data_root)
    dataset_slug = _dataset_slug(config.dataset_name)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "manifests" / "part_synthesis"
    manifest_path = output_dir / f"arts_mllm_{dataset_slug}.jsonl"
    train_mesh_list_path = output_dir / "train_mesh_list.txt"
    val_mesh_list_path = output_dir / "val_mesh_list.txt"
    meta_path = output_dir / "manifest_meta.json"
    skip_path = output_dir / "skip_report.json"
    existing = [path for path in (manifest_path, train_mesh_list_path, val_mesh_list_path, meta_path, skip_path) if path.exists()]
    if existing and not args.overwrite and not args.dry_run:
        raise FileExistsError("output file(s) already exist; pass --overwrite: " + ", ".join(str(path) for path in existing))

    t0 = time.time()
    rows, skips, skip_counter, split_counter, voxel_num_by_split, slat_root, completion_manifest, source_row_count = build_rows(config, args)
    created_at = _utc_now_iso()
    meta = {
        "manifest_version": MANIFEST_VERSION,
        "task": TASK_NAME,
        "created_at": created_at,
        "created_by": str(Path(__file__).resolve()),
        "config": str(Path(args.config).resolve()),
        "data_root": str(data_root),
        "slat_root": str(slat_root),
        "completion_manifest": str(completion_manifest),
        "source_of_truth": "Part Completion manifest target_part_names/view_indices; part_id.txt only validates cache consistency",
        "dataset_name": config.dataset_name,
        "dataset_slug": dataset_slug,
        "manifest": str(manifest_path),
        "train_mesh_list": str(train_mesh_list_path),
        "val_mesh_list": str(val_mesh_list_path),
        "skip_report": str(skip_path),
        "split_rule": "deterministic sha1(instance_id) bucket; val_ratio=%.6f" % args.val_ratio,
        "counts": {
            "source_completion_rows": source_row_count,
            "rows": len(rows),
            "skipped": len(skips),
            "splits": dict(split_counter),
            "skip_reasons": dict(skip_counter.most_common()),
        },
        "schema": {
            "mesh_list_line": "<instance_id> <voxel_num>",
            "omnipart_data_root": str(slat_root),
            "instance_id": "<object_id>_angle_<angle_idx>",
            "sample_unit": "object_id + angle_idx + exact Part Completion target_part_names",
            "views": "exact 4 Part Completion selected view_indices",
        },
        "runtime_seconds": round(time.time() - t0, 3),
    }
    skip_report = {
        "created_at": created_at,
        "manifest": str(manifest_path),
        "completion_manifest": str(completion_manifest),
        "rows": len(rows),
        "skipped": len(skips),
        "skip_reasons": dict(skip_counter.most_common()),
        "skips": skips,
    }

    print(f"Part synthesis rows: {len(rows)} skipped={len(skips)} splits={dict(split_counter)}")
    print(f"SLat root: {slat_root}")
    if args.dry_run:
        print("Dry run: no files written")
        if skips:
            print("ERROR: skipped rows found; strict manifest generation requires zero skips", file=sys.stderr)
            return 1
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if skips:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        skip_path.write_text(json.dumps(skip_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Meta: {meta_path}")
        print(f"Skip report: {skip_path}")
        print("ERROR: skipped rows found; strict manifest generation requires zero skips; manifest/list not written", file=sys.stderr)
        return 1

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    train_mesh_list_path.write_text(
        "".join(f"{instance_id} {voxel_num}\n" for instance_id, voxel_num in voxel_num_by_split["train"]),
        encoding="utf-8",
    )
    val_mesh_list_path.write_text(
        "".join(f"{instance_id} {voxel_num}\n" for instance_id, voxel_num in voxel_num_by_split["val"]),
        encoding="utf-8",
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    skip_path.write_text(json.dumps(skip_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    print(f"Train mesh list: {train_mesh_list_path}")
    print(f"Val mesh list: {val_mesh_list_path}")
    print(f"Meta: {meta_path}")
    print(f"Skip report: {skip_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    return write_outputs(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
