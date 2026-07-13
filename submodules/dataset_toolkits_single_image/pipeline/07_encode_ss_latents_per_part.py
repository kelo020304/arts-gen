#!/usr/bin/env python3
"""Encode TRELLIS sparse-structure latents from voxelized surface indices.

Input (from pipeline/03_voxelize.py):
    reconstruction/voxel_expanded/<object_id>/angle_<i>/64/surface.npy
    reconstruction/voxel_expanded/<object_id>/angle_<i>/64/ind_<part_name>.npy

Output:
    reconstruction/ss_latents_expanded/<object_id>/angle_<i>/latent.npz
    reconstruction/ss_latents_per_part/<object_id>/angle_<i>/<part_name>.npy

The overall ``latent.npz`` stores a ``mean`` array with shape (8, 16, 16, 16).
Each per-part output latent is a float32 array with shape (8, 16, 16, 16).
The encoder loading and occupancy-grid construction mirror TRELLIS' original
`dataset_toolkits/encode_ss_latent.py`, adapted from object-level PLY inputs to
this repository's voxel-index arrays.
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
from typing import Any, Iterable, Literal

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import PipelineConfig, load_config, resolve_repo_path  # noqa: E402


EXPECTED_RESOLUTION = 64
EXPECTED_LATENT_SHAPE = (8, 16, 16, 16)
EXPECTED_LATENT_DTYPE = np.float32
VALID_COVERAGE = ("voxel-kept", "part-info-all")
VALID_LATENT_SCOPE = ("all", "overall", "parts")


@dataclass(frozen=True)
class EncodeItem:
    object_id: str
    angle_idx: int
    kind: Literal["overall", "part"]
    ind_path: Path
    out_path: Path
    part_name: str | None = None


@dataclass
class Counters:
    objects_seen: int = 0
    angles_seen: int = 0
    overall_seen: int = 0
    parts_seen: int = 0
    queued: int = 0
    generated: int = 0
    existing_valid: int = 0
    overall_queued: int = 0
    overall_generated: int = 0
    overall_existing_valid: int = 0
    overall_failed: int = 0
    skipped_missing_voxel: int = 0
    failed: int = 0
    extra_files: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode TRELLIS SS latents per canonical part from 03_voxelize ind_*.npy files."
    )
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument(
        "--object-ids",
        help="Optional comma-separated object ID subset, e.g. 100013,100712.",
    )
    parser.add_argument(
        "--object-list",
        help="Optional newline-delimited object ID file. Mutually exclusive with --object-ids.",
    )
    parser.add_argument(
        "--coverage",
        choices=VALID_COVERAGE,
        default="voxel-kept",
        help=(
            "Expected part scope. 'voxel-kept' generates for parts with existing ind_*.npy "
            "and reports missing voxel inds as explicit skips. 'part-info-all' treats missing "
            "ind_*.npy as failures."
        ),
    )
    parser.add_argument(
        "--latent-scope",
        choices=VALID_LATENT_SCOPE,
        default="all",
        help=(
            "Latent outputs to process. 'all' encodes overall surface latent plus per-part latents; "
            "'overall' only processes reconstruction/ss_latents_expanded; "
            "'parts' preserves the historical per-part-only behavior."
        ),
    )
    parser.add_argument(
        "--enc-pretrained",
        help="Absolute or repo-relative TRELLIS sparse-structure encoder prefix/path. Default: trellis.ss_encoder from config.",
    )
    parser.add_argument(
        "--trellis-root",
        help="Absolute or repo-relative path containing the trellis Python package. Default: trellis.root from config.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for encoding. Use 'cuda' by default, or 'cpu' for debugging.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of part occupancy grids encoded per forward pass.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Shard rank for distributed/manual splitting.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Shard count for distributed/manual splitting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing outputs instead of accepting valid files as complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate work and write report without importing TRELLIS or writing latent files.",
    )
    parser.add_argument(
        "--report-path",
        help="JSON report path (default: /tmp/ss_latents_per_part_<dataset>_<ts>.json).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after per-item failures. Default stops after first encoding failure.",
    )
    return parser.parse_args(argv)


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace(" ", "_")


def _utc_now() -> tuple[int, str]:
    now = int(time.time())
    return now, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_object_ids(raw_value: str) -> list[str]:
    object_ids = [item.strip() for item in raw_value.split(",")]
    if not object_ids or any(not item for item in object_ids):
        raise ValueError("--object-ids must be a comma-separated list of non-empty IDs")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("--object-ids contains duplicate IDs")
    return object_ids


def _resolve_object_ids(cfg: PipelineConfig, args: argparse.Namespace) -> list[str]:
    if args.object_ids and args.object_list:
        raise ValueError("--object-ids and --object-list are mutually exclusive")

    available = cfg.list_object_ids()
    available_set = set(available)
    if args.object_ids:
        requested = _parse_object_ids(args.object_ids)
    elif args.object_list:
        list_path = Path(args.object_list)
        requested = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        if len(requested) != len(set(requested)):
            raise ValueError(f"--object-list contains duplicate IDs: {list_path}")
    else:
        return available

    unknown = [oid for oid in requested if oid not in available_set]
    if unknown:
        raise ValueError("Unknown or filtered-out object IDs: " + ", ".join(unknown))
    return requested


def _load_json_mapping(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object: {path}")
    return payload


def load_part_names(part_info_path: Path, object_id: str) -> list[str]:
    payload = _load_json_mapping(part_info_path, "part_info")
    if str(payload.get("object_id")) != str(object_id):
        raise ValueError(
            f"part_info object_id mismatch for {part_info_path}: "
            f"expected {object_id}, got {payload.get('object_id')!r}"
        )
    parts = payload.get("parts")
    label_to_key = payload.get("label_to_key")
    num_parts = payload.get("num_parts")
    if not isinstance(parts, dict):
        raise TypeError(f"part_info.parts must be a dict: {part_info_path}")
    if not isinstance(label_to_key, dict):
        raise TypeError(f"part_info.label_to_key must be a dict: {part_info_path}")
    if not isinstance(num_parts, int) or isinstance(num_parts, bool) or num_parts < 1:
        raise ValueError(f"part_info.num_parts must be a positive int: {part_info_path}")

    names: list[str] = []
    for part_idx in range(num_parts):
        key = str(part_idx)
        if key not in label_to_key:
            raise KeyError(f"part_info.label_to_key missing {key!r}: {part_info_path}")
        part_name = label_to_key[key]
        if not isinstance(part_name, str) or not part_name:
            raise TypeError(f"part_info.label_to_key[{key!r}] must be non-empty string")
        if part_name not in parts:
            raise KeyError(f"part {part_name!r} from label_to_key missing in parts")
        entry = parts[part_name]
        if not isinstance(entry, dict):
            raise TypeError(f"part_info.parts[{part_name!r}] must be a dict")
        if entry.get("part_index") != part_idx:
            raise ValueError(
                f"part_index mismatch for {part_name!r}: expected {part_idx}, got {entry.get('part_index')!r}"
            )
        if part_name in names:
            raise ValueError(f"duplicate part name in part_info: {part_name}")
        names.append(part_name)
    return names


def validate_latent_file(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 - report exact file-level issue
        return f"unreadable: {exc!r}"
    if arr.dtype != EXPECTED_LATENT_DTYPE:
        return f"dtype {arr.dtype} != float32"
    if tuple(arr.shape) != EXPECTED_LATENT_SHAPE:
        return f"shape {tuple(arr.shape)} != {EXPECTED_LATENT_SHAPE}"
    try:
        if not bool(np.isfinite(arr).all()):
            return "contains NaN or Inf"
    except Exception as exc:  # noqa: BLE001
        return f"finite check failed: {exc!r}"
    return None


def validate_overall_latent_file(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        with np.load(path) as data:
            if "mean" not in data.files:
                return "missing mean"
            arr = data["mean"]
            if arr.dtype != EXPECTED_LATENT_DTYPE:
                return f"mean dtype {arr.dtype} != float32"
            if tuple(arr.shape) != EXPECTED_LATENT_SHAPE:
                return f"mean shape {tuple(arr.shape)} != {EXPECTED_LATENT_SHAPE}"
            if not bool(np.isfinite(arr).all()):
                return "mean contains NaN or Inf"
    except Exception as exc:  # noqa: BLE001 - report exact file-level issue
        return f"unreadable: {exc!r}"
    return None


def validate_output_file(item: EncodeItem) -> str | None:
    if item.kind == "overall":
        return validate_overall_latent_file(item.out_path)
    return validate_latent_file(item.out_path)


def validate_ind_file(path: Path, resolution: int) -> tuple[np.ndarray | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable: {exc!r}"
    if arr.dtype != np.int64:
        return None, f"dtype {arr.dtype} != int64"
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None, f"shape {arr.shape} not (N,3)"
    if arr.shape[0] < 1:
        return None, "empty voxel index array"
    if int(arr.min()) < 0 or int(arr.max()) >= resolution:
        return None, f"coords out of [0,{resolution})"
    return arr, None


def iter_expected_items(
    cfg: PipelineConfig,
    object_ids: Iterable[str],
    coverage: str,
    latent_scope: str,
    counters: Counters,
    records: list[dict[str, Any]],
) -> list[EncodeItem]:
    part_info_root = Path(cfg.part_info_dir)
    voxel_root = Path(cfg.reconstruction_dir) / "voxel_expanded"
    overall_out_root = Path(cfg.reconstruction_dir) / "ss_latents_expanded"
    part_out_root = Path(cfg.reconstruction_dir) / "ss_latents_per_part"
    res = cfg.voxel.resolution

    items: list[EncodeItem] = []
    for oid in object_ids:
        counters.objects_seen += 1
        part_names: list[str] = []
        if latent_scope in ("all", "parts"):
            part_names = load_part_names(part_info_root / oid / "part_info.json", oid)
        num_angles = cfg.get_num_angles(oid)
        counters.angles_seen += num_angles
        for angle_i in range(num_angles):
            angle_voxel_dir = voxel_root / oid / f"angle_{angle_i}" / str(res)
            if latent_scope in ("all", "overall"):
                counters.overall_seen += 1
                surface_path = angle_voxel_dir / "surface.npy"
                overall_out_path = overall_out_root / oid / f"angle_{angle_i}" / "latent.npz"
                if not surface_path.is_file():
                    counters.failed += 1
                    counters.overall_failed += 1
                    records.append(
                        {
                            "status": "failed",
                            "kind": "overall",
                            "object_id": oid,
                            "angle_idx": angle_i,
                            "part_name": None,
                            "input": str(surface_path),
                            "output": str(overall_out_path),
                            "reason": "missing surface.npy; no fallback allowed",
                        }
                    )
                else:
                    items.append(EncodeItem(oid, angle_i, "overall", surface_path, overall_out_path))

            if latent_scope not in ("all", "parts"):
                continue

            angle_out_dir = part_out_root / oid / f"angle_{angle_i}"
            expected_names = set(part_names)
            if angle_out_dir.is_dir():
                for extra_path in angle_out_dir.glob("*.npy"):
                    if extra_path.stem not in expected_names:
                        counters.extra_files += 1
                        records.append(
                            {
                                "status": "extra_file",
                                "kind": "part",
                                "object_id": oid,
                                "angle_idx": angle_i,
                                "part_name": extra_path.stem,
                                "path": str(extra_path),
                                "reason": "latent basename is not a canonical part_info key",
                            }
                        )
            for part_name in part_names:
                counters.parts_seen += 1
                ind_path = angle_voxel_dir / f"ind_{part_name}.npy"
                out_path = angle_out_dir / f"{part_name}.npy"
                if not ind_path.is_file():
                    if coverage == "voxel-kept":
                        counters.skipped_missing_voxel += 1
                        records.append(
                            {
                                "status": "skipped",
                                "kind": "part",
                                "object_id": oid,
                                "angle_idx": angle_i,
                                "part_name": part_name,
                                "input": str(ind_path),
                                "output": str(out_path),
                                "reason": "missing voxel ind; skipped under coverage=voxel-kept",
                            }
                        )
                        continue
                    counters.failed += 1
                    records.append(
                        {
                            "status": "failed",
                            "kind": "part",
                            "object_id": oid,
                            "angle_idx": angle_i,
                            "part_name": part_name,
                            "input": str(ind_path),
                            "output": str(out_path),
                            "reason": "missing voxel ind under coverage=part-info-all",
                        }
                    )
                    continue
                items.append(EncodeItem(oid, angle_i, "part", ind_path, out_path, part_name))
    return items


def shard_items(items: list[EncodeItem], rank: int, world_size: int) -> list[EncodeItem]:
    if world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("--rank must satisfy 0 <= rank < --world-size")
    start = len(items) * rank // world_size
    end = len(items) * (rank + 1) // world_size
    return items[start:end]


def import_trellis_models(trellis_root: Path):
    trellis_root = Path(resolve_repo_path(trellis_root))
    if not trellis_root.is_absolute():
        raise ValueError(f"--trellis-root must be an absolute or repo-relative path: {trellis_root}")
    if not trellis_root.is_dir():
        raise FileNotFoundError(f"--trellis-root does not exist or is not a directory: {trellis_root}")
    sys.path.insert(0, str(trellis_root))

    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import torch. Run this stage in the official dataset_toolkits environment. "
            f"Original error: {exc!r}"
        ) from exc

    try:
        # TRELLIS' package __init__ eagerly imports pipelines/renderers/representations,
        # which can require unrelated optional deps (rembg, flexicubes, etc.). Step 07
        # only needs trellis.models.from_pretrained, so install a lightweight parent
        # package stub and import the models subpackage directly. This mirrors the
        # reference encoder's model API while avoiding non-encoder import failures.
        trellis_pkg_dir = trellis_root / "trellis"
        if trellis_pkg_dir.is_dir() and "trellis" not in sys.modules:
            pkg = types.ModuleType("trellis")
            pkg.__path__ = [str(trellis_pkg_dir)]  # type: ignore[attr-defined]
            pkg.__file__ = str(trellis_pkg_dir / "__init__.py")
            sys.modules["trellis"] = pkg
        import trellis.models as models  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import trellis.models. Pass --trellis-root pointing to the TRELLIS "
            "checkout that contains trellis/models, and run in the official dataset_toolkits environment. "
            f"Original error: {exc!r}"
        ) from exc
    torch.set_grad_enabled(False)
    return torch, models


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
    torch, models = import_trellis_models(Path(args.trellis_root))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    enc_pretrained = require_trellis_checkpoint_prefix(args.enc_pretrained, "--enc-pretrained")
    encoder = models.from_pretrained(enc_pretrained).eval().to(device)
    return torch, encoder, device


def make_occupancy_tensor(torch, coords: np.ndarray, resolution: int, device):
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.float32, device=device)
    coord_tensor = torch.as_tensor(coords, dtype=torch.long, device=device)
    ss[:, coord_tensor[:, 0], coord_tensor[:, 1], coord_tensor[:, 2]] = 1.0
    return ss


def normalize_encoder_output(torch, latent) -> np.ndarray:
    if isinstance(latent, (tuple, list)):
        latent = latent[0]
    if hasattr(latent, "mean") and not torch.is_tensor(latent):
        latent = latent.mean
    if not torch.is_tensor(latent):
        raise TypeError(f"encoder returned unsupported type: {type(latent).__name__}")
    if not torch.isfinite(latent).all():
        raise ValueError("encoder returned NaN or Inf")
    arr = latent.detach().cpu().numpy().astype(np.float32, copy=False)
    return arr


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, arr)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


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


def encode_items(
    items: list[EncodeItem],
    cfg: PipelineConfig,
    args: argparse.Namespace,
    counters: Counters,
    records: list[dict[str, Any]],
) -> None:
    def mark_queued(item: EncodeItem) -> None:
        counters.queued += 1
        if item.kind == "overall":
            counters.overall_queued += 1

    def mark_existing_valid(item: EncodeItem) -> None:
        counters.existing_valid += 1
        if item.kind == "overall":
            counters.overall_existing_valid += 1

    def mark_generated(item: EncodeItem) -> None:
        counters.generated += 1
        if item.kind == "overall":
            counters.overall_generated += 1

    def mark_failed(item: EncodeItem) -> None:
        counters.failed += 1
        if item.kind == "overall":
            counters.overall_failed += 1

    if args.dry_run:
        for item in items:
            reason = validate_output_file(item)
            if reason is None and not args.overwrite:
                mark_existing_valid(item)
                status = "existing_valid"
            else:
                mark_queued(item)
                status = "would_generate"
            records.append(
                {
                    "status": status,
                    "kind": item.kind,
                    "object_id": item.object_id,
                    "angle_idx": item.angle_idx,
                    "part_name": item.part_name,
                    "input": str(item.ind_path),
                    "output": str(item.out_path),
                    "reason": reason,
                }
            )
        return

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    torch, encoder, device = load_encoder(args)
    res = cfg.voxel.resolution

    batch_tensors = []
    batch_items: list[EncodeItem] = []

    def flush_batch() -> None:
        if not batch_items:
            return
        batch = torch.stack(batch_tensors, dim=0)
        latent = encoder(batch, sample_posterior=False)
        arr = normalize_encoder_output(torch, latent)
        expected_batch_shape = (len(batch_items), *EXPECTED_LATENT_SHAPE)
        if tuple(arr.shape) != expected_batch_shape:
            raise ValueError(f"encoder latent shape {arr.shape} != {expected_batch_shape}")
        for idx, item in enumerate(batch_items):
            one = np.ascontiguousarray(arr[idx], dtype=np.float32)
            if item.kind == "overall":
                atomic_save_npz(item.out_path, mean=one)
            else:
                atomic_save_npy(item.out_path, one)
            post_reason = validate_output_file(item)
            if post_reason is not None:
                raise ValueError(f"saved latent failed validation: {post_reason}")
            mark_generated(item)
            records.append(
                {
                    "status": "generated",
                    "kind": item.kind,
                    "object_id": item.object_id,
                    "angle_idx": item.angle_idx,
                    "part_name": item.part_name,
                    "input": str(item.ind_path),
                    "output": str(item.out_path),
                }
            )
        batch_tensors.clear()
        batch_items.clear()

    for item in tqdm(items, desc="Encoding SS latents"):
        try:
            existing_reason = validate_output_file(item)
            if existing_reason is None and not args.overwrite:
                mark_existing_valid(item)
                records.append(
                    {
                        "status": "existing_valid",
                        "kind": item.kind,
                        "object_id": item.object_id,
                        "angle_idx": item.angle_idx,
                        "part_name": item.part_name,
                        "input": str(item.ind_path),
                        "output": str(item.out_path),
                    }
                )
                continue

            coords, ind_reason = validate_ind_file(item.ind_path, res)
            if ind_reason is not None or coords is None:
                raise ValueError(f"invalid voxel coordinate input: {ind_reason}")
            batch_tensors.append(make_occupancy_tensor(torch, coords, res, device))
            batch_items.append(item)
            mark_queued(item)
            if len(batch_items) >= args.batch_size:
                flush_batch()
        except Exception as exc:  # noqa: BLE001
            mark_failed(item)
            records.append(
                {
                    "status": "failed",
                    "kind": item.kind,
                    "object_id": item.object_id,
                    "angle_idx": item.angle_idx,
                    "part_name": item.part_name,
                    "input": str(item.ind_path),
                    "output": str(item.out_path),
                    "reason": repr(exc),
                }
            )
            batch_tensors.clear()
            batch_items.clear()
            if not args.continue_on_error:
                raise
    flush_batch()


def write_report(
    cfg: PipelineConfig,
    config_path: Path,
    args: argparse.Namespace,
    counters: Counters,
    records: list[dict[str, Any]],
    report_path: Path,
) -> dict[str, Any]:
    ts, iso = _utc_now()
    report = {
        "dataset": cfg.dataset_name,
        "config_path": str(config_path.resolve()),
        "timestamp_unix": ts,
        "timestamp_iso": iso,
        "step": "07_encode_ss_latents_per_part",
        "output_root": str(Path(cfg.reconstruction_dir) / "ss_latents_per_part"),
        "output_roots": {
            "overall": str(Path(cfg.reconstruction_dir) / "ss_latents_expanded"),
            "parts": str(Path(cfg.reconstruction_dir) / "ss_latents_per_part"),
        },
        "latent_contract": {
            "overall": {
                "format": "npz",
                "field": "mean",
                "dtype": "float32",
                "shape": list(EXPECTED_LATENT_SHAPE),
                "input_template": "reconstruction/voxel_expanded/{object_id}/angle_{X}/64/surface.npy",
                "path_template": "reconstruction/ss_latents_expanded/{object_id}/angle_{X}/latent.npz",
            },
            "part": {
                "format": "npy",
                "dtype": "float32",
                "shape": list(EXPECTED_LATENT_SHAPE),
                "input_template": "reconstruction/voxel_expanded/{object_id}/angle_{X}/64/ind_{part_name}.npy",
                "path_template": "reconstruction/ss_latents_per_part/{object_id}/angle_{X}/{part_name}.npy",
            },
        },
        "options": {
            "coverage": args.coverage,
            "latent_scope": args.latent_scope,
            "object_ids": args.object_ids,
            "object_list": args.object_list,
            "rank": args.rank,
            "world_size": args.world_size,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "enc_pretrained": args.enc_pretrained,
            "trellis_root": args.trellis_root,
        },
        "summary": {
            "passed": counters.failed == 0 and counters.extra_files == 0,
            "objects_seen": counters.objects_seen,
            "angles_seen": counters.angles_seen,
            "overall_seen": counters.overall_seen,
            "parts_seen": counters.parts_seen,
            "queued": counters.queued,
            "generated": counters.generated,
            "existing_valid": counters.existing_valid,
            "overall_queued": counters.overall_queued,
            "overall_generated": counters.overall_generated,
            "overall_existing_valid": counters.overall_existing_valid,
            "overall_failed": counters.overall_failed,
            "skipped_missing_voxel": counters.skipped_missing_voxel,
            "failed": counters.failed,
            "extra_files": counters.extra_files,
        },
        "records": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def print_summary(report: dict[str, Any], report_path: Path) -> None:
    s = report["summary"]
    print(
        "[ss-latent] "
        f"dataset={report['dataset']} objects={s['objects_seen']} angles={s['angles_seen']} "
        f"overall={s['overall_seen']} parts={s['parts_seen']} "
        f"generated={s['generated']} existing_valid={s['existing_valid']} "
        f"overall_generated={s['overall_generated']} overall_existing_valid={s['overall_existing_valid']} "
        f"skipped_missing_voxel={s['skipped_missing_voxel']} failed={s['failed']} "
        f"extra_files={s['extra_files']}"
    )
    print(f"[report] written to {report_path.resolve()}")
    print(f"[summary] passed={'true' if s['passed'] else 'false'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    cfg = load_config(str(config_path))
    args.enc_pretrained = args.enc_pretrained or cfg.trellis.ss_encoder
    args.trellis_root = args.trellis_root or cfg.trellis.root
    if cfg.voxel.resolution != EXPECTED_RESOLUTION:
        raise ValueError(
            f"SS encoder contract expects voxel resolution {EXPECTED_RESOLUTION}, got {cfg.voxel.resolution}"
        )

    object_ids = _resolve_object_ids(cfg, args)
    counters = Counters()
    records: list[dict[str, Any]] = []
    all_items = iter_expected_items(cfg, object_ids, args.coverage, args.latent_scope, counters, records)
    sharded_items = shard_items(all_items, args.rank, args.world_size)
    try:
        encode_items(sharded_items, cfg, args, counters, records)
    except Exception as exc:  # noqa: BLE001 - keep a machine-readable report on fatal stops
        counters.failed += 1
        records.append(
            {
                "status": "fatal",
                "object_id": None,
                "angle_idx": None,
                "part_name": None,
                "reason": repr(exc),
            }
        )

    if args.report_path:
        report_path = Path(args.report_path)
    else:
        report_path = Path(
            f"/tmp/ss_latents_per_part_{_dataset_slug(cfg.dataset_name)}_{int(time.time())}.json"
        )
    report = write_report(cfg, config_path, args, counters, records, report_path)
    print_summary(report, report_path)
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
