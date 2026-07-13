#!/usr/bin/env python3
"""Decode TRELLIS sparse-structure latents back to 64^3 voxel coordinates.

Inputs:
    reconstruction/ss_latents_expanded/<object_id>/angle_<i>/latent.npz['mean']
    reconstruction/ss_latents_per_part/<object_id>/angle_<i>/<part_name>.npy

Outputs:
    reconstruction/ss_latent_decoded/<object_id>/angle_<i>/64/overall.npy
    reconstruction/ss_latent_decoded/<object_id>/angle_<i>/64/parts/<part_name>.npy
    reconstruction/ss_latent_decoded/<object_id>/angle_<i>/64/metrics.json

The decoder mirrors TRELLIS sparse-structure VAE inference:
    logits = sparse_structure_decoder(z)
    coords = argwhere(logits > logit_threshold)

Coordinates are kept in the repository's voxel XYZ convention and are intended
for QC against voxel_expanded/surface.npy and ind_<part_name>.npy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import PipelineConfig, load_config, resolve_repo_path  # noqa: E402


EXPECTED_RESOLUTION = 64
EXPECTED_LATENT_SHAPE = (8, 16, 16, 16)
EXPECTED_COORD_DTYPE = np.int64
VALID_SCOPE = ("vlm-targets", "all-parts", "overall-only")
IMAGE_RE = re.compile(r"/renders/(?P<object_id>[^/]+)/angle_(?P<angle>\d+)/rgb/view_(?P<view>\d+)\.png(?:$|[?#])")


@dataclass(frozen=True)
class DecodeItem:
    object_id: str
    angle_idx: int
    kind: Literal["overall", "part"]
    latent_path: Path
    gt_path: Path
    out_path: Path
    part_name: str | None = None

    @property
    def label(self) -> str:
        return "overall" if self.kind == "overall" else str(self.part_name)


@dataclass
class Counters:
    items_seen: int = 0
    queued: int = 0
    generated: int = 0
    existing_valid: int = 0
    dry_run: int = 0
    missing_latent: int = 0
    missing_gt: int = 0
    invalid_latent: int = 0
    invalid_gt: int = 0
    invalid_existing: int = 0
    failed: int = 0
    skipped_duplicate: int = 0


@dataclass
class AngleMetrics:
    object_id: str
    angle_idx: int
    resolution: int
    decoder: str
    logit_threshold: float
    overall: dict[str, Any] | None = None
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "angle_idx": self.angle_idx,
            "angle_name": f"angle_{self.angle_idx}",
            "resolution": self.resolution,
            "decoder": self.decoder,
            "logit_threshold": self.logit_threshold,
            "overall": self.overall,
            "parts": self.parts,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode TRELLIS SS latents to voxel coords for QC and web preview."
    )
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument(
        "--scope",
        choices=VALID_SCOPE,
        default="all-parts",
        help=(
            "Decode scope. Default 'all-parts' decodes overall + every existing per-part latent "
            "and does not require the legacy 4-image VLM JSONL. 'vlm-targets' decodes only "
            "overall + target parts from that legacy JSONL; 'overall-only' decodes only overall latents."
        ),
    )
    parser.add_argument("--jsonl", help="Optional VLM JSONL path for --scope vlm-targets.")
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset, e.g. 100013,100712.")
    parser.add_argument("--object-list", help="Optional newline-delimited object ID file. Mutually exclusive with --object-ids.")
    parser.add_argument(
        "--dec-pretrained",
        help="Absolute or repo-relative TRELLIS sparse-structure decoder prefix/path. Default: trellis.ss_decoder from config.",
    )
    parser.add_argument(
        "--trellis-root",
        help="Absolute or repo-relative path containing the trellis Python package. Default: trellis.root from config.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for decoding. Use 'cuda' or 'cpu'.")
    parser.add_argument("--batch-size", type=int, default=1, help="Latents decoded per forward pass.")
    parser.add_argument("--rank", type=int, default=0, help="Shard rank for distributed/manual splitting.")
    parser.add_argument("--world-size", type=int, default=1, help="Shard count for distributed/manual splitting.")
    parser.add_argument(
        "--logit-threshold",
        type=float,
        default=0.0,
        help="Decoder logit threshold; TRELLIS sparse-structure inference uses logits > 0.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing decoded coords.")
    parser.add_argument("--dry-run", action="store_true", help="Enumerate and validate work without importing TRELLIS or writing decoded files.")
    parser.add_argument("--max-items", type=int, help="Optional cap for smoke tests after sharding.")
    parser.add_argument("--report-path", help="JSON report path (default: /tmp/ss_latent_decoded_<dataset>_<ts>.json).")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after per-item failures.")
    return parser.parse_args(argv)


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace(" ", "-")


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


def default_jsonl_path(cfg: PipelineConfig) -> Path:
    return Path(cfg.vlm_dir) / "training_json" / f"arts_mllm_{_dataset_slug(cfg.dataset_name)}.jsonl"


def _load_assistant_json(record: dict[str, Any], line_no: int) -> dict[str, Any]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"line {line_no}: conversations must be a list")
    for message in reversed(conversations):
        if not isinstance(message, dict):
            continue
        raw = message.get("value", message.get("content"))
        if not isinstance(raw, str) or "components" not in raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("components"), dict):
            return parsed
    raise ValueError(f"line {line_no}: assistant JSON with components not found")


def _extract_image_target(image_path: str, line_no: int) -> tuple[str, int]:
    match = IMAGE_RE.search(image_path)
    if not match:
        raise ValueError(f"line {line_no}: cannot parse render object/angle from image path: {image_path}")
    return match.group("object_id"), int(match.group("angle"))


def load_vlm_target_map(jsonl_path: Path, object_ids: set[str]) -> dict[tuple[str, int], set[str]]:
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"VLM JSONL not found: {jsonl_path}")
    target_map: dict[tuple[str, int], set[str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no}: JSONL row must be an object")
            images = record.get("images")
            if not isinstance(images, list) or len(images) != 4:
                # The default JSONL is the 4-image VLM training file; skip sidecar rows defensively.
                continue
            parsed_targets = [_extract_image_target(str(image), line_no) for image in images]
            object_angle_set = set(parsed_targets)
            if len(object_angle_set) != 1:
                raise ValueError(f"line {line_no}: images do not share one object/angle: {sorted(object_angle_set)}")
            object_id, angle_idx = parsed_targets[0]
            if object_id not in object_ids:
                continue
            assistant = _load_assistant_json(record, line_no)
            components = assistant["components"]
            target_keys = [
                key
                for key, component in components.items()
                if isinstance(component, dict) and component.get("parent") is not None
            ]
            if not target_keys:
                raise ValueError(f"line {line_no}: no target components in assistant JSON")
            target_map.setdefault((object_id, angle_idx), set()).update(target_keys)
    return target_map


def overall_paths(cfg: PipelineConfig, object_id: str, angle_idx: int) -> tuple[Path, Path, Path]:
    reconstruction = Path(cfg.reconstruction_dir)
    res = str(cfg.voxel.resolution)
    latent_path = reconstruction / "ss_latents_expanded" / object_id / f"angle_{angle_idx}" / "latent.npz"
    gt_path = reconstruction / "voxel_expanded" / object_id / f"angle_{angle_idx}" / res / "surface.npy"
    out_path = reconstruction / "ss_latent_decoded" / object_id / f"angle_{angle_idx}" / res / "overall.npy"
    return latent_path, gt_path, out_path


def part_paths(cfg: PipelineConfig, object_id: str, angle_idx: int, part_name: str) -> tuple[Path, Path, Path]:
    reconstruction = Path(cfg.reconstruction_dir)
    res = str(cfg.voxel.resolution)
    latent_path = reconstruction / "ss_latents_per_part" / object_id / f"angle_{angle_idx}" / f"{part_name}.npy"
    gt_path = reconstruction / "voxel_expanded" / object_id / f"angle_{angle_idx}" / res / f"ind_{part_name}.npy"
    out_path = reconstruction / "ss_latent_decoded" / object_id / f"angle_{angle_idx}" / res / "parts" / f"{part_name}.npy"
    return latent_path, gt_path, out_path


def add_overall_item(items: list[DecodeItem], cfg: PipelineConfig, object_id: str, angle_idx: int) -> None:
    latent_path, gt_path, out_path = overall_paths(cfg, object_id, angle_idx)
    items.append(DecodeItem(object_id, angle_idx, "overall", latent_path, gt_path, out_path))


def build_decode_items(cfg: PipelineConfig, object_ids: list[str], args: argparse.Namespace) -> list[DecodeItem]:
    items: list[DecodeItem] = []
    selected = set(object_ids)
    if args.scope == "vlm-targets":
        target_map = load_vlm_target_map(Path(args.jsonl) if args.jsonl else default_jsonl_path(cfg), selected)
        for (object_id, angle_idx), part_names in sorted(target_map.items(), key=lambda item: (item[0][0], item[0][1])):
            add_overall_item(items, cfg, object_id, angle_idx)
            for part_name in sorted(part_names):
                latent_path, gt_path, out_path = part_paths(cfg, object_id, angle_idx, part_name)
                items.append(DecodeItem(object_id, angle_idx, "part", latent_path, gt_path, out_path, part_name))
        return items

    for object_id in object_ids:
        for angle_idx in range(cfg.get_num_angles(object_id)):
            add_overall_item(items, cfg, object_id, angle_idx)
            if args.scope == "overall-only":
                continue
            latent_dir = Path(cfg.reconstruction_dir) / "ss_latents_per_part" / object_id / f"angle_{angle_idx}"
            if not latent_dir.is_dir():
                continue
            for latent_path in sorted(latent_dir.glob("*.npy")):
                part_name = latent_path.stem
                _, gt_path, out_path = part_paths(cfg, object_id, angle_idx, part_name)
                items.append(DecodeItem(object_id, angle_idx, "part", latent_path, gt_path, out_path, part_name))
    return items


def dedupe_items(items: Iterable[DecodeItem], counters: Counters, records: list[dict[str, Any]]) -> list[DecodeItem]:
    seen: set[tuple[str, int, str, str | None]] = set()
    result: list[DecodeItem] = []
    for item in items:
        key = (item.object_id, item.angle_idx, item.kind, item.part_name)
        if key in seen:
            counters.skipped_duplicate += 1
            records.append(record_for_item(item, "skipped_duplicate", reason="duplicate decode target"))
            continue
        seen.add(key)
        result.append(item)
    return result


def shard_items(items: list[DecodeItem], rank: int, world_size: int) -> list[DecodeItem]:
    if world_size < 1:
        raise ValueError("--world-size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("--rank must satisfy 0 <= rank < --world-size")
    start = len(items) * rank // world_size
    end = len(items) * (rank + 1) // world_size
    return items[start:end]


def validate_coords(path: Path, resolution: int) -> tuple[np.ndarray | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable: {exc!r}"
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None, f"shape {arr.shape} not (N,3)"
    if arr.shape[0] == 0:
        return arr.astype(EXPECTED_COORD_DTYPE), None
    if not np.issubdtype(arr.dtype, np.integer):
        return None, f"dtype {arr.dtype} is not integer"
    if int(arr.min()) < 0 or int(arr.max()) >= resolution:
        return None, f"coords out of [0,{resolution})"
    arr = np.unique(arr.astype(EXPECTED_COORD_DTYPE, copy=False), axis=0)
    return arr, None


def load_latent(path: Path) -> tuple[np.ndarray | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        if path.suffix == ".npz":
            with np.load(path) as pack:
                if "mean" not in pack:
                    return None, "npz missing 'mean'"
                arr = pack["mean"]
        else:
            arr = np.load(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable: {exc!r}"
    if tuple(arr.shape) != EXPECTED_LATENT_SHAPE:
        return None, f"shape {tuple(arr.shape)} != {EXPECTED_LATENT_SHAPE}"
    if not np.issubdtype(arr.dtype, np.floating):
        return None, f"dtype {arr.dtype} is not floating"
    if not bool(np.isfinite(arr).all()):
        return None, "contains NaN or Inf"
    return arr.astype(np.float32, copy=False), None


def coord_metrics(decoded: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    decoded_set = {tuple(row) for row in decoded.astype(int).tolist()}
    gt_set = {tuple(row) for row in gt.astype(int).tolist()}
    intersection = len(decoded_set & gt_set)
    union = len(decoded_set | gt_set)
    decoded_count = len(decoded_set)
    gt_count = len(gt_set)
    return {
        "gt_count": gt_count,
        "decoded_count": decoded_count,
        "intersection": intersection,
        "union": union,
        "iou": (intersection / union) if union else 1.0,
        "precision": (intersection / decoded_count) if decoded_count else (1.0 if gt_count == 0 else 0.0),
        "recall": (intersection / gt_count) if gt_count else 1.0,
        "false_positive": decoded_count - intersection,
        "false_negative": gt_count - intersection,
    }


def record_for_item(item: DecodeItem, status: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "status": status,
        "object_id": item.object_id,
        "angle_idx": item.angle_idx,
        "angle_name": f"angle_{item.angle_idx}",
        "kind": item.kind,
        "part_name": item.part_name,
        "latent_path": str(item.latent_path),
        "gt_path": str(item.gt_path),
        "output_path": str(item.out_path),
    }
    payload.update(extra)
    return payload


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
        trellis_pkg_dir = trellis_root / "trellis"
        if trellis_pkg_dir.is_dir() and "trellis" not in sys.modules:
            pkg = types.ModuleType("trellis")
            pkg.__path__ = [str(trellis_pkg_dir)]  # type: ignore[attr-defined]
            pkg.__file__ = str(trellis_pkg_dir / "__init__.py")
            sys.modules["trellis"] = pkg
        import trellis.models as models  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import trellis.models. Pass --trellis-root pointing to the TRELLIS checkout "
            "that contains trellis/models, and run in the official dataset_toolkits environment. "
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


def load_decoder(args: argparse.Namespace):
    torch, models = import_trellis_models(Path(args.trellis_root))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    dec_pretrained = require_trellis_checkpoint_prefix(args.dec_pretrained, "--dec-pretrained")
    decoder = models.from_pretrained(dec_pretrained).eval().to(device)
    return torch, decoder, device


def decode_batch(torch, decoder, device, latents: list[np.ndarray], logit_threshold: float) -> list[np.ndarray]:
    batch = torch.from_numpy(np.stack(latents, axis=0)).to(device=device, dtype=torch.float32)
    logits = decoder(batch)
    occupied = logits > logit_threshold
    outputs: list[np.ndarray] = []
    for batch_idx in range(occupied.shape[0]):
        coords = torch.argwhere(occupied[batch_idx, 0])
        if coords.numel() == 0:
            arr = np.empty((0, 3), dtype=EXPECTED_COORD_DTYPE)
        else:
            arr = coords.detach().cpu().numpy().astype(EXPECTED_COORD_DTYPE, copy=False)
            arr = np.unique(arr, axis=0)
        outputs.append(arr)
    return outputs


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, arr.astype(EXPECTED_COORD_DTYPE, copy=False))
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def angle_metrics_for(
    metrics_by_angle: dict[tuple[str, int], AngleMetrics],
    item: DecodeItem,
    args: argparse.Namespace,
    resolution: int,
) -> AngleMetrics:
    key = (item.object_id, item.angle_idx)
    if key not in metrics_by_angle:
        metrics_by_angle[key] = AngleMetrics(
            object_id=item.object_id,
            angle_idx=item.angle_idx,
            resolution=resolution,
            decoder=args.dec_pretrained,
            logit_threshold=args.logit_threshold,
        )
    return metrics_by_angle[key]


def store_metric(
    metrics_by_angle: dict[tuple[str, int], AngleMetrics],
    item: DecodeItem,
    args: argparse.Namespace,
    resolution: int,
    metric: dict[str, Any],
) -> None:
    metric = dict(metric)
    metric.update(
        {
            "latent_path": str(item.latent_path),
            "gt_path": str(item.gt_path),
            "decoded_path": str(item.out_path),
        }
    )
    angle_metrics = angle_metrics_for(metrics_by_angle, item, args, resolution)
    if item.kind == "overall":
        angle_metrics.overall = metric
    else:
        assert item.part_name is not None
        angle_metrics.parts[item.part_name] = metric


def validate_item_inputs(
    item: DecodeItem,
    counters: Counters,
    records: list[dict[str, Any]],
    resolution: int,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    latent, latent_reason = load_latent(item.latent_path)
    if latent_reason is not None or latent is None:
        if latent_reason == "missing":
            counters.missing_latent += 1
        else:
            counters.invalid_latent += 1
        records.append(record_for_item(item, "missing_latent" if latent_reason == "missing" else "invalid_latent", reason=latent_reason))
        return None, None, False

    gt, gt_reason = validate_coords(item.gt_path, resolution)
    if gt_reason is not None or gt is None:
        if gt_reason == "missing":
            counters.missing_gt += 1
        else:
            counters.invalid_gt += 1
        records.append(record_for_item(item, "missing_gt" if gt_reason == "missing" else "invalid_gt", reason=gt_reason))
        return None, None, False
    return latent, gt, True


def process_items(
    cfg: PipelineConfig,
    args: argparse.Namespace,
    items: list[DecodeItem],
    counters: Counters,
    records: list[dict[str, Any]],
) -> dict[tuple[str, int], AngleMetrics]:
    resolution = cfg.voxel.resolution
    if resolution != EXPECTED_RESOLUTION:
        raise ValueError(f"This decoder stage expects voxel.resolution={EXPECTED_RESOLUTION}, got {resolution}")

    metrics_by_angle: dict[tuple[str, int], AngleMetrics] = {}
    batch_latents: list[np.ndarray] = []
    batch_gt: list[np.ndarray] = []
    batch_items: list[DecodeItem] = []
    torch = decoder = device = None

    def handle_decoded(item: DecodeItem, decoded: np.ndarray, gt: np.ndarray, status: str) -> None:
        if decoded.ndim != 2 or decoded.shape[1] != 3:
            raise ValueError(f"decoded coords for {item.label} have invalid shape {decoded.shape}")
        if decoded.shape[0] and (int(decoded.min()) < 0 or int(decoded.max()) >= resolution):
            raise ValueError(f"decoded coords for {item.label} out of [0,{resolution})")
        decoded = np.unique(decoded.astype(EXPECTED_COORD_DTYPE, copy=False), axis=0)
        if status == "generated":
            atomic_save_npy(item.out_path, decoded)
            counters.generated += 1
        metric = coord_metrics(decoded, gt)
        store_metric(metrics_by_angle, item, args, resolution, metric)
        records.append(record_for_item(item, status, metrics=metric))

    class BatchDecodeError(RuntimeError):
        pass

    def fail_batch(exc: Exception) -> None:
        nonlocal batch_latents, batch_gt, batch_items
        failed_items = list(batch_items)
        reason = repr(exc)
        for failed_item in failed_items:
            counters.failed += 1
            records.append(record_for_item(failed_item, "failed", reason=reason))
        batch_latents = []
        batch_gt = []
        batch_items = []

    def flush_batch() -> None:
        nonlocal batch_latents, batch_gt, batch_items, torch, decoder, device
        if not batch_items:
            return
        try:
            if torch is None or decoder is None or device is None:
                torch, decoder, device = load_decoder(args)
            decoded_arrays = decode_batch(torch, decoder, device, batch_latents, args.logit_threshold)
        except Exception as exc:  # noqa: BLE001
            fail_batch(exc)
            if not args.continue_on_error:
                raise BatchDecodeError(repr(exc)) from exc
            return

        failed_exc: Exception | None = None
        for item, decoded, gt in zip(batch_items, decoded_arrays, batch_gt):
            try:
                handle_decoded(item, decoded, gt, "generated")
            except Exception as exc:  # noqa: BLE001
                counters.failed += 1
                records.append(record_for_item(item, "failed", reason=repr(exc)))
                if not args.continue_on_error:
                    failed_exc = exc
                    break
        batch_latents = []
        batch_gt = []
        batch_items = []
        if failed_exc is not None:
            raise BatchDecodeError(repr(failed_exc)) from failed_exc

    for item in tqdm(items, desc="Decoding SS latents"):
        counters.items_seen += 1
        try:
            latent, gt, ok = validate_item_inputs(item, counters, records, resolution)
            if not ok or latent is None or gt is None:
                if not args.continue_on_error and records[-1]["status"].startswith("invalid"):
                    raise ValueError(records[-1]["reason"])
                continue

            existing, existing_reason = validate_coords(item.out_path, resolution)
            if existing_reason is None and existing is not None and not args.overwrite:
                counters.existing_valid += 1
                handle_decoded(item, existing, gt, "existing_valid")
                continue
            if existing_reason is not None and item.out_path.exists():
                counters.invalid_existing += 1
                records.append(record_for_item(item, "invalid_existing", reason=existing_reason))

            if args.dry_run:
                counters.dry_run += 1
                records.append(record_for_item(item, "dry_run"))
                continue

            counters.queued += 1
            batch_latents.append(latent)
            batch_gt.append(gt)
            batch_items.append(item)
            if len(batch_items) >= args.batch_size:
                flush_batch()
        except BatchDecodeError:
            if not args.continue_on_error:
                raise
        except Exception as exc:  # noqa: BLE001
            counters.failed += 1
            records.append(record_for_item(item, "failed", reason=repr(exc)))
            batch_latents = []
            batch_gt = []
            batch_items = []
            if not args.continue_on_error:
                raise
    flush_batch()
    return metrics_by_angle


def write_metrics_files(
    cfg: PipelineConfig,
    metrics_by_angle: dict[tuple[str, int], AngleMetrics],
    dry_run: bool,
) -> None:
    if dry_run:
        return
    root = Path(cfg.reconstruction_dir) / "ss_latent_decoded"
    res = str(cfg.voxel.resolution)
    for (object_id, angle_idx), payload in metrics_by_angle.items():
        path = root / object_id / f"angle_{angle_idx}" / res / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(
    cfg: PipelineConfig,
    config_path: Path,
    args: argparse.Namespace,
    counters: Counters,
    records: list[dict[str, Any]],
    metrics_by_angle: dict[tuple[str, int], AngleMetrics],
    report_path: Path,
) -> dict[str, Any]:
    ts, iso = _utc_now()
    failure_count = (
        counters.missing_latent
        + counters.missing_gt
        + counters.invalid_latent
        + counters.invalid_gt
        + counters.invalid_existing
        + counters.failed
    )
    report = {
        "dataset": cfg.dataset_name,
        "config_path": str(config_path.resolve()),
        "timestamp_unix": ts,
        "timestamp_iso": iso,
        "step": "08_decode_ss_latents",
        "output_root": str(Path(cfg.reconstruction_dir) / "ss_latent_decoded"),
        "decoded_contract": {
            "dtype": "int64",
            "shape": ["N", 3],
            "coord_range": [0, cfg.voxel.resolution - 1],
            "overall_template": "reconstruction/ss_latent_decoded/{object_id}/angle_{X}/64/overall.npy",
            "part_template": "reconstruction/ss_latent_decoded/{object_id}/angle_{X}/64/parts/{part_name}.npy",
            "metrics_template": "reconstruction/ss_latent_decoded/{object_id}/angle_{X}/64/metrics.json",
        },
        "options": {
            "scope": args.scope,
            "jsonl": args.jsonl,
            "object_ids": args.object_ids,
            "object_list": args.object_list,
            "rank": args.rank,
            "world_size": args.world_size,
            "max_items": args.max_items,
            "device": args.device,
            "batch_size": args.batch_size,
            "dec_pretrained": args.dec_pretrained,
            "trellis_root": args.trellis_root,
            "logit_threshold": args.logit_threshold,
            "dry_run": args.dry_run,
            "overwrite": args.overwrite,
            "continue_on_error": args.continue_on_error,
        },
        "summary": {
            **counters.__dict__,
            "angles_with_metrics": len(metrics_by_angle),
            "passed": failure_count == 0,
        },
        "records": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    cfg = load_config(str(config_path))
    args.dec_pretrained = args.dec_pretrained or cfg.trellis.ss_decoder
    args.trellis_root = args.trellis_root or cfg.trellis.root
    object_ids = _resolve_object_ids(cfg, args)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    records: list[dict[str, Any]] = []
    counters = Counters()
    items = build_decode_items(cfg, object_ids, args)
    items = dedupe_items(items, counters, records)
    items = shard_items(items, args.rank, args.world_size)
    if args.max_items is not None:
        if args.max_items < 1:
            raise ValueError("--max-items must be >= 1")
        items = items[: args.max_items]

    metrics_by_angle = process_items(cfg, args, items, counters, records)
    write_metrics_files(cfg, metrics_by_angle, args.dry_run)

    report_path = Path(args.report_path) if args.report_path else Path(
        f"/tmp/ss_latent_decoded_{_dataset_slug(cfg.dataset_name)}_{int(time.time())}.json"
    )
    report = write_report(cfg, config_path, args, counters, records, metrics_by_angle, report_path)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
