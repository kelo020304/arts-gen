from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from part_ss_eval_platform.eval_crf_0615 import _sample_part_stats  # noqa: E402
from part_ss_eval_platform.eval_real_0615 import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_SLAT_FLOW_CKPT,
    DEFAULT_SLAT_MESH_DECODER_CKPT,
    DEFAULT_SS_DECODER_CKPT,
    _command,
    _execute,
    _load_coords,
    _run_dir,
    _summarize,
    load_data_config,
    part_bucket,
    render_preview_voxel,
    render_trellis_slat_mesh,
)
from trellis.datasets.arts.part_ss_latent_flow import PartSSLatentFlowDataset  # noqa: E402


DEFAULT_OUT_DIR = Path("/robot/data-lab/jzh/art-gen-output/EE-eval/0617-1")
DEFAULT_PART_SEG_CKPT = Path(
    "/robot/data-lab/jzh/art-gen-output/part_promptable_seg_full_S_0615-5/ckpts/step_50000.pt"
)
DEFAULT_SS_FLOW_CKPT = Path(
    "/robot/data-lab/jzh/art-gen-output/tre_mf_4view_multiflow_0611/ckpts/denoiser_ema0.999_step0020000.pt"
)
DEFAULT_SPLIT_JSON_0617 = Path(
    "/robot/data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_verse_realappliance_0511dd_v4.json"
)
SUMMARY_FIELDS = [
    "split",
    "bucket",
    "n",
    "mean_IoU",
    "success@IoU0.5",
    "peak_vram_mib",
]
METRIC_FIELDS = [
    "split",
    "obj_id",
    "angle",
    "part_name",
    "bucket",
    "raw_voxels",
    "pred_voxels",
    "IoU",
    "hit@0.5",
    "peak_vram_mib",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    data_root: Path
    manifest_paths: tuple[str, ...]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _sample_dict(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        return dict(sample)
    if is_dataclass(sample):
        return asdict(sample)
    return dict(vars(sample))


def _normalize_sample(sample: Any) -> SimpleNamespace:
    payload = _sample_dict(sample)
    original_split = str(payload.get("split", ""))
    split = "held" if original_split in {"val", "heldout"} else original_split
    bucket = str(payload.get("priority_bucket") or payload.get("sample_bucket") or payload.get("bucket") or "")
    dataset_id = str(payload.get("dataset_id", ""))
    payload.update(
        {
            "split": split,
            "original_split": original_split,
            "bucket": bucket,
            "dataset_id": dataset_id,
            "obj_id": str(payload["obj_id"]),
            "angle_idx": int(payload["angle_idx"]),
        }
    )
    return SimpleNamespace(**payload)


def _sample_row(sample: SimpleNamespace) -> dict[str, Any]:
    return {
        "split": sample.split,
        "dataset_id": getattr(sample, "dataset_id", ""),
        "object_key": _object_key(sample),
        "obj_id": sample.obj_id,
        "angle_idx": int(sample.angle_idx),
        "data_root": getattr(sample, "data_root", ""),
        "manifest_path": getattr(sample, "manifest_path", ""),
        "bucket": getattr(sample, "bucket", ""),
        "sample_bucket": getattr(sample, "sample_bucket", ""),
        "priority_bucket": getattr(sample, "priority_bucket", ""),
        "part_count": int(getattr(sample, "part_count", 0)),
        "min_raw_voxels": int(getattr(sample, "min_raw_voxels", 0)),
        "max_raw_voxels": int(getattr(sample, "max_raw_voxels", 0)),
        "has_button": bool(getattr(sample, "has_button", False)),
        "has_large_keyword": bool(getattr(sample, "has_large_keyword", False)),
        "selected_reason": getattr(sample, "selected_reason", ""),
        "original_split": getattr(sample, "original_split", sample.split),
    }


def _samples_from_manifest(manifest: dict[str, Any]) -> list[SimpleNamespace]:
    rows: list[SimpleNamespace] = []
    for split in ("train", "held"):
        for item in manifest.get("samples", {}).get(split, []):
            rows.append(_normalize_sample({**item, "split": split}))
    if not rows:
        raise ValueError("selection manifest has no samples")
    return rows


def _selection_manifest(samples: list[SimpleNamespace], args: argparse.Namespace, source: dict[str, Any]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "held": []}
    for sample in samples:
        by_split[sample.split].append(_sample_row(sample))
    counts = {
        split: {
            "objects": len({row["object_key"] for row in rows}),
            "samples": len(rows),
            "dataset_id": dict(Counter(row["dataset_id"] for row in rows)),
            "sample_bucket": dict(Counter(row["sample_bucket"] for row in rows)),
            "priority_bucket": dict(Counter(row["priority_bucket"] for row in rows)),
            "has_button": sum(int(row["has_button"]) for row in rows),
            "has_large_keyword": sum(int(row["has_large_keyword"]) for row in rows),
        }
        for split, rows in by_split.items()
    }
    return {
        "name": "0617-1",
        "split_json": str(args.split_json),
        "train_count": int(args.train_count),
        "held_count": int(args.held_count),
        "unique_obj": True,
        "one_angle_per_obj": True,
        "selection_policy": source.get("selection_policy", ""),
        "source_counts": source.get("counts", {}),
        "datasets": source.get("datasets", []),
        "counts": counts,
        "samples": by_split,
    }


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _object_key(sample: Any) -> str:
    dataset_id = str(getattr(sample, "dataset_id", "") if not isinstance(sample, dict) else sample.get("dataset_id", ""))
    obj_id = str(getattr(sample, "obj_id", "") if not isinstance(sample, dict) else sample.get("obj_id", ""))
    return f"{dataset_id}::{obj_id}" if dataset_id else obj_id


def _path_obj_id(sample: Any) -> str:
    return _object_key(sample).replace("::", "__").replace("/", "_")


def _sample_key(sample: SimpleNamespace) -> tuple[str, str, str, int]:
    return (sample.split, str(getattr(sample, "dataset_id", "")), sample.obj_id, int(sample.angle_idx))


def _run_dir_for_sample(out_dir: Path, sample: SimpleNamespace) -> Path:
    return _run_dir(out_dir, sample.split, sample.obj_id, int(sample.angle_idx))


def _result_png_paths_for_sample(
    out_dir: Path,
    sample: SimpleNamespace,
    duplicate_counts: dict[tuple[str, str], int],
) -> tuple[Path, Path]:
    obj_dir = out_dir / sample.split / sample.obj_id
    if duplicate_counts[(sample.split, sample.obj_id)] > 1:
        stem = f"result_angle_{int(sample.angle_idx)}"
    else:
        stem = "result"
    return obj_dir / f"{stem}_voxel.png", obj_dir / f"{stem}_mesh.png"


def _split_ref_key(ref: Any) -> str:
    if isinstance(ref, dict):
        if "object_key" in ref:
            return str(ref["object_key"])
        dataset_id = str(ref.get("dataset_id", ""))
        obj_id = str(ref.get("obj_id", ref.get("object_id", "")))
        return f"{dataset_id}::{obj_id}" if dataset_id else obj_id
    return str(ref)


def _dataset_specs_from_split(split_data: dict[str, Any]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    raw_specs = split_data.get("datasets")
    if isinstance(raw_specs, list) and raw_specs:
        for item in raw_specs:
            data_root = Path(str(item["data_root"]))
            raw_manifest = item.get("manifest_paths", item.get("manifest_path"))
            if isinstance(raw_manifest, list):
                manifest_paths = tuple(str(path) for path in raw_manifest)
            elif raw_manifest:
                manifest_paths = (str(raw_manifest),)
            else:
                raise KeyError(f"split dataset entry missing manifest_path(s): {item}")
            specs.append(
                DatasetSpec(
                    dataset_id=str(item.get("dataset_id") or data_root.name),
                    data_root=data_root,
                    manifest_paths=manifest_paths,
                )
            )
        return specs
    data_root = Path(str(split_data["data_root"]))
    return [
        DatasetSpec(
            dataset_id=str(split_data.get("dataset_id") or ""),
            data_root=data_root,
            manifest_paths=(str(split_data["manifest_path"]),),
        )
    ]


def _base_data_config(data_config_path: Path) -> dict[str, Any]:
    cfg = load_data_config(Path(data_config_path))
    cfg.update(
        {
            "allow_missing_masks": False,
            "require_part_token": False,
            "use_mask_overlap_pooling": False,
            "filter_zero_mask_coverage": False,
        }
    )
    return cfg


def _make_dataset(spec: DatasetSpec, base_config: dict[str, Any]) -> PartSSLatentFlowDataset:
    manifest_paths = list(spec.manifest_paths)
    if not manifest_paths:
        raise ValueError(f"dataset {spec.dataset_id} has no manifests")
    cfg = dict(base_config)
    cfg["data_root"] = str(spec.data_root)
    cfg["manifest_path"] = str(manifest_paths[0])
    ds = PartSSLatentFlowDataset(cfg)
    for sample in ds.samples:
        sample["_eval_dataset_id"] = spec.dataset_id
        sample["_eval_data_root"] = str(spec.data_root)
        sample["_eval_manifest_path"] = str(manifest_paths[0])
    for extra_manifest in manifest_paths[1:]:
        extra_cfg = dict(cfg)
        extra_cfg["manifest_path"] = str(extra_manifest)
        extra_ds = PartSSLatentFlowDataset(extra_cfg)
        for sample in extra_ds.samples:
            sample["_eval_dataset_id"] = spec.dataset_id
            sample["_eval_data_root"] = str(spec.data_root)
            sample["_eval_manifest_path"] = str(extra_manifest)
        ds.samples.extend(extra_ds.samples)
        if hasattr(ds, "loads") and hasattr(extra_ds, "loads"):
            ds.loads.extend(extra_ds.loads)
    return ds


def _load_datasets(args: argparse.Namespace) -> tuple[dict[str, PartSSLatentFlowDataset], list[dict[str, str]]]:
    split_data = _read_json(Path(args.split_json))
    specs = _dataset_specs_from_split(split_data)
    base_config = _base_data_config(Path(args.data_config))
    datasets = {spec.dataset_id: _make_dataset(spec, base_config) for spec in specs}
    dataset_meta = [
        {
            "dataset_id": spec.dataset_id,
            "data_root": str(spec.data_root),
            "manifest_paths": [str(path) for path in spec.manifest_paths],
        }
        for spec in specs
    ]
    return datasets, dataset_meta


def _all_dataset_samples(datasets: dict[str, PartSSLatentFlowDataset]) -> list[tuple[str, PartSSLatentFlowDataset, dict[str, Any]]]:
    rows = []
    for dataset_id, ds in datasets.items():
        for sample in ds.samples:
            rows.append((dataset_id, ds, sample))
    return rows


def _select_one_angle_per_object(
    datasets: dict[str, PartSSLatentFlowDataset],
    split_json: Path,
    *,
    train_count: int,
    held_count: int,
    dataset_meta: list[dict[str, str]],
) -> tuple[list[SimpleNamespace], dict[str, Any]]:
    split_data = _read_json(split_json)
    split_ids = {
        "train": {_split_ref_key(x) for x in split_data["train_ids"]},
        "held": {_split_ref_key(x) for x in split_data["heldout_ids"]},
    }
    by_obj: dict[str, dict[str, list[dict[str, Any]]]] = {"train": defaultdict(list), "held": defaultdict(list)}
    for dataset_id, ds, sample in _all_dataset_samples(datasets):
        key = f"{dataset_id}::{sample['obj_id']}" if dataset_id else str(sample["obj_id"])
        split = "train" if key in split_ids["train"] else "held" if key in split_ids["held"] else ""
        if not split:
            continue
        stats = _sample_part_stats(ds, sample)
        by_obj[split][key].append(
            {
                "split": split,
                "dataset_id": dataset_id,
                "object_key": key,
                "obj_id": str(sample["obj_id"]),
                "angle_idx": int(sample["angle_idx"]),
                "data_root": str(sample.get("_eval_data_root") or ds.data_root),
                "manifest_path": str(sample.get("_eval_manifest_path") or ds.manifest_path),
                "stats": stats,
            }
        )

    def best_angle(records: list[dict[str, Any]]) -> dict[str, Any]:
        def score(rec: dict[str, Any]) -> tuple[int, int, int, int, int]:
            st = rec["stats"]
            return (
                int(st["has_large_keyword"]),
                int(st["part_count"]),
                int(st["keyword_score"]),
                int(st["max_raw_voxels"]),
                -int(rec["angle_idx"]),
            )

        return max(records, key=score)

    obj_records: dict[str, list[dict[str, Any]]] = {"train": [], "held": []}
    for split in ("train", "held"):
        for records in by_obj[split].values():
            obj_records[split].append(best_angle(records))

    quotas = {
        "train": {"large": 36, "medium": 16, "small": 12, "tiny": 8, "button": 13},
        "held": {"large": 18, "medium": 8, "small": 6, "tiny": 5, "button": 6},
    }
    targets = {"train": int(train_count), "held": int(held_count)}
    selected: dict[str, list[dict[str, Any]]] = {"train": [], "held": []}
    selected_keys: dict[str, set[str]] = {"train": set(), "held": set()}
    notes: list[str] = []

    def priority_bucket(record: dict[str, Any]) -> str:
        return str(record["stats"]["priority_bucket"])

    def record_sort(record: dict[str, Any]) -> tuple[int, int, int, int, str]:
        st = record["stats"]
        return (
            -int(st["has_large_keyword"]),
            -int(st["part_count"]),
            -int(st["keyword_score"]),
            -int(st["max_raw_voxels"]),
            str(record["object_key"]),
        )

    for split in ("train", "held"):
        records = sorted(obj_records[split], key=record_sort)
        for bucket in ("large", "medium", "small", "tiny", "button"):
            need = int(quotas[split].get(bucket, 0))
            for rec in [item for item in records if priority_bucket(item) == bucket]:
                if len(selected[split]) >= targets[split] or need <= 0:
                    break
                if rec["object_key"] in selected_keys[split]:
                    continue
                selected[split].append({**rec, "selected_reason": f"quota:{bucket}"})
                selected_keys[split].add(str(rec["object_key"]))
                need -= 1
            if need > 0:
                notes.append(f"{split}/{bucket} quota short by {need}")
        remaining = [rec for rec in records if str(rec["object_key"]) not in selected_keys[split]]
        for rec in remaining:
            if len(selected[split]) >= targets[split]:
                break
            selected[split].append({**rec, "selected_reason": "fill:large_multipart_priority"})
            selected_keys[split].add(str(rec["object_key"]))
        if len(selected[split]) < targets[split]:
            notes.append(f"{split} selected {len(selected[split])}/{targets[split]}")

    rows: list[SimpleNamespace] = []
    for split in ("train", "held"):
        selected[split].sort(key=lambda rec: (str(rec["dataset_id"]), priority_bucket(rec), str(rec["obj_id"])))
        for rec in selected[split]:
            st = rec["stats"]
            rows.append(
                _normalize_sample(
                    {
                        "split": split,
                        "dataset_id": rec["dataset_id"],
                        "obj_id": rec["obj_id"],
                        "angle_idx": int(rec["angle_idx"]),
                        "data_root": rec["data_root"],
                        "manifest_path": rec["manifest_path"],
                        "sample_bucket": str(st["sample_bucket"]),
                        "priority_bucket": str(st["priority_bucket"]),
                        "part_count": int(st["part_count"]),
                        "min_raw_voxels": int(st["min_raw_voxels"]),
                        "max_raw_voxels": int(st["max_raw_voxels"]),
                        "has_button": bool(st["has_button"]),
                        "has_large_keyword": bool(st["has_large_keyword"]),
                        "selected_reason": str(rec.get("selected_reason", "")),
                    }
                )
            )
    manifest = {
        "split_json": str(split_json),
        "train_count": int(train_count),
        "held_count": int(held_count),
        "selection_policy": (
            "object-level dedup over dataset_id::obj_id; choose one angle per object by large-part keyword, "
            "part_count, keyword score, max raw voxels; quotas prioritize large/multipart objects while retaining tiny/button cases"
        ),
        "notes": notes,
        "datasets": dataset_meta,
        "counts": {
            split: {
                "objects": len(selected[split]),
                "dataset_id": dict(Counter(str(item["dataset_id"]) for item in selected[split])),
                "priority_bucket": dict(Counter(priority_bucket(item) for item in selected[split])),
                "sample_bucket": dict(Counter(str(item["stats"]["sample_bucket"]) for item in selected[split])),
            }
            for split in ("train", "held")
        },
    }
    return rows, manifest


def load_or_build_selection(args: argparse.Namespace) -> list[SimpleNamespace]:
    out_dir = Path(args.out_dir)
    selection_path = out_dir / "selection.json"
    if selection_path.is_file() and not args.overwrite_selection:
        return _samples_from_manifest(json.loads(selection_path.read_text(encoding="utf-8")))

    datasets, dataset_meta = _load_datasets(args)
    samples, source_manifest = _select_one_angle_per_object(
        datasets,
        Path(args.split_json),
        train_count=int(args.train_count),
        held_count=int(args.held_count),
        dataset_meta=dataset_meta,
    )
    if len({_object_key(sample) for sample in samples}) != len(samples):
        raise RuntimeError("selection is not one-angle-per-object unique")
    manifest = _selection_manifest(samples, args, source_manifest)
    _write_json(selection_path, manifest)
    _write_csv(
        out_dir / "selection.csv",
        [_sample_row(sample) for sample in samples],
        [
            "split",
            "dataset_id",
            "object_key",
            "obj_id",
            "angle_idx",
            "data_root",
            "manifest_path",
            "bucket",
            "sample_bucket",
            "priority_bucket",
            "part_count",
            "min_raw_voxels",
            "max_raw_voxels",
            "has_button",
            "has_large_keyword",
            "selected_reason",
            "original_split",
        ],
    )
    return samples


def _find_dataset_sample(ds: Any, sample: SimpleNamespace) -> dict[str, Any]:
    for item in ds.samples:
        if str(item["obj_id"]) == sample.obj_id and int(item["angle_idx"]) == int(sample.angle_idx):
            return item
    raise KeyError(f"dataset sample not found: {sample.obj_id} angle={sample.angle_idx}")


def _dataset_for_sample(datasets: dict[str, PartSSLatentFlowDataset], sample: SimpleNamespace) -> PartSSLatentFlowDataset:
    dataset_id = str(getattr(sample, "dataset_id", ""))
    if dataset_id in datasets:
        return datasets[dataset_id]
    if not dataset_id and len(datasets) == 1:
        return next(iter(datasets.values()))
    raise KeyError(f"dataset_id={dataset_id!r} not found; available={sorted(datasets)}")


def _sample_data_config_path(out_dir: Path, sample: SimpleNamespace, ds: PartSSLatentFlowDataset) -> Path:
    safe_dataset = str(getattr(sample, "dataset_id", "") or "dataset").replace("/", "_")
    path = out_dir / "_data_configs" / safe_dataset / f"{sample.obj_id}-{int(sample.angle_idx)}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = str(getattr(sample, "manifest_path", "") or ds.manifest_path)
    data_root = str(getattr(sample, "data_root", "") or ds.data_root)
    text = "\n".join(
        [
            "stage: part_ss_latent_flow_eval_0617_1",
            "data:",
            f"  data_root: {data_root}",
            f"  recon_subdir: {ds.recon_subdir}",
            f"  mask_subdir: {ds.mask_subdir}",
            f"  manifest_path: {manifest_path}",
            "  num_views: 4",
            "  allow_missing_masks: false",
            "  require_part_token: false",
            "  use_mask_overlap_pooling: false",
            "  filter_zero_mask_coverage: false",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def _command_for_sample(out_dir: Path, sample: SimpleNamespace, args: argparse.Namespace, stage: str, ds: PartSSLatentFlowDataset):
    local_args = argparse.Namespace(**vars(args))
    local_args.data_config = str(_sample_data_config_path(out_dir, sample, ds))
    return _command(out_dir, sample, local_args, stage)


def _parts_complete(run_dir: Path, expected_parts: int) -> bool:
    parts = sorted((run_dir / "parts").glob("part_*_voxel.npz"))
    return len(parts) >= int(expected_parts) and int(expected_parts) > 0


def _stage_record(stage: str, sample: SimpleNamespace, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "split": sample.split,
        "dataset_id": getattr(sample, "dataset_id", ""),
        "object_key": _object_key(sample),
        "obj_id": sample.obj_id,
        "angle": int(sample.angle_idx),
        **extra,
    }


def _write_progress(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_stage_if_needed(
    out_dir: Path,
    sample: SimpleNamespace,
    args: argparse.Namespace,
    stage: str,
    progress_path: Path,
    records: list[dict[str, Any]],
    expected_parts: int,
    ds: PartSSLatentFlowDataset,
) -> int:
    run_dir = _run_dir_for_sample(out_dir, sample)
    if stage == "ss":
        done = (run_dir / "ss_latent.npy").is_file() and (run_dir / "voxel.npz").is_file()
    elif stage == "part":
        done = _parts_complete(run_dir, expected_parts)
    else:
        raise ValueError(f"unsupported stage: {stage}")
    if done and not args.force_stage:
        rec = _stage_record(stage, sample, "skipped", run_dir=str(run_dir))
        records.append(rec)
        _write_progress(progress_path, rec)
        return 0
    spec = _command_for_sample(out_dir, sample, args, stage, ds)
    rec = _execute(
        spec,
        gpu=str(args.gpu),
        progress_path=progress_path,
        label=f"B/{stage}/{sample.split}/{sample.obj_id}/{int(sample.angle_idx)}",
    )
    rec.update({"status": "done", "split": sample.split, "obj_id": sample.obj_id, "angle": int(sample.angle_idx)})
    records.append(rec)
    return int(rec.get("peak_vram_mib", 0))


def _render_if_needed(
    ds: Any,
    out_dir: Path,
    sample: SimpleNamespace,
    args: argparse.Namespace,
    duplicate_counts: dict[tuple[str, str], int],
    progress_path: Path,
    records: list[dict[str, Any]],
) -> int:
    run_dir = _run_dir_for_sample(out_dir, sample)
    ds_sample = _find_dataset_sample(ds, sample)
    part_items: list[tuple[str, np.ndarray]] = []
    for part_idx, part in enumerate(ds_sample["parts"]):
        path = run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz"
        if path.is_file():
            part_items.append((str(part["part_name"]), _load_coords(path)))
    whole_coords = _load_coords(run_dir / "voxel.npz") if (run_dir / "voxel.npz").is_file() else np.empty((0, 3), dtype=np.int64)
    voxel_path, mesh_path = _result_png_paths_for_sample(out_dir, sample, duplicate_counts)

    peak = 0
    if voxel_path.is_file() and not args.force_render:
        rec = _stage_record("voxel_png", sample, "skipped", path=str(voxel_path))
        records.append(rec)
        _write_progress(progress_path, rec)
    else:
        render_preview_voxel(whole_coords, part_items, voxel_path, sample.obj_id, int(sample.angle_idx))
        rec = _stage_record("voxel_png", sample, "done", path=str(voxel_path))
        records.append(rec)
        _write_progress(progress_path, rec)

    if mesh_path.is_file() and not args.force_render:
        rec = _stage_record("trellis_slat_mesh", sample, "skipped", path=str(mesh_path))
        records.append(rec)
        _write_progress(progress_path, rec)
        return peak
    try:
        mesh_rec = render_trellis_slat_mesh(
            ds,
            ds_sample,
            whole_coords,
            part_items,
            mesh_path,
            sample.obj_id,
            int(sample.angle_idx),
            args,
            progress_path=progress_path,
        )
        mesh_rec.update({"status": "done", "split": sample.split, "obj_id": sample.obj_id, "angle": int(sample.angle_idx)})
        records.append(mesh_rec)
        peak = max(peak, int(mesh_rec.get("peak_vram_mib", 0)))
    except Exception as exc:
        rec = _stage_record(
            "trellis_slat_mesh",
            sample,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            path=str(mesh_path),
        )
        records.append(rec)
        _write_progress(progress_path, rec)
        print(
            f"[eval_0617_1] mesh failed {sample.split} {sample.obj_id} "
            f"angle={int(sample.angle_idx)}: {type(exc).__name__}: {exc}",
            flush=True,
        )
    return peak


def _shard_samples(samples: list[SimpleNamespace], shard_id: int, shard_count: int) -> list[tuple[int, SimpleNamespace]]:
    return [(idx, sample) for idx, sample in enumerate(samples, 1) if (idx - 1) % int(shard_count) == int(shard_id)]


def _coords_iou(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.int64).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.int64).reshape(-1, 3)
    pred_set = set(map(tuple, pred.tolist()))
    gt_set = set(map(tuple, gt.tolist()))
    inter = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return {
        "IoU": float(inter / union) if union else 1.0,
        "pred_voxels": int(len(pred_set)),
        "raw_voxels": int(len(gt_set)),
    }


def collect_multi_metrics(
    datasets: dict[str, PartSSLatentFlowDataset],
    selected: list[SimpleNamespace],
    out_dir: Path,
    peak_vram: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in selected:
        ds = _dataset_for_sample(datasets, sample)
        ds_sample = _find_dataset_sample(ds, sample)
        run_dir = _run_dir_for_sample(out_dir, sample)
        for part_idx, part in enumerate(ds_sample["parts"]):
            part_name = str(part["part_name"])
            pred_path = run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz"
            pred = _load_coords(pred_path) if pred_path.is_file() else np.empty((0, 3), dtype=np.int64)
            raw = ds._load_raw_ind_coords(ds_sample, part).numpy().astype(np.int64)
            metric = _coords_iou(pred, raw)
            iou = float(metric["IoU"])
            raw_count = int(metric["raw_voxels"])
            rows.append(
                {
                    "split": sample.split,
                    "obj_id": sample.obj_id,
                    "angle": int(sample.angle_idx),
                    "part_name": part_name,
                    "bucket": part_bucket(part_name, part, raw_count),
                    "raw_voxels": raw_count,
                    "pred_voxels": int(metric["pred_voxels"]),
                    "IoU": iou,
                    "hit@0.5": int(iou >= 0.5),
                    "peak_vram_mib": int(peak_vram),
                }
            )
    return rows


def _checkpoint_meta(args: argparse.Namespace) -> dict[str, str]:
    return {
        "ss_flow_ckpt": str(args.ss_flow_ckpt),
        "part_seg_ckpt": str(args.part_seg_ckpt),
        "ss_decoder_ckpt": str(args.ss_decoder_ckpt),
        "slat_flow_ckpt": str(args.slat_flow_ckpt),
        "slat_mesh_decoder_ckpt": str(args.slat_mesh_decoder_ckpt),
        "slat_token_source": str(args.slat_token_source),
    }


def run_shard(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_or_build_selection(args)
    selected = _shard_samples(samples, int(args.shard_id), int(args.shard_count))
    progress_path = out_dir / f"progress_shard_{int(args.shard_id):02d}.jsonl"
    datasets, dataset_meta = _load_datasets(args)
    duplicate_counts: dict[tuple[str, str], int] = defaultdict(int)
    for sample in samples:
        duplicate_counts[(sample.split, sample.obj_id)] += 1

    records: list[dict[str, Any]] = []
    peak_vram = 0
    _write_json(
        out_dir / f"run_meta_shard_{int(args.shard_id):02d}.json",
        {
            "out_dir": str(out_dir),
            "shard_id": int(args.shard_id),
            "shard_count": int(args.shard_count),
            "gpu": str(args.gpu),
            "num_samples": len(selected),
            "datasets": dataset_meta,
            "checkpoints": _checkpoint_meta(args),
        },
    )
    for local_idx, (global_idx, sample) in enumerate(selected, 1):
        ds = _dataset_for_sample(datasets, sample)
        ds_sample = _find_dataset_sample(ds, sample)
        print(
            f"[eval_0617_1] shard {args.shard_id}/{args.shard_count} "
            f"{local_idx}/{len(selected)} global={global_idx}/{len(samples)} "
            f"{sample.split} {getattr(sample, 'dataset_id', '')}::{sample.obj_id} angle={int(sample.angle_idx)} parts={len(ds_sample['parts'])}",
            flush=True,
        )
        _write_progress(
            progress_path,
            _stage_record(
                "sample",
                sample,
                "started",
                global_idx=global_idx,
                local_idx=local_idx,
                total=len(selected),
            ),
        )
        peak_vram = max(
            peak_vram,
            _run_stage_if_needed(out_dir, sample, args, "ss", progress_path, records, len(ds_sample["parts"]), ds),
        )
        peak_vram = max(
            peak_vram,
            _run_stage_if_needed(out_dir, sample, args, "part", progress_path, records, len(ds_sample["parts"]), ds),
        )
        peak_vram = max(
            peak_vram,
            _render_if_needed(ds, out_dir, sample, args, duplicate_counts, progress_path, records),
        )
        _write_progress(
            progress_path,
            _stage_record(
                "sample",
                sample,
                "done",
                global_idx=global_idx,
                local_idx=local_idx,
                total=len(selected),
                peak_vram_mib=peak_vram,
            ),
        )

    shard_samples = [sample for _idx, sample in selected]
    metrics = collect_multi_metrics(datasets, shard_samples, out_dir, peak_vram)
    _write_csv(out_dir / f"metrics_shard_{int(args.shard_id):02d}.csv", metrics, METRIC_FIELDS)
    _write_json(out_dir / f"metrics_shard_{int(args.shard_id):02d}.json", metrics)
    _write_json(out_dir / f"records_shard_{int(args.shard_id):02d}.json", records)
    (out_dir / f"peak_vram_shard_{int(args.shard_id):02d}.txt").write_text(f"{peak_vram}\n", encoding="utf-8")
    summarize(args, quiet=False)
    print(f"[eval_0617_1] shard {args.shard_id} done -> {out_dir}", flush=True)
    return 0


def summarize(args: argparse.Namespace, *, quiet: bool = False) -> int:
    out_dir = Path(args.out_dir)
    samples = load_or_build_selection(args)
    datasets, _dataset_meta = _load_datasets(args)
    duplicate_counts: dict[tuple[str, str], int] = defaultdict(int)
    for sample in samples:
        duplicate_counts[(sample.split, sample.obj_id)] += 1
    peak_values = []
    for path in out_dir.glob("peak_vram_shard_*.txt"):
        try:
            peak_values.append(int(path.read_text(encoding="utf-8").strip()))
        except ValueError:
            pass
    progress_records = []
    for path in sorted(out_dir.glob("progress_shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "peak_vram_mib" in rec:
                peak_values.append(int(rec["peak_vram_mib"]))
            progress_records.append(rec)
    peak_vram = max(peak_values) if peak_values else 0
    artifact_done = set()
    mesh_done = set()
    for sample in samples:
        run_dir = _run_dir_for_sample(out_dir, sample)
        ds_sample = _find_dataset_sample(_dataset_for_sample(datasets, sample), sample)
        voxel_path, mesh_path = _result_png_paths_for_sample(out_dir, sample, duplicate_counts)
        key = _sample_key(sample)
        if mesh_path.is_file():
            mesh_done.add(key)
        if (
            (run_dir / "voxel.npz").is_file()
            and _parts_complete(run_dir, len(ds_sample["parts"]))
            and voxel_path.is_file()
            and mesh_path.is_file()
        ):
            artifact_done.add(key)
    metrics_samples = [
        sample
        for sample in samples
        if _sample_key(sample) in artifact_done
    ]
    metrics = collect_multi_metrics(datasets, metrics_samples, out_dir, peak_vram)
    summary = _summarize(metrics, peak_vram)
    completed_objects = {
        (rec.get("split"), rec.get("dataset_id", ""), rec.get("obj_id"), int(rec.get("angle", -1)))
        for rec in progress_records
        if rec.get("stage") == "sample" and rec.get("status") == "done"
    }
    status = {
        "total_samples": len(samples),
        "sample_done": max(len(completed_objects), len(artifact_done)),
        "mesh_done": len(mesh_done),
        "artifact_done": len(artifact_done),
        "metrics_rows": len(metrics),
        "peak_vram_mib": peak_vram,
    }
    _write_csv(out_dir / "metrics.csv", metrics, METRIC_FIELDS)
    _write_json(out_dir / "metrics.json", metrics)
    _write_csv(out_dir / "metrics_summary.csv", summary, SUMMARY_FIELDS)
    _write_json(out_dir / "metrics_summary.json", summary)
    _write_json(out_dir / "status.json", status)
    (out_dir / "peak_vram.txt").write_text(f"{peak_vram}\n", encoding="utf-8")
    if not quiet:
        print(f"[eval_0617_1] summary {status}", flush=True)
    return 0


def prepare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_or_build_selection(args)
    _datasets, dataset_meta = _load_datasets(args)
    _write_json(
        out_dir / "run_meta.json",
        {
            "name": "0617-1",
            "out_dir": str(out_dir),
            "num_samples": len(samples),
            "datasets": dataset_meta,
            "checkpoints": _checkpoint_meta(args),
            "ss_flow_rule": "TRELLIS multiflow from 4-view reconstruction/dinov2_tokens, velocity mean per step",
            "part_stage": "promptable_seg route=voxel, latest 0615-5 S checkpoint",
            "slat_stage": "TRELLIS SLat flow + TRELLIS mesh decoder; default condition is live TRELLIS RGBA tokens",
        },
    )
    print(f"[eval_0617_1] prepared {len(samples)} samples -> {out_dir}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="0617-1 EE eval: multiflow SS, latest promptable seg, TRELLIS SLat.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    p.add_argument("--split-json", default=str(DEFAULT_SPLIT_JSON_0617))
    p.add_argument("--part-seg-ckpt", default=str(DEFAULT_PART_SEG_CKPT))
    p.add_argument("--ss-flow-ckpt", default=str(DEFAULT_SS_FLOW_CKPT))
    p.add_argument("--ss-decoder-ckpt", default=str(DEFAULT_SS_DECODER_CKPT))
    p.add_argument("--slat-flow-ckpt", default=str(DEFAULT_SLAT_FLOW_CKPT))
    p.add_argument("--slat-mesh-decoder-ckpt", default=str(DEFAULT_SLAT_MESH_DECODER_CKPT))
    p.add_argument(
        "--slat-token-source",
        choices=("live", "cache"),
        default="live",
        help="SLat flow condition source. live matches accepted TRELLIS RGBA preprocessing.",
    )
    p.add_argument("--train-count", type=int, default=85)
    p.add_argument("--held-count", type=int, default=43)
    p.add_argument("--gpu", default="0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--slat-steps", type=int, default=25)
    p.add_argument("--slat-seed", type=int, default=42)
    p.add_argument("--mesh-render-resolution", type=int, default=768)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite-selection", action="store_true")
    p.add_argument("--force-stage", action="store_true")
    p.add_argument("--force-render", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir = str(Path(args.out_dir).expanduser())
    args.data_config = str(_require_file(Path(args.data_config), "data config"))
    args.split_json = str(_require_file(Path(args.split_json), "split json"))
    args.part_seg_ckpt = str(_require_file(Path(args.part_seg_ckpt), "part-seg ckpt"))
    args.ss_flow_ckpt = str(_require_file(Path(args.ss_flow_ckpt), "SS-flow ckpt"))
    args.ss_decoder_ckpt = str(_require_file(Path(args.ss_decoder_ckpt), "SS decoder ckpt"))
    args.slat_flow_ckpt = str(_require_file(Path(args.slat_flow_ckpt), "SLat flow ckpt"))
    args.slat_mesh_decoder_ckpt = str(_require_file(Path(args.slat_mesh_decoder_ckpt), "SLat mesh decoder ckpt"))
    _require_file(Path(args.slat_flow_ckpt).with_suffix(".json"), "SLat flow config")
    _require_file(Path(args.slat_mesh_decoder_ckpt).with_suffix(".json"), "SLat mesh decoder config")
    if int(args.shard_count) <= 0:
        raise ValueError("--shard-count must be positive")
    if not (0 <= int(args.shard_id) < int(args.shard_count)):
        raise ValueError("--shard-id must be in [0, shard_count)")
    if args.summarize_only:
        return summarize(args)
    if args.prepare_only:
        return prepare(args)
    if args.overwrite and int(args.shard_id) == 0:
        shutil.rmtree(Path(args.out_dir), ignore_errors=True)
    prepare(args)
    return run_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
