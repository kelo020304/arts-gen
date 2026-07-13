#!/usr/bin/env python3
"""Pack promptable part segmentation rows into shard files on vePFS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    OFFICIAL_SPLIT_PATH,
    MultiPromptableBaseDataset,
    PromptablePartDataset,
    audit_promptable_mask_visibility,
    boundary_band_mask,
    build_semantic_vocab,
    dataset_specs_from_split,
    enumerate_part_rows_multi,
    load_official_split,
    make_base_datasets,
    part_row_key,
    rows_for_obj_ids,
)

DEFAULT_BASE_PACKED_V1 = Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v1")
DEFAULT_PACKED_V6 = Path(os.environ.get("PACKED_DIR", "/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v6"))
PACK_COMPLETE_NAME = ".pack_complete"


def optional_path_arg(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "-"}:
        return None
    return Path(text)


def compact_sample(sample: dict) -> dict:
    raw_count = sample["raw_count"]
    if isinstance(raw_count, torch.Tensor):
        raw_count = int(raw_count.item())
    m_gt = sample["m_gt"].to(dtype=torch.uint8).contiguous()
    m_boundary = sample.get("m_boundary")
    if m_boundary is None:
        m_boundary = boundary_band_mask(m_gt, radius=1)
    return {
        "z_global": sample["z_global"].contiguous().float(),
        "latent_gt": sample["latent_gt"].contiguous().float(),
        "masks2d": sample["masks2d"].to(dtype=torch.uint8).contiguous(),
        "m_gt": m_gt,
        "m_boundary": m_boundary.to(dtype=torch.uint8).contiguous(),
        "raw_coords": sample["raw_coords"].to(dtype=torch.int16).contiguous(),
        "whole_coords": sample["whole_coords"].to(dtype=torch.int16).contiguous(),
        "raw_count": int(raw_count),
        "dataset_id": sample.get("dataset_id", ""),
        "obj_id": sample["obj_id"],
        "angle_idx": int(sample["angle_idx"]),
        "sample_id": sample["sample_id"],
        "part_name": sample["part_name"],
        "semantic_type": sample["semantic_type"],
        "part_idx": int(sample["part_idx"]),
        "original_label": int(sample["original_label"]),
        "view_indices": sample["view_indices"].to(dtype=torch.int16).contiguous(),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _legacy_part_row_key(row: Any) -> str:
    return f"{row.obj_id}|{int(row.angle_idx)}|{row.part_name}"


def _is_legacy_physx_row(row: Any) -> bool:
    dataset_id = str(getattr(row, "dataset_id", "") or "")
    data_root = str(getattr(row, "data_root", "") or "")
    manifest_path = str(getattr(row, "manifest_path", "") or "")
    if not dataset_id:
        return True
    joined = f"{dataset_id} {data_root} {manifest_path}".lower()
    return "physx-mobility" in joined or "physx_mobility" in joined


def _path_stat(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    out = {
        "path": str(path),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    if path.is_file() and path.stat().st_size <= 128 * 1024 * 1024:
        out["sha256"] = _sha256_file(path)
    return out


def source_fingerprint(split_json: Path, *, base_packed_dir: Path | None = None, pack_limit: int = 0) -> str:
    split_json = Path(split_json)
    split = load_official_split(split_json)
    specs = dataset_specs_from_split(split)
    payload: dict[str, Any] = {
        "split_json": _path_stat(split_json),
        "datasets": [],
        "base_packed_dir": None,
        "pack_limit": int(pack_limit),
    }
    for spec in specs:
        manifests = []
        for manifest in spec.manifest_paths:
            path = Path(manifest)
            if not path.is_absolute():
                path = spec.data_root / path
            manifests.append(_path_stat(path))
        payload["datasets"].append({
            "dataset_id": spec.dataset_id,
            "data_root": str(spec.data_root),
            "manifest_paths": manifests,
        })
    if base_packed_dir is not None:
        base_packed_dir = Path(base_packed_dir)
        payload["base_packed_dir"] = {
            "path": str(base_packed_dir),
            "index": _path_stat(base_packed_dir / "index.json"),
        }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pack_completion_status(
    packed_dir: Path,
    *,
    expected_fingerprint: str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    packed_dir = Path(packed_dir)
    marker_path = packed_dir / PACK_COMPLETE_NAME
    index_path = packed_dir / "index.json"
    if not marker_path.is_file():
        return False, f"missing {marker_path}", None
    if not index_path.is_file():
        return False, f"missing {index_path}", None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"failed to parse marker/index: {exc}", None
    rows = int(marker.get("rows", -1))
    entries = index.get("entries", [])
    if rows <= 0:
        return False, f"invalid marker rows={rows}", marker
    if len(entries) != rows:
        return False, f"marker rows={rows} but index entries={len(entries)}", marker
    index_rows = index.get("rows")
    if index_rows is not None and int(index_rows) != rows:
        return False, f"marker rows={rows} but index rows={index_rows}", marker
    fields = set(map(str, index.get("fields", [])))
    if "m_boundary" not in fields:
        return False, "packed index missing required field m_boundary", marker
    if expected_fingerprint is not None and marker.get("source_fingerprint") != expected_fingerprint:
        return False, "source fingerprint mismatch", marker
    shards = marker.get("shards")
    if not isinstance(shards, list) or not shards:
        return False, "marker has no shard metadata", marker
    index_rows_by_shard: dict[str, int] = {}
    for entry in entries:
        name = str(entry.get("shard", ""))
        index_rows_by_shard[name] = index_rows_by_shard.get(name, 0) + 1
    shard_rows = 0
    for shard in shards:
        name = str(shard.get("name", ""))
        shard_path = packed_dir / name
        if not shard_path.is_file():
            return False, f"missing shard {shard_path}", marker
        st = shard_path.stat()
        if int(shard.get("size_bytes", -1)) != int(st.st_size):
            return False, f"size mismatch for shard {name}", marker
        if int(shard.get("rows", -1)) != int(index_rows_by_shard.get(name, -1)):
            return False, f"row count mismatch for shard {name}", marker
        shard_rows += int(shard.get("rows", 0))
    if shard_rows != rows:
        return False, f"marker rows={rows} but shard rows={shard_rows}", marker
    indexed_shards = {str(entry.get("shard", "")) for entry in entries}
    marker_shards = {str(shard.get("name", "")) for shard in shards}
    if indexed_shards != marker_shards:
        return False, "index shard set differs from marker shard set", marker
    return True, "complete", marker


class PackedSourceReader:
    def __init__(self, packed_dir: Path | None) -> None:
        self.packed_dir = Path(packed_dir) if packed_dir is not None else None
        self.entries_by_key: dict[str, dict[str, Any]] = {}
        self._cached_shard_name: str | None = None
        self._cached_items: list[dict[str, Any]] | None = None
        if self.packed_dir is None:
            return
        index_path = self.packed_dir / "index.json"
        if not index_path.is_file():
            return
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.entries_by_key = {str(entry["key"]): entry for entry in index.get("entries", [])}

    def get(self, key: str) -> dict[str, Any] | None:
        if self.packed_dir is None:
            return None
        entry = self.entries_by_key.get(str(key))
        if entry is None:
            return None
        shard_name = str(entry["shard"])
        if shard_name != self._cached_shard_name:
            payload = torch.load(self.packed_dir / shard_name, map_location="cpu", weights_only=False)
            if not isinstance(payload, list):
                raise ValueError(f"{self.packed_dir / shard_name} expected list payload")
            self._cached_shard_name = shard_name
            self._cached_items = payload
        if self._cached_items is None:
            raise RuntimeError("packed source cache was not populated")
        return dict(self._cached_items[int(entry["index"])])


def _write_pack_complete_atomic(out_dir: Path, payload: dict[str, Any]) -> None:
    marker_path = Path(out_dir) / PACK_COMPLETE_NAME
    tmp_path = marker_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(marker_path)


def _shard_metadata(out_dir: Path, shard_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in shard_entries:
        path = Path(out_dir) / str(item["name"])
        st = path.stat()
        out.append({
            "name": str(item["name"]),
            "rows": int(item["rows"]),
            "size_bytes": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        })
    return out


def _sample_part_row_key(sample: Mapping[str, Any]) -> str:
    dataset_id = str(sample.get("dataset_id", "") or "")
    prefix = f"{dataset_id}::" if dataset_id else ""
    return f"{prefix}{sample['obj_id']}|{int(sample['angle_idx'])}|{sample['part_name']}"


def _existing_resume_shards(out_dir: Path) -> list[tuple[int, Path]]:
    shards = []
    for path in sorted(Path(out_dir).glob("shard_*.pt")):
        stem = path.stem
        try:
            shard_idx = int(stem.removeprefix("shard_"))
        except ValueError as exc:
            raise ValueError(f"unexpected shard name for resume: {path}") from exc
        shards.append((shard_idx, path))
    for expected, (shard_idx, path) in enumerate(shards):
        if shard_idx != expected:
            raise ValueError(f"resume requires contiguous shards from 0; expected {expected:06d}, got {path.name}")
    return shards


def _resume_state_from_shards(
    *,
    out_dir: Path,
    rows: list[Any],
    shard_size: int,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    shard_meta_entries: list[dict[str, Any]] = []
    shard_items: list[dict[str, Any]] = []
    resume_rows = 0
    next_shard_idx = 0
    shards = _existing_resume_shards(out_dir)
    if not shards:
        return {
            "entries": entries,
            "shard_meta_entries": shard_meta_entries,
            "shard_items": shard_items,
            "resume_rows": 0,
            "next_shard_idx": 0,
            "resume_shards": 0,
        }
    for pos, (shard_idx, path) in enumerate(shards):
        try:
            items = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"failed to load existing resume shard {path}") from exc
        if not isinstance(items, list):
            raise ValueError(f"{path} expected list payload for resume, got {type(items).__name__}")
        if len(items) > int(shard_size):
            raise ValueError(f"{path} has {len(items)} rows, exceeds shard_size={int(shard_size)}")
        is_last = pos == len(shards) - 1
        if not is_last and len(items) != int(shard_size):
            raise ValueError(f"{path} is not the last shard but has {len(items)} rows, expected {int(shard_size)}")
        shard_name = path.name
        for local_idx, sample in enumerate(items):
            global_idx = resume_rows + local_idx
            if global_idx >= len(rows):
                raise ValueError(f"existing resume shards contain more rows than current split rows at {path}:{local_idx}")
            expected_key = part_row_key(rows[global_idx])
            got_key = _sample_part_row_key(sample)
            if got_key != expected_key:
                raise ValueError(
                    f"resume shard prefix mismatch at global row {global_idx}: "
                    f"{path.name}[{local_idx}] got {got_key!r}, expected {expected_key!r}"
                )
            entries.append({
                "key": expected_key,
                "shard": shard_name,
                "index": int(local_idx),
                "dataset_id": rows[global_idx].dataset_id,
                "obj_id": rows[global_idx].obj_id,
                "angle_idx": int(rows[global_idx].angle_idx),
                "part_name": rows[global_idx].part_name,
                "raw_count": int(rows[global_idx].raw_count),
            })
        resume_rows += len(items)
        if is_last and len(items) < int(shard_size):
            shard_items = items
            next_shard_idx = shard_idx
        else:
            shard_meta_entries.append({"name": shard_name, "rows": len(items)})
            next_shard_idx = shard_idx + 1
    return {
        "entries": entries,
        "shard_meta_entries": shard_meta_entries,
        "shard_items": shard_items,
        "resume_rows": int(resume_rows),
        "next_shard_idx": int(next_shard_idx),
        "resume_shards": len(shards),
    }


def _is_realappliance_row(row: Any) -> bool:
    text = " ".join(
        str(getattr(row, name, "") or "")
        for name in ("dataset_id", "data_root", "manifest_path", "category")
    ).lower()
    return "realappliance" in text or "real appliance" in text


def _select_limited_rows(rows: list[Any], limit: int) -> list[Any]:
    limit = int(limit)
    if limit <= 0:
        return []
    if len(rows) <= limit:
        return list(rows)
    selected: list[Any] = []
    selected_keys: set[str] = set()

    def add(row: Any) -> None:
        key = part_row_key(row)
        if key not in selected_keys and len(selected) < limit:
            selected.append(row)
            selected_keys.add(key)

    realappliance_rows = [row for row in rows if _is_realappliance_row(row)]
    if realappliance_rows:
        realappliance_quota = min(len(realappliance_rows), max(1, min(limit, max(4, limit // 16))))
        for row in realappliance_rows[:realappliance_quota]:
            add(row)
    for row in rows:
        add(row)
        if len(selected) >= limit:
            break
    return selected


def pack_promptable_seg_dataset(
    *,
    split_json: Path = OFFICIAL_SPLIT_PATH,
    out_dir: Path = DEFAULT_PACKED_V6,
    shard_size: int = 512,
    limit: int = 0,
    include_heldout: bool = True,
    overwrite: bool = False,
    mask_audit_views: int = 12,
    filter_undetectable: bool = True,
    fail_label_absent_ratio: float = 0.02,
    base_packed_dir: Path | None = None,
    progress_every: int = 1000,
    source_fp: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    split_json = Path(split_json)
    out_dir = Path(out_dir)
    index_path = out_dir / "index.json"
    base_packed_dir = optional_path_arg(base_packed_dir)
    if base_packed_dir is not None and Path(base_packed_dir).resolve() == out_dir.resolve():
        raise ValueError(f"base_packed_dir must differ from out_dir, got {out_dir}")
    if bool(resume) and bool(overwrite):
        raise ValueError("resume and overwrite cannot both be true")
    if index_path.exists() and not bool(overwrite):
        raise FileExistsError(f"{index_path} exists; pass overwrite=True to rebuild")
    if bool(overwrite):
        for old_shard in out_dir.glob("shard_*.pt"):
            old_shard.unlink()
        for old_path in (index_path, out_dir / PACK_COMPLETE_NAME):
            if old_path.exists():
                old_path.unlink()

    split = load_official_split(split_json)
    specs = dataset_specs_from_split(split)
    bases = make_base_datasets(specs)
    base = MultiPromptableBaseDataset(bases)
    rows_all = enumerate_part_rows_multi(bases)
    train_refs = split.get("train_keys", split["train_ids"])
    heldout_refs = split.get("heldout_keys", split["heldout_ids"])
    train_rows_all = rows_for_obj_ids(rows_all, train_refs)
    heldout_rows = rows_for_obj_ids(rows_all, heldout_refs) if bool(include_heldout) else []
    train_rows = list(train_rows_all)
    rows = list(train_rows_all)
    if bool(include_heldout):
        rows.extend(heldout_rows)
    if int(limit) > 0:
        if bool(include_heldout) and heldout_rows:
            train_limit = min(len(train_rows_all), max(1, int(round(int(limit) * 0.75))))
            heldout_limit = min(len(heldout_rows), max(1, int(limit) - train_limit))
            while train_limit + heldout_limit < int(limit) and train_limit < len(train_rows_all):
                train_limit += 1
            while train_limit + heldout_limit < int(limit) and heldout_limit < len(heldout_rows):
                heldout_limit += 1
            train_rows = _select_limited_rows(train_rows_all, train_limit)
            rows = [*train_rows, *_select_limited_rows(heldout_rows, heldout_limit)]
        else:
            rows = _select_limited_rows(rows, int(limit))
            train_row_keys = {part_row_key(row) for row in train_rows_all}
            train_rows = [row for row in rows if part_row_key(row) in train_row_keys]
    mask_audit_meta = {}
    if int(mask_audit_views) > 0:
        audit = audit_promptable_mask_visibility(base, rows, expected_views=int(mask_audit_views))
        records = list(audit["records"])
        undetectable_selected = [rec for rec in records if rec["classification"] == "undetectable_selected_views"]
        undetectable_all = [rec for rec in records if rec["classification"] == "undetectable_all_views"]
        undetectable = [*undetectable_selected, *undetectable_all]
        absent = [rec for rec in records if rec["classification"] == "label_absent_all_views"]
        label_absent_ratio = len(absent) / max(1, len(records))
        mask_audit_meta = {
            key: value
            for key, value in audit.items()
            if key != "records"
        }
        mask_audit_meta.update({
            "filter_undetectable": bool(filter_undetectable),
            "undetectable_rows": len(undetectable),
            "undetectable_selected_rows": len(undetectable_selected),
            "undetectable_all_views_rows": len(undetectable_all),
            "label_absent_rows": len(absent),
        })
        audit_dir = out_dir / "mask_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (audit_dir / "summary.json").write_text(json.dumps(mask_audit_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (audit_dir / "undetectable.json").write_text(json.dumps(undetectable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (audit_dir / "undetectable_selected_views.json").write_text(json.dumps(undetectable_selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (audit_dir / "undetectable_all_views.json").write_text(json.dumps(undetectable_all, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (audit_dir / "label_absent_all_views.json").write_text(json.dumps(absent, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[pack-mask-audit] total={len(records)} visible={audit['class_counts'].get('visible_selected_views', 0)} "
            f"undetectable={len(undetectable)} ({len(undetectable) / max(1, len(records)):.4%}; "
            f"selected={len(undetectable_selected)} all_views={len(undetectable_all)}) "
            f"label_absent={len(absent)} ({label_absent_ratio:.4%}) out={audit_dir}",
            flush=True,
        )
        if label_absent_ratio > float(fail_label_absent_ratio):
            raise RuntimeError(
                f"label_absent_all_views ratio {label_absent_ratio:.4%} exceeds "
                f"--fail-label-absent-ratio={float(fail_label_absent_ratio):.4%}; inspect {audit_dir}"
            )
        if bool(filter_undetectable):
            drop_keys = {str(rec["key"]) for rec in undetectable}
            before = len(rows)
            rows = [row for row in rows if part_row_key(row) not in drop_keys]
            train_rows = [row for row in train_rows if part_row_key(row) not in drop_keys]
            print(f"[pack-mask-audit] filtered undetectable rows {before}->{len(rows)}", flush=True)
    semantic_vocab = build_semantic_vocab(rows)
    ds = PromptablePartDataset(
        base,
        rows,
        mask_size=512,
        semantic_vocab=semantic_vocab,
        include_whole_coords=True,
    )

    packed_source = PackedSourceReader(base_packed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    shard_items: list[dict[str, Any]] = []
    shard_meta_entries: list[dict[str, Any]] = []
    shard_idx = 0
    start_idx = 0
    resume_shards = 0
    if bool(resume):
        state = _resume_state_from_shards(out_dir=out_dir, rows=rows, shard_size=int(shard_size))
        entries = state["entries"]
        shard_items = state["shard_items"]
        shard_meta_entries = state["shard_meta_entries"]
        shard_idx = int(state["next_shard_idx"])
        start_idx = int(state["resume_rows"])
        resume_shards = int(state["resume_shards"])
        print(
            f"[pack-resume] validated shards={resume_shards} rows={start_idx} "
            f"next_shard={shard_idx:06d} partial_rows={len(shard_items)}",
            flush=True,
        )
    t0 = time.time()
    total_rows = len(rows)
    source_fp = source_fp or source_fingerprint(split_json, base_packed_dir=base_packed_dir, pack_limit=limit)
    reused_base_rows = 0
    materialized_rows = 0
    for idx in range(start_idx, len(rows)):
        row = rows[idx]
        packed_sample = None
        if _is_legacy_physx_row(row):
            packed_sample = packed_source.get(_legacy_part_row_key(row))
            if packed_sample is None:
                packed_sample = packed_source.get(part_row_key(row))
        if packed_sample is not None:
            packed_sample["dataset_id"] = row.dataset_id
            sample = compact_sample(packed_sample)
            reused_base_rows += 1
        else:
            sample = compact_sample(ds[idx])
            materialized_rows += 1
        shard_items.append(sample)
        entries.append({
            "key": part_row_key(row),
            "shard": f"shard_{shard_idx:06d}.pt",
            "index": len(shard_items) - 1,
            "dataset_id": row.dataset_id,
            "obj_id": row.obj_id,
            "angle_idx": int(row.angle_idx),
            "part_name": row.part_name,
            "raw_count": int(row.raw_count),
        })
        if int(progress_every) > 0 and ((idx + 1) == 1 or (idx + 1) % int(progress_every) == 0):
            pct = (idx + 1) / max(1, total_rows) * 100.0
            print(f"[pack-progress] rows={idx + 1}/{total_rows} pct={pct:.2f}", flush=True)
        if len(shard_items) >= int(shard_size):
            shard_name = f"shard_{shard_idx:06d}.pt"
            torch.save(shard_items, out_dir / shard_name)
            shard_meta_entries.append({"name": shard_name, "rows": len(shard_items)})
            print(f"[pack] shard={shard_idx:06d} rows={len(shard_items)} total={idx + 1}/{len(rows)}", flush=True)
            shard_idx += 1
            shard_items = []
    if shard_items:
        shard_name = f"shard_{shard_idx:06d}.pt"
        torch.save(shard_items, out_dir / shard_name)
        shard_meta_entries.append({"name": shard_name, "rows": len(shard_items)})
        print(f"[pack] shard={shard_idx:06d} rows={len(shard_items)} total={len(rows)}/{len(rows)}", flush=True)

    payload = {
        "format_version": 1,
        "split_json": str(split_json),
        "datasets": [
            {
                "dataset_id": spec.dataset_id,
                "data_root": str(spec.data_root),
                "manifest_paths": [str(path) for path in spec.manifest_paths],
            }
            for spec in specs
        ],
        "created_unix": time.time(),
        "rows": len(rows),
        "input_train_rows": len(train_rows_all),
        "train_rows": len(train_rows),
        "include_heldout": bool(include_heldout),
        "shard_size": int(shard_size),
        "pack_limit": int(limit),
        "base_packed_dir": str(base_packed_dir) if base_packed_dir is not None else None,
        "resume": bool(resume),
        "resume_existing_shards": int(resume_shards),
        "resume_existing_rows": int(start_idx),
        "reused_base_rows": int(reused_base_rows),
        "materialized_rows": int(materialized_rows),
        "semantic_vocab": semantic_vocab,
        "mask_audit": mask_audit_meta,
        "fields": ["z_global", "latent_gt", "masks2d", "m_gt", "m_boundary", "raw_coords", "whole_coords"],
        "entries": entries,
    }
    tmp_index_path = index_path.with_suffix(".json.tmp")
    tmp_index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_index_path.replace(index_path)
    shard_meta = _shard_metadata(out_dir, shard_meta_entries)
    size_bytes = sum(int(item["size_bytes"]) for item in shard_meta) + index_path.stat().st_size
    elapsed = time.time() - t0
    marker = {
        "format_version": 1,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "include_heldout": bool(include_heldout),
        "pack_limit": int(limit),
        "split_json": str(split_json),
        "base_packed_dir": str(base_packed_dir) if base_packed_dir is not None else None,
        "source_fingerprint": source_fp,
        "resume": bool(resume),
        "resume_existing_shards": int(resume_shards),
        "resume_existing_rows": int(start_idx),
        "created_unix": time.time(),
        "elapsed_s": elapsed,
        "size_bytes": int(size_bytes),
        "reused_base_rows": int(reused_base_rows),
        "materialized_rows": int(materialized_rows),
        "shards": shard_meta,
    }
    _write_pack_complete_atomic(out_dir, marker)
    print(
        f"[pack] done rows={len(rows)} reused_base={reused_base_rows} materialized={materialized_rows} "
        f"out={out_dir} size_gb={size_bytes / (1024 ** 3):.3f} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return {"index": payload, "marker": marker}


def ensure_packed_dataset(
    *,
    split_json: Path,
    out_dir: Path,
    shard_size: int = 512,
    include_heldout: bool = True,
    overwrite_incomplete: bool = True,
    base_packed_dir: Path | None = None,
    progress_every: int = 1000,
    mask_audit_views: int = 0,
    limit: int = 0,
) -> dict[str, Any]:
    base_packed_dir = optional_path_arg(base_packed_dir)
    fp = source_fingerprint(split_json, base_packed_dir=base_packed_dir, pack_limit=limit)
    ok, reason, marker = pack_completion_status(out_dir, expected_fingerprint=fp)
    if ok:
        print(f"[pack-check] complete out={out_dir} rows={marker.get('rows') if marker else 'unknown'}", flush=True)
        return {"status": "complete", "marker": marker}
    print(f"[pack-check] incomplete out={out_dir}: {reason}", flush=True)
    return {
        "status": "packed",
        **pack_promptable_seg_dataset(
            split_json=split_json,
            out_dir=out_dir,
            shard_size=shard_size,
            limit=limit,
            include_heldout=include_heldout,
            overwrite=bool(overwrite_incomplete),
            mask_audit_views=mask_audit_views,
            filter_undetectable=True,
            base_packed_dir=base_packed_dir,
            progress_every=progress_every,
            source_fp=fp,
            resume=False,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, default=OFFICIAL_SPLIT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PACKED_V6)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-heldout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-audit-views", type=int, default=12)
    parser.add_argument("--filter-undetectable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-label-absent-ratio", type=float, default=0.02)
    parser.add_argument("--base-packed-dir", type=optional_path_arg, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    pack_promptable_seg_dataset(
        split_json=args.split_json,
        out_dir=args.out_dir,
        shard_size=int(args.shard_size),
        limit=int(args.limit),
        include_heldout=bool(args.include_heldout),
        overwrite=bool(args.overwrite),
        mask_audit_views=int(args.mask_audit_views),
        filter_undetectable=bool(args.filter_undetectable),
        fail_label_absent_ratio=float(args.fail_label_absent_ratio),
        base_packed_dir=args.base_packed_dir,
        progress_every=int(args.progress_every),
        resume=bool(args.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
