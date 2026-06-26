#!/usr/bin/env python3
"""Repack a PromptablePartSeg object split from a large packed dataset into small local shards."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


DEFAULT_SOURCE = Path("/mnt/robot-data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5")
DEFAULT_SPLIT = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg_full_S_0618-2/"
    "eval/ab_subset_boundary/subset_split_seed20260622.json"
)
DEFAULT_OUT = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg_full_S_0618-2/"
    "eval/ab_subset_boundary/subset_pack_v5_seed20260622"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-packed-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def object_key(entry: dict[str, Any]) -> str:
    return f"{entry.get('dataset_id', '')}::{entry.get('obj_id', '')}"


def write_marker(out_dir: Path, marker: dict[str, Any]) -> None:
    tmp = out_dir / ".pack_complete.tmp"
    tmp.write_text(json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_dir / ".pack_complete")


def main() -> int:
    args = parse_args()
    src = Path(args.source_packed_dir)
    out = Path(args.out_dir)
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    wanted_objects = set(map(str, split.get("train_keys", split.get("train_ids", []))))
    wanted_objects |= set(map(str, split.get("heldout_keys", split.get("heldout_ids", []))))
    train_objects = set(map(str, split.get("train_keys", split.get("train_ids", []))))
    if not wanted_objects:
        raise RuntimeError(f"split has no train/heldout object keys: {args.split_json}")
    if out.exists() and any(out.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"{out} is not empty; pass --overwrite")
    if bool(args.overwrite):
        for path in out.glob("shard_*.pt"):
            path.unlink()
        for name in ("index.json", ".pack_complete"):
            path = out / name
            if path.exists():
                path.unlink()
    out.mkdir(parents=True, exist_ok=True)

    src_index = json.loads((src / "index.json").read_text(encoding="utf-8"))
    selected: list[tuple[int, dict[str, Any]]] = [
        (idx, entry)
        for idx, entry in enumerate(src_index.get("entries", []))
        if object_key(entry) in wanted_objects
    ]
    if not selected:
        raise RuntimeError("selected zero rows from source pack")
    selected.sort(key=lambda item: (str(item[1]["shard"]), int(item[1]["index"])))
    selected_global_indices = {idx for idx, _entry in selected}

    entries: list[dict[str, Any]] = []
    shard_items: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    shard_idx = 0
    out_row_idx = 0
    source_cache_name: str | None = None
    source_cache_payload: list[dict[str, Any]] | None = None
    t0 = time.time()

    def flush_shard() -> None:
        nonlocal shard_idx, shard_items
        if not shard_items:
            return
        name = f"shard_{shard_idx:06d}.pt"
        torch.save(shard_items, out / name)
        st = (out / name).stat()
        shard_rows.append({"name": name, "rows": len(shard_items), "size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
        print(f"[subset-repack] wrote {name} rows={len(shard_items)}", flush=True)
        shard_idx += 1
        shard_items = []

    for selected_pos, (src_global_idx, entry) in enumerate(selected, start=1):
        shard_name = str(entry["shard"])
        if shard_name != source_cache_name:
            source_cache_payload = torch.load(src / shard_name, map_location="cpu", weights_only=False)
            if not isinstance(source_cache_payload, list):
                raise ValueError(f"{src / shard_name} expected list payload")
            source_cache_name = shard_name
            if int(args.progress_every) > 0:
                print(
                    f"[subset-repack] source_shard={shard_name} selected={selected_pos}/{len(selected)} "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )
        assert source_cache_payload is not None
        sample = dict(source_cache_payload[int(entry["index"])])
        object_is_train = object_key(entry) in train_objects
        out_shard_name = f"shard_{shard_idx:06d}.pt"
        entries.append({
            **{k: v for k, v in entry.items() if k not in {"shard", "index"}},
            "key": str(entry["key"]),
            "shard": out_shard_name,
            "index": len(shard_items),
            "source_shard": shard_name,
            "source_index": int(entry["index"]),
            "source_global_index": int(src_global_idx),
            "split": "train" if object_is_train else "heldout",
        })
        shard_items.append(sample)
        out_row_idx += 1
        if len(shard_items) >= int(args.shard_size):
            flush_shard()
        if int(args.progress_every) > 0 and selected_pos % int(args.progress_every) == 0:
            print(f"[subset-repack] rows={selected_pos}/{len(selected)} out_rows={out_row_idx}", flush=True)
    flush_shard()

    train_rows = sum(1 for entry in entries if entry.get("split") == "train")
    payload = {
        **{k: v for k, v in src_index.items() if k not in {"entries", "rows", "train_rows", "created_unix", "shard_size", "base_packed_dir"}},
        "format_version": 1,
        "split_json": str(args.split_json),
        "source_packed_dir": str(src),
        "created_unix": time.time(),
        "rows": len(entries),
        "input_train_rows": train_rows,
        "train_rows": train_rows,
        "include_heldout": True,
        "shard_size": int(args.shard_size),
        "base_packed_dir": str(src),
        "subset_repack": True,
        "source_rows": int(src_index.get("rows", len(src_index.get("entries", [])))),
        "selected_source_global_indices": sorted(selected_global_indices),
        "entries": entries,
    }
    (out / "index.json.tmp").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "index.json.tmp").replace(out / "index.json")
    marker = {
        "format_version": 1,
        "rows": len(entries),
        "train_rows": train_rows,
        "include_heldout": True,
        "split_json": str(args.split_json),
        "base_packed_dir": str(src),
        "subset_repack": True,
        "created_unix": time.time(),
        "elapsed_s": time.time() - t0,
        "size_bytes": int(sum(item["size_bytes"] for item in shard_rows) + (out / "index.json").stat().st_size),
        "shards": shard_rows,
    }
    write_marker(out, marker)
    print(
        f"[subset-repack] done rows={len(entries)} train_rows={train_rows} shards={len(shard_rows)} "
        f"size_gb={marker['size_bytes'] / (1024 ** 3):.3f} elapsed={marker['elapsed_s']:.1f}s out={out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
