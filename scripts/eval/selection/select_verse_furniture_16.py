#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_JSON = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_verse_realappliance_v3.json"
)
DEFAULT_DATA_ROOT = Path("/robot/data-lab/jzh/art-gen/data/phyx-verse")


SELECTED_OBJECTS = [
    # Furniture with full drawer parts plus doors.
    "107e0185fbcb428584da42905bf094d3",  # Commode for Clothes
    "dfc9e3f0dd2c415db01462bf3a6de5fa",  # PPC400 Cabinet
    "34b4146f71d64617a97d3bbdb8a7682f",  # Wardrobe
    "4b1eeac388e641b9a6b975c32cf41439",  # Wooden Closet
    "05a035c3347645b8a7ceb6d65f825ac3",  # Closet
    # Furniture / storage furniture with lids.
    "0a6298cc21954fd59609834bc55663c7",  # Antique Chest
    "022f2092ded7436ca793f1948d7db12f",  # Ottoman
    "cb7da2136b864040bf7491e830071cc3",  # Treasure Chest
    "2319554b0d494cdb9687963cda06b7df",  # Crystal Chest
    "4b0c6f6db86e428080591506db783d17",  # Medieval Coffer
    "c7cf9af789824cc6ba7f53b6027dd03b",  # Takara Box
    "7227bc5a66294087ac1f8cb999d75ba9",  # Monster Chest
    "125ad3e93a95461cbad4e23a6fd234e4",  # Cofre V2
    "15559f47e22e44a3a3cb2c4d413a7c24",  # Tecno_cofre
    "1e3b71af22f042d9b9132ad5f7da1047",  # Treasure Chest
    "ca56d54791af482880752edfefa832cc",  # Borderlands Chest Lootbox
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _verse_manifest_paths(split_json: Path) -> list[Path]:
    split = _read_json(split_json)
    for item in split.get("datasets", []):
        if str(item.get("dataset_id")) == "phyx-verse":
            return [Path(str(path)) for path in item.get("manifest_paths", [])]
    raise KeyError(f"{split_json}: dataset_id='phyx-verse' not found")


def _part_text(part: dict[str, Any]) -> str:
    return " ".join(str(part.get(key, "")) for key in ("name", "type", "item_name")).lower()


def _is_full_drawer(part: dict[str, Any]) -> bool:
    text = _part_text(part)
    if "drawer" not in text:
        return False
    blocked = ("front_panel", "front panel", "handle", "knob", "pull")
    return not any(token in text for token in blocked)


def _is_door(part: dict[str, Any]) -> bool:
    text = _part_text(part)
    return "door" in text or "gate" in text


def _is_lid(part: dict[str, Any]) -> bool:
    text = _part_text(part)
    if "lid" in text or "cover" in text:
        blocked = ("lock", "latch", "tooth", "crystal")
        return not any(token in text for token in blocked)
    return False


def _keep_part(part: dict[str, Any]) -> bool:
    return _is_door(part) or _is_full_drawer(part) or _is_lid(part)


def _filter_record(rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for part in rec.get("target_parts") or []:
        if _keep_part(part):
            kept.append(copy.deepcopy(part))
    if not kept:
        raise ValueError(f"{rec.get('object_id')} angle={rec.get('angle_idx')}: no door/drawer/lid target parts kept")

    out = copy.deepcopy(rec)
    label_remap: dict[str, int] = {}
    local_to_component: dict[str, str] = {}
    target_names: list[str] = []
    original_labels: list[int] = []
    kinds = {"door": 0, "drawer": 0, "lid": 0}
    for slot, part in enumerate(kept, start=1):
        name = str(part.get("name") or part.get("item_name") or f"part_{slot:02d}")
        original_label = int(part.get("original_label", part.get("raw_label", slot)) or slot)
        part["local_label"] = slot
        target_names.append(name)
        original_labels.append(original_label)
        label_remap[str(original_label)] = slot
        local_to_component[str(slot)] = name
        if _is_door(part):
            kinds["door"] += 1
        if _is_full_drawer(part):
            kinds["drawer"] += 1
        if _is_lid(part):
            kinds["lid"] += 1

    if kinds["drawer"] and not kinds["door"]:
        raise ValueError(f"{rec.get('object_id')}: drawer candidate has no door")
    if not ((kinds["door"] and kinds["drawer"]) or kinds["lid"]):
        raise ValueError(f"{rec.get('object_id')}: does not satisfy door+drawer or lid policy after filtering")

    out["target_parts"] = kept
    out["target_part_names"] = target_names
    out["target_part_count"] = len(kept)
    out["target_original_labels"] = original_labels
    out["label_remap"] = label_remap
    out["local_label_to_component"] = local_to_component
    out["selection_override"] = {
        "policy": "0708-16-verse furniture: door+full-drawer or lid; one angle per object",
        "drawer_inner_cavity_proxy": "drawer parts are kept only when the part name is a full drawer, not a handle/knob/front_panel",
        "kept_kind_counts": kinds,
        "source_target_part_count": int(rec.get("target_part_count", len(rec.get("target_parts") or [])) or 0),
    }
    return out, kinds


def _sample_payload(rec: dict[str, Any], *, manifest_path: Path, kinds: dict[str, int]) -> dict[str, Any]:
    raw_counts = [int(part.get("raw_count", 0) or 0) for part in rec.get("target_parts") or []]
    return {
        "split": "train",
        "dataset_id": "phyx-verse",
        "object_key": f"phyx-verse::{rec['object_id']}",
        "obj_id": str(rec["object_id"]),
        "angle_idx": int(rec["angle_idx"]),
        "data_root": str(DEFAULT_DATA_ROOT),
        "manifest_path": str(manifest_path),
        "bucket": "verse_furniture_door_drawer_or_lid",
        "sample_bucket": "verse_furniture_door_drawer_or_lid",
        "priority_bucket": "verse_furniture_door_drawer_or_lid",
        "part_count": int(len(rec.get("target_parts") or [])),
        "min_raw_voxels": min(raw_counts) if raw_counts else 0,
        "max_raw_voxels": max(raw_counts) if raw_counts else 0,
        "has_button": False,
        "has_large_keyword": True,
        "selected_reason": "fixed_16_verse_furniture_door_drawer_or_lid_full_drawer_only",
        "original_split": "train",
        "category": str(rec.get("category") or ""),
        "object_name": str(rec.get("name") or ""),
        "kept_kind_counts": kinds,
        "target_part_names": list(rec.get("target_part_names") or []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select 16 phyx-verse furniture objects with door+drawer or lid.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--train-count", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.limit) != len(SELECTED_OBJECTS):
        raise ValueError(f"this curated selector expects --limit {len(SELECTED_OBJECTS)}, got {args.limit}")
    rows_by_object: dict[str, list[dict[str, Any]]] = {}
    for manifest_path in _verse_manifest_paths(args.split_json):
        for rec in _read_jsonl(manifest_path):
            obj_id = str(rec.get("object_id", ""))
            if obj_id in SELECTED_OBJECTS:
                rows_by_object.setdefault(obj_id, []).append(rec)

    filtered_rows: list[dict[str, Any]] = []
    kind_rows: list[dict[str, int]] = []
    missing = [obj_id for obj_id in SELECTED_OBJECTS if obj_id not in rows_by_object]
    if missing:
        raise KeyError(f"selected objects missing from verse manifests: {missing}")
    for obj_id in SELECTED_OBJECTS:
        rec = sorted(rows_by_object[obj_id], key=lambda row: int(row.get("angle_idx", 0)))[0]
        filtered, kinds = _filter_record(rec)
        filtered_rows.append(filtered)
        kind_rows.append(kinds)

    cfg_dir = args.out_dir / "_data_configs" / "phyx-verse"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    filtered_manifest = cfg_dir / "verse_furniture_door_drawer_lid_16.jsonl"
    with filtered_manifest.open("w", encoding="utf-8") as handle:
        for rec in filtered_rows:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    split_json = cfg_dir / "verse_furniture_door_drawer_lid_16_split.json"
    split_payload = {
        "name": "0708-16-verse-furniture-door-drawer-lid",
        "source_split_json": str(args.split_json),
        "selection_policy": (
            "16 curated unique phyx-verse furniture/storage-furniture objects; one angle per object; "
            "kept target parts are door/full-drawer or lid. Drawer candidates exclude handles, knobs, "
            "and front panels so drawers represent full cavity-bearing drawer parts."
        ),
        "datasets": [
            {
                "dataset_id": "phyx-verse",
                "data_root": str(DEFAULT_DATA_ROOT),
                "manifest_paths": [str(filtered_manifest)],
            }
        ],
        "train_ids": [
            {
                "dataset_id": "phyx-verse",
                "obj_id": str(rec["object_id"]),
                "object_key": f"phyx-verse::{rec['object_id']}",
            }
            for rec in filtered_rows
        ],
        "heldout_ids": [],
    }
    split_json.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    samples = [
        _sample_payload(rec, manifest_path=filtered_manifest, kinds=kinds)
        for rec, kinds in zip(filtered_rows, kind_rows)
    ]
    selection = {
        "name": "0708-16-verse-furniture-door-drawer-lid",
        "split_json": str(split_json),
        "filtered_manifest": str(filtered_manifest),
        "selection_policy": split_payload["selection_policy"],
        "sample_selection_unit": "objects",
        "selected_objects": [str(rec["object_id"]) for rec in filtered_rows],
        "selected_angles": [int(rec["angle_idx"]) for rec in filtered_rows],
        "counts": {
            "train": len(samples),
            "held": 0,
            "objects": len({sample["obj_id"] for sample in samples}),
            "samples": len(samples),
        },
        "samples": {"train": samples, "held": []},
    }
    out_path = args.out_dir / "selection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selection": str(out_path),
                "split_json": str(split_json),
                "filtered_manifest": str(filtered_manifest),
                "selected": len(samples),
                "unique_objects": len({sample["obj_id"] for sample in samples}),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
