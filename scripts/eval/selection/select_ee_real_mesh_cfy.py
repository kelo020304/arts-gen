#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PACKED_INDEX = Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5/index.json")
DEFAULT_SPLIT_JSON = Path(
    "/robot/data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_verse_realappliance_v3.json"
)
DEFAULT_OUT_DIR = Path("/robot/data-lab/jzh/art-gen/ee-eval/0707-32-cfy")
DATA_ROOTS = {
    "phyx-verse": Path("/robot/data-lab/jzh/art-gen/data/phyx-verse"),
    "realappliance": Path("/robot/data-lab/jzh/art-gen/data/realappliance"),
}
TARGETS = {"phyx-verse": 16, "realappliance": 16}

PRIMARY_KEYWORDS = (
    "door",
    "lid",
    "cover",
    "hatch",
    "\u95e8",
    "\u76d6",
)
DRAWER_KEYWORDS = ("drawer", "\u62bd\u5c49")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _text_for_part(name: str, part: dict[str, Any]) -> str:
    fields = [name, str(part.get("type", "")), str(part.get("joint", "")), str(part.get("joint_type", ""))]
    return " ".join(fields).lower()


def _part_matches(part_name: str, part: dict[str, Any]) -> tuple[bool, str]:
    text = _text_for_part(part_name, part)
    if any(word in text for word in DRAWER_KEYWORDS):
        return False, "drawer_excluded"
    for word in PRIMARY_KEYWORDS:
        if word in text:
            return True, word
    return False, ""


def _resolve_map(path: str, mtl_path: Path) -> Path:
    return (mtl_path.parent / path).resolve()


def _mtl_texture_paths(mtl_path: Path) -> list[Path]:
    textures: list[Path] = []
    if not mtl_path.is_file():
        return textures
    for raw_line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in {"map_kd", "map_ka", "map_ks", "map_bump", "bump"}:
            textures.append(_resolve_map(parts[1], mtl_path))
    return textures


def _asset_status(dataset_id: str, obj_id: str, stems: list[str]) -> dict[str, Any]:
    data_root = DATA_ROOTS[dataset_id]
    obj_dir = data_root / "raw" / "partseg" / obj_id / "objs"
    missing: list[str] = []
    textures: list[str] = []
    for stem in stems:
        obj_path = obj_dir / f"{stem}.obj"
        mtl_path = obj_dir / f"{stem}.mtl"
        if not obj_path.is_file():
            missing.append(str(obj_path))
        if not mtl_path.is_file():
            missing.append(str(mtl_path))
            continue
        for texture in _mtl_texture_paths(mtl_path):
            if texture.is_file():
                textures.append(str(texture))
            else:
                missing.append(str(texture))
    return {
        "obj_dir": str(obj_dir),
        "missing": missing,
        "texture_count": len(set(textures)),
        "textures": sorted(set(textures)),
    }


def _candidate_for_object(dataset_id: str, obj_id: str, angle_idx: int) -> dict[str, Any] | None:
    data_root = DATA_ROOTS[dataset_id]
    part_info_path = data_root / "reconstruction" / "part_info" / obj_id / "part_info.json"
    if not part_info_path.is_file():
        return None
    try:
        part_info = _load_json(part_info_path)
    except Exception:
        return None
    parts = part_info.get("parts")
    if not isinstance(parts, dict):
        return None

    matched_parts: list[dict[str, Any]] = []
    all_stems: list[str] = []
    trigger_stems: list[str] = []
    for part_name, part in parts.items():
        if not isinstance(part, dict):
            continue
        stems = [str(item) for item in part.get("obj_files", []) if str(item)]
        all_stems.extend(stems)
        matched, keyword = _part_matches(str(part_name), part)
        if not matched:
            continue
        trigger_stems.extend(stems)
        matched_parts.append(
            {
                "part_name": str(part_name),
                "type": str(part.get("type", "")),
                "joint": str(part.get("joint", "")),
                "keyword": keyword,
                "obj_files": stems,
            }
        )
    if not matched_parts or not all_stems or not trigger_stems:
        return None

    all_asset_status = _asset_status(dataset_id, obj_id, sorted(set(all_stems)))
    trigger_asset_status = _asset_status(dataset_id, obj_id, sorted(set(trigger_stems)))
    if all_asset_status["missing"] or trigger_asset_status["missing"]:
        return None
    if int(trigger_asset_status["texture_count"]) <= 0:
        return None

    keyword_score = 0
    for item in matched_parts:
        key = item["keyword"]
        keyword_score += 4 if key in {"door", "\u95e8"} else 3 if key in {"lid", "\u76d6"} else 2
    category = str(part_info.get("category", ""))
    category_score = 2 if any(word in category.lower() for word in ("furniture", "appliance", "cabinet")) else 0
    return {
        "dataset_id": dataset_id,
        "obj_id": obj_id,
        "angle_idx": int(angle_idx),
        "category": category,
        "num_parts": int(part_info.get("num_parts", len(parts))),
        "matched_parts": matched_parts,
        "all_obj_files": sorted(set(all_stems)),
        "texture_count": int(all_asset_status["texture_count"]),
        "trigger_texture_count": int(trigger_asset_status["texture_count"]),
        "score": int(keyword_score + category_score),
        "part_info": str(part_info_path),
        "raw_obj_dir": all_asset_status["obj_dir"],
    }


def _object_order(packed_index: Path) -> dict[str, list[tuple[str, int]]]:
    packed = _load_json(packed_index)
    order: dict[str, list[tuple[str, int]]] = {key: [] for key in TARGETS}
    seen: set[tuple[str, str]] = set()
    for entry in packed.get("entries", []):
        dataset_id = str(entry.get("dataset_id", ""))
        if dataset_id not in TARGETS:
            continue
        obj_id = str(entry.get("obj_id", ""))
        if not obj_id:
            continue
        key = (dataset_id, obj_id)
        if key in seen:
            continue
        seen.add(key)
        order[dataset_id].append((obj_id, int(entry.get("angle_idx", 0))))
    return order


def _select(packed_index: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_order = _object_order(packed_index)
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for dataset_id, target in TARGETS.items():
        candidates: list[dict[str, Any]] = []
        scanned = 0
        for obj_id, angle_idx in object_order.get(dataset_id, []):
            scanned += 1
            candidate = _candidate_for_object(dataset_id, obj_id, angle_idx)
            if candidate is not None:
                candidate["packed_order"] = scanned
                candidates.append(candidate)
            if len(candidates) >= max(target * 4, target + 20):
                break
        candidates.sort(key=lambda item: (-int(item["score"]), int(item["packed_order"]), item["obj_id"]))
        chosen = candidates[:target]
        if len(chosen) < target:
            raise RuntimeError(f"only found {len(chosen)} candidates for {dataset_id}, need {target}")
        selected.extend(chosen)
        audit.append(
            {
                "dataset_id": dataset_id,
                "target": target,
                "scanned_objects": scanned,
                "candidate_count": len(candidates),
                "chosen_count": len(chosen),
            }
        )
    return selected, audit


def _selection_payload(
    *,
    selected: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    split_json: Path,
    packed_index: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(selected):
        rows.append(
            {
                "split": "train",
                "dataset_id": item["dataset_id"],
                "object_key": f"{item['dataset_id']}::{item['obj_id']}",
                "obj_id": item["obj_id"],
                "angle_idx": int(item["angle_idx"]),
                "data_root": str(DATA_ROOTS[item["dataset_id"]]),
                "manifest_path": "",
                "bucket": "real_mesh_cfy",
                "sample_bucket": "real_mesh_cfy",
                "priority_bucket": "real_mesh_cfy",
                "part_count": int(item["num_parts"]),
                "min_raw_voxels": 0,
                "max_raw_voxels": 0,
                "has_button": False,
                "has_large_keyword": True,
                "selected_reason": "door_lid_cover_object_with_real_textured_source_mesh",
                "original_split": "train",
                "cfy_rank": idx,
                "cfy_category": item["category"],
                "cfy_matched_parts": item["matched_parts"],
                "cfy_all_obj_files": item["all_obj_files"],
                "cfy_texture_count": int(item["texture_count"]),
                "cfy_trigger_texture_count": int(item["trigger_texture_count"]),
                "cfy_part_info": item["part_info"],
                "cfy_raw_obj_dir": item["raw_obj_dir"],
            }
        )
    return {
        "name": "0707-32-cfy-real-mesh-door-lid-cover",
            "split_json": str(split_json),
            "packed_index": str(packed_index),
            "selection_policy": (
            "16 phyx-verse + 16 realappliance unique objects; part_info must contain a door/lid/cover/hatch/"
            "Chinese door/lid part; drawer parts are excluded; all source OBJ/MTL files must exist and the "
            "trigger part must have a resolvable texture map"
        ),
        "sample_selection_unit": "objects",
        "datasets": [
            {"dataset_id": dataset_id, "data_root": str(DATA_ROOTS[dataset_id]), "target": target}
            for dataset_id, target in TARGETS.items()
        ],
        "audit": audit,
        "samples": {"train": rows, "held": []},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select 32 cfy EE samples with real textured source meshes.")
    parser.add_argument("--packed-index", type=Path, default=DEFAULT_PACKED_INDEX)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selection-name", default="selection.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected, audit = _select(args.packed_index)
    payload = _selection_payload(selected=selected, audit=audit, split_json=args.split_json, packed_index=args.packed_index)
    selection_path = args.out_dir / args.selection_name
    _write_json(selection_path, payload)
    _write_json(args.out_dir / "selection_real_mesh_audit.json", {"audit": audit, "selected": selected})
    print(f"[select_ee_real_mesh_cfy] wrote {selection_path}")
    for item in audit:
        print(
            "[select_ee_real_mesh_cfy] "
            f"{item['dataset_id']} chosen={item['chosen_count']} candidates={item['candidate_count']} "
            f"scanned={item['scanned_objects']}"
        )
    for item in selected:
        matched = ", ".join(part["part_name"] for part in item["matched_parts"])
        print(
            f"{item['dataset_id']}::{item['obj_id']} angle={item['angle_idx']} "
            f"score={item['score']} tex={item['texture_count']} parts={matched}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
