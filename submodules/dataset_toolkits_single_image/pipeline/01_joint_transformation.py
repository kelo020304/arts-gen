#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import load_config
from joint_utils import generate_transforms_json


JOINT_TYPE_TO_NAME = {
    "A": "free_rotation",
    "B": "prismatic",
    "C": "revolute",
    "CB": "compound",
    "D": "pivot",
    "E": "fixed",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate joint transform archives and part_info metadata.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the dataset toolkit YAML config.",
    )
    parser.add_argument(
        "--object-ids",
        help="Optional comma-separated object ID subset, e.g. 100064,100283",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate outputs even when joint_transforms/{id}.json already exists.",
    )
    return parser.parse_args(argv)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _require_object_ids(raw_value: str) -> list[str]:
    object_ids = [item.strip() for item in raw_value.split(",")]
    if not object_ids or any(not item for item in object_ids):
        raise ValueError("--object-ids must be a comma-separated list of non-empty IDs")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("--object-ids contains duplicate IDs")
    return object_ids


def _normalize_part_type(raw_name: str) -> str:
    return raw_name.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def _load_finaljson(finaljson_path: Path) -> dict[str, Any]:
    if not finaljson_path.is_file():
        raise FileNotFoundError(f"Missing finaljson: {finaljson_path}")
    with finaljson_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _require_mapping(data, f"finaljson[{finaljson_path}]")


def _collect_group_assignments(
    group_info: dict[str, Any],
    num_parts: int,
) -> dict[int, dict[str, Any]]:
    assignments: dict[int, dict[str, Any]] = {}

    for raw_gid, raw_value in group_info.items():
        gid = str(raw_gid)
        if gid == "0":
            part_indices = _require_list(raw_value, "group_info['0']")
            parent_group: str | None = None
            params: list[Any] = []
            joint_type = "E"
        else:
            group_entry = _require_list(raw_value, f"group_info['{gid}']")
            if len(group_entry) != 4:
                raise ValueError(
                    f"group_info['{gid}'] must have length 4, got {len(group_entry)}"
                )

            raw_part_indices = group_entry[0]
            if isinstance(raw_part_indices, int):
                part_indices = [raw_part_indices]
            else:
                part_indices = _require_list(
                    raw_part_indices,
                    f"group_info['{gid}'][0]",
                )

            parent_group_value = group_entry[1]
            if not isinstance(parent_group_value, (int, str)):
                raise TypeError(
                    f"group_info['{gid}'][1] must be a string or integer parent group"
                )
            parent_group = str(parent_group_value)

            params = _require_list(group_entry[2], f"group_info['{gid}'][2]")
            joint_type = _require_string(group_entry[3], f"group_info['{gid}'][3]")
            if joint_type not in JOINT_TYPE_TO_NAME:
                raise ValueError(f"Unsupported joint type for group {gid}: {joint_type}")

        for raw_part_idx in part_indices:
            part_idx = _require_int(raw_part_idx, f"group_info['{gid}'] part index")
            if part_idx < 0 or part_idx >= num_parts:
                raise ValueError(
                    f"group {gid} references part index {part_idx}, outside [0, {num_parts - 1}]"
                )
            if part_idx in assignments:
                raise ValueError(
                    f"part index {part_idx} is assigned to multiple groups: "
                    f"{assignments[part_idx]['group_id']} and {gid}"
                )
            assignments[part_idx] = {
                "group_id": gid,
                "parent_group": parent_group,
                "joint_type": joint_type,
                "joint_name": JOINT_TYPE_TO_NAME[joint_type],
                "joint_params": params,
            }

    missing = sorted(set(range(num_parts)) - set(assignments))
    if missing:
        raise ValueError(f"Missing group assignment for part indices: {missing}")

    return assignments


def build_part_info(finaljson_data: dict[str, Any], object_id: str) -> dict[str, Any]:
    parts = _require_list(finaljson_data.get("parts"), "finaljson['parts']")
    group_info = _require_mapping(finaljson_data.get("group_info"), "finaljson['group_info']")
    category = _require_string(finaljson_data.get("category"), "finaljson['category']")

    assignments = _collect_group_assignments(group_info, len(parts))

    name_indices: defaultdict[str, int] = defaultdict(int)
    label_to_key: dict[str, str] = {}
    part_entries: dict[str, Any] = {}

    for part_idx, part in enumerate(parts):
        part_data = _require_mapping(part, f"finaljson['parts'][{part_idx}]")
        raw_name = _require_string(part_data.get("name"), f"parts[{part_idx}]['name']")
        normalized_type = _normalize_part_type(raw_name)
        canonical_idx = name_indices[normalized_type]
        name_indices[normalized_type] += 1
        canonical_key = f"{normalized_type}_{canonical_idx}"

        raw_label = _require_int(part_data.get("label"), f"parts[{part_idx}]['label']")
        obj_files = _require_list(part_data.get("obj"), f"parts[{part_idx}]['obj']")
        if not obj_files:
            raise ValueError(f"parts[{part_idx}]['obj'] must not be empty")
        for obj_idx, obj_name in enumerate(obj_files):
            _require_string(obj_name, f"parts[{part_idx}]['obj'][{obj_idx}]")

        joint_info = assignments[part_idx]
        label_to_key[str(part_idx)] = canonical_key
        part_entries[canonical_key] = {
            "label": part_idx + 1,
            "part_index": part_idx,
            "raw_label": raw_label,
            "type": normalized_type,
            "joint": joint_info["joint_name"],
            "joint_type": joint_info["joint_type"],
            "joint_group_id": joint_info["group_id"],
            "parent_group": joint_info["parent_group"],
            "joint_params": joint_info["joint_params"],
            "obj_files": obj_files,
        }

    return {
        "object_id": object_id,
        "category": category,
        "num_parts": len(parts),
        "label_to_key": label_to_key,
        "parts": part_entries,
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def generate_joint_archive(
    object_id: str,
    finaljson_data: dict[str, Any],
    obj_dir: Path,
    num_angles: int,
) -> dict[str, Any]:
    parts = _require_list(finaljson_data.get("parts"), "finaljson['parts']")
    num_parts = len(parts)
    if num_angles < 1:
        raise ValueError(f"num_angles must be >= 1, got {num_angles}")

    angles: dict[str, Any] = {}
    for angle_idx in range(num_angles):
        angle_data = generate_transforms_json(
            object_id=object_id,
            angle_idx=angle_idx,
            jsondata=finaljson_data,
            num_parts=num_parts,
            obj_dir=os.fspath(obj_dir),
        )
        angles[str(angle_idx)] = {
            "joint_states": angle_data["joint_states"],
            "part_transforms": angle_data["part_transforms"],
        }

    return {
        "object_id": object_id,
        "num_parts": num_parts,
        "angles": angles,
    }


def _resolve_object_ids(config, object_ids_arg: str | None) -> list[str]:
    available_object_ids = config.list_object_ids()
    if object_ids_arg is None:
        return available_object_ids

    requested_object_ids = _require_object_ids(object_ids_arg)
    available_set = set(available_object_ids)
    missing_object_ids = [
        object_id for object_id in requested_object_ids if object_id not in available_set
    ]
    if missing_object_ids:
        missing_text = ", ".join(missing_object_ids)
        raise ValueError(f"Unknown object IDs in --object-ids: {missing_text}")
    return requested_object_ids


def process_object(config, object_id: str, force: bool) -> tuple[str, int]:
    finaljson_path = Path(config.finaljson_dir) / f"{object_id}.json"
    obj_dir = Path(config.partseg_dir) / object_id / "objs"
    joint_output_path = Path(config.joint_transforms_dir) / f"{object_id}.json"
    part_info_path = Path(config.part_info_dir) / object_id / "part_info.json"

    if joint_output_path.exists() and not force:
        return "skipped", 0

    finaljson_data = _load_finaljson(finaljson_path)
    is_articulated = config.is_articulated(object_id)
    if is_articulated and not obj_dir.is_dir():
        raise FileNotFoundError(f"Missing articulated OBJ directory: {obj_dir}")

    num_angles = config.get_num_angles(object_id)
    joint_archive = generate_joint_archive(
        object_id=object_id,
        finaljson_data=finaljson_data,
        obj_dir=obj_dir,
        num_angles=num_angles,
    )
    part_info = build_part_info(finaljson_data, object_id)

    save_json(joint_output_path, joint_archive)
    save_json(part_info_path, part_info)
    return "done", num_angles


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    object_ids = _resolve_object_ids(config, args.object_ids)

    Path(config.joint_transforms_dir).mkdir(parents=True, exist_ok=True)
    total = len(object_ids)

    for index, object_id in enumerate(object_ids, start=1):
        status, angle_count = process_object(config, object_id, args.force)
        if status == "skipped":
            print(f"[{index}/{total}] {object_id} skipped (existing)")
        else:
            print(f"[{index}/{total}] {object_id} done ({angle_count} angles)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
