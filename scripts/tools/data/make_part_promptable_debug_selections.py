#!/usr/bin/env python3
"""Create fixed debug row selections for promptable part segmentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_PATH = PROJECT_ROOT / "scripts" / "dev"
if str(DEV_PATH) not in sys.path:
    sys.path.insert(0, str(DEV_PATH))


OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_debug_selections")


def write(name: str, rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


def main() -> int:
    p2_4 = [
        {"obj_id": "100283", "angle_idx": 0, "part_name": "button_0"},
        {"obj_id": "100283", "angle_idx": 0, "part_name": "door_0"},
        {"obj_id": "101943", "angle_idx": 0, "part_name": "knob_0"},
        {"obj_id": "101943", "angle_idx": 0, "part_name": "door_0"},
    ]
    same_obj_8 = [
        {"obj_id": "100283", "angle_idx": 0, "part_name": "button_0"},
        {"obj_id": "100283", "angle_idx": 0, "part_name": "button_2"},
        {"obj_id": "100283", "angle_idx": 0, "part_name": "button_3"},
        {"obj_id": "100283", "angle_idx": 0, "part_name": "button_5"},
        {"obj_id": "100283", "angle_idx": 0, "part_name": "button_6"},
        {"obj_id": "100283", "angle_idx": 0, "part_name": "button_7"},
        {"obj_id": "100283", "angle_idx": 0, "part_name": "button_8"},
        {"obj_id": "100283", "angle_idx": 0, "part_name": "door_0"},
    ]
    cross_obj_8 = [
        {"obj_id": "100283", "angle_idx": 0, "part_name": "button_0"},
        {"obj_id": "101943", "angle_idx": 0, "part_name": "knob_0"},
        {"obj_id": "102701", "angle_idx": 0, "part_name": "button_0"},
        {"obj_id": "101049", "angle_idx": 5, "part_name": "wheel_0"},
        {"obj_id": "101106", "angle_idx": 0, "part_name": "rotation_blade_0"},
        {"obj_id": "101253", "angle_idx": 9, "part_name": "rotation_blade_6"},
        {"obj_id": "100279", "angle_idx": 0, "part_name": "button_0"},
        {"obj_id": "100058", "angle_idx": 0, "part_name": "lid_0"},
    ]
    write("p2_4samples", p2_4)
    write("p2_same_obj_8samples", same_obj_8)
    write("p2_cross_obj_8samples", cross_obj_8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
