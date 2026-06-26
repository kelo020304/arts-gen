#!/usr/bin/env python3
"""Create fixed D2 Route-V memorization selections."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_PATH = PROJECT_ROOT / "scripts" / "dev"
if str(DEV_PATH) not in sys.path:
    sys.path.insert(0, str(DEV_PATH))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import enumerate_part_rows, make_base_dataset, pick_gate1_rows  # noqa: E402


OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_debug_selections")


def spec(row) -> dict[str, object]:
    return {"obj_id": row.obj_id, "angle_idx": int(row.angle_idx), "part_name": row.part_name}


def write(name: str, rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


def main() -> int:
    base = make_base_dataset()
    all_rows = enumerate_part_rows(base)
    gate1, _meta = pick_gate1_rows(all_rows)
    gate1_specs = [spec(row) for row in gate1]

    one = [{"obj_id": "100283", "angle_idx": 0, "part_name": "button_0"}]
    four = [
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
    write("d2_route_v_1sample", one)
    write("d2_route_v_4samples", four)
    write("d2_route_v_16samples", gate1_specs[:16])
    write("d2_route_v_68samples", gate1_specs)
    write("d2_route_v_same_obj_8samples", same_obj_8)
    write("d2_route_v_cross_obj_8samples", cross_obj_8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
