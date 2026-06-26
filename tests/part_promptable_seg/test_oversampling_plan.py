from __future__ import annotations

from dataclasses import dataclass
from scripts.train.part_promptable_seg.part_promptable_seg_utils import build_oversampling_plan, oversample_repeat_for_row


@dataclass(frozen=True)
class Row:
    obj_id: str
    angle_idx: int
    part_name: str
    semantic_type: str
    raw_count: int
    dataset_id: str = ""
    data_root: str = ""
    manifest_path: str = ""
    category: str = ""
    object_name: str = ""
    part_item_name: str = ""
    sample_part_names: str = ""


def test_oversampling_plan_stacks_domain_focus_and_small_repeats():
    rows = [
        Row("ra1", 0, "knob_0", "knob", 100, dataset_id="realappliance", category="Real Appliance"),
        Row("ra1", 0, "button_0", "button", 20, dataset_id="realappliance", category="Real Appliance"),
        Row("v1", 0, "panel_0", "panel", 700, dataset_id="phyx-verse", category="Kitchenware"),
        Row("v2", 0, "body_0", "body", 1200, dataset_id="phyx-verse", sample_part_names="body drawer"),
        Row("m1", 0, "body_0", "body", 1200, dataset_id="physx-mobility"),
    ]
    plan = build_oversampling_plan(
        rows,
        small_oversample=2,
        realappliance_oversample=3,
        verse_focus_oversample=2,
    )

    assert oversample_repeat_for_row(rows[0], realappliance_oversample=3, verse_focus_oversample=2, small_oversample=2) == 3
    assert oversample_repeat_for_row(rows[1], realappliance_oversample=3, verse_focus_oversample=2, small_oversample=2) == 6
    assert oversample_repeat_for_row(rows[2], realappliance_oversample=3, verse_focus_oversample=2, small_oversample=2) == 2
    assert oversample_repeat_for_row(rows[3], realappliance_oversample=3, verse_focus_oversample=2, small_oversample=2) == 2
    assert oversample_repeat_for_row(rows[4], realappliance_oversample=3, verse_focus_oversample=2, small_oversample=2) == 1
    assert plan["effective_rows"] == 14
    by_tier = {row["tier"]: row for row in plan["tiers"]}
    assert by_tier["realappliance"]["effective_rows"] == 9
    assert by_tier["verse_focus"]["effective_rows"] == 4
    assert by_tier["base"]["effective_rows"] == 1
