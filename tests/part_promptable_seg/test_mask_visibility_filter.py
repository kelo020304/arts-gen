from __future__ import annotations

import numpy as np

from scripts.train.part_promptable_seg.part_promptable_seg_utils import audit_promptable_mask_visibility


class DummyBase:
    def __init__(self, tmp_path):
        self.mask_root = tmp_path
        self.samples = [
            {"obj_id": "obj", "angle_idx": 0},
        ]


class Row:
    sample_idx = 0
    part_idx = 0
    obj_id = "obj"
    angle_idx = 0
    sample_id = "obj_angle_0"
    part_name = "hidden_part"
    semantic_type = "hidden"
    original_label = 7
    raw_count = 12
    view_indices = (0, 1)
    dataset_id = ""


def test_all_views_absent_is_undetectable_all_views(tmp_path):
    mask_dir = tmp_path / "obj" / "angle_0" / "mask"
    mask_dir.mkdir(parents=True)
    for idx in range(4):
        np.save(mask_dir / f"mask_{idx}.npy", np.zeros((4, 4), dtype=np.int16))

    audit = audit_promptable_mask_visibility(DummyBase(tmp_path), [Row()], expected_views=4)

    rec = audit["records"][0]
    assert rec["classification"] == "undetectable_all_views"
    assert rec["selected_visible_pixels"] == 0
    assert rec["all_visible_pixels"] == 0


def test_selected_visible_still_kept(tmp_path):
    mask_dir = tmp_path / "obj" / "angle_0" / "mask"
    mask_dir.mkdir(parents=True)
    for idx in range(4):
        arr = np.zeros((4, 4), dtype=np.int16)
        if idx == 1:
            arr[0, 0] = 7
        np.save(mask_dir / f"mask_{idx}.npy", arr)

    audit = audit_promptable_mask_visibility(DummyBase(tmp_path), [Row()], expected_views=4)

    rec = audit["records"][0]
    assert rec["classification"] == "visible_selected_views"
    assert rec["selected_visible_pixels"] == 1
    assert rec["all_visible_pixels"] == 1
