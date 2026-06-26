import torch


from scripts.train.part_promptable_seg.train_part_promptable_seg import (
    configure_no_prompt_tracker,
    record_no_prompt_from_output,
)
from scripts.train.part_promptable_seg.train_part_promptable_seg import apply_view_dropout


def test_view_dropout_keeps_at_least_one_nonempty_view():
    masks = torch.zeros(3, 4, 8, 8)
    masks[0, 2, 1:3, 1:3] = 1.0
    masks[1, 0, 2:4, 2:4] = 1.0
    masks[1, 3, 4:6, 4:6] = 1.0

    for _ in range(64):
        dropped = apply_view_dropout(masks, min_views=1)
        per_sample = dropped.flatten(2).sum(dim=2).sum(dim=1)
        assert per_sample[:2].gt(0).all()
        assert float(per_sample[2]) == 0.0
        assert torch.equal(dropped[0, 2], masks[0, 2])
        assert torch.equal(dropped[0, 0], torch.zeros_like(dropped[0, 0]))


def test_no_prompt_tracker_writes_trigger_manifest(tmp_path):
    configure_no_prompt_tracker(tmp_path, rank=3)
    batch = {
        "obj_id": ["o0", "o1"],
        "angle_idx": [0, 1],
        "sample_id": ["s0", "s1"],
        "part_name": ["button_0", "door_0"],
        "original_label": [5, 7],
        "view_indices": torch.tensor([[0, 4, 7, 11], [1, 3, 6, 9]], dtype=torch.long),
    }
    out = {"no_prompt_mask": torch.tensor([False, True])}

    hits = record_no_prompt_from_output(out, batch, context="test", step=12)

    assert hits == 1
    log_path = tmp_path / "logs" / "no_prompt_rank3.jsonl"
    text = log_path.read_text(encoding="utf-8")
    assert '"obj_id": "o1"' in text
    assert '"part_name": "door_0"' in text
