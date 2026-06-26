import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from trellis.trainers.arts.part_ss_latent_flow_eval import (
    compute_decode_metrics,
    decode_ss_latent_to_coords_with_stats,
    part_assignment_iou_matrix,
    summarize_assignment_matrix,
    summarize_bucketed_part_metrics,
    write_part_ss_inspection_sample,
    write_part_ss_inspection,
)


def test_write_part_ss_inspection_creates_index_and_png(tmp_path):
    rows = [{
        "obj_id": "100075",
        "sample_id": "sample-0",
        "target_part_name": "wheel_0",
        "target_slot": 1,
        "latent_mse": 0.1,
        "latent_l1": 0.05,
        "decode_iou_pred_vs_raw_ind": 0.2,
        "decode_iou_pred_vs_gt_decode": 0.3,
        "pred_count": 12,
        "raw_ind_count": 10,
        "zero_mse": 0.2,
        "mse_vs_zero": 0.5,
        "pred_logit_max": 1.5,
        "gt_logit_max": 2.0,
        "pred_count_at_thresholds": {0.0: 12, -0.5: 18},
        "png_name": "100075_wheel_0.png",
    }]
    panels = [{
        "obj_id": "100075",
        "target_part_name": "wheel_0",
        "pred_coords": np.array([[1, 2, 3]], dtype=np.int64),
        "gt_coords": np.array([[1, 2, 3]], dtype=np.int64),
        "global_coords": np.array([[0, 0, 0]], dtype=np.int64),
        "raw_gt_coords": np.array([[1, 2, 3]], dtype=np.int64),
        "rgb_views": [],
        "mask_views": [],
    }]
    object_panels = [{
        "obj_id": "100075",
        "part_count": 1,
        "surface_coords": np.array([[0, 0, 0], [0, 0, 1], [1, 2, 3]], dtype=np.int64),
        "parts": [{
            "name": "wheel_0",
            "pred_coords": np.array([[1, 2, 3]], dtype=np.int64),
            "gt_coords": np.array([[1, 2, 3]], dtype=np.int64),
            "raw_coords": np.array([[1, 2, 3]], dtype=np.int64),
        }],
        "png_name": "100075_complete_voxels.png",
    }]
    index_path = write_part_ss_inspection(tmp_path, 500, rows, panels, object_panels=object_panels)
    assert index_path.name == "index.txt"
    assert index_path.is_file()
    index_text = index_path.read_text(encoding="utf-8")
    assert "zero_mse" in index_text
    assert "pred_logit_max" in index_text
    assert "@-0.5" in index_text
    assert "Target Parts Voxel Panels" in index_text
    assert "target_parts_iou" in index_text
    assert "100075_complete_voxels.png" in index_text
    png_path = tmp_path / "step_000500" / "100075_wheel_0.png"
    assert png_path.is_file()
    with Image.open(png_path) as img:
        assert img.size[0] >= 2000
        assert img.size[1] >= 1800
    object_png_path = tmp_path / "step_000500" / "100075_complete_voxels.png"
    assert object_png_path.is_file()
    with Image.open(object_png_path) as img:
        assert img.size[0] >= 2000
        assert img.size[1] >= 600


def test_decode_iou_pred_vs_raw_ind_uses_dataset_ind_voxels():
    pred_decode = torch.tensor([[1, 1, 1], [2, 2, 2], [9, 9, 9]], dtype=torch.long)
    gt_decode = torch.tensor([[1, 1, 1], [5, 5, 5]], dtype=torch.long)
    raw_ind = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.long)
    metrics = compute_decode_metrics(
        pred_coords=pred_decode,
        gt_decode_coords=gt_decode,
        raw_ind_coords=raw_ind,
    )
    assert metrics["decode_iou_pred_vs_gt_decode"] == 0.25
    assert metrics["decode_iou_pred_vs_raw_ind"] == 0.5
    assert metrics["raw_ind_count"] == 3


def test_eval_bucket_summary_uses_size_and_k_rules():
    rows = [
        {
            "raw_ind_count": 10,
            "object_part_count": 1,
            "decode_iou_pred_vs_raw_ind": 0.1,
            "decode_recall_pred_vs_raw_ind": 0.2,
            "latent_cos": 0.9,
        },
        {
            "raw_ind_count": 100,
            "object_part_count": 4,
            "decode_iou_pred_vs_raw_ind": 0.3,
            "decode_recall_pred_vs_raw_ind": 0.4,
            "latent_cos": 0.8,
        },
        {
            "raw_ind_count": 1000,
            "object_part_count": 17,
            "decode_iou_pred_vs_raw_ind": 0.5,
            "decode_recall_pred_vs_raw_ind": 0.6,
            "latent_cos": 0.7,
        },
    ]
    summary = summarize_bucketed_part_metrics(rows, size_boundaries=(50.0, 500.0))
    assert summary["iou_size_small"] == 0.1
    assert summary["recall_size_medium"] == 0.4
    assert summary["cos_size_medium"] == 0.8
    assert summary["iou_size_large"] == 0.5
    assert summary["iou_k_1_2"] == 0.1
    assert summary["cos_k_1_2"] == 0.9
    assert summary["iou_k_3_5"] == 0.3
    assert summary["iou_k_16_plus"] == 0.5


def test_assignment_matrix_exposes_part_identity_swap():
    raw_a = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)
    raw_b = torch.tensor([[9, 9, 9], [9, 9, 8]], dtype=torch.long)
    matrix = part_assignment_iou_matrix([raw_b, raw_a], [raw_a, raw_b])
    assert matrix.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    summary = summarize_assignment_matrix(matrix)
    assert summary["assignment_diag_iou"] == 0.0
    assert summary["assignment_offdiag_max"] == 1.0


def test_decode_stats_expose_threshold_counts_and_logit_margin():
    class FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, z):
            logits = torch.tensor(
                [[[[[-0.75, -0.25], [0.25, 1.25]], [[-1.5, -0.1], [0.0, 0.5]]]]],
                device=z.device,
                dtype=z.dtype,
            )
            return logits.expand(z.shape[0], -1, -1, -1, -1)

    coords, stats = decode_ss_latent_to_coords_with_stats(
        FakeDecoder(),
        torch.zeros(1, 2, 2, 2),
        threshold=0.0,
        debug_thresholds=(0.0, -0.5),
    )
    assert coords.shape[0] == 3
    assert stats["logit_max"] == 1.25
    assert stats["counts"][0.0] == 3
    assert stats["counts"][-0.5] == 6


def test_sample_writer_passes_complete_object_panel_to_part_png(tmp_path, monkeypatch):
    import trellis.trainers.arts.part_ss_latent_flow_eval as eval_viz

    rows = [{
        "obj_id": "100075",
        "target_part_name": "wheel_0",
        "png_name": "part.png",
    }]
    panels = [{
        "obj_id": "100075",
        "target_part_name": "wheel_0",
    }]
    object_panel = {
        "obj_id": "100075",
        "part_count": 1,
        "png_name": "complete.png",
        "surface_coords": np.array([[0, 0, 0]], dtype=np.int64),
        "parts": [],
    }
    captured = []

    def fake_object_png(panel, out_path):
        out_path.write_bytes(b"complete")

    def fake_part_png(row, panel, out_path, complete_object_panel=None):
        captured.append(complete_object_panel)
        out_path.write_bytes(b"part")

    monkeypatch.setattr(eval_viz, "_make_object_panel_png", fake_object_png)
    monkeypatch.setattr(eval_viz, "_make_panel_png", fake_part_png)

    write_part_ss_inspection_sample(
        tmp_path,
        500,
        rows,
        panels,
        object_panel=object_panel,
    )

    assert captured == [object_panel]


def test_sample_writer_can_emit_complete_only_without_part_png(tmp_path, monkeypatch):
    import trellis.trainers.arts.part_ss_latent_flow_eval as eval_viz

    object_panel = {
        "obj_id": "100075",
        "part_count": 1,
        "png_name": "complete.png",
        "surface_coords": np.array([[0, 0, 0]], dtype=np.int64),
        "parts": [],
    }
    part_calls = []

    def fake_object_png(panel, out_path):
        out_path.write_bytes(b"complete")

    def fake_part_png(*args, **kwargs):
        part_calls.append(args)

    monkeypatch.setattr(eval_viz, "_make_object_panel_png", fake_object_png)
    monkeypatch.setattr(eval_viz, "_make_panel_png", fake_part_png)

    write_part_ss_inspection_sample(
        tmp_path,
        500,
        [],
        [],
        object_panel=object_panel,
    )

    assert (tmp_path / "step_000500" / "complete.png").is_file()
    assert part_calls == []


def test_complete_object_index_reports_per_part_raw_iou(tmp_path, monkeypatch):
    import trellis.trainers.arts.part_ss_latent_flow_eval as eval_viz

    object_panel = {
        "obj_id": "100075",
        "part_count": 2,
        "png_name": "complete.png",
        "surface_coords": np.array([[0, 0, 0], [1, 1, 1], [9, 9, 9]], dtype=np.int64),
        "parts": [
            {
                "name": "door_0",
                "pred_coords": np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
                "raw_coords": np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
            },
            {
                "name": "knob_0",
                "pred_coords": np.array([[8, 8, 8]], dtype=np.int64),
                "raw_coords": np.array([[9, 9, 9]], dtype=np.int64),
            },
        ],
    }

    def fake_object_png(panel, out_path):
        eval_viz._populate_complete_object_metrics(panel)
        out_path.write_bytes(b"complete")

    monkeypatch.setattr(eval_viz, "_make_object_panel_png", fake_object_png)

    index_path = write_part_ss_inspection(
        tmp_path,
        500,
        rows=[],
        panels=[],
        object_panels=[object_panel],
    )

    text = index_path.read_text(encoding="utf-8")
    assert "## Complete Object Part Metrics" in text
    assert "| 100075 | door_0 | 1.000000 |" in text
    assert "| 100075 | knob_0 | 0.000000 |" in text


def test_target_parts_iou_ignores_gt_body_context():
    import trellis.trainers.arts.part_ss_latent_flow_eval as eval_viz

    body_coords = np.array([[0, 0, idx] for idx in range(20)], dtype=np.int64)
    raw_part = np.array([[9, 9, 9]], dtype=np.int64)
    wrong_pred = np.array([[8, 8, 8]], dtype=np.int64)
    object_panel = {
        "obj_id": "100075",
        "part_count": 1,
        "surface_coords": np.concatenate([body_coords, raw_part], axis=0),
        "parts": [{
            "name": "door_0",
            "pred_coords": wrong_pred,
            "raw_coords": raw_part,
        }],
    }

    eval_viz._populate_complete_object_metrics(object_panel)

    assert object_panel["target_parts_iou_pred_vs_gt"] == 0.0
    assert object_panel["target_parts_precision_pred_vs_gt"] == 0.0
    assert object_panel["target_parts_recall_pred_vs_gt"] == 0.0
    assert object_panel["pred_target_parts_count"] == 1
    assert object_panel["gt_target_parts_count"] == 1


def test_batch_writer_matches_complete_object_panel_by_obj_id(tmp_path, monkeypatch):
    import trellis.trainers.arts.part_ss_latent_flow_eval as eval_viz

    rows = [
        {"obj_id": "100075", "target_part_name": "wheel_0", "png_name": "a.png"},
        {"obj_id": "100099", "target_part_name": "door_0", "png_name": "b.png"},
    ]
    panels = [
        {"obj_id": "100075", "target_part_name": "wheel_0"},
        {"obj_id": "100099", "target_part_name": "door_0"},
    ]
    object_panels = [
        {
            "obj_id": "100099",
            "part_count": 1,
            "png_name": "complete_b.png",
            "surface_coords": np.array([[0, 0, 0]], dtype=np.int64),
            "parts": [],
        },
        {
            "obj_id": "100075",
            "part_count": 1,
            "png_name": "complete_a.png",
            "surface_coords": np.array([[1, 1, 1]], dtype=np.int64),
            "parts": [],
        },
    ]
    captured = []

    def fake_object_png(panel, out_path):
        out_path.write_bytes(b"complete")

    def fake_part_png(row, panel, out_path, complete_object_panel=None):
        captured.append(complete_object_panel["obj_id"] if complete_object_panel else None)
        out_path.write_bytes(b"part")

    monkeypatch.setattr(eval_viz, "_make_object_panel_png", fake_object_png)
    monkeypatch.setattr(eval_viz, "_make_panel_png", fake_part_png)

    write_part_ss_inspection(
        tmp_path,
        500,
        rows,
        panels,
        object_panels=object_panels,
    )

    assert captured == ["100075", "100099"]
