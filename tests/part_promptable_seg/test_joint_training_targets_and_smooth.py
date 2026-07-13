from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.part_promptable_seg.train_part_promptable_seg import (
    _build_joint_target,
    joint_partial_label_unary_loss,
    joint_pairwise_smooth_loss,
    summarize_joint_eval_rows,
)


def test_build_joint_target_ignores_multi_claim_voxels() -> None:
    coord_keys = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    target, overlap, claim_mask = _build_joint_target(
        coord_keys,
        kept_part_keys=[torch.tensor([1, 2]), torch.tensor([2, 3])],
        dropped_part_keys=[torch.tensor([3, 4])],
    )

    assert target.tolist() == [0, 1, -100, -100, -100]
    assert overlap.tolist() == [False, False, True, True, False]
    assert claim_mask.tolist() == [
        [False, False, False],
        [False, False, False],
        [False, True, True],
        [False, False, False],
        [False, False, False],
    ]


def test_overlap_partial_unary_keeps_mass_in_claim_classes() -> None:
    claim_mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    body_logits = torch.tensor([[5.0, 0.0, 0.0]], requires_grad=True)
    claim_logits = torch.tensor([[-5.0, 0.0, 0.0]])

    body_loss, body_items = joint_partial_label_unary_loss(
        {"class_logits": body_logits, "overlap_claim_mask": claim_mask}
    )
    claim_loss, claim_items = joint_partial_label_unary_loss(
        {"class_logits": claim_logits, "overlap_claim_mask": claim_mask}
    )

    assert body_loss is not None and claim_loss is not None
    assert body_loss > claim_loss
    assert body_items["joint_overlap_supervised_voxels"] == 1.0
    assert body_items["joint_overlap_claim_mass"] < claim_items["joint_overlap_claim_mass"]
    body_loss.backward()
    assert body_logits.grad is not None
    assert float(body_logits.grad.abs().sum()) > 0.0


def test_overlap_band_gets_spatial_gradient_without_hard_ownership() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=torch.long)
    target = torch.tensor([1, -100, 2], dtype=torch.long)
    claim_mask = torch.tensor(
        [
            [False, False, False],
            [False, True, True],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    logits = torch.tensor(
        [[-5.0, 5.0, -5.0], [3.0, 0.0, 0.0], [-5.0, -5.0, 5.0]],
        requires_grad=True,
    )
    pred = {
        "class_logits": logits,
        "coords": coords,
        "target": target,
        "overlap_claim_mask": claim_mask,
    }

    partial_loss, _ = joint_partial_label_unary_loss(pred)
    spatial_loss, items = joint_pairwise_smooth_loss(
        pred,
        same_label_weight=1.0,
        all_label_weight=0.0,
        cross_label_weight=1.0,
        neighborhood=6,
    )

    assert partial_loss is not None and spatial_loss is not None
    assert items["joint_smooth_overlap_pairs"] == 2.0
    assert items["joint_smooth_cross_pairs"] == 0.0
    (partial_loss + spatial_loss).backward()
    assert logits.grad is not None
    assert float(logits.grad[1].abs().sum()) > 0.0
    assert torch.equal(logits.grad[[0, 2]], torch.zeros_like(logits.grad[[0, 2]]))


def test_joint_pairwise_cross_label_loss_repels_neighbor_probabilities() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)
    target = torch.tensor([0, 1], dtype=torch.long)
    same_logits = torch.tensor([[5.0, -5.0], [5.0, -5.0]], requires_grad=True)
    split_logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]], requires_grad=True)

    same_loss, same_items = joint_pairwise_smooth_loss(
        {"class_logits": same_logits, "coords": coords, "target": target},
        same_label_weight=0.0,
        all_label_weight=0.0,
        cross_label_weight=1.0,
        neighborhood=6,
    )
    split_loss, split_items = joint_pairwise_smooth_loss(
        {"class_logits": split_logits, "coords": coords, "target": target},
        same_label_weight=0.0,
        all_label_weight=0.0,
        cross_label_weight=1.0,
        neighborhood=6,
    )

    assert same_loss is not None and split_loss is not None
    assert same_loss > split_loss
    assert same_items["joint_smooth_cross_pairs"] == 1.0
    assert split_items["joint_smooth_cross_pairs"] == 1.0
    same_loss.backward()
    assert same_logits.grad is not None


def test_all_label_potts_does_not_cancel_cross_label_repulsion() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)
    target = torch.tensor([0, 1], dtype=torch.long)
    combined_logits = torch.tensor([[1.0, -1.0], [0.5, -0.5]], requires_grad=True)
    cross_logits = combined_logits.detach().clone().requires_grad_(True)

    combined_loss, combined_items = joint_pairwise_smooth_loss(
        {"class_logits": combined_logits, "coords": coords, "target": target},
        same_label_weight=0.0,
        all_label_weight=1.0,
        cross_label_weight=1.0,
        neighborhood=6,
    )
    cross_loss, _ = joint_pairwise_smooth_loss(
        {"class_logits": cross_logits, "coords": coords, "target": target},
        same_label_weight=0.0,
        all_label_weight=0.0,
        cross_label_weight=1.0,
        neighborhood=6,
    )

    assert combined_loss is not None and cross_loss is not None
    assert torch.allclose(combined_loss, cross_loss)
    assert combined_items["joint_smooth_cross_pairs"] == 1.0
    combined_loss.backward()
    assert combined_logits.grad is not None
    assert float(combined_logits.grad.abs().sum()) > 0.0


def test_joint_pairwise_same_label_loss_attracts_neighbor_probabilities() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)
    target = torch.tensor([0, 0], dtype=torch.long)
    same_logits = torch.tensor([[5.0, -5.0], [5.0, -5.0]])
    split_logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])

    same_loss, _ = joint_pairwise_smooth_loss(
        {"class_logits": same_logits, "coords": coords, "target": target},
        same_label_weight=1.0,
        all_label_weight=0.0,
        neighborhood=6,
    )
    split_loss, _ = joint_pairwise_smooth_loss(
        {"class_logits": split_logits, "coords": coords, "target": target},
        same_label_weight=1.0,
        all_label_weight=0.0,
        neighborhood=6,
    )

    assert same_loss is not None and split_loss is not None
    assert same_loss < split_loss


def test_raw_joint_summary_reads_evaluate_row_keys() -> None:
    rows = [
        {
            "joint_class_kind": "drawer",
            "cell_iou": 0.75,
            "e2e_recall": 0.5,
            "joint_voxel_share": 0.25,
        }
    ]

    summary = summarize_joint_eval_rows(rows)

    assert summary["drawer"] == {"n": 1.0, "iou": 0.75, "recall": 0.5, "voxel_share": 0.25}
