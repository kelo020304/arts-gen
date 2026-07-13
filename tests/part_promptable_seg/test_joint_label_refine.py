from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TRELLIS = ROOT / "TRELLIS-arts"
if str(TRELLIS) not in sys.path:
    sys.path.insert(0, str(TRELLIS))

from inference_pipeline.joint_label_refine import (  # noqa: E402
    joint_neighbor_pairs,
    refine_joint_labels,
    save_joint_partition,
)


def test_joint_neighbor_pairs_handles_unsorted_sparse_coords() -> None:
    coords = torch.tensor([[1, 1, 2], [1, 1, 0], [1, 1, 1], [8, 8, 8]], dtype=torch.long)

    left, right, weights = joint_neighbor_pairs(coords, neighborhood=6)

    pairs = {frozenset((int(a), int(b))) for a, b in zip(left.tolist(), right.tolist())}
    assert pairs == {frozenset((0, 2)), frozenset((1, 2))}
    assert torch.allclose(weights, torch.ones_like(weights))


def test_refine_joint_labels_changes_only_low_confidence_interface_voxel() -> None:
    coords = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [10, 10, 10]],
        dtype=torch.long,
    )
    logits = torch.tensor(
        [
            [6.0, -6.0],
            [0.0, 0.1],
            [6.0, -6.0],
            [-6.0, 6.0],
        ],
        dtype=torch.float32,
    )

    labels, record = refine_joint_labels(
        logits,
        coords,
        iterations=2,
        pairwise_weight=1.0,
        margin_threshold=0.2,
        neighborhood=6,
        preserve_small_classes=0,
    )

    assert logits.argmax(dim=1).tolist() == [0, 1, 0, 1]
    assert labels.tolist() == [0, 0, 0, 1]
    assert record["changed_voxels"] == 1
    assert record["ambiguous_voxels"] == 1
    assert record["class_counts_after"] == [3, 1]


def test_refine_joint_labels_preserves_high_confidence_tiny_class() -> None:
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=torch.long)
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0]], dtype=torch.float32)

    labels, record = refine_joint_labels(
        logits,
        coords,
        iterations=3,
        pairwise_weight=10.0,
        margin_threshold=0.2,
        neighborhood=6,
    )

    assert labels.tolist() == [0, 1, 0]
    assert record["changed_voxels"] == 0


def test_refine_joint_labels_locks_low_confidence_small_prediction_class() -> None:
    coords = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [10, 10, 10]],
        dtype=torch.long,
    )
    logits = torch.tensor(
        [[6.0, -6.0], [0.0, 0.1], [6.0, -6.0], [6.0, -6.0]],
        dtype=torch.float32,
    )

    labels, record = refine_joint_labels(
        logits,
        coords,
        iterations=2,
        pairwise_weight=2.0,
        margin_threshold=0.2,
        preserve_small_classes=1,
    )

    assert labels.tolist() == [0, 1, 0, 0]
    assert record["small_class_locked_voxels"] == 1


def test_refine_joint_labels_restores_classes_without_erasing_last_donor_voxel() -> None:
    coords = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]],
        dtype=torch.long,
    )
    logits = torch.tensor(
        [
            [-1.2554297, 1.4766254, -1.3915968],
            [0.7348717, 0.4222187, 0.2235404],
            [1.1480154, 0.1074529, 0.0081839],
            [-2.0852270, -0.8392951, 1.3647149],
            [0.3299694, -0.4936928, 0.1389263],
            [-0.2584978, 0.7898425, 0.2602480],
        ],
        dtype=torch.float32,
    )

    labels, record = refine_joint_labels(
        logits,
        coords,
        iterations=1,
        pairwise_weight=10.0,
        margin_threshold=1.0,
        margin_quantile=0.0,
        preserve_small_classes=0,
    )

    assert torch.all(torch.bincount(logits.argmax(dim=1), minlength=3) > 0)
    assert torch.all(torch.bincount(labels, minlength=3) > 0)
    assert all(count > 0 for count in record["class_counts_after"])


def test_save_joint_partition_keeps_soft_logits_and_refinement_metadata(tmp_path: Path) -> None:
    coords = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    path = tmp_path / "joint_partition.npz"

    save_joint_partition(
        path,
        coords=coords,
        logits=logits,
        labels=labels,
        class_names=["body", "drawer"],
        refinement={"enabled": True, "changed_voxels": 0},
    )

    with np.load(path, allow_pickle=False) as data:
        assert data["coords"].shape == (2, 3)
        assert data["logits"].dtype == np.float16
        assert data["class_names"].tolist() == ["body", "drawer"]
        assert data["labels_refined"].tolist() == [0, 1]
        metadata = json.loads(str(data["refinement_json"].item()))
    assert metadata == {"changed_voxels": 0, "enabled": True}
