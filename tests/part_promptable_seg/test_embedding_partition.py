from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TRELLIS = ROOT / "TRELLIS-arts"
if str(TRELLIS) not in sys.path:
    sys.path.insert(0, str(TRELLIS))
from scripts.train.part_promptable_seg.train_part_promptable_seg import (  # noqa: E402
    ObjectGroupBatchSampler,
    embedding_partition_loss,
    pairwise_overlap_from_coords,
    partition_coords_by_embedding,
)
from trellis.models.part_seg.promptable_latent_seg import PromptablePartLatentSegNet  # noqa: E402


@dataclass(frozen=True)
class Row:
    obj_id: str
    angle_idx: int
    part_name: str
    dataset_id: str = "ds"


def _batch(n: int = 2) -> dict:
    return {
        "dataset_id": ["ds"] * n,
        "obj_id": ["obj"] * n,
        "angle_idx": torch.zeros((n,), dtype=torch.long),
    }


def test_voxel_embedding_head_forward_shape() -> None:
    model = PromptablePartLatentSegNet(
        latent_channels=8,
        dim=32,
        num_views=4,
        depth=1,
        head_depth=1,
        heads=4,
        mask_size=32,
        mask_encoder="fg_points",
        point_k_boundary=4,
        point_k_interior=4,
        use_voxel_head=True,
        voxel_depth=1,
        voxel_embedding_dim=16,
    ).eval()
    z_global = torch.randn(1, 8, 16, 16, 16)
    masks = torch.zeros(1, 4, 32, 32)
    masks[0, 0, 4:8, 4:8] = 1.0
    candidate = torch.zeros(1, 16, 16, 16, dtype=torch.bool)
    candidate[:, 0, 0, 0] = True
    occ = torch.zeros(1, 1, 64, 64, 64)
    occ[:, :, :4, :4, :4] = 1.0

    with torch.no_grad():
        out = model.forward_voxels(z_global, masks, candidate, occ)

    assert out["voxel_logits"].shape == (1, 64)
    assert out["voxel_embeddings"].shape == (1, 64, 16)
    assert torch.isfinite(out["voxel_embeddings"]).all()
    norms = torch.linalg.vector_norm(out["voxel_embeddings"], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5)


def test_embedding_partition_loss_has_pull_and_push_terms() -> None:
    coords_list = [
        torch.tensor([[1, 1, 1], [1, 1, 2]], dtype=torch.long),
        torch.tensor([[2, 1, 1], [2, 1, 2]], dtype=torch.long),
    ]
    raw_coords = [coords.clone() for coords in coords_list]
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.9, 0.1], [0.8, 0.2]],
        ],
        dtype=torch.float32,
    )

    loss, items = embedding_partition_loss(
        embeddings,
        coords_list,
        raw_coords,
        _batch(2),
        pull_margin=0.05,
        push_margin=1.5,
        max_voxels_per_part=0,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0
    assert items["embed_pull"] > 0.0
    assert items["embed_push"] > 0.0
    assert items["embed_groups"] == 1.0
    assert items["embed_parts"] == 2.0


def test_embedding_partition_inference_removes_pairwise_overlap() -> None:
    coords_list = [
        torch.tensor([[1, 1, 1], [1, 1, 2]], dtype=torch.long),
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    ]
    logits = torch.full((2, 2), 8.0)
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    batch = _batch(2)

    before = pairwise_overlap_from_coords(coords_list, batch)
    partitioned = partition_coords_by_embedding(logits, coords_list, embeddings, batch, threshold=0.5)
    after = pairwise_overlap_from_coords(partitioned, batch)

    assert before[0]["object_overlap_voxels"] == 1
    assert before[1]["part_overlap_voxels"] == 1
    assert after[0]["object_overlap_voxels"] == 0
    assert after[1]["part_overlap_voxels"] == 0
    assert sum(coords.shape[0] for coords in partitioned) == 3


def test_embedding_partition_without_embeddings_keeps_promptable_predictions() -> None:
    coords_list = [
        torch.tensor([[1, 1, 1], [1, 1, 2]], dtype=torch.long),
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    ]
    logits = torch.full((2, 2), 8.0)

    partitioned = partition_coords_by_embedding(logits, coords_list, None, _batch(2), threshold=0.5)

    assert torch.equal(partitioned[0], coords_list[0])
    assert torch.equal(partitioned[1], coords_list[1])


def test_object_group_batch_sampler_groups_same_object_angle_rows() -> None:
    rows = [
        Row("a", 0, "body"),
        Row("a", 0, "drawer_0"),
        Row("a", 0, "drawer_1"),
        Row("b", 0, "body"),
        Row("b", 0, "door_0"),
        Row("c", 1, "body"),
    ]
    sampler = ObjectGroupBatchSampler(rows, batch_size=4, shuffle=False, seed=7)
    batches = list(iter(sampler))

    assert batches == [[0, 1, 2], [3, 4, 5]]
    for group in ([0, 1, 2], [3, 4], [5]):
        assert any(all(idx in batch for idx in group) for batch in batches)
    assert all(len(batch) <= 4 for batch in batches)
