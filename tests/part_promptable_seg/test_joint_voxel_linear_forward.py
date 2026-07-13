from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
TRELLIS = ROOT / "TRELLIS-arts"
if str(TRELLIS) not in sys.path:
    sys.path.insert(0, str(TRELLIS))

from trellis.models.part_seg.promptable_latent_seg import PromptablePartLatentSegNet  # noqa: E402


def test_joint_voxel_forward_scores_shared_voxels_against_queries() -> None:
    if not torch.cuda.is_available():
        pytest.skip("joint voxel bf16 regression needs CUDA")
    device = torch.device("cuda:0")
    model = PromptablePartLatentSegNet(
        latent_channels=8,
        dim=64,
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
        use_body_prompt=True,
    ).to(device)
    model.train()

    z_global = torch.randn(1, 8, 16, 16, 16, device=device)
    masks = torch.zeros(3, 4, 32, 32, device=device)
    masks[0, 0, 4:12, 4:12] = 1.0
    masks[1, 1, 12:20, 4:12] = 1.0
    masks[2, 2, 4:12, 12:20] = 1.0
    candidate = torch.zeros(1, 16, 16, 16, dtype=torch.bool, device=device)
    candidate[:, 0:2, 0:2, 0:2] = True
    full_occ = torch.zeros(1, 1, 64, 64, 64, device=device)
    full_occ[:, :, 0:8, 0:8, 0:8] = 1.0

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model(
            z_global,
            masks,
            candidate_cells=candidate,
            full_occ=full_occ,
            joint_voxels=True,
        )
        loss = out["joint_logits"].float().square().mean()
    loss.backward()

    assert out["joint_coords"].shape == (512, 3)
    assert out["joint_logits"].shape == (512, 4)
    assert model.joint_score_voxel.weight.grad is not None
    assert model.voxel_out.weight.grad is None
