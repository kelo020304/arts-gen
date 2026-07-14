from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
TRELLIS = ROOT / "TRELLIS-arts"
os.environ.setdefault("SPCONV_ALGO", "native")
if str(TRELLIS) not in sys.path:
    sys.path.insert(0, str(TRELLIS))

from trellis.models.part_seg.promptable_latent_seg import (  # noqa: E402
    PromptablePartLatentSegNet,
    joint_local_depth_from_ckpt,
    joint_local_mode_from_ckpt,
)


def _joint_inputs(device: torch.device):
    z_global = torch.randn(1, 8, 16, 16, 16, device=device)
    masks = torch.zeros(3, 4, 32, 32, device=device)
    masks[0, 0, 4:12, 4:12] = 1.0
    masks[1, 1, 12:20, 4:12] = 1.0
    masks[2, 2, 4:12, 12:20] = 1.0
    candidate = torch.zeros(1, 16, 16, 16, dtype=torch.bool, device=device)
    candidate[:, 0:2, 0:2, 0:2] = True
    full_occ = torch.zeros(1, 1, 64, 64, 64, device=device)
    full_occ[:, :, 0:8, 0:8, 0:8] = 1.0
    return z_global, masks, candidate, full_occ


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

    z_global, masks, candidate, full_occ = _joint_inputs(device)

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


@pytest.mark.parametrize("local_mode", ["post_spconv", "edge_graph"])
def test_joint_local_zero_gate_is_warm_start_compatible(local_mode: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("joint local compatibility regression needs CUDA")
    device = torch.device("cuda:0")
    common = dict(
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
    )
    torch.manual_seed(7)
    legacy = PromptablePartLatentSegNet(**common).to(device).eval()
    local = PromptablePartLatentSegNet(
        **common,
        joint_local_mode=local_mode,
        joint_local_depth=2,
    ).to(device).eval()
    result = local.load_state_dict(legacy.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert result.missing_keys
    assert all(key.startswith("joint_local_") for key in result.missing_keys)

    torch.manual_seed(11)
    inputs = _joint_inputs(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        legacy_out = legacy(*inputs[:2], candidate_cells=inputs[2], full_occ=inputs[3], joint_voxels=True)
        local_out = local(*inputs[:2], candidate_cells=inputs[2], full_occ=inputs[3], joint_voxels=True)
    assert torch.equal(legacy_out["joint_coords"], local_out["joint_coords"])
    assert torch.equal(legacy_out["joint_logits"], local_out["joint_logits"])

    ckpt = {
        "args": {"joint_local_mode": local_mode, "joint_local_depth": 2},
        "model": local.state_dict(),
    }
    assert joint_local_mode_from_ckpt(ckpt) == local_mode
    assert joint_local_depth_from_ckpt(ckpt) == 2
    reloaded = PromptablePartLatentSegNet(
        **common,
        joint_local_mode=joint_local_mode_from_ckpt(ckpt),
        joint_local_depth=joint_local_depth_from_ckpt(ckpt),
    ).to(device)
    reloaded.load_state_dict(ckpt["model"], strict=True)


def test_legacy_checkpoint_defaults_to_no_joint_local_model() -> None:
    ckpt = {"args": {"joint_seg": True}, "model": {"body_prompt": torch.zeros(1, 64)}}
    assert joint_local_mode_from_ckpt(ckpt) == "none"
    assert joint_local_depth_from_ckpt(ckpt) == 2
