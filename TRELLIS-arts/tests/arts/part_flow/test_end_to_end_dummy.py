"""Phase 8 integration gate: model + bridge + loss + export helpers."""

import os
from pathlib import Path

import numpy as np
import torch

from trellis.trainers.arts.part_flow_losses import FlowMatchingLoss
from trellis.models.part_flow.bridges import build_bridge
from trellis.models.part_flow.part_flow_predictor import PartFlowPredictor


def _batch():
    torch.manual_seed(0)
    k_max, K_b, N, VT, D = 6, 4, 24, 32, 16
    coords = torch.cat([
        torch.zeros(N, 1, dtype=torch.int32),
        torch.randint(0, 64, (N, 3), dtype=torch.int32),
    ], dim=1)
    labels = torch.randint(0, K_b, (N,), dtype=torch.long)
    labels[0] = -1
    mask_token_labels = torch.zeros(1, VT, dtype=torch.long)
    mask_token_labels[0, 1:K_b] = torch.arange(1, K_b)
    return {
        'coords': coords,
        'cond': torch.randn(1, VT, D),
        'mask_token_labels': mask_token_labels,
        'per_voxel_labels': labels,
        'is_on_surface': (torch.arange(N) % 3 == 0).long(),
        'voxel_layout': [slice(0, N)],
        'num_parts': [K_b],
    }


def test_model_dataset_loss_integration():
    bridge = build_bridge('fisher', k_max=6, t_max=1.0)
    model = PartFlowPredictor(k_max=6, hidden_dim=16, num_layers=0, num_heads=2, cond_dim=16)
    criterion = FlowMatchingLoss(bridge, empty_weight=0.05, part_weight=1.0, focal_gamma=2.0)
    loss, metrics = criterion(model, _batch())
    assert torch.isfinite(loss)
    assert metrics['ignore_frac'] > 0
    loss.backward()
    assert model.empty_token.grad is not None
    assert model.empty_token.grad.abs().max().item() > 0


def test_export_postprocess_roundtrip(tmp_path):
    # Phase 9 09-10 update: export helpers migrated from
    # scripts/inference/export_part_flow.py (deleted) to
    # trellis.utils.arts.part_flow_postprocess.
    from trellis.utils.arts.part_flow_postprocess import write_dual_output
    soft = np.zeros((64, 64, 64, 8), dtype=np.float16)
    hard = np.zeros((64, 64, 64), dtype=np.int64)
    hard[8:10, 8:10, 8:10] = 3
    soft_path, hard_path = write_dual_output(soft, hard, tmp_path)
    assert soft_path.is_file()
    assert hard_path.is_file()
    assert np.load(soft_path)['probs'].shape == (64, 64, 64, 8)
    assert np.load(hard_path).dtype == np.int64


def test_full_test_suite_runner_exists():
    """Phase 09 D-33: pytest auto-discovers TRELLIS-arts/tests/arts/, so the old
    ``run_all.sh`` wrapper was retired. Sanity-check the conftest is in place
    instead — that's what unblocks the package-rooted import path."""
    conftest = Path(__file__).resolve().parents[1] / 'conftest.py'
    assert conftest.is_file(), f'expected top-level conftest at {conftest}'
