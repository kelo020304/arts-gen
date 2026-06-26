"""D-05/D-09: surface_emb is nn.Embedding(2, H), summed into voxel tokens."""

import torch

from trellis.models.part_flow.part_flow_predictor import PartFlowPredictor


def _inputs():
    torch.manual_seed(0)
    k_max, K_b, N, VT, D = 5, 3, 8, 12, 8
    x_t = torch.zeros(N, k_max)
    x_t[:, :K_b] = torch.softmax(torch.randn(N, K_b), dim=-1)
    coords = torch.cat([
        torch.zeros(N, 1, dtype=torch.int32),
        torch.randint(0, 64, (N, 3), dtype=torch.int32),
    ], dim=1)
    cond = torch.randn(1, VT, D)
    mask_labels = torch.zeros(1, VT, dtype=torch.long)
    mask_labels[0, 0:3] = torch.tensor([1, 2, 1])
    return x_t, torch.tensor([0.5]), coords, cond, mask_labels, [K_b]


def test_surface_emb_exists_and_shape():
    model = PartFlowPredictor(k_max=8, hidden_dim=16, num_layers=0, num_heads=2, cond_dim=8)
    assert isinstance(model.surface_emb, torch.nn.Embedding)
    assert tuple(model.surface_emb.weight.shape) == (2, 16)


def test_surface_emb_zero_init():
    model = PartFlowPredictor(k_max=8, hidden_dim=16, num_layers=0, num_heads=2, cond_dim=8)
    assert torch.equal(model.surface_emb.weight, torch.zeros_like(model.surface_emb.weight))


def test_forward_accepts_is_on_surface():
    model = PartFlowPredictor(k_max=5, hidden_dim=16, num_layers=0, num_heads=2, cond_dim=8)
    x_t, t, coords, cond, mask_labels, num_parts = _inputs()
    is_on_surface = torch.zeros(x_t.shape[0], dtype=torch.long)
    out = model(x_t, t, coords, cond, mask_labels, num_parts, is_on_surface)
    assert out['endpoint_logits'].shape == (x_t.shape[0], 5)


def test_surface_emb_delta_affects_output():
    model = PartFlowPredictor(k_max=5, hidden_dim=16, num_layers=0, num_heads=2, cond_dim=8)
    with torch.no_grad():
        model.surface_emb.weight[1].copy_(torch.linspace(-0.5, 0.5, 16))
    x_t, t, coords, cond, mask_labels, num_parts = _inputs()
    zeros = torch.zeros(x_t.shape[0], dtype=torch.long)
    ones = torch.ones(x_t.shape[0], dtype=torch.long)
    out0 = model(x_t, t, coords, cond, mask_labels, num_parts, zeros)['endpoint_logits']
    out1 = model(x_t, t, coords, cond, mask_labels, num_parts, ones)['endpoint_logits']
    assert not torch.allclose(out0, out1)
