"""D-08: slot 0 of part_tokens is learnable empty_token, not mask-pooled."""

import torch

from trellis.models.part_flow.part_flow_predictor import PartFlowPredictor


def test_empty_token_exists():
    model = PartFlowPredictor(k_max=4, hidden_dim=12, num_layers=0, num_heads=2, cond_dim=6)
    assert isinstance(model.empty_token, torch.nn.Parameter)
    assert tuple(model.empty_token.shape) == (12,)


def test_empty_token_overrides_slot_0_regardless_of_mask():
    torch.manual_seed(0)
    model = PartFlowPredictor(k_max=4, hidden_dim=12, num_layers=0, num_heads=2, cond_dim=6)
    cond = torch.randn(2, 10, 6)
    cond[:, 0] = 1000.0
    labels = torch.zeros(2, 10, dtype=torch.long)
    labels[:, 0] = 0
    labels[:, 1] = 1
    labels[:, 2] = 2
    with torch.no_grad():
        cond_proj = model.rgb_proj(cond)
        part_tokens = model.build_part_tokens(cond_proj, labels, [3, 3])
    expected = model.empty_token.detach()
    assert torch.allclose(part_tokens[0, 0], expected, atol=1e-6)
    assert torch.allclose(part_tokens[1, 0], expected, atol=1e-6)
    assert part_tokens[:, 3].abs().max().item() < 1e-6
