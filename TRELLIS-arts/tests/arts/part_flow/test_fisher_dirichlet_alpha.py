import torch

from trellis.models.part_flow.bridges import FisherBridge, build_bridge


def test_fisher_bridge_accepts_dirichlet_alpha():
    bridge = FisherBridge(k_max=5, dirichlet_alpha=0.5)
    assert bridge.dirichlet_alpha == 0.5


def test_build_bridge_forwards_dirichlet_alpha():
    bridge = build_bridge('fisher', k_max=5, dirichlet_alpha=2.0)
    assert isinstance(bridge, FisherBridge)
    assert bridge.dirichlet_alpha == 2.0


def test_fisher_source_respects_valid_simplex_with_alpha():
    torch.manual_seed(0)
    bridge = FisherBridge(k_max=6, dirichlet_alpha=0.5)
    x0 = bridge.sample_source(
        num_parts=[3, 5],
        n_per_sample=[7, 11],
        device=torch.device('cpu'),
    )
    assert x0.shape == (18, 6)
    assert torch.allclose(x0[:7, :3].sum(dim=-1), torch.ones(7), atol=1e-5)
    assert torch.allclose(x0[7:, :5].sum(dim=-1), torch.ones(11), atol=1e-5)
    assert x0[:7, 3:].abs().max().item() == 0.0
    assert x0[7:, 5:].abs().max().item() == 0.0
