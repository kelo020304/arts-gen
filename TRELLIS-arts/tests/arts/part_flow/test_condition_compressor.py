import torch

from trellis.models.part_flow.part_flow_predictor import ConditionTokenCompressor


def test_condition_compressor_outputs_three_tokens_per_view_plus_global():
    torch.manual_seed(0)
    comp = ConditionTokenCompressor(
        dim=32,
        num_heads=4,
        num_view_tokens=3,
        max_views=8,
    )
    cond = torch.randn(2, 4 * 17, 32)
    out = comp(cond, num_views=4)
    assert out.shape == (2, 13, 32)


def test_condition_compressor_uses_distinct_positional_embeddings():
    torch.manual_seed(1)
    comp = ConditionTokenCompressor(
        dim=16,
        num_heads=4,
        num_view_tokens=3,
        max_views=4,
    )
    cond = torch.ones(1, 2 * 5, 16)
    out = comp(cond, num_views=2)
    assert out.shape == (1, 7, 16)
    assert not torch.allclose(out[:, 0], out[:, 1])
    assert not torch.allclose(out[:, 0], out[:, -1])
