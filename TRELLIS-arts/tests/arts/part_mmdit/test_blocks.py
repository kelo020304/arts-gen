import torch

from trellis.models.part_flow.part_mmdit import (
    DualStreamMMDiTBlock,
    GatedCrossPartBlock,
    PartConditionTokenBuilder,
)


def _enable_dual_stream_gates(block: DualStreamMMDiTBlock, value: float = 1.0) -> None:
    dim = block.part_mod.net[-1].out_features // 6
    for mod in (block.part_mod.net[-1], block.cond_mod.net[-1]):
        mod.bias.data[2 * dim : 3 * dim].fill_(value)
        mod.bias.data[5 * dim : 6 * dim].fill_(value)


def test_gated_crosspart_initial_noop():
    block = GatedCrossPartBlock(dim=32, num_heads=4).eval()
    x = torch.randn(3, 8, 32)
    part_valid = torch.ones(3, dtype=torch.bool)

    with torch.no_grad():
        y = block(x, part_valid)

    assert torch.allclose(y, x, atol=1e-5)


def test_gated_crosspart_masks_padding_when_gate_enabled():
    block = GatedCrossPartBlock(dim=32, num_heads=4).eval()
    block.gate.data.fill_(1.0)
    x = torch.randn(3, 8, 32)
    part_valid = torch.tensor([True, True, False])

    with torch.no_grad():
        y = block(x, part_valid)

    assert torch.allclose(y[2], x[2], atol=1e-5)
    assert torch.isfinite(y[:2]).all()


def test_gated_crosspart_attention_uses_only_valid_tokens(monkeypatch):
    block = GatedCrossPartBlock(dim=32, num_heads=4).eval()
    x = torch.randn(5, 8, 32)
    part_valid = torch.tensor([True, False, True, False, False])
    seen_token_counts = []
    original_forward = block.attn.forward

    def capture_forward(attn_x, key_padding_mask):
        seen_token_counts.append(int(attn_x.shape[1]))
        return original_forward(attn_x, key_padding_mask)

    monkeypatch.setattr(block.attn, "forward", capture_forward)

    with torch.no_grad():
        block(x, part_valid)

    assert seen_token_counts == [2 * 8]


def test_gated_crosspart_padding_path_supports_autocast_dtype():
    if not torch.cuda.is_available():
        return
    block = GatedCrossPartBlock(dim=32, num_heads=4).cuda().eval()
    block.gate.data.fill_(1.0)
    x = torch.randn(5, 8, 32, device="cuda")
    part_valid = torch.tensor([True, False, True, False, False], device="cuda")

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        y = block(x, part_valid)

    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_condition_token_builder_shapes_and_nulls():
    builder = PartConditionTokenBuilder(dim=32, name_dim=768).eval()
    name_tokens = torch.randn(2, 3, 5, 768)
    name_mask = torch.tensor(
        [
            [[True, True, True, False, False], [True, True, True, True, True], [False, False, False, False, False]],
            [[True, True, False, False, False], [True, True, True, True, False], [True, True, True, True, True]],
        ],
        dtype=torch.bool,
    )
    anchor = torch.rand(2, 3, 4, 4)
    anchor_valid = torch.ones(2, 3, 4, dtype=torch.bool)
    drop_name = torch.tensor([[False, True, False], [False, False, False]])
    drop_anchor = torch.tensor([[False, False, True], [False, False, False]])

    with torch.no_grad():
        cond, mask = builder(
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            drop_name=drop_name,
            drop_anchor=drop_anchor,
        )

    assert cond.shape == (6, 9, 32)
    assert mask.shape == (6, 9)
    assert mask.dtype == torch.bool
    assert mask[1, :5].all()
    assert mask[:, 5:].all()
    assert torch.isfinite(cond).all()


def test_dual_stream_mmdit_block_shape_and_zero_init_noop():
    block = DualStreamMMDiTBlock(dim=32, num_heads=4).eval()
    part = torch.randn(2, 8, 32)
    cond = torch.randn(2, 5, 32)
    t_emb = torch.randn(2, 32)
    cond_mask = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        part_out, cond_out = block(part, cond, t_emb, cond_mask)

    assert part_out.shape == part.shape
    assert cond_out.shape == cond.shape
    assert torch.allclose(part_out, part, atol=1e-6)
    assert torch.allclose(cond_out, cond, atol=1e-6)


def test_dual_stream_name_token_ablation_changes_part_output():
    builder = PartConditionTokenBuilder(dim=32, name_dim=768).eval()
    block = DualStreamMMDiTBlock(dim=32, num_heads=4).eval()
    _enable_dual_stream_gates(block)
    part = torch.randn(1, 8, 32)
    name_tokens = torch.randn(1, 1, 5, 768)
    name_mask = torch.ones(1, 1, 5, dtype=torch.bool)
    anchor = torch.rand(1, 1, 4, 4)
    anchor_valid = torch.ones(1, 1, 4, dtype=torch.bool)
    t_emb = torch.randn(1, 32)

    with torch.no_grad():
        cond_full, mask_full = builder(name_tokens, name_mask, anchor, anchor_valid)
        cond_null, mask_null = builder(
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            drop_name=torch.ones(1, 1, dtype=torch.bool),
        )
        out_full, _ = block(part, cond_full, t_emb, mask_full)
        out_null, _ = block(part, cond_null, t_emb, mask_null)

    assert torch.isfinite(out_null).all()
    assert out_null.abs().sum() > 0
    assert (out_full - out_null).abs().max() > 1e-5


def test_dual_stream_anchor_distinguishes_same_name_instance():
    builder = PartConditionTokenBuilder(dim=32, name_dim=768).eval()
    block = DualStreamMMDiTBlock(dim=32, num_heads=4).eval()
    _enable_dual_stream_gates(block)
    part = torch.randn(1, 8, 32)
    name_tokens = torch.randn(1, 1, 5, 768)
    name_mask = torch.ones(1, 1, 5, dtype=torch.bool)
    anchor_a = torch.zeros(1, 1, 4, 4)
    anchor_b = torch.ones(1, 1, 4, 4)
    anchor_valid = torch.ones(1, 1, 4, dtype=torch.bool)
    t_emb = torch.randn(1, 32)

    with torch.no_grad():
        cond_a, mask_a = builder(name_tokens, name_mask, anchor_a, anchor_valid)
        cond_b, mask_b = builder(name_tokens, name_mask, anchor_b, anchor_valid)
        out_a, _ = block(part, cond_a, t_emb, mask_a)
        out_b, _ = block(part, cond_b, t_emb, mask_b)

    assert (out_a - out_b).abs().max() > 1e-5
