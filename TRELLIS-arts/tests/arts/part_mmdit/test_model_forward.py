import torch

from trellis.models.part_flow import PartMMDiTModel


def _make_model(**overrides):
    cfg = dict(
        resolution=16,
        latent_channels=8,
        model_channels=64,
        cond_dim=64,
        num_blocks=4,
        num_heads=4,
        patch_size=2,
        num_views=4,
        max_parts=5,
        cross_part_layers=(1, 3),
        clip_name_dim=768,
        use_fp16=False,
        use_checkpoint=False,
    )
    cfg.update(overrides)
    return PartMMDiTModel(**cfg)


def test_patchify_unpatchify_roundtrip():
    model = _make_model().eval()
    x = torch.randn(2, 8, 16, 16, 16)

    with torch.no_grad():
        tokens = model.patchify_latent(x)
        y = model.unpatchify_latent(tokens)

    assert tokens.shape == (2, 512, 64)
    assert y.shape == x.shape


def test_forward_shape():
    model = _make_model().eval()
    batch_size, part_count, num_views, name_len = 2, 3, 4, 6
    x = torch.randn(batch_size, part_count, 8, 16, 16, 16)
    t = torch.rand(batch_size)
    z_global = torch.randn(batch_size, 8, 16, 16, 16)
    cond = torch.randn(batch_size, num_views * 7, 64)
    name_tokens = torch.randn(batch_size, part_count, name_len, 768)
    name_mask = torch.ones(batch_size, part_count, name_len, dtype=torch.bool)
    anchor = torch.rand(batch_size, part_count, num_views, 4)
    anchor_valid = torch.ones(batch_size, part_count, num_views, dtype=torch.bool)
    part_valid = torch.ones(batch_size, part_count, dtype=torch.bool)

    with torch.no_grad():
        v = model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
        )

    assert v.shape == x.shape
    assert torch.isfinite(v).all()


def test_name_null_fallback_nondegenerate():
    model = _make_model().eval()
    batch_size, part_count, num_views, name_len = 1, 2, 4, 6
    x = torch.randn(batch_size, part_count, 8, 16, 16, 16)
    t = torch.rand(batch_size)
    z_global = torch.randn(batch_size, 8, 16, 16, 16)
    cond = torch.randn(batch_size, num_views * 7, 64)
    name_tokens = torch.randn(batch_size, part_count, name_len, 768)
    name_mask = torch.ones(batch_size, part_count, name_len, dtype=torch.bool)
    anchor = torch.rand(batch_size, part_count, num_views, 4)
    anchor_valid = torch.ones(batch_size, part_count, num_views, dtype=torch.bool)
    part_valid = torch.ones(batch_size, part_count, dtype=torch.bool)
    drop_name = torch.ones(batch_size, part_count, dtype=torch.bool)

    with torch.no_grad():
        full = model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
        )
        name_null = model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
            drop_name=drop_name,
        )

    assert torch.isfinite(name_null).all()
    assert name_null.abs().sum() > 0
    assert (full - name_null).abs().max() > 1e-5


def test_anchor_swap_changes_same_name_instance_output():
    model = _make_model().eval()
    batch_size, part_count, num_views, name_len = 1, 2, 4, 6
    x = torch.randn(batch_size, part_count, 8, 16, 16, 16)
    t = torch.rand(batch_size)
    z_global = torch.randn(batch_size, 8, 16, 16, 16)
    cond = torch.randn(batch_size, num_views * 7, 64)
    one_name = torch.randn(batch_size, 1, name_len, 768)
    name_tokens = one_name.expand(batch_size, part_count, name_len, 768).clone()
    name_mask = torch.ones(batch_size, part_count, name_len, dtype=torch.bool)
    anchor = torch.rand(batch_size, part_count, num_views, 4)
    swapped_anchor = anchor.flip(dims=[1])
    anchor_valid = torch.ones(batch_size, part_count, num_views, dtype=torch.bool)
    part_valid = torch.ones(batch_size, part_count, dtype=torch.bool)

    with torch.no_grad():
        full = model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
        )
        swapped = model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            swapped_anchor,
            anchor_valid,
            part_valid,
        )

    assert (full - swapped).abs().max() > 1e-5
