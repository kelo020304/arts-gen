import torch

from trellis.trainers.arts.part_mmdit_losses import (
    foreground_weighted_part_mse,
    rectified_flow_loss,
)


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_name = None
        self.drop_anchor = None

    def forward(
        self,
        x_t_parts,
        t,
        z_global,
        cond,
        name_tokens,
        name_mask,
        anchor,
        anchor_valid,
        part_valid,
        *,
        drop_name=None,
        drop_anchor=None,
    ):
        self.drop_name = drop_name
        self.drop_anchor = drop_anchor
        return torch.zeros_like(x_t_parts)


def test_loss_finite_masks_padding_and_passes_independent_dropout():
    batch_size, part_count, num_views = 2, 3, 4
    x_1 = torch.randn(batch_size, part_count, 8, 16, 16, 16)
    part_valid = torch.tensor([[True, True, False], [True, False, False]])
    part_counts = torch.tensor([[100.0, 200.0, 0.0], [50.0, 0.0, 0.0]])
    model = CaptureModel()
    part_fg_mask = torch.ones(batch_size, part_count, 16, 16, 16, dtype=torch.bool)

    loss = rectified_flow_loss(
        model,
        x_1,
        torch.rand(batch_size),
        z_global=torch.randn(batch_size, 8, 16, 16, 16),
        cond=torch.randn(batch_size, num_views * 7, 64),
        name_tokens=torch.randn(batch_size, part_count, 5, 768),
        name_mask=torch.ones(batch_size, part_count, 5, dtype=torch.bool),
        anchor=torch.rand(batch_size, part_count, num_views, 4),
        anchor_valid=torch.ones(batch_size, part_count, num_views, dtype=torch.bool),
        part_valid=part_valid,
        part_raw_voxel_counts=part_counts,
        part_fg_mask=part_fg_mask,
        latent_scale=8.0,
        cfg_dropout_name=1.0,
        cfg_dropout_anchor=0.0,
        part_weight_kwargs=dict(
            mode="raw_voxel_count",
            alpha=0.5,
            min_w=0.5,
            max_w=3.0,
            ref_mode="median",
            normalize_per_object=True,
        ),
        object_balanced=True,
    )

    assert torch.isfinite(loss)
    assert model.drop_name.shape == part_valid.shape
    assert model.drop_anchor.shape == part_valid.shape
    assert model.drop_name[part_valid].all()
    assert not model.drop_anchor.any()


def test_loss_has_gradient_through_model_prediction():
    class BiasModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x_t_parts, t, *args, **kwargs):
            return torch.zeros_like(x_t_parts) + self.bias

    model = BiasModel()
    part_valid = torch.ones(1, 1, dtype=torch.bool)
    part_fg_mask = torch.ones(1, 1, 16, 16, 16, dtype=torch.bool)
    loss = rectified_flow_loss(
        model,
        torch.randn(1, 1, 8, 16, 16, 16),
        torch.rand(1),
        z_global=torch.randn(1, 8, 16, 16, 16),
        cond=torch.randn(1, 4 * 7, 64),
        name_tokens=torch.randn(1, 1, 5, 768),
        name_mask=torch.ones(1, 1, 5, dtype=torch.bool),
        anchor=torch.rand(1, 1, 4, 4),
        anchor_valid=torch.ones(1, 1, 4, dtype=torch.bool),
        part_valid=part_valid,
        part_raw_voxel_counts=torch.tensor([[10.0]]),
        part_fg_mask=part_fg_mask,
        latent_scale=8.0,
        part_weight_kwargs=dict(
            mode="none",
            alpha=0.5,
            min_w=0.5,
            max_w=3.0,
            ref_mode="median",
            normalize_per_object=True,
        ),
    )
    loss.backward()

    assert model.bias.grad is not None
    assert torch.isfinite(model.bias.grad)


def test_loss_scales_raw_latent_into_rf_space():
    class CaptureXT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_x_t = None

        def forward(self, x_t_parts, t, *args, **kwargs):
            self.seen_x_t = x_t_parts.detach().clone()
            return torch.zeros_like(x_t_parts)

    model = CaptureXT()
    part_valid = torch.ones(1, 1, dtype=torch.bool)
    x_1 = torch.ones(1, 1, 1, 1, 1, 1)
    part_fg_mask = torch.ones(1, 1, 1, 1, 1, dtype=torch.bool)

    rectified_flow_loss(
        model,
        x_1,
        torch.ones(1),
        z_global=torch.zeros(1, 1, 1, 1, 1),
        cond=torch.zeros(1, 1, 1),
        name_tokens=torch.zeros(1, 1, 5, 768),
        name_mask=torch.ones(1, 1, 5, dtype=torch.bool),
        anchor=torch.zeros(1, 1, 4, 4),
        anchor_valid=torch.ones(1, 1, 4, dtype=torch.bool),
        part_valid=part_valid,
        part_raw_voxel_counts=torch.tensor([[1.0]]),
        part_fg_mask=part_fg_mask,
        latent_scale=8.0,
        part_weight_kwargs=dict(
            mode="none",
            alpha=0.5,
            min_w=0.5,
            max_w=3.0,
            ref_mode="median",
            normalize_per_object=True,
        ),
    )

    assert torch.allclose(model.seen_x_t, torch.full_like(x_1, 8.0))


def test_foreground_weighted_mse_bg_one_matches_plain_mse():
    sq_error = torch.rand(2, 3, 4, 5, 5, 5)
    fg_mask = torch.zeros(2, 3, 5, 5, 5, dtype=torch.bool)
    fg_mask[:, :, 0, 0, 0] = True

    weighted = foreground_weighted_part_mse(
        sq_error,
        fg_mask,
        enabled=True,
        bg_weight=1.0,
    )
    plain = sq_error.mean(dim=(2, 3, 4, 5))

    assert torch.allclose(weighted, plain)


def test_foreground_weighted_mse_suppresses_background_contribution():
    sq_error = torch.ones(1, 1, 1, 2, 2, 2)
    sq_error[..., 0, 0, 0] = 9.0
    fg_mask = torch.zeros(1, 1, 2, 2, 2, dtype=torch.bool)
    fg_mask[..., 0, 0, 0] = True

    weighted = foreground_weighted_part_mse(
        sq_error,
        fg_mask,
        enabled=True,
        bg_weight=0.1,
    )
    plain = foreground_weighted_part_mse(
        sq_error,
        fg_mask,
        enabled=False,
        bg_weight=0.1,
    )

    assert weighted.item() > plain.item()
    assert torch.isclose(weighted, torch.tensor([[(9.0 + 0.7) / (1.0 + 0.7)]]))
