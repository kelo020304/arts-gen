import pytest
import torch
import torch.nn as nn

from trellis.trainers.arts.part_ss_latent_flow_losses import (
    PartSSLatentRFLoss,
    compute_object_loss_weights,
    compute_part_loss_weights,
    foreground_weighted_part_mse,
    sample_part_ss_latent,
)


class PerfectVelocity(nn.Module):
    def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots):
        return self.v_target.expand_as(x_t_parts)


class CaptureZGlobal(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output
        self.seen_z_global = None

    def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots):
        self.seen_z_global = z_global.detach().clone()
        return self.output.to(device=x_t_parts.device, dtype=x_t_parts.dtype).expand_as(x_t_parts)


class CapturePartTokenWeights(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output
        self.seen_part_token_weights = None

    def forward(
        self,
        x_t_parts,
        t,
        z_global,
        cond,
        mask_token_labels,
        part_valid,
        target_slots,
        *,
        part_token_weights=None,
    ):
        self.seen_part_token_weights = part_token_weights
        return self.output.to(device=x_t_parts.device, dtype=x_t_parts.dtype).expand_as(x_t_parts)


class FixedVelocity(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots, **kwargs):
        return self.output.to(device=x_t_parts.device, dtype=x_t_parts.dtype)


class TrainableFixedVelocity(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = nn.Parameter(output.clone())

    def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots, **kwargs):
        return self.output.to(device=x_t_parts.device, dtype=x_t_parts.dtype).expand_as(x_t_parts)


def _minimal_batch(x_1: torch.Tensor, valid: torch.Tensor, counts: torch.Tensor):
    return {
        "x_1_parts": x_1,
        "part_valid": valid,
        "part_raw_voxel_counts": counts,
        "part_fg_mask": torch.ones(
            x_1.shape[0],
            x_1.shape[1],
            *x_1.shape[3:],
            dtype=torch.bool,
        ),
        "z_global": torch.zeros(x_1.shape[0], 8, 16, 16, 16),
        "cond": torch.zeros(x_1.shape[0], 12, 1024),
        "mask_token_labels": torch.ones(x_1.shape[0], 12, dtype=torch.long),
        "target_slots": torch.ones(x_1.shape[:2], dtype=torch.long),
        "debug_t": torch.full((x_1.shape[0],), 0.5),
        "debug_noise": torch.zeros_like(x_1),
    }


def test_compute_part_loss_weights_boosts_small_part_and_normalizes():
    counts = torch.tensor([[10.0, 40.0, 160.0]])
    valid = torch.tensor([[True, True, True]])
    weights, stats = compute_part_loss_weights(
        counts,
        valid,
        mode="raw_voxel_count",
        alpha=0.5,
        min_w=0.5,
        max_w=3.0,
        ref_mode="median",
        normalize_per_object=True,
    )
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]
    assert torch.allclose(weights[0].mean(), torch.tensor(1.0), atol=1e-6)
    assert stats["part_count_zero"] == 0


def test_compute_part_loss_weights_clamps_large_part_min_weight():
    counts = torch.tensor([[10.0, 10000.0]])
    valid = torch.tensor([[True, True]])
    weights, _stats = compute_part_loss_weights(
        counts,
        valid,
        mode="raw_voxel_count",
        alpha=1.0,
        min_w=0.5,
        max_w=3.0,
        ref_mode="median",
        normalize_per_object=True,
    )
    assert weights[0, 1] < 1.0


def test_compute_part_loss_weights_count_zero_uses_unit_weight():
    counts = torch.tensor([[0.0, 10.0]])
    valid = torch.tensor([[True, True]])
    weights, stats = compute_part_loss_weights(
        counts,
        valid,
        mode="raw_voxel_count",
        alpha=0.5,
        min_w=0.5,
        max_w=3.0,
        ref_mode="median",
        normalize_per_object=True,
    )
    assert stats["part_count_zero"] == 1
    assert torch.isfinite(weights).all()


def test_compute_object_loss_weights_requires_valid_parts_for_sqrt_k():
    with pytest.raises(ValueError, match="K_valid"):
        compute_object_loss_weights(
            torch.tensor([[False, False]]),
            mode="sqrt_k",
            k_ref=5.0,
            min_w=0.75,
            max_w=2.0,
        )


def test_compute_object_loss_weights_sqrt_k_uses_fixed_ref_and_clamp():
    valid = torch.tensor([
        [True, False, False, False, False],
        [True, True, True, False, False],
        [True, True, True, True, True],
    ])
    weights = compute_object_loss_weights(valid, mode="sqrt_k", k_ref=1.0, min_w=0.75, max_w=2.0)
    assert torch.allclose(weights, torch.tensor([1.0, 3.0**0.5, 2.0]))


def test_part_weight_mode_none_matches_old_valid_mse():
    x_1 = torch.zeros(1, 2, 1, 1, 1, 2)
    pred = torch.tensor([[[[[[1.0, 3.0]]]], [[[[5.0, 7.0]]]]]])
    valid = torch.tensor([[True, True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[10.0, 100.0]]))
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        part_weight_mode="none",
        object_balanced=False,
        velocity_contrastive_weight=0.0,
        foreground_weight={"enabled": False},
    )
    loss, metrics = criterion(FixedVelocity(pred), batch)
    expected = torch.nn.functional.mse_loss(pred[valid], x_1[valid])
    assert torch.allclose(loss, expected)
    assert metrics["mse_unweighted"] == metrics["mse"]


def test_weighted_loss_preserves_unweighted_metrics():
    x_1 = torch.zeros(1, 2, 1, 1, 1, 2)
    pred = torch.tensor([[[[[[1.0, 1.0]]]], [[[[3.0, 3.0]]]]]])
    valid = torch.tensor([[True, True]])
    counts = torch.tensor([[10.0, 1000.0]])
    batch = _minimal_batch(x_1, valid, counts)
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        part_weight_mode="raw_voxel_count",
        part_weight_alpha=0.5,
        object_balanced=True,
    )
    loss, metrics = criterion(FixedVelocity(pred), batch)
    assert float(loss.item()) != metrics["mse_unweighted"]
    assert metrics["mse_unweighted"] == 5.0


def test_relative_endpoint_loss_normalizes_by_part_signal_energy():
    x_1 = torch.tensor([[[[[[1.0]]]], [[[[10.0]]]]]])
    pred = torch.zeros_like(x_1)
    valid = torch.tensor([[True, True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[10.0, 1000.0]]))
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        relative_endpoint_weight=2.0,
        relative_endpoint_eps=1.0e-8,
        velocity_contrastive_weight=0.0,
    )
    loss, metrics = criterion(FixedVelocity(pred), batch)
    assert metrics["relative_endpoint_loss"] == pytest.approx(0.25)
    assert metrics["relative_endpoint_weighted"] == pytest.approx(0.5)
    assert loss.item() == pytest.approx(metrics["mse"] + 0.5)


def test_relative_endpoint_loss_is_smaller_near_endpoint_t():
    x_1 = torch.tensor([[[[[[1.0]]]], [[[[10.0]]]]]])
    pred = torch.zeros_like(x_1)
    valid = torch.tensor([[True, True]])
    low_t_batch = _minimal_batch(x_1, valid, torch.tensor([[10.0, 1000.0]]))
    high_t_batch = _minimal_batch(x_1, valid, torch.tensor([[10.0, 1000.0]]))
    low_t_batch["debug_t"] = torch.full((1,), 0.1)
    high_t_batch["debug_t"] = torch.full((1,), 0.9)
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        relative_endpoint_weight=1.0,
        relative_endpoint_eps=1.0e-8,
    )

    _low_loss, low_metrics = criterion(FixedVelocity(pred), low_t_batch)
    _high_loss, high_metrics = criterion(FixedVelocity(pred), high_t_batch)

    assert low_metrics["relative_endpoint_loss"] == pytest.approx(0.81)
    assert high_metrics["relative_endpoint_loss"] == pytest.approx(0.01)
    assert high_metrics["relative_endpoint_loss"] < low_metrics["relative_endpoint_loss"]


def test_identity_contrastive_loss_penalizes_swapped_part_binding():
    x_1 = torch.tensor([[[[[[1.0]]]], [[[[3.0]]]]]])
    valid = torch.tensor([[True, True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[10.0, 1000.0]]))
    batch["debug_t"] = torch.zeros(1)

    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        velocity_contrastive_weight=0.0,
        identity_contrastive_weight=1.0,
        identity_contrastive_temperature=0.1,
        identity_contrastive_eps=1.0e-8,
    )

    perfect_loss, perfect_metrics = criterion(FixedVelocity(x_1), batch)
    swapped_pred = torch.tensor([[[[[[3.0]]]], [[[[1.0]]]]]])
    swapped_loss, swapped_metrics = criterion(FixedVelocity(swapped_pred), batch)

    assert perfect_metrics["identity_contrastive_loss"] < 0.1
    assert perfect_metrics["identity_contrastive_acc"] == 1.0
    assert swapped_metrics["identity_contrastive_loss"] > perfect_metrics["identity_contrastive_loss"]
    assert swapped_metrics["identity_contrastive_acc"] == 0.0
    assert swapped_loss.item() > perfect_loss.item()


def test_identity_contrastive_loss_skips_single_part_objects():
    x_1 = torch.tensor([[[[[[1.0]]]]]])
    valid = torch.tensor([[True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[10.0]]))
    batch["debug_t"] = torch.zeros(1)
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        identity_contrastive_weight=1.0,
        identity_contrastive_temperature=0.1,
        identity_contrastive_eps=1.0e-8,
    )

    loss, metrics = criterion(FixedVelocity(x_1), batch)

    assert loss.item() == pytest.approx(0.0)
    assert metrics["identity_contrastive_loss"] == pytest.approx(0.0)
    assert metrics["identity_contrastive_objects"] == 0
    assert metrics["identity_contrastive_acc"] != metrics["identity_contrastive_acc"]


def test_full_loss_recipe_is_finite_and_backwards_through():
    x_1 = torch.tensor([
        [[[[[1.0]]]], [[[[3.0]]]]],
        [[[[[2.0]]]], [[[[4.0]]]]],
    ])
    valid = torch.ones(2, 2, dtype=torch.bool)
    counts = torch.tensor([[10.0, 1000.0], [20.0, 2000.0]])
    batch = _minimal_batch(x_1, valid, counts)
    batch["debug_t"] = torch.full((2,), 0.25)
    model = TrainableFixedVelocity(torch.zeros(1, 2, 1, 1, 1, 1))
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        part_weight_mode="raw_voxel_count",
        part_weight_alpha=0.5,
        object_balanced=True,
        relative_endpoint_weight=0.25,
        identity_contrastive_weight=0.05,
        identity_contrastive_temperature=0.1,
    )

    loss, metrics = criterion(model, batch)
    loss.backward()

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert torch.isfinite(model.output.grad).all()
    assert metrics["loss_total"] == pytest.approx(loss.detach().item())
    assert metrics["relative_endpoint_loss"] >= 0.0
    assert metrics["identity_contrastive_objects"] == 2
    assert 0.0 <= metrics["identity_contrastive_acc"] <= 1.0


def test_weighted_loss_reports_size_bucket_metrics():
    x_1 = torch.zeros(1, 3, 1, 1, 1, 1)
    pred = torch.tensor([[[[[[1.0]]]], [[[[2.0]]]], [[[[3.0]]]]]])
    valid = torch.tensor([[True, True, True]])
    counts = torch.tensor([[10.0, 100.0, 1000.0]])
    batch = _minimal_batch(x_1, valid, counts)
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        part_weight_mode="none",
        object_balanced=True,
        size_bucket_boundaries=(20.0, 200.0),
    )
    _loss, metrics = criterion(FixedVelocity(pred), batch)
    assert metrics["mse_size_small"] == 1.0
    assert metrics["mse_size_medium"] == 4.0
    assert metrics["mse_size_large"] == 9.0


def test_foreground_weighted_part_mse_bg_one_matches_plain_mse():
    sq_error = torch.rand(2, 3, 4, 5, 5, 5)
    fg_mask = torch.zeros(2, 3, 5, 5, 5, dtype=torch.bool)
    fg_mask[:, :, 0, 0, 0] = True

    weighted = foreground_weighted_part_mse(sq_error, fg_mask, bg_weight=1.0)
    plain = sq_error.flatten(start_dim=2).mean(dim=2)

    assert torch.allclose(weighted, plain)


def test_foreground_weighted_part_mse_suppresses_background():
    sq_error = torch.ones(1, 1, 1, 2, 2, 2)
    sq_error[..., 0, 0, 0] = 9.0
    fg_mask = torch.zeros(1, 1, 2, 2, 2, dtype=torch.bool)
    fg_mask[..., 0, 0, 0] = True

    weighted = foreground_weighted_part_mse(sq_error, fg_mask, bg_weight=0.1)
    plain = foreground_weighted_part_mse(sq_error, fg_mask, enabled=False)

    assert weighted.item() > plain.item()
    assert torch.isclose(weighted, torch.tensor([[(9.0 + 0.7) / (1.0 + 0.7)]]))


def test_rf_loss_builds_foreground_mask_from_raw_coords():
    x_1 = torch.zeros(1, 1, 8, 16, 16, 16)
    pred = torch.ones_like(x_1)
    pred[..., 0, 0, 0] = 9.0
    batch = _minimal_batch(x_1, torch.ones(1, 1, dtype=torch.bool), torch.tensor([[2.0]]))
    batch.pop("part_fg_mask")
    batch["raw_ind_coords"] = [[torch.tensor([[0, 0, 0], [3, 3, 3]], dtype=torch.long)]]
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        part_weight_mode="none",
        velocity_contrastive_weight=0.0,
        foreground_weight={"enabled": True, "bg_weight": 0.1},
    )

    loss, metrics = criterion(FixedVelocity(pred), batch)

    assert torch.isfinite(loss)
    assert metrics["mse_unweighted"] > 1.0


def test_rf_loss_foreground_mask_matches_raw_coord_max_pool_to_latent_grid():
    x_1 = torch.zeros(1, 2, 8, 16, 16, 16)
    valid = torch.tensor([[True, True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[3.0, 2.0]]))
    batch.pop("part_fg_mask")
    batch["raw_ind_coords"] = [[
        torch.tensor([[0, 0, 0], [3, 3, 3], [4, 4, 4]], dtype=torch.long),
        torch.tensor([[63, 63, 63], [8, 0, 4]], dtype=torch.long),
    ]]
    criterion = PartSSLatentRFLoss(foreground_weight={"enabled": True, "bg_weight": 0.1})

    fg_mask = criterion._part_fg_mask_from_batch(batch, valid, device=x_1.device)
    expected = torch.zeros(1, 2, 16, 16, 16, dtype=torch.bool)
    expected[0, 0, 0, 0, 0] = True
    expected[0, 0, 1, 1, 1] = True
    expected[0, 1, 15, 15, 15] = True
    expected[0, 1, 2, 0, 1] = True

    assert torch.equal(fg_mask.cpu(), expected)


def test_rf_loss_bg_weight_one_matches_foreground_disabled_loss():
    x_1 = torch.zeros(1, 2, 1, 2, 2, 2)
    pred = torch.arange(16, dtype=torch.float32).reshape(1, 2, 1, 2, 2, 2)
    valid = torch.tensor([[True, True]])
    counts = torch.tensor([[10.0, 1000.0]])
    batch = _minimal_batch(x_1, valid, counts)
    batch["part_fg_mask"] = torch.zeros(1, 2, 2, 2, 2, dtype=torch.bool)
    batch["part_fg_mask"][0, 0, 0, 0, 0] = True
    batch["part_fg_mask"][0, 1, 1, 1, 1] = True
    common_kwargs = dict(
        t_min=0.0,
        t_max=1.0,
        part_weight_mode="raw_voxel_count",
        part_weight_alpha=0.5,
        object_balanced=True,
        relative_endpoint_weight=0.25,
        velocity_contrastive_weight=0.0,
        identity_contrastive_weight=0.0,
    )

    fg_disabled = PartSSLatentRFLoss(**common_kwargs, foreground_weight={"enabled": False})
    bg_one = PartSSLatentRFLoss(**common_kwargs, foreground_weight={"enabled": True, "bg_weight": 1.0})
    disabled_loss, disabled_metrics = fg_disabled(FixedVelocity(pred), batch)
    bg_one_loss, bg_one_metrics = bg_one(FixedVelocity(pred), batch)

    assert torch.allclose(bg_one_loss, disabled_loss)
    assert bg_one_metrics["mse"] == pytest.approx(disabled_metrics["mse"])
    assert bg_one_metrics["loss_total"] == pytest.approx(disabled_metrics["loss_total"])


def test_rf_loss_zero_for_perfect_velocity():
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, noise_scale=1.0, velocity_contrastive_weight=0.0)
    model = PerfectVelocity()
    batch = {
        "x_1_parts": torch.randn(2, 2, 8, 16, 16, 16),
        "part_valid": torch.ones(2, 2, dtype=torch.bool),
        "part_fg_mask": torch.ones(2, 2, 16, 16, 16, dtype=torch.bool),
        "z_global": torch.randn(2, 8, 16, 16, 16),
        "cond": torch.randn(2, 12, 1024),
        "mask_token_labels": torch.ones(2, 12, dtype=torch.long),
        "target_slots": torch.ones(2, 2, dtype=torch.long),
        "debug_t": torch.full((2,), 0.5),
        "debug_noise": torch.zeros(2, 2, 8, 16, 16, 16),
    }
    model.v_target = batch["x_1_parts"]
    loss, metrics = criterion(model, batch)
    assert loss.item() == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["latent_l1"] == 0.0
    assert metrics["parts"] == 4


def test_rf_loss_ignores_invalid_padded_parts():
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, noise_scale=1.0)
    model = PerfectVelocity()
    x_1 = torch.zeros(1, 2, 8, 16, 16, 16)
    x_1[:, 0] = 1.0
    batch = {
        "x_1_parts": x_1,
        "part_valid": torch.tensor([[True, False]]),
        "part_fg_mask": torch.ones(1, 2, 16, 16, 16, dtype=torch.bool),
        "z_global": torch.randn(1, 8, 16, 16, 16),
        "cond": torch.randn(1, 12, 1024),
        "mask_token_labels": torch.ones(1, 12, dtype=torch.long),
        "target_slots": torch.tensor([[1, 0]]),
        "debug_t": torch.full((1,), 0.5),
        "debug_noise": torch.zeros_like(x_1),
    }
    model.v_target = torch.zeros_like(x_1)
    model.v_target[:, 0] = 1.0
    model.v_target[:, 1] = 999.0
    loss, metrics = criterion(model, batch)
    assert loss.item() == 0.0
    assert metrics["parts"] == 1


def test_rf_loss_latent_scale_trains_scaled_target_but_reports_unscaled_endpoint():
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, noise_scale=1.0, latent_scale=4.0)
    model = PerfectVelocity()
    x_1 = torch.full((1, 1, 8, 16, 16, 16), 0.25)
    batch = {
        "x_1_parts": x_1,
        "part_valid": torch.ones(1, 1, dtype=torch.bool),
        "part_fg_mask": torch.ones(1, 1, 16, 16, 16, dtype=torch.bool),
        "z_global": torch.randn(1, 8, 16, 16, 16),
        "cond": torch.randn(1, 12, 1024),
        "mask_token_labels": torch.ones(1, 12, dtype=torch.long),
        "target_slots": torch.ones(1, 1, dtype=torch.long),
        "debug_t": torch.full((1,), 0.5),
        "debug_noise": torch.zeros_like(x_1),
    }
    model.v_target = x_1 * 4.0
    loss, metrics = criterion(model, batch)
    assert loss.item() == 0.0
    assert metrics["latent_l1"] == 0.0


def test_rf_loss_keeps_global_condition_in_raw_latent_scale():
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, noise_scale=1.0, latent_scale=4.0)
    x_1 = torch.zeros(1, 1, 8, 16, 16, 16)
    z_global = torch.full((1, 8, 16, 16, 16), 0.25)
    model = CaptureZGlobal(output=torch.zeros_like(x_1))
    batch = {
        "x_1_parts": x_1,
        "part_valid": torch.ones(1, 1, dtype=torch.bool),
        "part_fg_mask": torch.ones(1, 1, 16, 16, 16, dtype=torch.bool),
        "z_global": z_global,
        "cond": torch.randn(1, 12, 1024),
        "mask_token_labels": torch.ones(1, 12, dtype=torch.long),
        "target_slots": torch.ones(1, 1, dtype=torch.long),
        "debug_t": torch.full((1,), 0.5),
        "debug_noise": torch.zeros_like(x_1),
    }
    criterion(model, batch)
    assert torch.allclose(model.seen_z_global, z_global)


def test_rf_loss_forwards_part_token_weights_when_present():
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, noise_scale=1.0)
    x_1 = torch.zeros(1, 1, 8, 16, 16, 16)
    part_token_weights = torch.zeros(1, 1, 12)
    part_token_weights[0, 0, 3] = 1.0
    model = CapturePartTokenWeights(output=torch.zeros_like(x_1))
    batch = {
        "x_1_parts": x_1,
        "part_valid": torch.ones(1, 1, dtype=torch.bool),
        "part_fg_mask": torch.ones(1, 1, 16, 16, 16, dtype=torch.bool),
        "z_global": torch.randn(1, 8, 16, 16, 16),
        "cond": torch.randn(1, 12, 1024),
        "mask_token_labels": torch.zeros(1, 12, dtype=torch.long),
        "target_slots": torch.ones(1, 1, dtype=torch.long),
        "part_token_weights": part_token_weights,
        "debug_t": torch.full((1,), 0.5),
        "debug_noise": torch.zeros_like(x_1),
    }
    criterion(model, batch)
    assert model.seen_part_token_weights is part_token_weights


def test_sampler_returns_latent_shape():
    class ZeroVelocity(nn.Module):
        def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots):
            return torch.zeros_like(x_t_parts)

    z = sample_part_ss_latent(
        ZeroVelocity(),
        z_global=torch.randn(1, 8, 16, 16, 16),
        cond=torch.randn(1, 12, 1024),
        mask_token_labels=torch.ones(1, 12, dtype=torch.long),
        part_valid=torch.ones(1, 2, dtype=torch.bool),
        target_slots=torch.ones(1, 2, dtype=torch.long),
        num_steps=4,
        noise_scale=1.0,
    )
    assert z.shape == (1, 2, 8, 16, 16, 16)


def test_sampler_latent_scale_returns_decoder_scale_latent():
    class ConstantVelocity(nn.Module):
        def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots):
            return torch.full_like(x_t_parts, 2.0)

    z = sample_part_ss_latent(
        ConstantVelocity(),
        z_global=torch.zeros(1, 8, 16, 16, 16),
        cond=torch.randn(1, 12, 1024),
        mask_token_labels=torch.ones(1, 12, dtype=torch.long),
        part_valid=torch.ones(1, 1, dtype=torch.bool),
        target_slots=torch.ones(1, 1, dtype=torch.long),
        num_steps=2,
        noise_scale=0.0,
        latent_scale=4.0,
    )
    assert torch.allclose(z, torch.full_like(z, 0.5))


def test_sampler_keeps_global_condition_in_raw_latent_scale():
    z_global = torch.full((1, 8, 16, 16, 16), 0.25)
    model = CaptureZGlobal(output=torch.zeros(1, 1, 8, 16, 16, 16))
    sample_part_ss_latent(
        model,
        z_global=z_global,
        cond=torch.randn(1, 12, 1024),
        mask_token_labels=torch.ones(1, 12, dtype=torch.long),
        part_valid=torch.ones(1, 1, dtype=torch.bool),
        target_slots=torch.ones(1, 1, dtype=torch.long),
        num_steps=2,
        noise_scale=0.0,
        latent_scale=4.0,
    )
    assert torch.allclose(model.seen_z_global, z_global)


# ----------------------------------------------------------------------
# Fix 3: per-object slot<->part shuffle
# ----------------------------------------------------------------------
class CaptureInputs(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output
        self.seen = {}

    def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots, **kwargs):
        self.seen = {
            "x_t": x_t_parts.detach().clone(),
            "target_slots": target_slots.detach().clone(),
            "part_token_weights": None if kwargs.get("part_token_weights") is None
            else kwargs["part_token_weights"].detach().clone(),
        }
        return self.output.to(device=x_t_parts.device, dtype=x_t_parts.dtype).expand_as(x_t_parts)


def test_part_shuffle_permutes_slot_and_latent_consistently():
    torch.manual_seed(0)
    # Distinct per-part latents (constant per part) so we can detect a permutation.
    x_1 = torch.zeros(1, 3, 1, 1, 1, 1)
    x_1[0, 0] = 1.0
    x_1[0, 1] = 2.0
    x_1[0, 2] = 3.0
    valid = torch.tensor([[True, True, True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[10.0, 20.0, 30.0]]))
    batch["target_slots"] = torch.tensor([[5, 6, 7]])
    batch["debug_t"] = torch.ones(1)  # x_t == x_1 when t=1 and noise=0
    weights = torch.zeros(1, 3, 12)
    weights[0, 0, 0] = 1.0
    weights[0, 1, 1] = 1.0
    weights[0, 2, 2] = 1.0
    batch["part_token_weights"] = weights

    model = CaptureInputs(torch.zeros(1, 3, 1, 1, 1, 1))
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, part_shuffle=True, velocity_contrastive_weight=0.0)

    found_permutation = False
    base_latent = x_1[0, :, 0, 0, 0, 0]
    base_slots = torch.tensor([5, 6, 7])
    for _ in range(20):
        criterion(model, batch)
        seen_latent = model.seen["x_t"][0, :, 0, 0, 0, 0]
        seen_slots = model.seen["target_slots"][0]
        seen_weight_idx = model.seen["part_token_weights"][0].argmax(dim=1)
        # The same permutation must apply to latent, slot id, and token weights.
        perm = (seen_latent - 1.0).round().long()  # latent value k+1 -> original index k
        assert torch.equal(seen_slots, base_slots[perm])
        assert torch.equal(seen_weight_idx, perm)
        if not torch.equal(seen_latent, base_latent):
            found_permutation = True
    assert found_permutation, "part_shuffle never produced a non-identity permutation"


def test_part_shuffle_off_keeps_original_binding():
    x_1 = torch.zeros(1, 3, 1, 1, 1, 1)
    x_1[0, 0] = 1.0
    x_1[0, 1] = 2.0
    x_1[0, 2] = 3.0
    valid = torch.tensor([[True, True, True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[10.0, 20.0, 30.0]]))
    batch["target_slots"] = torch.tensor([[5, 6, 7]])
    batch["debug_t"] = torch.ones(1)
    model = CaptureInputs(torch.zeros(1, 3, 1, 1, 1, 1))
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, part_shuffle=False, velocity_contrastive_weight=0.0)
    criterion(model, batch)
    assert torch.equal(model.seen["x_t"][0, :, 0, 0, 0, 0], x_1[0, :, 0, 0, 0, 0])
    assert torch.equal(model.seen["target_slots"][0], torch.tensor([5, 6, 7]))


# ----------------------------------------------------------------------
# Fix 4: DeltaFM velocity-contrastive identity loss
# ----------------------------------------------------------------------
def test_velocity_contrastive_prefers_aligned_velocity_assignment():
    # noise=0, t=1 -> v_target = x_1; distinct per-part targets.
    x_1 = torch.zeros(1, 2, 4, 1, 1, 1)
    x_1[0, 0, :, 0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x_1[0, 1, :, 0, 0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    valid = torch.tensor([[True, True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[10.0, 20.0]]))
    batch["debug_t"] = torch.ones(1)
    batch["debug_noise"] = torch.zeros_like(x_1)
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        velocity_contrastive_weight=1.0,
        velocity_contrastive_lambda=0.05,
    )
    aligned_loss, aligned_metrics = criterion(FixedVelocity(x_1), batch)
    swapped = x_1.flip(dims=[1])
    swapped_loss, swapped_metrics = criterion(FixedVelocity(swapped), batch)

    assert aligned_metrics["velocity_contrastive_objects"] == 1
    assert aligned_metrics["velocity_contrastive_acc"] == 1.0
    assert swapped_metrics["velocity_contrastive_acc"] == 0.0
    assert swapped_loss.item() > aligned_loss.item()


def test_velocity_contrastive_skips_single_part_objects():
    x_1 = torch.zeros(1, 1, 4, 1, 1, 1)
    valid = torch.tensor([[True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[10.0]]))
    batch["debug_t"] = torch.ones(1)
    batch["debug_noise"] = torch.zeros_like(x_1)
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, velocity_contrastive_weight=1.0)
    loss, metrics = criterion(FixedVelocity(x_1), batch)
    assert metrics["velocity_contrastive_objects"] == 0
    assert metrics["velocity_contrastive_loss"] == pytest.approx(0.0)
    assert loss.item() == pytest.approx(metrics["mse"])


# ----------------------------------------------------------------------
# Fix 5: logit-normal continuous-time schedule
# ----------------------------------------------------------------------
def test_logit_normal_t_schedule_stays_in_unit_interval_and_skews_to_center():
    torch.manual_seed(0)
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, t_schedule="logit_normal")
    samples = torch.cat([criterion._sample_t(4096, device=torch.device("cpu"), dtype=torch.float32)])
    assert bool((samples >= 0.0).all()) and bool((samples <= 1.0).all())
    # logit_normal(0,1) is symmetric about 0.5 and concentrates mass there.
    assert abs(float(samples.mean()) - 0.5) < 0.02
    central = ((samples > 0.25) & (samples < 0.75)).float().mean().item()
    assert central > 0.5


def test_uniform_t_schedule_is_flat():
    torch.manual_seed(0)
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, t_schedule="uniform")
    samples = criterion._sample_t(8192, device=torch.device("cpu"), dtype=torch.float32)
    central = ((samples > 0.25) & (samples < 0.75)).float().mean().item()
    assert abs(central - 0.5) < 0.05


def test_invalid_t_schedule_raises():
    with pytest.raises(ValueError, match="t_schedule"):
        PartSSLatentRFLoss(t_schedule="cosine")


# ----------------------------------------------------------------------
# Fix 6: per-channel latent normalization
# ----------------------------------------------------------------------
def test_per_channel_latent_norm_round_trips_through_loss():
    channels = 8
    mean = torch.arange(channels, dtype=torch.float32)
    std = torch.arange(1, channels + 1, dtype=torch.float32)
    # x_1 == mean per channel -> normalized x_1 == 0 -> with noise=0, t=1 the RF
    # target velocity is 0; a zero-velocity model reproduces x_1 exactly.
    x_1 = mean.view(1, 1, channels, 1, 1, 1).expand(1, 1, channels, 2, 2, 2).contiguous()
    valid = torch.tensor([[True]])
    batch = _minimal_batch(x_1, valid, torch.tensor([[10.0]]))
    batch["z_global"] = torch.zeros(1, channels, 16, 16, 16)
    batch["debug_t"] = torch.ones(1)
    batch["debug_noise"] = torch.zeros_like(x_1)
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        latent_norm_mode="per_channel",
        latent_channels=channels,
        latent_mean=mean,
        latent_std=std,
        velocity_contrastive_weight=0.0,
    )
    loss, metrics = criterion(FixedVelocity(torch.zeros_like(x_1)), batch)
    # endpoint_raw should recover x_1 (== mean) exactly, so latent metrics are 0.
    assert metrics["latent_l1"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["latent_mse"] == pytest.approx(0.0, abs=1e-6)


def test_per_channel_latent_norm_requires_stats():
    with pytest.raises(ValueError, match="per_channel"):
        PartSSLatentRFLoss(latent_norm_mode="per_channel", latent_channels=8)


def test_sampler_per_channel_latent_norm_denormalizes():
    channels = 8
    mean = torch.arange(channels, dtype=torch.float32)
    std = torch.ones(channels)

    class ZeroVelocity(nn.Module):
        def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots, **kwargs):
            return torch.zeros_like(x_t_parts)

    z = sample_part_ss_latent(
        ZeroVelocity(),
        z_global=torch.zeros(1, channels, 16, 16, 16),
        cond=torch.randn(1, 12, 1024),
        mask_token_labels=torch.ones(1, 12, dtype=torch.long),
        part_valid=torch.ones(1, 1, dtype=torch.bool),
        target_slots=torch.ones(1, 1, dtype=torch.long),
        num_steps=2,
        noise_scale=0.0,
        latent_norm_mode="per_channel",
        latent_mean=mean,
        latent_std=std,
    )
    # x stays 0 in RF space (zero noise + zero velocity); denorm -> mean per channel.
    expected = mean.view(1, 1, channels, 1, 1, 1).expand_as(z)
    assert torch.allclose(z, expected, atol=1e-6)


# ----------------------------------------------------------------------
# Self-conditioning + classifier-free-guidance training support
# ----------------------------------------------------------------------
class CaptureKwargs(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output
        self.calls = []

    def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots, **kwargs):
        self.calls.append({
            "x_self_cond": None if kwargs.get("x_self_cond") is None else kwargs["x_self_cond"].detach().clone(),
            "drop_part_cond": None if kwargs.get("drop_part_cond") is None
            else kwargs["drop_part_cond"].detach().clone(),
        })
        return self.output.to(device=x_t_parts.device, dtype=x_t_parts.dtype).expand_as(x_t_parts)


def test_cfg_dropout_passes_drop_part_cond_mask():
    torch.manual_seed(0)
    x_1 = torch.zeros(1, 4, 1, 1, 1, 1)
    valid = torch.ones(1, 4, dtype=torch.bool)
    batch = _minimal_batch(x_1, valid, torch.ones(1, 4))
    model = CaptureKwargs(torch.zeros(1, 4, 1, 1, 1, 1))
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, cfg_dropout_prob=1.0, velocity_contrastive_weight=0.0)
    _loss, metrics = criterion(model, batch)
    assert len(model.calls) == 1
    drop = model.calls[0]["drop_part_cond"]
    assert drop is not None
    assert bool(drop.all())  # p=1.0 drops every valid part
    assert metrics["cfg_dropped_parts"] == 4


def test_cfg_dropout_off_passes_no_drop_mask():
    x_1 = torch.zeros(1, 2, 1, 1, 1, 1)
    valid = torch.ones(1, 2, dtype=torch.bool)
    batch = _minimal_batch(x_1, valid, torch.ones(1, 2))
    model = CaptureKwargs(torch.zeros(1, 2, 1, 1, 1, 1))
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, cfg_dropout_prob=0.0, velocity_contrastive_weight=0.0)
    _loss, metrics = criterion(model, batch)
    assert model.calls[0]["drop_part_cond"] is None
    assert metrics["cfg_dropped_parts"] == 0


def test_self_conditioning_double_pass_feeds_first_pass_estimate():
    x_1 = torch.zeros(1, 1, 1, 1, 1, 1)
    valid = torch.ones(1, 1, dtype=torch.bool)
    batch = _minimal_batch(x_1, valid, torch.ones(1, 1))
    model = CaptureKwargs(torch.zeros(1, 1, 1, 1, 1, 1))
    # prob=1.0 -> always run the double pass.
    criterion = PartSSLatentRFLoss(
        t_min=0.0,
        t_max=1.0,
        self_conditioning=True,
        self_conditioning_prob=1.0,
        velocity_contrastive_weight=0.0,
    )
    _loss, metrics = criterion(model, batch)
    assert metrics["self_cond_active"] == 1
    assert len(model.calls) == 2
    # First pass receives zeros, second pass receives a non-None estimate.
    assert model.calls[0]["x_self_cond"] is not None
    assert torch.equal(model.calls[0]["x_self_cond"], torch.zeros_like(x_1))
    assert model.calls[1]["x_self_cond"] is not None


def test_self_conditioning_disabled_runs_single_pass_without_self_cond():
    x_1 = torch.zeros(1, 1, 1, 1, 1, 1)
    valid = torch.ones(1, 1, dtype=torch.bool)
    batch = _minimal_batch(x_1, valid, torch.ones(1, 1))
    model = CaptureKwargs(torch.zeros(1, 1, 1, 1, 1, 1))
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, self_conditioning=False, velocity_contrastive_weight=0.0)
    _loss, metrics = criterion(model, batch)
    assert metrics["self_cond_active"] == 0
    assert len(model.calls) == 1
    assert model.calls[0]["x_self_cond"] is None
