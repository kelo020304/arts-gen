import torch
import pytest

from trellis.trainers.arts.part_ss_latent_flow import (
    _build_part_ss_latent_rf_loss,
    _latent_metric_values,
    _resolve_object_weight_k_ref,
)
from trellis.trainers.arts.part_ss_latent_flow_losses import PartSSLatentRFLoss


class FixedVelocity(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots, **kwargs):
        return self.output.to(device=x_t_parts.device, dtype=x_t_parts.dtype)


def test_latent_metrics_report_zero_baseline_and_ratio():
    target = torch.tensor([[[[[2.0, 0.0]]]]])
    pred = torch.zeros_like(target)
    metrics = _latent_metric_values(pred, target)
    assert metrics["latent_mse"] == metrics["zero_mse"]
    assert metrics["latent_l1"] == metrics["zero_l1"]
    assert metrics["mse_vs_zero"] == 1.0
    assert metrics["l1_vs_zero"] == 1.0

    perfect = _latent_metric_values(target, target)
    assert perfect["latent_mse"] == 0.0
    assert perfect["latent_l1"] == 0.0
    assert perfect["mse_vs_zero"] == 0.0
    assert perfect["l1_vs_zero"] == 0.0


def test_object_weight_k_ref_none_mode_does_not_touch_dataset_samples():
    class DatasetThatShouldNotBeScanned:
        @property
        def samples(self):
            raise AssertionError("samples should not be read for object_weight_mode=none")

    assert _resolve_object_weight_k_ref(
        DatasetThatShouldNotBeScanned(),
        {"object_weight_mode": "none", "object_weight_k_ref_source": "dataset_median"},
        rank=0,
        is_distributed=False,
    ) is None


def test_object_weight_k_ref_fixed_requires_positive_value():
    class EmptyDataset:
        samples = []

    with pytest.raises(ValueError, match="object_weight_k_ref > 0"):
        _resolve_object_weight_k_ref(
            EmptyDataset(),
            {"object_weight_mode": "sqrt_k", "object_weight_k_ref_source": "fixed"},
            rank=0,
            is_distributed=False,
        )


def test_object_weight_k_ref_dataset_median_uses_sample_part_counts():
    class Dataset:
        samples = [
            {"parts": [1]},
            {"parts": [1, 2, 3]},
            {"parts": [1, 2, 3, 4, 5]},
        ]

    assert _resolve_object_weight_k_ref(
        Dataset(),
        {"object_weight_mode": "sqrt_k", "object_weight_k_ref_source": "dataset_median"},
        rank=0,
        is_distributed=False,
    ) == 3.0


def test_trainer_build_returns_plain_rf_loss_and_threads_new_flags():
    criterion = _build_part_ss_latent_rf_loss(
        loss_cfg={
            "part_shuffle": True,
            "velocity_contrastive_weight": 0.05,
            "velocity_contrastive_lambda": 0.05,
            "identity_contrastive_weight": 0.0,
            "cfg_dropout_prob": 0.1,
        },
        flow_cfg={
            "t_min": 0.0,
            "t_max": 1.0,
            "noise_scale": 1.0,
            "latent_scale": 1.0,
            "t_schedule": "logit_normal",
        },
        resolved_object_weight_k_ref=None,
        model_cfg={"latent_channels": 8, "self_conditioning": True},
    )
    assert isinstance(criterion, PartSSLatentRFLoss)
    assert criterion.part_shuffle is True
    assert criterion.velocity_contrastive_weight == 0.05
    assert criterion.identity_contrastive_weight == 0.0
    assert criterion.t_schedule == "logit_normal"
    assert criterion.self_conditioning is True
    assert criterion.cfg_dropout_prob == 0.1
    assert not hasattr(criterion, "set_step")

    x_1 = torch.zeros(1, 1, 1, 1, 1, 1)
    batch = {
        "x_1_parts": x_1,
        "part_valid": torch.ones(1, 1, dtype=torch.bool),
        "part_fg_mask": torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
        "z_global": torch.zeros(1, 1, 1, 1, 1),
        "cond": torch.zeros(1, 1, 1),
        "mask_token_labels": torch.ones(1, 1, dtype=torch.long),
        "target_slots": torch.ones(1, 1, dtype=torch.long),
        "debug_t": torch.zeros(1),
        "debug_noise": torch.zeros_like(x_1),
    }
    _loss, metrics = criterion(FixedVelocity(torch.zeros_like(x_1)), batch)

    assert "velocity_contrastive_loss" in metrics
    assert not any(key.startswith("decode_aware") for key in metrics)
