import importlib
from pathlib import Path

import pytest
import torch


def test_part_mmdit_trainer_imports():
    module = importlib.import_module("trellis.trainers.arts.part_mmdit")
    assert hasattr(module, "train")


def test_train_arts_dispatch_has_part_mmdit():
    train_arts = importlib.import_module("train_arts")
    assert train_arts._STAGE_DISPATCH["part_mmdit"] == "trellis.trainers.arts.part_mmdit"


def test_build_eval_dataset_rejects_train_val_overlap():
    module = importlib.import_module("trellis.trainers.arts.part_mmdit")

    class FakeDataset:
        def __init__(self, cfg):
            self.cfg = dict(cfg)
            self.samples = [{"obj_id": obj_id, "parts": [1]} for obj_id in cfg["obj_ids"]]

        def __len__(self):
            return len(self.samples)

    train_dataset = FakeDataset({"obj_ids": ["train_a", "shared"]})

    with pytest.raises(ValueError, match="train/val obj_id overlap"):
        module._build_eval_dataset(
            {"data": {"obj_ids": ["shared", "val_b"]}},
            dataset_cls=FakeDataset,
            train_dataset=train_dataset,
        )


def test_append_eval_metrics_markdown(tmp_path):
    module = importlib.import_module("trellis.trainers.arts.part_mmdit")
    path = tmp_path / "code_update.md"
    metrics = {
        "train_latent_cos": 0.8,
        "train_latent_rel_l2": 0.06,
        "latent_cos": 0.9,
        "latent_rel_l2": 0.05,
        "target_iou": 0.1,
        "part_iou": 0.2,
        "recall": 0.3,
        "precision": 0.4,
        "small_recall": 0.5,
        "large_recall": 0.6,
        "count_error": 0.7,
        "offdiag": 0.8,
    }

    module._append_eval_metrics_markdown(path, step=2000, metrics=metrics)

    text = path.read_text(encoding="utf-8")
    assert (
        "| 2000 | 0.8000 | 0.0600 | 0.9000 | 0.0500 | 0.1000 | 0.2000 | 0.3000 | "
        "0.4000 | 0.5000 | 0.6000 | 0.7000 | 0.8000 |"
    ) in text


def test_part_mmdit_can_sample_logit_normal_t_schedule_when_configured():
    module = importlib.import_module("trellis.trainers.arts.part_mmdit")

    torch.manual_seed(0)
    t = module.sample_flow_timesteps(
        10_000,
        device=torch.device("cpu"),
        dtype=torch.float32,
        t_schedule="logit_normal",
        t_logit_normal_mean=0.0,
        t_logit_normal_std=1.0,
    )
    assert bool((t > 0.0).all()) and bool((t < 1.0).all())
    assert abs(float(t.mean()) - 0.5) < 0.02
    central = ((t > 0.25) & (t < 0.75)).float().mean().item()
    assert central > 0.65


def test_part_mmdit_t_schedule_can_still_sample_uniform():
    module = importlib.import_module("trellis.trainers.arts.part_mmdit")

    torch.manual_seed(0)
    t = module.sample_flow_timesteps(
        10_000,
        device=torch.device("cpu"),
        dtype=torch.float32,
        t_schedule="uniform",
    )
    central = ((t > 0.25) & (t < 0.75)).float().mean().item()
    assert abs(central - 0.5) < 0.03


def test_part_mmdit_default_t_schedule_is_uniform_and_sampler_shift_zero():
    module = importlib.import_module("trellis.trainers.arts.part_mmdit")

    class FakeModel:
        resolution = 16
        patch_size = 2
        model_channels = 1024

    torch.manual_seed(0)
    t = module.sample_flow_timesteps(
        10_000,
        device=torch.device("cpu"),
        dtype=torch.float32,
        t_schedule="uniform",
    )
    central = ((t > 0.25) & (t < 0.75)).float().mean().item()
    assert abs(central - 0.5) < 0.03

    assert module._sampler_timestep_shift(FakeModel(), {}) == pytest.approx(0.0)
    base_grid = torch.linspace(0.0, 1.0, 6)
    assert torch.allclose(module._apply_timestep_shift(base_grid, shift=0.0), base_grid)


def test_dynamic_timestep_shift_matches_token_dim_formula_when_enabled():
    module = importlib.import_module("trellis.trainers.arts.part_mmdit")

    class FakeModel:
        resolution = 16
        patch_size = 2
        model_channels = 1024

    shift = module._dynamic_timestep_shift(FakeModel(), {"dynamic_timestep_shift": True})
    assert abs(shift - (128.0 ** 0.5)) < 1e-6

    t = torch.tensor([0.0, 0.5, 1.0])
    shifted = module._apply_timestep_shift(t, shift=shift)
    assert torch.allclose(shifted[[0, 2]], torch.tensor([0.0, 1.0]))
    assert shifted[1] > 0.5
