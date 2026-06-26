import pytest
import torch

from trellis.trainers.arts.part_mmdit import (
    compute_part_mmdit_eval_metrics,
    compute_part_mmdit_latent_alignment,
    sample_part_mmdit_latent,
)


def test_eval_metrics_reports_small_recall_and_assignment():
    pred_coords = [
        [
            torch.tensor([[0, 0, 0], [0, 0, 1], [9, 9, 9]]),
            torch.tensor([[5, 5, 5], [5, 5, 6], [5, 5, 7]]),
        ]
    ]
    raw_coords = [
        [
            torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
            torch.tensor([[5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 5, 8]]),
        ]
    ]
    counts = torch.tensor([[100.0, 5000.0]])
    valid = torch.tensor([[True, True]])

    metrics = compute_part_mmdit_eval_metrics(
        pred_coords,
        raw_coords,
        counts,
        valid,
        size_boundaries=(500.0, 3000.0),
    )

    assert metrics["small_recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["large_recall"] == pytest.approx(3.0 / 4.0)
    assert metrics["recall"] == pytest.approx(5.0 / 7.0)
    assert metrics["precision"] == pytest.approx(5.0 / 6.0)
    assert metrics["target_iou"] == pytest.approx(5.0 / 8.0)
    assert metrics["offdiag"] == pytest.approx(0.0)
    assert metrics["count_error"] == pytest.approx(abs(6.0 - 7.0) / 7.0)


def test_eval_metrics_handles_padded_variable_part_counts():
    pred_coords = [
        [torch.tensor([[0, 0, 0]])],
        [torch.tensor([[1, 1, 1]]), torch.tensor([[2, 2, 2]])],
    ]
    raw_coords = [
        [torch.tensor([[0, 0, 0]])],
        [torch.tensor([[1, 1, 1]]), torch.tensor([[3, 3, 3]])],
    ]
    counts = torch.tensor([[10.0, 0.0], [10.0, 10.0]])
    valid = torch.tensor([[True, False], [True, True]])

    metrics = compute_part_mmdit_eval_metrics(pred_coords, raw_coords, counts, valid)

    assert metrics["small_recall"] == pytest.approx((1.0 + 1.0 + 0.0) / 3.0)


def test_latent_alignment_reports_cos_and_relative_l2_on_valid_parts():
    target = torch.zeros(1, 2, 1, 1, 1, 2)
    pred = torch.zeros_like(target)
    target[0, 0, 0, 0, 0] = torch.tensor([3.0, 4.0])
    pred[0, 0, 0, 0, 0] = torch.tensor([3.0, 4.0])
    pred[0, 1, 0, 0, 0] = torch.tensor([10.0, 10.0])
    valid = torch.tensor([[True, False]])

    metrics = compute_part_mmdit_latent_alignment(pred, target, valid)

    assert metrics["latent_cos"] == pytest.approx(1.0)
    assert metrics["latent_rel_l2"] == pytest.approx(0.0)


def test_eval_fixed_metrics_can_run_cos_only_without_decode(monkeypatch):
    from trellis.trainers.arts import part_mmdit as module

    calls = []

    def fake_latent(_model, dataset, _device, _flow_cfg, eval_cfg):
        calls.append((dataset, int(eval_cfg["max_samples"])))
        value = 0.5 + 0.1 * len(calls)
        return {"latent_cos": value, "latent_rel_l2": 1.0 / value}

    def fake_decode(*_args, **_kwargs):
        raise AssertionError("decode path should not run when eval.decode=false")

    monkeypatch.setattr(module, "_eval_latent_alignment", fake_latent)
    monkeypatch.setattr(module, "_eval_decode_metrics", fake_decode)

    train_dataset = object()
    val_dataset = object()
    metrics = module._eval_fixed_metrics(
        object(),
        train_dataset,
        val_dataset,
        torch.device("cpu"),
        {},
        {"decode": False, "max_samples": 3, "train_max_samples": 2},
    )

    assert calls == [(val_dataset, 3), (train_dataset, 2)]
    assert metrics["latent_cos"] == pytest.approx(0.6)
    assert metrics["train_latent_cos"] == pytest.approx(0.7)
    assert metrics["small_recall"] != metrics["small_recall"]


def test_eval_interval_rejects_mismatched_aliases():
    from trellis.trainers.arts import part_mmdit as module

    assert module._eval_interval_steps({"eval_every": 500}) == 500
    assert module._eval_interval_steps({"fixed_every": 250}) == 250
    assert module._eval_interval_steps({"eval_every": 250, "fixed_every": 250}) == 250
    with pytest.raises(ValueError, match="eval\\.eval_every .* eval\\.fixed_every"):
        module._eval_interval_steps({"eval_every": 250, "fixed_every": 500})


def test_eval_sample_indices_honor_preferred_obj_ids():
    from trellis.trainers.arts import part_mmdit as module

    class FakeDataset:
        samples = [
            {"obj_id": "multi_a"},
            {"obj_id": "single_b"},
            {"obj_id": "button_c"},
            {"obj_id": "extra_d"},
        ]

        def __len__(self):
            return len(self.samples)

    indices = module._eval_sample_indices(
        FakeDataset(),
        {
            "max_samples": 3,
            "prefer_obj_ids": ["single_b", "button_c"],
        },
        "max_samples",
        prefer_obj_ids_key="prefer_obj_ids",
    )
    assert indices == [1, 2, 0]

    with pytest.raises(ValueError, match="not present"):
        module._eval_sample_indices(
            FakeDataset(),
            {
                "max_samples": 3,
                "prefer_obj_ids": ["missing"],
            },
            "max_samples",
            prefer_obj_ids_key="prefer_obj_ids",
        )


def test_queue_eval_log_line_matches_contract():
    from trellis.trainers.arts import part_mmdit as module

    line = module._queue_eval_log_line(
        step=2000,
        wall=12.5,
        loss_value=0.1234,
        lr=2.0e-4,
        grad_norm=0.9,
        metrics={
            "train_cos_single": 0.1,
            "train_cos_multi": 0.2,
            "train_cos_buttons": 0.3,
            "val_cos_single": 0.4,
            "val_cos_multi": 0.5,
            "val_cos_buttons": 0.55,
            "small_recall": 0.6,
            "part_iou": 0.7,
            "target_iou": 0.8,
            "vel_err_t0.02": 0.9,
            "vel_err_t0.1": 1.0,
            "vel_err_t0.3": 1.1,
            "vel_err_t0.5": 1.2,
            "vel_err_t0.9": 1.3,
        },
        t_values=[0.02, 0.1, 0.3, 0.5, 0.9],
    )

    assert line.startswith("  [EVAL step=2000 wall=12.5s] loss=0.1234 lr=2.00e-04 gradnorm=0.9000 | ")
    assert "TRAIN cos_single=0.1000 cos_multi=0.2000 cos_buttons=0.3000" in line
    assert "VAL cos_single=0.4000 cos_multi=0.5000 cos_buttons=0.5500 small_recall=0.6000 part_iou=0.7000 target_iou=0.8000" in line
    assert "VEL_ERR t0.02=0.9000 t0.1=1.0000 t0.3=1.1000 t0.5=1.2000 t0.9=1.3000" in line


def test_queue_eval_log_line_prints_na_for_empty_groups():
    from trellis.trainers.arts import part_mmdit as module

    line = module._queue_eval_log_line(
        step=1,
        wall=1.0,
        loss_value=1.0,
        lr=1.0e-4,
        grad_norm=2.0,
        metrics={
            "train_cos_single": float("nan"),
            "train_cos_multi": 0.2,
            "train_cos_buttons": float("nan"),
            "val_cos_single": 0.4,
            "val_cos_multi": float("nan"),
            "val_cos_buttons": 0.5,
            "small_recall": float("nan"),
            "part_iou": 0.7,
            "target_iou": float("nan"),
            "vel_err_t0.02": float("nan"),
        },
        t_values=[0.02],
    )

    assert "TRAIN cos_single=n/a cos_multi=0.2000 cos_buttons=n/a" in line
    assert "VAL cos_single=0.4000 cos_multi=n/a cos_buttons=0.5000 small_recall=n/a part_iou=0.7000 target_iou=n/a" in line
    assert "VEL_ERR t0.02=n/a" in line


def test_queue_metrics_aggregates_train_val_decode_and_velocity(monkeypatch):
    from trellis.trainers.arts import part_mmdit as module

    def fake_group(_model, dataset, _device, _flow_cfg, _eval_cfg, *, max_samples_key, prefer_obj_ids_key=None):
        assert max_samples_key in {"train_max_samples", "max_samples"}
        assert prefer_obj_ids_key in {"train_prefer_obj_ids", "prefer_obj_ids"}
        prefix = "train" if dataset == "train" else "val"
        return {
            "cos_single": 1.0 if prefix == "train" else 4.0,
            "cos_multi": 2.0 if prefix == "train" else 5.0,
            "cos_buttons": 3.0 if prefix == "train" else 6.0,
        }

    def fake_decode(*_args, **_kwargs):
        return {
            "small_recall": 0.7,
            "part_iou": 0.8,
            "target_iou": 0.9,
        }

    def fake_velocity(*_args, **_kwargs):
        return {"vel_err_t0.02": 0.11, "vel_err_t0.1": 0.22}

    monkeypatch.setattr(module, "_eval_latent_group_cos", fake_group)
    monkeypatch.setattr(module, "_eval_decode_metrics", fake_decode)
    monkeypatch.setattr(module, "_eval_velocity_sweep_mean", fake_velocity)

    metrics = module._eval_queue_metrics(
        object(),
        "train",
        "val",
        torch.device("cpu"),
        {},
        {"decode": True},
    )

    assert metrics["train_cos_single"] == pytest.approx(1.0)
    assert metrics["train_cos_multi"] == pytest.approx(2.0)
    assert metrics["train_cos_buttons"] == pytest.approx(3.0)
    assert metrics["val_cos_single"] == pytest.approx(4.0)
    assert metrics["val_cos_multi"] == pytest.approx(5.0)
    assert metrics["val_cos_buttons"] == pytest.approx(6.0)
    assert metrics["small_recall"] == pytest.approx(0.7)
    assert metrics["part_iou"] == pytest.approx(0.8)
    assert metrics["target_iou"] == pytest.approx(0.9)
    assert metrics["vel_err_t0.02"] == pytest.approx(0.11)
    assert metrics["vel_err_t0.1"] == pytest.approx(0.22)


def test_eval_sample_raw_can_use_cond_only_sampler(monkeypatch):
    from trellis.trainers.arts import part_mmdit as module

    calls = []

    def fake_cond_only(_model, **kwargs):
        calls.append(("cond_only", kwargs["num_steps"], kwargs["timestep_shift"]))
        return torch.full((1, 1, 1, 2, 2, 2), 16.0)

    def fake_cfg(*_args, **_kwargs):
        raise AssertionError("CFG sampler should not run when eval.cond_only=true")

    monkeypatch.setattr(module, "sample_part_mmdit_latent_cond_only_scaled", fake_cond_only)
    monkeypatch.setattr(module, "sample_part_mmdit_latent", fake_cfg)

    batch = {
        "z_global": torch.zeros(1, 1, 2, 2, 2),
        "cond": torch.zeros(1, 2, 4),
        "name_tokens": torch.zeros(1, 1, 5, 768),
        "name_mask": torch.ones(1, 1, 5, dtype=torch.bool),
        "anchor": torch.zeros(1, 1, 4, 4),
        "anchor_valid": torch.ones(1, 1, 4, dtype=torch.bool),
        "part_valid": torch.ones(1, 1, dtype=torch.bool),
    }

    pred = module._sample_eval_part_latent_raw(
        object(),
        batch,
        {"latent_scale": 8.0, "num_steps": 7, "timestep_shift": 0.0},
        {"cond_only": True},
    )

    assert calls == [("cond_only", 7, 0.0)]
    assert torch.allclose(pred, torch.full((1, 1, 1, 2, 2, 2), 2.0))


def test_sampler_uses_name_tokens_masks_and_three_way_cfg():
    class CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(
            self,
            x,
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
            self.calls.append(
                {
                    "t": float(t[0].item()),
                    "name_tokens_shape": tuple(name_tokens.shape),
                    "name_mask_shape": tuple(name_mask.shape),
                    "drop_name": drop_name.detach().cpu().tolist(),
                    "drop_anchor": drop_anchor.detach().cpu().tolist(),
                }
            )
            return torch.zeros_like(x)

    model = CaptureModel()
    part_valid = torch.tensor([[True]])
    z_global = torch.zeros(1, 1, 2, 2, 2)
    initial_noise = torch.ones(1, 1, 1, 2, 2, 2)

    pred = sample_part_mmdit_latent(
        model,
        z_global=z_global,
        cond=torch.zeros(1, 2, 4),
        name_tokens=torch.zeros(1, 1, 5, 768),
        name_mask=torch.ones(1, 1, 5, dtype=torch.bool),
        anchor=torch.zeros(1, 1, 4, 4),
        anchor_valid=torch.ones(1, 1, 4, dtype=torch.bool),
        part_valid=part_valid,
        initial_noise=initial_noise,
        num_steps=2,
        noise_scale=1.0,
        latent_scale=2.0,
        s_name=4.0,
        s_anchor=4.0,
        timestep_shift=2.0,
    )

    assert pred.shape == initial_noise.shape
    assert torch.allclose(pred, initial_noise / 2.0)
    assert len(model.calls) == 6
    assert model.calls[0]["name_tokens_shape"] == (1, 1, 5, 768)
    assert model.calls[0]["name_mask_shape"] == (1, 1, 5)
    assert model.calls[0]["drop_name"] == [[True]]
    assert model.calls[0]["drop_anchor"] == [[True]]
    assert model.calls[1]["drop_name"] == [[False]]
    assert model.calls[1]["drop_anchor"] == [[True]]
    assert model.calls[2]["drop_name"] == [[True]]
    assert model.calls[2]["drop_anchor"] == [[False]]
    assert model.calls[3]["t"] == pytest.approx(2.0 / 3.0)


def test_sampler_shift_zero_uses_uniform_grid():
    class CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.t_values = []

        def forward(
            self,
            x,
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
            self.t_values.append(float(t[0].item()))
            return torch.zeros_like(x)

    model = CaptureModel()
    part_valid = torch.tensor([[True]])
    z_global = torch.zeros(1, 1, 2, 2, 2)
    initial_noise = torch.ones(1, 1, 1, 2, 2, 2)

    sample_part_mmdit_latent(
        model,
        z_global=z_global,
        cond=torch.zeros(1, 2, 4),
        name_tokens=torch.zeros(1, 1, 5, 768),
        name_mask=torch.ones(1, 1, 5, dtype=torch.bool),
        anchor=torch.zeros(1, 1, 4, 4),
        anchor_valid=torch.ones(1, 1, 4, dtype=torch.bool),
        part_valid=part_valid,
        initial_noise=initial_noise,
        num_steps=4,
        latent_scale=2.0,
        timestep_shift=0.0,
    )

    # Three CFG calls per Euler step, so inspect the first call of each step.
    assert model.t_values[0::3] == pytest.approx([0.0, 0.25, 0.5, 0.75])
