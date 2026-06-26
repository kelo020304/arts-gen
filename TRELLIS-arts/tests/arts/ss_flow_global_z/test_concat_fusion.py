import importlib
import os
import sys
import types

import pytest
import torch


def _install_trellis_stub():
    trellis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if trellis_path not in sys.path:
        sys.path.insert(0, trellis_path)
    if "trellis" not in sys.modules:
        pkg = types.ModuleType("trellis")
        pkg.__path__ = [os.path.join(trellis_path, "trellis")]
        pkg.__package__ = "trellis"
        sys.modules["trellis"] = pkg
    for sp in ("models", "modules", "trainers", "utils", "datasets", "pipelines"):
        name = f"trellis.{sp}"
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(trellis_path, "trellis", sp)]
            mod.__package__ = name
            sys.modules[name] = mod


_install_trellis_stub()

from trellis.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402


def _small_model(*, use_view_id_embedding=True, num_view_embeddings=4):
    model = SparseStructureFlowModel(
        resolution=4,
        in_channels=2,
        model_channels=16,
        cond_channels=16,
        out_channels=2,
        num_blocks=1,
        num_heads=4,
        patch_size=2,
        pe_mode="ape",
        use_fp16=False,
        use_camera_pose=False,
        use_view_id_embedding=use_view_id_embedding,
        num_view_embeddings=num_view_embeddings,
    )
    with torch.no_grad():
        model.out_layer.weight.fill_(0.03)
        model.out_layer.bias.zero_()
    return model


class _EchoDenoiser(torch.nn.Module):
    def forward(self, x_t, t, cond, **kwargs):
        while t.ndim < x_t.ndim:
            t = t.view(*t.shape, 1)
        scale = cond.mean(dim=tuple(range(1, cond.ndim))).view(-1, *([1] * (x_t.ndim - 1)))
        return x_t + scale + t.to(x_t.dtype)


class _ShapeDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x_t, t, cond, **kwargs):
        self.calls.append((tuple(x_t.shape), tuple(t.shape), tuple(cond.shape)))
        return torch.ones_like(x_t) * float(cond.ndim)


class _CaptureDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x_t, t, cond, **kwargs):
        self.calls.append({"x_t": x_t.detach().clone(), "t": t.detach().clone(), "cond": cond.detach().clone()})
        return torch.zeros_like(x_t)


def _trainer_class():
    mod = importlib.import_module("trellis.trainers.arts.ss_flow_global_z")
    return mod.SSFlowGlobalZTrainer


def test_concat_forward_shape_and_backward_cpu():
    torch.manual_seed(0)
    model = _small_model()
    x_t = torch.randn(2, 2, 4, 4, 4, requires_grad=True)
    t = torch.tensor([10.0, 20.0])
    cond = torch.randn(2, 4, 5, 16, requires_grad=True)

    out = model(x_t, t, cond)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == x_t.shape
    assert x_t.grad is not None
    assert cond.grad is not None
    assert model.view_id_embedding.weight.grad is not None


def test_view_id_embedding_affects_output_and_order_cpu():
    torch.manual_seed(1)
    model = _small_model()
    model.eval()
    x_t = torch.randn(1, 2, 4, 4, 4)
    t = torch.tensor([10.0])
    cond = torch.randn(1, 4, 5, 16)

    out_default = model(x_t, t, cond)
    out_shifted_ids = model(x_t, t, cond, view_ids=torch.tensor([1, 2, 3, 0]))
    out_permuted = model(x_t, t, cond[:, [1, 0, 2, 3]])

    assert not torch.allclose(out_default, out_shifted_ids)
    assert not torch.allclose(out_default, out_permuted)


def test_multidiffusion_fusion_path_matches_per_view_average():
    trainer_cls = _trainer_class()
    model = _EchoDenoiser()
    x_t = torch.zeros(2, 2, 4, 4, 4)
    t = torch.tensor([0.1, 0.2])
    cond = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)

    out = trainer_cls.predict_multiview(model, x_t, t, cond, fusion_mode="multidiffusion")

    manual = []
    for view in range(cond.shape[1]):
        manual.append(model(x_t, t, cond[:, view]))
    expected = torch.stack(manual, dim=1).mean(dim=1)
    assert torch.allclose(out, expected)


def test_concat_fusion_path_calls_model_once_with_4d_cond():
    trainer_cls = _trainer_class()
    model = _ShapeDenoiser()
    x_t = torch.zeros(2, 2, 4, 4, 4)
    t = torch.tensor([0.1, 0.2])
    cond = torch.zeros(2, 4, 3, 5)

    out = trainer_cls.predict_multiview(model, x_t, t, cond, fusion_mode="concat")

    assert out.shape == x_t.shape
    assert len(model.calls) == 1
    assert model.calls[0] == (tuple(x_t.shape), tuple(t.shape), tuple(cond.shape))
    assert torch.equal(out, torch.ones_like(x_t) * 4)


def test_concat_training_loss_uses_4d_cond_and_cfg_dropout():
    trainer_cls = _trainer_class()
    trainer = trainer_cls.__new__(trainer_cls)
    denoiser = _CaptureDenoiser()
    trainer.training_models = {"denoiser": denoiser}
    trainer.fusion_mode = "concat"
    trainer.p_uncond = 1.0
    trainer.sigma_min = 1.0e-5
    trainer.t_schedule = {"name": "uniform"}
    x_0 = torch.randn(2, 2, 4, 4, 4)
    cond = torch.randn(2, 4, 3, 5)

    terms, extras = trainer_cls.training_losses(trainer, x_0=x_0, cond=cond)

    assert extras == {}
    assert "loss" in terms and "mse" in terms
    assert float(terms["fusion_mode_concat"].item()) == 1.0
    assert len(denoiser.calls) == 1
    assert tuple(denoiser.calls[0]["cond"].shape) == tuple(cond.shape)
    assert torch.equal(denoiser.calls[0]["cond"], torch.zeros_like(cond))


@pytest.mark.parametrize("num_views", [1, 2, 5])
def test_concat_forward_accepts_non_four_view_counts(num_views):
    torch.manual_seed(2 + num_views)
    model = _small_model(use_view_id_embedding=True, num_view_embeddings=4)
    x_t = torch.randn(1, 2, 4, 4, 4)
    t = torch.tensor([10.0])
    cond = torch.randn(1, num_views, 3, 16)

    out = model(x_t, t, cond)

    assert out.shape == x_t.shape
