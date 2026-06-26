import importlib
import os
import sys
import types

import torch


def _install_train_arts_trellis_stub():
    trellis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if trellis_path not in sys.path:
        sys.path.insert(0, trellis_path)
    pkg = types.ModuleType("trellis")
    pkg.__path__ = [os.path.join(trellis_path, "trellis")]
    pkg.__package__ = "trellis"
    sys.modules["trellis"] = pkg
    for sp in ("models", "modules", "trainers", "utils", "datasets"):
        mod = types.ModuleType(f"trellis.{sp}")
        mod.__path__ = [os.path.join(trellis_path, "trellis", sp)]
        mod.__package__ = f"trellis.{sp}"
        sys.modules[f"trellis.{sp}"] = mod
    pipelines = types.ModuleType("trellis.pipelines")
    pipelines.__path__ = [os.path.join(trellis_path, "trellis", "pipelines")]
    pipelines.__package__ = "trellis.pipelines"
    sys.modules["trellis.pipelines"] = pipelines


def test_trainer_imports_and_stage_dispatch():
    _install_train_arts_trellis_stub()
    mod = importlib.import_module("trellis.trainers.arts.ss_flow_global_z")
    assert hasattr(mod, "train")
    assert hasattr(mod, "SSFlowGlobalZTrainer")

    train_arts = importlib.import_module("train_arts")
    assert train_arts._STAGE_DISPATCH["ss_flow_global_z"] == "trellis.trainers.arts.ss_flow_global_z"


def test_get_cond_uses_preencoded_tokens_without_image_encoder():
    _install_train_arts_trellis_stub()
    mod = importlib.import_module("trellis.trainers.arts.ss_flow_global_z")
    trainer = mod.SSFlowGlobalZTrainer.__new__(mod.SSFlowGlobalZTrainer)
    trainer.p_uncond = 0.0

    cond = torch.randn(2, 12, 5)
    out = mod.SSFlowGlobalZTrainer.get_cond(trainer, cond)
    inf = mod.SSFlowGlobalZTrainer.get_inference_cond(trainer, cond)

    assert out is cond
    assert torch.equal(inf["cond"], cond)
    assert torch.equal(inf["neg_cond"], torch.zeros_like(cond))


def test_get_cond_cfg_dropout_zeros_tokens():
    _install_train_arts_trellis_stub()
    mod = importlib.import_module("trellis.trainers.arts.ss_flow_global_z")
    trainer = mod.SSFlowGlobalZTrainer.__new__(mod.SSFlowGlobalZTrainer)
    trainer.p_uncond = 1.0

    cond = torch.ones(2, 12, 5)
    out = mod.SSFlowGlobalZTrainer.get_cond(trainer, cond)

    assert torch.equal(out, torch.zeros_like(cond))
