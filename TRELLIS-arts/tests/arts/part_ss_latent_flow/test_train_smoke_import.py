import importlib


def test_part_ss_latent_flow_trainer_imports():
    mod = importlib.import_module("trellis.trainers.arts.part_ss_latent_flow")
    assert hasattr(mod, "train")
