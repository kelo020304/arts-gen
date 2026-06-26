import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from inference_pipeline import object_inputs


class _FakeDataset:
    instances = 0

    def __init__(self, cfg):
        type(self).instances += 1
        self.cfg = cfg
        self.samples = []


def _setup(monkeypatch):
    """Stub both dataset classes + clear the module cache for isolation."""
    _FakeDataset.instances = 0
    object_inputs._DATASET_CACHE.clear()
    monkeypatch.setattr(object_inputs, "PartSSLatentFlowDataset", _FakeDataset)
    monkeypatch.setattr(object_inputs, "PartSSLatentFlowSingleViewDataset", _FakeDataset)


def test_dataset_for_caches_same_object(monkeypatch):
    _setup(monkeypatch)
    cfg = {"data_root": "/dev/root", "manifest_path": "manifests/m.jsonl", "num_views": 4}

    ds1 = object_inputs._dataset_for("multi", cfg)
    ds2 = object_inputs._dataset_for("multi", cfg)

    assert ds1 is ds2                       # same cached instance reused
    assert _FakeDataset.instances == 1      # constructor (and its manifest read/log) ran once


def test_dataset_for_separate_keys_not_shared(monkeypatch):
    _setup(monkeypatch)
    cfg = {"data_root": "/dev/root", "manifest_path": "manifests/m.jsonl", "num_views": 4}

    multi = object_inputs._dataset_for("multi", cfg)
    single = object_inputs._dataset_for("single", cfg)

    assert multi is not single              # view_mode is part of the cache key
    assert _FakeDataset.instances == 2
