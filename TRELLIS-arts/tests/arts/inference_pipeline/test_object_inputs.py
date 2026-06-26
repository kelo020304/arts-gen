import pytest
import inference_pipeline.object_inputs as oi

class _FakeDS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = [
            {"obj_id": "100013", "angle_idx": 0},
            {"obj_id": "100214", "angle_idx": 0},
            {"obj_id": "100214", "angle_idx": 1},
        ]
    def __getitem__(self, i):
        s = self.samples[i]
        return {"obj_id": s["obj_id"], "angle_idx": s["angle_idx"], "cond": "TOKENS",
                "target_part_names": ["wheel_0"], "raw_ind_coords": [[[0, 0, 0]]]}

class _FakeSingle(_FakeDS):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert int(cfg.get("num_views", 1)) == 1   # single-view forces 1

def test_load_object_inputs_selects_by_id_and_angle(monkeypatch):
    monkeypatch.setattr(oi, "PartSSLatentFlowDataset", _FakeDS)
    monkeypatch.setattr(oi, "PartSSLatentFlowSingleViewDataset", _FakeSingle)
    out = oi.load_object_inputs({"data_root": "x"}, object_id="100214", angle_idx=1, view_mode="four")
    assert out["dataset_index"] == 2 and out["obj_id"] == "100214" and out["angle_idx"] == 1
    assert out["cond"] == "TOKENS" and out["target_part_names"] == ["wheel_0"]

def test_single_view_uses_single_dataset(monkeypatch):
    monkeypatch.setattr(oi, "PartSSLatentFlowDataset", _FakeDS)
    monkeypatch.setattr(oi, "PartSSLatentFlowSingleViewDataset", _FakeSingle)
    out = oi.load_object_inputs({}, object_id="100013", angle_idx=0, view_mode="single")
    assert out["dataset_index"] == 0

def test_missing_object_raises(monkeypatch):
    monkeypatch.setattr(oi, "PartSSLatentFlowDataset", _FakeDS)
    monkeypatch.setattr(oi, "PartSSLatentFlowSingleViewDataset", _FakeSingle)
    with pytest.raises(KeyError):
        oi.load_object_inputs({}, object_id="999999", angle_idx=0, view_mode="four")
