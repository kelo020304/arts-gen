import sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from inference_pipeline import data_config_io


def test_load_with_root_override(tmp_path):
    cfg_yaml = tmp_path/"c.yaml"
    cfg_yaml.write_text(yaml.safe_dump({"data": {"data_root": "/dev/abs", "recon_subdir": "reconstruction",
        "mask_subdir": "renders", "manifest_path": "manifests/m.jsonl", "num_views": 4}}))
    dc = data_config_io.load_data_config(cfg_yaml, data_root_override="/local/smoke")
    assert dc["data_root"]=="/local/smoke" and dc["recon_subdir"]=="reconstruction"
    assert dc["manifest_path"]=="manifests/m.jsonl"


def test_missing_data_section_raises(tmp_path):
    import pytest
    bad = tmp_path/"b.yaml"; bad.write_text(yaml.safe_dump({"model": {}}))
    with pytest.raises(KeyError):
        data_config_io.load_data_config(bad)
