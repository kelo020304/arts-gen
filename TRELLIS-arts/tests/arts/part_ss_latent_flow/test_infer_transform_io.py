import json, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from inference_pipeline import transform_io


def test_meta_roundtrip(tmp_path):
    transform_io.write_meta(tmp_path, mode="A", view="four", object_id="123", run_id="r", ckpts={"part_flow":"/x"})
    m = transform_io.read_meta(tmp_path)
    assert m["mode"]=="A" and m["object_id"]=="123"
    assert m["stage_status"]=={"ss":"pending","part":"pending","slat":"pending","assemble":"pending"}


def test_set_status(tmp_path):
    transform_io.write_meta(tmp_path, mode="A", view="four", object_id="1", run_id="r", ckpts={})
    transform_io.set_stage_status(tmp_path, "ss", "done")
    assert transform_io.read_meta(tmp_path)["stage_status"]["ss"]=="done"


def test_write_meta_resets_status_when_mode_changes(tmp_path):
    transform_io.write_meta(tmp_path, mode="B", view="four", object_id="1", run_id="r", ckpts={})
    transform_io.set_stage_status(tmp_path, "ss", "done")
    transform_io.write_transform(tmp_path, resolution=64, scale=2.0, offset=[0.0, 0.0, 0.0])

    transform_io.write_meta(tmp_path, mode="A", view="four", object_id="1", run_id="r", ckpts={})
    meta = transform_io.read_meta(tmp_path)

    assert meta["mode"] == "A"
    assert meta["stage_status"] == {"ss": "pending", "part": "pending", "slat": "pending", "assemble": "pending"}
    assert meta["transform"] is None


def test_write_meta_preserves_status_for_same_run_contract(tmp_path):
    transform_io.write_meta(tmp_path, mode="A", view="four", object_id="1", run_id="r", ckpts={})
    transform_io.set_stage_status(tmp_path, "ss", "done")

    transform_io.write_meta(tmp_path, mode="A", view="four", object_id="1", run_id="r", ckpts={})

    assert transform_io.read_meta(tmp_path)["stage_status"]["ss"] == "done"


def test_transform_missing(tmp_path):
    transform_io.write_transform(tmp_path, resolution=64, scale=None, offset=None)
    t = json.loads((tmp_path/"transform.json").read_text())
    assert t["applied_to_assets"] is False and t["grid_to_world"] is None and t["normalization"] is None
    assert t["transform_source"]=="missing"


def test_transform_present(tmp_path):
    transform_io.write_transform(tmp_path, resolution=64, scale=2.0, offset=[0.0,0.0,0.0])
    t = json.loads((tmp_path/"transform.json").read_text())
    assert np.array(t["grid_to_world"]).shape==(4,4)
    assert t["normalization"]=={"offset":[0.0,0.0,0.0],"scale":2.0}
