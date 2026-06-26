import json
from pathlib import Path
from unittest.mock import patch

from post_process.kinematic_solver.utils.config import V1DatasetRoots
from post_process.kinematic_solver.utils.phase0_verify import write_dataset_fingerprint
from post_process.kinematic_solver.utils.run_validate import _run_one_model


def test_run_validate_writes_validation_jsonl(tmp_path):
    converter = tmp_path / "converter"
    source = tmp_path / "source"
    run = tmp_path / "run"
    obj_dir = converter / "raw/partseg/ra_007/objs"
    obj_dir.mkdir(parents=True)
    (obj_dir / "body.obj").write_text("v 0 0 0\n")
    (converter / "raw/vlm_oracle").mkdir(parents=True)
    (converter / "raw/vlm_oracle/ra_007.json").write_text(json.dumps({"joints": {"j": {}}}))
    (converter / "raw/stage_metadata").mkdir(parents=True)
    (converter / "raw/stage_metadata/ra_007.json").write_text(json.dumps({
        "meters_per_unit": 1.0,
        "joint_prim_paths": {"j": "/World/j"},
    }))
    (source / "source/model/007").mkdir(parents=True)
    (source / "source/model/007/Aligned.usd").write_text("#usda 1.0\n")
    (run / "ra_007").mkdir(parents=True)
    (run / "ra_007/predictions.jsonl").write_text(json.dumps({
        "object_id": "ra_007",
        "joint_name": "j",
        "type": "prismatic",
        "status": "partial",
        "predicted_lower": 0.0,
        "predicted_upper": None,
    }) + "\n")
    roots = V1DatasetRoots(converter_output_root=converter, source_root=source)
    write_dataset_fingerprint(roots, run, ids=["ra_007"])

    with patch("post_process.kinematic_solver.utils.run_validate.validate_joint") as validate:
        validate.return_value = {
            "object_id": "ra_007",
            "joint_name": "j",
            "validation_status": "skipped_non_ok",
        }
        _run_one_model("ra_007", roots, run)

    rows = [
        json.loads(line)
        for line in (run / "ra_007/validation.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows == [{
        "object_id": "ra_007",
        "joint_name": "j",
        "validation_status": "skipped_non_ok",
    }]


def test_run_validate_ok_prediction_writes_predicted_usd_and_validates_it(tmp_path):
    converter = tmp_path / "converter"
    source = tmp_path / "source"
    run = tmp_path / "run"
    obj_dir = converter / "raw/partseg/ra_007/objs"
    obj_dir.mkdir(parents=True)
    (obj_dir / "body.obj").write_text("v 0 0 0\n")
    (obj_dir / "part_00.obj").write_text("v 1 0 0\n")
    oracle = {
        "joints": {
            "j": {
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            }
        }
    }
    (converter / "raw/vlm_oracle").mkdir(parents=True)
    (converter / "raw/vlm_oracle/ra_007.json").write_text(json.dumps(oracle))
    stage_metadata = {
        "meters_per_unit": 1.0,
        "joint_prim_paths": {"j": "/World/j"},
    }
    (converter / "raw/stage_metadata").mkdir(parents=True)
    (converter / "raw/stage_metadata/ra_007.json").write_text(json.dumps(stage_metadata))
    (source / "source/model/007").mkdir(parents=True)
    source_usd = source / "source/model/007/Aligned.usd"
    source_usd.write_text("#usda 1.0\n")
    (run / "ra_007").mkdir(parents=True)
    prediction = {
        "object_id": "ra_007",
        "joint_name": "j",
        "type": "prismatic",
        "status": "ok",
        "status_lower": "ok",
        "status_upper": "ok",
        "predicted_lower": 0.0,
        "predicted_upper": 0.1,
    }
    (run / "ra_007/predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    roots = V1DatasetRoots(converter_output_root=converter, source_root=source)
    write_dataset_fingerprint(roots, run, ids=["ra_007"])
    expected_predicted = run / "ra_007/predicted_usd/j.usd"

    def fake_write_predicted_usd_for(**kwargs):
        assert kwargs["prediction"] == prediction
        assert kwargs["source_usd_path"] == source_usd
        assert kwargs["stage_metadata"] == stage_metadata
        assert kwargs["out_path"] == expected_predicted
        expected_predicted.parent.mkdir(parents=True, exist_ok=True)
        expected_predicted.write_text("#usda 1.0\n")
        return expected_predicted

    def fake_validate_joint(ctx):
        assert ctx.predicted_usd_path == expected_predicted
        assert ctx.usd_path == source_usd
        assert ctx.vlm_oracle_model == oracle
        assert ctx.part_to_obj_path == {
            "body": obj_dir / "body.obj",
            "part_00": obj_dir / "part_00.obj",
        }
        return {
            "object_id": ctx.object_id,
            "joint_name": ctx.joint_name,
            "validation_status": "passed",
        }

    with patch(
        "post_process.kinematic_solver.utils.run_validate.write_predicted_usd_for",
        side_effect=fake_write_predicted_usd_for,
    ) as write_predicted, \
         patch(
             "post_process.kinematic_solver.utils.run_validate.validate_joint",
             side_effect=fake_validate_joint,
         ) as validate:
        _run_one_model("ra_007", roots, run)

    write_predicted.assert_called_once()
    validate.assert_called_once()
    rows = [
        json.loads(line)
        for line in (run / "ra_007/validation.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows == [{
        "object_id": "ra_007",
        "joint_name": "j",
        "validation_status": "passed",
    }]
