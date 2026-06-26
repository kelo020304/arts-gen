import json
from pathlib import Path

from post_process.kinematic_solver.sdk.mjcf_preview import (
    write_iteration_mjcf_preview,
    write_rest_mjcf_preview,
)
from post_process.kinematic_solver.sdk.schemas import EstimateContext, LimitEstimate


def _write_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
            ]
        )
    )


def test_iteration_mjcf_preview_materializes_manifest_for_full_range_playback(tmp_path):
    converter_root = tmp_path / "converter"
    _write_obj(converter_root / "raw/partseg/ra_063/objs/body.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/part_00.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/part_02.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/static_trim.obj")
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_00": {
                "joint_name": "part_00",
                "type": "revolute",
                "canonical_unit": "radians",
                "origin_world": [0.0, -0.026, 0.099],
                "axis_world": [0.0, 1.0, 0.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body", "part_02"],
            },
            "part_02": {
                "joint_name": "part_02",
                "type": "prismatic",
                "canonical_unit": "meters",
                "origin_world": [0.0, -0.011, -0.054],
                "axis_world": [0.0, 1.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body", "part_00"],
            },
        },
        evidence={},
    )

    result = write_iteration_mjcf_preview(
        ctx,
        [
            LimitEstimate(
                joint_name="part_00",
                lower=-2.6,
                upper=0.0,
                axis_world=[0.0, 1.0, 0.0],
                axis_label="+Y",
            ),
            LimitEstimate(
                joint_name="part_02",
                lower=0.0,
                upper=0.15,
                axis_world=[0.0, -1.0, 0.0],
                axis_label="-Y",
            ),
        ],
        converter_output_root=converter_root,
        run_dir=tmp_path / "run",
        iteration=3,
    )

    assert result["iteration"] == 3
    assert result["asset_name"].startswith("ks_ra_063_")
    assert result["manifest"]["status"] == "ok"
    joints = {joint["name"]: joint for joint in result["manifest"]["joints"]}
    assert joints["part_00"]["type"] == "hinge"
    assert joints["part_00"]["range"] == [-2.6, 0.0]
    assert joints["part_00"]["axis"] == [0.0, 1.0, 0.0]
    assert joints["part_02"]["type"] == "slide"
    assert joints["part_02"]["range"] == [0.0, 0.15]
    assert joints["part_02"]["axis"] == [0.0, -1.0, 0.0]
    assert (Path(result["asset_dir"]) / "mjcf" / "assets" / "part_02.obj").is_file()


def test_iteration_mjcf_preview_writes_frontend_state_payload(tmp_path):
    converter_root = tmp_path / "converter"
    _write_obj(converter_root / "raw/partseg/ra_063/objs/body.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/part_02.obj")
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "joint_name": "part_02",
                "type": "prismatic",
                "canonical_unit": "meters",
                "origin_world": [0.0, 0.0, 0.0],
                "axis_world": [0.0, 1.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )

    result = write_iteration_mjcf_preview(
        ctx,
        [LimitEstimate(joint_name="part_02", lower=0.0, upper=0.1)],
        converter_output_root=converter_root,
        run_dir=tmp_path / "run",
        iteration=1,
        joint_states={
            "part_02": {
                "status": "need_fix",
                "errors": ["upper too small"],
            }
        },
    )

    state = json.loads((Path(result["run_dir"]) / "frontend_state.json").read_text())
    assert state["latest_iteration"] == 1
    assert state["latest_preview"]["asset_name"] == result["asset_name"]
    assert state["latest_preview"]["playback"]["mode"] == "sequential_full_range"
    assert state["latest_preview"]["joint_states"]["part_02"]["status"] == "need_fix"


def test_iteration_mjcf_preview_keeps_motion_search_as_diagnostics_not_playback(tmp_path):
    converter_root = tmp_path / "converter"
    _write_obj(converter_root / "raw/partseg/ra_063/objs/body.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/part_02.obj")
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "joint_name": "part_02",
                "type": "prismatic",
                "canonical_unit": "meters",
                "origin_world": [0.0, 0.0, 0.0],
                "axis_world": [0.0, 1.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )

    result = write_iteration_mjcf_preview(
        ctx,
        [LimitEstimate(joint_name="part_02", lower=0.0, upper=0.1, axis_world=[0.0, -1.0, 0.0])],
        converter_output_root=converter_root,
        run_dir=tmp_path / "run",
        iteration=1,
        motion_search=[
            {
                "joint_name": "part_02",
                "selected_axis_label": "-Y",
                "selected_limit": 0.1,
                "axis_trials": [
                    {
                        "axis_label": "+Y",
                        "axis_world": [0.0, 1.0, 0.0],
                        "limit": 0.02,
                        "samples": [{"q": 0.0, "valid": True}, {"q": 0.02, "valid": True}],
                    },
                    {
                        "axis_label": "-Y",
                        "axis_world": [0.0, -1.0, 0.0],
                        "limit": 0.1,
                        "samples": [{"q": 0.0, "valid": True}, {"q": 0.1, "valid": True}],
                    },
                ],
            }
        ],
    )

    playback = result["playback"]
    assert playback["mode"] == "sequential_full_range"
    assert "trials" not in playback
    state = json.loads((Path(result["run_dir"]) / "frontend_state.json").read_text())
    assert state["latest_preview"]["motion_search"][0]["selected_axis_label"] == "-Y"
    assert "trials" not in state["latest_preview"]["playback"]


def test_iteration_mjcf_preview_can_stop_at_manual_sliders_for_final_pass(tmp_path):
    converter_root = tmp_path / "converter"
    _write_obj(converter_root / "raw/partseg/ra_063/objs/body.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/part_02.obj")
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "joint_name": "part_02",
                "type": "prismatic",
                "canonical_unit": "meters",
                "origin_world": [0.0, 0.0, 0.0],
                "axis_world": [0.0, 1.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )

    result = write_iteration_mjcf_preview(
        ctx,
        [LimitEstimate(joint_name="part_02", lower=0.0, upper=0.1)],
        converter_output_root=converter_root,
        run_dir=tmp_path / "run",
        iteration=2,
        manual_sliders=True,
    )

    assert result["playback"]["mode"] == "manual_sliders"
    state = json.loads((Path(result["run_dir"]) / "frontend_state.json").read_text())
    assert state["latest_preview"]["playback"]["mode"] == "manual_sliders"


def test_rest_mjcf_preview_loads_asset_before_limits_are_known(tmp_path):
    converter_root = tmp_path / "converter"
    _write_obj(converter_root / "raw/partseg/ra_063/objs/body.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/part_00.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/part_02.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/static_trim.obj")
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_00": {
                "joint_name": "part_00",
                "type": "revolute",
                "origin_world": [0.0, -0.026, 0.099],
                "axis_world": [0.0, 0.0, 1.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body", "part_02"],
            },
            "part_02": {
                "joint_name": "part_02",
                "type": "prismatic",
                "origin_world": [0.0, -0.011, -0.054],
                "axis_world": [0.0, 1.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body", "part_00"],
            },
        },
        evidence={},
    )

    result = write_rest_mjcf_preview(
        ctx,
        converter_output_root=converter_root,
        run_dir=tmp_path / "run",
    )

    assert result["iteration"] == 0
    assert result["preview_kind"] == "rest"
    assert result["estimates"] == []
    assert result["manifest"]["status"] == "ok"
    assert result["manifest"]["joints"] == []
    body_names = {body["name"] for body in result["manifest"]["bodies"]}
    assert {"body", "part_00", "part_02", "static_trim"}.issubset(body_names)
    assert (Path(result["asset_dir"]) / "mjcf" / "assets" / "part_00.obj").is_file()
    assert (Path(result["asset_dir"]) / "mjcf" / "assets" / "static_trim.obj").is_file()
    state = json.loads((Path(result["run_dir"]) / "frontend_state.json").read_text())
    assert state["latest_iteration"] == 0
    assert state["latest_preview"]["preview_kind"] == "rest"
    assert state["latest_preview"]["playback"]["mode"] == "rest_pose"
