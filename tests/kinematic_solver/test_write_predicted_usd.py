from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from post_process.kinematic_solver.utils.write_predicted_usd import write_predicted_usd_for


def test_write_predicted_usd_for_returns_none_on_non_ok():
    out = write_predicted_usd_for(
        prediction={
            "status": "partial",
            "predicted_lower": 0.0,
            "predicted_upper": None,
            "joint_name": "j",
            "object_id": "ra_007",
            "type": "prismatic",
        },
        source_usd_path=None,
        stage_metadata={"meters_per_unit": 1.0, "joint_prim_paths": {}},
        out_path=None,
    )

    assert out is None


def test_write_predicted_usd_for_copies_source_and_saves_stage(tmp_path):
    source = tmp_path / "Aligned.usd"
    source.write_text("#usda 1.0\n")
    out_path = tmp_path / "predicted.usd"
    stage = MagicMock()
    prim = MagicMock()
    stage.GetPrimAtPath.return_value = prim

    with patch("post_process.kinematic_solver.utils.write_predicted_usd.Usd") as usd:
        usd.Stage.Open.return_value = stage
        written = write_predicted_usd_for(
            prediction={
                "status": "ok",
                "predicted_lower": -0.1,
                "predicted_upper": 0.2,
                "joint_name": "j",
                "object_id": "ra_007",
                "type": "prismatic",
            },
            source_usd_path=source,
            stage_metadata={
                "meters_per_unit": 1.0,
                "joint_prim_paths": {"j": "/World/joint"},
            },
            out_path=out_path,
        )

    assert written == out_path
    assert out_path.read_text() == "#usda 1.0\n"
    stage.GetPrimAtPath.assert_called_once_with("/World/joint")
    prim.GetLowerLimitAttr.return_value.Set.assert_called_once_with(pytest.approx(-0.1))
    prim.GetUpperLimitAttr.return_value.Set.assert_called_once_with(pytest.approx(0.2))
    stage.GetRootLayer.return_value.Save.assert_called_once()


def test_write_predicted_usd_for_ok_requires_paths():
    with pytest.raises(AssertionError):
        write_predicted_usd_for(
            prediction={
                "status": "ok",
                "predicted_lower": 0.0,
                "predicted_upper": 1.0,
                "joint_name": "j",
                "object_id": "ra_007",
                "type": "prismatic",
            },
            source_usd_path=None,
            stage_metadata={"meters_per_unit": 1.0, "joint_prim_paths": {"j": "/World/j"}},
            out_path=Path("/tmp/nope.usd"),
        )
