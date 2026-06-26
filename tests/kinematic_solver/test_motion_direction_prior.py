import pytest

from post_process.kinematic_solver.utils.motion_direction_prior import (
    apply_motion_direction_prior,
    part_semantics_from_gt_part,
)


def test_part_semantics_from_gt_part_inverts_label_mapping():
    semantics = part_semantics_from_gt_part({
        "pull-out drawer pan": "part_02",
        "temperature knob": "part_00",
    })

    assert semantics == {
        "part_02": ["pull-out drawer pan"],
        "part_00": ["temperature knob"],
    }


def test_pull_out_drawer_semantics_clamps_prismatic_prediction_to_positive_side():
    prediction = {
        "object_id": "ra_063",
        "joint_name": "part_02",
        "type": "prismatic",
        "canonical_unit": "meters",
        "predicted_lower": -0.15,
        "predicted_upper": 0.15,
        "status": "ok",
        "status_lower": "ok",
        "status_upper": "ok",
    }
    joint = {
        "type": "prismatic",
        "moving_parts": ["part_02"],
    }

    out = apply_motion_direction_prior(
        prediction,
        joint=joint,
        part_semantics={"part_02": ["pull-out drawer pan"]},
    )

    assert out["predicted_lower"] == 0.0
    assert out["predicted_upper"] == pytest.approx(0.15)
    assert out["status"] == "ok"
    assert out["status_lower"] == "ok"
    assert out["status_upper"] == "ok"
    assert out["motion_direction_prior"]["policy"] == "positive_only"
    assert out["motion_direction_prior"]["raw_predicted_lower"] == pytest.approx(-0.15)


def test_non_drawer_semantics_leaves_prediction_unchanged():
    prediction = {
        "object_id": "ra_063",
        "joint_name": "part_00",
        "type": "prismatic",
        "canonical_unit": "meters",
        "predicted_lower": -0.02,
        "predicted_upper": 0.02,
        "status": "ok",
        "status_lower": "ok",
        "status_upper": "ok",
    }

    out = apply_motion_direction_prior(
        prediction,
        joint={"type": "prismatic", "moving_parts": ["part_00"]},
        part_semantics={"part_00": ["temperature knob"]},
    )

    assert out == prediction


def test_panel_label_does_not_match_pan_keyword():
    prediction = {
        "object_id": "ra_test",
        "joint_name": "part_03",
        "type": "prismatic",
        "canonical_unit": "meters",
        "predicted_lower": -0.02,
        "predicted_upper": 0.02,
        "status": "ok",
        "status_lower": "ok",
        "status_upper": "ok",
    }

    out = apply_motion_direction_prior(
        prediction,
        joint={"type": "prismatic", "moving_parts": ["part_03"]},
        part_semantics={"part_03": ["sliding control panel"]},
    )

    assert out == prediction
