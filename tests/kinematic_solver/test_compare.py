from post_process.kinematic_solver.utils.compare import compare
from post_process.kinematic_solver.utils.config import ComparisonConfig


def test_compare_both_directions_ok_reports_errors_and_success():
    out = compare(
        predicted={
            "object_id": "ra_007",
            "joint_name": "joint0",
            "type": "prismatic",
            "canonical_unit": "meters",
            "predicted_lower": -0.02,
            "predicted_upper": 0.12,
            "status": "ok",
            "status_lower": "ok",
            "status_upper": "ok",
        },
        gt={"lower": 0.0, "upper": 0.10},
    )

    assert out["abs_err_lower"] == 0.02
    assert abs(out["rel_err_upper"] - 0.2) < 1e-9
    assert out["success_lower"] is False
    assert out["success_upper"] is False
    assert 0.0 <= out["iou_range"] <= 1.0


def test_compare_partial_direction_is_not_aggregate_evaluable():
    out = compare(
        predicted={
            "object_id": "ra_007",
            "joint_name": "joint0",
            "type": "prismatic",
            "canonical_unit": "meters",
            "predicted_lower": 0.0,
            "predicted_upper": None,
            "status": "partial",
            "status_lower": "ok",
            "status_upper": "initial_collision",
        },
        gt={"lower": 0.0, "upper": 0.10},
    )

    assert out["success_lower"] is True
    assert out["success_upper"] is None
    assert out["success"] is None
    assert out["iou_range"] is None


def test_compare_uses_configured_success_threshold():
    predicted = {
        "object_id": "ra_007",
        "joint_name": "joint0",
        "type": "prismatic",
        "canonical_unit": "meters",
        "predicted_lower": 0.0,
        "predicted_upper": 0.11,
        "status": "ok",
        "status_lower": "ok",
        "status_upper": "ok",
    }
    gt = {"lower": 0.0, "upper": 0.10}

    strict = compare(
        predicted,
        gt,
        config=ComparisonConfig(success_rel_err_threshold=0.05),
    )
    loose = compare(
        predicted,
        gt,
        config=ComparisonConfig(success_rel_err_threshold=0.20),
    )

    assert strict["success_upper"] is False
    assert strict["success"] is False
    assert loose["success_upper"] is True
    assert loose["success"] is True
