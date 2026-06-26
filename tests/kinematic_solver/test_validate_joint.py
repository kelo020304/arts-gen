from pathlib import Path
from unittest.mock import patch

import pytest

from post_process.kinematic_solver.utils.config import V1_COACD_RUN_PARAMS, V1_VHACD_CACHE_METADATA
from post_process.kinematic_solver.utils.errors import InvalidValidationContextError
from post_process.kinematic_solver.utils.validate import (
    ValidationContext,
    _geometry_sanity_validate,
    validate_joint,
)


def _make_ctx(prediction, predicted_usd_path):
    return ValidationContext(
        prediction=prediction,
        vlm_oracle_model={
            "joints": {
                "j": {
                    "type": "prismatic",
                    "axis_world": [1, 0, 0],
                    "origin_world": [0, 0, 0],
                    "moving_parts": ["part_00"],
                    "static_parts": ["body"],
                }
            }
        },
        joint_name="j",
        object_id="ra_007",
        usd_path=Path("/fake/Aligned.usd"),
        predicted_usd_path=predicted_usd_path,
        part_to_obj_path={"part_00": Path("/fake/part_00.obj"), "body": Path("/fake/body.obj")},
        vhacd_cache_root=Path("/fake/vhacd"),
        coacd_run_params=dict(V1_COACD_RUN_PARAMS),
        vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
        stage_metadata={"meters_per_unit": 1.0, "joint_prim_paths": {"j": "/World/j"}},
    )


def test_validate_joint_skipped_non_ok_returns_skipped():
    ctx = _make_ctx(
        prediction={
            "status": "partial",
            "predicted_lower": -0.1,
            "predicted_upper": None,
            "status_upper": "initial_collision",
            "status_lower": "ok",
        },
        predicted_usd_path=None,
    )

    out = validate_joint(ctx)

    assert out["validation_status"] == "skipped_non_ok"
    assert out["object_id"] == "ra_007"
    assert out["joint_name"] == "j"


def test_validate_joint_raises_when_non_ok_has_predicted_path():
    ctx = _make_ctx({"status": "partial"}, Path("/tmp/should-not-exist.usd"))

    with pytest.raises(InvalidValidationContextError):
        validate_joint(ctx)


def test_validate_joint_raises_when_ok_missing_predicted_path():
    ctx = _make_ctx(
        {
            "status": "ok",
            "predicted_lower": -0.1,
            "predicted_upper": 0.1,
            "status_upper": "ok",
            "status_lower": "ok",
        },
        None,
    )

    with pytest.raises(InvalidValidationContextError):
        validate_joint(ctx)


def test_validate_joint_falls_back_to_geometry_sanity_when_isaac_unavailable():
    ctx = _make_ctx(
        {
            "status": "ok",
            "predicted_lower": -0.1,
            "predicted_upper": 0.1,
            "status_upper": "ok",
            "status_lower": "ok",
        },
        Path("/tmp/predicted.usd"),
    )

    with patch("post_process.kinematic_solver.utils.validate._isaac_runtime_available", return_value=False), \
         patch(
             "post_process.kinematic_solver.utils.validate._geometry_sanity_validate",
             return_value={"validation_status": "skipped_backend_unavailable"},
         ):
        out = validate_joint(ctx)

    assert out["validation_status"] == "skipped_backend_unavailable"


def test_geometry_sanity_validate_loads_vhacd_checks_endpoints_and_clears(monkeypatch):
    calls = {"evaluated": []}

    class FakeBackend:
        def load_model(self, **kwargs):
            calls["load_model"] = kwargs

        def clear(self):
            calls["cleared"] = True

    class FakeCollisionConstraint:
        def __init__(self, moving_parts, static_parts, backend, config=None):
            calls["constraint"] = {
                "moving_parts": moving_parts,
                "static_parts": static_parts,
                "backend": backend,
                "config": config,
            }

    class FakeJointEvaluator:
        def __init__(self, *, joint, constraints, backend):
            calls["evaluator"] = {
                "joint": joint,
                "constraints": constraints,
                "backend": backend,
            }

        def __call__(self, q_value):
            calls["evaluated"].append(q_value)
            return q_value >= 0.0

        def calibrate_at_zero(self):
            calls["calibrated"] = True
            return True

    monkeypatch.setattr("post_process.kinematic_solver.utils._fcl_backend.FclBackend", FakeBackend)
    monkeypatch.setattr(
        "post_process.kinematic_solver.utils.constraints.CollisionConstraint",
        FakeCollisionConstraint,
    )
    monkeypatch.setattr(
        "post_process.kinematic_solver.utils.joint_evaluator.JointEvaluator",
        FakeJointEvaluator,
    )
    ctx = _make_ctx(
        {
            "status": "ok",
            "predicted_lower": -0.1,
            "predicted_upper": 0.2,
            "status_upper": "ok",
            "status_lower": "ok",
        },
        Path("/tmp/predicted.usd"),
    )

    out = _geometry_sanity_validate(ctx)

    assert calls["load_model"]["object_id"] == "ra_007"
    assert calls["load_model"]["part_to_obj_path"] == ctx.part_to_obj_path
    assert calls["constraint"]["moving_parts"] == ["part_00"]
    assert calls["constraint"]["static_parts"] == ["body"]
    assert calls["constraint"]["config"].allow_initial_penetration is True
    assert calls["calibrated"] is True
    assert calls["evaluated"] == [-0.1, 0.2]
    assert calls["cleared"] is True
    assert out["validation_status"] == "skipped_backend_unavailable"
    assert out["geometry_overlap_at_lower"] is True
    assert out["geometry_overlap_at_upper"] is False
    assert out["sanity_passed"] is False
