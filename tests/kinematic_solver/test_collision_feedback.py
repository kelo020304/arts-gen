from __future__ import annotations

import pytest

from post_process.kinematic_solver.sdk.collision_feedback import (
    CollisionFeedbackConfig,
    propose_collision_clear_interval,
)
from post_process.kinematic_solver.sdk.kin_agent import KinematicCandidate


trimesh = pytest.importorskip("trimesh")
pytest.importorskip("manifold3d")


def _candidate(lower=-1.0, upper=1.0, *, signals=None):
    return KinematicCandidate(
        joint_type="prismatic",
        axis_world=(1.0, 0.0, 0.0),
        origin_world=(0.0, 0.0, 0.0),
        lower=lower,
        upper=upper,
        score=1.0,
        signals=signals or {},
    )


def _meshes(obstacle_centers=(-0.6, 0.6)):
    obstacles = []
    for center in obstacle_centers:
        obstacle = trimesh.creation.box(extents=(0.2, 0.5, 0.5))
        obstacle.apply_translation((center, 0.0, 0.0))
        obstacles.append(obstacle)
    body = trimesh.util.concatenate(obstacles)
    moving = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    return body, moving


def _audit(*, q_samples=None, exact=True, volume=True):
    return {
        "mesh_state": {
            "body_is_volume": volume,
            "moving_is_volume": volume,
        },
        "q_samples": q_samples or [
            {"q": -1.0, "invalid": False},
            {"q": -0.5, "invalid": True},
            {"q": 0.0, "invalid": False},
            {"q": 0.5, "invalid": True},
            {"q": 1.0, "invalid": False},
        ],
        "narrow_phase": {
            "exact": exact,
            "invalid_threshold": 0.005,
        },
    }


def test_proposes_zero_connected_two_sided_clear_interval_with_bounded_search():
    body, moving = _meshes()

    result = propose_collision_clear_interval(body, moving, _candidate(), _audit())

    assert result["status"] == "accepted"
    assert result["accept_gate"]["accepted"] is True
    assert result["requires_review"] is False
    assert -0.41 < result["proposal"]["lower"] < -0.39
    assert 0.39 < result["proposal"]["upper"] < 0.41
    assert result["retained_fraction"] == pytest.approx(0.4, abs=0.01)
    assert result["proposal"]["negative_side"]["bisections"] == 6
    assert result["proposal"]["positive_side"]["bisections"] == 6
    # Baseline plus at most endpoint + six midpoint evaluations on each side.
    assert len(result["exact_evaluations"]) <= 15


def test_rejects_proposal_that_retains_less_than_default_fraction():
    body, moving = _meshes(obstacle_centers=(-0.25, 0.25))
    samples = [
        {"q": -1.0, "invalid": False},
        {"q": -0.1, "invalid": True},
        {"q": 0.0, "invalid": False},
        {"q": 0.1, "invalid": True},
        {"q": 1.0, "invalid": False},
    ]

    result = propose_collision_clear_interval(
        body, moving, _candidate(), _audit(q_samples=samples),
    )

    assert result["status"] == "rejected"
    assert result["accept_gate"]["accepted"] is False
    assert result["accept_gate"]["retained_fraction_ok"] is False
    assert "retained_fraction_below_minimum" in result["accept_gate"]["reasons"]
    assert result["requires_review"] is True


def test_rejects_shrink_that_conflicts_with_observed_motion_interval():
    body, moving = _meshes()
    candidate = _candidate(signals={
        "motion_observed_span": 1.0,
        "motion_observed_lower": -0.5,
        "motion_observed_upper": 0.5,
    })

    result = propose_collision_clear_interval(body, moving, candidate, _audit())

    assert result["status"] == "rejected"
    assert result["accept_gate"]["retained_fraction_ok"] is True
    assert result["accept_gate"]["observed_motion_preserved"] is False
    assert "proposal_conflicts_with_observed_motion" in result["review_reason"]


@pytest.mark.parametrize("exact,volume", [(False, True), (True, False)])
def test_exact_unavailable_or_non_volume_mesh_requires_review(exact, volume):
    body, moving = _meshes()

    result = propose_collision_clear_interval(
        body, moving, _candidate(), _audit(exact=exact, volume=volume),
    )

    assert result["status"] == "review"
    assert result["proposal"] is None
    assert result["accept_gate"]["accepted"] is False
    assert result["requires_review"] is True
    assert result["review_reason"] == "exact_audit_unavailable_or_mesh_non_watertight"


def test_actual_non_watertight_mesh_requires_review_even_if_audit_claims_volume():
    body, moving = _meshes()
    body.update_faces([index for index in range(len(body.faces)) if index != 0])

    result = propose_collision_clear_interval(body, moving, _candidate(), _audit())

    assert result["status"] == "review"
    assert result["requires_review"] is True
    assert result["review_reason"] == "decoded_mesh_non_watertight_or_not_volume"


def test_exact_clear_broad_false_positive_does_not_shrink_candidate():
    body, moving = _meshes(obstacle_centers=(-2.0, 2.0))

    result = propose_collision_clear_interval(body, moving, _candidate(), _audit())

    assert result["status"] == "no_change"
    assert result["proposal"]["lower"] == -1.0
    assert result["proposal"]["upper"] == 1.0
    assert result["proposal"]["changed"] is False
    assert result["requires_review"] is False


def test_config_enforces_six_bisection_cap():
    with pytest.raises(ValueError, match="in \\[0, 6\\]"):
        CollisionFeedbackConfig(max_bisections_per_side=7)
