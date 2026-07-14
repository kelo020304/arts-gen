from __future__ import annotations

import importlib.util
import math

import pytest

from post_process.kinematic_solver.sdk.collision_audit import (
    DecodedCollisionAuditConfig,
    audit_joint_collision,
)
from post_process.kinematic_solver.sdk.kin_agent import KinematicCandidate


trimesh = pytest.importorskip("trimesh")
pytest.importorskip("open3d")


MANIFOLD_AVAILABLE = importlib.util.find_spec("manifold3d") is not None


CONFIG = DecodedCollisionAuditConfig(
    max_surface_points=4000,
    min_q_samples=9,
    max_q_samples=41,
    max_narrow_samples=5,
)


def _candidate(kind, axis, origin, lower, upper):
    return KinematicCandidate(
        joint_type=kind,
        axis_world=axis,
        origin_world=origin,
        lower=lower,
        upper=upper,
        score=1.0,
    )


def test_clean_slide_is_collision_free():
    body = trimesh.creation.box(extents=(0.2, 1.0, 1.0))
    body.apply_translation((-0.4, 0.0, 0.0))
    moving = trimesh.creation.box(extents=(0.2, 0.4, 0.4))
    moving.apply_translation((0.1, 0.0, 0.0))
    candidate = _candidate("prismatic", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.5)

    report = audit_joint_collision(body, moving, candidate, config=CONFIG)

    assert report["status"] == ("clear" if MANIFOLD_AVAILABLE else "approximate_unverified")
    assert report["collision_detected"] is False
    assert report["requires_review"] is (not MANIFOLD_AVAILABLE)
    assert report["narrow_phase"]["exact"] is MANIFOLD_AVAILABLE
    if MANIFOLD_AVAILABLE:
        assert report["recommended_actions"] == []
    else:
        assert report["narrow_phase"]["error"] == "manifold3d unavailable"


def test_colliding_slide_is_detected_between_endpoints():
    body = trimesh.creation.box(extents=(0.12, 1.0, 1.0))
    body.apply_translation((0.48, 0.0, 0.0))
    moving = trimesh.creation.box(extents=(0.2, 0.4, 0.4))
    moving.apply_translation((0.1, 0.0, 0.0))
    candidate = _candidate("prismatic", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.8)

    report = audit_joint_collision(body, moving, candidate, config=CONFIG)

    assert report["status"] == ("collision" if MANIFOLD_AVAILABLE else "approximate_collision")
    assert report["collision_detected"] is True
    assert report["requires_review"] is True
    assert report["first_invalid_q"] is not None
    assert 0.2 < report["first_invalid_q"] < 0.7
    assert report["recommended_actions"] == [
        "verify_signed_axis",
        "shrink_range_or_repair_segmentation_boundary",
    ]


def test_clean_hinge_is_collision_free():
    body = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    body.apply_translation((-1.0, -1.0, 0.0))
    moving = trimesh.creation.box(extents=(1.0, 0.08, 0.08))
    moving.apply_translation((0.5, 0.0, 0.0))
    candidate = _candidate("revolute", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.0, math.pi / 2.0)

    report = audit_joint_collision(body, moving, candidate, config=CONFIG)

    assert report["status"] == ("clear" if MANIFOLD_AVAILABLE else "approximate_unverified")
    assert report["collision_detected"] is False
    assert report["requires_review"] is (not MANIFOLD_AVAILABLE)
    if MANIFOLD_AVAILABLE:
        assert report["recommended_actions"] == []
    else:
        assert report["narrow_phase"]["error"] == "manifold3d unavailable"


def test_colliding_hinge_is_detected_inside_sweep():
    body = trimesh.creation.box(extents=(0.18, 0.18, 0.3))
    body.apply_translation((0.5, 0.5, 0.0))
    moving = trimesh.creation.box(extents=(1.0, 0.1, 0.1))
    moving.apply_translation((0.5, 0.0, 0.0))
    candidate = _candidate("revolute", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.0, math.pi / 2.0)

    report = audit_joint_collision(body, moving, candidate, config=CONFIG)

    assert report["status"] == ("collision" if MANIFOLD_AVAILABLE else "approximate_collision")
    assert report["collision_detected"] is True
    assert report["first_invalid_q"] is not None
    assert 0.4 < report["first_invalid_q"] < 1.2
    assert report["recommended_actions"] == [
        "revise_hinge_origin",
        "shrink_range_if_no_clear_origin",
    ]


def test_non_watertight_mesh_never_silently_passes():
    body = trimesh.creation.box(extents=(0.2, 1.0, 1.0))
    body.update_faces([index for index in range(len(body.faces)) if index != 0])
    moving = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    moving.apply_translation((1.0, 0.0, 0.0))
    candidate = _candidate("prismatic", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.2)

    report = audit_joint_collision(body, moving, candidate, config=CONFIG)

    assert report["status"] == "approximate_unverified"
    assert report["confidence"] == "low"
    assert report["requires_review"] is True
    assert report["narrow_phase"]["exact"] is False
    assert report["recommended_actions"] == ["review_non_watertight_collision_geometry"]
