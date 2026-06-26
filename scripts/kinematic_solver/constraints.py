"""Collision constraints used by the geometric joint evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .backend import CollisionBackend
from .config import CollisionConstraintConfig
from .errors import DegenerateAxisExtentError


@dataclass
class CollisionConstraint:
    moving_parts: list[str]
    static_parts: list[str]
    backend: CollisionBackend
    config: CollisionConstraintConfig = CollisionConstraintConfig()
    initial_overlap_pairs: list[tuple[str, str]] | None = None

    def __call__(self) -> bool:
        return self.check()

    def calibrate_at_zero(self) -> bool:
        self.initial_overlap_pairs = [
            (moving, static)
            for moving in self.moving_parts
            for static in self.static_parts
            if self.backend.overlap([moving], [static])
        ]
        return not self.initial_overlap_pairs

    def check(self) -> bool:
        return not self.backend.overlap(self.moving_parts, self.static_parts)


class RetainedOverlapConstraint:
    """Keep a moving part's projected interval sufficiently overlapped with q=0."""

    def __init__(
        self,
        *,
        joint: Mapping,
        raw_vertices_by_part: Mapping[str, np.ndarray],
        min_retained_ratio: float = 0.2,
    ) -> None:
        self.joint = joint
        axis = np.asarray(joint["axis_world"], dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            raise DegenerateAxisExtentError(f"{joint.get('joint_name')}: zero axis")
        self.axis = axis / axis_norm
        self.min_retained_ratio = float(min_retained_ratio)
        projections = []
        for part in joint["moving_parts"]:
            projections.append(np.asarray(raw_vertices_by_part[part], dtype=np.float64) @ self.axis)
        proj0 = np.concatenate(projections)
        self._lo0 = float(proj0.min())
        self._hi0 = float(proj0.max())
        self._len0 = self._hi0 - self._lo0
        if self._len0 < 1e-6:
            raise DegenerateAxisExtentError(
                f"{joint.get('joint_name')}: retained-overlap axis extent {self._len0:.3e}"
            )
        self._current_q_signed = 0.0

    def calibrate_at_zero(self) -> bool:
        self._current_q_signed = 0.0
        return True

    def set_current_q(self, q_signed: float) -> None:
        self._current_q_signed = float(q_signed)

    def check(self) -> bool:
        lo = self._lo0 + self._current_q_signed
        hi = self._hi0 + self._current_q_signed
        inter = max(0.0, min(self._hi0, hi) - max(self._lo0, lo))
        return (inter / self._len0) >= self.min_retained_ratio

    def __call__(self) -> bool:
        return self.check()


class Body0GapRangeConstraint:
    """Infer a one-sided prismatic travel range from the body0-to-child axial gap."""

    def __init__(
        self,
        *,
        joint: Mapping,
        parent_vertices: np.ndarray,
        moving_vertices: np.ndarray,
        tolerance: float = 1e-9,
    ) -> None:
        axis = np.asarray(joint["axis_world"], dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            raise DegenerateAxisExtentError(f"{joint.get('joint_name')}: zero axis")
        self.axis = axis / axis_norm
        self.tolerance = float(tolerance)
        parent_lo, parent_hi = _project_interval(parent_vertices, self.axis)
        moving_lo, moving_hi = _project_interval(moving_vertices, self.axis)

        if moving_lo >= parent_hi:
            gap = moving_lo - parent_hi
            self.lower = 0.0
            self.upper = float(gap)
        elif moving_hi <= parent_lo:
            gap = parent_lo - moving_hi
            self.lower = -float(gap)
            self.upper = 0.0
        else:
            raise DegenerateAxisExtentError(
                f"{joint.get('joint_name')}: body0 and moving intervals overlap on joint axis"
            )
        self._current_q_signed = 0.0

    def calibrate_at_zero(self) -> bool:
        self._current_q_signed = 0.0
        return True

    def set_current_q(self, q_signed: float) -> None:
        self._current_q_signed = float(q_signed)

    def check(self) -> bool:
        return (
            self.lower - self.tolerance
            <= self._current_q_signed
            <= self.upper + self.tolerance
        )

    def __call__(self) -> bool:
        return self.check()


def _project_interval(vertices: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    points = np.asarray(vertices, dtype=np.float64)
    if points.size == 0:
        raise DegenerateAxisExtentError("empty vertices cannot define an axis interval")
    projections = points @ axis
    return float(projections.min()), float(projections.max())


def make_body0_gap_range_constraint(
    *,
    joint: Mapping,
    raw_vertices_by_part: Mapping[str, np.ndarray],
    tolerance: float = 1e-9,
) -> Body0GapRangeConstraint | None:
    """Create a body0 gap constraint only for separated nested prismatic joints."""
    if joint.get("type") != "prismatic":
        return None
    body0 = joint.get("body0_link_name")
    if not body0 or body0 == "body":
        return None
    moving_parts = list(joint.get("moving_parts", ()))
    if not moving_parts or body0 in moving_parts or body0 not in raw_vertices_by_part:
        return None

    moving_vertices = np.concatenate(
        [np.asarray(raw_vertices_by_part[part], dtype=np.float64) for part in moving_parts],
        axis=0,
    )
    axis = np.asarray(joint["axis_world"], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        raise DegenerateAxisExtentError(f"{joint.get('joint_name')}: zero axis")
    unit_axis = axis / axis_norm
    parent_lo, parent_hi = _project_interval(raw_vertices_by_part[body0], unit_axis)
    moving_lo, moving_hi = _project_interval(moving_vertices, unit_axis)
    if moving_lo < parent_hi and moving_hi > parent_lo:
        return None

    return Body0GapRangeConstraint(
        joint=joint,
        parent_vertices=np.asarray(raw_vertices_by_part[body0], dtype=np.float64),
        moving_vertices=moving_vertices,
        tolerance=tolerance,
    )
