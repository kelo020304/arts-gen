"""Symmetric-prior baseline for joint range estimation."""

from __future__ import annotations

import math

import numpy as np


def symmetric_prior_predict(
    *,
    joint_type: str,
    axis_world,
    moving_vertices: np.ndarray,
) -> dict:
    axis = np.asarray(axis_world, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("axis_world must be non-zero")
    axis = axis / norm

    if joint_type == "prismatic":
        projections = np.asarray(moving_vertices, dtype=np.float64) @ axis
        extent = float(projections.max() - projections.min())
        return {
            "status": "ok",
            "predicted_lower": -0.5 * extent,
            "predicted_upper": 0.5 * extent,
            "status_lower": "ok",
            "status_upper": "ok",
            "type": "prismatic",
            "canonical_unit": "meters",
        }
    if joint_type == "revolute":
        return {
            "status": "ok",
            "predicted_lower": -math.pi / 2,
            "predicted_upper": math.pi / 2,
            "status_lower": "ok",
            "status_upper": "ok",
            "type": "revolute",
            "canonical_unit": "radians",
        }
    raise ValueError(f"unsupported joint_type: {joint_type!r}")
