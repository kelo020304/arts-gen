"""Manual joint transforms in world-baked coordinates."""

from __future__ import annotations

import numpy as np


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError(f"zero-length axis is invalid: {vec}")
    return vec / norm


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = _unit(axis)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ])


def apply_joint_transform_world_baked(
    *,
    joint_type: str,
    direction: int,
    q_abs: float,
    axis_world: np.ndarray,
    origin_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(R, t)` that moves geometry from q=0 to signed q."""
    sign = 1 if direction >= 0 else -1
    axis = _unit(np.asarray(axis_world, dtype=float))
    origin = np.asarray(origin_world, dtype=float)
    q_signed = sign * float(q_abs)

    if joint_type == "prismatic":
        return np.eye(3), axis * q_signed
    if joint_type == "revolute":
        rotation = _axis_angle(axis, q_signed)
        translation = origin - rotation @ origin
        return rotation, translation
    raise ValueError(f"unsupported joint_type: {joint_type!r}")
