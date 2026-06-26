"""Coordinate-frame normalization for kinematic solver assets.

Converter outputs are world-baked in the source asset frame where -Y points
front, +X points right, and +Z points up. The agent/VLM contract uses the ROS
frame: +X front, +Y left, +Z up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .schemas import EstimateContext


CANONICAL_FRAME_NAME = "ros_x_front_y_left_z_up"
SOURCE_FRAME_NAME = "source_neg_y_axis_front_x_axis_right_z_up"
SOURCE_TO_CANONICAL_TRANSFORM = "source_neg_y_front_pos_x_right_to_ros_x_front_y_left"
CANONICAL_FRAME_EVIDENCE = {
    "name": CANONICAL_FRAME_NAME,
    "source": SOURCE_FRAME_NAME,
    "transform": SOURCE_TO_CANONICAL_TRANSFORM,
}


def context_uses_canonical_frame(ctx: EstimateContext) -> bool:
    frame = ctx.evidence.get("__coordinate_frame__", {}) or {}
    return isinstance(frame, dict) and frame.get("name") == CANONICAL_FRAME_NAME


def source_to_canonical_vector(raw: Any) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in raw)
    return (-y, -x, z)


def source_to_canonical_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return array.reshape((-1, 3))
    transformed = np.empty_like(array, dtype=np.float64)
    transformed[:, 0] = -array[:, 1]
    transformed[:, 1] = -array[:, 0]
    transformed[:, 2] = array[:, 2]
    return transformed


def with_canonical_coordinate_frame(ctx: EstimateContext) -> EstimateContext:
    if context_uses_canonical_frame(ctx):
        return ctx
    joints = {}
    for joint_name, joint in ctx.joints.items():
        updated = dict(joint)
        if _is_vec3(updated.get("axis_world")):
            updated["axis_world"] = list(source_to_canonical_vector(updated["axis_world"]))
        if _is_vec3(updated.get("origin_world")):
            updated["origin_world"] = list(source_to_canonical_vector(updated["origin_world"]))
        joints[joint_name] = updated
    evidence = {name: dict(value) if isinstance(value, dict) else value for name, value in ctx.evidence.items()}
    part_centers = evidence.get("__part_centers__")
    if isinstance(part_centers, dict):
        evidence["__part_centers__"] = {
            str(part): list(source_to_canonical_vector(center))
            for part, center in part_centers.items()
            if _is_vec3(center)
        }
    evidence["__coordinate_frame__"] = dict(CANONICAL_FRAME_EVIDENCE)
    return EstimateContext(object_id=ctx.object_id, joints=joints, evidence=evidence)


def copy_obj_as_canonical(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in Path(source).read_text(errors="ignore").splitlines():
        if line.startswith("v ") or line.startswith("vn "):
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = source_to_canonical_vector(parts[1:4])
                lines.append(f"{parts[0]} {x:.10g} {y:.10g} {z:.10g}")
                continue
        lines.append(line)
    Path(target).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_vec3(raw: Any) -> bool:
    return isinstance(raw, (list, tuple)) and len(raw) == 3
