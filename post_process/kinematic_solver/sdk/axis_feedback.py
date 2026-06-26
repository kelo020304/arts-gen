"""Bounded axis updates derived from validation feedback."""

from __future__ import annotations

import math
import re
from typing import Any


def next_axis_from_feedback(
    feedback: str,
    joint_name: str,
    fallback_axis: Any,
    *,
    max_step_degrees: float = 5.0,
) -> dict[str, Any] | None:
    """Return the next bounded axis action from a joint-axis feedback block.

    The validation error contains both the target motion axis and the candidate
    axis that was actually validated in the previous iteration. The candidate
    axis must be used as the current action state; otherwise each iteration
    restarts from the initial/recommended axis and never accumulates progress.
    """
    block = _feedback_block_for_joint(feedback, joint_name)
    if not block:
        return None
    target_axis = _parse_named_axis(block, "target_axis_world")
    if target_axis is None:
        return None
    previous_candidate_axis = _parse_named_axis(block, "candidate_axis_world")
    current_axis = previous_candidate_axis or _unit_axis(fallback_axis)
    if current_axis is None:
        return None
    next_axis = _rotate_toward(current_axis, target_axis, max_step_degrees)
    return {
        "current_axis": list(current_axis),
        "target_axis": list(target_axis),
        "next_axis": list(next_axis),
        "used_previous_candidate_axis": previous_candidate_axis is not None,
        "max_step_degrees": float(max_step_degrees),
        "remaining_degrees_before_step": _angle_degrees(current_axis, target_axis),
        "remaining_degrees_after_step": _angle_degrees(next_axis, target_axis),
    }


def _feedback_block_for_joint(feedback: str, joint_name: str) -> str:
    feedback_l = str(feedback or "").lower()
    joint_l = str(joint_name).lower()
    if joint_l not in feedback_l:
        return ""
    idx = feedback_l.find("joint=" + joint_l)
    if idx < 0:
        idx = feedback_l.find("joint: " + joint_l)
    if idx < 0:
        idx = feedback_l.find(joint_l)
    if idx < 0:
        return feedback_l
    nxt = feedback_l.find("\n[", idx + 1)
    end = nxt if nxt >= 0 else len(feedback_l)
    return feedback_l[idx:end]


def _parse_named_axis(block: str, field_name: str) -> tuple[float, float, float] | None:
    pattern = (
        re.escape(field_name.lower())
        + r"\s*=\s*\[\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
        + r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
        + r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\]"
    )
    match = re.search(pattern, block)
    if not match:
        return None
    return _unit_axis([float(match.group(1)), float(match.group(2)), float(match.group(3))])


def _unit_axis(raw: Any) -> tuple[float, float, float] | None:
    if raw is None or len(raw) != 3:
        return None
    axis = tuple(float(value) for value in raw)
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in axis)


def _angle_degrees(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))
    return math.degrees(math.acos(dot))


def _rotate_toward(
    current: tuple[float, float, float],
    target: tuple[float, float, float],
    max_step_degrees: float,
) -> tuple[float, float, float]:
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(current, target, strict=True))))
    angle = math.acos(dot)
    if angle <= 1e-12:
        return current
    step = min(math.radians(float(max_step_degrees)), angle)
    tangent = (
        target[0] - dot * current[0],
        target[1] - dot * current[1],
        target[2] - dot * current[2],
    )
    tangent_unit = _unit_axis(tangent)
    if tangent_unit is None:
        return current
    return _unit_axis(
        [
            math.cos(step) * current[0] + math.sin(step) * tangent_unit[0],
            math.cos(step) * current[1] + math.sin(step) * tangent_unit[1],
            math.cos(step) * current[2] + math.sin(step) * tangent_unit[2],
        ]
    ) or current
