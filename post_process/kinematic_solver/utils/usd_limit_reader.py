"""USD limit attr read helper for Phase 0 source metadata extraction."""

from __future__ import annotations

import math


def _joint_api(joint_prim, joint_type: str):
    if hasattr(joint_prim, "GetLowerLimitAttr") and hasattr(joint_prim, "GetUpperLimitAttr"):
        return joint_prim
    from pxr import UsdPhysics

    if joint_type == "revolute":
        return UsdPhysics.RevoluteJoint(joint_prim)
    if joint_type == "prismatic":
        return UsdPhysics.PrismaticJoint(joint_prim)
    raise ValueError(f"unsupported joint_type: {joint_type!r}")


def read_limits_from_source_usd(
    joint_prim,
    joint_type: str,
    meters_per_unit: float = 1.0,
) -> tuple[float, float]:
    """Return source limits in canonical V1 units: radians or meters."""
    joint = _joint_api(joint_prim, joint_type)
    lower = float(joint.GetLowerLimitAttr().Get())
    upper = float(joint.GetUpperLimitAttr().Get())
    if joint_type == "revolute":
        return math.radians(lower), math.radians(upper)
    if joint_type == "prismatic":
        return lower * float(meters_per_unit), upper * float(meters_per_unit)
    raise ValueError(f"unsupported joint_type: {joint_type!r}")
