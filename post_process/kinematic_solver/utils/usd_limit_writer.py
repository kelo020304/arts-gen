"""USD limit attr write helper for predicted limit USD copies."""

from __future__ import annotations

import math


def wrap_joint_for_limit_write(joint_prim, joint_type: str):
    """Return a schema object exposing writable limit attrs."""
    if hasattr(joint_prim, "GetLowerLimitAttr") and hasattr(joint_prim, "GetUpperLimitAttr"):
        return joint_prim
    from pxr import UsdPhysics

    if joint_type == "revolute":
        return UsdPhysics.RevoluteJoint(joint_prim)
    if joint_type == "prismatic":
        return UsdPhysics.PrismaticJoint(joint_prim)
    raise ValueError(f"unsupported joint_type: {joint_type!r}")


def write_predicted_limits(
    joint_prim,
    joint_type: str,
    pred_lower: float,
    pred_upper: float,
    meters_per_unit: float,
) -> None:
    """Write prediction limits in USD-authored units."""
    if joint_type == "revolute":
        joint_prim.GetLowerLimitAttr().Set(math.degrees(float(pred_lower)))
        joint_prim.GetUpperLimitAttr().Set(math.degrees(float(pred_upper)))
        return
    if joint_type == "prismatic":
        scale = float(meters_per_unit)
        joint_prim.GetLowerLimitAttr().Set(float(pred_lower) / scale)
        joint_prim.GetUpperLimitAttr().Set(float(pred_upper) / scale)
        return
    raise ValueError(f"unsupported joint_type: {joint_type!r}")
