"""VLM-provided initial joint guesses for the iterative limit agent."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schemas import EstimateContext, LimitEstimate


def load_vlm_initial_context(
    ctx: EstimateContext,
    json_path: Path,
) -> tuple[EstimateContext, list[LimitEstimate]]:
    """Overlay rough VLM axis/range guesses onto a context.

    Prismatic limits in the JSON are millimeters. Revolute limits are degrees.
    The returned context preserves geometry fields from the existing oracle while
    exposing the normalized initial estimate under each joint's evidence.
    """
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    object_id = payload.get("object_id")
    if object_id is not None and str(object_id) != ctx.object_id:
        raise ValueError(
            f"initial JSON object_id={object_id!r} does not match context object_id={ctx.object_id!r}"
        )

    raw_joints = payload.get("initial_joints", payload.get("joints"))
    if not isinstance(raw_joints, dict):
        raise ValueError("initial JSON must contain an object field 'initial_joints'")

    joints = {name: dict(joint) for name, joint in ctx.joints.items()}
    evidence = deepcopy(ctx.evidence)
    evidence["__action_space__"] = {
        "axis_mode": "continuous",
        "axis_step_degrees": 5.0,
        "prismatic_limit_steps_mm": [10.0, 5.0, 2.5, 1.0, 0.5],
        "revolute_limit_steps_degrees": [10.0, 5.0, 2.5, 1.0, 0.5],
    }
    estimates: list[LimitEstimate] = []

    for joint_name, raw in raw_joints.items():
        if not isinstance(raw, dict):
            raise ValueError(f"initial JSON joint {joint_name!r} must be an object")

        if joint_name in joints:
            joint = joints[joint_name]
        else:
            joint = _joint_from_initial_json(ctx.object_id, joint_name, raw, evidence)
            joints[joint_name] = joint
        joint_type = str(raw.get("type") or joint.get("type") or "")
        if joint_type not in {"revolute", "prismatic"}:
            raise ValueError(f"{joint_name}: initial joint type must be revolute or prismatic")
        joint["type"] = joint_type
        if raw.get("parent") is not None:
            joint["parent"] = str(raw["parent"])

        vlm_axis = _normalized_axis(raw.get("axis", raw.get("axis_world")), joint_name)
        axis = _normalized_axis(
            evidence.get(joint_name, {}).get("recommended_axis_world", vlm_axis),
            joint_name,
        )
        joint["axis_world"] = axis
        lower, upper, limit_unit = _convert_limit(joint_type, raw.get("limit"))
        initial_record = {
            "joint_name": joint_name,
            "type": joint_type,
            "axis_world": axis,
            "vlm_axis_world": vlm_axis,
            "lower": lower,
            "upper": upper,
            "raw_limit": raw.get("limit"),
            "limit_unit": limit_unit,
            "parent": raw.get("parent"),
            "source": "sdk_geometry_axis_with_vlm_limit",
        }
        joint_evidence = evidence.setdefault(joint_name, {})
        joint_evidence["initial_estimate"] = initial_record
        joint_evidence["joint_state"] = "need_fix"
        estimates.append(
            LimitEstimate(
                joint_name=joint_name,
                lower=lower,
                upper=upper,
                axis_world=axis,
                confidence=0.25,
                reason="VLM initial rough guess; expected to be refined by bounded agent actions.",
            )
        )

    estimates.sort(key=lambda estimate: estimate.joint_name)
    return (
        EstimateContext(object_id=ctx.object_id, joints=joints, evidence=evidence),
        estimates,
    )


def estimates_to_evidence_records(estimates: list[LimitEstimate]) -> list[dict[str, Any]]:
    return [asdict(estimate) for estimate in estimates]


def _joint_from_initial_json(
    object_id: str,
    joint_name: str,
    raw: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    available_parts = [
        str(part)
        for part in evidence.get("__available_parts__", [])
    ]
    if available_parts and joint_name not in available_parts:
        raise ValueError(
            f"initial JSON joint {joint_name!r} is not present in available mesh parts: "
            f"{', '.join(available_parts)}"
        )

    moving_parts = raw.get("moving_parts")
    if moving_parts is None:
        moving_parts = [joint_name]
    if not isinstance(moving_parts, (list, tuple)) or not moving_parts:
        raise ValueError(f"{joint_name}: moving_parts must be a non-empty array when provided")
    moving_parts = [str(part) for part in moving_parts]

    if available_parts:
        missing = sorted(set(moving_parts) - set(available_parts))
        if missing:
            raise ValueError(
                f"{joint_name}: moving_parts not present in available mesh parts: {', '.join(missing)}"
            )
        static_parts = [
            part for part in available_parts
            if part not in set(moving_parts)
        ]
    else:
        static_parts = [str(part) for part in raw.get("static_parts", [])]

    parent = str(raw.get("parent") or "body")
    origin = raw.get("origin_world", raw.get("origin"))
    if origin is None:
        part_centers = evidence.get("__part_centers__", {})
        origin = part_centers.get(joint_name, [0.0, 0.0, 0.0]) if isinstance(part_centers, dict) else [0.0, 0.0, 0.0]

    joint_type = str(raw.get("type") or "")
    return {
        "object_id": object_id,
        "joint_name": joint_name,
        "joint_path": str(raw.get("joint_path") or f"/World/{joint_name}/{joint_name}"),
        "type": joint_type,
        "canonical_unit": "meters" if joint_type == "prismatic" else "radians",
        "axis_world": _normalized_axis(raw.get("axis", raw.get("axis_world")), joint_name),
        "origin_world": _numeric_vec3(origin, f"{joint_name}: origin_world"),
        "moving_parts": moving_parts,
        "static_parts": static_parts,
        "body0_path": str(raw.get("body0_path") or f"/World/{parent}"),
        "child_body_path": str(raw.get("child_body_path") or f"/World/{joint_name}"),
        "body0_link_name": parent,
        "source": "initial_json_defined_joint",
    }


def _numeric_vec3(raw: Any, label: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{label} must be a length-3 array")
    values = [float(value) for value in raw]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain finite numbers")
    return values


def _normalized_axis(raw: Any, joint_name: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{joint_name}: initial axis must be a length-3 array")
    axis = [float(value) for value in raw]
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1e-12:
        raise ValueError(f"{joint_name}: initial axis must be non-zero")
    return [value / norm for value in axis]


def _convert_limit(joint_type: str, raw_limit: Any) -> tuple[float, float, str]:
    if raw_limit is None:
        return 0.0, 0.0, "none"
    if not isinstance(raw_limit, (list, tuple)) or len(raw_limit) != 2:
        raise ValueError("initial limit must be null or a two-number array")
    lower = float(raw_limit[0])
    upper = float(raw_limit[1])
    if joint_type == "prismatic":
        return lower / 1000.0, upper / 1000.0, "millimeter"
    return math.radians(lower), math.radians(upper), "degree"
