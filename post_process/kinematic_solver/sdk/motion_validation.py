"""Sample candidate joint motion and reject physically invalid actions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import numpy as np

from post_process.kinematic_solver.utils.backend import CollisionBackend, make_backend
from post_process.kinematic_solver.utils.config import (
    CollisionConstraintConfig,
    V1_COACD_RUN_PARAMS,
    V1_VHACD_CACHE_METADATA,
)
from post_process.kinematic_solver.utils.constraints import (
    CollisionConstraint,
    RetainedOverlapConstraint,
)
from post_process.kinematic_solver.utils.joint_evaluator import JointEvaluator
from post_process.kinematic_solver.utils.manual_transform import apply_joint_transform_world_baked

from .motion_search import AXIS_ACTIONS, AxisAction, search_axis_actions
from .schemas import EstimateContext, LimitEstimate
from .coordinate_frame import (
    SOURCE_TO_CANONICAL_TRANSFORM,
    context_uses_canonical_frame,
    source_to_canonical_points,
)


@dataclass(frozen=True)
class MotionSearchValidationResult:
    errors: list[str]
    traces: list[dict]


def validate_motion_samples(
    ctx: EstimateContext,
    estimates: Iterable[LimitEstimate],
    *,
    backend: CollisionBackend,
    sample_count: int = 17,
    strict_articraft: bool = True,
) -> list[str]:
    """Validate candidate ranges by applying sampled joint motion to geometry."""
    errors: list[str] = []
    for estimate in estimates:
        if estimate.joint_name not in ctx.joints:
            continue
        joint = _joint_with_estimate_axis(ctx, estimate)
        missing = [
            key
            for key in ("axis_world", "origin_world", "moving_parts", "static_parts")
            if not joint.get(key)
        ]
        if missing:
            errors.append(
                f"{estimate.joint_name}: cannot run motion validation without "
                "axis_world, origin_world, moving_parts, static_parts"
            )
            continue

        constraint = CollisionConstraint(
            list(joint["moving_parts"]),
            list(joint["static_parts"]),
            backend=backend,
            config=CollisionConstraintConfig(allow_initial_penetration=True),
            use_exact_mesh=strict_articraft,
        )
        evaluator = JointEvaluator(
            joint=joint,
            constraints=[constraint],
            backend=backend,
        )
        evaluator.calibrate_at_zero()
        for q in _range_samples(float(estimate.lower), float(estimate.upper), sample_count):
            if not evaluator(q):
                errors.append(
                    f"{estimate.joint_name}: candidate motion collides/interferes at q={q:.6g}"
                )
                break
    return errors


def validate_motion_search(
    ctx: EstimateContext,
    estimates: Iterable[LimitEstimate],
    *,
    backend: CollisionBackend,
    raw_vertices_by_part: dict[str, np.ndarray] | None = None,
    sample_count: int = 17,
) -> MotionSearchValidationResult:
    """Run SDK-owned signed-axis search and validate estimates against its trace."""
    estimates = list(estimates)
    if _uses_continuous_axis_action_space(ctx):
        return _validate_continuous_candidate_motion(
            ctx,
            estimates,
            backend=backend,
            raw_vertices_by_part=raw_vertices_by_part,
            sample_count=sample_count,
        )

    errors: list[str] = []
    traces: list[dict] = []

    for estimate in estimates:
        if estimate.joint_name not in ctx.joints:
            continue
        search_joint = _joint_from_context(ctx, estimate.joint_name)
        candidate_joint = _joint_with_estimate_axis(ctx, estimate)
        missing = [
            key
            for key in ("axis_world", "origin_world", "moving_parts", "static_parts")
            if not search_joint.get(key)
        ]
        if missing:
            errors.append(
                f"{estimate.joint_name}: cannot run motion validation without "
                "axis_world, origin_world, moving_parts, static_parts"
            )
            continue

        search_params = _search_params_for_joint(search_joint)
        axis_results = search_axis_actions(
            lambda action: _evaluator_for_axis_action(
                search_joint,
                action,
                backend=backend,
                raw_vertices_by_part=raw_vertices_by_part,
                labels=_joint_labels(ctx, estimate.joint_name),
            ),
            initial_step=search_params["initial_step"],
            max_limit=search_params["max_limit"],
            min_step=search_params["min_step"],
        )
        labels = _joint_labels(ctx, estimate.joint_name)
        axis_results = _postprocess_axis_results_for_joint_motion(
            search_joint,
            axis_results,
            backend=backend,
            labels=labels,
        )
        selected = _select_sdk_axis_result(
            search_joint,
            axis_results,
        )
        trace = {
            "joint_name": estimate.joint_name,
            "estimate_axis_label": estimate.axis_label
            or _axis_label(estimate.axis_world or candidate_joint.get("axis_world")),
            "estimate_axis_world": (
                [float(value) for value in estimate.axis_world]
                if estimate.axis_world is not None
                else [float(value) for value in candidate_joint.get("axis_world", [])]
            ),
            "selected_axis_label": selected.axis_label,
            "selected_axis_world": list(selected.axis_world),
            "selected_limit": float(selected.limit),
            "selection_reason": _selection_reason(search_joint, selected, axis_results),
            "axis_trials": [
                {
                    **asdict(result),
                    "axis_world": list(result.axis_world),
                }
                for result in axis_results
            ],
        }
        traces.append(trace)

        estimate_axis = _unit_axis(estimate.axis_world or candidate_joint.get("axis_world"))
        selected_axis = _unit_axis(selected.axis_world)
        if estimate_axis is None or selected_axis is None:
            errors.append(f"{estimate.joint_name}: axis_world must be a non-zero 3-vector")
            continue
        if not _same_signed_axis(estimate_axis, selected_axis):
            got = estimate.axis_label or _axis_label(estimate_axis)
            errors.append(
                f"{estimate.joint_name}: estimate axis {got} does not match SDK selected "
                f"action {selected.axis_label} from signed-axis motion search"
            )
            continue

        range_error = _validate_estimate_range_against_search(
            estimate,
            search_joint,
            selected_limit=float(selected.limit),
            search_max_limit=float(search_params["max_limit"]),
            labels=labels,
        )
        if range_error:
            errors.append(range_error)

    errors.extend(
        validate_motion_samples(
            ctx,
            estimates,
            backend=backend,
            sample_count=sample_count,
        )
    )
    return MotionSearchValidationResult(errors=errors, traces=traces)


def validate_motion_samples_from_roots(
    ctx: EstimateContext,
    estimates: Iterable[LimitEstimate],
    *,
    converter_output_root: Path,
    sample_count: int = 17,
) -> list[str]:
    """Load VHACD/FCL geometry and validate sampled candidate motion."""
    backend = make_backend()
    part_to_obj = _part_to_obj_paths(converter_output_root, ctx.object_id)
    coord_transform = (
        SOURCE_TO_CANONICAL_TRANSFORM
        if context_uses_canonical_frame(ctx)
        else None
    )
    backend.load_model(
        object_id=ctx.object_id,
        part_to_obj_path=part_to_obj,
        vhacd_cache_root=converter_output_root / f"raw/vhacd/{ctx.object_id}",
        coacd_run_params=dict(V1_COACD_RUN_PARAMS),
        vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
        coordinate_transform=coord_transform,
    )
    backend.load_exact_meshes(
        part_to_obj_path=part_to_obj,
        coordinate_transform=coord_transform,
    )
    try:
        return validate_motion_samples(
            ctx,
            estimates,
            backend=backend,
            sample_count=sample_count,
        )
    finally:
        backend.clear()


def validate_motion_search_from_roots(
    ctx: EstimateContext,
    estimates: Iterable[LimitEstimate],
    *,
    converter_output_root: Path,
    sample_count: int = 17,
) -> MotionSearchValidationResult:
    """Load geometry and run SDK-owned signed-axis search validation."""
    backend = make_backend()
    part_to_obj = _part_to_obj_paths(converter_output_root, ctx.object_id)
    coord_transform = (
        SOURCE_TO_CANONICAL_TRANSFORM
        if context_uses_canonical_frame(ctx)
        else None
    )
    backend.load_model(
        object_id=ctx.object_id,
        part_to_obj_path=part_to_obj,
        vhacd_cache_root=converter_output_root / f"raw/vhacd/{ctx.object_id}",
        coacd_run_params=dict(V1_COACD_RUN_PARAMS),
        vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
        coordinate_transform=coord_transform,
    )
    backend.load_exact_meshes(
        part_to_obj_path=part_to_obj,
        coordinate_transform=coord_transform,
    )
    try:
        return validate_motion_search(
            ctx,
            list(estimates),
            backend=backend,
            raw_vertices_by_part=_raw_vertices_by_part(
                converter_output_root,
                ctx.object_id,
                transform_source_frame=context_uses_canonical_frame(ctx),
            ),
            sample_count=sample_count,
        )
    finally:
        backend.clear()


def _joint_with_estimate_axis(ctx: EstimateContext, estimate: LimitEstimate) -> dict:
    joint = _joint_from_context(ctx, estimate.joint_name)
    if estimate.axis_world is not None:
        joint["axis_world"] = [float(value) for value in estimate.axis_world]
    return joint


def _joint_from_context(ctx: EstimateContext, joint_name: str) -> dict:
    joint = dict(ctx.joints[joint_name])
    joint.setdefault("object_id", ctx.object_id)
    joint.setdefault("joint_name", joint_name)
    return joint


def _evaluator_for_axis_action(
    joint: dict,
    action: AxisAction,
    *,
    backend: CollisionBackend,
    raw_vertices_by_part: dict[str, np.ndarray] | None,
    labels: list[str],
) -> JointEvaluator:
    trial_joint = dict(joint)
    trial_joint["axis_world"] = list(action.axis_world)
    constraints = _constraints_for_joint_motion(
        trial_joint,
        backend=backend,
        raw_vertices_by_part=raw_vertices_by_part,
        labels=labels,
    )
    evaluator = JointEvaluator(
        joint=trial_joint,
        constraints=constraints,
        backend=backend,
    )
    evaluator.calibrate_at_zero()
    return evaluator


def _postprocess_axis_results_for_joint_motion(
    joint: dict,
    axis_results: list,
    *,
    backend: CollisionBackend,
    labels: list[str],
) -> list:
    if joint.get("type") != "prismatic" or not _labels_describe_pull_out_drawer(labels):
        return axis_results

    adjusted = []
    for result in axis_results:
        limit = float(result.limit)
        trial_joint = dict(joint)
        trial_joint["axis_world"] = list(result.axis_world)
        if limit <= 1e-9 or _endpoint_clears_parent_static_geometry(
            trial_joint,
            backend=backend,
            q_signed=limit,
        ):
            adjusted.append(result)
            continue
        samples = [
            dict(sample)
            for sample in result.samples
        ]
        samples.append({
            "q": limit,
            "valid": False,
            "reason": "endpoint_overlap",
        })
        adjusted.append(replace(
            result,
            status="endpoint_overlap",
            limit=0.0,
            samples=samples,
        ))
    return sorted(adjusted, key=lambda item: item.limit, reverse=True)


def _endpoint_clears_parent_static_geometry(
    joint: dict,
    *,
    backend: CollisionBackend,
    q_signed: float,
) -> bool:
    evaluator = JointEvaluator(joint=joint, constraints=[], backend=backend)
    evaluator(float(q_signed))
    pairs = backend.overlapping_pairs(
        list(joint.get("moving_parts", [])),
        list(joint.get("static_parts", [])),
    )
    return not pairs


def _constraints_for_joint_motion(
    trial_joint: dict,
    *,
    backend: CollisionBackend,
    raw_vertices_by_part: dict[str, np.ndarray] | None,
    labels: list[str],
    strict_articraft: bool = True,
) -> list:
    constraints = [
        CollisionConstraint(
            list(trial_joint["moving_parts"]),
            list(trial_joint["static_parts"]),
            backend=backend,
            config=CollisionConstraintConfig(allow_initial_penetration=True),
            use_exact_mesh=strict_articraft,
        )
    ]
    if (
        trial_joint.get("type") == "prismatic"
        and raw_vertices_by_part is not None
        and all(part in raw_vertices_by_part for part in trial_joint.get("moving_parts", []))
    ):
        constraints.append(
            RetainedOverlapConstraint(
                joint=trial_joint,
                raw_vertices_by_part=raw_vertices_by_part,
                min_retained_ratio=0.05,
            )
        )
    if strict_articraft:
        return constraints
    if (
        trial_joint.get("type") == "prismatic"
        and raw_vertices_by_part is not None
        and all(part in raw_vertices_by_part for part in trial_joint.get("moving_parts", []))
    ):
        clearance = MotionClearanceConstraint(
            joint=trial_joint,
            raw_vertices_by_part=raw_vertices_by_part,
            probe_distance=0.10,
            max_overlap_ratio_at_probe=0.65,
        )
        if clearance.is_active:
            constraints.append(clearance)
    if (
        trial_joint.get("type") == "revolute"
        and raw_vertices_by_part is not None
        and _labels_describe_rotary_control(labels)
        and all(part in raw_vertices_by_part for part in trial_joint.get("moving_parts", []))
    ):
        spin_constraint = RotarySpinInPlaceConstraint(
            joint=trial_joint,
            raw_vertices_by_part=raw_vertices_by_part,
            max_center_sweep_ratio=0.25,
            max_center_sweep_m=0.006,
        )
        if spin_constraint.is_active:
            constraints.append(spin_constraint)
    return constraints


def _validate_continuous_candidate_motion(
    ctx: EstimateContext,
    estimates: list[LimitEstimate],
    *,
    backend: CollisionBackend,
    raw_vertices_by_part: dict[str, np.ndarray] | None,
    sample_count: int,
) -> MotionSearchValidationResult:
    errors: list[str] = []
    traces: list[dict] = []
    for estimate in estimates:
        if estimate.joint_name not in ctx.joints:
            continue
        joint = _joint_with_estimate_axis(ctx, estimate)
        missing = [
            key
            for key in ("axis_world", "origin_world", "moving_parts", "static_parts")
            if not joint.get(key)
        ]
        if missing:
            errors.append(
                f"{estimate.joint_name}: cannot run motion validation without "
                "axis_world, origin_world, moving_parts, static_parts"
            )
            continue
        labels = _joint_labels(ctx, estimate.joint_name)
        axis = _unit_axis(estimate.axis_world or joint.get("axis_world"))
        axis_error, axis_trace = _continuous_axis_validation(
            ctx,
            estimate.joint_name,
            joint,
            axis=axis,
            raw_vertices_by_part=raw_vertices_by_part,
            labels=labels,
        )
        if axis_error:
            errors.append(axis_error)
        constraints = _constraints_for_joint_motion(
            joint,
            backend=backend,
            raw_vertices_by_part=raw_vertices_by_part,
            labels=labels,
        )
        evaluator = JointEvaluator(joint=joint, constraints=constraints, backend=backend)
        evaluator.calibrate_at_zero()
        samples = []
        for q in _range_samples(float(estimate.lower), float(estimate.upper), sample_count):
            valid = bool(evaluator(q))
            samples.append({"q": q, "valid": valid})
            if not valid and not any(error.startswith(f"{estimate.joint_name}:") for error in errors):
                errors.append(
                    f"{estimate.joint_name}: candidate motion invalid at q={q:.6g} "
                    "under continuous-axis action validation"
                )
        endpoint_error = _validate_prismatic_parent_clearance_at_upper_endpoint(
            estimate,
            joint,
            backend=backend,
            labels=labels,
        )
        if endpoint_error:
            errors.append(endpoint_error)
        overlap_error, overlap_trace = _validate_prismatic_overlap_progress_at_endpoint(
            estimate,
            joint,
            raw_vertices_by_part=raw_vertices_by_part,
        )
        if overlap_error:
            errors.append(overlap_error)
        trace = {
            "joint_name": estimate.joint_name,
            "estimate_axis_label": estimate.axis_label or _axis_label(axis),
            "estimate_axis_world": list(axis) if axis is not None else None,
            "selected_axis_label": "candidate",
            "selected_axis_world": list(axis) if axis is not None else None,
            "selected_limit": max(abs(float(estimate.lower)), abs(float(estimate.upper))),
            "selection_reason": "continuous candidate axis supplied by bounded agent action space",
            "candidate_samples": samples,
            "axis_trials": [],
        }
        if axis_trace:
            trace["axis_validation"] = axis_trace
        if overlap_trace:
            trace["prismatic_overlap_progress"] = overlap_trace
        traces.append(trace)
    return MotionSearchValidationResult(errors=errors, traces=traces)


def _continuous_axis_validation(
    ctx: EstimateContext,
    joint_name: str,
    joint: dict,
    *,
    axis: tuple[float, float, float] | None,
    raw_vertices_by_part: dict[str, np.ndarray] | None,
    labels: list[str],
) -> tuple[str | None, dict | None]:
    if axis is None:
        return f"{joint_name}: axis_world must be a non-zero 3-vector", None

    target: tuple[float, float, float] | None = None
    target_kind: str | None = None
    signed = True
    if joint.get("type") == "revolute":
        recommended_axis = _recommended_axis(ctx, joint_name)
        if recommended_axis is not None:
            target = recommended_axis
            target_kind = "recommended axis"
        else:
            mount_axis = _rotary_mount_axis(
                joint,
                raw_vertices_by_part,
            )
            if mount_axis is not None:
                target = mount_axis
                target_kind = "rotary control mount axis"
        signed = False
    elif joint.get("type") == "prismatic":
        rest_face_axis = _prismatic_rest_face_exit_axis(
            joint,
            raw_vertices_by_part,
        )
        if rest_face_axis is not None:
            target = rest_face_axis
            target_kind = "prismatic rest-face exit axis"
            signed = True

    if target is None or target_kind is None:
        return None, None

    dot_signed = sum(a * b for a, b in zip(axis, target, strict=True))
    score = dot_signed if signed else abs(dot_signed)
    max_angle_degrees = 5.0 if target_kind in {
        "rotary control geometry axis",
        "rotary compact geometry axis",
        "rotary control mount axis",
        "prismatic rest-face exit axis",
    } else 15.0
    min_score = math.cos(math.radians(max_angle_degrees))
    angle_degrees = math.degrees(math.acos(max(-1.0, min(1.0, score))))
    trace = {
        "target_kind": target_kind,
        "target_axis_world": list(target),
        "target_axis_label": _axis_label(target),
        "signed": signed,
        "alignment": float(score),
        "angle_degrees": float(angle_degrees),
        "max_angle_degrees": max_angle_degrees,
    }
    if score >= min_score:
        return None, trace

    return (
        f"{joint_name}: candidate axis {_axis_label(axis)} deviates from {target_kind} "
        f"{_axis_label(target)} by {angle_degrees:.3g} degrees; "
        f"target_axis_world={[round(float(value), 8) for value in target]}; "
        f"candidate_axis_world={[round(float(value), 8) for value in axis]}; "
        f"max_angle_degrees={max_angle_degrees:.3g}; switch axis or rotate "
        "toward the SDK motion target before marking this joint correct",
        trace,
    )


def _validate_prismatic_parent_clearance_at_upper_endpoint(
    estimate: LimitEstimate,
    joint: dict,
    *,
    backend: CollisionBackend,
    labels: list[str],
) -> str | None:
    if joint.get("type") != "prismatic" or not _labels_describe_pull_out_drawer(labels):
        return None
    upper = float(estimate.upper)
    lower = float(estimate.lower)
    endpoint = upper if abs(upper) >= abs(lower) else lower
    if abs(endpoint) <= 1e-9:
        return (
            f"{estimate.joint_name}: drawer/pull-out candidate has zero endpoint motion; "
            "increase a finite limit before marking this joint correct"
        )
    evaluator = JointEvaluator(
        joint=joint,
        constraints=[],
        backend=backend,
    )
    evaluator(float(endpoint))
    pairs = list(backend.overlapping_pairs(
        list(joint.get("moving_parts", [])),
        list(joint.get("static_parts", [])),
    ))
    if not pairs:
        return None
    return (
        f"{estimate.joint_name}: candidate motion still intersects parent/static geometry "
        f"at upper endpoint q={endpoint:.6g}; pairs={pairs[:6]}. "
        "This is a sampled-motion failure, so fix axis or extend the limit before "
        "marking this joint correct"
    )


def _validate_prismatic_overlap_progress_at_endpoint(
    estimate: LimitEstimate,
    joint: dict,
    *,
    raw_vertices_by_part: dict[str, np.ndarray] | None,
) -> tuple[str | None, dict | None]:
    if joint.get("type") != "prismatic" or raw_vertices_by_part is None:
        return None, None
    moving = _vertices_for_parts(joint.get("moving_parts", []), raw_vertices_by_part)
    static = _vertices_for_static_body(joint, raw_vertices_by_part)
    axis = _unit_axis(estimate.axis_world or joint.get("axis_world"))
    if moving is None or static is None or axis is None:
        return None, None
    base_overlap = _aabb_intersection_volume(moving, static)
    if base_overlap <= 1e-12:
        return None, None
    lower = float(estimate.lower)
    upper = float(estimate.upper)
    endpoint = upper if abs(upper) >= abs(lower) else lower
    if abs(endpoint) <= 1e-9:
        return None, None
    candidate_overlap = _aabb_intersection_volume(
        moving + np.asarray(axis, dtype=np.float64) * endpoint,
        static,
    )
    candidate_ratio = candidate_overlap / base_overlap
    best = _best_prismatic_overlap_axis(
        moving,
        static,
        endpoint=endpoint,
        base_overlap=base_overlap,
    )
    best_axis = _unit_axis(best["axis_world"])
    angle_to_best = (
        0.0
        if best_axis is None
        else math.degrees(math.acos(max(-1.0, min(1.0, sum(
            a * b for a, b in zip(axis, best_axis, strict=True)
        )))))
    )
    max_allowed_overlap_ratio = 0.50
    max_axis_angle_degrees = 5.0
    trace = {
        "base_overlap_volume": float(base_overlap),
        "endpoint": float(endpoint),
        "candidate_axis_label": _axis_label(axis),
        "candidate_axis_world": list(axis),
        "candidate_overlap_volume": float(candidate_overlap),
        "candidate_overlap_ratio": float(candidate_ratio),
        "best_axis_label": best["axis_label"],
        "best_axis_world": list(best["axis_world"]),
        "best_overlap_ratio": float(best["overlap_ratio"]),
        "angle_to_best_axis_degrees": float(angle_to_best),
        "max_allowed_overlap_ratio": max_allowed_overlap_ratio,
        "max_axis_angle_degrees": max_axis_angle_degrees,
    }
    require_axis_alignment = candidate_ratio > max_allowed_overlap_ratio
    if candidate_ratio <= max_allowed_overlap_ratio:
        return None, trace
    reasons = []
    if candidate_ratio > max_allowed_overlap_ratio:
        reasons.append(
            f"range is too short: endpoint overlap ratio={candidate_ratio:.3g}, "
            f"required<={max_allowed_overlap_ratio:.3g}"
        )
    if require_axis_alignment and angle_to_best > max_axis_angle_degrees:
        reasons.append(
            f"axis is not aligned with the best exit direction: "
            f"angle={angle_to_best:.3g} degrees, required<={max_axis_angle_degrees:.3g}"
        )
    return (
        f"{estimate.joint_name}: Articraft-style prismatic pose QC failure at "
        f"q={endpoint:.6g}; {'; '.join(reasons)}. "
        f"best signed-axis probe is {best['axis_label']} with ratio={best['overlap_ratio']:.3g}; "
        f"target_axis_world={[round(float(value), 8) for value in best['axis_world']]}; "
        f"candidate_axis_world={[round(float(value), 8) for value in axis]}; "
        "this mirrors Articraft expect_gap/expect_overlap sampled-pose checks: "
        "the slider must visibly clear/retain the parent geometry before it can be marked correct",
        trace,
    )


def _best_prismatic_overlap_axis(
    moving: np.ndarray,
    static: np.ndarray,
    *,
    endpoint: float,
    base_overlap: float,
) -> dict:
    q = abs(float(endpoint))
    best: dict | None = None
    for action in AXIS_ACTIONS:
        moved = moving + np.asarray(action.axis_world, dtype=np.float64) * q
        overlap = _aabb_intersection_volume(moved, static)
        ratio = overlap / base_overlap
        candidate = {
            "axis_label": action.label,
            "axis_world": action.axis_world,
            "overlap_ratio": float(ratio),
        }
        if best is None or ratio < float(best["overlap_ratio"]):
            best = candidate
    assert best is not None
    return best


class MotionClearanceConstraint:
    """Reject prismatic actions that do not clear the static geometry when driven."""

    def __init__(
        self,
        *,
        joint: dict,
        raw_vertices_by_part: dict[str, np.ndarray],
        probe_distance: float,
        max_overlap_ratio_at_probe: float,
    ) -> None:
        self.joint = dict(joint)
        moving_parts = list(joint.get("moving_parts", []))
        static_parts = [
            part
            for part in joint.get("static_parts", [])
            if part in raw_vertices_by_part
        ]
        if not static_parts and "body" in raw_vertices_by_part:
            static_parts = ["body"]
        self.is_active = bool(moving_parts and static_parts)
        if not self.is_active:
            self._moving_vertices = np.zeros((0, 3), dtype=np.float64)
            self._static_vertices = np.zeros((0, 3), dtype=np.float64)
            self._base_overlap_volume = 0.0
        else:
            self._moving_vertices = np.concatenate(
                [np.asarray(raw_vertices_by_part[part], dtype=np.float64) for part in moving_parts],
                axis=0,
            )
            self._static_vertices = np.concatenate(
                [np.asarray(raw_vertices_by_part[part], dtype=np.float64) for part in static_parts],
                axis=0,
            )
            self._base_overlap_volume = _aabb_intersection_volume(
                self._moving_vertices,
                self._static_vertices,
            )
        self._probe_distance = float(probe_distance)
        self._max_overlap_ratio_at_probe = float(max_overlap_ratio_at_probe)
        self._current_q_signed = 0.0

    def calibrate_at_zero(self) -> bool:
        self._current_q_signed = 0.0
        return True

    def set_current_q(self, q_signed: float) -> None:
        self._current_q_signed = float(q_signed)

    def check(self) -> bool:
        if not self.is_active or self._current_q_signed <= 1e-9:
            return True
        current_overlap = _aabb_intersection_volume(
            self._transformed_moving_vertices(abs(self._current_q_signed)),
            self._static_vertices,
        )
        if self._base_overlap_volume <= 1e-12:
            return current_overlap <= 1e-9
        ratio = current_overlap / self._base_overlap_volume
        if self._current_q_signed + 1e-12 < self._probe_distance:
            return ratio <= 1.02
        return ratio <= self._max_overlap_ratio_at_probe

    def __call__(self) -> bool:
        return self.check()

    def _transformed_moving_vertices(self, q_abs: float) -> np.ndarray:
        rotation, translation = apply_joint_transform_world_baked(
            joint_type=self.joint["type"],
            direction=1,
            q_abs=float(q_abs),
            axis_world=np.asarray(self.joint["axis_world"], dtype=np.float64),
            origin_world=np.asarray(self.joint["origin_world"], dtype=np.float64),
        )
        return (rotation @ self._moving_vertices.T).T + translation


class RotarySpinInPlaceConstraint:
    """Reject rotary-control axes that sweep the knob instead of spinning it in place."""

    def __init__(
        self,
        *,
        joint: dict,
        raw_vertices_by_part: dict[str, np.ndarray],
        max_center_sweep_ratio: float,
        max_center_sweep_m: float,
    ) -> None:
        self.joint = dict(joint)
        moving_parts = list(joint.get("moving_parts", []))
        self.is_active = bool(moving_parts)
        if not self.is_active:
            self._moving_vertices = np.zeros((0, 3), dtype=np.float64)
            self._rest_center = np.zeros(3, dtype=np.float64)
            self._max_center_sweep = 0.0
        else:
            self._moving_vertices = np.concatenate(
                [np.asarray(raw_vertices_by_part[part], dtype=np.float64) for part in moving_parts],
                axis=0,
            )
            self._rest_center = self._moving_vertices.mean(axis=0)
            self._max_center_sweep = max(
                float(max_center_sweep_m),
                _perpendicular_radius(self._moving_vertices, self.joint["axis_world"])
                * float(max_center_sweep_ratio),
            )
        self._current_q_signed = 0.0

    def calibrate_at_zero(self) -> bool:
        self._current_q_signed = 0.0
        return True

    def set_current_q(self, q_signed: float) -> None:
        self._current_q_signed = float(q_signed)

    def check(self) -> bool:
        if not self.is_active or abs(self._current_q_signed) <= 1e-9:
            return True
        rotation, translation = apply_joint_transform_world_baked(
            joint_type=self.joint["type"],
            direction=1 if self._current_q_signed >= 0.0 else -1,
            q_abs=abs(float(self._current_q_signed)),
            axis_world=np.asarray(self.joint["axis_world"], dtype=np.float64),
            origin_world=np.asarray(self.joint["origin_world"], dtype=np.float64),
        )
        moved_center = rotation @ self._rest_center + translation
        return float(np.linalg.norm(moved_center - self._rest_center)) <= self._max_center_sweep

    def __call__(self) -> bool:
        return self.check()


def _search_params_for_joint(joint: dict) -> dict[str, float]:
    if joint.get("type") == "revolute":
        return {
            "initial_step": math.pi / 4.0,
            "max_limit": math.tau,
            "min_step": math.radians(2.0),
        }
    return {
        "initial_step": 0.10,
        "max_limit": 0.50,
        "min_step": 0.005,
    }


def _select_sdk_axis_result(joint: dict, axis_results):
    authored_axis = _unit_axis(joint.get("axis_world"))
    if authored_axis is not None:
        global_best = axis_results[0]
        line_results = [
            result
            for result in axis_results
            if _same_unsigned_axis(_unit_axis(result.axis_world), authored_axis)
        ]
        if line_results:
            best_line = max(line_results, key=lambda result: float(result.limit))
            if float(best_line.limit) > 1e-9:
                if float(global_best.limit) > float(best_line.limit) + 1e-9:
                    return global_best
                authored_label = _axis_label(authored_axis)
                for result in line_results:
                    if (
                        result.axis_label == authored_label
                        and abs(float(result.limit) - float(best_line.limit)) <= 1e-9
                    ):
                        return result
                return best_line
    return axis_results[0]


def _selection_reason(joint: dict, selected, axis_results) -> str:
    authored_axis = _unit_axis(joint.get("axis_world"))
    selected_axis = _unit_axis(selected.axis_world)
    if authored_axis is not None and selected_axis is not None:
        if _same_signed_axis(authored_axis, selected_axis):
            return "authored axis action selected by physical motion search"
        if _same_unsigned_axis(authored_axis, selected_axis):
            return "opposite signed action on authored joint axis selected by physical motion search"
    return "longest valid signed-axis motion"


def _validate_estimate_range_against_search(
    estimate: LimitEstimate,
    joint: dict,
    *,
    selected_limit: float,
    search_max_limit: float,
    labels: list[str],
) -> str | None:
    if joint.get("type") != "prismatic":
        return None
    lower = float(estimate.lower)
    upper = float(estimate.upper)
    if abs(lower) > 1e-6:
        return f"{estimate.joint_name}: drawer/pull-out SDK search expects lower=0 for the selected signed action"
    if selected_limit >= search_max_limit - 1e-9:
        return None
    tolerance = max(0.02, selected_limit * 0.15)
    if upper > selected_limit + tolerance:
        return (
            f"{estimate.joint_name}: upper={upper:.6g} exceeds SDK searched limit "
            f"{selected_limit:.6g} for selected action"
        )
    if upper < max(0.0, selected_limit - tolerance):
        return (
            f"{estimate.joint_name}: upper={upper:.6g} is below SDK searched limit "
            f"{selected_limit:.6g} for selected action"
        )
    return None


def _range_samples(lower: float, upper: float, sample_count: int) -> list[float]:
    count = max(int(sample_count), 2)
    if lower == upper:
        return [lower]
    return [
        lower + (upper - lower) * idx / (count - 1)
        for idx in range(count)
    ]


def _joint_labels(ctx: EstimateContext, joint_name: str) -> list[str]:
    evidence = ctx.evidence.get(joint_name, {}) or {}
    labels = evidence.get("labels") or []
    if isinstance(labels, (list, tuple)):
        return [str(label) for label in labels]
    return [str(labels)]


def _uses_continuous_axis_action_space(ctx: EstimateContext) -> bool:
    config = ctx.evidence.get("__action_space__", {}) or {}
    if not isinstance(config, dict):
        return False
    return config.get("axis_mode") == "continuous"


def _recommended_axis(ctx: EstimateContext, joint_name: str) -> tuple[float, float, float] | None:
    evidence = ctx.evidence.get(joint_name, {}) or {}
    raw = evidence.get("recommended_axis_world")
    if raw is None:
        raw_candidates = evidence.get("axis_candidates")
        if isinstance(raw_candidates, list) and raw_candidates:
            first = raw_candidates[0]
            if isinstance(first, dict):
                raw = first.get("axis_world")
    return _unit_axis(raw)


def _labels_describe_rotary_control(labels: list[str]) -> bool:
    text = " ".join(label.lower() for label in labels)
    return any(
        cue in text
        for cue in (
            "knob",
            "dial",
            "temperature",
            "timer",
            "control",
            "rotary",
            "旋钮",
            "表盘",
        )
    )


def _labels_describe_pull_out_drawer(labels: list[str]) -> bool:
    text = " ".join(label.lower() for label in labels)
    return any(
        cue in text
        for cue in (
            "pull-out",
            "pull out",
            "drawer",
            "pan",
            "tray",
            "basket",
            "bin",
            "fryer basket",
            "air fryer",
            "抽屉",
            "炸篮",
            "炸桶",
        )
    )


def _rotary_mount_axis(
    joint: dict,
    raw_vertices_by_part: dict[str, np.ndarray] | None,
) -> tuple[float, float, float] | None:
    if raw_vertices_by_part is None:
        return None
    moving = _vertices_for_parts(joint.get("moving_parts", []), raw_vertices_by_part)
    static = _vertices_for_static_body(joint, raw_vertices_by_part)
    if moving is None or static is None:
        return None
    moving_center = _aabb_center(moving)
    static_center = _aabb_center(static)
    half_extents = np.maximum((_aabb_max(static) - _aabb_min(static)) * 0.5, 1e-9)
    normalized_offset = (moving_center - static_center) / half_extents
    index = int(np.argmax(np.abs(normalized_offset)))
    if abs(float(normalized_offset[index])) <= 1e-6:
        return None
    axis = np.zeros(3, dtype=np.float64)
    axis[index] = 1.0 if normalized_offset[index] >= 0.0 else -1.0
    return tuple(float(value) for value in axis)


def _rotary_geometry_axis(
    joint: dict,
    raw_vertices_by_part: dict[str, np.ndarray] | None,
) -> tuple[float, float, float] | None:
    return _rotary_mount_axis(joint, raw_vertices_by_part)


def _prismatic_rest_face_exit_axis(
    joint: dict,
    raw_vertices_by_part: dict[str, np.ndarray] | None,
) -> tuple[float, float, float] | None:
    if raw_vertices_by_part is None:
        return None
    moving = _vertices_for_parts(joint.get("moving_parts", []), raw_vertices_by_part)
    static = _vertices_for_static_body(joint, raw_vertices_by_part)
    if moving is None or static is None:
        return None
    moving_lo = moving.min(axis=0)
    moving_hi = moving.max(axis=0)
    static_lo = static.min(axis=0)
    static_hi = static.max(axis=0)
    moving_extent = np.maximum(moving_hi - moving_lo, 1e-9)
    static_extent = np.maximum(static_hi - static_lo, 1e-9)
    best_score = 0.0
    best_axis: np.ndarray | None = None
    for axis_index in range(3):
        for sign in (1.0, -1.0):
            exposure = (
                moving_hi[axis_index] - static_hi[axis_index]
                if sign > 0.0
                else static_lo[axis_index] - moving_lo[axis_index]
            )
            if exposure <= 1e-5:
                continue
            overlap_ratios = []
            for other_index in range(3):
                if other_index == axis_index:
                    continue
                overlap = min(moving_hi[other_index], static_hi[other_index]) - max(
                    moving_lo[other_index],
                    static_lo[other_index],
                )
                overlap_ratios.append(
                    max(0.0, float(overlap)) / min(
                        moving_extent[other_index],
                        static_extent[other_index],
                    )
                )
            min_overlap_ratio = min(overlap_ratios) if overlap_ratios else 0.0
            if min_overlap_ratio < 0.45:
                continue
            score = float(exposure) * min_overlap_ratio
            if score <= best_score:
                continue
            axis = np.zeros(3, dtype=np.float64)
            axis[axis_index] = sign
            best_axis = axis
            best_score = score
    if best_axis is None:
        return None
    return tuple(float(value) for value in best_axis)


def _compact_revolute_geometry_axis(
    joint: dict,
    raw_vertices_by_part: dict[str, np.ndarray] | None,
) -> tuple[float, float, float] | None:
    if raw_vertices_by_part is None:
        return None
    moving = _vertices_for_parts(joint.get("moving_parts", []), raw_vertices_by_part)
    if moving is None or moving.shape[0] < 6:
        return None
    centered = moving - moving.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)
    thin_value = float(values[order[0]])
    next_value = float(values[order[1]])
    wide_value = float(values[order[2]])
    if not np.isfinite(thin_value) or not np.isfinite(next_value) or not np.isfinite(wide_value):
        return None
    if next_value <= 1e-12 or thin_value / next_value > 0.35:
        return None
    if wide_value <= 1e-12 or next_value / wide_value < 0.50:
        return None
    axis = vectors[:, int(order[0])]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    return tuple(float(value) for value in axis)


def _vertices_for_static_body(
    joint: dict,
    raw_vertices_by_part: dict[str, np.ndarray],
) -> np.ndarray | None:
    static_parts = list(joint.get("static_parts", []))
    if "body" in raw_vertices_by_part:
        return np.asarray(raw_vertices_by_part["body"], dtype=np.float64)
    return _vertices_for_parts(static_parts, raw_vertices_by_part)


def _vertices_for_parts(
    parts: Iterable,
    raw_vertices_by_part: dict[str, np.ndarray],
) -> np.ndarray | None:
    arrays = [
        np.asarray(raw_vertices_by_part[part], dtype=np.float64)
        for part in parts
        if part in raw_vertices_by_part
    ]
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)


def _aabb_min(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).min(axis=0)


def _aabb_max(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).max(axis=0)


def _aabb_center(points: np.ndarray) -> np.ndarray:
    return (_aabb_min(points) + _aabb_max(points)) * 0.5


def _unit_axis(raw) -> tuple[float, float, float] | None:
    if raw is None or len(raw) != 3:
        return None
    axis = tuple(float(value) for value in raw)
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in axis)


def _same_signed_axis(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> bool:
    return sum(a * b for a, b in zip(left, right, strict=True)) >= 0.999


def _same_unsigned_axis(
    left: tuple[float, float, float] | None,
    right: tuple[float, float, float] | None,
) -> bool:
    if left is None or right is None:
        return False
    return abs(sum(a * b for a, b in zip(left, right, strict=True))) >= 0.999


def _axis_label(axis) -> str | None:
    if axis is None or len(axis) != 3:
        return None
    values = [float(value) for value in axis]
    idx = max(range(3), key=lambda item: abs(values[item]))
    names = ("X", "Y", "Z")
    return ("+" if values[idx] >= 0.0 else "-") + names[idx]


def _aabb_intersection_volume(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    left_lo = left.min(axis=0)
    left_hi = left.max(axis=0)
    right_lo = right.min(axis=0)
    right_hi = right.max(axis=0)
    extents = np.maximum(0.0, np.minimum(left_hi, right_hi) - np.maximum(left_lo, right_lo))
    return float(np.prod(extents))


def _perpendicular_radius(points: np.ndarray, axis_world) -> float:
    axis = np.asarray(axis_world, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12 or points.size == 0:
        return 0.0
    axis = axis / norm
    extents = points.max(axis=0) - points.min(axis=0)
    projected_extent = abs(float(extents @ axis))
    perpendicular_extent = max(0.0, float(np.linalg.norm(extents) ** 2 - projected_extent ** 2)) ** 0.5
    return 0.5 * perpendicular_extent


def _part_to_obj_paths(converter_output_root: Path, object_id: str) -> dict[str, Path]:
    obj_dir = converter_output_root / f"raw/partseg/{object_id}/objs"
    return {
        path.stem: path
        for path in sorted(obj_dir.glob("*.obj"))
        if path.stem == "body" or path.stem.startswith("part_")
    }


def _raw_vertices_by_part(
    converter_output_root: Path,
    object_id: str,
    *,
    transform_source_frame: bool = False,
) -> dict[str, np.ndarray]:
    obj_dir = converter_output_root / f"raw/partseg/{object_id}/objs"
    return {
        path.stem: _load_obj_vertices(path, transform_source_frame=transform_source_frame)
        for path in sorted(obj_dir.glob("*.obj"))
        if path.stem == "body" or path.stem.startswith("part_")
    }


def _load_obj_vertices(obj_path: Path, *, transform_source_frame: bool = False) -> np.ndarray:
    vertices = []
    for line in obj_path.read_text(errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    points = np.asarray(vertices, dtype=np.float64)
    return source_to_canonical_points(points) if transform_source_frame else points
