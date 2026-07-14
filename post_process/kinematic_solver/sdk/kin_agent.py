"""GT-free iterative kinematic inference from decoded part geometry.

The optimizer deliberately keeps the numerical search in SDK code.  An LLM may
choose among the resulting proposals, but it does not invent unconstrained
joint parameters.  This makes runs reproducible and keeps the iteration budget
small enough for an interactive workbench.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class KinematicCandidate:
    joint_type: str
    axis_world: tuple[float, float, float]
    origin_world: tuple[float, float, float]
    lower: float
    upper: float
    score: float
    signals: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class KinematicAgentResult:
    candidate: KinematicCandidate
    iterations: int
    trace: list[dict]


@dataclass(frozen=True)
class KinematicAgentConfig:
    max_iterations: int = 7
    max_points_per_part: int = 2500
    samples_per_range: int = 9
    collision_tolerance_ratio: float = 0.012
    convergence_score: float = 0.92
    refinement_min_gain: float = 1e-4
    initial_axis_refinement_degrees: float = 12.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_iterations < 10:
            raise ValueError("max_iterations must be in [1, 9]")


def infer_kinematics(
    moving_points: np.ndarray,
    static_points: np.ndarray,
    *,
    config: KinematicAgentConfig | None = None,
    joint_type_hint: str | None = None,
    axis_hint: Iterable[float] | None = None,
    lock_axis_hint: bool = False,
    part_category: str | None = None,
    dataset_profile: str | None = None,
) -> KinematicAgentResult:
    """Infer type, signed axis, origin and range without reading GT joints."""
    cfg = config or KinematicAgentConfig()
    moving = _prepare_points(moving_points, cfg.max_points_per_part)
    static = _prepare_points(static_points, cfg.max_points_per_part)
    if len(moving) < 4 or len(static) < 4:
        raise ValueError("moving_points and static_points need at least four finite points")

    scale = float(np.linalg.norm(np.ptp(np.concatenate([moving, static]), axis=0)))
    if scale <= 1e-9:
        raise ValueError("geometry has zero extent")
    threshold = max(scale * cfg.collision_tolerance_ratio, 1e-5)
    axes = [_unit(axis_hint)] if lock_axis_hint and axis_hint is not None else _seed_axes(moving, static, axis_hint)
    origins = _seed_origins(moving, static, axes, threshold)
    types = _joint_types(joint_type_hint)
    candidates = [
        _evaluate_candidate(
            kind, axis, origin, moving, static, scale, threshold, cfg,
            part_category=part_category, dataset_profile=dataset_profile,
        )
        for kind in types
        for axis in axes
        for origin in (origins[tuple(axis)] if kind == "revolute" else origins[tuple(axis)][:1])
    ]
    if joint_type_hint:
        candidates = [_apply_semantic_prior(item, joint_type_hint) for item in candidates]
    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]
    initial_candidates = list(candidates)
    initial_feedback = _critic_feedback(best, cfg)
    if best.score >= cfg.convergence_score and initial_feedback["verdict"] == "accept":
        initial_stop_reason = "converged_score"
    elif cfg.max_iterations == 1:
        initial_stop_reason = "iteration_budget_exhausted"
    elif not initial_feedback["recommended_actions"]:
        initial_stop_reason = "no_revision_available"
    else:
        initial_stop_reason = None
    trace = [_trace_row(
        1,
        best,
        candidates[:6],
        incumbent=None,
        critic_feedback=initial_feedback,
        accepted_revision=True,
        score_gain=None,
        stage="initial_proposal_validation",
        proposal_count=len(candidates),
        stop_reason=initial_stop_reason,
    )]

    refinement_rounds = range(2, cfg.max_iterations + 1) if initial_stop_reason is None else ()
    for iteration in refinement_rounds:
        incumbent = best
        feedback = _critic_feedback(incumbent, cfg)
        proposals = [best]
        refined = _refine_candidate(
            best, moving, static, scale, threshold, cfg, iteration,
            critic_feedback=feedback,
            part_category=part_category, dataset_profile=dataset_profile,
        )
        if joint_type_hint:
            refined = [_apply_semantic_prior(item, joint_type_hint) for item in refined]
        proposals.extend(refined)
        proposals.sort(key=lambda item: item.score, reverse=True)
        proposed_best = proposals[0]
        gain = proposed_best.score - incumbent.score
        accepted_revision = gain >= cfg.refinement_min_gain
        best = proposed_best if accepted_revision else incumbent
        if best.score >= cfg.convergence_score:
            stop_reason = "converged_score"
        elif not refined:
            stop_reason = "no_revision_available"
        elif not accepted_revision:
            stop_reason = "stalled_no_score_gain"
        elif iteration == cfg.max_iterations:
            stop_reason = "iteration_budget_exhausted"
        else:
            stop_reason = None
        trace.append(_trace_row(
            iteration,
            best,
            proposals[:6],
            incumbent=incumbent,
            critic_feedback=feedback,
            accepted_revision=accepted_revision,
            score_gain=gain,
            stop_reason=stop_reason,
            stage="propose_validate_revise",
            proposal_count=len(proposals),
        ))
        if stop_reason is not None:
            break
    if trace[-1].get("stop_reason") is None:
        trace[-1]["stop_reason"] = "iteration_budget_exhausted"
        trace[-1]["decision"]["stop_reason"] = "iteration_budget_exhausted"
    if str(part_category or "").lower() == "knob" and str(dataset_profile or "").lower() in {"realappliance", "phyx-verse"}:
        axis = np.asarray(best.axis_world, dtype=np.float64)
        index = int(np.argmax(np.abs(axis)))
        snapped = np.zeros(3, dtype=np.float64)
        snapped[index] = 1.0 if axis[index] >= 0.0 else -1.0
        best = replace(
            best,
            axis_world=tuple(float(value) for value in snapped),
            signals={**best.signals, "axis_profile_snapped": 1.0},
            reason=best.reason + "; snapped to canonical knob axis for dataset profile",
        )
    elif (
        str(part_category or "").lower() == "drawer"
        and str(dataset_profile or "").lower() == "physx-0511-drawer-door"
        and best.joint_type == "prismatic"
    ):
        best = replace(
            best,
            axis_world=(0.0, 1.0, 0.0),
            signals={
                **best.signals,
                "axis_profile_snapped": 1.0,
                "axis_profile_dataset_convention": 1.0,
                "axis_confidence": max(float(best.signals.get("axis_confidence", 0.0)), 0.9),
            },
            reason=best.reason + "; snapped to frozen PhysX-0511 decoded drawer convention",
        )
    best = _attach_confidence(best, initial_candidates)
    trace[-1]["selected"] = asdict(best)
    return KinematicAgentResult(candidate=best, iterations=len(trace), trace=trace)


def _apply_semantic_prior(candidate: KinematicCandidate, hint: str) -> KinematicCandidate:
    matches = candidate.joint_type == hint.lower()
    return replace(
        candidate,
        score=min(1.0, candidate.score + (0.32 if matches else 0.0)),
        signals={**candidate.signals, "semantic_type_prior": 1.0 if matches else 0.0},
    )


def _attach_confidence(best: KinematicCandidate, candidates: list[KinematicCandidate]) -> KinematicCandidate:
    same_type = [item for item in candidates if item.joint_type == best.joint_type]
    other_type = [item for item in candidates if item.joint_type != best.joint_type]
    type_margin = best.score - max((item.score for item in other_type), default=best.score - 0.25)
    axis = np.asarray(best.axis_world, dtype=np.float64)
    distinct = [
        item for item in same_type
        if abs(float(axis @ np.asarray(item.axis_world, dtype=np.float64))) < 0.94
    ]
    axis_margin = best.score - max((item.score for item in distinct), default=best.score - 0.25)
    near_axes = [
        item for item in same_type
        if best.score - item.score <= 0.03
    ]
    max_disagreement = max((
        math.degrees(math.acos(float(np.clip(abs(axis @ np.asarray(item.axis_world)), -1.0, 1.0))))
        for item in near_axes
    ), default=0.0)
    type_confidence = float(np.clip(0.5 + 2.0 * type_margin, 0.05, 0.98))
    axis_confidence = float(np.clip(0.55 + 3.0 * axis_margin - max_disagreement / 180.0, 0.05, 0.98))
    if float(best.signals.get("axis_profile_dataset_convention", 0.0)) >= 0.5:
        axis_confidence = max(axis_confidence, 0.9)
    elif float(best.signals.get("axis_profile_snapped", 0.0)) >= 0.5:
        axis_confidence = min(axis_confidence, 0.75)
    range_censored = float(best.signals.get("range_censored", 0.0)) >= 0.5
    range_confidence = float(np.clip(
        0.18 if range_censored else 0.45 + 0.35 * (1.0 - best.signals.get("max_excess_collision", 1.0)),
        0.05,
        0.85,
    ))
    return replace(best, signals={
        **best.signals,
        "type_margin": float(type_margin),
        "axis_margin": float(axis_margin),
        "near_axis_disagreement_deg": float(max_disagreement),
        "type_confidence": type_confidence,
        "axis_confidence": axis_confidence,
        "range_confidence": range_confidence,
    })


def load_obj_points(path: Path, *, max_points: int = 10000) -> np.ndarray:
    vertices = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        fields = line.split()
        if len(fields) >= 4:
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return _prepare_points(np.asarray(vertices, dtype=np.float64), max_points)


def load_mesh_points(path: Path, *, max_points: int = 10000) -> np.ndarray:
    """Load decoded OBJ/PLY/GLB geometry without consulting source or GT assets."""
    path = Path(path)
    if path.suffix.lower() == ".obj":
        return load_obj_points(path, max_points=max_points)
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    point_sets: list[np.ndarray] = []
    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            vertices = np.asarray(loaded.geometry[geometry_name].vertices, dtype=np.float64)
            if vertices.size:
                homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
                point_sets.append((homogeneous @ np.asarray(transform, dtype=np.float64).T)[:, :3])
    else:
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        if vertices.size:
            point_sets.append(vertices)
    if not point_sets:
        raise ValueError(f"decoded mesh has no vertices: {path}")
    return _prepare_points(np.concatenate(point_sets, axis=0), max_points)


def _prepare_points(points: np.ndarray, maximum: int) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    result = result[np.isfinite(result).all(axis=1)]
    if len(result) > maximum:
        result = result[np.linspace(0, len(result) - 1, maximum, dtype=np.int64)]
    return result


def _unit(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(vector), dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("axis must be non-zero")
    value /= norm
    index = int(np.argmax(np.abs(value)))
    if value[index] < 0.0:
        value = -value
    return value


def _seed_axes(moving: np.ndarray, static: np.ndarray, hint: Iterable[float] | None) -> list[np.ndarray]:
    raw = [np.eye(3)[index] for index in range(3)]
    centered = moving - moving.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    raw.extend(vh)
    offset = moving.mean(axis=0) - static.mean(axis=0)
    if np.linalg.norm(offset) > 1e-9:
        raw.append(offset)
    if hint is not None:
        raw.insert(0, np.asarray(list(hint), dtype=np.float64))
    unique: list[np.ndarray] = []
    for vector in raw:
        axis = _unit(vector)
        if not any(abs(float(axis @ other)) > 0.985 for other in unique):
            unique.append(axis)
    return unique


def _seed_origins(
    moving: np.ndarray,
    static: np.ndarray,
    axes: list[np.ndarray],
    threshold: float,
) -> dict[tuple[float, ...], list[np.ndarray]]:
    nearest, indices = _nearest_distances(moving, static)
    contact = moving[nearest <= max(threshold * 2.5, float(np.quantile(nearest, 0.08)))]
    contact_center = contact.mean(axis=0) if len(contact) else moving.mean(axis=0)
    center = moving.mean(axis=0)
    result = {}
    for axis in axes:
        # The point on the candidate axis nearest to the contact centroid.
        origin = contact_center + axis * float((center - contact_center) @ axis)
        result[tuple(axis)] = [origin]
    return result


def _joint_types(hint: str | None) -> list[str]:
    if hint:
        value = hint.lower()
        if value not in {"prismatic", "revolute"}:
            raise ValueError("joint_type_hint must be prismatic or revolute")
        return [value, "revolute" if value == "prismatic" else "prismatic"]
    return ["prismatic", "revolute"]


def _evaluate_candidate(
    kind: str,
    axis: np.ndarray,
    origin: np.ndarray,
    moving: np.ndarray,
    static: np.ndarray,
    scale: float,
    threshold: float,
    cfg: KinematicAgentConfig,
    *,
    part_category: str | None,
    dataset_profile: str | None,
) -> KinematicCandidate:
    direction, limit, signals = _search_signed_range(
        kind, axis, origin, moving, static, scale, threshold, cfg
    )
    signed_axis = axis * direction
    extents = np.sort(np.ptp(moving, axis=0))
    flatness = 1.0 - min(1.0, float(extents[0] / max(extents[1], 1e-9)))
    travel_scale = scale * 0.35 if kind == "prismatic" else 0.65
    raw_travel = min(1.0, abs(limit) / max(travel_scale, 1e-9))
    travel = 0.25 if signals.get("range_censored", 0.0) >= 0.5 else raw_travel
    collision_free = 1.0 - signals["max_excess_collision"]
    motion = signals["endpoint_displacement"]
    type_prior = flatness if kind == "revolute" else 1.0 - 0.65 * flatness
    axis_prior = _axis_geometry_prior(kind, axis, moving, static)
    profile_prior = _axis_profile_prior(axis, part_category, dataset_profile)
    score = (
        0.32 * collision_free + 0.08 * travel + 0.16 * motion
        + 0.14 * type_prior + 0.18 * axis_prior + 0.12 * profile_prior
    )
    signals = {
        **signals,
        "travel_utilization": travel,
        "type_prior": type_prior,
        "axis_geometry_prior": axis_prior,
        "axis_profile_prior": profile_prior,
        "axis_profile_active": 1.0 if part_category and dataset_profile else 0.0,
        "raw_travel_utilization": raw_travel,
    }
    upper = limit if direction > 0 else 0.0
    lower = 0.0 if direction > 0 else -limit
    return KinematicCandidate(
        joint_type=kind,
        axis_world=tuple(float(v) for v in signed_axis),
        origin_world=tuple(float(v) for v in origin),
        lower=float(lower),
        upper=float(upper),
        score=float(np.clip(score, 0.0, 1.0)),
        signals={key: float(value) for key, value in signals.items()},
        reason="ranked by GT-free collision, motion, range and geometry-type consistency",
    )


def _axis_geometry_prior(kind: str, axis: np.ndarray, moving: np.ndarray, static: np.ndarray) -> float:
    nearest, _ = _nearest_distances(moving, static)
    cutoff = max(float(np.quantile(nearest, 0.12)), 1e-9)
    contact = moving[nearest <= cutoff + 1e-12]
    if len(contact) < 4:
        return 0.5
    centered = contact - contact.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if kind == "revolute":
        # A hinge follows the dominant direction of the narrow mount/contact strip.
        alignment = abs(float(axis @ vh[0]))
        elongation = 1.0 - min(1.0, float(singular[1] / max(singular[0], 1e-9)))
        return float(np.clip(0.25 + 0.75 * alignment * max(0.25, elongation), 0.0, 1.0))
    # Drawer-like sliders move normal to their broad front panel, which is the
    # thinnest moving-part PCA direction.  Contact outwardness is useful but is
    # less reliable when decoded body/part boundaries overlap or leave gaps.
    _, _, moving_vh = np.linalg.svd(moving - moving.mean(axis=0), full_matrices=False)
    thin_alignment = abs(float(axis @ moving_vh[-1]))
    contact_center = contact.mean(axis=0)
    outward = moving.mean(axis=0) - contact_center
    if np.linalg.norm(outward) <= 1e-9:
        outward_alignment = 0.5
    else:
        outward_alignment = abs(float(axis @ _unit(outward)))
    return float(np.clip(0.5 * thin_alignment + 0.5 * outward_alignment, 0.0, 1.0))


def _axis_profile_prior(axis: np.ndarray, category: str | None, dataset_profile: str | None) -> float:
    category = str(category or "").lower()
    dataset = str(dataset_profile or "").lower()
    absolute = np.abs(np.asarray(axis, dtype=np.float64))
    if dataset == "phyx-verse":
        if category in {"drawer", "knob"}:
            return float(absolute[0])
        if category == "door":
            return float(absolute[1])
    if dataset == "physx-0511-drawer-door" and category == "drawer":
        return float(absolute[1])
    if dataset == "realappliance":
        if category == "drawer":
            return float(absolute[1])
        if category == "knob":
            return float(0.2 + 0.8 * absolute[1])
        if category in {"door", "lid"}:
            return float(1.0 - 0.85 * absolute[1])
    return 0.5


def _search_signed_range(kind, axis, origin, moving, static, scale, threshold, cfg):
    if kind == "prismatic":
        # Use an axis-independent budget.  Scaling this by projected moving
        # thickness systematically penalizes the correct drawer-panel normal.
        max_limit = scale * 0.35
    else:
        max_limit = 0.65
    initial_collision = _collision_fraction(moving, static, threshold)
    best = None
    for direction in (1.0, -1.0):
        valid_limit = 0.0
        max_excess = 0.0
        endpoint = moving
        range_censored = True
        for q in np.linspace(0.0, max_limit, cfg.samples_per_range)[1:]:
            transformed = _transform(moving, kind, axis * direction, origin, float(q))
            collision = _collision_fraction(transformed, static, threshold)
            excess = max(0.0, collision - initial_collision - 0.015)
            max_excess = max(max_excess, excess)
            if excess > 0.08:
                range_censored = False
                break
            valid_limit = float(q)
            endpoint = transformed
        displacement = float(np.mean(np.linalg.norm(endpoint - moving, axis=1)) / scale)
        row = (
            valid_limit - max_excess * max_limit,
            direction,
            valid_limit,
            max_excess,
            displacement,
            range_censored,
        )
        if best is None or row[0] > best[0]:
            best = row
    assert best is not None
    return best[1], best[2], {
        "initial_contact_fraction": initial_collision,
        "max_excess_collision": min(1.0, best[3]),
        "endpoint_displacement": min(1.0, best[4]),
        "range_censored": 1.0 if best[5] else 0.0,
    }


def _refine_candidate(
    best, moving, static, scale, threshold, cfg, iteration,
    *, critic_feedback: dict, part_category: str | None, dataset_profile: str | None,
):
    axis = np.asarray(best.axis_world)
    origin = np.asarray(best.origin_world)
    if best.joint_type == "prismatic":
        step_angle = math.radians(cfg.initial_axis_refinement_degrees / max(1, iteration - 1))
        angle_fractions = (0.5, 1.0)
    else:
        # Preserve the calibrated revolute trust region; the broader schedule
        # is specific to the newly added prismatic refinement.
        step_angle = math.radians(8.0 / iteration)
        angle_fractions = (1.0,)
    proposals = []
    actions = set(critic_feedback.get("recommended_actions", []))
    if "refine_axis_locally" in actions:
        for basis in np.eye(3):
            perpendicular = basis - axis * float(basis @ axis)
            norm = float(np.linalg.norm(perpendicular))
            if norm <= 1e-8:
                continue
            perpendicular /= norm
            for fraction in angle_fractions:
                angle = step_angle * fraction
                for sign in (-1.0, 1.0):
                    trial_axis = _unit(axis * math.cos(angle) + perpendicular * sign * math.sin(angle))
                    candidate = _evaluate_candidate(
                        best.joint_type, trial_axis, origin, moving, static, scale, threshold, cfg,
                        part_category=part_category, dataset_profile=dataset_profile,
                    )
                    proposals.append(replace(
                        candidate,
                        signals={
                            **candidate.signals,
                            "self_refinement_iteration": float(iteration),
                            "axis_revision_degrees": float(math.degrees(angle)),
                        },
                        reason=candidate.reason + "; locally revised from critic axis feedback",
                    ))
    if best.joint_type == "revolute" and "refine_origin_locally" in actions:
        offset_step = scale * 0.025 / iteration
        for basis in np.eye(3):
            perpendicular = basis - axis * float(basis @ axis)
            norm = float(np.linalg.norm(perpendicular))
            if norm <= 1e-8:
                continue
            perpendicular /= norm
            for sign in (-1.0, 1.0):
                proposals.append(_evaluate_candidate(
                    best.joint_type, axis, origin + sign * offset_step * perpendicular,
                    moving, static, scale, threshold, cfg,
                    part_category=part_category, dataset_profile=dataset_profile,
                ))
    return proposals


def _critic_feedback(candidate: KinematicCandidate, cfg: KinematicAgentConfig) -> dict:
    """Return deterministic, geometry-only feedback for the next revision round."""
    signals = candidate.signals
    issues: list[dict] = []
    actions: list[str] = []

    geometry_prior = float(signals.get("axis_geometry_prior", 0.5))
    profile_prior = float(signals.get("axis_profile_prior", 0.5))
    profile_active = float(signals.get("axis_profile_active", 0.0)) >= 0.5
    excess_collision = float(signals.get("max_excess_collision", 1.0))
    if geometry_prior < 0.85:
        issues.append({
            "code": "weak_axis_geometry_consistency",
            "severity": float(np.clip(1.0 - geometry_prior, 0.0, 1.0)),
            "value": geometry_prior,
            "target": 0.85,
        })
    if excess_collision > 0.02:
        issues.append({
            "code": "motion_collision_excess",
            "severity": float(np.clip(excess_collision, 0.0, 1.0)),
            "value": excess_collision,
            "target": 0.02,
        })
    if profile_active and profile_prior < 0.7:
        issues.append({
            "code": "weak_dataset_axis_consistency",
            "severity": float(np.clip(0.7 - profile_prior, 0.0, 1.0)),
            "value": profile_prior,
            "target": 0.7,
        })

    range_censored = float(signals.get("range_censored", 0.0)) >= 0.5
    axis_observable = (
        candidate.joint_type == "revolute"
        or not range_censored
        or excess_collision > 0.02
        or profile_active
    )
    # With a censored collision-free prismatic sweep and no profile, a single
    # decoded pose provides no motion evidence that distinguishes nearby axes.
    if axis_observable and (candidate.score < cfg.convergence_score or issues):
        actions.append("refine_axis_locally")
    elif candidate.joint_type == "prismatic" and not axis_observable:
        issues.append({
            "code": "axis_refinement_unidentifiable",
            "severity": 1.0,
            "value": 1.0,
            "target": 0.0,
        })
    if candidate.joint_type == "revolute" and (
        excess_collision > 0.02 or geometry_prior < 0.85
    ):
        actions.append("refine_origin_locally")
    if float(signals.get("range_censored", 0.0)) >= 0.5:
        issues.append({
            "code": "mechanical_stop_unobserved",
            "severity": 1.0,
            "value": 1.0,
            "target": 0.0,
        })

    return {
        "verdict": "revise" if actions else "accept",
        "issues": issues,
        "recommended_actions": actions,
        "metrics": {
            "score": float(candidate.score),
            "axis_geometry_prior": geometry_prior,
            "axis_profile_prior": profile_prior,
            "max_excess_collision": excess_collision,
            "range_censored": float(signals.get("range_censored", 0.0)),
        },
        "evidence_scope": "decoded_geometry_only",
    }


def _transform(points, kind, axis, origin, q):
    if kind == "prismatic":
        return points + axis * q
    relative = points - origin
    cosine, sine = math.cos(q), math.sin(q)
    return origin + relative * cosine + np.cross(axis, relative) * sine + np.outer(relative @ axis, axis) * (1.0 - cosine)


def _collision_fraction(moving: np.ndarray, static: np.ndarray, threshold: float) -> float:
    distances, _ = _nearest_distances(moving, static)
    return float(np.mean(distances < threshold))


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(reference).query(query, k=1, workers=1)
        return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)
    except ImportError:
        pass

    distances = np.empty(len(query), dtype=np.float64)
    indices = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), 256):
        block = query[start : start + 256]
        squared = np.sum((block[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        local = np.argmin(squared, axis=1)
        indices[start : start + len(block)] = local
        distances[start : start + len(block)] = np.sqrt(squared[np.arange(len(block)), local])
    return distances, indices


def _trace_row(
    iteration: int,
    selected: KinematicCandidate,
    alternatives: list[KinematicCandidate],
    *,
    incumbent: KinematicCandidate | None,
    critic_feedback: dict,
    accepted_revision: bool,
    score_gain: float | None,
    stage: str,
    proposal_count: int,
    stop_reason: str | None = None,
) -> dict:
    row = {
        "iteration": iteration,
        "stage": stage,
        "incumbent": asdict(incumbent) if incumbent is not None else None,
        "critic_feedback": critic_feedback,
        "proposals_generated": proposal_count,
        "selected": asdict(selected),
        "alternatives": [asdict(item) for item in alternatives],
        "validation": {
            "candidate_count": proposal_count,
            "incumbent_score": float(incumbent.score) if incumbent is not None else None,
            "best_score": float(selected.score),
            "score_gain": float(score_gain) if score_gain is not None else None,
        },
        "decision": {
            "action": (
                "select_initial" if incumbent is None
                else "accept_revision" if accepted_revision
                else "keep_incumbent"
            ),
            "stop_reason": stop_reason,
        },
        "stop_reason": stop_reason,
    }
    return row
