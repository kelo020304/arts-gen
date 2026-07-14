"""Exact decoded-mesh feedback for shrinking a joint motion interval.

This module consumes an existing broad-phase collision audit.  It does not run
the bundle pipeline or mutate a :class:`KinematicCandidate`; instead it returns
a JSON-friendly proposal which an agent may accept after inspecting the gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .kin_agent import KinematicCandidate


FEEDBACK_VERSION = "decoded_collision_feedback_v1"


@dataclass(frozen=True)
class CollisionFeedbackConfig:
    """Safety gates for one bounded exact collision-feedback pass."""

    max_bisections_per_side: int = 6
    min_retained_fraction: float = 0.35
    observed_interval_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 <= self.max_bisections_per_side <= 6:
            raise ValueError("max_bisections_per_side must be in [0, 6]")
        if not 0.0 <= self.min_retained_fraction <= 1.0:
            raise ValueError("min_retained_fraction must be in [0, 1]")
        if self.observed_interval_tolerance < 0.0:
            raise ValueError("observed_interval_tolerance must be non-negative")


def propose_collision_clear_interval(
    body_mesh: Any,
    moving_mesh: Any,
    candidate: KinematicCandidate,
    audit: Mapping[str, Any],
    *,
    config: CollisionFeedbackConfig | None = None,
) -> dict[str, Any]:
    """Propose the zero-connected exact-clear sub-interval of ``candidate``.

    The broad audit identifies the first invalid bracket on each side of zero.
    A Manifold intersection-volume query then refines each bracket with at most
    six bisections.  The proposal is deliberately conservative: the returned
    endpoint is the last exact-clear value, not the first invalid value.
    """
    cfg = config or CollisionFeedbackConfig()
    lower, upper = float(candidate.lower), float(candidate.upper)
    original_span = upper - lower
    base = _base_result(candidate, cfg)
    if not math.isfinite(original_span) or original_span <= 1e-12:
        return _review(base, "candidate_interval_is_empty")
    if lower > 0.0 or upper < 0.0:
        return _review(base, "candidate_interval_does_not_contain_zero")

    samples = _normalized_samples(audit.get("q_samples"), lower, upper)
    if not samples or not any(abs(row["q"]) <= 1e-10 for row in samples):
        return _review(base, "audit_has_no_zero_connected_q_samples")

    audit_mesh_state = audit.get("mesh_state") or {}
    narrow = audit.get("narrow_phase") or {}
    if not (
        audit_mesh_state.get("body_is_volume")
        and audit_mesh_state.get("moving_is_volume")
        and narrow.get("exact")
    ):
        return _review(base, "exact_audit_unavailable_or_mesh_non_watertight")
    threshold = narrow.get("invalid_threshold")
    if threshold is None or not math.isfinite(float(threshold)):
        return _review(base, "exact_invalid_threshold_missing")

    try:
        body = _load_mesh(body_mesh)
        moving = _load_mesh(moving_mesh)
    except Exception as exc:
        return _review(base, f"decoded_mesh_load_failed:{type(exc).__name__}")
    if not (_is_closed_volume(body) and _is_closed_volume(moving)):
        return _review(base, "decoded_mesh_non_watertight_or_not_volume")
    if not _manifold_available():
        return _review(base, "manifold_exact_intersection_unavailable")

    denominator = abs(float(moving.volume))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return _review(base, "moving_mesh_has_zero_volume")

    evaluations: dict[float, dict[str, Any]] = {}

    def evaluate(q_value: float) -> dict[str, Any]:
        key = float(q_value)
        if key not in evaluations:
            overlap = _exact_overlap_fraction(
                body, moving, candidate, key, denominator=denominator,
            )
            evaluations[key] = {
                "q": key,
                "overlap_fraction": float(overlap),
                "invalid": bool(overlap > float(threshold)),
            }
        return evaluations[key]

    try:
        baseline = evaluate(0.0)
        if baseline["invalid"]:
            result = _review(base, "zero_state_exceeds_exact_collision_threshold")
            result["exact_evaluations"] = _sorted_evaluations(evaluations)
            return result
        negative = _refine_side(
            samples, direction=-1, evaluate=evaluate,
            max_bisections=cfg.max_bisections_per_side,
        )
        positive = _refine_side(
            samples, direction=1, evaluate=evaluate,
            max_bisections=cfg.max_bisections_per_side,
        )
    except Exception as exc:
        result = _review(base, f"manifold_exact_intersection_failed:{type(exc).__name__}")
        result["exact_evaluations"] = _sorted_evaluations(evaluations)
        return result

    proposed_lower = negative["clear_endpoint"] if negative["bracket"] else lower
    proposed_upper = positive["clear_endpoint"] if positive["bracket"] else upper
    proposed_span = max(0.0, float(proposed_upper) - float(proposed_lower))
    retained_fraction = proposed_span / original_span
    observed_interval = _observed_interval(candidate.signals)
    observed_preserved = _contains_observed_interval(
        proposed_lower, proposed_upper, observed_interval,
        tolerance=cfg.observed_interval_tolerance,
    )
    changed = proposed_lower > lower + 1e-10 or proposed_upper < upper - 1e-10
    retained_ok = retained_fraction + 1e-12 >= cfg.min_retained_fraction
    gate_reasons = []
    if not changed:
        gate_reasons.append("no_exact_invalid_bracket")
    if not retained_ok:
        gate_reasons.append("retained_fraction_below_minimum")
    if not observed_preserved:
        gate_reasons.append("proposal_conflicts_with_observed_motion")
    accepted = bool(changed and retained_ok and observed_preserved)
    requires_review = bool(changed and not accepted)

    return {
        **base,
        "status": "accepted" if accepted else "rejected" if changed else "no_change",
        "proposal": {
            "lower": float(proposed_lower),
            "upper": float(proposed_upper),
            "changed": changed,
            "negative_side": negative,
            "positive_side": positive,
        },
        "retained_fraction": float(retained_fraction),
        "accept_gate": {
            "accepted": accepted,
            "min_retained_fraction": float(cfg.min_retained_fraction),
            "retained_fraction_ok": retained_ok,
            "observed_interval": observed_interval,
            "observed_motion_preserved": observed_preserved,
            "reasons": gate_reasons,
        },
        "requires_review": requires_review,
        "review_reason": ";".join(gate_reasons) if requires_review else None,
        "exact_threshold": float(threshold),
        "exact_evaluations": _sorted_evaluations(evaluations),
    }


def _refine_side(samples, *, direction, evaluate, max_bisections):
    outward = sorted(
        (row for row in samples if direction * row["q"] > 1e-10),
        key=lambda row: direction * row["q"],
    )
    last_clear_q = 0.0
    broad_bracket = None
    invalid_q = None
    for row in outward:
        q_value = float(row["q"])
        if not row["invalid"]:
            last_clear_q = q_value
            continue
        broad_bracket = [float(last_clear_q), q_value]
        if evaluate(q_value)["invalid"]:
            invalid_q = q_value
            break
        # A broad false positive is exact-clear and can seed a later bracket.
        last_clear_q = q_value

    if invalid_q is None:
        return {
            "bracket": None,
            "broad_bracket": broad_bracket,
            "clear_endpoint": None,
            "bisections": 0,
        }

    if evaluate(last_clear_q)["invalid"]:
        last_clear_q = 0.0
    clear_q = float(last_clear_q)
    for _ in range(max_bisections):
        midpoint = 0.5 * (clear_q + invalid_q)
        if evaluate(midpoint)["invalid"]:
            invalid_q = midpoint
        else:
            clear_q = midpoint
    return {
        "bracket": [min(clear_q, invalid_q), max(clear_q, invalid_q)],
        "broad_bracket": broad_bracket,
        "clear_endpoint": float(clear_q),
        "first_invalid_endpoint": float(invalid_q),
        "bisections": int(max_bisections),
    }


def _normalized_samples(raw_samples, lower, upper):
    if not isinstance(raw_samples, (list, tuple)):
        return []
    by_q = {}
    for row in raw_samples:
        if not isinstance(row, Mapping) or "q" not in row or "invalid" not in row:
            continue
        q_value = float(row["q"])
        if math.isfinite(q_value) and lower - 1e-10 <= q_value <= upper + 1e-10:
            by_q[q_value] = {"q": q_value, "invalid": bool(row["invalid"])}
    if lower <= 0.0 <= upper and not any(abs(q) <= 1e-10 for q in by_q):
        return []
    return [by_q[q] for q in sorted(by_q)]


def _observed_interval(signals: Mapping[str, float]) -> list[float] | None:
    span = float(signals.get("motion_observed_span", 0.0))
    if not math.isfinite(span) or span <= 1e-8:
        return None
    lower = float(signals.get("motion_observed_lower", 0.0))
    upper = float(signals.get("motion_observed_upper", span))
    return [min(lower, upper), max(lower, upper)]


def _contains_observed_interval(lower, upper, observed, *, tolerance):
    if observed is None:
        return True
    return bool(observed[0] >= lower - tolerance and observed[1] <= upper + tolerance)


def _base_result(candidate, config):
    return {
        "version": FEEDBACK_VERSION,
        "candidate_interval": [float(candidate.lower), float(candidate.upper)],
        "proposal": None,
        "retained_fraction": None,
        "accept_gate": {
            "accepted": False,
            "min_retained_fraction": float(config.min_retained_fraction),
            "retained_fraction_ok": False,
            "observed_interval": _observed_interval(candidate.signals),
            "observed_motion_preserved": False,
            "reasons": [],
        },
        "requires_review": False,
        "review_reason": None,
        "exact_evaluations": [],
    }


def _review(result, reason):
    result["status"] = "review"
    result["requires_review"] = True
    result["review_reason"] = reason
    result["accept_gate"]["reasons"] = [reason]
    return result


def _load_mesh(value):
    import trimesh

    if hasattr(value, "vertices") and hasattr(value, "faces"):
        return value
    loaded = trimesh.load(Path(value), force="scene", process=False)
    return loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded


def _is_closed_volume(mesh):
    return bool(len(getattr(mesh, "faces", ())) and mesh.is_watertight and mesh.is_volume)


def _exact_overlap_fraction(body, moving, candidate, q_value, *, denominator):
    import trimesh

    transformed = trimesh.Trimesh(
        vertices=_transform_points(np.asarray(moving.vertices), candidate, q_value),
        faces=np.asarray(moving.faces),
        process=False,
    )
    intersection = trimesh.boolean.intersection([body, transformed], engine="manifold")
    overlap_volume = 0.0 if intersection is None else abs(float(intersection.volume))
    return overlap_volume / max(float(denominator), 1e-12)


def _transform_points(points, candidate, q_value):
    axis = np.asarray(candidate.axis_world, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("candidate axis has zero length")
    axis = axis / norm
    points = np.asarray(points, dtype=np.float64)
    if candidate.joint_type == "prismatic":
        return points + axis * float(q_value)
    origin = np.asarray(candidate.origin_world, dtype=np.float64)
    relative = points - origin
    cosine, sine = math.cos(q_value), math.sin(q_value)
    return (
        origin + relative * cosine + np.cross(axis, relative) * sine
        + np.outer(relative @ axis, axis) * (1.0 - cosine)
    )


def _manifold_available():
    try:
        import manifold3d  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def _sorted_evaluations(evaluations):
    return [evaluations[q] for q in sorted(evaluations)]
