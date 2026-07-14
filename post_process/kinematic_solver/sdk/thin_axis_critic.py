"""Bounded decoded-geometry axis-family critic for PhyX rotary knobs."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np


PHYX_KNOB_THIN_AXIS_MIN_CONFIDENCE = 0.80
PHYX_KNOB_THIN_AXIS_MAX_SCORE_DROP = 0.20


def decoded_thin_axis_evidence(moving_points: np.ndarray) -> dict | None:
    """Return canonical family evidence from a decoded moving-part point cloud."""
    points = np.asarray(moving_points, dtype=np.float64).reshape((-1, 3))
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        return None
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        return None
    thin = vectors[:, int(np.argmin(values))]
    components = np.abs(thin)
    order = np.argsort(components)[::-1]
    family = int(order[0])
    alignment = float(components[family])
    component_margin = float(components[order[0]] - components[order[1]])
    sorted_values = np.sort(np.maximum(values, 0.0))
    anisotropy = float(
        1.0 - sorted_values[0] / max(sorted_values[1], 1e-12)
    )
    # Alignment measures cardinality; the margin penalizes family ambiguity.
    # A nearly isotropic cloud has no identifiable thin direction even if an
    # eigensolver happens to return a cardinal basis vector.
    confidence = float(np.clip(alignment - 0.5 * components[order[1]], 0.0, 1.0))
    if anisotropy < 0.02:
        confidence *= max(0.0, anisotropy / 0.02)
    return {
        "family": family,
        "family_name": "XYZ"[family],
        "thin_axis": [float(value) for value in thin],
        "alignment": alignment,
        "component_margin": component_margin,
        "confidence": confidence,
        "anisotropy": float(np.clip(anisotropy, 0.0, 1.0)),
        "eigenvalues": [float(value) for value in sorted_values],
    }


def apply_phyx_knob_thin_axis_critic(
    result,
    *,
    dataset_id: str | None,
    part_category: str | None,
    moving_points: np.ndarray,
    max_iterations: int,
    min_confidence: float = PHYX_KNOB_THIN_AXIS_MIN_CONFIDENCE,
    max_score_drop: float = PHYX_KNOB_THIN_AXIS_MAX_SCORE_DROP,
    allowed_dataset_ids: tuple[str, ...] = ("phyx-verse",),
):
    """Rerank to an existing cardinal proposal when decoded knob geometry is decisive."""
    allowed = {str(value).lower() for value in allowed_dataset_ids}
    if str(dataset_id or "").lower() not in allowed or str(part_category or "").lower() != "knob":
        return result, None

    evidence = decoded_thin_axis_evidence(moving_points)
    if evidence is None:
        return result, {
            "model": "decoded_thin_axis_family_v1",
            "used": False,
            "review_required": True,
            "review_reason": "decoded knob thin axis unavailable",
        }
    evidence = {
        **evidence,
        "model": "decoded_thin_axis_family_v1",
        "min_confidence": float(min_confidence),
        "max_score_drop": float(max_score_drop),
        "used": False,
        "review_required": False,
        "review_reason": None,
    }
    if result.candidate.joint_type != "revolute":
        evidence.update({
            "review_required": True,
            "review_reason": "knob was not inferred as revolute",
        })
        return _attach_evidence_signals(result, evidence, score_drop=None), evidence

    family = int(evidence["family"])
    candidates = []
    for row in result.trace:
        candidates.extend(row.get("alternatives") or [])
    candidates.append(asdict(result.candidate))
    compatible = []
    for raw in candidates:
        axis = np.asarray(raw.get("axis_world") or (), dtype=np.float64)
        if axis.shape != (3,) or float(np.linalg.norm(axis)) <= 1e-12:
            continue
        if raw.get("joint_type") != result.candidate.joint_type:
            continue
        if int(np.argmax(np.abs(axis))) == family:
            compatible.append(raw)
    if not compatible:
        evidence.update({
            "review_required": True,
            "review_reason": "no validated proposal matches decoded knob thin-axis family",
        })
        return _attach_evidence_signals(result, evidence, score_drop=None), evidence

    selected = max(compatible, key=lambda raw: float(raw.get("score", -1.0)))
    selected_score = float(selected.get("score", -1.0))
    score_drop = max(0.0, float(result.candidate.score) - selected_score)
    evidence["selected_proposal_score"] = selected_score
    evidence["score_drop"] = score_drop
    if float(evidence["confidence"]) < float(min_confidence):
        evidence.update({
            "review_required": True,
            "review_reason": "decoded knob thin-axis family is ambiguous",
        })
        return _attach_evidence_signals(result, evidence, score_drop=score_drop), evidence
    if score_drop > float(max_score_drop) + 1e-12:
        evidence.update({
            "review_required": True,
            "review_reason": "decoded knob thin-axis proposal exceeds bounded score drop",
        })
        return _attach_evidence_signals(result, evidence, score_drop=score_drop), evidence

    selected_axis = np.asarray(selected["axis_world"], dtype=np.float64)
    snapped_axis = np.zeros(3, dtype=np.float64)
    snapped_axis[family] = 1.0 if selected_axis[family] >= 0.0 else -1.0
    refined = replace(
        result.candidate,
        axis_world=tuple(float(value) for value in snapped_axis),
        origin_world=tuple(float(value) for value in selected["origin_world"]),
        signals={
            **result.candidate.signals,
            "phyx_thin_axis_family": float(family),
            "phyx_thin_axis_alignment": float(evidence["alignment"]),
            "phyx_thin_axis_component_margin": float(evidence["component_margin"]),
            "phyx_thin_axis_family_confidence": float(evidence["confidence"]),
            "phyx_thin_axis_score_drop": score_drop,
            "phyx_thin_axis_used": 1.0,
            "decoded_knob_thin_axis_used": 1.0,
            "axis_confidence": max(
                float(result.candidate.signals.get("axis_confidence", 0.0)),
                float(evidence["confidence"]),
            ),
        },
        reason=result.candidate.reason + "; reranked by bounded decoded knob thin-axis critic",
    )
    evidence["used"] = True
    trace = list(result.trace)
    row = {
        "iteration": min(max_iterations, len(trace) + 1),
        "stage": "decoded_thin_axis_family_critic",
        "selected": asdict(refined),
        "thin_axis_critic": evidence,
        "alternatives": compatible[:6],
    }
    if len(trace) < max_iterations:
        trace.append(row)
    elif trace:
        trace[-1] = row
    return replace(result, candidate=refined, iterations=len(trace), trace=trace), evidence


def _attach_evidence_signals(result, evidence: dict, *, score_drop: float | None):
    candidate = replace(
        result.candidate,
        signals={
            **result.candidate.signals,
            "phyx_thin_axis_family": float(evidence.get("family", -1)),
            "phyx_thin_axis_alignment": float(evidence.get("alignment", 0.0)),
            "phyx_thin_axis_component_margin": float(evidence.get("component_margin", 0.0)),
            "phyx_thin_axis_family_confidence": float(evidence.get("confidence", 0.0)),
            "phyx_thin_axis_score_drop": float(score_drop if score_drop is not None else -1.0),
            "phyx_thin_axis_used": 0.0,
            "decoded_knob_thin_axis_used": 0.0,
        },
    )
    return replace(result, candidate=candidate)
