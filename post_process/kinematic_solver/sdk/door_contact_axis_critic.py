"""Conservative static hinge-axis critic for decoded PhyX door geometry."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np


PHYX_DOOR_CONTACT_MIN_CONFIDENCE = 0.65
PHYX_DOOR_CONTACT_MAX_SCORE_DROP = 0.15
PHYX_DOOR_CONTACT_QUANTILE = 0.25


def decoded_door_contact_axis_evidence(
    body_points: np.ndarray,
    moving_points: np.ndarray,
    *,
    contact_quantile: float = PHYX_DOOR_CONTACT_QUANTILE,
    maximum_points: int = 2500,
) -> dict | None:
    """Estimate a hinge family only when part extent and body contact agree.

    The moving-part dominant direction alone is not sufficient: broad doors
    often have a longest edge unrelated to the hinge.  Requiring the nearest
    body-contact strip to independently select the same cardinal family makes
    the critic conservative on decoded boundaries with missing or merged
    geometry.
    """
    body = _prepare_points(body_points, maximum_points)
    moving = _prepare_points(moving_points, maximum_points)
    if len(body) < 4 or len(moving) < 4:
        return None

    moving_axis, moving_confidence = _dominant_axis(moving)
    nearest = _nearest_distances(moving, body)
    cutoff = float(np.quantile(nearest, float(contact_quantile)))
    contact = moving[nearest <= cutoff + 1e-12]
    if len(contact) < 8:
        return None
    contact_axis, contact_confidence = _dominant_axis(contact)
    moving_family = int(np.argmax(np.abs(moving_axis)))
    contact_family = int(np.argmax(np.abs(contact_axis)))
    family_agreement = moving_family == contact_family
    confidence = min(moving_confidence, contact_confidence) if family_agreement else 0.0
    return {
        "family": moving_family if family_agreement else None,
        "family_name": "XYZ"[moving_family] if family_agreement else None,
        "moving_family": moving_family,
        "moving_family_name": "XYZ"[moving_family],
        "contact_family": contact_family,
        "contact_family_name": "XYZ"[contact_family],
        "moving_axis": [float(value) for value in moving_axis],
        "contact_axis": [float(value) for value in contact_axis],
        "moving_confidence": moving_confidence,
        "contact_confidence": contact_confidence,
        "confidence": float(confidence),
        "family_agreement": family_agreement,
        "contact_quantile": float(contact_quantile),
        "contact_points": int(len(contact)),
    }


def apply_phyx_door_contact_axis_critic(
    result,
    *,
    dataset_id: str | None,
    part_category: str | None,
    body_points: np.ndarray,
    moving_points: np.ndarray,
    max_iterations: int,
    min_confidence: float = PHYX_DOOR_CONTACT_MIN_CONFIDENCE,
    max_score_drop: float = PHYX_DOOR_CONTACT_MAX_SCORE_DROP,
):
    """Rerank a static PhyX door to an existing geometry-validated axis."""
    if str(dataset_id or "").lower() != "phyx-verse" or str(part_category or "").lower() != "door":
        return result, None

    raw = decoded_door_contact_axis_evidence(body_points, moving_points)
    if raw is None:
        evidence = _evidence_base(min_confidence, max_score_drop)
        evidence.update({
            "review_required": True,
            "review_reason": "decoded door contact-axis evidence unavailable",
        })
        return _attach_evidence_signals(result, evidence, score_drop=None), evidence
    evidence = {
        **raw,
        **_evidence_base(min_confidence, max_score_drop),
    }
    if result.candidate.joint_type != "revolute":
        evidence.update({
            "review_required": True,
            "review_reason": "PhyX door was not inferred as revolute",
        })
        return _attach_evidence_signals(result, evidence, score_drop=None), evidence
    if not bool(evidence["family_agreement"]):
        evidence.update({
            "review_required": True,
            "review_reason": "moving extent and body contact select different door-axis families",
        })
        return _attach_evidence_signals(result, evidence, score_drop=None), evidence

    family = int(evidence["family"])
    incumbent_family = int(np.argmax(np.abs(np.asarray(result.candidate.axis_world, dtype=np.float64))))
    if family == incumbent_family:
        evidence["review_reason"] = "incumbent already matches decoded door contact-axis family"
        return _attach_evidence_signals(result, evidence, score_drop=0.0), evidence
    # The frozen PhyX profile already handles the common Y-family doors.  This
    # critic is only calibrated to override it when independent decoded
    # geometry identifies a non-Y hinge family.
    if family == 1:
        evidence.update({
            "review_required": True,
            "review_reason": "door contact critic does not override toward the profile-prior family",
        })
        return _attach_evidence_signals(result, evidence, score_drop=None), evidence

    compatible = _compatible_candidates(result, family)
    if not compatible:
        evidence.update({
            "review_required": True,
            "review_reason": "no validated proposal matches decoded door contact-axis family",
        })
        return _attach_evidence_signals(result, evidence, score_drop=None), evidence
    selected = max(compatible, key=lambda raw_candidate: float(raw_candidate.get("score", -1.0)))
    selected_score = float(selected.get("score", -1.0))
    score_drop = max(0.0, float(result.candidate.score) - selected_score)
    evidence.update({"selected_proposal_score": selected_score, "score_drop": score_drop})
    if float(evidence["confidence"]) < float(min_confidence):
        evidence.update({
            "review_required": True,
            "review_reason": "decoded door contact-axis family is ambiguous",
        })
        return _attach_evidence_signals(result, evidence, score_drop=score_drop), evidence
    if score_drop > float(max_score_drop) + 1e-12:
        evidence.update({
            "review_required": True,
            "review_reason": "decoded door contact-axis proposal exceeds bounded score drop",
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
            "phyx_door_contact_axis_family": float(family),
            "phyx_door_contact_axis_confidence": float(evidence["confidence"]),
            "phyx_door_contact_axis_score_drop": score_drop,
            "phyx_door_contact_axis_used": 1.0,
            "axis_confidence": max(
                float(result.candidate.signals.get("axis_confidence", 0.0)),
                float(evidence["confidence"]),
            ),
        },
        reason=result.candidate.reason + "; reranked by bounded decoded door contact-axis critic",
    )
    evidence["used"] = True
    trace = list(result.trace)
    row = {
        "iteration": min(max_iterations, len(trace) + 1),
        "stage": "decoded_door_contact_axis_critic",
        "selected": asdict(refined),
        "door_contact_axis_critic": evidence,
        "alternatives": compatible[:6],
    }
    if len(trace) < max_iterations:
        trace.append(row)
    elif trace:
        trace[-1] = row
    return replace(result, candidate=refined, iterations=len(trace), trace=trace), evidence


def _prepare_points(points: np.ndarray, maximum: int) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    result = result[np.isfinite(result).all(axis=1)]
    if len(result) > maximum:
        result = result[np.linspace(0, len(result) - 1, maximum, dtype=np.int64)]
    return result


def _dominant_axis(points: np.ndarray) -> tuple[np.ndarray, float]:
    _, _, vh = np.linalg.svd(points - points.mean(axis=0, keepdims=True), full_matrices=False)
    axis = vh[0]
    components = np.abs(axis)
    order = np.argsort(components)[::-1]
    confidence = float(np.clip(components[order[0]] - 0.5 * components[order[1]], 0.0, 1.0))
    return axis, confidence


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        return np.asarray(cKDTree(reference).query(query, k=1, workers=1)[0], dtype=np.float64)
    except ImportError:
        pass
    distances = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 256):
        block = query[start : start + 256]
        squared = np.sum((block[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        distances[start : start + len(block)] = np.sqrt(np.min(squared, axis=1))
    return distances


def _compatible_candidates(result, family: int) -> list[dict]:
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
    return compatible


def _evidence_base(min_confidence: float, max_score_drop: float) -> dict:
    return {
        "model": "decoded_door_contact_axis_v1",
        "min_confidence": float(min_confidence),
        "max_score_drop": float(max_score_drop),
        "used": False,
        "review_required": False,
        "review_reason": None,
    }


def _attach_evidence_signals(result, evidence: dict, *, score_drop: float | None):
    family = evidence.get("family")
    candidate = replace(
        result.candidate,
        signals={
            **result.candidate.signals,
            "phyx_door_contact_axis_family": float(family if family is not None else -1),
            "phyx_door_contact_axis_confidence": float(evidence.get("confidence", 0.0)),
            "phyx_door_contact_axis_score_drop": float(score_drop if score_drop is not None else -1.0),
            "phyx_door_contact_axis_used": 0.0,
        },
    )
    return replace(result, candidate=candidate)
