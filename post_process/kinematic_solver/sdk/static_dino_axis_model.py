"""DINO-based static door hinge-family proposal critic."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


DEFAULT_STATIC_DINO_AXIS_MODEL = (
    Path(__file__).resolve().parents[1] / "priors" / "kin_static_dino_door_axis_v1.joblib"
)


def apply_static_dino_door_axis_reranker(
    result,
    *,
    dino_feature,
    max_iterations: int,
    model_path: Path | None = None,
):
    if dino_feature is None:
        return result, None
    path = Path(model_path or DEFAULT_STATIC_DINO_AXIS_MODEL)
    if not path.is_file():
        return result, None
    import joblib

    artifact = joblib.load(path)
    if artifact.get("format") != "arts_gen_kin_static_dino_door_axis_v1":
        return result, None
    values = np.asarray([list(dino_feature.feature)], dtype=np.float64)
    probabilities = artifact["model"].predict_proba(values)[0]
    index = int(np.argmax(probabilities))
    family = int(artifact["model"].classes_[index])
    confidence = float(probabilities[index])
    threshold = float(artifact.get("confidence_threshold", 0.55))
    evidence = {
        "family": family, "family_name": "XYZ"[family],
        "confidence": confidence, "threshold": threshold,
        "probabilities": [float(value) for value in probabilities],
        "view_indices": list(dino_feature.view_indices),
        "used": False, "model": "kin_static_dino_door_axis_v1",
    }
    if confidence < threshold:
        evidence["review_reason"] = "static DINO door proposal below confidence threshold"
        return result, evidence
    candidates = []
    for row in result.trace:
        candidates.extend(row.get("alternatives") or [])
    candidates.append(asdict(result.candidate))
    compatible = [
        raw for raw in candidates
        if raw.get("joint_type") == result.candidate.joint_type
        and int(np.argmax(np.abs(np.asarray(raw.get("axis_world"), dtype=np.float64)))) == family
    ]
    if not compatible:
        evidence["review_reason"] = "no validated proposal matches static DINO door family"
        return result, evidence
    selected = max(compatible, key=lambda raw: float(raw.get("score", -1.0)))
    selected_axis = np.asarray(selected["axis_world"], dtype=np.float64)
    snapped = np.zeros(3, dtype=np.float64)
    snapped[family] = 1.0 if selected_axis[family] >= 0.0 else -1.0
    refined = replace(
        result.candidate,
        axis_world=tuple(float(value) for value in snapped),
        origin_world=tuple(float(value) for value in selected["origin_world"]),
        signals={
            **result.candidate.signals,
            "static_dino_door_axis_used": 1.0,
            "static_dino_door_axis_family": float(family),
            "static_dino_door_axis_confidence": confidence,
            "axis_confidence": max(float(result.candidate.signals.get("axis_confidence", 0.0)), confidence),
        },
        reason=result.candidate.reason + "; reranked by frozen static DINO door proposal",
    )
    evidence["used"] = True
    trace = list(result.trace)
    row = {
        "iteration": min(max_iterations, len(trace) + 1),
        "stage": "static_dino_door_axis_proposal_critic",
        "selected": asdict(refined), "static_dino_door_axis_model": evidence,
        "alternatives": compatible[:6],
    }
    if len(trace) < max_iterations:
        trace.append(row)
    elif trace:
        trace[-1] = row
    return replace(result, candidate=refined, iterations=len(trace), trace=trace), evidence
