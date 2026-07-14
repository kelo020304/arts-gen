"""Train-only static visual-relation axis-family proposal critic."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .axis_family_model import axis_family_numeric_features


DEFAULT_STATIC_AXIS_FAMILY_MODEL = (
    Path(__file__).resolve().parents[1] / "priors" / "kin_static_axis_family_lr_v1.joblib"
)


def predict_static_axis_family(
    label: str,
    category: str,
    numeric_features,
    *,
    model_path: Path | None = None,
) -> tuple[int, float, list[float], float] | None:
    path = Path(model_path or DEFAULT_STATIC_AXIS_FAMILY_MODEL)
    if not path.is_file():
        return None
    import joblib
    from scipy.sparse import csr_matrix, hstack

    artifact = joblib.load(path)
    if artifact.get("format") != "arts_gen_kin_static_axis_family_lr_v1":
        return None
    text = artifact["vectorizer"].transform([f"{category} {label}"])
    numeric = artifact["scaler"].transform(
        np.asarray([list(numeric_features)], dtype=np.float64)
    )
    features = hstack((text, csr_matrix(numeric)))
    probabilities = artifact["classifier"].predict_proba(features)[0]
    index = int(np.argmax(probabilities))
    family = int(artifact["classifier"].classes_[index])
    threshold = float((artifact.get("category_thresholds") or {}).get(category, 1.01))
    return family, float(probabilities[index]), [float(value) for value in probabilities], threshold


def apply_static_axis_family_reranker(
    result,
    *,
    label: str,
    category: str,
    body_points: np.ndarray,
    moving_points: np.ndarray,
    static_observation,
    max_iterations: int,
    model_path: Path | None = None,
):
    """Select an existing proposal when a static visual relation is confident."""
    if static_observation is None:
        return result, None
    features = axis_family_numeric_features(
        body_points, moving_points, result.candidate, result.trace, static_observation,
    )
    prediction = predict_static_axis_family(
        label, category, features, model_path=model_path,
    )
    if prediction is None:
        return result, None
    family, confidence, probabilities, threshold = prediction
    evidence = {
        "family": family,
        "family_name": "XYZ"[family],
        "confidence": confidence,
        "probabilities": probabilities,
        "threshold": threshold,
        "used": False,
        "model": "kin_static_axis_family_lr_v1",
        "view_count": int(getattr(static_observation, "view_count", 0)),
        "support": float(getattr(static_observation, "support", 0.0)),
    }
    if confidence < threshold:
        evidence["review_reason"] = "static visual axis proposal below category threshold"
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
        evidence["review_reason"] = "no validated proposal matches static visual axis family"
        return result, evidence
    selected = max(compatible, key=lambda raw: float(raw.get("score", -1.0)))
    selected_axis = np.asarray(selected["axis_world"], dtype=np.float64)
    snapped_axis = np.zeros(3, dtype=np.float64)
    snapped_axis[family] = 1.0 if selected_axis[family] >= 0.0 else -1.0
    refined = replace(
        result.candidate,
        axis_world=tuple(float(value) for value in snapped_axis),
        origin_world=tuple(float(value) for value in selected["origin_world"]),
        signals={
            **result.candidate.signals,
            "static_visual_axis_model_used": 1.0,
            "static_visual_axis_family": float(family),
            "static_visual_axis_confidence": confidence,
            "static_visual_axis_support": evidence["support"],
            "axis_confidence": max(
                float(result.candidate.signals.get("axis_confidence", 0.0)), confidence,
            ),
        },
        reason=result.candidate.reason + "; reranked by frozen static visual-relation model",
    )
    evidence["used"] = True
    trace = list(result.trace)
    row = {
        "iteration": min(max_iterations, len(trace) + 1),
        "stage": "static_visual_axis_proposal_critic",
        "selected": asdict(refined),
        "static_axis_family_model": evidence,
        "alternatives": compatible[:6],
    }
    if len(trace) < max_iterations:
        trace.append(row)
    elif trace:
        trace[-1] = row
    return replace(result, candidate=refined, iterations=len(trace), trace=trace), evidence
