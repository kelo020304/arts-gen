"""Train-only axis-family reranker for otherwise unobservable knob spins."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_AXIS_FAMILY_MODEL = Path(__file__).resolve().parents[1] / "priors" / "kin_axis_family_lr_v2.joblib"


def axis_family_numeric_features(
    body_points: np.ndarray,
    moving_points: np.ndarray,
    candidate,
    trace: list[dict],
    observation,
) -> np.ndarray:
    body = np.asarray(body_points, dtype=np.float64).reshape((-1, 3))
    moving = np.asarray(moving_points, dtype=np.float64).reshape((-1, 3))
    body_low, body_high = body.min(axis=0), body.max(axis=0)
    body_extent = np.maximum(body_high - body_low, 1e-6)
    moving_extent = moving.max(axis=0) - moving.min(axis=0)
    if observation is not None and observation.trajectory_points:
        center = np.mean(np.asarray(observation.trajectory_points, dtype=np.float64), axis=0)
    else:
        center = moving.mean(axis=0)
    position = (center - body_low) / body_extent
    face_distances = np.concatenate((position, 1.0 - position))
    family_scores = np.full(3, -1.0, dtype=np.float64)
    for row in trace:
        for raw in row.get("alternatives") or []:
            axis = np.abs(np.asarray(raw.get("axis_world") or [0.0, 0.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(axis)) <= 1e-12:
                continue
            family = int(np.argmax(axis))
            family_scores[family] = max(family_scores[family], float(raw.get("score", -1.0)))
    base_axis = np.abs(np.asarray(candidate.axis_world, dtype=np.float64))
    base_family = int(np.argmax(base_axis))
    return np.concatenate((
        center,
        position,
        face_distances,
        body_extent,
        moving_extent / body_extent,
        family_scores,
        np.eye(3, dtype=np.float64)[base_family],
        [float(candidate.signals.get("axis_confidence", 0.0))],
    ))


def predict_axis_family(
    label: str,
    numeric_features: Iterable[float],
    *,
    model_path: Path | None = None,
) -> tuple[int, float, list[float]] | None:
    path = Path(model_path or DEFAULT_AXIS_FAMILY_MODEL)
    if not path.is_file():
        return None
    import joblib
    from scipy.sparse import csr_matrix, hstack

    artifact = joblib.load(path)
    if artifact.get("format") != "arts_gen_kin_axis_family_lr_v2":
        return None
    vectorizer = artifact["vectorizer"]
    scaler = artifact["scaler"]
    classifier = artifact["classifier"]
    text = vectorizer.transform([str(label)])
    numeric = scaler.transform(np.asarray([list(numeric_features)], dtype=np.float64))
    features = hstack((text, csr_matrix(numeric)))
    probabilities = classifier.predict_proba(features)[0]
    index = int(np.argmax(probabilities))
    family = int(classifier.classes_[index])
    return family, float(probabilities[index]), [float(value) for value in probabilities]


def apply_axis_family_reranker(
    result,
    *,
    label: str,
    body_points: np.ndarray,
    moving_points: np.ndarray,
    observation,
    max_iterations: int,
):
    features = axis_family_numeric_features(
        body_points, moving_points, result.candidate, result.trace, observation,
    )
    prediction = predict_axis_family(label, features)
    if prediction is None or prediction[1] < 0.65:
        return result, None
    family, confidence, probabilities = prediction
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
        return result, None
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
            "axis_family_model_used": 1.0,
            "axis_family_model_family": float(family),
            "axis_family_model_confidence": confidence,
            "axis_confidence": max(float(result.candidate.signals.get("axis_confidence", 0.0)), confidence),
        },
        reason=result.candidate.reason + "; reranked by frozen train-object axis-family model",
    )
    evidence = {
        "family": family,
        "family_name": "XYZ"[family],
        "confidence": confidence,
        "probabilities": probabilities,
        "model": "kin_axis_family_lr_v2",
    }
    trace = list(result.trace)
    row = {
        "iteration": min(max_iterations, len(trace) + 1),
        "stage": "axis_family_model_critic",
        "selected": asdict(refined),
        "axis_family_model": evidence,
        "alternatives": compatible[:6],
    }
    if len(trace) < max_iterations:
        trace.append(row)
    else:
        trace[-1] = row
    return replace(result, candidate=refined, iterations=len(trace), trace=trace), evidence
