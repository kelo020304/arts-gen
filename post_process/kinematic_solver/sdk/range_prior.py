"""Frozen train-only interval priors and range calibrators."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Mapping

import numpy as np


DEFAULT_RANGE_PRIOR = Path(__file__).resolve().parents[1] / "priors" / "kin_range_prior_v3.json"

RANGE_CALIBRATOR_FEATURES = (
    "motion_observed_span",
    "object_diagonal",
    "motion_observation_confidence",
    "trajectory_linearity",
    "visual_hull_support_mean",
    "visual_hull_support_min",
    "trajectory_secondary_ratio",
    "trajectory_tertiary_ratio",
)


@dataclass(frozen=True)
class RangePriorEstimate:
    dataset: str
    category: str
    joint_type: str
    q10: float
    q25: float
    median: float
    q75: float
    q90: float
    lower_q10: float
    lower_q25: float
    lower_median: float
    lower_q75: float
    lower_q90: float
    upper_q10: float
    upper_q25: float
    upper_median: float
    upper_q75: float
    upper_q90: float
    n_objects: int
    confidence: float
    strategy: str
    calibrator: dict | None
    artifact_version: str

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "category": self.category,
            "joint_type": self.joint_type,
            "q10": self.q10,
            "q25": self.q25,
            "median": self.median,
            "q75": self.q75,
            "q90": self.q90,
            "lower_q10": self.lower_q10,
            "lower_q25": self.lower_q25,
            "lower_median": self.lower_median,
            "lower_q75": self.lower_q75,
            "lower_q90": self.lower_q90,
            "upper_q10": self.upper_q10,
            "upper_q25": self.upper_q25,
            "upper_median": self.upper_median,
            "upper_q75": self.upper_q75,
            "upper_q90": self.upper_q90,
            "n_objects": self.n_objects,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "calibrator": self.calibrator,
            "artifact_version": self.artifact_version,
        }


def load_range_prior(
    dataset: str | None,
    category: str | None,
    joint_type: str,
    *,
    path: Path | None = None,
) -> RangePriorEstimate | None:
    prior_path = Path(path or DEFAULT_RANGE_PRIOR)
    if not prior_path.is_file() or not dataset or not category:
        return None
    payload = json.loads(prior_path.read_text(encoding="utf-8"))
    key = f"{str(dataset).lower()}|{str(category).lower()}|{str(joint_type).lower()}"
    cell = (payload.get("cells") or {}).get(key)
    if not isinstance(cell, dict) or not bool(cell.get("usable")):
        return None
    n_objects = int(cell.get("n_objects", 0))
    q10 = float(cell["q10"])
    q25 = float(cell["q25"])
    median = float(cell["q50"])
    q75 = float(cell["q75"])
    q90 = float(cell["q90"])
    relative_width = (q90 - q10) / max(median, 1e-8)
    support = min(1.0, n_objects / 50.0)
    confidence = max(0.15, min(0.72, 0.25 + 0.35 * support + 0.20 / (1.0 + relative_width)))
    lower = cell.get("lower") or {}
    upper = cell.get("upper") or {}
    # v1 remains readable for old frozen benchmark files. It has no signed
    # interval strategy and therefore retains the legacy one-sided behavior.
    lower_values = _endpoint_quantiles(lower, fallback=0.0)
    upper_values = _endpoint_quantiles(upper, fallback=median)
    calibrator = (payload.get("calibrators") or {}).get(key)
    return RangePriorEstimate(
        dataset=str(dataset).lower(),
        category=str(category).lower(),
        joint_type=str(joint_type).lower(),
        q10=q10,
        q25=q25,
        median=median,
        q75=q75,
        q90=q90,
        lower_q10=lower_values[0],
        lower_q25=lower_values[1],
        lower_median=lower_values[2],
        lower_q75=lower_values[3],
        lower_q90=lower_values[4],
        upper_q10=upper_values[0],
        upper_q25=upper_values[1],
        upper_median=upper_values[2],
        upper_q75=upper_values[3],
        upper_q90=upper_values[4],
        n_objects=n_objects,
        confidence=confidence,
        strategy=str(cell.get("runtime_strategy") or "legacy_span"),
        calibrator=dict(calibrator) if isinstance(calibrator, dict) else None,
        artifact_version=str(payload.get("format") or "unknown"),
    )


def range_calibration_features(
    signals: Mapping[str, float],
    observation_diagnostics: Mapping[str, float] | None,
    object_diagonal: float | None,
) -> np.ndarray:
    diagnostics = observation_diagnostics or {}
    values = {
        "motion_observed_span": float(signals.get("motion_observed_span", 0.0)),
        "object_diagonal": float(object_diagonal or 0.0),
        "motion_observation_confidence": float(signals.get("motion_observation_confidence", 0.0)),
        "trajectory_linearity": float(diagnostics.get("trajectory_linearity", 0.0)),
        "visual_hull_support_mean": float(diagnostics.get("visual_hull_support_mean", 0.0)),
        "visual_hull_support_min": float(diagnostics.get("visual_hull_support_min", 0.0)),
        "trajectory_secondary_ratio": float(diagnostics.get("trajectory_secondary_ratio", 0.0)),
        "trajectory_tertiary_ratio": float(diagnostics.get("trajectory_tertiary_ratio", 0.0)),
    }
    return np.asarray([values[name] for name in RANGE_CALIBRATOR_FEATURES], dtype=np.float64)


def calibrate_range_candidate(
    candidate,
    prior: RangePriorEstimate,
    *,
    object_diagonal: float | None = None,
    observation_diagnostics: Mapping[str, float] | None = None,
):
    """Return a calibrated candidate and structured, GT-free range evidence."""
    axis, candidate_lower, candidate_upper = _canonical_axis_interval(
        candidate.axis_world, candidate.lower, candidate.upper,
    )
    raw_axis = np.asarray(candidate.axis_world, dtype=np.float64)
    axis_flipped = float(raw_axis @ axis) < 0.0
    observed_interval = _observed_interval(candidate.signals)
    if observed_interval is not None and axis_flipped:
        observed_interval = [-observed_interval[1], -observed_interval[0]]
    observed_span = float(candidate.signals.get("motion_observed_span", 0.0))
    strategy = prior.strategy
    prediction_interval = None
    calibrated_span = None

    if strategy == "ridge_observed_span" and prior.calibrator and observed_span > 1e-8:
        calibrated_span = _predict_ridge_span(
            prior.calibrator,
            range_calibration_features(candidate.signals, observation_diagnostics, object_diagonal),
        )
        anchor = str(prior.calibrator.get("interval_anchor") or "upper_zero")
        if anchor == "upper_zero":
            lower, upper = -calibrated_span, 0.0
        elif anchor == "lower_zero":
            lower, upper = 0.0, calibrated_span
        else:
            lower, upper = prior.lower_median, prior.upper_median
        residuals = prior.calibrator.get("absolute_span_residual_quantiles") or {}
        residual = float(residuals.get("q90", 0.0))
        span_low = max(0.0, calibrated_span - residual)
        span_high = calibrated_span + residual
        prediction_interval = (
            [-span_high, 0.0] if anchor == "upper_zero"
            else [0.0, span_high] if anchor == "lower_zero"
            else [prior.lower_q10, prior.upper_q90]
        )
        prediction_interval_inner = (
            [-span_low, 0.0] if anchor == "upper_zero"
            else [0.0, span_low] if anchor == "lower_zero"
            else [prior.lower_q90, prior.upper_q10]
        )
    elif strategy == "span_envelope":
        fallback_span = max(float(prior.median), min(observed_span, float(prior.q90)))
        if observed_interval is not None:
            observed_lower, observed_upper = observed_interval
            if observed_lower >= -1e-6:
                lower, upper = 0.0, fallback_span
            elif observed_upper <= 1e-6:
                lower, upper = -fallback_span, 0.0
            else:
                scale = fallback_span / max(observed_upper - observed_lower, 1e-8)
                lower, upper = observed_lower * scale, observed_upper * scale
        elif candidate_upper > 0.0:
            lower, upper = 0.0, fallback_span
        else:
            lower, upper = -fallback_span, 0.0
        prediction_interval = [float(lower), float(upper)]
        prediction_interval_inner = [float(lower), float(upper)]
    elif strategy in {"signed_interval", "ridge_observed_span"}:
        lower, upper = prior.lower_median, prior.upper_median
        prediction_interval = [prior.lower_q10, prior.upper_q90]
        prediction_interval_inner = [prior.lower_q90, prior.upper_q10]
        if strategy == "ridge_observed_span":
            strategy = "signed_interval_fallback"
    else:
        return candidate, {
            "applied": False,
            "strategy": strategy,
            "observed_state_interval": observed_interval,
            "mechanical_stop_confirmed": False,
        }

    refined = replace(
        candidate,
        axis_world=tuple(float(value) for value in axis),
        lower=float(lower),
        upper=float(upper),
        signals={
            **candidate.signals,
            "range_prior_used": 1.0,
            "range_prior_q10": float(prior.q10),
            "range_prior_q50": float(prior.median),
            "range_prior_q90": float(prior.q90),
            "range_prior_objects": float(prior.n_objects),
            "range_confidence": max(
                float(candidate.signals.get("range_confidence", 0.0)),
                float(prior.confidence),
            ),
            "range_censored": 1.0,
            "mechanical_stop_confirmed": 0.0,
        },
        reason=candidate.reason + f"; calibrated with frozen train-only {strategy}",
    )
    evidence = {
        "applied": True,
        "strategy": strategy,
        "observed_state_interval": observed_interval,
        "estimated_usable_interval": [float(lower), float(upper)],
        "prediction_interval": {
            "outer_q90": prediction_interval,
            "inner_q90": prediction_interval_inner,
        },
        "calibrated_span": calibrated_span,
        "mechanical_stop_confirmed": False,
    }
    return refined, evidence


def _predict_ridge_span(calibrator: Mapping[str, object], features: np.ndarray) -> float:
    feature_names = tuple(calibrator.get("feature_names") or ())
    if feature_names != RANGE_CALIBRATOR_FEATURES:
        raise ValueError(f"range calibrator feature mismatch: {feature_names}")
    mean = np.asarray(calibrator["scaler_mean"], dtype=np.float64)
    scale = np.asarray(calibrator["scaler_scale"], dtype=np.float64)
    coefficients = np.asarray(calibrator["coefficients"], dtype=np.float64)
    if not (features.shape == mean.shape == scale.shape == coefficients.shape):
        raise ValueError("range calibrator coefficient shape mismatch")
    standardized = (features - mean) / np.where(scale > 1e-12, scale, 1.0)
    prediction = float(calibrator["intercept"]) + float(standardized @ coefficients)
    bounds = calibrator.get("prediction_clip") or [0.0, float("inf")]
    return float(np.clip(prediction, float(bounds[0]), float(bounds[1])))


def _canonical_axis_interval(axis_values, lower: float, upper: float):
    axis = np.asarray(axis_values, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("zero-length range axis")
    axis = axis / norm
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        return -axis, -float(upper), -float(lower)
    return axis, float(lower), float(upper)


def _observed_interval(signals: Mapping[str, float]) -> list[float] | None:
    span = float(signals.get("motion_observed_span", 0.0))
    if span <= 1e-8:
        return None
    return [
        float(signals.get("motion_observed_lower", 0.0)),
        float(signals.get("motion_observed_upper", span)),
    ]


def _endpoint_quantiles(values: Mapping[str, float], *, fallback: float) -> tuple[float, ...]:
    return tuple(float(values.get(key, fallback)) for key in ("q10", "q25", "q50", "q75", "q90"))
