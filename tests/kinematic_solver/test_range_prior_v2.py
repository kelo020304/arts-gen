import json
from pathlib import Path

import pytest

from post_process.kinematic_solver.build_range_prior import (
    _canonical_decoded_interval,
    build_range_prior,
    resolve_dataset_root,
)
from post_process.kinematic_solver.sdk.kin_agent import KinematicCandidate
from post_process.kinematic_solver.sdk.kin_agent import KinematicAgentResult
from post_process.kinematic_solver.run_kin_agent_bundle import _apply_range_prior
from post_process.kinematic_solver.sdk.range_prior import (
    RANGE_CALIBRATOR_FEATURES,
    RangePriorEstimate,
    calibrate_range_candidate,
    load_range_prior,
)


def _candidate(axis=(-1.0, 0.0, 0.0), lower=0.0, upper=2.0, signals=None):
    return KinematicCandidate(
        joint_type="revolute",
        axis_world=axis,
        origin_world=(0.0, 0.0, 0.0),
        lower=lower,
        upper=upper,
        score=0.8,
        signals=signals or {"range_censored": 1.0},
        reason="test",
    )


def test_realappliance_range_training_uses_decoded_delivery_frame():
    lower, upper = _canonical_decoded_interval(
        "realappliance", (0.0, 1.0, 0.0), 0.0, 1.5,
    )

    assert (lower, upper) == pytest.approx((-1.5, 0.0))


def _prior(**overrides):
    values = {
        "dataset": "phyx-verse",
        "category": "knob",
        "joint_type": "revolute",
        "q10": 2.0, "q25": 2.0, "median": 2.0, "q75": 2.0, "q90": 2.0,
        "lower_q10": -1.0, "lower_q25": -1.0, "lower_median": -1.0,
        "lower_q75": -1.0, "lower_q90": -1.0,
        "upper_q10": 1.0, "upper_q25": 1.0, "upper_median": 1.0,
        "upper_q75": 1.0, "upper_q90": 1.0,
        "n_objects": 123,
        "confidence": 0.72,
        "strategy": "signed_interval",
        "calibrator": None,
        "artifact_version": "arts_gen_kin_range_prior_v3",
    }
    values.update(overrides)
    return RangePriorEstimate(**values)


def test_signed_interval_prior_canonicalizes_axis_and_preserves_center():
    refined, evidence = calibrate_range_candidate(_candidate(), _prior())

    assert refined.axis_world == pytest.approx((1.0, 0.0, 0.0))
    assert (refined.lower, refined.upper) == pytest.approx((-1.0, 1.0))
    assert evidence["estimated_usable_interval"] == pytest.approx([-1.0, 1.0])
    assert evidence["mechanical_stop_confirmed"] is False


def test_range_critic_applies_one_round_and_stays_below_ten():
    result = KinematicAgentResult(candidate=_candidate(), iterations=3, trace=[{}, {}, {}])

    refined = _apply_range_prior(result, _prior(), max_iterations=7)

    assert refined.iterations == 4
    assert refined.iterations < 10
    assert refined.trace[-1]["stage"] == "range_calibration_critic"
    assert refined.trace[-1]["range_calibration"]["mechanical_stop_confirmed"] is False


def test_strategy_none_does_not_change_unsupported_cell():
    result = KinematicAgentResult(candidate=_candidate(lower=-0.4, upper=0.0), iterations=3, trace=[{}, {}, {}])
    door_prior = _prior(category="door", strategy="none")

    refined = _apply_range_prior(result, door_prior, max_iterations=7)

    assert refined is result
    assert (refined.candidate.lower, refined.candidate.upper) == (-0.4, 0.0)


def test_span_envelope_keeps_door_direction_and_caps_noisy_arc_span():
    candidate = _candidate(
        axis=(-1.0, 0.0, 0.0), lower=0.0, upper=2.4,
        signals={
            "range_censored": 1.0,
            "motion_observed_span": 2.4,
            "motion_observed_lower": 0.0,
            "motion_observed_upper": 2.4,
        },
    )
    prior = _prior(
        category="door", strategy="span_envelope",
        median=0.5, q90=0.72,
    )

    refined, evidence = calibrate_range_candidate(candidate, prior)

    assert (refined.lower, refined.upper) == pytest.approx((-0.72, 0.0))
    assert refined.axis_world == pytest.approx((1.0, 0.0, 0.0))
    assert evidence["observed_state_interval"] == pytest.approx([-2.4, 0.0])
    assert evidence["strategy"] == "span_envelope"
    assert evidence["mechanical_stop_confirmed"] is False


def test_ridge_calibrator_separates_observation_estimate_and_uncertainty():
    calibrator = {
        "feature_names": list(RANGE_CALIBRATOR_FEATURES),
        "scaler_mean": [0.0] * len(RANGE_CALIBRATOR_FEATURES),
        "scaler_scale": [1.0] * len(RANGE_CALIBRATOR_FEATURES),
        "coefficients": [1.0] + [0.0] * (len(RANGE_CALIBRATOR_FEATURES) - 1),
        "intercept": 0.2,
        "prediction_clip": [0.1, 1.0],
        "interval_anchor": "upper_zero",
        "absolute_span_residual_quantiles": {"q90": 0.1},
    }
    candidate = _candidate(
        axis=(0.0, -1.0, 0.0), lower=0.0, upper=0.4,
        signals={
            "range_censored": 1.0,
            "motion_observed_span": 0.4,
            "motion_observed_lower": 0.0,
            "motion_observed_upper": 0.4,
            "motion_observation_confidence": 0.8,
        },
    )
    refined, evidence = calibrate_range_candidate(
        candidate,
        _prior(
            dataset="physx-0511-drawer-door", category="drawer", joint_type="prismatic",
            strategy="ridge_observed_span", calibrator=calibrator,
        ),
        object_diagonal=1.5,
        observation_diagnostics={"trajectory_linearity": 0.9},
    )

    assert refined.axis_world == pytest.approx((0.0, 1.0, 0.0))
    assert (refined.lower, refined.upper) == pytest.approx((-0.6, 0.0))
    assert evidence["observed_state_interval"] == pytest.approx([-0.4, 0.0])
    assert evidence["estimated_usable_interval"] == pytest.approx([-0.6, 0.0])
    assert evidence["prediction_interval"]["outer_q90"] == pytest.approx([-0.7, 0.0])
    assert evidence["mechanical_stop_confirmed"] is False


def test_physx_nested_dataset_root_is_resolved(tmp_path: Path):
    nested = tmp_path / "PhysX-Mobility-full-4view-0511" / "PhysX-Mobility-full-4view-0511"
    (nested / "reconstruction" / "part_info").mkdir(parents=True)

    assert resolve_dataset_root(tmp_path, "physx-0511-drawer-door") == nested


def test_builder_writes_canonical_signed_endpoints_without_per_object_rows(tmp_path: Path):
    data_root = tmp_path / "data"
    dataset_root = data_root / "phyx-verse"
    train_ids = []
    for index in range(8):
        object_id = f"object-{index}"
        train_ids.append({"dataset_id": "phyx-verse", "obj_id": object_id})
        target = dataset_root / "reconstruction" / "part_info" / object_id / "part_info.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"parts": {
            "control_knob_0": {
                "type": "control_knob", "joint": "revolute",
                "joint_params": [0, 0, -1, 0, 0, 0, -1, 1],
            }
        }}), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_ids": train_ids}), encoding="utf-8")
    output = tmp_path / "prior.json"

    payload = build_range_prior(split, data_root, output)
    cell = payload["cells"]["phyx-verse|knob|revolute"]

    assert payload["format"] == "arts_gen_kin_range_prior_v3"
    assert cell["lower"]["q50"] == pytest.approx(-1.0)
    assert cell["upper"]["q50"] == pytest.approx(1.0)
    assert cell["runtime_strategy"] == "signed_interval"
    serialized = output.read_text(encoding="utf-8")
    assert "object-0" not in serialized
    assert "joint_params" not in serialized


def test_runtime_loader_reads_only_frozen_artifact(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "prior.json"
    artifact.write_text(json.dumps({
        "format": "arts_gen_kin_range_prior_v3",
        "cells": {
            "phyx-verse|knob|revolute": {
                "usable": True, "n_objects": 10, "runtime_strategy": "signed_interval",
                "q10": 2, "q25": 2, "q50": 2, "q75": 2, "q90": 2,
                "lower": {key: -1 for key in ("q10", "q25", "q50", "q75", "q90")},
                "upper": {key: 1 for key in ("q10", "q25", "q50", "q75", "q90")},
            }
        },
        "calibrators": {},
    }), encoding="utf-8")
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        assert Path(path) == artifact
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    prior = load_range_prior("phyx-verse", "knob", "revolute", path=artifact)

    assert prior is not None
    assert prior.lower_median == -1.0
