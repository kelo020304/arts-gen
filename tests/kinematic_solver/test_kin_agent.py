from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from post_process.kinematic_solver.sdk import kin_agent as kin_agent_module
from post_process.kinematic_solver.sdk import motion_observation as motion_observation_module
from post_process.kinematic_solver.sdk.kin_agent import (
    KinematicAgentConfig,
    KinematicAgentResult,
    KinematicCandidate,
    infer_kinematics,
)
from post_process.kinematic_solver.sdk.kin_export import (
    delivery_joint_payload,
    export_decoded_mesh_obj,
    write_kinematic_bundle_mjcf,
    write_kinematic_mjcf,
    write_kinematic_usda,
)
from post_process.kinematic_solver.run_kin_agent_bundle import (
    _assert_decoded_delivery_input,
    _apply_motion_observation,
    _gate_collision_revision_by_axis_evidence,
    _select_motion_observation,
    run_bundle,
)
from post_process.kinematic_solver.sdk.motion_observation import (
    MotionObservationEstimate,
    _to_source_frame,
    estimate_static_part_observation,
    estimate_motion_from_render_states,
    fit_motion_trajectory,
)
from post_process.kinematic_solver.sdk.range_prior import load_range_prior
from post_process.kinematic_solver.sdk.axis_family_model import predict_axis_family
from post_process.kinematic_solver.sdk.thin_axis_critic import (
    apply_phyx_knob_thin_axis_critic,
    decoded_thin_axis_evidence,
)
from post_process.kinematic_solver.sdk.door_contact_axis_critic import (
    apply_phyx_door_contact_axis_critic,
    decoded_door_contact_axis_evidence,
)


def _box(low, high, count=8):
    axes = [np.linspace(low[index], high[index], count) for index in range(3)]
    return np.asarray([
        (x, y, z)
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
        if x in (axes[0][0], axes[0][-1])
        or y in (axes[1][0], axes[1][-1])
        or z in (axes[2][0], axes[2][-1])
    ])


@pytest.mark.parametrize("token", [
    "raw/partseg/ra_001/body.obj",
    "reconstruction/part_info/001/part_info.json",
    "joint_transforms/001.json",
    "source/model/001/Aligned.usd",
])
def test_delivery_input_rejects_gt_and_source_paths(tmp_path, token):
    with pytest.raises(ValueError, match="decoded SLat assets only"):
        _assert_decoded_delivery_input(tmp_path / token)


def test_collision_range_revision_rejects_axis_model_without_independent_evidence():
    candidate = KinematicCandidate(
        joint_type="revolute", axis_world=(0.0, 0.0, 1.0), origin_world=(0.0, 0.0, 0.0),
        lower=0.0, upper=1.0, score=0.9,
        signals={"axis_confidence": 0.95, "axis_family_model_used": 1.0},
    )
    feedback = {
        "status": "accepted", "requires_review": False, "review_reason": None,
        "accept_gate": {"accepted": True, "reasons": []},
    }

    gated = _gate_collision_revision_by_axis_evidence(feedback, candidate)

    assert gated["accept_gate"]["accepted"] is False
    assert "axis_family_model_not_independently_observed" in gated["review_reason"]


def test_collision_range_revision_accepts_independent_motion_axis_evidence():
    candidate = KinematicCandidate(
        joint_type="revolute", axis_world=(0.0, 1.0, 0.0), origin_world=(0.0, 0.0, 0.0),
        lower=0.0, upper=1.0, score=0.9,
        signals={
            "axis_confidence": 0.9,
            "axis_family_model_used": 1.0,
            "motion_axis_family_used": 1.0,
        },
    )
    feedback = {
        "status": "accepted", "requires_review": False, "review_reason": None,
        "accept_gate": {"accepted": True, "reasons": []},
    }

    gated = _gate_collision_revision_by_axis_evidence(feedback, candidate)

    assert gated["accept_gate"]["accepted"] is True


def test_collision_range_revision_accepts_independent_door_contact_axis_evidence():
    candidate = KinematicCandidate(
        joint_type="revolute", axis_world=(0.0, 0.0, 1.0), origin_world=(0.0, 0.0, 0.0),
        lower=0.0, upper=1.0, score=0.9,
        signals={"axis_confidence": 0.9, "phyx_door_contact_axis_used": 1.0},
    )
    feedback = {
        "status": "accepted", "requires_review": False, "review_reason": None,
        "accept_gate": {"accepted": True, "reasons": []},
    }

    gated = _gate_collision_revision_by_axis_evidence(feedback, candidate)

    assert gated["accept_gate"]["accepted"] is True


def test_realappliance_motion_observation_is_mapped_into_decoded_frame(tmp_path):
    render_root = tmp_path / "realappliance" / "renders" / "001"
    mapped = _to_source_frame(
        np.asarray([1.0, 2.0, 3.0]),
        {"scale": 1.0, "offset": [0.0, 0.0, 0.0]},
        render_root,
    )

    assert mapped == pytest.approx((1.0, 3.0, -2.0))


def test_static_part_observation_uses_one_state_and_multiple_views(tmp_path, monkeypatch):
    render_root = tmp_path / "realappliance" / "renders" / "001"
    state = render_root / "angle_0"
    state.mkdir(parents=True)
    (state / "camera_transforms.json").write_text(
        '{"scale":1,"offset":[0,0,0],"frames":['
        '{"view_index":0},{"view_index":1},{"view_index":2}]}'
    )
    (state / "bbox_gt.json").write_text(
        '{"parts":{"timer_knob_0":{"views":{'
        '"0":{"bbox":[1,1,2,2]},"1":{"bbox":[1,1,2,2]},'
        '"2":{"bbox":[1,1,2,2]}}}}}'
    )
    monkeypatch.setattr(
        motion_observation_module,
        "_bbox_cone_center",
        lambda _cameras, _views: (np.asarray([1.0, 2.0, 3.0]), 0.9),
    )

    observation = estimate_static_part_observation(render_root, "timer_knob_0")

    assert observation is not None
    assert observation.center_world == pytest.approx((1.0, 3.0, -2.0))
    assert observation.state_index == 0
    assert observation.view_count == 3
    assert observation.trajectory_points == (observation.center_world,)


def test_agent_recovers_drawer_motion_without_gt():
    moving = _box((0.05, -0.4, -0.3), (0.8, 0.4, 0.3))
    static = np.concatenate([
        _box((-0.1, -0.5, -0.4), (0.0, 0.5, 0.4)),
        _box((0.0, -0.55, -0.45), (0.9, -0.5, 0.45)),
        _box((0.0, 0.5, -0.45), (0.9, 0.55, 0.45)),
    ])
    result = infer_kinematics(moving, static)

    assert result.iterations < 10
    assert result.candidate.joint_type == "prismatic"
    assert abs(result.candidate.axis_world[0]) > 0.98
    assert result.candidate.upper - result.candidate.lower > 0.4
    assert result.trace[-1]["selected"]["signals"]["max_excess_collision"] < 0.02
    assert 0.0 < result.candidate.signals["axis_confidence"] <= 1.0
    assert 0.0 < result.candidate.signals["type_confidence"] <= 1.0
    assert 0.0 < result.candidate.signals["range_confidence"] <= 1.0
    assert result.candidate.signals["range_censored"] in {0.0, 1.0}


def test_physx_0511_drawer_uses_frozen_decoded_axis_convention():
    moving = _box((0.05, -0.4, -0.3), (0.8, 0.4, 0.3))
    static = _box((-0.1, -0.5, -0.4), (0.0, 0.5, 0.4))

    result = infer_kinematics(
        moving,
        static,
        joint_type_hint="prismatic",
        part_category="drawer",
        dataset_profile="physx-0511-drawer-door",
    )

    assert result.candidate.axis_world == pytest.approx((0.0, 1.0, 0.0))
    assert result.candidate.signals["axis_profile_dataset_convention"] == 1.0
    assert result.candidate.signals["axis_confidence"] >= 0.9


def test_agent_recovers_door_hinge_direction_without_gt():
    moving = _box((0.0, -0.02, 0.0), (1.0, 0.02, 1.0), count=9)
    static = _box((-0.15, -0.2, -0.1), (-0.03, 0.2, 1.1), count=9)
    result = infer_kinematics(moving, static)

    assert result.candidate.joint_type == "revolute"
    assert abs(result.candidate.axis_world[2]) > 0.95
    assert abs(result.candidate.origin_world[0]) < 0.08


def test_agent_enforces_sub_ten_iteration_budget():
    with pytest.raises(ValueError, match=r"\[1, 9\]"):
        KinematicAgentConfig(max_iterations=10)


def test_prismatic_self_refinement_is_a_critic_driven_revision(monkeypatch):
    angle = np.deg2rad(12.0)
    initial_axis = np.asarray([np.cos(angle), np.sin(angle), 0.0])

    monkeypatch.setattr(
        kin_agent_module,
        "_seed_axes",
        lambda moving, static, hint: [initial_axis],
    )

    def evaluate(kind, axis, origin, *args, **kwargs):
        alignment = abs(float(np.asarray(axis) @ np.asarray([1.0, 0.0, 0.0])))
        score = 0.4 + 0.5 * alignment if kind == "prismatic" else 0.1
        return KinematicCandidate(
            joint_type=kind,
            axis_world=tuple(float(value) for value in axis),
            origin_world=tuple(float(value) for value in origin),
            lower=0.0,
            upper=0.4,
            score=score,
            signals={
                "axis_geometry_prior": alignment,
                "axis_profile_prior": 0.5,
                "axis_profile_active": 0.0,
                "max_excess_collision": 0.03,
                "range_censored": 0.0,
            },
        )

    monkeypatch.setattr(kin_agent_module, "_evaluate_candidate", evaluate)
    moving = _box((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2), count=4)
    static = _box((0.3, -0.3, -0.3), (0.5, 0.3, 0.3), count=4)

    result = infer_kinematics(
        moving,
        static,
        config=KinematicAgentConfig(max_iterations=7, convergence_score=0.899),
    )

    assert result.iterations == 2
    assert result.candidate.axis_world == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    revision = result.trace[1]
    assert revision["stage"] == "propose_validate_revise"
    assert revision["incumbent"]["axis_world"] == pytest.approx(initial_axis)
    assert revision["critic_feedback"]["verdict"] == "revise"
    assert "refine_axis_locally" in revision["critic_feedback"]["recommended_actions"]
    assert revision["proposals_generated"] > 1
    assert revision["validation"]["candidate_count"] == revision["proposals_generated"]
    assert revision["validation"]["score_gain"] > 0.0
    assert revision["decision"] == {
        "action": "accept_revision",
        "stop_reason": "converged_score",
    }


def test_censored_prismatic_axis_does_not_drift_without_observable_evidence():
    moving = _box((0.05, -0.4, -0.3), (0.8, 0.4, 0.3))
    static = np.concatenate([
        _box((-0.1, -0.5, -0.4), (0.0, 0.5, 0.4)),
        _box((0.0, -0.55, -0.45), (0.9, -0.5, 0.45)),
        _box((0.0, 0.5, -0.45), (0.9, 0.55, 0.45)),
    ])

    result = infer_kinematics(moving, static)

    assert result.candidate.axis_world == pytest.approx((1.0, 0.0, 0.0))
    final = result.trace[-1]
    issue_codes = {item["code"] for item in final["critic_feedback"]["issues"]}
    assert "axis_refinement_unidentifiable" in issue_codes
    assert result.iterations == 1
    assert final["proposals_generated"] > 1
    assert final["decision"]["action"] == "select_initial"
    assert final["stop_reason"] == "no_revision_available"


def test_single_iteration_budget_has_explicit_stop_reason():
    moving = _box((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2), count=4)
    static = _box((0.3, -0.3, -0.3), (0.5, 0.3, 0.3), count=4)

    result = infer_kinematics(
        moving,
        static,
        config=KinematicAgentConfig(max_iterations=1),
    )

    assert result.iterations == 1
    assert result.trace[0]["stop_reason"] == "iteration_budget_exhausted"
    assert result.trace[0]["decision"]["stop_reason"] == "iteration_budget_exhausted"


def test_motion_observation_fits_prismatic_trajectory():
    points = np.column_stack((np.linspace(-0.2, 0.4, 9), np.zeros(9), np.zeros(9)))

    estimate = fit_motion_trajectory(points, "prismatic")

    assert estimate is not None
    axis, _origin, span, confidence, diagnostics = estimate
    assert abs(axis[0]) > 0.999
    assert span == pytest.approx(0.6)
    assert confidence > 0.8
    assert diagnostics["trajectory_linearity"] > 0.99


def test_motion_observation_fits_revolute_circle():
    angles = np.linspace(0.0, np.pi / 2.0, 10)
    center = np.asarray([0.2, -0.3, 0.1])
    points = center + np.column_stack((0.4 * np.cos(angles), 0.4 * np.sin(angles), np.zeros(10)))

    estimate = fit_motion_trajectory(points, "revolute")

    assert estimate is not None
    axis, origin, span, confidence, diagnostics = estimate
    assert abs(axis[2]) > 0.999
    assert np.linalg.norm(origin[:2] - center[:2]) < 1e-6
    assert span == pytest.approx(np.pi / 2.0)
    assert confidence > 0.7
    assert diagnostics["trajectory_circle_residual"] < 1e-8


def _motion_estimate(points, joint_type, *, confidence=None):
    fitted = fit_motion_trajectory(points, joint_type)
    assert fitted is not None
    axis, origin, span, fitted_confidence, diagnostics = fitted
    return MotionObservationEstimate(
        joint_type=joint_type,
        axis_world=tuple(axis),
        origin_world=tuple(origin),
        observed_span=span,
        confidence=fitted_confidence if confidence is None else confidence,
        state_count=len(points),
        trajectory_points=tuple(tuple(point) for point in points),
        diagnostics=diagnostics,
        input_files=(),
    )


def _agent_result(axis=(1.0, 0.0, 0.0), *, axis_confidence=0.7, joint_type="revolute"):
    candidate = KinematicCandidate(
        joint_type=joint_type,
        axis_world=axis,
        origin_world=(0.0, 0.0, 0.0),
        lower=0.0,
        upper=0.5,
        score=0.8,
        signals={
            "axis_confidence": axis_confidence,
            "type_confidence": 0.7,
            "range_confidence": 0.2,
            "range_censored": 1.0,
        },
    )
    alternatives = [
        {**candidate.__dict__},
        {**candidate.__dict__, "axis_world": (0.0, 0.0, 1.0), "score": 0.77},
    ]
    return KinematicAgentResult(
        candidate=candidate,
        iterations=1,
        trace=[{"iteration": 1, "selected": alternatives[0], "alternatives": alternatives}],
    )


@pytest.mark.parametrize(
    ("points", "expected_type"),
    [
        (np.column_stack((np.linspace(0.0, 0.4, 10), np.zeros(10), np.zeros(10))), "prismatic"),
        (
            np.column_stack((
                0.4 * np.cos(np.linspace(0.0, np.pi / 2.0, 10)),
                0.4 * np.sin(np.linspace(0.0, np.pi / 2.0, 10)),
                np.zeros(10),
            )),
            "revolute",
        ),
    ],
)
def test_lid_motion_selector_compares_line_and_circle(points, expected_type):
    hypotheses = {
        kind: _motion_estimate(points, kind)
        for kind in ("prismatic", "revolute")
    }

    selected = _select_motion_observation(
        _agent_result(), hypotheses, part_category="lid",
    )

    assert selected is not None
    assert selected.joint_type == expected_type
    assert selected.diagnostics["motion_type_classifier_used"] == 1.0
    assert selected.diagnostics["motion_type_secondary_ratio_threshold"] == pytest.approx(0.10)


def test_lid_motion_selector_records_threshold_uncertainty():
    linear = _motion_estimate(
        np.column_stack((np.linspace(0.0, 0.4, 10), np.zeros(10), np.zeros(10))),
        "prismatic",
    )
    circular = _motion_estimate(
        np.column_stack((
            0.4 * np.cos(np.linspace(0.0, np.pi / 2.0, 10)),
            0.4 * np.sin(np.linspace(0.0, np.pi / 2.0, 10)),
            np.zeros(10),
        )),
        "revolute",
    )
    linear = MotionObservationEstimate(
        **{
            **linear.__dict__,
            "diagnostics": {**linear.diagnostics, "trajectory_secondary_ratio": 0.09},
        }
    )

    selected = _select_motion_observation(
        _agent_result(), {"prismatic": linear, "revolute": circular}, part_category="lid",
    )

    assert selected is not None
    assert selected.joint_type == "prismatic"
    assert selected.diagnostics["motion_type_uncertain"] == 1.0
    assert selected.diagnostics["motion_type_margin"] == pytest.approx(0.01)


@pytest.mark.parametrize("axis_confidence,should_switch", [(0.7, False), (0.2, True)])
def test_door_motion_family_gate_is_conservative_for_low_confidence(
    axis_confidence, should_switch,
):
    result = _agent_result(axis_confidence=axis_confidence)
    points = np.column_stack((np.zeros(10), np.zeros(10), np.linspace(0.0, 0.3, 10)))
    observation = _motion_estimate(points, "revolute", confidence=0.15)
    observation = MotionObservationEstimate(
        **{**observation.__dict__, "axis_world": (0.0, 0.0, 1.0)}
    )

    refined = _apply_motion_observation(
        result, observation, max_iterations=7, part_category="door",
    )

    assert (abs(refined.candidate.axis_world[2]) > 0.999) is should_switch
    assert refined.iterations < 10


def test_door_motion_family_gate_snaps_moderate_observation_to_existing_cardinal():
    result = _agent_result(axis_confidence=0.7)
    points = np.column_stack((np.zeros(10), np.zeros(10), np.linspace(0.0, 0.3, 10)))
    observation = _motion_estimate(points, "revolute", confidence=0.40)
    observation = MotionObservationEstimate(
        **{**observation.__dict__, "axis_world": (0.12, -0.18, 0.976)}
    )

    refined = _apply_motion_observation(
        result, observation, max_iterations=7, part_category="door",
    )

    assert refined.candidate.axis_world == pytest.approx((0.0, 0.0, 1.0))
    assert refined.candidate.signals["motion_axis_family_used"] == 1.0
    assert refined.candidate.signals["motion_observation_used"] == 0.0
    assert refined.trace[-1]["stage"] == "motion_axis_family_critic"


def test_frozen_range_prior_uses_train_only_artifact():
    prior = load_range_prior("realappliance", "knob", "revolute")

    assert prior is not None
    assert prior.n_objects >= 8
    assert 1.0 < prior.median <= 1.5
    assert prior.artifact_version == "arts_gen_kin_range_prior_v3"
    assert prior.strategy == "signed_interval"


def test_motion_observation_rejects_joint_annotation_path(tmp_path: Path):
    forbidden = tmp_path / "reconstruction" / "part_info" / "001"
    forbidden.mkdir(parents=True)

    with pytest.raises(ValueError, match="renders/<object>"):
        estimate_motion_from_render_states(forbidden, "door_0", "revolute")


def test_frozen_axis_family_model_loads_without_gt_rows():
    prediction = predict_axis_family("timer_knob_0", np.zeros(25, dtype=np.float64))

    assert prediction is not None
    family, confidence, probabilities = prediction
    assert family in {0, 1, 2}
    assert 0.0 < confidence <= 1.0
    assert sum(probabilities) == pytest.approx(1.0)


def _axis_family_result(axis=(1.0, 0.0, 0.0), z_score=0.82):
    incumbent = KinematicCandidate(
        joint_type="revolute",
        axis_world=axis,
        origin_world=(0.0, 0.0, 0.0),
        lower=-1.0,
        upper=1.0,
        score=0.95,
        signals={"axis_confidence": 0.7},
    )
    alternatives = [
        asdict(incumbent),
        asdict(KinematicCandidate(
            joint_type="revolute",
            axis_world=(0.0, 0.0, 1.0),
            origin_world=(0.1, 0.2, 0.3),
            lower=-0.8,
            upper=0.8,
            score=z_score,
        )),
    ]
    return KinematicAgentResult(
        candidate=incumbent,
        iterations=1,
        trace=[{"iteration": 1, "alternatives": alternatives, "selected": asdict(incumbent)}],
    )


def test_phyx_knob_thin_axis_critic_selects_existing_cardinal_proposal():
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    moving = np.column_stack((np.cos(theta), np.sin(theta), 0.05 * np.sin(3.0 * theta)))
    result = _axis_family_result(z_score=0.82)

    refined, evidence = apply_phyx_knob_thin_axis_critic(
        result,
        dataset_id="phyx-verse",
        part_category="knob",
        moving_points=moving,
        max_iterations=7,
    )

    assert refined.candidate.axis_world == pytest.approx((0.0, 0.0, 1.0))
    assert refined.candidate.origin_world == pytest.approx((0.1, 0.2, 0.3))
    assert refined.candidate.signals["phyx_thin_axis_used"] == 1.0
    assert evidence["used"] is True
    assert refined.trace[-1]["stage"] == "decoded_thin_axis_family_critic"
    assert refined.iterations < 10


def test_phyx_knob_thin_axis_critic_is_bounded_and_marks_review():
    moving = _box((-1.0, -1.0, -0.05), (1.0, 1.0, 0.05))
    result = _axis_family_result(z_score=0.70)

    refined, evidence = apply_phyx_knob_thin_axis_critic(
        result,
        dataset_id="phyx-verse",
        part_category="knob",
        moving_points=moving,
        max_iterations=7,
    )

    assert refined.candidate.axis_world == result.candidate.axis_world
    assert refined.candidate.signals["phyx_thin_axis_used"] == 0.0
    assert evidence["review_required"] is True
    assert "score drop" in evidence["review_reason"]
    assert refined.iterations == result.iterations


def test_decoded_thin_axis_evidence_rejects_isotropic_cloud():
    moving = _box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    evidence = decoded_thin_axis_evidence(moving)

    assert evidence is not None
    assert evidence["confidence"] < 0.8


def test_phyx_knob_thin_axis_critic_does_not_change_realappliance():
    moving = np.column_stack((np.arange(12), np.arange(12), np.zeros(12)))
    result = _axis_family_result()

    refined, evidence = apply_phyx_knob_thin_axis_critic(
        result,
        dataset_id="realappliance",
        part_category="knob",
        moving_points=moving,
        max_iterations=7,
    )

    assert refined is result
    assert evidence is None


def test_realappliance_knob_thin_axis_fallback_requires_explicit_strict_opt_in():
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    moving = np.column_stack((np.cos(theta), np.sin(theta), 0.02 * np.sin(3.0 * theta)))
    result = _axis_family_result(z_score=0.82)

    refined, evidence = apply_phyx_knob_thin_axis_critic(
        result,
        dataset_id="realappliance",
        part_category="knob",
        moving_points=moving,
        max_iterations=7,
        min_confidence=0.95,
        max_score_drop=0.15,
        allowed_dataset_ids=("realappliance",),
    )

    assert refined.candidate.axis_world == pytest.approx((0.0, 0.0, 1.0))
    assert refined.candidate.signals["decoded_knob_thin_axis_used"] == 1.0
    assert evidence["used"] is True


def _static_z_hinge_geometry():
    moving = _box((-0.05, -0.4, -1.0), (0.05, 0.4, 1.0), count=9)
    body = _box((-0.16, -0.06, -1.05), (-0.07, 0.06, 1.05), count=9)
    return body, moving


def test_phyx_door_contact_axis_critic_selects_existing_cardinal_proposal():
    body, moving = _static_z_hinge_geometry()
    result = _axis_family_result(axis=(0.0, 1.0, 0.0), z_score=0.82)

    refined, evidence = apply_phyx_door_contact_axis_critic(
        result,
        dataset_id="phyx-verse",
        part_category="door",
        body_points=body,
        moving_points=moving,
        max_iterations=7,
    )

    assert evidence["moving_family_name"] == "Z"
    assert evidence["contact_family_name"] == "Z"
    assert evidence["used"] is True
    assert refined.candidate.axis_world == pytest.approx((0.0, 0.0, 1.0))
    assert refined.candidate.origin_world == pytest.approx((0.1, 0.2, 0.3))
    assert refined.candidate.signals["phyx_door_contact_axis_used"] == 1.0
    assert refined.trace[-1]["stage"] == "decoded_door_contact_axis_critic"
    assert refined.iterations < 10


def test_phyx_door_contact_axis_critic_rejects_disagreeing_geometry():
    _, moving = _static_z_hinge_geometry()
    body = _box((-0.06, -0.5, -0.08), (0.06, 0.5, -0.06), count=9)
    result = _axis_family_result(axis=(0.0, 1.0, 0.0), z_score=0.82)

    refined, evidence = apply_phyx_door_contact_axis_critic(
        result,
        dataset_id="phyx-verse",
        part_category="door",
        body_points=body,
        moving_points=moving,
        max_iterations=7,
    )

    assert evidence["family_agreement"] is False
    assert evidence["review_required"] is True
    assert refined.candidate.axis_world == result.candidate.axis_world
    assert refined.candidate.signals["phyx_door_contact_axis_used"] == 0.0


def test_decoded_door_contact_axis_evidence_has_no_metadata_input():
    body, moving = _static_z_hinge_geometry()

    evidence = decoded_door_contact_axis_evidence(body, moving)

    assert evidence is not None
    assert evidence["family_name"] == "Z"
    assert evidence["confidence"] >= 0.65


def test_decoded_thin_axis_evidence_reports_family_without_metadata():
    points = _box((-1.0, -1.0, -0.05), (1.0, 1.0, 0.05))

    evidence = decoded_thin_axis_evidence(points)

    assert evidence is not None
    assert evidence["family_name"] == "Z"
    assert evidence["confidence"] >= 0.8


def test_decoded_mesh_export_contract(tmp_path: Path):
    body = tmp_path / "decoded_body.obj"
    moving = tmp_path / "decoded_part.obj"
    body.write_text("v 0 0 0\n")
    moving.write_text("v 0 0 0\n")
    candidate = KinematicCandidate(
        joint_type="prismatic",
        axis_world=(1.0, 0.0, 0.0),
        origin_world=(0.1, 0.2, 0.3),
        lower=0.0,
        upper=0.4,
        score=0.9,
    )
    xml_path = write_kinematic_mjcf(
        tmp_path / "object.xml",
        object_name="test",
        body_mesh=body,
        moving_mesh=moving,
        joint_name="drawer",
        candidate=candidate,
    )
    usda_path = write_kinematic_usda(
        tmp_path / "object.usda",
        object_name="test",
        body_mesh=body,
        moving_mesh=moving,
        joint_name="drawer",
        candidate=candidate,
    )

    root = ET.parse(xml_path).getroot()
    object_body = root.find("./worldbody/body")
    joint = root.find("./worldbody/body/body/joint")
    assert object_body.attrib["quat"] == "0.707106781 0.707106781 0 0"
    assert joint.attrib["axis"] == "0 0 1"
    assert root.find("compiler").attrib["balanceinertia"] == "true"
    delivery = delivery_joint_payload(candidate, True)
    assert delivery["axis_world"] == [0.0, 0.0, 1.0]
    assert delivery["lower"] == 0.0
    assert delivery["upper"] == pytest.approx(0.4)
    usda = usda_path.read_text()
    assert 'def Mesh "body"' in usda
    assert "primvars:displayColor" in usda
    assert "PhysicsPrismaticJoint" in usda
    assert 'uniform token physics:axis = "X"' in usda
    assert "physics:localRot0" in usda
    assert "physics:axisVector" not in usda


def test_bundle_runner_combines_decoded_parts(tmp_path: Path):
    body = tmp_path / "body.obj"
    drawer = tmp_path / "drawer.obj"
    door = tmp_path / "door.obj"
    for path, points in (
        (body, _box((-0.2, -0.6, -0.5), (0.0, 0.6, 1.2), count=5)),
        (drawer, _box((0.05, -0.4, -0.3), (0.8, 0.4, 0.3), count=5)),
        (door, _box((0.02, -0.02, 0.0), (0.9, 0.02, 1.0), count=5)),
    ):
        path.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in points), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(__import__("json").dumps({
        "object_name": "fixture",
        "parts": [
            {"part_id": -1, "label": "body", "kind": "body", "mesh_path": str(body)},
            {"part_id": 1, "label": "drawer", "kind": "part", "mesh_path": str(drawer)},
            {"part_id": 2, "label": "door", "kind": "part", "mesh_path": str(door)},
        ],
    }), encoding="utf-8")

    payload = run_bundle(summary, tmp_path / "run", max_iterations=4)

    assert payload["format"] == "arts_gen_kin_agent_v17"
    assert len(payload["parts"]) == 2
    assert all(1 <= item["iterations"] < 10 for item in payload["parts"])
    assert all(item["iterations"] <= payload["max_iterations"] for item in payload["parts"])
    assert all(item["trace"][-1]["stage"] == "decoded_collision_feedback" for item in payload["parts"])
    assert all("collision_feedback" in item for item in payload["parts"])
    root = ET.parse(payload["xml_path"]).getroot()
    assert len(root.findall("./worldbody/body/body/joint")) == 2
    assert len(root.findall("./worldbody/body/body/geom[@group='3']")) == 2
    assert payload["apply_root_correction"] is False
    assert payload["collision_audit"]["version"] == "decoded_collision_audit_v2"
    assert payload["collision_audit"]["requires_review"] is True
    assert all("collision_audit" in item for item in payload["parts"])
    assert all("decoded collision audit unverified" in item["review_reasons"] for item in payload["parts"])
    assert "quat" not in root.find("./worldbody/body").attrib
    usda = Path(payload["usd_path"]).read_text()
    assert "xformOp:orient" not in usda
    assert usda.count("def Mesh") == 3
    assert usda.count('def Cube "Collision"') == 2
    assert "PhysicsCollisionAPI" in usda


def test_bundle_runner_applies_phyx_knob_thin_axis_critic(tmp_path: Path, monkeypatch):
    body = tmp_path / "body.obj"
    knob = tmp_path / "knob.obj"
    body.write_text("".join(
        f"v {x} {y} {z}\n" for x, y, z in _box((-1, -1, -1), (1, 1, 1), count=3)
    ))
    knob.write_text("".join(
        f"v {x} {y} {z}\n" for x, y, z in _box((-1, -1, -0.04), (1, 1, 0.04), count=5)
    ))
    summary = tmp_path / "summary.json"
    summary.write_text(__import__("json").dumps({
        "object_name": "fixture",
        "parts": [
            {"part_id": -1, "label": "body", "kind": "body", "mesh_path": str(body)},
            {"part_id": 1, "label": "control_knob_0", "kind": "part", "mesh_path": str(knob)},
        ],
    }))
    monkeypatch.setattr(
        "post_process.kinematic_solver.run_kin_agent_bundle.infer_kinematics",
        lambda *args, **kwargs: _axis_family_result(z_score=0.82),
    )
    monkeypatch.setattr(
        "post_process.kinematic_solver.run_kin_agent_bundle.load_range_prior",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "post_process.kinematic_solver.run_kin_agent_bundle._run_mujoco_validation",
        lambda *args, **kwargs: {"ok": True},
    )

    payload = run_bundle(
        summary, tmp_path / "run", max_iterations=7, dataset_id="phyx-verse",
    )
    row = payload["parts"][0]

    assert row["candidate"]["axis_world"] == pytest.approx((0.0, 0.0, 1.0))
    assert row["phyx_knob_thin_axis_critic"]["used"] is True
    assert row["iterations"] < 10


def test_decoded_vertex_colors_export_as_power_of_two_texture(tmp_path: Path):
    import trimesh
    from PIL import Image

    mesh = trimesh.creation.box(extents=(0.2, 0.3, 0.4))
    mesh.visual.vertex_colors = np.tile(np.asarray([[20, 120, 220, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
    source = tmp_path / "decoded.glb"
    mesh.export(source)

    exported = export_decoded_mesh_obj(source, tmp_path / "assets" / "colored.obj")

    texture = exported.with_suffix(".png")
    assert texture.is_file()
    with Image.open(texture) as image:
        assert image.width == image.height
        assert image.width & (image.width - 1) == 0
    assert "map_Kd colored.png" in exported.with_suffix(".mtl").read_text()
