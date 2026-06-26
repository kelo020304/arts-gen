import pytest

import numpy as np

from post_process.kinematic_solver.sdk.motion_validation import (
    validate_motion_samples,
    validate_motion_search,
)
from post_process.kinematic_solver.sdk.schemas import EstimateContext, LimitEstimate


class FakeBackend:
    def __init__(self):
        self.q_values = []

    def reset_to_identity(self):
        pass

    def set_pose(self, part_name, rotation, translation):
        if part_name == "part_02":
            self.q_values.append(float(translation[1]))

    def overlapping_pairs(self, moving_parts, static_parts):
        if self.q_values and abs(self.q_values[-1]) > 0.11:
            return [("part_02", "body")]
        return []


class FakeDrawerClearanceBackend:
    def __init__(self):
        self.translation = np.zeros(3, dtype=float)

    def reset_to_identity(self):
        self.translation = np.zeros(3, dtype=float)

    def set_pose(self, part_name, rotation, translation):
        if part_name == "part_02":
            self.translation = np.asarray(translation, dtype=float)

    def overlapping_pairs(self, moving_parts, static_parts):
        if "part_02" not in moving_parts or "body" not in static_parts:
            return []
        moved_far_enough_out = self.translation[1] <= -0.12
        lateral_slip_is_small = abs(float(self.translation[0])) <= 0.02
        if moved_far_enough_out and lateral_slip_is_small:
            return []
        return [("part_02", "body")]


def test_validate_motion_samples_rejects_colliding_candidate():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0.0, 1.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )

    errors = validate_motion_samples(
        ctx,
        [LimitEstimate("part_02", 0.0, 0.16, axis_world=[0.0, 1.0, 0.0])],
        backend=FakeBackend(),
        sample_count=5,
    )

    assert errors
    assert "collides/interferes" in errors[0]


def test_validate_motion_samples_accepts_non_colliding_candidate():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0.0, 1.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )

    errors = validate_motion_samples(
        ctx,
        [LimitEstimate("part_02", 0.0, 0.10, axis_world=[0.0, 1.0, 0.0])],
        backend=FakeBackend(),
        sample_count=5,
    )

    assert errors == []


def test_validate_motion_samples_requires_joint_pose_fields():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_02": {"type": "prismatic"}},
        evidence={},
    )

    errors = validate_motion_samples(
        ctx,
        [LimitEstimate("part_02", 0.0, 0.10)],
        backend=FakeBackend(),
    )

    assert errors == ["part_02: cannot run motion validation without axis_world, origin_world, moving_parts, static_parts"]


def _drawer_clearance_vertices() -> dict[str, np.ndarray]:
    return {
        "body": np.asarray(
            [
                [-0.09, -0.07, -0.12],
                [0.09, 0.11, 0.12],
            ],
            dtype=float,
        ),
        "part_02": np.asarray(
            [
                [-0.087, -0.11, -0.11],
                [0.087, 0.088, 0.003],
            ],
            dtype=float,
        ),
    }


def test_validate_motion_search_rejects_drawer_axis_from_actual_clearance_not_recommendation():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0.0, 1.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_02", 0.0, 0.16, axis_world=[0.0, 1.0, 0.0], axis_label="+Y")],
        backend=FakeDrawerClearanceBackend(),
        raw_vertices_by_part=_drawer_clearance_vertices(),
    )

    assert result.errors
    assert "SDK selected action -Y" in result.errors[0]
    assert result.traces[0]["selected_axis_label"] == "-Y"
    trials = {
        trial["axis_label"]: trial
        for trial in result.traces[0]["axis_trials"]
    }
    assert trials["+Y"]["limit"] <= 0.005
    assert trials["+Y"]["samples"][1]["valid"] is False


def test_validate_motion_search_does_not_select_sideways_drawer_axis_that_never_clears_parent():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [1.0, 0.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={"part_02": {"labels": ["pull-out drawer pan"]}},
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_02", 0.0, 0.13, axis_world=[1.0, 0.0, 0.0], axis_label="+X")],
        backend=FakeDrawerClearanceBackend(),
        raw_vertices_by_part=_drawer_clearance_vertices(),
    )

    assert result.errors
    assert "SDK selected action -Y" in result.errors[0]
    assert result.traces[0]["selected_axis_label"] == "-Y"
    trials = {
        trial["axis_label"]: trial
        for trial in result.traces[0]["axis_trials"]
    }
    assert trials["+X"]["limit"] < 0.10
    assert any(
        sample["q"] >= 0.10 and sample["valid"] is False
        for sample in trials["+X"]["samples"]
    )


def test_validate_motion_search_accepts_drawer_matching_actual_motion_search_action():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0.0, 1.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_02", 0.0, 0.15, axis_world=[0.0, -1.0, 0.0], axis_label="-Y")],
        backend=FakeDrawerClearanceBackend(),
        raw_vertices_by_part=_drawer_clearance_vertices(),
    )

    assert result.errors == []
    assert result.traces[0]["selected_axis_label"] == "-Y"
    assert result.traces[0]["axis_trials"]
    trials = {
        trial["axis_label"]: trial
        for trial in result.traces[0]["axis_trials"]
    }
    assert trials["+Y"]["limit"] <= 0.005
    assert trials["-Y"]["limit"] > 0.0


def test_validate_continuous_motion_rejects_drawer_still_intersecting_parent_at_upper():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [1.0, 0.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_02": {"labels": ["drawer", "pull-out pan"]},
        },
    )

    result = validate_motion_search(
        ctx,
        [
            LimitEstimate(
                "part_02",
                0.0,
                0.07,
                axis_world=[0.9396926207859085, -0.34202014332566877, 0.0],
            )
        ],
        backend=FakeDrawerClearanceBackend(),
        raw_vertices_by_part=_drawer_clearance_vertices(),
    )

    assert result.errors
    assert "prismatic rest-face exit axis -Y" in result.errors[0]


def test_validate_continuous_motion_accepts_drawer_clear_at_upper_endpoint():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [1.0, 0.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_02": {"labels": ["drawer", "pull-out pan"]},
        },
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_02", 0.0, 0.13, axis_world=[0.0, -1.0, 0.0])],
        backend=FakeDrawerClearanceBackend(),
        raw_vertices_by_part=_drawer_clearance_vertices(),
    )

    assert result.errors == []
    assert result.traces[0]["candidate_samples"]


def test_validate_continuous_motion_rejects_prismatic_axis_without_overlap_progress():
    ctx = EstimateContext(
        object_id="ra_036",
        joints={
            "part_07": {
                "type": "prismatic",
                "axis_world": [0.0, 0.0, 1.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_07"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_07": {"labels": []},
        },
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_07", 0.0, 0.03, axis_world=[0.0, 0.0, 1.0])],
        backend=FakeBackend(),
        raw_vertices_by_part={
            "body": np.asarray([[-0.10, -0.10, -0.10], [0.10, 0.10, 0.10]], dtype=float),
            "part_07": np.asarray([[-0.08, -0.08, -0.04], [0.08, 0.08, 0.04]], dtype=float),
        },
    )

    assert result.errors
    assert "Articraft-style prismatic pose QC failure" in result.errors[0]
    assert "target_axis_world=" in result.errors[0]
    assert result.traces[0]["prismatic_overlap_progress"]["candidate_axis_label"] == "+Z"


def test_validate_continuous_motion_rejects_short_slanted_drawer_motion():
    ctx = EstimateContext(
        object_id="ra_036",
        joints={
            "part_07": {
                "type": "prismatic",
                "axis_world": [0.9443573954837094, 0.013836829969331997, 0.3286299617071223],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_07"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_07": {"labels": ["drawer"]},
        },
    )

    result = validate_motion_search(
        ctx,
        [
            LimitEstimate(
                "part_07",
                0.0,
                0.0225,
                axis_world=[0.9443573954837094, 0.013836829969331997, 0.3286299617071223],
            )
        ],
        backend=FakeBackend(),
        raw_vertices_by_part={
            "body": np.asarray([[-0.10, -0.10, -0.10], [0.10, 0.10, 0.10]], dtype=float),
            "part_07": np.asarray([[-0.08, -0.08, -0.04], [0.08, 0.08, 0.04]], dtype=float),
        },
    )

    assert result.errors
    assert "Articraft-style prismatic pose QC failure" in result.errors[0]
    assert "target_axis_world=" in result.errors[0]
    trace = result.traces[0]["prismatic_overlap_progress"]
    assert trace["candidate_overlap_ratio"] > trace["max_allowed_overlap_ratio"]
    assert trace["angle_to_best_axis_degrees"] > trace["max_axis_angle_degrees"]


def test_validate_continuous_motion_rejects_drawer_range_that_barely_exits_body():
    ctx = EstimateContext(
        object_id="ra_036",
        joints={
            "part_07": {
                "type": "prismatic",
                "axis_world": [1.0, 0.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_07"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_07": {"labels": ["drawer"]},
        },
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_07", 0.0, 0.0225, axis_world=[1.0, 0.0, 0.0])],
        backend=FakeBackend(),
        raw_vertices_by_part={
            "body": np.asarray([[-0.10, -0.10, -0.10], [0.10, 0.10, 0.10]], dtype=float),
            "part_07": np.asarray([[-0.08, -0.08, -0.04], [0.08, 0.08, 0.04]], dtype=float),
        },
    )

    assert result.errors
    assert "range is too short" in result.errors[0]


def test_validate_continuous_motion_rejects_prismatic_axis_that_exits_wrong_face():
    ctx = EstimateContext(
        object_id="ra_036",
        joints={
            "part_07": {
                "type": "prismatic",
                "axis_world": [0.0, 0.0, -1.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_07"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_07": {"labels": ["drawer"]},
        },
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_07", 0.0, 0.15, axis_world=[0.0, 0.0, -1.0])],
        backend=FakeBackend(),
        raw_vertices_by_part={
            "body": np.asarray([[-0.12, -0.10, -0.10], [0.10, 0.10, 0.10]], dtype=float),
            "part_07": np.asarray([[-0.08, -0.08, -0.04], [0.14, 0.08, 0.04]], dtype=float),
        },
    )

    assert result.errors
    assert "prismatic rest-face exit axis +X" in result.errors[0]
    assert result.traces[0]["axis_validation"]["target_axis_label"] == "+X"


def _top_knob_vertices() -> dict[str, np.ndarray]:
    return {
        "part_00": np.asarray(
            [
                [-0.018, -0.018, 0.004],
                [0.018, 0.018, 0.008],
            ],
            dtype=float,
        ),
        "body": np.asarray(
            [
                [-0.10, -0.10, -0.10],
                [0.10, 0.10, 0.00],
            ],
            dtype=float,
        ),
    }


def _slanted_top_knob_vertices(angle_degrees: float = 20.0) -> dict[str, np.ndarray]:
    hx, hy, hz = 0.018, 0.018, 0.004
    points = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )
    angle = np.deg2rad(angle_degrees)
    rotation_y = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=float,
    )
    points = (rotation_y @ points.T).T + np.asarray([0.0, 0.0, 0.008])
    return {
        "part_00": points,
        "body": np.asarray(
            [
                [-0.10, -0.10, -0.10],
                [0.10, 0.10, 0.00],
            ],
            dtype=float,
        ),
    }


def test_validate_motion_search_rejects_top_knob_axis_that_sweeps_instead_of_spinning():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_00": {
                "type": "revolute",
                "axis_world": [0.0, 1.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "part_00": {
                "labels": ["temperature knob"],
                "recommended_axis_label": "+Y",
                "recommended_axis_world": [0.0, 1.0, 0.0],
            }
        },
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_00", 0.0, 1.5, axis_world=[0.0, 1.0, 0.0], axis_label="+Y")],
        backend=FakeBackend(),
        raw_vertices_by_part=_top_knob_vertices(),
    )

    assert result.errors
    assert "SDK selected action +Z" in result.errors[0]
    assert result.traces[0]["selected_axis_label"] == "+Z"


def test_validate_continuous_motion_rejects_rotary_axis_self_certified_by_recommendation():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_00": {
                "type": "revolute",
                "axis_world": [0.0, 1.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_00": {
                "labels": ["temperature knob"],
                "recommended_axis_label": "+Y",
                "recommended_axis_world": [0.0, 1.0, 0.0],
            },
        },
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_00", 0.0, 0.5, axis_world=[0.0, 1.0, 0.0], axis_label="+Y")],
        backend=FakeBackend(),
        raw_vertices_by_part=_top_knob_vertices(),
    )

    assert result.errors
    assert "recommended axis +Y" in result.errors[0]
    assert result.traces[0]["axis_validation"]["target_axis_label"] == "+Z"


def test_validate_continuous_motion_uses_relation_recommended_rotary_axis():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_00": {
                "type": "revolute",
                "axis_world": [0.0, 0.0, 1.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_00": {
                "labels": ["temperature knob"],
                "recommended_axis_world": [0.9396926207859084, 0.0, 0.3420201433256687],
            },
        },
    )

    wrong = validate_motion_search(
        ctx,
        [LimitEstimate("part_00", 0.0, 0.5, axis_world=[0.0, 0.0, 1.0], axis_label="+Z")],
        backend=FakeBackend(),
        raw_vertices_by_part=_slanted_top_knob_vertices(20.0),
    )

    assert wrong.errors
    assert "recommended axis +X" in wrong.errors[0]
    target = wrong.traces[0]["axis_validation"]["target_axis_world"]

    correct = validate_motion_search(
        ctx,
        [LimitEstimate("part_00", 0.0, 0.5, axis_world=target)],
        backend=FakeBackend(),
        raw_vertices_by_part=_slanted_top_knob_vertices(20.0),
    )

    assert correct.errors == []


def test_validate_continuous_motion_does_not_invent_revolute_axis_without_relation():
    ctx = EstimateContext(
        object_id="ra_036",
        joints={
            "part_03": {
                "type": "revolute",
                "axis_world": [0.0, 0.0, 1.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_03"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_03": {"labels": []},
        },
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_03", 0.0, 0.5, axis_world=[0.0, 0.0, 1.0], axis_label="+Z")],
        backend=FakeBackend(),
        raw_vertices_by_part={
            "part_03": _slanted_top_knob_vertices(75.0)["part_00"],
            "body": _slanted_top_knob_vertices(75.0)["body"],
        },
    )

    assert result.errors == []
    assert "axis_validation" not in result.traces[0]


def test_validate_continuous_motion_rejects_rotary_axis_outside_five_degree_action_tolerance():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_00": {
                "type": "revolute",
                "axis_world": [0.0, 0.0, 1.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "__action_space__": {"axis_mode": "continuous"},
            "part_00": {"labels": ["temperature knob"]},
        },
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_00", 0.0, 0.5, axis_world=[0.0, 0.0, 1.0], axis_label="+Z")],
        backend=FakeBackend(),
        raw_vertices_by_part=_slanted_top_knob_vertices(8.0),
    )

    assert result.errors == []
    assert "axis_validation" not in result.traces[0]


def test_validate_motion_search_rejects_drawer_range_beyond_sdk_search_limit():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0.0, 1.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_02", 0.0, 0.40, axis_world=[0.0, -1.0, 0.0], axis_label="-Y")],
        backend=FakeDrawerClearanceBackend(),
        raw_vertices_by_part=_drawer_clearance_vertices(),
    )

    assert result.errors
    assert "exceeds SDK searched limit" in result.errors[0]


def test_validate_motion_search_tie_breaks_revolute_with_authored_axis_not_pca_candidate():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_00": {
                "type": "revolute",
                "axis_world": [0.0, 1.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            },
        },
        evidence={
            "part_00": {
                "labels": ["temperature knob"],
                "recommended_axis_label": "-Z",
                "recommended_axis_world": [0.0, 0.0, -1.0],
            }
        },
    )

    result = validate_motion_search(
        ctx,
        [LimitEstimate("part_00", 0.0, 1.0, axis_world=[0.0, 1.0, 0.0], axis_label="+Y")],
        backend=FakeBackend(),
    )

    assert result.errors == []
    assert result.traces[0]["selected_axis_label"] == "+Y"
    assert "authored" in result.traces[0]["selection_reason"]
