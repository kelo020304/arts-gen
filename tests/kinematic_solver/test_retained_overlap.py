import numpy as np
import pytest

from post_process.kinematic_solver.utils.constraints import (
    Body0GapRangeConstraint,
    RetainedOverlapConstraint,
    make_body0_gap_range_constraint,
)
from post_process.kinematic_solver.utils.errors import DegenerateAxisExtentError


def test_retained_overlap_q0_ratio_is_1():
    constraint = RetainedOverlapConstraint(
        joint={"joint_name": "j", "axis_world": [1, 0, 0], "moving_parts": ["part"]},
        raw_vertices_by_part={"part": np.array([[0, 0, 0], [2, 0, 0]], dtype=float)},
        min_retained_ratio=0.2,
    )

    constraint.set_current_q(0.0)

    assert constraint.check() is True


def test_retained_overlap_fails_when_shift_loses_too_much_projection_interval():
    constraint = RetainedOverlapConstraint(
        joint={"joint_name": "j", "axis_world": [1, 0, 0], "moving_parts": ["part"]},
        raw_vertices_by_part={"part": np.array([[0, 0, 0], [1, 0, 0]], dtype=float)},
        min_retained_ratio=0.2,
    )

    constraint.set_current_q(0.5)
    assert constraint.check() is True
    constraint.set_current_q(0.85)
    assert constraint.check() is False


def test_retained_overlap_degenerate_axis_extent_fails_loud():
    with pytest.raises(DegenerateAxisExtentError):
        RetainedOverlapConstraint(
            joint={"joint_name": "j", "axis_world": [1, 0, 0], "moving_parts": ["part"]},
            raw_vertices_by_part={"part": np.array([[0, 0, 0], [0, 1, 0]], dtype=float)},
            min_retained_ratio=0.2,
        )


def test_body0_gap_range_positive_side_allows_only_outward_gap():
    constraint = Body0GapRangeConstraint(
        joint={"joint_name": "j", "axis_world": [0, 0, 1]},
        parent_vertices=np.array([[0, 0, 0], [0, 0, 1.0]], dtype=float),
        moving_vertices=np.array([[0, 0, 1.12], [0, 0, 1.3]], dtype=float),
    )

    constraint.set_current_q(0.0)
    assert constraint.check() is True
    constraint.set_current_q(0.11)
    assert constraint.check() is True
    constraint.set_current_q(0.13)
    assert constraint.check() is False
    constraint.set_current_q(-0.01)
    assert constraint.check() is False


def test_body0_gap_range_negative_side_allows_only_outward_gap():
    constraint = Body0GapRangeConstraint(
        joint={"joint_name": "j", "axis_world": [0, 0, 1]},
        parent_vertices=np.array([[0, 0, 0], [0, 0, 1.0]], dtype=float),
        moving_vertices=np.array([[0, 0, -0.3], [0, 0, -0.12]], dtype=float),
    )

    constraint.set_current_q(0.0)
    assert constraint.check() is True
    constraint.set_current_q(-0.11)
    assert constraint.check() is True
    constraint.set_current_q(-0.13)
    assert constraint.check() is False
    constraint.set_current_q(0.01)
    assert constraint.check() is False


def test_body0_gap_range_factory_skips_body_or_overlapping_parent():
    assert make_body0_gap_range_constraint(
        joint={
            "joint_name": "root",
            "type": "prismatic",
            "axis_world": [0, 0, 1],
            "moving_parts": ["part_00"],
            "body0_link_name": "body",
        },
        raw_vertices_by_part={
            "body": np.array([[0, 0, 0], [0, 0, 1.0]], dtype=float),
            "part_00": np.array([[0, 0, 1.12], [0, 0, 1.3]], dtype=float),
        },
    ) is None

    assert make_body0_gap_range_constraint(
        joint={
            "joint_name": "overlap",
            "type": "prismatic",
            "axis_world": [0, 0, 1],
            "moving_parts": ["part_01"],
            "body0_link_name": "part_00",
        },
        raw_vertices_by_part={
            "part_00": np.array([[0, 0, 0], [0, 0, 1.0]], dtype=float),
            "part_01": np.array([[0, 0, 0.8], [0, 0, 1.3]], dtype=float),
        },
    ) is None
