import math

import numpy as np

from post_process.kinematic_solver.utils.manual_transform import apply_joint_transform_world_baked


def test_prismatic_transform_translates_along_signed_axis():
    rotation, translation = apply_joint_transform_world_baked(
        joint_type="prismatic",
        direction=-1,
        q_abs=0.25,
        axis_world=np.array([0.0, 0.0, 2.0]),
        origin_world=np.array([9.0, 9.0, 9.0]),
    )

    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, [0.0, 0.0, -0.25])


def test_revolute_transform_rotates_about_world_origin_point():
    rotation, translation = apply_joint_transform_world_baked(
        joint_type="revolute",
        direction=1,
        q_abs=math.pi / 2,
        axis_world=np.array([0.0, 0.0, 1.0]),
        origin_world=np.array([1.0, 0.0, 0.0]),
    )

    pivot = np.array([1.0, 0.0, 0.0])
    point = np.array([2.0, 0.0, 0.0])
    moved = rotation @ point + translation

    assert np.allclose(rotation @ pivot + translation, pivot)
    assert np.allclose(moved, [1.0, 1.0, 0.0], atol=1e-7)
