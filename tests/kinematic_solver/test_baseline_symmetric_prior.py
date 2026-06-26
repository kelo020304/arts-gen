import math

import numpy as np
import pytest

from post_process.kinematic_solver.utils.baseline_symmetric_prior import symmetric_prior_predict


def test_symmetric_prior_prismatic_uses_axis_extent():
    moving_vertices = np.array([
        [-1.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    out = symmetric_prior_predict(
        joint_type="prismatic",
        axis_world=[1, 0, 0],
        moving_vertices=moving_vertices,
    )

    assert out["predicted_lower"] == pytest.approx(-1.0)
    assert out["predicted_upper"] == pytest.approx(1.0)
    assert out["status"] == "ok"


def test_symmetric_prior_revolute_uses_pi_over_2():
    out = symmetric_prior_predict(
        joint_type="revolute",
        axis_world=[0, 0, 1],
        moving_vertices=np.zeros((1, 3)),
    )

    assert out["predicted_lower"] == pytest.approx(-math.pi / 2)
    assert out["predicted_upper"] == pytest.approx(math.pi / 2)
