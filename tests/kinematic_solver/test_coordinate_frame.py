import numpy as np

from post_process.kinematic_solver.sdk.coordinate_frame import (
    CANONICAL_FRAME_EVIDENCE,
    source_to_canonical_points,
    source_to_canonical_vector,
    with_canonical_coordinate_frame,
)
from post_process.kinematic_solver.sdk.schemas import EstimateContext


def test_source_asset_frame_maps_to_ros_front_left_up_axes():
    assert source_to_canonical_vector([0.0, -1.0, 0.0]) == (1.0, 0.0, 0.0)
    assert source_to_canonical_vector([1.0, 0.0, 0.0]) == (0.0, -1.0, 0.0)
    assert source_to_canonical_vector([0.0, 0.0, 1.0]) == (0.0, 0.0, 1.0)


def test_context_axes_and_origins_are_exposed_in_ros_frame():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0.0, -1.0, 0.0],
                "origin_world": [0.2, -0.3, 0.4],
            }
        },
        evidence={"__part_centers__": {"part_02": [0.2, -0.3, 0.4]}},
    )

    transformed = with_canonical_coordinate_frame(ctx)

    assert transformed.evidence["__coordinate_frame__"] == CANONICAL_FRAME_EVIDENCE
    assert transformed.joints["part_02"]["axis_world"] == [1.0, 0.0, 0.0]
    assert transformed.joints["part_02"]["origin_world"] == [0.3, -0.2, 0.4]
    assert transformed.evidence["__part_centers__"]["part_02"] == [0.3, -0.2, 0.4]


def test_source_points_transform_to_ros_frame():
    points = np.asarray([[1.0, -2.0, 3.0]], dtype=float)

    transformed = source_to_canonical_points(points)

    np.testing.assert_allclose(transformed, [[2.0, -1.0, 3.0]])
