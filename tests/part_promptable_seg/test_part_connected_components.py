from __future__ import annotations

import numpy as np

from scripts.eval.post.part_connected_components import filter_part_connected_components


def _component(y: int, count: int) -> np.ndarray:
    return np.asarray([[x, y, 0] for x in range(count)], dtype=np.int32)


def _filter(coords: np.ndarray, *, max_large_distance: int | None):
    return filter_part_connected_components(
        coords,
        part_index=0,
        part_name="door_0",
        min_component_voxels=5,
        min_component_fraction=0.2,
        max_component_distance=2,
        max_large_component_distance=max_large_distance,
    )


def test_default_keeps_existing_large_or_near_behavior() -> None:
    main = _component(0, 20)
    large_remote = _component(10, 10)
    small_remote = _component(20, 2)

    filtered, record = _filter(np.concatenate([main, large_remote, small_remote]), max_large_distance=None)

    assert filtered.shape[0] == 30
    assert record["reassigned_to_body_voxels"] == 2
    assert record["thresholds"]["max_large_component_distance"] is None
    assert record["components"][1]["reason"] == "large_within_distance_limit"


def test_hard_distance_limit_removes_large_remote_component() -> None:
    main = _component(0, 20)
    large_remote = _component(10, 10)

    filtered, record = _filter(np.concatenate([main, large_remote]), max_large_distance=4)

    assert filtered.shape[0] == 20
    assert record["reassigned_to_body_voxels"] == 10
    assert record["components"][1]["bbox_gap_to_largest"] == 9
    assert record["components"][1]["reason"] == "large_but_remote_reassigned_to_body"


def test_hard_distance_limit_still_keeps_near_and_large_within_limit() -> None:
    main = _component(0, 20)
    near_small = _component(3, 1)
    large_within_limit = _component(5, 5)

    filtered, record = _filter(
        np.concatenate([main, near_small, large_within_limit]),
        max_large_distance=4,
    )

    assert filtered.shape[0] == 26
    reasons = {item["bbox_gap_to_largest"]: item["reason"] for item in record["components"][1:]}
    assert reasons[2] == "near"
    assert reasons[4] == "large_within_distance_limit"
