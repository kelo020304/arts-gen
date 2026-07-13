from __future__ import annotations

import numpy as np

from scripts.inference.reconstruct import CkptConfig, _apply_part_cc_filter


def _component(y: int, count: int) -> np.ndarray:
    return np.asarray([[x, y, 0] for x in range(count)], dtype=np.int32)


def _coord_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(value) for value in row) for row in coords.tolist()}


def test_workbench_defaults_enable_strict_cc_filter_and_joint_refine() -> None:
    cfg = CkptConfig()

    assert cfg.part_joint_refine is True
    assert cfg.part_cc_filter is True
    assert cfg.part_cc_max_large_component_distance == 4
    assert str(cfg.part_seg_ckpt).endswith(
        "part-prompt-seg-L-0709-1-joint/ckpts/step_100000.pt"
    )


def test_cc_filter_reassigns_remote_part_component_to_recomputed_body() -> None:
    main = _component(0, 20)
    remote = _component(10, 10)
    fixed_body = np.asarray([[40, 40, 40]], dtype=np.int32)
    whole = np.concatenate([main, remote, fixed_body], axis=0)
    part_coords = {
        7: np.concatenate([main, remote], axis=0),
        -1: fixed_body.copy(),
    }

    records = _apply_part_cc_filter(
        part_coords,
        whole_coords=whole,
        part_ids=[7],
        part_names={7: "door"},
        min_component_voxels=5,
        min_component_fraction=0.2,
        max_component_distance=2,
        max_large_component_distance=4,
    )

    assert _coord_set(part_coords[7]) == _coord_set(main)
    assert _coord_set(part_coords[-1]) == _coord_set(np.concatenate([remote, fixed_body]))
    assert records[0]["part_id"] == 7
    assert records[0]["reassigned_to_body_voxels"] == 10

