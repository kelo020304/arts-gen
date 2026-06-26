from pathlib import Path
from unittest.mock import MagicMock

from post_process.kinematic_solver.utils.comparison_visualize import write_per_direction_overlays


def test_comparison_visualize_writes_only_ok_directions(tmp_path):
    backend = MagicMock()
    backend.iter_render_hulls.return_value = []

    written = write_per_direction_overlays(
        backend=backend,
        joint={
            "type": "prismatic",
            "axis_world": [1, 0, 0],
            "origin_world": [0, 0, 0],
            "moving_parts": ["part_00"],
        },
        prediction={
            "status_upper": "ok",
            "predicted_upper": 0.1,
            "status_lower": "initial_collision",
            "predicted_lower": None,
            "type": "prismatic",
        },
        gt={"lower": -0.1, "upper": 0.1},
        out_dir=tmp_path,
    )

    assert tmp_path / "pred_upper.png" in written
    assert tmp_path / "gt_upper.png" in written
    assert (tmp_path / "pred_upper.png").is_file()
    assert (tmp_path / "gt_upper.png").is_file()
    assert tmp_path / "pred_lower.png" not in written
    assert tmp_path / "gt_lower.png" not in written
