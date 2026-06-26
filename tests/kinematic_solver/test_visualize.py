import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image, UnidentifiedImageError

from post_process.kinematic_solver.utils.visualize import (
    _write_gif,
    render_final_predicted,
    visualize_one_joint,
    write_visualization_manifest,
)


def _write_png(path: Path) -> None:
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(path)


def test_visualize_writes_manifest_with_per_direction_status(tmp_path):
    manifest_path = tmp_path / "visualization_manifest.json"

    write_visualization_manifest(
        manifest_path=manifest_path,
        object_id="ra_007",
        joint_name="joint_part_00",
        upper_status="ok",
        upper_gif=tmp_path / "gif_upper.gif",
        upper_final=tmp_path / "final_predicted_upper.png",
        lower_status="skipped_non_ok",
        lower_gif=None,
        lower_final=None,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["object_id"] == "ra_007"
    assert manifest["joint_name"] == "joint_part_00"
    assert manifest["upper"] == {
        "status": "ok",
        "gif": "gif_upper.gif",
        "final_png": "final_predicted_upper.png",
    }
    assert manifest["lower"] == {
        "status": "skipped_non_ok",
        "gif": None,
        "final_png": None,
    }


def test_render_final_predicted_rejects_null_q():
    with pytest.raises(ValueError):
        render_final_predicted(
            backend=MagicMock(),
            joint={
                "type": "prismatic",
                "axis_world": [1, 0, 0],
                "origin_world": [0, 0, 0],
                "moving_parts": ["part_00"],
            },
            q_signed=None,
            out_path=Path("/tmp/final.png"),
        )


def test_render_final_predicted_uses_external_renderer(tmp_path, monkeypatch):
    class PoseOnlyBackend:
        def __init__(self):
            self.poses = []

        def set_pose(self, part_name, rotation, translation):
            self.poses.append((part_name, rotation, translation))

    backend = PoseOnlyBackend()
    rendered = []

    def fake_render_backend_frame(render_backend, out_path):
        rendered.append((render_backend, out_path))
        Path(out_path).write_bytes(b"png")

    monkeypatch.setattr(
        "post_process.kinematic_solver.utils.visualize.render_backend_frame",
        fake_render_backend_frame,
        raising=False,
    )

    render_final_predicted(
        backend=backend,
        joint={
            "type": "prismatic",
            "axis_world": [1, 0, 0],
            "origin_world": [0, 0, 0],
            "moving_parts": ["part_00"],
        },
        q_signed=0.1,
        out_path=tmp_path / "final.png",
    )

    assert rendered == [(backend, tmp_path / "final.png")]
    assert backend.poses[0][0] == "part_00"


def test_visualize_one_joint_writes_final_pngs_and_manifest(tmp_path):
    backend = MagicMock()
    backend.iter_render_hulls.return_value = []
    joint = {
        "type": "prismatic",
        "axis_world": [1, 0, 0],
        "origin_world": [0, 0, 0],
        "moving_parts": ["part_00"],
    }
    prediction = {
        "object_id": "ra_007",
        "joint_name": "joint0",
        "status_upper": "ok",
        "status_lower": "ok",
        "predicted_upper": 0.1,
        "predicted_lower": -0.1,
    }

    visualize_one_joint(
        backend=backend,
        joint=joint,
        prediction=prediction,
        trace={
            "trace_upper": [{"q": 0.0, "valid": True}, {"q": 0.1, "valid": True}],
            "trace_lower": [{"q": 0.0, "valid": True}, {"q": -0.1, "valid": True}],
        },
        out_dir=tmp_path,
        viz_stride=1,
        render_frames=True,
    )

    assert (tmp_path / "final_predicted_upper.png").is_file()
    assert (tmp_path / "final_predicted_lower.png").is_file()
    manifest = json.loads((tmp_path / "visualization_manifest.json").read_text())
    assert manifest["upper"]["status"] == "ok"
    assert manifest["lower"]["status"] == "ok"
    step_manifest = json.loads((tmp_path / "step_manifest.json").read_text())
    assert step_manifest["object_id"] == "ra_007"
    assert step_manifest["joint_name"] == "joint0"
    assert [step["side"] for step in step_manifest["steps"]] == [
        "final_range", "final_range", "final_range",
    ]
    assert [step["q"] for step in step_manifest["steps"]] == [-0.1, 0.0, 0.1]
    assert step_manifest["steps"][0]["frame"] == "frame_final_range_000.png"
    assert step_manifest["raw_scan_evidence"]["directions"]["lower"]["last_valid_q"] == -0.1
    html = (tmp_path / "step_viewer.html").read_text()
    assert "KinematicSolver Step Viewer" in html
    assert "TURN" in html
    assert "frame_final_range_000.png" in html


def test_visualize_one_joint_can_write_fast_geometry_step_viewer(tmp_path):
    class GeometryBackend:
        def __init__(self):
            self.render_calls = 0
            self.poses = {}

        def iter_render_hulls(self):
            from post_process.kinematic_solver.utils.render import RenderHull
            import numpy as np

            yield RenderHull(
                part_name="body",
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int32),
                rotation=np.eye(3),
                translation=np.zeros(3),
            )
            yield RenderHull(
                part_name="part_00",
                vertices=np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int32),
                rotation=np.eye(3),
                translation=np.zeros(3),
            )

        def set_pose(self, part_name, rotation, translation):
            self.poses[part_name] = (rotation, translation)

    backend = GeometryBackend()

    visualize_one_joint(
        backend=backend,
        joint={
            "type": "prismatic",
            "axis_world": [1, 0, 0],
            "origin_world": [0, 0, 0],
            "moving_parts": ["part_00"],
        },
        prediction={
            "object_id": "ra_007",
            "joint_name": "joint0",
            "status_upper": "ok",
            "status_lower": "ok",
            "predicted_upper": 0.1,
            "predicted_lower": -0.1,
        },
        trace={
            "trace_upper": [{"q": 0.0, "valid": True}, {"q": 0.1, "valid": True}],
            "trace_lower": [{"q": 0.0, "valid": True}, {"q": -0.1, "valid": True}],
        },
        out_dir=tmp_path,
        viz_stride=1,
        render_frames=False,
    )

    assert not (tmp_path / "frame_upper_000.png").exists()
    assert not (tmp_path / "gif_upper.gif").exists()
    step_manifest = json.loads((tmp_path / "step_manifest.json").read_text())
    assert len(step_manifest["render_hulls"]) == 2
    assert [step["q"] for step in step_manifest["steps"]] == [-0.1, 0.0, 0.1]
    assert step_manifest["steps"][2]["translation"] == [0.1, 0.0, 0.0]
    html = (tmp_path / "step_viewer.html").read_text()
    assert "THREE.PerspectiveCamera" in html
    assert "render_hulls" in html


def test_step_viewer_uses_final_range_not_clamped_raw_negative_trace(tmp_path):
    backend = MagicMock()
    backend.iter_render_hulls.return_value = []

    visualize_one_joint(
        backend=backend,
        joint={
            "type": "prismatic",
            "axis_world": [1, 0, 0],
            "origin_world": [0, 0, 0],
            "moving_parts": ["part_00"],
        },
        prediction={
            "object_id": "ra_063",
            "joint_name": "part_02",
            "status_upper": "ok",
            "status_lower": "ok",
            "predicted_upper": 0.15,
            "predicted_lower": 0.0,
            "motion_direction_prior": {
                "policy": "positive_only",
                "raw_predicted_lower": -0.15,
                "raw_predicted_upper": 0.15,
            },
        },
        trace={
            "trace_upper": [
                {"q": 0.0, "valid": True},
                {"q": 0.01, "valid": True},
                {"q": 0.15, "valid": True},
                {"q": 0.16, "valid": False},
            ],
            "trace_lower": [
                {"q": 0.0, "valid": True},
                {"q": -0.01, "valid": True},
                {"q": -0.15, "valid": True},
                {"q": -0.16, "valid": False},
            ],
        },
        out_dir=tmp_path,
        viz_stride=1,
        render_frames=False,
    )

    step_manifest = json.loads((tmp_path / "step_manifest.json").read_text())
    assert [step["q"] for step in step_manifest["steps"]] == [0.0, 0.01, 0.15]
    assert {step["side"] for step in step_manifest["steps"]} == {"final_range"}
    assert all(step["valid"] is True for step in step_manifest["steps"])
    evidence = step_manifest["raw_scan_evidence"]
    assert evidence["final_range"] == {"lower": 0.0, "upper": 0.15}
    assert evidence["directions"]["lower"]["last_valid_q"] == -0.15
    assert evidence["directions"]["lower"]["first_invalid_q"] == -0.16
    assert evidence["direction_prior"]["policy"] == "positive_only"


def test_visualize_one_joint_raises_when_ok_side_has_no_frames(tmp_path):
    backend = MagicMock()
    backend.iter_render_hulls.return_value = []

    with pytest.raises(RuntimeError, match="empty frames"):
        visualize_one_joint(
            backend=backend,
            joint={
                "type": "prismatic",
                "axis_world": [1, 0, 0],
                "origin_world": [0, 0, 0],
                "moving_parts": ["part_00"],
            },
            prediction={
                "object_id": "ra_007",
                "joint_name": "joint0",
                "status_upper": "ok",
                "status_lower": "initial_collision",
                "predicted_upper": 0.1,
                "predicted_lower": None,
            },
            trace={"trace_upper": [], "trace_lower": []},
            out_dir=tmp_path,
            viz_stride=1,
            render_frames=True,
        )


def test_visualize_one_joint_renders_non_ok_trace_for_debug_viewer(tmp_path):
    backend = MagicMock()
    backend.iter_render_hulls.return_value = []

    visualize_one_joint(
        backend=backend,
        joint={
            "type": "prismatic",
            "axis_world": [1, 0, 0],
            "origin_world": [0, 0, 0],
            "moving_parts": ["part_00"],
        },
        prediction={
            "object_id": "ra_063",
            "joint_name": "part_02",
            "status_upper": "initial_collision",
            "status_lower": "initial_collision",
            "predicted_upper": None,
            "predicted_lower": None,
        },
        trace={
            "trace_upper": [{"q": 0.0, "valid": False}],
            "trace_lower": [{"q": 0.0, "valid": False}],
        },
        out_dir=tmp_path,
        viz_stride=1,
        render_frames=True,
    )

    step_manifest = json.loads((tmp_path / "step_manifest.json").read_text())
    assert [step["side"] for step in step_manifest["steps"]] == ["upper", "lower"]
    assert step_manifest["steps"][0]["valid"] is False
    assert (tmp_path / "frame_upper_000.png").is_file()
    assert (tmp_path / "frame_lower_000.png").is_file()
    manifest = json.loads((tmp_path / "visualization_manifest.json").read_text())
    assert manifest["upper"]["gif"] is None
    assert manifest["lower"]["final_png"] is None


def test_write_gif_propagates_invalid_frame_error(tmp_path):
    bad_frame = tmp_path / "bad.png"
    bad_frame.write_bytes(b"not a png")

    with pytest.raises(UnidentifiedImageError):
        _write_gif(tmp_path / "out.gif", [bad_frame])
