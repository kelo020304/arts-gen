from post_process.kinematic_solver.sdk.object_visualization import write_estimate_viewers
from post_process.kinematic_solver.sdk.schemas import EstimateContext, LimitEstimate


class FakeBackend:
    def __init__(self):
        self.loaded = None
        self.reset_count = 0
        self.cleared = False

    def load_model(self, **kwargs):
        self.loaded = kwargs

    def reset_to_identity(self):
        self.reset_count += 1

    def clear(self):
        self.cleared = True


def test_write_estimate_viewers_uses_candidate_limits_and_relative_links(tmp_path, monkeypatch):
    converter_root = tmp_path / "converter"
    obj_dir = converter_root / "raw/partseg/ra_063/objs"
    obj_dir.mkdir(parents=True)
    (obj_dir / "body.obj").write_text("")
    (obj_dir / "part_02.obj").write_text("")
    (converter_root / "raw/vhacd/ra_063").mkdir(parents=True)
    backend = FakeBackend()
    calls = []

    monkeypatch.setattr(
        "post_process.kinematic_solver.sdk.object_visualization.make_backend",
        lambda spike_result=None: backend,
    )

    def fake_visualize_one_joint(**kwargs):
        calls.append(kwargs)
        kwargs["out_dir"].mkdir(parents=True)
        (kwargs["out_dir"] / "step_viewer.html").write_text("viewer")

    monkeypatch.setattr(
        "post_process.kinematic_solver.sdk.object_visualization.visualize_one_joint",
        fake_visualize_one_joint,
    )
    monkeypatch.setattr(
        "post_process.kinematic_solver.sdk.object_visualization._copy_viewer_vendor",
        lambda out_dir: None,
    )
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0, 1, 0],
                "origin_world": [0, 0, 0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            }
        },
        evidence={},
    )

    details = write_estimate_viewers(
        ctx,
        [LimitEstimate(joint_name="part_02", lower=0.0, upper=0.15)],
        converter_output_root=converter_root,
        out_dir=tmp_path / "out",
    )

    assert backend.loaded["object_id"] == "ra_063"
    assert backend.reset_count == 1
    assert backend.cleared is True
    assert details["object_viewers"] == [
        {
            "joint_name": "part_02",
            "href": "ra_063/agent_viz/part_02/step_viewer.html",
            "lower": 0.0,
            "upper": 0.15,
        }
    ]
    assert calls[0]["prediction"]["predicted_lower"] == 0.0
    assert calls[0]["prediction"]["predicted_upper"] == 0.15
    assert calls[0]["prediction"]["predicted_axis_world"] is None
    assert calls[0]["trace"]["trace_upper"][0]["q"] == 0.0
    assert calls[0]["trace"]["trace_upper"][-1]["q"] == 0.15


def test_write_estimate_viewers_prefers_estimate_axis_world_over_oracle_axis(tmp_path, monkeypatch):
    converter_root = tmp_path / "converter"
    obj_dir = converter_root / "raw/partseg/ra_063/objs"
    obj_dir.mkdir(parents=True)
    (obj_dir / "body.obj").write_text("")
    (obj_dir / "part_02.obj").write_text("")
    (converter_root / "raw/vhacd/ra_063").mkdir(parents=True)
    backend = FakeBackend()
    calls = []

    monkeypatch.setattr(
        "post_process.kinematic_solver.sdk.object_visualization.make_backend",
        lambda spike_result=None: backend,
    )
    monkeypatch.setattr(
        "post_process.kinematic_solver.sdk.object_visualization.visualize_one_joint",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "post_process.kinematic_solver.sdk.object_visualization._copy_viewer_vendor",
        lambda out_dir: None,
    )
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0, 1, 0],
                "origin_world": [0, 0, 0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            }
        },
        evidence={},
    )

    details = write_estimate_viewers(
        ctx,
        [
            LimitEstimate(
                joint_name="part_02",
                lower=0.0,
                upper=0.15,
                axis_world=[0.0, -1.0, 0.0],
                axis_label="-Y",
            )
        ],
        converter_output_root=converter_root,
        out_dir=tmp_path / "out",
    )

    assert calls[0]["joint"]["axis_world"] == [0.0, -1.0, 0.0]
    assert calls[0]["prediction"]["predicted_axis_world"] == [0.0, -1.0, 0.0]
    assert details["object_viewers"][0]["axis_label"] == "-Y"
