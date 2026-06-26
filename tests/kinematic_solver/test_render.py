from pathlib import Path

import numpy as np

from post_process.kinematic_solver.utils.render import RenderHull, render_backend_frame


class FakeRenderBackend:
    def iter_render_hulls(self):
        yield RenderHull(
            part_name="body",
            vertices=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            rotation=np.eye(3),
            translation=np.zeros(3),
        )


def test_render_backend_frame_writes_png_from_render_hulls(tmp_path):
    out_path = tmp_path / "frame.png"

    render_backend_frame(FakeRenderBackend(), out_path)

    assert out_path.is_file()
    assert out_path.read_bytes().startswith(b"\x89PNG")


def test_render_backend_frame_rejects_backend_without_render_hulls(tmp_path):
    class NotRenderable:
        pass

    try:
        render_backend_frame(NotRenderable(), Path(tmp_path / "frame.png"))
    except TypeError as exc:
        assert "iter_render_hulls" in str(exc)
    else:
        raise AssertionError("expected TypeError")
