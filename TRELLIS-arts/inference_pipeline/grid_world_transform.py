from __future__ import annotations
from pathlib import Path
import json
import numpy as np

Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def build_grid_to_world(*, resolution: int, scale: float, offset, obj_up_axis: str) -> np.ndarray:
    """4x4: voxel grid index([0,R)) -> world meters.
    grid->norm([-0.5,0.5]) -> norm->world((n-offset)/scale) -> optional Y->Z up rotation."""
    off = np.asarray(offset, dtype=np.float64).reshape(3)
    inv = 1.0 / float(resolution)
    m_grid_norm = np.array([
        [inv, 0,   0,   0.5 * inv - 0.5],
        [0,   inv, 0,   0.5 * inv - 0.5],
        [0,   0,   inv, 0.5 * inv - 0.5],
        [0,   0,   0,   1.0],
    ], dtype=np.float64)
    s = float(scale)
    m_norm_world = np.array([
        [1 / s, 0,     0,     -off[0] / s],
        [0,     1 / s, 0,     -off[1] / s],
        [0,     0,     1 / s, -off[2] / s],
        [0,     0,     0,     1.0],
    ], dtype=np.float64)
    M = m_norm_world @ m_grid_norm
    axis = str(obj_up_axis).upper()
    if axis == "Y":
        M = Y_UP_TO_Z_UP @ M
    elif axis != "Z":
        raise ValueError(f"obj_up_axis must be 'Y' or 'Z', got {obj_up_axis!r}")
    return M


def load_scale_offset(camera_transforms_path) -> tuple[float, list[float]]:
    payload = json.loads(Path(camera_transforms_path).read_text(encoding="utf-8"))
    if "scale" not in payload or "offset" not in payload:
        raise KeyError(f"camera_transforms 缺 scale/offset: {camera_transforms_path}")
    return float(payload["scale"]), [float(v) for v in payload["offset"]]
