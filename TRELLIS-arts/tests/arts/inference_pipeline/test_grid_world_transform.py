import json
import numpy as np
import pytest
from inference_pipeline.grid_world_transform import build_grid_to_world, load_scale_offset

def test_grid_origin_maps_to_norm_center():
    M = build_grid_to_world(resolution=64, scale=1.0, offset=[0, 0, 0], obj_up_axis="Z")
    w = M @ np.array([0, 0, 0, 1.0])
    c = (0 + 0.5) / 64 - 0.5
    np.testing.assert_allclose(w[:3], [c, c, c], atol=1e-9)

def test_scale_offset_inverse():
    # normalized = world*scale + offset  ->  world = (norm - offset)/scale
    M = build_grid_to_world(resolution=64, scale=2.0, offset=[0.1, 0.0, -0.2], obj_up_axis="Z")
    g = np.array([10, 20, 30, 1.0])
    n = (g[:3] + 0.5) / 64 - 0.5
    expect_world = (n - np.array([0.1, 0.0, -0.2])) / 2.0
    np.testing.assert_allclose((M @ g)[:3], expect_world, atol=1e-9)

def test_yup_rotates_axes():
    M = build_grid_to_world(resolution=64, scale=1.0, offset=[0, 0, 0], obj_up_axis="Y")
    ny = (10 + 0.5) / 64 - 0.5
    nz = (20 + 0.5) / 64 - 0.5
    w = M @ np.array([0, 10, 20, 1.0])
    np.testing.assert_allclose([w[1], w[2]], [-nz, ny], atol=1e-9)  # Y->Z up: (.,ny,nz)->(.,-nz,ny)

def test_bad_up_axis_raises():
    with pytest.raises(ValueError):
        build_grid_to_world(resolution=64, scale=1.0, offset=[0, 0, 0], obj_up_axis="X")

def test_load_scale_offset(tmp_path):
    p = tmp_path / "camera_transforms.json"
    p.write_text(json.dumps({"scale": 2.5, "offset": [1, 2, 3], "frames": []}))
    s, off = load_scale_offset(p)
    assert s == 2.5 and off == [1.0, 2.0, 3.0]

def test_load_scale_offset_missing_raises(tmp_path):
    p = tmp_path / "camera_transforms.json"
    p.write_text(json.dumps({"frames": []}))
    with pytest.raises(KeyError):
        load_scale_offset(p)
