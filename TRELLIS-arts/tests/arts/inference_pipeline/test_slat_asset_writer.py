import numpy as np
from pathlib import Path
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets

class _FakeGaussian:
    def save_ply(self, path): Path(path).write_text("ply-stub")

class _FakeMesh:
    success = True
    vertices = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    vertex_attrs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

def test_save_decoded_slat_assets_writes_named_files(tmp_path):
    asset_dir = tmp_path / "parts" / "part_00"
    decoded = {"gaussian": _FakeGaussian(), "mesh": _FakeMesh()}
    rec = save_decoded_slat_assets(decoded, asset_dir, mesh_name="part_00.glb", gaussian_name="part_00.ply")
    assert (asset_dir / "part_00.ply").is_file()
    assert (asset_dir / "part_00.glb").is_file()
    assert rec == {"gaussian": "part_00.ply", "mesh": "part_00.glb"}

def test_save_decoded_slat_assets_preserves_vertex_colors(tmp_path):
    import trimesh

    save_decoded_slat_assets({"mesh": _FakeMesh()}, tmp_path, mesh_name="mesh.glb")
    mesh = trimesh.load(tmp_path / "mesh.glb", force="mesh")
    colors = np.asarray(mesh.visual.vertex_colors)
    assert colors.shape == (3, 4)
    assert colors[:, :3].tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

def test_save_decoded_slat_assets_exports_y_up_vertices(tmp_path):
    import trimesh

    save_decoded_slat_assets({"mesh": _FakeMesh()}, tmp_path, mesh_name="mesh.glb")
    mesh = trimesh.load(tmp_path / "mesh.glb", force="mesh")
    expected = np.array(
        [
            [1.0, 3.0, -2.0],
            [4.0, 6.0, -5.0],
            [7.0, 9.0, -8.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(mesh.vertices), expected)

def test_save_decoded_slat_assets_raises_on_bad_gaussian(tmp_path):
    import pytest
    with pytest.raises(TypeError):
        save_decoded_slat_assets({"gaussian": object()}, tmp_path)  # object() has no save_ply -> must raise, not swallow

def test_save_decoded_slat_assets_handles_missing_keys(tmp_path):
    rec = save_decoded_slat_assets({}, tmp_path)   # neither gaussian nor mesh -> empty record, no crash
    assert rec == {}
