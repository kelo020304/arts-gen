import numpy as np
import trimesh
import pytest
from inference_pipeline.assemble import assemble_complete

def _make_box_glb(path):
    trimesh.creation.box(extents=(1, 1, 1)).export(str(path))

def _make_ply(path, n, layout=("x", "y", "z")):
    from plyfile import PlyData, PlyElement
    dt = [(name, "f4") for name in layout]
    verts = np.zeros(n, dtype=dt)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))

def test_assemble_meshes_into_named_scene(tmp_path):
    parts = tmp_path / "parts"; parts.mkdir()
    _make_box_glb(parts / "part_00.glb")
    _make_box_glb(parts / "part_01.glb")
    out = assemble_complete(tmp_path, part_mesh_names=["part_00.glb", "part_01.glb"], part_gaussian_names=[])
    assert out["complete_mesh"] == "complete.glb"
    scene = trimesh.load(str(tmp_path / "complete.glb"))
    assert hasattr(scene, "geometry") and len(scene.geometry) == 2   # two named part geometries

def test_assemble_concats_gaussian_ply(tmp_path):
    parts = tmp_path / "parts"; parts.mkdir()
    _make_ply(parts / "part_00.ply", 3)
    _make_ply(parts / "part_01.ply", 5)
    out = assemble_complete(tmp_path, part_mesh_names=[], part_gaussian_names=["part_00.ply", "part_01.ply"])
    assert out["complete_gaussian"] == "complete.ply"
    from plyfile import PlyData
    merged = PlyData.read(str(tmp_path / "complete.ply"))
    assert merged["vertex"].count == 8   # 3 + 5

def test_assemble_rejects_mismatched_ply_layout(tmp_path):
    parts = tmp_path / "parts"; parts.mkdir()
    _make_ply(parts / "a.ply", 2, layout=("x", "y", "z"))
    _make_ply(parts / "b.ply", 2, layout=("x", "y"))      # different property layout
    with pytest.raises(ValueError):
        assemble_complete(tmp_path, part_mesh_names=[], part_gaussian_names=["a.ply", "b.ply"])
