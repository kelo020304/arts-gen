import numpy as np
import pytest
from inference_pipeline.voxel_io import save_voxel, load_voxel_bin

def test_voxel_roundtrip(tmp_path):
    coords = np.array([[0, 1, 2], [63, 0, 5]], dtype=np.int64)
    save_voxel(tmp_path, coords, resolution=64, source="pred")
    npz = np.load(tmp_path / "voxel.npz")
    assert npz["coords"].dtype == np.int32 and npz["coords"].tolist() == coords.tolist()
    assert int(npz["resolution"]) == 64
    assert str(npz["coord_frame"]) == "canonical_grid"
    assert str(npz["source"]) == "pred"
    flat = load_voxel_bin(tmp_path / "voxel.bin")
    assert flat.dtype == np.uint16 and flat.tolist() == coords.tolist()
    raw = (tmp_path / "voxel.bin").read_bytes()
    assert len(raw) == coords.shape[0] * 3 * 2          # uint16 = 2 bytes/value

def test_save_voxel_custom_basename(tmp_path):
    save_voxel(tmp_path, np.zeros((1, 3), dtype=np.int64), resolution=64, source="gt", basename="part_00_voxel")
    assert (tmp_path / "part_00_voxel.npz").is_file()
    assert (tmp_path / "part_00_voxel.bin").is_file()

def test_save_voxel_rejects_out_of_range(tmp_path):
    with pytest.raises(ValueError):
        save_voxel(tmp_path, np.array([[64, 0, 0]], dtype=np.int64), resolution=64, source="pred")  # 64 >= resolution
