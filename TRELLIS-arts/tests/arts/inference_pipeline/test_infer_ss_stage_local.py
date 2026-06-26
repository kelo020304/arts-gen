import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from inference_pipeline import ss_stage_local


def test_mode_a_writes(tmp_path, monkeypatch):
    fake = {"z_global": torch.zeros(8, 16, 16, 16),
            "raw_surface_coords": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)}
    monkeypatch.setattr(ss_stage_local, "load_object_inputs", lambda *a, **k: fake)
    out = tmp_path / "run"
    r = ss_stage_local.run_mode_a({"data_root": "/x"}, object_id="o", angle_idx=0, view_mode="four", out_dir=out)
    z = np.load(out / "ss_latent.npy"); assert z.shape == (8, 16, 16, 16)
    v = np.load(out / "voxel.npz"); assert v["coords"].shape == (2, 3) and str(v["source"]) == "gt"
    assert (out / "voxel.bin").stat().st_size == 2 * 3 * 2
