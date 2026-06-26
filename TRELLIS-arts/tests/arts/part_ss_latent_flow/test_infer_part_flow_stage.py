import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from inference_pipeline import part_flow_stage

def test_save_part_voxels_preserves_order(tmp_path):
    out = tmp_path/"run"
    part_coords = {"wheel_1": np.array([[7,8,9]], np.int64),       # 真实返回：name 键、[N,3] int64、无 batch 列
                   "wheel_0": np.array([[1,2,3],[4,5,6]], np.int64)}
    names = ["wheel_0", "wheel_1"]                                 # 目标原序
    written = part_flow_stage.save_part_voxels(out, part_coords, target_part_names=names, resolution=64)
    assert written == ["part_00_voxel.npz", "part_01_voxel.npz"]
    z0 = np.load(out/"parts"/"part_00_voxel.npz")
    assert z0["coords"].shape==(2,3) and str(z0["target_part_name"])=="wheel_0" and int(z0["part_index"])==0
    z1 = np.load(out/"parts"/"part_01_voxel.npz")
    assert str(z1["target_part_name"])=="wheel_1"


def test_run_defaults_to_trellis_decode_and_forwards_part_token_weights(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.save(run_dir / "ss_latent.npy", np.zeros((8, 16, 16, 16), np.float32))
    part_token_weights = torch.zeros(1, 4, dtype=torch.float32)
    calls = []

    def fake_load_object_inputs(data_config, *, object_id, angle_idx, view_mode):
        return {
            "cond": torch.randn(4, 1024),
            "target_slots": torch.tensor([1], dtype=torch.long),
            "mask_token_labels": torch.tensor([1, 1, 0, 0], dtype=torch.long),
            "target_part_names": ["opener_0"],
            "part_token_weights": part_token_weights,
        }

    def fake_run_part_ss_latent_flow(*args, **kwargs):
        calls.append(kwargs)
        return {
            "part_coords": {"opener_0": np.array([[1, 2, 3]], np.int64)},
            "part_latents": {"opener_0": np.zeros((8, 16, 16, 16), np.float32)},
        }

    import inference
    import inference_pipeline.object_inputs as object_inputs

    monkeypatch.setattr(object_inputs, "load_object_inputs", fake_load_object_inputs)
    monkeypatch.setattr(inference, "run_part_ss_latent_flow", fake_run_part_ss_latent_flow)

    written = part_flow_stage.run(
        run_dir,
        {},
        object_id="102252",
        angle_idx=0,
        view_mode="four",
        part_flow_ckpt="/fake/part.pt",
        ss_decoder_ckpt="/fake/ss_dec_conv3d.safetensors",
    )

    assert written == ["part_00_voxel.npz"]
    assert (run_dir / "parts" / "part_00_voxel.npz").is_file()
    assert not (run_dir / "parts" / "part_00_latent.npy").exists()
    assert calls[0]["decode"] is True
    assert calls[0]["ss_decoder_ckpt"] == "/fake/ss_dec_conv3d.safetensors"
    assert calls[0]["part_token_weights"] is part_token_weights
