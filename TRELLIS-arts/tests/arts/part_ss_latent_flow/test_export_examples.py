from pathlib import Path

import pytest
import torch


class _FakeGaussian:
    def save_ply(self, path):
        Path(path).write_text("ply\n", encoding="utf-8")


class _FakeMesh:
    success = True
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])


class _FakeSLat:
    coords = torch.tensor([[0, 1, 2, 3], [0, 3, 2, 1]], dtype=torch.int32)
    feats = torch.tensor([[0.5, -0.5], [1.5, -1.5]], dtype=torch.float32)


class _FakeDataset:
    def __init__(self, *, data_root: Path, rgb_paths=None, mask_paths=None):
        self.data_root = data_root
        self._rgb_paths = list(rgb_paths or [])
        self._mask_paths = list(mask_paths or [])

    def _iter_rgb_paths(self, sample):
        return self._rgb_paths

    def _iter_mask_paths(self, sample):
        return self._mask_paths


def test_copy_rgb_views_uses_dataset_resolver_when_manifest_image_paths_empty(tmp_path):
    import export_part_ss_latent_flow_examples as export_examples

    source_rgb = tmp_path / "resolved" / "view_7.png"
    source_rgb.parent.mkdir(parents=True)
    source_rgb.write_bytes(b"rgb")
    dataset = _FakeDataset(data_root=tmp_path, rgb_paths=[source_rgb])
    out_dir = tmp_path / "example"

    copied = export_examples._copy_rgb_views(
        dataset,
        {"image_paths": [], "view_indices": [7]},
        out_dir,
    )

    assert len(copied) == 1
    assert copied[0].startswith("rgb/")
    assert (out_dir / copied[0]).read_bytes() == b"rgb"


def test_copy_mask_views_uses_dataset_resolver_and_writes_mask_folder(tmp_path):
    import export_part_ss_latent_flow_examples as export_examples

    source_mask = tmp_path / "resolved" / "mask_7.npy"
    source_mask.parent.mkdir(parents=True)
    source_mask.write_bytes(b"mask")
    dataset = _FakeDataset(data_root=tmp_path, mask_paths=[source_mask])
    out_dir = tmp_path / "example"

    assert hasattr(export_examples, "_copy_mask_views")
    copied = export_examples._copy_mask_views(
        dataset,
        {"view_indices": [7]},
        out_dir,
    )

    assert len(copied) == 1
    assert copied[0].startswith("mask/")
    assert (out_dir / copied[0]).read_bytes() == b"mask"


def test_save_decoded_slat_assets_records_mesh_and_gaussian(tmp_path):
    import export_part_ss_latent_flow_examples as export_examples

    asset_dir = tmp_path / "pred_assets" / "00_handle"
    record = export_examples._save_decoded_slat_assets(
        {"gaussian": _FakeGaussian(), "mesh": _FakeMesh()},
        asset_dir,
        tmp_path,
    )

    assert record == {
        "pred_gaussian": "pred_assets/00_handle/gaussians.ply",
        "pred_mesh": "pred_assets/00_handle/mesh.obj",
    }
    assert (asset_dir / "gaussians.ply").read_text(encoding="utf-8") == "ply\n"
    assert (asset_dir / "mesh.obj").is_file()


def test_write_part_slat_assets_writes_slat_pt_and_assets_for_nonempty_voxel(tmp_path, monkeypatch):
    import export_part_ss_latent_flow_examples as export_examples

    calls = []
    fake_slat = _FakeSLat()

    def fake_run_slat_flow_from_tokens(cond_tokens, pred_coords, ckpt_path, *, num_steps, seed=None):
        calls.append({
            "cond_tokens": cond_tokens.clone(),
            "pred_coords": pred_coords.clone(),
            "ckpt_path": ckpt_path,
            "num_steps": num_steps,
            "seed": seed,
        })
        return fake_slat

    monkeypatch.setattr(export_examples, "run_slat_flow_from_tokens", fake_run_slat_flow_from_tokens)
    monkeypatch.setattr(
        export_examples,
        "decode_slat_assets",
        lambda slat, **kwargs: {"gaussian": _FakeGaussian(), "mesh": _FakeMesh()},
    )

    cond_tokens = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    pred_coords = torch.tensor([[1, 2, 3]], dtype=torch.long)
    record = export_examples._write_part_slat_assets(
        pred_coords=pred_coords,
        cond_tokens=cond_tokens,
        example_dir=tmp_path,
        stem="00_handle",
        slat_cfg={
            "flow_ckpt": "flow.safetensors",
            "gs_decoder_ckpt": "gs.safetensors",
            "mesh_decoder_ckpt": "mesh.safetensors",
            "num_steps": 7,
            "empty_policy": "skip",
        },
        slat_seed=12345,
    )

    assert len(calls) == 1
    assert torch.equal(calls[0]["cond_tokens"], cond_tokens)
    assert torch.equal(calls[0]["pred_coords"], pred_coords)
    assert calls[0]["ckpt_path"] == "flow.safetensors"
    assert calls[0]["num_steps"] == 7
    assert calls[0]["seed"] == 12345
    assert record == {
        "pred_gaussian": "pred_assets/00_handle/gaussians.ply",
        "pred_mesh": "pred_assets/00_handle/mesh.obj",
        "slat_status": "generated",
        "pred_slat": "pred_slat/00_handle.pt",
        "pred_asset_dir": "pred_assets/00_handle",
        "slat_seed": 12345,
    }
    payload = torch.load(tmp_path / "pred_slat" / "00_handle.pt", map_location="cpu", weights_only=True)
    assert payload["format"] == "trellis_sparse_tensor_v1"
    assert payload["is_normalized"] is True
    assert torch.equal(payload["coords"], fake_slat.coords.cpu())
    assert torch.equal(payload["feats"], fake_slat.feats.cpu())
    assert (tmp_path / "pred_assets" / "00_handle" / "mesh.obj").is_file()
    assert (tmp_path / "pred_assets" / "00_handle" / "gaussians.ply").is_file()


def test_write_part_slat_assets_raises_on_empty_voxel_when_error_policy(tmp_path):
    import export_part_ss_latent_flow_examples as export_examples

    with pytest.raises(ValueError, match="empty predicted voxel part: 00_handle"):
        export_examples._write_part_slat_assets(
            pred_coords=torch.zeros(0, 3, dtype=torch.long),
            cond_tokens=torch.zeros(2, 4),
            example_dir=tmp_path,
            stem="00_handle",
            slat_cfg={
                "flow_ckpt": "flow.safetensors",
                "gs_decoder_ckpt": "gs.safetensors",
                "mesh_decoder_ckpt": "mesh.safetensors",
                "num_steps": 7,
                "empty_policy": "error",
            },
        )
