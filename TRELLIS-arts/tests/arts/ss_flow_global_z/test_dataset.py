import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


def _install_trellis_stub():
    trellis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if trellis_path not in sys.path:
        sys.path.insert(0, trellis_path)
    if "trellis" not in sys.modules:
        pkg = types.ModuleType("trellis")
        pkg.__path__ = [os.path.join(trellis_path, "trellis")]
        pkg.__package__ = "trellis"
        sys.modules["trellis"] = pkg
    for sp in ("datasets", "utils"):
        name = f"trellis.{sp}"
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(trellis_path, "trellis", sp)]
            mod.__package__ = name
            sys.modules[name] = mod


_install_trellis_stub()

from trellis.datasets.arts.ss_flow_global_z import SSFlowGlobalZDataset


def _write_sample(root: Path, obj_id: str = "obj_001", angle_idx: int = 0, *, views: int = 4):
    angle_dir = f"angle_{angle_idx}"
    latent_dir = root / "reconstruction" / "ss_latents_expanded" / obj_id / angle_dir
    token_dir = root / "reconstruction" / "dinov2_tokens" / obj_id / angle_dir
    latent_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    latent = np.arange(8 * 16 * 16 * 16, dtype=np.float32).reshape(8, 16, 16, 16)
    tokens = np.arange(views * 3 * 5, dtype=np.float32).reshape(views, 3, 5)
    np.savez_compressed(latent_dir / "latent.npz", mean=latent)
    np.savez_compressed(token_dir / "tokens.npz", tokens=tokens)
    return latent, tokens


def _write_json_manifest(root: Path, obj_id: str = "obj_001", angle_idx: int = 0):
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({
            "samples": [
                {"object_id": obj_id, "angle_idx": angle_idx, "complete": True, "view_indices": [0, 1, 2, 3]},
                {"object_id": "incomplete", "angle_idx": 0, "complete": False},
            ]
        }),
        encoding="utf-8",
    )
    return manifest


def _base_cfg(root: Path, manifest_path: str | None = "manifest.json"):
    return {
        "data_root": str(root),
        "recon_subdir": "reconstruction",
        "manifest_path": manifest_path,
        "num_views": 4,
        "condition_mode": "multiflow_view",
        "token_count": 3,
        "token_dim": 5,
    }


def test_dataset_reads_json_manifest_and_collates(tmp_path):
    latent, tokens = _write_sample(tmp_path)
    _write_json_manifest(tmp_path)

    ds = SSFlowGlobalZDataset(_base_cfg(tmp_path))
    assert len(ds) == 1
    assert ds.samples[0]["obj_id"] == "obj_001"

    sample = ds[0]
    assert torch.equal(sample["x_0"], torch.from_numpy(latent))
    assert sample["x_0"].shape == (8, 16, 16, 16)
    assert sample["cond"].shape == (4, 3, 5)
    assert torch.equal(sample["cond"], torch.from_numpy(tokens))

    batch = SSFlowGlobalZDataset.collate_fn([sample, sample])
    assert batch["x_0"].shape == (2, 8, 16, 16, 16)
    assert batch["cond"].shape == (2, 4, 3, 5)
    assert set(batch) == {"x_0", "cond"}


def test_dataset_reads_jsonl_part_manifest_paths(tmp_path):
    _write_sample(tmp_path, obj_id="obj_002", angle_idx=7)
    manifest = tmp_path / "parts.jsonl"
    record = {
        "object_id": "obj_002",
        "angle_idx": 7,
        "sample_id": "obj_002_angle_7_parts",
        "target_part_names": ["a", "b"],
        "view_indices": [0, 1, 2, 3],
        "paths": {
            "overall_latent": "reconstruction/ss_latents_expanded/obj_002/angle_7/latent.npz",
            "dinov2_tokens": "reconstruction/dinov2_tokens/obj_002/angle_7/tokens.npz",
        },
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    ds = SSFlowGlobalZDataset(_base_cfg(tmp_path, "parts.jsonl"))

    assert len(ds) == 1
    assert ds.samples[0]["sample_id"] == "obj_002_angle_7_parts"
    assert ds[0]["x_0"].shape == (8, 16, 16, 16)


def test_dataset_enumerates_directories_without_manifest(tmp_path):
    _write_sample(tmp_path, obj_id="obj_003", angle_idx=2)
    ds = SSFlowGlobalZDataset(_base_cfg(tmp_path, None))

    assert len(ds) == 1
    assert ds.samples[0]["obj_id"] == "obj_003"
    assert ds.samples[0]["angle_idx"] == 2


def test_dataset_filters_test_obj_ids_and_max_samples(tmp_path):
    _write_sample(tmp_path, obj_id="a", angle_idx=0)
    _write_sample(tmp_path, obj_id="b", angle_idx=0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "samples": [
                {"object_id": "a", "angle_idx": 0},
                {"object_id": "b", "angle_idx": 0},
            ]
        }),
        encoding="utf-8",
    )
    cfg = _base_cfg(tmp_path)
    cfg["test_obj_ids"] = ["b"]
    cfg["max_samples"] = 1

    ds = SSFlowGlobalZDataset(cfg)

    assert len(ds) == 1
    assert ds.samples[0]["obj_id"] == "b"


def test_dataset_uses_manifest_view_indices(tmp_path):
    _write_sample(tmp_path, views=5)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "samples": [
                {"object_id": "obj_001", "angle_idx": 0, "view_indices": [4, 2, 1, 0]},
            ]
        }),
        encoding="utf-8",
    )
    cfg = _base_cfg(tmp_path)
    cfg["num_views"] = 4

    sample = SSFlowGlobalZDataset(cfg)[0]

    all_tokens = np.load(
        tmp_path / "reconstruction" / "dinov2_tokens" / "obj_001" / "angle_0" / "tokens.npz"
    )["tokens"]
    expected = torch.from_numpy(all_tokens[[4, 2, 1, 0]]).float()
    assert torch.equal(sample["cond"], expected)
    assert SSFlowGlobalZDataset(cfg).samples[0]["view_indices"] == [4, 2, 1, 0]


def test_dataset_falls_back_to_config_view_indices_when_manifest_omits_them(tmp_path):
    _write_sample(tmp_path, views=5)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"samples": [{"object_id": "obj_001", "angle_idx": 0}]}),
        encoding="utf-8",
    )
    cfg = _base_cfg(tmp_path)
    cfg["view_indices"] = [4, 2, 1, 0]

    sample = SSFlowGlobalZDataset(cfg)[0]

    assert torch.equal(
        sample["cond"],
        torch.from_numpy(
            np.load(
                tmp_path / "reconstruction" / "dinov2_tokens" / "obj_001" / "angle_0" / "tokens.npz"
            )["tokens"][[4, 2, 1, 0]]
        ).float(),
    )


def test_dataset_rejects_negative_physical_view_ids(tmp_path):
    _write_sample(tmp_path, views=5)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "samples": [
                {"object_id": "obj_001", "angle_idx": 0, "view_indices": [0, 1, 2, -1]},
            ]
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-negative physical view ids"):
        SSFlowGlobalZDataset(_base_cfg(tmp_path))


def test_view_aug_flag_is_reserved(tmp_path):
    _write_sample(tmp_path)
    _write_json_manifest(tmp_path)
    cfg = _base_cfg(tmp_path)
    cfg["view_aug"] = {"enabled": True}

    with pytest.raises(NotImplementedError, match="view_aug"):
        SSFlowGlobalZDataset(cfg)


def test_dataset_rejects_missing_token_key(tmp_path):
    _write_sample(tmp_path)
    _write_json_manifest(tmp_path)
    token_path = tmp_path / "reconstruction" / "dinov2_tokens" / "obj_001" / "angle_0" / "tokens.npz"
    np.savez_compressed(token_path, bad=np.zeros((4, 3, 5), dtype=np.float32))
    ds = SSFlowGlobalZDataset(_base_cfg(tmp_path))

    with pytest.raises(KeyError, match="tokens"):
        _ = ds[0]


def test_dataset_rejects_bad_latent_shape(tmp_path):
    _write_sample(tmp_path)
    _write_json_manifest(tmp_path)
    latent_path = tmp_path / "reconstruction" / "ss_latents_expanded" / "obj_001" / "angle_0" / "latent.npz"
    np.savez_compressed(latent_path, mean=np.zeros((8, 8, 8, 8), dtype=np.float32))
    ds = SSFlowGlobalZDataset(_base_cfg(tmp_path))

    with pytest.raises(ValueError, match="8,16,16,16"):
        _ = ds[0]


def test_dataset_rejects_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        SSFlowGlobalZDataset(_base_cfg(tmp_path))
