import shutil
import json
from pathlib import Path

import pytest
import torch

from trellis.datasets.arts.part_ss_latent_flow import PartSSLatentFlowDataset


def _cfg():
    return {
        "data_root": "data/smoke_test/1",
        "recon_subdir": "reconstruction",
        "mask_subdir": "renders",
        "manifest_path": "manifests/part_completion/subset_hq20.train.jsonl",
        "num_views": 4,
        "allow_missing_masks": False,
        "require_part_token": True,
    }


def _manifest_records():
    manifest = Path(_cfg()["data_root"]) / _cfg()["manifest_path"]
    return [json.loads(line) for line in manifest.open() if line.strip()]


def test_dataset_is_object_level_not_target_part_level():
    ds = PartSSLatentFlowDataset(_cfg())
    records = _manifest_records()
    expected_rows = len(records)
    expected_parts = sum(len(rec["target_part_names"]) for rec in records)
    assert expected_rows == 20
    assert expected_parts == 58
    assert len(ds) == expected_rows
    sample = ds[0]
    assert sample["x_1_parts"].shape[1:] == (8, 16, 16, 16)
    assert sample["part_valid"].shape == (sample["x_1_parts"].shape[0],)
    assert sample["part_valid"].all()
    assert sample["target_slots"].shape == sample["part_valid"].shape
    assert len(sample["target_part_names"]) == int(sample["part_valid"].sum())
    assert len(sample["raw_ind_coords"]) == int(sample["part_valid"].sum())
    assert sample["raw_surface_coords"].dim() == 2
    assert sample["raw_surface_coords"].shape[1] == 3
    assert sample["z_global"].shape == (8, 16, 16, 16)
    assert sample["cond"].shape == (4 * 1370, 1024)
    assert sample["mask_token_labels"].shape == (sample["cond"].shape[0],)
    assert sample["obj_id"]


def test_dataset_returns_part_raw_voxel_counts():
    ds = PartSSLatentFlowDataset(_cfg())
    sample = ds[0]

    expected = torch.tensor(
        [coords.shape[0] for coords in sample["raw_ind_coords"]],
        dtype=torch.float32,
    )
    assert torch.equal(sample["part_raw_voxel_counts"], expected)
    assert sample["part_raw_voxel_counts"].shape == sample["part_valid"].shape


def test_dataset_collate_pads_variable_part_count():
    ds = PartSSLatentFlowDataset(_cfg())
    sample_two = next(ds[i] for i in range(len(ds)) if len(ds[i]["target_part_names"]) == 2)
    sample_three = next(ds[i] for i in range(len(ds)) if len(ds[i]["target_part_names"]) == 3)
    batch = PartSSLatentFlowDataset.collate_fn([sample_two, sample_three])
    assert batch["x_1_parts"].shape[:2] == (2, 3)
    assert batch["x_1_parts"].shape[2:] == (8, 16, 16, 16)
    assert batch["part_valid"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["target_slots"].shape == (2, 3)
    assert batch["target_slots"][0, 2].item() == 0
    assert batch["z_global"].shape == (2, 8, 16, 16, 16)
    assert batch["cond"].shape == (2, 4 * 1370, 1024)
    assert batch["mask_token_labels"].shape[:2] == batch["cond"].shape[:2]
    assert len(batch["raw_surface_coords"]) == 2
    assert len(batch["target_part_names"][0]) == 2
    assert len(batch["target_part_names"][1]) == 3
    assert batch["part_raw_voxel_counts"].shape == (2, 3)
    assert batch["part_raw_voxel_counts"][0, 2].item() == 0
    assert torch.all(batch["part_raw_voxel_counts"][batch["part_valid"]] > 0)


def test_dataset_returns_part_token_weights_when_enabled():
    cfg = _cfg()
    cfg["use_mask_overlap_pooling"] = True
    ds = PartSSLatentFlowDataset(cfg)
    sample = ds[0]
    weights = sample["part_token_weights"]
    assert weights.shape == (len(sample["target_part_names"]), sample["cond"].shape[0])
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(weights.shape[0], dtype=weights.dtype),
        atol=1e-5,
    )


def test_dataset_omits_part_token_weights_when_disabled():
    cfg = _cfg()
    cfg["use_mask_overlap_pooling"] = False
    ds = PartSSLatentFlowDataset(cfg)
    sample = ds[0]
    assert "part_token_weights" not in sample


def test_iter_rgb_paths_falls_back_to_render_tree_when_manifest_paths_are_missing(tmp_path):
    ds = PartSSLatentFlowDataset.__new__(PartSSLatentFlowDataset)
    ds.data_root = tmp_path
    ds.mask_root = tmp_path / "renders"

    rgb_path = tmp_path / "renders" / "100015" / "angle_0" / "rgb" / "view_3.png"
    rgb_path.parent.mkdir(parents=True)
    rgb_path.write_bytes(b"png")
    sample = {
        "obj_id": "100015",
        "angle_idx": 0,
        "view_indices": [3],
        "image_paths": [],
    }

    assert ds._iter_rgb_paths(sample) == [rgb_path]


def test_dataset_collate_pads_part_token_weights_when_enabled():
    cfg = _cfg()
    cfg["use_mask_overlap_pooling"] = True
    ds = PartSSLatentFlowDataset(cfg)
    sample_two = next(ds[i] for i in range(len(ds)) if len(ds[i]["target_part_names"]) == 2)
    sample_three = next(ds[i] for i in range(len(ds)) if len(ds[i]["target_part_names"]) == 3)
    batch = PartSSLatentFlowDataset.collate_fn([sample_two, sample_three])
    assert batch["part_token_weights"].shape == (2, 3, batch["cond"].shape[1])
    assert torch.allclose(batch["part_token_weights"][0, :2].sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(batch["part_token_weights"][1, :3].sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.all(batch["part_token_weights"][0, 2] == 0)


def test_dataset_rejects_missing_z_global(tmp_path):
    cfg = _cfg()
    cfg["data_root"] = str(tmp_path / "smoke")
    shutil.copytree(Path("data/smoke_test/1"), cfg["data_root"])
    ds = PartSSLatentFlowDataset(cfg)
    sample = ds.samples[0]
    z_global_path = (
        Path(cfg["data_root"]) / "reconstruction" / "ss_latents_expanded"
        / sample["obj_id"] / f"angle_{sample['angle_idx']}" / "latent.npz"
    )
    z_global_path.unlink()
    with pytest.raises(FileNotFoundError, match=sample["obj_id"]):
        _ = ds[0]


def test_dataset_rejects_missing_z_part_for_target(tmp_path):
    cfg = _cfg()
    cfg["data_root"] = str(tmp_path / "smoke")
    shutil.copytree(Path("data/smoke_test/1"), cfg["data_root"])
    ds = PartSSLatentFlowDataset(cfg)
    sample = ds.samples[0]
    part_name = sample["parts"][0]["part_name"]
    z_part_path = (
        Path(cfg["data_root"]) / "reconstruction" / "ss_latents_per_part"
        / sample["obj_id"] / f"angle_{sample['angle_idx']}" / f"{part_name}.npy"
    )
    z_part_path.unlink()
    with pytest.raises(FileNotFoundError, match=part_name):
        _ = ds[0]


def test_dataset_rejects_missing_mask_for_target(tmp_path):
    cfg = _cfg()
    cfg["data_root"] = str(tmp_path / "smoke")
    shutil.copytree(Path("data/smoke_test/1"), cfg["data_root"])
    ds = PartSSLatentFlowDataset(cfg)
    sample = ds.samples[0]
    first_view = sample["view_indices"][0]
    mask_path = (
        Path(cfg["data_root"]) / "renders" / sample["obj_id"]
        / f"angle_{sample['angle_idx']}" / "mask" / f"mask_{first_view}.npy"
    )
    mask_path.unlink()
    with pytest.raises(FileNotFoundError, match="mask"):
        _ = ds[0]


def test_dataset_filters_samples_with_zero_target_mask_coverage(tmp_path):
    cfg = _cfg()
    cfg["data_root"] = str(tmp_path / "smoke")
    cfg["filter_zero_mask_coverage"] = True
    cfg["zero_mask_coverage_report"] = "manifests/part_completion/zero_mask_report.json"
    shutil.copytree(Path("data/smoke_test/1"), cfg["data_root"])

    manifest = Path(cfg["data_root"]) / cfg["manifest_path"]
    records = [json.loads(line) for line in manifest.open() if line.strip()]
    bad_sample_id = records[0]["sample_id"]
    records[0]["label_remap"] = {"999999": 1}
    with manifest.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    ds = PartSSLatentFlowDataset(cfg)

    assert len(ds) == len(records) - 1
    assert bad_sample_id not in {sample["sample_id"] for sample in ds.samples}
    report_path = Path(cfg["data_root"]) / cfg["zero_mask_coverage_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["skipped_samples"] == 1
    assert report["skipped_parts"] >= 1
    assert report["records"][0]["sample_id"] == bad_sample_id
    assert report["records"][0]["reason"] == "zero_2d_mask_token_coverage"


def test_dataset_filters_include_and_exclude_obj_ids_without_loading_files():
    ds = PartSSLatentFlowDataset.__new__(PartSSLatentFlowDataset)
    ds.include_obj_ids = {"keep"}
    ds.exclude_obj_ids = None
    samples = [{"obj_id": "keep"}, {"obj_id": "drop"}]
    assert ds._filter_samples_by_obj_id(samples) == [{"obj_id": "keep"}]

    ds.include_obj_ids = None
    ds.exclude_obj_ids = {"drop"}
    assert ds._filter_samples_by_obj_id(samples) == [{"obj_id": "keep"}]

    ds.include_obj_ids = {"same"}
    ds.exclude_obj_ids = {"same"}
    with pytest.raises(ValueError, match="overlap"):
        ds._filter_samples_by_obj_id(samples)
