import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trellis.datasets.arts.part_mmdit import PartMMDiTDataset, raw_coords_to_part_fg_mask


LATENT_SHAPE = (8, 16, 16, 16)


def _write_npz_mean(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=array)


def _write_npz_tokens(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, tokens=array)


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _make_fixture(tmp_path: Path) -> dict:
    data_root = tmp_path / "data_root"
    recon_subdir = "reconstruction"
    obj_id = "objA"
    angle_idx = 0
    view_indices = [2, 5, 8, 9]
    latent = np.random.randn(*LATENT_SHAPE).astype(np.float32)

    cache_path = data_root / recon_subdir / "name_emb_cache" / "clip_vitl14_seq.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dim": 768,
            "seq": {
                "button": {
                    "tokens": torch.ones(5, 768),
                    "mask": torch.ones(5, dtype=torch.bool),
                },
                "body": {
                    "tokens": torch.full((7, 768), 2.0),
                    "mask": torch.ones(7, dtype=torch.bool),
                },
                "handle": {
                    "tokens": torch.full((4, 768), 3.0),
                    "mask": torch.ones(4, dtype=torch.bool),
                },
            },
        },
        cache_path,
    )

    _write_npz_mean(
        data_root / recon_subdir / "ss_latents_expanded" / obj_id / "angle_0" / "latent.npz",
        latent,
    )
    _write_npz_tokens(
        data_root / recon_subdir / "dinov2_tokens" / obj_id / "angle_0" / "tokens.npz",
        np.random.randn(12, 7, 1024).astype(np.float32),
    )

    parts = [
        ("button_0", "button", np.array([[1, 2, 3], [1, 2, 4]], dtype=np.int64)),
        ("body_0", "body", np.array([[3, 2, 1], [4, 2, 1], [5, 2, 1]], dtype=np.int64)),
    ]
    target_parts = []
    part_info = {"parts": {}}
    for part_name, part_type, coords in parts:
        _write_npy(
            data_root
            / recon_subdir
            / "ss_latents_per_part"
            / obj_id
            / "angle_0"
            / f"{part_name}.npy",
            latent,
        )
        _write_npy(
            data_root
            / recon_subdir
            / "voxel_expanded"
            / obj_id
            / "angle_0"
            / "64"
            / f"ind_{part_name}.npy",
            coords,
        )
        part_info["parts"][part_name] = {"type": part_type}
        target_parts.append(
            {
                "name": part_name,
                "local_label": len(target_parts) + 1,
                "paths": {
                    "part_latent": (
                        f"{recon_subdir}/ss_latents_per_part/{obj_id}/angle_0/"
                        f"{part_name}.npy"
                    ),
                    "part_voxel": (
                        f"{recon_subdir}/voxel_expanded/{obj_id}/angle_0/64/"
                        f"ind_{part_name}.npy"
                    ),
                },
            }
        )

    part_info_path = data_root / recon_subdir / "part_info" / obj_id / "part_info.json"
    part_info_path.parent.mkdir(parents=True, exist_ok=True)
    part_info_path.write_text(json.dumps(part_info), encoding="utf-8")

    bbox = {
        "resolution": 512,
        "parts": {
            "button_0": {
                "views": {
                    "2": {"bbox": [128, 64, 256, 192], "visible": True},
                    "5": {"bbox": [0, 0, 10, 10], "visible": False},
                    "8": {"bbox": [256, 256, 512, 512], "visible": True},
                    "9": {"bbox": [64, 64, 128, 128], "visible": True},
                }
            },
            "body_0": {
                "views": {
                    "2": {"bbox": [0, 0, 512, 512], "visible": True},
                    "5": {"bbox": [10, 20, 110, 220], "visible": True},
                    "8": {"bbox": [0, 0, 1, 1], "visible": False},
                    "9": {"bbox": [100, 100, 200, 300], "visible": True},
                }
            },
        },
    }
    bbox_path = data_root / "renders" / obj_id / "angle_0" / "bbox_gt.json"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    bbox_path.write_text(json.dumps(bbox), encoding="utf-8")

    manifest_path = data_root / "manifests" / "part_completion" / "train.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "object_id": obj_id,
        "angle_idx": angle_idx,
        "sample_id": "sampleA",
        "target_part_names": [name for name, _, _ in parts],
        "target_parts": target_parts,
        "view_indices": view_indices,
        "paths": {
            "overall_latent": (
                f"{recon_subdir}/ss_latents_expanded/{obj_id}/angle_0/latent.npz"
            ),
            "dinov2_tokens": (
                f"{recon_subdir}/dinov2_tokens/{obj_id}/angle_0/tokens.npz"
            ),
            "part_info": f"{recon_subdir}/part_info/{obj_id}/part_info.json",
        },
    }
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    return {
        "data_root": data_root,
        "manifest_path": Path("manifests/part_completion/train.jsonl"),
        "obj_id": obj_id,
    }


def _cfg(fixture: dict) -> dict:
    return {
        "data_root": str(fixture["data_root"]),
        "recon_subdir": "reconstruction",
        "mask_subdir": "renders",
        "manifest_path": str(fixture["manifest_path"]),
        "num_views": 4,
    }


def test_getitem_contract_and_anchor_view_selection(tmp_path):
    fixture = _make_fixture(tmp_path)
    ds = PartMMDiTDataset(_cfg(fixture))

    sample = ds[0]

    assert sample["x_1_parts"].shape == (2, 8, 16, 16, 16)
    assert sample["z_global"].shape == (8, 16, 16, 16)
    assert sample["cond"].shape == (4 * 7, 1024)
    assert sample["name_tokens"].shape == (2, 7, 768)
    assert sample["name_mask"].shape == (2, 7)
    assert sample["name_mask"].dtype == torch.bool
    assert sample["name_mask"].tolist() == [
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, True, True],
    ]
    assert torch.equal(sample["name_tokens"][0, 5:], torch.zeros(2, 768))
    assert sample["anchor"].shape == (2, 4, 4)
    assert sample["anchor_valid"].dtype == torch.bool
    assert sample["anchor_valid"].tolist() == [
        [True, False, True, True],
        [True, True, False, True],
    ]
    assert torch.allclose(sample["anchor"][0, 0], torch.tensor([0.375, 0.25, 0.25, 0.25]))
    assert torch.equal(sample["part_valid"], torch.tensor([True, True]))
    assert sample["part_raw_voxel_counts"].tolist() == [2.0, 3.0]
    assert sample["part_fg_mask"].shape == (2, 16, 16, 16)
    assert sample["part_fg_mask"].dtype == torch.bool
    assert int(sample["part_fg_mask"][0].sum().item()) == 2
    assert sample["part_fg_mask"][0, 0, 0, 0]
    assert sample["part_fg_mask"][1, 0, 0, 0]
    assert sample["part_fg_mask"][1, 1, 0, 0]
    assert sample["target_part_names"] == ["button_0", "body_0"]
    assert sample["target_part_types"] == ["button", "body"]
    assert sample["obj_id"] == fixture["obj_id"]
    assert sample["view_indices"] == [2, 5, 8, 9]


def test_collate_pads_variable_part_count(tmp_path):
    fixture = _make_fixture(tmp_path)
    ds = PartMMDiTDataset(_cfg(fixture))
    first = ds[0]
    second = dict(first)
    second["x_1_parts"] = first["x_1_parts"][:1]
    second["part_valid"] = first["part_valid"][:1]
    second["part_raw_voxel_counts"] = first["part_raw_voxel_counts"][:1]
    second["part_fg_mask"] = first["part_fg_mask"][:1]
    second["name_tokens"] = first["name_tokens"][:1, :5]
    second["name_mask"] = first["name_mask"][:1, :5]
    second["anchor"] = first["anchor"][:1]
    second["anchor_valid"] = first["anchor_valid"][:1]
    second["target_part_names"] = first["target_part_names"][:1]
    second["target_part_types"] = first["target_part_types"][:1]
    second["raw_ind_coords"] = first["raw_ind_coords"][:1]

    batch = PartMMDiTDataset.collate_fn([first, second])

    assert batch["x_1_parts"].shape == (2, 2, 8, 16, 16, 16)
    assert batch["part_valid"].tolist() == [[True, True], [True, False]]
    assert batch["name_tokens"].shape == (2, 2, 7, 768)
    assert batch["name_mask"].shape == (2, 2, 7)
    assert batch["name_mask"].tolist() == [
        [
            [True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True],
        ],
        [
            [True, True, True, True, True, False, False],
            [False, False, False, False, False, False, False],
        ],
    ]
    assert batch["anchor"].shape == (2, 2, 4, 4)
    assert batch["anchor_valid"].shape == (2, 2, 4)
    assert batch["part_raw_voxel_counts"].tolist() == [[2.0, 3.0], [2.0, 0.0]]
    assert batch["part_fg_mask"].shape == (2, 2, 16, 16, 16)
    assert batch["part_fg_mask"][0, 1].any()
    assert not batch["part_fg_mask"][1, 1].any()
    assert batch["z_global"].shape == (2, 8, 16, 16, 16)
    assert batch["cond"].shape == (2, 4 * 7, 1024)
    assert batch["target_part_names"] == [["button_0", "body_0"], ["button_0"]]


def test_raw_coords_to_part_fg_mask_max_pools_64_to_16_and_dilates():
    coords = torch.tensor([[0, 0, 0], [3, 3, 3], [4, 0, 0], [63, 63, 63]])

    mask = raw_coords_to_part_fg_mask(coords, dilate=0)

    assert int(mask.sum().item()) == 3
    assert mask[0, 0, 0]
    assert mask[1, 0, 0]
    assert mask[15, 15, 15]

    dilated = raw_coords_to_part_fg_mask(torch.tensor([[8, 8, 8]]), dilate=1)
    assert dilated[2, 2, 2]
    assert dilated[1, 2, 2]
    assert dilated[3, 3, 3]


def test_dataset_rejects_missing_name_embedding(tmp_path):
    fixture = _make_fixture(tmp_path)
    cache_path = (
        fixture["data_root"]
        / "reconstruction"
        / "name_emb_cache"
        / "clip_vitl14_seq.pt"
    )
    torch.save(
        {
            "dim": 768,
            "seq": {
                "button": {
                    "tokens": torch.ones(5, 768),
                    "mask": torch.ones(5, dtype=torch.bool),
                }
            },
        },
        cache_path,
    )
    ds = PartMMDiTDataset(_cfg(fixture))

    with pytest.raises(KeyError, match="body"):
        _ = ds[0]


def test_dataset_filters_include_and_exclude_obj_ids(tmp_path):
    fixture = _make_fixture(tmp_path)
    data_root = fixture["data_root"]
    recon_subdir = "reconstruction"
    obj_b = "objB"

    part_info_path = data_root / recon_subdir / "part_info" / obj_b / "part_info.json"
    part_info_path.parent.mkdir(parents=True, exist_ok=True)
    part_info_path.write_text(
        json.dumps({"parts": {"button_0": {"type": "button"}}}),
        encoding="utf-8",
    )

    manifest_path = data_root / fixture["manifest_path"]
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8")
        + json.dumps(
            {
                "object_id": obj_b,
                "angle_idx": 0,
                "sample_id": "sampleB",
                "target_part_names": ["button_0"],
                "target_parts": [{"name": "button_0"}],
                "view_indices": [2, 5, 8, 9],
                "paths": {
                    "part_info": f"{recon_subdir}/part_info/{obj_b}/part_info.json",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    include_cfg = dict(_cfg(fixture), include_obj_ids=[fixture["obj_id"]])
    include_ds = PartMMDiTDataset(include_cfg)
    assert [sample["obj_id"] for sample in include_ds.samples] == [fixture["obj_id"]]

    exclude_cfg = dict(_cfg(fixture), exclude_obj_ids=[fixture["obj_id"]])
    exclude_ds = PartMMDiTDataset(exclude_cfg)
    assert [sample["obj_id"] for sample in exclude_ds.samples] == [obj_b]
