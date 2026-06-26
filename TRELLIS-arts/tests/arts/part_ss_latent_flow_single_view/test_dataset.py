"""Tests for PartSSLatentFlowSingleViewDataset synthetic manifest handling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _make_fixture(
    tmp_path: Path,
    *,
    parts: list[tuple[str, int, int]] | None = None,
) -> tuple[Path, Path]:
    data_root = tmp_path / "data_root"
    obj_id = "100064"
    angle_idx = 0
    view_idx = 3
    sample_id = f"physx_mobility_{obj_id}_angle_{angle_idx}_view_{view_idx}"
    latent_shape = (8, 16, 16, 16)
    recon_root = data_root / "reconstruction"
    parts = parts or [("drawer_0", 1, 1)]

    _write_npz(
        recon_root / "ss_latents_expanded" / obj_id / f"angle_{angle_idx}" / "latent.npz",
        mean=np.random.randn(*latent_shape).astype(np.float32),
    )
    target_parts = []
    for idx, (part_name, original_label, local_label) in enumerate(parts):
        _write_npy(
            recon_root / "ss_latents_per_part" / obj_id / f"angle_{angle_idx}" / f"{part_name}.npy",
            np.random.randn(*latent_shape).astype(np.float32),
        )
        _write_npy(
            recon_root / "voxel_expanded" / obj_id / f"angle_{angle_idx}" / "64" / f"ind_{part_name}.npy",
            np.array([[10 + idx, 20, 30], [11 + idx, 20, 30]], dtype=np.int64),
        )
        target_parts.append(
            {
                "name": part_name,
                "original_label": original_label,
                "local_label": local_label,
                "paths": {
                    "part_latent": f"reconstruction/ss_latents_per_part/{obj_id}/angle_{angle_idx}/{part_name}.npy",
                    "part_voxel": f"reconstruction/voxel_expanded/{obj_id}/angle_{angle_idx}/64/ind_{part_name}.npy",
                },
            }
        )
    _write_npy(
        recon_root / "voxel_expanded" / obj_id / f"angle_{angle_idx}" / "64" / "surface.npy",
        np.array([[10, 20, 30], [11, 20, 30], [12, 21, 31]], dtype=np.int64),
    )
    _write_npz(
        recon_root / "dinov2_tokens" / obj_id / f"angle_{angle_idx}" / "part_complete" / "tokens.npz",
        tokens=np.random.randn(16, 10, 8).astype(np.float32),
    )

    mask = np.zeros((512, 512), dtype=np.int32)
    for idx, (_, original_label, _) in enumerate(parts):
        row = idx % 3
        col = (idx * 2) % 3
        mask[row * 14 : (row + 1) * 14, col * 14 : (col + 1) * 14] = original_label
    _write_npy(
        data_root / "renders" / obj_id / f"angle_{angle_idx}" / "part_complete" / "mask" / "label" / f"mask_{view_idx}.npy",
        mask,
    )

    manifest_rel = Path("vlm/training_json/synthetic_manifest.jsonl")
    manifest_abs = data_root / manifest_rel
    row = {
        "sample_id": sample_id,
        "object_id": obj_id,
        "angle_idx": angle_idx,
        "view_idx": view_idx,
        "view_indices": [view_idx],
        "target_part_names": [part_name for part_name, _, _ in parts],
        "target_parts": target_parts,
        "image_paths": [f"renders/{obj_id}/angle_{angle_idx}/part_complete/rgb/view_{view_idx}.png"],
        "mask_paths": [f"renders/{obj_id}/angle_{angle_idx}/part_complete/mask/label/mask_{view_idx}.npy"],
        "feature_path": f"reconstruction/dinov2_tokens/{obj_id}/angle_{angle_idx}/part_complete/tokens.npz",
        "feature_view_index": view_idx,
        "paths": {
            "overall_surface": f"reconstruction/voxel_expanded/{obj_id}/angle_{angle_idx}/64/surface.npy",
            "dinov2_tokens": f"reconstruction/dinov2_tokens/{obj_id}/angle_{angle_idx}/part_complete/tokens.npz",
        },
    }
    manifest_abs.parent.mkdir(parents=True, exist_ok=True)
    manifest_abs.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return data_root, manifest_rel


def test_init_requires_num_views_one(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="num_views=1"):
        PartSSLatentFlowSingleViewDataset(
            {
                "data_root": str(data_root),
                "manifest_path": str(manifest_rel),
                "num_views": 4,
            }
        )


def test_init_forces_num_views_one_when_missing(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
        }
    )
    assert ds.num_views == 1


def test_enumerate_reads_mask_paths(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
        }
    )
    assert len(ds.samples) == 1
    sample = ds.samples[0]
    assert sample["mask_paths"] == [
        "renders/100064/angle_0/part_complete/mask/label/mask_3.npy"
    ]


def test_enumerate_drops_samples_over_max_target_parts(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    manifest_abs = data_root / manifest_rel
    normal = json.loads(manifest_abs.read_text(encoding="utf-8"))
    too_many = dict(normal)
    too_many["sample_id"] = f"{normal['sample_id']}_too_many"
    too_many["target_part_names"] = ["part_0", "part_1", "part_2"]
    too_many["target_parts"] = [
        {
            "name": name,
            "original_label": idx,
            "local_label": idx,
            "paths": {
                "part_latent": f"reconstruction/ss_latents_per_part/100064/angle_0/{name}.npy",
                "part_voxel": f"reconstruction/voxel_expanded/100064/angle_0/64/ind_{name}.npy",
            },
        }
        for idx, name in enumerate(too_many["target_part_names"], start=1)
    ]
    manifest_abs.write_text(
        json.dumps(normal) + "\n" + json.dumps(too_many) + "\n",
        encoding="utf-8",
    )

    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
            "max_target_parts_per_sample": 2,
            "drop_over_max_parts": True,
        }
    )

    assert len(ds.samples) == 1
    assert ds.samples[0]["sample_id"] == normal["sample_id"]
    assert ds.dropped_over_max_parts_samples == 1
    assert ds.dropped_over_max_parts == 3
    assert ds.max_seen_target_parts == 3


def test_enumerate_drops_samples_with_missing_dino_tokens(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    manifest_abs = data_root / manifest_rel
    normal = json.loads(manifest_abs.read_text(encoding="utf-8"))
    missing = dict(normal)
    missing["sample_id"] = f"{normal['sample_id']}_missing_tokens"
    missing["paths"] = dict(normal["paths"])
    missing["paths"]["dinov2_tokens"] = (
        "reconstruction/dinov2_tokens/100064/angle_0/part_complete/missing_tokens.npz"
    )
    manifest_abs.write_text(
        json.dumps(normal) + "\n" + json.dumps(missing) + "\n",
        encoding="utf-8",
    )

    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
            "filter_missing_dino_tokens": True,
        }
    )

    assert len(ds.samples) == 1
    assert ds.samples[0]["sample_id"] == normal["sample_id"]
    assert ds.dropped_missing_dino_token_samples == 1


def test_mask_paths_len_mismatch_raises(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    manifest_abs = data_root / manifest_rel
    row = json.loads(manifest_abs.read_text(encoding="utf-8"))
    row["mask_paths"] = ["a", "b"]
    manifest_abs.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="len\\(mask_paths\\)==1"):
        PartSSLatentFlowSingleViewDataset(
            {
                "data_root": str(data_root),
                "manifest_path": str(manifest_rel),
                "num_views": 1,
            }
        )


def test_getitem_returns_expected_shapes(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
        }
    )
    sample = ds[0]
    assert sample["x_1_parts"].shape == (1, 8, 16, 16, 16)
    assert sample["part_valid"].tolist() == [True]
    assert sample["target_part_names"] == ["drawer_0"]
    assert sample["cond"].shape == (10, 8)
    assert sample["mask_token_labels"].shape == (10,)
    assert sample["z_global"].shape == (8, 16, 16, 16)


def test_getitem_selects_manifest_feature_view_index(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    tokens_path = (
        data_root
        / "reconstruction"
        / "dinov2_tokens"
        / "100064"
        / "angle_0"
        / "part_complete"
        / "tokens.npz"
    )
    tokens = np.arange(16 * 10 * 8, dtype=np.float32).reshape(16, 10, 8)
    np.savez(tokens_path, tokens=tokens)

    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
        }
    )
    sample = ds[0]
    np.testing.assert_array_equal(sample["cond"].numpy(), tokens[3])


def test_getitem_mask_label_coverage(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
        }
    )
    sample = ds[0]
    assert int((sample["mask_token_labels"] == 1).sum()) > 0


def test_multi_part_original_labels_remap_to_distinct_target_slots(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(
        tmp_path,
        parts=[
            ("drawer_0", 7, 1),
            ("door_0", 42, 2),
        ],
    )
    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
        }
    )

    sample = ds[0]
    labels = sample["mask_token_labels"]
    assert sample["target_part_names"] == ["drawer_0", "door_0"]
    assert sample["target_slots"].tolist() == [1, 2]
    assert int((labels == 1).sum()) > 0
    assert int((labels == 2).sum()) > 0
    assert int((labels == 7).sum()) == 0
    assert int((labels == 42).sum()) == 0


def test_single_image_manifest_original_local_labels_remap_to_compact_target_slots(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(
        tmp_path,
        parts=[
            ("drawer_0", 7, 7),
            ("door_0", 42, 42),
        ],
    )
    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
            "use_mask_overlap_pooling": True,
            "mask_overlap_patch_grid_h": 3,
            "mask_overlap_patch_grid_w": 3,
            "mask_overlap_patch_h": 14,
            "mask_overlap_patch_w": 14,
            "mask_overlap_patch_start_index": 1,
        }
    )

    sample = ds[0]
    labels = sample["mask_token_labels"]
    assert sample["target_slots"].tolist() == [1, 2]
    assert int((labels == 1).sum()) > 0
    assert int((labels == 2).sum()) > 0
    assert int((labels == 7).sum()) == 0
    assert int((labels == 42).sum()) == 0
    assert sample["part_token_weights"].shape == (2, sample["cond"].shape[0])
    assert np.allclose(sample["part_token_weights"].sum(dim=-1).numpy(), np.ones(2))


def test_small_visible_part_gets_overlap_weights_when_hard_label_loses_patch(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(
        tmp_path,
        parts=[
            ("panel_0", 1, 1),
            ("button_0", 2, 2),
        ],
    )
    mask_path = (
        data_root
        / "renders"
        / "100064"
        / "angle_0"
        / "part_complete"
        / "mask"
        / "label"
        / "mask_3.npy"
    )
    mask = np.zeros((512, 512), dtype=np.int32)
    mask[:14, :14] = 1
    mask[:2, :2] = 2
    np.save(mask_path, mask)

    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
            "use_mask_overlap_pooling": True,
            "mask_overlap_patch_grid_h": 3,
            "mask_overlap_patch_grid_w": 3,
            "mask_overlap_patch_h": 14,
            "mask_overlap_patch_w": 14,
            "mask_overlap_patch_start_index": 1,
        }
    )
    sample = ds[0]
    assert int((sample["mask_token_labels"] == 2).sum()) == 0
    weights = sample["part_token_weights"]
    assert weights.shape == (2, sample["cond"].shape[0])
    assert np.isclose(float(weights[1].sum().item()), 1.0)
    assert float(weights[1, 1].item()) > 0.0


def test_mask_paths_missing_falls_back_to_parent(tmp_path: Path) -> None:
    from trellis.datasets.arts.part_ss_latent_flow_single_view import (
        PartSSLatentFlowSingleViewDataset,
    )

    data_root, manifest_rel = _make_fixture(tmp_path)
    manifest_abs = data_root / manifest_rel
    row = json.loads(manifest_abs.read_text(encoding="utf-8"))
    del row["mask_paths"]
    manifest_abs.write_text(json.dumps(row) + "\n", encoding="utf-8")

    parent_mask = data_root / "renders" / "100064" / "angle_0" / "mask" / "mask_3.npy"
    parent_mask.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((512, 512), dtype=np.int32)
    mask[:42, :42] = 1
    np.save(parent_mask, mask)

    ds = PartSSLatentFlowSingleViewDataset(
        {
            "data_root": str(data_root),
            "manifest_path": str(manifest_rel),
            "num_views": 1,
        }
    )
    sample = ds[0]
    assert int((sample["mask_token_labels"] == 1).sum()) > 0
