import numpy as np
import torch

from trellis.models.part_seg.point_mask_encoder import (
    TYPE_BOUNDARY,
    TYPE_CENTROID,
    TYPE_INTERIOR,
    PointMaskEncoder,
    sample_mask_points,
)
from trellis.models.part_seg.promptable_latent_seg import PromptablePartLatentSegNet


def _tiny_mask(size=512):
    mask = np.zeros((size, size), dtype=np.uint8)
    coords = np.asarray(
        [
            [100, 200],
            [100, 201],
            [101, 200],
            [101, 201],
            [102, 200],
            [102, 201],
        ],
        dtype=np.int64,
    )
    mask[coords[:, 0], coords[:, 1]] = 1
    return mask, coords[:, ::-1].astype(np.float32)


def test_tiny_mask_keeps_all_pixels_and_centroid():
    mask, expected_xy = _tiny_mask()
    sample = sample_mask_points(mask, k_boundary=32, k_interior=32)

    assert sample.coords_xy.shape[0] == 13
    assert int((sample.point_types == TYPE_INTERIOR).sum()) == 6
    assert int((sample.point_types == TYPE_BOUNDARY).sum()) == 6
    assert int((sample.point_types == TYPE_CENTROID).sum()) == 1

    interior_xy = sample.coords_xy[sample.point_types == TYPE_INTERIOR]
    boundary_xy = sample.coords_xy[sample.point_types == TYPE_BOUNDARY]
    assert {tuple(v) for v in interior_xy.tolist()} == {tuple(v) for v in expected_xy.tolist()}
    assert {tuple(v) for v in boundary_xy.tolist()} == {tuple(v) for v in expected_xy.tolist()}

    centroid = sample.coords_xy[sample.point_types == TYPE_CENTROID][0]
    assert np.allclose(centroid, expected_xy.mean(axis=0))


def test_empty_views_padding_and_all_empty_no_prompt_mask():
    encoder = PointMaskEncoder(dim=32, num_views=4, mask_size=32, k_boundary=4, k_interior=4).eval()
    masks = torch.zeros(2, 4, 32, 32)
    masks[0, 2, 10:12, 20:22] = 1.0

    out = encoder(masks)

    assert out.tokens.shape[0] == 2
    assert out.key_padding_mask.dtype == torch.bool
    assert out.no_prompt_mask.tolist() == [False, True]
    assert int(out.counts[0]) == 9
    assert int(out.counts[1]) == 1
    assert not bool(out.key_padding_mask[0, : out.counts[0]].any())
    assert bool(out.key_padding_mask[0, out.counts[0] :].all())
    assert not bool(out.key_padding_mask[1, 0])
    assert bool(out.key_padding_mask[1, 1:].all())


def test_eval_sampling_is_deterministic():
    encoder = PointMaskEncoder(dim=32, num_views=4, mask_size=64, k_boundary=8, k_interior=8).eval()
    masks = torch.zeros(1, 4, 64, 64)
    masks[0, 0, 8:20, 9:21] = 1.0
    masks[0, 3, 30:45, 40:55] = 1.0

    out_a = encoder(masks)
    out_b = encoder(masks)

    assert torch.equal(out_a.counts, out_b.counts)
    assert torch.equal(out_a.key_padding_mask, out_b.key_padding_mask)
    assert torch.equal(out_a.coords_uv, out_b.coords_uv)
    assert torch.equal(out_a.point_types, out_b.point_types)
    assert torch.allclose(out_a.tokens, out_b.tokens)


def test_multicomponent_each_component_gets_boundary_and_interior():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:9, 4:9] = 1
    mask[40:47, 50:57] = 1
    sample = sample_mask_points(mask, k_boundary=8, k_interior=8)

    for y0, y1, x0, x1 in ((4, 9, 4, 9), (40, 47, 50, 57)):
        xy = sample.coords_xy
        inside = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
        assert int(((sample.point_types == TYPE_BOUNDARY) & inside).sum()) >= 1
        assert int(((sample.point_types == TYPE_INTERIOR) & inside).sum()) >= 1


def test_fg_point_encoder_drop_in_forward_shapes():
    model = PromptablePartLatentSegNet(
        latent_channels=8,
        dim=32,
        num_views=4,
        depth=1,
        head_depth=1,
        heads=4,
        mask_size=32,
        mask_encoder="fg_points",
        point_k_boundary=4,
        point_k_interior=4,
        use_voxel_head=True,
        voxel_depth=1,
    ).eval()
    z_global = torch.randn(2, 8, 16, 16, 16)
    masks = torch.zeros(2, 4, 32, 32)
    masks[0, 0, 5:8, 5:8] = 1.0
    masks[0, 2, 20:23, 10:15] = 1.0
    masks[1, 1, 12:18, 16:22] = 1.0
    empty = torch.zeros(8, 16, 16, 16)

    with torch.no_grad():
        out = model(z_global, masks, empty)

    assert out["mask_tokens"].shape[0] == 2
    assert out["mask_tokens"].shape[-1] == 32
    assert out["features"].shape == (2, 4096, 32)
    assert out["m_logit"].shape == (2, 4096)
    assert out["part_latent"].shape == z_global.shape
    assert torch.isfinite(out["part_latent"]).all()


def test_fg_point_encoder_voxel_head_padding_path_shape():
    model = PromptablePartLatentSegNet(
        latent_channels=8,
        dim=32,
        num_views=4,
        depth=1,
        head_depth=1,
        heads=4,
        mask_size=32,
        mask_encoder="fg_points",
        point_k_boundary=4,
        point_k_interior=4,
        use_voxel_head=True,
        voxel_depth=1,
    ).eval()
    z_global = torch.randn(1, 8, 16, 16, 16)
    masks = torch.zeros(1, 4, 32, 32)
    masks[0, 0, 5:9, 5:9] = 1.0
    candidate = torch.zeros(1, 16, 16, 16, dtype=torch.bool)
    candidate[:, 0, 0, 0] = True
    occ = torch.zeros(1, 1, 64, 64, 64)
    occ[:, :, :4, :4, :4] = 1.0

    with torch.no_grad():
        out = model.forward_voxels(z_global, masks, candidate, occ, max_voxels_per_sample=0)

    assert out["voxel_logits"].shape == (1, 64)
    assert out["voxel_pad_mask"].shape == (1, 64)
    assert len(out["voxel_coords"]) == 1
    assert torch.isfinite(out["voxel_logits"]).all()
