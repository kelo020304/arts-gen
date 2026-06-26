import torch
import torch.nn.functional as F


from scripts.train.part_promptable_seg.part_promptable_seg_utils import boundary_band_mask
from scripts.train.part_promptable_seg.train_part_promptable_seg import dice_loss_prob, mask_loss


def _manual_mask_loss(logits, target, *, boundary=None, boundary_weight=1.0, focal_gamma=0.0):
    pos = target.sum(dim=1).clamp_min(1.0)
    neg = target.shape[1] - pos
    weights = (neg / pos).clamp(4.0, 1000.0)
    terms = []
    for idx in range(logits.shape[0]):
        elem = F.binary_cross_entropy_with_logits(
            logits[idx],
            target[idx],
            pos_weight=weights[idx],
            reduction="none",
        )
        if focal_gamma > 0:
            prob = torch.sigmoid(logits[idx])
            pt = torch.where(target[idx] > 0.5, prob, 1.0 - prob).clamp(1.0e-6, 1.0 - 1.0e-6)
            elem = elem * torch.pow(1.0 - pt, focal_gamma)
        if boundary is not None and boundary_weight > 1.0:
            elem = elem * torch.where(boundary[idx].bool(), elem.new_full((), boundary_weight), elem.new_ones(()))
        terms.append(elem.mean())
    bce = torch.stack(terms).mean()
    dice = dice_loss_prob(logits.sigmoid(), target, dims=(1,))
    return bce + dice, bce


def test_boundary_band_marks_shell_not_deep_interior():
    mask = torch.zeros((10, 10, 10), dtype=torch.float32)
    mask[2:8, 2:8, 2:8] = 1.0

    boundary = boundary_band_mask(mask, radius=1)

    assert boundary[2, 3, 3]
    assert boundary[1, 3, 3]
    assert not boundary[4, 4, 4]


def test_mask_loss_boundary_weight_multiplies_only_boundary_bce_terms():
    target_3d = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    target_3d[:, 1:3, 1:3, 1:3] = 1.0
    target = target_3d.reshape(1, -1)
    boundary = torch.zeros_like(target, dtype=torch.bool)
    boundary[:, 21] = True
    boundary[:, 42] = True
    logits = torch.linspace(-1.5, 1.5, target.numel(), dtype=torch.float32).reshape_as(target)

    got, items = mask_loss(logits, target, boundary_flat=boundary, boundary_weight=2.5, focal_gamma=0.0)
    expected, expected_bce = _manual_mask_loss(logits, target, boundary=boundary, boundary_weight=2.5)

    assert torch.allclose(got, expected)
    assert abs(items["mask_bce"] - float(expected_bce)) < 1.0e-6
    assert items["boundary_weight"] == 2.5
    assert items["boundary_voxel_ratio"] == float(boundary.float().mean())


def test_mask_loss_boundary_weight_one_is_exact_regression():
    target = torch.tensor([[0, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.float32)
    logits = torch.tensor([[-1.1, 0.3, 0.9, -0.4, 1.2, -0.8, 0.5, -0.2]], dtype=torch.float32)
    boundary = torch.ones_like(target, dtype=torch.bool)

    old, old_items = mask_loss(logits, target, focal_gamma=1.5)
    new, new_items = mask_loss(logits, target, boundary_flat=boundary, boundary_weight=1.0, focal_gamma=1.5)

    assert torch.equal(new, old)
    assert new_items["mask_bce"] == old_items["mask_bce"]
    assert new_items["mask_dice"] == old_items["mask_dice"]
    assert new_items["boundary_voxel_ratio"] == 0.0
